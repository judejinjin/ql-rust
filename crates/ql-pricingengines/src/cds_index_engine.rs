//! CDS Index pricing engine.
//!
//! Prices a CDS index by treating it as a homogeneous portfolio of single-name
//! CDS contracts (flat spread / flat hazard rate assumption). The engine
//! computes protection leg and premium leg NPVs, fair spread, and
//! upfront ↔ running conversion.
//!
//! ## References
//!
//! - O'Kane, D. "Modelling Single-name and Multi-name Credit Derivatives"
//! - ISDA Standard CDS Model
//! - Morgan Stanley "CDS Index methodology" (2019)

use ql_instruments::cds_index::CdsIndex;
use ql_instruments::credit_default_swap::CdsProtectionSide;
use ql_termstructures::default_term_structure::DefaultProbabilityTermStructure;
use ql_termstructures::yield_term_structure::YieldTermStructure;
use ql_time::Date;
use std::sync::Arc;

/// Results from the CDS index pricing engine.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CdsIndexResult {
    /// Clean NPV (per unit notional, sign convention: positive for protection buyer).
    pub clean_npv: f64,
    /// Dirty NPV (includes accrued).
    pub dirty_npv: f64,
    /// Protection leg PV (unsigned).
    pub protection_leg_pv: f64,
    /// Premium leg PV per bp (unsigned, risky annuity / RPV01).
    pub rpv01: f64,
    /// Accrued premium.
    pub accrued: f64,
    /// Index fair spread (running spread that makes NPV = 0).
    pub fair_spread: f64,
    /// Upfront amount (positive = buyer pays) for the given coupon.
    pub upfront: f64,
    /// Points upfront (upfront / notional).
    pub points_upfront: f64,
}

/// Price a CDS index using the homogeneous flat-spread model.
///
/// # Arguments
/// * `index` — the CDS index instrument
/// * `default_curve` — the flat or bootstrapped default probability curve
/// * `yield_curve` — risk-free discount curve
/// * `valuation_date` — pricing date
pub fn price_cds_index(
    index: &CdsIndex,
    default_curve: &Arc<dyn DefaultProbabilityTermStructure>,
    yield_curve: &Arc<dyn YieldTermStructure>,
    valuation_date: Date,
) -> CdsIndexResult {
    let dc = yield_curve.day_counter();
    let notional = index.notional;
    let recovery = index.recovery_rate;
    let coupon = index.coupon;

    // === Premium leg ===
    // PV of premium payments: Σ Δt_i × S(t_i) × D(t_i)
    let mut rpv01 = 0.0;
    let mut accrued = 0.0;

    for period in &index.schedule {
        let t_start = dc.year_fraction(valuation_date, period.accrual_start);
        let t_end = dc.year_fraction(valuation_date, period.accrual_end);
        let t_pay = dc.year_fraction(valuation_date, period.payment_date);

        if t_pay <= 0.0 {
            // Check if we're in an accrual period that has started but not paid
            if t_start < 0.0 && t_end > 0.0 {
                // Pro-rata accrued from start to valuation
                let accrual_frac = (-t_start) / (t_end - t_start) * period.accrual_fraction;
                accrued = accrual_frac * coupon * notional;
            }
            continue;
        }

        // If valuation is within this period, compute accrued
        if t_start <= 0.0 && t_end > 0.0 {
            let accrual_frac = (-t_start) / (t_end - t_start) * period.accrual_fraction;
            accrued = accrual_frac * coupon * notional;
        }

        let s_end = default_curve.survival_probability(t_end.max(0.0));
        let d_pay = yield_curve.discount_t(t_pay.max(0.0));

        rpv01 += period.accrual_fraction * s_end * d_pay;
    }

    // === Protection leg ===
    // PV = (1 - R) × ∫₀ᵀ D(t) × dDefault(t)
    // Approximate with step-wise integration over premium periods
    let mut protection_leg_pv = 0.0;
    let num_steps = 100;

    let t_max = dc.year_fraction(valuation_date, index.maturity);
    if t_max > 0.0 {
        let dt = t_max / num_steps as f64;
        let mut prev_survival = default_curve.survival_probability(0.0);

        for step in 1..=num_steps {
            let t = dt * step as f64;
            let curr_survival = default_curve.survival_probability(t);
            let mid_t = t - 0.5 * dt;
            let df = yield_curve.discount_t(mid_t);

            // Probability of default in [t-dt, t]: S(t-dt) - S(t)
            let default_prob = prev_survival - curr_survival;
            protection_leg_pv += (1.0 - recovery) * default_prob * df;

            prev_survival = curr_survival;
        }
    }

    // Scale to notional
    let premium_leg_pv = rpv01 * coupon * notional;

    let sign = match index.side {
        CdsProtectionSide::Buyer => 1.0,
        CdsProtectionSide::Seller => -1.0,
    };

    let dirty_npv = sign * (protection_leg_pv * notional - premium_leg_pv);
    let clean_npv = dirty_npv + sign * accrued;

    // Fair spread: protection_leg / rpv01
    let fair_spread = if rpv01.abs() > 1e-15 {
        protection_leg_pv / rpv01
    } else {
        0.0
    };

    // Upfront: (coupon - fair_spread) × rpv01 × notional
    let upfront = (coupon - fair_spread) * rpv01 * notional;
    let points_upfront = if notional.abs() > 1e-15 {
        upfront / notional
    } else {
        0.0
    };

    CdsIndexResult {
        clean_npv,
        dirty_npv,
        protection_leg_pv: protection_leg_pv * notional,
        rpv01,
        accrued,
        fair_spread,
        upfront,
        points_upfront,
    }
}

/// Convert CDS index upfront to running spread.
///
/// Given an upfront amount and the risky annuity, compute the equivalent
/// running spread: running_spread = coupon + upfront / (rpv01 × notional).
pub fn cds_index_upfront_to_spread(
    coupon: f64,
    upfront: f64,
    rpv01: f64,
    notional: f64,
) -> f64 {
    if rpv01.abs() < 1e-15 || notional.abs() < 1e-15 {
        return coupon;
    }
    coupon - upfront / (rpv01 * notional)
}

/// Convert CDS index running spread to upfront.
///
/// upfront = (coupon - spread) × rpv01 × notional.
pub fn cds_index_spread_to_upfront(
    coupon: f64,
    spread: f64,
    rpv01: f64,
    notional: f64,
) -> f64 {
    (coupon - spread) * rpv01 * notional
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_instruments::cds_index::{CdsIndexFamily, CdsIndex};
    use ql_instruments::credit_default_swap::CdsPremiumPeriod;
    use ql_termstructures::{FlatForward, FlatHazardRate};
    use ql_time::{DayCounter, Month};

    fn make_test_index(settle: Date) -> CdsIndex {
        let schedule: Vec<CdsPremiumPeriod> = (0..20)
            .map(|i| {
                let year = settle.year() + i / 4;
                let months = [Month::March, Month::June, Month::September, Month::December];
                let start_m = months[(i % 4) as usize];
                let end_m = months[((i + 1) % 4) as usize];
                let end_y = if (i + 1) % 4 == 0 { year + 1 } else { year };
                CdsPremiumPeriod {
                    accrual_start: Date::from_ymd(year, start_m, 20),
                    accrual_end: Date::from_ymd(end_y, end_m, 20),
                    payment_date: Date::from_ymd(end_y, end_m, 20),
                    accrual_fraction: 0.25,
                }
            })
            .collect();

        CdsIndex::new(
            CdsIndexFamily::CdxNaIg,
            42, 1,
            CdsProtectionSide::Buyer,
            10_000_000.0,
            125,
            0.01,  // 100 bps
            0.0,
            Date::from_ymd(settle.year() + 5, Month::March, 20),
            0.40,
            schedule,
        )
    }

    #[test]
    fn cds_index_pricing_basic() {
        let settle = Date::from_ymd(2025, Month::January, 15);
        let index = make_test_index(settle);

        let hazard_rate = 0.01 / (1.0 - 0.4); // ~167 bps hazard for 100 bps spread
        let default_curve: Arc<dyn DefaultProbabilityTermStructure> = Arc::new(
            FlatHazardRate::new(settle, hazard_rate, DayCounter::Actual365Fixed),
        );
        let yield_curve: Arc<dyn YieldTermStructure> = Arc::new(
            FlatForward::new(settle, 0.03, DayCounter::Actual365Fixed),
        );

        let result = price_cds_index(&index, &default_curve, &yield_curve, settle);

        assert!(result.rpv01 > 0.0, "RPV01 should be positive");
        assert!(result.protection_leg_pv > 0.0, "Protection leg should be positive");
        // Fair spread should be close to 100 bps
        assert!(result.fair_spread > 0.005 && result.fair_spread < 0.03,
            "Fair spread {} should be near 0.01", result.fair_spread);
    }

    #[test]
    fn cds_index_at_fair_spread_npv_near_zero() {
        let settle = Date::from_ymd(2025, Month::January, 15);
        let mut index = make_test_index(settle);

        let hazard_rate = 0.02;
        let default_curve: Arc<dyn DefaultProbabilityTermStructure> = Arc::new(
            FlatHazardRate::new(settle, hazard_rate, DayCounter::Actual365Fixed),
        );
        let yield_curve: Arc<dyn YieldTermStructure> = Arc::new(
            FlatForward::new(settle, 0.03, DayCounter::Actual365Fixed),
        );

        let result1 = price_cds_index(&index, &default_curve, &yield_curve, settle);

        // Re-price at fair spread
        index.coupon = result1.fair_spread;
        let result2 = price_cds_index(&index, &default_curve, &yield_curve, settle);

        assert_abs_diff_eq!(result2.dirty_npv, 0.0, epsilon = 100.0);
    }

    #[test]
    fn upfront_spread_conversion_roundtrip() {
        let coupon = 0.01; // 100 bps
        let spread = 0.012; // 120 bps
        let rpv01 = 4.5;
        let notional = 10_000_000.0;

        let upfront = cds_index_spread_to_upfront(coupon, spread, rpv01, notional);
        let back = cds_index_upfront_to_spread(coupon, upfront, rpv01, notional);

        assert_abs_diff_eq!(back, spread, epsilon = 1e-10);
    }

    #[test]
    fn cds_index_seller_npv_sign() {
        let settle = Date::from_ymd(2025, Month::January, 15);
        let mut index_buyer = make_test_index(settle);
        let mut index_seller = make_test_index(settle);
        index_seller.side = CdsProtectionSide::Seller;

        let hazard_rate = 0.015;
        let default_curve: Arc<dyn DefaultProbabilityTermStructure> = Arc::new(
            FlatHazardRate::new(settle, hazard_rate, DayCounter::Actual365Fixed),
        );
        let yield_curve: Arc<dyn YieldTermStructure> = Arc::new(
            FlatForward::new(settle, 0.03, DayCounter::Actual365Fixed),
        );

        let buyer = price_cds_index(&index_buyer, &default_curve, &yield_curve, settle);
        let seller = price_cds_index(&index_seller, &default_curve, &yield_curve, settle);

        // Buyer and seller NPVs should have opposite signs
        assert!(buyer.dirty_npv * seller.dirty_npv <= 0.0 || buyer.dirty_npv.abs() < 1.0,
            "Buyer ({}) and seller ({}) NPVs should be opposite", buyer.dirty_npv, seller.dirty_npv);
    }
}
