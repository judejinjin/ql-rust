//! Mid-point CDS pricing engine.
//!
//! Prices a CDS by evaluating protection and premium legs using
//! mid-point integration of hazard rates and discount factors.

use ql_instruments::credit_default_swap::CreditDefaultSwap;
use std::sync::Arc;
use ql_termstructures::default_term_structure::DefaultProbabilityTermStructure;
use ql_termstructures::yield_term_structure::YieldTermStructure;

/// Result from CDS pricing.
#[derive(Debug, Clone)]
pub struct CdsResult {
    /// Net present value from the protection buyer's perspective.
    pub npv: f64,
    /// Fair (par) spread that makes NPV = 0.
    pub fair_spread: f64,
    /// Premium leg PV (unsigned).
    pub premium_leg_pv: f64,
    /// Protection leg PV (unsigned).
    pub protection_leg_pv: f64,
}

/// Price a CDS using the mid-point engine.
///
/// # Parameters
/// - `cds`: the CDS instrument
/// - `default_curve`: default probability term structure
/// - `yield_curve`: risk-free discount curve
/// - `valuation_time`: time from reference date to valuation (typically 0)
#[allow(clippy::too_many_arguments)]
pub fn midpoint_cds_engine(
    cds: &CreditDefaultSwap,
    default_curve: &Arc<dyn DefaultProbabilityTermStructure>,
    yield_curve: &Arc<dyn YieldTermStructure>,
    _valuation_time: f64,
) -> CdsResult {
    let recovery = cds.recovery_rate;
    let notional = cds.notional;
    let spread = cds.spread;
    let ref_date = yield_curve.reference_date();
    let dc = yield_curve.day_counter();

    // Premium leg: Σ spread * notional * Δt_k * S(t_k) * D(t_k)
    let mut premium_leg_pv = 0.0;
    for period in &cds.schedule {
        let t = dc.year_fraction(ref_date, period.payment_date);
        if t <= 0.0 {
            continue;
        }
        let s = default_curve.survival_probability(t);
        let d = yield_curve.discount_t(t);
        premium_leg_pv += period.accrual_fraction * s * d;
    }
    premium_leg_pv *= spread * notional;

    // Protection leg: (1-R) * notional * Σ [S(t_{k-1}) - S(t_k)] * D(t_mid)
    // where t_mid = (t_{k-1} + t_k) / 2
    let mut protection_leg_pv = 0.0;
    let mut prev_t = 0.0;
    for period in &cds.schedule {
        let t = dc.year_fraction(ref_date, period.payment_date);
        if t <= 0.0 {
            prev_t = t;
            continue;
        }
        let t_mid = 0.5 * (prev_t + t);
        let s_prev = default_curve.survival_probability(prev_t.max(0.0));
        let s_curr = default_curve.survival_probability(t);
        let d_mid = yield_curve.discount_t(t_mid.max(0.0));
        protection_leg_pv += (s_prev - s_curr) * d_mid;
        prev_t = t;
    }
    protection_leg_pv *= (1.0 - recovery) * notional;

    // NPV from buyer's perspective: protection - premium
    let npv = cds.side.sign() * (protection_leg_pv - premium_leg_pv);

    // Fair spread: the spread at which NPV = 0
    // protection_pv = fair_spread * risky_annuity
    let risky_annuity: f64 = cds
        .schedule
        .iter()
        .map(|p| {
            let t = dc.year_fraction(ref_date, p.payment_date);
            if t <= 0.0 {
                0.0
            } else {
                p.accrual_fraction
                    * default_curve.survival_probability(t)
                    * yield_curve.discount_t(t)
            }
        })
        .sum();

    let fair_spread = if risky_annuity > 0.0 {
        protection_leg_pv / (notional * risky_annuity)
    } else {
        0.0
    };

    CdsResult {
        npv,
        fair_spread,
        premium_leg_pv,
        protection_leg_pv,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_instruments::credit_default_swap::{CdsPremiumPeriod, CdsProtectionSide};
    use ql_termstructures::{FlatForward, default_term_structure::FlatHazardRate};
    use ql_time::{Date, DayCounter, Month};

    fn make_test_cds(spread: f64) -> CreditDefaultSwap {
        let schedule: Vec<CdsPremiumPeriod> = (1..=20)
            .map(|i| {
                let year_offset = (i - 1) / 4;
                let quarter = (i - 1) % 4;
                let start_month = [Month::March, Month::June, Month::September, Month::December][quarter];
                let end_quarter = i % 4;
                let end_month = [Month::March, Month::June, Month::September, Month::December][end_quarter];
                let end_year_offset = i / 4;
                CdsPremiumPeriod {
                    accrual_start: Date::from_ymd(2025 + year_offset as i32, start_month, 20),
                    accrual_end: Date::from_ymd(2025 + end_year_offset as i32, end_month, 20),
                    payment_date: Date::from_ymd(2025 + end_year_offset as i32, end_month, 20),
                    accrual_fraction: 0.25,
                }
            })
            .collect();

        CreditDefaultSwap::new(
            CdsProtectionSide::Buyer,
            10_000_000.0,
            spread,
            Date::from_ymd(2030, Month::March, 20),
            0.4,
            schedule,
        )
    }

    #[test]
    fn cds_fair_spread_near_input_for_flat_curve() {
        let ref_date = Date::from_ymd(2025, Month::March, 20);
        let dc = DayCounter::Actual365Fixed;

        let spread = 0.01; // 100bp
        let cds = make_test_cds(spread);

        let default_curve = Arc::new(FlatHazardRate::from_spread(ref_date, spread, 0.4, dc));
        let yield_curve = Arc::new(FlatForward::new(ref_date, 0.03, dc));

        let result = midpoint_cds_engine(
            &cds,
            &(default_curve as Arc<dyn DefaultProbabilityTermStructure>),
            &(yield_curve as Arc<dyn YieldTermStructure>),
            0.0,
        );

        // Fair spread should be close to 100bp
        assert_abs_diff_eq!(result.fair_spread, spread, epsilon = 5e-4);
    }

    #[test]
    fn cds_npv_near_zero_at_par() {
        let ref_date = Date::from_ymd(2025, Month::March, 20);
        let dc = DayCounter::Actual365Fixed;

        let spread = 0.005;
        let cds = make_test_cds(spread);

        let default_curve = Arc::new(FlatHazardRate::from_spread(ref_date, spread, 0.4, dc));
        let yield_curve = Arc::new(FlatForward::new(ref_date, 0.03, dc));

        let result = midpoint_cds_engine(
            &cds,
            &(default_curve as Arc<dyn DefaultProbabilityTermStructure>),
            &(yield_curve as Arc<dyn YieldTermStructure>),
            0.0,
        );

        // NPV should be near zero when spread = fair spread
        assert_abs_diff_eq!(result.npv, 0.0, epsilon = 5000.0); // within 5000 on 10M notional
    }

    #[test]
    fn cds_protection_buyer_positive_when_spread_below_fair() {
        let ref_date = Date::from_ymd(2025, Month::March, 20);
        let dc = DayCounter::Actual365Fixed;

        // Market spread is 200bp but CDS has 100bp coupon
        let cds = make_test_cds(0.01);
        let default_curve = Arc::new(FlatHazardRate::from_spread(ref_date, 0.02, 0.4, dc));
        let yield_curve = Arc::new(FlatForward::new(ref_date, 0.03, dc));

        let result = midpoint_cds_engine(
            &cds,
            &(default_curve as Arc<dyn DefaultProbabilityTermStructure>),
            &(yield_curve as Arc<dyn YieldTermStructure>),
            0.0,
        );

        // Buyer benefits when default prob is higher than priced
        assert!(result.npv > 0.0, "Buyer NPV should be positive");
    }

    #[test]
    fn cds_protection_and_premium_legs_positive() {
        let ref_date = Date::from_ymd(2025, Month::March, 20);
        let dc = DayCounter::Actual365Fixed;

        let cds = make_test_cds(0.01);
        let default_curve = Arc::new(FlatHazardRate::from_spread(ref_date, 0.01, 0.4, dc));
        let yield_curve = Arc::new(FlatForward::new(ref_date, 0.05, dc));

        let result = midpoint_cds_engine(
            &cds,
            &(default_curve as Arc<dyn DefaultProbabilityTermStructure>),
            &(yield_curve as Arc<dyn YieldTermStructure>),
            0.0,
        );

        assert!(result.premium_leg_pv > 0.0);
        assert!(result.protection_leg_pv > 0.0);
    }
}
