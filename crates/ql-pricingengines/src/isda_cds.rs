//! ISDA Standard CDS pricing engine and analytics.
//!
//! Implements the ISDA standard CDS model with:
//! - Accrual-on-default in the premium leg
//! - Protection/premium leg NPV with continuous default integration
//! - Upfront / points-upfront calculation (post-Big Bang convention)
//! - RPV01 (risky PV01) and CS01 (credit spread sensitivity)
//! - Clean vs. dirty price distinction
//! - IMM CDS schedule generation (Mar/Jun/Sep/Dec 20th)
//!
//! ## References
//!
//! - ISDA CDS Standard Model (2009 Big Bang Protocol)
//! - O'Kane, D. "Modelling Single-name and Multi-name Credit Derivatives"

use ql_instruments::credit_default_swap::{
    CdsPremiumPeriod, CdsProtectionSide, CreditDefaultSwap,
};
use ql_termstructures::default_term_structure::DefaultProbabilityTermStructure;
use ql_termstructures::yield_term_structure::YieldTermStructure;
use ql_time::{Date, DayCounter, Month};
use std::sync::Arc;

// ===========================================================================
//  ISDA CDS Result
// ===========================================================================

/// Full result from ISDA CDS pricing.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IsdaCdsResult {
    /// Dirty NPV (includes accrued).
    pub dirty_npv: f64,
    /// Clean NPV (dirty − accrued premium).
    pub clean_npv: f64,
    /// Fair (par) spread.
    pub fair_spread: f64,
    /// Premium leg PV (unsigned, dirty).
    pub premium_leg_bps: f64,
    /// Protection leg PV (unsigned).
    pub protection_leg_pv: f64,
    /// Accrual-on-default PV (unsigned).
    pub accrual_on_default_pv: f64,
    /// Risky annuity / RPV01 (per basis point of notional).
    pub rpv01: f64,
    /// Accrued premium at valuation date (unsigned).
    pub accrued_premium: f64,
}

// ===========================================================================
//  ISDA Standard Model Engine
// ===========================================================================

/// Price a CDS using the ISDA standard model with accrual-on-default.
///
/// This engine integrates the protection leg using sub-period steps (monthly)
/// for improved accuracy compared to the midpoint engine, and includes the
/// accrued premium paid on default within each coupon period.
///
/// # Parameters
/// - `cds`: CDS instrument
/// - `default_curve`: survival/default probability curve
/// - `yield_curve`: risk-free discount curve
/// - `valuation_date`: pricing date (if `None`, uses yield curve reference date)
/// - `steps_per_period`: sub-steps per coupon period for integration (default: 4)
pub fn isda_cds_engine(
    cds: &CreditDefaultSwap,
    default_curve: &Arc<dyn DefaultProbabilityTermStructure>,
    yield_curve: &Arc<dyn YieldTermStructure>,
    valuation_date: Option<Date>,
    steps_per_period: Option<usize>,
) -> IsdaCdsResult {
    let ref_date = valuation_date.unwrap_or_else(|| yield_curve.reference_date());
    let dc = yield_curve.day_counter();
    let recovery = cds.recovery_rate;
    let notional = cds.notional;
    let spread = cds.spread;
    let steps = steps_per_period.unwrap_or(4);

    // --- Premium leg (including accrual on default) ---
    let mut premium_leg_pv = 0.0;
    let mut accrual_on_default_pv = 0.0;
    let mut accrued_at_valuation = 0.0;

    for period in &cds.schedule {
        let t_start = dc.year_fraction(ref_date, period.accrual_start);
        let t_end = dc.year_fraction(ref_date, period.accrual_end);
        let t_pay = dc.year_fraction(ref_date, period.payment_date);

        if t_pay <= 0.0 {
            // Period already paid — but may contribute to accrued
            continue;
        }

        // Standard premium: Δt × S(t_end) × D(t_pay)
        let s_end = default_curve.survival_probability(t_end.max(0.0));
        let d_pay = yield_curve.discount_t(t_pay.max(0.0));
        premium_leg_pv += period.accrual_fraction * s_end * d_pay;

        // Accrual on default: approximate by sub-period integration
        // ∫_{t_start}^{t_end} accrual(u) × dDefault(u) × D(u)
        let eff_start = t_start.max(0.0);
        if t_end > eff_start {
            let dt = (t_end - eff_start) / steps as f64;
            let mut prev_s = default_curve.survival_probability(eff_start);
            for k in 1..=steps {
                let u = eff_start + k as f64 * dt;
                let curr_s = default_curve.survival_probability(u);
                let d_u = yield_curve.discount_t(u);
                // Fraction of period accrued at midpoint
                let accrual_at_u =
                    period.accrual_fraction * (u - eff_start) / (t_end - eff_start);
                // Default probability in this sub-period
                let dp = prev_s - curr_s;
                accrual_on_default_pv += accrual_at_u * dp * d_u;
                prev_s = curr_s;
            }
        }

        // Compute accrued premium at valuation date (for clean price)
        let val_t = 0.0; // valuation is at ref_date
        if t_start <= val_t && val_t < t_end {
            let frac_accrued = if (t_end - t_start).abs() > 1e-10 {
                (val_t - t_start) / (t_end - t_start)
            } else {
                0.0
            };
            accrued_at_valuation = spread * notional * period.accrual_fraction * frac_accrued;
        }
    }

    // Full premium = standard premium + accrual on default (both scaled by spread × notional)
    let total_premium_bps = premium_leg_pv + accrual_on_default_pv;
    let premium_value = spread * notional * total_premium_bps;

    // --- Protection leg ---
    let mut protection_leg_pv = 0.0;
    let mut prev_t = 0.0_f64;
    for period in &cds.schedule {
        let t_end = dc.year_fraction(ref_date, period.accrual_end);
        if t_end <= 0.0 {
            prev_t = t_end;
            continue;
        }
        let eff_start = prev_t.max(0.0);
        let dt = (t_end - eff_start) / steps as f64;
        let mut prev_s = default_curve.survival_probability(eff_start);
        for k in 1..=steps {
            let u = eff_start + k as f64 * dt;
            let curr_s = default_curve.survival_probability(u);
            let d_mid = yield_curve.discount_t((u - dt * 0.5).max(0.0));
            protection_leg_pv += (prev_s - curr_s) * d_mid;
            prev_s = curr_s;
        }
        prev_t = t_end;
    }
    protection_leg_pv *= (1.0 - recovery) * notional;

    // --- NPV ---
    let dirty_npv = cds.side.sign() * (protection_leg_pv - premium_value);
    let clean_npv = dirty_npv + cds.side.sign() * accrued_at_valuation;

    // --- Fair spread ---
    let rpv01 = total_premium_bps;
    let fair_spread = if rpv01 > 0.0 {
        protection_leg_pv / (notional * rpv01)
    } else {
        0.0
    };

    IsdaCdsResult {
        dirty_npv,
        clean_npv,
        fair_spread,
        premium_leg_bps: premium_leg_pv,
        protection_leg_pv,
        accrual_on_default_pv: spread * notional * accrual_on_default_pv,
        rpv01,
        accrued_premium: accrued_at_valuation,
    }
}

// ===========================================================================
//  Upfront / Points Upfront
// ===========================================================================

/// Compute the upfront amount for a CDS trading with a standard running coupon.
///
/// Post-Big Bang, CDS trades with a fixed coupon (typically 100bp or 500bp)
/// and an upfront payment. The upfront = (par_spread − running) × RPV01 × N.
///
/// Returns the upfront payment (positive means buyer pays upfront).
pub fn cds_upfront(
    cds: &CreditDefaultSwap,
    default_curve: &Arc<dyn DefaultProbabilityTermStructure>,
    yield_curve: &Arc<dyn YieldTermStructure>,
) -> f64 {
    let result = isda_cds_engine(cds, default_curve, yield_curve, None, None);
    // Upfront = protection_leg - premium_leg_at_running_coupon
    // Or equivalently: (fair_spread - running) * RPV01 * N
    (result.fair_spread - cds.spread) * result.rpv01 * cds.notional
}

/// Points upfront (as a fraction of notional).
pub fn cds_points_upfront(
    cds: &CreditDefaultSwap,
    default_curve: &Arc<dyn DefaultProbabilityTermStructure>,
    yield_curve: &Arc<dyn YieldTermStructure>,
) -> f64 {
    cds_upfront(cds, default_curve, yield_curve) / cds.notional
}

// ===========================================================================
//  CS01 — Credit Spread Sensitivity
// ===========================================================================

/// CS01: change in CDS NPV for a 1bp parallel shift in the hazard rate curve.
///
/// Uses central finite difference: CS01 = [NPV(h-1bp) − NPV(h+1bp)] / 2.
///
/// Requires a flat hazard rate curve for simplicity. For piecewise curves,
/// shifts should be applied to each segment.
pub fn cds_cs01(
    cds: &CreditDefaultSwap,
    default_curve: &Arc<dyn DefaultProbabilityTermStructure>,
    yield_curve: &Arc<dyn YieldTermStructure>,
) -> f64 {
    use ql_termstructures::default_term_structure::FlatHazardRate;

    // Extract hazard rate from the curve at 1Y
    let h = default_curve.hazard_rate(1.0);
    let bump = 1e-4 / (1.0 - cds.recovery_rate); // 1bp in spread terms
    let ref_date = yield_curve.reference_date();
    let dc = yield_curve.day_counter();

    let curve_up = Arc::new(FlatHazardRate::new(ref_date, h + bump, dc));
    let curve_dn = Arc::new(FlatHazardRate::new(ref_date, (h - bump).max(0.0), dc));

    let npv_up = isda_cds_engine(
        cds,
        &(curve_up as Arc<dyn DefaultProbabilityTermStructure>),
        yield_curve,
        None,
        None,
    )
    .dirty_npv;
    let npv_dn = isda_cds_engine(
        cds,
        &(curve_dn as Arc<dyn DefaultProbabilityTermStructure>),
        yield_curve,
        None,
        None,
    )
    .dirty_npv;

    (npv_dn - npv_up) / 2.0
}

// ===========================================================================
//  IMM CDS Schedule Generation
// ===========================================================================

/// Standard CDS IMM dates: Mar 20, Jun 20, Sep 20, Dec 20.
const IMM_MONTHS: [Month; 4] = [Month::March, Month::June, Month::September, Month::December];

/// Generate an ISDA standard CDS schedule.
///
/// CDS schedules run between IMM dates (20th of Mar/Jun/Sep/Dec).
/// The effective date is typically T+1 business day, and the first coupon
/// date is the next IMM date after the effective date.
///
/// # Parameters
/// - `effective_date`: protection start date
/// - `maturity_date`: last IMM date
/// - `day_counter`: for accrual fractions (typically Act/360)
///
/// # Returns
/// Vector of premium periods, suitable for `CreditDefaultSwap::new()`.
pub fn cds_imm_schedule(
    effective_date: Date,
    maturity_date: Date,
    day_counter: DayCounter,
) -> Vec<CdsPremiumPeriod> {
    // Find the first IMM date strictly after effective_date
    let mut periods = Vec::new();
    let mut current = if is_imm_date(effective_date) {
        next_imm_date(effective_date + 1)
    } else {
        next_imm_date(effective_date)
    };

    // Build from effective_date to maturity
    let mut prev = effective_date;
    while current <= maturity_date {
        let accrual_fraction = day_counter.year_fraction(prev, current);
        periods.push(CdsPremiumPeriod {
            accrual_start: prev,
            accrual_end: current,
            payment_date: current,
            accrual_fraction,
        });
        prev = current;
        current = next_imm_date(current + 1);
    }

    // Handle final stub if maturity doesn't fall exactly on an IMM date
    if prev < maturity_date {
        let accrual_fraction = day_counter.year_fraction(prev, maturity_date);
        periods.push(CdsPremiumPeriod {
            accrual_start: prev,
            accrual_end: maturity_date,
            payment_date: maturity_date,
            accrual_fraction,
        });
    }

    periods
}

/// Check if a date is an IMM CDS date (20th of Mar/Jun/Sep/Dec).
fn is_imm_date(date: Date) -> bool {
    let d = date.day_of_month();
    if d != 20 {
        return false;
    }
    let m = date.month() as u32;
    IMM_MONTHS.iter().any(|&imm_m| imm_m as u32 == m)
}

/// Find the next IMM date on or after the given date.
fn next_imm_date(from: Date) -> Date {
    let (y, m, d) = (from.year(), from.month() as u32, from.day_of_month());

    // Check if we're before the 20th of an IMM month
    for &imm_m in &IMM_MONTHS {
        if imm_m as u32 == m && d <= 20 {
            return Date::from_ymd(y, imm_m, 20);
        }
    }

    // Find next IMM month after current month
    for &imm_m in &IMM_MONTHS {
        if imm_m as u32 > m {
            return Date::from_ymd(y, imm_m, 20);
        }
    }

    // Wrap to next year
    Date::from_ymd(y + 1, Month::March, 20)
}

/// Build a standard CDS using ISDA conventions.
///
/// # Parameters
/// - `side`: buyer or seller
/// - `notional`: face amount
/// - `running_spread`: coupon (e.g. 0.01 for 100bp, 0.05 for 500bp)
/// - `effective`: protection start date
/// - `maturity`: CDS maturity (IMM date)
/// - `recovery`: assumed recovery rate
/// - `dc`: day counter for accrual (typically Act/360)
pub fn make_standard_cds(
    side: CdsProtectionSide,
    notional: f64,
    running_spread: f64,
    effective: Date,
    maturity: Date,
    recovery: f64,
    dc: DayCounter,
) -> CreditDefaultSwap {
    let schedule = cds_imm_schedule(effective, maturity, dc);
    CreditDefaultSwap::new(side, notional, running_spread, maturity, recovery, schedule)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_termstructures::default_term_structure::FlatHazardRate;
    use ql_termstructures::{FlatForward, YieldTermStructure};

    fn setup() -> (
        CreditDefaultSwap,
        Arc<dyn DefaultProbabilityTermStructure>,
        Arc<dyn YieldTermStructure>,
    ) {
        let ref_date = Date::from_ymd(2025, Month::March, 20);
        let dc = DayCounter::Actual360;

        let cds = make_standard_cds(
            CdsProtectionSide::Buyer,
            10_000_000.0,
            0.01,
            ref_date,
            Date::from_ymd(2030, Month::March, 20),
            0.4,
            dc,
        );

        let default_curve: Arc<dyn DefaultProbabilityTermStructure> =
            Arc::new(FlatHazardRate::from_spread(ref_date, 0.01, 0.4, DayCounter::Actual365Fixed));
        let yield_curve: Arc<dyn YieldTermStructure> =
            Arc::new(FlatForward::new(ref_date, 0.03, DayCounter::Actual365Fixed));

        (cds, default_curve, yield_curve)
    }

    #[test]
    fn isda_fair_spread_near_input() {
        let (cds, default_curve, yield_curve) = setup();
        let result = isda_cds_engine(&cds, &default_curve, &yield_curve, None, None);
        // Fair spread should be close to 100bp (the input spread)
        assert_abs_diff_eq!(result.fair_spread, 0.01, epsilon = 2e-3);
    }

    #[test]
    fn isda_npv_near_zero_at_par() {
        let (cds, default_curve, yield_curve) = setup();
        let result = isda_cds_engine(&cds, &default_curve, &yield_curve, None, None);
        // At par, NPV should be near zero relative to notional
        let npv_pct = result.dirty_npv.abs() / cds.notional;
        assert!(npv_pct < 0.02, "NPV/notional = {npv_pct} should be < 2%");
    }

    #[test]
    fn isda_accrual_on_default_positive() {
        let (cds, default_curve, yield_curve) = setup();
        let result = isda_cds_engine(&cds, &default_curve, &yield_curve, None, None);
        assert!(
            result.accrual_on_default_pv > 0.0,
            "Accrual on default should be positive"
        );
    }

    #[test]
    fn isda_rpv01_positive() {
        let (cds, default_curve, yield_curve) = setup();
        let result = isda_cds_engine(&cds, &default_curve, &yield_curve, None, None);
        assert!(result.rpv01 > 0.0, "RPV01 must be positive");
    }

    #[test]
    fn cds_upfront_positive_when_credit_wider() {
        let ref_date = Date::from_ymd(2025, Month::March, 20);
        let dc = DayCounter::Actual360;

        // CDS with 100bp running, but market spread is 200bp
        let cds = make_standard_cds(
            CdsProtectionSide::Buyer,
            10_000_000.0,
            0.01,
            ref_date,
            Date::from_ymd(2030, Month::March, 20),
            0.4,
            dc,
        );

        let default_curve: Arc<dyn DefaultProbabilityTermStructure> = Arc::new(
            FlatHazardRate::from_spread(ref_date, 0.02, 0.4, DayCounter::Actual365Fixed),
        );
        let yield_curve: Arc<dyn YieldTermStructure> =
            Arc::new(FlatForward::new(ref_date, 0.03, DayCounter::Actual365Fixed));

        let upfront = cds_upfront(&cds, &default_curve, &yield_curve);
        assert!(upfront > 0.0, "Buyer should receive upfront when market spread > running");
    }

    #[test]
    fn cs01_nonzero_for_buyer() {
        let (cds, default_curve, yield_curve) = setup();
        let cs01 = cds_cs01(&cds, &default_curve, &yield_curve);
        // CS01 should be meaningfully non-zero
        assert!(cs01.abs() > 100.0, "CS01 should be significant, got {cs01}");
    }

    #[test]
    fn imm_schedule_quarterly() {
        let eff = Date::from_ymd(2025, Month::March, 20);
        let mat = Date::from_ymd(2030, Month::March, 20);
        let schedule = cds_imm_schedule(eff, mat, DayCounter::Actual360);
        // 5 years × 4 quarters = 20 periods
        assert_eq!(schedule.len(), 20, "Expected 20 quarterly periods");
        // First period ends on Jun 20
        assert_eq!(schedule[0].accrual_end, Date::from_ymd(2025, Month::June, 20));
        // Last period ends on maturity
        assert_eq!(schedule.last().unwrap().accrual_end, mat);
    }

    #[test]
    fn imm_schedule_non_imm_start() {
        let eff = Date::from_ymd(2025, Month::April, 1);
        let mat = Date::from_ymd(2026, Month::March, 20);
        let schedule = cds_imm_schedule(eff, mat, DayCounter::Actual360);
        // Should have a short first stub + 3 full quarters
        assert!(schedule.len() >= 3);
        // First period: Apr 1 to Jun 20
        assert_eq!(schedule[0].accrual_start, eff);
        assert_eq!(schedule[0].accrual_end, Date::from_ymd(2025, Month::June, 20));
    }
}
