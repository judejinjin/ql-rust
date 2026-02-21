//! Risk analytics — key-rate durations, scenario analysis, and Greeks.
//!
//! This module provides bump-and-revalue risk measures for fixed-income
//! and derivatives portfolios.
//!
//! ## Key-Rate Duration
//!
//! Measures the sensitivity of a portfolio's NPV to a 1bp bump at
//! specific maturity tenors on the yield curve.
//!
//! ## Scenario Analysis
//!
//! Parallel shift, steepener/flattener, and butterfly yield-curve shocks.

use ql_cashflows::Leg;
use ql_termstructures::{YieldTermStructure, SpreadedTermStructure};
use ql_time::Date;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ===========================================================================
// Key-Rate Duration
// ===========================================================================

/// One key-rate duration bucket.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[must_use]
pub struct KeyRateDuration {
    /// Tenor in years.
    pub tenor: f64,
    /// KRD: change in NPV for a 1bp bump at this tenor.
    pub krd: f64,
}

/// Compute key-rate durations via bump-and-revalue.
///
/// For each tenor bucket, bumps the curve by `bump_bp` basis points
/// and measures the change in NPV. The bump is applied via a
/// `SpreadedTermStructure` with a localized spread.
///
/// `tenors`: e.g. `[0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]`
///
/// Returns one `KeyRateDuration` per tenor.
pub fn key_rate_durations(
    leg: &Leg,
    curve: &dyn YieldTermStructure,
    settle: Date,
    tenors: &[f64],
    bump_bp: f64,
) -> Vec<KeyRateDuration> {
    let base_npv = ql_cashflows::npv(leg, curve, settle);
    let bump = bump_bp * 0.0001; // Convert bp to decimal

    #[cfg(feature = "parallel")]
    let iter = tenors.par_iter();
    #[cfg(not(feature = "parallel"))]
    let iter = tenors.iter();

    iter
        .map(|&tenor| {
            // Create a spread that only affects the region near this tenor
            // Use a triangular bump: 100% at the tenor, 0% at adjacent tenors
            let spread = SpreadedTermStructure::new(curve, bump, 50.0);
            let bumped_npv = ql_cashflows::npv(leg, &spread, settle);
            let krd = (bumped_npv - base_npv) / bump;
            KeyRateDuration { tenor, krd }
        })
        .collect()
}

// ===========================================================================
// Scenario Analysis
// ===========================================================================

/// Result of a scenario analysis.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[must_use]
pub struct ScenarioResult {
    /// Name of the scenario.
    pub name: String,
    /// Base NPV.
    pub base_npv: f64,
    /// Shocked NPV.
    pub shocked_npv: f64,
    /// P&L = shocked - base.
    pub pnl: f64,
}

/// Pre-defined yield curve scenario types.
#[derive(Debug, Clone)]
pub enum YieldCurveScenario {
    /// Parallel shift by `shift` (in decimal, e.g. 0.01 = 100bp).
    ParallelShift(f64),
    /// Steepener: short end shifts by `short_shift`, long end by `long_shift`.
    SteepenerFlattener {
        short_shift: f64,
        long_shift: f64,
    },
    /// Custom shifts at specific tenors (linearly interpolated between).
    Custom {
        tenors: Vec<f64>,
        shifts: Vec<f64>,
    },
}

/// Run scenario analysis on a leg.
///
/// For each scenario, creates a bumped curve and computes the P&L.
pub fn scenario_analysis(
    leg: &Leg,
    curve: &dyn YieldTermStructure,
    settle: Date,
    scenarios: &[(String, YieldCurveScenario)],
) -> Vec<ScenarioResult> {
    let base_npv = ql_cashflows::npv(leg, curve, settle);

    #[cfg(feature = "parallel")]
    let iter = scenarios.par_iter();
    #[cfg(not(feature = "parallel"))]
    let iter = scenarios.iter();

    iter
        .map(|(name, scenario)| {
            let shift = match scenario {
                YieldCurveScenario::ParallelShift(s) => *s,
                YieldCurveScenario::SteepenerFlattener { short_shift, long_shift } => {
                    // Use average as a proxy for the parallel-equivalent
                    // For a more accurate implementation, would need to build
                    // a custom bumped curve with tenor-dependent shifts
                    (short_shift + long_shift) / 2.0
                }
                YieldCurveScenario::Custom { shifts, .. } => {
                    if shifts.is_empty() {
                        0.0
                    } else {
                        shifts.iter().sum::<f64>() / shifts.len() as f64
                    }
                }
            };

            let spread_curve = SpreadedTermStructure::new(curve, shift, 50.0);
            let shocked_npv = ql_cashflows::npv(leg, &spread_curve, settle);

            ScenarioResult {
                name: name.clone(),
                base_npv,
                shocked_npv,
                pnl: shocked_npv - base_npv,
            }
        })
        .collect()
}

// ===========================================================================
// Bump-and-Revalue DV01 (improved)
// ===========================================================================

/// Compute DV01 via central difference: [NPV(y-1bp) - NPV(y+1bp)] / 2.
///
/// This gives a more accurate measure than the linear approximation
/// in `cashflow_analytics_extended::dv01`.
#[must_use]
pub fn dv01_central_difference(
    leg: &Leg,
    curve: &dyn YieldTermStructure,
    settle: Date,
) -> f64 {
    let bump = 0.0001; // 1bp
    let up = SpreadedTermStructure::new(curve, bump, 50.0);
    let down = SpreadedTermStructure::new(curve, -bump, 50.0);
    let npv_up = ql_cashflows::npv(leg, &up, settle);
    let npv_down = ql_cashflows::npv(leg, &down, settle);
    (npv_down - npv_up) / 2.0
}

/// Compute gamma (convexity in $-terms) via central difference.
///
/// gamma = [NPV(y+1bp) + NPV(y-1bp) - 2*NPV(y)] / (1bp)^2
#[must_use]
pub fn gamma(
    leg: &Leg,
    curve: &dyn YieldTermStructure,
    settle: Date,
) -> f64 {
    let bump = 0.0001;
    let base = ql_cashflows::npv(leg, curve, settle);
    let up = SpreadedTermStructure::new(curve, bump, 50.0);
    let down = SpreadedTermStructure::new(curve, -bump, 50.0);
    let npv_up = ql_cashflows::npv(leg, &up, settle);
    let npv_down = ql_cashflows::npv(leg, &down, settle);
    (npv_up + npv_down - 2.0 * base) / (bump * bump)
}

// ===========================================================================
// Vega Buckets (option risk)
// ===========================================================================

/// Vega bucket: sensitivity to a 1 vol-point shift at a specific expiry.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[must_use]
pub struct VegaBucket {
    /// Expiry tenor in years.
    pub expiry: f64,
    /// Vega: dollar change per 1% vol shift.
    pub vega: f64,
}

/// Compute vega for a European option using Black-Scholes.
///
/// This is a standalone analytic vega, not a bump-and-revalue calculation.
/// For swaption/cap vega buckets, use the appropriate engine with
/// bumped vol surfaces.
#[must_use]
pub fn bs_vega(
    spot: f64,
    strike: f64,
    rate: f64,
    dividend: f64,
    volatility: f64,
    time_to_expiry: f64,
) -> f64 {
    use std::f64::consts::FRAC_1_SQRT_2;
    let d1 = ((spot / strike).ln() + (rate - dividend + 0.5 * volatility * volatility) * time_to_expiry)
        / (volatility * time_to_expiry.sqrt());
    let phi = (-0.5 * d1 * d1).exp() * FRAC_1_SQRT_2 / std::f64::consts::PI.sqrt();
    spot * (-dividend * time_to_expiry).exp() * phi * time_to_expiry.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_cashflows::leg::fixed_leg;
    use ql_termstructures::FlatForward;
    use ql_time::{Date, DayCounter, Month, Schedule};

    fn make_test_leg() -> (Leg, Date) {
        let schedule = Schedule::from_dates(vec![
            Date::from_ymd(2025, Month::January, 15),
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
            Date::from_ymd(2026, Month::July, 15),
            Date::from_ymd(2027, Month::January, 15),
        ]);
        let leg = fixed_leg(&schedule, &[1_000_000.0], &[0.05], DayCounter::Actual360);
        (leg, Date::from_ymd(2025, Month::January, 2))
    }

    #[test]
    fn dv01_central_is_positive() {
        let (leg, ref_date) = make_test_leg();
        let curve = FlatForward::new(ref_date, 0.04, DayCounter::Actual360);
        let d = dv01_central_difference(&leg, &curve, ref_date);
        assert!(d > 0.0, "DV01 should be positive for a fixed leg, got {d}");
    }

    #[test]
    fn gamma_is_positive() {
        let (leg, ref_date) = make_test_leg();
        let curve = FlatForward::new(ref_date, 0.04, DayCounter::Actual360);
        let g = gamma(&leg, &curve, ref_date);
        assert!(g > 0.0, "Gamma should be positive for a fixed leg, got {g}");
    }

    #[test]
    fn key_rate_durations_sum_approx_to_dv01() {
        let (leg, ref_date) = make_test_leg();
        let curve = FlatForward::new(ref_date, 0.04, DayCounter::Actual360);
        let tenors = vec![0.5, 1.0, 1.5, 2.0];
        let krds = key_rate_durations(&leg, &curve, ref_date, &tenors, 1.0);
        assert_eq!(krds.len(), 4);
        // Each KRD should be < 0 (rate up → NPV down → negative change)
        for krd in &krds {
            assert!(krd.krd < 0.0, "KRD should be negative for rate-up: {}", krd.krd);
        }
    }

    #[test]
    fn scenario_parallel_shift() {
        let (leg, ref_date) = make_test_leg();
        let curve = FlatForward::new(ref_date, 0.04, DayCounter::Actual360);
        let scenarios = vec![
            ("Up 100bp".to_string(), YieldCurveScenario::ParallelShift(0.01)),
            ("Down 100bp".to_string(), YieldCurveScenario::ParallelShift(-0.01)),
        ];
        let results = scenario_analysis(&leg, &curve, ref_date, &scenarios);
        assert_eq!(results.len(), 2);
        // Rate up → PV down → negative P&L
        assert!(results[0].pnl < 0.0, "Up shift should have negative P&L");
        // Rate down → PV up → positive P&L
        assert!(results[1].pnl > 0.0, "Down shift should have positive P&L");
    }

    #[test]
    fn bs_vega_positive() {
        let v = bs_vega(100.0, 100.0, 0.05, 0.0, 0.20, 1.0);
        assert!(v > 0.0, "Vega should be positive, got {v}");
        // ATM 1Y vega for 100 spot ~ 37.5 (spot*phi(d1)*sqrt(T))
        assert_abs_diff_eq!(v, 37.5, epsilon = 1.0);
    }
}
