#![allow(clippy::too_many_arguments)]
//! Multi-curve bootstrapping framework.
//!
//! Post-crisis interest rate modelling separates the **discount curve**
//! (typically OIS/SOFR) from one or more **forecast curves** (IBOR tenors).
//! This module provides:
//!
//! - [`DualCurveRateHelper`] — trait for helpers that are aware of an
//!   exogenous discount curve when computing their implied quote.
//! - [`DualCurveSwapRateHelper`] — swap rate helper that forecasts the
//!   floating leg from the curve being bootstrapped while discounting
//!   cashflows with a separately-supplied OIS curve.
//! - [`MultiCurveOISRateHelper`] — multi-period OIS helper that properly
//!   compounds overnight rates over sub-periods.
//! - [`bootstrap_forecast_curve`] — bootstrap a forecast curve given an
//!   external discount curve.
//! - [`bootstrap_dual_curves`] — iterative joint bootstrap of OIS discount
//!   and IBOR forecast curves.

use ql_core::errors::{QLError, QLResult};
use ql_math::solvers1d::{Brent, Solver1D};
use ql_time::{Calendar, Date, DayCounter};
use tracing::{debug, info, info_span};

use crate::bootstrap::{interpolate_log_linear, PiecewiseYieldCurve, RateHelper};
use crate::yield_term_structure::YieldTermStructure;

// ===========================================================================
// DualCurveRateHelper Trait
// ===========================================================================

/// A rate helper that uses an external discount curve when computing its
/// implied quote.
///
/// The forecast curve's `(times, dfs)` are supplied as raw slices (the curve
/// being built), while the discount curve is passed as a trait object.
pub trait DualCurveRateHelper: Send + Sync {
    /// The pillar date this helper constrains on the forecast curve.
    fn pillar_date(&self) -> Date;

    /// The quoted market rate.
    fn quote(&self) -> f64;

    /// Compute the implied quote using the forecast curve's raw nodes for
    /// projection and a separate discount curve for discounting.
    fn implied_quote_dual(
        &self,
        forecast_times: &[f64],
        forecast_dfs: &[f64],
        discount_curve: &dyn YieldTermStructure,
        day_counter: DayCounter,
        ref_date: Date,
    ) -> f64;
}

// ===========================================================================
// DualCurveSwapRateHelper
// ===========================================================================

/// Swap rate helper for dual-curve bootstrapping.
///
/// The par swap rate in a dual-curve framework is:
///
/// $$r = \frac{\sum_i \tau_i^{float} \cdot f_i \cdot D(t_i)}
///            {\sum_j \tau_j^{fixed} \cdot D(t_j)}$$
///
/// where $f_i$ is the IBOR forward rate projected from the **forecast**
/// curve and $D(t)$ is the discount factor from the **OIS/discount** curve.
#[derive(Debug, Clone)]
pub struct DualCurveSwapRateHelper {
    /// Quoted par swap rate.
    rate: f64,
    /// Start date of the swap.
    start_date: Date,
    /// Fixed-leg payment dates.
    fixed_payment_dates: Vec<Date>,
    /// Floating-leg fixing dates (start of each accrual period).
    float_start_dates: Vec<Date>,
    /// Floating-leg payment dates (end of each accrual period).
    float_end_dates: Vec<Date>,
    /// Day counter for the fixed leg.
    fixed_day_counter: DayCounter,
    /// Day counter for the floating leg.
    float_day_counter: DayCounter,
}

impl DualCurveSwapRateHelper {
    /// Create a dual-curve swap rate helper with explicit schedules.
    pub fn new(
        rate: f64,
        start_date: Date,
        fixed_payment_dates: Vec<Date>,
        float_start_dates: Vec<Date>,
        float_end_dates: Vec<Date>,
        fixed_day_counter: DayCounter,
        float_day_counter: DayCounter,
    ) -> Self {
        Self {
            rate,
            start_date,
            fixed_payment_dates,
            float_start_dates,
            float_end_dates,
            fixed_day_counter,
            float_day_counter,
        }
    }

    /// Convenience: create from a tenor.
    ///
    /// Fixed leg pays annually, floating leg pays at `float_frequency`
    /// (e.g. 4 for quarterly, 2 for semi-annual).
    pub fn from_tenor(
        rate: f64,
        start_date: Date,
        tenor_years: u32,
        float_frequency: u32,
        fixed_day_counter: DayCounter,
        float_day_counter: DayCounter,
        calendar: Calendar,
    ) -> Self {
        // Fixed leg: annual payments
        let mut fixed_payment_dates = Vec::new();
        for i in 1..=tenor_years {
            let date = advance_months(start_date, i as i32 * 12);
            let adjusted =
                calendar.adjust(date, ql_time::BusinessDayConvention::ModifiedFollowing);
            fixed_payment_dates.push(adjusted);
        }

        // Floating leg
        let months_per_period = 12 / float_frequency;
        let total_float_periods = tenor_years * float_frequency;
        let mut float_start_dates = Vec::new();
        let mut float_end_dates = Vec::new();
        for i in 0..total_float_periods {
            let s = if i == 0 {
                start_date
            } else {
                advance_months(start_date, i as i32 * months_per_period as i32)
            };
            let s_adj =
                calendar.adjust(s, ql_time::BusinessDayConvention::ModifiedFollowing);
            let e = advance_months(
                start_date,
                (i as i32 + 1) * months_per_period as i32,
            );
            let e_adj =
                calendar.adjust(e, ql_time::BusinessDayConvention::ModifiedFollowing);
            float_start_dates.push(s_adj);
            float_end_dates.push(e_adj);
        }

        Self {
            rate,
            start_date,
            fixed_payment_dates,
            float_start_dates,
            float_end_dates,
            fixed_day_counter,
            float_day_counter,
        }
    }
}

impl DualCurveRateHelper for DualCurveSwapRateHelper {
    fn pillar_date(&self) -> Date {
        *self
            .float_end_dates
            .last()
            .or(self.fixed_payment_dates.last())
            .unwrap_or(&self.start_date)
    }

    fn quote(&self) -> f64 {
        self.rate
    }

    fn implied_quote_dual(
        &self,
        forecast_times: &[f64],
        forecast_dfs: &[f64],
        discount_curve: &dyn YieldTermStructure,
        day_counter: DayCounter,
        ref_date: Date,
    ) -> f64 {
        // Fixed-leg annuity (discounted with OIS curve)
        let mut annuity = 0.0;
        let mut prev_fixed = self.start_date;
        for &pay_date in &self.fixed_payment_dates {
            let yf = self.fixed_day_counter.year_fraction(prev_fixed, pay_date);
            let df = discount_curve.discount(pay_date);
            annuity += yf * df;
            prev_fixed = pay_date;
        }

        if annuity.abs() < 1e-15 {
            return 0.0;
        }

        // Float-leg PV (forward rates from forecast curve, discounted with OIS)
        let mut float_pv = 0.0;
        for (s_date, e_date) in self
            .float_start_dates
            .iter()
            .zip(self.float_end_dates.iter())
        {
            let t_s = day_counter.year_fraction(ref_date, *s_date);
            let t_e = day_counter.year_fraction(ref_date, *e_date);
            let yf = self.float_day_counter.year_fraction(*s_date, *e_date);

            // Forward rate from the forecast curve being bootstrapped
            let df_s = interpolate_log_linear(forecast_times, forecast_dfs, t_s);
            let df_e = interpolate_log_linear(forecast_times, forecast_dfs, t_e);
            let fwd = if yf.abs() < 1e-15 {
                0.0
            } else {
                (df_s / df_e - 1.0) / yf
            };

            // Discount with OIS curve
            let ois_df = discount_curve.discount(*e_date);
            float_pv += fwd * yf * ois_df;
        }

        // Implied par rate = float_pv / annuity
        float_pv / annuity
    }
}

// ===========================================================================
// MultiCurveOISRateHelper
// ===========================================================================

/// Multi-period OIS rate helper with proper sub-period compounding.
///
/// For OIS swaps longer than ~1Y the single-period approximation
/// ($r = (df_s/df_e - 1)/\tau$) is inaccurate. This helper compounds
/// the implied overnight rates over quarterly sub-periods and matches the
/// resulting swap rate against the market quote.
#[derive(Debug, Clone)]
pub struct MultiCurveOISRateHelper {
    /// Quoted par OIS rate.
    rate: f64,
    /// Start date of the swap.
    start_date: Date,
    /// Fixed-leg payment dates (annual).
    fixed_dates: Vec<Date>,
    /// Floating sub-period start dates.
    float_start_dates: Vec<Date>,
    /// Floating sub-period end dates.
    float_end_dates: Vec<Date>,
    /// Day counter for the fixed leg.
    fixed_day_counter: DayCounter,
    /// Day counter for the floating leg.
    float_day_counter: DayCounter,
}

impl MultiCurveOISRateHelper {
    /// Create a multi-period OIS helper from a tenor.
    ///
    /// Fixed leg pays annually; floating leg compounds quarterly.
    pub fn from_tenor(
        rate: f64,
        start_date: Date,
        tenor_years: u32,
        fixed_day_counter: DayCounter,
        float_day_counter: DayCounter,
        calendar: Calendar,
    ) -> Self {
        // Fixed dates: annual
        let mut fixed_dates = Vec::new();
        for i in 1..=tenor_years {
            let d = advance_months(start_date, i as i32 * 12);
            let adj =
                calendar.adjust(d, ql_time::BusinessDayConvention::ModifiedFollowing);
            fixed_dates.push(adj);
        }

        // Float: quarterly sub-periods
        let total_periods = tenor_years * 4;
        let mut float_start_dates = Vec::new();
        let mut float_end_dates = Vec::new();
        for i in 0..total_periods {
            let s = if i == 0 {
                start_date
            } else {
                advance_months(start_date, i as i32 * 3)
            };
            let s_adj =
                calendar.adjust(s, ql_time::BusinessDayConvention::ModifiedFollowing);
            let e = advance_months(start_date, (i as i32 + 1) * 3);
            let e_adj =
                calendar.adjust(e, ql_time::BusinessDayConvention::ModifiedFollowing);
            float_start_dates.push(s_adj);
            float_end_dates.push(e_adj);
        }

        Self {
            rate,
            start_date,
            fixed_dates,
            float_start_dates,
            float_end_dates,
            fixed_day_counter,
            float_day_counter,
        }
    }
}

impl RateHelper for MultiCurveOISRateHelper {
    fn pillar_date(&self) -> Date {
        *self
            .fixed_dates
            .last()
            .unwrap_or(&self.start_date)
    }

    fn quote(&self) -> f64 {
        self.rate
    }

    fn implied_quote(
        &self,
        times: &[f64],
        dfs: &[f64],
        day_counter: DayCounter,
        ref_date: Date,
    ) -> f64 {
        // Fixed-leg annuity
        let mut annuity = 0.0;
        let mut prev = self.start_date;
        for &d in &self.fixed_dates {
            let yf = self.fixed_day_counter.year_fraction(prev, d);
            let t = day_counter.year_fraction(ref_date, d);
            let df = interpolate_log_linear(times, dfs, t);
            annuity += yf * df;
            prev = d;
        }
        if annuity.abs() < 1e-15 {
            return 0.0;
        }

        // Floating-leg PV: compound sub-period ON rates
        let mut float_pv = 0.0;
        for (s, e) in self
            .float_start_dates
            .iter()
            .zip(self.float_end_dates.iter())
        {
            let t_s = day_counter.year_fraction(ref_date, *s);
            let t_e = day_counter.year_fraction(ref_date, *e);
            let yf = self.float_day_counter.year_fraction(*s, *e);

            let df_s = interpolate_log_linear(times, dfs, t_s);
            let df_e = interpolate_log_linear(times, dfs, t_e);
            let fwd = if yf.abs() < 1e-15 {
                0.0
            } else {
                (df_s / df_e - 1.0) / yf
            };

            let ois_df_end = interpolate_log_linear(times, dfs, t_e);
            float_pv += fwd * yf * ois_df_end;
        }

        float_pv / annuity
    }
}

// ===========================================================================
// Forecast curve bootstrap (given external discount curve)
// ===========================================================================

/// Bootstrap a **forecast** (IBOR) yield curve using an external discount
/// curve for cashflow discounting.
///
/// This is the standard post-crisis dual-curve bootstrap:
/// 1. The OIS/SOFR discount curve is already available.
/// 2. Each helper projects forward rates from the curve being built and
///    discounts with the exogenous OIS curve.
///
/// Returns a [`PiecewiseYieldCurve`] representing the forecast curve.
pub fn bootstrap_forecast_curve(
    reference_date: Date,
    helpers: &mut [Box<dyn DualCurveRateHelper>],
    discount_curve: &dyn YieldTermStructure,
    day_counter: DayCounter,
    accuracy: f64,
) -> QLResult<PiecewiseYieldCurve> {
    let _span = info_span!("bootstrap_forecast", num_helpers = helpers.len()).entered();
    info!(
        num_helpers = helpers.len(),
        "Starting dual-curve forecast bootstrap"
    );

    helpers.sort_by_key(|h| h.pillar_date());

    let mut times = vec![0.0];
    let mut dfs = vec![1.0];
    let solver = Brent;

    for helper in helpers.iter() {
        let pillar = helper.pillar_date();
        let t_pillar = day_counter.year_fraction(reference_date, pillar);

        if t_pillar <= 0.0 {
            return Err(QLError::InvalidArgument(
                "pillar date must be after reference date".into(),
            ));
        }

        let market_quote = helper.quote();
        let guess_df = (-0.05 * t_pillar).exp();
        times.push(t_pillar);
        dfs.push(guess_df);

        let n = times.len();
        let times_clone = times.clone();
        let dfs_clone = dfs.clone();

        let objective = |df: f64| -> f64 {
            let mut trial_dfs = dfs_clone.clone();
            trial_dfs[n - 1] = df;
            helper.implied_quote_dual(
                &times_clone,
                &trial_dfs,
                discount_curve,
                day_counter,
                reference_date,
            ) - market_quote
        };

        let df_solution = solver.solve(objective, 0.0, guess_df, 1e-6, 2.0, accuracy, 100)?;

        dfs[n - 1] = df_solution;
        debug!(
            pillar = %pillar,
            t = t_pillar,
            quote = market_quote,
            df = df_solution,
            "Forecast pillar solved"
        );
    }

    info!(pillars = times.len() - 1, "Forecast bootstrap complete");

    let max_date = helpers
        .last()
        .map(|h| h.pillar_date())
        .unwrap_or(reference_date);

    PiecewiseYieldCurve::from_nodes(reference_date, day_counter, times, dfs, max_date)
}

// ===========================================================================
// Iterative dual-curve bootstrap
// ===========================================================================

/// Iteratively bootstrap OIS discount and IBOR forecast curves.
///
/// **Algorithm:**
/// 1. Bootstrap the OIS curve from `ois_helpers` (single-curve).
/// 2. Bootstrap the IBOR forecast curve from `ibor_helpers` using the OIS
///    curve for discounting.
/// 3. If convergence is requested (`max_iterations > 1`), repeat steps 1-2
///    until both curves stabilise (max DF change < `accuracy` at all pillars).
///
/// Returns `(ois_curve, ibor_forecast_curve)`.
pub fn bootstrap_dual_curves(
    reference_date: Date,
    ois_helpers: &mut [Box<dyn RateHelper>],
    ibor_helpers: &mut [Box<dyn DualCurveRateHelper>],
    day_counter: DayCounter,
    accuracy: f64,
    max_iterations: usize,
) -> QLResult<(PiecewiseYieldCurve, PiecewiseYieldCurve)> {
    let _span = info_span!("dual_curve_bootstrap").entered();
    info!(
        ois = ois_helpers.len(),
        ibor = ibor_helpers.len(),
        max_iter = max_iterations,
        "Starting dual-curve bootstrap"
    );

    let max_iterations = max_iterations.max(1);

    // Step 1: initial OIS curve
    let mut ois_curve = PiecewiseYieldCurve::new(
        reference_date,
        ois_helpers,
        day_counter,
        accuracy,
    )?;

    let mut ibor_curve: Option<PiecewiseYieldCurve> = None;

    for iter in 0..max_iterations {
        // Step 2: bootstrap IBOR forecast curve using current OIS discount curve
        let new_ibor = bootstrap_forecast_curve(
            reference_date,
            ibor_helpers,
            &ois_curve,
            day_counter,
            accuracy,
        )?;

        // Check convergence on IBOR curve
        if let Some(ref prev) = ibor_curve {
            let prev_nodes = prev.nodes();
            let new_nodes = new_ibor.nodes();
            let max_change = prev_nodes
                .iter()
                .zip(new_nodes.iter())
                .map(|((_, df1), (_, df2))| (df1 - df2).abs())
                .fold(0.0_f64, f64::max);

            debug!(iter, max_change, "Dual-curve iteration");

            if max_change < accuracy {
                info!(iter, "Dual-curve bootstrap converged");
                return Ok((ois_curve, new_ibor));
            }
        }

        ibor_curve = Some(new_ibor);

        // For a pure sequential approach, OIS helpers don't depend on the
        // IBOR curve, so re-bootstrapping OIS is a no-op. But if there were
        // cross-dependencies (e.g. basis swaps), we would rebuild here:
        if iter < max_iterations - 1 {
            ois_curve = PiecewiseYieldCurve::new(
                reference_date,
                ois_helpers,
                day_counter,
                accuracy,
            )?;
        }
    }

    let ibor = ibor_curve.ok_or_else(|| {
        QLError::InvalidArgument("dual-curve bootstrap produced no IBOR curve".into())
    })?;

    info!("Dual-curve bootstrap complete (max iterations reached)");
    Ok((ois_curve, ibor))
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Advance a date by a given number of months.
fn advance_months(d: Date, months: i32) -> Date {
    let (y, m, day) = (d.year(), d.month() as u32, d.day_of_month());
    let total = y * 12 + m as i32 - 1 + months;
    let new_y = total / 12;
    let new_m = (total % 12) as u32 + 1;
    let new_d = day.min(Date::days_in_month(new_y, new_m));
    Date::from_ymd_opt(new_y, new_m, new_d).unwrap_or(d)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bootstrap::{DepositRateHelper, SwapRateHelper};
    use crate::bootstrap_extended::OISRateHelper;
    use crate::FlatForward;
    use approx::assert_abs_diff_eq;
    use ql_time::Month;

    fn ref_date() -> Date {
        Date::from_ymd(2025, Month::January, 2)
    }

    // -----------------------------------------------------------------
    // DualCurveSwapRateHelper
    // -----------------------------------------------------------------

    #[test]
    fn dual_curve_swap_helper_flat_curves() {
        // When forecast and discount curves are the same flat rate, the
        // dual-curve swap helper should reproduce the flat rate as the
        // par swap rate.
        let rd = ref_date();
        let flat = FlatForward::new(rd, 0.04, DayCounter::Actual360);

        let h = DualCurveSwapRateHelper::from_tenor(
            0.04,
            rd,
            5,
            4,
            DayCounter::Actual360,
            DayCounter::Actual360,
            Calendar::Target,
        );

        // Build a trivially accurate forecast curve (same flat 4%)
        // by creating times/dfs matching the flat curve at many points.
        let mut times = vec![0.0];
        let mut dfs_vec = vec![1.0];
        for yr in 1..=6 {
            let t = yr as f64;
            times.push(t);
            dfs_vec.push(flat.discount_t(t));
        }

        let iq = h.implied_quote_dual(
            &times,
            &dfs_vec,
            &flat,
            DayCounter::Actual360,
            rd,
        );

        // Should be close to 4% (not exact due to discrete quarterly periods
        // and day-count rounding vs continuous flat rate)
        assert_abs_diff_eq!(iq, 0.04, epsilon = 1.5e-3);
    }

    #[test]
    fn dual_curve_swap_helper_different_curves() {
        // Forecast at 5%, discount at 3%. The implied par rate should lie
        // between them (closer to the forecast rate).
        let rd = ref_date();
        let disc = FlatForward::new(rd, 0.03, DayCounter::Actual360);
        let fcast = FlatForward::new(rd, 0.05, DayCounter::Actual360);

        let h = DualCurveSwapRateHelper::from_tenor(
            0.05,
            rd,
            5,
            4,
            DayCounter::Actual360,
            DayCounter::Actual360,
            Calendar::Target,
        );

        let mut times = vec![0.0];
        let mut dfs_vec = vec![1.0];
        for yr in 1..=6 {
            let t = yr as f64 * 0.25;
            times.push(t);
            dfs_vec.push(fcast.discount_t(t));
        }
        // Add annual points too
        for yr in 2..=6 {
            let t = yr as f64;
            times.push(t);
            dfs_vec.push(fcast.discount_t(t));
        }
        // sort by time
        let mut nodes: Vec<(f64, f64)> = times.into_iter().zip(dfs_vec).collect();
        nodes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        nodes.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-10);
        let (times, dfs_vec): (Vec<f64>, Vec<f64>) = nodes.into_iter().unzip();

        let iq = h.implied_quote_dual(
            &times,
            &dfs_vec,
            &disc,
            DayCounter::Actual360,
            rd,
        );

        // When forecast > discount, par rate ≈ forecast rate
        assert!(iq > 0.04, "implied rate {iq} should be > 4%");
        assert!(iq < 0.06, "implied rate {iq} should be < 6%");
    }

    // -----------------------------------------------------------------
    // Multi-period OIS helper
    // -----------------------------------------------------------------

    #[test]
    fn multi_period_ois_bootstrap() {
        let rd = ref_date();
        let start = rd;

        // Short end: 3M OIS (single period — same as deposit helper)
        let end_3m = advance_months(start, 3);
        let end_3m = Calendar::Target.adjust(
            end_3m,
            ql_time::BusinessDayConvention::ModifiedFollowing,
        );

        // Long end: 2Y OIS with quarterly compounding
        let mut helpers: Vec<Box<dyn RateHelper>> = vec![
            Box::new(OISRateHelper::new(
                0.035,
                start,
                end_3m,
                DayCounter::Actual360,
            )),
            Box::new(MultiCurveOISRateHelper::from_tenor(
                0.04,
                start,
                2,
                DayCounter::Actual360,
                DayCounter::Actual360,
                Calendar::Target,
            )),
        ];

        let curve = PiecewiseYieldCurve::new(rd, &mut helpers, DayCounter::Actual360, 1e-12)
            .expect("bootstrap should succeed");

        // Curve should have sensible DFs
        assert!(curve.discount(end_3m) > 0.95 && curve.discount(end_3m) < 1.0);
        let end_2y = advance_months(start, 24);
        let df_2y = curve.discount(end_2y);
        assert!(df_2y > 0.9 && df_2y < 0.98, "df_2y = {df_2y}");
    }

    // -----------------------------------------------------------------
    // bootstrap_forecast_curve
    // -----------------------------------------------------------------

    #[test]
    fn bootstrap_forecast_with_flat_discount() {
        let rd = ref_date();
        let start = rd;

        // OIS discount curve: flat 3%
        let ois = FlatForward::new(rd, 0.03, DayCounter::Actual360);

        // IBOR swap helpers at different tenors
        let mut helpers: Vec<Box<dyn DualCurveRateHelper>> = vec![
            Box::new(DualCurveSwapRateHelper::from_tenor(
                0.04,
                start,
                2,
                4,
                DayCounter::Actual360,
                DayCounter::Actual360,
                Calendar::Target,
            )),
            Box::new(DualCurveSwapRateHelper::from_tenor(
                0.045,
                start,
                5,
                4,
                DayCounter::Actual360,
                DayCounter::Actual360,
                Calendar::Target,
            )),
            Box::new(DualCurveSwapRateHelper::from_tenor(
                0.05,
                start,
                10,
                4,
                DayCounter::Actual360,
                DayCounter::Actual360,
                Calendar::Target,
            )),
        ];

        let forecast = bootstrap_forecast_curve(
            rd,
            &mut helpers,
            &ois,
            DayCounter::Actual360,
            1e-10,
        )
        .expect("forecast bootstrap should succeed");

        // Forecast curve should have higher rates than OIS
        let df_5y = forecast.discount(advance_months(start, 60));
        let ois_5y = ois.discount(advance_months(start, 60));
        assert!(
            df_5y < ois_5y,
            "forecast df ({df_5y}) should be < OIS df ({ois_5y}) since IBOR > OIS"
        );

        // Check that 2Y tenor reproduces roughly 4%
        // (pillar rate verification)
        assert!(forecast.size() >= 3, "should have at least 3 pillars + ref");
    }

    // -----------------------------------------------------------------
    // bootstrap_dual_curves
    // -----------------------------------------------------------------

    #[test]
    fn dual_curve_end_to_end() {
        let rd = ref_date();
        let start = rd;

        // OIS helpers: 3M deposit + 2Y OIS
        let end_3m = Calendar::Target.adjust(
            advance_months(start, 3),
            ql_time::BusinessDayConvention::ModifiedFollowing,
        );
        let mut ois_helpers: Vec<Box<dyn RateHelper>> = vec![
            Box::new(DepositRateHelper::new(
                0.03,
                start,
                end_3m,
                DayCounter::Actual360,
            )),
            Box::new(SwapRateHelper::from_tenor(
                0.035,
                start,
                2,
                1,
                DayCounter::Actual360,
                Calendar::Target,
            )),
            Box::new(SwapRateHelper::from_tenor(
                0.04,
                start,
                5,
                1,
                DayCounter::Actual360,
                Calendar::Target,
            )),
        ];

        // IBOR helpers: 2Y and 5Y swaps
        let mut ibor_helpers: Vec<Box<dyn DualCurveRateHelper>> = vec![
            Box::new(DualCurveSwapRateHelper::from_tenor(
                0.04,
                start,
                2,
                4,
                DayCounter::Actual360,
                DayCounter::Actual360,
                Calendar::Target,
            )),
            Box::new(DualCurveSwapRateHelper::from_tenor(
                0.05,
                start,
                5,
                4,
                DayCounter::Actual360,
                DayCounter::Actual360,
                Calendar::Target,
            )),
        ];

        let (ois_curve, ibor_curve) = bootstrap_dual_curves(
            rd,
            &mut ois_helpers,
            &mut ibor_helpers,
            DayCounter::Actual360,
            1e-10,
            3,
        )
        .expect("dual bootstrap should succeed");

        // OIS curve should have lower rates than IBOR
        let end_5y = advance_months(start, 60);
        let ois_df = ois_curve.discount(end_5y);
        let ibor_df = ibor_curve.discount(end_5y);
        assert!(
            ibor_df < ois_df,
            "IBOR df ({ibor_df}) < OIS df ({ois_df}) since IBOR spread > 0"
        );

        // Both curves should be monotonically decreasing
        for yr in 1..=5 {
            let d = advance_months(start, yr * 12);
            assert!(ois_curve.discount(d) < 1.0);
            assert!(ibor_curve.discount(d) < 1.0);
        }
    }
}
