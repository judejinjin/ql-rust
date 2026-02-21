//! Bootstrapping framework for piecewise yield curves.
//!
//! Provides:
//! - `RateHelper` trait for market instrument helpers
//! - `DepositRateHelper` for money-market deposits
//! - `SwapRateHelper` for par swap rates
//! - `PiecewiseYieldCurve` with iterative bootstrap using Brent solver

use ql_core::errors::{QLError, QLResult};
use ql_math::solvers1d::{Brent, Solver1D};
use ql_time::{Calendar, Date, DayCounter};
use tracing::{debug, info, info_span};

use crate::term_structure::TermStructure;
use crate::yield_term_structure::YieldTermStructure;

// ===========================================================================
// RateHelper Trait
// ===========================================================================

/// A market instrument helper used to bootstrap a yield curve.
///
/// Each helper corresponds to one market instrument (deposit, swap, etc.)
/// and defines a pillar date and the relationship between the quoted rate
/// and the discount factors on the curve.
pub trait RateHelper: Send + Sync {
    /// The pillar date — the maturity date that this helper constrains.
    fn pillar_date(&self) -> Date;

    /// The quoted market rate.
    fn quote(&self) -> f64;

    /// Compute the implied quote given a set of discount factors.
    ///
    /// `times` and `dfs` define the current state of the curve being built.
    /// The helper computes what the instrument's rate would be under these dfs.
    fn implied_quote(&self, times: &[f64], dfs: &[f64], day_counter: DayCounter, ref_date: Date) -> f64;
}

// ===========================================================================
// DepositRateHelper
// ===========================================================================

/// Helper for bootstrapping from money-market deposit rates.
///
/// A deposit has a simple rate: `df = 1 / (1 + rate * yearfrac)`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DepositRateHelper {
    /// Quoted deposit rate.
    rate: f64,
    /// Start date (value date).
    start_date: Date,
    /// End date (maturity).
    end_date: Date,
    /// Day counter for year fraction.
    day_counter: DayCounter,
}

impl DepositRateHelper {
    /// Create a deposit rate helper.
    pub fn new(rate: f64, start_date: Date, end_date: Date, day_counter: DayCounter) -> Self {
        Self {
            rate,
            start_date,
            end_date,
            day_counter,
        }
    }
}

impl RateHelper for DepositRateHelper {
    fn pillar_date(&self) -> Date {
        self.end_date
    }

    fn quote(&self) -> f64 {
        self.rate
    }

    fn implied_quote(&self, times: &[f64], dfs: &[f64], day_counter: DayCounter, ref_date: Date) -> f64 {
        let t_start = day_counter.year_fraction(ref_date, self.start_date);
        let t_end = day_counter.year_fraction(ref_date, self.end_date);
        let yf = self.day_counter.year_fraction(self.start_date, self.end_date);

        let df_start = interpolate_log_linear(times, dfs, t_start);
        let df_end = interpolate_log_linear(times, dfs, t_end);

        if yf.abs() < 1e-15 {
            return 0.0;
        }
        (df_start / df_end - 1.0) / yf
    }
}

// ===========================================================================
// SwapRateHelper
// ===========================================================================

/// Helper for bootstrapping from par swap rates.
///
/// Assumes a fixed-for-floating swap where the fixed leg pays a known rate
/// and we need to find discount factors such that the swap has zero NPV.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SwapRateHelper {
    /// Quoted par swap rate.
    rate: f64,
    /// Fixed leg payment dates (not including start date).
    payment_dates: Vec<Date>,
    /// Start date of the swap.
    start_date: Date,
    /// Day counter for the fixed leg.
    day_counter: DayCounter,
}

impl SwapRateHelper {
    /// Create a swap rate helper from a quoted rate and payment schedule.
    pub fn new(
        rate: f64,
        start_date: Date,
        payment_dates: Vec<Date>,
        day_counter: DayCounter,
    ) -> Self {
        Self {
            rate,
            payment_dates,
            start_date,
            day_counter,
        }
    }

    /// Convenience: create from a rate, tenor in years, and frequency (payments per year).
    ///
    /// Generates evenly-spaced payment dates.
    pub fn from_tenor(
        rate: f64,
        start_date: Date,
        tenor_years: u32,
        frequency: u32,
        day_counter: DayCounter,
        calendar: Calendar,
    ) -> Self {
        let mut payment_dates = Vec::new();
        let total_periods = tenor_years * frequency;
        let months_per_period = 12 / frequency;

        for i in 1..=total_periods {
            let months = (i * months_per_period) as i32;
            let (y, m, d) = (
                start_date.year(),
                start_date.month() as u32,
                start_date.day_of_month(),
            );
            let total_months = (y * 12 + m as i32 - 1) + months;
            let new_y = total_months / 12;
            let new_m = (total_months % 12) as u32 + 1;
            let new_d = d.min(Date::days_in_month(new_y, new_m));
            let date = Date::from_ymd_opt(new_y, new_m, new_d).unwrap_or(start_date);
            let adjusted = calendar.adjust(date, ql_time::BusinessDayConvention::ModifiedFollowing);
            payment_dates.push(adjusted);
        }

        Self {
            rate,
            payment_dates,
            start_date,
            day_counter,
        }
    }
}

impl RateHelper for SwapRateHelper {
    fn pillar_date(&self) -> Date {
        *self.payment_dates.last().unwrap_or(&self.start_date)
    }

    fn quote(&self) -> f64 {
        self.rate
    }

    fn implied_quote(&self, times: &[f64], dfs: &[f64], day_counter: DayCounter, ref_date: Date) -> f64 {
        // Par swap rate = (df_start - df_end) / annuity
        // where annuity = sum of (year_frac_i * df_i) for each fixed payment
        let t_start = day_counter.year_fraction(ref_date, self.start_date);
        let df_start = interpolate_log_linear(times, dfs, t_start);

        let mut annuity = 0.0;
        let mut prev_date = self.start_date;

        for &payment_date in &self.payment_dates {
            let yf = self.day_counter.year_fraction(prev_date, payment_date);
            let t = day_counter.year_fraction(ref_date, payment_date);
            let df = interpolate_log_linear(times, dfs, t);
            annuity += yf * df;
            prev_date = payment_date;
        }

        if annuity.abs() < 1e-15 {
            return 0.0;
        }

        let t_end = day_counter.year_fraction(ref_date, self.pillar_date());
        let df_end = interpolate_log_linear(times, dfs, t_end);

        (df_start - df_end) / annuity
    }
}

// ===========================================================================
// PiecewiseYieldCurve
// ===========================================================================

/// A yield curve constructed by iterative bootstrapping from rate helpers.
///
/// For each helper (sorted by pillar date), the bootstrap uses the Brent
/// solver to find the discount factor at the pillar date that makes the
/// helper's implied quote match its market quote.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PiecewiseYieldCurve {
    reference_date: Date,
    day_counter: DayCounter,
    calendar: Calendar,
    /// Bootstrapped times (year fractions).
    times: Vec<f64>,
    /// Bootstrapped discount factors.
    dfs: Vec<f64>,
    /// Max date (last pillar).
    max_date: Date,
}

impl PiecewiseYieldCurve {
    /// Bootstrap a yield curve from rate helpers.
    ///
    /// Helpers should cover non-overlapping pillars. They will be sorted by
    /// pillar date internally.
    pub fn new(
        reference_date: Date,
        helpers: &mut [Box<dyn RateHelper>],
        day_counter: DayCounter,
        accuracy: f64,
    ) -> QLResult<Self> {
        let _span = info_span!("bootstrap", num_helpers = helpers.len(), accuracy).entered();
        info!(num_helpers = helpers.len(), "Starting yield-curve bootstrap");

        // Sort helpers by pillar date
        helpers.sort_by_key(|h| h.pillar_date());

        // Initialize with reference date: t=0, df=1.0
        let mut times = vec![0.0];
        let mut dfs = vec![1.0];
        let calendar = Calendar::NullCalendar;

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

            // Temporarily add a guess for this pillar
            let guess_df = (-0.05 * t_pillar).exp(); // Initial guess: 5% flat
            times.push(t_pillar);
            dfs.push(guess_df);

            let n = times.len();

            // Objective: find df_pillar such that implied_quote == market_quote
            let times_clone = times.clone();
            let dfs_clone = dfs.clone();

            let objective = |df: f64| -> f64 {
                let mut trial_dfs = dfs_clone.clone();
                trial_dfs[n - 1] = df;
                helper.implied_quote(&times_clone, &trial_dfs, day_counter, reference_date) - market_quote
            };

            // Bracket: discount factor should be between 0.001 and 2.0
            let df_solution = solver.solve(
                objective,
                0.0,
                guess_df,
                1e-6,
                2.0,
                accuracy,
                100,
            )?;

            dfs[n - 1] = df_solution;
            debug!(pillar = %pillar, t = t_pillar, quote = market_quote, df = df_solution, "Bootstrap pillar solved");
        }

        info!(pillars = times.len() - 1, "Bootstrap complete");

        let max_date = helpers
            .last()
            .map(|h| h.pillar_date())
            .unwrap_or(reference_date);

        Ok(Self {
            reference_date,
            day_counter,
            calendar,
            times,
            dfs,
            max_date,
        })
    }

    /// Get the bootstrapped discount factors as (time, df) pairs.
    pub fn nodes(&self) -> Vec<(f64, f64)> {
        self.times
            .iter()
            .zip(self.dfs.iter())
            .map(|(&t, &df)| (t, df))
            .collect()
    }

    /// Number of pillar points (including t=0).
    pub fn size(&self) -> usize {
        self.times.len()
    }

    /// Construct directly from pre-computed nodes (used by multi-curve bootstrap).
    pub fn from_nodes(
        reference_date: Date,
        day_counter: DayCounter,
        times: Vec<f64>,
        dfs: Vec<f64>,
        max_date: Date,
    ) -> QLResult<Self> {
        if times.len() != dfs.len() || times.is_empty() {
            return Err(QLError::InvalidArgument(
                "times and dfs must be non-empty and same length".into(),
            ));
        }
        Ok(Self {
            reference_date,
            day_counter,
            calendar: Calendar::NullCalendar,
            times,
            dfs,
            max_date,
        })
    }
}

impl TermStructure for PiecewiseYieldCurve {
    fn reference_date(&self) -> Date {
        self.reference_date
    }

    fn day_counter(&self) -> DayCounter {
        self.day_counter
    }

    fn calendar(&self) -> &Calendar {
        &self.calendar
    }

    fn max_date(&self) -> Date {
        self.max_date
    }
}

impl YieldTermStructure for PiecewiseYieldCurve {
    fn discount_impl(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }
        interpolate_log_linear(&self.times, &self.dfs, t)
    }
}

// ===========================================================================
// Helper: log-linear interpolation on discount factors
// ===========================================================================

/// Interpolate discount factors log-linearly.
///
/// If `t` is before the first node, extrapolate flat at the first df.
/// If `t` is after the last node, extrapolate log-linearly from the last segment.
pub(crate) fn interpolate_log_linear(times: &[f64], dfs: &[f64], t: f64) -> f64 {
    if times.is_empty() || dfs.is_empty() {
        return 1.0;
    }

    if t <= times[0] {
        return dfs[0];
    }

    if t >= times[times.len() - 1] {
        // Log-linear extrapolation from last segment
        let n = times.len();
        if n < 2 {
            return dfs[dfs.len() - 1];
        }
        let t1 = times[n - 2];
        let t2 = times[n - 1];
        let ln1 = dfs[n - 2].ln();
        let ln2 = dfs[n - 1].ln();
        if (t2 - t1).abs() < 1e-15 {
            return dfs[n - 1];
        }
        let slope = (ln2 - ln1) / (t2 - t1);
        (ln2 + slope * (t - t2)).exp()
    } else {
        // Find segment
        let mut i = 0;
        for (j, tj) in times.iter().enumerate().skip(1) {
            if *tj >= t {
                i = j - 1;
                break;
            }
        }
        let t1 = times[i];
        let t2 = times[i + 1];
        let ln1 = dfs[i].ln();
        let ln2 = dfs[i + 1].ln();
        if (t2 - t1).abs() < 1e-15 {
            return dfs[i];
        }
        let frac = (t - t1) / (t2 - t1);
        (ln1 + frac * (ln2 - ln1)).exp()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_time::Month;

    fn ref_date() -> Date {
        Date::from_ymd(2025, Month::January, 2)
    }

    #[test]
    fn bootstrap_single_deposit() {
        let ref_date = ref_date();
        let start = ref_date;
        let end = Date::from_ymd(2025, Month::April, 2); // ~3M
        let rate = 0.04; // 4%

        let mut helpers: Vec<Box<dyn RateHelper>> = vec![Box::new(DepositRateHelper::new(
            rate,
            start,
            end,
            DayCounter::Actual360,
        ))];

        let curve = PiecewiseYieldCurve::new(
            ref_date,
            &mut helpers,
            DayCounter::Actual360,
            1e-12,
        )
        .unwrap();

        // The bootstrapped curve should give a discount factor at the deposit maturity
        // consistent with: df = 1/(1 + rate * yf)
        let yf = DayCounter::Actual360.year_fraction(start, end);
        let expected_df = 1.0 / (1.0 + rate * yf);
        let actual_df = curve.discount(end);
        assert_abs_diff_eq!(actual_df, expected_df, epsilon = 1e-8);
    }

    #[test]
    fn bootstrap_two_deposits() {
        let ref_date = ref_date();
        let start = ref_date;

        let end_3m = Date::from_ymd(2025, Month::April, 2);
        let end_6m = Date::from_ymd(2025, Month::July, 2);

        let mut helpers: Vec<Box<dyn RateHelper>> = vec![
            Box::new(DepositRateHelper::new(0.04, start, end_3m, DayCounter::Actual360)),
            Box::new(DepositRateHelper::new(0.045, start, end_6m, DayCounter::Actual360)),
        ];

        let curve = PiecewiseYieldCurve::new(
            ref_date,
            &mut helpers,
            DayCounter::Actual360,
            1e-12,
        )
        .unwrap();

        // Check 3M deposit
        let yf_3m = DayCounter::Actual360.year_fraction(start, end_3m);
        let expected_3m = 1.0 / (1.0 + 0.04 * yf_3m);
        assert_abs_diff_eq!(curve.discount(end_3m), expected_3m, epsilon = 1e-8);

        // Check 6M deposit
        let yf_6m = DayCounter::Actual360.year_fraction(start, end_6m);
        let expected_6m = 1.0 / (1.0 + 0.045 * yf_6m);
        assert_abs_diff_eq!(curve.discount(end_6m), expected_6m, epsilon = 1e-8);
    }

    #[test]
    fn bootstrap_deposits_and_swap() {
        let ref_date = ref_date();
        let start = ref_date;

        // Short end: 3M and 6M deposits
        let end_3m = Date::from_ymd(2025, Month::April, 2);
        let end_6m = Date::from_ymd(2025, Month::July, 2);

        // Long end: 2Y annual swap
        let swap_dates = vec![
            Date::from_ymd(2026, Month::January, 2),
            Date::from_ymd(2027, Month::January, 2),
        ];

        let mut helpers: Vec<Box<dyn RateHelper>> = vec![
            Box::new(DepositRateHelper::new(0.04, start, end_3m, DayCounter::Actual360)),
            Box::new(DepositRateHelper::new(0.045, start, end_6m, DayCounter::Actual360)),
            Box::new(SwapRateHelper::new(0.05, start, swap_dates.clone(), DayCounter::Actual360)),
        ];

        let curve = PiecewiseYieldCurve::new(
            ref_date,
            &mut helpers,
            DayCounter::Actual360,
            1e-12,
        )
        .unwrap();

        // Verify curve has correct number of nodes (1 ref + 3 pillars)
        assert_eq!(curve.size(), 4);

        // Verify discount factors are monotonically decreasing
        let nodes = curve.nodes();
        for i in 1..nodes.len() {
            assert!(
                nodes[i].1 < nodes[i - 1].1,
                "df should decrease: t={}, df={}",
                nodes[i].0,
                nodes[i].1
            );
        }

        // Verify the swap rate is reproduced
        let df_start = curve.discount(start);
        let df_1y = curve.discount(swap_dates[0]);
        let df_2y = curve.discount(swap_dates[1]);
        let yf1 = DayCounter::Actual360.year_fraction(start, swap_dates[0]);
        let yf2 = DayCounter::Actual360.year_fraction(swap_dates[0], swap_dates[1]);
        let annuity = yf1 * df_1y + yf2 * df_2y;
        let implied_swap = (df_start - df_2y) / annuity;
        assert_abs_diff_eq!(implied_swap, 0.05, epsilon = 1e-6);
    }

    #[test]
    fn bootstrap_consistency() {
        let ref_date = ref_date();
        let start = ref_date;

        let end_6m = Date::from_ymd(2025, Month::July, 2);

        let mut helpers: Vec<Box<dyn RateHelper>> = vec![
            Box::new(DepositRateHelper::new(0.04, start, end_6m, DayCounter::Actual360)),
        ];

        let curve = PiecewiseYieldCurve::new(
            ref_date,
            &mut helpers,
            DayCounter::Actual360,
            1e-12,
        )
        .unwrap();

        // discount(ref_date) should be 1.0
        assert_abs_diff_eq!(curve.discount(ref_date), 1.0, epsilon = 1e-14);

        // All discount factors should be positive
        for &(_, df) in curve.nodes().iter() {
            assert!(df > 0.0);
        }
    }

    #[test]
    fn swap_helper_from_tenor() {
        let ref_date = ref_date();
        let helper = SwapRateHelper::from_tenor(
            0.05,
            ref_date,
            5,
            1,
            DayCounter::Actual360,
            Calendar::Target,
        );
        assert_eq!(helper.payment_dates.len(), 5);
        assert!(helper.pillar_date() > ref_date);
    }

    #[test]
    fn interpolate_log_linear_basic() {
        let times = vec![0.0, 1.0, 2.0];
        let dfs = vec![1.0, 0.95, 0.90];
        // At t=0.5, should be between 1.0 and 0.95
        let df = interpolate_log_linear(&times, &dfs, 0.5);
        assert!(df > 0.94 && df < 1.0, "df = {df}");
        // At nodes
        assert_abs_diff_eq!(interpolate_log_linear(&times, &dfs, 0.0), 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(interpolate_log_linear(&times, &dfs, 1.0), 0.95, epsilon = 1e-12);
    }
}
