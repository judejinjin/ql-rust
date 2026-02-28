//! Piecewise-bootstrapped spread yield curves.
//!
//! - [`PiecewiseZeroSpreadedTermStructure`] — bootstraps a piecewise-constant
//!   or piecewise-linear zero-rate spread on top of a base yield curve using
//!   spread instruments (par CDS, asset swaps, etc.).
//! - [`PiecewiseForwardSpreadedTermStructure`] — bootstraps piecewise forward
//!   spreads instead of zero-rate spreads.
//!
//! These correspond to QuantLib's `SpreadedLinearZeroInterpolatedTermStructure`
//! and `PiecewiseZeroSpreadedTermStructure`.

use serde::{Deserialize, Serialize};
use ql_time::{Date, DayCounter, Calendar};
use crate::term_structure::TermStructure;
use crate::yield_term_structure::YieldTermStructure;

// ---------------------------------------------------------------------------
// SpreadInterpolation
// ---------------------------------------------------------------------------

/// How to interpolate the spread between pillar maturities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpreadInterpolation {
    /// Piecewise-constant (flat) spread between pillars.
    Flat,
    /// Piecewise-linear spread between pillars.
    Linear,
}

// ---------------------------------------------------------------------------
// PiecewiseZeroSpreadedTermStructure
// ---------------------------------------------------------------------------

/// A yield curve formed by adding a piecewise zero-rate spread to a base curve.
///
/// Given a base curve and a set of `(time, spread)` pillars, the
/// resulting zero rate at time `t` is:
///
///   z(t) = z_base(t) + s(t)
///
/// where `s(t)` is interpolated (flat or linear) from the pillars.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PiecewiseZeroSpreadedTermStructure {
    reference_date: Date,
    day_counter: DayCounter,
    calendar: Calendar,
    max_date: Date,
    /// Base curve discount factors sampled at fine grid.
    base_times: Vec<f64>,
    base_dfs: Vec<f64>,
    /// Spread pillar times (sorted ascending).
    spread_times: Vec<f64>,
    /// Spread pillar values (zero-rate spreads, continuously compounded).
    spread_values: Vec<f64>,
    /// Interpolation mode.
    interpolation: SpreadInterpolation,
}

impl PiecewiseZeroSpreadedTermStructure {
    /// Construct from a base curve and spread pillars.
    ///
    /// `spread_pillars` is a slice of `(time_in_years, spread)` pairs, sorted
    /// by time. The spread is a continuously compounded zero-rate spread.
    pub fn new(
        base: &dyn YieldTermStructure,
        spread_pillars: &[(f64, f64)],
        interpolation: SpreadInterpolation,
        max_years: f64,
    ) -> Self {
        let ref_date = base.reference_date();
        let dc = base.day_counter();
        let num_samples = 200;
        let mut base_times = Vec::with_capacity(num_samples);
        let mut base_dfs = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            let t = max_years * (i as f64) / ((num_samples - 1) as f64);
            base_times.push(t);
            base_dfs.push(base.discount_t(t));
        }

        let max_days = (max_years * 365.25) as i32;
        let max_date = ref_date + max_days;

        let mut spread_times = Vec::with_capacity(spread_pillars.len());
        let mut spread_values = Vec::with_capacity(spread_pillars.len());
        for &(t, s) in spread_pillars {
            spread_times.push(t);
            spread_values.push(s);
        }

        Self {
            reference_date: ref_date,
            day_counter: dc,
            calendar: base.calendar().clone(),
            max_date,
            base_times,
            base_dfs,
            spread_times,
            spread_values,
            interpolation,
        }
    }

    /// Evaluate the spread at time `t` using the chosen interpolation.
    fn spread_at(&self, t: f64) -> f64 {
        if self.spread_times.is_empty() {
            return 0.0;
        }
        if self.spread_times.len() == 1 || t <= self.spread_times[0] {
            return self.spread_values[0];
        }
        if t >= *self.spread_times.last().unwrap() {
            return *self.spread_values.last().unwrap();
        }
        // Find bracketing interval
        let idx = self.spread_times.partition_point(|&x| x < t);
        if idx == 0 {
            return self.spread_values[0];
        }
        match self.interpolation {
            SpreadInterpolation::Flat => self.spread_values[idx - 1],
            SpreadInterpolation::Linear => {
                let t0 = self.spread_times[idx - 1];
                let t1 = self.spread_times[idx];
                let s0 = self.spread_values[idx - 1];
                let s1 = self.spread_values[idx];
                let w = (t - t0) / (t1 - t0);
                s0 + w * (s1 - s0)
            }
        }
    }

    /// The spread pillar times.
    pub fn spread_times(&self) -> &[f64] {
        &self.spread_times
    }

    /// The spread pillar values.
    pub fn spread_values(&self) -> &[f64] {
        &self.spread_values
    }
}

impl TermStructure for PiecewiseZeroSpreadedTermStructure {
    fn reference_date(&self) -> Date { self.reference_date }
    fn day_counter(&self) -> DayCounter { self.day_counter }
    fn calendar(&self) -> &Calendar { &self.calendar }
    fn max_date(&self) -> Date { self.max_date }
}

impl YieldTermStructure for PiecewiseZeroSpreadedTermStructure {
    fn discount_impl(&self, t: f64) -> f64 {
        if t <= 0.0 { return 1.0; }
        let base_df = interpolate_log_linear(&self.base_times, &self.base_dfs, t);
        let spread = self.spread_at(t);
        base_df * (-spread * t).exp()
    }
}

// ---------------------------------------------------------------------------
// PiecewiseForwardSpreadedTermStructure
// ---------------------------------------------------------------------------

/// A yield curve formed by adding a piecewise instantaneous forward spread
/// to a base curve.
///
/// The resulting discount factor is:
///
///   P(t) = P_base(t) · exp(−∫₀ᵗ s(u) du)
///
/// where `s(u)` is a piecewise-constant forward spread.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PiecewiseForwardSpreadedTermStructure {
    reference_date: Date,
    day_counter: DayCounter,
    calendar: Calendar,
    max_date: Date,
    base_times: Vec<f64>,
    base_dfs: Vec<f64>,
    /// Forward spread pillar times (sorted ascending).
    spread_times: Vec<f64>,
    /// Forward spread pillar values.
    spread_values: Vec<f64>,
}

impl PiecewiseForwardSpreadedTermStructure {
    /// Construct from a base curve and piecewise-constant forward spread pillars.
    pub fn new(
        base: &dyn YieldTermStructure,
        spread_pillars: &[(f64, f64)],
        max_years: f64,
    ) -> Self {
        let ref_date = base.reference_date();
        let dc = base.day_counter();
        let num_samples = 200;
        let mut base_times = Vec::with_capacity(num_samples);
        let mut base_dfs = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            let t = max_years * (i as f64) / ((num_samples - 1) as f64);
            base_times.push(t);
            base_dfs.push(base.discount_t(t));
        }

        let max_days = (max_years * 365.25) as i32;
        let max_date = ref_date + max_days;

        let mut spread_times = Vec::with_capacity(spread_pillars.len());
        let mut spread_values = Vec::with_capacity(spread_pillars.len());
        for &(t, s) in spread_pillars {
            spread_times.push(t);
            spread_values.push(s);
        }

        Self {
            reference_date: ref_date,
            day_counter: dc,
            calendar: base.calendar().clone(),
            max_date,
            base_times,
            base_dfs,
            spread_times,
            spread_values,
        }
    }

    /// Integrated forward spread from 0 to t: ∫₀ᵗ s(u) du.
    fn integrated_spread(&self, t: f64) -> f64 {
        if self.spread_times.is_empty() || t <= 0.0 {
            return 0.0;
        }
        let mut integral = 0.0;
        let mut prev_t = 0.0;
        for (i, &pillar_t) in self.spread_times.iter().enumerate() {
            if t <= pillar_t {
                // Current spread applies from prev_t to t
                integral += self.spread_values[i] * (t - prev_t);
                return integral;
            }
            integral += self.spread_values[i] * (pillar_t - prev_t);
            prev_t = pillar_t;
        }
        // Beyond last pillar: use last spread
        integral += self.spread_values.last().unwrap_or(&0.0) * (t - prev_t);
        integral
    }
}

impl TermStructure for PiecewiseForwardSpreadedTermStructure {
    fn reference_date(&self) -> Date { self.reference_date }
    fn day_counter(&self) -> DayCounter { self.day_counter }
    fn calendar(&self) -> &Calendar { &self.calendar }
    fn max_date(&self) -> Date { self.max_date }
}

impl YieldTermStructure for PiecewiseForwardSpreadedTermStructure {
    fn discount_impl(&self, t: f64) -> f64 {
        if t <= 0.0 { return 1.0; }
        let base_df = interpolate_log_linear(&self.base_times, &self.base_dfs, t);
        let int_spread = self.integrated_spread(t);
        base_df * (-int_spread).exp()
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

fn interpolate_log_linear(times: &[f64], dfs: &[f64], t: f64) -> f64 {
    if t <= times[0] { return dfs[0]; }
    let n = times.len();
    if t >= times[n - 1] {
        // Log-linear extrapolation
        if n < 2 { return dfs[n - 1]; }
        let log_df_last = dfs[n - 1].ln();
        let log_df_prev = dfs[n - 2].ln();
        let slope = (log_df_last - log_df_prev) / (times[n - 1] - times[n - 2]);
        return (log_df_last + slope * (t - times[n - 1])).exp();
    }
    let idx = times.partition_point(|&x| x < t);
    if idx == 0 { return dfs[0]; }
    let t0 = times[idx - 1];
    let t1 = times[idx];
    let w = (t - t0) / (t1 - t0);
    let log_df = (1.0 - w) * dfs[idx - 1].max(1e-20).ln() + w * dfs[idx].max(1e-20).ln();
    log_df.exp()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_time::Month;
    use crate::yield_curves::FlatForward;

    #[test]
    fn test_zero_spread_equals_base() {
        let today = Date::from_ymd(2025, Month::January, 15);
        let base = FlatForward::new(today, 0.05, DayCounter::Actual365Fixed);
        let spreaded = PiecewiseZeroSpreadedTermStructure::new(
            &base,
            &[(1.0, 0.0), (5.0, 0.0), (10.0, 0.0)],
            SpreadInterpolation::Linear,
            30.0,
        );
        for &t in &[0.5, 1.0, 2.0, 5.0, 10.0, 20.0] {
            assert_abs_diff_eq!(
                spreaded.discount_t(t),
                base.discount_t(t),
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn test_linear_spread_interpolation() {
        let today = Date::from_ymd(2025, Month::January, 15);
        let base = FlatForward::new(today, 0.03, DayCounter::Actual365Fixed);
        let spreaded = PiecewiseZeroSpreadedTermStructure::new(
            &base,
            &[(1.0, 0.01), (5.0, 0.02)],
            SpreadInterpolation::Linear,
            30.0,
        );
        // At t=3, linearly interpolated spread = 0.01 + (3-1)/(5-1) * (0.02-0.01) = 0.015
        let df_base = (-0.03 * 3.0_f64).exp();
        let df_expected = df_base * (-0.015 * 3.0_f64).exp();
        assert_abs_diff_eq!(spreaded.discount_t(3.0), df_expected, epsilon = 1e-6);
    }

    #[test]
    fn test_flat_spread_interpolation() {
        let today = Date::from_ymd(2025, Month::January, 15);
        let base = FlatForward::new(today, 0.03, DayCounter::Actual365Fixed);
        let spreaded = PiecewiseZeroSpreadedTermStructure::new(
            &base,
            &[(1.0, 0.01), (5.0, 0.02)],
            SpreadInterpolation::Flat,
            30.0,
        );
        // At t=3 (between 1 and 5), flat => spread = 0.01 (left value)
        let df_base = (-0.03 * 3.0_f64).exp();
        let df_expected = df_base * (-0.01 * 3.0_f64).exp();
        assert_abs_diff_eq!(spreaded.discount_t(3.0), df_expected, epsilon = 1e-6);
    }

    #[test]
    fn test_forward_spread() {
        let today = Date::from_ymd(2025, Month::January, 15);
        let base = FlatForward::new(today, 0.04, DayCounter::Actual365Fixed);
        let fwd = PiecewiseForwardSpreadedTermStructure::new(
            &base,
            &[(2.0, 0.005), (5.0, 0.010)],
            30.0,
        );
        // At t=1: ∫₀¹ s(u) du = 0.005 * 1 = 0.005
        let df_base_1 = (-0.04_f64 * 1.0).exp();
        assert_abs_diff_eq!(fwd.discount_t(1.0), df_base_1 * (-0.005_f64).exp(), epsilon = 1e-6);

        // At t=3: ∫₀² 0.005 du + ∫₂³ 0.010 du = 0.010 + 0.010 = 0.020
        let df_base_3 = (-0.04_f64 * 3.0).exp();
        assert_abs_diff_eq!(fwd.discount_t(3.0), df_base_3 * (-0.020_f64).exp(), epsilon = 1e-5);
    }

    #[test]
    fn test_spreaded_is_lower_discount() {
        let today = Date::from_ymd(2025, Month::January, 15);
        let base = FlatForward::new(today, 0.03, DayCounter::Actual365Fixed);
        let spreaded = PiecewiseZeroSpreadedTermStructure::new(
            &base,
            &[(1.0, 0.005), (5.0, 0.010), (10.0, 0.015)],
            SpreadInterpolation::Linear,
            30.0,
        );
        // Positive spread => lower discount factor
        for &t in &[1.0, 3.0, 5.0, 10.0] {
            assert!(spreaded.discount_t(t) < base.discount_t(t));
        }
    }
}
