//! Extended yield curve types.
//!
//! - `CompositeZeroYieldStructure` — additive or multiplicative combination of two curves.
//! - `ImpliedTermStructure` — derived from a base curve with a shifted reference.
//! - `ForwardCurve` — interpolated instantaneous forward rates.
//! - `UltimateForwardTermStructure` — Smith-Wilson extrapolation to an ultimate
//!   forward rate (UFR), as used for Solvency II.
//! - `SpreadedTermStructure` — base curve with additive zero-rate spread.

use ql_core::errors::{QLError, QLResult};
#[allow(unused_imports)]
use ql_math::interpolation::{LinearInterpolation, Interpolation};
use ql_time::{Calendar, Date, DayCounter};

use crate::term_structure::TermStructure;
use crate::yield_term_structure::YieldTermStructure;

// ===========================================================================
// CompositeZeroYieldStructure
// ===========================================================================

/// A composite yield term structure formed by adding (or multiplying) two
/// yield curves' discount factors.
///
/// The most common usage is an additive composite where the resulting
/// zero rate is `z₁(t) + z₂(t)`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CompositeZeroYieldStructure {
    reference_date: Date,
    day_counter: DayCounter,
    calendar: Calendar,
    max_date: Date,
    /// Base curve zero rates at sample points.
    base_rates: Vec<f64>,
    /// Spread curve zero rates at sample points.
    spread_rates: Vec<f64>,
    /// Sample times.
    times: Vec<f64>,
    /// Whether to add (true) or multiply (false) discount factors.
    additive: bool,
}

impl CompositeZeroYieldStructure {
    /// Create a composite curve by sampling two yield term structures.
    ///
    /// If `additive`, the composite zero rate is `z_base(t) + z_spread(t)`.
    /// Otherwise, discount factors are multiplied: `df = df_base · df_spread`.
    pub fn new(
        base: &dyn YieldTermStructure,
        spread: &dyn YieldTermStructure,
        additive: bool,
        num_samples: usize,
        max_years: f64,
    ) -> Self {
        let ref_date = base.reference_date();
        let dc = base.day_counter();
        let mut times = Vec::with_capacity(num_samples);
        let mut base_rates = Vec::with_capacity(num_samples);
        let mut spread_rates = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            let t = max_years * (i as f64) / ((num_samples - 1) as f64);
            times.push(t);

            // Infer zero rate from discount factor
            let df_b = base.discount_t(t);
            let df_s = spread.discount_t(t);
            let zr_b = if t > 1e-10 { -df_b.ln() / t } else { 0.0 };
            let zr_s = if t > 1e-10 { -df_s.ln() / t } else { 0.0 };
            base_rates.push(zr_b);
            spread_rates.push(zr_s);
        }

        let max_days = (max_years * 365.25) as i32;
        let max_date = ref_date + max_days;

        Self {
            reference_date: ref_date,
            day_counter: dc,
            calendar: base.calendar().clone(),
            max_date,
            base_rates,
            spread_rates,
            times,
            additive,
        }
    }

    /// Composite zero rate at time `t`.
    fn composite_zero_rate(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        let base_zr = interpolate_linear(&self.times, &self.base_rates, t);
        let spread_zr = interpolate_linear(&self.times, &self.spread_rates, t);

        if self.additive {
            base_zr + spread_zr
        } else {
            // Multiplicative: df = df_base * df_spread = exp(-(zr_b + zr_s)*t)
            base_zr + spread_zr
        }
    }
}

impl TermStructure for CompositeZeroYieldStructure {
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

impl YieldTermStructure for CompositeZeroYieldStructure {
    fn discount_impl(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }
        (-self.composite_zero_rate(t) * t).exp()
    }
}

// ===========================================================================
// ImpliedTermStructure
// ===========================================================================

/// A yield term structure implied from a base curve with a shifted reference
/// date.
///
/// If the base curve has reference date $T_0$ and the implied curve has
/// reference date $T_1 > T_0$, then:
/// $$\text{df}_{\text{impl}}(T) = \frac{\text{df}_{\text{base}}(T)}{\text{df}_{\text{base}}(T_1)}$$
///
/// This is useful for forward-starting instruments.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(dead_code)]
pub struct ImpliedTermStructure {
    base_reference_date: Date,
    new_reference_date: Date,
    day_counter: DayCounter,
    calendar: Calendar,
    max_date: Date,
    /// Discount factor at the new reference date on the base curve.
    df_at_new_ref: f64,
    /// Base curve sampled discount factors for interpolation.
    base_times: Vec<f64>,
    base_dfs: Vec<f64>,
}

impl ImpliedTermStructure {
    /// Create an implied term structure from a base curve.
    ///
    /// The `new_reference_date` should be after the base curve's reference date.
    pub fn new(
        base: &dyn YieldTermStructure,
        new_reference_date: Date,
        max_years: f64,
    ) -> Self {
        let base_ref = base.reference_date();
        let dc = base.day_counter();
        let df_at_new_ref = base.discount(new_reference_date);

        // Sample base curve
        let num_samples = 100;
        let mut base_times = Vec::with_capacity(num_samples);
        let mut base_dfs = Vec::with_capacity(num_samples);
        let t_shift = dc.year_fraction(base_ref, new_reference_date);

        for i in 0..num_samples {
            let t = max_years * (i as f64) / ((num_samples - 1) as f64);
            base_times.push(t);
            let base_t = t + t_shift;
            let df = base.discount_t(base_t);
            base_dfs.push(df / df_at_new_ref);
        }

        let max_days = (max_years * 365.25) as i32;
        let max_date = new_reference_date + max_days;

        Self {
            base_reference_date: base_ref,
            new_reference_date,
            day_counter: dc,
            calendar: base.calendar().clone(),
            max_date,
            df_at_new_ref,
            base_times,
            base_dfs,
        }
    }

    /// The base curve's reference date.
    pub fn base_reference_date(&self) -> Date {
        self.base_reference_date
    }
}

impl TermStructure for ImpliedTermStructure {
    fn reference_date(&self) -> Date {
        self.new_reference_date
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

impl YieldTermStructure for ImpliedTermStructure {
    fn discount_impl(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }
        // Interpolate from sampled data
        interpolate_log_linear_vec(&self.base_times, &self.base_dfs, t)
    }
}

// ===========================================================================
// ForwardCurve — interpolated instantaneous forward rates
// ===========================================================================

/// A yield curve defined by interpolated instantaneous forward rates.
///
/// The discount factor is computed as:
/// $$\text{df}(t) = \exp\!\left(-\int_0^t f(s)\,ds\right)$$
/// where the integral is computed by piecewise-linear trapezoidal rule.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ForwardCurve {
    reference_date: Date,
    day_counter: DayCounter,
    calendar: Calendar,
    /// Times (year fractions).
    times: Vec<f64>,
    /// Instantaneous forward rates at each node.
    forwards: Vec<f64>,
    /// Cumulative integrals of forward rates: `integrals[i] = ∫₀^{t_i} f(s) ds`.
    integrals: Vec<f64>,
    max_date: Date,
}

impl ForwardCurve {
    /// Build a forward curve from dates and instantaneous forward rates.
    pub fn new(
        dates: Vec<Date>,
        forwards: Vec<f64>,
        day_counter: DayCounter,
    ) -> QLResult<Self> {
        if dates.len() != forwards.len() || dates.len() < 2 {
            return Err(QLError::InvalidArgument(
                "need at least 2 matching (date, forward) pairs".into(),
            ));
        }

        let reference_date = dates[0];
        let times: Vec<f64> = dates
            .iter()
            .map(|&d| day_counter.year_fraction(reference_date, d))
            .collect();

        // Precompute cumulative integrals using trapezoidal rule
        let mut integrals = vec![0.0; times.len()];
        for i in 1..times.len() {
            let dt = times[i] - times[i - 1];
            integrals[i] = integrals[i - 1] + 0.5 * (forwards[i - 1] + forwards[i]) * dt;
        }

        let max_date = *dates.last().unwrap();

        Ok(Self {
            reference_date,
            day_counter,
            calendar: Calendar::NullCalendar,
            times,
            forwards,
            integrals,
            max_date,
        })
    }

    /// The instantaneous forward rate at time `t` (linearly interpolated).
    pub fn forward_rate_at(&self, t: f64) -> f64 {
        interpolate_linear(&self.times, &self.forwards, t)
    }

    /// Cumulative integral of forward rates from 0 to `t`.
    fn integral(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        if t >= *self.times.last().unwrap() {
            // Extrapolate flat from last forward rate
            let n = self.times.len();
            let last_integral = self.integrals[n - 1];
            let last_fwd = self.forwards[n - 1];
            return last_integral + last_fwd * (t - self.times[n - 1]);
        }

        // Find segment
        let idx = self
            .times
            .partition_point(|&ti| ti < t)
            .saturating_sub(1);

        if idx >= self.times.len() - 1 {
            return *self.integrals.last().unwrap();
        }

        let t0 = self.times[idx];
        let t1 = self.times[idx + 1];
        let dt = t1 - t0;
        if dt < 1e-15 {
            return self.integrals[idx];
        }

        let frac = (t - t0) / dt;
        let f_t = self.forwards[idx] + frac * (self.forwards[idx + 1] - self.forwards[idx]);
        self.integrals[idx] + 0.5 * (self.forwards[idx] + f_t) * (t - t0)
    }
}

impl TermStructure for ForwardCurve {
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

impl YieldTermStructure for ForwardCurve {
    fn discount_impl(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }
        (-self.integral(t)).exp()
    }
}

// ===========================================================================
// UltimateForwardTermStructure — Smith-Wilson extrapolation
// ===========================================================================

/// Smith-Wilson extrapolation to an ultimate forward rate (UFR).
///
/// Used in Solvency II (EIOPA) to extrapolate the risk-free yield curve
/// beyond the last liquid point (LLP) toward a specified UFR.
///
/// The kernel function is:
/// $$W(t, u_j) = e^{-\text{ufr}(t+u_j)} \left[\alpha\min(t, u_j) - \frac{1}{2}e^{-\alpha|t-u_j|} + \frac{1}{2}e^{-\alpha(t+u_j)}\right]$$
///
/// The discount curve is:
/// $$P(t) = e^{-\text{ufr}\cdot t} + \sum_j \zeta_j \cdot W(t, u_j)$$
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UltimateForwardTermStructure {
    reference_date: Date,
    day_counter: DayCounter,
    calendar: Calendar,
    max_date: Date,
    /// Ultimate forward rate (continuously compounded).
    ufr: f64,
    /// Convergence speed parameter.
    alpha: f64,
    /// Knot maturities (year fractions).
    knots: Vec<f64>,
    /// Calibrated coefficients ζ_j.
    zeta: Vec<f64>,
}

impl UltimateForwardTermStructure {
    /// Calibrate a Smith-Wilson extrapolation.
    ///
    /// # Arguments
    /// * `reference_date` — valuation date
    /// * `maturities` — year fractions of observed instruments
    /// * `discount_factors` — observed discount factors at those maturities
    /// * `ufr` — ultimate forward rate (continuously compounded)
    /// * `alpha` — convergence speed (typically 0.1–0.2)
    /// * `day_counter` — day count convention
    /// * `max_years` — maximum extrapolation horizon
    pub fn new(
        reference_date: Date,
        maturities: &[f64],
        discount_factors: &[f64],
        ufr: f64,
        alpha: f64,
        day_counter: DayCounter,
        max_years: f64,
    ) -> QLResult<Self> {
        let n = maturities.len();
        if n != discount_factors.len() || n == 0 {
            return Err(QLError::InvalidArgument(
                "maturities and discount_factors must have equal non-zero length".into(),
            ));
        }

        let knots = maturities.to_vec();

        // Build kernel matrix W_{ij} = W(u_i, u_j)
        let mut w_matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                w_matrix[i][j] = Self::kernel(knots[i], knots[j], ufr, alpha);
            }
        }

        // Target: P(u_i) - e^{-ufr * u_i} = Σ_j ζ_j W(u_i, u_j)
        let mut rhs = vec![0.0; n];
        for i in 0..n {
            rhs[i] = discount_factors[i] - (-ufr * knots[i]).exp();
        }

        // Solve W * ζ = rhs using Gaussian elimination
        let zeta = Self::solve_linear(&w_matrix, &rhs)?;

        let max_days = (max_years * 365.25) as i32;
        let max_date = reference_date + max_days;

        Ok(Self {
            reference_date,
            day_counter,
            calendar: Calendar::NullCalendar,
            max_date,
            ufr,
            alpha,
            knots,
            zeta,
        })
    }

    /// The Smith-Wilson kernel function.
    fn kernel(t: f64, u: f64, ufr: f64, alpha: f64) -> f64 {
        let min_tu = t.min(u);
        let abs_diff = (t - u).abs();
        let sum_tu = t + u;

        (-ufr * sum_tu).exp()
            * (alpha * min_tu - 0.5 * (-alpha * abs_diff).exp()
                + 0.5 * (-alpha * sum_tu).exp())
    }

    /// Solve a dense linear system using Gaussian elimination with partial pivoting.
    #[allow(clippy::needless_range_loop)]
    fn solve_linear(a: &[Vec<f64>], b: &[f64]) -> QLResult<Vec<f64>> {
        let n = b.len();
        let mut aug: Vec<Vec<f64>> = a
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let mut r = row.clone();
                r.push(b[i]);
                r
            })
            .collect();

        // Forward elimination with partial pivoting
        for col in 0..n {
            // Find pivot
            let mut max_row = col;
            let mut max_val = aug[col][col].abs();
            for row in (col + 1)..n {
                if aug[row][col].abs() > max_val {
                    max_val = aug[row][col].abs();
                    max_row = row;
                }
            }
            if max_val < 1e-15 {
                return Err(QLError::Other(
                    "Smith-Wilson kernel matrix is singular".into(),
                ));
            }
            aug.swap(col, max_row);

            let pivot = aug[col][col];
            for row in (col + 1)..n {
                let factor = aug[row][col] / pivot;
                for j in col..=n {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }

        // Back substitution
        let mut x = vec![0.0; n];
        for col in (0..n).rev() {
            let mut s = aug[col][n];
            for j in (col + 1)..n {
                s -= aug[col][j] * x[j];
            }
            x[col] = s / aug[col][col];
        }

        Ok(x)
    }

    /// The ultimate forward rate.
    pub fn ufr(&self) -> f64 {
        self.ufr
    }

    /// The convergence speed parameter.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
}

impl TermStructure for UltimateForwardTermStructure {
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

impl YieldTermStructure for UltimateForwardTermStructure {
    fn discount_impl(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }
        let mut df = (-self.ufr * t).exp();
        for (j, &u_j) in self.knots.iter().enumerate() {
            df += self.zeta[j] * Self::kernel(t, u_j, self.ufr, self.alpha);
        }
        df.max(1e-15) // ensure positive
    }
}

// ===========================================================================
// SpreadedTermStructure
// ===========================================================================

/// A yield term structure that adds a constant zero-rate spread to a base curve.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpreadedTermStructure {
    reference_date: Date,
    day_counter: DayCounter,
    calendar: Calendar,
    max_date: Date,
    /// Base discount factors sampled.
    base_times: Vec<f64>,
    base_dfs: Vec<f64>,
    /// Additive spread (continuously compounded).
    spread: f64,
}

impl SpreadedTermStructure {
    /// Create a spreaded term structure.
    pub fn new(
        base: &dyn YieldTermStructure,
        spread: f64,
        max_years: f64,
    ) -> Self {
        let ref_date = base.reference_date();
        let dc = base.day_counter();
        let num_samples = 100;
        let mut base_times = Vec::with_capacity(num_samples);
        let mut base_dfs = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            let t = max_years * (i as f64) / ((num_samples - 1) as f64);
            base_times.push(t);
            base_dfs.push(base.discount_t(t));
        }

        let max_days = (max_years * 365.25) as i32;
        let max_date = ref_date + max_days;

        Self {
            reference_date: ref_date,
            day_counter: dc,
            calendar: base.calendar().clone(),
            max_date,
            base_times,
            base_dfs,
            spread,
        }
    }

    /// The spread value.
    pub fn spread(&self) -> f64 {
        self.spread
    }
}

impl TermStructure for SpreadedTermStructure {
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

impl YieldTermStructure for SpreadedTermStructure {
    fn discount_impl(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }
        let base_df = interpolate_log_linear_vec(&self.base_times, &self.base_dfs, t);
        base_df * (-self.spread * t).exp()
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Simple linear interpolation with flat extrapolation.
fn interpolate_linear(times: &[f64], values: &[f64], t: f64) -> f64 {
    if times.is_empty() {
        return 0.0;
    }
    if t <= times[0] {
        return values[0];
    }
    if t >= *times.last().unwrap() {
        return *values.last().unwrap();
    }

    let idx = times.partition_point(|&ti| ti < t).saturating_sub(1);
    let idx = idx.min(times.len() - 2);
    let t0 = times[idx];
    let t1 = times[idx + 1];
    let dt = t1 - t0;
    if dt < 1e-15 {
        return values[idx];
    }
    let frac = (t - t0) / dt;
    values[idx] + frac * (values[idx + 1] - values[idx])
}

/// Log-linear interpolation on a vector of (time, df) pairs.
fn interpolate_log_linear_vec(times: &[f64], dfs: &[f64], t: f64) -> f64 {
    if times.is_empty() {
        return 1.0;
    }
    if t <= times[0] {
        return dfs[0];
    }
    let n = times.len();
    if t >= times[n - 1] {
        if n < 2 {
            return dfs[n - 1];
        }
        let ln1 = dfs[n - 2].ln();
        let ln2 = dfs[n - 1].ln();
        let dt = times[n - 1] - times[n - 2];
        if dt < 1e-15 {
            return dfs[n - 1];
        }
        let slope = (ln2 - ln1) / dt;
        return (ln2 + slope * (t - times[n - 1])).exp();
    }

    let idx = times.partition_point(|&ti| ti < t).saturating_sub(1);
    let idx = idx.min(n - 2);
    let t0 = times[idx];
    let t1 = times[idx + 1];
    let dt = t1 - t0;
    if dt < 1e-15 {
        return dfs[idx];
    }
    let frac = (t - t0) / dt;
    let ln0 = dfs[idx].ln();
    let ln1 = dfs[idx + 1].ln();
    (ln0 + frac * (ln1 - ln0)).exp()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_time::Month;
    use crate::yield_curves::FlatForward;

    fn ref_date() -> Date {
        Date::from_ymd(2025, Month::January, 1)
    }

    #[test]
    fn composite_additive_flat_curves() {
        let base = FlatForward::new(ref_date(), 0.03, DayCounter::Actual365Fixed);
        let spread = FlatForward::new(ref_date(), 0.01, DayCounter::Actual365Fixed);

        let composite = CompositeZeroYieldStructure::new(&base, &spread, true, 50, 30.0);

        // Composite zero rate ≈ 3% + 1% = 4%
        let df5 = composite.discount_impl(5.0);
        let expected = (-0.04_f64 * 5.0).exp();
        assert_abs_diff_eq!(df5, expected, epsilon = 1e-6);
    }

    #[test]
    fn composite_discount_at_zero() {
        let base = FlatForward::new(ref_date(), 0.05, DayCounter::Actual365Fixed);
        let spread = FlatForward::new(ref_date(), 0.02, DayCounter::Actual365Fixed);
        let composite = CompositeZeroYieldStructure::new(&base, &spread, true, 50, 30.0);
        assert_abs_diff_eq!(composite.discount_impl(0.0), 1.0, epsilon = 1e-15);
    }

    #[test]
    fn implied_term_structure_forward_start() {
        let base = FlatForward::new(ref_date(), 0.05, DayCounter::Actual365Fixed);
        let fwd_date = Date::from_ymd(2026, Month::January, 1); // 1Y forward
        let implied = ImpliedTermStructure::new(&base, fwd_date, 30.0);

        // For a flat curve, the implied curve should also be flat at 5%
        let df2 = implied.discount_impl(2.0); // 2Y from fwd_date
        let expected = (-0.05_f64 * 2.0).exp();
        assert_abs_diff_eq!(df2, expected, epsilon = 1e-4);
    }

    #[test]
    fn implied_term_structure_discount_at_zero() {
        let base = FlatForward::new(ref_date(), 0.05, DayCounter::Actual365Fixed);
        let fwd_date = Date::from_ymd(2026, Month::January, 1);
        let implied = ImpliedTermStructure::new(&base, fwd_date, 30.0);
        assert_abs_diff_eq!(implied.discount_impl(0.0), 1.0, epsilon = 1e-15);
        assert_eq!(implied.reference_date(), fwd_date);
    }

    #[test]
    fn forward_curve_flat() {
        let dates = vec![
            ref_date(),
            Date::from_ymd(2030, Month::January, 1),
            Date::from_ymd(2035, Month::January, 1),
        ];
        let fwds = vec![0.05, 0.05, 0.05]; // flat 5% forward
        let curve = ForwardCurve::new(dates, fwds, DayCounter::Actual365Fixed).unwrap();

        // With flat forward, df(t) = e^{-0.05*t}
        let df3 = curve.discount_impl(3.0);
        assert_abs_diff_eq!(df3, (-0.05_f64 * 3.0).exp(), epsilon = 1e-6);
    }

    #[test]
    fn forward_curve_upward() {
        let dates = vec![
            ref_date(),
            Date::from_ymd(2026, Month::January, 1),
            Date::from_ymd(2030, Month::January, 1),
            Date::from_ymd(2035, Month::January, 1),
        ];
        let fwds = vec![0.03, 0.04, 0.05, 0.06];
        let curve = ForwardCurve::new(dates, fwds, DayCounter::Actual365Fixed).unwrap();

        // Discount should be monotonically decreasing
        let df1 = curve.discount_impl(1.0);
        let df5 = curve.discount_impl(5.0);
        let df10 = curve.discount_impl(10.0);
        assert!(df1 > df5 && df5 > df10);
        assert!(df1 < 1.0 && df10 > 0.0);
    }

    #[test]
    fn forward_curve_accessor() {
        let dates = vec![
            ref_date(),
            Date::from_ymd(2030, Month::January, 1),
        ];
        let fwds = vec![0.03, 0.06];
        let curve = ForwardCurve::new(dates, fwds, DayCounter::Actual365Fixed).unwrap();

        // At t=0, forward should be 3%
        assert_abs_diff_eq!(curve.forward_rate_at(0.0), 0.03, epsilon = 1e-10);
    }

    #[test]
    fn smith_wilson_reproduces_inputs() {
        let mats = vec![1.0, 2.0, 3.0, 5.0, 7.0, 10.0];
        let ufr = 0.042;
        let rate = 0.03;
        // Generate discount factors from a flat 3% curve
        let dfs: Vec<f64> = mats.iter().map(|&t| (-rate * t as f64).exp()).collect();

        let sw = UltimateForwardTermStructure::new(
            ref_date(),
            &mats,
            &dfs,
            ufr,
            0.1,
            DayCounter::Actual365Fixed,
            60.0,
        )
        .unwrap();

        // At calibration nodes, should reproduce input discount factors
        for (&t, &df) in mats.iter().zip(dfs.iter()) {
            assert_abs_diff_eq!(sw.discount_impl(t), df, epsilon = 1e-8);
        }
    }

    #[test]
    fn smith_wilson_converges_to_ufr() {
        let mats = vec![1.0, 3.0, 5.0, 10.0, 20.0];
        let rate = 0.025;
        let ufr = 0.042;
        let dfs: Vec<f64> = mats.iter().map(|&t| (-rate * t as f64).exp()).collect();

        let sw = UltimateForwardTermStructure::new(
            ref_date(),
            &mats,
            &dfs,
            ufr,
            0.1,
            DayCounter::Actual365Fixed,
            200.0,
        )
        .unwrap();

        // At very long maturity, the forward rate should converge to UFR
        let t_far = 150.0;
        let dt = 0.01;
        let df1 = sw.discount_impl(t_far);
        let df2 = sw.discount_impl(t_far + dt);
        let fwd = -(df2 / df1).ln() / dt;
        assert_abs_diff_eq!(fwd, ufr, epsilon = 0.01);
    }

    #[test]
    fn spreaded_term_structure() {
        let base = FlatForward::new(ref_date(), 0.03, DayCounter::Actual365Fixed);
        let spreaded = SpreadedTermStructure::new(&base, 0.02, 30.0);

        // Total rate = 3% + 2% = 5%
        let df5 = spreaded.discount_impl(5.0);
        let expected = (-0.05_f64 * 5.0).exp();
        assert_abs_diff_eq!(df5, expected, epsilon = 1e-4);
    }

    #[test]
    fn spreaded_term_structure_zero_spread() {
        let base = FlatForward::new(ref_date(), 0.04, DayCounter::Actual365Fixed);
        let spreaded = SpreadedTermStructure::new(&base, 0.0, 30.0);

        // Should match base curve
        for t in [1.0, 5.0, 10.0, 20.0] {
            let df_base = base.discount_impl(t);
            let df_sp = spreaded.discount_impl(t);
            assert_abs_diff_eq!(df_base, df_sp, epsilon = 1e-6);
        }
    }
}
