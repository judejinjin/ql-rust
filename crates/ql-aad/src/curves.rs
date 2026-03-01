//! AD-aware yield curve — generic discount factors for curve-level AAD.
//!
//! The key idea: pillar discount factors (or zero rates) are `T: Number`
//! inputs. The curve interpolates in log-DF space, and all operations
//! (ln, exp, linear interpolation) flow through the AD tape.
//!
//! This enables **exact** DV01, key-rate durations, and curve sensitivities
//! via a single reverse-mode adjoint pass — no bump-and-reprice needed.
//!
//! # Example
//!
//! ```
//! use ql_aad::curves::DiscountCurveAD;
//! use ql_aad::Number;
//!
//! // Build curve from pillar zero rates
//! let times = vec![0.25, 0.5, 1.0, 2.0, 5.0, 10.0];
//! let rates = vec![0.02, 0.025, 0.03, 0.032, 0.035, 0.04];
//! let curve = DiscountCurveAD::from_zero_rates(&times, &rates);
//!
//! // Query discount factor (f64 — no AD)
//! let df: f64 = curve.discount(1.5);
//! assert!((df - (-0.031 * 1.5_f64).exp()).abs() < 0.01);
//! ```

use crate::number::Number;

// ===========================================================================
// Helpers
// ===========================================================================

/// Binary search for the segment containing `x` in sorted `xs`.
/// Returns `i` such that `xs[i] <= x < xs[i+1]`, clamped to valid range.
fn locate(xs: &[f64], x: f64) -> usize {
    if x <= xs[0] {
        return 0;
    }
    if x >= xs[xs.len() - 1] {
        return xs.len().saturating_sub(2);
    }
    match xs.binary_search_by(|a| a.partial_cmp(&x).unwrap()) {
        Ok(i) => i.min(xs.len() - 2),
        Err(i) => (i - 1).min(xs.len() - 2),
    }
}

/// Linear interpolation with T-valued y-data and f64 x-query.
///
/// This is the complementary pattern to `interp::LinearInterp::eval`:
/// - `LinearInterp::eval<T>(x: T) -> T`: y-data is f64, x is generic
///   (for "evaluate at AD point" scenarios)
/// - `lerp_t(xs, ys, x) -> T`: y-data is generic T, x is f64
///   (for "curve sensitivity" scenarios where pillar values are AD inputs)
fn lerp_t<T: Number>(xs: &[f64], ys: &[T], x: f64) -> T {
    debug_assert!(xs.len() == ys.len() && xs.len() >= 2);
    let i = locate(xs, x);
    let dx = xs[i + 1] - xs[i];
    if dx.abs() < 1e-30 {
        return ys[i];
    }
    let frac = (x - xs[i]) / dx;
    ys[i] * T::from_f64(1.0 - frac) + ys[i + 1] * T::from_f64(frac)
}

// ===========================================================================
// DiscountCurveAD
// ===========================================================================

/// AD-aware discount curve with generic pillar values.
///
/// Constructed from pillar times and either zero rates or discount factors.
/// Log-discount factors are interpolated linearly (equivalent to log-linear
/// interpolation on DFs — the standard method).
///
/// When instantiated with `T = AReal`, each pillar value is a tape input
/// and the adjoint pass computes exact sensitivities to all pillars.
pub struct DiscountCurveAD<T: Number> {
    /// Pillar maturities (ascending, strictly positive).
    times: Vec<f64>,
    /// ln(DF) at each pillar — interpolated linearly in this space.
    log_dfs: Vec<T>,
}

impl<T: Number> DiscountCurveAD<T> {
    /// Build from pillar times and zero rates: `DF(t_i) = exp(-r_i · t_i)`.
    pub fn from_zero_rates(times: &[f64], zero_rates: &[T]) -> Self {
        assert!(
            times.len() == zero_rates.len() && times.len() >= 2,
            "need at least 2 pillars"
        );
        let log_dfs: Vec<T> = times
            .iter()
            .zip(zero_rates.iter())
            .map(|(&t, &r)| T::from_f64(-t) * r)
            .collect();
        Self {
            times: times.to_vec(),
            log_dfs,
        }
    }

    /// Build from pillar times and discount factors directly.
    pub fn from_dfs(times: &[f64], dfs: &[T]) -> Self {
        assert!(
            times.len() == dfs.len() && times.len() >= 2,
            "need at least 2 pillars"
        );
        let log_dfs: Vec<T> = dfs.iter().map(|&df| df.ln()).collect();
        Self {
            times: times.to_vec(),
            log_dfs,
        }
    }

    /// Build from pillar times and instantaneous forward rates.
    ///
    /// Assumes piecewise-constant forwards between pillars.
    /// `DF(t_i) = exp(-Σ f_j · (t_j - t_{j-1}))`.
    pub fn from_forwards(times: &[f64], fwd_rates: &[T]) -> Self {
        assert!(
            times.len() == fwd_rates.len() && times.len() >= 2,
            "need at least 2 pillars"
        );
        let mut log_dfs = Vec::with_capacity(times.len());
        // First pillar
        let mut cum = T::from_f64(-times[0]) * fwd_rates[0];
        log_dfs.push(cum);
        // Subsequent pillars
        for i in 1..times.len() {
            let dt = times[i] - times[i - 1];
            cum = cum + T::from_f64(-dt) * fwd_rates[i];
            log_dfs.push(cum);
        }
        Self {
            times: times.to_vec(),
            log_dfs,
        }
    }

    /// Number of pillar points.
    pub fn num_pillars(&self) -> usize {
        self.times.len()
    }

    /// Pillar times (immutable reference).
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    /// Discount factor at time `t` via log-linear interpolation.
    pub fn discount(&self, t: f64) -> T {
        if t <= 0.0 {
            return T::one();
        }
        lerp_t(&self.times, &self.log_dfs, t).exp()
    }

    /// Continuously compounded zero rate at time `t`: `r(t) = -ln(DF(t)) / t`.
    pub fn zero_rate(&self, t: f64) -> T {
        if t <= 1e-14 {
            // At t=0, return the short rate (first pillar rate)
            return self.log_dfs[0] * T::from_f64(-1.0 / self.times[0]);
        }
        let log_df = lerp_t(&self.times, &self.log_dfs, t);
        log_df * T::from_f64(-1.0 / t)
    }

    /// Forward rate between `t1` and `t2`:
    /// `f(t1, t2) = -[ln(DF(t2)) - ln(DF(t1))] / (t2 - t1)`.
    pub fn forward_rate(&self, t1: f64, t2: f64) -> T {
        assert!(t2 > t1, "t2 must be > t1");
        let log_df1 = lerp_t(&self.times, &self.log_dfs, t1);
        let log_df2 = lerp_t(&self.times, &self.log_dfs, t2);
        (log_df1 - log_df2) * T::from_f64(1.0 / (t2 - t1))
    }

    /// Instantaneous forward rate at time `t`: `f(t) = -d/dt ln(DF(t))`.
    ///
    /// Since log-DFs are piecewise linear, the instantaneous forward is
    /// piecewise constant (the negative slope of each segment).
    pub fn inst_forward(&self, t: f64) -> T {
        let i = locate(&self.times, t);
        let dt = self.times[i + 1] - self.times[i];
        if dt.abs() < 1e-30 {
            return T::zero();
        }
        (self.log_dfs[i] - self.log_dfs[i + 1]) * T::from_f64(1.0 / dt)
    }
}

// ===========================================================================
// Convenience constructors for f64 (non-AD)
// ===========================================================================

impl DiscountCurveAD<f64> {
    /// Build from pillar times and zero rates (f64 convenience).
    pub fn from_zero_rates_f64(times: &[f64], zero_rates: &[f64]) -> Self {
        Self::from_zero_rates(times, zero_rates)
    }

    /// Build from pillar times and discount factors (f64 convenience).
    pub fn from_dfs_f64(times: &[f64], dfs: &[f64]) -> Self {
        Self::from_dfs(times, dfs)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn sample_times() -> Vec<f64> {
        vec![0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    }

    fn sample_rates() -> Vec<f64> {
        vec![0.020, 0.025, 0.030, 0.032, 0.035, 0.040]
    }

    #[test]
    fn discount_at_pillars() {
        let times = sample_times();
        let rates = sample_rates();
        let curve = DiscountCurveAD::from_zero_rates(&times, &rates);

        for (&t, &r) in times.iter().zip(&rates) {
            let df: f64 = curve.discount(t);
            let expected = (-r * t).exp();
            assert_abs_diff_eq!(df, expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn discount_interpolated() {
        let times = sample_times();
        let rates = sample_rates();
        let curve = DiscountCurveAD::from_zero_rates(&times, &rates);

        // At t=1.5 (between 1.0 and 2.0 pillars)
        let df: f64 = curve.discount(1.5);
        // Log-linear: ln(df(1.5)) = lerp(ln(df(1.0)), ln(df(2.0)), 0.5)
        let ln_df1 = -0.030 * 1.0;
        let ln_df2 = -0.032 * 2.0;
        let expected = (0.5 * ln_df1 + 0.5 * ln_df2).exp();
        assert_abs_diff_eq!(df, expected, epsilon = 1e-12);
    }

    #[test]
    fn discount_at_zero() {
        let times = sample_times();
        let rates = sample_rates();
        let curve = DiscountCurveAD::from_zero_rates(&times, &rates);
        let df: f64 = curve.discount(0.0);
        assert_abs_diff_eq!(df, 1.0, epsilon = 1e-14);
    }

    #[test]
    fn zero_rate_at_pillars() {
        let times = sample_times();
        let rates = sample_rates();
        let curve = DiscountCurveAD::from_zero_rates(&times, &rates);

        for (&t, &r) in times.iter().zip(&rates) {
            let rate: f64 = curve.zero_rate(t);
            assert_abs_diff_eq!(rate, r, epsilon = 1e-12);
        }
    }

    #[test]
    fn forward_rate_between_pillars() {
        let times = sample_times();
        let rates = sample_rates();
        let curve = DiscountCurveAD::from_zero_rates(&times, &rates);

        // Forward between 1y and 2y
        let fwd: f64 = curve.forward_rate(1.0, 2.0);
        // f(1,2) = -(ln(df(2)) - ln(df(1))) / 1 = (r1*1 - r2*2) * (-1)
        //        = (0.032*2 - 0.030*1) = 0.034
        let expected = 0.032 * 2.0 - 0.030 * 1.0;
        assert_abs_diff_eq!(fwd, expected, epsilon = 1e-12);
    }

    #[test]
    fn from_dfs_roundtrip() {
        let times = sample_times();
        let rates = sample_rates();
        let dfs: Vec<f64> = times.iter().zip(&rates)
            .map(|(&t, &r)| (-r * t).exp()).collect();

        let curve = DiscountCurveAD::from_dfs(&times, &dfs);

        for (&t, &r) in times.iter().zip(&rates) {
            let rate: f64 = curve.zero_rate(t);
            assert_abs_diff_eq!(rate, r, epsilon = 1e-10);
        }
    }

    #[test]
    fn curve_sensitivity_areal() {
        use crate::tape::{with_tape, adjoint_tl, AReal};

        let times = vec![1.0, 2.0, 5.0];
        let rates_f64 = vec![0.03, 0.035, 0.04];

        // Compute ∂NPV/∂r_i for a single cashflow at t=3.0
        // NPV = 100 * DF(3.0)
        let (npv, grad) = with_tape(|tape| {
            let rates: Vec<AReal> = rates_f64.iter().map(|&r| tape.input(r)).collect();
            let curve = DiscountCurveAD::from_zero_rates(&times, &rates);

            let df = curve.discount(3.0);
            let npv = df * AReal::from_f64(100.0);

            let adj = adjoint_tl(npv);
            let g: Vec<f64> = rates.iter().map(|a| adj[a.idx]).collect();
            (npv.val, g)
        });

        // NPV should be 100 * DF(3.0)
        assert!(npv > 0.0 && npv < 100.0);

        // All sensitivities should be negative (higher rate → lower DF → lower NPV)
        for (i, &g) in grad.iter().enumerate() {
            assert!(g < 0.0 || g.abs() < 1e-10,
                    "gradient[{}] = {} should be <= 0", i, g);
        }

        // The 2y and 5y pillars bracket t=3.0, so they should have nonzero sensitivities
        // The 1y pillar should have zero sensitivity (t=3.0 is outside [1, 2) segment)
        assert!(grad[0].abs() < 1e-10, "1y pillar should have ~zero sensitivity for t=3");
        assert!(grad[1].abs() > 1e-5, "2y pillar should have nonzero sensitivity for t=3");
        assert!(grad[2].abs() > 1e-5, "5y pillar should have nonzero sensitivity for t=3");
    }

    #[test]
    fn curve_sensitivity_dual() {
        use crate::dual::Dual;

        let times = vec![1.0, 2.0, 5.0];

        // Sensitivity of DF(3.0) to the 2y rate
        // Seed the 2y rate with derivative 1
        let rates = vec![
            Dual::constant(0.03),   // 1y
            Dual::variable(0.035),  // 2y — seeded
            Dual::constant(0.04),   // 5y
        ];
        let curve = DiscountCurveAD::from_zero_rates(&times, &rates);
        let df: Dual = curve.discount(3.0);

        // df.dot = ∂DF(3)/∂r_2y
        assert!(df.dot < 0.0, "∂DF/∂r should be negative, got {}", df.dot);

        // Verify against finite difference
        let bump = 1e-6;
        let rates_up = vec![0.03, 0.035 + bump, 0.04];
        let curve_up = DiscountCurveAD::from_zero_rates(&times, &rates_up);
        let df_up: f64 = curve_up.discount(3.0);

        let rates_dn = vec![0.03, 0.035 - bump, 0.04];
        let curve_dn = DiscountCurveAD::from_zero_rates(&times, &rates_dn);
        let df_dn: f64 = curve_dn.discount(3.0);

        let fd_deriv = (df_up - df_dn) / (2.0 * bump);
        assert_abs_diff_eq!(df.dot, fd_deriv, epsilon = 1e-4);
    }

    #[test]
    fn key_rate_durations_via_areal() {
        use crate::tape::{with_tape, adjoint_tl, AReal};

        // Bond: 5 annual cashflows of 5% coupon + 100 principal at year 5
        let cf_times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cf_amounts = vec![5.0, 5.0, 5.0, 5.0, 105.0];

        let pillar_times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let pillar_rates = vec![0.03, 0.032, 0.034, 0.036, 0.038];

        let (npv, krd) = with_tape(|tape| {
            let rates: Vec<AReal> = pillar_rates.iter().map(|&r| tape.input(r)).collect();
            let curve = DiscountCurveAD::from_zero_rates(&pillar_times, &rates);

            let mut npv = AReal::zero();
            for (&t, &amt) in cf_times.iter().zip(&cf_amounts) {
                npv = npv + AReal::from_f64(amt) * curve.discount(t);
            }

            let adj = adjoint_tl(npv);
            let krd: Vec<f64> = rates.iter().map(|a| adj[a.idx]).collect();
            (npv.val, krd)
        });

        // NPV should be close to par (rates ≈ coupon)
        assert!((npv - 100.0).abs() < 10.0, "npv={}", npv);

        // KRDs should all be negative (higher rate → lower DF → lower NPV)
        for (i, &k) in krd.iter().enumerate() {
            assert!(k < 0.0, "KRD[{}] = {} should be negative", i, k);
        }

        // The 5y KRD should be largest in magnitude (principal cashflow)
        assert!(krd[4].abs() > krd[0].abs(),
                "5y KRD ({}) should exceed 1y KRD ({})", krd[4], krd[0]);
    }

    #[test]
    fn from_forwards() {
        let times = vec![1.0, 2.0, 5.0];
        let fwds = vec![0.03, 0.035, 0.04];
        let curve = DiscountCurveAD::from_forwards(&times, &fwds);

        // DF(1) = exp(-0.03 * 1)
        let df1: f64 = curve.discount(1.0);
        assert_abs_diff_eq!(df1, (-0.03_f64).exp(), epsilon = 1e-12);

        // DF(2) = exp(-0.03*1 - 0.035*1)
        let df2: f64 = curve.discount(2.0);
        assert_abs_diff_eq!(df2, (-0.03 - 0.035_f64).exp(), epsilon = 1e-12);
    }

    #[test]
    fn inst_forward_piecewise_constant() {
        let times = vec![1.0, 2.0, 5.0];
        let rates = vec![0.03, 0.035, 0.04];
        let curve = DiscountCurveAD::from_zero_rates(&times, &rates);

        // Instantaneous forward should be piecewise constant between pillars
        let f1: f64 = curve.inst_forward(0.5);
        let f2: f64 = curve.inst_forward(1.5);
        let f3: f64 = curve.inst_forward(3.0);

        // All should be finite and positive
        assert!(f1 > 0.0 && f1.is_finite());
        assert!(f2 > 0.0 && f2.is_finite());
        assert!(f3 > 0.0 && f3.is_finite());
    }
}
