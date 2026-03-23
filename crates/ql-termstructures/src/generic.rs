//! Generic term structure functions that work with any `T: Number`.
//!
//! These functions enable differentiation through term-structure operations
//! (discount factors, zero rates, forward rates) when combined with AD types.
//!
//! Unlike the trait-based [`YieldTermStructure`](crate::yield_term_structure::YieldTermStructure),
//! which is object-safe and returns `f64`, these free functions are fully
//! generic and preserve derivative information.
//!
//! # Examples
//!
//! ```
//! use ql_termstructures::generic::{flat_discount, flat_zero_rate, flat_forward_rate};
//! use ql_core::Number;
//!
//! let df = flat_discount(0.05_f64, 2.0_f64);
//! assert!((df - (-0.10_f64).exp()).abs() < 1e-14);
//! ```

use ql_core::Number;
use ql_math::generic::log_linear_interp;

// ===========================================================================
// GenericYieldCurve trait (INFRA-3)
// ===========================================================================

/// A yield curve that returns discount factors, zero rates, and forward rates
/// as generic `T: Number` values, enabling automatic differentiation through
/// curve operations.
///
/// Unlike the object-safe [`YieldTermStructure`](crate::yield_term_structure::YieldTermStructure)
/// which returns `f64`, this trait preserves derivative information for AD.
///
/// # Design notes
///
/// - Query times are `f64` (time is not a risk factor).
/// - Outputs are generic `T` (discount factors, rates are risk-sensitive).
/// - Default implementations for `zero_rate_t` and `forward_rate_t` are
///   derived from `discount_t`, so only one method needs implementation.
pub trait GenericYieldCurve<T: Number>: Send + Sync {
    /// Discount factor at year-fraction `t`.
    fn discount_t(&self, t: f64) -> T;

    /// Continuously compounded zero rate at time `t`: `r = -ln(DF)/t`.
    fn zero_rate_t(&self, t: f64) -> T {
        if t.abs() < 1e-14 {
            // Short-rate limit: use a small dt to avoid 0/0
            let dt = 1e-4;
            return self.discount_t(dt).ln() * T::from_f64(-1.0 / dt);
        }
        self.discount_t(t).ln() * T::from_f64(-1.0 / t)
    }

    /// Forward rate between `t1` and `t2`:
    /// `f = [ln(DF(t1)) - ln(DF(t2))] / (t2 - t1)`.
    fn forward_rate_t(&self, t1: f64, t2: f64) -> T {
        let df1 = self.discount_t(t1);
        let df2 = self.discount_t(t2);
        (df1 / df2).ln() * T::from_f64(1.0 / (t2 - t1))
    }
}

// ===========================================================================
// Concrete generic curve types
// ===========================================================================

/// Flat continuously-compounded rate curve, generic over `T: Number`.
///
/// `DF(t) = exp(-rate * t)` where `rate` is generic `T`.
pub struct FlatCurve<T: Number> {
    /// Rate.
    pub rate: T,
}

impl<T: Number> FlatCurve<T> {
    /// New.
    pub fn new(rate: T) -> Self {
        Self { rate }
    }
}

impl<T: Number> GenericYieldCurve<T> for FlatCurve<T> {
    #[inline]
    fn discount_t(&self, t: f64) -> T {
        (T::zero() - self.rate * T::from_f64(t)).exp()
    }

    #[inline]
    fn zero_rate_t(&self, _t: f64) -> T {
        self.rate
    }

    #[inline]
    fn forward_rate_t(&self, _t1: f64, _t2: f64) -> T {
        self.rate
    }
}

/// Log-linearly interpolated discount curve with generic `T: Number` pillar DFs.
///
/// Stores `ln(DF)` at each pillar and interpolates linearly in log-DF space.
/// When `T` is an AD type (e.g. `Dual`, `AReal`), sensitivities to each
/// pillar flow through the interpolation.
pub struct InterpDiscountCurve<T: Number> {
    /// Pillar times (ascending, positive).
    times: Vec<f64>,
    /// `ln(DF)` at each pillar — generic `T` for AD.
    log_dfs: Vec<T>,
}

impl<T: Number> InterpDiscountCurve<T> {
    /// Build from pillar times and discount factors.
    pub fn from_dfs(times: &[f64], dfs: &[T]) -> Self {
        assert!(times.len() == dfs.len() && times.len() >= 2, "need ≥ 2 pillars");
        let log_dfs = dfs.iter().map(|&df| df.ln()).collect();
        Self {
            times: times.to_vec(),
            log_dfs,
        }
    }

    /// Build from pillar times and continuously-compounded zero rates.
    pub fn from_zero_rates(times: &[f64], rates: &[T]) -> Self {
        assert!(times.len() == rates.len() && times.len() >= 2, "need ≥ 2 pillars");
        let log_dfs = times.iter().zip(rates).map(|(&t, &r)| T::from_f64(-t) * r).collect();
        Self {
            times: times.to_vec(),
            log_dfs,
        }
    }

    /// Build from pillar times and piecewise-constant forward rates.
    pub fn from_forwards(times: &[f64], fwd_rates: &[T]) -> Self {
        assert!(times.len() == fwd_rates.len() && times.len() >= 2, "need ≥ 2 pillars");
        let mut log_dfs = Vec::with_capacity(times.len());
        let mut cum = T::from_f64(-times[0]) * fwd_rates[0];
        log_dfs.push(cum);
        for i in 1..times.len() {
            let dt = times[i] - times[i - 1];
            cum += T::from_f64(-dt) * fwd_rates[i];
            log_dfs.push(cum);
        }
        Self {
            times: times.to_vec(),
            log_dfs,
        }
    }

    /// Number of pillar points.
    pub fn num_pillars(&self) -> usize { self.times.len() }

    /// Pillar times.
    pub fn times(&self) -> &[f64] { &self.times }
}

impl<T: Number> GenericYieldCurve<T> for InterpDiscountCurve<T> {
    fn discount_t(&self, t: f64) -> T {
        if t <= 0.0 {
            return T::one();
        }
        lerp_generic(&self.times, &self.log_dfs, t).exp()
    }
}

/// Linearly interpolated zero-rate curve with generic `T: Number` pillar rates.
///
/// `DF(t) = exp(-r(t) * t)` where `r(t)` is linearly interpolated from pillar rates.
pub struct InterpZeroCurve<T: Number> {
    times: Vec<f64>,
    rates: Vec<T>,
}

impl<T: Number> InterpZeroCurve<T> {
    /// New.
    pub fn new(times: &[f64], rates: &[T]) -> Self {
        assert!(times.len() == rates.len() && times.len() >= 2, "need ≥ 2 pillars");
        Self {
            times: times.to_vec(),
            rates: rates.to_vec(),
        }
    }
}

impl<T: Number> GenericYieldCurve<T> for InterpZeroCurve<T> {
    fn discount_t(&self, t: f64) -> T {
        if t <= 0.0 {
            return T::one();
        }
        let rate = lerp_generic(&self.times, &self.rates, t);
        (T::zero() - rate * T::from_f64(t)).exp()
    }

    fn zero_rate_t(&self, t: f64) -> T {
        if t.abs() < 1e-14 {
            return self.rates[0]; // first pillar rate
        }
        lerp_generic(&self.times, &self.rates, t)
    }
}

// ---------------------------------------------------------------------------
// Helper: linear interpolation with T-valued y-data and f64 x-query
// ---------------------------------------------------------------------------

/// Linear interpolation: `xs` is `f64`, `ys` is generic `T`, query `x` is `f64`.
///
/// This is the "curve sensitivity" pattern: pillar values are AD inputs,
/// query time is a plain f64.
fn lerp_generic<T: Number>(xs: &[f64], ys: &[T], x: f64) -> T {
    debug_assert!(xs.len() == ys.len() && xs.len() >= 2);
    // Locate segment
    let i = if x <= xs[0] {
        0
    } else if x >= xs[xs.len() - 1] {
        xs.len() - 2
    } else {
        match xs.binary_search_by(|a| a.partial_cmp(&x).unwrap()) {
            Ok(i) => i.min(xs.len() - 2),
            Err(i) => (i - 1).min(xs.len() - 2),
        }
    };
    let dx = xs[i + 1] - xs[i];
    if dx.abs() < 1e-30 {
        return ys[i];
    }
    let frac = (x - xs[i]) / dx;
    ys[i] * T::from_f64(1.0 - frac) + ys[i + 1] * T::from_f64(frac)
}

// ===========================================================================
// Adapter: wrap &dyn YieldTermStructure as GenericYieldCurve<f64>
// ===========================================================================

/// Adapter that wraps a reference to any `YieldTermStructure` (object-safe,
/// f64-returning) as a `GenericYieldCurve<f64>`.
///
/// This allows existing f64 curves to be used with the generic pricing
/// functions unchanged. Zero cost: just delegates to the underlying trait.
pub struct YieldCurveAdapter<'a> {
    inner: &'a dyn crate::yield_term_structure::YieldTermStructure,
}

impl<'a> YieldCurveAdapter<'a> {
    /// New.
    pub fn new(inner: &'a dyn crate::yield_term_structure::YieldTermStructure) -> Self {
        Self { inner }
    }
}

impl<'a> GenericYieldCurve<f64> for YieldCurveAdapter<'a> {
    #[inline]
    fn discount_t(&self, t: f64) -> f64 {
        self.inner.discount_t(t)
    }
}

// ===========================================================================
// Flat-rate discount functions
// ===========================================================================

/// Continuous-compounding discount factor: `exp(-r * t)`.
#[inline]
pub fn flat_discount<T: Number>(rate: T, t: T) -> T {
    (T::zero() - rate * t).exp()
}

/// Simply compounded discount factor: `1 / (1 + r * t)`.
#[inline]
pub fn simple_discount<T: Number>(rate: T, t: T) -> T {
    T::one() / (T::one() + rate * t)
}

/// Annually compounded discount factor: `(1 + r)^(-t)`.
#[inline]
pub fn annual_discount<T: Number>(rate: T, t: T) -> T {
    (T::one() + rate).powf(T::zero() - t)
}

/// Zero rate from a continuously-compounded discount factor: `r = -ln(df) / t`.
#[inline]
pub fn zero_rate_from_discount<T: Number>(df: T, t: T) -> T {
    T::zero() - df.ln() / t
}

/// Forward rate between `t1` and `t2` from two discount factors.
#[inline]
pub fn flat_forward_rate<T: Number>(df1: T, df2: T, t1: T, t2: T) -> T {
    T::zero() - (df2 / df1).ln() / (t2 - t1)
}

/// Instantaneous forward rate at time `t` for a flat curve.
///
/// For a constant rate this is just the rate itself, but the function
/// signature is generic so it composes properly with AD.
#[inline]
pub fn flat_zero_rate<T: Number>(rate: T, _t: T) -> T {
    rate
}

// ===========================================================================
// Interpolated discount curve (generic)
// ===========================================================================

/// Discount factor from a log-linearly interpolated discount curve.
///
/// `times` and `dfs` are the curve pillar data (f64).
/// The query time `t` is generic, enabling AD.
#[inline]
pub fn interp_discount<T: Number>(times: &[f64], dfs: &[f64], t: T) -> T {
    log_linear_interp(times, dfs, t)
}

/// Zero rate from an interpolated discount curve.
#[inline]
pub fn interp_zero_rate<T: Number>(times: &[f64], dfs: &[f64], t: T) -> T {
    let df = interp_discount(times, dfs, t);
    zero_rate_from_discount(df, t)
}

/// Forward rate between `t1` and `t2` from an interpolated discount curve.
#[inline]
pub fn interp_forward_rate<T: Number>(times: &[f64], dfs: &[f64], t1: T, t2: T) -> T {
    let df1 = interp_discount(times, dfs, t1);
    let df2 = interp_discount(times, dfs, t2);
    flat_forward_rate(df1, df2, t1, t2)
}

// ===========================================================================
// Nelson-Siegel (generic)
// ===========================================================================

/// Nelson-Siegel zero rate at time `t`, generic over `T: Number`.
///
/// ```text
/// r(t) = β₀ + β₁ · (1 - e^{-t/τ})/(t/τ)
///      + β₂ · ((1 - e^{-t/τ})/(t/τ) - e^{-t/τ})
/// ```
pub fn nelson_siegel_rate<T: Number>(beta0: T, beta1: T, beta2: T, tau: T, t: T) -> T {
    let x = t / tau;
    let em = (T::zero() - x).exp();
    let factor = if t.to_f64().abs() < 1e-14 {
        T::one()
    } else {
        (T::one() - em) / x
    };
    beta0 + beta1 * factor + beta2 * (factor - em)
}

/// Nelson-Siegel discount factor at time `t`.
pub fn nelson_siegel_discount<T: Number>(beta0: T, beta1: T, beta2: T, tau: T, t: T) -> T {
    let rate = nelson_siegel_rate(beta0, beta1, beta2, tau, t);
    (T::zero() - rate * t).exp()
}

/// Svensson zero rate at time `t`, generic over `T: Number`.
///
/// Extends Nelson-Siegel with a second hump term:
/// ```text
/// r(t) = β₀ + β₁·f₁(t/τ₁) + β₂·(f₁(t/τ₁) - e^{-t/τ₁})
///      + β₃·(f₁(t/τ₂) - e^{-t/τ₂})
/// ```
pub fn svensson_rate<T: Number>(
    beta0: T, beta1: T, beta2: T, beta3: T,
    tau1: T, tau2: T, t: T,
) -> T {
    let x1 = t / tau1;
    let x2 = t / tau2;
    let em1 = (T::zero() - x1).exp();
    let em2 = (T::zero() - x2).exp();

    let f1 = if t.to_f64().abs() < 1e-14 {
        T::one()
    } else {
        (T::one() - em1) / x1
    };
    let f2 = if t.to_f64().abs() < 1e-14 {
        T::one()
    } else {
        (T::one() - em2) / x2
    };

    beta0 + beta1 * f1 + beta2 * (f1 - em1) + beta3 * (f2 - em2)
}

/// Svensson discount factor at time `t`.
pub fn svensson_discount<T: Number>(
    beta0: T, beta1: T, beta2: T, beta3: T,
    tau1: T, tau2: T, t: T,
) -> T {
    let rate = svensson_rate(beta0, beta1, beta2, beta3, tau1, tau2, t);
    (T::zero() - rate * t).exp()
}

// ===========================================================================
// SABR volatility (generic)
// ===========================================================================

/// Hagan SABR implied volatility approximation, generic over `T: Number`.
///
/// Computes the Black implied volatility for a given strike/forward pair
/// under the SABR model. All parameters can be AD types for calibration
/// sensitivities.
pub fn sabr_vol_generic<T: Number>(
    strike: T,
    forward: T,
    expiry: T,
    alpha: T,
    beta: T,
    rho: T,
    nu: T,
) -> T {
    let one = T::one();
    let half = T::half();
    let eps = T::from_f64(1e-12);

    // ATM case
    let fk = forward * strike;
    let f_over_k = forward / strike;

    if (forward - strike).abs().to_f64() < eps.to_f64() {
        // ATM formula: σ ≈ α · F^{β-1} · [1 + ((1-β)²/24 · α²/F^{2(1-β)} + ρβνα/(4F^{1-β}) + (2-3ρ²)ν²/24) · T]
        let f_b1 = forward.powf(one - beta);
        let term1 = (one - beta) * (one - beta) * alpha * alpha
            / (T::from_f64(24.0) * f_b1 * f_b1);
        let term2 = rho * beta * nu * alpha / (T::from_f64(4.0) * f_b1);
        let term3 =
            (T::two() - T::from_f64(3.0) * rho * rho) * nu * nu / T::from_f64(24.0);
        return alpha / f_b1 * (one + (term1 + term2 + term3) * expiry);
    }

    let log_fk = f_over_k.ln();
    let fk_b = fk.powf(half * (one - beta));
    let fk_2b = fk_b * fk_b; // (FK)^{1-beta}

    let z = nu / alpha * fk_b * log_fk;
    let chi = ((one - T::two() * rho * z + z * z).sqrt() + z - rho).ln() - (one - rho).ln();

    // Avoid 0/0
    let z_over_chi = if chi.to_f64().abs() < 1e-14 {
        one
    } else {
        z / chi
    };

    let one_minus_beta = one - beta;
    let log_fk_sq = log_fk * log_fk;

    let numerator = alpha * z_over_chi;
    let denominator = fk_b
        * (one + one_minus_beta * one_minus_beta * log_fk_sq / T::from_f64(24.0)
            + one_minus_beta.powi(4) * log_fk_sq * log_fk_sq / T::from_f64(1920.0));

    let correction = one
        + (one_minus_beta * one_minus_beta * alpha * alpha
            / (T::from_f64(24.0) * fk_2b)
            + rho * beta * nu * alpha / (T::from_f64(4.0) * fk_b)
            + (T::two() - T::from_f64(3.0) * rho * rho) * nu * nu / T::from_f64(24.0))
            * expiry;

    numerator / denominator * correction
}

// ===========================================================================
// INFRA-4 — Generic Default Probability Term Structure
// ===========================================================================

/// Generic default-probability term structure (AD-compatible).
///
/// Mirrors [`crate::default_term_structure::DefaultProbabilityTermStructure`]
/// but with `T: Number` instead of `f64`.
pub trait GenericDefaultProbTS<T: Number> {
    /// Survival probability at time `t` (years).
    fn survival_probability(&self, t: T) -> T;

    /// Default probability in [0, t].
    fn default_probability(&self, t: T) -> T {
        T::one() - self.survival_probability(t)
    }

    /// Default probability in [t1, t2].
    fn default_probability_interval(&self, t1: T, t2: T) -> T {
        self.survival_probability(t1) - self.survival_probability(t2)
    }

    /// Hazard rate at time `t` (finite-difference approximation).
    fn hazard_rate(&self, t: T) -> T {
        let dt = T::from_f64(1e-4);
        let t1 = if t.to_f64() < 1e-4 { T::zero() } else { t - dt };
        let t2 = t + dt;
        let s1 = self.survival_probability(t1);
        let s2 = self.survival_probability(t2);
        if s2.to_f64() <= 0.0 { return T::zero(); }
        -(s2.ln() - s1.ln()) / (t2 - t1)
    }
}

/// Constant hazard-rate default curve, generic over `T: Number`.
///
/// S(t) = exp(-λ·t).
#[derive(Debug, Clone)]
pub struct FlatHazardRateGeneric<T: Number> {
    /// Constant hazard rate λ.
    pub hazard: T,
}

impl<T: Number> FlatHazardRateGeneric<T> {
    /// Create from a constant hazard rate.
    pub fn new(hazard: T) -> Self { Self { hazard } }

    /// Create from a CDS spread: λ ≈ spread / (1 − recovery).
    pub fn from_spread(spread: T, recovery: T) -> Self {
        Self { hazard: spread / (T::one() - recovery) }
    }
}

impl<T: Number> GenericDefaultProbTS<T> for FlatHazardRateGeneric<T> {
    fn survival_probability(&self, t: T) -> T {
        (-self.hazard * t).exp()
    }

    fn hazard_rate(&self, _t: T) -> T {
        self.hazard
    }
}

/// Piecewise-constant hazard-rate default curve, generic over `T: Number`.
///
/// Between knot points the hazard rate is constant.  Survival probabilities
/// at knots are stored directly.
#[derive(Debug, Clone)]
pub struct PiecewiseHazardGeneric<T: Number> {
    /// Time knots (f64 — grid geometry is not differentiated).
    pub times: Vec<f64>,
    /// Survival probability at each knot.
    pub survival_probs: Vec<T>,
}

impl<T: Number> PiecewiseHazardGeneric<T> {
    /// Build from knot times and corresponding survival probabilities.
    pub fn new(times: Vec<f64>, survival_probs: Vec<T>) -> Self {
        assert_eq!(times.len(), survival_probs.len());
        assert!(!times.is_empty());
        Self { times, survival_probs }
    }
}

impl<T: Number> GenericDefaultProbTS<T> for PiecewiseHazardGeneric<T> {
    fn survival_probability(&self, t: T) -> T {
        let tf = t.to_f64();
        if tf <= self.times[0] { return self.survival_probs[0]; }
        let n = self.times.len();
        if tf >= self.times[n - 1] {
            // Flat extrapolation of last hazard rate
            let s_last = self.survival_probs[n - 1];
            if n < 2 { return s_last; }
            let dt_last = self.times[n - 1] - self.times[n - 2];
            let s_prev = self.survival_probs[n - 2];
            let lambda = if dt_last > 1e-15 {
                -(s_last / s_prev).ln() / T::from_f64(dt_last)
            } else {
                T::zero()
            };
            return s_last * (-lambda * T::from_f64(tf - self.times[n - 1])).exp();
        }
        // Find interval
        let idx = self.times.partition_point(|&x| x < tf);
        let idx = idx.min(n - 1).max(1);
        let t0 = self.times[idx - 1];
        let t1 = self.times[idx];
        let s0 = self.survival_probs[idx - 1];
        let s1 = self.survival_probs[idx];
        // Log-linear interpolation: ln S(t) = lerp(ln S0, ln S1, alpha)
        let alpha = T::from_f64((tf - t0) / (t1 - t0));
        let ln_s = s0.ln() * (T::one() - alpha) + s1.ln() * alpha;
        ln_s.exp()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_discount_1y() {
        let df: f64 = flat_discount(0.05, 1.0);
        assert!((df - (-0.05_f64).exp()).abs() < 1e-14);
    }

    #[test]
    fn simple_discount_1y() {
        let df: f64 = simple_discount(0.05, 1.0);
        assert!((df - 1.0 / 1.05).abs() < 1e-14);
    }

    #[test]
    fn annual_discount_1y() {
        let df: f64 = annual_discount(0.05, 1.0);
        assert!((df - 1.0 / 1.05).abs() < 1e-14);
    }

    #[test]
    fn zero_rate_roundtrip() {
        let r = 0.05;
        let t = 2.0;
        let df: f64 = flat_discount(r, t);
        let r2: f64 = zero_rate_from_discount(df, t);
        assert!((r - r2).abs() < 1e-14);
    }

    #[test]
    fn forward_rate_flat() {
        let r = 0.05;
        let df1: f64 = flat_discount(r, 1.0);
        let df2: f64 = flat_discount(r, 2.0);
        let fwd: f64 = flat_forward_rate(df1, df2, 1.0, 2.0);
        assert!((fwd - r).abs() < 1e-13, "fwd = {fwd}");
    }

    #[test]
    fn interp_discount_test() {
        let times = &[0.0, 1.0, 2.0, 5.0];
        let dfs = &[1.0, 0.95, 0.90, 0.78];
        let df: f64 = interp_discount(times, dfs, 1.5);
        // log-linear: exp(0.5 * ln(0.95) + 0.5 * ln(0.90))
        let expected = (0.5 * 0.95_f64.ln() + 0.5 * 0.90_f64.ln()).exp();
        assert!((df - expected).abs() < 1e-10, "df = {df}, expected = {expected}");
    }

    #[test]
    fn nelson_siegel_rate_test() {
        // Flat curve: beta0=0.05, beta1=0, beta2=0
        let r: f64 = nelson_siegel_rate(0.05, 0.0, 0.0, 1.0, 2.0);
        assert!((r - 0.05).abs() < 1e-10);
    }

    #[test]
    fn sabr_atm_test() {
        let vol: f64 =
            sabr_vol_generic(0.05, 0.05, 1.0, 0.20, 0.5, -0.3, 0.4);
        assert!(vol > 0.0 && vol < 1.0, "vol = {vol}");
    }

    #[test]
    fn sabr_otm_test() {
        let vol: f64 =
            sabr_vol_generic(0.06, 0.05, 1.0, 0.20, 0.5, -0.3, 0.4);
        assert!(vol > 0.0 && vol < 1.0, "vol = {vol}");
    }

    // -----------------------------------------------------------------------
    // GenericYieldCurve trait + concrete types (INFRA-3 tests)
    // -----------------------------------------------------------------------

    #[test]
    fn flat_curve_discount() {
        let curve = FlatCurve::new(0.05_f64);
        let df = curve.discount_t(2.0);
        let expected = (-0.10_f64).exp();
        assert!((df - expected).abs() < 1e-14);
    }

    #[test]
    fn flat_curve_zero_rate() {
        let curve = FlatCurve::new(0.05_f64);
        let r = curve.zero_rate_t(3.0);
        assert!((r - 0.05).abs() < 1e-14);
    }

    #[test]
    fn flat_curve_forward_rate() {
        let curve = FlatCurve::new(0.05_f64);
        let fwd = curve.forward_rate_t(1.0, 3.0);
        assert!((fwd - 0.05).abs() < 1e-14);
    }

    #[test]
    fn interp_discount_curve_at_pillars() {
        let times = vec![0.5, 1.0, 2.0, 5.0];
        let rates = vec![0.02_f64, 0.03, 0.035, 0.04];
        let curve = InterpDiscountCurve::from_zero_rates(&times, &rates);
        for (&t, &r) in times.iter().zip(&rates) {
            let df = curve.discount_t(t);
            let expected = (-r * t).exp();
            assert!((df - expected).abs() < 1e-12, "t={t}, df={df}, expected={expected}");
        }
    }

    #[test]
    fn interp_discount_curve_interpolated() {
        let times = vec![1.0, 2.0, 5.0];
        let rates = vec![0.03_f64, 0.035, 0.04];
        let curve = InterpDiscountCurve::from_zero_rates(&times, &rates);
        let df = curve.discount_t(1.5);
        // ln(DF(1.0)) = -0.03, ln(DF(2.0)) = -0.07
        // lerp at 1.5: ln(DF) = 0.5 * (-0.03) + 0.5 * (-0.07) = -0.05
        let expected = (-0.05_f64).exp();
        assert!((df - expected).abs() < 1e-12);
    }

    #[test]
    fn interp_zero_curve_at_pillars() {
        let times = vec![0.5, 1.0, 2.0, 5.0];
        let rates = vec![0.02_f64, 0.03, 0.035, 0.04];
        let curve = InterpZeroCurve::new(&times, &rates);
        for (&t, &r) in times.iter().zip(&rates) {
            let df = curve.discount_t(t);
            let expected = (-r * t).exp();
            assert!((df - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn interp_zero_curve_zero_rate() {
        let times = vec![1.0, 2.0, 5.0];
        let rates = vec![0.03_f64, 0.035, 0.04];
        let curve = InterpZeroCurve::new(&times, &rates);
        // At t=1.5: linearly interpolated rate = 0.5*0.03 + 0.5*0.035 = 0.0325
        let r = curve.zero_rate_t(1.5);
        assert!((r - 0.0325).abs() < 1e-12);
    }

    #[test]
    fn generic_yield_curve_default_forward_rate() {
        let times = vec![1.0, 2.0, 5.0];
        let rates = vec![0.03_f64, 0.035, 0.04];
        let curve = InterpDiscountCurve::from_zero_rates(&times, &rates);
        let fwd = curve.forward_rate_t(1.0, 2.0);
        // f(1,2) = -[ln(DF(2))-ln(DF(1))] / 1 = -[(-0.07)-(-0.03)] = 0.04
        assert!((fwd - 0.04).abs() < 1e-12, "fwd={fwd}");
    }

    #[test]
    fn interp_discount_curve_from_dfs() {
        let times = vec![1.0, 2.0, 5.0];
        let dfs = vec![0.97_f64, 0.93, 0.82];
        let curve = InterpDiscountCurve::from_dfs(&times, &dfs);
        for (&t, &d) in times.iter().zip(&dfs) {
            let df = curve.discount_t(t);
            assert!((df - d).abs() < 1e-12);
        }
    }

    #[test]
    fn yield_curve_adapter() {
        use crate::yield_term_structure::YieldTermStructure;
        use crate::FlatForward;
        use ql_time::{Date, Month, DayCounter};
        let ref_date = Date::from_ymd(2025, Month::January, 15);
        let ff = FlatForward::new(ref_date, 0.05, DayCounter::Actual365Fixed);
        let adapter = YieldCurveAdapter::new(&ff);
        // Both should give the same discount factor
        let df_direct = ff.discount_t(2.0);
        let df_adapter = adapter.discount_t(2.0);
        assert!((df_direct - df_adapter).abs() < 1e-14);
    }

    // ── INFRA-4: Generic Default Probability TS ──────────────

    #[test]
    fn flat_hazard_survival() {
        let c = FlatHazardRateGeneric::new(0.02_f64);
        let s = c.survival_probability(5.0);
        let expected = (-0.10_f64).exp();
        assert!((s - expected).abs() < 1e-14, "s={s}, expected={expected}");
    }

    #[test]
    fn flat_hazard_default_prob() {
        let c = FlatHazardRateGeneric::new(0.02_f64);
        let dp = c.default_probability(5.0);
        let expected = 1.0 - (-0.10_f64).exp();
        assert!((dp - expected).abs() < 1e-14);
    }

    #[test]
    fn flat_hazard_from_spread() {
        let c = FlatHazardRateGeneric::<f64>::from_spread(0.01, 0.4);
        // lambda = 0.01/0.6 ≈ 0.016667
        let expected_h = 0.01 / 0.6;
        assert!((c.hazard - expected_h).abs() < 1e-12);
    }

    #[test]
    fn piecewise_hazard_interp() {
        // Two knots: S(1)=0.98, S(3)=0.92
        let times = vec![0.0, 1.0, 3.0];
        let sps = vec![1.0_f64, 0.98, 0.92];
        let c = PiecewiseHazardGeneric::new(times, sps);
        // At t=2 (midpoint of [1,3]): log-linear
        let s = c.survival_probability(2.0);
        let expected = (0.5 * 0.98_f64.ln() + 0.5 * 0.92_f64.ln()).exp();
        assert!((s - expected).abs() < 1e-12, "s={s}, expected={expected}");
    }

    #[test]
    fn piecewise_hazard_at_knots() {
        let times = vec![0.0, 1.0, 3.0, 5.0];
        let sps = vec![1.0_f64, 0.98, 0.93, 0.85];
        let c = PiecewiseHazardGeneric::new(times, sps);
        assert!((c.survival_probability(0.0) - 1.0).abs() < 1e-14);
        assert!((c.survival_probability(1.0) - 0.98).abs() < 1e-12);
        assert!((c.survival_probability(3.0) - 0.93).abs() < 1e-12);
        assert!((c.survival_probability(5.0) - 0.85).abs() < 1e-12);
    }
}
