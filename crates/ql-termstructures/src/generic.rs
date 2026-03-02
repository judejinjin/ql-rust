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
use ql_math::generic::{linear_interp, log_linear_interp};

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
}
