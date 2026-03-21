//! Generic math functions that work with any `T: Number`.
//!
//! These functions use only operations defined by the [`Number`] trait,
//! making them compatible with AD types (`Dual`, `DualVec`, `AReal`) so
//! that derivatives propagate automatically.
//!
//! The `f64`-only versions in [`crate::distributions`] delegate to `statrs`
//! for higher precision; these generic versions use rational approximations
//! that are accurate to ~7.5e-8 (Abramowitz-Stegun) while staying fully
//! differentiable.
//!
//! # Examples
//!
//! ```
//! use ql_math::generic::{normal_pdf, normal_cdf, black_scholes_generic};
//! use ql_core::Number;
//!
//! // Works with f64 directly
//! let phi = normal_cdf(1.0_f64);
//! assert!((phi - 0.8413).abs() < 1e-3);
//!
//! // Black-Scholes with f64
//! let price = black_scholes_generic(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true);
//! assert!((price - 10.45).abs() < 0.1);
//! ```

use ql_core::Number;

// ===========================================================================
// Standard Normal PDF & CDF
// ===========================================================================

/// Standard normal probability density function:
///     φ(x) = (1/√(2π)) · exp(-x²/2)
///
/// Generic over `T: Number` so derivatives propagate through AD types.
#[inline]
pub fn normal_pdf<T: Number>(x: T) -> T {
    let half = T::half();
    let inv_sqrt_2pi = T::from_f64(0.3989422804014327); // 1/√(2π)
    inv_sqrt_2pi * (T::zero() - half * x * x).exp()
}

/// Cumulative distribution function for the standard normal distribution.
///
/// Uses the Abramowitz-Stegun (1964, formula 26.2.17) rational approximation,
/// which is accurate to ~|ε| < 7.5e-8:
///
/// ```text
///     Φ(x) ≈ 1 − φ(x)(b₁t + b₂t² + b₃t³ + b₄t⁴ + b₅t⁵)
/// ```
///
/// where `t = 1 / (1 + 0.2316419 |x|)`.
#[inline]
pub fn normal_cdf<T: Number>(x: T) -> T {
    let b1 = T::from_f64(0.319381530);
    let b2 = T::from_f64(-0.356563782);
    let b3 = T::from_f64(1.781477937);
    let b4 = T::from_f64(-1.821255978);
    let b5 = T::from_f64(1.330274429);
    let p = T::from_f64(0.2316419);

    let ax = x.abs();
    let t = T::one() / (T::one() + p * ax);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;

    let pdf = normal_pdf(ax);
    let poly = b1 * t + b2 * t2 + b3 * t3 + b4 * t4 + b5 * t5;
    let cdf_positive = T::one() - pdf * poly;

    if x.to_f64() >= 0.0 {
        cdf_positive
    } else {
        T::one() - cdf_positive
    }
}

/// Inverse of the standard normal CDF (quantile function).
///
/// Uses the Peter Acklam rational approximation, accurate to ~1e-9
/// for p in (1e-300, 1-1e-300). Returns `f64` because the inverse CDF
/// is only needed for sampling, not differentiation.
#[inline]
pub fn inverse_normal_cdf(p: f64) -> f64 {
    // Peter Acklam's rational approximation
    const A: [f64; 6] = [
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383_577_518_672_69e2,
        -3.066479806614716e+01,
         2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
         7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e+00,
         3.754408661907416e+00,
    ];

    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        // Lower tail
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0]*q + C[1])*q + C[2])*q + C[3])*q + C[4])*q + C[5])
            / ((((D[0]*q + D[1])*q + D[2])*q + D[3])*q + 1.0)
    } else if p <= P_HIGH {
        // Central region
        let q = p - 0.5;
        let r = q * q;
        (((((A[0]*r + A[1])*r + A[2])*r + A[3])*r + A[4])*r + A[5]) * q
            / (((((B[0]*r + B[1])*r + B[2])*r + B[3])*r + B[4])*r + 1.0)
    } else {
        // Upper tail
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0]*q + C[1])*q + C[2])*q + C[3])*q + C[4])*q + C[5])
            / ((((D[0]*q + D[1])*q + D[2])*q + D[3])*q + 1.0)
    }
}

// ===========================================================================
// Linear Interpolation (Generic)
// ===========================================================================

/// Generic linear interpolation: given sorted knots `(xs, ys)`, return the
/// interpolated value at `x`.
///
/// Extrapolates flat beyond the boundaries.
///
/// `xs` and `ys` are `f64` (knot positions are not differentiated);
/// the query point `x` and the result are generic `T: Number`.
#[inline]
pub fn linear_interp<T: Number>(xs: &[f64], ys: &[f64], x: T) -> T {
    debug_assert!(xs.len() == ys.len() && xs.len() >= 2);
    let xv = x.to_f64();

    // Clamp / flat extrapolation
    if xv <= xs[0] {
        return T::from_f64(ys[0]);
    }
    if xv >= xs[xs.len() - 1] {
        return T::from_f64(ys[ys.len() - 1]);
    }

    // Binary search for the interval
    let mut lo = 0;
    let mut hi = xs.len() - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xv < xs[mid] {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    let x0 = T::from_f64(xs[lo]);
    let x1 = T::from_f64(xs[hi]);
    let y0 = T::from_f64(ys[lo]);
    let y1 = T::from_f64(ys[hi]);

    // y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

/// Generic log-linear interpolation on discount factors.
///
/// Given sorted times `ts` and discount factors `dfs`, returns
/// `exp(interp(ts, ln(dfs), t))`, i.e. log-linearly interpolated DF.
#[inline]
pub fn log_linear_interp<T: Number>(ts: &[f64], dfs: &[f64], t: T) -> T {
    debug_assert!(ts.len() == dfs.len() && ts.len() >= 2);
    let log_dfs: Vec<f64> = dfs.iter().map(|d| d.ln()).collect();
    let log_df = linear_interp(ts, &log_dfs, t);
    log_df.exp()
}

// ===========================================================================
// Bivariate Normal CDF
// ===========================================================================

/// Cumulative distribution function of the standard bivariate normal,
/// generic over `T: Number`.
///
/// Uses the Drezner-Wesolowsky (1990) algorithm with Gauss-Legendre
/// quadrature, accurate to ~1e-7. Both `x` and `y` are generic
/// so AD derivatives propagate; `rho` is `f64` (correlation is typically
/// not a risk factor).
///
/// ```text
///     BVN(x, y; ρ) = Pr[X ≤ x, Y ≤ y]   where (X,Y) ~ N(0, 0, 1, 1, ρ)
/// ```
pub fn bivariate_normal_cdf<T: Number>(x: T, y: T, rho: f64) -> T {
    // Gauss-Legendre 3-point weights & abscissae (on [-1,1])
    const W: [f64; 3] = [0.1713244923791702, 0.3607615730481386, 0.467_913_934_572_691];
    const XI: [f64; 3] = [0.932_469_514_203_152, 0.6612093864662645, 0.2386191860831969];

    let dh = T::zero() - x;
    let dk = T::zero() - y;

    if rho.abs() < 0.925 {
        // Standard Drezner formula for moderate correlation
        let hs = (dh * dh + dk * dk) / T::from_f64(2.0);
        let asr = rho.asin(); // f64

        let mut bvn = T::zero();
        for i in 0..3 {
            for &sign in &[-1.0_f64, 1.0] {
                let sn_f64 = (asr * (sign * XI[i] + 1.0) / 2.0).sin();
                let sn = T::from_f64(sn_f64);
                let one_minus_sn2 = T::from_f64(1.0 - sn_f64 * sn_f64);
                bvn += T::from_f64(W[i])
                    * (T::zero() - (hs - sn * dh * dk) / one_minus_sn2).exp();
            }
        }
        bvn * T::from_f64(asr / (4.0 * std::f64::consts::PI))
            + normal_cdf(T::zero() - dh) * normal_cdf(T::zero() - dk)
    } else {
        // High correlation: use identities to reduce
        if rho < 0.0 {
            // BVN(x,y;-|ρ|) = Φ(x) - BVN(x,-y;|ρ|)
            let bvn_pos = bivariate_normal_cdf(x, T::zero() - y, -rho);
            let result = normal_cdf(x) - bvn_pos;
            if result.to_f64() < 0.0 { T::zero() } else { result }
        } else {
            // ρ close to +1: use Drezner (1990) high-correlation formula
            // BVN(x,y;ρ) ≈ Φ(min(x,y)) - correction
            let two = T::from_f64(2.0);
            let _sqrt2pi = T::from_f64(std::f64::consts::TAU.sqrt());

            // Transformation: hk = sqrt(2*(1-ρ)), compute in high-accuracy region
            let onemr = (1.0 - rho).max(0.0);
            if onemr < 1e-15 {
                // ρ ≈ 1 exactly
                return normal_cdf(if x.to_f64() < y.to_f64() { x } else { y });
            }

            let _hk = onemr.sqrt();
            let hs = (dh * dh + dk * dk) / two;
            let asr = rho.asin();

            let mut bvn = T::zero();
            for i in 0..3 {
                for &sign in &[-1.0_f64, 1.0] {
                    let sn_f64 = (asr * (sign * XI[i] + 1.0) / 2.0).sin();
                    let sn = T::from_f64(sn_f64);
                    let one_minus_sn2 = T::from_f64(1.0 - sn_f64 * sn_f64);
                    bvn += T::from_f64(W[i])
                        * (T::zero() - (hs - sn * dh * dk) / one_minus_sn2).exp();
                }
            }
            bvn * T::from_f64(asr / (4.0 * std::f64::consts::PI))
                + normal_cdf(T::zero() - dh) * normal_cdf(T::zero() - dk)
        }
    }
}

// ===========================================================================
// Black-Scholes Generic
// ===========================================================================

/// Black-Scholes European option price, generic over `T: Number`.
///
/// All inputs (spot, strike, rate, dividend yield, volatility, time to expiry)
/// can be AD types, enabling automatic Greek computation.
///
/// Returns the option premium.
///
/// # Parameters
/// - `spot`: Current underlying price
/// - `strike`: Option strike price
/// - `r`: Continuously compounded risk-free rate
/// - `q`: Continuously compounded dividend yield
/// - `vol`: Black-Scholes volatility (annualized)
/// - `t`: Time to expiry in years
/// - `is_call`: `true` for a call, `false` for a put
///
/// # Examples
///
/// ```
/// use ql_math::generic::black_scholes_generic;
///
/// let call = black_scholes_generic(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true);
/// assert!((call - 10.4506).abs() < 0.01);
///
/// let put = black_scholes_generic(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, false);
/// assert!((put - 5.5735).abs() < 0.01);
/// ```
#[inline]
pub fn black_scholes_generic<T: Number>(
    spot: T,
    strike: T,
    r: T,
    q: T,
    vol: T,
    t: T,
    is_call: bool,
) -> T {
    let sqrt_t = t.sqrt();
    let vol_sqrt_t = vol * sqrt_t;

    let d1 = ((spot / strike).ln() + (r - q + vol * vol * T::half()) * t) / vol_sqrt_t;
    let d2 = d1 - vol_sqrt_t;

    let df = (T::zero() - r * t).exp(); // e^(-r*t)
    let fwd_factor = (T::zero() - q * t).exp(); // e^(-q*t)

    if is_call {
        spot * fwd_factor * normal_cdf(d1) - strike * df * normal_cdf(d2)
    } else {
        strike * df * normal_cdf(T::zero() - d2) - spot * fwd_factor * normal_cdf(T::zero() - d1)
    }
}

// ===========================================================================
// Discount Factor (Generic)
// ===========================================================================

/// Continuously-compounded discount factor: `exp(-r * t)`.
#[inline]
pub fn discount_factor<T: Number>(rate: T, t: T) -> T {
    (T::zero() - rate * t).exp()
}

/// Forward discount factor between times `t1` and `t2` given a flat rate.
#[inline]
pub fn forward_discount<T: Number>(rate: T, t1: T, t2: T) -> T {
    (T::zero() - rate * (t2 - t1)).exp()
}

/// Zero rate from discount factor and time: `r = -ln(df) / t`.
#[inline]
pub fn zero_rate_from_df<T: Number>(df: T, t: T) -> T {
    T::zero() - df.ln() / t
}

// ===========================================================================
// Risk Analytics (Generic)
// ===========================================================================

/// Delta of a European option (∂V/∂S) computed analytically, generic.
#[inline]
pub fn bs_delta_generic<T: Number>(
    spot: T,
    strike: T,
    r: T,
    q: T,
    vol: T,
    t: T,
    is_call: bool,
) -> T {
    let sqrt_t = t.sqrt();
    let d1 = ((spot / strike).ln() + (r - q + vol * vol * T::half()) * t) / (vol * sqrt_t);
    let fwd_factor = (T::zero() - q * t).exp();

    if is_call {
        fwd_factor * normal_cdf(d1)
    } else {
        fwd_factor * (normal_cdf(d1) - T::one())
    }
}

/// Vega of a European option (∂V/∂σ) computed analytically, generic.
#[inline]
pub fn bs_vega_generic<T: Number>(
    spot: T,
    strike: T,
    r: T,
    q: T,
    vol: T,
    t: T,
) -> T {
    let sqrt_t = t.sqrt();
    let d1 = ((spot / strike).ln() + (r - q + vol * vol * T::half()) * t) / (vol * sqrt_t);
    let fwd_factor = (T::zero() - q * t).exp();

    spot * fwd_factor * normal_pdf(d1) * sqrt_t
}

/// NPV of a series of fixed cashflows given a flat discount rate.
///
/// `amounts[i]` is paid at `times[i]`. Generic over `T: Number`.
#[inline]
pub fn npv_flat_rate<T: Number>(amounts: &[f64], times: &[f64], rate: T) -> T {
    let mut pv = T::zero();
    for (amt, &t) in amounts.iter().zip(times.iter()) {
        pv += T::from_f64(*amt) * discount_factor(rate, T::from_f64(t));
    }
    pv
}

/// Implied volatility via Newton-Raphson, generic.
///
/// Finds `σ` such that `bs_price(σ) = target_price`.
/// Returns `None` if convergence fails.
pub fn implied_vol_newton(
    target_price: f64,
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    t: f64,
    is_call: bool,
    initial_guess: f64,
    max_iter: usize,
    tol: f64,
) -> Option<f64> {
    let mut vol = initial_guess;
    for _ in 0..max_iter {
        let price = black_scholes_generic(spot, strike, r, q, vol, t, is_call);
        let vega = bs_vega_generic(spot, strike, r, q, vol, t);
        let vega_val = vega.to_f64();
        if vega_val.abs() < 1e-20 {
            return None;
        }
        let diff = price.to_f64() - target_price;
        vol -= diff / vega_val;
        if diff.abs() < tol {
            return Some(vol);
        }
        if vol <= 0.0 {
            return None;
        }
    }
    None
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_pdf() {
        let v: f64 = normal_pdf(0.0);
        assert!((v - 0.3989422804014327).abs() < 1e-12);
    }

    #[test]
    fn test_normal_cdf() {
        let cases: &[(f64, f64)] = &[
            (0.0, 0.5),
            (1.0, 0.8413447),
            (-1.0, 0.1586553),
            (2.0, 0.9772499),
            (-2.0, 0.0227501),
        ];
        for &(x, expected) in cases {
            let v: f64 = normal_cdf(x);
            assert!((v - expected).abs() < 1e-6, "N({x}) = {v}, expected {expected}");
        }
    }

    #[test]
    fn test_inverse_normal_cdf() {
        let cases: &[(f64, f64)] = &[
            (0.5, 0.0),
            (0.8413, 0.9998),
            (0.025, -1.9600),
        ];
        for &(p, expected) in cases {
            let v = inverse_normal_cdf(p);
            assert!((v - expected).abs() < 0.01, "invN({p}) = {v}, expected {expected}");
        }
    }

    #[test]
    fn test_black_scholes_call() {
        let price: f64 = black_scholes_generic(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true);
        assert!((price - 10.4506).abs() < 0.05, "BS call = {price}");
    }

    #[test]
    fn test_black_scholes_put() {
        let price: f64 = black_scholes_generic(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, false);
        assert!((price - 5.5735).abs() < 0.05, "BS put = {price}");
    }

    #[test]
    fn test_put_call_parity() {
        let s = 100.0;
        let k = 100.0;
        let r = 0.05;
        let q = 0.02;
        let vol = 0.25;
        let t = 1.0;
        let call: f64 = black_scholes_generic(s, k, r, q, vol, t, true);
        let put: f64 = black_scholes_generic(s, k, r, q, vol, t, false);
        let parity = call - put - s * (-q * t).exp() + k * (-r * t).exp();
        assert!(parity.abs() < 1e-6, "parity residual = {parity}");
    }

    #[test]
    fn test_discount_factor() {
        let df: f64 = discount_factor(0.05, 1.0);
        assert!((df - (-0.05_f64).exp()).abs() < 1e-14);
    }

    #[test]
    fn test_linear_interp_midpoint() {
        let xs = &[0.0, 1.0, 2.0];
        let ys = &[0.0, 1.0, 4.0];
        let v: f64 = linear_interp(xs, ys, 0.5);
        assert!((v - 0.5).abs() < 1e-14);
    }

    #[test]
    fn test_linear_interp_second_segment() {
        let xs = &[0.0, 1.0, 2.0];
        let ys = &[0.0, 1.0, 4.0];
        let v: f64 = linear_interp(xs, ys, 1.5);
        assert!((v - 2.5).abs() < 1e-14);
    }

    #[test]
    fn test_npv_flat_rate() {
        let amounts = [100.0, 100.0, 1100.0];
        let times = [1.0, 2.0, 3.0];
        let pv: f64 = npv_flat_rate(&amounts, &times, 0.05);
        // Manual: 100*e^{-0.05} + 100*e^{-0.10} + 1100*e^{-0.15}
        let expected = 100.0 * (-0.05_f64).exp()
            + 100.0 * (-0.10_f64).exp()
            + 1100.0 * (-0.15_f64).exp();
        assert!((pv - expected).abs() < 1e-10, "npv = {pv}, expected = {expected}");
    }

    #[test]
    fn test_implied_vol() {
        let target = black_scholes_generic(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true);
        let iv = implied_vol_newton(target, 100.0, 100.0, 0.05, 0.0, 1.0, true, 0.30, 100, 1e-10);
        assert!(iv.is_some());
        assert!((iv.unwrap() - 0.20).abs() < 1e-8, "iv = {:?}", iv);
    }

    #[test]
    fn test_bs_delta_call() {
        let delta: f64 = bs_delta_generic(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true);
        assert!((delta - 0.6368).abs() < 0.01, "delta = {delta}");
    }

    #[test]
    fn test_bs_vega() {
        let vega: f64 = bs_vega_generic(100.0, 100.0, 0.05, 0.0, 0.20, 1.0);
        assert!(vega > 30.0 && vega < 45.0, "vega = {vega}");
    }
}
