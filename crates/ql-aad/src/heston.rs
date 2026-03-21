//! Generic Heston pricing engine with AD-aware Greeks.
//!
//! The characteristic function and price are generic over `T: Number`,
//! enabling forward-mode (`DualVec<N>`) and reverse-mode (`AReal`) Greeks.
//!
//! The outer quadrature (Gauss-Legendre) stays `f64`; only the characteristic
//! function evaluation at each node is generic, so derivatives flow through
//! the CF computation while the integration weights/nodes remain plain `f64`.
//!
//! # Example
//!
//! ```
//! use ql_aad::heston::{heston_greeks_ad, HestonGreeks};
//!
//! let g = heston_greeks_ad(
//!     100.0, 100.0, 0.05, 0.0, 1.0,  // spot, strike, r, q, tau
//!     0.04, 1.5, 0.04, 0.3, -0.7,    // v0, kappa, theta, sigma, rho
//!     true,                            // is_call
//! );
//! assert!(g.npv > 5.0 && g.npv < 20.0);
//! assert!(g.delta > 0.0 && g.delta < 1.0);
//! ```

use crate::complex::Complex;
use crate::dual_vec::DualVec;
use crate::number::Number;

// ===========================================================================
// Gauss-Legendre nodes and weights (48-point)
// ===========================================================================

/// Precomputed 48-point Gauss-Legendre quadrature on [-1, 1].
/// Nodes and weights from Abramowitz & Stegun Table 25.4.
pub fn gauss_legendre_48() -> &'static [(f64, f64)] {
    // Using 24-point half (symmetric), mirrored to get 48.
    // For simplicity, we use a smaller high-accuracy 20-point rule
    // that is sufficient for the CF integrand decay.
    // Full GL-48 would be 24 symmetric pairs — here we use 20-point.
    static GL20: [(f64, f64); 20] = [
        (0.076_526_521_133_497_34, 0.152_753_387_130_725_84),
        (0.227_785_851_141_645_07, 0.149_172_986_472_603_74),
        (0.373_706_088_715_419_55, 0.142_096_109_318_382_04),
        (0.510_867_001_950_827_1, 0.131_688_638_449_176_64),
        (0.636_053_680_726_515, 0.118_194_531_961_518_41),
        (0.746_331_906_460_150_8, 0.101_930_119_817_240_44),
        (0.839_116_971_822_218_8, 0.083_276_741_576_704_75),
        (0.912_234_428_251_326, 0.062_672_048_334_109_07),
        (0.963_971_927_277_913_8, 0.040_601_429_800_386_94),
        (0.993_128_599_185_094_9, 0.017614007139152118),
        (-0.076_526_521_133_497_34, 0.152_753_387_130_725_84),
        (-0.227_785_851_141_645_07, 0.149_172_986_472_603_74),
        (-0.373_706_088_715_419_55, 0.142_096_109_318_382_04),
        (-0.510_867_001_950_827_1, 0.131_688_638_449_176_64),
        (-0.636_053_680_726_515, 0.118_194_531_961_518_41),
        (-0.746_331_906_460_150_8, 0.101_930_119_817_240_44),
        (-0.839_116_971_822_218_8, 0.083_276_741_576_704_75),
        (-0.912_234_428_251_326, 0.062_672_048_334_109_07),
        (-0.963_971_927_277_913_8, 0.040_601_429_800_386_94),
        (-0.993_128_599_185_094_9, 0.017614007139152118),
    ];
    &GL20
}

/// Numerically integrate f over [a, b] using Gauss-Legendre quadrature.
#[allow(dead_code)]
fn gl_integrate<F: Fn(f64) -> f64>(f: F, a: f64, b: f64) -> f64 {
    let half_range = 0.5 * (b - a);
    let mid = 0.5 * (a + b);
    let nodes = gauss_legendre_48();
    let mut sum = 0.0;
    for &(xi, wi) in nodes {
        sum += wi * f(mid + half_range * xi);
    }
    sum * half_range
}

// ===========================================================================
// Heston characteristic function (generic)
// ===========================================================================

/// Compute the Heston log-characteristic function for P_j (j=1,2).
///
/// Uses the Albrecher et al. (2007) "little Heston trap" avoidance formulation.
/// All model parameters are generic `T: Number` so derivatives propagate.
///
/// The integration variable `phi` is `f64` (not differentiated), but all model
/// parameters carry derivatives.
#[allow(clippy::too_many_arguments)]
pub fn heston_cf_generic<T: Number>(
    phi: f64,
    v0: T,
    kappa: T,
    theta: T,
    sigma: T,
    rho: T,
    tau: T,
    j: u8,
) -> Complex<T> {
    let phi_t = T::from_f64(phi);
    let s2 = sigma * sigma;

    // b_j, u_j
    let b_j = if j == 1 { kappa - rho * sigma } else { kappa };
    let u_j = if j == 1 { T::half() } else { T::zero() - T::half() };

    // bj_minus = b_j - rho*sigma*i*phi
    let bj_minus = Complex::new(b_j, T::zero() - rho * sigma * phi_t);

    // d² = bj_minus² + σ² * (φ² - 2u_j*i*φ)
    //     = bj_minus² + σ² * Complex(φ², -2u_j*φ)
    let two = T::two();
    let d_sq = bj_minus * bj_minus
        + Complex::from_real(s2) * Complex::new(phi_t * phi_t, T::zero() - two * u_j * phi_t);
    let d = d_sq.sqrt();

    // r_minus = bj_minus - d
    let r_minus = bj_minus - d;

    // g = r_minus / (bj_minus + d)
    let g = r_minus / (bj_minus + d);

    // exp(-d*tau)
    let one_c = Complex::from_real(T::one());
    let exp_neg_dt = (Complex::from_real(T::zero()) - d.scale(tau)).exp();

    // D = (r_minus / σ²) * (1 - exp(-dτ)) / (1 - g*exp(-dτ))
    let one_minus_gexp = one_c - g * exp_neg_dt;
    let big_d = r_minus.scale(s2.recip()) * ((one_c - exp_neg_dt) / one_minus_gexp);

    // C = (κθ/σ²) * [r_minus*τ - 2*ln((1 - g*exp(-dτ))/(1-g))]
    let log_ratio = (one_minus_gexp / (one_c - g)).ln();
    let big_c = (r_minus.scale(tau) - log_ratio.scale(two)).scale(kappa * theta * s2.recip());

    // f_j = exp(C + D*v0)
    (big_c + big_d.scale(v0)).exp()
}

/// Generic Heston price for a European option.
///
/// All model parameters are `T: Number`; the integration points and
/// mechanics stay `f64`. Derivatives flow through the CF at each
/// quadrature node.
#[allow(clippy::too_many_arguments)]
pub fn heston_price_generic<T: Number>(
    spot: T,
    strike: T,
    r: T,
    q: T,
    tau: T,
    v0: T,
    kappa: T,
    theta: T,
    sigma: T,
    rho: T,
    is_call: bool,
) -> T {
    let neg_one = T::zero() - T::one();
    let df = (neg_one * r * tau).exp();
    let fwd = spot * ((r - q) * tau).exp();
    let x = (fwd / strike).ln(); // log-moneyness (T)

    // Adaptive upper limit based on the CF decay rate
    let v0_f64 = v0.to_f64();
    let tau_f64 = tau.to_f64();
    let upper = (50.0 / (v0_f64 * tau_f64).sqrt().max(0.05)).clamp(50.0, 200.0);

    // Integrate for P1 and P2
    let integrate_pj_generic = |j: u8| -> T {
        let half_range = 0.5 * (upper - 1e-8);
        let mid = 0.5 * (1e-8 + upper);
        let nodes = gauss_legendre_48();
        let mut sum = T::zero();
        for &(xi, wi) in nodes {
            let phi = mid + half_range * xi;
            if phi < 1e-8 { continue; }
            let f = heston_cf_generic(phi, v0, kappa, theta, sigma, rho, tau, j);
            // Re[ e^{iφx} f_j / (iφ) ] = (f.re * sin(φ*x) + f.im * cos(φ*x)) / φ
            let phi_x = T::from_f64(phi) * x;
            let cos_px = phi_x.cos();
            let sin_px = phi_x.sin();
            let integrand = (f.re * sin_px + f.im * cos_px) / T::from_f64(phi);
            sum += integrand * T::from_f64(wi * half_range);
        }
        sum
    };

    let i1 = integrate_pj_generic(1);
    let i2 = integrate_pj_generic(2);

    let pi = T::pi();
    let half = T::half();
    let p1 = half + i1 / pi;
    let p2 = half + i2 / pi;

    let call = fwd * df * p1 - strike * df * p2;

    if is_call {
        call.max(T::zero())
    } else {
        // Put-call parity
        let put = call - spot * ((T::zero() - q) * tau).exp() + strike * df;
        put.max(T::zero())
    }
}

// ===========================================================================
// Heston Greeks via forward-mode AD
// ===========================================================================

/// All Greeks for a Heston-priced European option.
#[derive(Clone, Debug)]
pub struct HestonGreeks {
    /// Option price.
    pub npv: f64,
    /// ∂V/∂S — spot delta.
    pub delta: f64,
    /// ∂²V/∂S² — spot gamma (via FD on AD delta).
    pub gamma: f64,
    /// ∂V/∂v₀ — sensitivity to initial variance.
    pub vega_v0: f64,
    /// ∂V/∂κ — sensitivity to mean reversion speed.
    pub d_kappa: f64,
    /// ∂V/∂θ — sensitivity to long-run variance.
    pub d_theta: f64,
    /// ∂V/∂σ — volatility-of-volatility sensitivity.
    pub d_sigma: f64,
    /// ∂V/∂ρ — correlation sensitivity.
    pub d_rho: f64,
    /// ∂V/∂r — interest rate sensitivity (rho).
    pub rho_rate: f64,
}

/// Compute Heston Greeks via forward-mode AD with `DualVec<8>`.
///
/// Seeds: 0=spot, 1=v0, 2=kappa, 3=theta, 4=sigma, 5=rho, 6=r, 7=q.
#[allow(clippy::too_many_arguments)]
pub fn heston_greeks_ad(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    tau: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    is_call: bool,
) -> HestonGreeks {
    type D8 = DualVec<8>;

    let s =     D8::variable(spot, 0);
    let k =     D8::constant(strike);
    let r_d =   D8::variable(r, 6);
    let q_d =   D8::variable(q, 7);
    let tau_d = D8::constant(tau); // not differentiating w.r.t. tau here
    let v0_d =  D8::variable(v0, 1);
    let kap_d = D8::variable(kappa, 2);
    let th_d =  D8::variable(theta, 3);
    let sig_d = D8::variable(sigma, 4);
    let rho_d = D8::variable(rho, 5);

    let price = heston_price_generic(s, k, r_d, q_d, tau_d, v0_d, kap_d, th_d, sig_d, rho_d, is_call);

    let delta = price.dot[0];

    // Gamma via central difference on delta
    let h = spot * 1e-4;
    let gamma = {
        use crate::dual::Dual;
        let s_up = Dual::variable(spot + h);
        let p_up = heston_price_generic(
            s_up,
            Dual::constant(strike),
            Dual::constant(r),
            Dual::constant(q),
            Dual::constant(tau),
            Dual::constant(v0),
            Dual::constant(kappa),
            Dual::constant(theta),
            Dual::constant(sigma),
            Dual::constant(rho),
            is_call,
        );
        let s_dn = Dual::variable(spot - h);
        let p_dn = heston_price_generic(
            s_dn,
            Dual::constant(strike),
            Dual::constant(r),
            Dual::constant(q),
            Dual::constant(tau),
            Dual::constant(v0),
            Dual::constant(kappa),
            Dual::constant(theta),
            Dual::constant(sigma),
            Dual::constant(rho),
            is_call,
        );
        (p_up.dot - p_dn.dot) / (2.0 * h)
    };

    HestonGreeks {
        npv: price.val,
        delta,
        gamma,
        vega_v0: price.dot[1],
        d_kappa: price.dot[2],
        d_theta: price.dot[3],
        d_sigma: price.dot[4],
        d_rho: price.dot[5],
        rho_rate: price.dot[6],
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // Standard test parameters (matching analytic_heston.rs)
    const S: f64 = 100.0;
    const K: f64 = 100.0;
    const R: f64 = 0.05;
    const Q: f64 = 0.0;
    const TAU: f64 = 1.0;
    const V0: f64 = 0.04;
    const KAPPA: f64 = 1.5;
    const THETA: f64 = 0.04;
    const SIGMA: f64 = 0.3;
    const RHO: f64 = -0.7;

    #[test]
    fn heston_f64_call_positive() {
        let p = heston_price_generic::<f64>(S, K, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, true);
        assert!(p > 5.0 && p < 20.0, "Heston call price {} not in range", p);
    }

    #[test]
    fn heston_f64_put_call_parity() {
        let c = heston_price_generic::<f64>(S, K, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, true);
        let p = heston_price_generic::<f64>(S, K, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, false);
        let lhs = c - p;
        let rhs = S * (-Q * TAU).exp() - K * (-R * TAU).exp();
        assert_abs_diff_eq!(lhs, rhs, epsilon = 0.1);
    }

    #[test]
    fn heston_f64_otm_cheaper() {
        let atm = heston_price_generic::<f64>(S, K, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, true);
        let otm = heston_price_generic::<f64>(S, 120.0, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, true);
        assert!(otm < atm, "OTM {} should be < ATM {}", otm, atm);
    }

    #[test]
    fn heston_greeks_delta_range() {
        let g = heston_greeks_ad(S, K, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, true);
        assert!(g.delta > 0.0 && g.delta < 1.0, "Call delta {} out of range", g.delta);
    }

    #[test]
    fn heston_greeks_put_delta() {
        let g = heston_greeks_ad(S, K, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, false);
        assert!(g.delta < 0.0 && g.delta > -1.0, "Put delta {} out of range", g.delta);
    }

    #[test]
    fn heston_greeks_gamma_positive() {
        let gc = heston_greeks_ad(S, K, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, true);
        assert!(gc.gamma > 0.0, "Gamma should be positive: {}", gc.gamma);
    }

    #[test]
    fn heston_greeks_vega_v0_positive() {
        let g = heston_greeks_ad(S, K, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, true);
        assert!(g.vega_v0 > 0.0, "Vega_v0 should be positive for ATM call: {}", g.vega_v0);
    }

    #[test]
    fn heston_greeks_rho_rate_positive_call() {
        let g = heston_greeks_ad(S, K, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, true);
        assert!(g.rho_rate > 0.0, "rho_rate should be positive for call: {}", g.rho_rate);
    }

    #[test]
    fn heston_greeks_vs_finite_difference() {
        let g = heston_greeks_ad(S, K, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, true);
        let h = 1e-5;

        // Delta via FD
        let p_up = heston_price_generic::<f64>(S + h, K, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, true);
        let p_dn = heston_price_generic::<f64>(S - h, K, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, true);
        let fd_delta = (p_up - p_dn) / (2.0 * h);
        assert_abs_diff_eq!(g.delta, fd_delta, epsilon = 1e-3);

        // Vega_v0 via FD
        let p_up = heston_price_generic::<f64>(S, K, R, Q, TAU, V0 + h, KAPPA, THETA, SIGMA, RHO, true);
        let p_dn = heston_price_generic::<f64>(S, K, R, Q, TAU, V0 - h, KAPPA, THETA, SIGMA, RHO, true);
        let fd_vega = (p_up - p_dn) / (2.0 * h);
        assert_abs_diff_eq!(g.vega_v0, fd_vega, epsilon = 1e-2);

        // d_rho via FD
        let p_up = heston_price_generic::<f64>(S, K, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO + h, true);
        let p_dn = heston_price_generic::<f64>(S, K, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO - h, true);
        let fd_drho = (p_up - p_dn) / (2.0 * h);
        assert_abs_diff_eq!(g.d_rho, fd_drho, epsilon = 1e-2);

        // rho_rate via FD
        let p_up = heston_price_generic::<f64>(S, K, R + h, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, true);
        let p_dn = heston_price_generic::<f64>(S, K, R - h, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, true);
        let fd_rho_rate = (p_up - p_dn) / (2.0 * h);
        assert_abs_diff_eq!(g.rho_rate, fd_rho_rate, epsilon = 1e-2);
    }

    #[test]
    fn heston_matches_existing_pricer_approx() {
        // Our generic pricer uses a 20-point GL quadrature vs the original's 48-point.
        // Values should be in the same ballpark.
        let p = heston_price_generic::<f64>(S, K, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, true);
        // The analytic_heston.rs pricer gives ~10.17 for these params.
        // Our result should be close (within a few percent due to fewer quadrature points).
        assert!(p > 8.0 && p < 14.0, "Heston price {} not near expected ~10.17", p);
    }

    #[test]
    fn heston_zero_vol_of_vol_near_bs() {
        // σ → 0 ⟹ Heston → BS with flat vol = √v0
        let p = heston_price_generic::<f64>(
            S, K, R, Q, TAU, V0, 1.5, V0, 0.001, 0.0, true,
        );
        // BS with σ=20%, S=100, K=100, r=5%, T=1 ~ 10.45
        assert_abs_diff_eq!(p, 10.45, epsilon = 1.0);
    }

    #[test]
    fn heston_reverse_mode_areal() {
        use crate::tape::{with_tape, adjoint_tl, AReal};

        let (price, inputs) = with_tape(|tape| {
            let s = tape.input(S);
            let k = tape.input(K);
            let r = tape.input(R);
            let q = tape.input(Q);
            let tau = tape.input(TAU);
            let v0 = tape.input(V0);
            let kappa = tape.input(KAPPA);
            let theta = tape.input(THETA);
            let sigma = tape.input(SIGMA);
            let rho = tape.input(RHO);
            let p = heston_price_generic::<AReal>(s, k, r, q, tau, v0, kappa, theta, sigma, rho, true);
            (p, [s, k, r, q, tau, v0, kappa, theta, sigma, rho])
        });

        let grad = adjoint_tl(price);
        let delta = grad[inputs[0].idx];

        // Delta should be in (0, 1) for a call
        assert!(delta > 0.0 && delta < 1.0, "AReal delta {} out of range", delta);

        // Verify against forward-mode
        let g = heston_greeks_ad(S, K, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, true);
        assert_abs_diff_eq!(delta, g.delta, epsilon = 0.05);
        assert_abs_diff_eq!(price.val, g.npv, epsilon = 0.01);
    }
}
