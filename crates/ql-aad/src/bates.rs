//! Generic Bates (Heston + Merton jumps) pricing engine with AD-aware Greeks.
//!
//! Bates extends Heston by adding a log-normal jump component characterised by
//! three extra parameters: jump intensity `lambda`, mean log-jump `nu`, and
//! jump-size volatility `delta`.
//!
//! The Bates characteristic function is:
//!
//!   f_Bates(φ) = f_Heston(φ) · exp(λτ [e^{iφν − ½δ²φ²} − 1 − iφk̄])
//!
//! where k̄ = e^{ν + δ²/2} − 1 is the jump compensator ensuring the
//! risk-neutral drift is preserved.
//!
//! # Example
//!
//! ```
//! use ql_aad::bates::{bates_greeks_ad, BatesGreeks};
//!
//! let g = bates_greeks_ad(
//!     100.0, 100.0, 0.05, 0.0, 1.0,      // spot, strike, r, q, tau
//!     0.04, 1.5, 0.04, 0.3, -0.7,        // v0, kappa, theta, sigma, rho
//!     0.5, -0.1, 0.15,                     // lambda, nu, delta
//!     true,                                 // is_call
//! );
//! assert!(g.npv > 5.0 && g.npv < 25.0);
//! assert!(g.delta > 0.0 && g.delta < 1.0);
//! ```

use crate::complex::Complex;
use crate::dual_vec::DualVec;
use crate::heston::{gauss_legendre_48, heston_cf_generic};
use crate::number::Number;

// ===========================================================================
// Bates characteristic function (generic)
// ===========================================================================

/// Generic Bates CF: f_Bates = f_Heston · exp(jump_term).
///
/// `lambda` — jump intensity  
/// `nu`     — mean of log-jump size  
/// `delta`  — std-dev of log-jump size  
#[allow(clippy::too_many_arguments)]
pub fn bates_cf_generic<T: Number>(
    phi: f64,
    v0: T,
    kappa: T,
    theta: T,
    sigma: T,
    rho: T,
    tau: T,
    lambda: T,
    nu: T,
    delta_j: T,
    j: u8,
) -> Complex<T> {
    let f_heston = heston_cf_generic(phi, v0, kappa, theta, sigma, rho, tau, j);

    // Jump compensator: k_bar = exp(nu + delta^2/2) - 1
    let half = T::half();
    let k_bar = (nu + half * delta_j * delta_j).exp() - T::one();

    // Jump CF: exp(i*phi*nu - 0.5*delta^2*phi^2)
    let phi_t = T::from_f64(phi);
    let jump_cf_re = T::zero() - half * delta_j * delta_j * phi_t * phi_t;
    let jump_cf_im = phi_t * nu;
    let jump_cf = Complex::new(jump_cf_re, jump_cf_im).exp();

    // jump_term = lambda * tau * (jump_cf - 1 - i*phi*k_bar)
    let one_c = Complex::from_real(T::one());
    let i_phi_kbar = Complex::new(T::zero(), phi_t * k_bar);
    let jump_term = (jump_cf - one_c - i_phi_kbar).scale(lambda * tau);

    // f_Bates = f_Heston * exp(jump_term)
    f_heston * jump_term.exp()
}

// ===========================================================================
// Bates price (generic)
// ===========================================================================

/// Generic Bates price for a European option.
///
/// All 11 model parameters are `T: Number`.
#[allow(clippy::too_many_arguments)]
pub fn bates_price_generic<T: Number>(
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
    lambda: T,
    nu: T,
    delta_j: T,
    is_call: bool,
) -> T {
    let neg_one = T::zero() - T::one();
    let df = (neg_one * r * tau).exp();
    let fwd = spot * ((r - q) * tau).exp();
    let x = (fwd / strike).ln();

    let v0_f64 = v0.to_f64();
    let tau_f64 = tau.to_f64();
    let upper = (50.0 / (v0_f64 * tau_f64).sqrt().max(0.05)).clamp(50.0, 200.0);

    let integrate_pj = |j: u8| -> T {
        let half_range = 0.5 * (upper - 1e-8);
        let mid = 0.5 * (1e-8 + upper);
        let nodes = gauss_legendre_48();
        let mut sum = T::zero();
        for &(xi, wi) in nodes {
            let phi = mid + half_range * xi;
            if phi < 1e-8 {
                continue;
            }
            let f = bates_cf_generic(phi, v0, kappa, theta, sigma, rho, tau, lambda, nu, delta_j, j);
            let phi_x = T::from_f64(phi) * x;
            let cos_px = phi_x.cos();
            let sin_px = phi_x.sin();
            let integrand = (f.re * sin_px + f.im * cos_px) / T::from_f64(phi);
            sum += integrand * T::from_f64(wi * half_range);
        }
        sum
    };

    let i1 = integrate_pj(1);
    let i2 = integrate_pj(2);

    let pi = T::pi();
    let half = T::half();
    let p1 = half + i1 / pi;
    let p2 = half + i2 / pi;

    let call = fwd * df * p1 - strike * df * p2;

    if is_call {
        call.max(T::zero())
    } else {
        let put = call - spot * ((T::zero() - q) * tau).exp() + strike * df;
        put.max(T::zero())
    }
}

// ===========================================================================
// Bates Greeks via forward-mode AD
// ===========================================================================

/// All Greeks for a Bates-priced European option.
#[derive(Clone, Debug)]
pub struct BatesGreeks {
    /// Option price.
    pub npv: f64,
    /// ∂V/∂S — spot delta.
    pub delta: f64,
    /// ∂²V/∂S² — spot gamma (via FD on AD delta).
    pub gamma: f64,
    /// ∂V/∂v₀
    pub vega_v0: f64,
    /// ∂V/∂κ
    pub d_kappa: f64,
    /// ∂V/∂θ (long-run variance)
    pub d_theta: f64,
    /// ∂V/∂σ (vol-of-vol)
    pub d_sigma: f64,
    /// ∂V/∂ρ (correlation)
    pub d_rho: f64,
    /// ∂V/∂r (interest rate)
    pub rho_rate: f64,
    /// ∂V/∂λ (jump intensity)
    pub d_lambda: f64,
    /// ∂V/∂ν (mean log-jump)
    pub d_nu: f64,
    /// ∂V/∂δ (jump-size vol)
    pub d_delta_j: f64,
}

/// Compute Bates Greeks via forward-mode AD with `DualVec<11>`.
///
/// Seeds: 0=spot, 1=v0, 2=kappa, 3=theta, 4=sigma, 5=rho, 6=r,
///        7=lambda, 8=nu, 9=delta_j, 10=q.
#[allow(clippy::too_many_arguments)]
pub fn bates_greeks_ad(
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
    lambda: f64,
    nu: f64,
    delta_j: f64,
    is_call: bool,
) -> BatesGreeks {
    type D11 = DualVec<11>;

    let s      = D11::variable(spot, 0);
    let k      = D11::constant(strike);
    let r_d    = D11::variable(r, 6);
    let q_d    = D11::variable(q, 10);
    let tau_d  = D11::constant(tau);
    let v0_d   = D11::variable(v0, 1);
    let kap_d  = D11::variable(kappa, 2);
    let th_d   = D11::variable(theta, 3);
    let sig_d  = D11::variable(sigma, 4);
    let rho_d  = D11::variable(rho, 5);
    let lam_d  = D11::variable(lambda, 7);
    let nu_d   = D11::variable(nu, 8);
    let dj_d   = D11::variable(delta_j, 9);

    let price = bates_price_generic(s, k, r_d, q_d, tau_d, v0_d, kap_d, th_d, sig_d, rho_d, lam_d, nu_d, dj_d, is_call);

    let ad_delta = price.dot[0];

    // Gamma via central difference on delta
    let h = spot * 1e-4;
    let gamma = {
        let d_up = heston_price_dual(spot + h, strike, r, q, tau, v0, kappa, theta, sigma, rho, lambda, nu, delta_j, is_call);
        let d_dn = heston_price_dual(spot - h, strike, r, q, tau, v0, kappa, theta, sigma, rho, lambda, nu, delta_j, is_call);
        (d_up - d_dn) / (2.0 * h)
    };

    BatesGreeks {
        npv: price.val,
        delta: ad_delta,
        gamma,
        vega_v0: price.dot[1],
        d_kappa: price.dot[2],
        d_theta: price.dot[3],
        d_sigma: price.dot[4],
        d_rho: price.dot[5],
        rho_rate: price.dot[6],
        d_lambda: price.dot[7],
        d_nu: price.dot[8],
        d_delta_j: price.dot[9],
    }
}

/// Helper: compute Bates ∂V/∂S via Dual numberto get delta at a shifted spot.
#[allow(clippy::too_many_arguments)]
fn heston_price_dual(
    spot_val: f64,
    strike: f64,
    r: f64,
    q: f64,
    tau: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    lambda: f64,
    nu: f64,
    delta_j: f64,
    is_call: bool,
) -> f64 {
    use crate::dual::Dual;
    let p = bates_price_generic(
        Dual::variable(spot_val),
        Dual::constant(strike),
        Dual::constant(r),
        Dual::constant(q),
        Dual::constant(tau),
        Dual::constant(v0),
        Dual::constant(kappa),
        Dual::constant(theta),
        Dual::constant(sigma),
        Dual::constant(rho),
        Dual::constant(lambda),
        Dual::constant(nu),
        Dual::constant(delta_j),
        is_call,
    );
    p.dot // delta at this spot
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

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
    const LAMBDA: f64 = 0.5;
    const NU: f64 = -0.1;
    const DELTA_J: f64 = 0.15;

    #[test]
    fn bates_f64_call_positive() {
        let p = bates_price_generic::<f64>(S, K, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, LAMBDA, NU, DELTA_J, true);
        assert!(p > 5.0 && p < 25.0, "Bates call price {} not in range", p);
    }

    #[test]
    fn bates_f64_put_call_parity() {
        let c = bates_price_generic::<f64>(S, K, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, LAMBDA, NU, DELTA_J, true);
        let p = bates_price_generic::<f64>(S, K, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, LAMBDA, NU, DELTA_J, false);
        let lhs = c - p;
        let rhs = S * (-Q * TAU).exp() - K * (-R * TAU).exp();
        assert_abs_diff_eq!(lhs, rhs, epsilon = 0.1);
    }

    #[test]
    fn bates_reduces_to_heston_lambda_zero() {
        // λ=0 ⟹ Bates = Heston
        let heston_p = crate::heston::heston_price_generic::<f64>(S, K, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, true);
        let bates_p = bates_price_generic::<f64>(S, K, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, 0.0, 0.0, 0.01, true);
        assert_abs_diff_eq!(heston_p, bates_p, epsilon = 1e-6);
    }

    #[test]
    fn bates_jumps_shift_price() {
        let no_jump = bates_price_generic::<f64>(S, K, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, 0.0, 0.0, 0.01, true);
        let with_jump = bates_price_generic::<f64>(S, K, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, LAMBDA, NU, DELTA_J, true);
        assert!((with_jump - no_jump).abs() > 0.1, "Jumps should change price");
    }

    #[test]
    fn bates_greeks_delta_range() {
        let g = bates_greeks_ad(S, K, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, LAMBDA, NU, DELTA_J, true);
        assert!(g.delta > 0.0 && g.delta < 1.0, "Call delta {} out of range", g.delta);
    }

    #[test]
    fn bates_greeks_put_delta() {
        let g = bates_greeks_ad(S, K, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, LAMBDA, NU, DELTA_J, false);
        assert!(g.delta < 0.0 && g.delta > -1.0, "Put delta {} out of range", g.delta);
    }

    #[test]
    fn bates_greeks_gamma_positive() {
        let g = bates_greeks_ad(S, K, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, LAMBDA, NU, DELTA_J, true);
        assert!(g.gamma > 0.0, "Gamma should be positive: {}", g.gamma);
    }

    #[test]
    fn bates_greeks_vs_finite_difference() {
        let g = bates_greeks_ad(S, K, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, LAMBDA, NU, DELTA_J, true);
        let h = 1e-5;

        // Delta via FD
        let p_up = bates_price_generic::<f64>(S + h, K, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, LAMBDA, NU, DELTA_J, true);
        let p_dn = bates_price_generic::<f64>(S - h, K, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, LAMBDA, NU, DELTA_J, true);
        let fd_delta = (p_up - p_dn) / (2.0 * h);
        assert_abs_diff_eq!(g.delta, fd_delta, epsilon = 1e-3);

        // d_lambda via FD
        let p_up = bates_price_generic::<f64>(S, K, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, LAMBDA + h, NU, DELTA_J, true);
        let p_dn = bates_price_generic::<f64>(S, K, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, LAMBDA - h, NU, DELTA_J, true);
        let fd_dlam = (p_up - p_dn) / (2.0 * h);
        assert_abs_diff_eq!(g.d_lambda, fd_dlam, epsilon = 1e-2);

        // d_nu via FD
        let p_up = bates_price_generic::<f64>(S, K, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, LAMBDA, NU + h, DELTA_J, true);
        let p_dn = bates_price_generic::<f64>(S, K, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, LAMBDA, NU - h, DELTA_J, true);
        let fd_dnu = (p_up - p_dn) / (2.0 * h);
        assert_abs_diff_eq!(g.d_nu, fd_dnu, epsilon = 1e-2);
    }

    #[test]
    fn bates_reverse_mode_areal() {
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
            let lam = tape.input(LAMBDA);
            let nu = tape.input(NU);
            let dj = tape.input(DELTA_J);
            let p = bates_price_generic::<AReal>(s, k, r, q, tau, v0, kappa, theta, sigma, rho, lam, nu, dj, true);
            (p, [s, k, r, q, tau, v0, kappa, theta, sigma, rho, lam, nu, dj])
        });

        let grad = adjoint_tl(price);
        let delta = grad[inputs[0].idx];
        let d_lambda = grad[inputs[10].idx];

        assert!(delta > 0.0 && delta < 1.0, "AReal delta {} out of range", delta);
        assert!(d_lambda.abs() > 0.0, "d_lambda should be nonzero");

        // Check against forward-mode
        let g = bates_greeks_ad(S, K, R, Q, TAU, V0, KAPPA, THETA, SIGMA, RHO, LAMBDA, NU, DELTA_J, true);
        assert_abs_diff_eq!(delta, g.delta, epsilon = 0.05);
        assert_abs_diff_eq!(price.val, g.npv, epsilon = 0.01);
    }
}
