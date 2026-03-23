//! Generic Black-Scholes pricer and forward-mode AD-based Greeks calculator.
//!
//! The core function [`bs_price_generic`] computes the Black-Scholes price
//! using generic arithmetic over any `T: Number`. By instantiating `T` as
//! `DualVec<5>`, we obtain all first-order Greeks (delta, vega, theta, rho,
//! and gamma via second bump) in a single forward pass.
//!
//! # Example
//!
//! ```
//! use ql_aad::{OptionKind, bs_greeks_forward_ad};
//!
//! let greeks = bs_greeks_forward_ad(100.0, 100.0, 0.05, 0.02, 0.20, 1.0, OptionKind::Call);
//! assert!((greeks.npv - 9.227).abs() < 0.01);
//! assert!((greeks.delta - 0.587).abs() < 0.01);
//! ```

use crate::dual_vec::DualVec;
use crate::math::normal_cdf;
use crate::number::Number;

/// Option type: Call or Put.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionKind {
    /// Call.
    Call,
    /// Put.
    Put,
}

/// Black-Scholes Greeks computed via forward-mode AD.
#[derive(Debug, Clone)]
pub struct BSGreeks {
    /// Net present value (option price).
    pub npv: f64,
    /// Delta: ∂V/∂S.
    pub delta: f64,
    /// Gamma: ∂²V/∂S² (via finite difference on the AD delta).
    pub gamma: f64,
    /// Vega: ∂V/∂σ.
    pub vega: f64,
    /// Theta: -∂V/∂T (negative because time decreases).
    pub theta: f64,
    /// Rho: ∂V/∂r.
    pub rho: f64,
}

/// Generic Black-Scholes price for a European option.
///
/// # Arguments
///
/// * `spot`   – Current spot price S
/// * `strike` – Strike price K
/// * `r`      – Risk-free interest rate
/// * `q`      – Continuous dividend yield
/// * `vol`    – Volatility σ
/// * `t`      – Time to expiry in years
/// * `kind`   – Call or Put
///
/// # Formula
///
/// For a call:  V = S·e^{-qT}·N(d₁) − K·e^{-rT}·N(d₂)
/// For a put:   V = K·e^{-rT}·N(−d₂) − S·e^{-qT}·N(−d₁)
///
/// where d₁ = [ln(S/K) + (r − q + σ²/2)T] / (σ√T)
///       d₂ = d₁ − σ√T
pub fn bs_price_generic<T: Number>(
    spot: T,
    strike: T,
    r: T,
    q: T,
    vol: T,
    t: T,
    kind: OptionKind,
) -> T {
    let sqrt_t = t.sqrt();
    let vol_sqrt_t = vol * sqrt_t;
    let half = T::half();

    // d1, d2
    let d1 = ((spot / strike).ln() + (r - q + half * vol * vol) * t) / vol_sqrt_t;
    let d2 = d1 - vol_sqrt_t;

    // Discount factors
    let neg_one = T::zero() - T::one();
    let df_r = (neg_one * r * t).exp();
    let df_q = (neg_one * q * t).exp();

    match kind {
        OptionKind::Call => {
            spot * df_q * normal_cdf(d1) - strike * df_r * normal_cdf(d2)
        }
        OptionKind::Put => {
            strike * df_r * normal_cdf(T::zero() - d2) - spot * df_q * normal_cdf(T::zero() - d1)
        }
    }
}

/// Compute all Black-Scholes Greeks using forward-mode AD with `DualVec<5>`.
///
/// Seeds derivatives for: spot(0), vol(1), t(2), r(3), q(4).
/// Gamma is obtained via finite-difference bump on spot.
///
/// # Arguments
///
/// Same as [`bs_price_generic`] but all inputs are plain `f64`.
pub fn bs_greeks_forward_ad(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    kind: OptionKind,
) -> BSGreeks {
    // Seed indices: 0=spot, 1=vol, 2=t, 3=r, 4=q
    let s = DualVec::<5>::variable(spot, 0);
    let k = DualVec::<5>::constant(strike);
    let r_d = DualVec::<5>::variable(r, 3);
    let q_d = DualVec::<5>::variable(q, 4);
    let v_d = DualVec::<5>::variable(vol, 1);
    let t_d = DualVec::<5>::variable(t, 2);

    let price = bs_price_generic(s, k, r_d, q_d, v_d, t_d, kind);

    let delta = price.dot[0];
    let vega = price.dot[1];
    let dv_dt = price.dot[2];
    let rho = price.dot[3];

    // Gamma via central finite-difference on delta.
    // We compute delta at S+h and S-h using Dual (single-seed) and take
    // (delta_up - delta_dn) / (2h). This is more accurate than a one-sided bump.
    let h = spot * 1e-4;
    let gamma = {
        use crate::dual::Dual;
        let s_up = Dual::variable(spot + h);
        let k_c = Dual::constant(strike);
        let r_c = Dual::constant(r);
        let q_c = Dual::constant(q);
        let v_c = Dual::constant(vol);
        let t_c = Dual::constant(t);
        let price_up = bs_price_generic(s_up, k_c, r_c, q_c, v_c, t_c, kind);
        let delta_up = price_up.dot;

        let s_dn = Dual::variable(spot - h);
        let price_dn = bs_price_generic(s_dn, k_c, r_c, q_c, v_c, t_c, kind);
        let delta_dn = price_dn.dot;

        (delta_up - delta_dn) / (2.0 * h)
    };

    // Theta convention: negative of ∂V/∂T (time passes, T decreases)
    // Commonly expressed as per-day: theta_day = theta / 365
    let theta = -dv_dt;

    BSGreeks {
        npv: price.val,
        delta,
        gamma,
        vega,
        theta,
        rho,
    }
}

/// Compute Black-Scholes price using plain f64 (essentially [`bs_price_generic::<f64>`]).
///
/// Useful as a convenience function and for validation.
#[inline]
pub fn bs_price_f64(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    kind: OptionKind,
) -> f64 {
    bs_price_generic::<f64>(spot, strike, r, q, vol, t, kind)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // Reference values from QuantLib / analytic formulae
    // S=100, K=100, r=5%, q=2%, σ=20%, T=1Y, Call
    const S: f64 = 100.0;
    const K: f64 = 100.0;
    const R: f64 = 0.05;
    const Q: f64 = 0.02;
    const VOL: f64 = 0.20;
    const T: f64 = 1.0;

    #[test]
    fn bs_call_price() {
        let p = bs_price_f64(S, K, R, Q, VOL, T, OptionKind::Call);
        // Expected ~9.227 (BS analytic with q=2%)
        assert_abs_diff_eq!(p, 9.227, epsilon = 0.05);
    }

    #[test]
    fn bs_put_price() {
        let p = bs_price_f64(S, K, R, Q, VOL, T, OptionKind::Put);
        // Put-call parity: P = C - S*exp(-qT) + K*exp(-rT)
        let c = bs_price_f64(S, K, R, Q, VOL, T, OptionKind::Call);
        let parity = c - S * (-Q * T).exp() + K * (-R * T).exp();
        assert_abs_diff_eq!(p, parity, epsilon = 1e-6);
    }

    #[test]
    fn put_call_parity() {
        let c = bs_price_f64(S, K, R, Q, VOL, T, OptionKind::Call);
        let p = bs_price_f64(S, K, R, Q, VOL, T, OptionKind::Put);
        let lhs = c - p;
        let rhs = S * (-Q * T).exp() - K * (-R * T).exp();
        assert_abs_diff_eq!(lhs, rhs, epsilon = 1e-6);
    }

    #[test]
    fn greeks_call_delta() {
        let g = bs_greeks_forward_ad(S, K, R, Q, VOL, T, OptionKind::Call);
        // Delta should be in (0, 1) for a call.
        assert!(g.delta > 0.0 && g.delta < 1.0);
        // ATM call delta ~ 0.587 for these parameters (with q=2%)
        assert_abs_diff_eq!(g.delta, 0.587, epsilon = 0.01);
    }

    #[test]
    fn greeks_put_delta() {
        let g = bs_greeks_forward_ad(S, K, R, Q, VOL, T, OptionKind::Put);
        // Put delta should be in (-1, 0)
        assert!(g.delta > -1.0 && g.delta < 0.0);
    }

    #[test]
    fn greeks_gamma_positive() {
        let gc = bs_greeks_forward_ad(S, K, R, Q, VOL, T, OptionKind::Call);
        let gp = bs_greeks_forward_ad(S, K, R, Q, VOL, T, OptionKind::Put);
        // Gamma is the same for calls and puts
        assert!(gc.gamma > 0.0);
        assert_abs_diff_eq!(gc.gamma, gp.gamma, epsilon = 1e-4);
    }

    #[test]
    fn greeks_vega_positive() {
        let g = bs_greeks_forward_ad(S, K, R, Q, VOL, T, OptionKind::Call);
        assert!(g.vega > 0.0);
        // Vega for ATM ~38-40 (per 100% vol move; per 1% move ~ 0.38)
        assert_abs_diff_eq!(g.vega, 37.52, epsilon = 0.5);
    }

    #[test]
    fn greeks_theta_negative_for_call() {
        let g = bs_greeks_forward_ad(S, K, R, Q, VOL, T, OptionKind::Call);
        // Theta (annual) is typically negative for long options
        assert!(g.theta < 0.0);
    }

    #[test]
    fn greeks_rho_call_positive() {
        let g = bs_greeks_forward_ad(S, K, R, Q, VOL, T, OptionKind::Call);
        // Rho for a call is positive (higher rates → higher call value)
        assert!(g.rho > 0.0);
    }

    #[test]
    fn greeks_vs_finite_difference() {
        // Validate AD Greeks against finite-difference bumps
        let g = bs_greeks_forward_ad(S, K, R, Q, VOL, T, OptionKind::Call);
        let h = 1e-6;

        // Delta via FD
        let p_up = bs_price_f64(S + h, K, R, Q, VOL, T, OptionKind::Call);
        let p_dn = bs_price_f64(S - h, K, R, Q, VOL, T, OptionKind::Call);
        let fd_delta = (p_up - p_dn) / (2.0 * h);
        assert_abs_diff_eq!(g.delta, fd_delta, epsilon = 1e-4);

        // Gamma via FD — use larger bump for second derivative to avoid
        // catastrophic cancellation (h² in denominator amplifies roundoff).
        let h_gamma = 0.01;
        let p_up_g = bs_price_f64(S + h_gamma, K, R, Q, VOL, T, OptionKind::Call);
        let p_dn_g = bs_price_f64(S - h_gamma, K, R, Q, VOL, T, OptionKind::Call);
        let fd_gamma = (p_up_g - 2.0 * g.npv + p_dn_g) / (h_gamma * h_gamma);
        assert_abs_diff_eq!(g.gamma, fd_gamma, epsilon = 1e-4);

        // Vega via FD
        let pv_up = bs_price_f64(S, K, R, Q, VOL + h, T, OptionKind::Call);
        let pv_dn = bs_price_f64(S, K, R, Q, VOL - h, T, OptionKind::Call);
        let fd_vega = (pv_up - pv_dn) / (2.0 * h);
        assert_abs_diff_eq!(g.vega, fd_vega, epsilon = 1e-3);

        // Rho via FD
        let pr_up = bs_price_f64(S, K, R + h, Q, VOL, T, OptionKind::Call);
        let pr_dn = bs_price_f64(S, K, R - h, Q, VOL, T, OptionKind::Call);
        let fd_rho = (pr_up - pr_dn) / (2.0 * h);
        assert_abs_diff_eq!(g.rho, fd_rho, epsilon = 1e-3);

        // Theta via FD (remember theta = -dV/dT)
        let pt_up = bs_price_f64(S, K, R, Q, VOL, T + h, OptionKind::Call);
        let pt_dn = bs_price_f64(S, K, R, Q, VOL, T - h, OptionKind::Call);
        let fd_theta = -(pt_up - pt_dn) / (2.0 * h);
        assert_abs_diff_eq!(g.theta, fd_theta, epsilon = 1e-3);
    }

    #[test]
    fn zero_vol_intrinsic() {
        // With near-zero vol, call price → max(S·e^{-qT} - K·e^{-rT}, 0).
        let itm_call = bs_price_f64(120.0, 100.0, R, Q, 0.001, T, OptionKind::Call);
        let intrinsic = 120.0 * (-Q * T).exp() - 100.0 * (-R * T).exp();
        assert_abs_diff_eq!(itm_call, intrinsic, epsilon = 0.5);
    }

    #[test]
    fn deep_itm_call_delta_near_one() {
        let g = bs_greeks_forward_ad(200.0, 100.0, R, Q, VOL, T, OptionKind::Call);
        assert!(g.delta > 0.95);
    }

    #[test]
    fn deep_otm_call_delta_near_zero() {
        let g = bs_greeks_forward_ad(50.0, 100.0, R, Q, VOL, T, OptionKind::Call);
        assert!(g.delta < 0.05);
    }
}
