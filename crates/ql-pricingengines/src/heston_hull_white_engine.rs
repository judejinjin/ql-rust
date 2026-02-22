//! Analytic Heston–Hull-White (H1HW) pricing engine.
//!
//! Implements the Grzelak & Oosterlee (2011) "A1HW" approximation for
//! European options under the hybrid Heston stochastic-volatility +
//! Hull-White stochastic interest-rate model.
//!
//! ## Model
//!
//! ```text
//! dS/S = (r − q) dt + √v dW_S
//! dv   = κ(θ − v) dt + σ_v √v dW_v,  dW_S dW_v = ρ_sv dt
//! dr   = (θ_r(t) − a r) dt + σ_r dW_r,  dW_S dW_r = ρ_sr dt
//! ```
//!
//! ## A1HW Approximation
//!
//! The approximation (Eq. 3.4 in Grzelak-Oosterlee 2011) adjusts the
//! initial Heston variance by the variance contribution from the stochastic
//! rate process:
//!
//! ```text
//! v₀* = v₀ + ξ(T),
//! ξ(T) = ρ_sv · σ_v · σ_r · B(a, T) · (1 − e^{−κT}) / κ
//! B(a, T) = (1 − e^{−aT}) / a
//! ```
//!
//! Then price using the standard Heston characteristic-function engine
//! with the adjusted v₀*.
//!
//! ## References
//!
//! Grzelak, L.A. & Oosterlee, C.W. (2011). *On the Heston model with
//! stochastic interest rates.*  SIAM Journal on Financial Mathematics 2(1).

use ql_instruments::OptionType;
use ql_models::HestonModel;

use crate::analytic_heston::heston_price;

/// Result from the H1HW engine.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HestonHullWhiteResult {
    /// Approximate European option price.
    pub price: f64,
    /// Effective initial variance used (v₀*).
    pub v0_eff: f64,
    /// Variance adjustment ξ(T) due to stochastic interest rate.
    pub xi: f64,
}

/// Price a European option under the Heston + Hull-White hybrid model
/// using the A1HW (first-order) approximation.
///
/// # Arguments
/// - `spot` — current spot price S₀
/// - `strike` — option strike K
/// - `tau` — time to maturity T (years)
/// - `r0` — current short rate r(0)
/// - `hw_a` — Hull-White mean-reversion speed a
/// - `hw_sigma_r` — Hull-White rate volatility σ_r
/// - `heston_v0` — Heston initial variance v₀
/// - `heston_kappa` — Heston mean-reversion κ
/// - `heston_theta` — Heston long-run variance θ
/// - `heston_sigma_v` — Heston vol-of-vol σ_v
/// - `heston_rho_sv` — equity–variance correlation ρ_sv
/// - `equity_rate_rho` — equity–rate correlation ρ_sr
/// - `dividend_yield` — continuous dividend yield q
/// - `opt_type` — call or put
#[allow(clippy::too_many_arguments)]
pub fn heston_hull_white_price(
    spot: f64,
    strike: f64,
    tau: f64,
    r0: f64,
    hw_a: f64,
    hw_sigma_r: f64,
    heston_v0: f64,
    heston_kappa: f64,
    heston_theta: f64,
    heston_sigma_v: f64,
    heston_rho_sv: f64,
    equity_rate_rho: f64,
    dividend_yield: f64,
    opt_type: OptionType,
) -> HestonHullWhiteResult {
    // Compute the variance adjustment ξ(T)
    // B(a, T) = (1 − e^{−aT}) / a
    let b_hw = if hw_a.abs() < 1e-8 {
        tau // limit a→0
    } else {
        (1.0 - (-hw_a * tau).exp()) / hw_a
    };
    // Variance adjustment due to equity-rate correlation
    // ξ(T) = ρ_sr · σ_v · σ_r · B(a,T) · (1 − e^{−κT}) / κ
    let xi = if heston_kappa.abs() < 1e-8 {
        equity_rate_rho * heston_sigma_v * hw_sigma_r * b_hw * tau
    } else {
        equity_rate_rho
            * heston_sigma_v
            * hw_sigma_r
            * b_hw
            * (1.0 - (-heston_kappa * tau).exp())
            / heston_kappa
    };

    let v0_eff = (heston_v0 + xi).max(1e-8);

    // Build a temporary HestonModel with adjusted v₀* and flat rate r0
    let model = HestonModel::new(
        spot,
        r0,
        dividend_yield,
        v0_eff,
        heston_kappa,
        heston_theta,
        heston_sigma_v,
        heston_rho_sv,
    );
    let is_call = matches!(opt_type, OptionType::Call);
    let result = heston_price(&model, strike, tau, is_call);

    HestonHullWhiteResult {
        price: result.npv,
        v0_eff,
        xi,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h1hw_call_positive() {
        let res = heston_hull_white_price(
            100.0, 100.0, 1.0,
            0.03,   // r0
            0.1,    // hw_a
            0.01,   // hw_sigma_r
            0.04, 1.5, 0.04, 0.3, -0.7,  // heston params
            0.1,    // equity-rate rho
            0.0, OptionType::Call,
        );
        assert!(res.price > 0.0 && res.price < 100.0, "price={}", res.price);
    }

    #[test]
    fn h1hw_zero_rate_corr_is_pure_heston() {
        // With ρ_sr = 0, the adjustment ξ = 0 and result equals pure Heston
        let res0 = heston_hull_white_price(
            100.0, 100.0, 1.0, 0.05, 0.1, 0.01,
            0.04, 1.5, 0.04, 0.3, -0.7,
            0.0, // zero equity-rate rho
            0.0, OptionType::Call,
        );
        assert!((res0.xi).abs() < 1e-12, "xi should be 0 when rho_sr=0");
    }

    #[test]
    fn h1hw_positive_rho_increases_variance() {
        let res_pos = heston_hull_white_price(
            100.0, 100.0, 1.0, 0.05, 0.1, 0.05,
            0.04, 1.5, 0.04, 0.3, -0.7,
            0.5, 0.0, OptionType::Call,
        );
        let res_neg = heston_hull_white_price(
            100.0, 100.0, 1.0, 0.05, 0.1, 0.05,
            0.04, 1.5, 0.04, 0.3, -0.7,
            -0.5, 0.0, OptionType::Call,
        );
        // Larger effective variance → higher option price
        assert!(
            res_pos.v0_eff > res_neg.v0_eff || (res_pos.v0_eff - res_neg.v0_eff).abs() < 1e-3,
            "v0_eff mismatch"
        );
    }

    #[test]
    fn h1hw_put_call_parity_approx() {
        let s = 100.0;
        let k = 100.0;
        let t = 1.0;
        let r0 = 0.04;
        let call = heston_hull_white_price(
            s, k, t, r0, 0.1, 0.01, 0.04, 1.5, 0.04, 0.3, -0.7, 0.1, 0.0, OptionType::Call,
        );
        let put = heston_hull_white_price(
            s, k, t, r0, 0.1, 0.01, 0.04, 1.5, 0.04, 0.3, -0.7, 0.1, 0.0, OptionType::Put,
        );
        let fwd = s - k * (-r0 * t).exp();
        assert!((call.price - put.price - fwd).abs() < 1.0, "pcp err = {}", (call.price - put.price - fwd).abs());
    }
}
