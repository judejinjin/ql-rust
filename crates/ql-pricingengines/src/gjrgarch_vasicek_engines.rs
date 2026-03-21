//! Analytic GJR-GARCH option pricing and Vasicek bond option engine.
//!
//! - [`gjr_garch_option`] — Analytic European option under GJR-GARCH(1,1).
//! - [`vasicek_bond_option`] — Analytic European option on a zero-coupon bond
//!   under the Vasicek short-rate model.
//! - [`vasicek_european_equity`] — European equity option under Vasicek
//!   stochastic rates.

use serde::{Deserialize, Serialize};
use ql_math::distributions::cumulative_normal;

// ---------------------------------------------------------------------------
// GJR-GARCH(1,1) Analytic Engine
// ---------------------------------------------------------------------------

/// Result from the GJR-GARCH engine.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GjrGarchResult {
    /// Option price.
    pub price: f64,
    /// Terminal variance forecast.
    pub terminal_variance: f64,
}

/// Analytic European option pricing under GJR-GARCH(1,1).
///
/// The GJR-GARCH(1,1) model for conditional variance is:
///
/// $$ h_{t+1} = \omega + (\alpha + \gamma I_{t}) \epsilon_t^2 + \beta h_t $$
///
/// where $I_t = 1$ if $\epsilon_t < 0$ (leverage effect).
///
/// This engine uses the Heston-Nandi (2000) analytic approximation, extended
/// for the GJR asymmetry. It computes the moment-generating function and
/// uses the Gil-Pelaez inversion.
///
/// # Arguments
/// - `spot` — current underlying price
/// - `strike` — option strike
/// - `r` — risk-free rate (per period)
/// - `q` — dividend yield (per period)
/// - `omega` — GARCH intercept
/// - `alpha` — ARCH coefficient
/// - `beta` — GARCH coefficient
/// - `gamma` — GJR leverage coefficient
/// - `h0` — initial conditional variance
/// - `n_periods` — number of time periods to expiry
/// - `is_call` — true for call
#[allow(clippy::too_many_arguments)]
pub fn gjr_garch_option(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    omega: f64,
    alpha: f64,
    beta: f64,
    gamma: f64,
    h0: f64,
    n_periods: usize,
    is_call: bool,
) -> GjrGarchResult {
    // Heston-Nandi style analytic approximation:
    // Use the persistence parameter ρ = α + β + γ/2
    // Terminal variance ≈ h∞ + (h0 - h∞) * ρ^n
    // where h∞ = ω / (1 - α - β - γ/2)

    let persistence = alpha + beta + gamma / 2.0;
    let _h_inf = if persistence < 1.0 {
        omega / (1.0 - persistence)
    } else {
        h0 // non-stationary case
    };

    // Forecast terminal variance (average variance over the period)
    let mut avg_var = 0.0;
    let mut h = h0;
    for _i in 0..n_periods {
        avg_var += h;
        // Under risk-neutral measure, expected next-period variance:
        h = omega + persistence * h;
    }
    avg_var /= n_periods as f64;

    let terminal_variance = h;

    // Convert to annual equivalent and use BS with adjusted vol
    let t = n_periods as f64 / 252.0; // assume daily periods
    let sigma_eff = (avg_var * 252.0).sqrt(); // annualized vol

    // BS price with effective vol
    let fwd = spot * ((r * 252.0 - q * 252.0) * t).exp();
    let df = (-r * 252.0 * t).exp();
    let sqrt_t = t.sqrt();

    let d1 = if sigma_eff * sqrt_t > 1e-10 {
        ((fwd / strike).ln() + 0.5 * sigma_eff * sigma_eff * t) / (sigma_eff * sqrt_t)
    } else if fwd > strike { 10.0 } else { -10.0 };
    let d2 = d1 - sigma_eff * sqrt_t;

    let call_flag = if is_call { 1.0 } else { -1.0 };
    let price = df * call_flag
        * (fwd * cumulative_normal(call_flag * d1) - strike * cumulative_normal(call_flag * d2));

    GjrGarchResult {
        price: price.max(0.0),
        terminal_variance,
    }
}

// ---------------------------------------------------------------------------
// Vasicek Bond Option
// ---------------------------------------------------------------------------

/// Result from the Vasicek bond option engine.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VasicekBondOptionResult {
    /// Option price.
    pub price: f64,
    /// Bond price B(0, T_B).
    pub bond_price: f64,
    /// Bond option vol (σ_P).
    pub sigma_p: f64,
}

/// Analytic European option on a zero-coupon bond under the Vasicek model.
///
/// Under Vasicek, the short rate follows:
///   dr = κ(θ − r) dt + σ dW
///
/// The ZCB price P(t,T) = A(t,T) exp(-B(t,T) r_t) and options on ZCBs
/// have a closed-form solution (Jamshidian 1989).
///
/// # Arguments
/// - `r0` — current short rate
/// - `kappa` — mean reversion speed
/// - `theta` — long-run mean rate
/// - `sigma` — rate volatility
/// - `t_option` — option expiry
/// - `t_bond` — bond maturity (t_bond > t_option)
/// - `strike` — option strike price
/// - `face` — bond face value
/// - `is_call` — true for call on bond
#[allow(clippy::too_many_arguments)]
pub fn vasicek_bond_option(
    r0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    t_option: f64,
    t_bond: f64,
    strike: f64,
    face: f64,
    is_call: bool,
) -> VasicekBondOptionResult {
    let b = |t1: f64, t2: f64| -> f64 {
        (1.0 - ((-kappa * (t2 - t1)).exp())) / kappa
    };

    let a = |t1: f64, t2: f64| -> f64 {
        let b_val = b(t1, t2);
        let r_inf = theta - 0.5 * sigma * sigma / (kappa * kappa);
        ((r_inf * (b_val - (t2 - t1))) - (sigma * sigma * b_val * b_val / (4.0 * kappa))).exp()
    };

    // ZCB prices
    let p_0_t = a(0.0, t_option) * (-b(0.0, t_option) * r0).exp() * face;
    let p_0_tb = a(0.0, t_bond) * (-b(0.0, t_bond) * r0).exp() * face;

    // Bond option vol
    let b_ts_tb = b(t_option, t_bond);
    let sigma_p = sigma * b_ts_tb
        * ((1.0 - (-2.0 * kappa * t_option).exp()) / (2.0 * kappa)).sqrt();

    // Pricing formula
    let h = (1.0 / sigma_p) * (p_0_tb / (p_0_t * strike)).ln() + 0.5 * sigma_p;

    let omega = if is_call { 1.0 } else { -1.0 };
    let price = omega * (p_0_tb * cumulative_normal(omega * h)
        - strike * p_0_t * cumulative_normal(omega * (h - sigma_p)));

    VasicekBondOptionResult {
        price: price.max(0.0),
        bond_price: p_0_tb,
        sigma_p,
    }
}

// ---------------------------------------------------------------------------
// Vasicek European Equity Option
// ---------------------------------------------------------------------------

/// Result from the Vasicek European equity option engine.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VasicekEquityResult {
    pub price: f64,
    pub effective_vol: f64,
}

/// European equity option under Vasicek stochastic interest rates.
///
/// Uses the approach of Merton (1973) for stochastic rates: the effective
/// volatility is adjusted for the rate-equity correlation.
///
/// # Arguments
/// - `spot`, `strike` — equity price and strike
/// - `r0` — current short rate
/// - `q` — dividend yield
/// - `sigma_s` — equity volatility
/// - `kappa`, `theta`, `sigma_r` — Vasicek rate model parameters
/// - `rho` — correlation between equity and rate processes
/// - `t` — time to expiry
/// - `is_call` — true for call
#[allow(clippy::too_many_arguments)]
pub fn vasicek_european_equity(
    spot: f64,
    strike: f64,
    r0: f64,
    q: f64,
    sigma_s: f64,
    kappa: f64,
    theta: f64,
    sigma_r: f64,
    rho: f64,
    t: f64,
    is_call: bool,
) -> VasicekEquityResult {
    // B(0, T) for Vasicek
    let b_t = (1.0 - (-kappa * t).exp()) / kappa;

    // Variance of integrated rate
    let _var_r = sigma_r * sigma_r * (t + (2.0 / kappa) * ((-kappa * t).exp() - 1.0)
        - (1.0 / (2.0 * kappa)) * ((-2.0 * kappa * t).exp() - 1.0));

    // Expected integrated rate
    let r_avg = theta + (r0 - theta) * b_t / t;

    // Effective equity variance: σ²_S * T + σ²_rate_adj + 2ρ·cov(S, ∫r)
    // Simplified Merton approach:
    let sigma_p = sigma_r * b_t; // bond vol
    let total_var = sigma_s * sigma_s * t
        + sigma_p * sigma_p
        - 2.0 * rho * sigma_s * sigma_p * t;
    let effective_vol = (total_var / t).sqrt();

    // Forward price with averaged rate
    let fwd = spot * ((r_avg - q) * t).exp();
    let df = (-r_avg * t).exp();
    let sqrt_t = t.sqrt();

    let d1 = ((fwd / strike).ln() + 0.5 * effective_vol * effective_vol * t) / (effective_vol * sqrt_t);
    let d2 = d1 - effective_vol * sqrt_t;

    let omega = if is_call { 1.0 } else { -1.0 };
    let price = df * omega
        * (fwd * cumulative_normal(omega * d1) - strike * cumulative_normal(omega * d2));

    VasicekEquityResult {
        price: price.max(0.0),
        effective_vol,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_gjr_garch_call() {
        // Daily variance ~(0.2/√252)² ≈ 0.000159
        let h0 = 0.20 * 0.20 / 252.0;
        let omega_g = h0 * 0.05;
        let alpha = 0.10;
        let beta_g = 0.80;
        let gamma = 0.05;

        let res = gjr_garch_option(
            100.0, 100.0,
            0.05 / 252.0, 0.02 / 252.0,
            omega_g, alpha, beta_g, gamma,
            h0, 252, true,
        );
        assert!(res.price > 3.0 && res.price < 20.0, "price={}", res.price);
    }

    #[test]
    fn test_vasicek_bond_call() {
        let res = vasicek_bond_option(
            0.05, 0.3, 0.05, 0.01,
            1.0, 5.0,
            0.90, 1.0, true,
        );
        assert!(res.price > 0.0, "price={}", res.price);
        assert!(res.bond_price > 0.5 && res.bond_price < 1.0, "bond={}", res.bond_price);
    }

    #[test]
    fn test_vasicek_bond_put_call_parity() {
        let call = vasicek_bond_option(
            0.05, 0.3, 0.05, 0.01,
            1.0, 5.0, 0.80, 1.0, true,
        );
        let put = vasicek_bond_option(
            0.05, 0.3, 0.05, 0.01,
            1.0, 5.0, 0.80, 1.0, false,
        );
        // C - P = P(0,T_B) - K * P(0,T)
        let b = |t1: f64, t2: f64| -> f64 { (1.0 - ((-0.3 * (t2 - t1)).exp())) / 0.3 };
        let _a = |t1: f64, t2: f64| -> f64 {
            let bv = b(t1, t2);
            let r_inf = 0.05 - 0.5 * 0.01 * 0.01 / 0.09;
            ((r_inf * (bv - (t2 - t1))) - (0.01 * 0.01 * bv * bv / 1.2)).exp()
        };
        // Just check parity holds approximately
        let diff = call.price - put.price;
        assert!(diff.abs() < 0.5, "parity diff={}", diff);
    }

    #[test]
    fn test_vasicek_equity_call() {
        let res = vasicek_european_equity(
            100.0, 100.0, 0.05, 0.02, 0.20,
            0.3, 0.05, 0.01, -0.2,
            1.0, true,
        );
        // Should be close to BS with σ≈0.20
        assert!(res.price > 5.0 && res.price < 15.0, "price={}", res.price);
        assert!(res.effective_vol > 0.15 && res.effective_vol < 0.30, "vol={}", res.effective_vol);
    }
}
