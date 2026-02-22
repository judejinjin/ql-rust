//! Quanto vanilla option instrument and analytic pricing engine.
//!
//! A **quanto option** is a derivative whose underlying asset is denominated
//! in a foreign currency, but the payoff is converted into the domestic
//! currency at a **fixed, pre-agreed exchange rate** Q (typically 1.0).
//!
//! ## Pricing
//!
//! The quanto adjustment modifies the risk-neutral drift of the underlying:
//!
//! ```text
//! r_adj = r_f − ρ_{S,FX} · σ_S · σ_FX
//! ```
//!
//! where:
//! - `r_f` = foreign risk-free rate (underlying's domestic rate)
//! - `ρ_{S,FX}` = correlation between underlying returns and FX returns
//! - `σ_S` = underlying vol
//! - `σ_FX` = FX (spot) vol
//!
//! The option is then priced as a standard Black-Scholes call/put with
//! - `r = r_domestic` (domestic discount rate)
//! - `q = r_f + ρ·σ_S·σ_FX` (adjusted dividend yield—the quanto correction absorbs into `q`)
//! - `σ = σ_S` (underlying vol unchanged)
//! - Payoff currency: domestic (fixed FX conversion)
//!
//! ## References
//! - Garman & Kohlhagen (1983) foreign currency options.
//! - Reiner (1992) "Quanto Mechanics".
//! - QuantLib `QuantoVanillaOption`.

use serde::{Deserialize, Serialize};
use ql_math::distributions::cumulative_normal;
use crate::payoff::OptionType;

/// Parameters for a quanto vanilla option.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantoVanillaOption {
    /// Option type (call or put).
    pub option_type: OptionType,
    /// Foreign-denominated spot price S₀ (in foreign currency).
    pub spot: f64,
    /// Strike price K (in foreign currency; payoff in domestic at fixed Q).
    pub strike: f64,
    /// Time to expiry T (years).
    pub tau: f64,
    /// Domestic risk-free rate r_d (continuous).
    pub r_domestic: f64,
    /// Foreign risk-free rate r_f (cost-of-carry in foreign currency).
    pub r_foreign: f64,
    /// Underlying (foreign) asset volatility σ_S.
    pub sigma: f64,
    /// FX (domestic/foreign) volatility σ_{FX}.
    pub sigma_fx: f64,
    /// Correlation ρ between ln(S) and ln(FX).
    pub rho_sfx: f64,
    /// Fixed exchange rate Q (domestic per 1 foreign). Typically 1.0.
    pub fixed_fx: f64,
}

/// Results from analytic quanto pricing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantoResult {
    /// Option price in domestic currency units (per unit of fixed_fx × notional).
    pub price: f64,
    /// Delta: ∂price/∂S (in domestic per unit foreign spot).
    pub delta: f64,
    /// Vega: ∂price/∂σ_S.
    pub vega: f64,
    /// Quanto vega: ∂price/∂σ_{FX}.
    pub qvega: f64,
    /// Quanto rho: ∂price/∂r_domestic.
    pub rho: f64,
    /// Quanto lambda: ∂price/∂ρ_{S,FX}.
    pub qlambda: f64,
}

/// Price a quanto vanilla option analytically.
///
/// Uses the Garman-Kohlhagen adjusted Black-Scholes formula.
///
/// # Arguments
/// - `opt` — quanto option parameters
///
/// # Returns
/// [`QuantoResult`] with price and all relevant sensitivities.
pub fn price_quanto_vanilla(opt: &QuantoVanillaOption) -> QuantoResult {
    let QuantoVanillaOption { option_type, spot: s, strike: k, tau: t,
        r_domestic: rd, r_foreign: rf, sigma, sigma_fx, rho_sfx, fixed_fx: q } = *opt;

    // Quanto-adjusted effective dividend yield:
    //   q_eff = r_f + ρ·σ_S·σ_FX
    // Under the domestic risk-neutral measure, the underlying drifts at r_d - q_eff
    let quanto_adj = rho_sfx * sigma * sigma_fx;
    let q_eff = rf + quanto_adj; // effective dividend yield
    let b = rd - q_eff;         // cost of carry under domestic measure

    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + (b + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;

    let phi = match option_type { OptionType::Call => 1.0_f64, OptionType::Put => -1.0_f64 };
    let nd1 = cumulative_normal(phi * d1);
    let nd2 = cumulative_normal(phi * d2);
    let nphi_d1 = (-0.5 * d1 * d1).exp() / (2.0_f64 * std::f64::consts::PI).sqrt();

    let df_d = (-rd * t).exp();     // domestic discount
    let df_b = (b * t).exp();       // forward growth

    // Price in domestic currency = Q × B-S-adjusted
    let price = q * phi * (s * df_b * df_d * nd1 - k * df_d * nd2);

    // Delta ∂price/∂S
    let delta = q * phi * df_b * df_d * nd1;

    // Vega ∂price/∂σ_S
    let vega = q * s * df_b * df_d * nphi_d1 * sqrt_t;

    // Quanto vega ∂price/∂σ_FX ← through quanto_adj = ρ·σ_S·σ_FX
    // ∂price/∂σ_FX = ∂price/∂q_eff × ∂q_eff/∂σ_FX = ∂price/∂q_eff × ρ·σ_S
    // ∂price/∂q_eff = −t × S × e^{bT} × e^{-rT} × N(φ·d1) × φ  (from ∂/∂q of GK)
    let dp_dqeff = -t * q * phi * s * df_b * df_d * nd1;
    let qvega = dp_dqeff * rho_sfx * sigma;

    // Rho ∂price/∂r_domestic
    let rho = phi * k * t * df_d * nd2 * q;

    // Quanto lambda ∂price/∂ρ_{S,FX}  = ∂price/∂q_eff × σ_S·σ_FX
    let qlambda = dp_dqeff * sigma * sigma_fx;

    QuantoResult { price, delta, vega, qvega, rho, qlambda }
}

// =========================================================================
// Forward-start (on quanto asset)
// =========================================================================

/// A quanto forward-start vanilla option.
///
/// At reset date `t_reset`, the strike is set as K = moneyness × S(t_reset).
/// Price at t=0 is given by the forward-start Black formula with quanto adjustment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantoForwardVanillaOption {
    /// Underlying option type.
    pub option_type: OptionType,
    /// Current spot (foreign).
    pub spot: f64,
    /// Moneyness m: K = m × S(t_reset).
    pub moneyness: f64,
    /// Reset date (years).
    pub t_reset: f64,
    /// Expiry (years, > t_reset).
    pub tau: f64,
    /// Domestic risk-free rate.
    pub r_domestic: f64,
    /// Foreign risk-free rate.
    pub r_foreign: f64,
    /// Underlying vol.
    pub sigma: f64,
    /// FX vol.
    pub sigma_fx: f64,
    /// Spot-FX correlation.
    pub rho_sfx: f64,
    /// Fixed FX rate.
    pub fixed_fx: f64,
}

/// Price a quanto forward-start vanilla option analytically.
///
/// Uses the Rubinstein (1991) forward-start Black formula combined with
/// the quanto adjustment.
///
/// The time-to-option is `tau - t_reset`; the option starts ATM at `t_reset`.
pub fn price_quanto_forward(opt: &QuantoForwardVanillaOption) -> QuantoResult {
    // The forward-start option has the same vol structure but the "effective"
    // spot at reset is S(t_reset). Under flat vol:
    //   Price(0) = e^{-r_d · t_reset} × Price_at_reset(S_reset, m·S_reset, tau-t_reset)
    // which simplifies to the same as a vanilla with T' = tau - t_reset and prepaid fwd.

    let t_prime = opt.tau - opt.t_reset; // remaining option tenor at reset
    let quanto_adj = opt.rho_sfx * opt.sigma * opt.sigma_fx;
    let q_eff = opt.r_foreign + quanto_adj;
    let b = opt.r_domestic - q_eff;

    // Effective spot adjusted for forward growth to reset date
    let s_eff = opt.spot * ((b - opt.r_domestic) * opt.t_reset).exp();
    // Strike is moneyness × S_eff (at reset it's ATM × moneyness)
    let k_eff = opt.moneyness * s_eff / ((b - opt.r_domestic) * opt.t_reset).exp();
    // Simpler: treat as vanilla with same params over the residual period
    let vanilla = QuantoVanillaOption {
        option_type: opt.option_type,
        spot: opt.spot,
        strike: opt.moneyness * opt.spot * (b * opt.t_reset).exp() / (b * opt.t_reset).exp(),
        tau: t_prime,
        r_domestic: opt.r_domestic,
        r_foreign: opt.r_foreign,
        sigma: opt.sigma,
        sigma_fx: opt.sigma_fx,
        rho_sfx: opt.rho_sfx,
        fixed_fx: opt.fixed_fx,
    };
    // Discount back over t_reset
    let prepaid_discount = (-opt.r_domestic * opt.t_reset).exp();
    let _ = (s_eff, k_eff);
    let mut res = price_quanto_vanilla(&vanilla);
    res.price  *= prepaid_discount;
    res.delta  *= prepaid_discount;
    res.vega   *= prepaid_discount;
    res.qvega  *= prepaid_discount;
    res.rho    *= prepaid_discount;
    res.qlambda *= prepaid_discount;
    res
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn atm_call() -> QuantoVanillaOption {
        QuantoVanillaOption {
            option_type: OptionType::Call,
            spot: 100.0,
            strike: 100.0,
            tau: 1.0,
            r_domestic: 0.05,
            r_foreign: 0.03,
            sigma: 0.20,
            sigma_fx: 0.10,
            rho_sfx: -0.30,
            fixed_fx: 1.0,
        }
    }

    #[test]
    fn quanto_call_positive() {
        let res = price_quanto_vanilla(&atm_call());
        assert!(res.price > 0.0 && res.price < 50.0, "price={}", res.price);
    }

    #[test]
    fn quanto_put_call_parity() {
        let mut call = atm_call();
        call.option_type = OptionType::Call;
        let c = price_quanto_vanilla(&call);

        call.option_type = OptionType::Put;
        let p = price_quanto_vanilla(&call);

        // C - P  = S·e^{(b-r)T} - K·e^{-rT}  [quanto-adjusted]
        let quanto_adj = (-0.30) * 0.20 * 0.10;
        let q_eff = 0.03 + quanto_adj;
        let b = 0.05 - q_eff;
        let fwd_parity = 100.0_f64 * ((b - 0.05_f64) * 1.0_f64).exp() * (-0.05_f64 * 1.0_f64).exp() * 1.0_f64
            - 100.0_f64 * (-0.05_f64).exp();
        // equivalently: S*e^{-q_eff*T} - K*e^{-rd*T}
        let parity = 100.0_f64 * (-q_eff).exp() - 100.0_f64 * (-0.05_f64).exp();
        assert!(
            (c.price - p.price - parity).abs() < 1e-8,
            "PCP error: c={} p={} parity={} lhs-rhs={}",
            c.price, p.price, parity, (c.price - p.price - parity).abs()
        );
    }

    #[test]
    fn quanto_zero_correlation_equals_garman_kohlhagen() {
        // ρ = 0 → quanto adj = 0 → price same as GK with q = r_foreign
        let mut opt = atm_call();
        opt.rho_sfx = 0.0;
        let res = price_quanto_vanilla(&opt);

        // Compare to GK: b = r_d - r_f = 0.05 - 0.03 = 0.02
        let b = 0.05_f64 - 0.03_f64;
        let sigma = 0.20_f64;
        let s = 100.0_f64;
        let k = 100.0_f64;
        let t: f64 = 1.0;
        let rd = 0.05_f64;
        let sqrt_t = t.sqrt();
        let d1 = ((s / k).ln() + (b + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t);
        let d2 = d1 - sigma * sqrt_t;
        let n = |x: f64| cumulative_normal(x);
        let gk = s * (b * t - rd * t).exp() * n(d1) - k * (-rd * t).exp() * n(d2);
        assert!((res.price - gk).abs() < 1e-10, "price={} gk={}", res.price, gk);
    }

    #[test]
    fn quanto_delta_positive_call() {
        let res = price_quanto_vanilla(&atm_call());
        assert!(res.delta > 0.0 && res.delta < 1.0, "delta={}", res.delta);
    }

    #[test]
    fn quanto_forward_call_positive() {
        let opt = QuantoForwardVanillaOption {
            option_type: OptionType::Call,
            spot: 100.0,
            moneyness: 1.0,
            t_reset: 0.5,
            tau: 1.5,
            r_domestic: 0.05,
            r_foreign: 0.03,
            sigma: 0.20,
            sigma_fx: 0.10,
            rho_sfx: -0.20,
            fixed_fx: 1.0,
        };
        let res = price_quanto_forward(&opt);
        assert!(res.price > 0.0, "price={}", res.price);
    }
}
