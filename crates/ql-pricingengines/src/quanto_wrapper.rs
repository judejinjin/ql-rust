//! Generic Quanto adjustment wrapper and Quanto term structure.
//!
//! A quanto option is an option denominated in a foreign currency but
//! settled in a domestic currency at a fixed FX rate. The quanto
//! adjustment modifies the drift of the underlying asset.
//!
//! - [`quanto_adjustment`] — Adjust BS parameters for quanto pricing.
//! - [`QuantoTermStructure`] — A yield term structure with quanto adjustment.

/// Parameters for a quanto adjustment.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct QuantoAdjustment {
    /// Adjusted (domestic) risk-free rate.
    pub r_domestic: f64,
    /// Adjusted dividend yield (includes quanto correction).
    pub q_adjusted: f64,
    /// Quanto-corrected forward price.
    pub forward_adjusted: f64,
}

/// Compute the quanto adjustment for a foreign-denominated option
/// settled in domestic currency.
///
/// The adjustment changes the cost-of-carry from (r_f - q) to
/// (r_f - q - ρ·σ_S·σ_FX), where:
/// - `r_foreign` — foreign risk-free rate
/// - `r_domestic` — domestic risk-free rate
/// - `q` — dividend yield of the underlying
/// - `sigma_s` — volatility of the underlying
/// - `sigma_fx` — volatility of the FX rate
/// - `rho_s_fx` — correlation between underlying and FX rate
/// - `spot` — current underlying price (in foreign currency)
/// - `t` — time to expiry
pub fn quanto_adjustment(
    spot: f64,
    r_foreign: f64,
    r_domestic: f64,
    q: f64,
    sigma_s: f64,
    sigma_fx: f64,
    rho_s_fx: f64,
    t: f64,
) -> QuantoAdjustment {
    // The quanto correction subtracts ρ·σ_S·σ_FX from the drift
    let quanto_correction = rho_s_fx * sigma_s * sigma_fx;
    let q_adjusted = q + quanto_correction;
    let forward_adjusted = spot * ((r_foreign - q_adjusted) * t).exp();

    QuantoAdjustment {
        r_domestic,
        q_adjusted,
        forward_adjusted,
    }
}

/// Quanto term structure.
///
/// Wraps a foreign yield curve with a quanto adjustment, producing
/// domestic-equivalent discount factors.
#[derive(Clone, Debug)]
pub struct QuantoTermStructure {
    /// Foreign risk-free rate (flat).
    pub r_foreign: f64,
    /// Domestic risk-free rate (flat).
    pub r_domestic: f64,
    /// Underlying volatility.
    pub sigma_s: f64,
    /// FX volatility.
    pub sigma_fx: f64,
    /// Correlation between asset and FX.
    pub rho_s_fx: f64,
}

impl QuantoTermStructure {
    /// Create a new QuantoTermStructure.
    pub fn new(r_foreign: f64, r_domestic: f64, sigma_s: f64, sigma_fx: f64, rho_s_fx: f64) -> Self {
        Self { r_foreign, r_domestic, sigma_s, sigma_fx, rho_s_fx }
    }

    /// Discount factor at time t (domestic measure).
    pub fn discount(&self, t: f64) -> f64 {
        (-self.r_domestic * t).exp()
    }

    /// Forward rate from t₁ to t₂.
    pub fn forward_rate(&self, t1: f64, t2: f64) -> f64 {
        if (t2 - t1).abs() < 1e-14 { return self.r_domestic; }
        (self.discount(t1) / self.discount(t2)).ln() / (t2 - t1)
    }

    /// Adjusted dividend yield for quanto pricing.
    pub fn adjusted_dividend_yield(&self, q: f64) -> f64 {
        q + self.rho_s_fx * self.sigma_s * self.sigma_fx
    }

    /// Adjusted forward price.
    pub fn adjusted_forward(&self, spot: f64, q: f64, t: f64) -> f64 {
        let q_adj = self.adjusted_dividend_yield(q);
        spot * ((self.r_foreign - q_adj) * t).exp()
    }
}

/// Price a quanto European option (generic wrapper).
///
/// Takes any single-asset European pricer and applies the quanto adjustment.
/// Returns the price in domestic currency.
///
/// # Arguments
/// - `spot` — asset price in foreign currency
/// - `strike` — option strike in foreign currency
/// - `r_foreign`, `r_domestic` — interest rates
/// - `q` — foreign dividend yield
/// - `sigma_s` — asset volatility
/// - `sigma_fx` — FX volatility
/// - `rho_s_fx` — asset/FX correlation
/// - `t` — time to expiry
/// - `is_call` — true for call
/// - `fx_rate` — fixed FX rate for settlement
#[allow(clippy::too_many_arguments)]
pub fn quanto_vanilla(
    spot: f64,
    strike: f64,
    r_foreign: f64,
    r_domestic: f64,
    q: f64,
    sigma_s: f64,
    sigma_fx: f64,
    rho_s_fx: f64,
    t: f64,
    is_call: bool,
    fx_rate: f64,
) -> QuantoVanillaResult {
    // Delegate pricing to the generic implementation (AD-ready).
    let price = crate::generic::quanto_vanilla_generic::<f64>(
        spot, strike, r_foreign, r_domestic, q, sigma_s, sigma_fx, rho_s_fx, t, is_call, fx_rate,
    );

    // Compute the extra fields from the quanto adjustment.
    let adj = quanto_adjustment(spot, r_foreign, r_domestic, q, sigma_s, sigma_fx, rho_s_fx, t);

    QuantoVanillaResult {
        price,
        forward_adjusted: adj.forward_adjusted,
        q_adjusted: adj.q_adjusted,
    }
}

/// Result from the quanto vanilla engine.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct QuantoVanillaResult {
    /// Option price in domestic currency.
    pub price: f64,
    /// Quanto-adjusted forward.
    pub forward_adjusted: f64,
    /// Quanto-adjusted dividend yield.
    pub q_adjusted: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_quanto_adjustment() {
        let adj = quanto_adjustment(100.0, 0.02, 0.05, 0.01, 0.20, 0.10, -0.3, 1.0);
        // q_adjusted = 0.01 + (-0.3)*0.20*0.10 = 0.01 - 0.006 = 0.004
        assert_abs_diff_eq!(adj.q_adjusted, 0.004, epsilon = 1e-10);
    }

    #[test]
    fn test_quanto_term_structure() {
        let qts = QuantoTermStructure::new(0.02, 0.05, 0.20, 0.10, -0.3);
        let df = qts.discount(1.0);
        assert_abs_diff_eq!(df, (-0.05_f64).exp(), epsilon = 1e-10);
        let q_adj = qts.adjusted_dividend_yield(0.01);
        assert_abs_diff_eq!(q_adj, 0.004, epsilon = 1e-10);
    }

    #[test]
    fn test_quanto_vanilla_call() {
        let res = quanto_vanilla(
            100.0, 100.0, 0.02, 0.05, 0.01,
            0.20, 0.10, -0.3, 1.0, true, 1.0,
        );
        assert!(res.price > 0.0, "price={}", res.price);
    }

    #[test]
    fn test_quanto_vanilla_put_call_parity() {
        let call = quanto_vanilla(
            100.0, 100.0, 0.02, 0.05, 0.01,
            0.20, 0.10, -0.3, 1.0, true, 1.0,
        );
        let put = quanto_vanilla(
            100.0, 100.0, 0.02, 0.05, 0.01,
            0.20, 0.10, -0.3, 1.0, false, 1.0,
        );
        let fwd = call.forward_adjusted;
        let df = (-0.05_f64).exp();
        // C - P = df * (F - K) * fx_rate
        let parity = call.price - put.price - df * (fwd - 100.0);
        assert_abs_diff_eq!(parity, 0.0, epsilon = 0.5);
    }
}
