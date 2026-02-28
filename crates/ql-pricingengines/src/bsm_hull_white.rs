//! Analytic BSM Hull-White engine.
//!
//! Prices equity options with stochastic interest rates under the
//! combined Black-Scholes-Merton + Hull-White model.
//!
//! The equity process follows GBM and the short rate follows the
//! Hull-White one-factor model. The correlation between the two
//! Brownian motions is ρ.
//!
//! This corresponds to QuantLib's `AnalyticBSMHullWhiteEngine`.
//!
//! Reference: Brigo & Mercurio, "Interest Rate Models — Theory and Practice",
//!            Chapter 24.

use serde::{Deserialize, Serialize};

/// Parameters for the BSM Hull-White hybrid model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BsmHullWhiteParams {
    /// Spot price.
    pub spot: f64,
    /// Strike price.
    pub strike: f64,
    /// Time to maturity (years).
    pub maturity: f64,
    /// Equity volatility.
    pub equity_vol: f64,
    /// Hull-White mean-reversion speed (a).
    pub hw_mean_reversion: f64,
    /// Hull-White short-rate volatility (σ_r).
    pub hw_vol: f64,
    /// Correlation between equity and short rate.
    pub rho: f64,
    /// Risk-free rate (continuous, flat).
    pub rate: f64,
    /// Dividend yield (continuous).
    pub div_yield: f64,
    /// True for call, false for put.
    pub is_call: bool,
}

/// Result from the BSM Hull-White engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BsmHullWhiteResult {
    /// Option price.
    pub npv: f64,
    /// Effective total volatility.
    pub effective_vol: f64,
    /// Delta (∂V/∂S).
    pub delta: f64,
}

/// Price a European option under BSM + Hull-White stochastic rates.
///
/// Uses the Brigo-Mercurio formula: the corrected total variance is
///   Σ² = σ²_S T + σ²_r B(T)² T − 2ρ σ_S σ_r B(T) T
/// where B(τ) = (1 − e^{−aτ})/a and the integrated variance from HW is
///   ∫₀ᵀ σ²_r B(T−t)² dt.
pub fn price_bsm_hull_white(params: &BsmHullWhiteParams) -> BsmHullWhiteResult {
    let t = params.maturity;
    let a = params.hw_mean_reversion;
    let sigma_r = params.hw_vol;
    let sigma_s = params.equity_vol;
    let rho = params.rho;

    // Hull-White B function
    let b_hw = if a.abs() < 1e-10 { t } else { (1.0 - (-a * t).exp()) / a };

    // Integrated HW variance: ∫₀ᵀ σ²_r B(T−t)² dt
    let hw_integrated_var = if a.abs() < 1e-10 {
        sigma_r * sigma_r * t * t * t / 3.0
    } else {
        sigma_r * sigma_r / (a * a) * (
            t - 2.0 / a * (1.0 - (-a * t).exp()) + 1.0 / (2.0 * a) * (1.0 - (-2.0 * a * t).exp())
        )
    };

    // Cross-term: ∫₀ᵀ σ_S σ_r B(T−t) dt
    let cross_term = if a.abs() < 1e-10 {
        sigma_s * sigma_r * t * t / 2.0
    } else {
        sigma_s * sigma_r / a * (t - b_hw)
    };

    // Total effective variance
    let total_var = sigma_s * sigma_s * t + hw_integrated_var - 2.0 * rho * cross_term;
    let total_var = total_var.max(1e-15);
    let effective_vol = (total_var / t).sqrt();

    // Forward price
    let df = (-params.rate * t).exp();
    let fwd = params.spot * ((params.rate - params.div_yield) * t).exp();

    // Black-76 formula with effective vol
    let sqrt_t = t.sqrt();
    let total_std = effective_vol * sqrt_t;
    let d1 = (fwd / params.strike).ln() / total_std + 0.5 * total_std;
    let d2 = d1 - total_std;

    let (npv, delta) = if params.is_call {
        let price = df * (fwd * norm_cdf(d1) - params.strike * norm_cdf(d2));
        let delta_val = (-params.div_yield * t).exp() * norm_cdf(d1);
        (price, delta_val)
    } else {
        let price = df * (params.strike * norm_cdf(-d2) - fwd * norm_cdf(-d1));
        let delta_val = -(-params.div_yield * t).exp() * norm_cdf(-d1);
        (price, delta_val)
    };

    BsmHullWhiteResult {
        npv: npv.max(0.0),
        effective_vol,
        delta,
    }
}

/// Rational approximation of the standard normal CDF (Abramowitz & Stegun 7.1.26 + erf→Φ).
fn norm_cdf(x: f64) -> f64 {
    if x >= 8.0 { return 1.0; }
    if x <= -8.0 { return 0.0; }

    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let z = x.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + p * z);
    let erf_approx = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-z * z).exp();
    0.5 * (1.0 + sign * erf_approx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bsm_hw_reduces_to_bs() {
        // With zero HW vol, should reduce to Black-Scholes
        let params = BsmHullWhiteParams {
            spot: 100.0,
            strike: 100.0,
            maturity: 1.0,
            equity_vol: 0.20,
            hw_mean_reversion: 0.05,
            hw_vol: 0.0,   // no rate vol
            rho: 0.0,
            rate: 0.05,
            div_yield: 0.0,
            is_call: true,
        };
        let res = price_bsm_hull_white(&params);
        // BS price for ATM call ≈ 10.45
        assert!((res.npv - 10.45).abs() < 0.5, "npv={}", res.npv);
        assert!((res.effective_vol - 0.20).abs() < 0.01, "evol={}", res.effective_vol);
    }

    #[test]
    fn test_bsm_hw_stochastic_rates_increase_vol() {
        let base = BsmHullWhiteParams {
            spot: 100.0,
            strike: 100.0,
            maturity: 5.0,
            equity_vol: 0.20,
            hw_mean_reversion: 0.05,
            hw_vol: 0.0,
            rho: 0.0,
            rate: 0.03,
            div_yield: 0.01,
            is_call: true,
        };
        let base_res = price_bsm_hull_white(&base);

        let stoch = BsmHullWhiteParams {
            hw_vol: 0.01,
            ..base.clone()
        };
        let stoch_res = price_bsm_hull_white(&stoch);

        // Stochastic rates increase effective vol with zero correlation
        assert!(stoch_res.effective_vol > base_res.effective_vol,
            "base_evol={}, stoch_evol={}", base_res.effective_vol, stoch_res.effective_vol);
    }

    #[test]
    fn test_bsm_hw_negative_correlation_reduces_vol() {
        // Positive ρ (equity rises with rates) → cross-term reduces total variance
        // Negative ρ → cross-term increases total variance
        let pos = BsmHullWhiteParams {
            spot: 100.0, strike: 100.0, maturity: 2.0,
            equity_vol: 0.20, hw_mean_reversion: 0.05,
            hw_vol: 0.01, rho: 0.3,
            rate: 0.05, div_yield: 0.0, is_call: true,
        };
        let neg = BsmHullWhiteParams { rho: -0.3, ..pos.clone() };

        let pos_res = price_bsm_hull_white(&pos);
        let neg_res = price_bsm_hull_white(&neg);

        // Positive correlation reduces effective vol (cross term is subtracted)
        assert!(pos_res.effective_vol < neg_res.effective_vol,
            "pos_evol={}, neg_evol={}", pos_res.effective_vol, neg_res.effective_vol);
    }

    #[test]
    fn test_bsm_hw_put_call_parity() {
        let call = BsmHullWhiteParams {
            spot: 100.0, strike: 95.0, maturity: 1.0,
            equity_vol: 0.25, hw_mean_reversion: 0.1,
            hw_vol: 0.008, rho: 0.2,
            rate: 0.05, div_yield: 0.02, is_call: true,
        };
        let put = BsmHullWhiteParams { is_call: false, ..call.clone() };

        let cr = price_bsm_hull_white(&call);
        let pr = price_bsm_hull_white(&put);

        let fwd = call.spot * ((call.rate - call.div_yield) * call.maturity).exp();
        let df = (-call.rate * call.maturity).exp();
        let parity = cr.npv - pr.npv - df * (fwd - call.strike);

        assert!(parity.abs() < 0.01, "put-call parity error={}", parity);
    }
}
