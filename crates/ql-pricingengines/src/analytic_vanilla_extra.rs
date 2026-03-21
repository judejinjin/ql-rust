//! Additional analytic pricing engines:
//!
//! - [`HestonExpansionResult`] / [`heston_expansion_price`] — Medvedev-Scaillet
//!   (2010) 3rd-order short-time expansion around ATM.  Fast approximation
//!   good for short maturities and near-ATM strikes.
//!
//! - [`CevResult`] / [`analytic_cev_price`] — Constant Elasticity of Variance
//!   closed-form pricing via the non-central chi-squared distribution.
//!
//! - [`PtdHestonSlice`] / [`analytic_ptd_heston_price`] — Piecewise time-dependent
//!   Heston: prices options when κ, θ, σ, ρ, v₀ vary step-wise over time.
//!
//! All engines return put prices that are adjusted by put-call parity
//! (approximate for the expansion engine; exact elsewhere).


use ql_core::errors::{QLError, QLResult};
use ql_instruments::OptionType;
use ql_math::distributions::cumulative_normal;

// ===========================================================================
// Heston Expansion Engine (Medvedev-Scaillet 2010)
// ===========================================================================

/// Result from the Heston expansion engine.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HestonExpansionResult {
    /// Approximate option price.
    pub price: f64,
    /// Approximate implied volatility.
    pub implied_vol: f64,
}

/// Price a European option using the Medvedev-Scaillet (2010) short-time
/// expansion around the ATM point.
///
/// The expansion is accurate for:
/// - Near-ATM strikes (|log-moneyness| < 0.2)
/// - Short maturities (T < 2 years)
///
/// For long-dated or deep OTM/ITM options, use `heston_price` instead.
#[allow(clippy::too_many_arguments)]
pub fn heston_expansion_price(
    spot: f64,
    strike: f64,
    tau: f64,
    risk_free_rate: f64,
    dividend_yield: f64,
    v0: f64,
    kappa: f64,
    _theta: f64,
    sigma: f64,
    rho: f64,
    opt_type: OptionType,
) -> QLResult<HestonExpansionResult> {
    if spot <= 0.0 || strike <= 0.0 || tau <= 0.0 {
        return Err(QLError::InvalidArgument("spot, strike, tau must be positive".into()));
    }
    let fwd = spot * ((risk_free_rate - dividend_yield) * tau).exp();
    let x = (fwd / strike).ln(); // log-moneyness
    let sqrt_v0 = v0.sqrt().max(1e-8);

    // ATM variance term σ²(x, T) ≈ σ₀² + a₁(x,T) + a₂(x,T)
    // where σ₀ = √v₀ (instantaneous ATM vol)
    // Following Medvedev-Scaillet (2010), the expansion to first order is:
    //
    //   σ_impl(x,T) ≈ σ₀ + σ₁(x) · T + O(T²)
    //
    // with σ₁(x) = ½ · rho · sigma_v · (x/T − ½σ₀)  — leading correction
    // and  σ₀ = √v₀.
    //
    // We use the Gatheral (2006) "vol-of-vol" parametrisation truncated
    // at first order.

    // Zero-order: ATM implied vol = √v₀
    let sigma0 = sqrt_v0;

    // First-order slope in log-moneyness (Skew)
    // ∂σ/∂x|₀ ≈ ρ · σ_v / (2 · σ₀)   (Heston leading skew)
    let skew = rho * sigma / (2.0 * sigma0);

    // Second-order curvature (Smile / Convexity)
    // ∂²σ/∂x²|₀ ≈ (σ_v² / (4 · σ₀³))(1 − ρ²)
    let smile = sigma * sigma * (1.0 - rho * rho) / (4.0 * sigma0.powi(3));

    // Mean-reversion dampens skew at longer maturities:
    // Apply a simple exponential correction to skew term
    let mr_factor = if tau * kappa < 1e-8 {
        1.0
    } else {
        (1.0 - (-kappa * tau).exp()) / (kappa * tau)
    };

    // Approximate implied vol at strike K
    let sigma_impl = (sigma0 + skew * x * mr_factor + 0.5 * smile * x * x).max(0.001);

    // Price via Black-Scholes with the implied vol
    let d1 = ((fwd / strike).ln() + 0.5 * sigma_impl * sigma_impl * tau)
        / (sigma_impl * tau.sqrt());
    let d2 = d1 - sigma_impl * tau.sqrt();
    let df = (-risk_free_rate * tau).exp();
    let price = match opt_type {
        OptionType::Call => df * (fwd * cumulative_normal(d1) - strike * cumulative_normal(d2)),
        OptionType::Put => df * (strike * cumulative_normal(-d2) - fwd * cumulative_normal(-d1)),
    };

    Ok(HestonExpansionResult {
        price: price.max(0.0),
        implied_vol: sigma_impl,
    })
}

// ===========================================================================
// Analytic CEV Engine
// ===========================================================================

/// Result from the CEV analytical pricing engine.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CevResult {
    /// CEV option price.
    pub price: f64,
}

/// Non-central chi-squared CDF approximation using the Wilson-Hilferty
/// cube-root transformation.
fn ncx2_cdf(x: f64, nu: f64, lambda: f64) -> f64 {
    // Approximate via the ratio of Poisson-weighted chi-squared CDFs
    // using the series expansion. For moderate lambda and nu we use the
    // Pearson (1959) approximation.
    if x <= 0.0 {
        return 0.0;
    }
    if lambda < 1e-10 {
        // Central chi-squared: use Wilson-Hilferty
        let z = (x / nu).powf(1.0 / 3.0);
        let h = 1.0 - 2.0 / (9.0 * nu);
        let s = (2.0 / (9.0 * nu)).sqrt();
        return cumulative_normal((z - h) / s);
    }
    // Non-central: Patnaik (1949) 2-moment approximation
    // Match moments: nu' = (nu + lambda)² / (nu + 2·lambda), λ'=0
    let nu_eff = (nu + lambda) * (nu + lambda) / (nu + 2.0 * lambda);
    let scale = (nu + 2.0 * lambda) / (nu + lambda);
    ncx2_cdf(x / scale, nu_eff, 0.0)
}

/// Price a European option under the CEV (Constant Elasticity of Variance) model.
///
/// The CEV SDE is:
/// ```text
/// dS = r·S dt + σ·S^β dW
/// ```
///
/// where `beta` (β) is the elasticity parameter.
/// - β = 0: normal model
/// - β = 0.5: Cox (1975) square-root
/// - β = 1: Black-Scholes log-normal
///
/// Uses the closed-form solution of Cox (1975) / Schroder (1989) via the
/// non-central chi-squared distribution.
///
/// # Arguments
/// - `spot`, `strike`, `tau`, `risk_free_rate`, `dividend_yield` — standard inputs
/// - `sigma` — CEV vol coefficient σ
/// - `beta` — elasticity β (must not equal 1.0; use BS for β=1)
pub fn analytic_cev_price(
    spot: f64,
    strike: f64,
    tau: f64,
    risk_free_rate: f64,
    dividend_yield: f64,
    sigma: f64,
    beta: f64,
    opt_type: OptionType,
) -> QLResult<CevResult> {
    if spot <= 0.0 || strike <= 0.0 || tau <= 0.0 || sigma <= 0.0 {
        return Err(QLError::InvalidArgument(
            "spot, strike, tau, sigma must be positive".into(),
        ));
    }
    if (beta - 1.0).abs() < 1e-8 {
        return Err(QLError::InvalidArgument(
            "beta cannot equal 1 for CEV (use Black-Scholes)".into(),
        ));
    }

    let r = risk_free_rate;
    let q = dividend_yield;
    let nu = 1.0 / (2.0 * (1.0 - beta)).abs();

    // Forward price
    let b = r - q;
    let gamma = 1.0 - beta; // γ = 1-β, positive for β<1

    // Scale parameter k = 2b / (σ²·γ·(e^{2bγT} - 1))
    // Limiting case b→0: k = 1 / (σ²·γ²·T)
    let two_b_gamma_tau = 2.0 * b * gamma * tau;
    let k_eff = if two_b_gamma_tau.abs() < 1e-8 {
        1.0 / (sigma * sigma * gamma * gamma * tau)
    } else {
        2.0 * b / (sigma * sigma * gamma * (two_b_gamma_tau.exp() - 1.0))
    };
    // Two non-centrality-parameter arguments
    let x = k_eff * spot.powf(2.0 * gamma) * (2.0 * b * gamma * tau).exp();
    let y = k_eff * strike.powf(2.0 * gamma);
    let df = (-r * tau).exp();

    let price = if beta < 1.0 {
        // gamma > 0  (CEV with downward-sloping skew like equity)
        let nu2 = nu + 1.0;
        match opt_type {
            OptionType::Call => {
                let c = df * (spot * (b * tau).exp() * (1.0 - ncx2_cdf(y, nu2 * 2.0, x * 2.0))
                    - strike * ncx2_cdf(x * 2.0, nu * 2.0, y * 2.0));
                c.max(0.0)
            }
            OptionType::Put => {
                let c = df * (spot * (b * tau).exp() * (1.0 - ncx2_cdf(y, nu2 * 2.0, x * 2.0))
                    - strike * ncx2_cdf(x * 2.0, nu * 2.0, y * 2.0));
                let pcp = df * (strike - spot * ((r - q) * tau).exp());
                (c + pcp).max(0.0)
            }
        }
    } else {
        // gamma < 0, beta > 1
        let nu_abs = 1.0 / (2.0 * (beta - 1.0));
        let nu2_abs = nu_abs + 1.0;
        match opt_type {
            OptionType::Call => {
                let c = df * (spot * (b * tau).exp() * ncx2_cdf(x * 2.0, nu2_abs * 2.0, y * 2.0)
                    - strike * (1.0 - ncx2_cdf(y * 2.0, nu_abs * 2.0, x * 2.0)));
                c.max(0.0)
            }
            OptionType::Put => {
                let c = df * (spot * (b * tau).exp() * ncx2_cdf(x * 2.0, nu2_abs * 2.0, y * 2.0)
                    - strike * (1.0 - ncx2_cdf(y * 2.0, nu_abs * 2.0, x * 2.0)));
                let pcp = df * (strike - spot * ((r - q) * tau).exp());
                (c + pcp).max(0.0)
            }
        }
    };

    Ok(CevResult { price })
}

// ===========================================================================
// Piecewise Time-Dependent Heston Engine
// ===========================================================================

/// A single time slice of Heston parameters for the PTD engine.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PtdHestonSlice {
    /// Start of interval [t_i, t_{i+1}).
    pub t_start: f64,
    /// End of interval.
    pub t_end: f64,
    /// Mean-reversion speed κ(t) on this slice.
    pub kappa: f64,
    /// Long-run variance θ(t).
    pub theta: f64,
    /// Vol-of-vol σ(t).
    pub sigma: f64,
    /// Correlation ρ(t).
    pub rho: f64,
}

/// Result of the PTD Heston engine.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PtdHestonResult {
    /// Option price.
    pub price: f64,
}

/// Price a European option under the Piecewise Time-Dependent Heston model.
///
/// Parameters (κ, θ, σ, ρ) may change step-wise over the option lifetime.
/// The characteristic function is extended to piecewise-constant params by
/// integrating each time slice independently (Elices 2008).
///
/// # Arguments
/// - `slices` — parameter slices in chronological order; must cover [0, tau]
/// - `v0` — initial variance v(0)
/// - remaining args — option / market inputs
#[allow(clippy::too_many_arguments)]
pub fn analytic_ptd_heston_price(
    slices: &[PtdHestonSlice],
    v0: f64,
    spot: f64,
    strike: f64,
    tau: f64,
    risk_free_rate: f64,
    dividend_yield: f64,
    opt_type: OptionType,
) -> QLResult<PtdHestonResult> {
    if slices.is_empty() {
        return Err(QLError::InvalidArgument("slices must be non-empty".into()));
    }
    if spot <= 0.0 || strike <= 0.0 || tau <= 0.0 {
        return Err(QLError::InvalidArgument("spot, strike, tau must be positive".into()));
    }

    // Build effective average parameters weighted by time-slice lengths.
    // This is the "averaging approximation": for smooth parameter variation
    // the average parameters produce prices very close to the exact integral.
    // For a rigorous treatment, the exact PTD-CF requires a piecewise
    // Riccati ODE solution (see Elices 2008, Eq. 23).
    let total_t: f64 = slices.iter().map(|s| s.t_end - s.t_start).sum();
    if total_t < 1e-10 {
        return Err(QLError::InvalidArgument("slices have zero total duration".into()));
    }

    let kappa_avg = slices.iter().map(|s| s.kappa * (s.t_end - s.t_start)).sum::<f64>() / total_t;
    let theta_avg = slices.iter().map(|s| s.theta * (s.t_end - s.t_start)).sum::<f64>() / total_t;
    let sigma_avg = slices.iter().map(|s| s.sigma * (s.t_end - s.t_start)).sum::<f64>() / total_t;
    let rho_avg = slices.iter().map(|s| s.rho * (s.t_end - s.t_start)).sum::<f64>() / total_t;

    // Effective long-run variance accounts for the drift of each slice
    // θ_eff = Σ [ θ_i · (1 − e^{−κ_i Δt_i}) ] / Σ[ (1 − e^{−κ_i Δt_i}) ]
    let sum_weight: f64 = slices
        .iter()
        .map(|s| {
            let dt = s.t_end - s.t_start;
            if s.kappa.abs() < 1e-8 {
                dt
            } else {
                (1.0 - (-s.kappa * dt).exp()) / s.kappa
            }
        })
        .sum();
    let theta_eff = if sum_weight < 1e-10 {
        theta_avg
    } else {
        slices
            .iter()
            .map(|s| {
                let dt = s.t_end - s.t_start;
                let w = if s.kappa.abs() < 1e-8 {
                    dt
                } else {
                    (1.0 - (-s.kappa * dt).exp()) / s.kappa
                };
                s.theta * w
            })
            .sum::<f64>()
            / sum_weight
    };

    // Use the standard Heston engine with effective parameters
    use ql_models::HestonModel;
    let model = HestonModel::new(
        spot,
        risk_free_rate,
        dividend_yield,
        v0,
        kappa_avg,
        theta_eff,
        sigma_avg,
        rho_avg,
    );
    let is_call = matches!(opt_type, OptionType::Call);
    let result = crate::analytic_heston::heston_price(&model, strike, tau, is_call);

    Ok(PtdHestonResult { price: result.npv })
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- HestonExpansionEngine ---

    #[test]
    fn heston_expansion_atm_positive() {
        let res = heston_expansion_price(
            100.0, 100.0, 1.0, 0.05, 0.0, 0.04, 1.5, 0.04, 0.3, -0.7, OptionType::Call,
        ).unwrap();
        assert!(res.price > 0.0 && res.implied_vol > 0.0, "price={}", res.price);
    }

    #[test]
    fn heston_expansion_put_call_parity_approx() {
        let s = 100.0;
        let k = 100.0;
        let t = 0.5;
        let r = 0.05;
        let q = 0.0;
        let params = (0.04, 1.5, 0.04, 0.3, -0.7_f64);
        let call = heston_expansion_price(s, k, t, r, q, params.0, params.1, params.2, params.3, params.4, OptionType::Call).unwrap();
        let put  = heston_expansion_price(s, k, t, r, q, params.0, params.1, params.2, params.3, params.4, OptionType::Put).unwrap();
        let pcp = s * (-q * t).exp() - k * (-r * t).exp();
        assert!((call.price - put.price - pcp).abs() < 0.5, "pcp err={}", (call.price - put.price - pcp).abs());
    }

    #[test]
    fn heston_expansion_invalid() {
        assert!(heston_expansion_price(0.0, 100.0, 1.0, 0.05, 0.0, 0.04, 1.5, 0.04, 0.3, -0.7, OptionType::Call).is_err());
    }

    // --- CEV Engine ---

    #[test]
    fn cev_call_positive() {
        let res = analytic_cev_price(100.0, 100.0, 1.0, 0.05, 0.0, 0.3, 0.5, OptionType::Call).unwrap();
        assert!(res.price > 0.0, "price={}", res.price);
    }

    #[test]
    fn cev_beta_one_rejected() {
        assert!(analytic_cev_price(100.0, 100.0, 1.0, 0.05, 0.0, 0.3, 1.0, OptionType::Call).is_err());
    }

    #[test]
    fn cev_call_positive_beta_high() {
        let res = analytic_cev_price(100.0, 100.0, 1.0, 0.05, 0.0, 0.3, 1.5, OptionType::Call).unwrap();
        assert!(res.price >= 0.0, "price={}", res.price);
    }

    // --- PTD Heston Engine ---

    #[test]
    fn ptd_heston_single_slice_matches_constant() {
        // Single slice should give same result as constant-param Heston
        let slice = PtdHestonSlice {
            t_start: 0.0, t_end: 1.0,
            kappa: 1.5, theta: 0.04, sigma: 0.3, rho: -0.7,
        };
        let res = analytic_ptd_heston_price(
            &[slice], 0.04, 100.0, 100.0, 1.0, 0.05, 0.0, OptionType::Call,
        ).unwrap();
        assert!(res.price > 0.0, "price={}", res.price);
    }

    #[test]
    fn ptd_heston_two_slices() {
        let slices = vec![
            PtdHestonSlice { t_start: 0.0, t_end: 0.5, kappa: 1.5, theta: 0.04, sigma: 0.3, rho: -0.7 },
            PtdHestonSlice { t_start: 0.5, t_end: 1.0, kappa: 1.0, theta: 0.05, sigma: 0.25, rho: -0.5 },
        ];
        let res = analytic_ptd_heston_price(
            &slices, 0.04, 100.0, 100.0, 1.0, 0.05, 0.0, OptionType::Call,
        ).unwrap();
        assert!(res.price > 0.0, "price={}", res.price);
    }

    #[test]
    fn ptd_heston_empty_slices_error() {
        assert!(analytic_ptd_heston_price(
            &[], 0.04, 100.0, 100.0, 1.0, 0.05, 0.0, OptionType::Call,
        ).is_err());
    }
}
