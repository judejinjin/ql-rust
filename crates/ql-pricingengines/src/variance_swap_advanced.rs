//! Advanced variance swap engines.
//!
//! - [`replicating_variance_swap`] — Replicating variance swap engine using
//!   a strip of European options.
//! - [`mc_variance_swap`] — Monte Carlo variance swap engine.

use serde::{Deserialize, Serialize};

/// Result from variance swap engines.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VarianceSwapResult {
    /// Fair variance strike (K_var = E[σ²_realized]).
    pub fair_variance: f64,
    /// Fair volatility strike (√fair_variance).
    pub fair_volatility: f64,
    /// Present value of the swap (long variance).
    pub pv: f64,
}

// ---------------------------------------------------------------------------
// Replicating Variance Swap Engine
// ---------------------------------------------------------------------------

/// Price a variance swap using the model-free replication formula.
///
/// The fair variance is computed from a strip of OTM European option prices:
///
/// $$ K_{var} = \frac{2}{T} \left[ \int_0^F \frac{P(K)}{K^2} dK
///              + \int_F^\infty \frac{C(K)}{K^2} dK \right] $$
///
/// where F is the forward price, P and C are OTM put/call prices.
///
/// The option prices are computed from a Black-Scholes implied volatility
/// surface given as a set of strikes and volatilities.
///
/// # Arguments
/// - `spot` — current underlying price
/// - `r`, `q` — risk-free rate, dividend yield
/// - `t` — time to expiry
/// - `strikes` — array of strike prices for the replicating portfolio
/// - `implied_vols` — implied volatilities at each strike
/// - `variance_strike` — agreed variance strike (for PV computation)
/// - `notional` — variance notional
#[allow(clippy::too_many_arguments)]
pub fn replicating_variance_swap(
    spot: f64,
    r: f64,
    q: f64,
    t: f64,
    strikes: &[f64],
    implied_vols: &[f64],
    variance_strike: f64,
    notional: f64,
) -> VarianceSwapResult {
    assert_eq!(strikes.len(), implied_vols.len(), "strikes and vols must have same length");
    assert!(strikes.len() >= 2, "need at least 2 strikes");

    let forward = spot * ((r - q) * t).exp();
    let df = (-r * t).exp();

    // Compute option prices at each strike
    let n = strikes.len();
    let mut option_prices = vec![0.0; n];
    for i in 0..n {
        let k = strikes[i];
        let sigma = implied_vols[i];
        let is_call = k >= forward; // Use OTM: calls above forward, puts below
        option_prices[i] = bs_price(spot, k, r, q, sigma, t, is_call);
    }

    // Numerical integration: ∑ (2/T) * Δk * option_price / K²
    // The integration log-contract approach
    let mut fair_variance = 0.0;

    // Add the log-contract term: 2/T * [F/K_0 - 1 - ln(F/K_0)]
    // where K_0 is the nearest strike below forward
    let k0_idx = strikes.iter().position(|&k| k >= forward).unwrap_or(n - 1);
    let k0_idx = if k0_idx > 0 { k0_idx - 1 } else { 0 };
    let k0 = strikes[k0_idx];

    // Log-contract correction
    fair_variance += (2.0 / t) * (forward / k0 - 1.0 - (forward / k0).ln());

    // Sum over strikes (trapezoidal integration)
    for i in 0..n {
        let k = strikes[i];
        let dk = if i == 0 {
            strikes[1] - strikes[0]
        } else if i == n - 1 {
            strikes[n - 1] - strikes[n - 2]
        } else {
            (strikes[i + 1] - strikes[i - 1]) / 2.0
        };

        fair_variance += (2.0 / t) * dk * option_prices[i] / (k * k * df);
    }

    let fair_volatility = fair_variance.sqrt();
    let pv = notional * df * (fair_variance - variance_strike);

    VarianceSwapResult {
        fair_variance,
        fair_volatility,
        pv,
    }
}

// ---------------------------------------------------------------------------
// MC Variance Swap Engine
// ---------------------------------------------------------------------------

/// Monte Carlo variance swap engine.
///
/// Simulates paths and computes realized variance from discrete log-returns.
///
/// # Arguments
/// - `spot` — current underlying price
/// - `r`, `q` — risk-free rate, dividend yield
/// - `sigma` — constant volatility (for GBM simulation)
/// - `t` — time to expiry
/// - `n_fixings` — number of monitoring dates
/// - `n_paths` — number of Monte Carlo paths
/// - `variance_strike` — agreed variance strike
/// - `notional` — variance notional
/// - `seed` — RNG seed
#[allow(clippy::too_many_arguments)]
pub fn mc_variance_swap(
    spot: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    n_fixings: usize,
    n_paths: usize,
    variance_strike: f64,
    notional: f64,
    seed: u64,
) -> VarianceSwapResult {
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    let dt = t / n_fixings as f64;
    let drift = (r - q - 0.5 * sigma * sigma) * dt;
    let vol_sqrt_dt = sigma * dt.sqrt();
    let df = (-r * t).exp();

    let mut rng = StdRng::seed_from_u64(seed);
    let mut sum_variance = 0.0;
    let mut _sum_variance_sq = 0.0;

    for _ in 0..n_paths {
        let mut s = spot;
        let mut sum_log_return_sq = 0.0;

        for _ in 0..n_fixings {
            let z: f64 = sample_normal(&mut rng);
            let s_new = s * (drift + vol_sqrt_dt * z).exp();
            let log_ret = (s_new / s).ln();
            sum_log_return_sq += log_ret * log_ret;
            s = s_new;
        }

        // Annualized realized variance
        let realized_var = sum_log_return_sq / t;
        sum_variance += realized_var;
        _sum_variance_sq += realized_var * realized_var;
    }

    let fair_variance = sum_variance / n_paths as f64;
    let fair_volatility = fair_variance.sqrt();
    let pv = notional * df * (fair_variance - variance_strike);

    VarianceSwapResult {
        fair_variance,
        fair_volatility,
        pv,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn bs_price(spot: f64, strike: f64, r: f64, q: f64, sigma: f64, t: f64, is_call: bool) -> f64 {
    use ql_math::distributions::cumulative_normal;

    let omega = if is_call { 1.0 } else { -1.0 };
    let fwd = spot * ((r - q) * t).exp();
    let df = (-r * t).exp();
    let sqrt_t = t.sqrt();
    let d1 = ((fwd / strike).ln() + 0.5 * sigma * sigma * t) / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;
    df * omega * (fwd * cumulative_normal(omega * d1) - strike * cumulative_normal(omega * d2))
}

fn sample_normal(rng: &mut impl rand::Rng) -> f64 {
    use rand_distr::{Distribution, StandardNormal};
    StandardNormal.sample(rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_replicating_variance_swap_flat_vol() {
        // With flat vol, fair variance should ≈ σ²
        let sigma = 0.20;
        let n = 21;
        let strikes: Vec<f64> = (0..n).map(|i| 60.0 + i as f64 * 4.0).collect();
        let vols: Vec<f64> = vec![sigma; n];

        let res = replicating_variance_swap(
            100.0, 0.05, 0.02, 1.0,
            &strikes, &vols,
            sigma * sigma, 100_000.0,
        );

        // Fair variance should be close to σ² = 0.04
        assert_abs_diff_eq!(res.fair_variance, sigma * sigma, epsilon = 0.01);
    }

    #[test]
    fn test_mc_variance_swap_gbm() {
        // Under GBM, realized variance → σ² as n_paths → ∞
        let sigma = 0.25;
        let res = mc_variance_swap(
            100.0, 0.05, 0.02, sigma,
            1.0, 252, 100_000, sigma * sigma,
            100_000.0, 42,
        );

        // Fair variance ≈ σ²
        assert_abs_diff_eq!(res.fair_variance, sigma * sigma, epsilon = 0.01);
        // PV ≈ 0 when variance_strike = fair variance
        assert!(res.pv.abs() < 1500.0, "pv={}", res.pv);
    }

    #[test]
    fn test_mc_variance_swap_pv_sign() {
        let sigma = 0.25;
        // variance_strike < σ² → long variance should have positive PV
        let res = mc_variance_swap(
            100.0, 0.05, 0.02, sigma,
            1.0, 252, 50_000, 0.04,
            100_000.0, 42,
        );
        assert!(res.pv > 0.0, "pv={} should be positive for low strike", res.pv);
    }
}
