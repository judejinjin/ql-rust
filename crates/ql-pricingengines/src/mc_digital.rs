//! Monte Carlo digital option engine.
//!
//! Prices cash-or-nothing and asset-or-nothing options via Monte Carlo
//! simulation under GBM (with optional barrier).

use serde::{Deserialize, Serialize};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

/// Type of digital option payoff.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum McDigitalType {
    /// Pays fixed cash amount if S_T > K (call) or S_T < K (put).
    CashOrNothing,
    /// Pays S_T if S_T > K (call) or S_T < K (put).
    AssetOrNothing,
}

/// Result from the MC digital engine.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct McDigitalResult {
    pub price: f64,
    pub std_error: f64,
    pub delta: f64,
}

/// Monte Carlo digital option engine.
///
/// Prices European digital (binary) options using Monte Carlo simulation.
/// Supports cash-or-nothing (pays fixed amount) and asset-or-nothing
/// (pays S_T) with call or put direction.
///
/// # Arguments
/// - `spot` — current underlying price
/// - `strike` — barrier level
/// - `r`, `q` — risk-free rate, dividend yield
/// - `sigma` — volatility
/// - `t` — time to expiry
/// - `is_call` — true for call (S_T > K), false for put (S_T < K)
/// - `digital_type` — CashOrNothing or AssetOrNothing
/// - `cash_amount` — fixed cash payoff (for CashOrNothing)
/// - `n_paths` — number of simulation paths
/// - `seed` — RNG seed
#[allow(clippy::too_many_arguments)]
pub fn mc_digital(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    is_call: bool,
    digital_type: McDigitalType,
    cash_amount: f64,
    n_paths: usize,
    seed: u64,
) -> McDigitalResult {
    let mut rng = StdRng::seed_from_u64(seed);
    let drift = (r - q - 0.5 * sigma * sigma) * t;
    let vol_sqrt_t = sigma * t.sqrt();
    let df = (-r * t).exp();

    let mut sum_payoff = 0.0;
    let mut sum_payoff_sq = 0.0;

    // For delta: use pathwise method (bump & reprice)
    let bump = spot * 0.001;
    let mut sum_payoff_up = 0.0;

    for _ in 0..n_paths {
        let z: f64 = StandardNormal.sample(&mut rng);
        let s_t = spot * (drift + vol_sqrt_t * z).exp();
        let s_t_up = (spot + bump) * (drift + vol_sqrt_t * z).exp();

        // Antithetic
        let s_t_anti = spot * (drift - vol_sqrt_dt_anti(vol_sqrt_t, z)).exp();

        let payoff = digital_payoff(s_t, strike, is_call, digital_type, cash_amount);
        let payoff_anti = digital_payoff(s_t_anti, strike, is_call, digital_type, cash_amount);
        let avg_payoff = 0.5 * (payoff + payoff_anti);

        sum_payoff += avg_payoff;
        sum_payoff_sq += avg_payoff * avg_payoff;

        let payoff_up = digital_payoff(s_t_up, strike, is_call, digital_type, cash_amount);
        sum_payoff_up += payoff_up;
    }

    let mean = sum_payoff / n_paths as f64;
    let var = sum_payoff_sq / n_paths as f64 - mean * mean;
    let std_error = (var / n_paths as f64).sqrt();

    let price = df * mean;
    let mean_up = sum_payoff_up / n_paths as f64;
    let delta = df * (mean_up - mean) / bump;

    McDigitalResult { price, std_error, delta }
}

fn vol_sqrt_dt_anti(vol_sqrt_t: f64, z: f64) -> f64 {
    // For antithetic: use -z
    vol_sqrt_t * (-z)
        + (vol_sqrt_t * z) // cancel to get drift - vol*sqrt(t)*z
        - vol_sqrt_t * z
}

fn digital_payoff(s_t: f64, strike: f64, is_call: bool, digital_type: McDigitalType, cash: f64) -> f64 {
    let in_the_money = if is_call { s_t > strike } else { s_t < strike };
    if !in_the_money {
        return 0.0;
    }
    match digital_type {
        McDigitalType::CashOrNothing => cash,
        McDigitalType::AssetOrNothing => s_t,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_mc_cash_or_nothing_call() {
        // Analytic: df * N(d2) * cash
        let res = mc_digital(
            100.0, 100.0, 0.05, 0.02, 0.20, 1.0,
            true, McDigitalType::CashOrNothing, 1.0,
            200_000, 42,
        );
        // d2 for ATM≈ (r-q-σ²/2)*T/(σ√T) = (0.05-0.02-0.02)/0.20 = 0.05
        // N(0.05) ≈ 0.5199, df ≈ 0.9512  => ~0.4945
        assert_abs_diff_eq!(res.price, 0.495, epsilon = 0.03);
    }

    #[test]
    fn test_mc_asset_or_nothing_call() {
        let res = mc_digital(
            100.0, 100.0, 0.05, 0.02, 0.20, 1.0,
            true, McDigitalType::AssetOrNothing, 1.0,
            200_000, 42,
        );
        // Analytic: S*exp(-q*T)*N(d1) ≈ 100*exp(-0.02)*N(0.25) ≈ 98.02*0.5987≈ 58.7
        assert!(res.price > 45.0 && res.price < 70.0, "price={}", res.price);
    }

    #[test]
    fn test_mc_cash_or_nothing_put() {
        let call = mc_digital(
            100.0, 100.0, 0.05, 0.02, 0.20, 1.0,
            true, McDigitalType::CashOrNothing, 1.0,
            200_000, 42,
        );
        let put = mc_digital(
            100.0, 100.0, 0.05, 0.02, 0.20, 1.0,
            false, McDigitalType::CashOrNothing, 1.0,
            200_000, 42,
        );
        // C + P = df * cash for binary options
        let df = (-0.05_f64).exp();
        assert_abs_diff_eq!(call.price + put.price, df, epsilon = 0.05);
    }
}
