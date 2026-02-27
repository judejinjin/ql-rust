//! Monte Carlo forward-start option engines.
//!
//! - [`mc_forward_european_bs`] — MC forward-start European under GBM
//! - [`mc_forward_european_heston`] — MC forward-start European under Heston

use rand::prelude::*;
use rand_distr::StandardNormal;

/// Result from MC forward-start engines.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct McForwardResult {
    /// Option price.
    pub price: f64,
    /// Standard error.
    pub std_error: f64,
}

/// Monte Carlo forward-start European option under GBM.
///
/// A forward-start option has its strike set at some future date `t_start`
/// as a fraction `alpha` of the spot at that time: K = α·S(t_start).
///
/// # Arguments
/// - `spot` — current price
/// - `alpha` — strike ratio (K = α·S(t_start))
/// - `r` — risk-free rate
/// - `q` — dividend yield
/// - `sigma` — volatility
/// - `t_start` — time to forward-start date
/// - `t_expiry` — time to expiry
/// - `is_call` — true for call
/// - `n_paths` — MC paths
/// - `seed` — optional seed
#[allow(clippy::too_many_arguments)]
pub fn mc_forward_european_bs(
    spot: f64,
    alpha: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t_start: f64,
    t_expiry: f64,
    is_call: bool,
    n_paths: usize,
    seed: Option<u64>,
) -> McForwardResult {
    let omega = if is_call { 1.0 } else { -1.0 };
    let df = (-r * t_expiry).exp();
    let dt1 = t_start;
    let dt2 = t_expiry - t_start;

    let drift1 = (r - q - 0.5 * sigma * sigma) * dt1;
    let vol1 = sigma * dt1.sqrt();
    let drift2 = (r - q - 0.5 * sigma * sigma) * dt2;
    let vol2 = sigma * dt2.sqrt();

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::seed_from_u64(42),
    };

    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    for _ in 0..n_paths / 2 {
        let z1: f64 = rng.sample(StandardNormal);
        let z2: f64 = rng.sample(StandardNormal);

        // Path 1
        let s_start = spot * (drift1 + vol1 * z1).exp();
        let strike = alpha * s_start;
        let s_expiry = s_start * (drift2 + vol2 * z2).exp();
        let payoff1 = (omega * (s_expiry - strike)).max(0.0);

        // Antithetic
        let s_start_a = spot * (drift1 - vol1 * z1).exp();
        let strike_a = alpha * s_start_a;
        let s_expiry_a = s_start_a * (drift2 - vol2 * z2).exp();
        let payoff2 = (omega * (s_expiry_a - strike_a)).max(0.0);

        let payoff = 0.5 * (payoff1 + payoff2);
        sum += payoff;
        sum_sq += payoff * payoff;
    }

    let n_eff = (n_paths / 2) as f64;
    let mean = sum / n_eff;
    let variance = (sum_sq / n_eff - mean * mean).max(0.0);
    let std_error = (variance / n_eff).sqrt();

    McForwardResult {
        price: df * mean,
        std_error: df * std_error,
    }
}

/// Monte Carlo forward-start European option under Heston dynamics.
///
/// Simulates the Heston SDE from t=0 to t_expiry, with the strike
/// set at t_start as K = α·S(t_start).
#[allow(clippy::too_many_arguments)]
pub fn mc_forward_european_heston(
    spot: f64,
    alpha: f64,
    r: f64,
    q: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    t_start: f64,
    t_expiry: f64,
    is_call: bool,
    n_steps: usize,
    n_paths: usize,
    seed: Option<u64>,
) -> McForwardResult {
    let omega = if is_call { 1.0 } else { -1.0 };
    let df = (-r * t_expiry).exp();
    let dt = t_expiry / n_steps as f64;
    let rho_bar = (1.0 - rho * rho).sqrt();
    let start_step = (t_start / dt).round() as usize;

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::seed_from_u64(42),
    };

    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    for _ in 0..n_paths {
        let mut s = spot;
        let mut v = v0;
        let mut strike = alpha * spot; // will be reset at start_step

        for step in 0..n_steps {
            let z1: f64 = rng.sample(StandardNormal);
            let z2: f64 = rng.sample(StandardNormal);
            let w1 = z1;
            let w2 = rho * z1 + rho_bar * z2;

            let v_pos = v.max(0.0);
            let sqrt_v = v_pos.sqrt();
            s *= ((r - q - 0.5 * v_pos) * dt + sqrt_v * dt.sqrt() * w1).exp();
            v = (v + kappa * (theta - v_pos) * dt + sigma * sqrt_v * dt.sqrt() * w2).max(0.0);

            if step == start_step {
                strike = alpha * s;
            }
        }

        let payoff = (omega * (s - strike)).max(0.0);
        sum += payoff;
        sum_sq += payoff * payoff;
    }

    let n = n_paths as f64;
    let mean = sum / n;
    let variance = (sum_sq / n - mean * mean).max(0.0);
    let std_error = (variance / n).sqrt();

    McForwardResult {
        price: df * mean,
        std_error: df * std_error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mc_forward_bs_call() {
        let res = mc_forward_european_bs(
            100.0, 1.0, 0.05, 0.0, 0.20,
            0.5, 1.5, true, 50000, Some(42),
        );
        // Forward-start ATM call with 1y remaining
        assert!(res.price > 5.0 && res.price < 20.0, "price={}", res.price);
    }

    #[test]
    fn test_mc_forward_bs_put() {
        let res = mc_forward_european_bs(
            100.0, 1.0, 0.05, 0.0, 0.20,
            0.5, 1.5, false, 50000, Some(42),
        );
        assert!(res.price > 1.0 && res.price < 15.0, "price={}", res.price);
    }

    #[test]
    fn test_mc_forward_heston_call() {
        let res = mc_forward_european_heston(
            100.0, 1.0, 0.05, 0.0,
            0.04, 1.5, 0.04, 0.3, -0.7,
            0.5, 1.5, true, 200, 20000, Some(42),
        );
        assert!(res.price > 2.0 && res.price < 20.0, "price={}", res.price);
    }
}
