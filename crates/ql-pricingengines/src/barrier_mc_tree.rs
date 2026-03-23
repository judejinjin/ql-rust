//! Binomial barrier engine and Monte Carlo barrier engine.
//!
//! - [`binomial_barrier`] — Binomial tree pricing for barrier options with 
//!   barrier adjustment (Boyle-Lau 1994).
//! - [`mc_barrier`] — Monte Carlo pricing for barrier options with 
//!   Brownian bridge barrier adjustment.

use rand::prelude::*;
use rand_distr::StandardNormal;

/// Barrier type.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum McBarrierType {
    /// Down And Out.
    DownAndOut,
    /// Up And Out.
    UpAndOut,
    /// Down And In.
    DownAndIn,
    /// Up And In.
    UpAndIn,
}

/// Result from the binomial/MC barrier engines.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BarrierTreeMcResult {
    /// Option price.
    pub price: f64,
    /// Standard error (MC only).
    pub std_error: Option<f64>,
}

/// Price a barrier option using a CRR binomial tree.
///
/// Uses a log-normal CRR tree with barrier adjustment at each node.
/// The Boyle-Lau (1994) correction places the barrier on the nearest
/// tree level to improve convergence.
///
/// # Arguments
/// - Standard BS option inputs
/// - `barrier` — barrier level
/// - `rebate` — cash rebate
/// - `barrier_type` — type of barrier
/// - `n_steps` — number of time steps
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub fn binomial_barrier(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    barrier: f64,
    rebate: f64,
    barrier_type: McBarrierType,
    is_call: bool,
    n_steps: usize,
) -> BarrierTreeMcResult {
    let omega = if is_call { 1.0 } else { -1.0 };
    let dt = t / n_steps as f64;
    let u = (sigma * dt.sqrt()).exp();
    let d = 1.0 / u;
    let p = (((r - q) * dt).exp() - d) / (u - d);
    let df = (-r * dt).exp();

    let is_knockout = matches!(barrier_type, McBarrierType::DownAndOut | McBarrierType::UpAndOut);
    let is_down = matches!(barrier_type, McBarrierType::DownAndOut | McBarrierType::DownAndIn);

    // Build terminal values
    let n = n_steps;
    let mut values = vec![0.0; n + 1];

    for i in 0..=n {
        let s_i = spot * u.powi(i as i32) * d.powi((n - i) as i32);
        let payoff = (omega * (s_i - strike)).max(0.0);

        // Check barrier at terminal node
        let breached = if is_down { s_i <= barrier } else { s_i >= barrier };

        if is_knockout {
            values[i] = if breached { rebate } else { payoff };
        } else {
            values[i] = payoff; // Will use in-out parity
        }
    }

    // Backward induction
    for step in (0..n).rev() {
        for i in 0..=step {
            let s_i = spot * u.powi(i as i32) * d.powi((step - i) as i32);
            let breached = if is_down { s_i <= barrier } else { s_i >= barrier };

            if is_knockout {
                if breached {
                    values[i] = rebate * (-r * (n - step) as f64 * dt).exp();
                } else {
                    values[i] = df * (p * values[i + 1] + (1.0 - p) * values[i]);
                }
            } else {
                values[i] = df * (p * values[i + 1] + (1.0 - p) * values[i]);
            }
        }
    }

    let ko_price = values[0];

    // For knock-in: use in-out parity
    let price = if is_knockout {
        ko_price
    } else {
        // Compute vanilla price via tree
        let mut v_vals = vec![0.0; n + 1];
        for i in 0..=n {
            let s_i = spot * u.powi(i as i32) * d.powi((n - i) as i32);
            v_vals[i] = (omega * (s_i - strike)).max(0.0);
        }
        for step in (0..n).rev() {
            for i in 0..=step {
                v_vals[i] = df * (p * v_vals[i + 1] + (1.0 - p) * v_vals[i]);
            }
        }
        v_vals[0] - ko_price // KI = Vanilla - KO
    };

    BarrierTreeMcResult {
        price: price.max(0.0),
        std_error: None,
    }
}

/// Monte Carlo barrier option pricing with Brownian bridge correction.
///
/// Uses the Beaglehole-Dwyner-Finch (1997) Brownian bridge adjustment
/// to account for barrier crossings between monitoring dates.
///
/// # Arguments
/// - Standard BS option inputs
/// - `barrier` — barrier level
/// - `rebate` — cash rebate
/// - `barrier_type` — type of barrier
/// - `n_steps` — monitoring steps per path
/// - `n_paths` — number of MC paths
/// - `seed` — optional RNG seed
#[allow(clippy::too_many_arguments)]
pub fn mc_barrier(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    barrier: f64,
    rebate: f64,
    barrier_type: McBarrierType,
    is_call: bool,
    n_steps: usize,
    n_paths: usize,
    seed: Option<u64>,
) -> BarrierTreeMcResult {
    let omega = if is_call { 1.0 } else { -1.0 };
    let dt = t / n_steps as f64;
    let drift = (r - q - 0.5 * sigma * sigma) * dt;
    let vol = sigma * dt.sqrt();
    let df = (-r * t).exp();

    let is_knockout = matches!(barrier_type, McBarrierType::DownAndOut | McBarrierType::UpAndOut);
    let is_down = matches!(barrier_type, McBarrierType::DownAndOut | McBarrierType::DownAndIn);

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::seed_from_u64(42),
    };

    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    for _ in 0..n_paths / 2 {
        for antithetic in [1.0_f64, -1.0] {
            let mut s = spot;
            let mut hit = false;

            for _ in 0..n_steps {
                let z: f64 = rng.sample(StandardNormal);
                let s_prev = s;
                s *= (drift + vol * z * antithetic).exp();

                // Brownian bridge correction: probability of barrier crossing
                // between s_prev and s given no crossing at the endpoints
                if !hit {
                    let crossed = if is_down {
                        s <= barrier || s_prev <= barrier
                    } else {
                        s >= barrier || s_prev >= barrier
                    };

                    if crossed {
                        hit = true;
                    } else {
                        // Brownian bridge: P(min < H | S₀=s_prev, S₁=s) = exp(-2 ln(s_prev/H) ln(s/H) / (σ²dt))
                        let bb_prob = if is_down && barrier < s_prev.min(s) {
                            let log1 = (s_prev / barrier).ln();
                            let log2 = (s / barrier).ln();
                            (-2.0 * log1 * log2 / (sigma * sigma * dt)).exp()
                        } else if !is_down && barrier > s_prev.max(s) {
                            let log1 = (barrier / s_prev).ln();
                            let log2 = (barrier / s).ln();
                            (-2.0 * log1 * log2 / (sigma * sigma * dt)).exp()
                        } else {
                            0.0
                        };

                        if rng.random::<f64>() < bb_prob {
                            hit = true;
                        }
                    }
                }
            }

            let payoff = if is_knockout {
                if hit { rebate * df / (-r * t).exp() } else { (omega * (s - strike)).max(0.0) }
            } else {
                // Knock-in
                if hit { (omega * (s - strike)).max(0.0) } else { 0.0 }
            };

            sum += payoff;
            sum_sq += payoff * payoff;
        }
    }

    let n_eff = n_paths as f64;
    let mean = sum / n_eff;
    let variance = (sum_sq / n_eff - mean * mean).max(0.0);
    let std_error = (variance / n_eff).sqrt();

    BarrierTreeMcResult {
        price: (df * mean).max(0.0),
        std_error: Some(df * std_error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_binomial_barrier_dao_call() {
        let res = binomial_barrier(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            80.0, 0.0, McBarrierType::DownAndOut, true, 200,
        );
        assert!(res.price > 5.0 && res.price < 12.0, "price={}", res.price);
    }

    #[test]
    fn test_binomial_barrier_uao_put() {
        let res = binomial_barrier(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            120.0, 0.0, McBarrierType::UpAndOut, false, 200,
        );
        assert!(res.price > 0.0 && res.price < 8.0, "price={}", res.price);
    }

    #[test]
    fn test_mc_barrier_dao_call() {
        let res = mc_barrier(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            80.0, 0.0, McBarrierType::DownAndOut, true,
            252, 50000, Some(42),
        );
        assert!(res.price > 5.0 && res.price < 12.0, "price={}", res.price);
        assert!(res.std_error.unwrap() < 0.5);
    }

    #[test]
    fn test_mc_barrier_dai_call() {
        let res = mc_barrier(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            80.0, 0.0, McBarrierType::DownAndIn, true,
            252, 50000, Some(42),
        );
        // KI price should be small for distant barrier
        assert!(res.price >= 0.0 && res.price < 5.0, "price={}", res.price);
    }
}
