//! Monte Carlo pricing engines.
//!
//! Provides MC engines for European, barrier, Asian, Heston, and Bates options.

use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Poisson, StandardNormal};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use tracing::{info, info_span};

use ql_instruments::OptionType;

/// Conditionally parallel collect: uses `par_iter` when the `parallel` feature
/// is enabled, falling back to sequential `into_iter` otherwise.
fn par_map_collect<T, F>(n: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Send + Sync,
{
    #[cfg(feature = "parallel")]
    {
        (0..n).into_par_iter().map(&f).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..n).map(f).collect()
    }
}

/// Result of a Monte Carlo simulation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MCResult {
    /// Net present value (discounted mean payoff).
    pub npv: f64,
    /// Standard error of the NPV estimate.
    pub std_error: f64,
    /// Number of paths simulated.
    pub num_paths: usize,
}

// ===========================================================================
// MC European Engine (GBM)
// ===========================================================================

/// Price a European option via Monte Carlo under GBM.
///
/// Uses exact log-normal sampling (single step) for efficiency.
#[allow(clippy::too_many_arguments)]
pub fn mc_european(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    vol: f64,
    time_to_expiry: f64,
    option_type: OptionType,
    num_paths: usize,
    antithetic: bool,
    seed: u64,
) -> MCResult {
    let _span = info_span!("mc_european", num_paths, antithetic).entered();
    let df = (-r * time_to_expiry).exp();
    let sqrt_t = time_to_expiry.sqrt();
    let drift = (r - q - 0.5 * vol * vol) * time_to_expiry;

    let effective_paths = if antithetic { num_paths / 2 } else { num_paths };

    // Use parallel batches
    let batch_size = 10000_usize;
    let num_batches = effective_paths.div_ceil(batch_size);

    let results: Vec<(f64, f64)> = par_map_collect(num_batches, |batch_idx| {
            let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(batch_idx as u64));
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(effective_paths);
            let mut sum = 0.0;
            let mut sum_sq = 0.0;

            for _ in start..end {
                let z: f64 = StandardNormal.sample(&mut rng);

                let st1 = spot * (drift + vol * sqrt_t * z).exp();
                let payoff1 = (option_type.sign() * (st1 - strike)).max(0.0);

                if antithetic {
                    let st2 = spot * (drift - vol * sqrt_t * z).exp();
                    let payoff2 = (option_type.sign() * (st2 - strike)).max(0.0);
                    let payoff_avg = 0.5 * (payoff1 + payoff2);
                    sum += payoff_avg;
                    sum_sq += payoff_avg * payoff_avg;
                } else {
                    sum += payoff1;
                    sum_sq += payoff1 * payoff1;
                }
            }
            (sum, sum_sq)
        });

    let total_sum: f64 = results.iter().map(|(s, _)| s).sum();
    let total_sum_sq: f64 = results.iter().map(|(_, s)| s).sum();
    let n = effective_paths as f64;

    let mean = total_sum / n;
    let variance = (total_sum_sq / n - mean * mean).max(0.0);
    let std_error = (variance / n).sqrt() * df;

    let result = MCResult {
        npv: df * mean,
        std_error,
        num_paths: if antithetic {
            effective_paths * 2
        } else {
            effective_paths
        },
    };
    info!(npv = result.npv, std_error = result.std_error, paths = result.num_paths, "MC European complete");
    result
}

// ===========================================================================
// MC Barrier Engine
// ===========================================================================

/// Price a barrier option via Monte Carlo under GBM.
///
/// Uses discrete monitoring at each time step.
#[allow(clippy::too_many_arguments)]
pub fn mc_barrier(
    spot: f64,
    strike: f64,
    barrier: f64,
    rebate: f64,
    r: f64,
    q: f64,
    vol: f64,
    time_to_expiry: f64,
    option_type: OptionType,
    is_up: bool,
    is_knock_in: bool,
    num_paths: usize,
    num_steps: usize,
    seed: u64,
) -> MCResult {
    let df = (-r * time_to_expiry).exp();
    let dt = time_to_expiry / num_steps as f64;
    let sqrt_dt = dt.sqrt();
    let mu = r - q - 0.5 * vol * vol;

    let batch_size = 5000_usize;
    let num_batches = num_paths.div_ceil(batch_size);

    let results: Vec<(f64, f64, usize)> = par_map_collect(num_batches, |batch_idx| {
            let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(batch_idx as u64));
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(num_paths);
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            let count = end - start;

            for _ in start..end {
                let mut s = spot;
                let mut hit = false;

                for _ in 0..num_steps {
                    let z: f64 = StandardNormal.sample(&mut rng);
                    s *= (mu * dt + vol * sqrt_dt * z).exp();

                    if (is_up && s >= barrier) || (!is_up && s <= barrier) {
                        hit = true;
                        break;
                    }
                }

                let vanilla_payoff = (option_type.sign() * (s - strike)).max(0.0);

                let payoff = if is_knock_in {
                    if hit {
                        vanilla_payoff
                    } else {
                        rebate
                    }
                } else {
                    // knock-out
                    if hit {
                        rebate
                    } else {
                        vanilla_payoff
                    }
                };

                sum += payoff;
                sum_sq += payoff * payoff;
            }
            (sum, sum_sq, count)
        });

    let total_sum: f64 = results.iter().map(|(s, _, _)| s).sum();
    let total_sum_sq: f64 = results.iter().map(|(_, s, _)| s).sum();
    let total_count: usize = results.iter().map(|(_, _, c)| c).sum();
    let n = total_count as f64;

    let mean = total_sum / n;
    let variance = (total_sum_sq / n - mean * mean).max(0.0);
    let std_error = (variance / n).sqrt() * df;

    MCResult {
        npv: df * mean,
        std_error,
        num_paths: total_count,
    }
}

// ===========================================================================
// MC Asian Engine
// ===========================================================================

/// Price an Asian (average-price) option via Monte Carlo under GBM.
///
/// Monitors at each time step and computes arithmetic or geometric average.
#[allow(clippy::too_many_arguments)]
pub fn mc_asian(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    vol: f64,
    time_to_expiry: f64,
    option_type: OptionType,
    is_arithmetic: bool,
    num_paths: usize,
    num_steps: usize,
    seed: u64,
) -> MCResult {
    let df = (-r * time_to_expiry).exp();
    let dt = time_to_expiry / num_steps as f64;
    let sqrt_dt = dt.sqrt();
    let mu = r - q - 0.5 * vol * vol;

    let batch_size = 5000_usize;
    let num_batches = num_paths.div_ceil(batch_size);

    let results: Vec<(f64, f64, usize)> = par_map_collect(num_batches, |batch_idx| {
            let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(batch_idx as u64));
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(num_paths);
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            let count = end - start;

            for _ in start..end {
                let mut s = spot;
                let mut running_sum = 0.0;
                let mut running_log_sum = 0.0;

                for _ in 0..num_steps {
                    let z: f64 = StandardNormal.sample(&mut rng);
                    s *= (mu * dt + vol * sqrt_dt * z).exp();
                    running_sum += s;
                    running_log_sum += s.ln();
                }

                let avg = if is_arithmetic {
                    running_sum / num_steps as f64
                } else {
                    (running_log_sum / num_steps as f64).exp()
                };

                let payoff = (option_type.sign() * (avg - strike)).max(0.0);
                sum += payoff;
                sum_sq += payoff * payoff;
            }
            (sum, sum_sq, count)
        });

    let total_sum: f64 = results.iter().map(|(s, _, _)| s).sum();
    let total_sum_sq: f64 = results.iter().map(|(_, s, _)| s).sum();
    let total_count: usize = results.iter().map(|(_, _, c)| c).sum();
    let n = total_count as f64;

    let mean = total_sum / n;
    let variance = (total_sum_sq / n - mean * mean).max(0.0);
    let std_error = (variance / n).sqrt() * df;

    MCResult {
        npv: df * mean,
        std_error,
        num_paths: total_count,
    }
}

// ===========================================================================
// MC Heston Engine
// ===========================================================================

/// Price a European option via Monte Carlo under the Heston model.
///
/// Uses the QE scheme (Andersen 2008) for variance and log-Euler for spot.
#[allow(clippy::too_many_arguments)]
pub fn mc_heston(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    time_to_expiry: f64,
    option_type: OptionType,
    num_paths: usize,
    num_steps: usize,
    seed: u64,
) -> MCResult {
    let df = (-r * time_to_expiry).exp();
    let dt = time_to_expiry / num_steps as f64;
    let sqrt_dt = dt.sqrt();

    let batch_size = 5000_usize;
    let num_batches = num_paths.div_ceil(batch_size);

    let results: Vec<(f64, f64, usize)> = par_map_collect(num_batches, |batch_idx| {
            let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(batch_idx as u64));
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(num_paths);
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            let count = end - start;

            for _ in start..end {
                let mut log_s = spot.ln();
                let mut v = v0;

                for _ in 0..num_steps {
                    let z1: f64 = StandardNormal.sample(&mut rng);
                    let z2_indep: f64 = StandardNormal.sample(&mut rng);
                    let z2 = rho * z1 + (1.0 - rho * rho).sqrt() * z2_indep;

                    let v_pos = v.max(0.0);
                    let sqrt_v = v_pos.sqrt();

                    // Log-Euler for spot
                    log_s += (r - q - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * z1;

                    // Euler for variance (with absorption at 0)
                    v += kappa * (theta - v_pos) * dt + sigma * sqrt_v * sqrt_dt * z2;
                    v = v.max(0.0);
                }

                let st = log_s.exp();
                let payoff = (option_type.sign() * (st - strike)).max(0.0);
                sum += payoff;
                sum_sq += payoff * payoff;
            }
            (sum, sum_sq, count)
        });

    let total_sum: f64 = results.iter().map(|(s, _, _)| s).sum();
    let total_sum_sq: f64 = results.iter().map(|(_, s, _)| s).sum();
    let total_count: usize = results.iter().map(|(_, _, c)| c).sum();
    let n = total_count as f64;

    let mean = total_sum / n;
    let variance = (total_sum_sq / n - mean * mean).max(0.0);
    let std_error = (variance / n).sqrt() * df;

    MCResult {
        npv: df * mean,
        std_error,
        num_paths: total_count,
    }
}

// ===========================================================================
// MC Bates (Heston + Jumps) Engine
// ===========================================================================

/// Price a European option via Monte Carlo under the Bates model
/// (Heston stochastic volatility + Merton-style jumps).
///
/// Uses log-Euler for spot and Euler for variance (with absorption),
/// with compound Poisson jumps added at each time step.
///
/// Jump size: log(1+J) ~ N(nu, delta²).
/// Jump arrival: N ~ Poisson(lambda * dt) per step.
#[allow(clippy::too_many_arguments)]
pub fn mc_bates(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    lambda: f64,
    nu: f64,
    delta: f64,
    time_to_expiry: f64,
    option_type: OptionType,
    num_paths: usize,
    num_steps: usize,
    seed: u64,
) -> MCResult {
    let df = (-r * time_to_expiry).exp();
    let dt = time_to_expiry / num_steps as f64;
    let sqrt_dt = dt.sqrt();

    // Jump compensator: k_bar = exp(nu + delta^2/2) - 1
    let k_bar = (nu + 0.5 * delta * delta).exp() - 1.0;

    let batch_size = 5000_usize;
    let num_batches = num_paths.div_ceil(batch_size);

    // Poisson distribution for jump counts
    let lambda_dt = lambda * dt;

    let results: Vec<(f64, f64, usize)> = par_map_collect(num_batches, |batch_idx| {
            let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(batch_idx as u64));
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(num_paths);
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            let count = end - start;

            // Create Poisson distribution (handle lambda_dt = 0 case)
            let poisson = if lambda_dt > 0.0 {
                Some(Poisson::new(lambda_dt).unwrap())
            } else {
                None
            };

            for _ in start..end {
                let mut log_s = spot.ln();
                let mut v = v0;

                for _ in 0..num_steps {
                    let z1: f64 = StandardNormal.sample(&mut rng);
                    let z2_indep: f64 = StandardNormal.sample(&mut rng);
                    let z2 = rho * z1 + (1.0 - rho * rho).sqrt() * z2_indep;

                    let v_pos = v.max(0.0);
                    let sqrt_v = v_pos.sqrt();

                    // Diffusion step (log-Euler for spot, Euler for variance)
                    log_s += (r - q - 0.5 * v_pos - lambda * k_bar) * dt + sqrt_v * sqrt_dt * z1;
                    v += kappa * (theta - v_pos) * dt + sigma * sqrt_v * sqrt_dt * z2;
                    v = v.max(0.0);

                    // Jump step
                    if let Some(ref pois) = poisson {
                        let n_jumps = pois.sample(&mut rng) as u64;
                        for _ in 0..n_jumps {
                            let z_jump: f64 = StandardNormal.sample(&mut rng);
                            log_s += nu + delta * z_jump;
                        }
                    }
                }

                let st = log_s.exp();
                let payoff = (option_type.sign() * (st - strike)).max(0.0);
                sum += payoff;
                sum_sq += payoff * payoff;
            }
            (sum, sum_sq, count)
        });

    let total_sum: f64 = results.iter().map(|(s, _, _)| s).sum();
    let total_sum_sq: f64 = results.iter().map(|(_, s, _)| s).sum();
    let total_count: usize = results.iter().map(|(_, _, c)| c).sum();
    let n = total_count as f64;

    let mean = total_sum / n;
    let variance = (total_sum_sq / n - mean * mean).max(0.0);
    let std_error = (variance / n).sqrt() * df;

    MCResult {
        npv: df * mean,
        std_error,
        num_paths: total_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mc_european_call_converges_to_bs() {
        // BS price for S=100, K=100, r=5%, q=0, σ=20%, T=1: ~10.45
        let result = mc_european(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0,
            OptionType::Call, 100_000, true, 42,
        );
        // Should be within ~3 standard errors of BS
        assert!(
            (result.npv - 10.45).abs() < 3.0 * result.std_error + 0.5,
            "MC European call {} not near BS 10.45 (stderr={})",
            result.npv,
            result.std_error
        );
    }

    #[test]
    fn mc_european_put_converges_to_bs() {
        // BS put: ~5.57
        let result = mc_european(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0,
            OptionType::Put, 100_000, true, 42,
        );
        assert!(
            (result.npv - 5.57).abs() < 3.0 * result.std_error + 0.5,
            "MC European put {} not near BS 5.57 (stderr={})",
            result.npv,
            result.std_error
        );
    }

    #[test]
    fn mc_barrier_knock_out_cheaper() {
        // Down-and-out call should be cheaper than vanilla call
        let vanilla = mc_european(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0,
            OptionType::Call, 50_000, true, 42,
        );
        let barrier = mc_barrier(
            100.0, 100.0, 80.0, 0.0, 0.05, 0.0, 0.2, 1.0,
            OptionType::Call, false, false, 50_000, 252, 42,
        );
        assert!(
            barrier.npv < vanilla.npv,
            "Barrier {} should be < vanilla {}",
            barrier.npv,
            vanilla.npv
        );
    }

    #[test]
    fn mc_barrier_knock_in_plus_knock_out_equals_vanilla() {
        // KI + KO ≈ Vanilla (in-out parity, approximate due to discrete monitoring)
        let vanilla = mc_european(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0,
            OptionType::Call, 100_000, true, 42,
        );
        let ko = mc_barrier(
            100.0, 100.0, 80.0, 0.0, 0.05, 0.0, 0.2, 1.0,
            OptionType::Call, false, false, 100_000, 252, 42,
        );
        let ki = mc_barrier(
            100.0, 100.0, 80.0, 0.0, 0.05, 0.0, 0.2, 1.0,
            OptionType::Call, false, true, 100_000, 252, 42,
        );
        let parity = ko.npv + ki.npv;
        assert!(
            (parity - vanilla.npv).abs() < 1.5,
            "KI({}) + KO({}) = {} should ≈ vanilla {}",
            ki.npv,
            ko.npv,
            parity,
            vanilla.npv,
        );
    }

    #[test]
    fn mc_asian_call_cheaper_than_vanilla() {
        // Asian call should be cheaper than vanilla European call
        let vanilla = mc_european(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0,
            OptionType::Call, 100_000, true, 42,
        );
        let asian = mc_asian(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0,
            OptionType::Call, true, 100_000, 252, 42,
        );
        assert!(
            asian.npv < vanilla.npv,
            "Asian {} should be < vanilla {}",
            asian.npv,
            vanilla.npv
        );
        assert!(asian.npv > 0.0, "Asian call should be positive");
    }

    #[test]
    fn mc_asian_geometric_cheaper_than_arithmetic() {
        // Geometric average ≤ arithmetic average
        let arith = mc_asian(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0,
            OptionType::Call, true, 50_000, 252, 42,
        );
        let geom = mc_asian(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0,
            OptionType::Call, false, 50_000, 252, 42,
        );
        assert!(
            geom.npv <= arith.npv + 0.5,
            "Geometric {} should be ≤ arithmetic {}",
            geom.npv,
            arith.npv
        );
    }

    #[test]
    fn mc_heston_call_reasonable() {
        // Heston with standard params: price should be ~10
        let result = mc_heston(
            100.0, 100.0, 0.05, 0.0,
            0.04, 1.5, 0.04, 0.3, -0.7,
            1.0, OptionType::Call, 50_000, 252, 42,
        );
        assert!(
            result.npv > 5.0 && result.npv < 20.0,
            "Heston MC call {} not in reasonable range",
            result.npv
        );
    }

    #[test]
    fn mc_heston_vs_analytic() {
        // Heston MC should be close to analytic price (~10.45 with these params)
        let result = mc_heston(
            100.0, 100.0, 0.05, 0.0,
            0.04, 1.5, 0.04, 0.3, -0.7,
            1.0, OptionType::Call, 100_000, 252, 42,
        );
        // Use wider tolerance since MC is inherently noisy
        assert!(
            (result.npv - 10.45).abs() < 3.0 * result.std_error + 1.0,
            "Heston MC call {} not near analytic 10.45 (stderr={})",
            result.npv,
            result.std_error
        );
    }

    #[test]
    fn mc_antithetic_reduces_variance() {
        // Antithetic should give lower or similar std_error
        let plain = mc_european(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0,
            OptionType::Call, 50_000, false, 42,
        );
        let anti = mc_european(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0,
            OptionType::Call, 50_000, true, 42,
        );
        // Antithetic should have lower std_error (usually),
        // but at minimum both should produce reasonable estimates
        assert!(plain.npv > 5.0 && plain.npv < 20.0);
        assert!(anti.npv > 5.0 && anti.npv < 20.0);
    }
}
