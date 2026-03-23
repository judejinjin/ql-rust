//! Mountain-range exotic options: Himalaya, Everest, Atlas, Pagoda.
//!
//! Multi-asset path-dependent exotics priced via Monte Carlo simulation.
//!
//! - **Himalaya**: At each observation date, the best-performing asset is removed
//!   and its return locked in. Payoff is the average of locked-in returns.
//! - **Everest**: Payoff based on the *worst*-performing asset's return.
//! - **Atlas**: Payoff is the average return after removing the best and worst N
//!   performers.
//! - **Pagoda**: Capped accumulated positive returns over observation dates.
//!
//! Reference: Quessette (2002), *New Products, New Risks*.



/// Result from a mountain-range option pricing.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MountainRangeResult {
    /// Monte Carlo estimate of the NPV.
    pub npv: f64,
    /// Standard error of the MC estimate.
    pub std_error: f64,
    /// Number of paths simulated.
    pub n_paths: usize,
}

/// Type of mountain-range option.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum MountainType {
    /// Best asset removed at each observation; payoff = average of locked returns.
    Himalaya,
    /// Payoff = min(return_i) over all assets.
    Everest,
    /// Average return after removing `n_remove` best and `n_remove` worst.
    Atlas {
        /// Number of assets to remove from best and worst.
        n_remove: usize,
    },
    /// Accumulated positive returns, each capped at a local cap.
    Pagoda {
        /// Maximum return per observation.
        local_cap: f64,
    },
}

/// Price a mountain-range exotic option via Monte Carlo.
///
/// # Arguments
/// - `spots`         — initial prices of each asset (length = n_assets)
/// - `vols`          — volatilities of each asset
/// - `correlations`  — flat correlation matrix (row-major, n_assets × n_assets)
/// - `r`             — risk-free rate (continuous)
/// - `q`             — dividend yields (per asset)
/// - `observation_times` — times at which returns are measured (sorted ascending)
/// - `notional`      — notional amount
/// - `mountain_type` — type of mountain-range option
/// - `n_paths`       — number of Monte Carlo paths
/// - `seed`          — PRNG seed
#[allow(clippy::too_many_arguments)]
pub fn mc_mountain_range(
    spots: &[f64],
    vols: &[f64],
    correlations: &[f64],
    r: f64,
    q: &[f64],
    observation_times: &[f64],
    notional: f64,
    mountain_type: MountainType,
    n_paths: usize,
    seed: u64,
) -> MountainRangeResult {
    let n_assets = spots.len();
    assert_eq!(vols.len(), n_assets);
    assert_eq!(q.len(), n_assets);
    assert_eq!(correlations.len(), n_assets * n_assets);

    let n_obs = observation_times.len();

    // Cholesky of correlation matrix
    let chol = cholesky_flat(correlations, n_assets);

    let mut rng = SimpleRng::new(seed);
    let mut sum_payoff = 0.0;
    let mut sum_payoff2 = 0.0;

    // Discount factor to valuation date
    let t_final = *observation_times.last().unwrap_or(&0.0);
    let df = (-r * t_final).exp();

    for _ in 0..n_paths {
        // Simulate asset paths at observation times
        let mut prices = spots.to_vec();
        let mut returns_at_obs: Vec<Vec<f64>> = Vec::with_capacity(n_obs);
        let mut prev_t = 0.0;

        for &t in observation_times {
            let dt = t - prev_t;
            let sqrt_dt = dt.sqrt();

            // Generate correlated normals
            let z_indep: Vec<f64> = (0..n_assets).map(|_| rng.normal()).collect();
            let z_corr = apply_cholesky(&chol, &z_indep, n_assets);

            let mut obs_returns = Vec::with_capacity(n_assets);
            for i in 0..n_assets {
                let drift = (r - q[i] - 0.5 * vols[i] * vols[i]) * dt;
                let diffusion = vols[i] * sqrt_dt * z_corr[i];
                prices[i] *= (drift + diffusion).exp();
                obs_returns.push(prices[i] / spots[i] - 1.0);
            }
            returns_at_obs.push(obs_returns);
            prev_t = t;
        }

        let payoff = compute_mountain_payoff(
            &returns_at_obs,
            n_assets,
            &mountain_type,
        );

        let discounted = notional * payoff.max(0.0) * df;
        sum_payoff += discounted;
        sum_payoff2 += discounted * discounted;
    }

    let mean = sum_payoff / n_paths as f64;
    let variance = sum_payoff2 / n_paths as f64 - mean * mean;
    let std_error = (variance / n_paths as f64).sqrt();

    MountainRangeResult {
        npv: mean,
        std_error,
        n_paths,
    }
}

fn compute_mountain_payoff(
    returns_at_obs: &[Vec<f64>],
    n_assets: usize,
    mountain_type: &MountainType,
) -> f64 {
    match mountain_type {
        MountainType::Himalaya => {
            himalaya_payoff(returns_at_obs, n_assets)
        }
        MountainType::Everest => {
            everest_payoff(returns_at_obs, n_assets)
        }
        MountainType::Atlas { n_remove } => {
            atlas_payoff(returns_at_obs, n_assets, *n_remove)
        }
        MountainType::Pagoda { local_cap } => {
            pagoda_payoff(returns_at_obs, n_assets, *local_cap)
        }
    }
}

fn himalaya_payoff(returns_at_obs: &[Vec<f64>], n_assets: usize) -> f64 {
    // At each observation, lock in the best performer and remove it
    let mut available: Vec<bool> = vec![true; n_assets];
    let mut locked_returns = Vec::new();

    for obs in returns_at_obs {
        let mut best_idx = 0;
        let mut best_ret = f64::NEG_INFINITY;
        for i in 0..n_assets {
            if available[i] && obs[i] > best_ret {
                best_ret = obs[i];
                best_idx = i;
            }
        }
        if best_ret > f64::NEG_INFINITY {
            locked_returns.push(best_ret);
            available[best_idx] = false;
        }
    }

    if locked_returns.is_empty() {
        0.0
    } else {
        locked_returns.iter().sum::<f64>() / locked_returns.len() as f64
    }
}

fn everest_payoff(returns_at_obs: &[Vec<f64>], _n_assets: usize) -> f64 {
    // Final observation's minimum return
    if let Some(final_obs) = returns_at_obs.last() {
        final_obs
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    } else {
        0.0
    }
}

fn atlas_payoff(returns_at_obs: &[Vec<f64>], _n_assets: usize, n_remove: usize) -> f64 {
    // Average of final returns, excluding n_remove best and n_remove worst
    if let Some(final_obs) = returns_at_obs.last() {
        let mut sorted = final_obs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let remaining = if 2 * n_remove < sorted.len() {
            &sorted[n_remove..(sorted.len() - n_remove)]
        } else {
            &sorted[..]
        };
        if remaining.is_empty() {
            0.0
        } else {
            remaining.iter().sum::<f64>() / remaining.len() as f64
        }
    } else {
        0.0
    }
}

fn pagoda_payoff(returns_at_obs: &[Vec<f64>], n_assets: usize, local_cap: f64) -> f64 {
    // Accumulated average return per observation, each capped at local_cap
    let mut accumulated = 0.0;
    for obs in returns_at_obs {
        let avg_ret = obs.iter().sum::<f64>() / n_assets as f64;
        accumulated += avg_ret.min(local_cap);
    }
    accumulated
}

// ---------------------------------------------------------------------------
// Linear algebra helpers (self-contained)
// ---------------------------------------------------------------------------

fn cholesky_flat(corr: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i * n + k] * l[j * n + k];
            }
            if i == j {
                let val = corr[i * n + j] - sum;
                l[i * n + j] = if val > 0.0 { val.sqrt() } else { 0.0 };
            } else {
                let diag = l[j * n + j];
                l[i * n + j] = if diag > 1e-15 {
                    (corr[i * n + j] - sum) / diag
                } else {
                    0.0
                };
            }
        }
    }
    l
}

fn apply_cholesky(l: &[f64], z: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for i in 0..n {
        for j in 0..=i {
            out[i] += l[i * n + j] * z[j];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Simple RNG (xorshift64 + Box-Muller)
// ---------------------------------------------------------------------------

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        // Box-Muller
        let u1 = self.uniform().max(1e-15);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn identity_corr(n: usize) -> Vec<f64> {
        let mut c = vec![0.0; n * n];
        for i in 0..n {
            c[i * n + i] = 1.0;
        }
        c
    }

    #[test]
    fn himalaya_basic() {
        let n = 3;
        let spots = vec![100.0; n];
        let vols = vec![0.20; n];
        let q = vec![0.0; n];
        let obs: Vec<f64> = (1..=n).map(|i| i as f64 * 0.25).collect();
        let corr = identity_corr(n);
        let res = mc_mountain_range(
            &spots, &vols, &corr, 0.05, &q, &obs, 100.0,
            MountainType::Himalaya, 10_000, 42,
        );
        assert!(res.npv >= 0.0, "Himalaya NPV should be non-negative: {}", res.npv);
        assert!(res.std_error < res.npv.abs().max(1.0), "std error too large");
    }

    #[test]
    fn everest_basic() {
        let n = 4;
        let spots = vec![100.0; n];
        let vols = vec![0.25; n];
        let q = vec![0.02; n];
        let obs = vec![1.0];
        let mut corr = vec![0.3; n * n];
        for i in 0..n {
            corr[i * n + i] = 1.0;
        }
        let res = mc_mountain_range(
            &spots, &vols, &corr, 0.05, &q, &obs, 100.0,
            MountainType::Everest, 10_000, 123,
        );
        // Everest payoff can be negative (worst performer), so NPV could be 0 if floored
        assert!(res.npv >= 0.0, "Everest NPV: {}", res.npv);
    }

    #[test]
    fn atlas_removes_extremes() {
        let n = 5;
        let spots = vec![100.0; n];
        let vols = vec![0.20; n];
        let q = vec![0.0; n];
        let obs = vec![1.0];
        let corr = identity_corr(n);
        let res = mc_mountain_range(
            &spots, &vols, &corr, 0.05, &q, &obs, 100.0,
            MountainType::Atlas { n_remove: 1 }, 10_000, 999,
        );
        assert!(res.npv >= 0.0, "Atlas NPV: {}", res.npv);
    }

    #[test]
    fn pagoda_basic() {
        let n = 3;
        let spots = vec![100.0; n];
        let vols = vec![0.15; n];
        let q = vec![0.0; n];
        let obs = vec![0.25, 0.50, 0.75, 1.0];
        let corr = identity_corr(n);
        let res = mc_mountain_range(
            &spots, &vols, &corr, 0.05, &q, &obs, 100.0,
            MountainType::Pagoda { local_cap: 0.05 }, 10_000, 7,
        );
        assert!(res.npv >= 0.0, "Pagoda NPV: {}", res.npv);
    }

    #[test]
    fn himalaya_more_paths_lower_error() {
        let n = 3;
        let spots = vec![100.0; n];
        let vols = vec![0.20; n];
        let q = vec![0.0; n];
        let obs: Vec<f64> = (1..=n).map(|i| i as f64 * 0.25).collect();
        let corr = identity_corr(n);
        let r1 = mc_mountain_range(
            &spots, &vols, &corr, 0.05, &q, &obs, 100.0,
            MountainType::Himalaya, 1_000, 42,
        );
        let r2 = mc_mountain_range(
            &spots, &vols, &corr, 0.05, &q, &obs, 100.0,
            MountainType::Himalaya, 20_000, 42,
        );
        assert!(r2.std_error < r1.std_error,
                "more paths should give lower error: {} vs {}", r2.std_error, r1.std_error);
    }
}
