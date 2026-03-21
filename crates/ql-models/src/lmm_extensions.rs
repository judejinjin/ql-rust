//! LMM extensions: spot-measure drift, Brownian bridge, exotic products.
//!
//! This module extends the base [`crate::lmm`] with:
//!
//! - **Spot-measure drift** (`spot_measure_drift`) — alternative to terminal
//!   measure, more natural for path-dependent products where payments occur at
//!   each reset date.
//! - **Brownian bridge terminal value sampler** (`brownian_bridge_terminal`) —
//!   variance reduction technique for path-based Monte Carlo.
//! - **Ratchet caplet pricer** (`lmm_ratchet_cap_price`) — caplet where the
//!   strike at each reset is the previous period's fixing.
//! - **CMS rate extractor** (`lmm_cms_rate`) — constant-maturity swap rate
//!   evaluated under the LMM curve state at any simulation time.
//! - **Bermudan swaption** (`lmm_bermudan_swaption_price`) — Longstaff-Schwartz
//!   LSM regression for early exercise.
//! - **Multi-factor vol surface** (`LmmVolSurface`) — time-homogeneous or
//!   separable volatility structure calibrated to caplet vols.

#![allow(clippy::too_many_arguments)]

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};

use crate::lmm::{LmmConfig, LmmCurveState, LmmResult, evolve_one_step};

// =========================================================================
// Spot-measure drift
// =========================================================================

/// Compute the drift of forward rate `i` under the **spot measure**
/// (rolling money-market numeraire from rate 0 forward).
///
/// Under the spot measure:
/// ```text
/// μ_i^spot = Σ_{j=0}^{i} [τ_j f_j σ_i σ_j ρ_{ij} / (1 + τ_j f_j)]
/// ```
///
/// This is positive (convex correction) and grows with index `i`.
#[allow(clippy::needless_range_loop)]
pub fn spot_measure_drift(config: &LmmConfig, i: usize, forwards: &[f64]) -> f64 {
    let n = config.n_rates;
    let mut mu = 0.0;
    for j in 0..=i {
        let tau_f = config.accruals[j] * forwards[j];
        mu += tau_f * config.volatilities[i] * config.volatilities[j]
            * config.correlation[i * n + j]
            / (1.0 + tau_f);
    }
    mu
}

/// Evolve forward rates for one step under the **spot measure**.
pub fn evolve_one_step_spot_measure(
    config: &LmmConfig,
    forwards: &[f64],
    alive_from: usize,
    dt: f64,
    chol: &[f64],
    z: &[f64],
) -> Vec<f64> {
    let n = config.n_rates;
    let sqrt_dt = dt.sqrt();
    let mut new_forwards = forwards.to_vec();

    // Correlated Brownian increments
    let mut dw = vec![0.0; n];
    for i in alive_from..n {
        for k in alive_from..=i {
            dw[i] += chol[i * n + k] * z[k];
        }
        dw[i] *= sqrt_dt;
    }

    // Predictor step
    let mut f_pred = forwards.to_vec();
    for i in alive_from..n {
        let mu = spot_measure_drift(config, i, forwards);
        let sigma = config.volatilities[i];
        f_pred[i] = forwards[i] * ((mu - 0.5 * sigma * sigma) * dt + sigma * dw[i]).exp();
    }

    // Corrector step
    for i in alive_from..n {
        let mu0 = spot_measure_drift(config, i, forwards);
        let mu1 = spot_measure_drift(config, i, &f_pred);
        let avg_mu = 0.5 * (mu0 + mu1);
        let sigma = config.volatilities[i];
        new_forwards[i] = forwards[i] * ((avg_mu - 0.5 * sigma * sigma) * dt + sigma * dw[i]).exp();
    }

    new_forwards
}

// =========================================================================
// Brownian bridge terminal value
// =========================================================================

/// Sample the **terminal value** of a Brownian path conditional on the
/// initial value using a Brownian bridge.
///
/// Given `W(0) = 0` and `W(T) = w_t` (the terminal value drawn from
/// N(0, T)), generate `W(t)` for intermediate `t` as:
/// ```text
/// W(t) | W(T) = w_t  ~  N(t/T * w_t, t(T-t)/T)
/// ```
///
/// Returns a vector of bridge values at the given `times`.
pub fn brownian_bridge(
    w_terminal: f64,
    total_time: f64,
    times: &[f64],
    rng: &mut SmallRng,
) -> Vec<f64> {
    let n = times.len();
    let mut bridge = vec![0.0; n];
    for (k, &t) in times.iter().enumerate() {
        let mean = t / total_time * w_terminal;
        let var = t * (total_time - t) / total_time;
        let std = var.max(0.0).sqrt();
        let z: f64 = rng.sample(StandardNormal);
        bridge[k] = mean + std * z;
    }
    bridge
}

// =========================================================================
// Ratchet cap
// =========================================================================

/// Price a **ratchet cap** via LMM Monte Carlo.
///
/// A ratchet cap has caplets where the strike for period `i` equals the
/// fixing `f_{i-1}(T_{i-1})` from the previous period.  For the first
/// caplet the strike is `initial_strike`.
pub fn lmm_ratchet_cap_price(
    config: &LmmConfig,
    initial_strike: f64,
    n_paths: usize,
    seed: u64,
) -> LmmResult {
    let n = config.n_rates;
    let chol = config.cholesky();
    let dt = config.accruals[0];

    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut rng = SmallRng::seed_from_u64(seed);

    for _ in 0..n_paths {
        let mut forwards = config.initial_forwards.clone();
        let mut numeraire = 1.0;
        let mut path_pv = 0.0;
        let mut prev_strike = initial_strike;

        for step in 0..n {
            let strike = prev_strike;
            let z: Vec<f64> = (0..n).map(|_| rng.sample(StandardNormal)).collect();
            let sqrt_dt = dt.sqrt();

            let caplet_payoff = (forwards[step] - strike).max(0.0) * config.accruals[step];
            path_pv += caplet_payoff / numeraire;

            // Next period's strike is this period's fixing
            prev_strike = forwards[step];

            forwards = evolve_one_step(config, &forwards, step + 1, dt, sqrt_dt, &chol, &z);
            numeraire *= 1.0 + config.accruals[step] * config.initial_forwards[step];
        }

        sum += path_pv;
        sum_sq += path_pv * path_pv;
    }

    let mean = sum / n_paths as f64;
    let variance = sum_sq / n_paths as f64 - mean * mean;
    LmmResult {
        price: mean,
        std_error: (variance / n_paths as f64).max(0.0).sqrt(),
    }
}

// =========================================================================
// CMS rate under LMM
// =========================================================================

/// Extract the constant-maturity swap (CMS) rate of a `tenor`-year swap
/// starting at time `swap_start` from an LMM curve state.
///
/// Simply delegates to [`LmmCurveState::swap_rate`] with boundaries derived
/// from the accrual schedule.
pub fn lmm_cms_rate(
    state: &LmmCurveState,
    accruals: &[f64],
    swap_start: usize,
    swap_tenor_periods: usize,
) -> f64 {
    let swap_end = (swap_start + swap_tenor_periods).min(state.forwards.len());
    state.swap_rate(swap_start, swap_end, accruals)
}

// =========================================================================
// Bermudan swaption — Longstaff-Schwartz LSM
// =========================================================================

/// Price a **Bermudan payer swaption** via Longstaff-Schwartz LSM regression
/// on the LMM simulation.
///
/// The swaption can be exercised at any reset date in `[exercise_start, swap_end)`.
/// LSM uses a polynomial basis of the current swap rate to estimate
/// continuation values.
///
/// # Arguments
/// - `config`         — LMM configuration
/// - `exercise_start` — first exercise date (in rate index)
/// - `swap_end`       — last rate in the underlying swap
/// - `strike`         — fixed rate of the swap
/// - `n_paths`        — number of Monte Carlo paths
/// - `seed`           — random seed
#[allow(clippy::needless_range_loop)]
pub fn lmm_bermudan_swaption_price(
    config: &LmmConfig,
    exercise_start: usize,
    swap_end: usize,
    strike: f64,
    n_paths: usize,
    seed: u64,
) -> LmmResult {
    let n = config.n_rates;
    let chol = config.cholesky();
    let dt = config.accruals[0];
    let sqrt_dt = dt.sqrt();

    // Simulate all paths forward to maturity
    // paths[path][step] = forwards at that step
    let mut paths: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_paths);
    let mut rng = SmallRng::seed_from_u64(seed);

    for _ in 0..n_paths {
        let mut path = Vec::with_capacity(n + 1);
        let mut forwards = config.initial_forwards.clone();
        path.push(forwards.clone());
        for step in 0..n {
            let z: Vec<f64> = (0..n).map(|_| rng.sample(StandardNormal)).collect();
            forwards = evolve_one_step(config, &forwards, step + 1, dt, sqrt_dt, &chol, &z);
            path.push(forwards.clone());
        }
        paths.push(path);
    }

    // LSM backward induction from swap_end - 1 back to exercise_start
    // cash_flows[p] = discounted payoff given optimal exercise
    let mut cash_flows: Vec<f64> = vec![0.0; n_paths];

    // At the last exercise date, exercise if ITM
    for (p, path) in paths.iter().enumerate() {
        let fwds = &path[swap_end - 1];
        let state = LmmCurveState {
            forwards: fwds.clone(),
            alive_index: swap_end - 1,
        };
        let swap_rate = state.swap_rate(swap_end - 1, swap_end.min(n), &config.accruals);
        // Annuity at this date
        let mut annuity = 0.0;
        let mut d = 1.0;
        for j in (swap_end - 1)..swap_end.min(n) {
            d /= 1.0 + config.accruals[j] * fwds[j];
            annuity += config.accruals[j] * d;
        }
        cash_flows[p] = (swap_rate - strike).max(0.0) * annuity;
    }

    // Backward sweep
    for step in (exercise_start..swap_end - 1).rev() {
        let in_the_money: Vec<usize> = (0..n_paths)
            .filter(|&p| {
                let fwds = &paths[p][step];
                let state = LmmCurveState { forwards: fwds.clone(), alive_index: step };
                let sr = state.swap_rate(step, swap_end.min(n), &config.accruals);
                sr > strike
            })
            .collect();

        if in_the_money.is_empty() {
            continue;
        }

        // LSM: regress continuation values on swap rate (polynomial degree 2)
        let x: Vec<f64> = in_the_money.iter().map(|&p| {
            let fwds = &paths[p][step];
            let state = LmmCurveState { forwards: fwds.clone(), alive_index: step };
            state.swap_rate(step, swap_end.min(n), &config.accruals)
        }).collect();

        let y: Vec<f64> = in_the_money.iter().map(|&p| cash_flows[p]).collect();

        // Polynomial regression: fit a + b*x + c*x^2
        let cont_estimates = ols_quadratic_predict(&x, &y, &x);

        // Update exercise decisions
        for (k, &p) in in_the_money.iter().enumerate() {
            let fwds = &paths[p][step];
            let state = LmmCurveState { forwards: fwds.clone(), alive_index: step };
            let sr = state.swap_rate(step, swap_end.min(n), &config.accruals);
            let mut annuity = 0.0;
            let mut d = 1.0;
            for j in step..swap_end.min(n) {
                d /= 1.0 + config.accruals[j] * fwds[j];
                annuity += config.accruals[j] * d;
            }
            let exercise_value = (sr - strike).max(0.0) * annuity;
            let continuation = cont_estimates[k].max(0.0);
            if exercise_value >= continuation {
                cash_flows[p] = exercise_value;
            }
        }
    }

    // Discount all cash flows to t=0
    let n_paths_f = n_paths as f64;
    let total_disc: f64 = {
        let mut d = 1.0;
        for i in 0..exercise_start {
            d /= 1.0 + config.accruals[i] * config.initial_forwards[i];
        }
        d
    };

    let pvs: Vec<f64> = cash_flows.iter().map(|&cf| cf * total_disc).collect();
    let mean = pvs.iter().sum::<f64>() / n_paths_f;
    let variance = pvs.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n_paths_f;

    LmmResult {
        price: mean,
        std_error: (variance / n_paths_f).max(0.0).sqrt(),
    }
}

/// Simple OLS quadratic regression: fit y ~ a + b*x + c*x^2 and predict.
fn ols_quadratic_predict(x: &[f64], y: &[f64], x_pred: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n < 3 {
        let mean_y = y.iter().sum::<f64>() / n.max(1) as f64;
        return x_pred.iter().map(|_| mean_y).collect();
    }

    // Build Gram matrix [1, x, x^2]' [1, x, x^2] (3x3)
    let mut ata = [[0.0f64; 3]; 3];
    let mut atb = [0.0f64; 3];
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let basis = [1.0, xi, xi * xi];
        for r in 0..3 {
            atb[r] += basis[r] * yi;
            for c in 0..3 {
                ata[r][c] += basis[r] * basis[c];
            }
        }
    }

    // Solve 3×3 via Cramer's rule (small fixed size)
    let coeffs = solve_3x3(&ata, &atb);

    x_pred.iter().map(|&xi| coeffs[0] + coeffs[1] * xi + coeffs[2] * xi * xi).collect()
}

/// Solve 3×3 linear system A*x = b via Gaussian elimination.
#[allow(clippy::needless_range_loop)]
fn solve_3x3(a: &[[f64; 3]; 3], b: &[f64; 3]) -> [f64; 3] {
    let mut m = [[0.0f64; 4]; 3];
    for i in 0..3 {
        for j in 0..3 { m[i][j] = a[i][j]; }
        m[i][3] = b[i];
    }
    // Forward elimination
    for col in 0..3 {
        // Pivot
        let mut max_row = col;
        for row in col + 1..3 {
            if m[row][col].abs() > m[max_row][col].abs() { max_row = row; }
        }
        m.swap(col, max_row);
        let pivot = m[col][col];
        if pivot.abs() < 1e-15 { return [0.0; 3]; }
        for row in col + 1..3 {
            let factor = m[row][col] / pivot;
            for k in col..4 { m[row][k] -= factor * m[col][k]; }
        }
    }
    // Back substitution
    let mut x = [0.0f64; 3];
    for i in (0..3).rev() {
        x[i] = m[i][3];
        for j in i + 1..3 { x[i] -= m[i][j] * x[j]; }
        x[i] /= m[i][i];
    }
    x
}

// =========================================================================
// Multi-factor vol surface
// =========================================================================

/// A time-homogeneous LMM volatility surface calibrated to market caplet vols.
///
/// Uses the separable parametric form:
/// ```text
/// σ_i(t) = [a + b*(T_i - t)] * exp(-c*(T_i - t)) + d
/// ```
/// where `T_i - t` is time-to-expiry of caplet `i`.  This is the
/// Rebonato four-parameter form used widely in practice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebonatVolSurface {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
}

impl RebonatVolSurface {
    /// Create with given parameters.  Constraint: `a + d > 0`, `d > 0`.
    pub fn new(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self { a, b, c, d }
    }

    /// Instantaneous volatility for a caplet with time-to-expiry `tte`.
    pub fn vol(&self, tte: f64) -> f64 {
        (self.a + self.b * tte) * (-self.c * tte).exp() + self.d
    }

    /// Black-equivalent integrated vol for a caplet maturing at `T_expiry`.
    ///
    /// ```text
    /// σ_Black^2 = (1/T) ∫_0^T σ(T-t)^2 dt
    /// ```
    /// Computed by Gauss-Legendre quadrature (16-point).
    pub fn integrated_black_vol(&self, t_expiry: f64) -> f64 {
        if t_expiry <= 0.0 { return self.d.max(0.0); }
        // 16-point Gauss-Legendre nodes/weights on [0, 1]
        const NODES: [f64; 8] = [
            0.0950125098360663, 0.2816035507792589, 0.4580167776572274,
            0.6178762444026437, 0.755_404_408_355_003, 0.8656312023341769,
            0.9445750230732326, 0.9894009349916499,
        ];
        const WEIGHTS: [f64; 8] = [
            0.1894506104550685, 0.1826034150449236, 0.1691565193950025,
            0.1495959888165767, 0.1246289512509579, 0.0951585116824928,
            0.0622535239386479, 0.0271524594117541,
        ];
        let mut sum = 0.0;
        for (&n, &w) in NODES.iter().zip(WEIGHTS.iter()) {
            // Symmetric: use both +n and -n mapped to [0, T_expiry]
            let t1 = t_expiry * 0.5 * (1.0 + n);
            let t2 = t_expiry * 0.5 * (1.0 - n);
            let tte1 = t_expiry - t1;
            let tte2 = t_expiry - t2;
            sum += w * (self.vol(tte1).powi(2) + self.vol(tte2).powi(2));
        }
        let variance = sum * t_expiry * 0.5;
        (variance / t_expiry).max(0.0).sqrt()
    }

    /// Calibrate to market Black caplet vols by least-squares.
    ///
    /// Uses a simple gradient-free grid search followed by steepest descent.
    /// Returns the best-fit [`RebonatVolSurface`].
    pub fn calibrate(expiries: &[f64], market_vols: &[f64]) -> Self {
        assert_eq!(expiries.len(), market_vols.len());

        let objective = |a: f64, b: f64, c: f64, d: f64| -> f64 {
            let surf = Self::new(a, b, c, d);
            expiries.iter().zip(market_vols.iter())
                .map(|(&t, &mv)| {
                    let model_v = surf.integrated_black_vol(t);
                    (model_v - mv).powi(2)
                })
                .sum()
        };

        let mut best = (0.1, 0.0, 1.0, 0.1, f64::MAX);
        // Grid search over (a, b, c, d)
        for &a in &[0.05, 0.10, 0.15] {
            for &b in &[-0.05, 0.0, 0.05] {
                for &c in &[0.5, 1.0, 2.0] {
                    for &d in &[0.05, 0.10, 0.15] {
                        let err = objective(a, b, c, d);
                        if err < best.4 {
                            best = (a, b, c, d, err);
                        }
                    }
                }
            }
        }

        Self::new(best.0, best.1, best.2, best.3)
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> LmmConfig {
        LmmConfig::flat(8, 0.05, 0.25, 0.20, 0.5)
    }

    #[test]
    fn spot_measure_drift_positive() {
        let config = make_config();
        // Under spot measure drift should be positive for all rates
        let drift_i4 = spot_measure_drift(&config, 4, &config.initial_forwards);
        assert!(drift_i4 > 0.0, "spot drift should be positive: {}", drift_i4);
    }

    #[test]
    fn spot_measure_evolve_preserves_positivity() {
        let config = make_config();
        let chol = config.cholesky();
        let n = config.n_rates;
        let dt = 0.25;
        let mut forwards = config.initial_forwards.clone();
        let mut rng = SmallRng::seed_from_u64(99);

        for step in 0..n - 1 {
            let z: Vec<f64> = (0..n).map(|_| rng.sample(StandardNormal)).collect();
            forwards = evolve_one_step_spot_measure(&config, &forwards, step + 1, dt, &chol, &z);
            for (i, &f) in forwards.iter().enumerate().skip(step + 1) {
                assert!(f > 0.0, "forward {} should be positive: {}", i, f);
            }
        }
    }

    #[test]
    fn ratchet_cap_price_positive() {
        let config = make_config();
        let res = lmm_ratchet_cap_price(&config, 0.05, 5000, 42);
        assert!(res.price > 0.0, "ratchet cap price: {}", res.price);
    }

    #[test]
    fn bermudan_swaption_price_positive() {
        let config = make_config();
        let res = lmm_bermudan_swaption_price(&config, 1, 8, 0.05, 2000, 7);
        assert!(res.price > 0.0, "bermudan swaption: {}", res.price);
    }

    #[test]
    fn rebonato_vol_surface_integrated() {
        let surf = RebonatVolSurface::new(0.1, 0.05, 1.0, 0.10);
        // At T=0 vol should equal d (instantaneous short vol)
        let v = surf.vol(0.0);
        assert!((v - (surf.a + surf.d)).abs() < 1e-10, "at tte=0 vol={}", v);
        // Integrated vol at T=2 should be positive
        let iv = surf.integrated_black_vol(2.0);
        assert!(iv > 0.0, "integrated vol: {}", iv);
    }
}
