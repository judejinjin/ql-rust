#![allow(clippy::too_many_arguments)]
//! LIBOR Market Model (BGM) framework.
//!
//! Implements the log-normal forward rate model (Brace-Gatarek-Musiela)
//! with predictor-corrector time stepping, drift correction,
//! and pricing of caps, swaptions, and multi-step products.
//!
//! Key types:
//! - `LmmCurveState` — state of forward rates at a given time
//! - `LmmEvolver` — predictor-corrector evolution of forward rates
//! - `lmm_cap_price` — cap pricing via LMM Monte Carlo
//! - `lmm_swaption_price` — swaption pricing via LMM Monte Carlo

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

/// Configuration for an LMM simulation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LmmConfig {
    /// Number of forward rates.
    pub n_rates: usize,
    /// Accrual fractions (τ_i) for each period. Length = n_rates.
    pub accruals: Vec<f64>,
    /// Initial forward rates f_i(0). Length = n_rates.
    pub initial_forwards: Vec<f64>,
    /// Instantaneous volatilities σ_i. Length = n_rates.
    pub volatilities: Vec<f64>,
    /// Correlation matrix (n_rates × n_rates), stored row-major.
    pub correlation: Vec<f64>,
}

impl LmmConfig {
    /// Create with flat vol and exponential correlation.
    pub fn flat(
        n_rates: usize,
        forward: f64,
        accrual: f64,
        vol: f64,
        corr_decay: f64,
    ) -> Self {
        let accruals = vec![accrual; n_rates];
        let initial_forwards = vec![forward; n_rates];
        let volatilities = vec![vol; n_rates];

        let mut correlation = vec![0.0; n_rates * n_rates];
        for i in 0..n_rates {
            for j in 0..n_rates {
                correlation[i * n_rates + j] =
                    (-corr_decay * (i as f64 - j as f64).abs()).exp();
            }
        }

        Self {
            n_rates,
            accruals,
            initial_forwards,
            volatilities,
            correlation,
        }
    }

    /// Compute the drift correction for forward rate i in the forward measure
    /// (terminal measure at the last rate's maturity).
    ///
    /// Under the terminal measure:
    ///   μ_i = - Σ_{j=i+1}^{N-1} [τ_j f_j σ_i σ_j ρ_{ij} / (1 + τ_j f_j)]
    fn drift(&self, i: usize, forwards: &[f64]) -> f64 {
        let n = self.n_rates;
        let mut mu = 0.0;
        for (j, (&accrual_j, &fwd_j)) in self.accruals.iter().zip(forwards.iter()).enumerate().skip(i + 1) {
            let tau_f = accrual_j * fwd_j;
            mu -= tau_f * self.volatilities[i] * self.volatilities[j]
                * self.correlation[i * n + j]
                / (1.0 + tau_f);
        }
        mu
    }

    /// Cholesky decomposition of the correlation matrix.
    /// Returns lower triangular matrix L such that ρ = L L^T.
    fn cholesky(&self) -> Vec<f64> {
        let n = self.n_rates;
        let mut l = vec![0.0; n * n];

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[i * n + k] * l[j * n + k];
                }
                if i == j {
                    l[i * n + j] = (self.correlation[i * n + i] - sum).max(0.0).sqrt();
                } else {
                    let diag = l[j * n + j];
                    if diag > 1e-15 {
                        l[i * n + j] = (self.correlation[i * n + j] - sum) / diag;
                    }
                }
            }
        }
        l
    }
}

/// State of forward rates at a given time.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LmmCurveState {
    /// Current forward rates.
    pub forwards: Vec<f64>,
    /// Index of first alive rate (rates before this have expired).
    pub alive_index: usize,
}

impl LmmCurveState {
    /// Create initial state.
    pub fn new(forwards: Vec<f64>) -> Self {
        Self {
            forwards,
            alive_index: 0,
        }
    }

    /// Discount factor from rate i to rate j (j > i).
    pub fn discount(&self, from: usize, to: usize, accruals: &[f64]) -> f64 {
        let mut d = 1.0;
        for (&tau, &f) in accruals[from..to].iter().zip(self.forwards[from..to].iter()) {
            d /= 1.0 + tau * f;
        }
        d
    }

    /// Swap rate for a swap from rate `start` to rate `end`.
    pub fn swap_rate(&self, start: usize, end: usize, accruals: &[f64]) -> f64 {
        let mut annuity = 0.0;
        let mut d = 1.0;
        for (&tau, &f) in accruals[start..end].iter().zip(self.forwards[start..end].iter()) {
            d /= 1.0 + tau * f;
            annuity += tau * d;
        }
        if annuity.abs() < 1e-15 {
            return 0.0;
        }
        (1.0 - d) / annuity
    }
}

/// Evolve forward rates using a predictor-corrector (Glasserman) scheme.
///
/// Returns evolved forward rates at the next time step.
fn evolve_one_step(
    config: &LmmConfig,
    forwards: &[f64],
    alive_from: usize,
    dt: f64,
    sqrt_dt: f64,
    chol: &[f64],
    z: &[f64], // independent standard normals, length n_rates
) -> Vec<f64> {
    let n = config.n_rates;
    let mut new_forwards = forwards.to_vec();

    // Correlated Brownian increments: W = L · z
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
        let drift = config.drift(i, forwards);
        let sigma = config.volatilities[i];
        f_pred[i] = forwards[i]
            * ((drift - 0.5 * sigma * sigma) * dt + sigma * dw[i]).exp();
    }

    // Corrector step: average drift at initial and predicted state
    for i in alive_from..n {
        let drift_0 = config.drift(i, forwards);
        let drift_1 = config.drift(i, &f_pred);
        let avg_drift = 0.5 * (drift_0 + drift_1);
        let sigma = config.volatilities[i];
        new_forwards[i] = forwards[i]
            * ((avg_drift - 0.5 * sigma * sigma) * dt + sigma * dw[i]).exp();
    }

    new_forwards
}

/// Result of an LMM Monte Carlo simulation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LmmResult {
    /// Expected price.
    pub price: f64,
    /// Standard error.
    pub std_error: f64,
}

/// Price a cap via LMM Monte Carlo.
///
/// A cap is a portfolio of caplets. Each caplet i pays max(f_i(T_i) − K, 0) × τ_i
/// at time T_{i+1}, where T_i is the fixing date for rate i.
pub fn lmm_cap_price(
    config: &LmmConfig,
    strike: f64,
    n_paths: usize,
    seed: u64,
) -> LmmResult {
    let n = config.n_rates;
    let chol = config.cholesky();
    let dt = config.accruals[0]; // time step = accrual period

    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut rng = SmallRng::seed_from_u64(seed);

    for _ in 0..n_paths {
        let mut forwards = config.initial_forwards.clone();
        let mut numeraire = 1.0; // money market account

        for step in 0..n {
            // Generate independent normals
            let z: Vec<f64> = (0..n).map(|_| rng.sample(StandardNormal)).collect();
            let sqrt_dt = dt.sqrt();

            // Caplet payment for rate `step`
            let caplet_payoff = (forwards[step] - strike).max(0.0) * config.accruals[step];

            // Discount: we accumulate numeraire
            let pv = caplet_payoff / numeraire;
            sum += pv;
            sum_sq += pv * pv;

            // Evolve forwards
            forwards = evolve_one_step(config, &forwards, step + 1, dt, sqrt_dt, &chol, &z);

            // Update numeraire
            numeraire *= 1.0 + config.accruals[step] * config.initial_forwards[step];
        }
    }

    let mean = sum / n_paths as f64;
    let variance = sum_sq / n_paths as f64 - mean * mean;
    let std_error = (variance / n_paths as f64).max(0.0).sqrt();

    LmmResult {
        price: mean,
        std_error,
    }
}

/// Price a European swaption via LMM Monte Carlo.
///
/// A payer swaption pays max(S(T) − K, 0) × A(T) at exercise date T,
/// where S(T) is the swap rate and A(T) is the annuity.
pub fn lmm_swaption_price(
    config: &LmmConfig,
    swap_start: usize, // index of first rate in the swap
    swap_end: usize,   // index past last rate in the swap
    strike: f64,
    n_paths: usize,
    is_payer: bool,
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

        // Evolve to swaption exercise date (step = swap_start)
        for step in 0..swap_start {
            let z: Vec<f64> = (0..n).map(|_| rng.sample(StandardNormal)).collect();
            let sqrt_dt = dt.sqrt();
            forwards = evolve_one_step(config, &forwards, step + 1, dt, sqrt_dt, &chol, &z);
            numeraire *= 1.0 + config.accruals[step] * config.initial_forwards[step];
        }

        // Compute swap rate and annuity at exercise
        let state = LmmCurveState::new(forwards.clone());
        let swap_rate = state.swap_rate(swap_start, swap_end, &config.accruals);

        // Annuity
        let mut annuity = 0.0;
        let mut d = 1.0;
        for (&tau, &f) in config.accruals[swap_start..swap_end].iter().zip(forwards[swap_start..swap_end].iter()) {
            d /= 1.0 + tau * f;
            annuity += tau * d;
        }

        let payoff = if is_payer {
            (swap_rate - strike).max(0.0) * annuity
        } else {
            (strike - swap_rate).max(0.0) * annuity
        };

        let pv = payoff / numeraire;
        sum += pv;
        sum_sq += pv * pv;
    }

    let mean = sum / n_paths as f64;
    let variance = sum_sq / n_paths as f64 - mean * mean;
    let std_error = (variance / n_paths as f64).max(0.0).sqrt();

    LmmResult {
        price: mean,
        std_error,
    }
}

/// Black's formula for a caplet (for comparison).
pub fn black_caplet_price(
    forward: f64,
    strike: f64,
    vol: f64,
    expiry: f64,
    accrual: f64,
    discount: f64,
) -> f64 {
    if vol <= 0.0 || expiry <= 0.0 {
        return discount * accrual * (forward - strike).max(0.0);
    }
    let std_dev = vol * expiry.sqrt();
    let d1 = ((forward / strike).ln() + 0.5 * std_dev * std_dev) / std_dev;
    let d2 = d1 - std_dev;
    let n = ql_math::distributions::NormalDistribution::standard();
    discount * accrual * (forward * n.cdf(d1) - strike * n.cdf(d2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_config() -> LmmConfig {
        // 10 quarterly forward rates, flat at 5%, flat 20% vol, 0.5 correlation decay
        LmmConfig::flat(10, 0.05, 0.25, 0.20, 0.5)
    }

    #[test]
    fn config_flat_creates_valid_correlation() {
        let c = make_config();
        // Diagonal = 1
        for i in 0..c.n_rates {
            assert_abs_diff_eq!(c.correlation[i * c.n_rates + i], 1.0, epsilon = 1e-12);
        }
        // Symmetric
        for i in 0..c.n_rates {
            for j in 0..c.n_rates {
                assert_abs_diff_eq!(
                    c.correlation[i * c.n_rates + j],
                    c.correlation[j * c.n_rates + i],
                    epsilon = 1e-12
                );
            }
        }
    }

    #[test]
    fn cholesky_reconstructs_correlation() {
        let c = make_config();
        let l = c.cholesky();
        let n = c.n_rates;

        // Verify L L^T = ρ
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += l[i * n + k] * l[j * n + k];
                }
                assert_abs_diff_eq!(sum, c.correlation[i * n + j], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn curve_state_discount_flat() {
        let config = make_config();
        let state = LmmCurveState::new(config.initial_forwards.clone());
        // Discount from 0 to 1 (first period)
        let d = state.discount(0, 1, &config.accruals);
        assert_abs_diff_eq!(d, 1.0 / (1.0 + 0.25 * 0.05), epsilon = 1e-12);
    }

    #[test]
    fn curve_state_swap_rate_flat() {
        let config = make_config();
        let state = LmmCurveState::new(config.initial_forwards.clone());
        // When forward rates are flat, swap rate ≈ forward rate
        let sr = state.swap_rate(0, 10, &config.accruals);
        assert_abs_diff_eq!(sr, 0.05, epsilon = 0.001);
    }

    #[test]
    fn drift_is_negative_terminal_measure() {
        let config = make_config();
        // Under terminal measure, drift for early rates should be negative
        let drift = config.drift(0, &config.initial_forwards);
        assert!(
            drift < 0.0,
            "Drift under terminal measure should be negative: {drift}"
        );
    }

    #[test]
    fn lmm_cap_price_positive() {
        let config = make_config();
        let result = lmm_cap_price(&config, 0.05, 5000, 42);
        assert!(
            result.price > 0.0,
            "ATM cap should have positive price: {}",
            result.price
        );
    }

    #[test]
    fn lmm_cap_price_monotone_in_strike() {
        let config = make_config();
        let p_low = lmm_cap_price(&config, 0.03, 5000, 42);
        let p_high = lmm_cap_price(&config, 0.07, 5000, 42);
        assert!(
            p_low.price > p_high.price,
            "Cap with lower strike should be more expensive: low={}, high={}",
            p_low.price,
            p_high.price
        );
    }

    #[test]
    fn lmm_cap_vs_black_order_of_magnitude() {
        let config = make_config();
        let lmm_price = lmm_cap_price(&config, 0.05, 10000, 42);

        // Sum Black caplet prices for comparison
        let mut black_total = 0.0;
        let mut discount = 1.0;
        for i in 0..config.n_rates {
            let expiry = (i + 1) as f64 * config.accruals[i];
            discount /= 1.0 + config.accruals[i] * config.initial_forwards[i];
            black_total += black_caplet_price(
                config.initial_forwards[i],
                0.05,
                config.volatilities[i],
                expiry,
                config.accruals[i],
                discount,
            );
        }

        // LMM should be in the same ballpark as Black
        let ratio = lmm_price.price / black_total;
        assert!(
            ratio > 0.5 && ratio < 2.0,
            "LMM cap ({}) should be near Black cap ({}), ratio={}",
            lmm_price.price,
            black_total,
            ratio
        );
    }

    #[test]
    fn lmm_swaption_price_positive() {
        let config = make_config();
        let result = lmm_swaption_price(&config, 2, 10, 0.05, 5000, true, 42);
        assert!(
            result.price > 0.0,
            "Payer swaption should have positive price: {}",
            result.price
        );
    }

    #[test]
    fn lmm_swaption_payer_receiver_parity() {
        let config = make_config();
        let n_paths = 20000;
        let payer = lmm_swaption_price(&config, 2, 10, 0.05, n_paths, true, 42);
        let receiver = lmm_swaption_price(&config, 2, 10, 0.05, n_paths, false, 42);

        // When strike = ATM forward swap rate, payer ≈ receiver
        // They should at least be of similar magnitude
        let ratio = payer.price / receiver.price.max(1e-10);
        assert!(
            ratio > 0.3 && ratio < 3.0,
            "ATM payer ({}) and receiver ({}) should be similar order",
            payer.price,
            receiver.price
        );
    }

    #[test]
    fn evolve_preserves_positivity() {
        let config = make_config();
        let chol = config.cholesky();
        let n = config.n_rates;
        let dt = config.accruals[0];
        let sqrt_dt = dt.sqrt();

        let mut rng = SmallRng::seed_from_u64(12345);
        let mut forwards = config.initial_forwards.clone();

        for step in 0..n {
            let z: Vec<f64> = (0..n).map(|_| rng.sample(StandardNormal)).collect();
            forwards = evolve_one_step(&config, &forwards, step + 1, dt, sqrt_dt, &chol, &z);
            for (i, &f) in forwards.iter().enumerate().skip(step + 1) {
                assert!(f > 0.0, "Forward rate {i} should stay positive: {f}");
            }
        }
    }
}
