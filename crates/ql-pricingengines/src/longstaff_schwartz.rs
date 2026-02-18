//! Longstaff-Schwartz Monte Carlo engine for American options.
//!
//! Implements the Longstaff-Schwartz (2001) least-squares regression
//! approach for pricing American-style options via backward induction
//! on simulated paths.
//!
//! # Algorithm
//! 1. Simulate `num_paths` GBM paths forward.
//! 2. At maturity, compute terminal payoff.
//! 3. Walk backwards through exercise dates:
//!    a. Identify in-the-money paths.
//!    b. Regress discounted continuation values on polynomial basis of S.
//!    c. Compare immediate exercise to fitted continuation; exercise if better.
//! 4. The option value is the discounted average of optimal exercise payoffs.
//!
//! # References
//! - Longstaff, F. and Schwartz, E. (2001), "Valuing American Options
//!   by Simulation: A Simple Least-Squares Approach", *Review of Financial
//!   Studies* 14(1).

use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

use ql_instruments::OptionType;
use ql_methods::MCResult;

/// Basis function system for Longstaff-Schwartz regression.
#[derive(Debug, Clone, Copy)]
pub enum LSMBasis {
    /// Simple monomials: 1, x, x², ..., x^degree.
    Monomial,
    /// Laguerre polynomials (weighted): L_0, L_1, ..., L_degree.
    Laguerre,
}

/// Price an American option using Longstaff-Schwartz Monte Carlo.
///
/// Uses multi-step GBM paths with backward regression to determine
/// the optimal exercise strategy.
///
/// # Parameters
/// - `spot`: initial underlying price
/// - `strike`: option strike price
/// - `r`: risk-free rate
/// - `q`: dividend yield
/// - `vol`: volatility
/// - `time_to_expiry`: time to expiry in years
/// - `option_type`: `Call` or `Put`
/// - `num_paths`: number of MC paths (more = lower variance)
/// - `num_steps`: number of time steps (exercise opportunities)
/// - `basis_degree`: degree of polynomial basis for regression (2–4 recommended)
/// - `basis`: which basis function system to use
/// - `seed`: RNG seed for reproducibility
///
/// # Returns
/// `MCResult` with NPV, standard error, and number of paths.
#[allow(clippy::too_many_arguments)]
pub fn mc_american_longstaff_schwartz(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    vol: f64,
    time_to_expiry: f64,
    option_type: OptionType,
    num_paths: usize,
    num_steps: usize,
    basis_degree: usize,
    basis: LSMBasis,
    seed: u64,
) -> MCResult {
    let dt = time_to_expiry / num_steps as f64;
    let df_step = (-r * dt).exp(); // discount factor per step
    let sqrt_dt = dt.sqrt();
    let drift = (r - q - 0.5 * vol * vol) * dt;
    let omega = option_type.sign();

    // Step 1: Simulate paths forward (all at once for regression)
    // paths[i][j] = spot at time step j for path i
    let paths = simulate_paths(spot, drift, vol, sqrt_dt, num_paths, num_steps, seed);

    // Step 2: Initialize with terminal payoff
    // cashflow_time[i] = step at which path i exercises (or num_steps for terminal)
    // cashflow_value[i] = undiscounted payoff at exercise
    let mut cashflow_time = vec![num_steps; num_paths];
    let mut cashflow_value = vec![0.0_f64; num_paths];

    for i in 0..num_paths {
        cashflow_value[i] = (omega * (paths[i][num_steps] - strike)).max(0.0);
    }

    // Step 3: Backward induction
    for step in (1..num_steps).rev() {
        // Find in-the-money paths at this step
        let mut itm_indices: Vec<usize> = Vec::new();
        let mut itm_x: Vec<f64> = Vec::new();
        let mut itm_y: Vec<f64> = Vec::new();

        for i in 0..num_paths {
            let s = paths[i][step];
            let exercise_val = omega * (s - strike);
            if exercise_val > 0.0 {
                // Discount the future cashflow back to this step
                let steps_ahead = cashflow_time[i] - step;
                let disc = df_step.powi(steps_ahead as i32);
                let cont_val = cashflow_value[i] * disc;

                itm_indices.push(i);
                itm_x.push(s);
                itm_y.push(cont_val);
            }
        }

        if itm_indices.len() < basis_degree + 1 {
            // Not enough ITM paths for regression, skip this step
            continue;
        }

        // Regression: fit continuation value = f(S)
        let basis_matrix = build_basis_matrix(&itm_x, basis_degree, basis);
        let coeffs = least_squares_fit(&basis_matrix, &itm_y);

        // Compare exercise vs fitted continuation
        for &i in itm_indices.iter() {
            let s = paths[i][step];
            let exercise_val = omega * (s - strike);
            let fitted_cont = evaluate_basis(s, &coeffs, basis);

            if exercise_val > fitted_cont && exercise_val > 0.0 {
                cashflow_time[i] = step;
                cashflow_value[i] = exercise_val;
            }
        }
    }

    // Step 4: Compute NPV as discounted average
    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    for i in 0..num_paths {
        let steps = cashflow_time[i];
        let disc = df_step.powi(steps as i32);
        let pv = cashflow_value[i] * disc;
        sum += pv;
        sum_sq += pv * pv;
    }

    let n = num_paths as f64;
    let mean = sum / n;
    let variance = (sum_sq / n - mean * mean).max(0.0);
    let std_error = (variance / n).sqrt();

    MCResult {
        npv: mean,
        std_error,
        num_paths,
    }
}

// ===========================================================================
// Path simulation
// ===========================================================================

/// Simulate GBM paths forward.
/// Returns `paths[path_idx][step]` = spot value.
fn simulate_paths(
    spot: f64,
    drift_dt: f64,
    vol: f64,
    sqrt_dt: f64,
    num_paths: usize,
    num_steps: usize,
    seed: u64,
) -> Vec<Vec<f64>> {
    // Use parallel batches for speed
    let batch_size = 5000_usize;
    let num_batches = num_paths.div_ceil(batch_size);

    let batches: Vec<Vec<Vec<f64>>> = (0..num_batches)
        .into_par_iter()
        .map(|batch_idx| {
            let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(batch_idx as u64));
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(num_paths);
            let mut batch_paths = Vec::with_capacity(end - start);

            for _ in start..end {
                let mut path = Vec::with_capacity(num_steps + 1);
                path.push(spot);
                let mut s = spot;
                for _ in 0..num_steps {
                    let z: f64 = StandardNormal.sample(&mut rng);
                    s *= (drift_dt + vol * sqrt_dt * z).exp();
                    path.push(s);
                }
                batch_paths.push(path);
            }
            batch_paths
        })
        .collect();

    batches.into_iter().flatten().collect()
}

// ===========================================================================
// Regression (least-squares polynomial fit)
// ===========================================================================

/// Build the basis matrix for regression.
///
/// Each row corresponds to one observation, columns are basis function values.
fn build_basis_matrix(x: &[f64], degree: usize, basis: LSMBasis) -> Vec<Vec<f64>> {
    let n = x.len();
    let p = degree + 1; // number of basis functions
    let mut matrix = vec![vec![0.0; p]; n];

    for (i, xi) in x.iter().enumerate() {
        match basis {
            LSMBasis::Monomial => {
                let mut xp = 1.0;
                for col in matrix[i].iter_mut().take(p) {
                    *col = xp;
                    xp *= xi;
                }
            }
            LSMBasis::Laguerre => {
                // Weighted Laguerre polynomials L_k(x) evaluated at normalized x
                // L_0 = 1
                // L_1 = 1 - x
                // L_2 = 1 - 2x + x²/2
                matrix[i][0] = 1.0;
                if p > 1 {
                    matrix[i][1] = 1.0 - xi;
                }
                if p > 2 {
                    matrix[i][2] = 1.0 - 2.0 * xi + 0.5 * xi * xi;
                }
                // For higher degrees, fall back to monomial
                for (j, col) in matrix[i].iter_mut().enumerate().skip(3).take(p.saturating_sub(3)) {
                    *col = xi.powi(j as i32);
                }
            }
        }
    }

    matrix
}

/// Evaluate the fitted polynomial at a given point.
fn evaluate_basis(x: f64, coeffs: &[f64], basis: LSMBasis) -> f64 {
    let p = coeffs.len();
    match basis {
        LSMBasis::Monomial => {
            let mut val = 0.0;
            let mut xp = 1.0;
            for c in coeffs.iter().take(p) {
                val += c * xp;
                xp *= x;
            }
            val
        }
        LSMBasis::Laguerre => {
            let mut val = coeffs[0]; // L_0 = 1
            if p > 1 {
                val += coeffs[1] * (1.0 - x);
            }
            if p > 2 {
                val += coeffs[2] * (1.0 - 2.0 * x + 0.5 * x * x);
            }
            for (j, c) in coeffs.iter().enumerate().skip(3).take(p - 3) {
                val += c * x.powi(j as i32);
            }
            val
        }
    }
}

/// Solve the least-squares problem A·c = y using the normal equations.
///
/// Uses A^T·A·c = A^T·y with simple Cholesky factorization.
fn least_squares_fit(basis_matrix: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let n = basis_matrix.len();
    let p = if n > 0 { basis_matrix[0].len() } else { return vec![]; };

    // Build A^T A (p × p) and A^T y (p × 1)
    let mut ata = vec![vec![0.0; p]; p];
    let mut aty = vec![0.0; p];

    for (row, yi) in basis_matrix.iter().zip(y.iter()) {
        for (j, &bj) in row.iter().enumerate() {
            aty[j] += bj * yi;
            for (k, &bk) in row.iter().enumerate().skip(j) {
                ata[j][k] += bj * bk;
            }
        }
    }
    // Symmetrize — we need cross-row indexing, so iterator style is impractical
    #[allow(clippy::needless_range_loop)]
    for j in 1..p {
        for k in 0..j {
            ata[j][k] = ata[k][j];
        }
    }

    // Add small regularization for numerical stability
    for (j, row) in ata.iter_mut().enumerate().take(p) {
        row[j] += 1e-8;
    }

    // Cholesky solve: A^T A = L L^T
    let l = cholesky(p, &ata);

    // Forward substitution: L z = A^T y
    let mut z = vec![0.0; p];
    for i in 0..p {
        let mut sum = aty[i];
        for j in 0..i {
            sum -= l[i][j] * z[j];
        }
        z[i] = sum / l[i][i];
    }

    // Back substitution: L^T c = z
    let mut c = vec![0.0; p];
    for i in (0..p).rev() {
        let mut sum = z[i];
        for j in (i + 1)..p {
            sum -= l[j][i] * c[j];
        }
        c[i] = sum / l[i][i];
    }

    c
}

/// Cholesky decomposition of a symmetric positive-definite matrix.
fn cholesky(n: usize, a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let sum: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
            if i == j {
                let val = a[i][i] - sum;
                l[i][j] = if val > 0.0 { val.sqrt() } else { 1e-15 };
            } else {
                l[i][j] = (a[i][j] - sum) / l[j][j];
            }
        }
    }
    l
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const SPOT: f64 = 100.0;
    const STRIKE: f64 = 100.0;
    const R: f64 = 0.05;
    const Q: f64 = 0.02;
    const VOL: f64 = 0.20;
    const T: f64 = 1.0;

    fn fd_reference(is_call: bool) -> f64 {
        ql_methods::fd_black_scholes(SPOT, STRIKE, R, Q, VOL, T, is_call, true, 800, 800).npv
    }

    #[test]
    fn ls_american_put_positive() {
        let res = mc_american_longstaff_schwartz(
            SPOT, STRIKE, R, Q, VOL, T,
            OptionType::Put,
            50_000, 50, 3, LSMBasis::Monomial, 42,
        );
        assert!(res.npv > 0.0, "LS put NPV should be positive: {}", res.npv);
    }

    #[test]
    fn ls_american_put_exceeds_european() {
        let res = mc_american_longstaff_schwartz(
            SPOT, STRIKE, R, Q, VOL, T,
            OptionType::Put,
            50_000, 50, 3, LSMBasis::Monomial, 42,
        );
        let euro = ql_methods::mc_european(SPOT, STRIKE, R, Q, VOL, T, OptionType::Put, 100_000, true, 42);
        assert!(
            res.npv >= euro.npv - 3.0 * euro.std_error,
            "LS American put {} should be >= Euro MC put {} (within 3 SE)",
            res.npv,
            euro.npv
        );
    }

    #[test]
    fn ls_american_put_vs_fd() {
        let res = mc_american_longstaff_schwartz(
            SPOT, STRIKE, R, Q, VOL, T,
            OptionType::Put,
            100_000, 50, 3, LSMBasis::Monomial, 42,
        );
        let fd = fd_reference(false);
        // LS should agree with FD within ~3 standard errors
        let diff = (res.npv - fd).abs();
        assert!(
            diff < fd * 0.05 + 3.0 * res.std_error,
            "LS put {:.4} ± {:.4} vs FD {:.4}: diff {:.4}",
            res.npv,
            res.std_error,
            fd,
            diff
        );
    }

    #[test]
    fn ls_american_call_no_dividend_near_european() {
        // With no dividends, American call should ≈ European call
        let res = mc_american_longstaff_schwartz(
            SPOT, STRIKE, R, 0.0, VOL, T,
            OptionType::Call,
            50_000, 50, 3, LSMBasis::Monomial, 42,
        );
        let euro = ql_methods::mc_european(SPOT, STRIKE, R, 0.0, VOL, T, OptionType::Call, 100_000, true, 42);
        let diff = (res.npv - euro.npv).abs();
        assert!(
            diff < 1.0,
            "LS American call {:.4} should be near European call {:.4}",
            res.npv,
            euro.npv
        );
    }

    #[test]
    fn ls_laguerre_basis_works() {
        let res = mc_american_longstaff_schwartz(
            SPOT, STRIKE, R, Q, VOL, T,
            OptionType::Put,
            50_000, 50, 3, LSMBasis::Laguerre, 42,
        );
        let fd = fd_reference(false);
        let diff = (res.npv - fd).abs();
        assert!(
            diff < fd * 0.10,
            "LS (Laguerre) put {:.4} vs FD {:.4}: diff {:.4}",
            res.npv,
            fd,
            diff
        );
    }

    #[test]
    fn ls_convergence_with_more_paths() {
        let res_small = mc_american_longstaff_schwartz(
            SPOT, STRIKE, R, Q, VOL, T,
            OptionType::Put,
            10_000, 50, 3, LSMBasis::Monomial, 42,
        );
        let res_large = mc_american_longstaff_schwartz(
            SPOT, STRIKE, R, Q, VOL, T,
            OptionType::Put,
            50_000, 50, 3, LSMBasis::Monomial, 42,
        );
        // Standard error should decrease with more paths
        assert!(
            res_large.std_error < res_small.std_error + 0.01,
            "More paths should reduce SE: small {:.4} vs large {:.4}",
            res_small.std_error,
            res_large.std_error
        );
    }

    #[test]
    fn ls_deep_itm_put() {
        // S=50, K=100 — deep ITM put, should exercise early ≈ 50
        let res = mc_american_longstaff_schwartz(
            50.0, 100.0, R, Q, VOL, T,
            OptionType::Put,
            30_000, 50, 3, LSMBasis::Monomial, 42,
        );
        assert!(
            res.npv > 45.0,
            "Deep ITM LS put should be near intrinsic 50: got {:.4}",
            res.npv
        );
    }
}
