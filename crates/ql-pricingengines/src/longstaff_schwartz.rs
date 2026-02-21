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
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use ql_instruments::OptionType;
use ql_methods::MCResult;

/// Basis function system for Longstaff-Schwartz regression.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
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
    let df_step = (-r * dt).exp();
    let sqrt_dt = dt.sqrt();
    let drift_dt = (r - q - 0.5 * vol * vol) * dt;
    let vol_sqrt_dt = vol * sqrt_dt;
    let omega = option_type.sign();
    let ln_spot = spot.ln();
    let ln_strike = strike.ln();

    // Step 1: Simulate paths forward as LOG-spots (no exp() in hot loop).
    // Generate row-major, then transpose to column-major so backward
    // induction scans sequential memory (huge cache win on 20 MB arrays).
    let stride = num_steps + 1;
    let row_paths =
        simulate_paths_log(ln_spot, drift_dt, vol_sqrt_dt, num_paths, num_steps, seed);

    // Transpose to column-major: paths[step * num_paths + path_idx]
    let paths = transpose_row_to_col(&row_paths, num_paths, stride);
    drop(row_paths); // free 20 MB

    // Precompute discount-factor table: df_table[k] = df_step^k
    let mut df_table = Vec::with_capacity(num_steps + 1);
    df_table.push(1.0);
    for _ in 1..=num_steps {
        df_table.push(df_table.last().unwrap() * df_step);
    }

    // Step 2: Terminal payoff (exp only at terminal nodes — sequential access)
    let mut cashflow_time = vec![num_steps; num_paths];
    let mut cashflow_value = vec![0.0_f64; num_paths];

    let terminal_col = &paths[num_steps * num_paths..];
    for i in 0..num_paths {
        let s = terminal_col[i].exp();
        cashflow_value[i] = (omega * (s - strike)).max(0.0);
    }

    // Step 3: Backward induction (column-major gives sequential reads)
    let mut itm_indices: Vec<usize> = Vec::with_capacity(num_paths);
    let mut itm_x: Vec<f64> = Vec::with_capacity(num_paths);
    let mut itm_y: Vec<f64> = Vec::with_capacity(num_paths);
    let p = basis_degree + 1;
    let mut basis_buf: Vec<f64> = Vec::with_capacity(num_paths * p);

    for step in (1..num_steps).rev() {
        itm_indices.clear();
        itm_x.clear();
        itm_y.clear();

        let step_col = &paths[step * num_paths..(step + 1) * num_paths];

        for i in 0..num_paths {
            let ln_s = step_col[i]; // sequential access!
            // ITM check in log-space — avoids exp() for OTM paths
            let is_itm = if omega > 0.0 {
                ln_s > ln_strike
            } else {
                ln_s < ln_strike
            };
            if is_itm {
                let s = ln_s.exp();
                let steps_ahead = cashflow_time[i] - step;
                let disc = df_table[steps_ahead];
                let cont_val = cashflow_value[i] * disc;

                itm_indices.push(i);
                itm_x.push(s);
                itm_y.push(cont_val);
            }
        }

        if itm_indices.len() < p {
            continue;
        }

        build_basis_matrix_into(&itm_x, basis_degree, basis, &mut basis_buf);
        let coeffs = least_squares_fit(&basis_buf, p, &itm_y);

        // Compare exercise vs fitted continuation (reuse cached spot from itm_x)
        for (idx, &i) in itm_indices.iter().enumerate() {
            let s = itm_x[idx];
            let exercise_val = omega * (s - strike);
            let fitted_cont = evaluate_basis(s, &coeffs, basis);

            if exercise_val > fitted_cont {
                cashflow_time[i] = step;
                cashflow_value[i] = exercise_val;
            }
        }
    }

    // Step 4: Compute NPV as discounted average (use table, not powi)
    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    for i in 0..num_paths {
        let disc = df_table[cashflow_time[i]];
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

/// Simulate GBM paths as LOG-spot values in a single pre-allocated array.
///
/// Uses antithetic variates: each batch generates path pairs (original +
/// mirror) sharing the same random draws, halving RNG calls and reducing
/// variance. Returns flat row-major array where `paths[path_idx * stride + step]`
/// is `ln(S)` at that step, with `stride = num_steps + 1`.
fn simulate_paths_log(
    ln_spot: f64,
    drift_dt: f64,
    vol_sqrt_dt: f64,
    num_paths: usize,
    num_steps: usize,
    seed: u64,
) -> Vec<f64> {
    let stride = num_steps + 1;
    let batch_size = 5000_usize;
    let chunk_elems = batch_size * stride;
    let mut paths = vec![0.0_f64; num_paths * stride];

    // Zero-copy parallel generation with antithetic variates.
    #[cfg(feature = "parallel")]
    let chunks_iter = paths.par_chunks_mut(chunk_elems);
    #[cfg(not(feature = "parallel"))]
    let chunks_iter = paths.chunks_mut(chunk_elems);

    chunks_iter
        .enumerate()
        .for_each(|(batch_idx, chunk)| {
            let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(batch_idx as u64));
            let count = chunk.len() / stride;
            let half = count / 2;

            // Generate original + antithetic pairs
            for p in 0..half {
                let base_a = p * stride;
                let base_b = (half + p) * stride;
                chunk[base_a] = ln_spot;
                chunk[base_b] = ln_spot;
                let mut ln_a = ln_spot;
                let mut ln_b = ln_spot;
                for j in 1..stride {
                    let z: f64 = StandardNormal.sample(&mut rng);
                    ln_a += drift_dt + vol_sqrt_dt * z;
                    ln_b += drift_dt - vol_sqrt_dt * z;
                    chunk[base_a + j] = ln_a;
                    chunk[base_b + j] = ln_b;
                }
            }
            // Odd path without an antithetic partner
            if count & 1 == 1 {
                let base = (count - 1) * stride;
                chunk[base] = ln_spot;
                let mut ln_s = ln_spot;
                for j in 1..stride {
                    let z: f64 = StandardNormal.sample(&mut rng);
                    ln_s += drift_dt + vol_sqrt_dt * z;
                    chunk[base + j] = ln_s;
                }
            }
        });

    paths
}

/// Cache-friendly tiled transpose: row-major → column-major.
///
/// Input:  `row[i * ncols + j]`  (path i, step j)
/// Output: `col[j * nrows + i]`  (step j, path i)
///
/// Block tiling keeps both source and destination in L1/L2 cache,
/// making the 20 MB transposition fast (~3-5 ms).
fn transpose_row_to_col(row: &[f64], nrows: usize, ncols: usize) -> Vec<f64> {
    let mut col = vec![0.0_f64; nrows * ncols];
    const BLOCK: usize = 64;
    for bi in (0..nrows).step_by(BLOCK) {
        let bi_end = (bi + BLOCK).min(nrows);
        for bj in (0..ncols).step_by(BLOCK) {
            let bj_end = (bj + BLOCK).min(ncols);
            for i in bi..bi_end {
                let src_base = i * ncols;
                for j in bj..bj_end {
                    col[j * nrows + i] = row[src_base + j];
                }
            }
        }
    }
    col
}

// ===========================================================================
// Regression (least-squares polynomial fit)
// ===========================================================================

/// Build the basis matrix for regression (flat row-major), reusing a buffer.
///
/// Each row corresponds to one observation, columns are basis function values.
/// Writes into `buf` which is resized as needed. Size: `n * p` where `p = degree + 1`.
fn build_basis_matrix_into(x: &[f64], degree: usize, basis: LSMBasis, buf: &mut Vec<f64>) {
    let n = x.len();
    let p = degree + 1;
    buf.clear();
    buf.resize(n * p, 0.0);

    for (i, xi) in x.iter().enumerate() {
        let row = i * p;
        match basis {
            LSMBasis::Monomial => {
                let mut xp = 1.0;
                for col in 0..p {
                    buf[row + col] = xp;
                    xp *= xi;
                }
            }
            LSMBasis::Laguerre => {
                buf[row] = 1.0;
                if p > 1 {
                    buf[row + 1] = 1.0 - xi;
                }
                if p > 2 {
                    buf[row + 2] = 1.0 - 2.0 * xi + 0.5 * xi * xi;
                }
                for j in 3..p {
                    buf[row + j] = xi.powi(j as i32);
                }
            }
        }
    }
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
/// `basis_matrix` is flat row-major with `p` columns.
fn least_squares_fit(basis_matrix: &[f64], p: usize, y: &[f64]) -> Vec<f64> {
    let n = y.len();
    if n == 0 || p == 0 {
        return vec![];
    }

    // Build A^T A (p × p) and A^T y (p × 1) — flat row-major
    let mut ata = vec![0.0; p * p];
    let mut aty = vec![0.0; p];

    for (i, &yi) in y.iter().enumerate() {
        let row = i * p;
        for j in 0..p {
            let bj = basis_matrix[row + j];
            aty[j] += bj * yi;
            for k in j..p {
                ata[j * p + k] += bj * basis_matrix[row + k];
            }
        }
    }
    // Symmetrize
    for j in 1..p {
        for k in 0..j {
            ata[j * p + k] = ata[k * p + j];
        }
    }

    // Add small regularization for numerical stability
    for j in 0..p {
        ata[j * p + j] += 1e-8;
    }

    // Cholesky solve: A^T A = L L^T  (flat row-major)
    let l = cholesky_flat(p, &ata);

    // Forward substitution: L z = A^T y
    let mut z = vec![0.0; p];
    for i in 0..p {
        let mut sum = aty[i];
        for j in 0..i {
            sum -= l[i * p + j] * z[j];
        }
        z[i] = sum / l[i * p + i];
    }

    // Back substitution: L^T c = z
    let mut c = vec![0.0; p];
    for i in (0..p).rev() {
        let mut sum = z[i];
        for j in (i + 1)..p {
            sum -= l[j * p + i] * c[j];
        }
        c[i] = sum / l[i * p + i];
    }

    c
}

/// Cholesky decomposition of a symmetric positive-definite matrix (flat row-major).
fn cholesky_flat(n: usize, a: &[f64]) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let sum: f64 = (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum();
            if i == j {
                let val = a[i * n + i] - sum;
                l[i * n + j] = if val > 0.0 { val.sqrt() } else { 1e-15 };
            } else {
                l[i * n + j] = (a[i * n + j] - sum) / l[j * n + j];
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
