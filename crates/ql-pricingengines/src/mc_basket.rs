//! Monte Carlo basket option engines.
//!
//! - [`mc_european_basket`] — MC European basket option (min/max/sum payoffs)
//! - [`mc_american_basket`] — MC American basket option via LSM

use rand::prelude::*;
use rand_distr::StandardNormal;

/// Basket payoff type.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum BasketPayoffType {
    /// max(ω·(Σ wᵢSᵢ − K), 0)
    WeightedSum,
    /// max(ω·(max(Sᵢ) − K), 0)
    MaxOfAssets,
    /// max(ω·(min(Sᵢ) − K), 0)
    MinOfAssets,
    /// max(ω·(S₁ − S₂ − K), 0)
    Spread,
}

/// Result from MC basket engines.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct McBasketResult {
    /// Option price.
    pub price: f64,
    /// Standard error.
    pub std_error: f64,
}

/// Cholesky decomposition of a correlation matrix.
#[allow(clippy::needless_range_loop)]
fn cholesky(corr: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = corr.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = 0.0;
            for k in 0..j {
                s += l[i][k] * l[j][k];
            }
            if i == j {
                l[i][j] = (corr[i][i] - s).max(0.0).sqrt();
            } else {
                l[i][j] = if l[j][j].abs() > 1e-14 {
                    (corr[i][j] - s) / l[j][j]
                } else {
                    0.0
                };
            }
        }
    }
    l
}

/// Monte Carlo European basket option pricing.
///
/// Simulates correlated GBM paths for multiple assets and averages payoffs.
///
/// # Arguments
/// - `spots` — vector of current asset prices
/// - `strike` — option strike
/// - `r` — risk-free rate
/// - `dividends` — vector of dividend yields
/// - `vols` — vector of volatilities
/// - `corr` — correlation matrix (n×n)
/// - `weights` — portfolio weights (for WeightedSum type)
/// - `t` — time to expiry
/// - `payoff_type` — basket payoff type
/// - `is_call` — true for call, false for put
/// - `n_paths` — number of MC paths
/// - `seed` — optional RNG seed
#[allow(clippy::too_many_arguments)]
pub fn mc_european_basket(
    spots: &[f64],
    strike: f64,
    r: f64,
    dividends: &[f64],
    vols: &[f64],
    corr: &[Vec<f64>],
    weights: &[f64],
    t: f64,
    payoff_type: BasketPayoffType,
    is_call: bool,
    n_paths: usize,
    seed: Option<u64>,
) -> McBasketResult {
    let omega = if is_call { 1.0 } else { -1.0 };
    let n_assets = spots.len();
    let df = (-r * t).exp();
    let chol = cholesky(corr);

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::seed_from_u64(42),
    };

    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    for _ in 0..n_paths {
        // Generate correlated normals
        let z_indep: Vec<f64> = (0..n_assets).map(|_| rng.sample(StandardNormal)).collect();
        let mut z_corr = vec![0.0; n_assets];
        for i in 0..n_assets {
            for j in 0..=i {
                z_corr[i] += chol[i][j] * z_indep[j];
            }
        }

        // Simulate terminal prices
        let mut terminals = vec![0.0; n_assets];
        for i in 0..n_assets {
            let drift = (r - dividends[i] - 0.5 * vols[i] * vols[i]) * t;
            let vol = vols[i] * t.sqrt();
            terminals[i] = spots[i] * (drift + vol * z_corr[i]).exp();
        }

        // Compute payoff
        let basket_value = match payoff_type {
            BasketPayoffType::WeightedSum => {
                terminals.iter().zip(weights).map(|(s, w)| s * w).sum::<f64>()
            }
            BasketPayoffType::MaxOfAssets => {
                terminals.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            }
            BasketPayoffType::MinOfAssets => {
                terminals.iter().cloned().fold(f64::INFINITY, f64::min)
            }
            BasketPayoffType::Spread => {
                if n_assets >= 2 { terminals[0] - terminals[1] } else { terminals[0] }
            }
        };

        let payoff = (omega * (basket_value - strike)).max(0.0);
        sum += payoff;
        sum_sq += payoff * payoff;
    }

    let n = n_paths as f64;
    let mean = sum / n;
    let variance = (sum_sq / n - mean * mean).max(0.0);
    let std_error = (variance / n).sqrt();

    McBasketResult {
        price: df * mean,
        std_error: df * std_error,
    }
}

/// Monte Carlo American basket option via Longstaff-Schwartz.
///
/// Uses LSM with polynomial regression for early exercise.
///
/// # Arguments
/// - Same as `mc_european_basket` plus `n_steps` for exercise monitoring
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub fn mc_american_basket(
    spots: &[f64],
    strike: f64,
    r: f64,
    dividends: &[f64],
    vols: &[f64],
    corr: &[Vec<f64>],
    weights: &[f64],
    t: f64,
    payoff_type: BasketPayoffType,
    is_call: bool,
    n_steps: usize,
    n_paths: usize,
    seed: Option<u64>,
) -> McBasketResult {
    let omega = if is_call { 1.0 } else { -1.0 };
    let n_assets = spots.len();
    let dt = t / n_steps as f64;
    let df_step = (-r * dt).exp();
    let chol = cholesky(corr);

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::seed_from_u64(42),
    };

    // Simulate all paths and store basket values at each step
    let mut paths = vec![vec![vec![0.0_f64; n_assets]; n_steps + 1]; n_paths];
    for p in 0..n_paths {
        paths[p][0][..n_assets].copy_from_slice(&spots[..n_assets]);
        for step in 1..=n_steps {
            let z_indep: Vec<f64> = (0..n_assets).map(|_| rng.sample(StandardNormal)).collect();
            let mut z_corr = vec![0.0; n_assets];
            for i in 0..n_assets {
                for j in 0..=i {
                    z_corr[i] += chol[i][j] * z_indep[j];
                }
            }
            for i in 0..n_assets {
                let drift = (r - dividends[i] - 0.5 * vols[i] * vols[i]) * dt;
                let vol = vols[i] * dt.sqrt();
                paths[p][step][i] = paths[p][step - 1][i] * (drift + vol * z_corr[i]).exp();
            }
        }
    }

    let basket_value = |path: &[f64]| -> f64 {
        match payoff_type {
            BasketPayoffType::WeightedSum => {
                path.iter().zip(weights).map(|(s, w)| s * w).sum::<f64>()
            }
            BasketPayoffType::MaxOfAssets => {
                path.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            }
            BasketPayoffType::MinOfAssets => {
                path.iter().cloned().fold(f64::INFINITY, f64::min)
            }
            BasketPayoffType::Spread => {
                if n_assets >= 2 { path[0] - path[1] } else { path[0] }
            }
        }
    };

    // Cash flows: initialize at terminal
    let mut cashflows = vec![0.0_f64; n_paths];
    let mut ex_time = vec![n_steps; n_paths];

    for p in 0..n_paths {
        let bv = basket_value(&paths[p][n_steps]);
        cashflows[p] = (omega * (bv - strike)).max(0.0);
    }

    // Backward induction
    for step in (1..n_steps).rev() {
        // Find in-the-money paths
        let mut itm_indices = Vec::new();
        let mut itm_x = Vec::new();
        let mut itm_y = Vec::new();

        for p in 0..n_paths {
            let bv = basket_value(&paths[p][step]);
            let exercise_val = (omega * (bv - strike)).max(0.0);
            if exercise_val > 0.0 {
                itm_indices.push(p);
                itm_x.push(bv);
                // Discounted continuation value
                let disc_steps = ex_time[p] - step;
                itm_y.push(cashflows[p] * df_step.powi(disc_steps as i32));
            }
        }

        if itm_indices.len() < 4 { continue; }

        // Polynomial regression: y = a₀ + a₁x + a₂x²
        let m = itm_indices.len();
        let x_mean: f64 = itm_x.iter().sum::<f64>() / m as f64;
        let x_std = (itm_x.iter().map(|x| (x - x_mean).powi(2)).sum::<f64>() / m as f64).sqrt().max(1e-10);

        let mut ata = [[0.0_f64; 3]; 3];
        let mut atb = [0.0_f64; 3];

        for i in 0..m {
            let xn = (itm_x[i] - x_mean) / x_std;
            let basis = [1.0, xn, xn * xn];
            for r in 0..3 {
                for c in 0..3 {
                    ata[r][c] += basis[r] * basis[c];
                }
                atb[r] += basis[r] * itm_y[i];
            }
        }

        // Solve 3x3 system (simple Gaussian elimination)
        let coeffs = solve_3x3(&ata, &atb);

        // Exercise decision
        for (idx, &p) in itm_indices.iter().enumerate() {
            let bv = basket_value(&paths[p][step]);
            let exercise_val = (omega * (bv - strike)).max(0.0);
            let xn = (itm_x[idx] - x_mean) / x_std;
            let continuation = coeffs[0] + coeffs[1] * xn + coeffs[2] * xn * xn;

            if exercise_val > continuation {
                cashflows[p] = exercise_val;
                ex_time[p] = step;
            }
        }
    }

    // Average discounted cash flows
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for p in 0..n_paths {
        let pv = cashflows[p] * df_step.powi(ex_time[p] as i32);
        sum += pv;
        sum_sq += pv * pv;
    }

    let n = n_paths as f64;
    let mean = sum / n;
    let variance = (sum_sq / n - mean * mean).max(0.0);
    let std_error = (variance / n).sqrt();

    McBasketResult {
        price: mean,
        std_error,
    }
}

#[allow(clippy::needless_range_loop)]
fn solve_3x3(a: &[[f64; 3]; 3], b: &[f64; 3]) -> [f64; 3] {
    let mut m = [[0.0; 4]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = a[i][j];
        }
        m[i][3] = b[i];
    }
    // Forward elimination
    for col in 0..3 {
        let mut max_row = col;
        for row in col + 1..3 {
            if m[row][col].abs() > m[max_row][col].abs() {
                max_row = row;
            }
        }
        m.swap(col, max_row);
        if m[col][col].abs() < 1e-14 { continue; }
        for row in col + 1..3 {
            let f = m[row][col] / m[col][col];
            for j in col..4 {
                m[row][j] -= f * m[col][j];
            }
        }
    }
    // Back substitution
    let mut x = [0.0; 3];
    for i in (0..3).rev() {
        if m[i][i].abs() < 1e-14 { continue; }
        x[i] = m[i][3];
        for j in i + 1..3 {
            x[i] -= m[i][j] * x[j];
        }
        x[i] /= m[i][i];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mc_european_basket_call() {
        let corr = vec![
            vec![1.0, 0.5],
            vec![0.5, 1.0],
        ];
        let res = mc_european_basket(
            &[100.0, 100.0], 100.0, 0.05, &[0.0, 0.0],
            &[0.20, 0.25], &corr, &[0.5, 0.5],
            1.0, BasketPayoffType::WeightedSum, true,
            50000, Some(42),
        );
        assert!(res.price > 5.0 && res.price < 15.0, "price={}", res.price);
    }

    #[test]
    fn test_mc_european_basket_max() {
        let corr = vec![
            vec![1.0, 0.3],
            vec![0.3, 1.0],
        ];
        let res = mc_european_basket(
            &[100.0, 100.0], 100.0, 0.05, &[0.0, 0.0],
            &[0.20, 0.20], &corr, &[0.5, 0.5],
            1.0, BasketPayoffType::MaxOfAssets, true,
            50000, Some(42),
        );
        // Max call > single call
        assert!(res.price > 10.0, "price={}", res.price);
    }

    #[test]
    fn test_mc_american_basket_put() {
        let corr = vec![
            vec![1.0, 0.5],
            vec![0.5, 1.0],
        ];
        let res = mc_american_basket(
            &[100.0, 100.0], 100.0, 0.05, &[0.0, 0.0],
            &[0.20, 0.25], &corr, &[0.5, 0.5],
            1.0, BasketPayoffType::WeightedSum, false,
            50, 20000, Some(42),
        );
        assert!(res.price > 0.0, "price={}", res.price);
    }

    #[test]
    fn test_mc_basket_spread() {
        let corr = vec![
            vec![1.0, 0.7],
            vec![0.7, 1.0],
        ];
        let res = mc_european_basket(
            &[100.0, 95.0], 5.0, 0.05, &[0.0, 0.0],
            &[0.20, 0.25], &corr, &[1.0, -1.0],
            1.0, BasketPayoffType::Spread, true,
            50000, Some(42),
        );
        assert!(res.price > 0.0, "price={}", res.price);
    }
}
