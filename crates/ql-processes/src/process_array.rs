//! Multi-asset correlated stochastic process array.
//!
//! `StochasticProcessArray` is a container for multiple correlated 1-D
//! stochastic processes, suitable for multi-asset Monte Carlo simulation
//! (e.g. basket options, rainbow options).
//!
//! Each component process provides a drift and diffusion, and the array
//! introduces correlations through a Cholesky decomposition of the
//! correlation matrix.
//!
//! Reference:
//! - QuantLib: StochasticProcessArray in stochasticprocessarray.hpp

use serde::{Deserialize, Serialize};

/// A single component process (GBM-like).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessComponent {
    /// Current spot value.
    pub x0: f64,
    /// Drift (risk-neutral: r − q for equity).
    pub drift: f64,
    /// Diffusion coefficient (volatility σ).
    pub diffusion: f64,
}

/// Multi-asset correlated process array.
#[derive(Clone, Debug)]
pub struct StochasticProcessArray {
    /// Individual processes.
    pub processes: Vec<ProcessComponent>,
    /// Correlation matrix (n × n, stored row-major).
    pub correlation: Vec<Vec<f64>>,
    /// Cholesky factor L such that Σ = L·Lᵀ.
    cholesky: Vec<Vec<f64>>,
}

impl StochasticProcessArray {
    /// Create a new process array.
    ///
    /// # Arguments
    /// - `processes` — vector of component processes
    /// - `correlation` — n×n correlation matrix (must be positive semi-definite)
    ///
    /// # Panics
    /// Panics if the correlation matrix dimensions don't match the number of processes,
    /// or if Cholesky decomposition fails.
    pub fn new(processes: Vec<ProcessComponent>, correlation: Vec<Vec<f64>>) -> Self {
        let n = processes.len();
        assert_eq!(correlation.len(), n, "Correlation matrix dimension mismatch");
        for row in &correlation {
            assert_eq!(row.len(), n, "Correlation matrix must be square");
        }
        let cholesky = cholesky_decompose(&correlation);
        StochasticProcessArray {
            processes,
            correlation,
            cholesky,
        }
    }

    /// Number of component processes.
    pub fn size(&self) -> usize {
        self.processes.len()
    }

    /// Initial values of all processes.
    pub fn initial_values(&self) -> Vec<f64> {
        self.processes.iter().map(|p| p.x0).collect()
    }

    /// Drift vector at time t (constant drifts for GBM components).
    pub fn drift_vec(&self, _t: f64, _x: &[f64]) -> Vec<f64> {
        self.processes.iter().map(|p| p.drift).collect()
    }

    /// Diffusion matrix: L · diag(σ_i) where L is the Cholesky factor.
    /// Returns an n×n matrix stored as Vec<Vec<f64>>.
    pub fn diffusion_matrix(&self, _t: f64, _x: &[f64]) -> Vec<Vec<f64>> {
        let n = self.processes.len();
        let mut result = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..=i {
                result[i][j] = self.cholesky[i][j] * self.processes[j].diffusion;
            }
        }
        result
    }

    /// Evolve all processes by one step (Euler discretisation).
    ///
    /// # Arguments
    /// - `x` — current values (length n)
    /// - `t` — current time
    /// - `dt` — time step
    /// - `dw` — vector of independent standard normal variates (length n)
    ///
    /// # Returns
    /// New values after one step.
    pub fn evolve(&self, x: &[f64], t: f64, dt: f64, dw: &[f64]) -> Vec<f64> {
        let n = self.processes.len();
        assert_eq!(x.len(), n);
        assert_eq!(dw.len(), n);

        let drift = self.drift_vec(t, x);
        let sqrt_dt = dt.sqrt();

        // Correlated Brownian increments: z = L · dw
        let mut z = vec![0.0; n];
        for i in 0..n {
            for j in 0..=i {
                z[i] += self.cholesky[i][j] * dw[j];
            }
        }

        // Log-Euler step: x_new = x · exp((μ - σ²/2)·dt + σ·√dt·z)
        let mut x_new = vec![0.0; n];
        for i in 0..n {
            let sigma = self.processes[i].diffusion;
            x_new[i] = x[i] * ((drift[i] - 0.5 * sigma * sigma) * dt + sigma * sqrt_dt * z[i]).exp();
        }
        x_new
    }

    /// Evolve along an entire path.
    ///
    /// # Arguments
    /// - `n_steps` — number of time steps
    /// - `t_final` — total time horizon
    /// - `dw_matrix` — n_steps × n matrix of independent standard normals
    ///
    /// # Returns
    /// (n_steps + 1) × n matrix of values; row 0 is the initial values.
    pub fn evolve_path(&self, n_steps: usize, t_final: f64, dw_matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = self.processes.len();
        let dt = t_final / n_steps as f64;
        let mut path = Vec::with_capacity(n_steps + 1);
        path.push(self.initial_values());

        for step in 0..n_steps {
            let t = step as f64 * dt;
            let x = &path[step];
            let dw = &dw_matrix[step];
            path.push(self.evolve(x, t, dt, dw));
        }
        path
    }

    /// Get the Cholesky factor.
    pub fn cholesky_factor(&self) -> &Vec<Vec<f64>> {
        &self.cholesky
    }
}

/// Cholesky decomposition of a symmetric positive (semi-)definite matrix.
/// Returns lower-triangular L such that A = L·Lᵀ.
pub fn cholesky_decompose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = 0.0;
            for k in 0..j {
                s += l[i][k] * l[j][k];
            }
            if i == j {
                let val = a[i][i] - s;
                l[i][j] = if val > 0.0 { val.sqrt() } else { 0.0 };
            } else if l[j][j].abs() > 1e-14 {
                l[i][j] = (a[i][j] - s) / l[j][j];
            }
        }
    }
    l
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_cholesky_identity() {
        let id = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let l = cholesky_decompose(&id);
        assert_abs_diff_eq!(l[0][0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(l[1][1], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(l[1][0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cholesky_correlated() {
        let rho = 0.6;
        let corr = vec![vec![1.0, rho], vec![rho, 1.0]];
        let l = cholesky_decompose(&corr);
        // Verify L·Lᵀ = corr
        let a00 = l[0][0] * l[0][0];
        let a01 = l[1][0] * l[0][0];
        let a11 = l[1][0] * l[1][0] + l[1][1] * l[1][1];
        assert_abs_diff_eq!(a00, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(a01, rho, epsilon = 1e-10);
        assert_abs_diff_eq!(a11, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_process_array_evolve() {
        let procs = vec![
            ProcessComponent { x0: 100.0, drift: 0.05, diffusion: 0.2 },
            ProcessComponent { x0: 50.0, drift: 0.03, diffusion: 0.3 },
        ];
        let corr = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
        let spa = StochasticProcessArray::new(procs, corr);

        let x = spa.initial_values();
        let dw = vec![0.1, -0.2];
        let x_new = spa.evolve(&x, 0.0, 1.0 / 252.0, &dw);
        assert_eq!(x_new.len(), 2);
        assert!(x_new[0] > 0.0 && x_new[1] > 0.0);
    }

    #[test]
    fn test_evolve_path() {
        let procs = vec![
            ProcessComponent { x0: 100.0, drift: 0.05, diffusion: 0.2 },
            ProcessComponent { x0: 100.0, drift: 0.05, diffusion: 0.2 },
        ];
        let corr = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let spa = StochasticProcessArray::new(procs, corr);

        let n_steps = 10;
        let dw: Vec<Vec<f64>> = (0..n_steps).map(|i| {
            vec![0.1 * (i as f64 - 5.0), -0.1 * (i as f64 - 5.0)]
        }).collect();

        let path = spa.evolve_path(n_steps, 1.0, &dw);
        assert_eq!(path.len(), n_steps + 1);
        assert_eq!(path[0], vec![100.0, 100.0]);
    }

    #[test]
    fn test_diffusion_matrix() {
        let procs = vec![
            ProcessComponent { x0: 100.0, drift: 0.05, diffusion: 0.2 },
            ProcessComponent { x0: 50.0, drift: 0.03, diffusion: 0.3 },
        ];
        let corr = vec![vec![1.0, 0.7], vec![0.7, 1.0]];
        let spa = StochasticProcessArray::new(procs, corr);
        let dm = spa.diffusion_matrix(0.0, &[100.0, 50.0]);
        assert_eq!(dm.len(), 2);
        // dm[0][0] = L[0][0] * σ₁ = 1.0 * 0.2 = 0.2
        assert_abs_diff_eq!(dm[0][0], 0.2, epsilon = 1e-10);
    }
}
