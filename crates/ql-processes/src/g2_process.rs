//! G2++ (two-factor Gaussian) short-rate process.
//!
//! The G2++ model describes the short rate as $r(t) = x(t) + y(t) + \varphi(t)$,
//! where $(x, y)$ follow correlated Ornstein-Uhlenbeck processes:
//!
//! $$dx = -a\,x\,dt + \sigma\,dW_1$$
//! $$dy = -b\,y\,dt + \eta\,dW_2$$
//! $$dW_1\,dW_2 = \rho\,dt$$
//!
//! The deterministic shift $\varphi(t)$ is calibrated to the initial term
//! structure (not stored here — see `G2Model` in `ql-models`).

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::process::StochasticProcess;

/// G2++ two-factor Gaussian short-rate process.
///
/// State vector: `[x, y]` where `x(0) = y(0) = 0`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct G2Process {
    /// Mean-reversion speed of factor x.
    pub a: f64,
    /// Volatility of factor x.
    pub sigma: f64,
    /// Mean-reversion speed of factor y.
    pub b: f64,
    /// Volatility of factor y.
    pub eta: f64,
    /// Instantaneous correlation between factors.
    pub rho: f64,
}

impl G2Process {
    /// Create a new G2++ process.
    ///
    /// # Parameters
    /// - `a` — mean-reversion speed of x (> 0)
    /// - `sigma` — volatility of x (> 0)
    /// - `b` — mean-reversion speed of y (> 0)
    /// - `eta` — volatility of y (> 0)
    /// - `rho` — correlation ∈ (−1, 1)
    pub fn new(a: f64, sigma: f64, b: f64, eta: f64, rho: f64) -> Self {
        assert!(a > 0.0, "a must be positive");
        assert!(sigma > 0.0, "sigma must be positive");
        assert!(b > 0.0, "b must be positive");
        assert!(eta > 0.0, "eta must be positive");
        assert!(rho.abs() < 1.0, "rho must be in (-1, 1)");
        Self { a, sigma, b, eta, rho }
    }

    // ── Analytical moments ──────────────────────────────

    /// $E[x(t) | x(0) = x_0]$
    pub fn mean_x(&self, x0: f64, t: f64) -> f64 {
        x0 * (-self.a * t).exp()
    }

    /// $E[y(t) | y(0) = y_0]$
    pub fn mean_y(&self, y0: f64, t: f64) -> f64 {
        y0 * (-self.b * t).exp()
    }

    /// $\text{Var}[x(t)]$ (unconditional from $x(0) = 0$).
    pub fn variance_x(&self, t: f64) -> f64 {
        self.sigma * self.sigma / (2.0 * self.a) * (1.0 - (-2.0 * self.a * t).exp())
    }

    /// $\text{Var}[y(t)]$ (unconditional from $y(0) = 0$).
    pub fn variance_y(&self, t: f64) -> f64 {
        self.eta * self.eta / (2.0 * self.b) * (1.0 - (-2.0 * self.b * t).exp())
    }

    /// $\text{Cov}[x(t), y(t)]$ (unconditional from $x(0) = y(0) = 0$).
    pub fn covariance_xy(&self, t: f64) -> f64 {
        self.rho * self.sigma * self.eta / (self.a + self.b)
            * (1.0 - (-(self.a + self.b) * t).exp())
    }

    /// Full 2×2 covariance matrix of $[x(t), y(t)]$.
    pub fn covariance_matrix(&self, t: f64) -> DMatrix<f64> {
        let vx = self.variance_x(t);
        let vy = self.variance_y(t);
        let cov = self.covariance_xy(t);
        DMatrix::from_row_slice(2, 2, &[vx, cov, cov, vy])
    }

    /// $B_a(\tau) = (1 - e^{-a\tau})/a$ — used in bond pricing.
    pub fn b_a(&self, tau: f64) -> f64 {
        if self.a.abs() < 1e-15 {
            tau
        } else {
            (1.0 - (-self.a * tau).exp()) / self.a
        }
    }

    /// $B_b(\tau) = (1 - e^{-b\tau})/b$ — used in bond pricing.
    pub fn b_b(&self, tau: f64) -> f64 {
        if self.b.abs() < 1e-15 {
            tau
        } else {
            (1.0 - (-self.b * tau).exp()) / self.b
        }
    }

    /// Exact simulation step: draw `[x(t+dt), y(t+dt)]` from the conditional
    /// Gaussian distribution given `[x(t), y(t)]` and standard normal
    /// increments `[z1, z2]`.
    pub fn evolve_exact(
        &self,
        x: f64,
        y: f64,
        dt: f64,
        z1: f64,
        z2: f64,
    ) -> (f64, f64) {
        let mx = self.mean_x(x, dt);
        let my = self.mean_y(y, dt);

        // Conditional variances (from t to t+dt)
        let vx = self.sigma * self.sigma / (2.0 * self.a) * (1.0 - (-2.0 * self.a * dt).exp());
        let vy = self.eta * self.eta / (2.0 * self.b) * (1.0 - (-2.0 * self.b * dt).exp());
        let cov = self.rho * self.sigma * self.eta / (self.a + self.b)
            * (1.0 - (-(self.a + self.b) * dt).exp());

        let sx = vx.sqrt();
        let sy = vy.sqrt();
        let rho_cond = if (sx * sy) > 1e-20 { cov / (sx * sy) } else { 0.0 };

        // Cholesky-style: x_new = mx + sx * z1
        //                  y_new = my + sy * (rho_cond * z1 + sqrt(1 - rho^2) * z2)
        let x_new = mx + sx * z1;
        let y_new = my + sy * (rho_cond * z1 + (1.0 - rho_cond * rho_cond).max(0.0).sqrt() * z2);

        (x_new, y_new)
    }
}

impl StochasticProcess for G2Process {
    fn size(&self) -> usize { 2 }

    fn factors(&self) -> usize { 2 }

    fn initial_values(&self) -> DVector<f64> {
        DVector::from_element(2, 0.0)
    }

    fn drift(&self, _t: f64, x: &DVector<f64>) -> DVector<f64> {
        DVector::from_vec(vec![-self.a * x[0], -self.b * x[1]])
    }

    fn diffusion(&self, _t: f64, _x: &DVector<f64>) -> DMatrix<f64> {
        // Cholesky factor of the correlation matrix, scaled by volatilities:
        // [[sigma, 0], [rho*eta, eta*sqrt(1-rho^2)]]
        let s11 = self.sigma;
        let s21 = self.rho * self.eta;
        let s22 = self.eta * (1.0 - self.rho * self.rho).max(0.0).sqrt();
        DMatrix::from_row_slice(2, 2, &[s11, 0.0, s21, s22])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn sample_process() -> G2Process {
        G2Process::new(0.1, 0.01, 0.15, 0.012, -0.3)
    }

    #[test]
    fn construction() {
        let p = sample_process();
        assert_abs_diff_eq!(p.a, 0.1, epsilon = 1e-15);
        assert_abs_diff_eq!(p.rho, -0.3, epsilon = 1e-15);
    }

    #[test]
    fn mean_reverts_to_zero() {
        let p = sample_process();
        let mx = p.mean_x(1.0, 100.0);
        let my = p.mean_y(1.0, 100.0);
        assert!(mx.abs() < 1e-3, "x should mean-revert to 0");
        assert!(my.abs() < 1e-3, "y should mean-revert to 0");
    }

    #[test]
    fn variance_increases_and_saturates() {
        let p = sample_process();
        let v1 = p.variance_x(1.0);
        let v5 = p.variance_x(5.0);
        let v100 = p.variance_x(100.0);
        assert!(v5 > v1);
        // Should saturate at sigma^2 / (2*a)
        let limit = p.sigma * p.sigma / (2.0 * p.a);
        assert_abs_diff_eq!(v100, limit, epsilon = 1e-10);
    }

    #[test]
    fn covariance_sign_matches_rho() {
        let p = sample_process();
        let cov = p.covariance_xy(2.0);
        // rho = -0.3, so covariance should be negative
        assert!(cov < 0.0);
    }

    #[test]
    fn b_a_limit() {
        let p = sample_process();
        // B_a(0) = 0 (limit)
        assert_abs_diff_eq!(p.b_a(0.0), 0.0, epsilon = 1e-12);
        // B_a(large) → 1/a
        assert_abs_diff_eq!(p.b_a(100.0), 1.0 / p.a, epsilon = 1e-3);
    }

    #[test]
    fn stochastic_process_trait() {
        let p = sample_process();
        assert_eq!(p.size(), 2);
        assert_eq!(p.factors(), 2);

        let x0 = p.initial_values();
        assert_abs_diff_eq!(x0[0], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(x0[1], 0.0, epsilon = 1e-15);

        let drift = p.drift(0.0, &x0);
        assert_abs_diff_eq!(drift[0], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(drift[1], 0.0, epsilon = 1e-15);

        let diff = p.diffusion(0.0, &x0);
        assert_eq!(diff.nrows(), 2);
        assert_eq!(diff.ncols(), 2);
        assert_abs_diff_eq!(diff[(0, 0)], p.sigma, epsilon = 1e-15);
        assert_abs_diff_eq!(diff[(0, 1)], 0.0, epsilon = 1e-15);
    }

    #[test]
    fn exact_evolve_mean() {
        let p = sample_process();
        // From (0,0) evolve with z1=z2=0 (no noise) → should stay at (0,0)
        let (x, y) = p.evolve_exact(0.0, 0.0, 1.0, 0.0, 0.0);
        assert_abs_diff_eq!(x, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(y, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn covariance_matrix_symmetric() {
        let p = sample_process();
        let cov = p.covariance_matrix(3.0);
        assert_abs_diff_eq!(cov[(0, 1)], cov[(1, 0)], epsilon = 1e-15);
    }

    #[test]
    fn clone_roundtrip() {
        let p = sample_process();
        let p2 = p.clone();
        assert_abs_diff_eq!(p.a, p2.a, epsilon = 1e-15);
        assert_abs_diff_eq!(p.sigma, p2.sigma, epsilon = 1e-15);
        assert_abs_diff_eq!(p.b, p2.b, epsilon = 1e-15);
        assert_abs_diff_eq!(p.eta, p2.eta, epsilon = 1e-15);
        assert_abs_diff_eq!(p.rho, p2.rho, epsilon = 1e-15);
    }
}
