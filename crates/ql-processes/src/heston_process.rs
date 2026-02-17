//! Heston stochastic volatility process.
//!
//! The Heston model describes the joint dynamics of spot price S and variance v:
//!
//!   dS = (r − q) S dt + √v S dW₁
//!   dv = κ(θ − v) dt + σ √v dW₂
//!   dW₁ dW₂ = ρ dt
//!
//! where κ is the mean-reversion speed, θ is the long-run variance,
//! σ is the vol-of-vol, and ρ is the correlation.

use nalgebra::{DMatrix, DVector};

use crate::process::StochasticProcess;

/// Parameters for the Heston process.
#[derive(Clone, Debug)]
pub struct HestonProcess {
    /// Initial spot price.
    pub s0: f64,
    /// Risk-free rate (constant).
    pub risk_free_rate: f64,
    /// Dividend yield (constant).
    pub dividend_yield: f64,
    /// Initial variance v(0).
    pub v0: f64,
    /// Mean-reversion speed κ.
    pub kappa: f64,
    /// Long-run variance θ.
    pub theta: f64,
    /// Vol-of-vol σ.
    pub sigma: f64,
    /// Correlation ρ between spot and variance Brownian motions.
    pub rho: f64,
}

impl HestonProcess {
    /// Create a new Heston process.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        s0: f64,
        risk_free_rate: f64,
        dividend_yield: f64,
        v0: f64,
        kappa: f64,
        theta: f64,
        sigma: f64,
        rho: f64,
    ) -> Self {
        Self {
            s0,
            risk_free_rate,
            dividend_yield,
            v0,
            kappa,
            theta,
            sigma,
            rho,
        }
    }

    /// Check the Feller condition: 2κθ > σ².
    ///
    /// When satisfied, the variance process stays strictly positive.
    pub fn feller_satisfied(&self) -> bool {
        2.0 * self.kappa * self.theta > self.sigma * self.sigma
    }

    /// Quadratic-exponential (QE) discretization for variance.
    ///
    /// This is the Andersen (2008) QE scheme which handles the variance
    /// process more accurately than Euler, especially near zero.
    ///
    /// Returns (S(t+dt), v(t+dt)) given (S(t), v(t)) and two independent
    /// uniform random variables u1, u2 ∈ (0, 1).
    pub fn evolve_qe(
        &self,
        _t: f64,
        s: f64,
        v: f64,
        dt: f64,
        z1: f64,
        z2: f64,
    ) -> (f64, f64) {
        let kappa = self.kappa;
        let theta = self.theta;
        let sigma = self.sigma;
        let rho = self.rho;
        let r = self.risk_free_rate;
        let q = self.dividend_yield;

        // --- Variance step (QE scheme) ---
        let e_kdt = (-kappa * dt).exp();
        let m = theta + (v - theta) * e_kdt;
        let s2 = v * sigma * sigma * e_kdt * (1.0 - e_kdt) / kappa
            + theta * sigma * sigma * (1.0 - e_kdt).powi(2) / (2.0 * kappa);

        let psi = s2 / (m * m).max(1e-30);
        let psi_crit = 1.5;

        let v_next = if psi <= psi_crit {
            // Use moment-matched non-central chi-squared via inverse CDF
            let b2 = 2.0 / psi - 1.0 + (2.0 / psi).sqrt() * (2.0 / psi - 1.0).max(0.0).sqrt();
            let a = m / (1.0 + b2);
            let b = b2.sqrt();
            a * (b + z2).powi(2)
        } else {
            // Exponential approximation
            let p = (psi - 1.0) / (psi + 1.0);
            let beta = (1.0 - p) / m.max(1e-30);
            // Convert z2 (standard normal) to uniform
            let u = ql_math::distributions::cumulative_normal(z2);
            if u <= p {
                0.0
            } else {
                (1.0 / beta) * (-(1.0 - (u - p) / (1.0 - p)).max(1e-30).ln())
            }
        };

        // --- Spot step (log-Euler with exact conditional integral) ---
        let k0 = (r - q) * dt - rho * kappa * theta * dt / sigma;
        let k1 = 0.5 * dt * (rho * kappa / sigma - 0.5) - rho / sigma;
        let k2 = 0.5 * dt * (rho * kappa / sigma - 0.5) + rho / sigma;
        let k3 = 0.5 * dt * (1.0 - rho * rho);

        let ln_s_next =
            s.ln() + k0 + k1 * v + k2 * v_next + (k3 * (v + v_next).max(0.0)).sqrt() * z1;

        (ln_s_next.exp(), v_next.max(0.0))
    }
}

impl StochasticProcess for HestonProcess {
    fn size(&self) -> usize {
        2
    }

    fn factors(&self) -> usize {
        2
    }

    fn initial_values(&self) -> DVector<f64> {
        DVector::from_vec(vec![self.s0, self.v0])
    }

    /// Drift vector: [(r-q)*S, κ(θ-v)].
    fn drift(&self, _t: f64, x: &DVector<f64>) -> DVector<f64> {
        let s = x[0];
        let v = x[1].max(0.0);
        DVector::from_vec(vec![
            (self.risk_free_rate - self.dividend_yield) * s,
            self.kappa * (self.theta - v),
        ])
    }

    /// Diffusion matrix incorporating correlation ρ.
    ///
    /// We decompose dW = L dZ where L is the Cholesky factor:
    /// L = [1    0  ]  so  σ_matrix = [√v*S   0        ]
    ///     [ρ  √(1-ρ²)]              [σ√v*ρ  σ√v*√(1-ρ²)]
    fn diffusion(&self, _t: f64, x: &DVector<f64>) -> DMatrix<f64> {
        let s = x[0];
        let v = x[1].max(0.0);
        let sqrt_v = v.sqrt();
        let rho2 = (1.0 - self.rho * self.rho).max(0.0).sqrt();

        DMatrix::from_row_slice(
            2,
            2,
            &[
                sqrt_v * s,
                0.0,
                self.sigma * sqrt_v * self.rho,
                self.sigma * sqrt_v * rho2,
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_heston() -> HestonProcess {
        HestonProcess::new(
            100.0, // s0
            0.05,  // r
            0.02,  // q
            0.04,  // v0
            1.5,   // kappa
            0.04,  // theta
            0.3,   // sigma (vol of vol)
            -0.7,  // rho
        )
    }

    #[test]
    fn heston_initial_values() {
        let h = make_heston();
        let iv = h.initial_values();
        assert_abs_diff_eq!(iv[0], 100.0);
        assert_abs_diff_eq!(iv[1], 0.04);
    }

    #[test]
    fn heston_drift() {
        let h = make_heston();
        let x = DVector::from_vec(vec![100.0, 0.04]);
        let d = h.drift(0.0, &x);
        // S drift: (0.05 - 0.02) * 100 = 3.0
        assert_abs_diff_eq!(d[0], 3.0, epsilon = 1e-12);
        // v drift: 1.5 * (0.04 - 0.04) = 0.0
        assert_abs_diff_eq!(d[1], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn heston_diffusion_shape() {
        let h = make_heston();
        let x = DVector::from_vec(vec![100.0, 0.04]);
        let diff = h.diffusion(0.0, &x);
        assert_eq!(diff.nrows(), 2);
        assert_eq!(diff.ncols(), 2);
        // diff[0][0] = sqrt(0.04) * 100 = 20.0
        assert_abs_diff_eq!(diff[(0, 0)], 20.0, epsilon = 1e-12);
        // diff[0][1] = 0 (upper-right of Cholesky)
        assert_abs_diff_eq!(diff[(0, 1)], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn heston_feller_condition() {
        let h = make_heston();
        // 2 * 1.5 * 0.04 = 0.12 > 0.09 = 0.3^2
        assert!(h.feller_satisfied());

        let h2 = HestonProcess::new(100.0, 0.05, 0.0, 0.04, 0.5, 0.02, 0.8, -0.7);
        // 2 * 0.5 * 0.02 = 0.02 < 0.64 = 0.8^2
        assert!(!h2.feller_satisfied());
    }

    #[test]
    fn heston_qe_positive_variance() {
        let h = make_heston();
        // Even with adverse random draws, variance should stay non-negative
        let (s, v) = h.evolve_qe(0.0, 100.0, 0.04, 0.01, -2.0, -2.0);
        assert!(s > 0.0, "Spot should be positive");
        assert!(v >= 0.0, "Variance should be non-negative");
    }

    #[test]
    fn heston_euler_evolve_zero_noise() {
        let h = make_heston();
        let x0 = h.initial_values();
        let dw = DVector::from_vec(vec![0.0, 0.0]);
        let x1 = h.evolve(0.0, &x0, 0.01, &dw);
        // S drift: 100 + 3.0 * 0.01 = 100.03
        assert_abs_diff_eq!(x1[0], 100.03, epsilon = 1e-10);
        // v drift: 0.04 + 0.0 * 0.01 = 0.04
        assert_abs_diff_eq!(x1[1], 0.04, epsilon = 1e-10);
    }

    #[test]
    fn heston_size_and_factors() {
        let h = make_heston();
        assert_eq!(h.size(), 2);
        assert_eq!(h.factors(), 2);
    }
}
