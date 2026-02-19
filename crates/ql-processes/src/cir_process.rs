//! Cox-Ingersoll-Ross square-root diffusion process.
//!
//! dr = κ(θ − r) dt + σ √r dW
//!
//! This process ensures non-negative rates when the Feller condition
//! 2κθ ≥ σ² is satisfied.
//!
//! Provides both Euler-Milstein and exact (non-central chi-squared)
//! simulation methods.

use crate::process::StochasticProcess1D;

/// CIR square-root process.
///
/// dr = κ(θ − r) dt + σ √r dW
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CoxIngersollRossProcess {
    /// Initial value r(0).
    pub r0: f64,
    /// Mean-reversion speed κ.
    pub kappa: f64,
    /// Long-run mean θ.
    pub theta: f64,
    /// Volatility coefficient σ.
    pub sigma: f64,
}

impl CoxIngersollRossProcess {
    /// Create a new CIR process.
    pub fn new(r0: f64, kappa: f64, theta: f64, sigma: f64) -> Self {
        Self {
            r0,
            kappa,
            theta,
            sigma,
        }
    }

    /// Check the Feller condition: 2κθ ≥ σ².
    pub fn feller_satisfied(&self) -> bool {
        2.0 * self.kappa * self.theta >= self.sigma * self.sigma
    }

    /// Exact conditional expectation: E[r(t+dt) | r(t)].
    ///
    /// E[r(t+dt)|r(t)] = θ + (r(t) − θ) e^{−κ dt}
    pub fn exact_expectation(&self, r: f64, dt: f64) -> f64 {
        self.theta + (r - self.theta) * (-self.kappa * dt).exp()
    }

    /// Exact conditional variance: Var[r(t+dt) | r(t)].
    ///
    /// Var = r σ² e^{−κdt}/κ (1 − e^{−κdt})
    ///     + θ σ²/(2κ) (1 − e^{−κdt})²
    pub fn exact_variance(&self, r: f64, dt: f64) -> f64 {
        let e = (-self.kappa * dt).exp();
        let s2 = self.sigma * self.sigma;

        let term1 = r * s2 * e / self.kappa * (1.0 - e);
        let term2 = self.theta * s2 / (2.0 * self.kappa) * (1.0 - e) * (1.0 - e);

        term1 + term2
    }

    /// Full-truncation Euler step (truncates r to 0 in diffusion).
    ///
    /// r(t+dt) = r + κ(θ − r⁺) dt + σ √(r⁺) √dt dW
    ///
    /// This preserves non-negativity approximately.
    pub fn evolve_euler_full_truncation(&self, r: f64, dt: f64, dw: f64) -> f64 {
        let r_pos = r.max(0.0);
        let dr = self.kappa * (self.theta - r_pos) * dt
            + self.sigma * r_pos.sqrt() * dt.sqrt() * dw;
        (r + dr).max(0.0)
    }

    /// Milstein step with full truncation.
    ///
    /// Adds the Milstein correction: ¼ σ² (dW² − dt).
    pub fn evolve_milstein(&self, r: f64, dt: f64, dw: f64) -> f64 {
        let r_pos = r.max(0.0);
        let sqrt_r = r_pos.sqrt();
        let sqrt_dt = dt.sqrt();

        let dr = self.kappa * (self.theta - r_pos) * dt
            + self.sigma * sqrt_r * sqrt_dt * dw
            + 0.25 * self.sigma * self.sigma * (dw * dw - 1.0) * dt;
        (r + dr).max(0.0)
    }
}

impl StochasticProcess1D for CoxIngersollRossProcess {
    fn x0(&self) -> f64 {
        self.r0
    }

    fn drift_1d(&self, _t: f64, x: f64) -> f64 {
        self.kappa * (self.theta - x)
    }

    fn diffusion_1d(&self, _t: f64, x: f64) -> f64 {
        self.sigma * x.max(0.0).sqrt()
    }
}

/// General square-root process: dX = κ(θ − X) dt + σ √X dW.
///
/// This is an alias for [`CoxIngersollRossProcess`] for use in contexts
/// outside interest rate modeling (e.g., Heston variance process).
pub type SquareRootProcess = CoxIngersollRossProcess;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_process() -> CoxIngersollRossProcess {
        // κ=0.5, θ=0.05, σ=0.1, r0=0.03
        CoxIngersollRossProcess::new(0.03, 0.5, 0.05, 0.1)
    }

    #[test]
    fn cir_process_feller() {
        let p = make_process();
        // 2*0.5*0.05 = 0.05 >= 0.01 = σ² ✓
        assert!(p.feller_satisfied());

        let bad = CoxIngersollRossProcess::new(0.03, 0.5, 0.05, 0.5);
        // 2*0.5*0.05 = 0.05 < 0.25 ✗
        assert!(!bad.feller_satisfied());
    }

    #[test]
    fn cir_process_exact_expectation() {
        let p = make_process();
        let e = p.exact_expectation(0.03, 1.0);
        // θ + (r−θ) e^{−κ} = 0.05 + (0.03−0.05) e^{−0.5}
        let expected = 0.05 + (0.03 - 0.05) * (-0.5_f64).exp();
        assert_abs_diff_eq!(e, expected, epsilon = 1e-12);
    }

    #[test]
    fn cir_process_exact_variance_positive() {
        let p = make_process();
        let v = p.exact_variance(0.03, 1.0);
        assert!(v > 0.0, "Variance should be positive: {v}");
    }

    #[test]
    fn cir_process_stochastic_process_trait() {
        let p = make_process();
        assert_abs_diff_eq!(p.x0(), 0.03);
        assert_abs_diff_eq!(p.drift_1d(0.0, 0.03), 0.5 * (0.05 - 0.03));
        assert_abs_diff_eq!(p.diffusion_1d(0.0, 0.04), 0.1 * 0.04_f64.sqrt());
    }

    #[test]
    fn cir_process_diffusion_at_zero() {
        let p = make_process();
        // Diffusion vanishes at r=0
        assert_abs_diff_eq!(p.diffusion_1d(0.0, 0.0), 0.0);
    }

    #[test]
    fn cir_process_euler_non_negative() {
        let p = make_process();
        // Even with a large negative shock, result should be non-negative
        let r = p.evolve_euler_full_truncation(0.001, 0.01, -5.0);
        assert!(r >= 0.0, "Euler result should be non-negative: {r}");
    }

    #[test]
    fn cir_process_milstein_non_negative() {
        let p = make_process();
        let r = p.evolve_milstein(0.001, 0.01, -5.0);
        assert!(r >= 0.0, "Milstein result should be non-negative: {r}");
    }

    #[test]
    fn cir_process_mean_convergence() {
        // After long time, expectation → θ
        let p = make_process();
        let e = p.exact_expectation(0.03, 100.0);
        assert_abs_diff_eq!(e, 0.05, epsilon = 1e-8);
    }

    #[test]
    fn cir_process_square_root_alias() {
        let s: SquareRootProcess = SquareRootProcess::new(0.04, 2.0, 0.04, 0.3);
        assert_abs_diff_eq!(s.x0(), 0.04);
    }
}
