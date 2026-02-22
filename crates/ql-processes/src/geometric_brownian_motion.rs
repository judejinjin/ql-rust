//! Geometric Brownian Motion process.
//!
//! A pure GBM with constant drift and volatility:
//!
//!   dS = μ S dt + σ S dW
//!
//! Unlike [`GeneralizedBlackScholesProcess`], this exposes a simple (μ, σ)
//! pair rather than the (r, q, σ(t)) market-data structure. It is useful
//! for educational examples and as the building block for basket / Rainbow
//! option Monte Carlo paths (where each asset has an independent drift).

use crate::process::StochasticProcess1D;

/// Simple constant-parameter Geometric Brownian Motion.
///
/// dS = μ S dt + σ S dW
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GeometricBrownianMotionProcess {
    /// Initial spot price S(0).
    pub s0: f64,
    /// Drift μ (e.g. r − q under risk-neutral measure).
    pub mu: f64,
    /// Constant volatility σ > 0.
    pub sigma: f64,
}

impl GeometricBrownianMotionProcess {
    /// Create a new GBM process.
    ///
    /// # Panics
    /// Panics if `s0 <= 0` or `sigma <= 0`.
    pub fn new(s0: f64, mu: f64, sigma: f64) -> Self {
        assert!(s0 > 0.0, "s0 must be positive");
        assert!(sigma > 0.0, "sigma must be positive");
        Self { s0, mu, sigma }
    }

    /// Exact simulation via the log-normal transition density.
    ///
    /// Returns S(t + dt) given S(t) = s and a standard normal draw z.
    /// Uses the exact solution:
    ///
    ///   S(t+dt) = S(t) · exp((μ − σ²/2) dt + σ √dt · z)
    pub fn evolve_exact(&self, s: f64, dt: f64, z: f64) -> f64 {
        s * ((self.mu - 0.5 * self.sigma * self.sigma) * dt
            + self.sigma * dt.sqrt() * z)
            .exp()
    }

    /// Log-return over interval dt for a standard normal draw z.
    pub fn log_return(&self, dt: f64, z: f64) -> f64 {
        (self.mu - 0.5 * self.sigma * self.sigma) * dt
            + self.sigma * dt.sqrt() * z
    }
}

impl StochasticProcess1D for GeometricBrownianMotionProcess {
    fn x0(&self) -> f64 {
        self.s0
    }

    fn drift_1d(&self, _t: f64, x: f64) -> f64 {
        self.mu * x
    }

    fn diffusion_1d(&self, _t: f64, x: f64) -> f64 {
        self.sigma * x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gbm_initial_value() {
        let p = GeometricBrownianMotionProcess::new(100.0, 0.05, 0.2);
        assert_eq!(p.x0(), 100.0);
    }

    #[test]
    fn gbm_drift_diffusion() {
        let p = GeometricBrownianMotionProcess::new(100.0, 0.05, 0.2);
        assert!((p.drift_1d(0.0, 50.0) - 2.5).abs() < 1e-12);
        assert!((p.diffusion_1d(0.0, 50.0) - 10.0).abs() < 1e-12);
    }

    #[test]
    fn gbm_exact_zero_vol_growth() {
        // With σ=0 and μ=0.1, S(1) = S(0)·exp(0.1)
        let p = GeometricBrownianMotionProcess::new(100.0, 0.1, 1e-9);
        let s1 = p.evolve_exact(100.0, 1.0, 0.0);
        assert!((s1 - 100.0 * 0.1_f64.exp()).abs() < 1e-4);
    }

    #[test]
    fn gbm_log_return_positive_drift() {
        let p = GeometricBrownianMotionProcess::new(100.0, 0.05, 0.2);
        // With z=0 the expected log-return is (μ − σ²/2)·dt
        let expected = (0.05 - 0.02) * 1.0;
        assert!((p.log_return(1.0, 0.0) - expected).abs() < 1e-12);
    }
}
