//! Stochastic process traits.
//!
//! Defines the multi-dimensional [`StochasticProcess`] and scalar
//! [`StochasticProcess1D`] interfaces that all concrete processes implement.

use nalgebra::{DMatrix, DVector};

// ---------------------------------------------------------------------------
// Multi-dimensional process
// ---------------------------------------------------------------------------

/// A multi-dimensional stochastic process.
///
/// The SDE is: dX = μ(t,X) dt + σ(t,X) dW
pub trait StochasticProcess: Send + Sync {
    /// Number of state variables.
    fn size(&self) -> usize;

    /// Number of Brownian factors (may differ from size).
    fn factors(&self) -> usize {
        self.size()
    }

    /// Initial values of the state variables.
    fn initial_values(&self) -> DVector<f64>;

    /// Drift vector μ(t, x).
    fn drift(&self, t: f64, x: &DVector<f64>) -> DVector<f64>;

    /// Diffusion matrix σ(t, x).  Dimensions: size() × factors().
    fn diffusion(&self, t: f64, x: &DVector<f64>) -> DMatrix<f64>;

    /// Euler discretization: x0 + μ dt + σ √dt dw.
    fn evolve(
        &self,
        t0: f64,
        x0: &DVector<f64>,
        dt: f64,
        dw: &DVector<f64>,
    ) -> DVector<f64> {
        let mu = self.drift(t0, x0);
        let sigma = self.diffusion(t0, x0);
        x0 + mu * dt + sigma * dw * dt.sqrt()
    }

    /// Expected value E[X(t0+dt) | X(t0) = x0] under Euler.
    fn expectation(
        &self,
        t0: f64,
        x0: &DVector<f64>,
        dt: f64,
    ) -> DVector<f64> {
        x0 + self.drift(t0, x0) * dt
    }
}

// ---------------------------------------------------------------------------
// 1-D process
// ---------------------------------------------------------------------------

/// A one-dimensional stochastic process.
///
/// The SDE is: dX = μ(t,X) dt + σ(t,X) dW
pub trait StochasticProcess1D: Send + Sync {
    /// The initial value X(0).
    fn x0(&self) -> f64;

    /// Drift μ(t, x).
    fn drift_1d(&self, t: f64, x: f64) -> f64;

    /// Diffusion σ(t, x).
    fn diffusion_1d(&self, t: f64, x: f64) -> f64;

    /// Euler discretization: x0 + μ dt + σ √dt dw.
    fn evolve_1d(&self, t0: f64, x0: f64, dt: f64, dw: f64) -> f64 {
        x0 + self.drift_1d(t0, x0) * dt + self.diffusion_1d(t0, x0) * dt.sqrt() * dw
    }

    /// Expected value E[X(t0+dt) | X(t0) = x0].
    fn expectation_1d(&self, t0: f64, x0: f64, dt: f64) -> f64 {
        x0 + self.drift_1d(t0, x0) * dt
    }

    /// Variance Var[X(t0+dt) | X(t0) = x0].
    fn variance_1d(&self, t0: f64, x0: f64, dt: f64) -> f64 {
        let s = self.diffusion_1d(t0, x0);
        s * s * dt
    }

    /// Standard deviation of X(t0+dt) | X(t0) = x0.
    fn std_deviation_1d(&self, t0: f64, x0: f64, dt: f64) -> f64 {
        self.variance_1d(t0, x0, dt).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A simple constant-drift, constant-diffusion 1D process for testing
    struct ConstantProcess {
        initial: f64,
        mu: f64,
        sigma: f64,
    }

    impl StochasticProcess1D for ConstantProcess {
        fn x0(&self) -> f64 {
            self.initial
        }
        fn drift_1d(&self, _t: f64, _x: f64) -> f64 {
            self.mu
        }
        fn diffusion_1d(&self, _t: f64, _x: f64) -> f64 {
            self.sigma
        }
    }

    #[test]
    fn euler_evolution_1d() {
        let p = ConstantProcess {
            initial: 100.0,
            mu: 0.05,
            sigma: 0.2,
        };
        let dt = 1.0;
        let dw = 0.0; // no randomness
        let x1 = p.evolve_1d(0.0, p.x0(), dt, dw);
        // x1 = 100 + 0.05*1 + 0.2*1*0 = 100.05
        assert!((x1 - 100.05).abs() < 1e-12);
    }

    #[test]
    fn expectation_1d() {
        let p = ConstantProcess {
            initial: 50.0,
            mu: 0.1,
            sigma: 0.3,
        };
        let e = p.expectation_1d(0.0, 50.0, 0.5);
        assert!((e - 50.05).abs() < 1e-12);
    }

    #[test]
    fn variance_1d() {
        let p = ConstantProcess {
            initial: 1.0,
            mu: 0.0,
            sigma: 0.2,
        };
        let v = p.variance_1d(0.0, 1.0, 0.25);
        // sigma^2 * dt = 0.04 * 0.25 = 0.01
        assert!((v - 0.01).abs() < 1e-12);
    }

    // Multi-dimensional test
    struct ConstantProcess2D;

    impl StochasticProcess for ConstantProcess2D {
        fn size(&self) -> usize {
            2
        }
        fn initial_values(&self) -> DVector<f64> {
            DVector::from_vec(vec![1.0, 2.0])
        }
        fn drift(&self, _t: f64, _x: &DVector<f64>) -> DVector<f64> {
            DVector::from_vec(vec![0.1, -0.05])
        }
        fn diffusion(&self, _t: f64, _x: &DVector<f64>) -> DMatrix<f64> {
            DMatrix::from_row_slice(2, 2, &[0.2, 0.0, 0.0, 0.3])
        }
    }

    #[test]
    fn multi_dim_expectation() {
        let p = ConstantProcess2D;
        let x0 = p.initial_values();
        let e = p.expectation(0.0, &x0, 1.0);
        assert!((e[0] - 1.1).abs() < 1e-12);
        assert!((e[1] - 1.95).abs() < 1e-12);
    }

    #[test]
    fn multi_dim_evolve_zero_noise() {
        let p = ConstantProcess2D;
        let x0 = p.initial_values();
        let dw = DVector::from_vec(vec![0.0, 0.0]);
        let x1 = p.evolve(0.0, &x0, 1.0, &dw);
        assert!((x1[0] - 1.1).abs() < 1e-12);
        assert!((x1[1] - 1.95).abs() < 1e-12);
    }
}
