//! Ornstein-Uhlenbeck and Hull-White short-rate processes.
//!
//! ## Ornstein-Uhlenbeck
//! dX = κ(θ − X) dt + σ dW
//!
//! ## Hull-White (extended Vasicek)
//! dr = (θ(t) − a r) dt + σ dW
//!
//! The Hull-White process is an OU process with a time-dependent mean
//! level that exactly fits the initial yield curve.

use crate::process::StochasticProcess1D;

// ===========================================================================
// Ornstein-Uhlenbeck
// ===========================================================================

/// A mean-reverting Ornstein-Uhlenbeck process.
///
/// dX = κ(θ − X) dt + σ dW
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct OrnsteinUhlenbeckProcess {
    /// Initial value X(0).
    pub x0: f64,
    /// Mean-reversion speed κ.
    pub kappa: f64,
    /// Long-run mean θ.
    pub theta: f64,
    /// Volatility σ.
    pub sigma: f64,
}

impl OrnsteinUhlenbeckProcess {
    /// Create a new OU process.
    pub fn new(x0: f64, kappa: f64, theta: f64, sigma: f64) -> Self {
        Self {
            x0,
            kappa,
            theta,
            sigma,
        }
    }

    /// Exact expected value: θ + (x − θ) e^{−κ dt}.
    pub fn exact_expectation(&self, x: f64, dt: f64) -> f64 {
        self.theta + (x - self.theta) * (-self.kappa * dt).exp()
    }

    /// Exact variance: (σ²/2κ)(1 − e^{−2κ dt}).
    pub fn exact_variance(&self, dt: f64) -> f64 {
        let s2 = self.sigma * self.sigma;
        if self.kappa.abs() < 1e-15 {
            return s2 * dt;
        }
        (s2 / (2.0 * self.kappa)) * (1.0 - (-2.0 * self.kappa * dt).exp())
    }

    /// Exact evolution: sample from the conditional Gaussian.
    pub fn evolve_exact(&self, x: f64, dt: f64, dw: f64) -> f64 {
        self.exact_expectation(x, dt) + self.exact_variance(dt).sqrt() * dw
    }
}

impl StochasticProcess1D for OrnsteinUhlenbeckProcess {
    fn x0(&self) -> f64 {
        self.x0
    }

    fn drift_1d(&self, _t: f64, x: f64) -> f64 {
        self.kappa * (self.theta - x)
    }

    fn diffusion_1d(&self, _t: f64, _x: f64) -> f64 {
        self.sigma
    }
}

// ===========================================================================
// Hull-White (constant θ(t) = θ for simplicity, calibration deferred)
// ===========================================================================

/// A Hull-White one-factor short-rate process.
///
/// dr = (θ − a r) dt + σ dW
///
/// In Phase 6 we use constant θ. The time-dependent θ(t) that fits
/// an initial yield curve is implemented during calibration (Phase 8).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HullWhiteProcess {
    /// Initial short rate r(0).
    pub r0: f64,
    /// Mean-reversion speed a.
    pub a: f64,
    /// Long-run rate level θ/a (constant approximation).
    pub theta: f64,
    /// Volatility σ.
    pub sigma: f64,
}

impl HullWhiteProcess {
    /// Create a new Hull-White process.
    pub fn new(r0: f64, a: f64, theta: f64, sigma: f64) -> Self {
        Self { r0, a, theta, sigma }
    }

    /// Exact expected value: θ/a + (r − θ/a) e^{−a dt}.
    pub fn exact_expectation(&self, r: f64, dt: f64) -> f64 {
        let mean_level = self.theta / self.a;
        mean_level + (r - mean_level) * (-self.a * dt).exp()
    }

    /// Exact variance: (σ²/2a)(1 − e^{−2a dt}).
    pub fn exact_variance(&self, dt: f64) -> f64 {
        let s2 = self.sigma * self.sigma;
        if self.a.abs() < 1e-15 {
            return s2 * dt;
        }
        (s2 / (2.0 * self.a)) * (1.0 - (-2.0 * self.a * dt).exp())
    }

    /// Exact evolution.
    pub fn evolve_exact(&self, r: f64, dt: f64, dw: f64) -> f64 {
        self.exact_expectation(r, dt) + self.exact_variance(dt).sqrt() * dw
    }

    /// Discount factor B(t, T) for the affine model.
    ///
    /// P(t, T) = A(t,T) exp(−B(t,T) r(t))
    /// B(t,T) = (1 − e^{−a(T−t)}) / a
    pub fn bond_b(&self, tau: f64) -> f64 {
        if self.a.abs() < 1e-15 {
            return tau;
        }
        (1.0 - (-self.a * tau).exp()) / self.a
    }
}

impl StochasticProcess1D for HullWhiteProcess {
    fn x0(&self) -> f64 {
        self.r0
    }

    fn drift_1d(&self, _t: f64, x: f64) -> f64 {
        self.theta - self.a * x
    }

    fn diffusion_1d(&self, _t: f64, _x: f64) -> f64 {
        self.sigma
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // --- Ornstein-Uhlenbeck tests ---

    #[test]
    fn ou_drift_at_mean() {
        let ou = OrnsteinUhlenbeckProcess::new(0.05, 0.5, 0.05, 0.01);
        // At mean, drift = 0
        assert_abs_diff_eq!(ou.drift_1d(0.0, 0.05), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn ou_drift_above_mean() {
        let ou = OrnsteinUhlenbeckProcess::new(0.05, 0.5, 0.05, 0.01);
        // Above mean, drift is negative (mean-reverting)
        assert!(ou.drift_1d(0.0, 0.10) < 0.0);
    }

    #[test]
    fn ou_exact_expectation() {
        let ou = OrnsteinUhlenbeckProcess::new(0.10, 1.0, 0.05, 0.01);
        let e = ou.exact_expectation(0.10, 1.0);
        // θ + (x - θ) e^{-κ} = 0.05 + 0.05 * e^{-1}
        let expected = 0.05 + 0.05 * (-1.0_f64).exp();
        assert_abs_diff_eq!(e, expected, epsilon = 1e-12);
    }

    #[test]
    fn ou_exact_variance() {
        let ou = OrnsteinUhlenbeckProcess::new(0.0, 2.0, 0.0, 0.1);
        let v = ou.exact_variance(1.0);
        // (σ²/2κ)(1 - e^{-2κ}) = (0.01/4)(1 - e^{-4})
        let expected = (0.01 / 4.0) * (1.0 - (-4.0_f64).exp());
        assert_abs_diff_eq!(v, expected, epsilon = 1e-12);
    }

    #[test]
    fn ou_exact_no_noise() {
        let ou = OrnsteinUhlenbeckProcess::new(0.10, 1.0, 0.05, 0.01);
        let x1 = ou.evolve_exact(0.10, 1.0, 0.0);
        let expected = ou.exact_expectation(0.10, 1.0);
        assert_abs_diff_eq!(x1, expected, epsilon = 1e-12);
    }

    // --- Hull-White tests ---

    #[test]
    fn hw_drift() {
        let hw = HullWhiteProcess::new(0.03, 0.1, 0.005, 0.01);
        // drift = θ - a*r = 0.005 - 0.1*0.03 = 0.002
        assert_abs_diff_eq!(hw.drift_1d(0.0, 0.03), 0.002, epsilon = 1e-15);
    }

    #[test]
    fn hw_diffusion_constant() {
        let hw = HullWhiteProcess::new(0.03, 0.1, 0.005, 0.01);
        assert_abs_diff_eq!(hw.diffusion_1d(0.0, 0.03), 0.01, epsilon = 1e-15);
        assert_abs_diff_eq!(hw.diffusion_1d(1.0, 0.10), 0.01, epsilon = 1e-15);
    }

    #[test]
    fn hw_exact_expectation() {
        let hw = HullWhiteProcess::new(0.03, 0.1, 0.005, 0.01);
        let e = hw.exact_expectation(0.03, 1.0);
        let mean_level = 0.005 / 0.1; // = 0.05
        let expected = mean_level + (0.03 - mean_level) * (-0.1_f64).exp();
        assert_abs_diff_eq!(e, expected, epsilon = 1e-12);
    }

    #[test]
    fn hw_bond_b() {
        let hw = HullWhiteProcess::new(0.03, 0.1, 0.005, 0.01);
        let b = hw.bond_b(1.0);
        let expected = (1.0 - (-0.1_f64).exp()) / 0.1;
        assert_abs_diff_eq!(b, expected, epsilon = 1e-12);
    }

    #[test]
    fn hw_exact_no_noise() {
        let hw = HullWhiteProcess::new(0.03, 0.1, 0.005, 0.01);
        let r1 = hw.evolve_exact(0.03, 0.5, 0.0);
        let expected = hw.exact_expectation(0.03, 0.5);
        assert_abs_diff_eq!(r1, expected, epsilon = 1e-12);
    }
}
