//! Bates stochastic volatility with jumps process.
//!
//! Extends the Heston model with Merton-style compound Poisson jumps:
//!
//!   dS/S = (r − q − λk̄) dt + √v dW₁ + J dN
//!   dv   = κ(θ − v) dt + σ √v dW₂
//!   dW₁ dW₂ = ρ dt
//!
//! where N is a Poisson process with intensity λ, and log(1+J) ~ N(ν, δ²).
//! The drift compensator k̄ = E[J] = exp(ν + δ²/2) − 1 ensures the
//! risk-neutral drift is (r − q).

use nalgebra::{DMatrix, DVector};

use crate::heston_process::HestonProcess;
use crate::process::StochasticProcess;

/// Bates process parameters (Heston + jumps).
#[derive(Clone, Debug)]
pub struct BatesProcess {
    /// The underlying Heston process.
    pub heston: HestonProcess,
    /// Jump intensity (mean number of jumps per year).
    pub lambda: f64,
    /// Mean of log-jump size: log(1+J) ~ N(nu, delta²).
    pub nu: f64,
    /// Volatility of log-jump size.
    pub delta: f64,
}

impl BatesProcess {
    /// Create a new Bates process.
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
        lambda: f64,
        nu: f64,
        delta: f64,
    ) -> Self {
        let heston = HestonProcess::new(s0, risk_free_rate, dividend_yield, v0, kappa, theta, sigma, rho);
        Self {
            heston,
            lambda,
            nu,
            delta,
        }
    }

    /// The compensator k̄ = E[e^J − 1] = exp(ν + δ²/2) − 1.
    pub fn jump_compensator(&self) -> f64 {
        (self.nu + 0.5 * self.delta * self.delta).exp() - 1.0
    }

    /// Evolve one time step using QE (Andersen 2008) for variance
    /// and log-Euler for spot, with compound Poisson jumps added.
    ///
    /// `z1`, `z2` are standard normal variates for diffusion.
    /// `u_poisson` is a uniform(0,1) for the Poisson draw.
    /// `jump_normals` provides normal variates for jump sizes
    /// (only the first N are used, where N ~ Poisson(λ dt)).
    #[allow(clippy::too_many_arguments)]
    pub fn evolve_qe_jump(
        &self,
        t: f64,
        s: f64,
        v: f64,
        dt: f64,
        z1: f64,
        z2: f64,
        num_jumps: usize,
        jump_normals: &[f64],
    ) -> (f64, f64) {
        // Heston QE step (gives back spot, variance)
        let (s_diff, v_next) = self.heston.evolve_qe(t, s, v, dt, z1, z2);

        // Add jump component to log-spot
        let k_bar = self.jump_compensator();
        let mut log_jump = 0.0;
        for jn in jump_normals.iter().take(num_jumps) {
            log_jump += self.nu + self.delta * jn;
        }

        // Drift compensation: subtract λ k̄ dt from log-spot
        let s_next = s_diff * (log_jump - self.lambda * k_bar * dt).exp();

        (s_next, v_next)
    }
}

impl StochasticProcess for BatesProcess {
    fn size(&self) -> usize {
        2
    }

    fn factors(&self) -> usize {
        2 // Diffusion part only; jumps handled separately
    }

    fn initial_values(&self) -> DVector<f64> {
        self.heston.initial_values()
    }

    fn drift(&self, t: f64, x: &DVector<f64>) -> DVector<f64> {
        // Drift includes the jump compensator in the spot component
        let mut d = self.heston.drift(t, x);
        let s = x[0];
        d[0] -= self.lambda * self.jump_compensator() * s;
        d
    }

    fn diffusion(&self, t: f64, x: &DVector<f64>) -> DMatrix<f64> {
        self.heston.diffusion(t, x)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_bates() -> BatesProcess {
        BatesProcess::new(
            100.0, // s0
            0.05,  // r
            0.02,  // q
            0.04,  // v0
            1.5,   // kappa
            0.04,  // theta
            0.3,   // sigma
            -0.7,  // rho
            0.5,   // lambda (0.5 jumps/year)
            -0.1,  // nu (negative mean jump)
            0.15,  // delta (jump vol)
        )
    }

    #[test]
    fn bates_jump_compensator() {
        let p = make_bates();
        // k_bar = exp(-0.1 + 0.5 * 0.15^2) - 1 = exp(-0.08875) - 1 ≈ -0.08489
        let k_bar = p.jump_compensator();
        let expected = (-0.1 + 0.5 * 0.15 * 0.15_f64).exp() - 1.0;
        assert_abs_diff_eq!(k_bar, expected, epsilon = 1e-10);
    }

    #[test]
    fn bates_reduces_to_heston_when_no_jumps() {
        let mut p = make_bates();
        p.lambda = 0.0;
        p.nu = 0.0;
        p.delta = 0.0;

        // QE step with no jumps should match Heston
        let (s_bates, v_bates) =
            p.evolve_qe_jump(0.0, 100.0, 0.04, 0.01, 0.5, -0.3, 0, &[]);
        let (s_heston, v_heston) =
            p.heston.evolve_qe(0.0, 100.0, 0.04, 0.01, 0.5, -0.3);

        assert_abs_diff_eq!(s_bates, s_heston, epsilon = 1e-10);
        assert_abs_diff_eq!(v_bates, v_heston, epsilon = 1e-10);
    }

    #[test]
    fn bates_initial_values() {
        let p = make_bates();
        let iv = p.initial_values();
        assert_abs_diff_eq!(iv[0], 100.0, epsilon = 1e-10);
        assert_abs_diff_eq!(iv[1], 0.04, epsilon = 1e-10);
    }

    #[test]
    fn bates_size_and_factors() {
        let p = make_bates();
        assert_eq!(p.size(), 2);
        assert_eq!(p.factors(), 2);
    }

    #[test]
    fn bates_drift_includes_jump_compensator() {
        let p = make_bates();
        let x = DVector::from_vec(vec![100.0, 0.04]);

        let d_bates = p.drift(0.0, &x);
        let d_heston = p.heston.drift(0.0, &x);

        // Bates drift should be smaller (lambda * k_bar * S subtracted)
        // k_bar ≈ -0.085, so lambda * k_bar * S ≈ -4.24, meaning drift is LARGER
        // (subtracting a negative number adds)
        let expected_diff = -p.lambda * p.jump_compensator() * 100.0;
        assert_abs_diff_eq!(d_bates[0] - d_heston[0], expected_diff, epsilon = 1e-8);
        // Variance drift unchanged
        assert_abs_diff_eq!(d_bates[1], d_heston[1], epsilon = 1e-10);
    }

    #[test]
    fn bates_evolve_with_jumps_changes_spot() {
        let p = make_bates();
        // No jumps
        let (s_no_jump, _) =
            p.evolve_qe_jump(0.0, 100.0, 0.04, 0.01, 0.5, -0.3, 0, &[]);
        // One jump
        let (s_one_jump, _) =
            p.evolve_qe_jump(0.0, 100.0, 0.04, 0.01, 0.5, -0.3, 1, &[0.0]);

        // They should differ because the jump adds nu to log-spot
        assert!(
            (s_no_jump - s_one_jump).abs() > 1e-6,
            "With jump should differ from no jump"
        );
    }
}
