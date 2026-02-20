//! Merton (1976) jump-diffusion process.
//!
//! SDE:  dS/S = (r − q − λk̄) dt + σ dW + J dN
//!
//! where N ~ Poisson(λ), log(1+J) ~ N(ν, δ²), and k̄ = E[e^J − 1].
//!
//! This process can be used with Monte Carlo and finite-difference engines,
//! unlike the analytic-only `merton_jump_diffusion()` function.

use crate::process::StochasticProcess1D;

/// Merton jump-diffusion process for a single equity underlying.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Merton76Process {
    /// Initial spot price S(0).
    pub s0: f64,
    /// Risk-free rate (continuous).
    pub risk_free_rate: f64,
    /// Dividend yield (continuous).
    pub dividend_yield: f64,
    /// Diffusion volatility σ.
    pub sigma: f64,
    /// Jump intensity λ (mean number of jumps per year).
    pub lambda: f64,
    /// Mean of log-jump size: log(1+J) ~ N(ν, δ²).
    pub nu: f64,
    /// Standard deviation of log-jump size.
    pub delta: f64,
}

impl Merton76Process {
    /// Create a new Merton jump-diffusion process.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        s0: f64,
        risk_free_rate: f64,
        dividend_yield: f64,
        sigma: f64,
        lambda: f64,
        nu: f64,
        delta: f64,
    ) -> Self {
        Self {
            s0,
            risk_free_rate,
            dividend_yield,
            sigma,
            lambda,
            nu,
            delta,
        }
    }

    /// Jump compensator k̄ = E[e^J − 1] = exp(ν + δ²/2) − 1.
    pub fn jump_compensator(&self) -> f64 {
        (self.nu + 0.5 * self.delta * self.delta).exp() - 1.0
    }

    /// Effective drift rate including jump compensation: r − q − λk̄.
    pub fn compensated_drift(&self) -> f64 {
        self.risk_free_rate - self.dividend_yield - self.lambda * self.jump_compensator()
    }

    /// Evolve one time step with jumps.
    ///
    /// * `t`, `s` — current time and spot
    /// * `dt` — time step
    /// * `dw` — standard normal for diffusion
    /// * `num_jumps` — Poisson-drawn number of jumps in [t, t+dt]
    /// * `jump_normals` — standard normals for jump sizes (first `num_jumps` used)
    #[allow(clippy::too_many_arguments)]
    pub fn evolve_with_jumps(
        &self,
        _t: f64,
        s: f64,
        dt: f64,
        dw: f64,
        num_jumps: usize,
        jump_normals: &[f64],
    ) -> f64 {
        let k_bar = self.jump_compensator();
        let mu = self.risk_free_rate - self.dividend_yield - self.lambda * k_bar;

        // Log-Euler for diffusion
        let log_s = s.ln()
            + (mu - 0.5 * self.sigma * self.sigma) * dt
            + self.sigma * dt.sqrt() * dw;

        // Add jumps
        let mut log_jump = 0.0;
        for jn in jump_normals.iter().take(num_jumps) {
            log_jump += self.nu + self.delta * jn;
        }

        (log_s + log_jump).exp()
    }

    /// Total variance per unit time (diffusion + jump variance).
    ///
    /// σ²_total = σ² + λ(ν² + δ²)
    pub fn total_variance_rate(&self) -> f64 {
        self.sigma * self.sigma + self.lambda * (self.nu * self.nu + self.delta * self.delta)
    }
}

/// Standard 1D process interface — diffusion part only (no jumps).
///
/// For full jump evolution, use [`Merton76Process::evolve_with_jumps`].
impl StochasticProcess1D for Merton76Process {
    fn x0(&self) -> f64 {
        self.s0
    }

    fn drift_1d(&self, _t: f64, x: f64) -> f64 {
        self.compensated_drift() * x
    }

    fn diffusion_1d(&self, _t: f64, x: f64) -> f64 {
        self.sigma * x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_process() -> Merton76Process {
        Merton76Process::new(100.0, 0.05, 0.02, 0.20, 0.5, -0.1, 0.15)
    }

    #[test]
    fn initial_value() {
        let p = make_process();
        assert_eq!(p.x0(), 100.0);
    }

    #[test]
    fn jump_compensator_positive_for_negative_nu() {
        let p = make_process();
        // nu = -0.1, delta = 0.15 ⟹ k̄ = exp(-0.1 + 0.5*0.0225) - 1 ≈ -0.0862
        let k = p.jump_compensator();
        assert!(k < 0.0, "k_bar should be negative when mean jump is negative: {k}");
    }

    #[test]
    fn drift_is_risk_neutral() {
        let p = make_process();
        let mu = p.compensated_drift();
        // r - q - λk̄ ≈ 0.05 - 0.02 - 0.5*(-0.0862) ≈ 0.0731
        assert!(mu > 0.0);
        assert!((mu - (0.05 - 0.02 - 0.5 * p.jump_compensator())).abs() < 1e-12);
    }

    #[test]
    fn diffusion_is_proportional_to_spot() {
        let p = make_process();
        let d100 = p.diffusion_1d(0.0, 100.0);
        let d200 = p.diffusion_1d(0.0, 200.0);
        assert!((d200 / d100 - 2.0).abs() < 1e-12);
    }

    #[test]
    fn evolve_with_no_jumps_preserves_positivity() {
        let p = make_process();
        let s = p.evolve_with_jumps(0.0, 100.0, 0.01, -2.0, 0, &[]);
        assert!(s > 0.0, "Spot must stay positive: {s}");
    }

    #[test]
    fn evolve_with_jumps_applies_jump() {
        let p = make_process();
        let s_no_jump = p.evolve_with_jumps(0.0, 100.0, 0.01, 0.0, 0, &[]);
        let s_with_jump = p.evolve_with_jumps(0.0, 100.0, 0.01, 0.0, 1, &[0.5]);
        // One jump with z=0.5: jump_size = exp(nu + delta*0.5) = exp(-0.1+0.075) ≈ 0.975
        // Prices should differ
        assert!((s_no_jump - s_with_jump).abs() > 0.01);
    }

    #[test]
    fn total_variance_exceeds_diffusion_variance() {
        let p = make_process();
        let total = p.total_variance_rate();
        let diff_only = p.sigma * p.sigma;
        assert!(total > diff_only, "Total var {total} should exceed diffusion {diff_only}");
    }
}
