//! Bates stochastic volatility model with jumps.
//!
//! Extends the Heston model with Merton-style jumps:
//! - λ: jump intensity (mean number of jumps per year)
//! - ν: mean log-jump size
//! - δ: volatility of log-jump size
//!
//! Implements [`CalibratedModel`] with 8 parameters:
//! [v0, κ, θ, σ, ρ, λ, ν, δ].

use ql_processes::BatesProcess;

use crate::calibrated_model::CalibratedModel;
use crate::parameter::{BoundaryConstraint, NoConstraint, Parameter, PositiveConstraint};

/// The Bates stochastic volatility + jumps model.
///
/// Parameters:
/// - v0: initial variance
/// - kappa (κ): mean-reversion speed
/// - theta (θ): long-run variance
/// - sigma (σ): vol-of-vol
/// - rho (ρ): correlation between spot and variance
/// - lambda (λ): jump intensity
/// - nu (ν): mean log-jump size
/// - delta (δ): volatility of log-jump size
pub struct BatesModel {
    /// The underlying Bates process.
    process: BatesProcess,
    /// Model parameters: [v0, kappa, theta, sigma, rho, lambda, nu, delta].
    params: Vec<Parameter>,
}

impl BatesModel {
    /// Create a new Bates model.
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
        let process =
            BatesProcess::new(s0, risk_free_rate, dividend_yield, v0, kappa, theta, sigma, rho, lambda, nu, delta);
        let params = vec![
            Parameter::new(v0, Box::new(PositiveConstraint)),
            Parameter::new(kappa, Box::new(PositiveConstraint)),
            Parameter::new(theta, Box::new(PositiveConstraint)),
            Parameter::new(sigma, Box::new(PositiveConstraint)),
            Parameter::new(rho, Box::new(BoundaryConstraint::new(vec![-1.0], vec![1.0]))),
            Parameter::new(lambda, Box::new(PositiveConstraint)),
            Parameter::new(nu, Box::new(NoConstraint)),
            Parameter::new(delta, Box::new(PositiveConstraint)),
        ];
        Self { process, params }
    }

    /// The underlying Bates process.
    pub fn process(&self) -> &BatesProcess {
        &self.process
    }

    /// Initial variance v0.
    pub fn v0(&self) -> f64 {
        self.params[0].value()
    }

    /// Mean-reversion speed κ.
    pub fn kappa(&self) -> f64 {
        self.params[1].value()
    }

    /// Long-run variance θ.
    pub fn theta(&self) -> f64 {
        self.params[2].value()
    }

    /// Vol-of-vol σ.
    pub fn sigma(&self) -> f64 {
        self.params[3].value()
    }

    /// Correlation ρ.
    pub fn rho(&self) -> f64 {
        self.params[4].value()
    }

    /// Jump intensity λ.
    pub fn lambda(&self) -> f64 {
        self.params[5].value()
    }

    /// Mean log-jump size ν.
    pub fn nu(&self) -> f64 {
        self.params[6].value()
    }

    /// Jump-size volatility δ.
    pub fn delta(&self) -> f64 {
        self.params[7].value()
    }

    /// Spot price.
    pub fn spot(&self) -> f64 {
        self.process.heston.s0
    }

    /// Risk-free rate.
    pub fn risk_free_rate(&self) -> f64 {
        self.process.heston.risk_free_rate
    }

    /// Dividend yield.
    pub fn dividend_yield(&self) -> f64 {
        self.process.heston.dividend_yield
    }

    /// Jump compensator k̄ = exp(ν + δ²/2) − 1.
    pub fn jump_compensator(&self) -> f64 {
        self.process.jump_compensator()
    }

    /// Check the Feller condition for the variance process.
    pub fn feller_satisfied(&self) -> bool {
        self.process.heston.feller_satisfied()
    }
}

impl CalibratedModel for BatesModel {
    fn parameters(&self) -> &[Parameter] {
        &self.params
    }

    fn set_params(&mut self, vals: &[f64]) {
        assert!(vals.len() >= 8, "BatesModel requires 8 parameters");
        for (i, &v) in vals.iter().enumerate().take(8) {
            self.params[i].set_value(v);
        }
        // Update the process
        self.process.heston.v0 = vals[0];
        self.process.heston.kappa = vals[1];
        self.process.heston.theta = vals[2];
        self.process.heston.sigma = vals[3];
        self.process.heston.rho = vals[4];
        self.process.lambda = vals[5];
        self.process.nu = vals[6];
        self.process.delta = vals[7];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_bates() -> BatesModel {
        BatesModel::new(
            100.0, 0.05, 0.02, // s0, r, q
            0.04, 1.5, 0.04, 0.3, -0.7, // v0, κ, θ, σ, ρ
            0.5, -0.1, 0.15, // λ, ν, δ
        )
    }

    #[test]
    fn bates_model_parameters() {
        let m = make_bates();
        assert_abs_diff_eq!(m.v0(), 0.04);
        assert_abs_diff_eq!(m.kappa(), 1.5);
        assert_abs_diff_eq!(m.theta(), 0.04);
        assert_abs_diff_eq!(m.sigma(), 0.3);
        assert_abs_diff_eq!(m.rho(), -0.7);
        assert_abs_diff_eq!(m.lambda(), 0.5);
        assert_abs_diff_eq!(m.nu(), -0.1);
        assert_abs_diff_eq!(m.delta(), 0.15);
    }

    #[test]
    fn bates_model_set_params() {
        let mut m = make_bates();
        m.set_params(&[0.05, 2.0, 0.06, 0.4, -0.5, 1.0, 0.0, 0.2]);
        assert_abs_diff_eq!(m.lambda(), 1.0);
        assert_abs_diff_eq!(m.nu(), 0.0);
        assert_abs_diff_eq!(m.delta(), 0.2);
        // Process should also be updated
        assert_abs_diff_eq!(m.process().lambda, 1.0);
        assert_abs_diff_eq!(m.process().heston.kappa, 2.0);
    }

    #[test]
    fn bates_model_spot_and_rates() {
        let m = make_bates();
        assert_abs_diff_eq!(m.spot(), 100.0);
        assert_abs_diff_eq!(m.risk_free_rate(), 0.05);
        assert_abs_diff_eq!(m.dividend_yield(), 0.02);
    }

    #[test]
    fn bates_params_roundtrip() {
        let m = make_bates();
        let v = m.params_as_vec();
        assert_eq!(v.len(), 8);
        assert_abs_diff_eq!(v[5], 0.5); // lambda
        assert_abs_diff_eq!(v[7], 0.15); // delta
    }
}
