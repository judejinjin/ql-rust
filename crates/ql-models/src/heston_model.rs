//! Heston stochastic volatility model.
//!
//! The model has five parameters: v0, κ, θ, σ, ρ.
//! It wraps a [`HestonProcess`] and implements [`CalibratedModel`].

use ql_processes::HestonProcess;

use crate::calibrated_model::CalibratedModel;
use crate::parameter::{BoundaryConstraint, Parameter, PositiveConstraint};

/// The Heston stochastic volatility model.
///
/// Parameters:
/// - v0: initial variance
/// - kappa (κ): mean-reversion speed
/// - theta (θ): long-run variance
/// - sigma (σ): vol-of-vol
/// - rho (ρ): correlation between spot and variance
pub struct HestonModel {
    /// The underlying Heston process.
    process: HestonProcess,
    /// Model parameters: [v0, kappa, theta, sigma, rho].
    params: Vec<Parameter>,
}

impl HestonModel {
    /// Create a new Heston model.
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
        let process = HestonProcess::new(s0, risk_free_rate, dividend_yield, v0, kappa, theta, sigma, rho);
        let params = vec![
            Parameter::new(v0, Box::new(PositiveConstraint)),
            Parameter::new(kappa, Box::new(PositiveConstraint)),
            Parameter::new(theta, Box::new(PositiveConstraint)),
            Parameter::new(sigma, Box::new(PositiveConstraint)),
            Parameter::new(rho, Box::new(BoundaryConstraint::new(vec![-1.0], vec![1.0]))),
        ];
        Self { process, params }
    }

    /// The underlying process.
    pub fn process(&self) -> &HestonProcess {
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

    /// Spot price.
    pub fn spot(&self) -> f64 {
        self.process.s0
    }

    /// Risk-free rate.
    pub fn risk_free_rate(&self) -> f64 {
        self.process.risk_free_rate
    }

    /// Dividend yield.
    pub fn dividend_yield(&self) -> f64 {
        self.process.dividend_yield
    }

    /// Check the Feller condition: 2κθ > σ².
    pub fn feller_satisfied(&self) -> bool {
        2.0 * self.kappa() * self.theta() > self.sigma() * self.sigma()
    }
}

impl CalibratedModel for HestonModel {
    fn parameters(&self) -> &[Parameter] {
        &self.params
    }

    fn set_params(&mut self, vals: &[f64]) {
        assert!(vals.len() >= 5, "HestonModel requires 5 parameters");
        self.params[0].set_value(vals[0]);
        self.params[1].set_value(vals[1]);
        self.params[2].set_value(vals[2]);
        self.params[3].set_value(vals[3]);
        self.params[4].set_value(vals[4]);
        // Update the process
        self.process.v0 = vals[0];
        self.process.kappa = vals[1];
        self.process.theta = vals[2];
        self.process.sigma = vals[3];
        self.process.rho = vals[4];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_model() -> HestonModel {
        HestonModel::new(
            100.0, // s0
            0.05,  // r
            0.02,  // q
            0.04,  // v0
            1.5,   // kappa
            0.04,  // theta
            0.3,   // sigma
            -0.7,  // rho
        )
    }

    #[test]
    fn model_parameters() {
        let m = make_model();
        assert_abs_diff_eq!(m.v0(), 0.04);
        assert_abs_diff_eq!(m.kappa(), 1.5);
        assert_abs_diff_eq!(m.theta(), 0.04);
        assert_abs_diff_eq!(m.sigma(), 0.3);
        assert_abs_diff_eq!(m.rho(), -0.7);
    }

    #[test]
    fn model_set_params() {
        let mut m = make_model();
        m.set_params(&[0.05, 2.0, 0.06, 0.4, -0.5]);
        assert_abs_diff_eq!(m.v0(), 0.05);
        assert_abs_diff_eq!(m.kappa(), 2.0);
        assert_abs_diff_eq!(m.theta(), 0.06);
        assert_abs_diff_eq!(m.sigma(), 0.4);
        assert_abs_diff_eq!(m.rho(), -0.5);
        // Process should also be updated
        assert_abs_diff_eq!(m.process().v0, 0.05);
        assert_abs_diff_eq!(m.process().kappa, 2.0);
    }

    #[test]
    fn model_feller() {
        let m = make_model();
        assert!(m.feller_satisfied());
    }

    #[test]
    fn params_as_vec_roundtrip() {
        let m = make_model();
        let v = m.params_as_vec();
        assert_eq!(v.len(), 5);
        assert_abs_diff_eq!(v[0], 0.04);
        assert_abs_diff_eq!(v[4], -0.7);
    }
}
