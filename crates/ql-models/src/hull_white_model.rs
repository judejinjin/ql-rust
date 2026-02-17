//! Hull-White one-factor short-rate model.
//!
//! dr = (θ(t) − a r) dt + σ dW
//!
//! Parameters: a (mean-reversion speed), σ (volatility).
//! Implements [`CalibratedModel`] and [`ShortRateModel`].

use crate::calibrated_model::{CalibratedModel, ShortRateModel};
use crate::parameter::{Parameter, PositiveConstraint};

/// A Hull-White one-factor model with constant parameters.
///
/// The time-dependent θ(t) that exactly fits the initial yield curve
/// is computed during calibration or pricing. For Phase 6, we use
/// a constant θ = a × long_rate.
pub struct HullWhiteModel {
    /// Mean-reversion speed a.
    a: f64,
    /// Volatility σ.
    sigma: f64,
    /// Long-run rate level (constant approximation of θ(t)/a).
    long_rate: f64,
    /// Model parameters: [a, sigma].
    params: Vec<Parameter>,
}

impl HullWhiteModel {
    /// Create a new Hull-White model.
    ///
    /// `long_rate` is the initial estimate of the long-run rate level.
    pub fn new(a: f64, sigma: f64, long_rate: f64) -> Self {
        let params = vec![
            Parameter::new(a, Box::new(PositiveConstraint)),
            Parameter::new(sigma, Box::new(PositiveConstraint)),
        ];
        Self {
            a,
            sigma,
            long_rate,
            params,
        }
    }

    /// Mean-reversion speed a.
    pub fn a(&self) -> f64 {
        self.a
    }

    /// Volatility σ.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// B(τ) = (1 − e^{−aτ}) / a — appears in the affine bond pricing formula.
    pub fn bond_b(&self, tau: f64) -> f64 {
        if self.a.abs() < 1e-15 {
            return tau;
        }
        (1.0 - (-self.a * tau).exp()) / self.a
    }

    /// A(τ) = exp{ [B(τ) − τ](a²θ − σ²/2) / a² − σ²B(τ)² / (4a) }
    ///
    /// Simplified version using constant θ = a × long_rate.
    pub fn bond_a(&self, tau: f64) -> f64 {
        let b = self.bond_b(tau);
        let theta = self.a * self.long_rate;
        let a2 = self.a * self.a;
        let s2 = self.sigma * self.sigma;

        if self.a.abs() < 1e-15 {
            return (-0.5 * s2 * tau * tau * tau / 3.0).exp();
        }

        ((b - tau) * (a2 * theta - 0.5 * s2) / a2 - s2 * b * b / (4.0 * self.a)).exp()
    }
}

impl CalibratedModel for HullWhiteModel {
    fn parameters(&self) -> &[Parameter] {
        &self.params
    }

    fn set_params(&mut self, vals: &[f64]) {
        assert!(vals.len() >= 2, "HullWhiteModel requires 2 parameters");
        self.params[0].set_value(vals[0]);
        self.params[1].set_value(vals[1]);
        self.a = vals[0];
        self.sigma = vals[1];
    }
}

impl ShortRateModel for HullWhiteModel {
    fn short_rate(&self, _t: f64, x: f64) -> f64 {
        x
    }

    fn discount(&self, t: f64) -> f64 {
        self.bond_a(t) * (-self.bond_b(t) * self.long_rate).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_model() -> HullWhiteModel {
        HullWhiteModel::new(0.1, 0.01, 0.05)
    }

    #[test]
    fn hw_model_parameters() {
        let m = make_model();
        assert_abs_diff_eq!(m.a(), 0.1);
        assert_abs_diff_eq!(m.sigma(), 0.01);
    }

    #[test]
    fn hw_model_set_params() {
        let mut m = make_model();
        m.set_params(&[0.2, 0.02]);
        assert_abs_diff_eq!(m.a(), 0.2);
        assert_abs_diff_eq!(m.sigma(), 0.02);
    }

    #[test]
    fn hw_bond_b_small_a() {
        let m = HullWhiteModel::new(1e-20, 0.01, 0.05);
        let b = m.bond_b(1.0);
        assert_abs_diff_eq!(b, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn hw_bond_b() {
        let m = make_model();
        let b = m.bond_b(1.0);
        let expected = (1.0 - (-0.1_f64).exp()) / 0.1;
        assert_abs_diff_eq!(b, expected, epsilon = 1e-12);
    }

    #[test]
    fn hw_discount_at_zero() {
        let m = make_model();
        let d = m.discount(0.0);
        // A(0) = 1, B(0) = 0, so P(0,0) = 1
        assert_abs_diff_eq!(d, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn hw_discount_decreasing() {
        let m = make_model();
        let d1 = m.discount(1.0);
        let d2 = m.discount(5.0);
        assert!(d1 > d2, "Longer maturity should have smaller discount factor");
        assert!(d1 < 1.0);
        assert!(d2 > 0.0);
    }

    #[test]
    fn hw_params_as_vec() {
        let m = make_model();
        let v = m.params_as_vec();
        assert_eq!(v.len(), 2);
        assert_abs_diff_eq!(v[0], 0.1);
        assert_abs_diff_eq!(v[1], 0.01);
    }
}
