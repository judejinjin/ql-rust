//! Calibrated model trait and calibration helpers.
//!
//! The [`CalibratedModel`] trait provides calibration infrastructure:
//! parameters, constraints, and a `calibrate` free function that minimizes
//! the sum of squared pricing errors against market quotes.

use ql_core::errors::QLResult;
use ql_math::optimization::{CostFunction, EndCriteria, Simplex};
use tracing::{info, info_span};

use crate::parameter::Parameter;

// ---------------------------------------------------------------------------
// CalibrationHelper trait
// ---------------------------------------------------------------------------

/// A calibration helper computes the model price and market price
/// for a single calibration instrument.
///
/// The `model_value_with_params` method re-prices the instrument
/// given a candidate parameter vector, enabling the optimizer to
/// evaluate candidates without mutating the model.
pub trait CalibrationHelper: Send + Sync {
    /// The market (or target) price.
    fn market_value(&self) -> f64;

    /// The model price given a candidate parameter vector.
    fn model_value_with_params(&self, params: &[f64]) -> f64;

    /// Calibration error given a candidate parameter vector.
    fn calibration_error_with_params(&self, params: &[f64]) -> f64 {
        self.model_value_with_params(params) - self.market_value()
    }
}

// ---------------------------------------------------------------------------
// CalibratedModel trait
// ---------------------------------------------------------------------------

/// A model that can be calibrated to market data.
///
/// Implementors provide a parameter vector and the ability to re-price
/// calibration helpers after parameter changes.
pub trait CalibratedModel: Send + Sync {
    /// The model's parameters.
    fn parameters(&self) -> &[Parameter];

    /// Set the model's parameter values from a flat vector.
    fn set_params(&mut self, params: &[f64]);

    /// Get current parameter values as a flat vector.
    fn params_as_vec(&self) -> Vec<f64> {
        self.parameters()
            .iter()
            .flat_map(|p| p.values.iter().copied())
            .collect()
    }
}

/// Calibrate a model to the given helpers using Simplex optimization.
///
/// Minimizes Σᵢ (model_price_i(params) − market_price_i)² .
///
/// Returns the optimal cost (sum of squared errors).
pub fn calibrate<M: CalibratedModel>(
    model: &mut M,
    helpers: &[Box<dyn CalibrationHelper>],
    criteria: &EndCriteria,
) -> QLResult<f64> {
    let _span = info_span!("calibrate", num_helpers = helpers.len(), num_params = model.params_as_vec().len()).entered();
    let initial = model.params_as_vec();
    let n = initial.len();
    info!(num_helpers = helpers.len(), num_params = n, "Starting model calibration");

    let cost = CalibrationCost { helpers, n };

    let simplex = Simplex::new(0.05);
    let result = simplex.minimize(&cost, &initial, criteria)?;

    model.set_params(&result.parameters);
    info!(cost = result.value, iterations = result.iterations, "Calibration complete");
    Ok(result.value)
}

/// Internal cost function for calibration.
struct CalibrationCost<'a> {
    helpers: &'a [Box<dyn CalibrationHelper>],
    n: usize,
}

impl CostFunction for CalibrationCost<'_> {
    fn value(&self, params: &[f64]) -> f64 {
        self.helpers
            .iter()
            .map(|h| {
                let err = h.calibration_error_with_params(params);
                err * err
            })
            .sum()
    }

    fn dimension(&self) -> usize {
        self.n
    }
}

// ---------------------------------------------------------------------------
// ShortRateModel trait
// ---------------------------------------------------------------------------

/// A short-rate model that can be calibrated and provides a lattice.
pub trait ShortRateModel: CalibratedModel {
    /// The short rate at time t for a given state variable.
    fn short_rate(&self, t: f64, x: f64) -> f64;

    /// Discount factor P(0, T) implied by the model.
    fn discount(&self, t: f64) -> f64;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameter::{NoConstraint, PositiveConstraint};

    // A simple quadratic model: f(x) = (x[0] - 3)^2 + (x[1] - 5)^2
    struct QuadraticModel {
        params: Vec<Parameter>,
    }

    impl QuadraticModel {
        fn new() -> Self {
            Self {
                params: vec![
                    Parameter::new(0.0, Box::new(NoConstraint)),
                    Parameter::new(0.0, Box::new(NoConstraint)),
                ],
            }
        }
    }

    impl CalibratedModel for QuadraticModel {
        fn parameters(&self) -> &[Parameter] {
            &self.params
        }
        fn set_params(&mut self, vals: &[f64]) {
            self.params[0].set_value(vals[0]);
            self.params[1].set_value(vals[1]);
        }
    }

    struct QuadHelper {
        target: f64,
        idx: usize,
    }

    impl CalibrationHelper for QuadHelper {
        fn market_value(&self) -> f64 {
            self.target
        }
        fn model_value_with_params(&self, params: &[f64]) -> f64 {
            params[self.idx]
        }
    }

    #[test]
    fn calibrate_simple_model() {
        let mut model = QuadraticModel::new();
        let helpers: Vec<Box<dyn CalibrationHelper>> = vec![
            Box::new(QuadHelper {
                target: 3.0,
                idx: 0,
            }),
            Box::new(QuadHelper {
                target: 5.0,
                idx: 1,
            }),
        ];

        let criteria = EndCriteria {
            max_iterations: 5000,
            function_epsilon: 1e-12,
            ..Default::default()
        };

        let cost = calibrate(&mut model, &helpers, &criteria).unwrap();
        assert!(cost < 1e-6, "Calibration should converge, got cost = {cost}");
        assert!(
            (model.params[0].value() - 3.0).abs() < 0.01,
            "param[0] should be ~3.0, got {}",
            model.params[0].value()
        );
        assert!(
            (model.params[1].value() - 5.0).abs() < 0.01,
            "param[1] should be ~5.0, got {}",
            model.params[1].value()
        );
    }

    #[test]
    fn short_rate_model_trait() {
        // Just verify the trait is object-safe by using it
        struct DummySR {
            params: Vec<Parameter>,
        }
        impl CalibratedModel for DummySR {
            fn parameters(&self) -> &[Parameter] {
                &self.params
            }
            fn set_params(&mut self, vals: &[f64]) {
                self.params[0].set_value(vals[0]);
            }
        }
        impl ShortRateModel for DummySR {
            fn short_rate(&self, _t: f64, _x: f64) -> f64 {
                self.params[0].value()
            }
            fn discount(&self, t: f64) -> f64 {
                (-self.params[0].value() * t).exp()
            }
        }

        let sr = DummySR {
            params: vec![Parameter::new(0.05, Box::new(PositiveConstraint))],
        };
        assert!((sr.short_rate(0.0, 0.0) - 0.05).abs() < 1e-12);
        assert!((sr.discount(1.0) - (-0.05_f64).exp()).abs() < 1e-12);
    }
}
