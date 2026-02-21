//! Instrument wrappers for Python.

use pyo3::prelude::*;
use ql_instruments::{VanillaOption, Payoff, Exercise, OptionType};

use crate::time::PyDate;

// ---------------------------------------------------------------------------
// VanillaOption
// ---------------------------------------------------------------------------

/// A vanilla European or American option.
///
/// Construct with ``VanillaOption(strike, expiry, is_call, is_american=False)``.
#[pyclass(name = "VanillaOption")]
#[derive(Clone)]
pub struct PyVanillaOption {
    pub(crate) inner: VanillaOption,
}

#[pymethods]
impl PyVanillaOption {
    /// Create a vanilla option.
    ///
    /// Parameters:
    ///   strike — strike price
    ///   expiry — expiration date
    ///   is_call — True for call, False for put
    ///   is_american — True for American exercise (default False for European)
    #[new]
    #[pyo3(signature = (strike, expiry, is_call=true, is_american=false))]
    fn new(strike: f64, expiry: &PyDate, is_call: bool, is_american: bool) -> Self {
        let opt_type = if is_call { OptionType::Call } else { OptionType::Put };
        let payoff = Payoff::PlainVanilla {
            option_type: opt_type,
            strike,
        };
        let exercise = if is_american {
            Exercise::American {
                earliest: expiry.inner, // simplified: earliest = expiry
                expiry: expiry.inner,
            }
        } else {
            Exercise::European { expiry: expiry.inner }
        };
        PyVanillaOption {
            inner: VanillaOption::new(payoff, exercise),
        }
    }

    #[getter]
    fn strike(&self) -> f64 {
        self.inner.strike()
    }

    #[getter]
    fn is_call(&self) -> bool {
        matches!(self.inner.option_type(), OptionType::Call)
    }

    fn __repr__(&self) -> String {
        let kind = if self.is_call() { "Call" } else { "Put" };
        format!("VanillaOption({} K={:.2})", kind, self.strike())
    }
}
