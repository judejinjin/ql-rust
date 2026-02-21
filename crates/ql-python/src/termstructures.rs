//! Term structure wrappers for Python.

use pyo3::prelude::*;
use ql_termstructures::yield_curves::FlatForward;
use ql_termstructures::yield_term_structure::YieldTermStructure;
use ql_termstructures::term_structure::TermStructure;

use crate::time::{PyDate, parse_day_counter};

// ---------------------------------------------------------------------------
// FlatForward
// ---------------------------------------------------------------------------

/// A flat yield curve at a constant continuously-compounded rate.
///
/// Construct with ``FlatForward(reference_date, rate, day_counter)``.
#[pyclass(name = "FlatForward")]
#[derive(Clone)]
pub struct PyFlatForward {
    pub(crate) inner: FlatForward,
}

#[pymethods]
impl PyFlatForward {
    /// Create a flat forward curve.
    ///
    /// Parameters:
    ///   reference_date — valuation date
    ///   rate — continuously compounded rate
    ///   day_counter — e.g. "Actual365Fixed", "Actual360"
    #[new]
    #[pyo3(signature = (reference_date, rate, day_counter="Actual365Fixed"))]
    fn new(reference_date: &PyDate, rate: f64, day_counter: &str) -> PyResult<Self> {
        let dc = parse_day_counter(day_counter)?;
        Ok(PyFlatForward {
            inner: FlatForward::new(reference_date.inner, rate, dc),
        })
    }

    /// The flat rate.
    #[getter]
    fn rate(&self) -> f64 {
        self.inner.rate()
    }

    /// Reference date.
    #[getter]
    fn reference_date(&self) -> PyDate {
        PyDate { inner: self.inner.reference_date() }
    }

    /// Discount factor for a given time (year fraction from reference).
    fn discount_t(&self, t: f64) -> f64 {
        self.inner.discount_t(t)
    }

    /// Discount factor to a given date.
    fn discount(&self, date: &PyDate) -> f64 {
        self.inner.discount(date.inner)
    }

    /// Instantaneous forward rate at time t.
    fn forward_rate_t(&self, t: f64) -> f64 {
        self.inner.forward_rate_t(t)
    }

    fn __repr__(&self) -> String {
        format!(
            "FlatForward(rate={:.6}, ref={})",
            self.rate(),
            self.inner.reference_date()
        )
    }
}
