//! Term structure wrappers for Python.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use ql_termstructures::yield_curves::FlatForward;
use ql_termstructures::yield_term_structure::YieldTermStructure;
use ql_termstructures::term_structure::TermStructure;
use ql_termstructures::nelson_siegel::{NelsonSiegelFitting, SvenssonFitting};
use ql_termstructures::bootstrap::{
    DepositRateHelper, SwapRateHelper, PiecewiseYieldCurve, RateHelper,
};

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

// ---------------------------------------------------------------------------
// NelsonSiegelCurve
// ---------------------------------------------------------------------------

/// Nelson-Siegel 4-parameter yield curve: z(t) = β₀ + β₁·f₁(t/τ) + β₂·f₂(t/τ).
///
/// Construct directly with parameters or calibrate with ``NelsonSiegelCurve.fit()``.
#[pyclass(name = "NelsonSiegelCurve")]
#[derive(Clone)]
pub struct PyNelsonSiegelCurve {
    inner: NelsonSiegelFitting,
}

#[pymethods]
impl PyNelsonSiegelCurve {
    /// Create from known parameters β₀, β₁, β₂, τ.
    #[new]
    #[pyo3(signature = (beta0, beta1, beta2, tau))]
    fn new(beta0: f64, beta1: f64, beta2: f64, tau: f64) -> Self {
        Self {
            inner: NelsonSiegelFitting::new(beta0, beta1, beta2, tau),
        }
    }

    /// Calibrate to market zero rates.
    ///
    /// Parameters:
    ///   maturities — list of maturities in years
    ///   market_rates — list of corresponding zero rates
    ///
    /// Returns a fitted ``NelsonSiegelCurve``.
    #[staticmethod]
    #[pyo3(signature = (maturities, market_rates))]
    fn fit(maturities: Vec<f64>, market_rates: Vec<f64>) -> PyResult<Self> {
        NelsonSiegelFitting::fit(&maturities, &market_rates)
            .map(|ns| Self { inner: ns })
            .map_err(|e| PyValueError::new_err(format!("Nelson-Siegel fit error: {e}")))
    }

    /// Zero rate at maturity t (years).
    fn zero_rate(&self, t: f64) -> f64 {
        self.inner.zero_rate(t)
    }

    /// Discount factor at maturity t (years).
    fn discount(&self, t: f64) -> f64 {
        self.inner.discount(t)
    }

    /// The four parameters [β₀, β₁, β₂, τ].
    #[getter]
    fn params(&self) -> [f64; 4] {
        self.inner.params
    }

    fn __repr__(&self) -> String {
        let p = &self.inner.params;
        format!(
            "NelsonSiegelCurve(β₀={:.6}, β₁={:.6}, β₂={:.6}, τ={:.4})",
            p[0], p[1], p[2], p[3]
        )
    }
}

// ---------------------------------------------------------------------------
// SvenssonCurve
// ---------------------------------------------------------------------------

/// Svensson 6-parameter extension of Nelson-Siegel.
///
/// Construct directly or calibrate with ``SvenssonCurve.fit()``.
#[pyclass(name = "SvenssonCurve")]
#[derive(Clone)]
pub struct PySvenssonCurve {
    inner: SvenssonFitting,
}

#[pymethods]
impl PySvenssonCurve {
    /// Create from known parameters β₀, β₁, β₂, β₃, τ₁, τ₂.
    #[new]
    #[pyo3(signature = (beta0, beta1, beta2, beta3, tau1, tau2))]
    fn new(beta0: f64, beta1: f64, beta2: f64, beta3: f64, tau1: f64, tau2: f64) -> Self {
        Self {
            inner: SvenssonFitting::new(beta0, beta1, beta2, beta3, tau1, tau2),
        }
    }

    /// Calibrate to market zero rates.
    #[staticmethod]
    #[pyo3(signature = (maturities, market_rates))]
    fn fit(maturities: Vec<f64>, market_rates: Vec<f64>) -> PyResult<Self> {
        SvenssonFitting::fit(&maturities, &market_rates)
            .map(|sv| Self { inner: sv })
            .map_err(|e| PyValueError::new_err(format!("Svensson fit error: {e}")))
    }

    /// Zero rate at maturity t (years).
    fn zero_rate(&self, t: f64) -> f64 {
        self.inner.zero_rate(t)
    }

    /// Discount factor at maturity t (years).
    fn discount(&self, t: f64) -> f64 {
        self.inner.discount(t)
    }

    /// The six parameters [β₀, β₁, β₂, β₃, τ₁, τ₂].
    #[getter]
    fn params(&self) -> [f64; 6] {
        self.inner.params
    }

    fn __repr__(&self) -> String {
        let p = &self.inner.params;
        format!(
            "SvenssonCurve(β₀={:.6}, β₁={:.6}, β₂={:.6}, β₃={:.6}, τ₁={:.4}, τ₂={:.4})",
            p[0], p[1], p[2], p[3], p[4], p[5]
        )
    }
}

// ---------------------------------------------------------------------------
// PiecewiseYieldCurve (bootstrapped)
// ---------------------------------------------------------------------------

/// A yield curve bootstrapped from deposit and swap rate helpers.
///
/// Use the ``bootstrap_yield_curve()`` function to construct one.
#[pyclass(name = "PiecewiseYieldCurve")]
#[derive(Clone)]
pub struct PyPiecewiseYieldCurve {
    inner: PiecewiseYieldCurve,
}

#[pymethods]
impl PyPiecewiseYieldCurve {
    /// Discount factor to a given date.
    fn discount(&self, date: &PyDate) -> f64 {
        self.inner.discount(date.inner)
    }

    /// Discount factor at time t (year fraction from reference).
    fn discount_t(&self, t: f64) -> f64 {
        self.inner.discount_t(t)
    }

    /// Forward rate at time t.
    fn forward_rate_t(&self, t: f64) -> f64 {
        self.inner.forward_rate_t(t)
    }

    /// The bootstrapped (time, df) node pairs.
    fn nodes(&self) -> Vec<(f64, f64)> {
        self.inner.nodes()
    }

    /// Number of pillar points (including t=0).
    fn size(&self) -> usize {
        self.inner.size()
    }

    fn __repr__(&self) -> String {
        format!(
            "PiecewiseYieldCurve(pillars={}, ref={})",
            self.inner.size(),
            self.inner.reference_date()
        )
    }
}

/// Bootstrap a piecewise yield curve from deposit and swap rate helpers.
///
/// Parameters:
///   reference_date — valuation date
///   deposit_rates — list of (rate, start_date, end_date) tuples for deposits
///   swap_rates — list of (rate, start_date, tenor_years, frequency) tuples for swaps
///   day_counter — e.g. "Actual365Fixed" (default)
///   accuracy — solver accuracy (default 1e-12)
///
/// Returns a ``PiecewiseYieldCurve``.
#[pyfunction]
#[pyo3(signature = (reference_date, deposit_rates, swap_rates, day_counter="Actual365Fixed", accuracy=1e-12))]
pub fn bootstrap_yield_curve(
    reference_date: &PyDate,
    deposit_rates: Vec<(f64, PyDate, PyDate)>,
    swap_rates: Vec<(f64, PyDate, u32, u32)>,
    day_counter: &str,
    accuracy: f64,
) -> PyResult<PyPiecewiseYieldCurve> {
    let dc = parse_day_counter(day_counter)?;
    let calendar = ql_time::Calendar::NullCalendar;

    let mut helpers: Vec<Box<dyn RateHelper>> = Vec::new();

    for (rate, start, end) in deposit_rates {
        helpers.push(Box::new(DepositRateHelper::new(rate, start.inner, end.inner, dc)));
    }

    for (rate, start, tenor_years, frequency) in swap_rates {
        helpers.push(Box::new(SwapRateHelper::from_tenor(
            rate,
            start.inner,
            tenor_years,
            frequency,
            dc,
            calendar.clone(),
        )));
    }

    PiecewiseYieldCurve::new(reference_date.inner, &mut helpers, dc, accuracy)
        .map(|curve| PyPiecewiseYieldCurve { inner: curve })
        .map_err(|e| PyValueError::new_err(format!("Bootstrap error: {e}")))
}
