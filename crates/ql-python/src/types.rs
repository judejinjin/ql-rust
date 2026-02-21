//! Result wrapper types exposed to Python.

use pyo3::prelude::*;

/// Analytic European pricing results (BS Greeks).
#[pyclass(name = "AnalyticResults")]
#[derive(Clone)]
pub struct PyAnalyticResults {
    #[pyo3(get)]
    pub npv: f64,
    #[pyo3(get)]
    pub delta: f64,
    #[pyo3(get)]
    pub gamma: f64,
    #[pyo3(get)]
    pub vega: f64,
    #[pyo3(get)]
    pub theta: f64,
    #[pyo3(get)]
    pub rho: f64,
}

#[pymethods]
impl PyAnalyticResults {
    fn __repr__(&self) -> String {
        format!(
            "AnalyticResults(npv={:.6}, delta={:.6}, gamma={:.6}, vega={:.6}, theta={:.6}, rho={:.6})",
            self.npv, self.delta, self.gamma, self.vega, self.theta, self.rho
        )
    }
}

/// Monte Carlo pricing result.
#[pyclass(name = "MCResult")]
#[derive(Clone)]
pub struct PyMCResult {
    #[pyo3(get)]
    pub npv: f64,
    #[pyo3(get)]
    pub std_error: f64,
    #[pyo3(get)]
    pub num_paths: usize,
}

#[pymethods]
impl PyMCResult {
    fn __repr__(&self) -> String {
        format!(
            "MCResult(npv={:.6}, std_error={:.6}, num_paths={})",
            self.npv, self.std_error, self.num_paths
        )
    }
}

/// Binomial lattice pricing result.
#[pyclass(name = "LatticeResult")]
#[derive(Clone)]
pub struct PyLatticeResult {
    #[pyo3(get)]
    pub npv: f64,
    #[pyo3(get)]
    pub delta: f64,
    #[pyo3(get)]
    pub gamma: f64,
    #[pyo3(get)]
    pub theta: f64,
}

#[pymethods]
impl PyLatticeResult {
    fn __repr__(&self) -> String {
        format!(
            "LatticeResult(npv={:.6}, delta={:.6}, gamma={:.6}, theta={:.6})",
            self.npv, self.delta, self.gamma, self.theta
        )
    }
}

/// Swap pricing results.
#[pyclass(name = "SwapResults")]
#[derive(Clone)]
pub struct PySwapResults {
    #[pyo3(get)]
    pub npv: f64,
    #[pyo3(get)]
    pub fixed_leg_npv: f64,
    #[pyo3(get)]
    pub floating_leg_npv: f64,
    #[pyo3(get)]
    pub fair_rate: f64,
}

#[pymethods]
impl PySwapResults {
    fn __repr__(&self) -> String {
        format!(
            "SwapResults(npv={:.6}, fair_rate={:.6})",
            self.npv, self.fair_rate
        )
    }
}

/// Bond pricing results.
#[pyclass(name = "BondResults")]
#[derive(Clone)]
pub struct PyBondResults {
    #[pyo3(get)]
    pub npv: f64,
    #[pyo3(get)]
    pub clean_price: f64,
    #[pyo3(get)]
    pub dirty_price: f64,
    #[pyo3(get)]
    pub accrued_interest: f64,
}

#[pymethods]
impl PyBondResults {
    fn __repr__(&self) -> String {
        format!(
            "BondResults(npv={:.6}, clean={:.6}, dirty={:.6}, accrued={:.6})",
            self.npv, self.clean_price, self.dirty_price, self.accrued_interest
        )
    }
}
