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

/// American option approximation results.
#[pyclass(name = "AmericanResult")]
#[derive(Clone)]
pub struct PyAmericanResult {
    #[pyo3(get)]
    pub npv: f64,
    #[pyo3(get)]
    pub early_exercise_premium: f64,
    #[pyo3(get)]
    pub critical_price: f64,
}

#[pymethods]
impl PyAmericanResult {
    fn __repr__(&self) -> String {
        format!(
            "AmericanResult(npv={:.6}, early_ex_premium={:.6}, critical_price={:.6})",
            self.npv, self.early_exercise_premium, self.critical_price
        )
    }
}

/// Finite difference result.
#[pyclass(name = "FDResult")]
#[derive(Clone)]
pub struct PyFDResult {
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
impl PyFDResult {
    fn __repr__(&self) -> String {
        format!(
            "FDResult(npv={:.6}, delta={:.6}, gamma={:.6}, theta={:.6})",
            self.npv, self.delta, self.gamma, self.theta
        )
    }
}

/// Heston analytic pricing result.
#[pyclass(name = "HestonResult")]
#[derive(Clone)]
pub struct PyHestonResult {
    #[pyo3(get)]
    pub npv: f64,
    #[pyo3(get)]
    pub p1: f64,
    #[pyo3(get)]
    pub p2: f64,
}

#[pymethods]
impl PyHestonResult {
    fn __repr__(&self) -> String {
        format!(
            "HestonResult(npv={:.6}, p1={:.6}, p2={:.6})",
            self.npv, self.p1, self.p2
        )
    }
}

/// Variance Gamma pricing result.
#[pyclass(name = "VGResult")]
#[derive(Clone)]
pub struct PyVgResult {
    #[pyo3(get)]
    pub npv: f64,
    #[pyo3(get)]
    pub delta: f64,
    #[pyo3(get)]
    pub vega: f64,
}

#[pymethods]
impl PyVgResult {
    fn __repr__(&self) -> String {
        format!("VGResult(npv={:.6}, delta={:.6}, vega={:.6})", self.npv, self.delta, self.vega)
    }
}

/// Quanto option pricing result.
#[pyclass(name = "QuantoResult")]
#[derive(Clone)]
pub struct PyQuantoResult {
    #[pyo3(get)]
    pub npv: f64,
    #[pyo3(get)]
    pub delta: f64,
    #[pyo3(get)]
    pub vega: f64,
    #[pyo3(get)]
    pub qvega: f64,
    #[pyo3(get)]
    pub rho: f64,
    #[pyo3(get)]
    pub qlambda: f64,
}

#[pymethods]
impl PyQuantoResult {
    fn __repr__(&self) -> String {
        format!(
            "QuantoResult(npv={:.6}, delta={:.6}, vega={:.6}, qvega={:.6})",
            self.npv, self.delta, self.vega, self.qvega
        )
    }
}

/// COS (Fourier-cosine) Heston pricing result.
#[pyclass(name = "CosHestonResult")]
#[derive(Clone)]
pub struct PyCosHestonResult {
    #[pyo3(get)]
    pub price: f64,
    #[pyo3(get)]
    pub n_terms: usize,
    #[pyo3(get)]
    pub a: f64,
    #[pyo3(get)]
    pub b: f64,
}

#[pymethods]
impl PyCosHestonResult {
    fn __repr__(&self) -> String {
        format!("CosHestonResult(price={:.6}, n_terms={}, a={:.4}, b={:.4})",
            self.price, self.n_terms, self.a, self.b)
    }
}

/// Binary barrier option result.
#[pyclass(name = "BinaryBarrierResult")]
#[derive(Clone)]
pub struct PyBinaryBarrierResult {
    #[pyo3(get)]
    pub price: f64,
    #[pyo3(get)]
    pub delta: f64,
}

#[pymethods]
impl PyBinaryBarrierResult {
    fn __repr__(&self) -> String {
        format!("BinaryBarrierResult(price={:.6}, delta={:.6})", self.price, self.delta)
    }
}

/// CEV model option result.
#[pyclass(name = "CevResult")]
#[derive(Clone)]
pub struct PyCevResult {
    #[pyo3(get)]
    pub price: f64,
}

#[pymethods]
impl PyCevResult {
    fn __repr__(&self) -> String {
        format!("CevResult(price={:.6})", self.price)
    }
}

/// FD Heston barrier result.
#[pyclass(name = "FdHestonBarrierResult")]
#[derive(Clone)]
pub struct PyFdHestonBarrierResult {
    #[pyo3(get)]
    pub price: f64,
    #[pyo3(get)]
    pub delta: f64,
    #[pyo3(get)]
    pub gamma: f64,
    #[pyo3(get)]
    pub vega: f64,
    #[pyo3(get)]
    pub ns: usize,
    #[pyo3(get)]
    pub nv: usize,
    #[pyo3(get)]
    pub nt: usize,
}

#[pymethods]
impl PyFdHestonBarrierResult {
    fn __repr__(&self) -> String {
        format!("FdHestonBarrierResult(price={:.6}, delta={:.6}, gamma={:.6}, vega={:.6})",
            self.price, self.delta, self.gamma, self.vega)
    }
}

/// Heston + Hull-White hybrid pricing result.
#[pyclass(name = "HestonHullWhiteResult")]
#[derive(Clone)]
pub struct PyHhwResult {
    #[pyo3(get)]
    pub price: f64,
    #[pyo3(get)]
    pub v0_eff: f64,
    #[pyo3(get)]
    pub xi: f64,
}

#[pymethods]
impl PyHhwResult {
    fn __repr__(&self) -> String {
        format!("HestonHullWhiteResult(price={:.6}, v0_eff={:.6}, xi={:.6})",
            self.price, self.v0_eff, self.xi)
    }
}

/// CDO tranche spread result.
#[pyclass(name = "CdoTrancheSpread")]
#[derive(Clone)]
pub struct PyCdoTranche {
    #[pyo3(get)]
    pub attachment: f64,
    #[pyo3(get)]
    pub detachment: f64,
    #[pyo3(get)]
    pub expected_loss: f64,
    #[pyo3(get)]
    pub fair_spread: f64,
}

#[pymethods]
impl PyCdoTranche {
    fn __repr__(&self) -> String {
        format!("CdoTrancheSpread([{:.0}%-{:.0}%] el={:.4} spread={:.4})",
            self.attachment * 100.0, self.detachment * 100.0,
            self.expected_loss, self.fair_spread)
    }
}

/// Credit Valuation Adjustment result.
#[pyclass(name = "CVAResult")]
#[derive(Clone)]
pub struct PyCvaResult {
    #[pyo3(get)]
    pub cva: f64,
    #[pyo3(get)]
    pub dva: f64,
    #[pyo3(get)]
    pub bcva: f64,
}

#[pymethods]
impl PyCvaResult {
    fn __repr__(&self) -> String {
        format!("CVAResult(cva={:.6}, dva={:.6}, bcva={:.6})", self.cva, self.dva, self.bcva)
    }
}

// ---------------------------------------------------------------------------
// New engine result types (Phase 33 / QuantLib parity gap items 2-9)
// ---------------------------------------------------------------------------

/// Asian option pricing result (analytic engines).
#[pyclass(name = "AsianResult")]
#[derive(Clone)]
pub struct PyAsianResult {
    /// Net present value.
    #[pyo3(get)]
    pub npv: f64,
    /// Effective volatility used internally (σ_A).
    #[pyo3(get)]
    pub effective_vol: f64,
    /// Effective forward / adjusted mean.
    #[pyo3(get)]
    pub effective_forward: f64,
}

#[pymethods]
impl PyAsianResult {
    fn __repr__(&self) -> String {
        format!(
            "AsianResult(npv={:.6}, effective_vol={:.6}, effective_forward={:.6})",
            self.npv, self.effective_vol, self.effective_forward
        )
    }
}

/// Ju-Zhong (1999) American option result.
#[pyclass(name = "JuAmericanResult")]
#[derive(Clone)]
pub struct PyJuAmericanResult {
    /// Net present value.
    #[pyo3(get)]
    pub npv: f64,
    /// Delta (∂V/∂S via finite difference).
    #[pyo3(get)]
    pub delta: f64,
    /// Critical asset price (early-exercise boundary).
    #[pyo3(get)]
    pub critical_price: f64,
}

#[pymethods]
impl PyJuAmericanResult {
    fn __repr__(&self) -> String {
        format!(
            "JuAmericanResult(npv={:.6}, delta={:.6}, critical_price={:.6})",
            self.npv, self.delta, self.critical_price
        )
    }
}

/// Integral European engine result.
#[pyclass(name = "IntegralResult")]
#[derive(Clone)]
pub struct PyIntegralResult {
    /// Net present value.
    #[pyo3(get)]
    pub npv: f64,
}

#[pymethods]
impl PyIntegralResult {
    fn __repr__(&self) -> String {
        format!("IntegralResult(npv={:.6})", self.npv)
    }
}

/// Basket / spread option pricing result.
#[pyclass(name = "BasketSpreadResult")]
#[derive(Clone)]
pub struct PyBasketSpreadResult {
    /// Net present value.
    #[pyo3(get)]
    pub npv: f64,
    /// Delta with respect to asset 1.
    #[pyo3(get)]
    pub delta1: f64,
    /// Delta with respect to asset 2.
    #[pyo3(get)]
    pub delta2: f64,
}

#[pymethods]
impl PyBasketSpreadResult {
    fn __repr__(&self) -> String {
        format!(
            "BasketSpreadResult(npv={:.6}, delta1={:.6}, delta2={:.6})",
            self.npv, self.delta1, self.delta2
        )
    }
}

/// Partial-time barrier option result.
#[pyclass(name = "PartialBarrierResult")]
#[derive(Clone)]
pub struct PyPartialBarrierResult {
    /// Net present value.
    #[pyo3(get)]
    pub npv: f64,
}

#[pymethods]
impl PyPartialBarrierResult {
    fn __repr__(&self) -> String {
        format!("PartialBarrierResult(npv={:.6})", self.npv)
    }
}

/// Two-asset correlation option result.
#[pyclass(name = "TwoAssetCorrelationResult")]
#[derive(Clone)]
pub struct PyTwoAssetCorrelationResult {
    /// Net present value.
    #[pyo3(get)]
    pub npv: f64,
    /// Delta with respect to asset 1.
    #[pyo3(get)]
    pub delta1: f64,
    /// Delta with respect to asset 2.
    #[pyo3(get)]
    pub delta2: f64,
}

#[pymethods]
impl PyTwoAssetCorrelationResult {
    fn __repr__(&self) -> String {
        format!(
            "TwoAssetCorrelationResult(npv={:.6}, delta1={:.6}, delta2={:.6})",
            self.npv, self.delta1, self.delta2
        )
    }
}

/// Extensible option result (holder or writer extensible).
#[pyclass(name = "ExtensibleOptionResult")]
#[derive(Clone)]
pub struct PyExtensibleOptionResult {
    /// Net present value.
    #[pyo3(get)]
    pub npv: f64,
}

#[pymethods]
impl PyExtensibleOptionResult {
    fn __repr__(&self) -> String {
        format!("ExtensibleOptionResult(npv={:.6})", self.npv)
    }
}

// ---------------------------------------------------------------------------
// Phase 34: Expanded Python bindings — new result types
// ---------------------------------------------------------------------------

/// Merton jump-diffusion pricing result.
#[pyclass(name = "MertonJdResult")]
#[derive(Clone)]
pub struct PyMertonJdResult {
    /// Net present value.
    #[pyo3(get)]
    pub npv: f64,
    /// Number of Poisson series terms used.
    #[pyo3(get)]
    pub num_terms: usize,
}

#[pymethods]
impl PyMertonJdResult {
    fn __repr__(&self) -> String {
        format!("MertonJdResult(npv={:.6}, num_terms={})", self.npv, self.num_terms)
    }
}

/// Variance swap pricing result.
#[pyclass(name = "VarianceSwapResult")]
#[derive(Clone)]
pub struct PyVarianceSwapResult {
    /// Net present value.
    #[pyo3(get)]
    pub npv: f64,
    /// Fair variance.
    #[pyo3(get)]
    pub fair_variance: f64,
    /// Fair volatility (sqrt of fair variance).
    #[pyo3(get)]
    pub fair_volatility: f64,
}

#[pymethods]
impl PyVarianceSwapResult {
    fn __repr__(&self) -> String {
        format!("VarianceSwapResult(npv={:.6}, fair_var={:.6}, fair_vol={:.6})",
            self.npv, self.fair_variance, self.fair_volatility)
    }
}

/// Monte Carlo Asian arithmetic pricing result.
#[pyclass(name = "McAsianArithResult")]
#[derive(Clone)]
pub struct PyMcAsianArithResult {
    /// Net present value (price).
    #[pyo3(get)]
    pub price: f64,
    /// Standard error.
    #[pyo3(get)]
    pub std_error: f64,
}

#[pymethods]
impl PyMcAsianArithResult {
    fn __repr__(&self) -> String {
        format!("McAsianArithResult(price={:.6}, std_error={:.6})",
            self.price, self.std_error)
    }
}
