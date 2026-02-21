//! Python bindings for the ql-rust quantitative finance library.
//!
//! This crate exposes key types and pricing functions from ql-rust to Python
//! via PyO3. Install with `maturin develop` or `pip install .` from the
//! `crates/ql-python` directory.
//!
//! # Quick start (Python)
//!
//! ```python
//! import ql_python as ql
//!
//! # Date arithmetic
//! d = ql.Date(2025, 6, 15)
//! print(d.year, d.month, d.day)  # 2025, 6, 15
//!
//! # European option pricing
//! greeks = ql.price_european_bs(
//!     spot=100.0, strike=100.0, r=0.05, q=0.02,
//!     vol=0.20, t=1.0, is_call=True
//! )
//! print(f"NPV={greeks.npv:.4f}, delta={greeks.delta:.4f}")
//! ```

mod types;
mod time;
mod termstructures;
mod instruments;
mod pricing;

use pyo3::prelude::*;

/// ql_python — Python bindings for ql-rust quantitative finance library.
#[pymodule]
fn ql_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Time types
    m.add_class::<time::PyDate>()?;
    m.add_class::<time::PyPeriod>()?;
    m.add_class::<time::PySchedule>()?;

    // Term structures
    m.add_class::<termstructures::PyFlatForward>()?;
    m.add_class::<termstructures::PyNelsonSiegelCurve>()?;
    m.add_class::<termstructures::PySvenssonCurve>()?;
    m.add_class::<termstructures::PyPiecewiseYieldCurve>()?;

    // Instruments
    m.add_class::<instruments::PyVanillaOption>()?;

    // Result types
    m.add_class::<types::PyAnalyticResults>()?;
    m.add_class::<types::PyMCResult>()?;
    m.add_class::<types::PyLatticeResult>()?;
    m.add_class::<types::PySwapResults>()?;
    m.add_class::<types::PyBondResults>()?;
    m.add_class::<types::PyAmericanResult>()?;
    m.add_class::<types::PyFDResult>()?;
    m.add_class::<types::PyHestonResult>()?;

    // Pricing functions — analytic
    m.add_function(wrap_pyfunction!(pricing::price_european_bs, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::implied_vol, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::barone_adesi_whaley_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::bjerksund_stensland_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::heston_price_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::kirk_spread_call_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::kirk_spread_put_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::sabr_vol_py, m)?)?;

    // Pricing functions — numerical
    m.add_function(wrap_pyfunction!(pricing::mc_european_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::mc_barrier_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::binomial_crr_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::fd_black_scholes_py, m)?)?;

    // Term structure bootstrapping
    m.add_function(wrap_pyfunction!(termstructures::bootstrap_yield_curve, m)?)?;

    // Calendar utilities
    m.add_function(wrap_pyfunction!(time::is_business_day, m)?)?;
    m.add_function(wrap_pyfunction!(time::advance_date, m)?)?;
    m.add_function(wrap_pyfunction!(time::year_fraction, m)?)?;
    m.add_function(wrap_pyfunction!(time::business_days_between, m)?)?;

    Ok(())
}
