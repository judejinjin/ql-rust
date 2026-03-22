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
mod fixed_income;

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
    m.add_class::<types::PyVgResult>()?;
    m.add_class::<types::PyQuantoResult>()?;
    m.add_class::<types::PyCvaResult>()?;
    m.add_class::<types::PyCosHestonResult>()?;
    m.add_class::<types::PyBinaryBarrierResult>()?;
    m.add_class::<types::PyCevResult>()?;
    m.add_class::<types::PyFdHestonBarrierResult>()?;
    m.add_class::<types::PyHhwResult>()?;
    m.add_class::<types::PyCdoTranche>()?;
    // New result types (QuantLib parity gap items 2-9)
    m.add_class::<types::PyAsianResult>()?;
    m.add_class::<types::PyJuAmericanResult>()?;
    m.add_class::<types::PyIntegralResult>()?;
    m.add_class::<types::PyBasketSpreadResult>()?;
    m.add_class::<types::PyPartialBarrierResult>()?;
    m.add_class::<types::PyTwoAssetCorrelationResult>()?;
    m.add_class::<types::PyExtensibleOptionResult>()?;
    // Phase 34 result types
    m.add_class::<types::PyMertonJdResult>()?;
    m.add_class::<types::PyVarianceSwapResult>()?;
    m.add_class::<types::PyMcAsianArithResult>()?;
    // Phase 36 result types — fixed income, credit, models, risk
    m.add_class::<types::PyHWAnalyticResult>()?;
    m.add_class::<types::PyTreeResult>()?;
    m.add_class::<types::PyCdsResult>()?;
    m.add_class::<types::PyBatesResult>()?;
    m.add_class::<types::PyCliquetResult>()?;
    m.add_class::<types::PyCallableBondResult>()?;
    m.add_class::<types::PyCdoTrancheResult>()?;
    m.add_class::<types::PySensitivity>()?;

    // Pricing functions — analytic
    m.add_function(wrap_pyfunction!(pricing::price_european_bs, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::implied_vol, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::barone_adesi_whaley_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::bjerksund_stensland_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::heston_price_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::kirk_spread_call_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::kirk_spread_put_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::sabr_vol_py, m)?)?;

    // Pricing functions — advanced models
    m.add_function(wrap_pyfunction!(pricing::vg_price_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::price_quanto_vanilla_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::bilateral_cva_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::cos_heston_price_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::analytic_binary_barrier_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::analytic_cev_price_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::fd_heston_barrier_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::heston_hull_white_price_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::cdo_spread_ladder_py, m)?)?;

    // Pricing functions — numerical
    m.add_function(wrap_pyfunction!(pricing::mc_european_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::mc_barrier_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::binomial_crr_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::fd_black_scholes_py, m)?)?;

    // Pricing functions — Asian options
    m.add_function(wrap_pyfunction!(pricing::asian_continuous_geo_avg_price_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::asian_discrete_geo_avg_price_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::asian_continuous_geo_avg_strike_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::asian_discrete_geo_avg_strike_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::asian_turnbull_wakeman_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::asian_levy_py, m)?)?;

    // Pricing functions — basket / spread
    m.add_function(wrap_pyfunction!(pricing::choi_basket_spread_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::dlz_basket_price_py, m)?)?;

    // Pricing functions — American / integral
    m.add_function(wrap_pyfunction!(pricing::ju_quadratic_american_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::integral_european_py, m)?)?;

    // Pricing functions — exotic options
    m.add_function(wrap_pyfunction!(pricing::partial_time_barrier_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::two_asset_correlation_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::holder_extensible_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::writer_extensible_py, m)?)?;

    // Phase 34: Expanded pricing functions
    m.add_function(wrap_pyfunction!(pricing::black76_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::bachelier_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::qd_plus_american_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::merton_jd_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::chooser_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::compound_option_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::lookback_floating_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::forward_start_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::power_option_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::digital_american_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::digital_barrier_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::double_barrier_knockout_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::margrabe_exchange_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::stulz_max_call_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::stulz_min_call_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::variance_swap_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::black_swaption_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::bachelier_swaption_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::black_caplet_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::mc_american_lsm_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::mc_asian_arithmetic_py, m)?)?;

    // Phase 36: Models, MC engines, vol surfaces, risk
    m.add_function(wrap_pyfunction!(pricing::bates_price_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::mc_heston_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::mc_bates_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::mc_asian_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::cliquet_price_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::equity_risk_ladder_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::sabr_smile_vol_py, m)?)?;
    m.add_function(wrap_pyfunction!(pricing::svi_smile_vol_py, m)?)?;

    // Phase 36: Fixed income, credit, trees
    m.add_function(wrap_pyfunction!(fixed_income::price_swap_py, m)?)?;
    m.add_function(wrap_pyfunction!(fixed_income::price_bond_py, m)?)?;
    m.add_function(wrap_pyfunction!(fixed_income::bond_duration_py, m)?)?;
    m.add_function(wrap_pyfunction!(fixed_income::bond_convexity_py, m)?)?;
    m.add_function(wrap_pyfunction!(fixed_income::bond_dv01_py, m)?)?;
    m.add_function(wrap_pyfunction!(fixed_income::bond_z_spread_py, m)?)?;
    m.add_function(wrap_pyfunction!(fixed_income::hw_bond_option_py, m)?)?;
    m.add_function(wrap_pyfunction!(fixed_income::hw_caplet_py, m)?)?;
    m.add_function(wrap_pyfunction!(fixed_income::hw_jamshidian_swaption_py, m)?)?;
    m.add_function(wrap_pyfunction!(fixed_income::tree_swaption_py, m)?)?;
    m.add_function(wrap_pyfunction!(fixed_income::tree_cap_floor_py, m)?)?;
    m.add_function(wrap_pyfunction!(fixed_income::tree_bond_price_py, m)?)?;
    m.add_function(wrap_pyfunction!(fixed_income::midpoint_cds_py, m)?)?;
    m.add_function(wrap_pyfunction!(fixed_income::price_callable_bond_py, m)?)?;
    m.add_function(wrap_pyfunction!(fixed_income::cdo_tranche_py, m)?)?;

    // Term structure bootstrapping
    m.add_function(wrap_pyfunction!(termstructures::bootstrap_yield_curve, m)?)?;

    // Calendar utilities
    m.add_function(wrap_pyfunction!(time::is_business_day, m)?)?;
    m.add_function(wrap_pyfunction!(time::advance_date, m)?)?;
    m.add_function(wrap_pyfunction!(time::year_fraction, m)?)?;
    m.add_function(wrap_pyfunction!(time::business_days_between, m)?)?;

    Ok(())
}
