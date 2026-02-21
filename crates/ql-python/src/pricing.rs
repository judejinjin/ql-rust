//! Pricing function wrappers for Python.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use ql_instruments::{OptionType, Payoff, Exercise, VanillaOption};
use ql_pricingengines::analytic_european::{price_european, implied_volatility};
use ql_methods::mc_engines::mc_european;
use ql_methods::lattice::binomial_crr;

use crate::types::{PyAnalyticResults, PyMCResult, PyLatticeResult};

// ---------------------------------------------------------------------------
// Black-Scholes analytic pricing
// ---------------------------------------------------------------------------

/// Price a European option using the Black-Scholes formula.
///
/// Parameters:
///   spot — underlying price
///   strike — strike price
///   r — risk-free rate (continuous)
///   q — dividend yield (continuous)
///   vol — volatility (annualized)
///   t — time to expiry in years
///   is_call — True for call, False for put
///
/// Returns an ``AnalyticResults`` with npv, delta, gamma, vega, theta, rho.
#[pyfunction]
#[pyo3(signature = (spot, strike, r, q, vol, t, is_call=true))]
pub fn price_european_bs(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    is_call: bool,
) -> PyAnalyticResults {
    let opt_type = if is_call { OptionType::Call } else { OptionType::Put };
    let payoff = Payoff::PlainVanilla { option_type: opt_type, strike };
    // Use a dummy date since price_european only needs the payoff+exercise type
    let expiry = ql_time::date::Date::from_ymd(2099, ql_time::date::Month::December, 31);
    let exercise = Exercise::European { expiry };
    let option = VanillaOption::new(payoff, exercise);

    let res = price_european(&option, spot, r, q, vol, t);
    PyAnalyticResults {
        npv: res.npv,
        delta: res.delta,
        gamma: res.gamma,
        vega: res.vega,
        theta: res.theta,
        rho: res.rho,
    }
}

/// Compute the implied volatility from a market price.
///
/// Parameters:
///   target_price — observed option price
///   spot — underlying price
///   strike — strike price
///   r — risk-free rate
///   q — dividend yield
///   t — time to expiry
///   is_call — True for call, False for put
///
/// Returns the implied volatility as a float.
///
/// Raises ValueError if no solution is found.
#[pyfunction]
#[pyo3(signature = (target_price, spot, strike, r, q, t, is_call=true))]
pub fn implied_vol(
    target_price: f64,
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    t: f64,
    is_call: bool,
) -> PyResult<f64> {
    let opt_type = if is_call { OptionType::Call } else { OptionType::Put };
    let payoff = Payoff::PlainVanilla { option_type: opt_type, strike };
    let expiry = ql_time::date::Date::from_ymd(2099, ql_time::date::Month::December, 31);
    let exercise = Exercise::European { expiry };
    let option = VanillaOption::new(payoff, exercise);

    implied_volatility(&option, target_price, spot, r, q, t)
        .map_err(|e| PyValueError::new_err(format!("implied vol error: {e}")))
}

// ---------------------------------------------------------------------------
// Monte Carlo
// ---------------------------------------------------------------------------

/// Price a European option using Monte Carlo simulation.
///
/// Parameters:
///   spot, strike, r, q, vol, t — market/option parameters
///   is_call — True for call, False for put
///   num_paths — number of simulation paths (default 100,000)
///   antithetic — use antithetic variates (default True)
///   seed — RNG seed (default 42)
///
/// Returns an ``MCResult`` with npv, std_error, num_paths.
#[pyfunction]
#[pyo3(signature = (spot, strike, r, q, vol, t, is_call=true, num_paths=100_000, antithetic=true, seed=42))]
pub fn mc_european_py(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    is_call: bool,
    num_paths: usize,
    antithetic: bool,
    seed: u64,
) -> PyMCResult {
    let opt_type = if is_call { OptionType::Call } else { OptionType::Put };
    let res = mc_european(spot, strike, r, q, vol, t, opt_type, num_paths, antithetic, seed);
    PyMCResult {
        npv: res.npv,
        std_error: res.std_error,
        num_paths: res.num_paths,
    }
}

// ---------------------------------------------------------------------------
// Binomial CRR
// ---------------------------------------------------------------------------

/// Price an option using the Cox-Ross-Rubinstein binomial tree.
///
/// Returns a ``LatticeResult`` with npv, delta, gamma, theta.
#[pyfunction]
#[pyo3(signature = (spot, strike, r, q, vol, t, is_call=true, is_american=false, num_steps=200))]
pub fn binomial_crr_py(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    is_call: bool,
    is_american: bool,
    num_steps: usize,
) -> PyLatticeResult {
    let res = binomial_crr(spot, strike, r, q, vol, t, is_call, is_american, num_steps);
    PyLatticeResult {
        npv: res.npv,
        delta: res.delta,
        gamma: res.gamma,
        theta: res.theta,
    }
}
