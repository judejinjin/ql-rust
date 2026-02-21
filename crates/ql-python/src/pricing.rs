//! Pricing function wrappers for Python.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use ql_instruments::{OptionType, Payoff, Exercise, VanillaOption};
use ql_pricingengines::analytic_european::{price_european, implied_volatility};
use ql_pricingengines::american_engines::{barone_adesi_whaley, bjerksund_stensland};
use ql_pricingengines::analytic_heston::heston_price;
use ql_pricingengines::multi_asset::{kirk_spread_call, kirk_spread_put};
use ql_methods::mc_engines::{mc_european, mc_barrier};
use ql_methods::finite_differences::fd_black_scholes;
use ql_methods::lattice::binomial_crr;
use ql_models::HestonModel;
use ql_termstructures::sabr::sabr_volatility;

use crate::types::{
    PyAnalyticResults, PyMCResult, PyLatticeResult,
    PyAmericanResult, PyFDResult, PyHestonResult,
};

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

// ---------------------------------------------------------------------------
// American option approximations
// ---------------------------------------------------------------------------

/// Barone-Adesi-Whaley quadratic approximation for American options.
///
/// Returns an ``AmericanResult`` with npv, early_exercise_premium, critical_price.
#[pyfunction]
#[pyo3(signature = (spot, strike, r, q, vol, t, is_call=true))]
pub fn barone_adesi_whaley_py(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    is_call: bool,
) -> PyAmericanResult {
    let res = barone_adesi_whaley(spot, strike, r, q, vol, t, is_call);
    PyAmericanResult {
        npv: res.npv,
        early_exercise_premium: res.early_exercise_premium,
        critical_price: res.critical_price,
    }
}

/// Bjerksund-Stensland flat-boundary approximation for American options.
///
/// Returns an ``AmericanResult`` with npv, early_exercise_premium, critical_price.
#[pyfunction]
#[pyo3(signature = (spot, strike, r, q, vol, t, is_call=true))]
pub fn bjerksund_stensland_py(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    is_call: bool,
) -> PyAmericanResult {
    let res = bjerksund_stensland(spot, strike, r, q, vol, t, is_call);
    PyAmericanResult {
        npv: res.npv,
        early_exercise_premium: res.early_exercise_premium,
        critical_price: res.critical_price,
    }
}

// ---------------------------------------------------------------------------
// Finite Differences
// ---------------------------------------------------------------------------

/// Price an option using Crank-Nicolson finite differences (Black-Scholes PDE).
///
/// Parameters:
///   spot, strike, r, q, vol, t — market parameters
///   is_call — True for call, False for put
///   is_american — True for American exercise constraint
///   num_space — spatial grid points (default 200)
///   num_time — time steps (default 200)
///
/// Returns an ``FDResult`` with npv, delta, gamma, theta.
#[pyfunction]
#[pyo3(signature = (spot, strike, r, q, vol, t, is_call=true, is_american=false, num_space=200, num_time=200))]
pub fn fd_black_scholes_py(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    is_call: bool,
    is_american: bool,
    num_space: usize,
    num_time: usize,
) -> PyFDResult {
    let res = fd_black_scholes(spot, strike, r, q, vol, t, is_call, is_american, num_space, num_time);
    PyFDResult {
        npv: res.npv,
        delta: res.delta,
        gamma: res.gamma,
        theta: res.theta,
    }
}

// ---------------------------------------------------------------------------
// Heston pricing
// ---------------------------------------------------------------------------

/// Price a European option under the Heston stochastic volatility model.
///
/// Parameters:
///   spot — underlying price
///   strike — strike price
///   r — risk-free rate
///   q — dividend yield
///   v0 — initial variance
///   kappa — mean-reversion speed
///   theta — long-run variance
///   sigma — vol-of-vol
///   rho — correlation (spot vs variance)
///   t — time to expiry
///   is_call — True for call, False for put
///
/// Returns a ``HestonResult`` with npv, p1, p2.
#[pyfunction]
#[pyo3(signature = (spot, strike, r, q, v0, kappa, theta, sigma, rho, t, is_call=true))]
#[allow(clippy::too_many_arguments)]
pub fn heston_price_py(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    t: f64,
    is_call: bool,
) -> PyHestonResult {
    let model = HestonModel::new(spot, r, q, v0, kappa, theta, sigma, rho);
    let res = heston_price(&model, strike, t, is_call);
    PyHestonResult {
        npv: res.npv,
        p1: res.p1,
        p2: res.p2,
    }
}

// ---------------------------------------------------------------------------
// MC Barrier
// ---------------------------------------------------------------------------

/// Price a barrier option via Monte Carlo under GBM.
///
/// Parameters:
///   spot, strike — underlying and strike price
///   barrier — barrier level
///   rebate — rebate paid on knock-out or if knock-in not triggered (default 0)
///   r, q, vol, t — market parameters
///   is_call — True for call, False for put
///   is_up — True for up-barrier, False for down-barrier
///   is_knock_in — True for knock-in, False for knock-out
///   num_paths — MC paths (default 100_000)
///   num_steps — time steps per path (default 252)
///   seed — RNG seed (default 42)
///
/// Returns an ``MCResult`` with npv, std_error, num_paths.
#[pyfunction]
#[pyo3(signature = (spot, strike, barrier, rebate=0.0, r=0.05, q=0.0, vol=0.2, t=1.0, is_call=true, is_up=true, is_knock_in=false, num_paths=100_000, num_steps=252, seed=42))]
#[allow(clippy::too_many_arguments)]
pub fn mc_barrier_py(
    spot: f64,
    strike: f64,
    barrier: f64,
    rebate: f64,
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    is_call: bool,
    is_up: bool,
    is_knock_in: bool,
    num_paths: usize,
    num_steps: usize,
    seed: u64,
) -> PyMCResult {
    let opt_type = if is_call { OptionType::Call } else { OptionType::Put };
    let res = mc_barrier(spot, strike, barrier, rebate, r, q, vol, t, opt_type, is_up, is_knock_in, num_paths, num_steps, seed);
    PyMCResult {
        npv: res.npv,
        std_error: res.std_error,
        num_paths: res.num_paths,
    }
}

// ---------------------------------------------------------------------------
// Kirk spread option
// ---------------------------------------------------------------------------

/// Kirk's approximation for a European call on the spread S₁ − S₂.
///
/// Returns the option price as a float.
#[pyfunction]
#[pyo3(signature = (s1, s2, strike, r, q1, q2, vol1, vol2, rho, t))]
pub fn kirk_spread_call_py(
    s1: f64,
    s2: f64,
    strike: f64,
    r: f64,
    q1: f64,
    q2: f64,
    vol1: f64,
    vol2: f64,
    rho: f64,
    t: f64,
) -> f64 {
    kirk_spread_call(s1, s2, strike, r, q1, q2, vol1, vol2, rho, t)
}

/// Kirk's approximation for a European put on the spread S₁ − S₂.
///
/// Returns the option price as a float.
#[pyfunction]
#[pyo3(signature = (s1, s2, strike, r, q1, q2, vol1, vol2, rho, t))]
pub fn kirk_spread_put_py(
    s1: f64,
    s2: f64,
    strike: f64,
    r: f64,
    q1: f64,
    q2: f64,
    vol1: f64,
    vol2: f64,
    rho: f64,
    t: f64,
) -> f64 {
    kirk_spread_put(s1, s2, strike, r, q1, q2, vol1, vol2, rho, t)
}

// ---------------------------------------------------------------------------
// SABR volatility
// ---------------------------------------------------------------------------

/// Compute SABR implied Black volatility (Hagan et al. 2002).
///
/// Parameters:
///   strike — option strike (> 0)
///   forward — forward price (> 0)
///   expiry — time to expiry in years (> 0)
///   alpha — SABR alpha (initial vol level)
///   beta — SABR beta (0 = normal, 1 = log-normal)
///   rho — correlation [-1, 1]
///   nu — vol of vol (≥ 0)
///
/// Returns the Black implied volatility.
#[pyfunction]
#[pyo3(signature = (strike, forward, expiry, alpha, beta, rho, nu))]
pub fn sabr_vol_py(
    strike: f64,
    forward: f64,
    expiry: f64,
    alpha: f64,
    beta: f64,
    rho: f64,
    nu: f64,
) -> f64 {
    sabr_volatility(strike, forward, expiry, alpha, beta, rho, nu)
}
