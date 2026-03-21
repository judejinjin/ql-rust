//! Pricing function wrappers for Python.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use ql_instruments::{OptionType, Payoff, Exercise, VanillaOption};
use ql_pricingengines::analytic_european::{price_european, implied_volatility};
use ql_pricingengines::american_engines::{barone_adesi_whaley, bjerksund_stensland};
use ql_pricingengines::analytic_heston::heston_price;
use ql_pricingengines::multi_asset::{kirk_spread_call, kirk_spread_put};
use ql_pricingengines::variance_gamma_engine::vg_cos_price;
use ql_pricingengines::cos_heston::cos_heston_price;
use ql_pricingengines::analytic_binary_barrier::{
    analytic_binary_barrier, BinaryBarrierType, BinaryPayoff, BinaryDirection,
};
use ql_pricingengines::analytic_vanilla_extra::{analytic_cev_price};
use ql_pricingengines::fd_heston_barrier::{fd_heston_barrier, FdBarrierType, FdHestonGridParams};
use ql_pricingengines::heston_hull_white_engine::heston_hull_white_price;
use ql_pricingengines::portfolio_credit::{bilateral_cva, cdo_spread_ladder};
use ql_instruments::quanto_option::{QuantoVanillaOption, price_quanto_vanilla};
use ql_methods::mc_engines::{mc_european, mc_barrier};
use ql_methods::finite_differences::fd_black_scholes;
use ql_methods::lattice::binomial_crr;
use ql_models::HestonModel;
use ql_models::VarianceGammaModel;
use ql_termstructures::sabr::sabr_volatility;

// New engines (QuantLib parity gap items 2-9)
use ql_pricingengines::analytic_asian::{
    asian_geometric_continuous_avg_price, asian_geometric_discrete_avg_price,
    asian_geometric_continuous_avg_strike, asian_geometric_discrete_avg_strike,
    asian_turnbull_wakeman, asian_levy,
};
use ql_pricingengines::basket_engines::{choi_basket_spread, dlz_basket_price};
use ql_pricingengines::vanilla_extra_engines::{
    ju_quadratic_american, integral_european_vanilla,
};
use ql_pricingengines::exotic_options::{
    partial_time_barrier, PartialBarrierType,
    two_asset_correlation, holder_extensible, writer_extensible,
};

use crate::types::{
    PyAnalyticResults, PyMCResult, PyLatticeResult,
    PyAmericanResult, PyFDResult, PyHestonResult,
    PyVgResult, PyQuantoResult, PyCvaResult,
    PyCosHestonResult, PyBinaryBarrierResult, PyCevResult,
    PyFdHestonBarrierResult, PyHhwResult, PyCdoTranche,
    // New result types
    PyAsianResult, PyJuAmericanResult, PyIntegralResult,
    PyBasketSpreadResult, PyPartialBarrierResult,
    PyTwoAssetCorrelationResult, PyExtensibleOptionResult,
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

// ---------------------------------------------------------------------------
// Variance Gamma pricing
// ---------------------------------------------------------------------------

/// Price a European option under the Variance Gamma model using the COS method.
///
/// Parameters:
///   sigma  — VG diffusion parameter (σ > 0)
///   nu     — variance rate (ν > 0)
///   theta  — drift of gamma subordinator (θ)
///   spot   — current underlying price
///   strike — option strike
///   tau    — time to expiry in years
///   r      — risk-free rate
///   q      — dividend yield
///   is_call — True for call, False for put
///
/// Returns a ``VGResult`` with npv, delta, vega.
#[pyfunction]
#[pyo3(signature = (sigma, nu, theta, spot, strike, tau, r, q=0.0, is_call=true, n_terms=128, l=10.0))]
#[allow(clippy::too_many_arguments)]
pub fn vg_price_py(
    sigma: f64,
    nu: f64,
    theta: f64,
    spot: f64,
    strike: f64,
    tau: f64,
    r: f64,
    q: f64,
    is_call: bool,
    n_terms: usize,
    l: f64,
) -> PyResult<PyVgResult> {
    let model = VarianceGammaModel::new(sigma, nu, theta);
    let opt_type = if is_call { OptionType::Call } else { OptionType::Put };
    let res = vg_cos_price(&model, spot, strike, tau, r, q, opt_type, n_terms, l)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyVgResult { npv: res.price, delta: res.delta, vega: res.vega })
}

// ---------------------------------------------------------------------------
// Quanto vanilla option pricing
// ---------------------------------------------------------------------------

/// Price a quanto vanilla option (fixed FX conversion).
///
/// The quanto adjustment modifies the effective dividend yield:
///   q_eff = r_foreign + rho_sfx * sigma_S * sigma_FX
///
/// Parameters:
///   spot         — underlying price in domestic currency
///   strike       — option strike
///   tau          — time to expiry
///   r_domestic   — domestic risk-free rate
///   r_foreign    — foreign risk-free rate (used as dividend yield)
///   sigma        — underlying volatility
///   sigma_fx     — FX volatility
///   rho_sfx      — correlation between underlying and FX
///   fixed_fx     — fixed FX rate (conversion multiplier)
///   is_call      — True for call
///
/// Returns a ``QuantoResult``.
#[pyfunction]
#[pyo3(signature = (spot, strike, tau, r_domestic, r_foreign, sigma, sigma_fx, rho_sfx, fixed_fx=1.0, is_call=true))]
pub fn price_quanto_vanilla_py(
    spot: f64,
    strike: f64,
    tau: f64,
    r_domestic: f64,
    r_foreign: f64,
    sigma: f64,
    sigma_fx: f64,
    rho_sfx: f64,
    fixed_fx: f64,
    is_call: bool,
) -> PyQuantoResult {
    let opt = QuantoVanillaOption {
        option_type: if is_call { OptionType::Call } else { OptionType::Put },
        spot,
        strike,
        tau,
        r_domestic,
        r_foreign,
        sigma,
        sigma_fx,
        rho_sfx,
        fixed_fx,
    };
    let res = price_quanto_vanilla(&opt);
    PyQuantoResult {
        npv: res.price,
        delta: res.delta,
        vega: res.vega,
        qvega: res.qvega,
        rho: res.rho,
        qlambda: res.qlambda,
    }
}

// ---------------------------------------------------------------------------
// Bilateral CVA / DVA
// ---------------------------------------------------------------------------

/// Compute bilateral CVA and DVA for an OTC derivative position.
///
/// Parameters:
///   times               — list of time grid points in years
///   expected_exposure   — expected exposure EE(t) at each time
///   negative_exposure   — NEE(t) = −E[min(V,0)] at each time
///   hazard_c            — counterparty hazard rate
///   hazard_b            — own hazard rate
///   recovery_c          — counterparty recovery rate
///   recovery_b          — own recovery rate
///   discount_factors    — risk-free discount factors at each time
///
/// Returns a ``CVAResult`` with cva, dva, bcva.
#[pyfunction]
pub fn bilateral_cva_py(
    times: Vec<f64>,
    expected_exposure: Vec<f64>,
    negative_exposure: Vec<f64>,
    hazard_c: f64,
    hazard_b: f64,
    recovery_c: f64,
    recovery_b: f64,
    discount_factors: Vec<f64>,
) -> PyResult<PyCvaResult> {
    if times.len() != expected_exposure.len()
        || times.len() != negative_exposure.len()
        || times.len() != discount_factors.len()
    {
        return Err(PyValueError::new_err("All input arrays must have the same length"));
    }
    let res = bilateral_cva(
        &times, &expected_exposure, &negative_exposure,
        hazard_c, hazard_b, recovery_c, recovery_b, &discount_factors,
    );
    Ok(PyCvaResult { cva: res.cva, dva: res.dva, bcva: res.bcva })
}

// ---------------------------------------------------------------------------
// COS Heston pricing
// ---------------------------------------------------------------------------

/// Price a European option under the Heston model using the COS (Fourier-cosine) method.
///
/// Parameters:
///   spot, strike, tau, r, q — standard market inputs
///   v0 — initial variance
///   kappa — mean-reversion speed
///   theta — long-run variance
///   sigma — vol-of-vol
///   rho — spot-variance correlation
///   is_call — True for call, False for put
///   n_terms — number of cosine series terms (0 = default 128)
///   l — truncation parameter (0 = default 12)
///
/// Returns a ``CosHestonResult``.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn cos_heston_price_py(
    spot: f64,
    strike: f64,
    tau: f64,
    r: f64,
    q: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    is_call: bool,
    n_terms: usize,
    l: f64,
) -> PyResult<PyCosHestonResult> {
    let model = HestonModel::new(spot, r, q, v0, kappa, theta, sigma, rho);
    let opt_type = if is_call { OptionType::Call } else { OptionType::Put };
    cos_heston_price(&model, spot, strike, tau, r, q, opt_type, n_terms, l)
        .map(|res| PyCosHestonResult {
            price: res.price,
            n_terms: res.n_terms,
            a: res.ab.0,
            b: res.ab.1,
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

// ---------------------------------------------------------------------------
// Binary barrier option
// ---------------------------------------------------------------------------

/// Price a European binary barrier option under Black-Scholes.
///
/// Parameters:
///   spot, strike, barrier, r, q, sigma, tau — market inputs
///   barrier_type — ``"DownAndIn"``, ``"DownAndOut"``, ``"UpAndIn"``, ``"UpAndOut"``
///   payoff_type  — ``"Cash"`` (pays K on exercise) or ``"Asset"`` (pays S)
///   direction    — ``"Call"`` or ``"Put"``
///
/// Returns a ``BinaryBarrierResult``.
#[pyfunction]
pub fn analytic_binary_barrier_py(
    spot: f64,
    strike: f64,
    barrier: f64,
    r: f64,
    q: f64,
    sigma: f64,
    tau: f64,
    barrier_type: &str,
    payoff_type: &str,
    direction: &str,
) -> PyResult<PyBinaryBarrierResult> {
    let bt = match barrier_type {
        "DownAndIn"  => BinaryBarrierType::DownAndIn,
        "DownAndOut" => BinaryBarrierType::DownAndOut,
        "UpAndIn"    => BinaryBarrierType::UpAndIn,
        "UpAndOut"   => BinaryBarrierType::UpAndOut,
        other => return Err(PyValueError::new_err(format!("Unknown barrier_type: {other}"))),
    };
    let pt = match payoff_type {
        "Cash" | "CashOrNothing" => BinaryPayoff::CashOrNothing,
        "Asset" | "AssetOrNothing" => BinaryPayoff::AssetOrNothing,
        other => return Err(PyValueError::new_err(format!("Unknown payoff_type: {other}"))),
    };
    let dir = match direction {
        "Call" => BinaryDirection::Call,
        "Put"  => BinaryDirection::Put,
        other => return Err(PyValueError::new_err(format!("Unknown direction: {other}"))),
    };
    let res = analytic_binary_barrier(spot, strike, barrier, r, q, sigma, tau, bt, pt, dir);
    Ok(PyBinaryBarrierResult { price: res.price, delta: res.delta })
}

// ---------------------------------------------------------------------------
// CEV model pricing
// ---------------------------------------------------------------------------

/// Price a European option under the Constant Elasticity of Variance (CEV) model.
///
/// Parameters:
///   spot, strike, tau, r, q — standard inputs
///   sigma — CEV coefficient σ
///   beta  — elasticity (β ≠ 1; use BS for β = 1)
///   is_call — True for call, False for put
///
/// Returns a ``CevResult``.
#[pyfunction]
pub fn analytic_cev_price_py(
    spot: f64,
    strike: f64,
    tau: f64,
    r: f64,
    q: f64,
    sigma: f64,
    beta: f64,
    is_call: bool,
) -> PyResult<PyCevResult> {
    let opt_type = if is_call { OptionType::Call } else { OptionType::Put };
    analytic_cev_price(spot, strike, tau, r, q, sigma, beta, opt_type)
        .map(|res| PyCevResult { price: res.price })
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

// ---------------------------------------------------------------------------
// FD Heston barrier
// ---------------------------------------------------------------------------

/// Price a European barrier option under the Heston model using a 2D PDE solver.
///
/// Parameters:
///   spot, strike, barrier, tau, r, q — market inputs
///   v0, kappa, theta, sigma_v, rho — Heston parameters
///   barrier_type — ``"DownAndOut"``, ``"UpAndOut"``, ``"DownAndIn"``, ``"UpAndIn"``
///   is_call — True for call, False for put
///   ns, nv, nt — grid dimensions (0 = defaults: 100/50/100)
///
/// Returns a ``FdHestonBarrierResult``.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn fd_heston_barrier_py(
    spot: f64,
    strike: f64,
    barrier: f64,
    tau: f64,
    r: f64,
    q: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma_v: f64,
    rho: f64,
    barrier_type: &str,
    is_call: bool,
    ns: usize,
    nv: usize,
    nt: usize,
) -> PyResult<PyFdHestonBarrierResult> {
    let bt = match barrier_type {
        "DownAndOut" => FdBarrierType::DownAndOut,
        "UpAndOut"   => FdBarrierType::UpAndOut,
        "DownAndIn"  => FdBarrierType::DownAndIn,
        "UpAndIn"    => FdBarrierType::UpAndIn,
        other => return Err(PyValueError::new_err(format!("Unknown barrier_type: {other}"))),
    };
    let model = HestonModel::new(spot, r, q, v0, kappa, theta, sigma_v, rho);
    let grid = FdHestonGridParams {
        ns: if ns == 0 { 100 } else { ns },
        nv: if nv == 0 { 50 } else { nv },
        nt: if nt == 0 { 100 } else { nt },
        ..FdHestonGridParams::default()
    };
    let res = fd_heston_barrier(&model, strike, tau, is_call, barrier, bt, &grid);
    Ok(PyFdHestonBarrierResult {
        price: res.price,
        delta: res.delta,
        gamma: res.gamma,
        vega:  res.vega,
        ns: res.ns, nv: res.nv, nt: res.nt,
    })
}

// ---------------------------------------------------------------------------
// Heston + Hull-White hybrid
// ---------------------------------------------------------------------------

/// Price a European option under the Heston + Hull-White hybrid model (A1HW approximation).
///
/// Parameters:
///   spot, strike, tau, r0, q — market inputs
///   hw_a, hw_sigma_r — Hull-White mean-reversion and rate vol
///   v0, kappa, theta, sigma_v, rho_sv — Heston parameters
///   equity_rate_rho — equity–rate correlation
///   is_call — True for call, False for put
///
/// Returns a ``HestonHullWhiteResult``.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn heston_hull_white_price_py(
    spot: f64,
    strike: f64,
    tau: f64,
    r0: f64,
    q: f64,
    hw_a: f64,
    hw_sigma_r: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma_v: f64,
    rho_sv: f64,
    equity_rate_rho: f64,
    is_call: bool,
) -> PyHhwResult {
    let opt_type = if is_call { OptionType::Call } else { OptionType::Put };
    let res = heston_hull_white_price(
        spot, strike, tau, r0, hw_a, hw_sigma_r,
        v0, kappa, theta, sigma_v, rho_sv,
        equity_rate_rho, q, opt_type,
    );
    PyHhwResult { price: res.price, v0_eff: res.v0_eff, xi: res.xi }
}

// ---------------------------------------------------------------------------
// CDO spread ladder
// ---------------------------------------------------------------------------

/// Compute CDO tranche fair spreads across a waterfall using the Gaussian copula LHP model.
///
/// Parameters:
///   attachments    — list of attachment points (e.g. [0.0, 0.03, 0.06, 0.09])
///   detachments    — list of detachment points (e.g. [0.03, 0.06, 0.09, 0.12])
///   default_prob   — portfolio-average 5Y default probability
///   correlation    — Gaussian copula factor correlation ρ
///   recovery       — recovery rate (e.g. 0.4)
///   n_names        — number of names in portfolio
///   maturity       — tranche maturity in years
///   flat_rate      — flat risk-free rate for discounting
///
/// Returns a list of ``CdoTrancheSpread`` objects.
#[pyfunction]
pub fn cdo_spread_ladder_py(
    attachments: Vec<f64>,
    detachments: Vec<f64>,
    default_prob: f64,
    correlation: f64,
    recovery: f64,
    n_names: usize,
    maturity: f64,
    flat_rate: f64,
) -> PyResult<Vec<PyCdoTranche>> {
    if attachments.len() != detachments.len() {
        return Err(PyValueError::new_err("attachments and detachments must have the same length"));
    }
    let tranches: Vec<(f64, f64)> = attachments.iter().zip(detachments.iter()).map(|(&a, &d)| (a, d)).collect();
    let res = cdo_spread_ladder(&tranches, default_prob, correlation, recovery, n_names, maturity, flat_rate);
    Ok(res.into_iter().map(|t| PyCdoTranche {
        attachment: t.attachment,
        detachment: t.detachment,
        expected_loss: t.expected_loss,
        fair_spread: t.fair_spread,
    }).collect())
}

// ---------------------------------------------------------------------------
// Asian option engines
// ---------------------------------------------------------------------------

/// Price a **continuous geometric average-price** Asian option (Kemna-Vorst).
///
/// Parameters:
///   spot, strike, r, q, vol, t — standard Black-Scholes inputs
///   is_call — True for call, False for put
///
/// Returns an ``AsianResult`` with npv, effective_vol, effective_forward.
#[pyfunction]
#[pyo3(signature = (spot, strike, r, q, vol, t, is_call=true))]
pub fn asian_continuous_geo_avg_price_py(
    spot: f64, strike: f64, r: f64, q: f64, vol: f64, t: f64, is_call: bool,
) -> PyAsianResult {
    let opt = if is_call { OptionType::Call } else { OptionType::Put };
    let res = asian_geometric_continuous_avg_price(spot, strike, r, q, vol, t, opt);
    PyAsianResult { npv: res.npv, effective_vol: res.effective_vol, effective_forward: res.effective_forward }
}

/// Price a **discrete geometric average-price** Asian option.
///
/// Parameters:
///   spot, strike, r, q, vol, t — standard Black-Scholes inputs
///   n — number of equally spaced averaging dates
///   is_call — True for call, False for put
#[pyfunction]
#[pyo3(signature = (spot, strike, r, q, vol, t, n, is_call=true))]
pub fn asian_discrete_geo_avg_price_py(
    spot: f64, strike: f64, r: f64, q: f64, vol: f64, t: f64, n: usize, is_call: bool,
) -> PyAsianResult {
    let opt = if is_call { OptionType::Call } else { OptionType::Put };
    let res = asian_geometric_discrete_avg_price(spot, strike, r, q, vol, t, n, opt);
    PyAsianResult { npv: res.npv, effective_vol: res.effective_vol, effective_forward: res.effective_forward }
}

/// Price a **continuous geometric average-strike** Asian option.
///
/// Parameters:
///   spot, r, q, vol, t — standard Black-Scholes inputs (no strike — payoff is S_T - G_T)
///   is_call — True for call, False for put
#[pyfunction]
#[pyo3(signature = (spot, r, q, vol, t, is_call=true))]
pub fn asian_continuous_geo_avg_strike_py(
    spot: f64, r: f64, q: f64, vol: f64, t: f64, is_call: bool,
) -> PyAsianResult {
    let opt = if is_call { OptionType::Call } else { OptionType::Put };
    let res = asian_geometric_continuous_avg_strike(spot, r, q, vol, t, opt);
    PyAsianResult { npv: res.npv, effective_vol: res.effective_vol, effective_forward: res.effective_forward }
}

/// Price a **discrete geometric average-strike** Asian option.
///
/// Parameters:
///   spot, r, q, vol, t — standard Black-Scholes inputs
///   n — number of averaging dates
///   is_call — True for call, False for put
#[pyfunction]
#[pyo3(signature = (spot, r, q, vol, t, n, is_call=true))]
pub fn asian_discrete_geo_avg_strike_py(
    spot: f64, r: f64, q: f64, vol: f64, t: f64, n: usize, is_call: bool,
) -> PyAsianResult {
    let opt = if is_call { OptionType::Call } else { OptionType::Put };
    let res = asian_geometric_discrete_avg_strike(spot, r, q, vol, t, n, opt);
    PyAsianResult { npv: res.npv, effective_vol: res.effective_vol, effective_forward: res.effective_forward }
}

/// Price an **arithmetic average-price** Asian option via Turnbull-Wakeman (1991).
///
/// Parameters:
///   spot, strike, r, q, vol, t — standard Black-Scholes inputs
///   t0 — time already elapsed in averaging window (0 for fresh option)
///   a  — accumulated average so far (0 if t0 == 0)
///   is_call — True for call, False for put
#[pyfunction]
#[pyo3(signature = (spot, strike, r, q, vol, t, t0=0.0, a=0.0, is_call=true))]
pub fn asian_turnbull_wakeman_py(
    spot: f64, strike: f64, r: f64, q: f64, vol: f64, t: f64,
    t0: f64, a: f64, is_call: bool,
) -> PyAsianResult {
    let opt = if is_call { OptionType::Call } else { OptionType::Put };
    let res = asian_turnbull_wakeman(spot, strike, r, q, vol, t, t0, a, opt);
    PyAsianResult { npv: res.npv, effective_vol: res.effective_vol, effective_forward: res.effective_forward }
}

/// Price an **arithmetic average-price** Asian option via Levy (1992).
///
/// Equivalent to `asian_turnbull_wakeman` with t0=0, a=0.
///
/// Parameters:
///   spot, strike, r, q, vol, t — standard Black-Scholes inputs
///   is_call — True for call, False for put
#[pyfunction]
#[pyo3(signature = (spot, strike, r, q, vol, t, is_call=true))]
pub fn asian_levy_py(
    spot: f64, strike: f64, r: f64, q: f64, vol: f64, t: f64, is_call: bool,
) -> PyAsianResult {
    let opt = if is_call { OptionType::Call } else { OptionType::Put };
    let res = asian_levy(spot, strike, r, q, vol, t, opt);
    PyAsianResult { npv: res.npv, effective_vol: res.effective_vol, effective_forward: res.effective_forward }
}

// ---------------------------------------------------------------------------
// Basket / spread engines
// ---------------------------------------------------------------------------

/// Price a European **spread option** (S1 - S2 - K) via Choi (2018).
///
/// Parameters:
///   s1, s2 — asset prices
///   r       — risk-free rate
///   q1, q2  — dividend yields
///   v1, v2  — volatilities
///   rho     — correlation between log-returns
///   t       — time to expiry
///   k       — spread strike (usually ≥ 0)
///   is_call — True for max(S1-S2-K,0), False for max(K-S1+S2,0)
///
/// Returns a ``BasketSpreadResult`` with npv, delta1, delta2.
#[pyfunction]
#[pyo3(signature = (s1, s2, r, q1, q2, v1, v2, rho, t, k=0.0, is_call=true))]
#[allow(clippy::too_many_arguments)]
pub fn choi_basket_spread_py(
    s1: f64, s2: f64, r: f64, q1: f64, q2: f64,
    v1: f64, v2: f64, rho: f64, t: f64, k: f64, is_call: bool,
) -> PyBasketSpreadResult {
    let res = choi_basket_spread(s1, s2, r, q1, q2, v1, v2, rho, t, k, is_call);
    PyBasketSpreadResult { npv: res.npv, delta1: res.delta1, delta2: res.delta2 }
}

/// Price an **N-asset arithmetic basket** option via Deng-Li-Zhou (2008).
///
/// Parameters:
///   spots   — list of current asset prices (length N)
///   weights — list of basket weights (must sum to something sensible, e.g. 1.0)
///   r       — risk-free rate
///   divs    — list of dividend yields (length N)
///   vols    — list of volatilities (length N)
///   corr    — correlation matrix, row-major flat list (N×N)
///   t       — time to expiry
///   strike  — strike on the weighted basket
///   is_call — True for call, False for put
///
/// Returns the NPV as a float.
///
/// Raises ValueError if dimension mismatches are detected.
#[pyfunction]
#[pyo3(signature = (spots, weights, r, divs, vols, corr, t, strike, is_call=true))]
pub fn dlz_basket_price_py(
    spots: Vec<f64>,
    weights: Vec<f64>,
    r: f64,
    divs: Vec<f64>,
    vols: Vec<f64>,
    corr: Vec<f64>,
    t: f64,
    strike: f64,
    is_call: bool,
) -> PyResult<f64> {
    let n = spots.len();
    if weights.len() != n || divs.len() != n || vols.len() != n || corr.len() != n * n {
        return Err(PyValueError::new_err(
            "spots/weights/divs/vols must all be length N; corr must be N×N (flat)"
        ));
    }
    Ok(dlz_basket_price(&spots, &weights, r, &divs, &vols, &corr, t, strike, is_call))
}

// ---------------------------------------------------------------------------
// Vanilla extra engines
// ---------------------------------------------------------------------------

/// Price an **American option** using the Ju-Zhong (1999) quadratic approximation.
///
/// More accurate than BAW near the early-exercise boundary. Includes a
/// correction term beyond the standard quadratic formula.
///
/// Parameters:
///   spot, strike, r, q, sigma, t — standard Black-Scholes inputs
///   is_call — True for call, False for put
///
/// Returns a ``JuAmericanResult`` with npv, delta, critical_price.
#[pyfunction]
#[pyo3(signature = (spot, strike, r, q, sigma, t, is_call=true))]
pub fn ju_quadratic_american_py(
    spot: f64, strike: f64, r: f64, q: f64, sigma: f64, t: f64, is_call: bool,
) -> PyJuAmericanResult {
    let res = ju_quadratic_american(spot, strike, r, q, sigma, t, is_call);
    PyJuAmericanResult { npv: res.npv, delta: res.delta, critical_price: res.critical_price }
}

/// Price a **European option** via 20-point Gauss-Hermite integration.
///
/// Supports an arbitrary payoff; this wrapper prices a vanilla call or put.
/// Useful for comparing with analytic prices or pricing near-vanilla exotics.
///
/// Parameters:
///   spot, strike, r, q, sigma, t — standard Black-Scholes inputs
///   is_call — True for call, False for put
///
/// Returns an ``IntegralResult`` with npv.
#[pyfunction]
#[pyo3(signature = (spot, strike, r, q, sigma, t, is_call=true))]
pub fn integral_european_py(
    spot: f64, strike: f64, r: f64, q: f64, sigma: f64, t: f64, is_call: bool,
) -> PyIntegralResult {
    let res = integral_european_vanilla(spot, strike, r, q, sigma, t, is_call);
    PyIntegralResult { npv: res.npv }
}

// ---------------------------------------------------------------------------
// Exotic option engines
// ---------------------------------------------------------------------------

/// Price a **partial-time barrier** option (Heynen-Kat 1994).
///
/// Barrier is active only during [0, t1]; full option expiry is at t.
///
/// Parameters:
///   spot, strike, barrier — asset price, strike, barrier level
///   r, q, sigma           — risk-free rate, dividend yield, volatility
///   t                     — total time to expiry
///   t1                    — end of barrier monitoring window (≤ t)
///   barrier_type          — one of ``"down_out"``, ``"down_in"``, ``"up_out"``, ``"up_in"``
///   is_call               — True for call, False for put
///
/// Returns a ``PartialBarrierResult`` with npv.
///
/// Raises ValueError on unknown barrier_type.
#[pyfunction]
#[pyo3(signature = (spot, strike, barrier, r, q, sigma, t, t1, barrier_type, is_call=true))]
pub fn partial_time_barrier_py(
    spot: f64, strike: f64, barrier: f64,
    r: f64, q: f64, sigma: f64,
    t: f64, t1: f64,
    barrier_type: &str,
    is_call: bool,
) -> PyResult<PyPartialBarrierResult> {
    let bt = match barrier_type {
        "down_out" => PartialBarrierType::B1DownOut,
        "down_in"  => PartialBarrierType::B1DownIn,
        "up_out"   => PartialBarrierType::B1UpOut,
        "up_in"    => PartialBarrierType::B1UpIn,
        other => return Err(PyValueError::new_err(format!(
            "Unknown barrier_type '{}'; use down_out/down_in/up_out/up_in", other
        ))),
    };
    let res = partial_time_barrier(spot, strike, barrier, r, q, sigma, t, t1, bt, is_call);
    Ok(PyPartialBarrierResult { npv: res.npv })
}

/// Price a **two-asset correlation** option (Zhang 1995).
///
/// Call payoff: max(S1 - K1, 0) * 1{S2 > K2}
/// Put payoff:  max(K1 - S1, 0) * 1{S2 < K2}
///
/// Parameters:
///   s1, s2     — current prices of assets 1 and 2
///   k1, k2     — strike on asset 1; barrier/condition on asset 2
///   r          — risk-free rate
///   q1, q2     — dividend yields
///   v1, v2     — volatilities
///   rho        — correlation between log-returns
///   t          — time to expiry
///   is_call    — True for call, False for put
///
/// Returns a ``TwoAssetCorrelationResult`` with npv, delta1, delta2.
#[pyfunction]
#[pyo3(signature = (s1, s2, k1, k2, r, q1, q2, v1, v2, rho, t, is_call=true))]
#[allow(clippy::too_many_arguments)]
pub fn two_asset_correlation_py(
    s1: f64, s2: f64, k1: f64, k2: f64,
    r: f64, q1: f64, q2: f64, v1: f64, v2: f64, rho: f64,
    t: f64, is_call: bool,
) -> PyTwoAssetCorrelationResult {
    let res = two_asset_correlation(s1, s2, k1, k2, r, q1, q2, v1, v2, rho, t, is_call);
    PyTwoAssetCorrelationResult { npv: res.npv, delta1: res.delta1, delta2: res.delta2 }
}

/// Price a **holder-extensible** option (Longstaff 1990).
///
/// The holder can extend the option at T1 by paying a premium,
/// resetting the strike to K2 and extending expiry to T2.
///
/// Parameters:
///   spot                — current asset price
///   k1, k2             — original and extension strikes
///   r, q, sigma        — market parameters
///   t1, t2             — original and extended expiries
///   extension_premium  — cost to the holder of extending
///   is_call            — True for call, False for put
///
/// Returns an ``ExtensibleOptionResult`` with npv.
#[pyfunction]
#[pyo3(signature = (spot, k1, k2, r, q, sigma, t1, t2, extension_premium=0.0, is_call=true))]
pub fn holder_extensible_py(
    spot: f64, k1: f64, k2: f64,
    r: f64, q: f64, sigma: f64,
    t1: f64, t2: f64,
    extension_premium: f64,
    is_call: bool,
) -> PyExtensibleOptionResult {
    let res = holder_extensible(spot, k1, k2, r, q, sigma, t1, t2, extension_premium, is_call);
    PyExtensibleOptionResult { npv: res.npv }
}

/// Price a **writer-extensible** option.
///
/// At T1 the writer (not the holder) may extend to T2 at strike K2
/// if doing so reduces their liability.
///
/// Parameters:
///   spot            — current asset price
///   k1, k2         — original and extension strikes
///   r, q, sigma    — market parameters
///   t1, t2         — original and extended expiries
///   is_call        — True for call, False for put
///
/// Returns an ``ExtensibleOptionResult`` with npv.
#[pyfunction]
#[pyo3(signature = (spot, k1, k2, r, q, sigma, t1, t2, is_call=true))]
pub fn writer_extensible_py(
    spot: f64, k1: f64, k2: f64,
    r: f64, q: f64, sigma: f64,
    t1: f64, t2: f64,
    is_call: bool,
) -> PyExtensibleOptionResult {
    let res = writer_extensible(spot, k1, k2, r, q, sigma, t1, t2, is_call);
    PyExtensibleOptionResult { npv: res.npv }
}
