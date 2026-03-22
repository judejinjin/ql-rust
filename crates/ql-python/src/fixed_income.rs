//! Fixed income, credit, and tree pricing wrappers for Python.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::sync::Arc;

use ql_instruments::vanilla_swap::{VanillaSwap, SwapType};
use ql_instruments::bond::FixedRateBond;
use ql_instruments::credit_default_swap::{CreditDefaultSwap, CdsProtectionSide, CdsPremiumPeriod};
use ql_instruments::callable_bond::{CallableBond, CallabilityType, CallabilityScheduleEntry};
use ql_pricingengines::discounting::{price_swap, price_bond};
use ql_pricingengines::hw_analytic::{hw_bond_option, hw_caplet, hw_jamshidian_swaption};
use ql_pricingengines::tree_swaption::{tree_swaption, tree_cap_floor, tree_bond_price};
use ql_pricingengines::cds_engine::midpoint_cds_engine;
use ql_pricingengines::callable_bond_engine::price_callable_bond;
use ql_pricingengines::credit_portfolio::{
    price_cdo_tranche, CreditBasket, CdoTranche, Issuer,
};
use ql_termstructures::yield_curves::FlatForward;
use ql_termstructures::yield_term_structure::YieldTermStructure;
use ql_termstructures::default_term_structure::{FlatHazardRate, DefaultProbabilityTermStructure};
use ql_cashflows::leg::fixed_leg;
use ql_cashflows::cashflow_analytics::duration as leg_duration;
use ql_cashflows::cashflow_analytics_extended::{convexity as leg_convexity, dv01 as leg_dv01, z_spread as leg_z_spread};
use ql_indexes::IborIndex;

use crate::time::{PyDate, parse_day_counter};
use crate::types::{
    PySwapResults, PyBondResults, PyHWAnalyticResult, PyTreeResult,
    PyCdsResult, PyCallableBondResult, PyCdoTrancheResult,
};

// ---------------------------------------------------------------------------
// Swap pricing (from schedules)
// ---------------------------------------------------------------------------

/// Price a fixed-for-floating vanilla interest rate swap.
///
/// Parameters:
///   swap_type — "Payer" or "Receiver"
///   nominal — notional amount
///   fixed_dates — list of fixed leg dates (schedule)
///   fixed_rate — fixed coupon rate
///   float_dates — list of floating leg dates (schedule)
///   float_spread — floating leg spread over index
///   reference_date — valuation date
///   flat_rate — flat yield curve rate for discounting
///   day_counter — e.g. "Actual365Fixed"
///
/// Returns a ``SwapResults`` with npv, fixed_leg_npv, floating_leg_npv, fair_rate.
#[pyfunction]
#[pyo3(signature = (swap_type, nominal, fixed_dates, fixed_rate, float_dates, float_spread, reference_date, flat_rate, day_counter="Actual365Fixed"))]
#[allow(clippy::too_many_arguments)]
pub fn price_swap_py(
    swap_type: &str,
    nominal: f64,
    fixed_dates: Vec<PyDate>,
    fixed_rate: f64,
    float_dates: Vec<PyDate>,
    float_spread: f64,
    reference_date: &PyDate,
    flat_rate: f64,
    day_counter: &str,
) -> PyResult<PySwapResults> {
    let dc = parse_day_counter(day_counter)?;
    let st = match swap_type {
        "Payer" | "payer" => SwapType::Payer,
        "Receiver" | "receiver" => SwapType::Receiver,
        o => return Err(PyValueError::new_err(format!("Unknown swap type '{o}'; use Payer or Receiver"))),
    };
    let fixed_sched = ql_time::Schedule::from_dates(
        fixed_dates.iter().map(|d| d.inner).collect(),
    );
    let float_sched = ql_time::Schedule::from_dates(
        float_dates.iter().map(|d| d.inner).collect(),
    );
    let index = IborIndex::euribor_6m();
    let swap = VanillaSwap::from_schedules(
        st, nominal, &fixed_sched, fixed_rate, dc,
        &float_sched, &index, float_spread, dc,
    );
    let curve = FlatForward::new(reference_date.inner, flat_rate, dc);
    let res = price_swap(&swap, &curve, reference_date.inner);
    Ok(PySwapResults {
        npv: res.npv,
        fixed_leg_npv: res.fixed_leg_npv,
        floating_leg_npv: res.floating_leg_npv,
        fair_rate: res.fair_rate,
    })
}

// ---------------------------------------------------------------------------
// Bond pricing (from schedule)
// ---------------------------------------------------------------------------

/// Price a fixed-rate bond on a flat yield curve.
///
/// Parameters:
///   face — face amount
///   settlement_days — number of settlement days
///   schedule_dates — coupon schedule dates
///   coupon_rate — annual coupon rate
///   reference_date — valuation date
///   flat_rate — flat yield curve rate
///   day_counter — day count convention
///
/// Returns a ``BondResults`` with npv, clean_price, dirty_price, accrued_interest.
#[pyfunction]
#[pyo3(signature = (face, settlement_days, schedule_dates, coupon_rate, reference_date, flat_rate, day_counter="Actual365Fixed"))]
#[allow(clippy::too_many_arguments)]
pub fn price_bond_py(
    face: f64,
    settlement_days: u32,
    schedule_dates: Vec<PyDate>,
    coupon_rate: f64,
    reference_date: &PyDate,
    flat_rate: f64,
    day_counter: &str,
) -> PyResult<PyBondResults> {
    let dc = parse_day_counter(day_counter)?;
    let sched = ql_time::Schedule::from_dates(
        schedule_dates.iter().map(|d| d.inner).collect(),
    );
    let bond = FixedRateBond::new(face, settlement_days, &sched, coupon_rate, dc);
    let curve = FlatForward::new(reference_date.inner, flat_rate, dc);
    let res = price_bond(&bond, &curve, reference_date.inner);
    Ok(PyBondResults {
        npv: res.npv,
        clean_price: res.clean_price,
        dirty_price: res.dirty_price,
        accrued_interest: res.accrued_interest,
    })
}

// ---------------------------------------------------------------------------
// Bond analytics (duration, convexity, DV01, Z-spread)
// ---------------------------------------------------------------------------

/// Macaulay duration of a fixed‐rate bond on a flat curve.
#[pyfunction]
#[pyo3(signature = (face, schedule_dates, coupon_rate, reference_date, flat_rate, day_counter="Actual365Fixed"))]
#[allow(clippy::too_many_arguments)]
pub fn bond_duration_py(
    face: f64,
    schedule_dates: Vec<PyDate>,
    coupon_rate: f64,
    reference_date: &PyDate,
    flat_rate: f64,
    day_counter: &str,
) -> PyResult<f64> {
    let dc = parse_day_counter(day_counter)?;
    let sched = ql_time::Schedule::from_dates(schedule_dates.iter().map(|d| d.inner).collect());
    let n = sched.dates().len().saturating_sub(1);
    let notionals = vec![face; n];
    let rates = vec![coupon_rate; n];
    let leg = fixed_leg(&sched, &notionals, &rates, dc);
    let curve = FlatForward::new(reference_date.inner, flat_rate, dc);
    Ok(leg_duration(&leg, &curve, reference_date.inner))
}

/// Convexity of a fixed‐rate bond on a flat curve.
#[pyfunction]
#[pyo3(signature = (face, schedule_dates, coupon_rate, reference_date, flat_rate, day_counter="Actual365Fixed"))]
#[allow(clippy::too_many_arguments)]
pub fn bond_convexity_py(
    face: f64,
    schedule_dates: Vec<PyDate>,
    coupon_rate: f64,
    reference_date: &PyDate,
    flat_rate: f64,
    day_counter: &str,
) -> PyResult<f64> {
    let dc = parse_day_counter(day_counter)?;
    let sched = ql_time::Schedule::from_dates(schedule_dates.iter().map(|d| d.inner).collect());
    let n = sched.dates().len().saturating_sub(1);
    let notionals = vec![face; n];
    let rates = vec![coupon_rate; n];
    let leg = fixed_leg(&sched, &notionals, &rates, dc);
    let curve = FlatForward::new(reference_date.inner, flat_rate, dc);
    Ok(leg_convexity(&leg, &curve, reference_date.inner))
}

/// DV01 (dollar duration) of a fixed‐rate bond.
#[pyfunction]
#[pyo3(signature = (face, schedule_dates, coupon_rate, reference_date, flat_rate, day_counter="Actual365Fixed"))]
#[allow(clippy::too_many_arguments)]
pub fn bond_dv01_py(
    face: f64,
    schedule_dates: Vec<PyDate>,
    coupon_rate: f64,
    reference_date: &PyDate,
    flat_rate: f64,
    day_counter: &str,
) -> PyResult<f64> {
    let dc = parse_day_counter(day_counter)?;
    let sched = ql_time::Schedule::from_dates(schedule_dates.iter().map(|d| d.inner).collect());
    let n = sched.dates().len().saturating_sub(1);
    let notionals = vec![face; n];
    let rates = vec![coupon_rate; n];
    let leg = fixed_leg(&sched, &notionals, &rates, dc);
    let curve = FlatForward::new(reference_date.inner, flat_rate, dc);
    Ok(leg_dv01(&leg, &curve, reference_date.inner))
}

/// Z-spread of a fixed-rate bond given a target price.
#[pyfunction]
#[pyo3(signature = (face, schedule_dates, coupon_rate, reference_date, flat_rate, target_price, day_counter="Actual365Fixed", accuracy=1e-7, max_iterations=100))]
#[allow(clippy::too_many_arguments)]
pub fn bond_z_spread_py(
    face: f64,
    schedule_dates: Vec<PyDate>,
    coupon_rate: f64,
    reference_date: &PyDate,
    flat_rate: f64,
    target_price: f64,
    day_counter: &str,
    accuracy: f64,
    max_iterations: usize,
) -> PyResult<f64> {
    let dc = parse_day_counter(day_counter)?;
    let sched = ql_time::Schedule::from_dates(schedule_dates.iter().map(|d| d.inner).collect());
    let n = sched.dates().len().saturating_sub(1);
    let notionals = vec![face; n];
    let rates = vec![coupon_rate; n];
    let leg = fixed_leg(&sched, &notionals, &rates, dc);
    let curve = FlatForward::new(reference_date.inner, flat_rate, dc);
    leg_z_spread(&leg, &curve, reference_date.inner, target_price, accuracy, max_iterations)
        .map_err(|e| PyValueError::new_err(format!("Z-spread solver: {e}")))
}

// ---------------------------------------------------------------------------
// Hull-White analytic engines
// ---------------------------------------------------------------------------

/// Hull-White analytic bond option price.
///
/// Parameters:
///   a — mean-reversion speed
///   sigma — short-rate vol
///   p_option — discount factor to option expiry
///   p_bond — discount factor to bond maturity
///   option_expiry — option expiry (years)
///   bond_maturity — bond maturity (years)
///   strike — option strike price
///   is_call — True for call, False for put
#[pyfunction]
#[pyo3(signature = (a, sigma, p_option, p_bond, option_expiry, bond_maturity, strike, is_call=true))]
#[allow(clippy::too_many_arguments)]
pub fn hw_bond_option_py(
    a: f64, sigma: f64, p_option: f64, p_bond: f64,
    option_expiry: f64, bond_maturity: f64, strike: f64, is_call: bool,
) -> PyHWAnalyticResult {
    let r = hw_bond_option(a, sigma, p_option, p_bond, option_expiry, bond_maturity, strike, is_call);
    PyHWAnalyticResult { npv: r.npv }
}

/// Hull-White analytic caplet price.
#[pyfunction]
#[pyo3(signature = (a, sigma, p_fixing, p_payment, fixing_date, payment_date, strike_rate, notional))]
#[allow(clippy::too_many_arguments)]
pub fn hw_caplet_py(
    a: f64, sigma: f64, p_fixing: f64, p_payment: f64,
    fixing_date: f64, payment_date: f64, strike_rate: f64, notional: f64,
) -> PyHWAnalyticResult {
    let r = hw_caplet(a, sigma, p_fixing, p_payment, fixing_date, payment_date, strike_rate, notional);
    PyHWAnalyticResult { npv: r.npv }
}

/// Hull-White Jamshidian decomposition swaption price.
#[pyfunction]
#[pyo3(signature = (a, sigma, option_expiry, swap_tenors, fixed_rate, discount_factors, p_option, notional, is_payer=true))]
#[allow(clippy::too_many_arguments)]
pub fn hw_jamshidian_swaption_py(
    a: f64, sigma: f64, option_expiry: f64,
    swap_tenors: Vec<f64>, fixed_rate: f64,
    discount_factors: Vec<f64>, p_option: f64,
    notional: f64, is_payer: bool,
) -> PyHWAnalyticResult {
    let r = hw_jamshidian_swaption(
        a, sigma, option_expiry, &swap_tenors, fixed_rate,
        &discount_factors, p_option, notional, is_payer,
    );
    PyHWAnalyticResult { npv: r.npv }
}

// ---------------------------------------------------------------------------
// Trinomial tree engines
// ---------------------------------------------------------------------------

/// Price a European swaption on a Hull-White trinomial tree.
#[pyfunction]
#[pyo3(signature = (a, sigma, r0, option_expiry, swap_tenors, fixed_rate, notional, is_payer=true, n_steps=100))]
#[allow(clippy::too_many_arguments)]
pub fn tree_swaption_py(
    a: f64, sigma: f64, r0: f64, option_expiry: f64,
    swap_tenors: Vec<f64>, fixed_rate: f64,
    notional: f64, is_payer: bool, n_steps: usize,
) -> PyTreeResult {
    let r = tree_swaption(a, sigma, r0, option_expiry, &swap_tenors, fixed_rate, notional, is_payer, n_steps);
    PyTreeResult { npv: r.npv }
}

/// Price a cap or floor on a Hull-White trinomial tree.
#[pyfunction]
#[pyo3(signature = (a, sigma, r0, fixing_times, payment_times, strike, notional, is_cap=true, n_steps_per_period=50))]
#[allow(clippy::too_many_arguments)]
pub fn tree_cap_floor_py(
    a: f64, sigma: f64, r0: f64,
    fixing_times: Vec<f64>, payment_times: Vec<f64>,
    strike: f64, notional: f64, is_cap: bool, n_steps_per_period: usize,
) -> PyTreeResult {
    let r = tree_cap_floor(a, sigma, r0, &fixing_times, &payment_times, strike, notional, is_cap, n_steps_per_period);
    PyTreeResult { npv: r.npv }
}

/// Price a zero-coupon bond on a Hull-White trinomial tree.
#[pyfunction]
#[pyo3(signature = (a, sigma, r0, maturity, n_steps=100))]
pub fn tree_bond_price_py(a: f64, sigma: f64, r0: f64, maturity: f64, n_steps: usize) -> PyTreeResult {
    let r = tree_bond_price(a, sigma, r0, maturity, n_steps);
    PyTreeResult { npv: r.npv }
}

// ---------------------------------------------------------------------------
// Credit: CDS pricing
// ---------------------------------------------------------------------------

/// Price a credit default swap using the midpoint engine.
///
/// Uses a flat hazard rate for default probabilities and a flat
/// yield curve for discounting. Builds quarterly premium periods
/// internally.
///
/// Parameters:
///   notional — CDS notional
///   spread — running spread (annual, e.g. 0.01 = 100bp)
///   maturity_years — CDS maturity in years (integer, quarterly periods)
///   recovery_rate — recovery rate (e.g. 0.40)
///   hazard_rate — constant hazard rate
///   risk_free_rate — flat risk-free rate
///   is_buyer — True for protection buyer
///
/// Returns a ``CdsResult`` with npv, fair_spread, premium_leg_pv, protection_leg_pv.
#[pyfunction]
#[pyo3(signature = (notional, spread, maturity_years, recovery_rate, hazard_rate, risk_free_rate, is_buyer=true))]
#[allow(clippy::too_many_arguments)]
pub fn midpoint_cds_py(
    notional: f64, spread: f64, maturity_years: u32,
    recovery_rate: f64, hazard_rate: f64, risk_free_rate: f64,
    is_buyer: bool,
) -> PyResult<PyCdsResult> {
    let today = ql_time::Date::from_ymd(2025, ql_time::Month::January, 15);
    let dc = ql_time::DayCounter::Actual365Fixed;

    let side = if is_buyer { CdsProtectionSide::Buyer } else { CdsProtectionSide::Seller };

    // Build quarterly premium periods
    let n_quarters = maturity_years * 4;
    let mut periods = Vec::with_capacity(n_quarters as usize);
    for i in 0..n_quarters {
        let start = today + (i as i32 * 91);
        let end = today + ((i + 1) as i32 * 91);
        periods.push(CdsPremiumPeriod {
            accrual_start: start,
            accrual_end: end,
            payment_date: end,
            accrual_fraction: 0.25,
        });
    }

    let maturity = today + (n_quarters as i32 * 91);
    let cds = CreditDefaultSwap::new(side, notional, spread, maturity, recovery_rate, periods);

    let default_curve: Arc<dyn DefaultProbabilityTermStructure> =
        Arc::new(FlatHazardRate::new(today, hazard_rate, dc));
    let yield_curve: Arc<dyn YieldTermStructure> =
        Arc::new(FlatForward::new(today, risk_free_rate, dc));

    let res = midpoint_cds_engine(&cds, &default_curve, &yield_curve, 0.0);
    Ok(PyCdsResult {
        npv: res.npv,
        fair_spread: res.fair_spread,
        premium_leg_pv: res.premium_leg_pv,
        protection_leg_pv: res.protection_leg_pv,
    })
}

// ---------------------------------------------------------------------------
// Callable bond pricing
// ---------------------------------------------------------------------------

/// Price a callable bond using a trinomial tree.
///
/// Parameters:
///   face — face amount
///   schedule_dates — coupon schedule dates
///   coupon_rate — annual coupon rate
///   call_dates — list of call dates
///   call_prices — list of call prices (one per call date)
///   r — risk-free rate
///   rate_vol — volatility of the short rate
///   reference_date — valuation date
///   num_steps — tree steps (default 100)
///   day_counter — day count convention
#[pyfunction]
#[pyo3(signature = (face, schedule_dates, coupon_rate, call_dates, call_prices, r, rate_vol, reference_date, num_steps=100, day_counter="Actual365Fixed"))]
#[allow(clippy::too_many_arguments)]
pub fn price_callable_bond_py(
    face: f64,
    schedule_dates: Vec<PyDate>,
    coupon_rate: f64,
    call_dates: Vec<PyDate>,
    call_prices: Vec<f64>,
    r: f64,
    rate_vol: f64,
    reference_date: &PyDate,
    num_steps: usize,
    day_counter: &str,
) -> PyResult<PyCallableBondResult> {
    let dc = parse_day_counter(day_counter)?;
    let sched = ql_time::Schedule::from_dates(schedule_dates.iter().map(|d| d.inner).collect());
    let call_sched: Vec<CallabilityScheduleEntry> = call_dates.iter().zip(call_prices.iter())
        .map(|(d, &p)| CallabilityScheduleEntry { date: d.inner, price: p })
        .collect();
    let bond = CallableBond::new(face, 0, &sched, coupon_rate, dc, CallabilityType::Call, call_sched);
    let res = price_callable_bond(&bond, r, rate_vol, reference_date.inner, num_steps);
    Ok(PyCallableBondResult { npv: res.npv, oas_hint: res.oas_hint })
}

// ---------------------------------------------------------------------------
// CDO tranche pricing (LHP)
// ---------------------------------------------------------------------------

/// Price a CDO tranche using the Gaussian copula LHP model.
///
/// Parameters:
///   issuer_notionals — per-issuer notionals
///   default_probs — per-issuer default probabilities
///   recovery_rates — per-issuer recovery rates
///   correlation — asset correlation
///   maturity — tranche maturity (years)
///   attachment — tranche attachment point (e.g. 0.03)
///   detachment — tranche detachment point (e.g. 0.07)
///   tranche_spread — tranche spread
///   tranche_notional — tranche notional
///   risk_free_rate — flat discount rate
///   n_integration — Gauss-Hermite points (default 50)
#[pyfunction]
#[pyo3(signature = (issuer_notionals, default_probs, recovery_rates, correlation, maturity, attachment, detachment, tranche_spread, tranche_notional, risk_free_rate, n_integration=50))]
#[allow(clippy::too_many_arguments)]
pub fn cdo_tranche_py(
    issuer_notionals: Vec<f64>,
    default_probs: Vec<f64>,
    recovery_rates: Vec<f64>,
    correlation: f64,
    maturity: f64,
    attachment: f64,
    detachment: f64,
    tranche_spread: f64,
    tranche_notional: f64,
    risk_free_rate: f64,
    n_integration: usize,
) -> PyResult<PyCdoTrancheResult> {
    if issuer_notionals.len() != default_probs.len() || issuer_notionals.len() != recovery_rates.len() {
        return Err(PyValueError::new_err("issuer_notionals, default_probs, recovery_rates must have equal length"));
    }
    let issuers: Vec<Issuer> = issuer_notionals.iter().enumerate().map(|(i, &n)| {
        Issuer {
            name: format!("Issuer{}", i),
            notional: n,
            default_probability: default_probs[i],
            recovery_rate: recovery_rates[i],
            seniority: 0,
        }
    }).collect();
    let basket = CreditBasket { issuers, correlation, maturity };
    let tranche = CdoTranche { attachment, detachment, spread: tranche_spread, notional: tranche_notional };
    let res = price_cdo_tranche(&basket, &tranche, risk_free_rate, n_integration);
    Ok(PyCdoTrancheResult {
        expected_loss: res.expected_loss,
        fair_spread: res.fair_spread,
        protection_leg: res.protection_leg,
        premium_leg: res.premium_leg,
        delta: res.delta,
    })
}
