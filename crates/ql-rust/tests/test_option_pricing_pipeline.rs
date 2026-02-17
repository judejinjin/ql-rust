//! Integration test: vol surface → process → engine → Greeks.
//!
//! Validates the full option pricing pipeline from market parameters
//! through pricing engines to Greeks and implied volatility.

use approx::assert_abs_diff_eq;
use ql_instruments::{OptionType, VanillaOption};
use ql_methods::{binomial_crr, fd_black_scholes, mc_european};
use ql_pricingengines::{implied_volatility, price_european};
use ql_time::{Date, DayCounter, Month, Schedule};

/// Black-Scholes: price a call and verify put-call parity.
#[test]
fn bs_put_call_parity() {
    let today = Date::from_ymd(2025, Month::January, 15);
    let spot = 100.0;
    let strike = 105.0;
    let vol = 0.20;
    let rate = 0.05;
    let div = 0.02;
    let expiry = 1.0;

    let call = VanillaOption::european_call(strike, today + 365);
    let put = VanillaOption::european_put(strike, today + 365);

    let call_result = price_european(&call, spot, rate, div, vol, expiry);
    let put_result = price_european(&put, spot, rate, div, vol, expiry);

    // Put-call parity: C - P = S·exp(-qT) - K·exp(-rT)
    let lhs = call_result.npv - put_result.npv;
    let rhs = spot * (-div * expiry).exp() - strike * (-rate * expiry).exp();
    assert_abs_diff_eq!(lhs, rhs, epsilon = 1e-10);
}

/// Implied volatility round-trip: price → implied vol → price.
#[test]
fn implied_vol_round_trip() {
    let today = Date::from_ymd(2025, Month::January, 15);
    let spot = 100.0;
    let strike = 110.0;
    let vol = 0.25;
    let rate = 0.05;
    let div = 0.0;
    let expiry = 0.5;

    let call = VanillaOption::european_call(strike, today + 182);
    let result = price_european(&call, spot, rate, div, vol, expiry);

    let iv = implied_volatility(&call, result.npv, spot, rate, div, expiry)
        .expect("Implied vol should converge");

    assert_abs_diff_eq!(iv, vol, epsilon = 1e-8);
}

/// Greeks: delta of ATM call ≈ 0.5 (with adjustments).
#[test]
fn atm_call_delta_approximately_half() {
    let today = Date::from_ymd(2025, Month::January, 15);
    let spot = 100.0;
    let strike = 100.0;
    let vol = 0.20;
    let rate = 0.05;
    let div = 0.0;
    let expiry = 1.0;

    let call = VanillaOption::european_call(strike, today + 365);
    let result = price_european(&call, spot, rate, div, vol, expiry);

    // ATM call delta should be around 0.5-0.6 (shifted by drift)
    assert!(result.delta > 0.4 && result.delta < 0.7,
        "ATM call delta = {} should be around 0.5", result.delta);

    // Gamma should be positive
    assert!(result.gamma > 0.0, "Gamma should be positive");

    // Vega should be positive
    assert!(result.vega > 0.0, "Vega should be positive");
}

/// Cross-validate: BS analytic vs MC vs FD vs Binomial for European call.
#[test]
fn cross_validate_european_call_four_methods() {
    let spot = 100.0;
    let strike = 105.0;
    let rate = 0.05;
    let div = 0.0;
    let vol = 0.20;
    let expiry = 1.0;

    let today = Date::from_ymd(2025, Month::January, 15);
    let call = VanillaOption::european_call(strike, today + 365);

    // 1. Analytic Black-Scholes
    let bs = price_european(&call, spot, rate, div, vol, expiry);

    // 2. Monte Carlo (high path count for accuracy)
    let mc = mc_european(
        spot, strike, rate, div, vol, expiry,
        OptionType::Call, 500_000, true, 42,
    );

    // 3. Finite Differences
    let fd = fd_black_scholes(
        spot, strike, rate, div, vol, expiry,
        true, false, 200, 200,
    );

    // 4. Binomial CRR
    let crr = binomial_crr(
        spot, strike, rate, div, vol, expiry,
        true, false, 500,
    );

    // All should agree within 1% of BS price
    let bs_price = bs.npv;
    let tol = bs_price * 0.01;

    assert_abs_diff_eq!(mc.npv, bs_price, epsilon = tol);
    assert_abs_diff_eq!(fd.npv, bs_price, epsilon = tol);
    assert_abs_diff_eq!(crr.npv, bs_price, epsilon = tol);
}

/// American put should be worth at least as much as European put.
#[test]
fn american_put_exceeds_european() {
    let spot = 100.0;
    let strike = 110.0; // ITM put
    let rate = 0.05;
    let div = 0.0;
    let vol = 0.30;
    let expiry = 1.0;

    let today = Date::from_ymd(2025, Month::January, 15);
    let put = VanillaOption::european_put(strike, today + 365);
    let euro_put = price_european(&put, spot, rate, div, vol, expiry);

    // American via FD
    let american_fd = fd_black_scholes(
        spot, strike, rate, div, vol, expiry,
        false, true, 200, 200,
    );

    // American via CRR
    let american_crr = binomial_crr(
        spot, strike, rate, div, vol, expiry,
        false, true, 500,
    );

    assert!(
        american_fd.npv >= euro_put.npv - 1e-6,
        "American put (FD) = {} should >= European put = {}",
        american_fd.npv, euro_put.npv
    );

    assert!(
        american_crr.npv >= euro_put.npv - 1e-6,
        "American put (CRR) = {} should >= European put = {}",
        american_crr.npv, euro_put.npv
    );
}

/// Bond pricing: coupon rate = discount rate → price ≈ par.
#[test]
fn bond_at_par_rate_prices_near_par() {
    let today = Date::from_ymd(2025, Month::January, 15);
    let dc = DayCounter::Actual365Fixed;
    let rate = 0.05;

    let schedule = Schedule::from_dates(vec![
        Date::from_ymd(2025, Month::January, 15),
        Date::from_ymd(2025, Month::July, 15),
        Date::from_ymd(2026, Month::January, 15),
        Date::from_ymd(2026, Month::July, 15),
        Date::from_ymd(2027, Month::January, 15),
    ]);

    let bond = ql_instruments::FixedRateBond::new(100.0, 2, &schedule, rate, dc);
    let curve = ql_termstructures::FlatForward::new(today, rate, dc);
    let result = ql_pricingengines::price_bond(&bond, &curve, today);

    // Clean price should be close to 100 (par)
    assert_abs_diff_eq!(result.clean_price, 100.0, epsilon = 1.5);
}
