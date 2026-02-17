//! Property-based tests using proptest.
//!
//! These tests verify mathematical invariants that must hold for all valid
//! parameter combinations, not just specific examples.

use proptest::prelude::*;
use ql_instruments::VanillaOption;
use ql_methods::binomial_crr;
use ql_pricingengines::price_european;
use ql_termstructures::{FlatForward, YieldTermStructure};
use ql_time::{Date, DayCounter, Month};

/// Strategy for generating valid Black-Scholes parameters.
fn bs_params() -> impl Strategy<Value = (f64, f64, f64, f64, f64, f64)> {
    (
        50.0..200.0_f64,   // spot
        50.0..200.0_f64,   // strike
        0.001..0.15_f64,   // risk-free rate
        0.0..0.08_f64,     // dividend yield
        0.05..0.80_f64,    // volatility
        0.1..3.0_f64,      // time to expiry
    )
}

proptest! {
    /// Put-call parity must hold for all valid BS parameters:
    /// C - P = S·exp(-qT) - K·exp(-rT)
    #[test]
    fn prop_put_call_parity(
        (spot, strike, rate, div, vol, expiry) in bs_params()
    ) {
        let today = Date::from_ymd(2025, Month::January, 15);
        let expiry_date = today + (expiry * 365.0) as i32;

        let call = VanillaOption::european_call(strike, expiry_date);
        let put = VanillaOption::european_put(strike, expiry_date);

        let call_price = price_european(&call, spot, rate, div, vol, expiry).npv;
        let put_price = price_european(&put, spot, rate, div, vol, expiry).npv;

        let lhs = call_price - put_price;
        let rhs = spot * (-div * expiry).exp() - strike * (-rate * expiry).exp();

        let tol = 1e-8 * (spot + strike);
        prop_assert!((lhs - rhs).abs() < tol,
            "Put-call parity violated: |{} - {}| = {} > {} (S={}, K={}, r={}, q={}, σ={}, T={})",
            lhs, rhs, (lhs - rhs).abs(), tol, spot, strike, rate, div, vol, expiry);
    }

    /// Call price must be non-negative and bounded: 0 ≤ C ≤ S.
    #[test]
    fn prop_call_price_bounds(
        (spot, strike, rate, div, vol, expiry) in bs_params()
    ) {
        let today = Date::from_ymd(2025, Month::January, 15);
        let expiry_date = today + (expiry * 365.0) as i32;
        let call = VanillaOption::european_call(strike, expiry_date);
        let price = price_european(&call, spot, rate, div, vol, expiry).npv;

        prop_assert!(price >= 0.0,
            "Call price should be non-negative, got {}", price);
        prop_assert!(price <= spot,
            "Call price {} should not exceed spot {}", price, spot);
    }

    /// Put price must be non-negative and bounded: 0 ≤ P ≤ K·exp(-rT).
    #[test]
    fn prop_put_price_bounds(
        (spot, strike, rate, div, vol, expiry) in bs_params()
    ) {
        let today = Date::from_ymd(2025, Month::January, 15);
        let expiry_date = today + (expiry * 365.0) as i32;
        let put = VanillaOption::european_put(strike, expiry_date);
        let price = price_european(&put, spot, rate, div, vol, expiry).npv;
        let upper = strike * (-rate * expiry).exp();

        prop_assert!(price >= 0.0,
            "Put price should be non-negative, got {}", price);
        prop_assert!(price <= upper + 1e-10,
            "Put price {} should not exceed K·e^(-rT) = {}", price, upper);
    }

    /// Call delta must be in [0, 1].
    #[test]
    fn prop_call_delta_range(
        (spot, strike, rate, div, vol, expiry) in bs_params()
    ) {
        let today = Date::from_ymd(2025, Month::January, 15);
        let expiry_date = today + (expiry * 365.0) as i32;
        let call = VanillaOption::european_call(strike, expiry_date);
        let result = price_european(&call, spot, rate, div, vol, expiry);

        prop_assert!(result.delta >= -0.01 && result.delta <= 1.01,
            "Call delta = {} should be in [0, 1]", result.delta);
    }

    /// Put delta must be in [-1, 0].
    #[test]
    fn prop_put_delta_range(
        (spot, strike, rate, div, vol, expiry) in bs_params()
    ) {
        let today = Date::from_ymd(2025, Month::January, 15);
        let expiry_date = today + (expiry * 365.0) as i32;
        let put = VanillaOption::european_put(strike, expiry_date);
        let result = price_european(&put, spot, rate, div, vol, expiry);

        prop_assert!(result.delta >= -1.01 && result.delta <= 0.01,
            "Put delta = {} should be in [-1, 0]", result.delta);
    }

    /// Gamma must be non-negative for both calls and puts.
    #[test]
    fn prop_gamma_non_negative(
        (spot, strike, rate, div, vol, expiry) in bs_params()
    ) {
        let today = Date::from_ymd(2025, Month::January, 15);
        let expiry_date = today + (expiry * 365.0) as i32;
        let call = VanillaOption::european_call(strike, expiry_date);
        let result = price_european(&call, spot, rate, div, vol, expiry);

        prop_assert!(result.gamma >= -1e-12,
            "Gamma should be non-negative, got {}", result.gamma);
    }

    /// Vega must be non-negative for European options.
    #[test]
    fn prop_vega_non_negative(
        (spot, strike, rate, div, vol, expiry) in bs_params()
    ) {
        let today = Date::from_ymd(2025, Month::January, 15);
        let expiry_date = today + (expiry * 365.0) as i32;
        let call = VanillaOption::european_call(strike, expiry_date);
        let result = price_european(&call, spot, rate, div, vol, expiry);

        prop_assert!(result.vega >= -1e-12,
            "Vega should be non-negative, got {}", result.vega);
    }

    /// Discount factor must be in (0, 1] for positive rates and t > 0.
    #[test]
    fn prop_discount_factor_range(
        rate in 0.001..0.20_f64,
        t in 0.01..30.0_f64,
    ) {
        let today = Date::from_ymd(2025, Month::January, 15);
        let dc = DayCounter::Actual365Fixed;
        let curve = FlatForward::new(today, rate, dc);
        let df = curve.discount_t(t);

        prop_assert!(df > 0.0, "Discount factor should be positive, got {}", df);
        prop_assert!(df <= 1.0, "Discount factor should be ≤ 1, got {}", df);
    }

    /// Discount factors must be monotonically decreasing with time.
    #[test]
    fn prop_discount_factor_monotonic(
        rate in 0.001..0.20_f64,
        t1 in 0.01..15.0_f64,
        dt in 0.01..15.0_f64,
    ) {
        let today = Date::from_ymd(2025, Month::January, 15);
        let dc = DayCounter::Actual365Fixed;
        let curve = FlatForward::new(today, rate, dc);

        let t2 = t1 + dt;
        let df1 = curve.discount_t(t1);
        let df2 = curve.discount_t(t2);

        prop_assert!(df1 >= df2,
            "df({}) = {} should be >= df({}) = {} for rate {}", t1, df1, t2, df2, rate);
    }

    /// American option value must be >= European option value.
    /// We use near-ATM strikes to avoid numerical noise at deep OTM.
    #[test]
    fn prop_american_geq_european(
        spot in 80.0..120.0_f64,
        moneyness in 0.8..1.2_f64,
        vol in 0.15..0.50_f64,
        expiry in 0.5..2.0_f64,
    ) {
        let strike = spot * moneyness;
        let rate = 0.05;
        let div = 0.0;
        let today = Date::from_ymd(2025, Month::January, 15);
        let expiry_date = today + (expiry * 365.0) as i32;

        let put = VanillaOption::european_put(strike, expiry_date);
        let euro = price_european(&put, spot, rate, div, vol, expiry).npv;

        // American via CRR with 200 steps for accuracy
        let american = binomial_crr(
            spot, strike, rate, div, vol, expiry,
            false, true, 200,
        );

        // Allow small numerical tolerance proportional to price
        let tol = 1e-3 * euro.max(0.01);
        prop_assert!(american.npv >= euro - tol,
            "American put {} should be >= European put {} (S={}, K={}, σ={}, T={})",
            american.npv, euro, spot, strike, vol, expiry);
    }

    /// Higher volatility → higher option price (for both calls and puts).
    #[test]
    fn prop_price_increases_with_vol(
        spot in 60.0..150.0_f64,
        strike in 60.0..150.0_f64,
        rate in 0.001..0.10_f64,
        vol1 in 0.05..0.40_f64,
        dvol in 0.01..0.40_f64,
    ) {
        let div = 0.0;
        let expiry = 1.0;
        let vol2 = vol1 + dvol;
        let today = Date::from_ymd(2025, Month::January, 15);
        let expiry_date = today + 365;

        let call = VanillaOption::european_call(strike, expiry_date);
        let p1 = price_european(&call, spot, rate, div, vol1, expiry).npv;
        let p2 = price_european(&call, spot, rate, div, vol2, expiry).npv;

        prop_assert!(p2 >= p1 - 1e-10,
            "Higher vol {} → {} should give higher price {} → {}",
            vol1, vol2, p1, p2);
    }
}
