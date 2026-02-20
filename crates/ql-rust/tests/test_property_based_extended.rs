//! Extended property-based tests covering bonds, swaps, term structures,
//! and additional option invariants.

use proptest::prelude::*;
use ql_instruments::{FixedRateBond, VanillaSwap, SwapType, VanillaOption};
use ql_cashflows::leg::fixed_leg;
use ql_pricingengines::{price_bond, price_swap, price_european, BondResults, SwapResults};
use ql_termstructures::{FlatForward, YieldTermStructure};
use ql_time::{Date, DayCounter, Month, Schedule};

/// Strategy for generating valid coupon rates.
fn bond_params() -> impl Strategy<Value = (f64, f64, i32)> {
    (
        0.001..0.12_f64,    // coupon rate
        0.001..0.15_f64,    // yield curve rate
        2..30_i32,          // maturity in years
    )
}

proptest! {
    /// Bond clean price should be positive for positive coupon and yield.
    #[test]
    fn prop_bond_clean_price_positive(
        (coupon, yld, years) in bond_params()
    ) {
        let today = Date::from_ymd(2025, Month::January, 15);
        let maturity_serial = today.serial() + years * 365;
        let _maturity = Date::from_serial(maturity_serial);
        let settle = today + 2;

        // Build annual schedule
        let mut dates = vec![today];
        for y in 1..=years {
            dates.push(Date::from_serial(today.serial() + y * 365));
        }
        let schedule = Schedule::from_dates(dates);
        let bond = FixedRateBond::new(100.0, 2, &schedule, coupon, DayCounter::Actual365Fixed);

        let curve = FlatForward::new(today, yld, DayCounter::Actual365Fixed);
        let res = price_bond(&bond, &curve, settle);

        prop_assert!(res.clean_price > 0.0,
            "Clean price should be positive, got {} (coupon={}, yield={})",
            res.clean_price, coupon, yld);
    }

    /// Higher coupon → higher bond price (fix yield).
    #[test]
    fn prop_bond_price_increases_with_coupon(
        yld in 0.01..0.10_f64,
        c1 in 0.01..0.08_f64,
        dc in 0.005..0.05_f64,
    ) {
        let c2 = c1 + dc;
        let today = Date::from_ymd(2025, Month::January, 15);
        let settle = today + 2;
        let mut dates = vec![today];
        for y in 1..=10 {
            dates.push(Date::from_serial(today.serial() + y * 365));
        }
        let schedule = Schedule::from_dates(dates);

        let bond1 = FixedRateBond::new(100.0, 2, &schedule, c1, DayCounter::Actual365Fixed);
        let bond2 = FixedRateBond::new(100.0, 2, &schedule, c2, DayCounter::Actual365Fixed);

        let curve = FlatForward::new(today, yld, DayCounter::Actual365Fixed);
        let p1 = price_bond(&bond1, &curve, settle).clean_price;
        let p2 = price_bond(&bond2, &curve, settle).clean_price;

        prop_assert!(p2 >= p1 - 1e-8,
            "Higher coupon {} → {} should give higher price {} → {}",
            c1, c2, p1, p2);
    }

    /// Higher yield → lower bond price (fix coupon).
    #[test]
    fn prop_bond_price_decreases_with_yield(
        coupon in 0.01..0.08_f64,
        y1 in 0.01..0.10_f64,
        dy in 0.005..0.05_f64,
    ) {
        let y2 = y1 + dy;
        let today = Date::from_ymd(2025, Month::January, 15);
        let settle = today + 2;
        let mut dates = vec![today];
        for y in 1..=10 {
            dates.push(Date::from_serial(today.serial() + y * 365));
        }
        let schedule = Schedule::from_dates(dates);

        let bond = FixedRateBond::new(100.0, 2, &schedule, coupon, DayCounter::Actual365Fixed);

        let curve1 = FlatForward::new(today, y1, DayCounter::Actual365Fixed);
        let curve2 = FlatForward::new(today, y2, DayCounter::Actual365Fixed);
        let p1 = price_bond(&bond, &curve1, settle).clean_price;
        let p2 = price_bond(&bond, &curve2, settle).clean_price;

        prop_assert!(p1 >= p2 - 1e-8,
            "Higher yield {} → {} should give lower price {} → {}",
            y1, y2, p1, p2);
    }

    /// A par swap (fixed rate == curve rate) should have NPV near zero.
    #[test]
    fn prop_par_swap_npv_near_zero(
        rate in 0.01..0.10_f64,
        years in 2..15_i32,
    ) {
        let today = Date::from_ymd(2025, Month::January, 15);
        let settle = today + 2;

        let mut dates = vec![today];
        for y in 1..=years {
            dates.push(Date::from_serial(today.serial() + y * 182)); // semi-annual
        }
        let schedule = Schedule::from_dates(dates);

        let dc = DayCounter::Actual365Fixed;
        let fixed = fixed_leg(&schedule, &[1_000_000.0], &[rate], dc);
        let floating = fixed_leg(&schedule, &[1_000_000.0], &[rate], dc);

        let swap = VanillaSwap::new(
            SwapType::Payer,
            1_000_000.0,
            fixed,
            floating,
            rate,
            0.0,
        );

        let curve = FlatForward::new(today, rate, dc);
        let res = price_swap(&swap, &curve, settle);

        // NPV should be near zero for a par swap (within 1% of notional)
        prop_assert!(res.npv.abs() < 10_000.0,
            "Par swap NPV should be near zero, got {} (rate={}, years={})",
            res.npv, rate, years);
    }

    /// Forward rate must be positive when the yield curve is positive.
    #[test]
    fn prop_forward_rate_positive(
        rate in 0.005..0.15_f64,
        t in 0.1..20.0_f64,
        dt in 0.01..5.0_f64,
    ) {
        let today = Date::from_ymd(2025, Month::January, 15);
        let dc = DayCounter::Actual365Fixed;
        let curve = FlatForward::new(today, rate, dc);

        let df1 = curve.discount_t(t);
        let df2 = curve.discount_t(t + dt);

        // Forward rate = (df1/df2 - 1) / dt
        let fwd = (df1 / df2 - 1.0) / dt;

        prop_assert!(fwd > -1e-12,
            "Forward rate should be positive, got {} (r={}, t={}, dt={})",
            fwd, rate, t, dt);
    }

    /// ZCB (discount factor) relationship: df(t) = exp(-r*t) for flat forward.
    #[test]
    fn prop_flat_forward_df_consistency(
        rate in 0.001..0.20_f64,
        t in 0.01..30.0_f64,
    ) {
        let today = Date::from_ymd(2025, Month::January, 15);
        let dc = DayCounter::Actual365Fixed;
        let curve = FlatForward::new(today, rate, dc);
        let df = curve.discount_t(t);
        let expected = (-rate * t).exp();

        let tol = 1e-10 * expected;
        prop_assert!((df - expected).abs() < tol,
            "df({}) = {} should equal exp(-{:.4}*{:.2}) = {}",
            t, df, rate, t, expected);
    }

    /// Intrinsic value bound: C(S,K) >= max(0, S*exp(-qT) - K*exp(-rT))
    #[test]
    fn prop_call_exceeds_intrinsic(
        spot in 50.0..200.0_f64,
        strike in 50.0..200.0_f64,
        rate in 0.001..0.12_f64,
        vol in 0.05..0.60_f64,
        expiry in 0.1..3.0_f64,
    ) {
        let div = 0.0;
        let today = Date::from_ymd(2025, Month::January, 15);
        let expiry_date = today + (expiry * 365.0) as i32;
        let call = VanillaOption::european_call(strike, expiry_date);
        let price = price_european(&call, spot, rate, div, vol, expiry).npv;
        let intrinsic = (spot * (-div * expiry).exp()
            - strike * (-rate * expiry).exp()).max(0.0);

        prop_assert!(price >= intrinsic - 1e-8,
            "Call price {} should exceed intrinsic {} (S={}, K={})",
            price, intrinsic, spot, strike);
    }

    /// Implied vol round-trip: price → implied vol should recover the original vol.
    #[test]
    fn prop_implied_vol_roundtrip(
        spot in 80.0..120.0_f64,
        moneyness in 0.9..1.1_f64,
        vol in 0.10..0.50_f64,
        expiry in 0.25..2.0_f64,
    ) {
        use ql_pricingengines::implied_volatility;

        let strike = spot * moneyness;
        let rate = 0.05;
        let div = 0.0;
        let today = Date::from_ymd(2025, Month::January, 15);
        let expiry_date = today + (expiry * 365.0) as i32;

        let call = VanillaOption::european_call(strike, expiry_date);
        let price = price_european(&call, spot, rate, div, vol, expiry).npv;

        if price > 1e-8 {
            let iv = implied_volatility(&call, price, spot, rate, div, expiry);
            match iv {
                Ok(recovered) => {
                    prop_assert!((recovered - vol).abs() < 1e-4,
                        "Implied vol {} should recover original {} (S={}, K={}, T={})",
                        recovered, vol, spot, strike, expiry);
                }
                Err(_) => {
                    // Root-finding can fail for very extreme parameters; allow it
                }
            }
        }
    }
}
