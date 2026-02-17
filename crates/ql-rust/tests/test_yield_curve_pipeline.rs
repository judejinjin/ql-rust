//! Integration test: market data → bootstrap → discount → forward rate.
//!
//! Validates the full yield curve pipeline from raw market quotes through
//! bootstrapping to derived quantities (discount factors, zero rates, forward rates).

use approx::assert_abs_diff_eq;
use ql_termstructures::{
    DepositRateHelper, FlatForward, PiecewiseYieldCurve, RateHelper, SwapRateHelper,
    YieldTermStructure,
};
use ql_time::{Date, DayCounter, Month};

/// Build a realistic yield curve from deposit + swap quotes and verify consistency.
#[test]
fn bootstrap_and_query_curve() {
    let today = Date::from_ymd(2025, Month::January, 15);
    let dc = DayCounter::Actual365Fixed;

    let mut helpers: Vec<Box<dyn RateHelper>> = vec![
        // 3M deposit at 4.5%
        Box::new(DepositRateHelper::new(0.045, today, today + 91, dc)),
        // 6M deposit at 4.6%
        Box::new(DepositRateHelper::new(0.046, today, today + 182, dc)),
        // 2Y swap at 4.8%
        Box::new(SwapRateHelper::new(
            0.048,
            today,
            vec![today + 365, today + 730],
            dc,
        )),
        // 5Y swap at 5.0%
        Box::new(SwapRateHelper::new(
            0.050,
            today,
            vec![
                today + 365,
                today + 730,
                today + 1095,
                today + 1461,
                today + 1826,
            ],
            dc,
        )),
    ];

    let curve = PiecewiseYieldCurve::new(today, &mut helpers, dc, 1e-12)
        .expect("Bootstrap should succeed");

    // Verify: discount factor at t=0 is 1
    assert_abs_diff_eq!(curve.discount_t(0.0), 1.0, epsilon = 1e-14);

    // Verify: discount factors are monotonically decreasing
    let times = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0];
    let dfs: Vec<f64> = times.iter().map(|&t| curve.discount_t(t)).collect();
    for w in dfs.windows(2) {
        assert!(w[0] > w[1], "df({}) = {} should be > df(next) = {}", 0.0, w[0], w[1]);
    }

    // Verify: all discount factors in (0, 1]
    for &df in &dfs {
        assert!(df > 0.0 && df <= 1.0, "df = {} out of range (0, 1]", df);
    }

    // Verify: zero rates are positive (computed from discount factors)
    for &t in &times {
        let df = curve.discount_t(t);
        let zero = -df.ln() / t;
        assert!(zero > 0.0, "zero rate at t={} should be positive, got {}", t, zero);
    }

    // Verify: forward rates are positive (approximate)
    for i in 0..times.len() - 1 {
        let t1 = times[i];
        let t2 = times[i + 1];
        let df1 = curve.discount_t(t1);
        let df2 = curve.discount_t(t2);
        let fwd = -(df2 / df1).ln() / (t2 - t1);
        assert!(fwd > 0.0, "forward rate [{}, {}] should be positive, got {}", t1, t2, fwd);
    }
}

/// Flat forward curve: zero rate and discount factor are self-consistent.
#[test]
fn flat_forward_consistency() {
    let today = Date::from_ymd(2025, Month::January, 15);
    let rate = 0.05;
    let dc = DayCounter::Actual365Fixed;
    let curve = FlatForward::new(today, rate, dc);

    for &t in &[0.25, 0.5, 1.0, 2.0, 5.0, 10.0] {
        let df = curve.discount_t(t);
        let expected_df = (-rate * t).exp();
        assert_abs_diff_eq!(df, expected_df, epsilon = 1e-12);

        // Zero rate from df should match the flat rate
        let zero = -df.ln() / t;
        assert_abs_diff_eq!(zero, rate, epsilon = 1e-12);
    }
}

/// Bootstrap should reproduce the input deposit rates.
#[test]
fn bootstrap_reproduces_deposit_rate() {
    let today = Date::from_ymd(2025, Month::June, 1);
    let dc = DayCounter::Actual365Fixed;
    let dep_rate = 0.05;
    let end_date = today + 91; // ~3M

    let mut helpers: Vec<Box<dyn RateHelper>> = vec![
        Box::new(DepositRateHelper::new(dep_rate, today, end_date, dc)),
    ];

    let curve = PiecewiseYieldCurve::new(today, &mut helpers, dc, 1e-12)
        .expect("Bootstrap should succeed");

    // The implied deposit rate should match the input
    let t = dc.year_fraction(today, end_date);
    let df = curve.discount_t(t);
    let implied_rate = (1.0 / df - 1.0) / t;
    assert_abs_diff_eq!(implied_rate, dep_rate, epsilon = 1e-6);
}
