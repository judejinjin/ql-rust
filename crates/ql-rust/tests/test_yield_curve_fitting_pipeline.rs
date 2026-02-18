//! Integration tests: Yield curve fitting pipeline.

use ql_termstructures::{
    CompositeZeroYieldStructure, FittedBondDiscountCurve,
    FlatForward, NelsonSiegelFitting, SpreadedTermStructure,
    SvenssonFitting, UltimateForwardTermStructure, YieldTermStructure,
};
use ql_time::{Date, DayCounter, Month};

/// Nelson-Siegel fitting should reproduce discount factors reasonably.
#[test]
fn nelson_siegel_fit_and_reprice() {
    let maturities = vec![0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0];
    let yields = vec![0.045, 0.046, 0.048, 0.050, 0.052, 0.054, 0.055, 0.056];

    let ns = NelsonSiegelFitting::fit(&maturities, &yields).unwrap();
    let today = Date::from_ymd(2025, Month::January, 15);
    let dc = DayCounter::Actual365Fixed;
    let fitted = FittedBondDiscountCurve::from_nelson_siegel(today, ns.clone(), dc, 30.0);

    for (&t, &y) in maturities.iter().zip(yields.iter()) {
        let df = fitted.discount_t(t);
        let implied_y = -df.ln() / t;
        let err = (implied_y - y).abs();
        assert!(err < 0.005, "NS fit error at t={}: implied {:.4} vs market {:.4}", t, implied_y, y);
    }
}

/// Svensson (6-param) should fit at least as well as Nelson-Siegel (4-param).
#[test]
fn svensson_fits_better_than_nelson_siegel() {
    let maturities = vec![0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0];
    let yields = vec![
        0.045, 0.046, 0.048, 0.050, 0.051, 0.053, 0.054, 0.055, 0.054, 0.053, 0.052,
    ];

    let ns = NelsonSiegelFitting::fit(&maturities, &yields).unwrap();
    let sv = SvenssonFitting::fit(&maturities, &yields).unwrap();

    let ns_err: f64 = maturities.iter().zip(yields.iter())
        .map(|(&t, &y)| (ns.zero_rate(t) - y).powi(2)).sum();
    let sv_err: f64 = maturities.iter().zip(yields.iter())
        .map(|(&t, &y)| (sv.zero_rate(t) - y).powi(2)).sum();

    assert!(sv_err <= ns_err + 1e-8, "Svensson SSE {:.8} should be <= NS SSE {:.8}", sv_err, ns_err);
}

/// Composite curve: base + spread = expected rate.
#[test]
fn composite_zero_yield_structure() {
    let today = Date::from_ymd(2025, Month::January, 15);
    let dc = DayCounter::Actual365Fixed;
    let base = FlatForward::new(today, 0.03, dc);
    let spread = FlatForward::new(today, 0.01, dc);

    let composite = CompositeZeroYieldStructure::new(&base, &spread, true, 100, 30.0);
    let df = composite.discount_t(5.0);
    let implied = -df.ln() / 5.0;
    assert!((implied - 0.04).abs() < 0.001, "Composite rate {:.4} != 0.04", implied);
}

/// SpreadedTermStructure: base + constant spread.
#[test]
fn spreaded_term_structure() {
    let today = Date::from_ymd(2025, Month::January, 15);
    let dc = DayCounter::Actual365Fixed;
    let base = FlatForward::new(today, 0.05, dc);
    let spreaded = SpreadedTermStructure::new(&base, 0.005, 30.0);

    let df = spreaded.discount_t(2.0);
    let implied = -df.ln() / 2.0;
    assert!((implied - 0.055).abs() < 0.001, "Spreaded rate {:.4} != 0.055", implied);
}

/// Smith-Wilson / UFR: extrapolation should converge to UFR.
#[test]
fn smith_wilson_ufr_convergence() {
    let today = Date::from_ymd(2025, Month::January, 15);
    let dc = DayCounter::Actual365Fixed;

    let maturities = vec![1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0];
    // Discount factors from flat 4%
    let dfs: Vec<f64> = maturities.iter().map(|&t| (-0.04_f64 * t).exp()).collect();
    let ufr = 0.042;
    let alpha = 0.1;

    let sw = UltimateForwardTermStructure::new(today, &maturities, &dfs, ufr, alpha, dc, 120.0)
        .unwrap();

    // At very long maturity, forward rate should converge towards UFR
    let df_99 = sw.discount_t(99.0);
    let df_100 = sw.discount_t(100.0);
    let fwd = -(df_100 / df_99).ln();
    assert!(
        (fwd - ufr).abs() < 0.005,
        "Forward at t=99 ({:.4}) should be near UFR ({:.4})",
        fwd, ufr,
    );
}
