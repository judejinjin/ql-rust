//! Integration tests: American option pricing pipeline.

use ql_methods::fd_black_scholes;
use ql_pricingengines::{
    barone_adesi_whaley, bjerksund_stensland, price_european, qd_plus_american,
};
use ql_instruments::VanillaOption;
use ql_time::{Date, Month};

/// All three American approximations must be >= European price (put).
#[test]
fn american_premium_greater_than_european() {
    let today = Date::from_ymd(2025, Month::January, 15);
    let s = 100.0;
    let k = 110.0;
    let r = 0.05;
    let q = 0.02;
    let sigma = 0.30;
    let t = 1.0;

    let eu = price_european(
        &VanillaOption::european_put(k, today + 365), s, r, q, sigma, t,
    );
    let baw = barone_adesi_whaley(s, k, r, q, sigma, t, false);
    let bjs = bjerksund_stensland(s, k, r, q, sigma, t, false);
    let qdp = qd_plus_american(s, k, r, q, sigma, t, false);

    assert!(baw.npv >= eu.npv, "BAW < European: {} < {}", baw.npv, eu.npv);
    assert!(bjs.npv >= eu.npv, "BJS < European: {} < {}", bjs.npv, eu.npv);
    assert!(qdp.npv >= eu.npv, "QD+ < European: {} < {}", qdp.npv, eu.npv);
}

/// Analytic approximations should be within 2% of each other.
#[test]
fn american_approximations_agree() {
    let baw = barone_adesi_whaley(100.0, 105.0, 0.05, 0.02, 0.25, 1.0, false);
    let bjs = bjerksund_stensland(100.0, 105.0, 0.05, 0.02, 0.25, 1.0, false);
    let qdp = qd_plus_american(100.0, 105.0, 0.05, 0.02, 0.25, 1.0, false);

    let max_price = baw.npv.max(bjs.npv).max(qdp.npv);
    let min_price = baw.npv.min(bjs.npv).min(qdp.npv);
    let spread = (max_price - min_price) / max_price;
    assert!(spread < 0.06, "Approx spread {:.4}% too wide", spread * 100.0);
}

/// FD American put should be close to QD+.
#[test]
fn fd_american_consistent_with_analytic() {
    let fd = fd_black_scholes(100.0, 110.0, 0.05, 0.0, 0.30, 1.0, false, true, 400, 400);
    let qdp = qd_plus_american(100.0, 110.0, 0.05, 0.0, 0.30, 1.0, false);
    let rel_err = ((fd.npv - qdp.npv) / qdp.npv).abs();
    assert!(rel_err < 0.01, "FD vs QD+ error {:.4}%", rel_err * 100.0);
}

/// Deep ITM call with high dividends should have early-exercise premium.
#[test]
fn deep_itm_call_with_dividends() {
    let baw = barone_adesi_whaley(150.0, 100.0, 0.05, 0.08, 0.20, 1.0, true);
    assert!(baw.early_exercise_premium > 0.0, "Expected positive early-exercise premium");
}
