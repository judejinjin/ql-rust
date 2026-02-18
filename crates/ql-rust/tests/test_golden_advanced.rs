//! Golden cross-validation tests for Phase 13–24 features.

use ql_instruments::VanillaOption;
use ql_pricingengines::{
    barone_adesi_whaley, bjerksund_stensland, kirk_spread_call, margrabe_exchange,
    merton_jump_diffusion, price_european, qd_plus_american,
};
use ql_termstructures::{NelsonSiegelFitting, SvenssonFitting};
use ql_time::{Date, Month};

/// BAW American put S=100, K=100, r=8%, q=0, σ=20%, T=0.25 ≈ 3.25.
#[test]
fn golden_baw_american_put() {
    let result = barone_adesi_whaley(100.0, 100.0, 0.08, 0.0, 0.20, 0.25, false);
    assert!((result.npv - 3.25).abs() < 0.30, "BAW ATM put {:.4} not near 3.25", result.npv);
}

/// Deep ITM American put should be near intrinsic.
#[test]
fn golden_deep_itm_american_put() {
    let result = qd_plus_american(50.0, 100.0, 0.05, 0.0, 0.20, 1.0, false);
    assert!(result.npv >= 49.0, "Deep ITM put {:.4} too far from intrinsic 50", result.npv);
    assert!(result.npv <= 52.0, "Deep ITM put {:.4} exceeds bound", result.npv);
}

/// Merton JD with λ=0 equals Black-Scholes.
#[test]
fn golden_merton_zero_jumps_equals_bs() {
    let today = Date::from_ymd(2025, Month::January, 15);
    let call = VanillaOption::european_call(105.0, today + 365);
    let bs = price_european(&call, 100.0, 0.05, 0.02, 0.20, 1.0);
    let mjd = merton_jump_diffusion(100.0, 105.0, 0.05, 0.02, 0.20, 1.0, 0.0, 0.0, 0.10, true);
    let err = (mjd.npv - bs.npv).abs();
    assert!(err < 0.01, "Merton(λ=0) {:.4} != BS {:.4}", mjd.npv, bs.npv);
}

/// Kirk spread call with near-zero vol should converge to discounted intrinsic.
#[test]
fn golden_kirk_low_vol() {
    let result = kirk_spread_call(100.0, 90.0, 5.0, 0.05, 0.0, 0.0, 0.001, 0.001, 0.0, 1.0);
    let intrinsic = ((100.0 - 90.0 - 5.0) * (-0.05_f64).exp()).max(0.0);
    // Both Kirk and intrinsic should be close to 5 * df
    assert!(result > 0.0, "Kirk should be positive");
    assert!((result - intrinsic).abs() < 1.0, "Kirk low-vol {:.4} too far from intrinsic {:.4}", result, intrinsic);
}

/// Margrabe exchange with equal spots should give positive value.
#[test]
fn golden_margrabe_equal_spots() {
    let result = margrabe_exchange(100.0, 100.0, 0.0, 0.0, 0.20, 0.25, 0.5, 1.0);
    assert!(result > 5.0 && result < 25.0, "Margrabe ATM {:.4} not in [5,25]", result);
}

/// Nelson-Siegel flat curve: all fitted rates near the flat level.
#[test]
fn golden_nelson_siegel_flat_curve() {
    let maturities = vec![0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0];
    let flat_rate = 0.05;
    let yields: Vec<f64> = maturities.iter().map(|_| flat_rate).collect();
    let ns = NelsonSiegelFitting::fit(&maturities, &yields).unwrap();
    for &t in &maturities {
        let fitted = ns.zero_rate(t);
        assert!((fitted - flat_rate).abs() < 0.002, "NS flat at t={}: {:.4} != {:.4}", t, fitted, flat_rate);
    }
}

/// Svensson upward-sloping: all fitted rates within 50bp of market.
#[test]
fn golden_svensson_upward_slope() {
    let maturities = vec![0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0];
    let yields = vec![0.030, 0.032, 0.035, 0.040, 0.043, 0.048, 0.050, 0.052, 0.053, 0.054, 0.055];
    let sv = SvenssonFitting::fit(&maturities, &yields).unwrap();
    for (&t, &y) in maturities.iter().zip(yields.iter()) {
        let fitted = sv.zero_rate(t);
        assert!((fitted - y).abs() < 0.005, "Svensson at t={}: {:.4} vs {:.4}", t, fitted, y);
    }
}

/// BJS vs BAW should produce similar American call prices.
#[test]
fn golden_bjs_vs_baw_call() {
    let baw = barone_adesi_whaley(100.0, 100.0, 0.05, 0.03, 0.25, 1.0, true);
    let bjs = bjerksund_stensland(100.0, 100.0, 0.05, 0.03, 0.25, 1.0, true);
    let rel_err = ((baw.npv - bjs.npv) / baw.npv).abs();
    assert!(rel_err < 0.10, "BAW {:.4} vs BJS {:.4} differ {:.2}%", baw.npv, bjs.npv, rel_err * 100.0);
}
