//! Integration tests: Multi-asset / basket option pricing pipeline.

use ql_pricingengines::{
    kirk_spread_call, kirk_spread_put, margrabe_exchange,
    stulz_max_call, stulz_min_call,
};

/// Stulz max-call should be >= min-call for equal spots.
#[test]
fn stulz_max_geq_min() {
    let max_c = stulz_max_call(100.0, 100.0, 100.0, 0.05, 0.02, 0.02, 0.20, 0.25, 0.5, 1.0);
    let min_c = stulz_min_call(100.0, 100.0, 100.0, 0.05, 0.02, 0.02, 0.20, 0.25, 0.5, 1.0);
    assert!(max_c > 0.0);
    assert!(min_c > 0.0);
    assert!(max_c >= min_c, "Max-call {} should be >= min-call {}", max_c, min_c);
}

/// Kirk spread call/put must satisfy put-call parity for spreads.
#[test]
fn kirk_spread_put_call_parity() {
    let (s1, s2, k, r, q1, q2, v1, v2, rho, t) =
        (100.0, 90.0, 5.0, 0.05, 0.01, 0.01, 0.20, 0.25, 0.6, 1.0);

    let call = kirk_spread_call(s1, s2, k, r, q1, q2, v1, v2, rho, t);
    let put = kirk_spread_put(s1, s2, k, r, q1, q2, v1, v2, rho, t);

    assert!(call > 0.0);
    assert!(put > 0.0);

    // C - P ≈ df * (S1*e^{-q1T} - S2*e^{-q2T} - K)
    let df = (-r * t).exp();
    let fwd_spread = s1 * (-q1 * t).exp() - s2 * (-q2 * t).exp() - k;
    let expected_diff = df * fwd_spread;
    let actual_diff = call - put;
    let err = (actual_diff - expected_diff).abs();
    assert!(err < 0.5, "Kirk parity error {:.4} too large", err);
}

/// Margrabe exchange option value must be positive and bounded.
#[test]
fn margrabe_exchange_bounds() {
    let ex = margrabe_exchange(100.0, 95.0, 0.01, 0.02, 0.20, 0.25, 0.5, 1.0);
    assert!(ex > 0.0, "Exchange option should be positive");
    let upper = 100.0 * (-0.01_f64).exp();
    assert!(ex < upper, "Exchange option {:.4} exceeds upper bound {:.4}", ex, upper);
}

/// Kirk spread call with zero strike should approximate Margrabe exchange.
#[test]
fn kirk_zero_strike_approx_margrabe() {
    let kirk = kirk_spread_call(100.0, 95.0, 0.0, 0.05, 0.01, 0.02, 0.20, 0.25, 0.5, 1.0);
    let marg = margrabe_exchange(100.0, 95.0, 0.01, 0.02, 0.20, 0.25, 0.5, 1.0);
    let rel_err = ((kirk - marg) / marg).abs();
    assert!(rel_err < 0.05, "Kirk(K=0) {:.4} vs Margrabe {:.4}, err {:.2}%", kirk, marg, rel_err * 100.0);
}
