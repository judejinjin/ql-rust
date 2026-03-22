//! Integration tests for barrier option variants.
//!
//! Covers double-barrier knockout/knockin, digital barriers,
//! and MC barriers.

use ql_instruments::OptionType;
use ql_methods::mc_barrier;
use ql_pricingengines::{
    double_barrier_knockin, double_barrier_knockout, digital_barrier, DigitalBarrierType,
};

// ── Double barrier knockout ─────────────────────────────────────

#[test]
fn double_barrier_ko_call_positive() {
    // Spot 100, barriers [80, 120], strike 100
    let r = double_barrier_knockout(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 80.0, 120.0, OptionType::Call, 50);
    assert!(r.npv >= 0.0 && r.npv.is_finite());
}

#[test]
fn double_barrier_ko_put_positive() {
    let r = double_barrier_knockout(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 80.0, 120.0, OptionType::Put, 50);
    assert!(r.npv >= 0.0 && r.npv.is_finite());
}

#[test]
fn double_barrier_ko_leq_vanilla() {
    // KO option <= vanilla (can only knock out, reducing value)
    let ko = double_barrier_knockout(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 80.0, 120.0, OptionType::Call, 50);
    let vanilla = ql_pricingengines::black_scholes_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Call);
    assert!(ko.npv <= vanilla.npv + 0.01, "KO {} > vanilla {}", ko.npv, vanilla.npv);
}

#[test]
fn double_barrier_ko_wider_barriers_more_valuable() {
    // Wider barriers → less likely to knock out → higher price
    let narrow = double_barrier_knockout(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 90.0, 110.0, OptionType::Call, 50);
    let wide = double_barrier_knockout(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 70.0, 130.0, OptionType::Call, 50);
    assert!(wide.npv >= narrow.npv - 0.01,
        "Wide {} < narrow {}", wide.npv, narrow.npv);
}

#[test]
#[should_panic(expected = "spot must lie between barriers")]
fn double_barrier_ko_at_barrier_zero() {
    // Spot at lower barrier → function panics (requires spot strictly between barriers)
    let _r = double_barrier_knockout(80.0, 100.0, 0.05, 0.0, 0.20, 1.0, 80.0, 120.0, OptionType::Call, 50);
}

// ── Double barrier knockin ──────────────────────────────────────

#[test]
fn double_barrier_ki_call_positive() {
    let r = double_barrier_knockin(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 80.0, 120.0, OptionType::Call, 50);
    assert!(r.npv >= 0.0 && r.npv.is_finite());
}

#[test]
fn double_barrier_ki_ko_parity() {
    // KI + KO = vanilla (in-out parity)
    let ki = double_barrier_knockin(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 80.0, 120.0, OptionType::Call, 50);
    let ko = double_barrier_knockout(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 80.0, 120.0, OptionType::Call, 50);
    let vanilla = ql_pricingengines::black_scholes_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Call);
    let parity = (ki.npv + ko.npv - vanilla.npv).abs();
    assert!(parity < 0.5, "KI({}) + KO({}) ≠ vanilla({}), diff={}", ki.npv, ko.npv, vanilla.npv, parity);
}

// ── Digital barrier ─────────────────────────────────────────────

#[test]
fn digital_barrier_one_touch_upper_positive() {
    let r = digital_barrier(100.0, 110.0, 1.0, 0.05, 0.0, 0.20, 1.0, DigitalBarrierType::OneTouch, true);
    assert!(r.price >= 0.0 && r.price.is_finite());
    assert!(r.price <= 1.0, "OneTouch rebate=1 should be ≤ 1, got {}", r.price);
}

#[test]
fn digital_barrier_no_touch_upper_positive() {
    let r = digital_barrier(100.0, 110.0, 1.0, 0.05, 0.0, 0.20, 1.0, DigitalBarrierType::NoTouch, true);
    assert!(r.price >= 0.0 && r.price.is_finite());
    assert!(r.price <= 1.0);
}

#[test]
fn digital_barrier_one_touch_no_touch_sum() {
    // OneTouch + NoTouch ≈ PV(rebate)  (probability sums to 1)
    let ot = digital_barrier(100.0, 110.0, 1.0, 0.05, 0.0, 0.20, 1.0, DigitalBarrierType::OneTouch, true);
    let nt = digital_barrier(100.0, 110.0, 1.0, 0.05, 0.0, 0.20, 1.0, DigitalBarrierType::NoTouch, true);
    let pv_rebate = 1.0 * (-0.05_f64 * 1.0).exp(); // PV of 1 at r=5%, t=1
    let sum = ot.price + nt.price;
    assert!((sum - pv_rebate).abs() < 0.15,
        "OT({}) + NT({}) = {} ≠ PV(1)={}", ot.price, nt.price, sum, pv_rebate);
}

#[test]
fn digital_barrier_far_barrier_one_touch_small() {
    // Very far barrier → low probability of touching
    let r = digital_barrier(100.0, 200.0, 1.0, 0.05, 0.0, 0.20, 1.0, DigitalBarrierType::OneTouch, true);
    assert!(r.price < 0.1, "Far barrier OneTouch should be small, got {}", r.price);
}

// ── MC barrier (ql-methods) ─────────────────────────────────────

#[test]
fn mc_barrier_up_and_out_call() {
    let r = mc_barrier(
        100.0, 100.0, 120.0, 0.0,      // spot, strike, barrier, rebate
        0.05, 0.0, 0.20, 1.0,           // r, q, vol, T
        OptionType::Call, true, false,   // call, up, knock-out
        100_000, 100, 42,               // paths, steps, seed
    );
    assert!(r.npv >= 0.0 && r.npv.is_finite());
}

#[test]
fn mc_barrier_down_and_out_put() {
    let r = mc_barrier(
        100.0, 100.0, 80.0, 0.0,
        0.05, 0.0, 0.20, 1.0,
        OptionType::Put, false, false,  // put, down, knock-out
        100_000, 100, 42,
    );
    assert!(r.npv >= 0.0 && r.npv.is_finite());
}

#[test]
fn mc_barrier_knock_out_leq_vanilla() {
    let ko = mc_barrier(
        100.0, 100.0, 120.0, 0.0,
        0.05, 0.0, 0.20, 1.0,
        OptionType::Call, true, false,
        200_000, 100, 42,
    );
    let vanilla = ql_methods::mc_european(
        100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
        OptionType::Call, 200_000, true, 42,
    );
    assert!(ko.npv <= vanilla.npv + 1.0,
        "KO MC {} > vanilla MC {}", ko.npv, vanilla.npv);
}
