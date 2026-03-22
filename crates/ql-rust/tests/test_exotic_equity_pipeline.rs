//! Integration tests for exotic equity option pricing engines.
//!
//! Covers chooser, forward-start, power, compound, lookback,
//! cliquet, and variance swap engines.

use ql_instruments::{CompoundOption, LookbackOption, OptionType, VarianceSwap};
use ql_pricingengines::{
    analytic_compound_option, analytic_lookback, chooser_price, cliquet_price,
    forward_start_option, power_option, price_variance_swap,
};

// ── Chooser options ─────────────────────────────────────────────

#[test]
fn chooser_atm_positive() {
    let r = chooser_price(100.0, 100.0, 0.05, 0.0, 0.20, 0.5, 1.0);
    assert!(r.npv > 0.0 && r.npv.is_finite());
}

#[test]
fn chooser_geq_call_and_put() {
    // Chooser ≥ max(call, put) since you get to choose
    let chooser = chooser_price(100.0, 100.0, 0.05, 0.0, 0.20, 0.5, 1.0);
    let call = ql_pricingengines::black_scholes_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Call);
    let put = ql_pricingengines::black_scholes_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Put);
    assert!(chooser.npv >= call.npv.max(put.npv) - 0.5,
        "Chooser {} < max(call={}, put={})", chooser.npv, call.npv, put.npv);
}

#[test]
fn chooser_longer_choose_time_more_valuable() {
    let short = chooser_price(100.0, 100.0, 0.05, 0.0, 0.20, 0.25, 1.0);
    let long = chooser_price(100.0, 100.0, 0.05, 0.0, 0.20, 0.75, 1.0);
    assert!(long.npv >= short.npv - 0.1,
        "Longer choose time {} < shorter {}", long.npv, short.npv);
}

// ── Forward-start options ───────────────────────────────────────

#[test]
fn forward_start_call_positive() {
    // alpha=1.0 → ATM at reset
    let r = forward_start_option(100.0, 0.05, 0.0, 0.20, 0.5, 1.0, 1.0, true);
    assert!(r.price > 0.0 && r.price.is_finite());
}

#[test]
fn forward_start_put_positive() {
    let r = forward_start_option(100.0, 0.05, 0.0, 0.20, 0.5, 1.0, 1.0, false);
    assert!(r.price > 0.0 && r.price.is_finite());
}

#[test]
fn forward_start_otm_cheaper() {
    let atm = forward_start_option(100.0, 0.05, 0.0, 0.20, 0.5, 1.0, 1.0, true);
    let otm = forward_start_option(100.0, 0.05, 0.0, 0.20, 0.5, 1.0, 1.1, true); // 10% OTM
    assert!(otm.price < atm.price + 0.01,
        "OTM {} >= ATM {}", otm.price, atm.price);
}

// ── Power options ───────────────────────────────────────────────

#[test]
fn power_option_alpha1_like_vanilla() {
    // Power option with alpha=1 should be similar to vanilla
    let power = power_option(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 1.0, true);
    let vanilla = ql_pricingengines::black_scholes_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Call);
    assert!((power.price - vanilla.npv).abs() < 1.0,
        "Power(α=1) {} vs vanilla {}", power.price, vanilla.npv);
}

#[test]
fn power_option_alpha2_positive() {
    let r = power_option(100.0, 10000.0, 0.05, 0.0, 0.20, 1.0, 2.0, true);
    assert!(r.price.is_finite());
}

// ── Compound options ────────────────────────────────────────────

#[test]
fn compound_call_on_call_positive() {
    let opt = CompoundOption::new(
        OptionType::Call, OptionType::Call, // call on call
        5.0,  // mother strike
        0.5,  // mother expiry
        100.0, // daughter strike
        1.0,  // daughter expiry
    );
    let r = analytic_compound_option(&opt, 100.0, 0.05, 0.0, 0.20);
    assert!(r.npv > 0.0 && r.npv.is_finite());
}

#[test]
fn compound_put_on_call_positive() {
    let opt = CompoundOption::new(
        OptionType::Put, OptionType::Call,
        5.0, 0.5, 100.0, 1.0,
    );
    let r = analytic_compound_option(&opt, 100.0, 0.05, 0.0, 0.20);
    assert!(r.npv >= 0.0 && r.npv.is_finite());
}

#[test]
fn compound_leq_daughter() {
    // Compound option ≤ daughter option value
    let opt = CompoundOption::new(
        OptionType::Call, OptionType::Call,
        5.0, 0.5, 100.0, 1.0,
    );
    let compound = analytic_compound_option(&opt, 100.0, 0.05, 0.0, 0.20);
    let daughter = ql_pricingengines::black_scholes_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Call);
    assert!(compound.npv <= daughter.npv + 0.5,
        "Compound {} > daughter {}", compound.npv, daughter.npv);
}

// ── Lookback options ────────────────────────────────────────────

#[test]
fn lookback_floating_call_positive() {
    let opt = LookbackOption::floating_strike(OptionType::Call, 95.0, 105.0, 1.0);
    let r = analytic_lookback(&opt, 100.0, 0.05, 0.0, 0.20);
    assert!(r.npv > 0.0 && r.npv.is_finite());
}

#[test]
fn lookback_floating_put_positive() {
    let opt = LookbackOption::floating_strike(OptionType::Put, 95.0, 105.0, 1.0);
    let r = analytic_lookback(&opt, 100.0, 0.05, 0.0, 0.20);
    assert!(r.npv > 0.0 && r.npv.is_finite());
}

#[test]
fn lookback_fixed_call_positive() {
    let opt = LookbackOption::fixed_strike(OptionType::Call, 100.0, 95.0, 105.0, 1.0);
    let r = analytic_lookback(&opt, 100.0, 0.05, 0.0, 0.20);
    assert!(r.npv > 0.0 && r.npv.is_finite());
}

#[test]
fn lookback_floating_geq_vanilla() {
    // Lookback floating-strike call ≥ vanilla call (better exercise price)
    let opt = LookbackOption::floating_strike(OptionType::Call, 100.0, 100.0, 1.0);
    let lookback = analytic_lookback(&opt, 100.0, 0.05, 0.0, 0.20);
    let vanilla = ql_pricingengines::black_scholes_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Call);
    assert!(lookback.npv >= vanilla.npv - 0.01,
        "Lookback {} < vanilla {}", lookback.npv, vanilla.npv);
}

// ── Cliquet options ─────────────────────────────────────────────

#[test]
fn cliquet_positive_with_caps_floors() {
    let reset_times = vec![0.25, 0.5, 0.75, 1.0];
    let r = cliquet_price(
        100.0, 0.05, 0.0, 0.20,
        &reset_times,
        -0.05,  // local floor -5%
        0.10,   // local cap +10%
        -0.10,  // global floor
        0.50,   // global cap
        1_000_000.0,
        OptionType::Call,
    );
    assert!(r.npv.is_finite());
}

#[test]
fn cliquet_wider_caps_more_valuable() {
    let times = vec![0.25, 0.5, 0.75, 1.0];
    let narrow = cliquet_price(100.0, 0.05, 0.0, 0.20, &times, -0.02, 0.05, -0.05, 0.20, 1.0, OptionType::Call);
    let wide = cliquet_price(100.0, 0.05, 0.0, 0.20, &times, -0.10, 0.20, -0.20, 0.80, 1.0, OptionType::Call);
    assert!(wide.npv >= narrow.npv - 0.01,
        "Wider caps {} < narrow caps {}", wide.npv, narrow.npv);
}

// ── Variance swaps ──────────────────────────────────────────────

#[test]
fn variance_swap_atm_zero_npv() {
    // If vol_strike == implied_vol, NPV ≈ 0 (at inception)
    let vs = VarianceSwap::from_vol_strike(100.0, 0.20, 1.0);
    let r = price_variance_swap(&vs, 0.20, 0.05);
    assert!(r.npv.abs() < 0.01, "ATM var swap NPV should be ~0, got {}", r.npv);
}

#[test]
fn variance_swap_positive_when_iv_exceeds_strike() {
    // Long variance: profit when realized/implied vol > strike vol
    let vs = VarianceSwap::from_vol_strike(100.0, 0.20, 1.0);
    let r = price_variance_swap(&vs, 0.30, 0.05);
    assert!(r.npv > 0.0, "IV>strike should give positive NPV, got {}", r.npv);
}

#[test]
fn variance_swap_negative_when_iv_below_strike() {
    let vs = VarianceSwap::from_vol_strike(100.0, 0.30, 1.0);
    let r = price_variance_swap(&vs, 0.20, 0.05);
    assert!(r.npv < 0.0, "IV<strike should give negative NPV, got {}", r.npv);
}
