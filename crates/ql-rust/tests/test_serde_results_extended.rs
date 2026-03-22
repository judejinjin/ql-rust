//! Integration tests for serde round-trips on recently-added result types.
//!
//! Ensures that all pricing result structs can be serialized to JSON
//! and deserialized back without data loss.

use ql_instruments::{CompoundOption, LookbackOption, OptionType, VarianceSwap};
use ql_models::HestonModel;
use ql_pricingengines::*;

fn round_trip<T: serde::Serialize + serde::de::DeserializeOwned>(val: &T) -> T {
    let json = serde_json::to_string(val).expect("serialize");
    serde_json::from_str(&json).expect("deserialize")
}

#[test]
fn serde_american_approx_result() {
    // Use an American put which has a well-defined critical price
    let r = barone_adesi_whaley(100.0, 100.0, 0.05, 0.02, 0.20, 1.0, false);
    // Verify finite values before round-trip
    assert!(r.npv.is_finite() && r.early_exercise_premium.is_finite() && r.critical_price.is_finite());
    let r2 = round_trip(&r);
    assert!((r.npv - r2.npv).abs() < 1e-12);
    assert!((r.early_exercise_premium - r2.early_exercise_premium).abs() < 1e-12);
}

#[test]
fn serde_asian_result() {
    let r = asian_geometric_continuous_avg_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Call);
    let r2 = round_trip(&r);
    assert!((r.npv - r2.npv).abs() < 1e-12);
    assert!((r.effective_vol - r2.effective_vol).abs() < 1e-12);
}

#[test]
fn serde_double_barrier_result() {
    let r = double_barrier_knockout(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 80.0, 120.0, OptionType::Call, 50);
    let r2 = round_trip(&r);
    assert!((r.npv - r2.npv).abs() < 1e-12);
}

#[test]
fn serde_chooser_result() {
    let r = chooser_price(100.0, 100.0, 0.05, 0.0, 0.20, 0.5, 1.0);
    let r2 = round_trip(&r);
    assert!((r.npv - r2.npv).abs() < 1e-12);
}

#[test]
fn serde_forward_start_result() {
    let r = forward_start_option(100.0, 0.05, 0.0, 0.20, 0.5, 1.0, 1.0, true);
    let r2 = round_trip(&r);
    assert!((r.price - r2.price).abs() < 1e-12);
}

#[test]
fn serde_power_result() {
    let r = power_option(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 1.5, true);
    let r2 = round_trip(&r);
    assert!((r.price - r2.price).abs() < 1e-12);
}

#[test]
fn serde_digital_barrier_result() {
    let r = digital_barrier(100.0, 110.0, 1.0, 0.05, 0.0, 0.20, 1.0, DigitalBarrierType::OneTouch, true);
    let r2 = round_trip(&r);
    assert!((r.price - r2.price).abs() < 1e-12);
}

#[test]
fn serde_quanto_result() {
    let r = quanto_european(100.0, 100.0, 0.05, 0.03, 0.20, 0.10, -0.3, 1.0, 1.5, true);
    let r2 = round_trip(&r);
    assert!((r.price - r2.price).abs() < 1e-12);
    assert!((r.delta - r2.delta).abs() < 1e-12);
}

#[test]
fn serde_merton_jd_result() {
    let r = merton_jump_diffusion(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 0.5, -0.1, 0.1, true);
    let r2 = round_trip(&r);
    assert!((r.npv - r2.npv).abs() < 1e-12);
    assert_eq!(r.num_terms, r2.num_terms);
}

#[test]
fn serde_lookback_result() {
    let opt = LookbackOption::floating_strike(OptionType::Call, 95.0, 105.0, 1.0);
    let r = analytic_lookback(&opt, 100.0, 0.05, 0.0, 0.20);
    let r2 = round_trip(&r);
    assert!((r.npv - r2.npv).abs() < 1e-12);
}

#[test]
fn serde_compound_option_result() {
    let opt = CompoundOption::new(OptionType::Call, OptionType::Call, 5.0, 0.5, 100.0, 1.0);
    let r = analytic_compound_option(&opt, 100.0, 0.05, 0.0, 0.20);
    let r2 = round_trip(&r);
    assert!((r.npv - r2.npv).abs() < 1e-12);
}

#[test]
fn serde_variance_swap_result() {
    let vs = VarianceSwap::from_vol_strike(100.0, 0.20, 1.0);
    let r = price_variance_swap(&vs, 0.25, 0.05);
    let r2 = round_trip(&r);
    assert!((r.npv - r2.npv).abs() < 1e-12);
}

#[test]
fn serde_cliquet_result() {
    let r = cliquet_price(100.0, 0.05, 0.0, 0.20, &[0.25, 0.5, 0.75, 1.0], -0.05, 0.10, -0.10, 0.50, 1e6, OptionType::Call);
    let r2 = round_trip(&r);
    assert!((r.npv - r2.npv).abs() < 1e-12);
}

#[test]
fn serde_heston_result() {
    let model = HestonModel::new(100.0, 0.05, 0.0, 0.04, 2.0, 0.04, 0.3, -0.7);
    let r = heston_price(&model, 100.0, 1.0, true);
    let r2 = round_trip(&r);
    assert!((r.npv - r2.npv).abs() < 1e-12);
}

#[test]
fn serde_mc_result() {
    let r = ql_methods::mc_european(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Call, 1000, true, 42);
    let r2 = round_trip(&r);
    assert!((r.npv - r2.npv).abs() < 1e-12);
    assert_eq!(r.num_paths, r2.num_paths);
}

#[test]
fn serde_fd_result() {
    let r = ql_methods::fd_black_scholes(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true, false, 50, 50);
    let r2 = round_trip(&r);
    assert!((r.npv - r2.npv).abs() < 1e-12);
    assert!((r.delta - r2.delta).abs() < 1e-12);
}
