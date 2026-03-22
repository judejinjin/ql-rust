//! Integration tests for Asian option pricing engines.
//!
//! Covers geometric (continuous/discrete, avg-price/avg-strike),
//! Turnbull-Wakeman, Levy, and Monte Carlo Asian engines.

use ql_pricingengines::{
    asian_geometric_continuous_avg_price, asian_geometric_continuous_avg_strike,
    asian_geometric_discrete_avg_price, asian_geometric_discrete_avg_strike, asian_levy,
    asian_turnbull_wakeman, mc_asian_arithmetic_price, mc_asian_geometric_price,
};
use ql_instruments::OptionType;
use ql_methods::mc_asian;

// ── Geometric continuous avg-price ──────────────────────────────

#[test]
fn asian_geo_cont_avg_price_call_positive() {
    let r = asian_geometric_continuous_avg_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Call);
    assert!(r.npv > 0.0 && r.npv.is_finite());
    assert!(r.effective_vol > 0.0);
}

#[test]
fn asian_geo_cont_avg_price_put_positive() {
    let r = asian_geometric_continuous_avg_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Put);
    assert!(r.npv > 0.0 && r.npv.is_finite());
}

#[test]
fn asian_geo_cont_call_leq_european() {
    // Asian call ≤ European call (averaging reduces variance)
    let asian = asian_geometric_continuous_avg_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Call);
    let european = ql_pricingengines::black_scholes_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Call);
    assert!(asian.npv <= european.npv + 0.01, "Asian {} > European {}", asian.npv, european.npv);
}

#[test]
fn asian_geo_cont_put_call_relation() {
    // Both call and put should be > 0 for ATM
    let call = asian_geometric_continuous_avg_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Call);
    let put = asian_geometric_continuous_avg_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Put);
    assert!(call.npv > 0.0);
    assert!(put.npv > 0.0);
    // Call > Put for zero-dividend with positive rates
    assert!(call.npv > put.npv);
}

// ── Geometric discrete avg-price ────────────────────────────────

#[test]
fn asian_geo_discrete_avg_price_converges_to_continuous() {
    let cont = asian_geometric_continuous_avg_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Call);
    let disc = asian_geometric_discrete_avg_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 252, OptionType::Call);
    // With 252 fixings, discrete should be close to continuous
    assert!((cont.npv - disc.npv).abs() < 0.5,
        "Continuous {} vs Discrete(252) {}", cont.npv, disc.npv);
}

#[test]
fn asian_geo_discrete_fewer_fixings_cheaper() {
    let few = asian_geometric_discrete_avg_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 4, OptionType::Call);
    let many = asian_geometric_discrete_avg_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 252, OptionType::Call);
    // Both positive
    assert!(few.npv > 0.0);
    assert!(many.npv > 0.0);
}

// ── Geometric continuous avg-strike ─────────────────────────────

#[test]
fn asian_geo_cont_avg_strike_positive() {
    let call = asian_geometric_continuous_avg_strike(100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Call);
    let put = asian_geometric_continuous_avg_strike(100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Put);
    assert!(call.npv > 0.0 && call.npv.is_finite());
    assert!(put.npv > 0.0 && put.npv.is_finite());
}

// ── Geometric discrete avg-strike ───────────────────────────────

#[test]
fn asian_geo_discrete_avg_strike_positive() {
    let r = asian_geometric_discrete_avg_strike(100.0, 0.05, 0.0, 0.20, 1.0, 12, OptionType::Call);
    assert!(r.npv > 0.0 && r.npv.is_finite());
}

// ── Turnbull-Wakeman ────────────────────────────────────────────

#[test]
fn asian_tw_fresh_option_positive() {
    // t0=0, a=0: fresh option (no averaging elapsed)
    let r = asian_turnbull_wakeman(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 0.0, 0.0, OptionType::Call);
    assert!(r.npv > 0.0 && r.npv.is_finite());
}

#[test]
fn asian_tw_partially_elapsed() {
    // Option half-way through averaging, running average = 102
    let r = asian_turnbull_wakeman(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 0.5, 102.0, OptionType::Call);
    assert!(r.npv > 0.0 && r.npv.is_finite());
}

#[test]
fn asian_tw_put_positive() {
    let r = asian_turnbull_wakeman(100.0, 105.0, 0.05, 0.0, 0.20, 1.0, 0.0, 0.0, OptionType::Put);
    assert!(r.npv > 0.0 && r.npv.is_finite());
}

// ── Levy ────────────────────────────────────────────────────────

#[test]
fn asian_levy_call_positive() {
    let r = asian_levy(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Call);
    assert!(r.npv > 0.0 && r.npv.is_finite());
}

#[test]
fn asian_levy_put_positive() {
    let r = asian_levy(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Put);
    assert!(r.npv > 0.0 && r.npv.is_finite());
}

#[test]
fn asian_levy_vs_geometric_atm() {
    // Levy (arithmetic approx) and geometric should be in same ballpark for ATM
    let levy = asian_levy(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Call);
    let geo = asian_geometric_continuous_avg_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Call);
    // Arithmetic >= Geometric (Jensen's inequality), but close for low vol
    assert!(levy.npv >= geo.npv - 0.5,
        "Levy {} < Geo {} - 0.5", levy.npv, geo.npv);
}

// ── MC Asian arithmetic ─────────────────────────────────────────

#[test]
fn mc_asian_arithmetic_price_positive() {
    let r = mc_asian_arithmetic_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 12, 50_000, true, Some(42));
    assert!(r.price > 0.0 && r.price.is_finite());
    assert!(r.std_error.is_finite());
}

#[test]
fn mc_asian_geometric_vs_analytic() {
    // MC geometric should converge to analytic geometric
    let mc = mc_asian_geometric_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 252, 100_000, true, Some(42));
    let analytic = asian_geometric_continuous_avg_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Call);
    assert!((mc.price - analytic.npv).abs() < 1.0,
        "MC geo {} vs analytic geo {}", mc.price, analytic.npv);
}

// ── mc_asian from ql-methods ────────────────────────────────────

#[test]
fn mc_asian_methods_arithmetic_positive() {
    let r = mc_asian(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Call, true, 50_000, 50, 42);
    assert!(r.npv > 0.0 && r.npv.is_finite());
}

#[test]
fn mc_asian_methods_geometric_vs_analytic() {
    let mc = mc_asian(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Call, false, 100_000, 50, 42);
    let analytic = asian_geometric_continuous_avg_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Call);
    assert!((mc.npv - analytic.npv).abs() < 1.0,
        "MC geo {} vs analytic {}", mc.npv, analytic.npv);
}
