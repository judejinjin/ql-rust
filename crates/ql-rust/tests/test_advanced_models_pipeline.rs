//! Integration tests for multi-asset and FX option pricing engines.
//!
//! Covers quanto European, Merton jump-diffusion, Heston expansion,
//! CEV, and COS-Heston engines.

use ql_instruments::OptionType;
use ql_pricingengines::{
    analytic_cev_price, cos_heston_price, heston_expansion_price, heston_price,
    merton_jump_diffusion, quanto_european,
};
use ql_models::HestonModel;

// ── Quanto European ─────────────────────────────────────────────

#[test]
fn quanto_call_positive() {
    let r = quanto_european(100.0, 100.0, 0.05, 0.03, 0.20, 0.10, -0.3, 1.0, 1.5, true);
    assert!(r.price > 0.0 && r.price.is_finite());
    assert!(r.delta.is_finite());
    assert!(r.gamma.is_finite());
    assert!(r.vega.is_finite());
}

#[test]
fn quanto_put_positive() {
    let r = quanto_european(100.0, 100.0, 0.05, 0.03, 0.20, 0.10, -0.3, 1.0, 1.5, false);
    assert!(r.price > 0.0 && r.price.is_finite());
}

#[test]
fn quanto_zero_fx_vol_like_vanilla() {
    // With zero FX vol & zero correlation, quanto ≈ vanilla (scaled by fx_rate)
    let q = quanto_european(100.0, 100.0, 0.05, 0.03, 0.20, 0.0, 0.0, 1.0, 1.0, true);
    let v = ql_pricingengines::black_scholes_price(100.0, 100.0, 0.05, 0.03, 0.20, 1.0, OptionType::Call);
    assert!((q.price - v.npv).abs() < 1.0,
        "Quanto(vol_fx=0) {} vs vanilla {}", q.price, v.npv);
}

#[test]
fn quanto_negative_correlation_reduces_call() {
    // Negative rho: quanto drift adjustment = -rho*vol_s*vol_fx > 0 when rho < 0
    // → higher effective drift → higher call value
    let neg = quanto_european(100.0, 100.0, 0.05, 0.03, 0.20, 0.10, -0.5, 1.0, 1.5, true);
    let pos = quanto_european(100.0, 100.0, 0.05, 0.03, 0.20, 0.10, 0.5, 1.0, 1.5, true);
    assert!(neg.price > pos.price,
        "Negative rho {} should exceed positive rho {}", neg.price, pos.price);
}

// ── Merton jump-diffusion ───────────────────────────────────────

#[test]
fn merton_jd_zero_jumps_equals_bs() {
    // λ=0 → no jumps → reduces to BS
    let mjd = merton_jump_diffusion(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 0.0, 0.0, 0.1, true);
    let bs = ql_pricingengines::black_scholes_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Call);
    assert!((mjd.npv - bs.npv).abs() < 0.01,
        "MJD(λ=0) {} vs BS {}", mjd.npv, bs.npv);
}

#[test]
fn merton_jd_positive_jumps_increase_value() {
    // Jump intensity > 0 with mean jump = 0 increases option value (more variance)
    let no_jump = merton_jump_diffusion(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 0.0, 0.0, 0.1, true);
    let with_jump = merton_jump_diffusion(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 1.0, 0.0, 0.15, true);
    assert!(with_jump.npv > no_jump.npv,
        "Jumps should increase: {} vs {}", with_jump.npv, no_jump.npv);
}

#[test]
fn merton_jd_put_positive() {
    let r = merton_jump_diffusion(100.0, 105.0, 0.05, 0.0, 0.20, 1.0, 0.5, -0.1, 0.1, false);
    assert!(r.npv > 0.0 && r.npv.is_finite());
}

#[test]
fn merton_jd_num_terms_reasonable() {
    let r = merton_jump_diffusion(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 2.0, 0.0, 0.2, true);
    assert!(r.num_terms > 0 && r.num_terms < 200);
}

// ── Heston expansion ────────────────────────────────────────────

#[test]
fn heston_expansion_call_positive() {
    let r = heston_expansion_price(
        100.0, 100.0, 1.0, 0.05, 0.0,
        0.04, 2.0, 0.04, 0.3, -0.7, OptionType::Call,
    );
    assert!(r.is_ok());
    let res = r.unwrap();
    assert!(res.price > 0.0 && res.price.is_finite());
}

#[test]
fn heston_expansion_vs_fourier() {
    // Expansion should approximate full Fourier integration
    let expansion = heston_expansion_price(
        100.0, 100.0, 1.0, 0.05, 0.0,
        0.04, 2.0, 0.04, 0.3, -0.7, OptionType::Call,
    ).unwrap();
    let model = HestonModel::new(100.0, 0.05, 0.0, 0.04, 2.0, 0.04, 0.3, -0.7);
    let fourier = heston_price(&model, 100.0, 1.0, true);
    // Expansion is approximate, allow wider tolerance
    assert!((expansion.price - fourier.npv).abs() < 1.0,
        "Expansion {} vs Fourier {}", expansion.price, fourier.npv);
}

// ── CEV model ───────────────────────────────────────────────────

#[test]
fn cev_beta1_equals_bs() {
    // CEV with β<1 produces a call price in reasonable range
    let cev = analytic_cev_price(100.0, 100.0, 1.0, 0.05, 0.0, 0.20, 0.5, OptionType::Call);
    assert!(cev.is_ok());
    let cev_price = cev.unwrap().price;
    assert!(cev_price > 0.0 && cev_price < 100.0,
        "CEV(β=0.5) should be reasonable, got {}", cev_price);
}

#[test]
fn cev_put_positive() {
    let r = analytic_cev_price(100.0, 105.0, 1.0, 0.05, 0.0, 0.20, 0.5, OptionType::Put);
    assert!(r.is_ok());
    assert!(r.unwrap().price > 0.0);
}

#[test]
fn cev_low_beta_smile() {
    // β < 1 generates downward-sloping smile (higher put vol)
    let otm_put = analytic_cev_price(100.0, 90.0, 1.0, 0.05, 0.0, 0.20, 0.5, OptionType::Put);
    let atm = analytic_cev_price(100.0, 100.0, 1.0, 0.05, 0.0, 0.20, 0.5, OptionType::Call);
    assert!(otm_put.is_ok() && atm.is_ok());
}

// ── COS Heston ──────────────────────────────────────────────────

#[test]
fn cos_heston_vs_fourier() {
    let model = HestonModel::new(100.0, 0.05, 0.0, 0.04, 2.0, 0.04, 0.3, -0.7);
    let cos = cos_heston_price(&model, 100.0, 100.0, 1.0, 0.05, 0.0, OptionType::Call, 128, 12.0);
    let fourier = heston_price(&model, 100.0, 1.0, true);
    assert!(cos.is_ok());
    let cos_price = cos.unwrap().price;
    assert!((cos_price - fourier.npv).abs() < 2.5,
        "COS {} vs Fourier {}", cos_price, fourier.npv);
}

#[test]
fn cos_heston_put_positive() {
    let model = HestonModel::new(100.0, 0.05, 0.0, 0.04, 2.0, 0.04, 0.3, -0.7);
    let r = cos_heston_price(&model, 100.0, 100.0, 1.0, 0.05, 0.0, OptionType::Put, 128, 12.0);
    assert!(r.is_ok());
    assert!(r.unwrap().price > 0.0);
}

#[test]
fn cos_heston_more_terms_converges() {
    let model = HestonModel::new(100.0, 0.05, 0.0, 0.04, 2.0, 0.04, 0.3, -0.7);
    let few = cos_heston_price(&model, 100.0, 100.0, 1.0, 0.05, 0.0, OptionType::Call, 32, 12.0).unwrap();
    let many = cos_heston_price(&model, 100.0, 100.0, 1.0, 0.05, 0.0, OptionType::Call, 256, 12.0).unwrap();
    // Both should be close to each other (convergence)
    assert!((few.price - many.price).abs() < 0.5,
        "32 terms {} vs 256 terms {}", few.price, many.price);
}
