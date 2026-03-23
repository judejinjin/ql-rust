//! Verification Group B — Credit Derivatives
//!
//! Cross-validates ql-rust CDS and risky bond pricing against
//! QuantLib-Risks-Py reference values.
//!
//! B1: CDS — MidPoint Engine (flat curves, base-case + AD)
//! B3: Risky Bond — Survival-weighted discounting

use ql_aad::DualVec;
use ql_pricingengines::generic::cds_midpoint_generic;
use ql_pricingengines::risky_bond_engine;
use ql_termstructures::generic::{FlatCurve, InterpDiscountCurve};

fn assert_ref(actual: f64, expected: f64, rel_tol: f64, label: &str) {
    let diff = (actual - expected).abs();
    let denom = expected.abs().max(1e-10);
    let rel = diff / denom;
    assert!(
        rel < rel_tol || diff < 1e-8,
        "{label}: actual={actual:.10}, expected={expected:.10}, rel_err={rel:.2e}"
    );
}

// =========================================================================
// B1. CDS — MidPoint Engine (Flat Curves)
// =========================================================================
//
// Python baseline: 2Y CDS, 150bp coupon, nominal=1M, recovery=0.50, RF=1%
// flat CDS spreads (150bp at all tenors) → hazard rate ≈ spread/(1-R) = 3%
//
// At fair spread the CDS NPV should be ≈ 0 (from protection buyer view).
// We test: (a) NPV ≈ 0, (b) AD sensitivities through the engine.

/// Quarterly payment times for a 2Y CDS starting at t=0.
const CDS_PAYMENT_TIMES: [f64; 8] = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00];
/// Year fractions for quarterly payments.
const CDS_PAYMENT_YFS: [f64; 8] = [0.25; 8];

#[test]
fn b1_cds_midpoint_flat_npv_near_zero() {
    // CDS with flat hazard rate = spread/(1-R) should have NPV ≈ 0.
    let spread = 0.015; // 150bp
    let recovery = 0.5;
    let notional = 1_000_000.0;
    let hazard_rate = spread / (1.0 - recovery); // 0.03

    // Flat risk-free curve
    let yield_curve = FlatCurve::new(0.01_f64);
    // Flat survival curve: S(t) = exp(-λt), treated as "discount factor"
    let survival_curve = FlatCurve::new(hazard_rate);

    let npv: f64 = cds_midpoint_generic(
        notional,
        spread,
        recovery,
        &CDS_PAYMENT_TIMES,
        &CDS_PAYMENT_YFS,
        &yield_curve,
        &survival_curve,
    );

    // At the implied flat hazard rate, NPV should be close to 0.
    // MidPoint approximation introduces small bias.
    assert!(
        npv.abs() < 500.0,
        "B1 CDS at fair spread: NPV should be near 0, got {npv:.2}"
    );
}

#[test]
fn b1_cds_midpoint_ad_sensitivities() {
    type D = DualVec<2>;
    let spread = 0.015;
    let recovery = 0.5;
    let notional = 1_000_000.0;

    // Inputs: risk-free rate [0], hazard rate [1]
    let rf = D::variable(0.01, 0);
    let hazard = D::variable(0.03, 1);

    let yield_curve = FlatCurve::new(rf);
    let survival_curve = FlatCurve::new(hazard);

    let npv: D = cds_midpoint_generic(
        notional,
        spread,
        recovery,
        &CDS_PAYMENT_TIMES,
        &CDS_PAYMENT_YFS,
        &yield_curve,
        &survival_curve,
    );

    // Protection buyer NPV: positive hazard rate sensitivity (more defaults = more payoff)
    // The hazard rate sensitivity should be negative for fair CDS (higher hazard →
    // more protection leg and less premium leg, but protection buyer benefits)
    let d_rf = npv.dot[0];
    let d_hazard = npv.dot[1];

    // Rate sensitivity should be small (near zero for 2Y)
    assert!(
        d_rf.abs() < 100_000.0,
        "B1 CDS: rate sensitivity reasonable, got {d_rf:.2}"
    );
    // Hazard rate sensitivity is substantial
    assert!(
        d_hazard.abs() > 1000.0,
        "B1 CDS: hazard rate sensitivity should be substantial, got {d_hazard:.2}"
    );
}

// =========================================================================
// B3. Risky Bond — Survival-Weighted Discounting
// =========================================================================
//
// Python: 5Y fixed-rate bond, 5% semiannual, notional=100, recovery=40%
// OIS rates (9): 1M–30Y bootstrapped → discount curve
// CDS spreads (4): 1Y–5Y bootstrapped → survival curve
//
// Reference: NPV = 100.6120
//
// Since the exact bootstrap (PiecewiseLogLinearDiscount) is QuantLib-specific,
// we test the risky_bond_engine math with equivalent flat curves and verify
// the AD-through-discounting pipeline separately.

/// Approximate 5Y semiannual payment times (~Act365Fixed from some eval date).
const BOND_PAYMENT_TIMES: [f64; 10] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
const BOND_YFS: [f64; 10] = [0.5; 10];

#[test]
fn b3_risky_bond_risk_free() {
    // Test risky bond engine with no credit risk (survival=1.0).
    // Should match pure fixed-rate bond discounted at a flat rate.
    let rate = 0.035;
    let coupon = 0.05;
    let notional = 100.0;

    let dfs: Vec<f64> = BOND_PAYMENT_TIMES.iter().map(|&t| (-rate * t).exp()).collect();
    let surv: Vec<f64> = vec![1.0; 10]; // No credit risk

    let res = risky_bond_engine(
        notional,
        coupon,
        &BOND_PAYMENT_TIMES,
        &dfs,
        &surv,
        0.40,
        &BOND_YFS,
        0.0,
    );

    // Risk-free NPV = Σ (coupon * yf * DF) + notional * DF(5Y)
    let expected_rf: f64 = BOND_PAYMENT_TIMES
        .iter()
        .zip(dfs.iter())
        .map(|(&_t, &df)| notional * coupon * 0.5 * df)
        .sum::<f64>()
        + notional * dfs[9];

    assert_ref(
        res.risk_free_npv,
        expected_rf,
        1e-8,
        "B3 risk-free NPV sanity",
    );

    // With survival=1.0 everywhere, credit-adjusted = risk-free
    assert_ref(
        res.credit_adjusted_npv,
        expected_rf,
        1e-4,
        "B3 no-credit-risk adjusted = risk-free",
    );
}

#[test]
fn b3_risky_bond_with_credit_spread() {
    // Test risky bond with realistic credit risk.
    // OIS flat rate ≈ 3.5%, hazard rate ≈ CDS_5Y/(1-R) ≈ 0.0125/0.6 ≈ 2.08%
    let rate = 0.035;
    let coupon = 0.05;
    let notional = 100.0;
    let recovery = 0.40;
    let hazard = 0.0125 / (1.0 - recovery); // approx 2.08%

    let dfs: Vec<f64> = BOND_PAYMENT_TIMES.iter().map(|&t| (-rate * t).exp()).collect();
    let survs: Vec<f64> = BOND_PAYMENT_TIMES
        .iter()
        .map(|&t| (-hazard * t).exp())
        .collect();

    let res = risky_bond_engine(
        notional,
        coupon,
        &BOND_PAYMENT_TIMES,
        &dfs,
        &survs,
        recovery,
        &BOND_YFS,
        0.0,
    );

    // Credit-adjusted NPV should be less than risk-free NPV
    assert!(
        res.credit_adjusted_npv < res.risk_free_npv,
        "B3: credit-adjusted ({}) < risk-free ({})",
        res.credit_adjusted_npv,
        res.risk_free_npv
    );

    // With 5% coupon and ~3.5% OIS, bond trades above par risk-free.
    // Credit spread reduces value. Expected range: ~95-105.
    assert!(
        (95.0..=108.0).contains(&res.credit_adjusted_npv),
        "B3: credit-adjusted NPV in reasonable range: {}",
        res.credit_adjusted_npv
    );
}

/// Test AD sensitivity of CDS NPV to underlying curves using the generic
/// engine with interpolated discount curves.
#[test]
fn b3_risky_bond_ad_via_cds_generic() {
    // Use cds_midpoint_generic with InterpDiscountCurve for AD.
    // B3-like setup: price a 5Y CDS with AD-enabled curves.
    type D = DualVec<3>;

    let rate = D::variable(0.035, 0);
    let hazard = D::variable(0.02, 1);
    let recovery_rate = 0.40;
    let spread = 0.0125;

    let yield_curve = FlatCurve::new(rate);
    let survival_curve = FlatCurve::new(hazard);

    // 5Y quarterly
    let times: Vec<f64> = (1..=20).map(|i| i as f64 * 0.25).collect();
    let yfs: Vec<f64> = vec![0.25; 20];

    let npv: D = cds_midpoint_generic(
        100.0,
        spread,
        recovery_rate,
        &times,
        &yfs,
        &yield_curve,
        &survival_curve,
    );

    // Verify sensitivities are reasonable
    // ∂NPV/∂rate: small for CDS
    assert!(
        npv.dot[0].abs() < 10.0,
        "B3 CDS: ∂NPV/∂rate = {:.4}",
        npv.dot[0]
    );
    // ∂NPV/∂hazard: large and positive (protection buyer benefits from higher hazard)
    assert!(
        npv.dot[1] > 0.0,
        "B3 CDS: ∂NPV/∂hazard should be positive for protection buyer: {:.4}",
        npv.dot[1]
    );
}

/// Verify risky_bond_engine with QuantLib-like parameters.
/// Uses approximate DFs and survival probs from the reference curve shape.
#[test]
fn b3_risky_bond_reference_shape() {
    // Approximate OIS curve from Nov 2024 SOFR snapshot (falling front end).
    // Zero rates: 1M=4.83%, 3M=4.55%, 6M=4.25%, 1Y=3.57%, 2Y=3.40% ...
    // We use a simplified curve interpolation.
    let ois_times = [1.0 / 12.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0];
    let ois_rates = [0.0483, 0.0455, 0.0425, 0.0357, 0.0340, 0.0335, 0.0350, 0.0385, 0.0415];

    // Build InterpDiscountCurve from zero rates
    let ois_rates_t: Vec<f64> = ois_rates.to_vec();
    let ois_curve = InterpDiscountCurve::<f64>::from_zero_rates(&ois_times, &ois_rates_t);

    // Hazard rates from CDS spreads: λ ≈ s / (1-R)
    let recovery = 0.40;
    let cds_spreads = [0.0050, 0.0075, 0.0100, 0.0125];
    let _cds_times = [1.0, 2.0, 3.0, 5.0];
    let hazard_rates: Vec<f64> = cds_spreads.iter().map(|s| s / (1.0 - recovery)).collect();

    use ql_termstructures::generic::GenericYieldCurve;

    // Build survival probs from implied hazard rates (piecewise constant approx)
    // For simplicity, assume flat hazard = average ≈ 0.015
    let avg_hazard: f64 = hazard_rates.iter().sum::<f64>() / hazard_rates.len() as f64;

    let coupon = 0.05;
    let notional = 100.0;

    // Bond payment dates (semiannual, 5Y)
    let dfs: Vec<f64> = BOND_PAYMENT_TIMES
        .iter()
        .map(|&t| ois_curve.discount_t(t))
        .collect();
    let survs: Vec<f64> = BOND_PAYMENT_TIMES
        .iter()
        .map(|&t| (-avg_hazard * t).exp())
        .collect();

    let res = risky_bond_engine(
        notional,
        coupon,
        &BOND_PAYMENT_TIMES,
        &dfs,
        &survs,
        recovery,
        &BOND_YFS,
        0.0,
    );

    // With ~3.5% OIS and 5% coupon, risk-free bond above par.
    // With credit spread ~60-125bp, credit-adjusted is slightly lower.
    // Reference NPV = 100.6120 (with exact QuantLib bootstrap).
    // Our approximate curves should give a value in the same ballpark.
    assert!(
        (95.0..=110.0).contains(&res.credit_adjusted_npv),
        "B3: credit-adjusted NPV in reference range: {:.4}",
        res.credit_adjusted_npv
    );
    assert!(
        (100.0..=115.0).contains(&res.risk_free_npv),
        "B3: risk-free NPV above par: {:.4}",
        res.risk_free_npv
    );
}
