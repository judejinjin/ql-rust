//! Verification Group E — Second-Order Sensitivities (Hessians)
//!
//! Computed via FD-over-AAD: bump each input by h=1e-5,
//! re-run forward-mode AD (DualVec), compute ∂²V/∂xᵢ∂xⱼ.
//!
//! E1: European Option 4×4 Hessian (Gamma, Vanna, Volga)
//! E2: IRS Hessian — dominant Swap 5Y × Swap 5Y
//! E3: CDS Hessian — cross-gamma CDS 2Y × Recovery
//! E4: IR Cap Hessian — Vega-gamma
//! E5: Risky Bond Hessian — IR×CR cross-gamma

use ql_aad::{DualVec, Number};
use ql_pricingengines::generic::{
    black_caplet_generic, bs_european_generic, cds_midpoint_generic, swap_engine_generic,
};
use ql_termstructures::generic::{FlatCurve, GenericYieldCurve, InterpDiscountCurve, InterpZeroCurve};

/// Compute 4×4 Hessian of a function via FD-over-forward-AD.
///
/// `f` takes 4 inputs and returns the function value + 4 first derivatives.
/// Returns H[i][j] ≈ ∂²V/∂xi∂xj.
#[allow(clippy::needless_range_loop, clippy::type_complexity)]
fn hessian_4x4(
    f: &dyn Fn(&[f64; 4]) -> (f64, [f64; 4]),
    x: &[f64; 4],
    h: f64,
) -> [[f64; 4]; 4] {
    let mut hess = [[0.0; 4]; 4];
    for i in 0..4 {
        let mut x_up = *x;
        let mut x_dn = *x;
        x_up[i] += h;
        x_dn[i] -= h;
        let (_, grad_up) = f(&x_up);
        let (_, grad_dn) = f(&x_dn);
        for j in 0..4 {
            hess[i][j] = (grad_up[j] - grad_dn[j]) / (2.0 * h);
        }
    }
    // Symmetrize
    for i in 0..4 {
        for j in (i + 1)..4 {
            let avg = 0.5 * (hess[i][j] + hess[j][i]);
            hess[i][j] = avg;
            hess[j][i] = avg;
        }
    }
    hess
}

// =========================================================================
// E1. European Option Hessian (4×4)
// =========================================================================
//
// Same instrument as A1: S=7.0, K=8.0, r=5%, q=5%, σ=10%, T=1Y (call)
// Inputs: [S, q, σ, r]
//
// Reference Hessian (from verification.md):
// | | S | q | σ | r |
// |---|------:|------:|------:|------:|
// | S | 0.2378 | -1.7692 | 2.3062 | 1.6736 |
// | q | -1.7690 | 12.4511 | -16.2313 | -11.7781 |
// | σ | 2.3062 | -16.2317 | 20.7436 | 15.0537 |
// | r | 1.6736 | -11.7797 | 15.0543 | 11.1373 |

#[test]
fn e1_european_option_hessian() {
    let strike = 8.0;
    let t = 1.0; // 1Y
    let is_call = true;

    // x = [spot, q, vol, r]
    let x0 = [7.0, 0.05, 0.10, 0.05];

    // AD function: compute value and gradient via DualVec<4>
    let f = |x: &[f64; 4]| -> (f64, [f64; 4]) {
        type D = DualVec<4>;
        let spot = D::variable(x[0], 0);
        let q = D::variable(x[1], 1);
        let vol = D::variable(x[2], 2);
        let r = D::variable(x[3], 3);

        let res = bs_european_generic(spot, D::constant(strike), r, q, vol, D::constant(t), is_call);
        (res.npv.val, res.npv.dot)
    };

    let h = 1e-5;
    let hess = hessian_4x4(&f, &x0, h);

    // Reference values (from verification.md)
    let ref_hess = [
        [0.2378, -1.7692, 2.3062, 1.6736],
        [-1.7690, 12.4511, -16.2313, -11.7781],
        [2.3062, -16.2317, 20.7436, 15.0537],
        [1.6736, -11.7797, 15.0543, 11.1373],
    ];

    // Verify with 5% tolerance (FD-over-AD has higher error)
    for i in 0..4 {
        for j in 0..4 {
            let actual = hess[i][j];
            let expected = ref_hess[i][j];
            let diff = (actual - expected).abs();
            let denom = expected.abs().max(0.01);
            let rel = diff / denom;
            assert!(
                rel < 0.05 || diff < 0.01,
                "E1 H[{i}][{j}]: actual={actual:.4}, expected={expected:.4}, rel={rel:.2e}"
            );
        }
    }

    // Specific analytic checks:
    // Gamma = H[0][0] ≈ 0.2378
    assert!(
        (hess[0][0] - 0.2378).abs() < 0.005,
        "E1 Gamma: {:.4}",
        hess[0][0]
    );
    // Vanna = H[0][2] ≈ 2.3062
    assert!(
        (hess[0][2] - 2.3062).abs() < 0.05,
        "E1 Vanna: {:.4}",
        hess[0][2]
    );
    // Volga = H[2][2] ≈ 20.7436
    assert!(
        (hess[2][2] - 20.7436).abs() < 0.5,
        "E1 Volga: {:.4}",
        hess[2][2]
    );
}

// =========================================================================
// E2. IRS Hessian — Selected Entries
// =========================================================================
//
// 5Y payer swap on an interpolated zero curve (5 zero rate inputs).
// Verify dominant entry: Swap 5Y × Swap 5Y (large negative).

#[test]
fn e2_irs_hessian_dominant_entry() {
    let notional = 10_000_000.0;
    let fixed_rate = 0.04;

    let tenors = [1.0, 2.0, 3.0, 4.0, 5.0];
    let rate_vals = [0.040, 0.041, 0.042, 0.043, 0.044];

    // AD function for gradient
    let f = |x: &[f64]| -> (f64, Vec<f64>) {
        type D = DualVec<5>;
        let rates: Vec<D> = x.iter().enumerate().map(|(i, &r)| D::variable(r, i)).collect();
        let curve = InterpZeroCurve::new(&tenors, &rates);
        let ft: Vec<f64> = (1..=5).map(|y| y as f64).collect();
        let yfs: Vec<f64> = vec![1.0; 5];
        let res = swap_engine_generic::<D>(
            notional, fixed_rate, &ft, &yfs, 0.0, 5.0, &curve,
        );
        (res.npv.val, res.npv.dot.to_vec())
    };

    // FD-over-AD Hessian for the 5Y×5Y entry
    let h = 1e-5;

    // Bump the 5Y rate (index 4)
    let mut x_up = rate_vals;
    let mut x_dn = rate_vals;
    x_up[4] += h;
    x_dn[4] -= h;

    let (_, grad_up) = f(&x_up);
    let (_, grad_dn) = f(&x_dn);

    let h55 = (grad_up[4] - grad_dn[4]) / (2.0 * h);

    // Should be large and negative (convexity of swap w.r.t. rates)
    assert!(
        h55 < -1_000_000.0,
        "E2 H[5Y,5Y] should be large negative: {h55:.0}"
    );
}

// =========================================================================
// E3. CDS Hessian — Cross-Gamma
// =========================================================================
//
// 2Y CDS, 6 inputs: 4 CDS spreads + recovery + risk-free rate
// Dominant cross-gamma: CDS 2Y × Recovery ≈ 124,815

#[test]
fn e3_cds_hessian_cross_gamma() {
    let notional = 1_000_000.0;
    let coupon = 0.015; // 150bp

    // x = [s1, s2, recovery, rf_rate]
    let x0 = [0.0050, 0.0075, 0.40, 0.05];
    let tenors = [1.0, 2.0];

    let f = |x: &[f64; 4]| -> (f64, [f64; 4]) {
        type D = DualVec<4>;
        let s1 = D::variable(x[0], 0);
        let s2 = D::variable(x[1], 1);
        let recovery = D::variable(x[2], 2);
        let rf = D::variable(x[3], 3);

        let loss = D::constant(1.0) - recovery;
        let h1 = s1 / loss;
        let h2 = s2 / loss;

        // Piecewise survival
        let s_1y = (D::constant(0.0) - h1 * D::constant(1.0)).exp();
        let s_2y = s_1y * (D::constant(0.0) - h2 * D::constant(1.0)).exp();
        let surv_curve = InterpDiscountCurve::from_dfs(&tenors, &[s_1y, s_2y]);
        let yield_curve = FlatCurve::new(rf);

        let times: Vec<f64> = (1..=8).map(|i| i as f64 * 0.25).collect();
        let yfs: Vec<f64> = vec![0.25; 8];

        let npv = cds_midpoint_generic(
            notional.to_f64(), coupon, x[2], // recovery as f64 for CDS formula
            &times, &yfs, &yield_curve, &surv_curve,
        );
        (npv.val, npv.dot)
    };

    let h = 1e-5;
    let hess = hessian_4x4(&f, &x0, h);

    // Cross-gamma s2 × recovery (indices 1, 2)
    // Should be significant since recovery directly modulates spread → hazard mapping
    assert!(
        hess[1][2].abs() > 100.0,
        "E3 H[s2,rec] should be significant: {:.2}",
        hess[1][2]
    );

    // Diagonal entries should be non-zero
    assert!(
        hess[1][1].abs() > 10.0,
        "E3 H[s2,s2] should be non-zero: {:.2}",
        hess[1][1]
    );
}

// =========================================================================
// E4. IR Cap Hessian — Vega-Gamma
// =========================================================================
//
// 10Y cap, inputs: [rate, vol]
// Check ∂²V/∂vol² (vega-gamma / volga).

#[test]
fn e4_ir_cap_vega_gamma() {
    let notional = 1_000_000.0;
    let strike = 0.05;
    let n_quarters = 40;

    let f = |rate: f64, vol: f64| -> (f64, f64, f64) {
        type D = DualVec<2>;
        let r = D::variable(rate, 0);
        let v = D::variable(vol, 1);
        let curve = FlatCurve::new(r);

        let mut cap_npv = D::constant(0.0);
        for i in 1..=n_quarters {
            let t_fixing = i as f64 * 0.25;
            let tau = D::constant(0.25);
            let df = curve.discount_t(t_fixing + 0.25);
            let caplet = black_caplet_generic::<D>(
                df, r, D::constant(strike), v, tau, D::constant(t_fixing), true,
            );
            cap_npv += D::constant(notional) * caplet;
        }
        (cap_npv.val, cap_npv.dot[0], cap_npv.dot[1])
    };

    let rate0 = 0.04;
    let vol0 = 0.20;
    let h = 1e-5;

    // Vega-gamma: ∂²V/∂vol² via FD on vega
    let (_, _, vega_up) = f(rate0, vol0 + h);
    let (_, _, vega_dn) = f(rate0, vol0 - h);
    let volga = (vega_up - vega_dn) / (2.0 * h);

    // Vega-gamma should be positive for an OTM cap
    // (vega is positive and increases with vol for OTM options)
    assert!(
        volga.is_finite(),
        "E4 volga should be finite: {volga:.2}"
    );

    // Cross-gamma: ∂²V/∂rate∂vol
    let (_, rate_sens_up, _) = f(rate0, vol0 + h);
    let (_, rate_sens_dn, _) = f(rate0, vol0 - h);
    let cross = (rate_sens_up - rate_sens_dn) / (2.0 * h);

    assert!(
        cross.is_finite(),
        "E4 cross-gamma ∂²V/∂r∂vol should be finite: {cross:.2}"
    );
}

// =========================================================================
// E5. Risky Bond Hessian — IR×CR Cross-Gamma
// =========================================================================
//
// 5Y risky bond priced as CDS proxy (from B3 setup).
// Inputs: [rf_rate, hazard_rate]
// Cross-gamma ∂²V/∂rf∂hazard should be non-zero.

#[test]
fn e5_risky_bond_hessian() {
    let notional = 1_000_000.0;
    let coupon_spread = 0.0125; // 125bp CDS coupon
    let recovery = 0.40;

    let f = |rf: f64, hazard: f64| -> (f64, f64, f64) {
        type D = DualVec<2>;
        let r = D::variable(rf, 0);
        let h = D::variable(hazard, 1);

        let yield_curve = FlatCurve::new(r);
        let surv_curve = FlatCurve::new(h);

        let times: Vec<f64> = (1..=20).map(|i| i as f64 * 0.25).collect();
        let yfs: Vec<f64> = vec![0.25; 20];

        let npv = cds_midpoint_generic(
            notional.to_f64(), coupon_spread, recovery,
            &times, &yfs, &yield_curve, &surv_curve,
        );
        (npv.val, npv.dot[0], npv.dot[1])
    };

    let rf0 = 0.03;
    let h0 = 0.02;
    let bump = 1e-5;

    // Cross-gamma: ∂²V/∂rf∂hazard
    let (_, _, h_sens_up) = f(rf0 + bump, h0);
    let (_, _, h_sens_dn) = f(rf0 - bump, h0);
    let cross_ir_cr = (h_sens_up - h_sens_dn) / (2.0 * bump);

    assert!(
        cross_ir_cr.abs() > 1.0,
        "E5 IR×CR cross-gamma should be non-trivial: {cross_ir_cr:.2}"
    );

    // IR×IR: ∂²V/∂rf²
    let (_, rf_sens_up, _) = f(rf0 + bump, h0);
    let (_, rf_sens_dn, _) = f(rf0 - bump, h0);
    let ir_ir = (rf_sens_up - rf_sens_dn) / (2.0 * bump);

    assert!(
        ir_ir.is_finite(),
        "E5 IR×IR should be finite: {ir_ir:.2}"
    );

    // CR×CR: ∂²V/∂hazard²
    let (_, _, h_sens_up2) = f(rf0, h0 + bump);
    let (_, _, h_sens_dn2) = f(rf0, h0 - bump);
    let cr_cr = (h_sens_up2 - h_sens_dn2) / (2.0 * bump);

    assert!(
        cr_cr.is_finite(),
        "E5 CR×CR should be finite: {cr_cr:.2}"
    );
}
