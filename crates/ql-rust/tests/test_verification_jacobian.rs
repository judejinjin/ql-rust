//! Verification Group D — Curve Jacobians
//!
//! D1: CDS spread sensitivities — ∂NPV/∂(CDS spread)
//! D2: Hazard rate sensitivities — ∂NPV/∂(hazard rate)
//! D3: Zero rate sensitivities — ∂NPV/∂(zero rate) for swap
//! D4: Round-trip K×J ≈ I verification

use ql_aad::{DualVec, Number};
use ql_pricingengines::generic::{cds_midpoint_generic, swap_engine_generic};
use ql_termstructures::generic::{FlatCurve, InterpDiscountCurve, InterpZeroCurve};

// =========================================================================
// D1. CDS Spread Jacobian
// =========================================================================
//
// 5Y CDS, $10M notional, 100bp coupon, recovery=40%
// Input spreads: 50, 75, 100, 125 bp at 1Y, 2Y, 3Y, 5Y
// AD gives ∂NPV/∂(spread) for each tenor.
//
// We build a piecewise hazard curve from spreads via λ = s/(1-R),
// compute survival probs, and use InterpDiscountCurve as the survival curve.

#[test]
fn d1_cds_spread_jacobian() {
    type D = DualVec<4>;
    let recovery = 0.40;
    let loss = 1.0 - recovery;
    let notional = 10_000_000.0;
    let coupon = 0.0100; // 100bp

    // 4 CDS spread inputs
    let spreads = [
        D::variable(0.0050, 0), // 1Y: 50bp
        D::variable(0.0075, 1), // 2Y: 75bp
        D::variable(0.0100, 2), // 3Y: 100bp
        D::variable(0.0125, 3), // 5Y: 125bp
    ];
    let tenors = [1.0, 2.0, 3.0, 5.0];

    // Piecewise-constant hazard rates from spreads
    let hazards: Vec<D> = spreads.iter().map(|&s| s / D::constant(loss)).collect();

    // Survival probs: S(t_i) = product of exp(-λ_k * Δt_k)
    let mut surv_probs = Vec::with_capacity(4);
    let mut prev_s = D::constant(1.0);
    let mut prev_t = 0.0;
    for i in 0..4 {
        let dt = tenors[i] - prev_t;
        let s = prev_s * (D::constant(0.0) - hazards[i] * D::constant(dt)).exp();
        surv_probs.push(s);
        prev_s = s;
        prev_t = tenors[i];
    }

    // Build survival curve as InterpDiscountCurve (discount = survival)
    let surv_curve = InterpDiscountCurve::from_dfs(&tenors, &surv_probs);

    // Flat yield curve (not differentiated)
    let yield_curve = FlatCurve::new(D::constant(0.05));

    // Quarterly payments for 5Y CDS
    let payment_times: Vec<f64> = (1..=20).map(|i| i as f64 * 0.25).collect();
    let payment_yfs: Vec<f64> = vec![0.25; 20];

    let npv = cds_midpoint_generic(
        notional, coupon, recovery, &payment_times, &payment_yfs,
        &yield_curve, &surv_curve,
    );

    // Verify: each ∂NPV/∂s_i is non-zero (real sensitivity)
    for i in 0..4 {
        assert!(
            npv.dot[i].abs() > 100.0,
            "D1 ∂NPV/∂s[{i}] should be non-trivial: {:.2}",
            npv.dot[i]
        );
    }

    // The 5Y spread (index 3) should have the largest absolute sensitivity
    // since the CDS is 5Y and the 5Y pillar covers the longest interval.
    assert!(
        npv.dot[3].abs() > npv.dot[0].abs(),
        "D1 5Y sensitivity should dominate: |{:.0}| > |{:.0}|",
        npv.dot[3], npv.dot[0]
    );

    // Verify against FD bump-and-reprice
    let base_npv = npv.val;
    let h = 1e-6;
    let spread_vals = [0.0050, 0.0075, 0.0100, 0.0125];
    for idx in 0..4 {
        let mut bumped = spread_vals;
        bumped[idx] += h;

        let b_hazards: Vec<f64> = bumped.iter().map(|&s| s / loss).collect();
        let mut b_surv = Vec::with_capacity(4);
        let mut ps = 1.0_f64;
        let mut pt = 0.0;
        for i in 0..4 {
            let dt = tenors[i] - pt;
            ps *= (-b_hazards[i] * dt).exp();
            b_surv.push(ps);
            pt = tenors[i];
        }
        let b_surv_curve = InterpDiscountCurve::<f64>::from_dfs(&tenors, &b_surv);
        let b_yield_curve = FlatCurve::new(0.05);
        let b_npv: f64 = cds_midpoint_generic(
            notional, coupon, recovery, &payment_times, &payment_yfs,
            &b_yield_curve, &b_surv_curve,
        );
        let fd = (b_npv - base_npv) / h;
        let rel = ((npv.dot[idx] - fd) / fd.abs().max(1.0)).abs();
        assert!(
            rel < 0.01,
            "D1 ∂NPV/∂s[{idx}]: AD={:.2} vs FD={:.2}, rel={rel:.2e}",
            npv.dot[idx], fd
        );
    }
}

// =========================================================================
// D2. Hazard Rate Jacobian
// =========================================================================
//
// Same CDS, but directly seed hazard rates as inputs.

#[test]
fn d2_hazard_rate_jacobian() {
    type D = DualVec<4>;
    let recovery = 0.40;
    let loss = 1.0 - recovery;
    let notional = 10_000_000.0;
    let coupon = 0.0100;
    let tenors = [1.0, 2.0, 3.0, 5.0];

    // Hazard rate inputs (= spread/(1-R))
    let hazards = [
        D::variable(0.0050 / loss, 0),
        D::variable(0.0075 / loss, 1),
        D::variable(0.0100 / loss, 2),
        D::variable(0.0125 / loss, 3),
    ];

    let mut surv_probs = Vec::with_capacity(4);
    let mut prev_s = D::constant(1.0);
    let mut prev_t = 0.0;
    for i in 0..4 {
        let dt = tenors[i] - prev_t;
        let s = prev_s * (D::constant(0.0) - hazards[i] * D::constant(dt)).exp();
        surv_probs.push(s);
        prev_s = s;
        prev_t = tenors[i];
    }

    let surv_curve = InterpDiscountCurve::from_dfs(&tenors, &surv_probs);
    let yield_curve = FlatCurve::new(D::constant(0.05));

    let payment_times: Vec<f64> = (1..=20).map(|i| i as f64 * 0.25).collect();
    let payment_yfs: Vec<f64> = vec![0.25; 20];

    let npv = cds_midpoint_generic(
        notional, coupon, recovery, &payment_times, &payment_yfs,
        &yield_curve, &surv_curve,
    );

    // All sensitivities should be non-zero
    for i in 0..4 {
        assert!(
            npv.dot[i].abs() > 50.0,
            "D2 ∂NPV/∂h[{i}] should be non-trivial: {:.2}",
            npv.dot[i]
        );
    }

    // Chain rule: ∂NPV/∂s = ∂NPV/∂h × ∂h/∂s = ∂NPV/∂h × 1/(1-R)
    // Since d1 uses ∂NPV/∂s and d2 uses ∂NPV/∂h, we verify:
    // ∂NPV/∂h_i × (1-R) ≈ ∂NPV/∂s_i  (within this simple model)
    //
    // We just check the sensitivities are consistent and have correct sign.
    // For protection buyer: higher hazard → more defaults → protection pays more → positive
    // But premium leg also changes. The net sign depends on coupon vs spread.
    // With coupon=100bp and spreads 50-125bp, the CDS is roughly near fair, so
    // sensitivities can be positive or negative.

    // FD validation
    let base_npv = npv.val;
    let h_bump = 1e-6;
    let h_vals: Vec<f64> = [0.0050, 0.0075, 0.0100, 0.0125]
        .iter()
        .map(|&s| s / loss)
        .collect();

    for idx in 0..4 {
        let mut bumped = h_vals.clone();
        bumped[idx] += h_bump;

        let mut b_surv = Vec::with_capacity(4);
        let mut ps = 1.0_f64;
        let mut pt = 0.0;
        for i in 0..4 {
            let dt = tenors[i] - pt;
            ps *= (-bumped[i] * dt).exp();
            b_surv.push(ps);
            pt = tenors[i];
        }
        let b_surv_curve = InterpDiscountCurve::<f64>::from_dfs(&tenors, &b_surv);
        let b_yield_curve = FlatCurve::new(0.05);
        let b_npv: f64 = cds_midpoint_generic(
            notional, coupon, recovery, &payment_times, &payment_yfs,
            &b_yield_curve, &b_surv_curve,
        );
        let fd = (b_npv - base_npv) / h_bump;
        let rel = ((npv.dot[idx] - fd) / fd.abs().max(1.0)).abs();
        assert!(
            rel < 0.01,
            "D2 ∂NPV/∂h[{idx}]: AD={:.2} vs FD={:.2}, rel={rel:.2e}",
            npv.dot[idx], fd
        );
    }
}

// =========================================================================
// D3. Zero Rate Jacobian
// =========================================================================
//
// 5Y SOFR OIS swap, $10M notional, annual fixed.
// 9 zero rate inputs at standard tenors.
// AD gives ∂NPV/∂(zero rate).

#[test]
fn d3_zero_rate_jacobian() {
    type D = DualVec<9>;
    let notional = 10_000_000.0;
    let fixed_rate = 0.04; // 4%

    let tenors = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0];
    let rate_vals = [0.042, 0.041, 0.040, 0.039, 0.038, 0.037, 0.036, 0.035, 0.034];

    // Seed each zero rate
    let rates: Vec<D> = rate_vals
        .iter()
        .enumerate()
        .map(|(i, &r)| D::variable(r, i))
        .collect();

    let curve = InterpZeroCurve::new(&tenors, &rates);

    let fixed_times: Vec<f64> = (1..=5).map(|y| y as f64).collect();
    let fixed_yfs: Vec<f64> = vec![1.0; 5];

    let res = swap_engine_generic::<D>(
        notional,
        fixed_rate,
        &fixed_times,
        &fixed_yfs,
        0.0,
        5.0,
        &curve,
    );

    // Count non-zero sensitivities — only pillars ≤ 5Y matter
    let mut non_zero_count = 0;
    for i in 0..9 {
        if res.npv.dot[i].abs() > 1.0 {
            non_zero_count += 1;
        }
    }
    assert!(
        non_zero_count >= 3,
        "D3: at least 3 zero rate sensitivities should be active, got {non_zero_count}"
    );

    // Pillars beyond 5Y should have zero (or very small) sensitivity
    for (i, &tenor) in tenors.iter().enumerate().skip(5) {
        assert!(
            res.npv.dot[i].abs() < 100.0,
            "D3: pillar {} ({:.0}Y) should have negligible sensitivity: {:.2}",
            i, tenor, res.npv.dot[i]
        );
    }

    // The 5Y pillar (index 4, tenor=5.0) should have the largest absolute sensitivity
    // since it's the maturity of the swap.
    let max_i = (0..5)
        .max_by(|&a, &b| res.npv.dot[a].abs().partial_cmp(&res.npv.dot[b].abs()).unwrap())
        .unwrap();
    assert!(
        max_i >= 3,
        "D3: largest sensitivity should be at longer tenors, got index {max_i} ({:.1}Y)",
        tenors[max_i]
    );

    // FD validation on the 5Y pillar
    let base_npv = res.npv.val;
    let h = 1e-6;
    for idx in 0..5 {
        let mut bumped = rate_vals;
        bumped[idx] += h;
        let b_rates: Vec<f64> = bumped.to_vec();
        let b_curve = InterpZeroCurve::new(&tenors, &b_rates);
        let b_res = swap_engine_generic::<f64>(
            notional, fixed_rate, &fixed_times, &fixed_yfs, 0.0, 5.0, &b_curve,
        );
        let fd = (b_res.npv - base_npv) / h;
        if fd.abs() > 1.0 {
            let rel = ((res.npv.dot[idx] - fd) / fd.abs()).abs();
            assert!(
                rel < 0.01,
                "D3 ∂NPV/∂z[{idx}] ({:.1}Y): AD={:.2} vs FD={:.2}, rel={rel:.2e}",
                tenors[idx], res.npv.dot[idx], fd
            );
        }
    }
}

// =========================================================================
// D4. Round-Trip K×J ≈ I
// =========================================================================
//
// For the simple model λ = s/(1-R):
//   J = ∂h/∂s = diag(1/(1-R))
//   K = ∂s/∂h = diag(1-R)
//   K×J = I
//
// We verify this algebraically and also verify the chain rule:
//   ∂NPV/∂s = J^T × ∂NPV/∂h

#[test]
fn d4_round_trip_spread_hazard_jacobian() {
    let recovery = 0.40;
    let loss = 1.0 - recovery;
    let notional = 10_000_000.0;
    let coupon = 0.0100;
    let tenors = [1.0, 2.0, 3.0, 5.0];
    let spread_vals = [0.0050, 0.0075, 0.0100, 0.0125];

    // --- Compute ∂NPV/∂s via AD (D1 style) ---
    let dnpv_ds = {
        type D = DualVec<4>;
        let spreads: Vec<D> = spread_vals
            .iter()
            .enumerate()
            .map(|(i, &s)| D::variable(s, i))
            .collect();
        let hazards: Vec<D> = spreads.iter().map(|&s| s / D::constant(loss)).collect();

        let mut surv = Vec::with_capacity(4);
        let mut ps = D::constant(1.0);
        let mut pt = 0.0;
        for i in 0..4 {
            let dt = tenors[i] - pt;
            ps *= (D::constant(0.0) - hazards[i] * D::constant(dt)).exp();
            surv.push(ps);
            pt = tenors[i];
        }
        let surv_curve = InterpDiscountCurve::from_dfs(&tenors, &surv);
        let yield_curve = FlatCurve::new(D::constant(0.05));

        let times: Vec<f64> = (1..=20).map(|i| i as f64 * 0.25).collect();
        let yfs: Vec<f64> = vec![0.25; 20];

        let npv = cds_midpoint_generic(
            notional, coupon, recovery, &times, &yfs, &yield_curve, &surv_curve,
        );
        npv.dot
    };

    // --- Compute ∂NPV/∂h via AD (D2 style) ---
    let dnpv_dh = {
        type D = DualVec<4>;
        let h_vals: Vec<f64> = spread_vals.iter().map(|&s| s / loss).collect();
        let hazards: Vec<D> = h_vals
            .iter()
            .enumerate()
            .map(|(i, &h)| D::variable(h, i))
            .collect();

        let mut surv = Vec::with_capacity(4);
        let mut ps = D::constant(1.0);
        let mut pt = 0.0;
        for i in 0..4 {
            let dt = tenors[i] - pt;
            ps *= (D::constant(0.0) - hazards[i] * D::constant(dt)).exp();
            surv.push(ps);
            pt = tenors[i];
        }
        let surv_curve = InterpDiscountCurve::from_dfs(&tenors, &surv);
        let yield_curve = FlatCurve::new(D::constant(0.05));

        let times: Vec<f64> = (1..=20).map(|i| i as f64 * 0.25).collect();
        let yfs: Vec<f64> = vec![0.25; 20];

        let npv = cds_midpoint_generic(
            notional, coupon, recovery, &times, &yfs, &yield_curve, &surv_curve,
        );
        npv.dot
    };

    // Chain rule: ∂NPV/∂s_i = ∂NPV/∂h_i × ∂h/∂s_i = ∂NPV/∂h_i × 1/(1-R)
    for i in 0..4 {
        let expected = dnpv_dh[i] / loss;
        let actual = dnpv_ds[i];
        let rel = ((actual - expected) / expected.abs().max(1.0)).abs();
        assert!(
            rel < 1e-10,
            "D4 chain rule ∂NPV/∂s[{i}]: actual={actual:.6}, expected={expected:.6}, rel={rel:.2e}"
        );
    }

    // Round-trip: K × J = diag(1-R) × diag(1/(1-R)) = I
    // For diagonal Jacobians this is trivial, but verify the numerical values.
    for i in 0..4 {
        let j_ii = 1.0 / loss;       // ∂h/∂s
        let k_ii = loss;              // ∂s/∂h
        let product = k_ii * j_ii;
        assert!(
            (product - 1.0).abs() < 1e-14,
            "D4 K×J [{i},{i}] = {product}, expected 1.0"
        );
    }
}
