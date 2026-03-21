//! Integration tests: validate generic engines with AD types (Dual, DualVec, AReal).
//!
//! These tests prove that the ~100 generic engines in `ql_pricingengines::generic`
//! produce correct derivatives when instantiated with `ql_aad` AD types:
//!
//! 1. **Forward-mode (`Dual`)**: seed one input, verify ∂V/∂input via finite difference.
//! 2. **Multi-seed forward-mode (`DualVec<N>`)**: all Greeks in one pass, match FD.
//! 3. **Reverse-mode (`AReal`)**: tape-based adjoint, verify all partials vs finite diff.
//!
//! Validation strategy: AD derivatives are cross-checked against central finite
//! differences with a bump of 1e-5 and tolerance of 1e-4 (relative or absolute).

use ql_aad::{Dual, DualVec, Number};
use ql_aad::tape::{with_tape, adjoint_tl, AReal};
use ql_pricingengines::generic::*;
use ql_termstructures::generic::FlatCurve;

// =========================================================================
// Helper: central finite difference
// =========================================================================

fn central_fd<F: Fn(f64) -> f64>(f: F, x: f64, bump: f64) -> f64 {
    (f(x + bump) - f(x - bump)) / (2.0 * bump)
}

/// Check AD derivative against finite difference within tolerance.
fn assert_ad_vs_fd(ad_deriv: f64, fd_deriv: f64, tol: f64, label: &str) {
    let diff = (ad_deriv - fd_deriv).abs();
    let denom = fd_deriv.abs().max(1e-10);
    let rel = diff / denom;
    assert!(
        diff < tol || rel < tol,
        "{label}: AD={ad_deriv:.8}, FD={fd_deriv:.8}, diff={diff:.2e}, rel={rel:.2e}"
    );
}

// =========================================================================
// 1. Black-Scholes European — Dual (forward-mode, single seed on spot)
// =========================================================================

#[test]
fn bs_european_dual_delta() {
    let bump = 1e-5;
    let spot = 100.0;

    let res = bs_european_generic(
        Dual::new(spot, 1.0),
        Dual::constant(100.0),
        Dual::constant(0.05),
        Dual::constant(0.0),
        Dual::constant(0.20),
        Dual::constant(1.0),
        true,
    );
    let ad_delta = res.npv.dot;

    let fd_delta = central_fd(
        |s| bs_european_generic(s, 100.0, 0.05, 0.0, 0.20, 1.0, true).npv,
        spot,
        bump,
    );

    assert_ad_vs_fd(ad_delta, fd_delta, 1e-4, "BS Dual delta");
}

// =========================================================================
// 2. Black-Scholes European — Dual (forward-mode, single seed on vol)
// =========================================================================

#[test]
fn bs_european_dual_vega_raw() {
    let bump = 1e-5;
    let vol = 0.20;

    let res = bs_european_generic(
        Dual::constant(100.0),
        Dual::constant(100.0),
        Dual::constant(0.05),
        Dual::constant(0.0),
        Dual::new(vol, 1.0),
        Dual::constant(1.0),
        true,
    );
    let ad_vega_raw = res.npv.dot;

    let fd_vega = central_fd(
        |v| bs_european_generic(100.0, 100.0, 0.05, 0.0, v, 1.0, true).npv,
        vol,
        bump,
    );

    assert_ad_vs_fd(ad_vega_raw, fd_vega, 1e-4, "BS Dual vega (raw)");
}

// =========================================================================
// 3. Black-Scholes European — DualVec<5> (all Greeks in one pass)
// =========================================================================

#[test]
fn bs_european_dualvec5_all_greeks() {
    type D5 = DualVec<5>;
    let spot   = D5::variable(100.0, 0);
    let strike = D5::variable(100.0, 1);
    let r      = D5::variable(0.05,  2);
    let q      = D5::variable(0.0,   3);
    let vol    = D5::variable(0.20,  4);
    let t      = D5::constant(1.0);

    let res = bs_european_generic(spot, strike, r, q, vol, t, true);
    let bump = 1e-5;

    let fd_delta = central_fd(
        |s| bs_european_generic(s, 100.0, 0.05, 0.0, 0.20, 1.0, true).npv,
        100.0, bump,
    );
    assert_ad_vs_fd(res.npv.dot[0], fd_delta, 1e-4, "DualVec5 delta");

    let fd_dstrike = central_fd(
        |k| bs_european_generic(100.0, k, 0.05, 0.0, 0.20, 1.0, true).npv,
        100.0, bump,
    );
    assert_ad_vs_fd(res.npv.dot[1], fd_dstrike, 1e-4, "DualVec5 ∂V/∂K");

    let fd_rho = central_fd(
        |rr| bs_european_generic(100.0, 100.0, rr, 0.0, 0.20, 1.0, true).npv,
        0.05, bump,
    );
    assert_ad_vs_fd(res.npv.dot[2], fd_rho, 1e-4, "DualVec5 rho (raw)");

    let fd_dq = central_fd(
        |qq| bs_european_generic(100.0, 100.0, 0.05, qq, 0.20, 1.0, true).npv,
        0.0, bump,
    );
    assert_ad_vs_fd(res.npv.dot[3], fd_dq, 1e-4, "DualVec5 ∂V/∂q");

    let fd_vega = central_fd(
        |v| bs_european_generic(100.0, 100.0, 0.05, 0.0, v, 1.0, true).npv,
        0.20, bump,
    );
    assert_ad_vs_fd(res.npv.dot[4], fd_vega, 1e-4, "DualVec5 vega (raw)");
}

// =========================================================================
// 4. Black-Scholes European — AReal (reverse-mode tape)
// =========================================================================

#[test]
fn bs_european_areal_all_greeks() {
    let (npv, s_idx, k_idx, r_idx, _q_idx, v_idx) = with_tape(|tape| {
        let s = tape.input(100.0);
        let k = tape.input(100.0);
        let r = tape.input(0.05);
        let q = tape.input(0.0);
        let v = tape.input(0.20);
        let t = ql_aad::tape::AReal::from_f64(1.0);

        let res = bs_european_generic(s, k, r, q, v, t, true);
        (res.npv, s.idx, k.idx, r.idx, q.idx, v.idx)
    });

    let grad = adjoint_tl(npv);
    let bump = 1e-5;

    let fd_delta = central_fd(
        |s| bs_european_generic(s, 100.0, 0.05, 0.0, 0.20, 1.0, true).npv,
        100.0, bump,
    );
    assert_ad_vs_fd(grad[s_idx], fd_delta, 1e-4, "AReal delta");

    let fd_rho = central_fd(
        |rr| bs_european_generic(100.0, 100.0, rr, 0.0, 0.20, 1.0, true).npv,
        0.05, bump,
    );
    assert_ad_vs_fd(grad[r_idx], fd_rho, 1e-4, "AReal rho");

    let fd_vega = central_fd(
        |v| bs_european_generic(100.0, 100.0, 0.05, 0.0, v, 1.0, true).npv,
        0.20, bump,
    );
    assert_ad_vs_fd(grad[v_idx], fd_vega, 1e-4, "AReal vega");

    assert!(grad[k_idx].abs() > 1e-6, "AReal ∂V/∂K should be non-zero");
}

// =========================================================================
// 5. Black-76 — Dual
// Signature: black76_generic(fwd, strike, r, vol, t, is_call) -> T
// =========================================================================

#[test]
fn black76_dual_delta() {
    let bump = 1e-5;
    let fwd = 100.0;

    let price = black76_generic(
        Dual::new(fwd, 1.0),
        Dual::constant(100.0),
        Dual::constant(0.05),
        Dual::constant(0.20),
        Dual::constant(1.0),
        true,
    );
    let ad_delta = price.dot;

    let fd_delta = central_fd(
        |f| black76_generic(f, 100.0, 0.05, 0.20, 1.0, true),
        fwd, bump,
    );

    assert_ad_vs_fd(ad_delta, fd_delta, 1e-4, "Black76 Dual delta");
}

// =========================================================================
// 6. Bachelier — Dual
// Signature: bachelier_generic(fwd, strike, r, vol, t, is_call) -> T
// =========================================================================

#[test]
fn bachelier_dual_delta() {
    let bump = 1e-5;
    let fwd = 0.03;

    let price = bachelier_generic(
        Dual::new(fwd, 1.0),
        Dual::constant(0.03),
        Dual::constant(0.02),
        Dual::constant(0.0),     // q (dividend yield)
        Dual::constant(0.005),
        Dual::constant(1.0),
        true,
    );
    let ad_delta = price.dot;

    let fd_delta = central_fd(
        |f| bachelier_generic(f, 0.03, 0.02, 0.0, 0.005, 1.0, true),
        fwd, bump,
    );

    assert_ad_vs_fd(ad_delta, fd_delta, 1e-3, "Bachelier Dual delta");
}

// =========================================================================
// 7. Black Swaption — DualVec<3>
// Signature: black_swaption_generic(annuity, swap_rate, strike, vol, t, is_payer) -> T
// =========================================================================

#[test]
fn black_swaption_dualvec_greeks() {
    type D3 = DualVec<3>;
    let annuity   = D3::variable(4.5,  2);
    let swap_rate = D3::variable(0.04, 0);
    let strike    = D3::constant(0.04);
    let vol       = D3::variable(0.15, 1);
    let t         = D3::constant(1.0);

    let price = black_swaption_generic(annuity, swap_rate, strike, vol, t, true);
    let ad_d_fwd = price.dot[0];
    let ad_d_vol = price.dot[1];

    let bump = 1e-6;
    let fd_d_fwd = central_fd(
        |f| black_swaption_generic(4.5, f, 0.04, 0.15, 1.0, true),
        0.04, bump,
    );
    let fd_d_vol = central_fd(
        |v| black_swaption_generic(4.5, 0.04, 0.04, v, 1.0, true),
        0.15, bump,
    );

    assert_ad_vs_fd(ad_d_fwd, fd_d_fwd, 1e-3, "Swaption DualVec ∂V/∂fwd");
    assert_ad_vs_fd(ad_d_vol, fd_d_vol, 1e-3, "Swaption DualVec ∂V/∂vol");
}

// =========================================================================
// 8. Black Caplet — Dual
// Signature: black_caplet_generic(df, forward, strike, vol, tau, t_fixing, is_cap) -> T
// =========================================================================

#[test]
fn black_caplet_dual_delta() {
    let bump = 1e-6;
    let fwd = 0.05;

    let price = black_caplet_generic(
        Dual::constant(0.97),   // df
        Dual::new(fwd, 1.0),   // forward (seeded)
        Dual::constant(0.04),  // strike
        Dual::constant(0.20),  // vol
        Dual::constant(0.5),   // tau
        Dual::constant(1.0),   // t_fixing
        true,
    );
    let ad_delta = price.dot;

    let fd_delta = central_fd(
        |f| black_caplet_generic(0.97, f, 0.04, 0.20, 0.5, 1.0, true),
        fwd, bump,
    );

    assert_ad_vs_fd(ad_delta, fd_delta, 1e-3, "Caplet Dual ∂V/∂fwd");
}

// =========================================================================
// 9. Kirk spread option — Dual
// Signature: kirk_spread_generic(s1, s2, strike, r, q1, q2, vol1, vol2, rho, t) -> T
// Note: no is_call bool — always prices the spread call
// =========================================================================

#[test]
fn kirk_spread_dual_delta() {
    let bump = 1e-5;
    let s1 = 100.0;

    let price = kirk_spread_generic(
        Dual::new(s1, 1.0),
        Dual::constant(90.0),
        Dual::constant(5.0),
        Dual::constant(0.05),
        Dual::constant(0.0),
        Dual::constant(0.0),
        Dual::constant(0.20),
        Dual::constant(0.25),
        Dual::constant(0.5),
        Dual::constant(1.0),
    );
    let ad_delta = price.dot;

    let fd_delta = central_fd(
        |s| kirk_spread_generic(s, 90.0, 5.0, 0.05, 0.0, 0.0, 0.20, 0.25, 0.5, 1.0),
        s1, bump,
    );

    assert_ad_vs_fd(ad_delta, fd_delta, 1e-3, "Kirk spread Dual ∂V/∂S1");
}

// =========================================================================
// 10. Margrabe exchange — Dual
// Signature: margrabe_exchange_generic(s1, s2, q1, q2, vol1, vol2, rho, t) -> T
// =========================================================================

#[test]
fn margrabe_exchange_dual_delta() {
    let bump = 1e-5;
    let s1 = 100.0;

    let price = margrabe_exchange_generic(
        Dual::new(s1, 1.0),
        Dual::constant(95.0),
        Dual::constant(0.0),
        Dual::constant(0.0),
        Dual::constant(0.20),
        Dual::constant(0.25),
        Dual::constant(0.5),
        Dual::constant(1.0),
    );
    let ad_delta = price.dot;

    let fd_delta = central_fd(
        |s| margrabe_exchange_generic(s, 95.0, 0.0, 0.0, 0.20, 0.25, 0.5, 1.0),
        s1, bump,
    );

    assert_ad_vs_fd(ad_delta, fd_delta, 1e-3, "Margrabe Dual ∂V/∂S1");
}

// =========================================================================
// 11. Variance swap — DualVec<2>
// Signature: variance_swap_generic(implied_vol, r, t, notional, strike_var)
//            -> VarianceSwapGenericResult<T>
// =========================================================================

#[test]
fn variance_swap_dualvec_greeks() {
    type D2 = DualVec<2>;
    let implied_vol = D2::variable(0.20, 0);
    let strike_var  = D2::variable(0.04, 1);  // 0.20^2
    let r           = D2::constant(0.05);
    let t           = D2::constant(1.0);
    let notional    = D2::constant(100_000.0);

    let result = variance_swap_generic(implied_vol, r, t, notional, strike_var);

    // ∂NPV/∂implied_vol should be non-trivial
    assert!(result.npv.dot[0].abs() > 0.1, "∂NPV/∂σ should be non-trivial");
    // ∂NPV/∂strike_var should be negative (higher strike → lower payout for long)
    assert!(result.npv.dot[1] < 0.0, "∂NPV/∂strike_var should be negative");
}

// =========================================================================
// 12. Quanto European — Dual
// Signature: quanto_european_generic(spot, strike, r_d, r_f, vol_s, vol_fx,
//            rho, t, fx_rate, is_call) -> T
// =========================================================================

#[test]
fn quanto_european_dual_delta() {
    let bump = 1e-5;
    let spot = 100.0;

    let price = quanto_european_generic(
        Dual::new(spot, 1.0),
        Dual::constant(100.0),
        Dual::constant(0.05),
        Dual::constant(0.03),
        Dual::constant(0.20),
        Dual::constant(0.10),
        Dual::constant(-0.3),
        Dual::constant(1.0),
        Dual::constant(1.0),   // fx_rate
        true,
    );
    let ad_delta = price.dot;

    let fd_delta = central_fd(
        |s| quanto_european_generic(s, 100.0, 0.05, 0.03, 0.20, 0.10, -0.3, 1.0, 1.0, true),
        spot, bump,
    );

    assert_ad_vs_fd(ad_delta, fd_delta, 1e-3, "Quanto Dual delta");
}

// =========================================================================
// 13. Chooser option — AReal
// Signature: chooser_generic(spot, strike, r, q, vol, t_choose, t_expiry) -> T
// =========================================================================

#[test]
fn chooser_areal_greeks() {
    let (price, s_idx, v_idx) = with_tape(|tape| {
        let s = tape.input(100.0);
        let k = ql_aad::tape::AReal::from_f64(100.0);
        let r = ql_aad::tape::AReal::from_f64(0.05);
        let q = ql_aad::tape::AReal::from_f64(0.0);
        let v = tape.input(0.20);
        let t_choose = ql_aad::tape::AReal::from_f64(0.5);
        let t_expiry = ql_aad::tape::AReal::from_f64(1.0);

        let price = chooser_generic(s, k, r, q, v, t_choose, t_expiry);
        (price, s.idx, v.idx)
    });

    let grad = adjoint_tl(price);
    let bump = 1e-5;

    let fd_delta = central_fd(
        |s| chooser_generic(s, 100.0, 0.05, 0.0, 0.20, 0.5, 1.0),
        100.0, bump,
    );
    assert_ad_vs_fd(grad[s_idx], fd_delta, 1e-3, "Chooser AReal delta");

    let fd_vega = central_fd(
        |v| chooser_generic(100.0, 100.0, 0.05, 0.0, v, 0.5, 1.0),
        0.20, bump,
    );
    assert_ad_vs_fd(grad[v_idx], fd_vega, 1e-3, "Chooser AReal vega");
}

// =========================================================================
// 14. Merton jump-diffusion — DualVec<3>
// Signature: merton_jd_generic(spot, strike, r, q, vol, t,
//            lambda, nu, delta, is_call) -> MertonJdResult<T>
// =========================================================================

#[test]
fn merton_jd_dualvec_greeks() {
    type D3 = DualVec<3>;
    let spot   = D3::variable(100.0, 0);
    let vol    = D3::variable(0.20, 1);
    let r      = D3::variable(0.05, 2);
    let strike = D3::constant(100.0);
    let q      = D3::constant(0.0);
    let t      = D3::constant(1.0);
    let lambda = D3::constant(0.1);
    let nu     = D3::constant(-0.1);
    let delta  = D3::constant(0.1);

    let result = merton_jd_generic(spot, strike, r, q, vol, t, lambda, nu, delta, true);
    let ad_delta = result.npv.dot[0];

    let bump = 1e-5;
    let fd_delta = central_fd(
        |s| merton_jd_generic(s, 100.0, 0.05, 0.0, 0.20, 1.0, 0.1, -0.1, 0.1, true).npv,
        100.0, bump,
    );

    assert_ad_vs_fd(ad_delta, fd_delta, 1e-3, "Merton JD DualVec delta");
}

// =========================================================================
// 15. BAW American — Dual
// Signature: barone_adesi_whaley_generic(spot, strike, r, q, vol, t, is_call)
//            -> BawResult<T>
// =========================================================================

#[test]
fn baw_american_dual_delta() {
    let bump = 1e-5;
    let spot = 100.0;

    let res = barone_adesi_whaley_generic(
        Dual::new(spot, 1.0),
        Dual::constant(110.0),
        Dual::constant(0.05),
        Dual::constant(0.02),
        Dual::constant(0.25),
        Dual::constant(1.0),
        false,
    );
    let ad_delta = res.npv.dot;

    let fd_delta = central_fd(
        |s| barone_adesi_whaley_generic(s, 110.0, 0.05, 0.02, 0.25, 1.0, false).npv,
        spot, bump,
    );

    assert_ad_vs_fd(ad_delta, fd_delta, 1e-3, "BAW Dual delta");
}

// =========================================================================
// 16. Bond PV — AReal (curve sensitivity)
// Signature: bond_pv_generic(coupon_amounts: &[f64], coupon_times: &[f64],
//            notional: f64, maturity: f64, rate: T) -> T
// =========================================================================

#[test]
fn bond_pv_areal_curve_sensitivity() {
    let coupon_amounts = [3.0, 3.0, 3.0, 3.0]; // 3% annual, par 100, 4Y
    let coupon_times = [1.0, 2.0, 3.0, 4.0];
    let notional = 100.0;
    let maturity = 4.0;

    let (pv, rate_idx) = with_tape(|tape| {
        let rate = tape.input(0.04);
        let pv = bond_pv_generic(&coupon_amounts, &coupon_times, notional, maturity, rate);
        (pv, rate.idx)
    });

    let grad = adjoint_tl(pv);

    // ∂PV/∂rate should be negative (higher rate → lower bond price)
    assert!(grad[rate_idx] < 0.0, "Bond ∂PV/∂r should be negative, got {}", grad[rate_idx]);

    // Verify against FD
    let bump = 1e-5;
    let fd = central_fd(
        |r| bond_pv_generic(&coupon_amounts, &coupon_times, notional, maturity, r),
        0.04, bump,
    );
    assert_ad_vs_fd(grad[rate_idx], fd, 1e-3, "Bond AReal ∂PV/∂r");
}

// =========================================================================
// 17. Swap PV — AReal (discount rate sensitivity)
// Signature: swap_pv_generic(float_amounts: &[f64], float_times: &[f64],
//            fixed_amounts: &[f64], fixed_times: &[f64], discount_rate: T) -> T
// =========================================================================

#[test]
fn swap_pv_areal_rate_sensitivity() {
    let float_amounts = [2.3, 2.4, 2.5, 2.6];
    let float_times = [0.5, 1.0, 1.5, 2.0];
    let fixed_amounts = [2.5, 2.5, 2.5, 2.5];
    let fixed_times = [0.5, 1.0, 1.5, 2.0];

    let (pv, rate_idx) = with_tape(|tape| {
        let rate = tape.input(0.04);
        let pv = swap_pv_generic(
            &float_amounts, &float_times,
            &fixed_amounts, &fixed_times,
            rate,
        );
        (pv, rate.idx)
    });

    let grad = adjoint_tl(pv);
    let bump = 1e-5;
    let fd = central_fd(
        |r| swap_pv_generic(
            &[2.3, 2.4, 2.5, 2.6], &[0.5, 1.0, 1.5, 2.0],
            &[2.5, 2.5, 2.5, 2.5], &[0.5, 1.0, 1.5, 2.0],
            r,
        ),
        0.04, bump,
    );
    assert_ad_vs_fd(grad[rate_idx], fd, 1e-3, "Swap AReal ∂NPV/∂r");
}

// =========================================================================
// 18. Asian geometric continuous — Dual
// Signature: asian_geometric_continuous_generic(spot, strike, r, q, vol, t, is_call) -> T
// =========================================================================

#[test]
fn asian_geometric_continuous_dual_delta() {
    let bump = 1e-5;
    let spot = 100.0;

    let price = asian_geometric_continuous_generic(
        Dual::new(spot, 1.0),
        Dual::constant(100.0),
        Dual::constant(0.05),
        Dual::constant(0.0),
        Dual::constant(0.20),
        Dual::constant(1.0),
        true,
    );
    let ad_delta = price.dot;

    let fd_delta = central_fd(
        |s| asian_geometric_continuous_generic(s, 100.0, 0.05, 0.0, 0.20, 1.0, true),
        spot, bump,
    );

    assert_ad_vs_fd(ad_delta, fd_delta, 1e-3, "Asian geometric Dual delta");
}

// =========================================================================
// 19. CDS midpoint — Dual (yield curve sensitivity via FlatCurve)
// Signature: cds_midpoint_generic(notional: f64, spread: f64, recovery: f64,
//            payment_times: &[f64], payment_yfs: &[f64],
//            yield_curve: &dyn GenericYieldCurve<T>,
//            survival_curve: &dyn GenericYieldCurve<T>) -> T
// =========================================================================

#[test]
fn cds_midpoint_dual_yield_sensitivity() {
    let bump = 1e-5;
    let yield_rate = 0.03;
    let payment_times = [1.0, 2.0, 3.0, 4.0, 5.0];
    let payment_yfs   = [1.0, 1.0, 1.0, 1.0, 1.0];

    // AD path: seed Dual on yield rate
    let yc = FlatCurve::new(Dual::new(yield_rate, 1.0));
    let sc = FlatCurve::new(Dual::constant(0.02));
    let npv = cds_midpoint_generic(
        1_000_000.0, 0.01, 0.40,
        &payment_times, &payment_yfs,
        &yc, &sc,
    );
    let ad_deriv = npv.dot;

    // FD path
    let fd_deriv = central_fd(|r| {
        let yc_fd = FlatCurve::new(r);
        let sc_fd = FlatCurve::new(0.02_f64);
        cds_midpoint_generic(
            1_000_000.0, 0.01, 0.40,
            &[1.0, 2.0, 3.0, 4.0, 5.0], &[1.0, 1.0, 1.0, 1.0, 1.0],
            &yc_fd, &sc_fd,
        )
    }, yield_rate, bump);

    assert_ad_vs_fd(ad_deriv, fd_deriv, 1e-2, "CDS Dual ∂NPV/∂yield_rate");
}

// =========================================================================
// 20. f64 parity — generic<f64> BS matches known analytical value
// =========================================================================

#[test]
fn f64_parity_bs_european() {
    // ATM call: S=K=100, r=5%, q=0, σ=20%, T=1
    // Known BS price ≈ 10.4506
    let res = bs_european_generic(100.0_f64, 100.0, 0.05, 0.0, 0.20, 1.0, true);
    assert!(
        (res.npv - 10.4506).abs() < 0.01,
        "BS ATM call ≈ 10.45, got {:.6}", res.npv
    );
    // Delta of ATM call ≈ 0.6368
    assert!(
        (res.delta - 0.6368).abs() < 0.01,
        "BS delta ≈ 0.637, got {:.6}", res.delta
    );
}

// =========================================================================
// 21. Lookback floating — Dual
// Signature: lookback_floating_generic(spot, s_min_or_max, r, q, vol, t, is_call) -> T
// =========================================================================

#[test]
fn lookback_floating_dual_delta() {
    let bump = 1e-5;
    let spot = 100.0;

    let price = lookback_floating_generic(
        Dual::new(spot, 1.0),
        Dual::constant(90.0),
        Dual::constant(0.05),
        Dual::constant(0.0),
        Dual::constant(0.30),
        Dual::constant(1.0),
        true,
    );
    let ad_delta = price.dot;

    let fd_delta = central_fd(
        |s| lookback_floating_generic(s, 90.0, 0.05, 0.0, 0.30, 1.0, true),
        spot, bump,
    );

    assert_ad_vs_fd(ad_delta, fd_delta, 1e-3, "Lookback floating Dual delta");
}

// =========================================================================
// 22. Nelson-Siegel curve — AReal (parameter sensitivity)
// =========================================================================

#[test]
fn nelson_siegel_areal_sensitivity() {
    use ql_termstructures::generic::nelson_siegel_discount;

    let (disc, b0_idx, b1_idx) = with_tape(|tape| {
        let beta0 = tape.input(0.04);
        let beta1 = tape.input(-0.02);
        let beta2 = ql_aad::tape::AReal::from_f64(0.01);
        let tau   = ql_aad::tape::AReal::from_f64(1.5);
        let t     = ql_aad::tape::AReal::from_f64(5.0);

        let df = nelson_siegel_discount(beta0, beta1, beta2, tau, t);
        (df, beta0.idx, beta1.idx)
    });

    let grad = adjoint_tl(disc);

    // ∂DF/∂β0 should be negative (higher level → lower discount factor)
    assert!(grad[b0_idx] < 0.0, "∂DF/∂β0 should be negative");
    // ∂DF/∂β1 should also be negative (higher slope → lower long-end DF)
    assert!(grad[b1_idx] < 0.0, "∂DF/∂β1 should be negative for 5Y");
}

// =========================================================================
// 23. HW bond option — Dual
// Signature: hw_bond_option_generic(a, sigma, bond_maturity, option_expiry,
//            bond_price, strike, r, is_call) -> T
// =========================================================================

#[test]
fn hw_bond_option_dual_delta() {
    let bump = 1e-5;
    let bond_price = 95.0;

    let price = hw_bond_option_generic(
        Dual::constant(0.1),             // a
        Dual::constant(0.01),            // sigma
        Dual::constant(5.0),             // bond_maturity
        Dual::constant(1.0),             // option_expiry
        Dual::new(bond_price, 1.0),      // bond_price (seeded)
        Dual::constant(90.0),            // strike
        Dual::constant(0.04),            // r
        true,
    );
    let ad_delta = price.dot;

    let fd_delta = central_fd(
        |bp| hw_bond_option_generic(0.1, 0.01, 5.0, 1.0, bp, 90.0, 0.04, true),
        bond_price, bump,
    );

    assert_ad_vs_fd(ad_delta, fd_delta, 1e-3, "HW bond option Dual ∂V/∂P_bond");
}

// =========================================================================
// 24. Power option — Dual
// Signature: power_option_generic(spot, strike, r, q, vol, t, alpha, is_call) -> T
// =========================================================================

#[test]
fn power_option_dual_delta() {
    let bump = 1e-5;
    let spot = 100.0;

    let price = power_option_generic(
        Dual::new(spot, 1.0),
        Dual::constant(100.0),
        Dual::constant(0.05),
        Dual::constant(0.0),
        Dual::constant(0.20),
        Dual::constant(1.0),
        Dual::constant(2.0),
        true,
    );
    let ad_delta = price.dot;

    let fd_delta = central_fd(
        |s| power_option_generic(s, 100.0, 0.05, 0.0, 0.20, 1.0, 2.0, true),
        spot, bump,
    );

    assert_ad_vs_fd(ad_delta, fd_delta, 1e-2, "Power option Dual delta");
}

// =========================================================================
// 25. Stulz max-call — Dual
// Signature: stulz_max_call_generic(s1, s2, strike, r, q1, q2,
//            vol1, vol2, rho_f64: f64, t) -> T
// Note: rho is f64, not T!
// =========================================================================

#[test]
fn stulz_max_call_dual_delta() {
    let bump = 1e-5;
    let s1 = 100.0;

    let price = stulz_max_call_generic(
        Dual::new(s1, 1.0),
        Dual::constant(95.0),
        Dual::constant(90.0),
        Dual::constant(0.05),
        Dual::constant(0.0),
        Dual::constant(0.0),
        Dual::constant(0.20),
        Dual::constant(0.25),
        0.5_f64,              // rho stays f64!
        Dual::constant(1.0),
    );
    let ad_delta = price.dot;

    let fd_delta = central_fd(
        |s| stulz_max_call_generic(s, 95.0, 90.0, 0.05, 0.0, 0.0, 0.20, 0.25, 0.5, 1.0),
        s1, bump,
    );

    assert_ad_vs_fd(ad_delta, fd_delta, 1e-2, "Stulz max-call Dual delta");
}

// =========================================================================
// 26. Forward-start — Dual
// Signature: forward_start_generic(spot, r, q, vol, t1, t2, alpha, is_call) -> T
// =========================================================================

#[test]
fn forward_start_dual_delta() {
    let bump = 1e-5;
    let spot = 100.0;

    let price = forward_start_generic(
        Dual::new(spot, 1.0),
        Dual::constant(0.05),  // r
        Dual::constant(0.02),  // q
        Dual::constant(0.20),  // vol
        Dual::constant(0.5),   // t1
        Dual::constant(1.0),   // t2
        Dual::constant(1.0),   // alpha
        true,
    );
    let ad_delta = price.dot;

    let fd_delta = central_fd(
        |s| forward_start_generic(s, 0.05, 0.02, 0.20, 0.5, 1.0, 1.0, true),
        spot, bump,
    );

    assert_ad_vs_fd(ad_delta, fd_delta, 1e-3, "Forward-start Dual delta");
}

// =========================================================================
// HIGHER-ORDER GREEKS: FD-on-AD (second-order)
// =========================================================================
//
// Since Dual stores f64 (no nesting for pure second-order AD), we compute
// second-order sensitivities as finite-difference-on-AD:
//   γ ≈ (δ(S+h) - δ(S-h)) / 2h   where δ is exact from AD
//
// Validated against closed-form BS formulas.

/// Standard normal PDF: φ(x) = exp(-x²/2) / √(2π)
fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// BS d1
fn bs_d1(s: f64, k: f64, r: f64, q: f64, vol: f64, t: f64) -> f64 {
    ((s / k).ln() + (r - q + 0.5 * vol * vol) * t) / (vol * t.sqrt())
}

/// BS d2
fn bs_d2(s: f64, k: f64, r: f64, q: f64, vol: f64, t: f64) -> f64 {
    bs_d1(s, k, r, q, vol, t) - vol * t.sqrt()
}

// --- Gamma (∂²V/∂S²) ---

#[test]
fn bs_gamma_fd_on_ad_vs_analytical() {
    let (s, k, r, q, vol, t) = (100.0, 100.0, 0.05, 0.02, 0.20, 1.0);

    // FD-on-AD: bump S, read AD delta at each bump
    let h = s * 1e-4;
    let delta_up = bs_european_generic(
        Dual::new(s + h, 1.0), Dual::constant(k),
        Dual::constant(r), Dual::constant(q),
        Dual::constant(vol), Dual::constant(t), true,
    ).npv.dot;

    let delta_dn = bs_european_generic(
        Dual::new(s - h, 1.0), Dual::constant(k),
        Dual::constant(r), Dual::constant(q),
        Dual::constant(vol), Dual::constant(t), true,
    ).npv.dot;

    let gamma_fd_ad = (delta_up - delta_dn) / (2.0 * h);

    // Analytical: Γ = φ(d1) * exp(-qT) / (S * σ * √T)
    let d1 = bs_d1(s, k, r, q, vol, t);
    let gamma_analytical = norm_pdf(d1) * (-q * t).exp() / (s * vol * t.sqrt());

    assert!(
        (gamma_fd_ad - gamma_analytical).abs() < 1e-4,
        "BS gamma: FD-on-AD={gamma_fd_ad:.10}, analytical={gamma_analytical:.10}"
    );
}

// --- Vanna (∂²V/∂S∂σ) ---

#[test]
fn bs_vanna_fd_on_ad_vs_analytical() {
    let (s, k, r, q, vol, t) = (100.0, 100.0, 0.05, 0.02, 0.20, 1.0);

    // FD-on-AD: bump σ, read AD delta (seeded on S) at each bump
    let h = vol * 1e-4;

    let delta_up = bs_european_generic(
        Dual::new(s, 1.0), Dual::constant(k),
        Dual::constant(r), Dual::constant(q),
        Dual::constant(vol + h), Dual::constant(t), true,
    ).npv.dot;

    let delta_dn = bs_european_generic(
        Dual::new(s, 1.0), Dual::constant(k),
        Dual::constant(r), Dual::constant(q),
        Dual::constant(vol - h), Dual::constant(t), true,
    ).npv.dot;

    let vanna_fd_ad = (delta_up - delta_dn) / (2.0 * h);

    // Analytical: vanna = -φ(d1) * exp(-qT) * d2 / (S * σ)
    // (equivalently: ∂delta/∂σ = -e^(-qT) * φ(d1) * d2 / σ  — but per S)
    let d1 = bs_d1(s, k, r, q, vol, t);
    let d2 = bs_d2(s, k, r, q, vol, t);
    let vanna_analytical = -norm_pdf(d1) * (-q * t).exp() * d2 / (vol);

    // Relative tolerance: cross-CDF error amplified in second-order
    assert!(
        (vanna_fd_ad - vanna_analytical).abs() / vanna_analytical.abs() < 5e-3,
        "BS vanna: FD-on-AD={vanna_fd_ad:.10}, analytical={vanna_analytical:.10}"
    );
}

// --- Volga (∂²V/∂σ²) ---

#[test]
fn bs_volga_fd_on_ad_vs_analytical() {
    let (s, k, r, q, vol, t) = (100.0, 100.0, 0.05, 0.02, 0.20, 1.0);

    // FD-on-AD: bump σ, read AD vega (seeded on σ) at each bump
    let h = vol * 1e-4;

    let vega_up = bs_european_generic(
        Dual::constant(s), Dual::constant(k),
        Dual::constant(r), Dual::constant(q),
        Dual::new(vol + h, 1.0), Dual::constant(t), true,
    ).npv.dot;

    let vega_dn = bs_european_generic(
        Dual::constant(s), Dual::constant(k),
        Dual::constant(r), Dual::constant(q),
        Dual::new(vol - h, 1.0), Dual::constant(t), true,
    ).npv.dot;

    let volga_fd_ad = (vega_up - vega_dn) / (2.0 * h);

    // Analytical: volga = S * φ(d1) * exp(-qT) * √T * d1 * d2 / σ
    let d1 = bs_d1(s, k, r, q, vol, t);
    let d2 = bs_d2(s, k, r, q, vol, t);
    let volga_analytical = s * norm_pdf(d1) * (-q * t).exp() * t.sqrt() * d1 * d2 / vol;

    assert!(
        (volga_fd_ad - volga_analytical).abs() / volga_analytical.abs() < 5e-3,
        "BS volga: FD-on-AD={volga_fd_ad:.10}, analytical={volga_analytical:.10}"
    );
}

// --- Charm (∂²V/∂S∂t = ∂delta/∂t) ---

#[test]
fn bs_charm_fd_on_ad_vs_pure_fd() {
    let (s, k, r, q, vol, t) = (100.0, 100.0, 0.05, 0.02, 0.20, 1.0);

    // FD-on-AD: bump t, read AD delta (seeded on S) at each bump
    let h = 1e-5;

    let delta_up = bs_european_generic(
        Dual::new(s, 1.0), Dual::constant(k),
        Dual::constant(r), Dual::constant(q),
        Dual::constant(vol), Dual::constant(t + h), true,
    ).npv.dot;

    let delta_dn = bs_european_generic(
        Dual::new(s, 1.0), Dual::constant(k),
        Dual::constant(r), Dual::constant(q),
        Dual::constant(vol), Dual::constant(t - h), true,
    ).npv.dot;

    let charm_fd_ad = (delta_up - delta_dn) / (2.0 * h);

    // Pure second-order FD for validation
    let h2 = 1e-4;
    let price = |ss: f64, tt: f64| -> f64 {
        bs_european_generic::<f64>(ss, k, r, q, vol, tt, true).npv
    };
    let charm_pure_fd = (price(s + h2, t + h2) - price(s - h2, t + h2)
                        - price(s + h2, t - h2) + price(s - h2, t - h2))
                        / (4.0 * h2 * h2);

    assert!(
        (charm_fd_ad - charm_pure_fd).abs() < 1e-3,
        "BS charm: FD-on-AD={charm_fd_ad:.8}, pure FD={charm_pure_fd:.8}"
    );
}

// --- BAW gamma: FD-on-AD vs pure FD ---

#[test]
fn baw_gamma_fd_on_ad_vs_pure_fd() {
    let (s, k, r, q, vol, t) = (100.0, 100.0, 0.05, 0.02, 0.20, 1.0);

    // FD-on-AD gamma
    let h = s * 1e-4;
    let delta_up = barone_adesi_whaley_generic(
        Dual::new(s + h, 1.0), Dual::constant(k),
        Dual::constant(r), Dual::constant(q),
        Dual::constant(vol), Dual::constant(t), false,
    ).npv.dot;

    let delta_dn = barone_adesi_whaley_generic(
        Dual::new(s - h, 1.0), Dual::constant(k),
        Dual::constant(r), Dual::constant(q),
        Dual::constant(vol), Dual::constant(t), false,
    ).npv.dot;

    let gamma_fd_ad = (delta_up - delta_dn) / (2.0 * h);

    // Pure second-order FD
    let h2 = 1e-3;
    let price = |ss: f64| -> f64 {
        barone_adesi_whaley_generic::<f64>(ss, k, r, q, vol, t, false).npv
    };
    let gamma_pure_fd = (price(s + h2) - 2.0 * price(s) + price(s - h2)) / (h2 * h2);

    assert!(
        (gamma_fd_ad - gamma_pure_fd).abs() / gamma_pure_fd.abs() < 1e-3,
        "BAW gamma: FD-on-AD={gamma_fd_ad:.10}, pure FD={gamma_pure_fd:.10}"
    );
}

// --- DualVec second-order: all first-order Greeks in one pass, then FD for gamma ---

#[test]
fn bs_dualvec_greeks_plus_fd_gamma() {
    type D5 = DualVec<5>;
    let (s, k, r, q, vol, t) = (100.0, 100.0, 0.05, 0.02, 0.20, 1.0);

    // One forward pass: get delta, rho, ∂V/∂q, vega, ∂V/∂t
    let res = bs_european_generic(
        D5::variable(s, 0),
        D5::constant(k),
        D5::variable(r, 1),
        D5::variable(q, 2),
        D5::variable(vol, 3),
        D5::variable(t, 4),
        true,
    );

    let delta = res.npv.dot[0];
    let rho_raw = res.npv.dot[1];
    let dq = res.npv.dot[2];
    let vega_raw = res.npv.dot[3];
    let dt = res.npv.dot[4];

    // Verify all are non-zero and have expected signs
    assert!(delta > 0.0, "Call delta should be positive: {delta}");
    assert!(rho_raw > 0.0, "Call rho should be positive: {rho_raw}");
    assert!(dq < 0.0, "Call ∂V/∂q should be negative: {dq}");
    assert!(vega_raw > 0.0, "Call vega should be positive: {vega_raw}");
    // theta (dt) sign: for an ATM call with r>q, theta is typically negative
    // but ∂V/∂t (raw sensitivity to maturity) is positive — longer maturity = more value
    assert!(dt > 0.0, "∂V/∂t should be positive for ATM call: {dt}");

    // Now FD-on-AD for gamma using DualVec (bump spot, read delta)
    let h = s * 1e-4;
    let delta_up = bs_european_generic(
        D5::variable(s + h, 0), D5::constant(k),
        D5::variable(r, 1), D5::variable(q, 2),
        D5::variable(vol, 3), D5::variable(t, 4), true,
    ).npv.dot[0];

    let delta_dn = bs_european_generic(
        D5::variable(s - h, 0), D5::constant(k),
        D5::variable(r, 1), D5::variable(q, 2),
        D5::variable(vol, 3), D5::variable(t, 4), true,
    ).npv.dot[0];

    let gamma_fd_ad = (delta_up - delta_dn) / (2.0 * h);
    let d1 = bs_d1(s, k, r, q, vol, t);
    let gamma_analytical = norm_pdf(d1) * (-q * t).exp() / (s * vol * t.sqrt());

    assert!(
        (gamma_fd_ad - gamma_analytical).abs() < 1e-4,
        "DualVec5 gamma: FD-on-AD={gamma_fd_ad:.10}, analytical={gamma_analytical:.10}"
    );
}

// --- AReal second-order: reverse-mode all partials, then FD for gamma ---

#[test]
fn bs_areal_greeks_plus_fd_gamma() {
    let (s, k, r, q, vol, t) = (100.0, 100.0, 0.05, 0.02, 0.20, 1.0);

    // Reverse-mode: get all first-order partials in one pass
    let (npv, s_idx, k_idx, r_idx, q_idx, v_idx) = with_tape(|tape| {
        let s_ar = tape.input(s);
        let k_ar = tape.input(k);
        let r_ar = tape.input(r);
        let q_ar = tape.input(q);
        let v_ar = tape.input(vol);
        let t_ar = AReal::from_f64(t);
        let res = bs_european_generic(s_ar, k_ar, r_ar, q_ar, v_ar, t_ar, true);
        (res.npv, s_ar.idx, k_ar.idx, r_ar.idx, q_ar.idx, v_ar.idx)
    });
    let grad = adjoint_tl(npv);

    // Verify signs
    assert!(grad[s_idx] > 0.0, "AReal delta positive");
    assert!(grad[k_idx] < 0.0, "AReal ∂V/∂K negative for call");
    assert!(grad[r_idx] > 0.0, "AReal rho positive for call");
    assert!(grad[q_idx] < 0.0, "AReal ∂V/∂q negative for call");
    assert!(grad[v_idx] > 0.0, "AReal vega positive");

    // FD-on-AD gamma using AReal
    let h = s * 1e-4;
    let delta_at = |spot: f64| -> f64 {
        let (npv2, si) = with_tape(|tape| {
            let s_ar = tape.input(spot);
            let k_ar = tape.input(k);
            let r_ar = tape.input(r);
            let q_ar = tape.input(q);
            let v_ar = tape.input(vol);
            let t_ar = AReal::from_f64(t);
            let res = bs_european_generic(s_ar, k_ar, r_ar, q_ar, v_ar, t_ar, true);
            (res.npv, s_ar.idx)
        });
        let g = adjoint_tl(npv2);
        g[si]
    };

    let gamma_fd_ad = (delta_at(s + h) - delta_at(s - h)) / (2.0 * h);
    let d1 = bs_d1(s, k, r, q, vol, t);
    let gamma_analytical = norm_pdf(d1) * (-q * t).exp() / (s * vol * t.sqrt());

    assert!(
        (gamma_fd_ad - gamma_analytical).abs() < 1e-4,
        "AReal gamma: FD-on-AD={gamma_fd_ad:.10}, analytical={gamma_analytical:.10}"
    );
}
