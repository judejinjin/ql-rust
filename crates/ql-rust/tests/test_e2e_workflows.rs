//! End-to-end integration tests: full workflows from market data through
//! pricing and risk.
//!
//! These tests exercise the complete ql-rust stack:
//! 1. Curve bootstrap from deposit/swap quotes
//! 2. Swap pricing (single & multi-curve)
//! 3. Bond pricing
//! 4. Bump-and-reprice delta / DV01
//! 5. Option pricing with engine adapters
//! 6. Implied vol round-trip
//! 7. Calendar → schedule → cashflow → NPV pipeline

use approx::assert_abs_diff_eq;
use ql_cashflows::{fixed_leg, ibor_leg, npv as leg_npv};
use ql_core::engine::PricingEngine;
use ql_indexes::IborIndex;
use ql_instruments::{FixedRateBond, SwapType, VanillaOption, VanillaSwap};
use ql_pricingengines::{
    engine_adapters::{AnalyticEuropeanEngine, BinomialCRREngine, MCEuropeanEngine},
    implied_volatility, price_bond, price_european, price_swap, price_swap_multicurve,
};
use ql_termstructures::{FlatForward, PiecewiseYieldCurve, DepositRateHelper, SwapRateHelper, YieldTermStructure};
use ql_time::{
    BusinessDayConvention, Calendar, Date, DayCounter, Month, Period, Schedule,
};

// ═══════════════════════════════════════════════════════════════════
// 1. Curve bootstrap → swap pricing → DV01
// ═══════════════════════════════════════════════════════════════════

/// Bootstrap a yield curve from market quotes, price a swap, bump the
/// curve by 1bp, reprice, and verify DV01 is reasonable.
#[test]
fn e2e_bootstrap_swap_dv01() {
    let today = Date::from_ymd(2025, Month::January, 15);
    let dc = DayCounter::Actual365Fixed;

    // --- Market quotes (flat at 3%) ---
    let mut helpers: Vec<Box<dyn ql_termstructures::RateHelper>> = vec![
        Box::new(DepositRateHelper::new(0.03, today, today + 90, dc)),
        Box::new(DepositRateHelper::new(0.03, today, today + 182, dc)),
        Box::new(SwapRateHelper::from_tenor(0.03, today, 1, 2, dc, Calendar::Target)),
        Box::new(SwapRateHelper::from_tenor(0.03, today, 2, 2, dc, Calendar::Target)),
        Box::new(SwapRateHelper::from_tenor(0.03, today, 5, 2, dc, Calendar::Target)),
    ];

    let curve = PiecewiseYieldCurve::new(today, &mut helpers, dc, 1e-12)
        .expect("bootstrap should succeed");

    // Verify: discount at 1y ≈ e^{-0.03}
    let df_1y = curve.discount(today + 365);
    assert!((df_1y - (-0.03_f64).exp()).abs() < 0.005,
        "1y discount ≈ e^(-0.03), got {df_1y}");

    // --- Build a 2y par swap at 3% ---
    let schedule = Schedule::from_dates(vec![
        Date::from_ymd(2025, Month::January, 15),
        Date::from_ymd(2025, Month::July, 15),
        Date::from_ymd(2026, Month::January, 15),
        Date::from_ymd(2026, Month::July, 15),
        Date::from_ymd(2027, Month::January, 15),
    ]);

    let notional = 10_000_000.0;
    let n = &[notional; 4];
    let fixed_rates = &[0.03; 4];
    let index = IborIndex::euribor_6m();
    let fixed = fixed_leg(&schedule, n, fixed_rates, dc);
    let floating = ibor_leg(&schedule, n, &index, &[0.0; 4], dc);
    let swap = VanillaSwap::new(SwapType::Payer, notional, fixed, floating, 0.03, 0.0);

    let base_result = price_swap(&swap, &curve, today);

    // --- Bump curve by +1bp, reprice ---
    let mut helpers_up: Vec<Box<dyn ql_termstructures::RateHelper>> = vec![
        Box::new(DepositRateHelper::new(0.0301, today, today + 90, dc)),
        Box::new(DepositRateHelper::new(0.0301, today, today + 182, dc)),
        Box::new(SwapRateHelper::from_tenor(0.0301, today, 1, 2, dc, Calendar::Target)),
        Box::new(SwapRateHelper::from_tenor(0.0301, today, 2, 2, dc, Calendar::Target)),
        Box::new(SwapRateHelper::from_tenor(0.0301, today, 5, 2, dc, Calendar::Target)),
    ];
    let curve_up = PiecewiseYieldCurve::new(today, &mut helpers_up, dc, 1e-12).unwrap();

    // Rebuild legs (CashFlow is not Clone)
    let fixed_up = fixed_leg(&schedule, n, fixed_rates, dc);
    let floating_up = ibor_leg(&schedule, n, &index, &[0.0; 4], dc);
    let swap_up = VanillaSwap::new(SwapType::Payer, notional, fixed_up, floating_up, 0.03, 0.0);

    let bumped_result = price_swap(&swap_up, &curve_up, today);

    // DV01 = change in NPV per 1bp parallel shift
    let dv01 = bumped_result.npv - base_result.npv;
    // For a 2y 10MM swap, DV01 should be in the tens to low thousands
    // (depends on whether floating rates are set from fixing)
    assert!(dv01.abs() > 10.0, "DV01 too small: {dv01}");
    assert!(dv01.abs() < 5000.0, "DV01 too large: {dv01}");
}

// ═══════════════════════════════════════════════════════════════════
// 2. Bond pricing from bootstrapped curve
// ═══════════════════════════════════════════════════════════════════

/// Build a yield curve, price a fixed-rate bond, verify clean price
/// is near par when coupon ≈ yield.
#[test]
fn e2e_curve_to_bond_pricing() {
    let today = Date::from_ymd(2025, Month::January, 15);
    let dc = DayCounter::Actual365Fixed;
    let curve = FlatForward::new(today, 0.04, dc);

    // 5-year annual coupon bond at 4% (≈ par)
    let schedule = Schedule::from_dates(vec![
        Date::from_ymd(2025, Month::January, 15),
        Date::from_ymd(2026, Month::January, 15),
        Date::from_ymd(2027, Month::January, 15),
        Date::from_ymd(2028, Month::January, 15),
        Date::from_ymd(2029, Month::January, 15),
        Date::from_ymd(2030, Month::January, 15),
    ]);

    let face = 100.0;
    let bond = FixedRateBond::new(
        face,
        2,              // settlement days
        &schedule,
        0.04,           // coupon rate
        dc,
    );

    let result = price_bond(&bond, &curve, today);

    // At par yield, clean price ≈ 100
    assert!((result.clean_price - 100.0).abs() < 2.0,
        "near-par bond clean price ≈ 100, got {}", result.clean_price);
    assert!(result.dirty_price >= result.clean_price - 1.0,
        "dirty ≥ clean - small accrued");
}

// ═══════════════════════════════════════════════════════════════════
// 3. Multi-curve swap pricing
// ═══════════════════════════════════════════════════════════════════

/// Price a swap with separate OIS discount and IBOR forecast curves,
/// verify that the result differs from single-curve pricing when
/// curves diverge.
#[test]
fn e2e_multicurve_swap() {
    let today = Date::from_ymd(2025, Month::January, 15);
    let dc = DayCounter::Actual365Fixed;

    let ois_curve = FlatForward::new(today, 0.03, dc);   // OIS at 3%
    let ibor_curve = FlatForward::new(today, 0.035, dc);  // IBOR at 3.5% (+50bp spread)

    let schedule = Schedule::from_dates(vec![
        Date::from_ymd(2025, Month::January, 15),
        Date::from_ymd(2025, Month::July, 15),
        Date::from_ymd(2026, Month::January, 15),
    ]);

    let notional = 1_000_000.0;
    let n = &[notional; 2];
    let index = IborIndex::euribor_6m();

    // Single-curve at 3%
    let fixed_sc = fixed_leg(&schedule, n, &[0.035; 2], dc);
    let float_sc = ibor_leg(&schedule, n, &index, &[0.0; 2], dc);
    let swap_sc = VanillaSwap::new(SwapType::Payer, notional, fixed_sc, float_sc, 0.035, 0.0);
    let sc_result = price_swap(&swap_sc, &ois_curve, today);

    // Multi-curve: forecast from IBOR, discount with OIS
    let fixed_mc = fixed_leg(&schedule, n, &[0.035; 2], dc);
    let float_mc = ibor_leg(&schedule, n, &index, &[0.0; 2], dc);
    let swap_mc = VanillaSwap::new(SwapType::Payer, notional, fixed_mc, float_mc, 0.035, 0.0);
    let mc_result = price_swap_multicurve(&swap_mc, &ibor_curve, &ois_curve, today);

    // The two approaches should give different results when curves differ
    // Multi-curve should show different floating leg NPV because projection
    // comes from a different curve
    // (The difference is the basis between IBOR and OIS)
    assert!(
        (mc_result.floating_leg_npv - sc_result.floating_leg_npv).abs() > 1.0
        || (mc_result.npv - sc_result.npv).abs() > 0.01,
        "multi-curve should differ from single-curve when there's a basis"
    );
}

// ═══════════════════════════════════════════════════════════════════
// 4. Option pricing: engine adapters + convergence
// ═══════════════════════════════════════════════════════════════════

/// Price a European call with three different engines (BS, MC, CRR)
/// and verify they all agree within tolerance.
#[test]
fn e2e_option_three_engines_agree() {
    let expiry = Date::from_ymd(2026, Month::January, 15);
    let option = VanillaOption::european_call(100.0, expiry);

    let (s, r, q, vol, t) = (100.0, 0.05, 0.0, 0.20, 1.0);

    // Analytic BS
    let bs = AnalyticEuropeanEngine {
        spot: s, risk_free_rate: r, dividend_yield: q,
        volatility: vol, time_to_expiry: t,
    };
    let bs_npv = bs.calculate(&option).unwrap().npv;

    // Monte Carlo
    let mc = MCEuropeanEngine {
        spot: s, risk_free_rate: r, dividend_yield: q,
        volatility: vol, time_to_expiry: t,
        num_paths: 500_000, antithetic: true, seed: 12345,
    };
    let mc_npv = mc.calculate(&option).unwrap().npv;

    // Binomial CRR
    let crr = BinomialCRREngine {
        spot: s, risk_free_rate: r, dividend_yield: q,
        volatility: vol, time_to_expiry: t,
        num_steps: 1000,
    };
    let crr_npv = crr.calculate(&option).unwrap().npv;

    // All should be close to BS ≈ 10.4506
    assert!((bs_npv - 10.45).abs() < 0.1, "BS: {bs_npv}");
    assert!((mc_npv - bs_npv).abs() < 0.3, "MC={mc_npv} vs BS={bs_npv}");
    assert!((crr_npv - bs_npv).abs() < 0.1, "CRR={crr_npv} vs BS={bs_npv}");
}

// ═══════════════════════════════════════════════════════════════════
// 5. Implied vol round-trip
// ═══════════════════════════════════════════════════════════════════

/// Price with known vol, extract implied vol, verify round-trip.
#[test]
fn e2e_implied_vol_round_trip() {
    let expiry = Date::from_ymd(2026, Month::January, 15);
    let call = VanillaOption::european_call(100.0, expiry);
    let put = VanillaOption::european_put(100.0, expiry);

    let true_vol = 0.25;
    let (s, r, q, t) = (100.0, 0.05, 0.02, 1.0);

    let call_price = price_european(&call, s, r, q, true_vol, t).npv;
    let put_price = price_european(&put, s, r, q, true_vol, t).npv;

    let call_iv = implied_volatility(&call, call_price, s, r, q, t).unwrap();
    let put_iv = implied_volatility(&put, put_price, s, r, q, t).unwrap();

    assert!((call_iv - true_vol).abs() < 1e-8, "call IV: {call_iv}");
    assert!((put_iv - true_vol).abs() < 1e-8, "put IV: {put_iv}");
}

// ═══════════════════════════════════════════════════════════════════
// 6. Put-call parity
// ═══════════════════════════════════════════════════════════════════

/// Verify put-call parity: C - P = S·e^(-qT) - K·e^(-rT)
#[test]
fn e2e_put_call_parity() {
    let expiry = Date::from_ymd(2026, Month::January, 15);
    let call = VanillaOption::european_call(105.0, expiry);
    let put = VanillaOption::european_put(105.0, expiry);

    let (s, r, q, vol, t) = (100.0, 0.05, 0.02, 0.30, 1.0);
    let c = price_european(&call, s, r, q, vol, t).npv;
    let p = price_european(&put, s, r, q, vol, t).npv;

    let parity_rhs = s * (-q * t).exp() - 105.0 * (-r * t).exp();
    assert_abs_diff_eq!(c - p, parity_rhs, epsilon = 1e-8);
}

// ═══════════════════════════════════════════════════════════════════
// 7. Calendar → Schedule → Cashflows → NPV pipeline
// ═══════════════════════════════════════════════════════════════════

/// Build a schedule using a real calendar, create cashflows, compute NPV.
#[test]
fn e2e_calendar_schedule_cashflow_npv() {
    let today = Date::from_ymd(2025, Month::January, 15);
    let dc = DayCounter::Actual365Fixed;
    let curve = FlatForward::new(today, 0.04, dc);

    // Build a quarterly schedule for 1 year using TARGET calendar
    let cal = Calendar::Target;
    let start = today;
    let _end = cal.advance(today, Period::years(1), BusinessDayConvention::ModifiedFollowing, false);

    let mut dates = vec![start];
    let mut d = start;
    for _ in 0..4 {
        d = cal.advance(d, Period::months(3), BusinessDayConvention::ModifiedFollowing, false);
        dates.push(d);
    }
    let schedule = Schedule::from_dates(dates);

    let notional = 1_000_000.0;
    let notionals = [notional; 4];
    let rates = [0.04; 4];
    let cashflows = fixed_leg(&schedule, &notionals, &rates, dc);

    let npv = leg_npv(&cashflows, &curve, today);
    // 4 quarterly coupons of ~$10k each, discounted at 4% → NPV ≈ $38-40k
    assert!(npv > 30_000.0 && npv < 50_000.0, "NPV = {npv}");
}

// ═══════════════════════════════════════════════════════════════════
// 8. Greeks consistency (delta, gamma, vega)
// ═══════════════════════════════════════════════════════════════════

/// Verify numerical Greeks match analytic Greeks from BS engine.
#[test]
fn e2e_greeks_numerical_vs_analytic() {
    let expiry = Date::from_ymd(2026, Month::January, 15);
    let option = VanillaOption::european_call(100.0, expiry);

    let (s, r, q, vol, t) = (100.0, 0.05, 0.0, 0.20, 1.0);
    let ds = 0.01;
    let dvol = 0.0001;

    let res = price_european(&option, s, r, q, vol, t);
    let res_up = price_european(&option, s + ds, r, q, vol, t);
    let res_dn = price_european(&option, s - ds, r, q, vol, t);

    // Numerical delta
    let num_delta = (res_up.npv - res_dn.npv) / (2.0 * ds);
    assert!((num_delta - res.delta).abs() < 1e-4,
        "analytic delta={}, numerical={num_delta}", res.delta);

    // Numerical gamma
    let num_gamma = (res_up.npv - 2.0 * res.npv + res_dn.npv) / (ds * ds);
    assert!((num_gamma - res.gamma).abs() < 1e-3,
        "analytic gamma={}, numerical={num_gamma}", res.gamma);

    // Numerical vega (per 1% = 0.01 vol shift)
    let res_vol_up = price_european(&option, s, r, q, vol + dvol, t);
    let res_vol_dn = price_european(&option, s, r, q, vol - dvol, t);
    let num_vega = (res_vol_up.npv - res_vol_dn.npv) / (2.0 * dvol) * 0.01;
    assert!((num_vega - res.vega).abs() < 1e-4,
        "analytic vega={}, numerical={num_vega}", res.vega);
}
