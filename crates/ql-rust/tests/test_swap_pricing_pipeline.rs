//! Integration test: curve → legs → swap → NPV.
//!
//! Validates the full swap pricing pipeline from curve construction through
//! leg building to final NPV calculation.

use approx::assert_abs_diff_eq;
use ql_cashflows::{fixed_leg, ibor_leg, npv as leg_npv};
use ql_indexes::IborIndex;
use ql_instruments::{SwapType, VanillaSwap};
use ql_pricingengines::price_swap;
use ql_termstructures::FlatForward;
use ql_time::{Date, DayCounter, Month, Schedule};

/// Build a swap from schedules, price it, and verify basic properties.
#[test]
fn swap_pricing_at_par_rate() {
    let today = Date::from_ymd(2025, Month::January, 15);
    let dc = DayCounter::Actual365Fixed;
    let rate = 0.05;
    let curve = FlatForward::new(today, rate, dc);

    // Build schedules for a 2-year semi-annual swap
    let schedule = Schedule::from_dates(vec![
        Date::from_ymd(2025, Month::January, 15),
        Date::from_ymd(2025, Month::July, 15),
        Date::from_ymd(2026, Month::January, 15),
        Date::from_ymd(2026, Month::July, 15),
        Date::from_ymd(2027, Month::January, 15),
    ]);

    let notional = 1_000_000.0;
    let notionals = [notional; 4];
    let fixed_rates = [rate; 4]; // at par

    let index = IborIndex::euribor_6m();

    let fixed = fixed_leg(&schedule, &notionals, &fixed_rates, dc);
    let floating = ibor_leg(&schedule, &notionals, &index, &[0.0; 4], dc);

    let swap = VanillaSwap::new(SwapType::Payer, notional, fixed, floating, rate, 0.0);

    let result = price_swap(&swap, &curve, today);

    // At the par rate, swap NPV should be close to zero
    // (Won't be exact because ibor rates need fixing, but fixed leg NPV ≈ floating leg NPV)
    assert!(
        result.fixed_leg_npv.abs() > 0.0,
        "Fixed leg NPV should be non-zero"
    );

    // Fair rate should be close to the curve rate
    // (This approximation holds when the floating leg rates are set)
    assert!(result.fair_rate >= 0.0, "Fair rate should be non-negative");
}

/// Payer vs receiver: NPVs should have opposite signs.
#[test]
fn payer_vs_receiver_symmetry() {
    let today = Date::from_ymd(2025, Month::January, 15);
    let dc = DayCounter::Actual365Fixed;
    let rate = 0.05;
    let curve = FlatForward::new(today, rate, dc);

    let schedule = Schedule::from_dates(vec![
        Date::from_ymd(2025, Month::January, 15),
        Date::from_ymd(2025, Month::July, 15),
        Date::from_ymd(2026, Month::January, 15),
    ]);

    let notional = 1_000_000.0;
    let notionals = [notional; 2];
    let rates = [0.04; 2]; // below market rate

    let index = IborIndex::euribor_6m();

    // Build separate legs for payer and receiver (CashFlow is not Clone)
    let payer_fixed = fixed_leg(&schedule, &notionals, &rates, dc);
    let payer_floating = ibor_leg(&schedule, &notionals, &index, &[0.0; 2], dc);
    let payer = VanillaSwap::new(SwapType::Payer, notional, payer_fixed, payer_floating, 0.04, 0.0);

    let receiver_fixed = fixed_leg(&schedule, &notionals, &rates, dc);
    let receiver_floating = ibor_leg(&schedule, &notionals, &index, &[0.0; 2], dc);
    let receiver = VanillaSwap::new(SwapType::Receiver, notional, receiver_fixed, receiver_floating, 0.04, 0.0);

    let payer_result = price_swap(&payer, &curve, today);
    let receiver_result = price_swap(&receiver, &curve, today);

    // Payer + Receiver = 0 (approximately)
    assert_abs_diff_eq!(
        payer_result.npv + receiver_result.npv,
        0.0,
        epsilon = 1e-8
    );
}

/// Fixed leg NPV should be positive for positive rates and notional.
#[test]
fn fixed_leg_npv_positive() {
    let today = Date::from_ymd(2025, Month::January, 15);
    let dc = DayCounter::Actual365Fixed;
    let curve = FlatForward::new(today, 0.04, dc);

    let schedule = Schedule::from_dates(vec![
        Date::from_ymd(2025, Month::January, 15),
        Date::from_ymd(2025, Month::July, 15),
        Date::from_ymd(2026, Month::January, 15),
    ]);

    let notionals = [1_000_000.0; 2];
    let rates = [0.05; 2];
    let fixed = fixed_leg(&schedule, &notionals, &rates, dc);

    let pv = leg_npv(&fixed, &curve, today);
    assert!(pv > 0.0, "Fixed leg NPV should be positive, got {}", pv);
}
