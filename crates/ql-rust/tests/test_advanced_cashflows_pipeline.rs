//! Integration tests: Advanced cash flow analytics pipeline.

use ql_cashflows::{
    convexity, dv01, fixed_leg, modified_duration, z_spread,
    time_bucketed_cashflows,
};
use ql_termstructures::FlatForward;
use ql_time::{Date, DayCounter, Month, Schedule};

fn make_leg() -> (ql_cashflows::Leg, FlatForward) {
    let today = Date::from_ymd(2025, Month::January, 15);
    let dc = DayCounter::Actual365Fixed;
    let schedule = Schedule::from_dates((0..=10).map(|y| today + y * 365).collect());
    let notionals = [1_000_000.0; 10];
    let rates = [0.05; 10];
    let leg = fixed_leg(&schedule, &notionals, &rates, dc);
    let curve = FlatForward::new(today, 0.05, dc);
    (leg, curve)
}

/// Convexity should be positive for a standard bond.
#[test]
fn convexity_positive() {
    let (leg, curve) = make_leg();
    let today = Date::from_ymd(2025, Month::January, 15);
    let cx = convexity(&leg, &curve, today);
    assert!(cx > 0.0, "Convexity should be positive, got {:.4}", cx);
}

/// DV01 should be positive for a standard fixed leg.
#[test]
fn dv01_positive() {
    let (leg, curve) = make_leg();
    let today = Date::from_ymd(2025, Month::January, 15);
    let dv = dv01(&leg, &curve, today);
    assert!(dv > 0.0, "DV01 should be positive, got {:.4}", dv);
}

/// Modified duration should be positive.
#[test]
fn modified_duration_positive() {
    let (leg, curve) = make_leg();
    let today = Date::from_ymd(2025, Month::January, 15);
    let md = modified_duration(&leg, &curve, today, 0.05, 1);
    assert!(md > 0.0, "Modified duration should be positive, got {:.4}", md);
}

/// Z-spread round-trip: pricing with z-spread should reproduce target price.
#[test]
fn z_spread_round_trip() {
    let today = Date::from_ymd(2025, Month::January, 15);
    let dc = DayCounter::Actual365Fixed;
    let schedule = Schedule::from_dates((0..=5).map(|y| today + y * 365).collect());
    let notionals = [100.0; 5];
    let rates = [0.06; 5];
    let leg = fixed_leg(&schedule, &notionals, &rates, dc);

    // Price at 5% flat curve
    let curve_5pct = FlatForward::new(today, 0.05, dc);
    let target_price = ql_cashflows::npv(&leg, &curve_5pct, today);

    // Use 4% curve and find z-spread to hit target_price
    let curve_4pct = FlatForward::new(today, 0.04, dc);
    let zs = z_spread(&leg, &curve_4pct, today, target_price, 1e-8, 200).unwrap();

    // z-spread should be approximately 1% (5% - 4%)
    assert!((zs - 0.01).abs() < 0.002, "Z-spread {:.4} should be ~0.01", zs);
}

/// Time-bucketed cash flows should sum to total NPV.
#[test]
fn time_buckets_sum_to_npv() {
    let (leg, curve) = make_leg();
    let today = Date::from_ymd(2025, Month::January, 15);

    let boundaries: Vec<f64> = (0..=10).map(|y| y as f64 + 0.5).collect();
    let buckets = time_bucketed_cashflows(&leg, &curve, today, &boundaries);
    let bucket_sum: f64 = buckets.iter().map(|b| b.pv).sum();
    let total_pv = ql_cashflows::npv(&leg, &curve, today);

    let err = (bucket_sum - total_pv).abs() / total_pv.abs();
    assert!(err < 0.01, "Bucket sum {:.2} != NPV {:.2}", bucket_sum, total_pv);
}
