//! # JSON Serialization Example
//!
//! Demonstrates round-trip serialization of instruments, term structures,
//! and pricing results through `serde_json`.
//!
//! Run with:
//! ```sh
//! cargo run --example serde_round_trip
//! ```

use ql_rust::*;
use ql_time::schedule::DateGenerationRule;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ─── 1. VanillaOption ────────────────────────────────────────────────
    println!("═══ VanillaOption ═══");
    let option = VanillaOption::european_call(100.0, Date::from_ymd(2026, Month::June, 15));
    let json = serde_json::to_string_pretty(&option)?;
    println!("{json}");
    let deser: VanillaOption = serde_json::from_str(&json)?;
    assert_eq!(deser.strike(), option.strike());
    println!("✓ round-trip OK\n");

    // ─── 2. BarrierOption ────────────────────────────────────────────────
    println!("═══ BarrierOption ═══");
    let barrier = BarrierOption::new(
        BarrierType::DownOut,
        80.0,
        0.0,
        Payoff::PlainVanilla {
            option_type: OptionType::Call,
            strike: 100.0,
        },
        Exercise::European {
            expiry: Date::from_ymd(2026, Month::June, 15),
        },
    );
    let json = serde_json::to_string_pretty(&barrier)?;
    println!("{json}");
    let _: BarrierOption = serde_json::from_str(&json)?;
    println!("✓ round-trip OK\n");

    // ─── 3. FlatForward ──────────────────────────────────────────────────
    println!("═══ FlatForward ═══");
    let curve = FlatForward::new(
        Date::from_ymd(2025, Month::January, 15),
        0.05,
        DayCounter::Actual365Fixed,
    );
    let json = serde_json::to_string_pretty(&curve)?;
    println!("{json}");
    let deser: FlatForward = serde_json::from_str(&json)?;
    assert!((deser.rate() - 0.05).abs() < 1e-12);
    println!("✓ round-trip OK\n");

    // ─── 4. NelsonSiegelFitting ──────────────────────────────────────────
    println!("═══ NelsonSiegelFitting ═══");
    let ns = NelsonSiegelFitting::new(0.05, -0.02, 0.01, 1.5);
    let json = serde_json::to_string_pretty(&ns)?;
    println!("{json}");
    let deser: NelsonSiegelFitting = serde_json::from_str(&json)?;
    assert_eq!(deser.params, ns.params);
    println!("✓ round-trip OK\n");

    // ─── 5. Schedule ─────────────────────────────────────────────────────
    println!("═══ Schedule ═══");
    let sched = Schedule::builder()
        .effective_date(Date::from_ymd(2025, Month::January, 15))
        .termination_date(Date::from_ymd(2030, Month::January, 15))
        .frequency(Frequency::Semiannual)
        .calendar(Calendar::Target)
        .convention(BusinessDayConvention::ModifiedFollowing)
        .termination_convention(BusinessDayConvention::ModifiedFollowing)
        .rule(DateGenerationRule::Forward)
        .end_of_month(false)
        .build()?;
    let json = serde_json::to_string_pretty(&sched)?;
    println!("Schedule dates: {} (JSON length: {} bytes)", sched.len(), json.len());
    let deser: Schedule = serde_json::from_str(&json)?;
    assert_eq!(deser.len(), sched.len());
    println!("✓ round-trip OK\n");

    // ─── 6. Pricing results ─────────────────────────────────────────────
    println!("═══ AnalyticEuropeanResults ═══");
    let opt = VanillaOption::european_call(100.0, Date::from_ymd(2026, Month::June, 15));
    let result = price_european(&opt, 100.0, 0.05, 0.0, 0.20, 1.0);
    let json = serde_json::to_string_pretty(&result)?;
    println!("{json}");
    let deser: AnalyticEuropeanResults = serde_json::from_str(&json)?;
    assert!((deser.npv - result.npv).abs() < 1e-12);
    println!("✓ round-trip OK\n");

    // ─── 7. CreditDefaultSwap ───────────────────────────────────────────
    println!("═══ CreditDefaultSwap ═══");
    let cds = CreditDefaultSwap::new(
        CdsProtectionSide::Buyer,
        1_000_000.0,
        0.01, // 100bp spread
        Date::from_ymd(2030, Month::March, 20),
        0.40, // recovery rate
        vec![CdsPremiumPeriod {
            accrual_start: Date::from_ymd(2025, Month::March, 20),
            accrual_end: Date::from_ymd(2025, Month::June, 20),
            payment_date: Date::from_ymd(2025, Month::June, 20),
            accrual_fraction: 0.25,
        }],
    );
    let json = serde_json::to_string_pretty(&cds)?;
    println!("CDS JSON length: {} bytes", json.len());
    let _: CreditDefaultSwap = serde_json::from_str(&json)?;
    println!("✓ round-trip OK\n");

    // ─── 8. Date ─────────────────────────────────────────────────────────
    println!("═══ Date ═══");
    let d = Date::from_ymd(2025, Month::June, 15);
    let json = serde_json::to_string(&d)?;
    println!("Date JSON: {json}");
    let deser: Date = serde_json::from_str(&json)?;
    assert_eq!(deser, d);
    println!("✓ round-trip OK\n");

    // ─── Summary ─────────────────────────────────────────────────────────
    println!("All round-trip tests passed!");
    println!("\nql-rust supports serde Serialize + Deserialize on ~190 types ");
    println!("across 13 crates: instruments, term structures, pricing results,");
    println!("models, processes, dates, schedules, and more.");
    println!("\nUse serde_json, serde_yaml, rmp-serde, or any serde backend.");

    Ok(())
}
