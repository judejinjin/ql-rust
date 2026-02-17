//! End-to-end example: load market data → bootstrap curves → price portfolio → persist results.
//!
//! Run with:
//! ```sh
//! cargo run --example end_to_end
//! ```

use ql_instruments::VanillaOption;
use ql_pricingengines::{price_european, implied_volatility, price_bond};
use ql_termstructures::{
    DepositRateHelper, FlatForward, PiecewiseYieldCurve, RateHelper, SwapRateHelper,
};
use ql_time::{Date, DayCounter, Month, Schedule};
use ql_instruments::FixedRateBond;
use ql_persistence::{
    Direction, EmbeddedStore, InstrumentType, ObjectStore, Trade,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  QuantLib-Rust End-to-End Example");
    println!("═══════════════════════════════════════════════════════════\n");

    let today = Date::from_ymd(2025, Month::January, 15);
    let dc = DayCounter::Actual365Fixed;

    // ─── Step 1: Bootstrap a yield curve from market data ────────────
    println!("Step 1: Bootstrapping yield curve...");

    let mut helpers: Vec<Box<dyn RateHelper>> = vec![
        // Money-market deposits
        Box::new(DepositRateHelper::new(0.045, today, today + 90, dc)),   // 3M deposit at 4.5%
        Box::new(DepositRateHelper::new(0.046, today, today + 180, dc)),  // 6M deposit at 4.6%
        // Par swap rates
        Box::new(SwapRateHelper::new(
            0.048,
            today,
            vec![today + 365, today + 730],
            dc,
        )), // 2Y swap at 4.8%
        Box::new(SwapRateHelper::new(
            0.050,
            today,
            vec![
                today + 365,
                today + 730,
                today + 1095,
                today + 1461,
                today + 1826,
            ],
            dc,
        )), // 5Y swap at 5.0%
    ];

    let curve = PiecewiseYieldCurve::new(today, &mut helpers, dc, 1e-12)?;

    println!("  Bootstrapped {} pillar(s):", curve.size());
    for (t, df) in curve.nodes() {
        let zero = if t > 1e-10 { -(df.ln()) / t } else { 0.0 };
        println!("    t = {:.4}y  df = {:.8}  zero = {:.4}%", t, df, zero * 100.0);
    }
    println!();

    // ─── Step 2: Price a European call option ────────────────────────
    println!("Step 2: Pricing European call option (Black-Scholes)...");

    let spot = 100.0;
    let strike = 105.0;
    let vol = 0.20;
    let rate = 0.05;
    let dividend = 0.02;
    let expiry = 1.0; // 1 year

    let call = VanillaOption::european_call(strike, today + 365);
    let result = price_european(&call, spot, rate, dividend, vol, expiry);

    println!("  Spot     = {:.2}", spot);
    println!("  Strike   = {:.2}", strike);
    println!("  Vol      = {:.1}%", vol * 100.0);
    println!("  Rate     = {:.1}%", rate * 100.0);
    println!("  Dividend = {:.1}%", dividend * 100.0);
    println!("  Expiry   = {:.1}Y", expiry);
    println!("  ─────────────────────────");
    println!("  NPV      = {:.6}", result.npv);
    println!("  Delta    = {:.6}", result.delta);
    println!("  Gamma    = {:.6}", result.gamma);
    println!("  Vega     = {:.6}", result.vega);
    println!("  Theta    = {:.6}", result.theta);
    println!("  Rho      = {:.6}", result.rho);
    println!();

    // ─── Step 3: Compute implied volatility ──────────────────────────
    println!("Step 3: Computing implied volatility...");

    let target_price = result.npv;
    let iv = implied_volatility(&call, target_price, spot, rate, dividend, expiry)?;
    println!("  Target price = {:.6}", target_price);
    println!("  Implied vol  = {:.6} ({:.2}%)", iv, iv * 100.0);
    println!();

    // ─── Step 4: Price a fixed-rate bond ─────────────────────────────
    println!("Step 4: Pricing a fixed-rate bond...");

    let bond_schedule = Schedule::from_dates(vec![
        Date::from_ymd(2025, Month::January, 15),
        Date::from_ymd(2025, Month::July, 15),
        Date::from_ymd(2026, Month::January, 15),
        Date::from_ymd(2026, Month::July, 15),
        Date::from_ymd(2027, Month::January, 15),
    ]);
    let bond = FixedRateBond::new(100.0, 2, &bond_schedule, 0.05, dc);
    let flat_curve = FlatForward::new(today, 0.045, dc);
    let bond_result = price_bond(&bond, &flat_curve, today);

    println!("  Face amount  = 100.00");
    println!("  Coupon       = 5.0% semi-annual");
    println!("  Discount     = 4.5% (flat)");
    println!("  ─────────────────────────");
    println!("  NPV          = {:.6}", bond_result.npv);
    println!("  Clean price  = {:.6}", bond_result.clean_price);
    println!("  Dirty price  = {:.6}", bond_result.dirty_price);
    println!("  Accrued      = {:.6}", bond_result.accrued_interest);
    println!();

    // ─── Step 5: Persist the option trade ────────────────────────────
    println!("Step 5: Persisting trade to embedded store...");

    let db_path = "/tmp/ql_rust_e2e_example.redb";
    let store = EmbeddedStore::open(db_path)?;

    let trade = Trade::new(
        InstrumentType::Option,
        serde_json::json!({
            "type": "european_call",
            "spot": spot,
            "strike": strike,
            "vol": vol,
            "rate": rate,
            "expiry_years": expiry,
            "npv": result.npv,
            "delta": result.delta,
        }),
        "ACME Corp",
        "equity-derivatives",
        1_000_000.0,
        Direction::Buy,
        "2025-01-15",
        "2025-01-17",
        "example-user",
    );

    let version = store.put_trade(&trade, "example-user")?;
    println!("  Trade ID : {}", trade.trade_id.as_str());
    println!("  Version  : {}", version);

    // Retrieve it back
    let loaded = store.get_trade(&trade.trade_id)?;
    println!("  Loaded   : {:?} ({:?})", loaded.instrument_type, loaded.status);
    println!();

    println!("═══════════════════════════════════════════════════════════");
    println!("  All steps completed successfully!");
    println!("═══════════════════════════════════════════════════════════");

    // Clean up
    std::fs::remove_file(db_path).ok();

    Ok(())
}
