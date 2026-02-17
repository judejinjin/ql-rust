//! Bootstrap a yield curve from deposit and swap rates, then query it.
//!
//! Run with:
//! ```sh
//! cargo run -p ql-rust --example bootstrap_yield_curve
//! ```

use ql_termstructures::{
    DepositRateHelper, FlatForward, PiecewiseYieldCurve, RateHelper, SwapRateHelper,
    YieldTermStructure,
};
use ql_time::{Date, DayCounter, Month};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let today = Date::from_ymd(2025, Month::January, 15);
    let dc = DayCounter::Actual365Fixed;

    println!("Yield Curve Bootstrapping");
    println!("═════════════════════════\n");

    // ── Market data (deposits + par swaps) ───────────────────────────
    println!("Market instruments:");
    println!("  3M deposit  @ 4.50%");
    println!("  6M deposit  @ 4.60%");
    println!("  2Y swap     @ 4.80%");
    println!("  5Y swap     @ 5.00%");
    println!("  10Y swap    @ 5.20%\n");

    let mut helpers: Vec<Box<dyn RateHelper>> = vec![
        Box::new(DepositRateHelper::new(0.045, today, today + 91, dc)),
        Box::new(DepositRateHelper::new(0.046, today, today + 182, dc)),
        Box::new(SwapRateHelper::new(
            0.048,
            today,
            vec![today + 365, today + 730],
            dc,
        )),
        Box::new(SwapRateHelper::new(
            0.050,
            today,
            (1..=5).map(|y| today + y * 365).collect(),
            dc,
        )),
        Box::new(SwapRateHelper::new(
            0.052,
            today,
            (1..=10).map(|y| today + y * 365).collect(),
            dc,
        )),
    ];

    // ── Bootstrap ────────────────────────────────────────────────────
    let curve = PiecewiseYieldCurve::new(today, &mut helpers, dc, 1e-12)?;

    println!("Bootstrapped curve ({} pillars):", curve.size());
    println!("  {:>8}  {:>12}  {:>10}", "Tenor", "Discount", "Zero Rate");
    println!("  {:>8}  {:>12}  {:>10}", "─────", "────────", "─────────");

    for (t, df) in curve.nodes() {
        let zero = if t > 1e-10 { -df.ln() / t } else { 0.0 };
        println!("  {:>7.2}y  {:>12.8}  {:>9.4}%", t, df, zero * 100.0);
    }
    println!();

    // ── Query at arbitrary tenors ────────────────────────────────────
    println!("Interpolated values:");
    for &t in &[0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0] {
        let df = curve.discount_t(t);
        let zero = if t > 0.0 { -df.ln() / t } else { 0.0 };
        println!("  t = {:>5.2}y   df = {:.8}   zero = {:.4}%", t, df, zero * 100.0);
    }
    println!();

    // ── Compare with flat curve ──────────────────────────────────────
    let flat = FlatForward::new(today, 0.05, dc);
    println!("Comparison with flat 5% curve:");
    for &t in &[1.0, 5.0, 10.0] {
        let df_boot = curve.discount_t(t);
        let df_flat = flat.discount_t(t);
        println!(
            "  t = {:>5.1}y   bootstrapped = {:.8}   flat = {:.8}   diff = {:+.6}",
            t,
            df_boot,
            df_flat,
            df_boot - df_flat
        );
    }

    Ok(())
}
