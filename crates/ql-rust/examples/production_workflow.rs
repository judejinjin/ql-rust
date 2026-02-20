//! Production workflow: exotic pricing, credit, and stochastic local vol.
//!
//! This example demonstrates the advanced capabilities added in the latest phase:
//! 1. Swaption vol surfaces
//! 2. Exotic options (quanto, power, forward-start, digital barrier)
//! 3. CDS pricing and credit analytics (ISDA standard model)
//! 4. Stochastic Local Volatility Monte Carlo
//!
//! Run with:
//! ```sh
//! cargo run --example production_workflow
//! ```

use std::sync::Arc;
use ql_instruments::CdsProtectionSide;
use ql_pricingengines::{
    // ISDA CDS
    isda_cds_engine, cds_upfront, cds_cs01,
    // Advanced exotics
    quanto_european, power_option, forward_start_option,
    digital_barrier, DigitalBarrierType,
    // SLV
    DupireLocalVol, SlvModel, mc_slv,
};
use ql_termstructures::{
    FlatForward, SwaptionConstantVol, YieldTermStructure,
    default_term_structure::{FlatHazardRate, DefaultProbabilityTermStructure},
};
use ql_time::{Date, DayCounter, Month};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  ql-rust Production Workflow Example");
    println!("═══════════════════════════════════════════════════════════\n");

    // ──────────────────────────────────────────────────────────────────
    // 1. Swaption Volatility Surface
    // ──────────────────────────────────────────────────────────────────
    println!("── 1. Swaption Vol Surface ───────────────────────────────");
    use ql_termstructures::SwaptionVolatilityStructure;
    let swvol = SwaptionConstantVol::new(0.20);
    println!("  ATM swaption vol (1Y into 5Y)  = {:.2}%",
             swvol.volatility(1.0, 5.0, 0.0) * 100.0);
    println!("  ATM swaption vol (5Y into 10Y) = {:.2}%\n",
             swvol.volatility(5.0, 10.0, 0.0) * 100.0);

    // ──────────────────────────────────────────────────────────────────
    // 2. Exotic Options
    // ──────────────────────────────────────────────────────────────────
    println!("── 2. Exotic Options ─────────────────────────────────────");
    let spot = 100.0;
    let strike = 100.0;

    // Quanto European Call (10 args: spot, strike, r_d, r_f, vol_s, vol_fx, rho, t, fx_rate, is_call)
    let quanto = quanto_european(spot, strike, 0.05, 0.0, 0.20, 0.10, -0.3, 1.0, 1.0, true);
    println!("  Quanto European Call   = {:.4}  (fx_vol=10%, corr=-0.3)", quanto.price);

    // Power option (α = 2)
    let power = power_option(spot, strike * strike, 0.05, 0.0, 0.20, 1.0, 2.0, true);
    println!("  Power Option (α=2)     = {:.4}", power.price);

    // Forward-start Call (strike set at 6M, matures 1Y)
    let fwd_start = forward_start_option(spot, 1.0, 0.05, 0.0, 0.20, 0.5, 1.0, true);
    println!("  Forward-Start Call     = {:.4}  (α=1.0, start=6M, mat=1Y)", fwd_start.price);

    // Digital Barrier: one-touch down
    let digi = digital_barrier(
        spot, 90.0, 0.05, 0.0, 0.20, 1.0, 1.0,
        DigitalBarrierType::OneTouch, true,
    );
    println!("  One-Touch Down (B=90)  = {:.4}\n", digi.price);

    // ──────────────────────────────────────────────────────────────────
    // 3. ISDA CDS Pricing
    // ──────────────────────────────────────────────────────────────────
    println!("── 3. ISDA CDS Pricing ─────────────────────────────────");
    let ref_date = Date::from_ymd(2025, Month::March, 20);
    let dc_cds = DayCounter::Actual360;

    let cds = ql_pricingengines::make_standard_cds(
        CdsProtectionSide::Buyer,
        10_000_000.0,
        0.01,      // 100bp running
        ref_date,
        Date::from_ymd(2030, Month::March, 20),
        0.40,
        dc_cds,
    );

    let default_curve: Arc<dyn DefaultProbabilityTermStructure> = Arc::new(
        FlatHazardRate::from_spread(ref_date, 0.01, 0.4, DayCounter::Actual365Fixed),
    );
    let yield_curve: Arc<dyn YieldTermStructure> =
        Arc::new(FlatForward::new(ref_date, 0.03, DayCounter::Actual365Fixed));

    let cds_result = isda_cds_engine(&cds, &default_curve, &yield_curve, None, None);
    println!("  5Y CDS (100bp running, 40% recovery):");
    println!("    Dirty NPV    = {:>14.2}", cds_result.dirty_npv);
    println!("    Clean NPV    = {:>14.2}", cds_result.clean_npv);
    println!("    Fair Spread  = {:>10.1} bp", cds_result.fair_spread * 10_000.0);
    println!("    RPV01        = {:>14.6}", cds_result.rpv01);

    let upfront = cds_upfront(&cds, &default_curve, &yield_curve);
    println!("    Upfront      = {:>14.2}", upfront);

    let cs01 = cds_cs01(&cds, &default_curve, &yield_curve);
    println!("    CS01         = {:>14.2}\n", cs01);

    // ──────────────────────────────────────────────────────────────────
    // 4. Stochastic Local Volatility Monte Carlo
    // ──────────────────────────────────────────────────────────────────
    println!("── 4. Stochastic Local Volatility MC ─────────────────────");
    let heston_slv = SlvModel::heston(100.0, 0.05, 0.0, 0.04, 1.5, 0.04, 0.3, -0.7);
    let slv_call = mc_slv(&heston_slv, 100.0, 1.0, true, 100_000, 200);
    let slv_put  = mc_slv(&heston_slv, 100.0, 1.0, false, 100_000, 200);
    println!("  Heston (v0=0.04, κ=1.5, θ=0.04, ξ=0.3, ρ=-0.7):");
    println!("    ATM Call = {:.4} ± {:.4}", slv_call.price, slv_call.std_error);
    println!("    ATM Put  = {:.4} ± {:.4}", slv_put.price, slv_put.std_error);
    let parity = slv_call.price - slv_put.price - (100.0 - 100.0 * (-0.05_f64).exp());
    println!("    Put-Call parity error = {:.4}\n", parity);

    // SLV with leverage function
    let leverage = DupireLocalVol::new(
        vec![0.5, 1.0],
        vec![80.0, 90.0, 100.0, 110.0, 120.0],
        vec![
            vec![1.05, 1.02, 1.00, 0.98, 0.95],
            vec![1.03, 1.01, 1.00, 0.99, 0.97],
        ],
    );
    let slv_lev = SlvModel::with_leverage(
        100.0, 0.05, 0.0,
        0.04, 1.5, 0.04, 0.3, -0.7,
        leverage,
    );
    let slv_lev_call = mc_slv(&slv_lev, 100.0, 1.0, true, 50_000, 200);
    println!("  SLV with leverage function:");
    println!("    ATM Call = {:.4} ± {:.4}\n", slv_lev_call.price, slv_lev_call.std_error);

    // ──────────────────────────────────────────────────────────────────
    println!("═══════════════════════════════════════════════════════════");
    println!("  All computations completed successfully.");
    println!("═══════════════════════════════════════════════════════════");

    Ok(())
}
