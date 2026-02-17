//! Price a European call option using the Black-Scholes analytic engine.
//!
//! Run with:
//! ```sh
//! cargo run -p ql-rust --example price_european_option
//! ```

use ql_instruments::VanillaOption;
use ql_pricingengines::{implied_volatility, price_european};
use ql_time::{Date, Month};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let today = Date::from_ymd(2025, Month::January, 15);

    // ── Option parameters ────────────────────────────────────────────
    let spot = 100.0;
    let strike = 105.0;
    let vol = 0.20;
    let rate = 0.05;
    let dividend = 0.02;
    let expiry = 1.0; // 1 year

    let call = VanillaOption::european_call(strike, today + 365);

    // ── Price ────────────────────────────────────────────────────────
    let result = price_european(&call, spot, rate, dividend, vol, expiry);

    println!("European Call Option (Black-Scholes)");
    println!("════════════════════════════════════");
    println!("  Spot     = {:.2}", spot);
    println!("  Strike   = {:.2}", strike);
    println!("  Vol      = {:.1}%", vol * 100.0);
    println!("  Rate     = {:.1}%", rate * 100.0);
    println!("  Dividend = {:.1}%", dividend * 100.0);
    println!("  Expiry   = {:.1}Y", expiry);
    println!("────────────────────────────────────");
    println!("  NPV      = {:.6}", result.npv);
    println!("  Delta    = {:.6}", result.delta);
    println!("  Gamma    = {:.6}", result.gamma);
    println!("  Vega     = {:.6}", result.vega);
    println!("  Theta    = {:.6}", result.theta);
    println!("  Rho      = {:.6}", result.rho);

    // ── Implied volatility round-trip ────────────────────────────────
    let iv = implied_volatility(&call, result.npv, spot, rate, dividend, expiry)?;
    println!("────────────────────────────────────");
    println!("  Implied vol (round-trip) = {:.6} ({:.2}%)", iv, iv * 100.0);

    Ok(())
}
