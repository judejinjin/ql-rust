//! `ql-cli price` subcommand implementation.

use anyhow::{bail, Result};
use ql_instruments::{OptionType, VanillaOption};
use ql_pricingengines::price_european;

use crate::PriceInstrument;

pub fn run(instrument: PriceInstrument) -> Result<()> {
    match instrument {
        PriceInstrument::VanillaOption {
            spot,
            strike,
            vol,
            rate,
            dividend,
            expiry,
            option_type,
        } => price_vanilla_option(spot, strike, vol, rate, dividend, expiry, &option_type),
    }
}

fn price_vanilla_option(
    spot: f64,
    strike: f64,
    vol: f64,
    rate: f64,
    dividend: f64,
    expiry: f64,
    option_type: &str,
) -> Result<()> {
    let option = match option_type.to_lowercase().as_str() {
        "call" => VanillaOption::european_call(strike, ql_time::Date::from_serial(0)),
        "put" => VanillaOption::european_put(strike, ql_time::Date::from_serial(0)),
        _ => bail!("Unknown option type '{}'. Use 'call' or 'put'.", option_type),
    };

    let result = price_european(&option, spot, rate, dividend, vol, expiry);

    let type_label = match option.option_type() {
        OptionType::Call => "Call",
        OptionType::Put => "Put",
    };

    println!("╔══════════════════════════════════════════╗");
    println!("║     European {type_label} Option (Black-Scholes)   ║");
    println!("╠══════════════════════════════════════════╣");
    println!("║  Spot       : {:>24.6}  ║", spot);
    println!("║  Strike     : {:>24.6}  ║", strike);
    println!("║  Volatility : {:>24.4}  ║", vol);
    println!("║  Rate       : {:>24.4}  ║", rate);
    println!("║  Dividend   : {:>24.4}  ║", dividend);
    println!("║  Expiry (Y) : {:>24.4}  ║", expiry);
    println!("╠══════════════════════════════════════════╣");
    println!("║  NPV        : {:>24.6}  ║", result.npv);
    println!("║  Delta      : {:>24.6}  ║", result.delta);
    println!("║  Gamma      : {:>24.6}  ║", result.gamma);
    println!("║  Vega       : {:>24.6}  ║", result.vega);
    println!("║  Theta      : {:>24.6}  ║", result.theta);
    println!("║  Rho        : {:>24.6}  ║", result.rho);
    println!("╚══════════════════════════════════════════╝");

    Ok(())
}
