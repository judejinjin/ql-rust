//! `ql-cli curve` subcommand implementation.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;

use ql_termstructures::{
    DepositRateHelper, PiecewiseYieldCurve, RateHelper, SwapRateHelper,
};
use ql_time::{Date, DayCounter, Month};

use crate::CurveAction;

/// JSON schema for market data input.
#[derive(Debug, Deserialize)]
struct MarketData {
    /// Reference date (YYYY-MM-DD).
    reference_date: String,
    /// Day counter (e.g. "Actual365Fixed", "Actual360").
    #[serde(default = "default_day_counter")]
    day_counter: String,
    /// Deposit rate quotes.
    #[serde(default)]
    deposits: Vec<DepositQuote>,
    /// Swap rate quotes.
    #[serde(default)]
    swaps: Vec<SwapQuote>,
}

fn default_day_counter() -> String {
    "Actual365Fixed".to_string()
}

#[derive(Debug, Deserialize)]
struct DepositQuote {
    /// Quoted rate.
    rate: f64,
    /// Maturity in months.
    maturity_months: u32,
}

#[derive(Debug, Deserialize)]
struct SwapQuote {
    /// Quoted par swap rate.
    rate: f64,
    /// Tenor in years.
    tenor_years: u32,
    /// Payment frequency per year (e.g. 2 for semi-annual).
    #[serde(default = "default_frequency")]
    frequency: u32,
}

fn default_frequency() -> u32 {
    2
}

/// Output format for bootstrapped curve.
#[derive(Debug, Serialize)]
struct CurveOutput {
    reference_date: String,
    day_counter: String,
    nodes: Vec<CurveNode>,
}

#[derive(Debug, Serialize)]
struct CurveNode {
    date: String,
    time: f64,
    discount_factor: f64,
    zero_rate: f64,
}

/// Execute the `curve` subcommand (e.g., bootstrap a yield curve).
pub fn run(action: CurveAction) -> Result<()> {
    match action {
        CurveAction::Bootstrap { input, output } => bootstrap(&input, output.as_deref()),
    }
}

fn parse_date(s: &str) -> Result<Date> {
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 3 {
        anyhow::bail!("Invalid date format '{}'. Expected YYYY-MM-DD.", s);
    }
    let year: i32 = parts[0].parse().context("Invalid year")?;
    let month_num: u32 = parts[1].parse().context("Invalid month")?;
    let day: u32 = parts[2].parse().context("Invalid day")?;
    let month = match month_num {
        1 => Month::January,
        2 => Month::February,
        3 => Month::March,
        4 => Month::April,
        5 => Month::May,
        6 => Month::June,
        7 => Month::July,
        8 => Month::August,
        9 => Month::September,
        10 => Month::October,
        11 => Month::November,
        12 => Month::December,
        _ => anyhow::bail!("Invalid month: {}", month_num),
    };
    Ok(Date::from_ymd(year, month, day))
}

fn parse_day_counter(s: &str) -> Result<DayCounter> {
    match s {
        "Actual365Fixed" | "Act365" => Ok(DayCounter::Actual365Fixed),
        "Actual360" | "Act360" => Ok(DayCounter::Actual360),
        "Thirty360" | "30/360" => Ok(DayCounter::Thirty360(ql_time::day_counter::Thirty360Convention::BondBasis)),
        _ => anyhow::bail!("Unknown day counter '{}'. Use Actual365Fixed, Actual360, or Thirty360.", s),
    }
}

fn bootstrap(input_path: &str, output_path: Option<&str>) -> Result<()> {
    let json_str = fs::read_to_string(input_path)
        .with_context(|| format!("Failed to read market data from '{}'", input_path))?;
    let market_data: MarketData =
        serde_json::from_str(&json_str).context("Failed to parse market data JSON")?;

    let ref_date = parse_date(&market_data.reference_date)?;
    let dc = parse_day_counter(&market_data.day_counter)?;

    // Build rate helpers
    let mut helpers: Vec<Box<dyn RateHelper>> = Vec::new();

    for dep in &market_data.deposits {
        let maturity_days = dep.maturity_months * 30; // approximate
        let end_date = ref_date + maturity_days as i32;
        helpers.push(Box::new(DepositRateHelper::new(
            dep.rate, ref_date, end_date, dc,
        )));
    }

    for swap in &market_data.swaps {
        let freq = swap.frequency;
        let n_periods = swap.tenor_years * freq;
        let months_per_period = 12 / freq;
        let mut dates = Vec::new();
        for i in 1..=n_periods {
            let total_months = i * months_per_period;
            let total_days = total_months * 30; // approximate
            dates.push(ref_date + total_days as i32);
        }
        helpers.push(Box::new(SwapRateHelper::new(
            swap.rate, ref_date, dates, dc,
        )));
    }

    if helpers.is_empty() {
        anyhow::bail!("No market data instruments found in input file.");
    }

    let curve = PiecewiseYieldCurve::new(ref_date, &mut helpers, dc, 1e-12)
        .context("Curve bootstrap failed")?;

    // Build output
    let nodes: Vec<CurveNode> = curve
        .nodes()
        .iter()
        .map(|&(t, df)| {
            let zero = if t > 1e-10 { -(df.ln()) / t } else { 0.0 };
            // Approximate the date from time
            let days = (t * 365.25) as i32;
            let node_date = ref_date + days;
            CurveNode {
                date: format!("{}", node_date),
                time: t,
                discount_factor: df,
                zero_rate: zero,
            }
        })
        .collect();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          Bootstrapped Yield Curve                          ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Reference date : {:<40} ║", market_data.reference_date);
    println!("║  Day counter    : {:<40} ║", market_data.day_counter);
    println!("║  Pillars        : {:<40} ║", nodes.len());
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  {:>8}  {:>14}  {:>14}  {:>12}     ║", "Time", "Discount", "Zero Rate", "Date");
    println!("╠══════════════════════════════════════════════════════════════╣");
    for node in &nodes {
        println!(
            "║  {:>8.4}  {:>14.8}  {:>14.6}  {:>12}     ║",
            node.time, node.discount_factor, node.zero_rate, node.date
        );
    }
    println!("╚══════════════════════════════════════════════════════════════╝");

    if let Some(out_path) = output_path {
        let output = CurveOutput {
            reference_date: market_data.reference_date.clone(),
            day_counter: market_data.day_counter.clone(),
            nodes,
        };
        let json = serde_json::to_string_pretty(&output)?;
        fs::write(out_path, &json)
            .with_context(|| format!("Failed to write output to '{}'", out_path))?;
        println!("\nCurve data written to: {}", out_path);
    }

    Ok(())
}
