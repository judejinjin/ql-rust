//! QuantLib-Rust CLI – a command-line interface for pricing, curve
//! bootstrapping, and trade persistence.

mod commands;

use anyhow::Result;
use clap::{Parser, Subcommand};

/// QuantLib-Rust command-line tool.
#[derive(Parser)]
#[command(name = "ql-cli", version, about = "QuantLib-Rust command-line interface")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Price a financial instrument.
    Price {
        #[command(subcommand)]
        instrument: PriceInstrument,
    },
    /// Yield-curve operations.
    Curve {
        #[command(subcommand)]
        action: CurveAction,
    },
    /// Trade booking and lifecycle.
    Trade {
        #[command(subcommand)]
        action: TradeAction,
    },
}

#[derive(Subcommand)]
enum PriceInstrument {
    /// Price a European vanilla option (Black-Scholes).
    VanillaOption {
        /// Current spot price.
        #[arg(long)]
        spot: f64,
        /// Strike price.
        #[arg(long)]
        strike: f64,
        /// Annualized volatility (e.g. 0.2 for 20%).
        #[arg(long)]
        vol: f64,
        /// Continuously compounded risk-free rate.
        #[arg(long)]
        rate: f64,
        /// Dividend yield (default: 0).
        #[arg(long, default_value_t = 0.0)]
        dividend: f64,
        /// Time to expiry in years (e.g. 1.0).
        #[arg(long)]
        expiry: f64,
        /// Option type: "call" or "put".
        #[arg(long, default_value = "call")]
        option_type: String,
    },
}

#[derive(Subcommand)]
enum CurveAction {
    /// Bootstrap a yield curve from a JSON market-data file.
    Bootstrap {
        /// Path to JSON market-data file.
        #[arg(long)]
        input: String,
        /// Optional output JSON file for the bootstrapped curve.
        #[arg(long)]
        output: Option<String>,
    },
}

#[derive(Subcommand)]
enum TradeAction {
    /// Book a new trade.
    Book {
        /// Instrument type (e.g. "option", "swap", "bond").
        #[arg(long, name = "type")]
        instrument_type: String,
        /// Counterparty name.
        #[arg(long)]
        counterparty: String,
        /// Trading book.
        #[arg(long)]
        book: String,
        /// Notional amount.
        #[arg(long)]
        notional: f64,
        /// Direction: "buy" or "sell".
        #[arg(long, default_value = "buy")]
        direction: String,
        /// Trade date (YYYY-MM-DD).
        #[arg(long)]
        trade_date: String,
        /// Settlement date (YYYY-MM-DD).
        #[arg(long)]
        settlement_date: String,
        /// Path to the redb database file.
        #[arg(long, default_value = "trades.redb")]
        db: String,
    },
    /// Show lifecycle events for a trade.
    Lifecycle {
        /// Trade ID.
        #[arg(long)]
        trade_id: String,
        /// Path to the redb database file.
        #[arg(long, default_value = "trades.redb")]
        db: String,
    },
    /// List trades matching optional filters.
    List {
        /// Filter by counterparty.
        #[arg(long)]
        counterparty: Option<String>,
        /// Filter by book.
        #[arg(long)]
        book: Option<String>,
        /// Path to the redb database file.
        #[arg(long, default_value = "trades.redb")]
        db: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Price { instrument } => commands::price::run(instrument),
        Commands::Curve { action } => commands::curve::run(action),
        Commands::Trade { action } => commands::trade::run(action),
    }
}
