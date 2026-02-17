//! `ql-cli trade` subcommand implementation.

use anyhow::{Context, Result};
use ql_persistence::{
    Direction, EmbeddedStore, InstrumentType, ObjectId, ObjectStore, Trade, TradeFilter,
};

use crate::TradeAction;

/// Execute the `trade` subcommand (book, list, or inspect trades).
pub fn run(action: TradeAction) -> Result<()> {
    match action {
        TradeAction::Book {
            instrument_type,
            counterparty,
            book,
            notional,
            direction,
            trade_date,
            settlement_date,
            db,
        } => book_trade(
            &instrument_type,
            &counterparty,
            &book,
            notional,
            &direction,
            &trade_date,
            &settlement_date,
            &db,
        ),
        TradeAction::Lifecycle { trade_id, db } => show_lifecycle(&trade_id, &db),
        TradeAction::List {
            counterparty,
            book,
            db,
        } => list_trades(counterparty.as_deref(), book.as_deref(), &db),
    }
}

fn parse_instrument_type(s: &str) -> Result<InstrumentType> {
    match s.to_lowercase().as_str() {
        "option" => Ok(InstrumentType::Option),
        "swap" | "irs" => Ok(InstrumentType::Swap),
        "bond" => Ok(InstrumentType::Bond),
        "cds" => Ok(InstrumentType::CDS),
        "swaption" => Ok(InstrumentType::Swaption),
        "capfloor" | "cap" => Ok(InstrumentType::Cap),
        "floor" => Ok(InstrumentType::Floor),
        _ => anyhow::bail!(
            "Unknown instrument type '{}'. Use: option, swap, bond, cds, swaption, capfloor.",
            s
        ),
    }
}

fn parse_direction(s: &str) -> Result<Direction> {
    match s.to_lowercase().as_str() {
        "buy" | "long" => Ok(Direction::Buy),
        "sell" | "short" => Ok(Direction::Sell),
        _ => anyhow::bail!("Unknown direction '{}'. Use 'buy' or 'sell'.", s),
    }
}

#[allow(clippy::too_many_arguments)]
fn book_trade(
    instrument_type: &str,
    counterparty: &str,
    book: &str,
    notional: f64,
    direction: &str,
    trade_date: &str,
    settlement_date: &str,
    db_path: &str,
) -> Result<()> {
    let inst_type = parse_instrument_type(instrument_type)?;
    let dir = parse_direction(direction)?;

    let trade = Trade::new(
        inst_type,
        serde_json::json!({}), // empty instrument data for now
        counterparty,
        book,
        notional,
        dir,
        trade_date,
        settlement_date,
        "cli-user",
    );

    let store = EmbeddedStore::open(db_path).context("Failed to open trade store")?;
    let version = store
        .put_trade(&trade, "cli-user")
        .context("Failed to persist trade")?;

    println!("╔══════════════════════════════════════════╗");
    println!("║            Trade Booked                  ║");
    println!("╠══════════════════════════════════════════╣");
    println!("║  Trade ID    : {:<24}  ║", trade.trade_id.as_str());
    println!("║  Type        : {:<24}  ║", instrument_type);
    println!("║  Counterparty: {:<24}  ║", counterparty);
    println!("║  Book        : {:<24}  ║", book);
    println!("║  Notional    : {:<24.2}  ║", notional);
    println!("║  Direction   : {:<24}  ║", direction);
    println!("║  Trade Date  : {:<24}  ║", trade_date);
    println!("║  Settle Date : {:<24}  ║", settlement_date);
    println!("║  Version     : {:<24}  ║", version);
    println!("╚══════════════════════════════════════════╝");

    Ok(())
}

fn show_lifecycle(trade_id: &str, db_path: &str) -> Result<()> {
    let store = EmbeddedStore::open(db_path).context("Failed to open trade store")?;
    let id = ObjectId::from_string(trade_id);

    // First show the trade
    let trade = store
        .get_trade(&id)
        .context("Failed to retrieve trade")?;

    println!("╔══════════════════════════════════════════╗");
    println!("║          Trade Lifecycle                 ║");
    println!("╠══════════════════════════════════════════╣");
    println!("║  Trade ID    : {:<24}  ║", trade.trade_id.as_str());
    println!("║  Status      : {:<24?}  ║", trade.status);
    println!("║  Counterparty: {:<24}  ║", trade.counterparty);
    println!("╠══════════════════════════════════════════╣");

    let events = store
        .replay_events(&id)
        .context("Failed to retrieve lifecycle events")?;

    if events.is_empty() {
        println!("║  (no lifecycle events)                   ║");
    } else {
        for event in &events {
            println!(
                "║  {:?} on {} by {}",
                event.event_type, event.event_date, event.entered_by
            );
        }
    }
    println!("╚══════════════════════════════════════════╝");

    Ok(())
}

fn list_trades(counterparty: Option<&str>, book: Option<&str>, db_path: &str) -> Result<()> {
    let store = EmbeddedStore::open(db_path).context("Failed to open trade store")?;

    let mut filter = TradeFilter::new();
    if let Some(cp) = counterparty {
        filter = filter.with_counterparty(cp);
    }
    if let Some(b) = book {
        filter = filter.with_book(b);
    }

    let trades = store
        .query_trades(&filter)
        .context("Failed to query trades")?;

    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║  {:>36}  {:>10}  {:>8}  {:>8}  ║", "Trade ID", "Type", "Book", "Status");
    println!("╠════════════════════════════════════════════════════════════════════════╣");
    if trades.is_empty() {
        println!("║  (no trades found)                                                   ║");
    } else {
        for t in &trades {
            println!(
                "║  {:>36}  {:>10?}  {:>8}  {:>8?}  ║",
                t.trade_id.as_str(),
                t.instrument_type,
                t.book,
                t.status
            );
        }
    }
    println!("║  Total: {} trade(s)                                                   ║", trades.len());
    println!("╚════════════════════════════════════════════════════════════════════════╝");

    Ok(())
}
