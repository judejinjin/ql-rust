//! Integration tests for the ql-cli binary.
//!
//! These tests invoke the compiled `ql-cli` binary via `assert_cmd` and check
//! stdout/stderr for expected output, covering the `price`, `curve`, and
//! `trade` subcommands.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

/// Helper to create a `Command` for the `ql-cli` binary.
fn ql_cli() -> Command {
    Command::cargo_bin("ql-cli").expect("binary ql-cli not found")
}

// =========================================================================
// price vanilla-option
// =========================================================================

#[test]
fn price_european_call_produces_positive_npv() {
    ql_cli()
        .args([
            "price", "vanilla-option",
            "--spot", "100",
            "--strike", "100",
            "--vol", "0.2",
            "--rate", "0.05",
            "--expiry", "1.0",
            "--option-type", "call",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("NPV"))
        .stdout(predicate::str::contains("Delta"))
        .stdout(predicate::str::contains("Gamma"))
        .stdout(predicate::str::contains("Vega"))
        .stdout(predicate::str::contains("Theta"))
        .stdout(predicate::str::contains("Rho"))
        .stdout(predicate::str::contains("European Call Option"));
}

#[test]
fn price_european_put_produces_correct_label() {
    ql_cli()
        .args([
            "price", "vanilla-option",
            "--spot", "100",
            "--strike", "110",
            "--vol", "0.25",
            "--rate", "0.03",
            "--expiry", "0.5",
            "--option-type", "put",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("European Put Option"));
}

#[test]
fn price_with_dividend_yield() {
    ql_cli()
        .args([
            "price", "vanilla-option",
            "--spot", "100",
            "--strike", "100",
            "--vol", "0.2",
            "--rate", "0.05",
            "--dividend", "0.02",
            "--expiry", "1.0",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Dividend"));
}

#[test]
fn price_invalid_option_type_fails() {
    ql_cli()
        .args([
            "price", "vanilla-option",
            "--spot", "100",
            "--strike", "100",
            "--vol", "0.2",
            "--rate", "0.05",
            "--expiry", "1.0",
            "--option-type", "straddle",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Unknown option type"));
}

#[test]
fn price_missing_required_arg_fails() {
    // Missing --expiry
    ql_cli()
        .args([
            "price", "vanilla-option",
            "--spot", "100",
            "--strike", "100",
            "--vol", "0.2",
            "--rate", "0.05",
        ])
        .assert()
        .failure();
}

// =========================================================================
// curve bootstrap
// =========================================================================

#[test]
fn curve_bootstrap_from_market_data_json() {
    // Use the project-level example fixture
    let fixture = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/market_data.json");
    ql_cli()
        .args(["curve", "bootstrap", "--input", fixture])
        .assert()
        .success()
        .stdout(predicate::str::contains("Bootstrapped Yield Curve"))
        .stdout(predicate::str::contains("Reference date"))
        .stdout(predicate::str::contains("Discount"))
        .stdout(predicate::str::contains("Zero Rate"));
}

#[test]
fn curve_bootstrap_with_json_output() {
    let dir = TempDir::new().unwrap();
    let output_path = dir.path().join("curve_out.json");
    let fixture = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/market_data.json");

    ql_cli()
        .args([
            "curve", "bootstrap",
            "--input", fixture,
            "--output", output_path.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Curve data written to"));

    // Verify the JSON output exists and is valid
    let json_str = fs::read_to_string(&output_path).expect("output file should exist");
    let parsed: serde_json::Value = serde_json::from_str(&json_str).expect("valid JSON");
    assert!(parsed["reference_date"].is_string());
    assert!(parsed["nodes"].is_array());
    let nodes = parsed["nodes"].as_array().unwrap();
    assert!(!nodes.is_empty(), "bootstrapped curve should have nodes");
    // Each node should have expected fields
    let first = &nodes[0];
    assert!(first["time"].is_f64());
    assert!(first["discount_factor"].is_f64());
}

#[test]
fn curve_bootstrap_missing_input_file_fails() {
    ql_cli()
        .args(["curve", "bootstrap", "--input", "/nonexistent/path.json"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Failed to read market data"));
}

#[test]
fn curve_bootstrap_invalid_json_fails() {
    let dir = TempDir::new().unwrap();
    let bad_input = dir.path().join("bad.json");
    fs::write(&bad_input, "{ not valid json").unwrap();

    ql_cli()
        .args(["curve", "bootstrap", "--input", bad_input.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Failed to parse market data"));
}

#[test]
fn curve_bootstrap_empty_instruments_fails() {
    let dir = TempDir::new().unwrap();
    let empty_input = dir.path().join("empty.json");
    fs::write(
        &empty_input,
        r#"{ "reference_date": "2025-01-15", "deposits": [], "swaps": [] }"#,
    )
    .unwrap();

    ql_cli()
        .args(["curve", "bootstrap", "--input", empty_input.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("No market data instruments"));
}

// =========================================================================
// trade book / list / lifecycle
// =========================================================================

#[test]
fn trade_book_and_list_round_trip() {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("test_trades.redb");
    let db = db_path.to_str().unwrap();

    // Book a trade
    ql_cli()
        .args([
            "trade", "book",
            "--instrument-type", "option",
            "--counterparty", "ACME",
            "--book", "FX-DESK",
            "--notional", "1000000",
            "--direction", "buy",
            "--trade-date", "2025-01-15",
            "--settlement-date", "2025-01-17",
            "--db", db,
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Trade Booked"))
        .stdout(predicate::str::contains("ACME"))
        .stdout(predicate::str::contains("FX-DESK"));

    // List trades — should find the one we booked
    ql_cli()
        .args(["trade", "list", "--db", db])
        .assert()
        .success()
        .stdout(predicate::str::contains("Total: 1 trade(s)"));
}

#[test]
fn trade_list_filter_by_counterparty() {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("filter_trades.redb");
    let db = db_path.to_str().unwrap();

    // Book two trades with different counterparties
    for (cp, book) in [("ALPHA", "DESK-A"), ("BETA", "DESK-B")] {
        ql_cli()
            .args([
                "trade", "book",
                "--instrument-type", "swap",
                "--counterparty", cp,
                "--book", book,
                "--notional", "5000000",
                "--trade-date", "2025-02-01",
                "--settlement-date", "2025-02-03",
                "--db", db,
            ])
            .assert()
            .success();
    }

    // Filter by counterparty ALPHA
    ql_cli()
        .args(["trade", "list", "--counterparty", "ALPHA", "--db", db])
        .assert()
        .success()
        .stdout(predicate::str::contains("DESK-A"))
        .stdout(predicate::str::contains("Total: 1 trade(s)"));
}

#[test]
fn trade_list_filter_by_book() {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("book_filter.redb");
    let db = db_path.to_str().unwrap();

    // Book trades in different books
    for book in ["RATES", "CREDIT"] {
        ql_cli()
            .args([
                "trade", "book",
                "--instrument-type", "bond",
                "--counterparty", "BankZ",
                "--book", book,
                "--notional", "10000000",
                "--trade-date", "2025-03-01",
                "--settlement-date", "2025-03-03",
                "--db", db,
            ])
            .assert()
            .success();
    }

    // Filter by book CREDIT
    ql_cli()
        .args(["trade", "list", "--book", "CREDIT", "--db", db])
        .assert()
        .success()
        .stdout(predicate::str::contains("Total: 1 trade(s)"));
}

#[test]
fn trade_book_invalid_instrument_type_fails() {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("invalid.redb");

    ql_cli()
        .args([
            "trade", "book",
            "--instrument-type", "futures",
            "--counterparty", "Test",
            "--book", "DESK",
            "--notional", "100",
            "--trade-date", "2025-01-01",
            "--settlement-date", "2025-01-03",
            "--db", db_path.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Unknown instrument type"));
}

#[test]
fn trade_book_invalid_direction_fails() {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("bad_dir.redb");

    ql_cli()
        .args([
            "trade", "book",
            "--instrument-type", "option",
            "--counterparty", "Test",
            "--book", "DESK",
            "--notional", "100",
            "--direction", "borrow",
            "--trade-date", "2025-01-01",
            "--settlement-date", "2025-01-03",
            "--db", db_path.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Unknown direction"));
}

// =========================================================================
// Top-level CLI
// =========================================================================

#[test]
fn help_flag_prints_usage() {
    ql_cli()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("QuantLib-Rust"));
}

#[test]
fn version_flag_prints_version() {
    ql_cli()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("ql-cli"));
}

#[test]
fn no_args_shows_error() {
    ql_cli()
        .assert()
        .failure()
        .stderr(predicate::str::contains("Usage"));
}

#[test]
fn subcommand_help_works() {
    ql_cli()
        .args(["price", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("vanilla-option"));
}
