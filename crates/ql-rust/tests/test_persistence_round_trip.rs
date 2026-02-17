//! Integration test: create trade → persist → load → verify.
//!
//! Validates the full persistence round-trip using the embedded store.

use ql_persistence::{
    Direction, EmbeddedStore, EventType, InstrumentType, LifecycleEvent,
    ObjectStore, Trade, TradeFilter, TradeStatus,
};

/// Book a trade, retrieve it, verify fields.
#[test]
fn trade_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test_rt.redb");
    let store = EmbeddedStore::open(db_path.to_str().unwrap()).unwrap();

    let trade = Trade::new(
        InstrumentType::Option,
        serde_json::json!({
            "type": "european_call",
            "strike": 105.0,
            "vol": 0.20,
        }),
        "ACME Corp",
        "equity-derivatives",
        1_000_000.0,
        Direction::Buy,
        "2025-01-15",
        "2025-01-17",
        "test-user",
    );

    let trade_id = trade.trade_id.clone();
    let version = store.put_trade(&trade, "test-user").unwrap();
    assert_eq!(version, 1);

    // Retrieve
    let loaded = store.get_trade(&trade_id).unwrap();
    assert_eq!(loaded.trade_id.as_str(), trade_id.as_str());
    assert_eq!(loaded.counterparty, "ACME Corp");
    assert_eq!(loaded.book, "equity-derivatives");
    assert_eq!(loaded.notional, 1_000_000.0);
    assert!(matches!(loaded.instrument_type, InstrumentType::Option));
    assert!(matches!(loaded.status, TradeStatus::Active));
    assert!(matches!(loaded.direction, Direction::Buy));
}

/// Book multiple trades, query by filter.
#[test]
fn trade_query_filter() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test_filter.redb");
    let store = EmbeddedStore::open(db_path.to_str().unwrap()).unwrap();

    // Book 3 trades
    let t1 = Trade::new(
        InstrumentType::Swap,
        serde_json::json!({}),
        "Bank A",
        "rates",
        5_000_000.0,
        Direction::Buy,
        "2025-01-15",
        "2025-01-17",
        "user1",
    );
    let t2 = Trade::new(
        InstrumentType::Option,
        serde_json::json!({}),
        "Bank B",
        "equity-derivatives",
        2_000_000.0,
        Direction::Sell,
        "2025-01-16",
        "2025-01-18",
        "user1",
    );
    let t3 = Trade::new(
        InstrumentType::Swap,
        serde_json::json!({}),
        "Bank A",
        "rates",
        10_000_000.0,
        Direction::Buy,
        "2025-01-17",
        "2025-01-19",
        "user2",
    );

    store.put_trade(&t1, "user1").unwrap();
    store.put_trade(&t2, "user1").unwrap();
    store.put_trade(&t3, "user2").unwrap();

    // Query all
    let all = store.query_trades(&TradeFilter::new()).unwrap();
    assert_eq!(all.len(), 3);

    // Query by counterparty
    let bank_a = store
        .query_trades(&TradeFilter::new().with_counterparty("Bank A"))
        .unwrap();
    assert_eq!(bank_a.len(), 2);

    // Query by instrument type
    let swaps = store
        .query_trades(&TradeFilter::new().with_instrument_type(InstrumentType::Swap))
        .unwrap();
    assert_eq!(swaps.len(), 2);

    // Query by book
    let equity = store
        .query_trades(&TradeFilter::new().with_book("equity-derivatives"))
        .unwrap();
    assert_eq!(equity.len(), 1);
}

/// Append lifecycle events and replay them.
#[test]
fn lifecycle_event_sourcing() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test_events.redb");
    let store = EmbeddedStore::open(db_path.to_str().unwrap()).unwrap();

    let trade = Trade::new(
        InstrumentType::Bond,
        serde_json::json!({"coupon": 0.05}),
        "Issuer X",
        "credit",
        50_000_000.0,
        Direction::Buy,
        "2025-01-15",
        "2025-01-20",
        "trader1",
    );

    let trade_id = trade.trade_id.clone();
    store.put_trade(&trade, "trader1").unwrap();

    // Append amendment event
    let event1 = LifecycleEvent::new(
        &trade_id,
        EventType::Amended,
        "2025-02-01",
        "trader1",
        serde_json::json!({"reason": "notional increase", "new_notional": 60_000_000.0}),
    );
    store.append_event(&trade_id, &event1, "trader1").unwrap();

    // Append another event
    let event2 = LifecycleEvent::new(
        &trade_id,
        EventType::CashSettled,
        "2025-07-15",
        "ops-team",
        serde_json::json!({"amount": 1_500_000.0, "type": "coupon"}),
    );
    store.append_event(&trade_id, &event2, "ops-team").unwrap();

    // Replay
    let events = store.replay_events(&trade_id).unwrap();
    assert_eq!(events.len(), 2);
    assert!(matches!(events[0].event_type, EventType::Amended));
    assert!(matches!(events[1].event_type, EventType::CashSettled));
}

/// Versioning: put same trade twice, version increments.
#[test]
fn trade_versioning() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test_version.redb");
    let store = EmbeddedStore::open(db_path.to_str().unwrap()).unwrap();

    let trade = Trade::new(
        InstrumentType::Option,
        serde_json::json!({}),
        "CP1",
        "eq",
        100_000.0,
        Direction::Buy,
        "2025-01-15",
        "2025-01-17",
        "user",
    );

    let v1 = store.put_trade(&trade, "user").unwrap();
    let v2 = store.put_trade(&trade, "user").unwrap();

    assert_eq!(v1, 1);
    assert_eq!(v2, 2);
}
