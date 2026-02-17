//! In-memory implementation of [`ObjectStore`].
//!
//! This store keeps all data in `HashMap`s behind `RwLock`s. It is intended
//! for unit testing and short-lived sessions — data is lost when the process
//! exits.

use std::collections::HashMap;
use std::sync::RwLock;

use chrono::{DateTime, Utc};
use ql_core::{QLError, QLResult};

use crate::domain::{
    LifecycleEvent, MarketSnapshot, ObjectId, SnapshotType, Trade, TradeFilter,
};
use crate::store::ObjectStore;

/// An in-memory object store backed by `HashMap`s.
///
/// Thread-safe via `RwLock`. Supports bitemporal versioning, event sourcing,
/// trade queries, and market snapshots — all in RAM.
pub struct InMemoryStore {
    /// trade_id → Vec<Trade> (one entry per version, newest last)
    trades: RwLock<HashMap<String, Vec<Trade>>>,
    /// trade_id → Vec<LifecycleEvent> (append-only, chronological)
    events: RwLock<HashMap<String, Vec<LifecycleEvent>>>,
    /// (date, snapshot_type_key) → MarketSnapshot
    snapshots: RwLock<HashMap<String, MarketSnapshot>>,
}

impl InMemoryStore {
    /// Create a new empty in-memory store.
    pub fn new() -> Self {
        Self {
            trades: RwLock::new(HashMap::new()),
            events: RwLock::new(HashMap::new()),
            snapshots: RwLock::new(HashMap::new()),
        }
    }

    /// Helper: composite key for snapshots.
    fn snapshot_key(date: &str, snapshot_type: SnapshotType) -> String {
        format!("{}:{:?}", date, snapshot_type)
    }
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl ObjectStore for InMemoryStore {
    fn get_trade(&self, id: &ObjectId) -> QLResult<Trade> {
        let trades = self.trades.read().map_err(|e| QLError::Other(e.to_string()))?;
        let versions = trades.get(id.as_str()).ok_or(QLError::NotFound)?;
        // Return the latest version that is still current (valid_to is None)
        versions
            .iter()
            .rev()
            .find(|t| t.is_current())
            .cloned()
            .ok_or(QLError::NotFound)
    }

    fn get_trade_as_of(&self, id: &ObjectId, as_of: DateTime<Utc>) -> QLResult<Trade> {
        let trades = self.trades.read().map_err(|e| QLError::Other(e.to_string()))?;
        let versions = trades.get(id.as_str()).ok_or(QLError::NotFound)?;
        // Find the version that was valid at `as_of`:
        //   valid_from <= as_of AND (valid_to is None OR valid_to > as_of)
        versions
            .iter()
            .rev()
            .find(|t| {
                t.valid_from <= as_of
                    && match t.valid_to {
                        None => true,
                        Some(vt) => vt > as_of,
                    }
            })
            .cloned()
            .ok_or(QLError::NotFound)
    }

    fn put_trade(&self, trade: &Trade, user: &str) -> QLResult<u64> {
        let mut trades = self.trades.write().map_err(|e| QLError::Other(e.to_string()))?;
        let versions = trades
            .entry(trade.trade_id.as_str().to_string())
            .or_default();

        let now = Utc::now();

        // Close the current version
        if let Some(current) = versions.iter_mut().rev().find(|t| t.is_current()) {
            current.valid_to = Some(now);
        }

        // Determine next version number
        let next_version = versions.iter().map(|t| t.version).max().unwrap_or(0) + 1;

        // Insert new version
        let mut new_trade = trade.clone();
        new_trade.version = next_version;
        new_trade.valid_from = now;
        new_trade.valid_to = None;
        new_trade.created_by = user.to_string();

        versions.push(new_trade);

        Ok(next_version)
    }

    fn append_event(
        &self,
        trade_id: &ObjectId,
        event: &LifecycleEvent,
        _user: &str,
    ) -> QLResult<ObjectId> {
        let mut events = self.events.write().map_err(|e| QLError::Other(e.to_string()))?;
        let event_list = events
            .entry(trade_id.as_str().to_string())
            .or_default();

        let event_id = event.event_id.clone();
        event_list.push(event.clone());
        Ok(event_id)
    }

    fn replay_events(&self, trade_id: &ObjectId) -> QLResult<Vec<LifecycleEvent>> {
        let events = self.events.read().map_err(|e| QLError::Other(e.to_string()))?;
        Ok(events
            .get(trade_id.as_str())
            .cloned()
            .unwrap_or_default())
    }

    fn query_trades(&self, filter: &TradeFilter) -> QLResult<Vec<Trade>> {
        let trades = self.trades.read().map_err(|e| QLError::Other(e.to_string()))?;
        let mut results = Vec::new();
        for versions in trades.values() {
            // Get only the current version of each trade
            if let Some(current) = versions.iter().rev().find(|t| t.is_current()) {
                if filter.matches(current) {
                    results.push(current.clone());
                }
            }
        }
        Ok(results)
    }

    fn save_snapshot(&self, snapshot: &MarketSnapshot) -> QLResult<ObjectId> {
        let mut snapshots = self
            .snapshots
            .write()
            .map_err(|e| QLError::Other(e.to_string()))?;
        let key = Self::snapshot_key(&snapshot.snapshot_date, snapshot.snapshot_type);
        let id = snapshot.snapshot_id.clone();
        snapshots.insert(key, snapshot.clone());
        Ok(id)
    }

    fn load_snapshot(
        &self,
        date: &str,
        snapshot_type: SnapshotType,
    ) -> QLResult<MarketSnapshot> {
        let snapshots = self
            .snapshots
            .read()
            .map_err(|e| QLError::Other(e.to_string()))?;
        let key = Self::snapshot_key(date, snapshot_type);
        snapshots.get(&key).cloned().ok_or(QLError::NotFound)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::*;

    fn make_test_trade(counterparty: &str, book: &str) -> Trade {
        Trade::new(
            InstrumentType::Swap,
            serde_json::json!({"fixed_rate": 0.035, "tenor": "5Y"}),
            counterparty,
            book,
            50_000_000.0,
            Direction::Buy,
            "2025-06-15",
            "2025-06-17",
            "trader1",
        )
    }

    #[test]
    fn test_put_and_get_trade() {
        let store = InMemoryStore::new();
        let trade = make_test_trade("JPMorgan", "rates_nyc");
        let trade_id = trade.trade_id.clone();

        let v = store.put_trade(&trade, "trader1").unwrap();
        assert_eq!(v, 1);

        let retrieved = store.get_trade(&trade_id).unwrap();
        assert_eq!(retrieved.counterparty, "JPMorgan");
        assert_eq!(retrieved.version, 1);
    }

    #[test]
    fn test_versioning() {
        let store = InMemoryStore::new();
        let mut trade = make_test_trade("GS", "credit");
        let trade_id = trade.trade_id.clone();

        store.put_trade(&trade, "t1").unwrap();

        // Amend the trade
        trade.notional = 40_000_000.0;
        trade.status = TradeStatus::Amended;
        let v2 = store.put_trade(&trade, "t1").unwrap();
        assert_eq!(v2, 2);

        let current = store.get_trade(&trade_id).unwrap();
        assert_eq!(current.version, 2);
    }

    #[test]
    fn test_bitemporal_query() {
        let store = InMemoryStore::new();
        let trade = make_test_trade("Citi", "fx");
        let trade_id = trade.trade_id.clone();

        store.put_trade(&trade, "t1").unwrap();
        let after_v1 = Utc::now();

        // Small delay to ensure timestamps differ
        std::thread::sleep(std::time::Duration::from_millis(10));

        let mut amended = trade.clone();
        amended.notional = 30_000_000.0;
        store.put_trade(&amended, "t1").unwrap();

        // Query "as of" after v1 but before v2 → should get v1
        let v1 = store.get_trade_as_of(&trade_id, after_v1).unwrap();
        assert_eq!(v1.version, 1);

        // Current should be v2
        let current = store.get_trade(&trade_id).unwrap();
        assert_eq!(current.version, 2);
    }

    #[test]
    fn test_event_sourcing() {
        let store = InMemoryStore::new();
        let trade_id = ObjectId::from_string("TRD-100");

        let events = vec![
            LifecycleEvent::new(
                &trade_id,
                EventType::Executed,
                "2025-06-15",
                "t1",
                serde_json::json!({"price": 99.5}),
            ),
            LifecycleEvent::new(
                &trade_id,
                EventType::Amended,
                "2025-07-01",
                "t1",
                serde_json::json!({"field": "notional", "new": 40e6}),
            ),
            LifecycleEvent::new(
                &trade_id,
                EventType::CashSettled,
                "2025-12-17",
                "ops",
                serde_json::json!({"amount": -437500.0}),
            ),
            LifecycleEvent::new(
                &trade_id,
                EventType::CashSettled,
                "2026-06-17",
                "ops",
                serde_json::json!({"amount": -425000.0}),
            ),
            LifecycleEvent::new(
                &trade_id,
                EventType::Matured,
                "2030-06-15",
                "system",
                serde_json::json!({}),
            ),
        ];

        for e in &events {
            store.append_event(&trade_id, e, &e.entered_by).unwrap();
        }

        let replayed = store.replay_events(&trade_id).unwrap();
        assert_eq!(replayed.len(), 5);
        assert_eq!(replayed[0].event_type, EventType::Executed);
        assert_eq!(replayed[4].event_type, EventType::Matured);
    }

    #[test]
    fn test_query_trades() {
        let store = InMemoryStore::new();
        store
            .put_trade(&make_test_trade("JPMorgan", "rates"), "t1")
            .unwrap();
        store
            .put_trade(&make_test_trade("JPMorgan", "credit"), "t1")
            .unwrap();
        store
            .put_trade(&make_test_trade("GS", "rates"), "t1")
            .unwrap();

        let jpm = store
            .query_trades(&TradeFilter::new().with_counterparty("JPMorgan"))
            .unwrap();
        assert_eq!(jpm.len(), 2);

        let rates = store
            .query_trades(&TradeFilter::new().with_book("rates"))
            .unwrap();
        assert_eq!(rates.len(), 2);

        let jpm_rates = store
            .query_trades(
                &TradeFilter::new()
                    .with_counterparty("JPMorgan")
                    .with_book("rates"),
            )
            .unwrap();
        assert_eq!(jpm_rates.len(), 1);
    }

    #[test]
    fn test_market_snapshot() {
        let store = InMemoryStore::new();
        let snap = MarketSnapshot::new(
            "2025-06-15",
            SnapshotType::EOD,
            serde_json::json!({
                "USD_SOFR": [0.04, 0.042, 0.045],
                "AAPL": 195.50
            }),
        );

        store.save_snapshot(&snap).unwrap();

        let loaded = store.load_snapshot("2025-06-15", SnapshotType::EOD).unwrap();
        assert_eq!(loaded.snapshot_date, "2025-06-15");
        assert_eq!(loaded.snapshot_type, SnapshotType::EOD);

        // Non-existent snapshot
        let result = store.load_snapshot("2025-06-15", SnapshotType::Intraday);
        assert!(result.is_err());
    }

    #[test]
    fn test_not_found() {
        let store = InMemoryStore::new();
        let result = store.get_trade(&ObjectId::from_string("nonexistent"));
        assert!(result.is_err());
    }
}
