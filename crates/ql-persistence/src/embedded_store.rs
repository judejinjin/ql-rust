//! Embedded on-disk implementation of [`ObjectStore`] backed by [`redb`].
//!
//! Uses `redb` — a pure-Rust, ACID, embedded key-value store — for durable
//! persistence with zero external infrastructure. Data is serialized with
//! `serde_json` (compatible with `serde_json::Value` fields in domain types).
//!
//! # Table Layout
//!
//! | Table | Key | Value |
//! |---|---|---|
//! | `trades` | `"{trade_id}:{version:016}"` | JSON-encoded `Trade` |
//! | `trade_current` | `"{trade_id}"` | version number as `u64` LE bytes |
//! | `events` | `"{trade_id}:{seq:016}"` | JSON-encoded `LifecycleEvent` |
//! | `event_counters` | `"{trade_id}"` | event counter as `u64` LE bytes |
//! | `snapshots` | `"{date}:{type}"` | JSON-encoded `MarketSnapshot` |

use ql_core::{QLError, QLResult};
use redb::{Database, ReadableTable, TableDefinition};

use crate::domain::{
    LifecycleEvent, MarketSnapshot, ObjectId, SnapshotType, Trade, TradeFilter,
};
use crate::store::ObjectStore;

use chrono::{DateTime, Utc};

// ---------------------------------------------------------------------------
// Table definitions
// ---------------------------------------------------------------------------

/// Versioned trade storage: key = "trade_id:version"
const TRADES: TableDefinition<&str, &[u8]> = TableDefinition::new("trades");

/// Current version pointer: key = "trade_id", value = u64 LE bytes
const TRADE_CURRENT: TableDefinition<&str, &[u8]> = TableDefinition::new("trade_current");

/// Lifecycle events: key = "trade_id:seq_number(zero-padded)"
const EVENTS: TableDefinition<&str, &[u8]> = TableDefinition::new("events");

/// Event counters: key = "trade_id", value = u64 LE bytes
const EVENT_COUNTERS: TableDefinition<&str, &[u8]> = TableDefinition::new("event_counters");

/// Market snapshots: key = "date:snapshot_type"
const SNAPSHOTS: TableDefinition<&str, &[u8]> = TableDefinition::new("snapshots");

// ---------------------------------------------------------------------------
// Helpers for u64 encoding (simple LE bytes, no serde needed)
// ---------------------------------------------------------------------------

fn encode_u64(v: u64) -> [u8; 8] {
    v.to_le_bytes()
}

fn decode_u64(bytes: &[u8]) -> QLResult<u64> {
    let arr: [u8; 8] = bytes
        .try_into()
        .map_err(|_| QLError::Other("invalid u64 bytes".into()))?;
    Ok(u64::from_le_bytes(arr))
}

// ---------------------------------------------------------------------------
// Helpers for JSON encoding of domain types
// ---------------------------------------------------------------------------

fn to_json_bytes<T: serde::Serialize>(val: &T) -> QLResult<Vec<u8>> {
    serde_json::to_vec(val).map_err(|e| QLError::Other(format!("json encode: {}", e)))
}

fn from_json_bytes<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> QLResult<T> {
    serde_json::from_slice(bytes).map_err(|e| QLError::Other(format!("json decode: {}", e)))
}

// ---------------------------------------------------------------------------
// EmbeddedStore
// ---------------------------------------------------------------------------

/// An on-disk object store backed by `redb`.
///
/// Provides durable, ACID-transactional storage for trades, lifecycle events,
/// and market snapshots with bitemporal versioning.
///
/// # Example
///
/// ```no_run
/// use ql_persistence::embedded_store::EmbeddedStore;
/// let store = EmbeddedStore::open("/tmp/ql_data.redb").unwrap();
/// ```
pub struct EmbeddedStore {
    db: Database,
}

impl EmbeddedStore {
    /// Open (or create) a redb database at the given file path.
    pub fn open(path: &str) -> QLResult<Self> {
        let db =
            Database::create(path).map_err(|e| QLError::Other(format!("redb open: {}", e)))?;
        Ok(Self { db })
    }

    /// Helper: encode a versioned trade key.
    fn trade_key(trade_id: &str, version: u64) -> String {
        format!("{}:{:016}", trade_id, version)
    }

    /// Helper: encode a snapshot key.
    fn snapshot_key(date: &str, snapshot_type: SnapshotType) -> String {
        format!("{}:{:?}", date, snapshot_type)
    }

    /// Helper: encode an event key.
    fn event_key(trade_id: &str, seq: u64) -> String {
        format!("{}:{:016}", trade_id, seq)
    }

    /// Get the current version number for a trade.
    fn current_version(&self, trade_id: &str) -> QLResult<u64> {
        let txn = self
            .db
            .begin_read()
            .map_err(|e| QLError::Other(format!("redb read txn: {}", e)))?;
        let table = txn
            .open_table(TRADE_CURRENT)
            .map_err(|e| QLError::Other(format!("redb open table: {}", e)))?;
        let guard = table
            .get(trade_id)
            .map_err(|e| QLError::Other(format!("redb get: {}", e)))?
            .ok_or(QLError::NotFound)?;
        decode_u64(guard.value())
    }

    /// Get the next event sequence number for a trade, incrementing the counter.
    fn next_event_seq(
        &self,
        txn: &redb::WriteTransaction,
        trade_id: &str,
    ) -> QLResult<u64> {
        let mut table = txn
            .open_table(EVENT_COUNTERS)
            .map_err(|e| QLError::Other(format!("redb open table: {}", e)))?;

        let current = match table
            .get(trade_id)
            .map_err(|e| QLError::Other(format!("redb get: {}", e)))?
        {
            Some(guard) => decode_u64(guard.value())?,
            None => 0,
        };

        let next = current + 1;
        let bytes = encode_u64(next);
        table
            .insert(trade_id, bytes.as_slice())
            .map_err(|e| QLError::Other(format!("redb insert: {}", e)))?;

        Ok(next)
    }
}

impl ObjectStore for EmbeddedStore {
    fn get_trade(&self, id: &ObjectId) -> QLResult<Trade> {
        let version = self.current_version(id.as_str())?;
        let key = Self::trade_key(id.as_str(), version);

        let txn = self
            .db
            .begin_read()
            .map_err(|e| QLError::Other(format!("redb read txn: {}", e)))?;
        let table = txn
            .open_table(TRADES)
            .map_err(|e| QLError::Other(format!("redb open table: {}", e)))?;
        let guard = table
            .get(key.as_str())
            .map_err(|e| QLError::Other(format!("redb get: {}", e)))?
            .ok_or(QLError::NotFound)?;

        from_json_bytes(guard.value())
    }

    fn get_trade_as_of(&self, id: &ObjectId, as_of: DateTime<Utc>) -> QLResult<Trade> {
        let current_ver = self.current_version(id.as_str())?;

        let txn = self
            .db
            .begin_read()
            .map_err(|e| QLError::Other(format!("redb read txn: {}", e)))?;
        let table = txn
            .open_table(TRADES)
            .map_err(|e| QLError::Other(format!("redb open table: {}", e)))?;

        // Scan versions from newest to oldest to find the one valid at `as_of`
        for v in (1..=current_ver).rev() {
            let key = Self::trade_key(id.as_str(), v);
            if let Some(guard) = table
                .get(key.as_str())
                .map_err(|e| QLError::Other(format!("redb get: {}", e)))?
            {
                let trade: Trade = from_json_bytes(guard.value())?;
                if trade.valid_from <= as_of {
                    match trade.valid_to {
                        None => return Ok(trade),
                        Some(vt) if vt > as_of => return Ok(trade),
                        _ => continue,
                    }
                }
            }
        }

        Err(QLError::NotFound)
    }

    fn put_trade(&self, trade: &Trade, user: &str) -> QLResult<u64> {
        let trade_id = trade.trade_id.as_str();
        let now = Utc::now();

        let txn = self
            .db
            .begin_write()
            .map_err(|e| QLError::Other(format!("redb write txn: {}", e)))?;

        let next_version;
        {
            let mut current_table = txn
                .open_table(TRADE_CURRENT)
                .map_err(|e| QLError::Other(format!("redb open table: {}", e)))?;
            let mut trades_table = txn
                .open_table(TRADES)
                .map_err(|e| QLError::Other(format!("redb open table: {}", e)))?;

            let prev_version = match current_table
                .get(trade_id)
                .map_err(|e| QLError::Other(format!("redb get: {}", e)))?
            {
                Some(guard) => Some(decode_u64(guard.value())?),
                None => None,
            };

            // Close previous version by updating its valid_to
            if let Some(pv) = prev_version {
                let prev_key = Self::trade_key(trade_id, pv);
                let prev_trade_bytes = trades_table
                    .get(prev_key.as_str())
                    .map_err(|e| QLError::Other(format!("redb get: {}", e)))?
                    .map(|guard| guard.value().to_vec());

                if let Some(bytes) = prev_trade_bytes {
                    let mut prev_trade: Trade = from_json_bytes(&bytes)?;
                    prev_trade.valid_to = Some(now);
                    let updated_bytes = to_json_bytes(&prev_trade)?;
                    trades_table
                        .insert(prev_key.as_str(), updated_bytes.as_slice())
                        .map_err(|e| QLError::Other(format!("redb insert: {}", e)))?;
                }
            }

            next_version = prev_version.unwrap_or(0) + 1;

            // Write new version
            let mut new_trade = trade.clone();
            new_trade.version = next_version;
            new_trade.valid_from = now;
            new_trade.valid_to = None;
            new_trade.created_by = user.to_string();

            let new_key = Self::trade_key(trade_id, next_version);
            let bytes = to_json_bytes(&new_trade)?;
            trades_table
                .insert(new_key.as_str(), bytes.as_slice())
                .map_err(|e| QLError::Other(format!("redb insert: {}", e)))?;

            // Update current version pointer
            let ver_bytes = encode_u64(next_version);
            current_table
                .insert(trade_id, ver_bytes.as_slice())
                .map_err(|e| QLError::Other(format!("redb insert: {}", e)))?;
        }

        txn.commit()
            .map_err(|e| QLError::Other(format!("redb commit: {}", e)))?;

        Ok(next_version)
    }

    fn append_event(
        &self,
        trade_id: &ObjectId,
        event: &LifecycleEvent,
        _user: &str,
    ) -> QLResult<ObjectId> {
        let txn = self
            .db
            .begin_write()
            .map_err(|e| QLError::Other(format!("redb write txn: {}", e)))?;

        let event_id;
        {
            let seq = self.next_event_seq(&txn, trade_id.as_str())?;
            let key = Self::event_key(trade_id.as_str(), seq);
            let bytes = to_json_bytes(event)?;

            let mut table = txn
                .open_table(EVENTS)
                .map_err(|e| QLError::Other(format!("redb open table: {}", e)))?;
            table
                .insert(key.as_str(), bytes.as_slice())
                .map_err(|e| QLError::Other(format!("redb insert: {}", e)))?;

            event_id = event.event_id.clone();
        }

        txn.commit()
            .map_err(|e| QLError::Other(format!("redb commit: {}", e)))?;

        Ok(event_id)
    }

    fn replay_events(&self, trade_id: &ObjectId) -> QLResult<Vec<LifecycleEvent>> {
        let txn = self
            .db
            .begin_read()
            .map_err(|e| QLError::Other(format!("redb read txn: {}", e)))?;
        let table = txn
            .open_table(EVENTS)
            .map_err(|e| QLError::Other(format!("redb open table: {}", e)))?;

        let prefix = format!("{}:", trade_id.as_str());
        let start = format!("{}:{:016}", trade_id.as_str(), 0u64);
        let end = format!("{}:{:016}", trade_id.as_str(), u64::MAX);

        let mut events = Vec::new();
        let range = table
            .range(start.as_str()..=end.as_str())
            .map_err(|e| QLError::Other(format!("redb range: {}", e)))?;

        for entry in range {
            let (key_guard, val_guard) =
                entry.map_err(|e| QLError::Other(format!("redb iter: {}", e)))?;
            let k = key_guard.value();
            if !k.starts_with(&prefix) {
                break;
            }
            let event: LifecycleEvent = from_json_bytes(val_guard.value())?;
            events.push(event);
        }

        Ok(events)
    }

    fn query_trades(&self, filter: &TradeFilter) -> QLResult<Vec<Trade>> {
        let txn = self
            .db
            .begin_read()
            .map_err(|e| QLError::Other(format!("redb read txn: {}", e)))?;
        let current_table = txn
            .open_table(TRADE_CURRENT)
            .map_err(|e| QLError::Other(format!("redb open table: {}", e)))?;
        let trades_table = txn
            .open_table(TRADES)
            .map_err(|e| QLError::Other(format!("redb open table: {}", e)))?;

        let mut results = Vec::new();

        let range = current_table
            .iter()
            .map_err(|e| QLError::Other(format!("redb iter: {}", e)))?;

        for entry in range {
            let (key_guard, val_guard) =
                entry.map_err(|e| QLError::Other(format!("redb iter: {}", e)))?;
            let tid = key_guard.value();
            let version = decode_u64(val_guard.value())?;

            let trade_key = Self::trade_key(tid, version);
            if let Some(trade_guard) = trades_table
                .get(trade_key.as_str())
                .map_err(|e| QLError::Other(format!("redb get: {}", e)))?
            {
                let trade: Trade = from_json_bytes(trade_guard.value())?;
                if filter.matches(&trade) {
                    results.push(trade);
                }
            }
        }

        Ok(results)
    }

    fn save_snapshot(&self, snapshot: &MarketSnapshot) -> QLResult<ObjectId> {
        let key = Self::snapshot_key(&snapshot.snapshot_date, snapshot.snapshot_type);
        let bytes = to_json_bytes(snapshot)?;

        let txn = self
            .db
            .begin_write()
            .map_err(|e| QLError::Other(format!("redb write txn: {}", e)))?;
        {
            let mut table = txn
                .open_table(SNAPSHOTS)
                .map_err(|e| QLError::Other(format!("redb open table: {}", e)))?;
            table
                .insert(key.as_str(), bytes.as_slice())
                .map_err(|e| QLError::Other(format!("redb insert: {}", e)))?;
        }
        txn.commit()
            .map_err(|e| QLError::Other(format!("redb commit: {}", e)))?;

        Ok(snapshot.snapshot_id.clone())
    }

    fn load_snapshot(
        &self,
        date: &str,
        snapshot_type: SnapshotType,
    ) -> QLResult<MarketSnapshot> {
        let key = Self::snapshot_key(date, snapshot_type);

        let txn = self
            .db
            .begin_read()
            .map_err(|e| QLError::Other(format!("redb read txn: {}", e)))?;
        let table = txn
            .open_table(SNAPSHOTS)
            .map_err(|e| QLError::Other(format!("redb open table: {}", e)))?;
        let guard = table
            .get(key.as_str())
            .map_err(|e| QLError::Other(format!("redb get: {}", e)))?
            .ok_or(QLError::NotFound)?;

        from_json_bytes(guard.value())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::*;
    use tempfile::NamedTempFile;

    fn temp_store() -> EmbeddedStore {
        let file = NamedTempFile::new().unwrap();
        EmbeddedStore::open(file.path().to_str().unwrap()).unwrap()
    }

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
        let store = temp_store();
        let trade = make_test_trade("JPMorgan", "rates_nyc");
        let trade_id = trade.trade_id.clone();

        let v = store.put_trade(&trade, "trader1").unwrap();
        assert_eq!(v, 1);

        let retrieved = store.get_trade(&trade_id).unwrap();
        assert_eq!(retrieved.counterparty, "JPMorgan");
        assert_eq!(retrieved.version, 1);
        assert!(retrieved.is_current());
    }

    #[test]
    fn test_versioning() {
        let store = temp_store();
        let mut trade = make_test_trade("GS", "credit");
        let trade_id = trade.trade_id.clone();

        store.put_trade(&trade, "t1").unwrap();

        trade.notional = 40_000_000.0;
        trade.status = TradeStatus::Amended;
        let v2 = store.put_trade(&trade, "t1").unwrap();
        assert_eq!(v2, 2);

        let current = store.get_trade(&trade_id).unwrap();
        assert_eq!(current.version, 2);
    }

    #[test]
    fn test_bitemporal_query() {
        let store = temp_store();
        let trade = make_test_trade("Citi", "fx");
        let trade_id = trade.trade_id.clone();

        store.put_trade(&trade, "t1").unwrap();
        let after_v1 = Utc::now();

        std::thread::sleep(std::time::Duration::from_millis(10));

        let mut amended = trade.clone();
        amended.notional = 30_000_000.0;
        store.put_trade(&amended, "t1").unwrap();

        // Query "as of" after v1 but before v2
        let v1 = store.get_trade_as_of(&trade_id, after_v1).unwrap();
        assert_eq!(v1.version, 1);

        // Current should be v2
        let current = store.get_trade(&trade_id).unwrap();
        assert_eq!(current.version, 2);
    }

    #[test]
    fn test_event_sourcing() {
        let store = temp_store();
        let trade_id = ObjectId::from_string("TRD-200");

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
        let store = temp_store();
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

        let all = store.query_trades(&TradeFilter::new()).unwrap();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_market_snapshot() {
        let store = temp_store();
        let snap = MarketSnapshot::new(
            "2025-06-15",
            SnapshotType::EOD,
            serde_json::json!({
                "USD_SOFR": [0.04, 0.042, 0.045],
                "AAPL": 195.50
            }),
        );

        store.save_snapshot(&snap).unwrap();

        let loaded = store
            .load_snapshot("2025-06-15", SnapshotType::EOD)
            .unwrap();
        assert_eq!(loaded.snapshot_date, "2025-06-15");
        assert_eq!(loaded.snapshot_type, SnapshotType::EOD);

        // Non-existent
        let result = store.load_snapshot("2025-06-15", SnapshotType::Intraday);
        assert!(result.is_err());
    }

    #[test]
    fn test_not_found() {
        let store = temp_store();
        let result = store.get_trade(&ObjectId::from_string("nonexistent"));
        assert!(result.is_err());
    }

    #[test]
    fn test_persistence_across_reopen() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_str().unwrap().to_string();

        let trade_id;

        // Write
        {
            let store = EmbeddedStore::open(&path).unwrap();
            let trade = make_test_trade("Barclays", "structured");
            trade_id = trade.trade_id.clone();
            store.put_trade(&trade, "t1").unwrap();

            let snap = MarketSnapshot::new(
                "2025-12-31",
                SnapshotType::EOD,
                serde_json::json!({"VIX": 15.5}),
            );
            store.save_snapshot(&snap).unwrap();
        }

        // Re-open and read
        {
            let store = EmbeddedStore::open(&path).unwrap();

            let trade = store.get_trade(&trade_id).unwrap();
            assert_eq!(trade.counterparty, "Barclays");

            let trades = store.query_trades(&TradeFilter::new()).unwrap();
            assert_eq!(trades.len(), 1);

            let snap = store
                .load_snapshot("2025-12-31", SnapshotType::EOD)
                .unwrap();
            assert_eq!(snap.data["VIX"], 15.5);
        }
    }
}
