//! Domain types for the persistence layer.
//!
//! These types model trades, lifecycle events, market snapshots,
//! and query filters in a SecDB/Beacon-inspired object model.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// ObjectId
// ---------------------------------------------------------------------------

/// Universal object identifier used as a key in the object store.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ObjectId(pub String);

impl ObjectId {
    /// Create a new random ObjectId backed by a UUID v4.
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Create an ObjectId from a string.
    pub fn from_string(s: &str) -> Self {
        Self(s.to_string())
    }

    /// Return the inner string reference.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for ObjectId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ObjectId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Trade
// ---------------------------------------------------------------------------

/// Direction of a trade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    Buy,
    Sell,
}

/// Current status of a trade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeStatus {
    Active,
    Amended,
    Novated,
    Terminated,
    Matured,
    Settled,
}

/// Type of instrument for query filtering.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InstrumentType {
    Bond,
    Swap,
    Option,
    Future,
    FRA,
    Cap,
    Floor,
    Swaption,
    CDS,
    CallableBond,
    ConvertibleBond,
    VarianceSwap,
    Other(String),
}

/// A trade record — the central business object for the persistence layer.
///
/// A `Trade` wraps an instrument description (stored as a serialized JSON value)
/// with business metadata: counterparty, book, notional, direction, status, dates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Unique trade identifier.
    pub trade_id: ObjectId,
    /// Type of the underlying instrument.
    pub instrument_type: InstrumentType,
    /// Serialized instrument description / parameters.
    pub instrument_data: serde_json::Value,
    /// Counterparty name or LEI.
    pub counterparty: String,
    /// Trading book.
    pub book: String,
    /// Notional amount.
    pub notional: f64,
    /// Buy or Sell.
    pub direction: Direction,
    /// Current lifecycle status.
    pub status: TradeStatus,
    /// Date the trade was executed.
    pub trade_date: String,
    /// Settlement date.
    pub settlement_date: String,
    /// Version number (incremented on each amendment).
    pub version: u64,
    /// Timestamp when this version was created.
    pub valid_from: DateTime<Utc>,
    /// Timestamp when this version was superseded (infinity if current).
    pub valid_to: Option<DateTime<Utc>>,
    /// User who created / last modified this version.
    pub created_by: String,
}

impl Trade {
    /// Create a new trade with initial metadata.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        instrument_type: InstrumentType,
        instrument_data: serde_json::Value,
        counterparty: &str,
        book: &str,
        notional: f64,
        direction: Direction,
        trade_date: &str,
        settlement_date: &str,
        user: &str,
    ) -> Self {
        Self {
            trade_id: ObjectId::new(),
            instrument_type,
            instrument_data,
            counterparty: counterparty.to_string(),
            book: book.to_string(),
            notional,
            direction,
            status: TradeStatus::Active,
            trade_date: trade_date.to_string(),
            settlement_date: settlement_date.to_string(),
            version: 1,
            valid_from: Utc::now(),
            valid_to: None,
            created_by: user.to_string(),
        }
    }

    /// Check if this version is the current (non-superseded) version.
    pub fn is_current(&self) -> bool {
        self.valid_to.is_none()
    }
}

// ---------------------------------------------------------------------------
// LifecycleEvent
// ---------------------------------------------------------------------------

/// Types of lifecycle events that can occur on a trade.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EventType {
    Executed,
    Amended,
    Novated,
    Terminated,
    Matured,
    CashSettled,
    MarginCall,
    Custom(String),
}

/// A lifecycle event — an immutable record of something that happened to a trade.
///
/// Events are append-only and form the authoritative audit trail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleEvent {
    /// Unique event identifier.
    pub event_id: ObjectId,
    /// The trade this event applies to.
    pub trade_id: ObjectId,
    /// Type of event.
    pub event_type: EventType,
    /// Business date of the event.
    pub event_date: String,
    /// Timestamp when the event was recorded.
    pub entered_at: DateTime<Utc>,
    /// User who entered the event.
    pub entered_by: String,
    /// Arbitrary payload (amendment details, settlement amounts, etc.).
    pub payload: serde_json::Value,
    /// Resulting trade version after this event (if applicable).
    pub resulting_version: Option<u64>,
}

impl LifecycleEvent {
    /// Create a new lifecycle event.
    pub fn new(
        trade_id: &ObjectId,
        event_type: EventType,
        event_date: &str,
        user: &str,
        payload: serde_json::Value,
    ) -> Self {
        Self {
            event_id: ObjectId::new(),
            trade_id: trade_id.clone(),
            event_type,
            event_date: event_date.to_string(),
            entered_at: Utc::now(),
            entered_by: user.to_string(),
            payload,
            resulting_version: None,
        }
    }

    /// Create a new event with a resulting version.
    pub fn with_version(mut self, version: u64) -> Self {
        self.resulting_version = Some(version);
        self
    }
}

// ---------------------------------------------------------------------------
// MarketSnapshot
// ---------------------------------------------------------------------------

/// Type of market data snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SnapshotType {
    /// End-of-day official closing data.
    EOD,
    /// Intraday snapshot.
    Intraday,
    /// Custom / ad-hoc snapshot.
    Custom,
}

/// A market data snapshot — a point-in-time capture of quotes, curves, and surfaces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketSnapshot {
    /// Unique snapshot identifier.
    pub snapshot_id: ObjectId,
    /// Business date for the snapshot.
    pub snapshot_date: String,
    /// Exact timestamp when the snapshot was taken.
    pub snapshot_time: DateTime<Utc>,
    /// Type of snapshot.
    pub snapshot_type: SnapshotType,
    /// The market data payload: quotes, curve points, vol surface entries, etc.
    pub data: serde_json::Value,
}

impl MarketSnapshot {
    /// Create a new market data snapshot.
    pub fn new(
        snapshot_date: &str,
        snapshot_type: SnapshotType,
        data: serde_json::Value,
    ) -> Self {
        Self {
            snapshot_id: ObjectId::new(),
            snapshot_date: snapshot_date.to_string(),
            snapshot_time: Utc::now(),
            snapshot_type,
            data,
        }
    }
}

// ---------------------------------------------------------------------------
// TradeFilter
// ---------------------------------------------------------------------------

/// Query filter for searching trades.
///
/// All fields are optional — `None` means "match any".
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TradeFilter {
    /// Filter by trade status.
    pub status: Option<TradeStatus>,
    /// Filter by counterparty name (exact match).
    pub counterparty: Option<String>,
    /// Filter by trading book.
    pub book: Option<String>,
    /// Filter by instrument type.
    pub instrument_type: Option<InstrumentType>,
    /// Filter by trade date range (inclusive).
    pub date_from: Option<String>,
    /// Filter by trade date range (inclusive).
    pub date_to: Option<String>,
}

impl TradeFilter {
    /// Create an empty filter that matches all trades.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the status filter.
    pub fn with_status(mut self, status: TradeStatus) -> Self {
        self.status = Some(status);
        self
    }

    /// Set the counterparty filter.
    pub fn with_counterparty(mut self, counterparty: &str) -> Self {
        self.counterparty = Some(counterparty.to_string());
        self
    }

    /// Set the book filter.
    pub fn with_book(mut self, book: &str) -> Self {
        self.book = Some(book.to_string());
        self
    }

    /// Set the instrument type filter.
    pub fn with_instrument_type(mut self, instrument_type: InstrumentType) -> Self {
        self.instrument_type = Some(instrument_type);
        self
    }

    /// Check whether a trade matches this filter.
    pub fn matches(&self, trade: &Trade) -> bool {
        if let Some(ref s) = self.status {
            if trade.status != *s {
                return false;
            }
        }
        if let Some(ref c) = self.counterparty {
            if trade.counterparty != *c {
                return false;
            }
        }
        if let Some(ref b) = self.book {
            if trade.book != *b {
                return false;
            }
        }
        if let Some(ref it) = self.instrument_type {
            if trade.instrument_type != *it {
                return false;
            }
        }
        if let Some(ref from) = self.date_from {
            if trade.trade_date < *from {
                return false;
            }
        }
        if let Some(ref to) = self.date_to {
            if trade.trade_date > *to {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// VersionedObject — generic bitemporal wrapper
// ---------------------------------------------------------------------------

/// A versioned wrapper for any persistable object, providing bitemporal metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionedObject<T> {
    /// The unique identifier of the object.
    pub id: ObjectId,
    /// The version number (monotonically increasing).
    pub version: u64,
    /// Timestamp when this version became effective.
    pub valid_from: DateTime<Utc>,
    /// Timestamp when this version was superseded (`None` = current).
    pub valid_to: Option<DateTime<Utc>>,
    /// Who created this version.
    pub created_by: String,
    /// The actual object data.
    pub data: T,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_object_id_creation() {
        let id1 = ObjectId::new();
        let id2 = ObjectId::new();
        assert_ne!(id1, id2);
        assert!(!id1.as_str().is_empty());
    }

    #[test]
    fn test_object_id_from_str() {
        let id = ObjectId::from_string("TRD-001");
        assert_eq!(id.as_str(), "TRD-001");
        assert_eq!(format!("{}", id), "TRD-001");
    }

    #[test]
    fn test_trade_creation() {
        let trade = Trade::new(
            InstrumentType::Swap,
            serde_json::json!({"fixed_rate": 0.035, "tenor": "5Y"}),
            "JPMorgan",
            "rates_nyc",
            50_000_000.0,
            Direction::Buy,
            "2025-06-15",
            "2025-06-17",
            "trader1",
        );
        assert_eq!(trade.status, TradeStatus::Active);
        assert_eq!(trade.version, 1);
        assert!(trade.is_current());
        assert_eq!(trade.counterparty, "JPMorgan");
        assert_eq!(trade.notional, 50_000_000.0);
    }

    #[test]
    fn test_trade_serialization_roundtrip() {
        let trade = Trade::new(
            InstrumentType::Bond,
            serde_json::json!({"coupon": 0.05}),
            "GoldmanSachs",
            "credit",
            10_000_000.0,
            Direction::Sell,
            "2025-01-10",
            "2025-01-12",
            "trader2",
        );
        let json = serde_json::to_string(&trade).unwrap();
        let restored: Trade = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.trade_id, trade.trade_id);
        assert_eq!(restored.counterparty, "GoldmanSachs");
        assert_eq!(restored.notional, 10_000_000.0);
    }

    #[test]
    fn test_trade_json_bytes_roundtrip() {
        let trade = Trade::new(
            InstrumentType::Option,
            serde_json::json!({"strike": 100.0, "type": "Call"}),
            "Citadel",
            "equity_vol",
            1_000_000.0,
            Direction::Buy,
            "2025-03-01",
            "2025-03-03",
            "trader3",
        );
        let bytes = serde_json::to_vec(&trade).unwrap();
        let restored: Trade = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(restored.trade_id, trade.trade_id);
        assert_eq!(restored.notional, 1_000_000.0);
    }

    #[test]
    fn test_lifecycle_event() {
        let trade_id = ObjectId::from_string("TRD-001");
        let event = LifecycleEvent::new(
            &trade_id,
            EventType::Executed,
            "2025-06-15",
            "trader1",
            serde_json::json!({"price": 99.5}),
        );
        assert_eq!(event.trade_id, trade_id);
        assert_eq!(event.event_type, EventType::Executed);

        let event2 = event.clone().with_version(1);
        assert_eq!(event2.resulting_version, Some(1));
    }

    #[test]
    fn test_lifecycle_event_serialization() {
        let event = LifecycleEvent::new(
            &ObjectId::from_string("TRD-002"),
            EventType::Amended,
            "2025-07-01",
            "risk_mgr",
            serde_json::json!({"field": "notional", "old": 50e6, "new": 40e6}),
        );
        let bytes = serde_json::to_vec(&event).unwrap();
        let restored: LifecycleEvent = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(restored.event_type, EventType::Amended);
    }

    #[test]
    fn test_market_snapshot() {
        let snap = MarketSnapshot::new(
            "2025-06-15",
            SnapshotType::EOD,
            serde_json::json!({
                "quotes": {"AAPL": 195.50, "MSFT": 420.00},
                "curves": {"USD_SOFR": [0.04, 0.042, 0.045]}
            }),
        );
        assert_eq!(snap.snapshot_date, "2025-06-15");
        assert_eq!(snap.snapshot_type, SnapshotType::EOD);
    }

    #[test]
    fn test_trade_filter() {
        let trade = Trade::new(
            InstrumentType::Swap,
            serde_json::json!({}),
            "JPMorgan",
            "rates_nyc",
            50e6,
            Direction::Buy,
            "2025-06-15",
            "2025-06-17",
            "trader1",
        );

        // Empty filter matches everything
        assert!(TradeFilter::new().matches(&trade));

        // Matching filters
        assert!(TradeFilter::new().with_status(TradeStatus::Active).matches(&trade));
        assert!(TradeFilter::new().with_counterparty("JPMorgan").matches(&trade));
        assert!(TradeFilter::new().with_book("rates_nyc").matches(&trade));
        assert!(TradeFilter::new().with_instrument_type(InstrumentType::Swap).matches(&trade));

        // Non-matching filters
        assert!(!TradeFilter::new().with_counterparty("Citadel").matches(&trade));
        assert!(!TradeFilter::new().with_status(TradeStatus::Terminated).matches(&trade));
        assert!(!TradeFilter::new().with_book("credit").matches(&trade));
    }

    #[test]
    fn test_trade_filter_date_range() {
        let trade = Trade::new(
            InstrumentType::Bond,
            serde_json::json!({}),
            "GS",
            "credit",
            10e6,
            Direction::Buy,
            "2025-06-15",
            "2025-06-17",
            "t1",
        );

        let filter_in = TradeFilter {
            date_from: Some("2025-01-01".to_string()),
            date_to: Some("2025-12-31".to_string()),
            ..Default::default()
        };
        assert!(filter_in.matches(&trade));

        let filter_out = TradeFilter {
            date_from: Some("2025-07-01".to_string()),
            ..Default::default()
        };
        assert!(!filter_out.matches(&trade));
    }

    #[test]
    fn test_versioned_object() {
        let obj = VersionedObject {
            id: ObjectId::from_string("OBJ-001"),
            version: 3,
            valid_from: Utc::now(),
            valid_to: None,
            created_by: "system".to_string(),
            data: serde_json::json!({"rate": 0.05}),
        };
        let json = serde_json::to_string(&obj).unwrap();
        let restored: VersionedObject<serde_json::Value> = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.version, 3);
        assert!(restored.valid_to.is_none());
    }
}
