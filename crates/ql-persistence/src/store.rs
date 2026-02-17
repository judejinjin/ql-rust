//! Abstract persistence traits.
//!
//! The [`ObjectStore`] trait defines a backend-agnostic API for storing and
//! retrieving trades, lifecycle events, and market snapshots with bitemporal
//! versioning. The [`Persistable`] marker trait constrains types that can be
//! stored through the `ObjectStore`.

use chrono::{DateTime, Utc};
use ql_core::QLResult;
use serde::{de::DeserializeOwned, Serialize};

use crate::domain::{
    LifecycleEvent, MarketSnapshot, ObjectId, SnapshotType, Trade, TradeFilter,
};

// ---------------------------------------------------------------------------
// Persistable marker trait
// ---------------------------------------------------------------------------

/// Marker trait for types that can be persisted through an [`ObjectStore`].
///
/// Any type implementing `Persistable` must be serializable, deserializable,
/// thread-safe, and provide a static string identifying its object type.
pub trait Persistable: Serialize + DeserializeOwned + Send + Sync {
    /// Return the canonical type name used as a storage discriminator.
    fn object_type() -> &'static str;
}

// Blanket implementations for domain types
impl Persistable for Trade {
    fn object_type() -> &'static str {
        "Trade"
    }
}

impl Persistable for LifecycleEvent {
    fn object_type() -> &'static str {
        "LifecycleEvent"
    }
}

impl Persistable for MarketSnapshot {
    fn object_type() -> &'static str {
        "MarketSnapshot"
    }
}

// ---------------------------------------------------------------------------
// ObjectStore trait
// ---------------------------------------------------------------------------

/// Abstract object store — swap implementations without changing business logic.
///
/// This is the core persistence API, designed for a SecDB/Beacon-style system:
///
/// * **Versioned objects** — each `put` creates a new version.
/// * **Bitemporal queries** — retrieve objects "as of" a past knowledge time.
/// * **Event sourcing** — append immutable lifecycle events and replay them.
/// * **Market snapshots** — save and load point-in-time market data.
///
/// # Implementations
///
/// * [`InMemoryStore`](crate::memory_store::InMemoryStore) — for testing.
/// * [`EmbeddedStore`](crate::embedded_store::EmbeddedStore) — redb-backed on-disk store.
pub trait ObjectStore: Send + Sync {
    // -- Trade CRUD --------------------------------------------------------

    /// Retrieve the current (latest) version of a trade.
    fn get_trade(&self, id: &ObjectId) -> QLResult<Trade>;

    /// Retrieve a trade as it was known at a specific point in time (bitemporal).
    fn get_trade_as_of(&self, id: &ObjectId, as_of: DateTime<Utc>) -> QLResult<Trade>;

    /// Save a new version of a trade. Returns the new version number.
    fn put_trade(&self, trade: &Trade, user: &str) -> QLResult<u64>;

    // -- Lifecycle events --------------------------------------------------

    /// Append an immutable lifecycle event. Returns the event's ObjectId.
    fn append_event(
        &self,
        trade_id: &ObjectId,
        event: &LifecycleEvent,
        user: &str,
    ) -> QLResult<ObjectId>;

    /// Replay all lifecycle events for a trade, in chronological order.
    fn replay_events(&self, trade_id: &ObjectId) -> QLResult<Vec<LifecycleEvent>>;

    // -- Trade queries -----------------------------------------------------

    /// Query trades matching a filter predicate.
    fn query_trades(&self, filter: &TradeFilter) -> QLResult<Vec<Trade>>;

    // -- Market snapshots --------------------------------------------------

    /// Save a market data snapshot. Returns the snapshot's ObjectId.
    fn save_snapshot(&self, snapshot: &MarketSnapshot) -> QLResult<ObjectId>;

    /// Load a market data snapshot for a given date and type.
    fn load_snapshot(
        &self,
        date: &str,
        snapshot_type: SnapshotType,
    ) -> QLResult<MarketSnapshot>;
}
