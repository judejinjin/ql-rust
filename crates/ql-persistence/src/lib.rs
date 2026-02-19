//! # ql-persistence
//!
//! SecDB/Beacon-style persistence: object store trait, embedded (redb),
//! PostgreSQL, and analytics (DuckDB + Parquet) backends.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────┐
//! │  ObjectStore trait (abstract persistence)   │
//! └─────────────────┬───────────────────────────┘
//!                   │
//!        ┌──────────┼──────────┐
//!        │          │          │
//!   InMemoryStore  EmbeddedStore  (future: PostgresStore)
//!   (testing)      (redb, on-disk)
//! ```
//!
//! ## Domain Types
//!
//! * [`Trade`] — trade record with instrument data, counterparty, book, notional.
//! * [`LifecycleEvent`] — immutable event log entries (executed, amended, settled, etc.).
//! * [`MarketSnapshot`] — point-in-time capture of quotes, curves, vol surfaces.
//! * [`TradeFilter`] — predicate-based trade query DSL.
//! * [`ObjectId`] — universal object identifier.
//!
//! ## Stores
//!
//! * [`InMemoryStore`] — `HashMap`-backed, for unit tests and ephemeral sessions.
//! * [`EmbeddedStore`] — `redb`-backed, durable ACID storage with zero infrastructure.
//!
//! ## Quick Start
//!
//! ```rust
//! use ql_persistence::{InMemoryStore, ObjectStore, Trade,
//!                      InstrumentType, Direction};
//!
//! let store = InMemoryStore::new();
//! let trade = Trade::new(
//!     InstrumentType::Swap,
//!     serde_json::json!({"tenor": "5Y"}),
//!     "ACME",          // counterparty
//!     "RATES_1",       // book
//!     1_000_000.0,     // notional
//!     Direction::Buy,
//!     "2025-01-15",    // trade_date
//!     "2025-01-17",    // settlement_date
//!     "desk_user",     // created_by
//! );
//! let id = trade.trade_id.clone();
//! store.put_trade(&trade, "desk_user").unwrap();
//! let loaded = store.get_trade(&id).unwrap();
//! assert_eq!(loaded.notional, 1_000_000.0);
//! ```

pub mod domain;
#[cfg(feature = "redb")]
pub mod embedded_store;
pub mod memory_store;
pub mod store;

pub use domain::{
    Direction, EventType, InstrumentType, LifecycleEvent, MarketSnapshot, ObjectId,
    SnapshotType, Trade, TradeFilter, TradeStatus, VersionedObject,
};
#[cfg(feature = "redb")]
pub use embedded_store::EmbeddedStore;
pub use memory_store::InMemoryStore;
pub use store::{ObjectStore, Persistable};
