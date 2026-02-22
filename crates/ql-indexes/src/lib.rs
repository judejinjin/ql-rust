//! # ql-indexes
//!
//! Financial index trait and implementations: IBOR, overnight, swap, and
//! inflation indexes.
//!
//! ## Overview
//!
//! | Module | Purpose |
//! |---|---|
//! | [`index`] | [`Index`] trait, [`IndexManager`] global fixing store, [`TimeSeries`] |
//! | [`interest_rate`] | [`InterestRate`] with compounding and day-count conventions |
//! | [`ibor_index`] | [`IborIndex`] — Euribor, USD LIBOR, etc. |
//! | [`overnight_index`] | [`OvernightIndex`] — SOFR, ESTR, SONIA, TONA |
//! | [`swap_index`] | [`SwapIndex`] — EUR/USD par swap rate indexes |
//!
//! ## Quick Start
//!
//! ```rust
//! use ql_indexes::{InterestRate, Compounding, IborIndex, OvernightIndex, Index};
//! use ql_time::DayCounter;
//!
//! // Compute a discount factor
//! let rate = InterestRate::new(0.05, DayCounter::Actual365Fixed,
//!                              Compounding::Continuous, 1);
//! let df = rate.discount_factor(1.0).unwrap();
//! assert!((df - (-0.05_f64).exp()).abs() < 1e-12);
//!
//! // Create an index
//! let sofr = OvernightIndex::sofr();
//! assert_eq!(sofr.name(), "SOFR");
//! ```

pub mod ibor_index;
pub mod index;
pub mod inflation_index;
pub mod interest_rate;
pub mod overnight_index;
pub mod swap_index;
pub mod equity_index;
pub mod bma_index;

pub use ibor_index::IborIndex;
pub use index::{Index, IndexManager, TimeSeries};
pub use inflation_index::{InflationIndex, CpiInterpolation};
pub use interest_rate::{Compounding, InterestRate};
pub use overnight_index::OvernightIndex;
pub use swap_index::SwapIndex;
pub use equity_index::EquityIndex;
pub use bma_index::{BmaIndex, BmaSwapFixedLeg};
