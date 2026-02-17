//! # ql-indexes
//!
//! Financial index trait and implementations: IBOR, overnight, swap, and inflation indexes.

pub mod ibor_index;
pub mod index;
pub mod interest_rate;
pub mod overnight_index;

pub use ibor_index::IborIndex;
pub use index::{Index, IndexManager, TimeSeries};
pub use interest_rate::{Compounding, InterestRate};
pub use overnight_index::OvernightIndex;
