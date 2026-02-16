//! # ql-core
//!
//! Foundational types, error handling, observer pattern, lazy evaluation,
//! relinkable handles, and global settings for the ql-rust quantitative
//! finance library.

pub mod errors;
pub mod handle;
pub mod lazy;
pub mod observable;
pub mod quote;
pub mod settings;
pub mod types;

pub use errors::{QLError, QLResult};
pub use types::*;
