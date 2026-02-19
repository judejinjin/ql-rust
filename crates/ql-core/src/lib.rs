//! # ql-core
//!
//! Foundational types, error handling, observer pattern, lazy evaluation,
//! relinkable handles, and global settings for the ql-rust quantitative
//! finance library.
//!
//! ## Overview
//!
//! This crate provides the core infrastructure that every other `ql-*` crate
//! depends on:
//!
//! | Module | Purpose |
//! |---|---|
//! | [`errors`] | [`QLError`] enum and [`QLResult<T>`] alias |
//! | [`types`] | Semantic type aliases (`Real`, `Rate`, `DiscountFactor`, …) |
//! | [`quote`] | Market-observable values ([`quote::SimpleQuote`]) |
//! | [`observable`] | Observer / Observable pattern for dependency tracking |
//! | [`handle`] | Relinkable shared references ([`handle::Handle<T>`]) |
//! | [`lazy`] | Lazy evaluation & caching ([`lazy::LazyCache`], [`lazy::Cached<T>`]) |
//! | [`settings`] | Global evaluation-date singleton ([`settings::Settings`]) |
//!
//! ## Quick Start
//!
//! ```rust
//! use ql_core::{QLResult, QLError};
//! use ql_core::quote::{Quote, SimpleQuote};
//!
//! // Create a market quote and read its value
//! let spot = SimpleQuote::new(100.0);
//! assert_eq!(spot.value().unwrap(), 100.0);
//!
//! // All library functions return QLResult<T>
//! fn safe_sqrt(x: f64) -> QLResult<f64> {
//!     if x < 0.0 {
//!         Err(QLError::NegativeValue { quantity: "input", value: x })
//!     } else {
//!         Ok(x.sqrt())
//!     }
//! }
//! assert!(safe_sqrt(-1.0).is_err());
//! ```

pub mod errors;
pub mod handle;
pub mod lazy;
pub mod observable;
pub mod quote;
pub mod settings;
pub mod types;

pub use errors::{QLError, QLResult};
pub use types::*;
