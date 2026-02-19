//! # ql-currencies
//!
//! Currency definitions, money arithmetic, and exchange rate management.
//!
//! ## Overview
//!
//! | Module | Purpose |
//! |---|---|
//! | [`currency`] | ISO 4217 [`Currency`] type with rounding conventions |
//! | [`money`] | [`Money`] — amount + currency with arithmetic operations |
//! | [`exchange_rate`] | [`ExchangeRate`] for converting between currencies |
//!
//! ## Quick Start
//!
//! ```rust
//! use ql_currencies::{Currency, Money, ExchangeRate};
//!
//! let usd = Currency::usd();
//! let eur = Currency::eur();
//!
//! // Money arithmetic
//! let a = Money::new(1000.0, usd.clone());
//! let b = Money::new(500.0, usd.clone());
//! let total = a.add(&b).unwrap();
//! assert!((total.amount - 1500.0).abs() < 1e-10);
//!
//! // Exchange rates
//! let fx = ExchangeRate::new(eur, usd, 1.10).unwrap();
//! let converted = fx.exchange(&Money::new(100.0, Currency::eur())).unwrap();
//! assert!((converted.amount - 110.0).abs() < 1e-10);
//! ```

pub mod currency;
pub mod exchange_rate;
pub mod money;

pub use currency::Currency;
pub use exchange_rate::ExchangeRate;
pub use money::Money;
