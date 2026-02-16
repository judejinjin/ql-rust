//! Fundamental type aliases for the quantitative finance library.
//!
//! These aliases make signatures self-documenting and match QuantLib conventions.

/// Floating-point real number.
pub type Real = f64;

/// Interest rate as a decimal (e.g., 0.05 for 5%).
pub type Rate = f64;

/// Spread over a reference rate, in decimal.
pub type Spread = f64;

/// Discount factor in (0, 1].
pub type DiscountFactor = f64;

/// Volatility as a decimal (e.g., 0.20 for 20%).
pub type Volatility = f64;

/// Year fraction / time in years.
pub type Time = f64;

/// Non-negative integer.
pub type Natural = u32;

/// Signed integer.
pub type Integer = i32;

/// Size / count.
pub type Size = usize;
