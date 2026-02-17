//! # ql-instruments
//!
//! Financial instrument trait and implementations: options, swaps, bonds, forwards.

pub mod payoff;
pub mod vanilla_option;
pub mod vanilla_swap;
pub mod bond;

// Re-exports
pub use payoff::{OptionType, Payoff, Exercise};
pub use vanilla_option::VanillaOption;
pub use vanilla_swap::{VanillaSwap, SwapType};
pub use bond::FixedRateBond;
