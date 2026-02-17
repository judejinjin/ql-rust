//! # ql-instruments
//!
//! Financial instrument trait and implementations: options, swaps, bonds, forwards.

pub mod payoff;
pub mod vanilla_option;
pub mod vanilla_swap;
pub mod bond;
pub mod barrier_option;
pub mod asian_option;
pub mod swaption;
pub mod cap_floor;

// Re-exports
pub use payoff::{OptionType, Payoff, Exercise};
pub use vanilla_option::VanillaOption;
pub use vanilla_swap::{VanillaSwap, SwapType};
pub use bond::FixedRateBond;
pub use barrier_option::{BarrierOption, BarrierType};
pub use asian_option::{AsianOption, AveragingType};
pub use swaption::{Swaption, SwaptionType, SettlementType};
pub use cap_floor::{CapFloor, CapFloorType, Caplet};
