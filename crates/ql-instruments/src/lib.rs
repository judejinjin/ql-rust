//! # ql-instruments
//!
//! Financial instrument definitions: options, swaps, bonds, forwards, and
//! exotic derivatives.
//!
//! ## Overview
//!
//! ### Vanilla Instruments
//! - [`VanillaOption`] — European/American equity options
//! - [`VanillaSwap`] — fixed-for-floating interest rate swaps
//! - [`FixedRateBond`] — coupon-bearing bonds
//!
//! ### Interest Rate Derivatives
//! - [`Swaption`] — options on interest rate swaps
//! - [`CapFloor`] — interest rate caps and floors
//!
//! ### Exotic Options
//! - [`BarrierOption`] — knock-in/knock-out barrier options
//! - [`AsianOption`] — arithmetic/geometric Asian options
//! - [`LookbackOption`] — floating/fixed strike lookback options
//! - [`CompoundOption`] — options on options
//! - [`VarianceSwap`] — variance and volatility swaps
//!
//! ### Credit
//! - [`CreditDefaultSwap`] — CDS with protection/premium legs
//! - [`CallableBond`] — bonds with embedded call/put options
//! - [`ConvertibleBond`] — bonds convertible to equity
//!
//! ## Common Types
//!
//! - [`OptionType`] — `Call` or `Put`
//! - [`Payoff`] — option payoff functions
//! - [`Exercise`] — exercise style (European, American, Bermudan)

pub mod payoff;
pub mod vanilla_option;
pub mod vanilla_swap;
pub mod bond;
pub mod barrier_option;
pub mod asian_option;
pub mod swaption;
pub mod cap_floor;
pub mod credit_default_swap;
pub mod callable_bond;
pub mod convertible_bond;
pub mod lookback_option;
pub mod compound_option;
pub mod variance_swap;

// Re-exports
pub use payoff::{OptionType, Payoff, Exercise};
pub use vanilla_option::VanillaOption;
pub use vanilla_swap::{VanillaSwap, SwapType};
pub use bond::FixedRateBond;
pub use barrier_option::{BarrierOption, BarrierType};
pub use asian_option::{AsianOption, AveragingType};
pub use swaption::{Swaption, SwaptionType, SettlementType};
pub use cap_floor::{CapFloor, CapFloorType, Caplet};
pub use credit_default_swap::{CreditDefaultSwap, CdsProtectionSide, CdsPremiumPeriod};
pub use callable_bond::{CallableBond, CallabilityType, CallabilityScheduleEntry};
pub use convertible_bond::ConvertibleBond;
pub use lookback_option::{LookbackOption, LookbackType};
pub use compound_option::CompoundOption;
pub use variance_swap::VarianceSwap;
