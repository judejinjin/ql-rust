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
//! - [`OISSwap`] — overnight indexed swaps (SOFR, ESTR, etc.)
//! - [`FixedRateBond`] — coupon-bearing bonds
//! - [`ZeroCouponBond`] — discount bonds (no coupons)
//! - [`AmortizingBond`] — bonds with scheduled principal repayments
//!
//! ### Interest Rate Derivatives
//! - [`Swaption`] — options on interest rate swaps
//! - [`CapFloor`] — interest rate caps and floors
//!
//! ### Exotic Options
//! - [`BarrierOption`] — knock-in/knock-out barrier options
//! - [`DoubleBarrierOption`] — double-barrier knock-in/knock-out options
//! - [`AsianOption`] — arithmetic/geometric Asian options
//! - [`LookbackOption`] — floating/fixed strike lookback options
//! - [`CompoundOption`] — options on options
//! - [`ChooserOption`] — simple chooser (as-you-like-it) options
//! - [`CliquetOption`] — cliquet / ratchet options
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
pub mod zero_coupon_bond;
pub mod amortizing_bond;
pub mod ois_swap;
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
pub mod inflation_linked_bond;
pub mod double_barrier_option;
pub mod chooser_option;
pub mod cliquet_option;

// Re-exports
pub use payoff::{OptionType, Payoff, Exercise};
pub use vanilla_option::VanillaOption;
pub use vanilla_swap::{VanillaSwap, SwapType};
pub use bond::FixedRateBond;
pub use zero_coupon_bond::ZeroCouponBond;
pub use amortizing_bond::AmortizingBond;
pub use ois_swap::OISSwap;
pub use barrier_option::{BarrierOption, BarrierType};
pub use double_barrier_option::{DoubleBarrierOption, DoubleBarrierType};
pub use asian_option::{AsianOption, AveragingType};
pub use swaption::{Swaption, SwaptionType, SettlementType};
pub use cap_floor::{CapFloor, CapFloorType, Caplet};
pub use credit_default_swap::{CreditDefaultSwap, CdsProtectionSide, CdsPremiumPeriod};
pub use callable_bond::{CallableBond, CallabilityType, CallabilityScheduleEntry};
pub use convertible_bond::ConvertibleBond;
pub use lookback_option::{LookbackOption, LookbackType};
pub use compound_option::CompoundOption;
pub use chooser_option::ChooserOption;
pub use cliquet_option::CliquetOption;
pub use variance_swap::VarianceSwap;
pub use inflation_linked_bond::InflationLinkedBond;
