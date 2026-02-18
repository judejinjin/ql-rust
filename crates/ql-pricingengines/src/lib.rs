//! # ql-pricingengines
//!
//! Pricing engine trait and implementations: analytic, Monte Carlo, finite difference, lattice.

pub mod american_engines;
pub mod analytic_european;
pub mod analytic_heston;
pub mod analytic_bates;
pub mod discounting;
pub mod swaption_engines;
pub mod cap_floor_engines;
pub mod cds_engine;
pub mod callable_bond_engine;
pub mod convertible_bond_engine;
pub mod longstaff_schwartz;
pub mod lookback_engine;
pub mod compound_engine;
pub mod variance_swap_engine;
pub mod merton_jump_diffusion;

// Re-exports
pub use american_engines::{
    barone_adesi_whaley, bjerksund_stensland, qd_plus_american, AmericanApproxResult,
};
pub use analytic_european::{price_european, implied_volatility, AnalyticEuropeanResults};
pub use analytic_heston::{heston_price, HestonResult};
pub use analytic_bates::{bates_price, bates_price_flat, BatesResult};
pub use discounting::{price_swap, price_bond, SwapResults, BondResults};
pub use swaption_engines::{black_swaption, bachelier_swaption, SwaptionResult};
pub use cap_floor_engines::{black_cap_floor, bachelier_cap_floor, CapFloorResult};
pub use cds_engine::{midpoint_cds_engine, CdsResult};
pub use callable_bond_engine::{price_callable_bond, CallableBondResult};
pub use convertible_bond_engine::{price_convertible_bond, ConvertibleBondResult};
pub use longstaff_schwartz::{mc_american_longstaff_schwartz, LSMBasis};
pub use lookback_engine::{analytic_lookback, LookbackResult};
pub use compound_engine::{analytic_compound_option, CompoundOptionResult};
pub use variance_swap_engine::{price_variance_swap, VarianceSwapResult};
pub use merton_jump_diffusion::{merton_jump_diffusion, MertonJDResult};
