//! # ql-pricingengines
//!
//! Pricing engine implementations: analytic, Monte Carlo, finite difference,
//! and lattice engines for the full range of financial instruments.
//!
//! ## Overview
//!
//! ### Equity / FX Options
//! - [`analytic_european`] — Black-Scholes closed-form with full Greeks
//! - [`american_engines`] — Barone-Adesi-Whaley, Bjerksund-Stensland, QD+
//! - [`analytic_heston`] — Heston (1993) semi-analytic pricing
//! - [`analytic_bates`] — Bates model (Heston + jumps)
//! - [`merton_jump_diffusion()`] — Merton (1976) jump-diffusion
//! - [`longstaff_schwartz`] — LSM Monte Carlo for American options
//! - [`multi_asset`] — basket options, spread options, exchange options
//! - [`lookback_engine`] — analytic lookback options
//! - [`compound_engine`] — analytic compound options
//! - [`variance_swap_engine`] — variance/volatility swap pricing
//!
//! ### Interest Rate Derivatives
//! - [`discounting`] — swap and bond discounting engines
//! - [`swaption_engines`] — Black and Bachelier swaption engines
//! - [`cap_floor_engines`] — cap/floor pricing (Black, Bachelier)
//! - [`hw_analytic`] — Hull-White analytic bond options, caplets, swaptions
//! - [`tree_swaption()`] — tree/FD/MC engines for HW swaptions
//!
//! ### Credit
//! - [`cds_engine`] — midpoint CDS engine
//! - [`callable_bond_engine`] — callable bond pricing
//! - [`convertible_bond_engine`] — convertible bond pricing
//! - [`credit_models`] — Gaussian copula, nth-to-default, CDS options

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
pub mod multi_asset;
pub mod hw_analytic;

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
pub use multi_asset::{
    mc_basket, stulz_max_call, stulz_min_call, kirk_spread_call, kirk_spread_put,
    margrabe_exchange, BasketType, BasketResult,
};
pub use hw_analytic::{
    hw_bond_option, hw_caplet, hw_floorlet, hw_jamshidian_swaption, HWAnalyticResult,
};
pub mod tree_swaption;
pub use tree_swaption::{
    tree_bond_price, tree_swaption, tree_bermudan_swaption, tree_cap_floor,
    fd_hw_swaption, mc_hw_cap_floor,
    TreeResult, FdResult, McHwResult,
};
pub mod credit_models;
pub use credit_models::{
    GaussianCopulaLHP, NtdResult, nth_to_default_mc, cds_option_black,
};
