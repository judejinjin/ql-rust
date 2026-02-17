//! # ql-termstructures
//!
//! Term structure traits and implementations: yield curves, volatility surfaces,
//! default probability curves, inflation curves, and bootstrapping framework.

pub mod term_structure;
pub mod yield_term_structure;
pub mod yield_curves;
pub mod bootstrap;
pub mod vol_term_structure;
pub mod sabr;

// Re-exports
pub use term_structure::TermStructure;
pub use yield_term_structure::YieldTermStructure;
pub use yield_curves::{FlatForward, DiscountCurve, ZeroCurve};
pub use bootstrap::{
    RateHelper, DepositRateHelper, SwapRateHelper, PiecewiseYieldCurve,
};
pub use vol_term_structure::{
    BlackVolTermStructure, LocalVolTermStructure,
    BlackConstantVol, BlackVarianceSurface, LocalVolSurface,
};
pub use sabr::{sabr_volatility, SabrSmileSection};
