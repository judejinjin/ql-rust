//! # ql-termstructures
//!
//! Term structure traits and implementations: yield curves, volatility surfaces,
//! default probability curves, inflation curves, and bootstrapping framework.
//!
//! ## Overview
//!
//! ### Yield Curves
//! - [`FlatForward`] — constant forward rate
//! - [`DiscountCurve`], [`ZeroCurve`] — interpolated from market data
//! - [`PiecewiseYieldCurve`] — bootstrapped from rate helpers
//! - [`NelsonSiegelFitting`], [`SvenssonFitting`] — parametric curve fitting
//!
//! ### Volatility Surfaces
//! - [`BlackConstantVol`], [`BlackVarianceSurface`] — Black vol term structures
//! - [`SabrSmileSection`], [`SviSmileSection`], [`ZabrSmileSection`] — smile models
//!
//! ### Credit & Inflation
//! - [`FlatHazardRate`], [`PiecewiseDefaultCurve`] — default probability curves
//! - [`FlatZeroInflationCurve`], [`PiecewiseZeroInflationCurve`] — inflation curves
//!
//! ## Quick Start
//!
//! ```rust
//! use ql_termstructures::{FlatForward, YieldTermStructure};
//! use ql_time::{Date, Month, DayCounter};
//!
//! let today = Date::from_ymd(2025, Month::January, 15);
//! let curve = FlatForward::new(today, 0.05, DayCounter::Actual365Fixed);
//! let df = curve.discount(today + 365);
//! assert!((df - (-0.05_f64).exp()).abs() < 0.01);
//! ```

pub mod term_structure;
pub mod yield_term_structure;
pub mod yield_curves;
pub mod yield_curves_extended;
pub mod bootstrap;
pub mod bootstrap_extended;
pub mod bootstrap_multicurve;
pub mod nelson_siegel;
pub mod fitted_bond_curve_extended;
pub mod vol_term_structure;
pub mod sabr;
pub mod svi;
pub mod zabr;
pub mod optionlet_stripper;
pub mod vol_interpolation;
pub mod default_term_structure;
pub mod inflation_term_structure;
pub mod swaption_vol;
pub mod andreasen_huge;
pub mod noarb_sabr;
pub mod kahale_smile_section;
pub mod spreaded_smile_section;
pub mod abcd_vol_term_structure;
pub mod gaussian1d_smile_section;
pub mod ufr_term_structure;
pub mod global_bootstrap;
pub mod seasonality;
pub mod inflation_vol;
pub mod heston_vol_surface;
pub mod xccy_helpers;

// Re-exports
pub use term_structure::TermStructure;
pub use yield_term_structure::YieldTermStructure;
pub use yield_curves::{FlatForward, DiscountCurve, ZeroCurve};
pub use yield_curves_extended::{
    CompositeZeroYieldStructure, ImpliedTermStructure, ForwardCurve,
    UltimateForwardTermStructure, SpreadedTermStructure, QuantoTermStructure,
};
pub use bootstrap::{
    RateHelper, DepositRateHelper, SwapRateHelper, PiecewiseYieldCurve,
};
pub use bootstrap_extended::{
    OISRateHelper, BondHelper, FuturesRateHelper, FRAHelper,
};
pub use nelson_siegel::{
    NelsonSiegelFitting, SvenssonFitting, FittedBondDiscountCurve, FittingMethod,
};
pub use fitted_bond_curve_extended::{
    PolynomialDiscountCurve, ExponentialSplineCurve,
};
pub use vol_term_structure::{
    BlackVolTermStructure, LocalVolTermStructure,
    BlackConstantVol, BlackVarianceSurface, LocalVolSurface,
};
pub use sabr::{sabr_volatility, SabrSmileSection};
pub use noarb_sabr::{NoArbSabrSmileSection, check_smile_arbitrage, kahale_call_prices, ArbitrageCheckResult};
pub use svi::{svi_volatility, svi_total_variance, svi_calibrate, SviSmileSection};
pub use zabr::{zabr_volatility, ZabrSmileSection};
pub use optionlet_stripper::{strip_optionlet_volatilities, interpolate_optionlet_vol, StrippedOptionletVolatilities};
pub use vol_interpolation::{BlackVarianceCurve, SmileSectionSurface};
pub use default_term_structure::{
    DefaultProbabilityTermStructure, FlatHazardRate, PiecewiseDefaultCurve,
};
pub use inflation_term_structure::{
    ZeroInflationTermStructure, FlatZeroInflationCurve,
    PiecewiseZeroInflationCurve, ZeroCouponInflationSwap,
    YoYInflationTermStructure, FlatYoYInflationCurve, YearOnYearInflationSwap,
};
pub use swaption_vol::{
    SwaptionVolatilityStructure, SwaptionConstantVol, SwaptionVolMatrix,
    SwaptionVolCube, SabrSwaptionVolCube, SabrParams,
    CapFloorTermVolStructure, ConstantCapFloorTermVol, CapFloorTermVolSurface,
};
pub use andreasen_huge::{
    AndreasenHugeVolSurface, AndreasenHugeConfig, AndreasenHugeCalibrationResult,
    VolQuote, andreasen_huge_calibrate,
};
pub use kahale_smile_section::KahaleSmileSection;
pub use spreaded_smile_section::{
    SpreadType, FlatVolQuote, TabulatedSmileQuote,
    SpreadedFlatSmileSection, SpreadedSmileSection,
};
pub use abcd_vol_term_structure::{AbcdParameters, AbcdVolTermStructure, integrate_abcd_sq};
pub use gaussian1d_smile_section::{HullWhiteParams, Gaussian1dSmileSection};
pub use global_bootstrap::{GlobalBootstrapHelper, GlobalHelperType, GlobalBootstrapResult, global_bootstrap};
pub use seasonality::{Seasonality, SeasonalityType, estimate_seasonality};
pub use inflation_vol::{
    CpiVolSurface, YoYOptionletVolSurface, ConstantCpiVol, ConstantYoYVol,
    strip_yoy_optionlet_vols,
};
pub use heston_vol_surface::{HestonBlackVolSurface, HestonVolParams};
pub use xccy_helpers::{
    CrossCurrencyBasisSwapHelper, TenorBasisSwapHelper, XccyBootstrapResult,
    bootstrap_xccy_curve, bootstrap_tenor_basis_curve,
};
pub use ufr_term_structure::{
    SmithWilsonParams, UltimateForwardTermStructure as SmithWilsonCurve,
    OisFutureRateHelper,
};
