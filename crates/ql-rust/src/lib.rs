//! # ql-rust
//!
//! Convenience facade crate that re-exports the most commonly used types
//! from the QuantLib-Rust library. Import this single crate to get started:
//!
//! ```rust
//! use ql_rust::*;
//! ```

// Core types and errors
pub use ql_core::{QLError, QLResult};
pub use ql_core::quote::SimpleQuote;

// Time types
pub use ql_time::{
    BusinessDayConvention, Calendar, Date, DayCounter, Frequency, Month, Period, Schedule,
    TimeUnit, Weekday,
};

// Instruments
pub use ql_instruments::{
    BarrierOption, BarrierType, CallableBond, CallabilityScheduleEntry, CallabilityType,
    CapFloor, CapFloorType, Caplet, CompoundOption, ConvertibleBond,
    CreditDefaultSwap, CdsProtectionSide, CdsPremiumPeriod, Exercise,
    FixedRateBond, LookbackOption, LookbackType, OptionType, Payoff,
    Swaption, SwaptionType, SettlementType, SwapType, VanillaOption,
    VanillaSwap, VarianceSwap, AsianOption, AveragingType,
};

// Term structures
pub use ql_termstructures::{
    BlackConstantVol, BlackVarianceSurface, BlackVolTermStructure,
    DefaultProbabilityTermStructure, DepositRateHelper, DiscountCurve,
    FlatForward, FlatHazardRate, FlatZeroInflationCurve, LocalVolSurface,
    LocalVolTermStructure, PiecewiseDefaultCurve, PiecewiseYieldCurve,
    PiecewiseZeroInflationCurve, RateHelper, SabrSmileSection, SwapRateHelper,
    TermStructure, YieldTermStructure, ZeroCurve, ZeroCouponInflationSwap,
    ZeroInflationTermStructure, sabr_volatility,
};

// Cash flows
pub use ql_cashflows::{CashFlow, Leg};

// Pricing engines
pub use ql_pricingengines::{
    price_european, implied_volatility, AnalyticEuropeanResults,
    heston_price, HestonResult,
    price_swap, price_bond, SwapResults, BondResults,
    black_swaption, bachelier_swaption, SwaptionResult,
    black_cap_floor, bachelier_cap_floor, CapFloorResult,
    midpoint_cds_engine, CdsResult,
    price_callable_bond, CallableBondResult,
    price_convertible_bond, ConvertibleBondResult,
    analytic_lookback, LookbackResult,
    analytic_compound_option, CompoundOptionResult,
    price_variance_swap, VarianceSwapResult,
};

// Stochastic processes
pub use ql_processes::{GeneralizedBlackScholesProcess, HestonProcess};

// Models
pub use ql_models::HestonModel;

// Numerical methods
pub use ql_methods::{
    MCResult, mc_european, mc_barrier, mc_asian, mc_heston,
    FDResult, fd_black_scholes,
    LatticeResult, binomial_crr,
};

// Persistence
pub use ql_persistence::{
    Direction, EventType, InstrumentType, LifecycleEvent, MarketSnapshot, ObjectId,
    SnapshotType, Trade, TradeFilter, TradeStatus, VersionedObject,
    EmbeddedStore, InMemoryStore, ObjectStore, Persistable,
};

// Re-export serde_json for convenience (instrument_data uses Value)
pub use serde_json;
