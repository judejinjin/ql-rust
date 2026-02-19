//! # ql-rust
//!
//! **QuantLib-Rust** — a comprehensive quantitative finance library for Rust,
//! faithfully reimplementing QuantLib's instrument/engine/term-structure model
//! with idiomatic Rust patterns.
//!
//! This facade crate re-exports the most commonly used types from the 14
//! underlying `ql-*` crates. Import this single crate to get started:
//!
//! ```rust,no_run
//! use ql_rust::*;
//! ```
//!
//! ## Feature Flags
//!
//! | Flag | Default | Effect |
//! |---|---|---|
//! | `parallel` | off | Enable `rayon` parallelism in Monte Carlo / FD methods |
//! | `redb` | off | Enable embedded `redb` persistence backend |
//!
//! ## Crate Map
//!
//! | Crate | Purpose |
//! |---|---|
//! | `ql-core` | Error types, type aliases, observer pattern, handles |
//! | `ql-time` | Dates, calendars, day counters, schedules |
//! | `ql-math` | Interpolation, root-finding, optimization, distributions |
//! | `ql-currencies` | ISO 4217 currencies, money, exchange rates |
//! | `ql-indexes` | IBOR/overnight indexes, interest rate conventions |
//! | `ql-termstructures` | Yield curves, vol surfaces, credit/inflation curves |
//! | `ql-cashflows` | Coupons, legs, cash-flow analytics |
//! | `ql-instruments` | Options, swaps, bonds, exotic derivatives |
//! | `ql-processes` | Stochastic processes (GBM, Heston, Hull-White, …) |
//! | `ql-models` | Calibrated models (Heston, HW, Bates, LMM, …) |
//! | `ql-methods` | MC simulation, FD solvers, lattice methods |
//! | `ql-pricingengines` | Analytic, MC, FD, and tree pricing engines |
//! | `ql-persistence` | Trade storage, lifecycle events, market snapshots |
//! | `ql-cli` | Command-line interface for pricing and curve ops |

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
    // Core yield curves
    BlackConstantVol, BlackVarianceSurface, BlackVolTermStructure,
    DefaultProbabilityTermStructure, DepositRateHelper, DiscountCurve,
    FlatForward, FlatHazardRate, FlatZeroInflationCurve, LocalVolSurface,
    LocalVolTermStructure, PiecewiseDefaultCurve, PiecewiseYieldCurve,
    PiecewiseZeroInflationCurve, RateHelper, SabrSmileSection, SwapRateHelper,
    TermStructure, YieldTermStructure, ZeroCurve, ZeroCouponInflationSwap,
    ZeroInflationTermStructure, sabr_volatility,
    // Extended yield curves (Phase 24)
    CompositeZeroYieldStructure, ImpliedTermStructure, ForwardCurve,
    UltimateForwardTermStructure, SpreadedTermStructure,
    // Extended rate helpers (Phase 24)
    OISRateHelper, BondHelper, FuturesRateHelper, FRAHelper,
    // Nelson-Siegel / Svensson fitting (Phase 24)
    NelsonSiegelFitting, SvenssonFitting, FittedBondDiscountCurve, FittingMethod,
    // SVI / ZABR smile models (Phase 18)
    SviSmileSection, ZabrSmileSection, svi_volatility, svi_total_variance,
    svi_calibrate, zabr_volatility,
    // Optionlet stripping (Phase 18)
    strip_optionlet_volatilities, interpolate_optionlet_vol, StrippedOptionletVolatilities,
    // Vol interpolation (Phase 18)
    BlackVarianceCurve, SmileSectionSurface,
};

// Cash flows
pub use ql_cashflows::{
    CashFlow, Leg,
    // CMS coupons (Phase 23)
    CmsCoupon, cms_convexity_adjustment, cms_caplet_price,
    // Digital / capped-floored / range-accrual / sub-period coupons (Phase 23)
    DigitalCoupon, CapFlooredCoupon, RangeAccrualCoupon, SubPeriodCoupon, SubPeriodType,
    // Extended analytics (Phase 23)
    convexity, modified_duration, dv01, z_spread, atm_rate,
    time_bucketed_cashflows, TimeBucket,
};

// Pricing engines
pub use ql_pricingengines::{
    // Analytic European
    price_european, implied_volatility, AnalyticEuropeanResults,
    // Heston
    heston_price, HestonResult,
    // Discounting (swap, bond)
    price_swap, price_bond, SwapResults, BondResults,
    // Swaptions
    black_swaption, bachelier_swaption, SwaptionResult,
    // Caps/floors
    black_cap_floor, bachelier_cap_floor, CapFloorResult,
    // Credit
    midpoint_cds_engine, CdsResult,
    // Callable bonds
    price_callable_bond, CallableBondResult,
    // Convertible bonds
    price_convertible_bond, ConvertibleBondResult,
    // Lookbacks
    analytic_lookback, LookbackResult,
    // Compound options
    analytic_compound_option, CompoundOptionResult,
    // Variance swaps
    price_variance_swap, VarianceSwapResult,
    // American engines (Phase 13)
    barone_adesi_whaley, bjerksund_stensland, qd_plus_american, AmericanApproxResult,
    // Bates / jump-diffusion (Phase 14)
    bates_price, bates_price_flat, BatesResult,
    merton_jump_diffusion, MertonJDResult,
    // Longstaff-Schwartz MC American (Phase 13)
    mc_american_longstaff_schwartz, LSMBasis,
    // Multi-asset / basket (Phase 15)
    mc_basket, stulz_max_call, stulz_min_call, kirk_spread_call, kirk_spread_put,
    margrabe_exchange, BasketType, BasketResult,
    // Hull-White analytic (Phase 17)
    hw_bond_option, hw_caplet, hw_floorlet, hw_jamshidian_swaption, HWAnalyticResult,
    // Tree / FD / MC swaption engines (Phase 17)
    tree_bond_price, tree_swaption, tree_bermudan_swaption, tree_cap_floor,
    fd_hw_swaption, mc_hw_cap_floor, TreeResult, FdResult, McHwResult,
    // Credit models (Phase 21)
    GaussianCopulaLHP, NtdResult, nth_to_default_mc, cds_option_black,
};

// Stochastic processes
pub use ql_processes::{
    GeneralizedBlackScholesProcess, HestonProcess,
    // Short-rate processes (Phase 16)
    OrnsteinUhlenbeckProcess, HullWhiteProcess,
    BatesProcess, CoxIngersollRossProcess, SquareRootProcess,
};

// Models
pub use ql_models::{
    HestonModel,
    // Short-rate models (Phase 16)
    HullWhiteModel, BatesModel, VasicekModel, CIRModel,
    BlackKarasinskiModel, G2Model,
    // LMM (Phase 20)
    LmmConfig, LmmCurveState, LmmResult, lmm_cap_price, lmm_swaption_price,
};

// Numerical methods
pub use ql_methods::{
    MCResult, mc_european, mc_barrier, mc_asian, mc_heston,
    FDResult, fd_black_scholes,
    LatticeResult, binomial_crr,
    // Bates MC (Phase 14)
    mc_bates,
    // FDM framework (Phase 19)
    Mesher1d, FdmMesherComposite, Fd1dResult, HestonFdResult,
    fd_1d_bs_solve, fd_heston_solve,
};

// Persistence
pub use ql_persistence::{
    Direction, EventType, InstrumentType, LifecycleEvent, MarketSnapshot, ObjectId,
    SnapshotType, Trade, TradeFilter, TradeStatus, VersionedObject,
    InMemoryStore, ObjectStore, Persistable,
};
#[cfg(feature = "redb")]
pub use ql_persistence::EmbeddedStore;

// Re-export serde_json for convenience (instrument_data uses Value)
pub use serde_json;
