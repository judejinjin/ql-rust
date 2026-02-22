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
pub use ql_core::engine::{PricingEngine, LazyInstrument, ClosureEngine};
pub use ql_core::market_data::{
    FeedCallback, FeedDrivenQuote, FeedEvent, FeedField,
    InMemoryFeed, MarketDataFeed, SubscriptionId,
};
pub use ql_core::portfolio::{HasNpv, NpvProvider, ReactivePortfolio, wire_entry};
pub use ql_core::observable::{Observable, Observer};

// Time types
pub use ql_time::{
    BusinessDayConvention, Calendar, Date, DayCounter, Frequency, JointRule, Month, Period,
    Schedule, TimeUnit, Weekday,
};

// Indexes
pub use ql_indexes::{
    Compounding, CpiInterpolation, IborIndex, InflationIndex, InterestRate,
    OvernightIndex, SwapIndex, Index, IndexManager,
};

// Instruments
pub use ql_instruments::{
    AmortizingBond, BondForward, BondForwardType,
    BarrierOption, BarrierType, CallableBond, CallabilityScheduleEntry, CallabilityType,
    CapFloor, CapFloorType, Caplet, ChooserOption, CliquetOption,
    CompositeInstrument, CompositeComponent, CompoundOption,
    ConvertibleBond, CreditDefaultSwap, CdsProtectionSide, CdsPremiumPeriod,
    DoubleBarrierOption, DoubleBarrierType, Exercise, FixedRateBond, FloatingRateBond,
    InflationLinkedBond, Instrument, LookbackOption, LookbackType,
    OISSwap,
    ForwardRateAgreement, BasisSwap, CrossCurrencySwap, XCcyLeg, FloatFloatSwap,
    OptionType, Payoff, Stock, Swaption, SwaptionType, SettlementType, SwapType,
    VanillaOption, VanillaSwap, VarianceSwap, AsianOption, AveragingType,
    ZeroCouponBond,
    // Inflation cap/floor instruments (Phase 29)
    InflationCapFloorType, YoYInflationCapFloor, YoYInflationCaplet,
    ZeroCouponInflationCapFloor, build_yoy_cap_floor,
    // Nonstandard (amortizing/step-up) swaps (Phase 29)
    NonstandardSwap, NonstandardSwapResults, AmortizationType,
    price_nonstandard_swap,
    // Asset swap + Equity TRS (Phase 29)
    AssetSwap, AssetSwapConvention, AssetSwapResult, price_asset_swap,
    EquityTRS, EquityTRSResult, price_equity_trs, equity_trs_fair_spread,
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
    // YoY inflation (Phase 25)
    YoYInflationTermStructure, FlatYoYInflationCurve, YearOnYearInflationSwap,
    // Extended yield curves (Phase 24)
    CompositeZeroYieldStructure, ImpliedTermStructure, ForwardCurve,
    UltimateForwardTermStructure, SpreadedTermStructure, QuantoTermStructure,
    // Extended rate helpers (Phase 24)
    OISRateHelper, BondHelper, FuturesRateHelper, FRAHelper,
    // Nelson-Siegel / Svensson fitting (Phase 24)
    NelsonSiegelFitting, SvenssonFitting, FittedBondDiscountCurve, FittingMethod,
    // SVI / ZABR smile models (Phase 18)
    SviSmileSection, ZabrSmileSection, svi_volatility, svi_total_variance,
    svi_calibrate, zabr_volatility,
    // NoArb-SABR + Kahale arbitrage-free interpolation (Phase 29)
    NoArbSabrSmileSection, ArbitrageCheckResult, check_smile_arbitrage,
    kahale_call_prices,
    // Optionlet stripping (Phase 18)
    strip_optionlet_volatilities, interpolate_optionlet_vol, StrippedOptionletVolatilities,
    // Vol interpolation (Phase 18)
    BlackVarianceCurve, SmileSectionSurface,
    // Swaption vol structures
    SwaptionVolatilityStructure, SwaptionConstantVol, SwaptionVolMatrix,
    SwaptionVolCube, SabrSwaptionVolCube, SabrParams,
    CapFloorTermVolStructure, ConstantCapFloorTermVol, CapFloorTermVolSurface,
    // Andreasen-Huge vol surface
    AndreasenHugeVolSurface, AndreasenHugeConfig, AndreasenHugeCalibrationResult,
    VolQuote, andreasen_huge_calibrate,
};

// Cash flows
pub use ql_cashflows::{
    CashFlow, Leg,
    // Leg builders
    fixed_leg, ibor_leg, overnight_leg, add_notional_exchange,
    // CMS coupons (Phase 23)
    CmsCoupon, cms_convexity_adjustment, cms_caplet_price,
    // Digital / capped-floored / range-accrual / sub-period coupons (Phase 23)
    DigitalCoupon, CapFlooredCoupon, RangeAccrualCoupon, SubPeriodCoupon, SubPeriodType,
    // Extended analytics (Phase 23)
    convexity, modified_duration, dv01, z_spread, atm_rate,
    time_bucketed_cashflows, TimeBucket,
    // CPI / inflation coupons (Phase 25)
    CPICoupon, generate_cpi_coupons,
    // Discrete dividends (Phase 29)
    Dividend, DividendSchedule,
};

// Pricing engines
pub use ql_pricingengines::{
    // Analytic European
    price_european, implied_volatility, AnalyticEuropeanResults,
    // Heston
    heston_price, HestonResult,
    // Discounting (swap, bond)
    price_swap, price_swap_multicurve, price_ois, price_bond, price_floating_bond,
    SwapResults, BondResults,
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
    // Double-barrier / chooser / cliquet engines (Phase 26)
    double_barrier_knockout, double_barrier_knockin, DoubleBarrierResult,
    chooser_price, ChooserResult,
    cliquet_price, CliquetResult,
    black_scholes_price,
    // Risk analytics
    key_rate_durations, KeyRateDuration,
    scenario_analysis, ScenarioResult, YieldCurveScenario,
    dv01_central_difference, gamma, bs_vega, VegaBucket,
    // ISDA CDS engine
    isda_cds_engine, IsdaCdsResult,
    cds_upfront, cds_points_upfront, cds_cs01,
    cds_imm_schedule, make_standard_cds,
    // Advanced exotics
    quanto_european, QuantoResult,
    power_option, PowerResult,
    forward_start_option, ForwardStartResult,
    digital_barrier, DigitalBarrierResult, DigitalBarrierType,
    // Discrete-dividend European & inflation cap/floor engines (Phase 29)
    price_european_discrete_dividends,
    InflationCapFloorResult,
    black_yoy_inflation_cap_floor, bachelier_yoy_inflation_cap_floor,
    black_zc_inflation_cap_floor, bachelier_zc_inflation_cap_floor,
    // Stochastic Local Vol
    DupireLocalVol, SlvModel, SlvCalibrationResult,
    calibrate_slv, mc_slv, SlvMcResult,
    // Gaussian 1-factor engine
    gaussian1d_swaption, gaussian1d_zcb_option, Gaussian1dResult,
    // LMM multi-step products
    LmmProduct, ExerciseType, BermudanSwaption, CmsSpreadOption,
    CallableRangeAccrual, lmm_product_mc,
};

// Stochastic processes
pub use ql_processes::{
    GeneralizedBlackScholesProcess, HestonProcess,
    // Short-rate processes (Phase 16)
    OrnsteinUhlenbeckProcess, HullWhiteProcess,
    BatesProcess, CoxIngersollRossProcess, SquareRootProcess,
    // G2++ two-factor process
    G2Process,
    // Merton76 jump-diffusion process
    Merton76Process,
    // GJR-GARCH process (Phase 29)
    GjrGarchProcess,
};

// Models
pub use ql_models::{
    HestonModel,
    // Short-rate models (Phase 16)
    HullWhiteModel, BatesModel, VasicekModel, CIRModel,
    BlackKarasinskiModel, G2Model,
    // LMM (Phase 20)
    LmmConfig, LmmCurveState, LmmResult, lmm_cap_price, lmm_swaption_price, evolve_one_step,
    // GSR / Markov-Functional
    Gsr1d, MarkovFunctional,
};

// Numerical methods
pub use ql_methods::{
    MCResult, mc_european, mc_barrier, mc_asian, mc_heston,
    FDResult, fd_black_scholes,
    LatticeResult, binomial_crr,
    // Bates MC (Phase 14)
    mc_bates,
    // MC control variates (Phase 29)
    mc_asian_cv, mc_european_cv, geometric_asian_cf,
    // Binomial with discrete dividends (Phase 29)
    binomial_crr_discrete_dividends,
    // FDM framework (Phase 19)
    Mesher1d, FdmMesherComposite, Fd1dResult, HestonFdResult,
    fd_1d_bs_solve, fd_heston_solve,
    // ADI schemes (Phase 29)
    AdiScheme, fd_heston_solve_adi,
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
