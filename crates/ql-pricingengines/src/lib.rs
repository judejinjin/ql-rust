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
//! - [`double_barrier_engine`] — Ikeda-Kunitomo double-barrier options
//! - [`chooser_engine`] — Rubinstein (1991) simple chooser options
//! - [`cliquet_engine`] — cliquet / ratchet option pricing
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
pub mod engine_adapters;
pub mod sensitivity;
pub mod swaption_engines;
pub mod cap_floor_engines;
pub mod inflation_cap_floor_engine;
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
pub mod double_barrier_engine;
pub mod chooser_engine;
pub mod cliquet_engine;
pub mod cos_heston;
pub mod heston_hull_white_engine;
pub mod analytic_vanilla_extra;
pub mod analytic_binary_barrier;
pub mod fd_heston_barrier;
pub mod variance_gamma_engine;
pub mod analytic_asian;
pub mod basket_engines;
pub mod vanilla_extra_engines;
pub mod exotic_options;

// Re-exports
pub use american_engines::{
    barone_adesi_whaley, bjerksund_stensland, qd_plus_american, AmericanApproxResult,
};
pub use analytic_european::{price_european, price_european_discrete_dividends, implied_volatility, black_scholes_price, AnalyticEuropeanResults};
pub use analytic_heston::{heston_price, HestonResult};
pub use analytic_bates::{bates_price, bates_price_flat, BatesResult};
pub use discounting::{
    price_swap, price_swap_multicurve, price_ois, price_bond, price_floating_bond,
    SwapResults, BondResults,
};
pub use swaption_engines::{black_swaption, bachelier_swaption, SwaptionResult};
pub use cap_floor_engines::{black_cap_floor, bachelier_cap_floor, CapFloorResult};
pub use inflation_cap_floor_engine::{
    InflationCapFloorResult,
    black_yoy_inflation_cap_floor, bachelier_yoy_inflation_cap_floor,
    black_zc_inflation_cap_floor, bachelier_zc_inflation_cap_floor,
};
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
    fd_hw_swaption, fd_hw_bermudan_fitted, mc_hw_cap_floor,
    TreeResult, FdResult, McHwResult,
};
pub mod credit_models;
pub use credit_models::{
    GaussianCopulaLHP, NtdResult, nth_to_default_mc, cds_option_black,
};
pub mod portfolio_credit;
pub use portfolio_credit::{
    StudentTCopulaLHP, ContagionResult, infectious_default_mc,
    CvaResult, bilateral_cva,
    CdoTrancheSpread, cdo_spread_ladder,
};
pub use double_barrier_engine::{double_barrier_knockout, double_barrier_knockin, DoubleBarrierResult};
pub use chooser_engine::{chooser_price, ChooserResult};
pub use cliquet_engine::{cliquet_price, CliquetResult};
pub mod risk_analytics;
pub use risk_analytics::{
    key_rate_durations, KeyRateDuration,
    scenario_analysis, ScenarioResult, YieldCurveScenario,
    dv01_central_difference, gamma, bs_vega, VegaBucket,
};
pub mod isda_cds;
pub use isda_cds::{
    isda_cds_engine, IsdaCdsResult,
    cds_upfront, cds_points_upfront, cds_cs01,
    cds_imm_schedule, make_standard_cds,
};
pub mod advanced_exotics;
pub use advanced_exotics::{
    quanto_european, QuantoResult,
    power_option, PowerResult,
    forward_start_option, ForwardStartResult,
    digital_barrier, DigitalBarrierResult, DigitalBarrierType,
};
pub mod stochastic_local_vol;
pub use stochastic_local_vol::{
    DupireLocalVol, SlvModel, SlvCalibrationResult,
    calibrate_slv, mc_slv, SlvMcResult,
};
pub mod gaussian1d_engine;
pub use gaussian1d_engine::{
    gaussian1d_swaption, gaussian1d_zcb_option, Gaussian1dResult,
};
pub mod lmm_products;
pub mod lmm_multi_step;
pub use lmm_products::{
    LmmProduct, ExerciseType, BermudanSwaption, CmsSpreadOption,
    CallableRangeAccrual, lmm_product_mc,
};
pub use lmm_multi_step::{
    MultiStepSwap, MultiStepSwaption, MultiStepOptionlets, MultiStepForwards,
    MultiStepCoterminalSwaps, MultiStepCoterminalSwaptions, MultiStepCoinitialSwaps,
    MultiStepInverseFloater, MultiStepRatchet, MultiStepTarn, MultiStepNothing,
    OneStepForwards, OneStepOptionlets, OneStepCoterminalSwaps, OneStepCoinitialSwaps,
    CallSpecifiedMultiProduct, ExerciseAdapter, CashRebate,
};
pub use engine_adapters::{
    AnalyticEuropeanEngine, DiscountingSwapEngine, DiscountingBondEngine,
    MCEuropeanEngine, BinomialCRREngine,
};
pub use sensitivity::{
    Sensitivity, RiskLadder, RiskFactor, EquityMarketParams,
    equity_risk_ladder, sensitivity_from_npvs, sensitivity_central, curve_sensitivities,
};
pub use cos_heston::{cos_heston_price, CosHestonResult};
pub use heston_hull_white_engine::{heston_hull_white_price, HestonHullWhiteResult};
pub use analytic_vanilla_extra::{
    heston_expansion_price, HestonExpansionResult,
    analytic_cev_price, CevResult,
    analytic_ptd_heston_price, PtdHestonSlice, PtdHestonResult,
};
pub use analytic_binary_barrier::{
    analytic_binary_barrier, BinaryBarrierResult, BinaryBarrierType, BinaryPayoff, BinaryDirection,
};
pub use fd_heston_barrier::{
    fd_heston_barrier, FdHestonBarrierResult, FdBarrierType, FdHestonGridParams,
    fd_heston_double_barrier, FdHestonDoubleBarrierResult,
};
pub use variance_gamma_engine::{vg_cos_price, VgResult};
pub use analytic_asian::{
    asian_geometric_continuous_avg_price, asian_geometric_discrete_avg_price,
    asian_geometric_continuous_avg_strike, asian_geometric_discrete_avg_strike,
    asian_turnbull_wakeman, asian_levy, AsianResult,
};
pub use basket_engines::{
    choi_basket_spread, dlz_basket_price, BasketSpreadResult,
};
pub use vanilla_extra_engines::{
    ju_quadratic_american, integral_european, integral_european_vanilla,
    JuAmericanResult, IntegralResult,
};
pub use exotic_options::{
    partial_time_barrier, PartialBarrierResult, PartialBarrierType,
    two_asset_correlation, TwoAssetCorrelationResult,
    holder_extensible, writer_extensible, ExtensibleOptionResult,
};

// --- New engines (QuantLib parity) ---
pub mod heston_pdf_engine;
pub mod qdfp_american;
pub mod digital_american;
pub mod mc_asian;
pub mod fd_asian;
pub mod choi_asian;
pub mod fd_bs_barrier;
pub mod double_binary_barrier;
pub mod barrier_mc_tree;
pub mod mc_basket;
pub mod fd_2d_engine;
pub mod spread_engines;
pub mod mc_forward;
pub mod quanto_wrapper;
pub mod fd_vanilla_extensions;
pub mod shout_swing;
pub mod variance_swap_advanced;
pub mod mc_digital;
pub mod gjrgarch_vasicek_engines;
pub mod vanna_volga;
pub mod integral_cds;
pub mod cva_swap_engine;
pub mod credit_portfolio;
pub mod fd_g2_swaption;
pub mod soft_barrier;
pub mod bsm_hull_white;
pub mod gaussian1d_nonstd_swaption;
pub mod mc_heston_hw;
pub mod mountain_range;
pub mod swaption_capfloor_extended;

pub mod xccy_swap_engine;
pub mod fx_forward_engine;
pub mod cds_index_engine;
pub mod tf_convertible_engine;
pub mod quanto_barrier_engine;

pub use heston_pdf_engine::{heston_pdf_price, HestonPdfResult, exponential_fitting_heston, ExpFitHestonResult};
pub use qdfp_american::{qdfp_american, QdFpAmericanResult};
pub use digital_american::{digital_american, DigitalAmericanResult, DigitalAmericanType};
pub use mc_asian::{mc_asian_arithmetic_price, mc_asian_arithmetic_strike, mc_asian_geometric_price, mc_asian_heston_price, McAsianResult};
pub use fd_asian::{fd_asian, FdAsianResult};
pub use choi_asian::{choi_asian, ChoiAsianResult};
pub use fd_bs_barrier::{fd_bs_barrier, fd_bs_rebate, FdBsBarrierResult, FdBsBarrierType};
pub use double_binary_barrier::{double_binary_barrier, DoubleBinaryBarrierResult, DoubleBinaryType};
pub use barrier_mc_tree::{binomial_barrier, mc_barrier, BarrierTreeMcResult, McBarrierType};
pub use mc_basket::{mc_european_basket, mc_american_basket, McBasketResult, BasketPayoffType};
pub use fd_2d_engine::{fd_2d_vanilla, Fd2dResult, Fd2dPayoff};
pub use spread_engines::{operator_splitting_spread, single_factor_bsm_basket, SpreadEngineResult};
pub use mc_forward::{mc_forward_european_bs, mc_forward_european_heston, McForwardResult};
pub use quanto_wrapper::{quanto_adjustment, quanto_vanilla, QuantoAdjustment, QuantoVanillaResult, QuantoTermStructure};
pub use fd_vanilla_extensions::{fd_bates_vanilla, fd_sabr_vanilla, fd_cev_vanilla, fd_cir_vanilla, fd_heston_hull_white, FdExtVanillaResult};
pub use shout_swing::{fd_shout_option, ShoutOptionResult, fd_swing_option, SwingOptionResult};
pub use variance_swap_advanced::{replicating_variance_swap, mc_variance_swap, VarianceSwapResult as AdvVarianceSwapResult};
pub use mc_digital::{mc_digital, McDigitalResult, McDigitalType};
pub use gjrgarch_vasicek_engines::{gjr_garch_option, GjrGarchResult, vasicek_bond_option, VasicekBondOptionResult, vasicek_european_equity, VasicekEquityResult};
pub use vanna_volga::{vanna_volga_barrier, VannaVolgaBarrierResult, VvBarrierType};
pub use integral_cds::{integral_cds_engine, IntegralCdsResult, risky_bond_engine, RiskyBondResult};
pub use cva_swap_engine::{cva_swap_engine, CvaSwapResult};
pub use credit_portfolio::{
    Issuer, CreditBasket, CdoTranche, LossDistribution, CdoTrancheResult,
    loss_distribution_lhp, price_cdo_tranche,
};
pub use fd_g2_swaption::{fd_g2_swaption, FdG2SwaptionResult, FdG2GridParams};
pub use soft_barrier::{price_soft_barrier, SoftBarrierResult, SoftBarrierConfig, SoftBarrierType};
pub use bsm_hull_white::{price_bsm_hull_white, BsmHullWhiteResult, BsmHullWhiteParams};
pub use gaussian1d_nonstd_swaption::{
    price_gaussian1d_nonstd_swaption, Gaussian1dNonstdSwaptionResult, Gaussian1dNonstdSwaptionParams,
};
pub use mc_heston_hw::{price_mc_heston_hw, McHestonHwResult, McHestonHwParams};
pub use mountain_range::{mc_mountain_range, MountainRangeResult, MountainType};

pub use xccy_swap_engine::{price_xccy_swap, XccySwapResult};
pub use fx_forward_engine::{price_fx_forward, fx_forward_rate, FxForwardResult};
pub use cds_index_engine::{price_cds_index, cds_index_upfront_to_spread, cds_index_spread_to_upfront, CdsIndexResult};
pub use tf_convertible_engine::{
    price_tf_convertible, TFConvertibleResult, CallabilityEntry, CallPutType,
};
pub use quanto_barrier_engine::{price_quanto_barrier, QuantoBarrierOption, QuantoBarrierResult};

pub use swaption_capfloor_extended::{
    gaussian1d_cap_floor, Gaussian1dCapFloorResult,
    gaussian1d_float_float_swaption, Gaussian1dFloatFloatSwaptionResult, FloatFloatSwaptionParams,
    mc_hw_swaption, McHwSwaptionResult,
    fd_hw_swaption_gsr, FdHwSwaptionGsrResult,
    TreeCapFloorEngine, TreeCapFloorEngineResult,
    IrregularSwap, IrregularSwapResult, IrregularSwaption, IrregularSwaptionResult,
    hagan_irregular_swaption, HaganIrregularSwaptionResult, StandardSwaptionSpec,
    basket_generating_engine, BasketGeneratingResult, BasketComponent,
    lattice_short_rate_engine, LatticeShortRateResult,
};

pub mod generic;

pub mod engines_extended;
pub use engines_extended::{
    BlackCalculator, BondFunctions,
    AnalyticDividendEuropeanResult, analytic_dividend_european,
    G2SwaptionResult, g2_swaption_price,
    TreeSwapResult, tree_swap_engine,
    AnalyticPerformanceResult, analytic_performance,
    ForwardPerformanceResult, forward_performance,
};

pub mod credit_extensions;
pub use credit_extensions::{
    DefaultLossModel, LossModelResult,
    GaussianLHPLossModel, BinomialLossModel, RecursiveLossModel,
    SaddlepointLossModel, RandomDefaultLossModel,
    OneFactorGaussianCopula, OneFactorStudentCopula, OneFactorAffineSurvival,
    DefaultProbabilityLatentModel, ConstantLossLatentModel, SpotLossLatentModel,
    RandomLossLatentModel, RandomDefaultLatentModel,
    CorrelationStructure, BaseCorrelationStructure, BaseCorrelationLossModel,
    integral_cdo_engine, midpoint_cdo_engine, integral_ntd_engine,
    CdoEngineResult, NtdEngineResult,
};

pub mod fd_engines_extended;
pub use fd_engines_extended::{
    fd_simple_bs_swing, FdSwingResult,
    fd_multi_period, FdMultiPeriodResult, ExercisePeriod,
};
