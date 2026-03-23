//! # ql-methods
//!
//! Numerical pricing methods: Monte Carlo simulation, finite difference solvers,
//! and lattice (binomial/trinomial tree) frameworks.
//!
//! ## Overview
//!
//! | Module | Purpose |
//! |---|---|
//! | [`monte_carlo`] | Path generation ([`Path`], [`MultiPath`]) with pseudo/quasi-random |
//! | [`mc_engines`] | MC pricing: European, barrier, Asian, Heston, Bates |
//! | [`finite_differences`] | Classical 1D finite difference solver |
//! | [`fdm_meshers`] | Mesh generation for FD grids (uniform, concentrating, log-spot) |
//! | [`fdm_operators`] | FD operators, ADI schemes, Crank-Nicolson, 2D Heston solver |
//! | [`lattice`] | Binomial CRR tree for option pricing |
//!
//! ## Quick Start
//!
//! ```rust
//! use ql_methods::{mc_european, MCResult};
//! use ql_instruments::OptionType;
//!
//! // Monte Carlo European call
//! let result: MCResult = mc_european(
//!     100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
//!     OptionType::Call, 50_000, true, 42,
//! );
//! assert!((result.npv - 10.45).abs() < 1.0); // approximate
//! ```
#![warn(missing_docs)]

pub mod monte_carlo;
pub mod mc_engines;
pub mod mc_control_variates;
pub mod finite_differences;
pub mod lattice;
pub mod generic;
pub mod fdm_meshers;
pub mod fdm_meshers_extended;
pub mod fdm_operators;
pub mod fdm_schemes;
pub mod lattice_2d;
pub mod fdm_extended;

// Re-exports
pub use monte_carlo::{Path, PathGenerator, MultiPath, MultiPathGenerator};
pub use mc_engines::{MCResult, mc_european, mc_barrier, mc_asian, mc_heston, mc_bates};
pub use mc_control_variates::{mc_asian_cv, mc_european_cv, geometric_asian_cf};
pub use finite_differences::{FDResult, fd_black_scholes};
pub use lattice::{LatticeResult, binomial_crr, binomial_crr_discrete_dividends};
pub use fdm_meshers::{
    Mesher1d, FdmMesherComposite,
    uniform_1d_mesher, concentrating_1d_mesher, log_spot_mesher,
    heston_variance_mesher,
};
pub use fdm_operators::{
    TripleBandOp, Heston2dOps, Fd1dResult, HestonFdResult, AdiScheme,
    build_bs_operator, build_heston_ops,
    crank_nicolson_step, implicit_step, douglas_adi_step,
    apply_american_condition, apply_cross_derivative,
    hundsdorfer_verwer_step, modified_craig_sneyd_step,
    fd_1d_bs_solve, fd_heston_solve, fd_heston_solve_adi,
};
pub use fdm_schemes::{
    FdmScheme1d, FdmScheme2d, BarrierDirection,
    BarrierCondition, AmericanExerciseCondition, AveragingCondition, RunningExtremeCondition,
    Fd1dSolver, Fd1dStepResult, heston_adi_solve,
};
pub use lattice_2d::{lattice_2d, trinomial_2d_bond_option, Lattice2dResult, FactorParams};
pub use fdm_extended::{
    FdmDirichletBoundary, BoundarySide, FdmBermudanStepCondition,
    Fdm3DimSolver, Fdm3dResult,
    build_cev_operator, build_cir_operator,
    build_hull_white_operator, build_g2_operators,
    craig_sneyd_step, trbdf2_step, method_of_lines_step,
};
pub use fdm_meshers_extended::{
    FdmBlackScholesMesherParams, fdm_black_scholes_mesher,
    FdmBlackScholesMultiStrikeMesherParams, fdm_black_scholes_multi_strike_mesher,
    FdmHestonVarianceMesherParams, fdm_heston_variance_mesher,
    FdmSimpleProcess1DMesherParams, fdm_simple_process_1d_mesher,
    ExponentialJump1DMesherParams, exponential_jump_1d_mesher,
    FdmCEV1DMesherParams, fdm_cev_1d_mesher,
};

pub mod fdm_operators_extended;
pub use fdm_operators_extended::{
    FdmLinearOp, FdmLinearOpComposite,
    NinePointLinearOp, FirstDerivativeOp, SecondDerivativeOp,
    SecondOrderMixedDerivativeOp, NthOrderDerivativeOp,
    ModTripleBandLinearOp,
    FdmBlackScholesOp, Fdm2dBlackScholesOp,
    FdmHestonOp, FdmHestonFwdOp, FdmHestonHullWhiteOp,
    FdmBatesOp, FdmBlackScholesFwdOp, FdmLocalVolFwdOp,
    FdmSquareRootFwdOp, FdmOrnsteinUhlenbeckOp, FdmSABROp,
};

pub mod fdm_infrastructure;
pub use fdm_infrastructure::{
    FdmLinearOpLayout, FdmLinearOpIterator, FdmBackwardSolver,
    FdmSolverDesc, Fdm1DimSolver, Fdm2DimSolver, FdmNDimSolver,
    FdmQuantoHelper, FdmDividendHandler,
    FdmInnerValueCalculator, VanillaInnerValue,
    FdmAffineModelTermStructure, FdmMesherIntegral,
    FdmIndicesOnBoundary, FdmHestonGreensFct,
    FdmSimpleStorageCondition, FdmSimpleSwingCondition,
    FdmSnapshotCondition, FdmStepConditionComposite,
    FdmArithmeticAverageCondition,
    RndCalculator, BSMRndCalculator, HestonRndCalculator,
    LocalVolRndCalculator, CEVRndCalculator, GBSMRndCalculator,
    SquareRootProcessRndCalculator,
    FdmDiscountDirichletBoundary, FdmTimeDependentDirichletBoundary,
    FdmBoundaryConditionSet,
};
