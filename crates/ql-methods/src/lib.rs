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

pub mod monte_carlo;
pub mod mc_engines;
pub mod finite_differences;
pub mod lattice;
pub mod fdm_meshers;
pub mod fdm_operators;

// Re-exports
pub use monte_carlo::{Path, PathGenerator, MultiPath, MultiPathGenerator};
pub use mc_engines::{MCResult, mc_european, mc_barrier, mc_asian, mc_heston, mc_bates};
pub use finite_differences::{FDResult, fd_black_scholes};
pub use lattice::{LatticeResult, binomial_crr};
pub use fdm_meshers::{
    Mesher1d, FdmMesherComposite,
    uniform_1d_mesher, concentrating_1d_mesher, log_spot_mesher,
    heston_variance_mesher,
};
pub use fdm_operators::{
    TripleBandOp, Heston2dOps, Fd1dResult, HestonFdResult,
    build_bs_operator, build_heston_ops,
    crank_nicolson_step, implicit_step, douglas_adi_step,
    apply_american_condition,
    fd_1d_bs_solve, fd_heston_solve,
};
