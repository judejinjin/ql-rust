//! # ql-methods
//!
//! Numerical pricing methods: Monte Carlo path generation, finite difference solvers,
//! and lattice (binomial/trinomial tree) frameworks.

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
