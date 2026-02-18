//! # ql-methods
//!
//! Numerical pricing methods: Monte Carlo path generation, finite difference solvers,
//! and lattice (binomial/trinomial tree) frameworks.

pub mod monte_carlo;
pub mod mc_engines;
pub mod finite_differences;
pub mod lattice;

// Re-exports
pub use monte_carlo::{Path, PathGenerator, MultiPath, MultiPathGenerator};
pub use mc_engines::{MCResult, mc_european, mc_barrier, mc_asian, mc_heston, mc_bates};
pub use finite_differences::{FDResult, fd_black_scholes};
pub use lattice::{LatticeResult, binomial_crr};
