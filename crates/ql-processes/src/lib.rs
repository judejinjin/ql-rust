//! # ql-processes
//!
//! Stochastic process traits and implementations for derivative pricing.
//!
//! ## Overview
//!
//! | Process | SDE | Use Case |
//! |---|---|---|
//! | [`GeneralizedBlackScholesProcess`] | dS = (r−q)S dt + σS dW | Equity options |
//! | [`HestonProcess`] | dS/S = (r−q) dt + √v dW₁, dv = κ(θ−v) dt + σ√v dW₂ | Stochastic vol |
//! | [`BatesProcess`] | Heston + Merton jumps | Vol smiles + jumps |
//! | [`HullWhiteProcess`] | dr = (θ(t)−a·r) dt + σ dW | Interest rates |
//! | [`CoxIngersollRossProcess`] | dr = κ(θ−r) dt + σ√r dW | Non-negative rates |
//! | [`GeometricBrownianMotionProcess`] | dS = μS dt + σS dW | Simple GBM / basket MC |
//! | [`HybridHestonHullWhiteProcess`] | Heston equity + HW rates | Equity/rates hybrids |
//! | [`HestonSLVProcess`] | Heston + local-vol leverage | Stochastic local vol |
//!
//! ## Traits
//!
//! - [`StochasticProcess`] — multi-dimensional process interface
//! - [`StochasticProcess1D`] — single-factor convenience trait

pub mod process;
pub mod black_scholes_process;
pub mod heston_process;
pub mod hull_white_process;
pub mod bates_process;
pub mod cir_process;
pub mod g2_process;
pub mod merton76_process;
pub mod gjr_garch_process;
pub mod geometric_brownian_motion;
pub mod hybrid_heston_hull_white;
pub mod heston_slv_process;
pub mod process_array;

// Re-exports
pub use process::{StochasticProcess, StochasticProcess1D};
pub use black_scholes_process::GeneralizedBlackScholesProcess;
pub use heston_process::HestonProcess;
pub use hull_white_process::{OrnsteinUhlenbeckProcess, HullWhiteProcess};
pub use bates_process::BatesProcess;
pub use cir_process::{CoxIngersollRossProcess, SquareRootProcess};
pub use g2_process::G2Process;
pub use merton76_process::Merton76Process;
pub use gjr_garch_process::GjrGarchProcess;
pub use geometric_brownian_motion::GeometricBrownianMotionProcess;
pub use hybrid_heston_hull_white::HybridHestonHullWhiteProcess;
pub use heston_slv_process::{HestonSLVProcess, LeverageFunction};
pub use process_array::{StochasticProcessArray, ProcessComponent, cholesky_decompose};
