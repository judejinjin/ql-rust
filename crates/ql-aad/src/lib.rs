//! # ql-aad — Adjoint Algorithmic Differentiation
//!
//! This crate provides forward-mode and reverse-mode automatic differentiation
//! for the ql-rust quantitative finance library.
//!
//! ## Overview
//!
//! | Type | Mode | Use Case |
//! |------|------|----------|
//! | [`Dual`] | Forward | Few inputs (≤5), e.g. BS Greeks |
//! | [`DualVec`] | Forward (multi-seed) | All BS Greeks in one pass |
//! | [`AReal`] | Reverse (tape-based) | Many inputs, e.g. Heston, curve sensitivities |
//!
//! All three types implement the [`Number`] trait, which also has a zero-cost
//! implementation for `f64`. Generic pricing functions `fn price<T: Number>(...)`
//! can be instantiated with any of these types.
//!
//! ## Quick Start — Forward-Mode BS Greeks
//!
//! ```
//! use ql_aad::{Number, DualVec, bs_greeks_forward_ad};
//! use ql_aad::OptionKind;
//!
//! let greeks = bs_greeks_forward_ad(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionKind::Call);
//! assert!((greeks.delta - 0.6368).abs() < 0.01);
//! ```

pub mod number;
pub mod dual;
pub mod dual_vec;
pub mod tape;
pub mod math;
pub mod bs;
pub mod complex;
pub mod heston;
pub mod bates;
pub mod interp;
pub mod integrate;
pub mod solvers;

pub use number::Number;
pub use dual::Dual;
pub use dual_vec::DualVec;
pub use tape::{Tape, AReal};
pub use bs::{OptionKind, BSGreeks, bs_price_generic, bs_greeks_forward_ad};
pub use complex::Complex;
pub use heston::{HestonGreeks, heston_price_generic, heston_greeks_ad};
pub use bates::{BatesGreeks, bates_price_generic, bates_greeks_ad};
pub use interp::{LinearInterp, LogLinearInterp, CubicSplineInterp, MonotoneConvexInterp};
pub use integrate::{gl_integrate, gl_integrate_f64_to_t, simpson_integrate, trapezoid_integrate};
pub use solvers::{newton_1d, newton_ad, halley_1d, implicit_diff, solve_and_diff, SolverError};
