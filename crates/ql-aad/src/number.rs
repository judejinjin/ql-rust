//! The `Number` trait — generic scalar abstraction for AD-enabled pricing.
//!
//! The canonical definition lives in [`ql_core::Number`]. This module
//! re-exports the trait so that downstream users of `ql-aad` can continue
//! to write `use ql_aad::Number;`.
//!
//! Implemented by:
//! - `f64` (zero-cost, no derivative computation)
//! - `Dual` (forward-mode, single directional derivative)
//! - `DualVec<N>` (forward-mode, N simultaneous directional derivatives)
//! - `AReal` (reverse-mode, tape-based adjoint)

// Re-export the canonical trait from ql-core.
pub use ql_core::Number;
