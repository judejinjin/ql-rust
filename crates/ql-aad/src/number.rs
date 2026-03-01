//! The `Number` trait — generic scalar abstraction for AD-enabled pricing.
//!
//! Implemented by:
//! - `f64` (zero-cost, no derivative computation)
//! - `Dual` (forward-mode, single directional derivative)
//! - `DualVec<N>` (forward-mode, N simultaneous directional derivatives)
//! - `AReal` (reverse-mode, tape-based adjoint)

use std::fmt;
use std::ops::{Add, Sub, Mul, Div, Neg, AddAssign, SubAssign, MulAssign, DivAssign};

/// Marker + arithmetic trait for scalars that can be used in pricing computations.
///
/// All pricing functions should be generic over `T: Number` to support
/// automatic differentiation without code duplication.
pub trait Number:
    Copy
    + Clone
    + Send
    + Sync
    + Default
    + PartialEq
    + PartialOrd
    + fmt::Debug
    + fmt::Display
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + 'static
{
    /// Create from an f64 constant.
    fn from_f64(v: f64) -> Self;

    /// Extract the scalar value (dropping derivative information).
    fn to_f64(self) -> f64;

    // --- Transcendental functions ---

    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
    fn powf(self, n: Self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;
    fn recip(self) -> Self;

    // --- Constants ---

    fn zero() -> Self;
    fn one() -> Self;
    fn two() -> Self { Self::from_f64(2.0) }
    fn half() -> Self { Self::from_f64(0.5) }
    fn pi() -> Self;
    fn epsilon() -> Self;

    // --- Comparisons (on value only for AD types) ---

    fn is_positive(self) -> bool;
    fn is_negative(self) -> bool;
    fn is_zero(self) -> bool { self.to_f64() == 0.0 }
}

// ===========================================================================
// f64 implementation — zero-cost passthrough
// ===========================================================================

impl Number for f64 {
    #[inline(always)]
    fn from_f64(v: f64) -> Self { v }
    #[inline(always)]
    fn to_f64(self) -> f64 { self }

    #[inline(always)]
    fn exp(self) -> Self { f64::exp(self) }
    #[inline(always)]
    fn ln(self) -> Self { f64::ln(self) }
    #[inline(always)]
    fn sqrt(self) -> Self { f64::sqrt(self) }
    #[inline(always)]
    fn abs(self) -> Self { f64::abs(self) }
    #[inline(always)]
    fn powf(self, n: Self) -> Self { f64::powf(self, n) }
    #[inline(always)]
    fn powi(self, n: i32) -> Self { f64::powi(self, n) }
    #[inline(always)]
    fn sin(self) -> Self { f64::sin(self) }
    #[inline(always)]
    fn cos(self) -> Self { f64::cos(self) }
    #[inline(always)]
    fn max(self, other: Self) -> Self { f64::max(self, other) }
    #[inline(always)]
    fn min(self, other: Self) -> Self { f64::min(self, other) }
    #[inline(always)]
    fn recip(self) -> Self { 1.0 / self }

    #[inline(always)]
    fn zero() -> Self { 0.0 }
    #[inline(always)]
    fn one() -> Self { 1.0 }
    #[inline(always)]
    fn pi() -> Self { std::f64::consts::PI }
    #[inline(always)]
    fn epsilon() -> Self { f64::EPSILON }

    #[inline(always)]
    fn is_positive(self) -> bool { self > 0.0 }
    #[inline(always)]
    fn is_negative(self) -> bool { self < 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f64_number_trait() {
        let x: f64 = Number::from_f64(3.0);
        assert_eq!(x.to_f64(), 3.0);
        assert_eq!(f64::zero(), 0.0);
        assert_eq!(f64::one(), 1.0);
        assert!((f64::pi() - std::f64::consts::PI).abs() < 1e-15);
    }

    #[test]
    fn f64_transcendentals() {
        let x: f64 = 2.0;
        assert!((Number::exp(x) - x.exp()).abs() < 1e-15);
        assert!((Number::ln(x) - x.ln()).abs() < 1e-15);
        assert!((Number::sqrt(x) - x.sqrt()).abs() < 1e-15);
    }
}
