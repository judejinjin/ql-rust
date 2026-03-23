//! The `Number` trait — generic scalar abstraction for AD-enabled pricing.
//!
//! This trait lives in `ql-core` so that all downstream crates can write
//! functions generic over `T: Number` without depending on `ql-aad`.
//!
//! Implemented by:
//! - `f64` (zero-cost, no derivative computation) — in this module
//! - `Dual` (forward-mode, single directional derivative) — in `ql-aad`
//! - `DualVec<N>` (forward-mode, N simultaneous derivatives) — in `ql-aad`
//! - `AReal` (reverse-mode, tape-based adjoint) — in `ql-aad`

use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Marker + arithmetic trait for scalars used in pricing computations.
///
/// All pricing functions should be generic over `T: Number` to support
/// automatic differentiation without code duplication.
///
/// # Examples
///
/// ```
/// use ql_core::Number;
///
/// fn discount<T: Number>(rate: T, t: T) -> T {
///     (T::zero() - rate * t).exp()
/// }
///
/// let df = discount(0.05_f64, 1.0);
/// assert!((df - 0.05_f64.exp().recip()).abs() < 1e-14);
/// ```
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

    /// Exp.
    fn exp(self) -> Self;
    /// Ln.
    fn ln(self) -> Self;
    /// Sqrt.
    fn sqrt(self) -> Self;
    /// Abs.
    fn abs(self) -> Self;
    /// Powf.
    fn powf(self, n: Self) -> Self;
    /// Powi.
    fn powi(self, n: i32) -> Self;
    /// Sin.
    fn sin(self) -> Self;
    /// Cos.
    fn cos(self) -> Self;
    /// Max.
    fn max(self, other: Self) -> Self;
    /// Min.
    fn min(self, other: Self) -> Self;
    /// Recip.
    fn recip(self) -> Self;
    /// Atan2.
    fn atan2(self, other: Self) -> Self;
    /// Tan.
    fn tan(self) -> Self;
    /// Asin.
    fn asin(self) -> Self;
    /// Acos.
    fn acos(self) -> Self;
    /// Atan.
    fn atan(self) -> Self;
    /// Sinh.
    fn sinh(self) -> Self;
    /// Cosh.
    fn cosh(self) -> Self;
    /// Tanh.
    fn tanh(self) -> Self;
    /// Log2.
    fn log2(self) -> Self;
    /// Log10.
    fn log10(self) -> Self;
    /// Floor.
    fn floor(self) -> Self;
    /// Ceil.
    fn ceil(self) -> Self;
    /// Fma.
    fn fma(self, a: Self, b: Self) -> Self {
        self * a + b
    }

    // --- Constants ---

    /// Zero.
    fn zero() -> Self;
    /// One.
    fn one() -> Self;
    /// Two.
    fn two() -> Self {
        Self::from_f64(2.0)
    }
    /// Half.
    fn half() -> Self {
        Self::from_f64(0.5)
    }
    /// Pi.
    fn pi() -> Self;
    /// Epsilon.
    fn epsilon() -> Self;

    // --- Comparisons (on value only for AD types) ---

    /// Is positive.
    fn is_positive(self) -> bool;
    /// Is negative.
    fn is_negative(self) -> bool;
    /// Is zero.
    fn is_zero(self) -> bool {
        self.to_f64() == 0.0
    }
}

// ===========================================================================
// f64 implementation — zero-cost passthrough
// ===========================================================================

impl Number for f64 {
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }

    #[inline(always)]
    fn exp(self) -> Self {
        f64::exp(self)
    }
    #[inline(always)]
    fn ln(self) -> Self {
        f64::ln(self)
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    #[inline(always)]
    fn abs(self) -> Self {
        f64::abs(self)
    }
    #[inline(always)]
    fn powf(self, n: Self) -> Self {
        f64::powf(self, n)
    }
    #[inline(always)]
    fn powi(self, n: i32) -> Self {
        f64::powi(self, n)
    }
    #[inline(always)]
    fn sin(self) -> Self {
        f64::sin(self)
    }
    #[inline(always)]
    fn cos(self) -> Self {
        f64::cos(self)
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        f64::max(self, other)
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        f64::min(self, other)
    }
    #[inline(always)]
    fn recip(self) -> Self {
        1.0 / self
    }
    #[inline(always)]
    fn atan2(self, other: Self) -> Self {
        f64::atan2(self, other)
    }
    #[inline(always)]
    fn tan(self) -> Self {
        f64::tan(self)
    }
    #[inline(always)]
    fn asin(self) -> Self {
        f64::asin(self)
    }
    #[inline(always)]
    fn acos(self) -> Self {
        f64::acos(self)
    }
    #[inline(always)]
    fn atan(self) -> Self {
        f64::atan(self)
    }
    #[inline(always)]
    fn sinh(self) -> Self {
        f64::sinh(self)
    }
    #[inline(always)]
    fn cosh(self) -> Self {
        f64::cosh(self)
    }
    #[inline(always)]
    fn tanh(self) -> Self {
        f64::tanh(self)
    }
    #[inline(always)]
    fn log2(self) -> Self {
        f64::log2(self)
    }
    #[inline(always)]
    fn log10(self) -> Self {
        f64::log10(self)
    }
    #[inline(always)]
    fn floor(self) -> Self {
        f64::floor(self)
    }
    #[inline(always)]
    fn ceil(self) -> Self {
        f64::ceil(self)
    }

    #[inline(always)]
    fn zero() -> Self {
        0.0
    }
    #[inline(always)]
    fn one() -> Self {
        1.0
    }
    #[inline(always)]
    fn pi() -> Self {
        std::f64::consts::PI
    }
    #[inline(always)]
    fn epsilon() -> Self {
        f64::EPSILON
    }

    #[inline(always)]
    fn is_positive(self) -> bool {
        self > 0.0
    }
    #[inline(always)]
    fn is_negative(self) -> bool {
        self < 0.0
    }
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

    #[test]
    fn generic_discount() {
        fn discount<T: Number>(rate: T, t: T) -> T {
            (T::zero() - rate * t).exp()
        }
        let df = discount(0.05_f64, 1.0_f64);
        assert!((df - (-0.05_f64).exp()).abs() < 1e-14);
    }
}
