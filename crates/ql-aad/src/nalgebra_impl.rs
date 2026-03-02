//! `nalgebra::RealField` implementation for [`Dual`] and [`AReal`].
//!
//! Enabled by the `nalgebra` cargo feature. This allows using `Dual` and
//! `AReal` as generic scalars in nalgebra matrices, enabling
//! differentiable linear algebra (e.g. Cholesky, eigendecomposition).
//!
//! # Example
//!
//! ```
//! # #[cfg(feature = "nalgebra")]
//! # {
//! use ql_aad::Dual;
//! use nalgebra::Matrix2;
//!
//! let m = Matrix2::new(
//!     Dual::new(1.0, 1.0), Dual::new(0.5, 0.0),
//!     Dual::new(0.5, 0.0), Dual::new(2.0, 0.0),
//! );
//! let det = m.determinant();
//! assert!((det.val - 1.75).abs() < 1e-12);
//! // det = a*d - b*c, ∂det/∂a = d = 2.0
//! assert!((det.dot - 2.0).abs() < 1e-12);
//! # }
//! ```

use crate::dual::Dual;
use crate::number::Number;
use crate::tape::AReal;

// ===========================================================================
// Helper: Rem / RemAssign for Dual and AReal
// ===========================================================================

impl std::ops::Rem for Dual {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Self) -> Self {
        // a % b, derivative: d(a%b)/da = 1, d(a%b)/db = -floor(a/b)
        let v = self.val % rhs.val;
        let d = self.dot - rhs.dot * (self.val / rhs.val).floor();
        Dual { val: v, dot: d }
    }
}

impl std::ops::RemAssign for Dual {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

impl std::ops::Rem for AReal {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Self) -> Self {
        // a % b = a - b * floor(a/b)
        // Just store the f64 result as a constant (no smooth derivative)
        let v = self.val % rhs.val;
        crate::tape::push_tl(v, smallvec::smallvec![(self.idx, 1.0), (rhs.idx, -(self.val / rhs.val).floor())])
    }
}

impl std::ops::RemAssign for AReal {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

// ===========================================================================
// num_traits impls
// ===========================================================================

macro_rules! impl_num_traits_for {
    ($T:ty) => {
        impl num_traits::Zero for $T {
            #[inline]
            fn zero() -> Self { <$T as Number>::zero() }
            #[inline]
            fn is_zero(&self) -> bool { Number::is_zero(*self) }
        }

        impl num_traits::One for $T {
            #[inline]
            fn one() -> Self { <$T as Number>::one() }
        }

        impl num_traits::Num for $T {
            type FromStrRadixErr = num_traits::ParseFloatError;
            fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
                let v = f64::from_str_radix(s, radix)?;
                Ok(<$T as Number>::from_f64(v))
            }
        }

        impl num_traits::FromPrimitive for $T {
            #[inline]
            fn from_i64(n: i64) -> Option<Self> { Some(<$T as Number>::from_f64(n as f64)) }
            #[inline]
            fn from_u64(n: u64) -> Option<Self> { Some(<$T as Number>::from_f64(n as f64)) }
            #[inline]
            fn from_f64(n: f64) -> Option<Self> { Some(<$T as Number>::from_f64(n)) }
            #[inline]
            fn from_f32(n: f32) -> Option<Self> { Some(<$T as Number>::from_f64(n as f64)) }
        }

        impl num_traits::Signed for $T {
            #[inline]
            fn abs(&self) -> Self { Number::abs(*self) }
            #[inline]
            fn abs_sub(&self, other: &Self) -> Self {
                if *self <= *other {
                    <$T as Number>::zero()
                } else {
                    *self - *other
                }
            }
            #[inline]
            fn signum(&self) -> Self {
                if self.to_f64() > 0.0 {
                    <$T as Number>::one()
                } else if self.to_f64() < 0.0 {
                    -<$T as Number>::one()
                } else {
                    <$T as Number>::zero()
                }
            }
            #[inline]
            fn is_positive(&self) -> bool { Number::is_positive(*self) }
            #[inline]
            fn is_negative(&self) -> bool { Number::is_negative(*self) }
        }

        impl num_traits::Inv for $T {
            type Output = Self;
            #[inline]
            fn inv(self) -> Self { Number::recip(self) }
        }
    };
}

impl_num_traits_for!(Dual);
impl_num_traits_for!(AReal);

// ===========================================================================
// approx impls
// ===========================================================================

macro_rules! impl_approx_for {
    ($T:ty) => {
        impl approx::AbsDiffEq for $T {
            type Epsilon = Self;
            #[inline]
            fn default_epsilon() -> Self { <$T as Number>::from_f64(f64::EPSILON) }
            #[inline]
            fn abs_diff_eq(&self, other: &Self, epsilon: Self) -> bool {
                (self.to_f64() - other.to_f64()).abs() < epsilon.to_f64()
            }
        }

        impl approx::RelativeEq for $T {
            #[inline]
            fn default_max_relative() -> Self { <$T as Number>::from_f64(f64::EPSILON) }
            #[inline]
            fn relative_eq(&self, other: &Self, epsilon: Self, max_relative: Self) -> bool {
                let a = self.to_f64();
                let b = other.to_f64();
                let diff = (a - b).abs();
                if diff < epsilon.to_f64() {
                    return true;
                }
                let largest = a.abs().max(b.abs());
                diff <= largest * max_relative.to_f64()
            }
        }

        impl approx::UlpsEq for $T {
            #[inline]
            fn default_max_ulps() -> u32 { 4 }
            #[inline]
            fn ulps_eq(&self, other: &Self, epsilon: Self, max_ulps: u32) -> bool {
                if approx::AbsDiffEq::abs_diff_eq(self, other, epsilon) {
                    return true;
                }
                // Fall back to relative comparison
                let a = self.to_f64();
                let b = other.to_f64();
                a.ulps_eq(&b, f64::EPSILON, max_ulps)
            }
        }
    };
}

impl_approx_for!(Dual);
impl_approx_for!(AReal);

// ===========================================================================
// simba::scalar::SimdValue
// ===========================================================================

macro_rules! impl_simd_value_for {
    ($T:ty) => {
        impl simba::simd::SimdValue for $T {
            type Element = Self;
            type SimdBool = bool;

            const LANES: usize = 1;

            #[inline]
            fn splat(val: Self::Element) -> Self { val }
            #[inline]
            fn extract(&self, _: usize) -> Self::Element { *self }
            #[inline]
            unsafe fn extract_unchecked(&self, _: usize) -> Self::Element { *self }
            #[inline]
            fn replace(&mut self, _: usize, val: Self::Element) { *self = val; }
            #[inline]
            unsafe fn replace_unchecked(&mut self, _: usize, val: Self::Element) { *self = val; }
            #[inline]
            fn select(self, cond: bool, other: Self) -> Self {
                if cond { self } else { other }
            }
        }
    };
}

impl_simd_value_for!(Dual);
impl_simd_value_for!(AReal);

// ===========================================================================
// simba::scalar::SubsetOf / SupersetOf
// ===========================================================================

impl simba::scalar::SubsetOf<Dual> for Dual {
    #[inline]
    fn to_superset(&self) -> Dual { *self }
    #[inline]
    fn from_superset_unchecked(element: &Dual) -> Dual { *element }
    #[inline]
    fn is_in_subset(_: &Dual) -> bool { true }
}

impl simba::scalar::SubsetOf<Dual> for f64 {
    #[inline]
    fn to_superset(&self) -> Dual { Dual::new(*self, 0.0) }
    #[inline]
    fn from_superset_unchecked(element: &Dual) -> f64 { element.val }
    #[inline]
    fn is_in_subset(_: &Dual) -> bool { true }
}

impl simba::scalar::SubsetOf<Dual> for f32 {
    #[inline]
    fn to_superset(&self) -> Dual { Dual::new(*self as f64, 0.0) }
    #[inline]
    fn from_superset_unchecked(element: &Dual) -> f32 { element.val as f32 }
    #[inline]
    fn is_in_subset(_: &Dual) -> bool { true }
}

impl simba::scalar::SubsetOf<AReal> for AReal {
    #[inline]
    fn to_superset(&self) -> AReal { *self }
    #[inline]
    fn from_superset_unchecked(element: &AReal) -> AReal { *element }
    #[inline]
    fn is_in_subset(_: &AReal) -> bool { true }
}

impl simba::scalar::SubsetOf<AReal> for f64 {
    #[inline]
    fn to_superset(&self) -> AReal { AReal::from_f64(*self) }
    #[inline]
    fn from_superset_unchecked(element: &AReal) -> f64 { element.val }
    #[inline]
    fn is_in_subset(_: &AReal) -> bool { true }
}

impl simba::scalar::SubsetOf<AReal> for f32 {
    #[inline]
    fn to_superset(&self) -> AReal { AReal::from_f64(*self as f64) }
    #[inline]
    fn from_superset_unchecked(element: &AReal) -> f32 { element.val as f32 }
    #[inline]
    fn is_in_subset(_: &AReal) -> bool { true }
}

// ===========================================================================
// simba::scalar::Field  (needs explicit impl — no blanket impl in simba)
// ===========================================================================
// ClosedNeg has a blanket impl for any T: Neg<Output=T> — already satisfied.
// Field requires SimdValue + NumAssign + ClosedNeg.

impl simba::scalar::Field for Dual {}
impl simba::scalar::Field for AReal {}

// ===========================================================================
// simba::scalar::ComplexField
// ===========================================================================

macro_rules! impl_complex_field_for {
    ($T:ty) => {
        impl simba::scalar::ComplexField for $T {
            type RealField = Self;

            #[inline] fn from_real(re: Self) -> Self { re }
            #[inline] fn real(self) -> Self { self }
            #[inline] fn imaginary(self) -> Self { <$T as Number>::zero() }
            #[inline] fn modulus(self) -> Self { Number::abs(self) }
            #[inline] fn modulus_squared(self) -> Self { self * self }
            #[inline] fn argument(self) -> Self {
                if self.to_f64() >= 0.0 {
                    <$T as Number>::zero()
                } else {
                    <$T as Number>::pi()
                }
            }
            #[inline] fn norm1(self) -> Self { Number::abs(self) }
            #[inline] fn scale(self, factor: Self) -> Self { self * factor }
            #[inline] fn unscale(self, factor: Self) -> Self { self / factor }
            #[inline] fn conjugate(self) -> Self { self }
            #[inline] fn floor(self) -> Self { Number::floor(self) }
            #[inline] fn ceil(self) -> Self { Number::ceil(self) }
            #[inline] fn round(self) -> Self {
                <$T as Number>::from_f64(self.to_f64().round())
            }
            #[inline] fn trunc(self) -> Self {
                <$T as Number>::from_f64(self.to_f64().trunc())
            }
            #[inline] fn fract(self) -> Self {
                self - <$T as Number>::from_f64(self.to_f64().trunc())
            }
            #[inline] fn mul_add(self, a: Self, b: Self) -> Self {
                Number::fma(self, a, b)
            }
            #[inline] fn abs(self) -> Self { Number::abs(self) }
            #[inline]
            fn hypot(self, other: Self) -> Self {
                Number::sqrt(self * self + other * other)
            }
            #[inline] fn recip(self) -> Self { Number::recip(self) }
            #[inline] fn sin(self) -> Self { Number::sin(self) }
            #[inline] fn cos(self) -> Self { Number::cos(self) }
            #[inline] fn sin_cos(self) -> (Self, Self) {
                (Number::sin(self), Number::cos(self))
            }
            #[inline] fn tan(self) -> Self { Number::tan(self) }
            #[inline] fn asin(self) -> Self { Number::asin(self) }
            #[inline] fn acos(self) -> Self { Number::acos(self) }
            #[inline] fn atan(self) -> Self { Number::atan(self) }
            #[inline] fn sinh(self) -> Self { Number::sinh(self) }
            #[inline] fn cosh(self) -> Self { Number::cosh(self) }
            #[inline] fn tanh(self) -> Self { Number::tanh(self) }
            #[inline]
            fn asinh(self) -> Self {
                // asinh(x) = ln(x + √(x² + 1))
                Number::ln(self + Number::sqrt(self * self + <$T as Number>::one()))
            }
            #[inline]
            fn acosh(self) -> Self {
                // acosh(x) = ln(x + √(x² - 1))
                Number::ln(self + Number::sqrt(self * self - <$T as Number>::one()))
            }
            #[inline]
            fn atanh(self) -> Self {
                // atanh(x) = 0.5 * ln((1+x)/(1-x))
                let one = <$T as Number>::one();
                <$T as Number>::half() * Number::ln((one + self) / (one - self))
            }
            #[inline] fn log(self, base: Self) -> Self {
                Number::ln(self) / Number::ln(base)
            }
            #[inline] fn log2(self) -> Self { Number::log2(self) }
            #[inline] fn log10(self) -> Self { Number::log10(self) }
            #[inline] fn ln(self) -> Self { Number::ln(self) }
            #[inline]
            fn ln_1p(self) -> Self {
                Number::ln(self + <$T as Number>::one())
            }
            #[inline] fn sqrt(self) -> Self { Number::sqrt(self) }
            #[inline] fn exp(self) -> Self { Number::exp(self) }
            #[inline]
            fn exp2(self) -> Self {
                Number::exp(self * <$T as Number>::from_f64(std::f64::consts::LN_2))
            }
            #[inline]
            fn exp_m1(self) -> Self {
                Number::exp(self) - <$T as Number>::one()
            }
            #[inline] fn powi(self, n: i32) -> Self { Number::powi(self, n) }
            #[inline] fn powf(self, n: Self) -> Self { Number::powf(self, n) }
            #[inline] fn powc(self, n: Self) -> Self { Number::powf(self, n) }
            #[inline]
            fn cbrt(self) -> Self {
                Number::powf(self, <$T as Number>::from_f64(1.0 / 3.0))
            }
            #[inline]
            fn is_finite(&self) -> bool { self.to_f64().is_finite() }
            #[inline]
            fn try_sqrt(self) -> Option<Self> {
                if self.to_f64() >= 0.0 {
                    Some(Number::sqrt(self))
                } else {
                    None
                }
            }

            #[inline]
            fn to_polar(self) -> (Self, Self) {
                (Number::abs(self), simba::scalar::ComplexField::argument(self))
            }

            #[inline]
            fn to_exp(self) -> (Self, Self) {
                let r = Number::abs(self);
                if r.to_f64() == 0.0 {
                    (<$T as Number>::one(), <$T as Number>::zero())
                } else {
                    (r, self / r)
                }
            }

            #[inline]
            fn signum(self) -> Self {
                if self.to_f64() > 0.0 {
                    <$T as Number>::one()
                } else if self.to_f64() < 0.0 {
                    -<$T as Number>::one()
                } else {
                    <$T as Number>::zero()
                }
            }

            #[inline]
            fn sinc(self) -> Self {
                if self.to_f64().abs() < 1e-15 {
                    <$T as Number>::one()
                } else {
                    Number::sin(self) / self
                }
            }

            #[inline]
            fn sinhc(self) -> Self {
                if self.to_f64().abs() < 1e-15 {
                    <$T as Number>::one()
                } else {
                    Number::sinh(self) / self
                }
            }

            #[inline]
            fn cosc(self) -> Self {
                if self.to_f64().abs() < 1e-15 {
                    <$T as Number>::zero()
                } else {
                    (Number::cos(self) - <$T as Number>::one()) / self
                }
            }

            #[inline]
            fn coshc(self) -> Self {
                if self.to_f64().abs() < 1e-15 {
                    <$T as Number>::zero()
                } else {
                    (Number::cosh(self) - <$T as Number>::one()) / self
                }
            }
        }
    };
}

impl_complex_field_for!(Dual);
impl_complex_field_for!(AReal);

// ===========================================================================
// simba::scalar::RealField
// ===========================================================================

macro_rules! impl_real_field_for {
    ($T:ty) => {
        impl simba::scalar::RealField for $T {
            #[inline] fn is_sign_positive(&self) -> bool { self.to_f64() > 0.0 }
            #[inline] fn is_sign_negative(&self) -> bool { self.to_f64() < 0.0 }
            #[inline]
            fn copysign(self, sign: Self) -> Self {
                if sign.to_f64() >= 0.0 { Number::abs(self) } else { -Number::abs(self) }
            }
            #[inline] fn max(self, other: Self) -> Self { Number::max(self, other) }
            #[inline] fn min(self, other: Self) -> Self { Number::min(self, other) }
            #[inline] fn clamp(self, min: Self, max: Self) -> Self {
                Number::max(Number::min(self, max), min)
            }
            #[inline] fn atan2(self, other: Self) -> Self { Number::atan2(self, other) }

            #[inline] fn min_value() -> Option<Self> { Some(<$T as Number>::from_f64(f64::MIN)) }
            #[inline] fn max_value() -> Option<Self> { Some(<$T as Number>::from_f64(f64::MAX)) }

            #[inline] fn pi() -> Self { <$T as Number>::pi() }
            #[inline] fn two_pi() -> Self { <$T as Number>::from_f64(std::f64::consts::TAU) }
            #[inline] fn frac_pi_2() -> Self { <$T as Number>::from_f64(std::f64::consts::FRAC_PI_2) }
            #[inline] fn frac_pi_3() -> Self { <$T as Number>::from_f64(std::f64::consts::FRAC_PI_3) }
            #[inline] fn frac_pi_4() -> Self { <$T as Number>::from_f64(std::f64::consts::FRAC_PI_4) }
            #[inline] fn frac_pi_6() -> Self { <$T as Number>::from_f64(std::f64::consts::FRAC_PI_6) }
            #[inline] fn frac_pi_8() -> Self { <$T as Number>::from_f64(std::f64::consts::FRAC_PI_8) }
            #[inline] fn frac_1_pi() -> Self { <$T as Number>::from_f64(std::f64::consts::FRAC_1_PI) }
            #[inline] fn frac_2_pi() -> Self { <$T as Number>::from_f64(std::f64::consts::FRAC_2_PI) }
            #[inline] fn frac_2_sqrt_pi() -> Self { <$T as Number>::from_f64(std::f64::consts::FRAC_2_SQRT_PI) }
            #[inline] fn e() -> Self { <$T as Number>::from_f64(std::f64::consts::E) }
            #[inline] fn log2_e() -> Self { <$T as Number>::from_f64(std::f64::consts::LOG2_E) }
            #[inline] fn log10_e() -> Self { <$T as Number>::from_f64(std::f64::consts::LOG10_E) }
            #[inline] fn ln_2() -> Self { <$T as Number>::from_f64(std::f64::consts::LN_2) }
            #[inline] fn ln_10() -> Self { <$T as Number>::from_f64(std::f64::consts::LN_10) }
        }
    };
}

impl_real_field_for!(Dual);
impl_real_field_for!(AReal);

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::{adjoint_tl, with_tape};

    #[test]
    fn dual_determinant_2x2() {
        use nalgebra::Matrix2;

        // M = [[a, b], [c, d]] with a as the variable
        let m = Matrix2::new(
            Dual::new(3.0, 1.0), Dual::new(1.0, 0.0),
            Dual::new(2.0, 0.0), Dual::new(4.0, 0.0),
        );
        let det = m.determinant();
        // det = a*d - b*c = 3*4 - 1*2 = 10
        assert!((det.val - 10.0).abs() < 1e-12, "det.val = {}", det.val);
        // d(det)/da = d = 4.0
        assert!((det.dot - 4.0).abs() < 1e-12, "det.dot = {}", det.dot);
    }

    #[test]
    fn dual_matrix_inverse() {
        use nalgebra::Matrix2;

        // Differentiate the (0,0) element of the inverse w.r.t. a
        let m = Matrix2::new(
            Dual::new(2.0, 1.0), Dual::new(1.0, 0.0),
            Dual::new(1.0, 0.0), Dual::new(3.0, 0.0),
        );
        let inv = m.try_inverse().unwrap();
        let inv_00 = inv[(0, 0)];
        // M^-1 (0,0) = d / (ad - bc) = 3/(2*3-1) = 3/5 = 0.6
        assert!((inv_00.val - 0.6).abs() < 1e-10, "inv_00.val = {}", inv_00.val);
        // d/da (d/(ad-bc)) = -d²/(ad-bc)² = -9/25 = -0.36
        assert!((inv_00.dot - (-0.36)).abs() < 1e-10, "inv_00.dot = {}", inv_00.dot);
    }

    #[test]
    fn dual_matrix_vector_mul() {
        use nalgebra::{Matrix2, Vector2};

        let a = Dual::new(2.0, 1.0); // differentiate w.r.t. this entry
        let m = Matrix2::new(a, Dual::new(0.0, 0.0),
                             Dual::new(0.0, 0.0), Dual::new(1.0, 0.0));
        let v = Vector2::new(Dual::new(3.0, 0.0), Dual::new(5.0, 0.0));
        let result = m * v;
        // result[0] = a*3 + 0*5 = 6, d/da = 3
        assert!((result[0].val - 6.0).abs() < 1e-12);
        assert!((result[0].dot - 3.0).abs() < 1e-12);
        // result[1] = 0*3 + 1*5 = 5, d/da = 0
        assert!((result[1].val - 5.0).abs() < 1e-12);
        assert!((result[1].dot - 0.0).abs() < 1e-12);
    }

    #[test]
    fn dual_zero_one() {
        let z = <Dual as num_traits::Zero>::zero();
        assert!(num_traits::Zero::is_zero(&z));
        let o = <Dual as num_traits::One>::one();
        assert!((o.val - 1.0).abs() < 1e-15);
    }

    #[test]
    fn dual_rem() {
        let a = Dual::new(7.5, 1.0);
        let b = Dual::new(3.0, 0.0);
        let r = a % b;
        assert!((r.val - 1.5).abs() < 1e-12);
    }

    #[test]
    fn dual_trig_via_complex_field() {
        use simba::scalar::ComplexField;
        let x = Dual::new(0.5, 1.0);
        let (s, c) = x.sin_cos();
        assert!((s.val - 0.5_f64.sin()).abs() < 1e-12);
        assert!((c.val - 0.5_f64.cos()).abs() < 1e-12);
        // d(sin(x))/dx = cos(x)
        assert!((s.dot - 0.5_f64.cos()).abs() < 1e-12);
        // d(cos(x))/dx = -sin(x)
        assert!((c.dot - (-0.5_f64.sin())).abs() < 1e-12);
    }

    #[test]
    fn dual_real_field_constants() {
        use simba::scalar::RealField;
        let p: Dual = RealField::pi();
        assert!((p.val - std::f64::consts::PI).abs() < 1e-15);
        let e: Dual = RealField::e();
        assert!((e.val - std::f64::consts::E).abs() < 1e-15);
    }

    #[test]
    fn areal_determinant_2x2() {
        use nalgebra::Matrix2;

        let (det_val, det_adj) = with_tape(|tape| {
            let a = tape.input(3.0);
            let b = AReal::from_f64(1.0);
            let c = AReal::from_f64(2.0);
            let d = AReal::from_f64(4.0);
            let m = Matrix2::new(a, b, c, d);
            let det = m.determinant();
            let adj = adjoint_tl(det);
            (det.val, adj[0]) // adj[0] = ∂det/∂a
        });
        // det = a*d - b*c = 10
        assert!((det_val - 10.0).abs() < 1e-10, "det_val = {}", det_val);
        // ∂det/∂a = d = 4.0
        assert!((det_adj - 4.0).abs() < 1e-10, "det_adj = {}", det_adj);
    }

    #[test]
    fn areal_real_field_constants() {
        use simba::scalar::RealField;

        with_tape(|_tape| {
            let p: AReal = RealField::pi();
            assert!((p.val - std::f64::consts::PI).abs() < 1e-15);
        });
    }

    #[test]
    fn dual_simd_value() {
        use simba::simd::SimdValue;
        let d = Dual::new(3.0, 1.0);
        assert_eq!(<Dual as SimdValue>::splat(d), d);
        assert_eq!(d.extract(0), d);
    }
}
