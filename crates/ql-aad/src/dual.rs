//! Forward-mode dual number: value + one directional derivative.
//!
//! `Dual` carries `(val, dot)` where `dot = ∂val/∂(seeded input)`.
//! Arithmetic and transcendental operations propagate the derivative
//! via the chain rule.
//!
//! # Example
//!
//! ```
//! use ql_aad::Dual;
//! use ql_aad::Number;
//!
//! // Seed x with derivative 1.0: f(x) = x², f'(x) = 2x
//! let x = Dual::new(3.0, 1.0);
//! let y = x * x;
//! assert!((y.val - 9.0).abs() < 1e-14);
//! assert!((y.dot - 6.0).abs() < 1e-14);
//! ```

use crate::number::Number;
use std::fmt;
use std::ops::{Add, Sub, Mul, Div, Neg, AddAssign, SubAssign, MulAssign, DivAssign};

/// Forward-mode dual number carrying value and one directional derivative.
#[derive(Clone, Copy, Debug, Default)]
pub struct Dual {
    /// The primal value.
    pub val: f64,
    /// The derivative component ∂val / ∂(seeded input).
    pub dot: f64,
}

impl Dual {
    /// Create a new dual number.
    #[inline(always)]
    pub fn new(val: f64, dot: f64) -> Self {
        Self { val, dot }
    }

    /// Create a constant (derivative = 0).
    #[inline(always)]
    pub fn constant(val: f64) -> Self {
        Self { val, dot: 0.0 }
    }

    /// Create a variable seeded with derivative = 1.
    #[inline(always)]
    pub fn variable(val: f64) -> Self {
        Self { val, dot: 1.0 }
    }
}

// ===========================================================================
// Display
// ===========================================================================

impl fmt::Display for Dual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dual({}, {})", self.val, self.dot)
    }
}

// ===========================================================================
// PartialEq / PartialOrd — compare on value only
// ===========================================================================

impl PartialEq for Dual {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val
    }
}

impl PartialOrd for Dual {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.val.partial_cmp(&other.val)
    }
}

// ===========================================================================
// Arithmetic operators
// ===========================================================================

impl Add for Dual {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self { val: self.val + rhs.val, dot: self.dot + rhs.dot }
    }
}

impl Sub for Dual {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self { val: self.val - rhs.val, dot: self.dot - rhs.dot }
    }
}

impl Mul for Dual {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // (a, a')(b, b') = (ab, ab' + a'b)
        Self {
            val: self.val * rhs.val,
            dot: self.val * rhs.dot + self.dot * rhs.val,
        }
    }
}

impl Div for Dual {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        // (a, a')/(b, b') = (a/b, (a'b - ab')/(b²))
        let inv_b = 1.0 / rhs.val;
        Self {
            val: self.val * inv_b,
            dot: (self.dot * rhs.val - self.val * rhs.dot) * inv_b * inv_b,
        }
    }
}

impl Neg for Dual {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self { val: -self.val, dot: -self.dot }
    }
}

impl AddAssign for Dual {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
}
impl SubAssign for Dual {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs; }
}
impl MulAssign for Dual {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; }
}
impl DivAssign for Dual {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) { *self = *self / rhs; }
}

// ===========================================================================
// Number trait implementation
// ===========================================================================

impl Number for Dual {
    #[inline(always)]
    fn from_f64(v: f64) -> Self { Self::constant(v) }

    #[inline(always)]
    fn to_f64(self) -> f64 { self.val }

    #[inline(always)]
    fn exp(self) -> Self {
        let e = self.val.exp();
        Self { val: e, dot: self.dot * e }
    }

    #[inline(always)]
    fn ln(self) -> Self {
        Self { val: self.val.ln(), dot: self.dot / self.val }
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        let s = self.val.sqrt();
        Self { val: s, dot: self.dot / (2.0 * s) }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        if self.val >= 0.0 {
            self
        } else {
            Self { val: -self.val, dot: -self.dot }
        }
    }

    #[inline(always)]
    fn powf(self, n: Self) -> Self {
        // d/dx(x^n) = n * x^(n-1) * dx + x^n * ln(x) * dn
        let val = self.val.powf(n.val);
        let dot = val * (n.dot * self.val.ln() + n.val * self.dot / self.val);
        Self { val, dot }
    }

    #[inline(always)]
    fn powi(self, n: i32) -> Self {
        let val = self.val.powi(n);
        let dot = n as f64 * self.val.powi(n - 1) * self.dot;
        Self { val, dot }
    }

    #[inline(always)]
    fn sin(self) -> Self {
        Self { val: self.val.sin(), dot: self.dot * self.val.cos() }
    }

    #[inline(always)]
    fn cos(self) -> Self {
        Self { val: self.val.cos(), dot: -self.dot * self.val.sin() }
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        // Subgradient: derivative of the one that's larger
        if self.val >= other.val { self } else { other }
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        if self.val <= other.val { self } else { other }
    }

    #[inline(always)]
    fn recip(self) -> Self {
        let inv = 1.0 / self.val;
        Self { val: inv, dot: -self.dot * inv * inv }
    }

    #[inline(always)]
    fn atan2(self, other: Self) -> Self {
        // d/d(y,x) atan2(y, x) = (x*dy - y*dx) / (x² + y²)
        let denom = self.val * self.val + other.val * other.val;
        Self {
            val: self.val.atan2(other.val),
            dot: (other.val * self.dot - self.val * other.dot) / denom,
        }
    }

    #[inline(always)]
    fn tan(self) -> Self {
        let c = self.val.cos();
        Self { val: self.val.tan(), dot: self.dot / (c * c) }
    }

    #[inline(always)]
    fn asin(self) -> Self {
        // d/dx asin(x) = 1/sqrt(1-x²)
        Self { val: self.val.asin(), dot: self.dot / (1.0 - self.val * self.val).sqrt() }
    }

    #[inline(always)]
    fn acos(self) -> Self {
        // d/dx acos(x) = -1/sqrt(1-x²)
        Self { val: self.val.acos(), dot: -self.dot / (1.0 - self.val * self.val).sqrt() }
    }

    #[inline(always)]
    fn atan(self) -> Self {
        // d/dx atan(x) = 1/(1+x²)
        Self { val: self.val.atan(), dot: self.dot / (1.0 + self.val * self.val) }
    }

    #[inline(always)]
    fn sinh(self) -> Self {
        Self { val: self.val.sinh(), dot: self.dot * self.val.cosh() }
    }

    #[inline(always)]
    fn cosh(self) -> Self {
        Self { val: self.val.cosh(), dot: self.dot * self.val.sinh() }
    }

    #[inline(always)]
    fn tanh(self) -> Self {
        let t = self.val.tanh();
        Self { val: t, dot: self.dot * (1.0 - t * t) }
    }

    #[inline(always)]
    fn log2(self) -> Self {
        Self { val: self.val.log2(), dot: self.dot / (self.val * std::f64::consts::LN_2) }
    }

    #[inline(always)]
    fn log10(self) -> Self {
        Self { val: self.val.log10(), dot: self.dot / (self.val * std::f64::consts::LN_10) }
    }

    #[inline(always)]
    fn floor(self) -> Self { Self { val: self.val.floor(), dot: 0.0 } }

    #[inline(always)]
    fn ceil(self) -> Self { Self { val: self.val.ceil(), dot: 0.0 } }

    #[inline(always)]
    fn zero() -> Self { Self::constant(0.0) }
    #[inline(always)]
    fn one() -> Self { Self::constant(1.0) }
    #[inline(always)]
    fn pi() -> Self { Self::constant(std::f64::consts::PI) }
    #[inline(always)]
    fn epsilon() -> Self { Self::constant(f64::EPSILON) }

    #[inline(always)]
    fn is_positive(self) -> bool { self.val > 0.0 }
    #[inline(always)]
    fn is_negative(self) -> bool { self.val < 0.0 }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn dual_add() {
        let a = Dual::new(2.0, 1.0);
        let b = Dual::new(3.0, 0.0);
        let c = a + b;
        assert_abs_diff_eq!(c.val, 5.0, epsilon = 1e-15);
        assert_abs_diff_eq!(c.dot, 1.0, epsilon = 1e-15);
    }

    #[test]
    fn dual_sub() {
        let a = Dual::new(5.0, 1.0);
        let b = Dual::new(3.0, 2.0);
        let c = a - b;
        assert_abs_diff_eq!(c.val, 2.0, epsilon = 1e-15);
        assert_abs_diff_eq!(c.dot, -1.0, epsilon = 1e-15);
    }

    #[test]
    fn dual_mul_product_rule() {
        // f(x) = x * (x+1), f'(x) = 2x+1
        let x = Dual::variable(3.0);
        let one = Dual::constant(1.0);
        let y = x * (x + one);
        assert_abs_diff_eq!(y.val, 12.0, epsilon = 1e-15);
        assert_abs_diff_eq!(y.dot, 7.0, epsilon = 1e-15); // 2*3+1
    }

    #[test]
    fn dual_div_quotient_rule() {
        // f(x) = 1/x, f'(x) = -1/x²
        let x = Dual::variable(2.0);
        let one = Dual::constant(1.0);
        let y = one / x;
        assert_abs_diff_eq!(y.val, 0.5, epsilon = 1e-15);
        assert_abs_diff_eq!(y.dot, -0.25, epsilon = 1e-15);
    }

    #[test]
    fn dual_exp() {
        // f(x) = e^x, f'(x) = e^x
        let x = Dual::variable(1.0);
        let y = x.exp();
        let e = 1.0_f64.exp();
        assert_abs_diff_eq!(y.val, e, epsilon = 1e-14);
        assert_abs_diff_eq!(y.dot, e, epsilon = 1e-14);
    }

    #[test]
    fn dual_ln() {
        // f(x) = ln(x), f'(x) = 1/x
        let x = Dual::variable(2.0);
        let y = x.ln();
        assert_abs_diff_eq!(y.val, 2.0_f64.ln(), epsilon = 1e-14);
        assert_abs_diff_eq!(y.dot, 0.5, epsilon = 1e-14);
    }

    #[test]
    fn dual_sqrt() {
        // f(x) = √x, f'(x) = 1/(2√x)
        let x = Dual::variable(4.0);
        let y = x.sqrt();
        assert_abs_diff_eq!(y.val, 2.0, epsilon = 1e-14);
        assert_abs_diff_eq!(y.dot, 0.25, epsilon = 1e-14);
    }

    #[test]
    fn dual_powi() {
        // f(x) = x³, f'(x) = 3x²
        let x = Dual::variable(2.0);
        let y = x.powi(3);
        assert_abs_diff_eq!(y.val, 8.0, epsilon = 1e-14);
        assert_abs_diff_eq!(y.dot, 12.0, epsilon = 1e-14);
    }

    #[test]
    fn dual_sin_cos() {
        let x = Dual::variable(std::f64::consts::FRAC_PI_4);
        let s = x.sin();
        let c = x.cos();
        assert_abs_diff_eq!(s.dot, c.val, epsilon = 1e-14); // d/dx sin(x) = cos(x)
        assert_abs_diff_eq!(c.dot, -s.val, epsilon = 1e-14); // d/dx cos(x) = -sin(x)
    }

    #[test]
    fn dual_chain_rule() {
        // f(x) = exp(x²), f'(x) = 2x·exp(x²)
        let x = Dual::variable(1.5);
        let y = (x * x).exp();
        let expected_val = (1.5_f64 * 1.5).exp();
        let expected_dot = 2.0 * 1.5 * expected_val;
        assert_abs_diff_eq!(y.val, expected_val, epsilon = 1e-12);
        assert_abs_diff_eq!(y.dot, expected_dot, epsilon = 1e-12);
    }

    #[test]
    fn dual_max_subgradient() {
        let x = Dual::variable(3.0);
        let zero = Dual::constant(0.0);
        let y = Number::max(x, zero); // max(x, 0) at x=3 → value=3, deriv=1
        assert_abs_diff_eq!(y.val, 3.0, epsilon = 1e-15);
        assert_abs_diff_eq!(y.dot, 1.0, epsilon = 1e-15);

        let x2 = Dual::variable(-1.0);
        let y2 = Number::max(x2, zero); // max(x, 0) at x=-1 → value=0, deriv=0
        assert_abs_diff_eq!(y2.val, 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(y2.dot, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn dual_neg() {
        let x = Dual::new(3.0, 1.0);
        let y = -x;
        assert_abs_diff_eq!(y.val, -3.0, epsilon = 1e-15);
        assert_abs_diff_eq!(y.dot, -1.0, epsilon = 1e-15);
    }

    #[test]
    fn dual_abs() {
        let x = Dual::variable(-3.0);
        let y = x.abs();
        assert_abs_diff_eq!(y.val, 3.0, epsilon = 1e-15);
        assert_abs_diff_eq!(y.dot, -1.0, epsilon = 1e-15); // d|x|/dx = -1 for x<0
    }

    #[test]
    fn dual_recip() {
        // f(x) = 1/x, f'(x) = -1/x²
        let x = Dual::variable(4.0);
        let y = x.recip();
        assert_abs_diff_eq!(y.val, 0.25, epsilon = 1e-15);
        assert_abs_diff_eq!(y.dot, -1.0 / 16.0, epsilon = 1e-15);
    }
}
