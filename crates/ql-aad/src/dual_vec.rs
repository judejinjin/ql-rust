//! Multi-directional forward-mode dual numbers: `DualVec<N>`.
//!
//! `DualVec<N>` carries a value and `N` simultaneous directional derivatives,
//! enabling all N partial derivatives in a single forward pass.
//!
//! Ideal for Black-Scholes where N=5 (spot, vol, r, q, t) or N=6 (+strike).

use crate::number::Number;
use std::fmt;
use std::ops::{Add, Sub, Mul, Div, Neg, AddAssign, SubAssign, MulAssign, DivAssign};

/// Forward-mode dual number with `N` simultaneous derivative directions.
#[derive(Clone, Copy, Debug)]
pub struct DualVec<const N: usize> {
    /// The primal value.
    pub val: f64,
    /// Derivative components: `dot[i] = ∂val / ∂(input i)`.
    pub dot: [f64; N],
}

impl<const N: usize> Default for DualVec<N> {
    fn default() -> Self {
        Self { val: 0.0, dot: [0.0; N] }
    }
}

impl<const N: usize> DualVec<N> {
    /// Create a constant (all derivatives zero).
    #[inline(always)]
    pub fn constant(val: f64) -> Self {
        Self { val, dot: [0.0; N] }
    }

    /// Create an input variable seeded in direction `idx`.
    ///
    /// `dot[idx] = 1.0`, all other derivatives are 0.
    ///
    /// # Panics
    /// Panics if `idx >= N`.
    #[inline]
    pub fn variable(val: f64, idx: usize) -> Self {
        assert!(idx < N, "DualVec<{}>: seed index {} out of range", N, idx);
        let mut dot = [0.0; N];
        dot[idx] = 1.0;
        Self { val, dot }
    }
}

// ===========================================================================
// Display
// ===========================================================================

impl<const N: usize> fmt::Display for DualVec<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DualVec({}, {:?})", self.val, &self.dot[..])
    }
}

// ===========================================================================
// PartialEq / PartialOrd — compare on value only
// ===========================================================================

impl<const N: usize> PartialEq for DualVec<N> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool { self.val == other.val }
}

impl<const N: usize> PartialOrd for DualVec<N> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.val.partial_cmp(&other.val)
    }
}

// ===========================================================================
// Helper: element-wise dot-array operations
// ===========================================================================

#[inline(always)]
fn dot_add<const N: usize>(a: &[f64; N], b: &[f64; N]) -> [f64; N] {
    let mut r = [0.0; N];
    for i in 0..N { r[i] = a[i] + b[i]; }
    r
}

#[inline(always)]
fn dot_sub<const N: usize>(a: &[f64; N], b: &[f64; N]) -> [f64; N] {
    let mut r = [0.0; N];
    for i in 0..N { r[i] = a[i] - b[i]; }
    r
}

#[inline(always)]
fn dot_neg<const N: usize>(a: &[f64; N]) -> [f64; N] {
    let mut r = [0.0; N];
    for i in 0..N { r[i] = -a[i]; }
    r
}

#[inline(always)]
fn dot_scale<const N: usize>(a: &[f64; N], s: f64) -> [f64; N] {
    let mut r = [0.0; N];
    for i in 0..N { r[i] = a[i] * s; }
    r
}

/// a_val * b_dot + a_dot * b_val  (product rule)
#[inline(always)]
fn dot_product_rule<const N: usize>(a_val: f64, b_val: f64, a_dot: &[f64; N], b_dot: &[f64; N]) -> [f64; N] {
    let mut r = [0.0; N];
    for i in 0..N { r[i] = a_val * b_dot[i] + a_dot[i] * b_val; }
    r
}

/// (a_dot * b_val - a_val * b_dot) / (b_val²)  (quotient rule)
#[inline(always)]
fn dot_quotient_rule<const N: usize>(a_val: f64, b_val: f64, a_dot: &[f64; N], b_dot: &[f64; N]) -> [f64; N] {
    let inv_b2 = 1.0 / (b_val * b_val);
    let mut r = [0.0; N];
    for i in 0..N { r[i] = (a_dot[i] * b_val - a_val * b_dot[i]) * inv_b2; }
    r
}

// ===========================================================================
// Arithmetic operators
// ===========================================================================

impl<const N: usize> Add for DualVec<N> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self { val: self.val + rhs.val, dot: dot_add(&self.dot, &rhs.dot) }
    }
}

impl<const N: usize> Sub for DualVec<N> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self { val: self.val - rhs.val, dot: dot_sub(&self.dot, &rhs.dot) }
    }
}

impl<const N: usize> Mul for DualVec<N> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self {
            val: self.val * rhs.val,
            dot: dot_product_rule(self.val, rhs.val, &self.dot, &rhs.dot),
        }
    }
}

impl<const N: usize> Div for DualVec<N> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self {
            val: self.val / rhs.val,
            dot: dot_quotient_rule(self.val, rhs.val, &self.dot, &rhs.dot),
        }
    }
}

impl<const N: usize> Neg for DualVec<N> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self { val: -self.val, dot: dot_neg(&self.dot) }
    }
}

impl<const N: usize> AddAssign for DualVec<N> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
}
impl<const N: usize> SubAssign for DualVec<N> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs; }
}
impl<const N: usize> MulAssign for DualVec<N> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; }
}
impl<const N: usize> DivAssign for DualVec<N> {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) { *self = *self / rhs; }
}

// ===========================================================================
// Number trait implementation
// ===========================================================================

impl<const N: usize> Number for DualVec<N> {
    #[inline(always)]
    fn from_f64(v: f64) -> Self { Self::constant(v) }

    #[inline(always)]
    fn to_f64(self) -> f64 { self.val }

    #[inline(always)]
    fn exp(self) -> Self {
        let e = self.val.exp();
        Self { val: e, dot: dot_scale(&self.dot, e) }
    }

    #[inline(always)]
    fn ln(self) -> Self {
        Self { val: self.val.ln(), dot: dot_scale(&self.dot, 1.0 / self.val) }
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        let s = self.val.sqrt();
        Self { val: s, dot: dot_scale(&self.dot, 0.5 / s) }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        if self.val >= 0.0 { self } else { -self }
    }

    #[inline(always)]
    fn powf(self, n: Self) -> Self {
        let val = self.val.powf(n.val);
        // d/d(params) of x^n = x^n * (n' * ln(x) + n * x'/x)
        let mut dot = [0.0; N];
        let ln_x = self.val.ln();
        let inv_x = 1.0 / self.val;
        for i in 0..N {
            dot[i] = val * (n.dot[i] * ln_x + n.val * self.dot[i] * inv_x);
        }
        Self { val, dot }
    }

    #[inline(always)]
    fn powi(self, n: i32) -> Self {
        let val = self.val.powi(n);
        let coeff = n as f64 * self.val.powi(n - 1);
        Self { val, dot: dot_scale(&self.dot, coeff) }
    }

    #[inline(always)]
    fn sin(self) -> Self {
        Self { val: self.val.sin(), dot: dot_scale(&self.dot, self.val.cos()) }
    }

    #[inline(always)]
    fn cos(self) -> Self {
        Self { val: self.val.cos(), dot: dot_scale(&self.dot, -self.val.sin()) }
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        if self.val >= other.val { self } else { other }
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        if self.val <= other.val { self } else { other }
    }

    #[inline(always)]
    fn recip(self) -> Self {
        let inv = 1.0 / self.val;
        Self { val: inv, dot: dot_scale(&self.dot, -inv * inv) }
    }

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

    type D5 = DualVec<5>;

    #[test]
    fn dualvec_constant_has_zero_derivatives() {
        let c = D5::constant(42.0);
        assert_abs_diff_eq!(c.val, 42.0, epsilon = 1e-15);
        for &d in &c.dot {
            assert_abs_diff_eq!(d, 0.0, epsilon = 1e-15);
        }
    }

    #[test]
    fn dualvec_variable_seeded() {
        let x = D5::variable(3.0, 2);
        assert_abs_diff_eq!(x.val, 3.0, epsilon = 1e-15);
        assert_abs_diff_eq!(x.dot[0], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(x.dot[1], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(x.dot[2], 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(x.dot[3], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(x.dot[4], 0.0, epsilon = 1e-15);
    }

    #[test]
    fn dualvec_mul_two_vars() {
        // f(x, y) = x*y,  ∂f/∂x = y,  ∂f/∂y = x
        type D2 = DualVec<2>;
        let x = D2::variable(3.0, 0);
        let y = D2::variable(5.0, 1);
        let z = x * y;
        assert_abs_diff_eq!(z.val, 15.0, epsilon = 1e-14);
        assert_abs_diff_eq!(z.dot[0], 5.0, epsilon = 1e-14); // ∂/∂x = y
        assert_abs_diff_eq!(z.dot[1], 3.0, epsilon = 1e-14); // ∂/∂y = x
    }

    #[test]
    fn dualvec_exp_chain() {
        // f(x) = exp(2x), ∂f/∂x = 2·exp(2x)
        type D1 = DualVec<1>;
        let x = D1::variable(1.0, 0);
        let two = D1::constant(2.0);
        let y = (two * x).exp();
        let expected = (2.0_f64).exp();
        assert_abs_diff_eq!(y.val, expected, epsilon = 1e-13);
        assert_abs_diff_eq!(y.dot[0], 2.0 * expected, epsilon = 1e-13);
    }

    #[test]
    fn dualvec_multi_var_polynomial() {
        // f(x, y, z) = x²y + z,  ∂f/∂x = 2xy,  ∂f/∂y = x²,  ∂f/∂z = 1
        type D3 = DualVec<3>;
        let x = D3::variable(2.0, 0);
        let y = D3::variable(3.0, 1);
        let z = D3::variable(1.0, 2);
        let f = x * x * y + z;
        assert_abs_diff_eq!(f.val, 13.0, epsilon = 1e-14); // 4*3 + 1
        assert_abs_diff_eq!(f.dot[0], 12.0, epsilon = 1e-14); // 2*2*3
        assert_abs_diff_eq!(f.dot[1], 4.0, epsilon = 1e-14);  // 2²
        assert_abs_diff_eq!(f.dot[2], 1.0, epsilon = 1e-14);  // 1
    }
}
