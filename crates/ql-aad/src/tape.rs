//! Reverse-mode tape-based automatic differentiation.
//!
//! The [`Tape`] records a computation graph (Wengert list) during the forward
//! pass. After computing the output, [`Tape::adjoint`] propagates derivatives
//! backward to produce gradients w.r.t. all inputs simultaneously.
//!
//! # Example
//!
//! ```
//! use ql_aad::tape::Tape;
//!
//! let mut tape = Tape::new();
//! let x = tape.input(3.0);
//! let y = tape.input(5.0);
//! // z = x * y + x
//! let xy = tape.mul(x, y);
//! let z = tape.add(xy, x);
//! let grad = tape.adjoint(z);
//! assert!((grad[x.idx] - 6.0).abs() < 1e-14); // ∂z/∂x = y + 1 = 6
//! assert!((grad[y.idx] - 3.0).abs() < 1e-14); // ∂z/∂y = x = 3
//! ```

use smallvec::{smallvec, SmallVec};
use std::cell::RefCell;
use std::fmt;
use std::ops::{Add, Sub, Mul, Div, Neg, AddAssign, SubAssign, MulAssign, DivAssign};

use crate::number::Number;

/// A node in the computation tape.
#[derive(Clone, Debug)]
struct Node {
    /// Value computed at this node.
    #[allow(dead_code)]
    value: f64,
    /// Partial derivatives w.r.t. children: (child_index, ∂self/∂child).
    partials: SmallVec<[(usize, f64); 2]>,
}

/// Thread-local computation tape (Wengert list) for reverse-mode AD.
///
/// Records operations during the forward pass and computes adjoints
/// (gradients) in a single backward pass.
#[derive(Clone, Debug)]
pub struct Tape {
    nodes: Vec<Node>,
}

/// Active real on the tape — records operations during forward computation.
///
/// Lightweight (just an index + cached value). Clone/Copy is allowed since
/// the actual data lives on the tape.
#[derive(Clone, Copy, Debug)]
pub struct AReal {
    /// Position on the tape.
    pub idx: usize,
    /// Cached forward value.
    pub val: f64,
}

impl Tape {
    /// Create a new empty tape.
    pub fn new() -> Self {
        Self { nodes: Vec::with_capacity(4096) }
    }

    /// Create a tape with pre-allocated capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self { nodes: Vec::with_capacity(cap) }
    }

    /// Clear the tape for reuse (keeps allocation).
    pub fn clear(&mut self) {
        self.nodes.clear();
    }

    /// Number of nodes on the tape.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the tape is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Register an input variable on the tape.
    pub fn input(&mut self, val: f64) -> AReal {
        let idx = self.nodes.len();
        self.nodes.push(Node {
            value: val,
            partials: SmallVec::new(),
        });
        AReal { idx, val }
    }

    /// Push a new node with given value and partial derivatives.
    fn push(&mut self, value: f64, partials: SmallVec<[(usize, f64); 2]>) -> AReal {
        let idx = self.nodes.len();
        self.nodes.push(Node { value, partials });
        AReal { idx, val: value }
    }

    // --- Unary operations ---

    /// Negate: `-a`.
    pub fn neg(&mut self, a: AReal) -> AReal {
        self.push(-a.val, smallvec![(a.idx, -1.0)])
    }

    /// Absolute value: `|a|`.
    pub fn abs(&mut self, a: AReal) -> AReal {
        let d = if a.val >= 0.0 { 1.0 } else { -1.0 };
        self.push(a.val.abs(), smallvec![(a.idx, d)])
    }

    /// Exponential: `exp(a)`.
    pub fn exp(&mut self, a: AReal) -> AReal {
        let e = a.val.exp();
        self.push(e, smallvec![(a.idx, e)])
    }

    /// Natural logarithm: `ln(a)`.
    pub fn ln(&mut self, a: AReal) -> AReal {
        self.push(a.val.ln(), smallvec![(a.idx, 1.0 / a.val)])
    }

    /// Square root: `√a`.
    pub fn sqrt(&mut self, a: AReal) -> AReal {
        let s = a.val.sqrt();
        self.push(s, smallvec![(a.idx, 0.5 / s)])
    }

    /// Reciprocal: `1/a`.
    pub fn recip(&mut self, a: AReal) -> AReal {
        let inv = 1.0 / a.val;
        self.push(inv, smallvec![(a.idx, -inv * inv)])
    }

    /// Integer power: `a^n`.
    pub fn powi(&mut self, a: AReal, n: i32) -> AReal {
        let val = a.val.powi(n);
        let d = n as f64 * a.val.powi(n - 1);
        self.push(val, smallvec![(a.idx, d)])
    }

    /// Sine: `sin(a)`.
    pub fn sin(&mut self, a: AReal) -> AReal {
        self.push(a.val.sin(), smallvec![(a.idx, a.val.cos())])
    }

    /// Cosine: `cos(a)`.
    pub fn cos(&mut self, a: AReal) -> AReal {
        self.push(a.val.cos(), smallvec![(a.idx, -a.val.sin())])
    }

    // --- Binary operations ---

    /// Addition: `a + b`.
    pub fn add(&mut self, a: AReal, b: AReal) -> AReal {
        self.push(a.val + b.val, SmallVec::from_buf([(a.idx, 1.0), (b.idx, 1.0)]))
    }

    /// Subtraction: `a - b`.
    pub fn sub(&mut self, a: AReal, b: AReal) -> AReal {
        self.push(a.val - b.val, SmallVec::from_buf([(a.idx, 1.0), (b.idx, -1.0)]))
    }

    /// Multiplication: `a * b`.
    pub fn mul(&mut self, a: AReal, b: AReal) -> AReal {
        self.push(a.val * b.val, SmallVec::from_buf([(a.idx, b.val), (b.idx, a.val)]))
    }

    /// Division: `a / b`.
    pub fn div(&mut self, a: AReal, b: AReal) -> AReal {
        let inv_b = 1.0 / b.val;
        self.push(
            a.val * inv_b,
            SmallVec::from_buf([(a.idx, inv_b), (b.idx, -a.val * inv_b * inv_b)]),
        )
    }

    /// Power: `a^b`.
    pub fn powf(&mut self, a: AReal, b: AReal) -> AReal {
        let val = a.val.powf(b.val);
        let da = b.val * a.val.powf(b.val - 1.0); // ∂(a^b)/∂a = b * a^(b-1)
        let db = val * a.val.ln();                  // ∂(a^b)/∂b = a^b * ln(a)
        self.push(val, SmallVec::from_buf([(a.idx, da), (b.idx, db)]))
    }

    /// Max: `max(a, b)` with subgradient.
    pub fn max(&mut self, a: AReal, b: AReal) -> AReal {
        if a.val >= b.val {
            self.push(a.val, smallvec![(a.idx, 1.0)])
        } else {
            self.push(b.val, smallvec![(b.idx, 1.0)])
        }
    }

    /// Min: `min(a, b)` with subgradient.
    pub fn min(&mut self, a: AReal, b: AReal) -> AReal {
        if a.val <= b.val {
            self.push(a.val, smallvec![(a.idx, 1.0)])
        } else {
            self.push(b.val, smallvec![(b.idx, 1.0)])
        }
    }

    /// Add a scalar constant: `a + c`.
    pub fn add_const(&mut self, a: AReal, c: f64) -> AReal {
        self.push(a.val + c, smallvec![(a.idx, 1.0)])
    }

    /// Multiply by a scalar constant: `a * c`.
    pub fn mul_const(&mut self, a: AReal, c: f64) -> AReal {
        self.push(a.val * c, smallvec![(a.idx, c)])
    }

    /// atan2(y, x): `atan2(a, b)`.
    pub fn atan2(&mut self, a: AReal, b: AReal) -> AReal {
        let denom = a.val * a.val + b.val * b.val;
        let da = b.val / denom;    // ∂atan2/∂y = x / (x² + y²)
        let db = -a.val / denom;   // ∂atan2/∂x = -y / (x² + y²)
        self.push(a.val.atan2(b.val), SmallVec::from_buf([(a.idx, da), (b.idx, db)]))
    }

    // --- Adjoint computation ---

    /// Compute the adjoint (gradient) from the given output node.
    ///
    /// Returns a vector where `result[i]` = ∂output/∂node_i.
    /// For input nodes, this gives the gradient w.r.t. each input.
    pub fn adjoint(&self, output: AReal) -> Vec<f64> {
        let n = self.nodes.len();
        let mut adj = vec![0.0; n];
        adj[output.idx] = 1.0;

        // Reverse sweep
        for i in (0..=output.idx).rev() {
            let a_i = adj[i];
            if a_i == 0.0 { continue; }
            for &(child_idx, partial) in &self.nodes[i].partials {
                adj[child_idx] += a_i * partial;
            }
        }

        adj
    }

    /// Compute gradients w.r.t. specific input nodes only.
    ///
    /// More convenient when you only need a few gradients:
    /// ```
    /// # use ql_aad::tape::Tape;
    /// let mut tape = Tape::new();
    /// let x = tape.input(2.0);
    /// let y = tape.input(3.0);
    /// let z = tape.mul(x, y);
    /// let grads = tape.gradient(z, &[x, y]);
    /// assert!((grads[0] - 3.0).abs() < 1e-14);  // ∂z/∂x = y
    /// assert!((grads[1] - 2.0).abs() < 1e-14);  // ∂z/∂y = x
    /// ```
    pub fn gradient(&self, output: AReal, inputs: &[AReal]) -> Vec<f64> {
        let adj = self.adjoint(output);
        inputs.iter().map(|inp| adj[inp.idx]).collect()
    }
}

impl Default for Tape {
    fn default() -> Self { Self::new() }
}

// ===========================================================================
// Thread-local tape for `Number`-based `AReal` arithmetic
// ===========================================================================

thread_local! {
    static TAPE: RefCell<Tape> = RefCell::new(Tape::new());
}

/// Activate the thread-local tape, run the closure, and return the result.
///
/// The tape is cleared before the closure runs. After the closure returns,
/// you can call [`adjoint_tl`] to compute gradients.
///
/// The closure receives a [`TapeHandle`] which supports `.input(val)` to
/// register input variables. AReal arithmetic operations inside the closure
/// automatically record onto the thread-local tape.
///
/// # Example
///
/// ```
/// use ql_aad::tape::{with_tape, adjoint_tl, AReal};
/// use ql_aad::Number;
///
/// let (z, x_idx, y_idx) = with_tape(|tape| {
///     let x = tape.input(3.0);
///     let y = tape.input(5.0);
///     let z = x * y + x; // uses Number trait ops
///     (z, x.idx, y.idx)
/// });
/// let grad = adjoint_tl(z);
/// assert!((grad[x_idx] - 6.0).abs() < 1e-14);
/// assert!((grad[y_idx] - 3.0).abs() < 1e-14);
/// ```
pub fn with_tape<F, R>(f: F) -> R
where
    F: FnOnce(&mut TapeHandle) -> R,
{
    // Clear the tape, releasing the borrow immediately so that AReal ops
    // inside the closure can acquire their own short-lived borrows.
    TAPE.with(|cell| {
        cell.borrow_mut().clear();
    });
    f(&mut TapeHandle)
}

/// A handle passed to [`with_tape`] closures that safely delegates to
/// the thread-local tape without holding a long-lived borrow.
pub struct TapeHandle;

impl TapeHandle {
    /// Register an input variable on the thread-local tape.
    pub fn input(&mut self, val: f64) -> AReal {
        input_tl(val)
    }
}

/// Compute adjoints on the thread-local tape from the given output.
pub fn adjoint_tl(output: AReal) -> Vec<f64> {
    TAPE.with(|cell| {
        let tape = cell.borrow();
        tape.adjoint(output)
    })
}

/// Push an input onto the thread-local tape.
pub fn input_tl(val: f64) -> AReal {
    TAPE.with(|cell| {
        cell.borrow_mut().input(val)
    })
}

// Helper: push a node onto the thread-local tape.
pub(crate) fn push_tl(value: f64, partials: SmallVec<[(usize, f64); 2]>) -> AReal {
    TAPE.with(|cell| {
        cell.borrow_mut().push(value, partials)
    })
}

// ===========================================================================
// AReal: Display, PartialEq, PartialOrd, Default
// ===========================================================================

impl fmt::Display for AReal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.val)
    }
}

impl PartialEq for AReal {
    fn eq(&self, other: &Self) -> bool { self.val == other.val }
}

impl PartialOrd for AReal {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.val.partial_cmp(&other.val)
    }
}

impl Default for AReal {
    fn default() -> Self { Self { idx: 0, val: 0.0 } }
}

// ===========================================================================
// AReal: std::ops for Number trait
// ===========================================================================

impl Add for AReal {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        push_tl(self.val + rhs.val, SmallVec::from_buf([(self.idx, 1.0), (rhs.idx, 1.0)]))
    }
}

impl Sub for AReal {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        push_tl(self.val - rhs.val, SmallVec::from_buf([(self.idx, 1.0), (rhs.idx, -1.0)]))
    }
}

impl Mul for AReal {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        push_tl(self.val * rhs.val, SmallVec::from_buf([(self.idx, rhs.val), (rhs.idx, self.val)]))
    }
}

impl Div for AReal {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let inv_b = 1.0 / rhs.val;
        push_tl(
            self.val * inv_b,
            SmallVec::from_buf([(self.idx, inv_b), (rhs.idx, -self.val * inv_b * inv_b)]),
        )
    }
}

impl Neg for AReal {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        push_tl(-self.val, smallvec![(self.idx, -1.0)])
    }
}

impl AddAssign for AReal {
    #[inline]
    fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
}
impl SubAssign for AReal {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs; }
}
impl MulAssign for AReal {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; }
}
impl DivAssign for AReal {
    #[inline]
    fn div_assign(&mut self, rhs: Self) { *self = *self / rhs; }
}

// ===========================================================================
// Number trait for AReal
// ===========================================================================

impl Number for AReal {
    #[inline]
    fn from_f64(v: f64) -> Self {
        // Constants don't need derivatives — push a node with no children
        push_tl(v, SmallVec::new())
    }

    #[inline]
    fn to_f64(self) -> f64 { self.val }

    #[inline]
    fn exp(self) -> Self {
        let e = self.val.exp();
        push_tl(e, smallvec![(self.idx, e)])
    }

    #[inline]
    fn ln(self) -> Self {
        push_tl(self.val.ln(), smallvec![(self.idx, 1.0 / self.val)])
    }

    #[inline]
    fn sqrt(self) -> Self {
        let s = self.val.sqrt();
        push_tl(s, smallvec![(self.idx, 0.5 / s)])
    }

    #[inline]
    fn abs(self) -> Self {
        let d = if self.val >= 0.0 { 1.0 } else { -1.0 };
        push_tl(self.val.abs(), smallvec![(self.idx, d)])
    }

    #[inline]
    fn powf(self, n: Self) -> Self {
        let val = self.val.powf(n.val);
        let da = n.val * self.val.powf(n.val - 1.0);
        let db = val * self.val.ln();
        push_tl(val, SmallVec::from_buf([(self.idx, da), (n.idx, db)]))
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        let val = self.val.powi(n);
        let d = n as f64 * self.val.powi(n - 1);
        push_tl(val, smallvec![(self.idx, d)])
    }

    #[inline]
    fn sin(self) -> Self {
        push_tl(self.val.sin(), smallvec![(self.idx, self.val.cos())])
    }

    #[inline]
    fn cos(self) -> Self {
        push_tl(self.val.cos(), smallvec![(self.idx, -self.val.sin())])
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        if self.val >= other.val {
            push_tl(self.val, smallvec![(self.idx, 1.0)])
        } else {
            push_tl(other.val, smallvec![(other.idx, 1.0)])
        }
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        if self.val <= other.val {
            push_tl(self.val, smallvec![(self.idx, 1.0)])
        } else {
            push_tl(other.val, smallvec![(other.idx, 1.0)])
        }
    }

    #[inline]
    fn recip(self) -> Self {
        let inv = 1.0 / self.val;
        push_tl(inv, smallvec![(self.idx, -inv * inv)])
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        let denom = self.val * self.val + other.val * other.val;
        let da = other.val / denom;
        let db = -self.val / denom;
        push_tl(self.val.atan2(other.val), SmallVec::from_buf([(self.idx, da), (other.idx, db)]))
    }

    #[inline]
    fn tan(self) -> Self {
        let c = self.val.cos();
        push_tl(self.val.tan(), smallvec![(self.idx, 1.0 / (c * c))])
    }

    #[inline]
    fn asin(self) -> Self {
        push_tl(self.val.asin(), smallvec![(self.idx, 1.0 / (1.0 - self.val * self.val).sqrt())])
    }

    #[inline]
    fn acos(self) -> Self {
        push_tl(self.val.acos(), smallvec![(self.idx, -1.0 / (1.0 - self.val * self.val).sqrt())])
    }

    #[inline]
    fn atan(self) -> Self {
        push_tl(self.val.atan(), smallvec![(self.idx, 1.0 / (1.0 + self.val * self.val))])
    }

    #[inline]
    fn sinh(self) -> Self {
        push_tl(self.val.sinh(), smallvec![(self.idx, self.val.cosh())])
    }

    #[inline]
    fn cosh(self) -> Self {
        push_tl(self.val.cosh(), smallvec![(self.idx, self.val.sinh())])
    }

    #[inline]
    fn tanh(self) -> Self {
        let t = self.val.tanh();
        push_tl(t, smallvec![(self.idx, 1.0 - t * t)])
    }

    #[inline]
    fn log2(self) -> Self {
        push_tl(self.val.log2(), smallvec![(self.idx, 1.0 / (self.val * std::f64::consts::LN_2))])
    }

    #[inline]
    fn log10(self) -> Self {
        push_tl(self.val.log10(), smallvec![(self.idx, 1.0 / (self.val * std::f64::consts::LN_10))])
    }

    #[inline]
    fn floor(self) -> Self {
        push_tl(self.val.floor(), SmallVec::new()) // not differentiable
    }

    #[inline]
    fn ceil(self) -> Self {
        push_tl(self.val.ceil(), SmallVec::new()) // not differentiable
    }

    #[inline]
    fn zero() -> Self { Self::from_f64(0.0) }
    #[inline]
    fn one() -> Self { Self::from_f64(1.0) }
    #[inline]
    fn pi() -> Self { Self::from_f64(std::f64::consts::PI) }
    #[inline]
    fn epsilon() -> Self { Self::from_f64(f64::EPSILON) }

    #[inline]
    fn is_positive(self) -> bool { self.val > 0.0 }
    #[inline]
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
    fn tape_add() {
        let mut tape = Tape::new();
        let x = tape.input(3.0);
        let y = tape.input(5.0);
        let z = tape.add(x, y);
        assert_abs_diff_eq!(z.val, 8.0, epsilon = 1e-15);
        let g = tape.gradient(z, &[x, y]);
        assert_abs_diff_eq!(g[0], 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(g[1], 1.0, epsilon = 1e-15);
    }

    #[test]
    fn tape_mul() {
        let mut tape = Tape::new();
        let x = tape.input(3.0);
        let y = tape.input(5.0);
        let z = tape.mul(x, y);
        assert_abs_diff_eq!(z.val, 15.0, epsilon = 1e-15);
        let g = tape.gradient(z, &[x, y]);
        assert_abs_diff_eq!(g[0], 5.0, epsilon = 1e-15); // ∂(xy)/∂x = y
        assert_abs_diff_eq!(g[1], 3.0, epsilon = 1e-15); // ∂(xy)/∂y = x
    }

    #[test]
    fn tape_polynomial() {
        // f(x) = x³ + 2x, f'(x) = 3x² + 2
        let mut tape = Tape::new();
        let x = tape.input(2.0);
        let x2 = tape.mul(x, x);
        let x3 = tape.mul(x2, x);
        let two_x = tape.mul_const(x, 2.0);
        let f = tape.add(x3, two_x);
        assert_abs_diff_eq!(f.val, 12.0, epsilon = 1e-14); // 8 + 4
        let g = tape.gradient(f, &[x]);
        assert_abs_diff_eq!(g[0], 14.0, epsilon = 1e-13); // 3*4 + 2
    }

    #[test]
    fn tape_exp() {
        let mut tape = Tape::new();
        let x = tape.input(1.0);
        let e = tape.exp(x);
        let g = tape.gradient(e, &[x]);
        assert_abs_diff_eq!(g[0], 1.0_f64.exp(), epsilon = 1e-14);
    }

    #[test]
    fn tape_ln() {
        let mut tape = Tape::new();
        let x = tape.input(2.0);
        let l = tape.ln(x);
        let g = tape.gradient(l, &[x]);
        assert_abs_diff_eq!(g[0], 0.5, epsilon = 1e-14); // 1/x
    }

    #[test]
    fn tape_sqrt() {
        let mut tape = Tape::new();
        let x = tape.input(4.0);
        let s = tape.sqrt(x);
        assert_abs_diff_eq!(s.val, 2.0, epsilon = 1e-14);
        let g = tape.gradient(s, &[x]);
        assert_abs_diff_eq!(g[0], 0.25, epsilon = 1e-14); // 1/(2√4)
    }

    #[test]
    fn tape_div() {
        let mut tape = Tape::new();
        let x = tape.input(6.0);
        let y = tape.input(3.0);
        let z = tape.div(x, y); // x/y = 2
        assert_abs_diff_eq!(z.val, 2.0, epsilon = 1e-14);
        let g = tape.gradient(z, &[x, y]);
        assert_abs_diff_eq!(g[0], 1.0 / 3.0, epsilon = 1e-14); // 1/y
        assert_abs_diff_eq!(g[1], -2.0 / 3.0, epsilon = 1e-14); // -x/y²
    }

    #[test]
    fn tape_chain_rule() {
        // f(x) = exp(x²), f'(x) = 2x·exp(x²)
        let mut tape = Tape::new();
        let x = tape.input(1.5);
        let x2 = tape.mul(x, x);
        let e = tape.exp(x2);
        let g = tape.gradient(e, &[x]);
        let expected = 2.0 * 1.5 * (1.5 * 1.5_f64).exp();
        assert_abs_diff_eq!(g[0], expected, epsilon = 1e-11);
    }

    #[test]
    fn tape_clear_and_reuse() {
        let mut tape = Tape::new();
        let x = tape.input(2.0);
        let _y = tape.mul(x, x);
        assert_eq!(tape.len(), 2); // input + mul

        tape.clear();
        assert_eq!(tape.len(), 0);

        // Reuse
        let a = tape.input(5.0);
        let b = tape.exp(a);
        let g = tape.gradient(b, &[a]);
        assert_abs_diff_eq!(g[0], 5.0_f64.exp(), epsilon = 1e-13);
    }

    #[test]
    fn tape_max_subgradient() {
        let mut tape = Tape::new();
        let x = tape.input(3.0);
        let zero = tape.input(0.0);
        let m = tape.max(x, zero);
        let g = tape.gradient(m, &[x, zero]);
        assert_abs_diff_eq!(g[0], 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(g[1], 0.0, epsilon = 1e-15);

        tape.clear();
        let x2 = tape.input(-2.0);
        let zero2 = tape.input(0.0);
        let m2 = tape.max(x2, zero2);
        let g2 = tape.gradient(m2, &[x2, zero2]);
        assert_abs_diff_eq!(g2[0], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(g2[1], 1.0, epsilon = 1e-15);
    }

    #[test]
    fn tape_many_inputs() {
        // f(x₁, ..., x₁₀) = Σ xᵢ², ∂f/∂xᵢ = 2xᵢ
        let mut tape = Tape::new();
        let inputs: Vec<AReal> = (1..=10).map(|i| tape.input(i as f64)).collect();
        let mut sum = tape.mul(inputs[0], inputs[0]);
        for &xi in &inputs[1..] {
            let xi2 = tape.mul(xi, xi);
            sum = tape.add(sum, xi2);
        }
        let g = tape.gradient(sum, &inputs);
        for (i, &grad) in g.iter().enumerate() {
            assert_abs_diff_eq!(grad, 2.0 * (i + 1) as f64, epsilon = 1e-12);
        }
    }

    #[test]
    fn tape_sin_cos() {
        let mut tape = Tape::new();
        let x = tape.input(std::f64::consts::FRAC_PI_4);
        let s = tape.sin(x);
        let _c = tape.cos(x);

        let gs = tape.gradient(s, &[x]);
        assert_abs_diff_eq!(gs[0], x.val.cos(), epsilon = 1e-14);

        // Need fresh tape for cos since adjoint is computed from cos node
        let mut tape2 = Tape::new();
        let x2 = tape2.input(std::f64::consts::FRAC_PI_4);
        let c2 = tape2.cos(x2);
        let gc = tape2.gradient(c2, &[x2]);
        assert_abs_diff_eq!(gc[0], -x2.val.sin(), epsilon = 1e-14);
    }
}
