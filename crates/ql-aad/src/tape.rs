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
