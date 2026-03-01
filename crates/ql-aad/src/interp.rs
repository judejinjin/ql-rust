//! Generic interpolation — AD-aware evaluation of pre-built `f64` interpolants.
//!
//! The pattern is **"build with `f64`, evaluate with `T: Number`"**: constructors
//! accept `f64` data and precompute coefficients, while `eval` methods are generic
//! so derivatives propagate through the interpolated value.
//!
//! This mirrors the interpolation types in `ql-math` but adds generic evaluation
//! for automatic differentiation.
//!
//! # Example
//!
//! ```
//! use ql_aad::interp::LinearInterp;
//! use ql_aad::{Dual, Number};
//!
//! let xs = vec![0.0, 1.0, 2.0, 3.0];
//! let ys = vec![0.0, 1.0, 4.0, 9.0]; // roughly x²
//! let interp = LinearInterp::new(xs, ys).unwrap();
//!
//! // f64 evaluation
//! assert!((interp.eval(1.5_f64) - 2.5).abs() < 1e-14);
//!
//! // AD evaluation — derivative is the local slope
//! let x = Dual::variable(1.5);
//! let y = interp.eval(x);
//! assert!((y.val - 2.5).abs() < 1e-14);
//! assert!((y.dot - 3.0).abs() < 1e-14); // slope in [1,2] = (4-1)/(2-1) = 3
//! ```

use crate::number::Number;

// ===========================================================================
// Helpers
// ===========================================================================

/// Locate segment index `i` such that `xs[i] <= x < xs[i+1]`.
///
/// Performs binary search; always returns a valid index in `[0, n-2]`.
/// Input `x` is f64 (the differentiation target is the *value*, not the location).
#[inline]
fn locate(xs: &[f64], x: f64) -> usize {
    let n = xs.len();
    debug_assert!(n >= 2);
    if x <= xs[0] {
        return 0;
    }
    if x >= xs[n - 2] {
        return n - 2;
    }
    // For small arrays, linear scan is faster
    if n <= 20 {
        for j in 1..n {
            if xs[j] > x {
                return j - 1;
            }
        }
        return n - 2;
    }
    // Binary search
    let pos = xs.partition_point(|&xi| xi <= x);
    if pos == 0 { 0 } else { (pos - 1).min(n - 2) }
}

fn validate(xs: &[f64], ys: &[f64]) -> Result<(), &'static str> {
    if xs.len() != ys.len() {
        return Err("x and y must have equal length");
    }
    if xs.len() < 2 {
        return Err("need at least 2 points");
    }
    for w in xs.windows(2) {
        if w[1] <= w[0] {
            return Err("x values must be strictly increasing");
        }
    }
    Ok(())
}

// ===========================================================================
// Linear Interpolation
// ===========================================================================

/// Piecewise linear interpolation with AD-aware evaluation.
///
/// Stores `f64` data; evaluates generically over `T: Number`.
#[derive(Clone, Debug)]
pub struct LinearInterp {
    xs: Vec<f64>,
    ys: Vec<f64>,
    slopes: Vec<f64>,
}

impl LinearInterp {
    /// Build from sorted (x, y) data.
    pub fn new(xs: Vec<f64>, ys: Vec<f64>) -> Result<Self, &'static str> {
        validate(&xs, &ys)?;
        let slopes: Vec<f64> = xs
            .windows(2)
            .zip(ys.windows(2))
            .map(|(xw, yw)| (yw[1] - yw[0]) / (xw[1] - xw[0]))
            .collect();
        Ok(Self { xs, ys, slopes })
    }

    /// Evaluate at `x` with automatic differentiation support.
    ///
    /// The derivative ∂y/∂x equals the local piecewise slope.
    /// When `x` carries derivatives w.r.t. model parameters (e.g. through
    /// `AReal` or `Dual`), the interpolated value correctly propagates them.
    #[inline]
    pub fn eval<T: Number>(&self, x: T) -> T {
        let i = locate(&self.xs, x.to_f64());
        let y_i = T::from_f64(self.ys[i]);
        let slope = T::from_f64(self.slopes[i]);
        let x_i = T::from_f64(self.xs[i]);
        y_i + slope * (x - x_i)
    }

    /// Evaluate the derivative at `x` (returns `f64`; this is the slope itself).
    #[inline]
    pub fn deriv(&self, x: f64) -> f64 {
        let i = locate(&self.xs, x);
        self.slopes[i]
    }

    /// Domain bounds.
    pub fn domain(&self) -> (f64, f64) {
        (self.xs[0], *self.xs.last().unwrap())
    }
}

// ===========================================================================
// Log-Linear Interpolation
// ===========================================================================

/// Piecewise linear interpolation in log-space with AD-aware evaluation.
///
/// Interpolates `ln(y)` linearly: `y(x) = exp(a + b*(x - x_i))`.
/// All y-values must be strictly positive.
#[derive(Clone, Debug)]
pub struct LogLinearInterp {
    xs: Vec<f64>,
    log_ys: Vec<f64>,
    slopes: Vec<f64>,
}

impl LogLinearInterp {
    /// Build from sorted (x, y) data. All `y` must be > 0.
    pub fn new(xs: Vec<f64>, ys: Vec<f64>) -> Result<Self, &'static str> {
        validate(&xs, &ys)?;
        if ys.iter().any(|&y| y <= 0.0) {
            return Err("all y values must be positive for log-linear");
        }
        let log_ys: Vec<f64> = ys.iter().map(|y| y.ln()).collect();
        let slopes: Vec<f64> = xs
            .windows(2)
            .zip(log_ys.windows(2))
            .map(|(xw, lw)| (lw[1] - lw[0]) / (xw[1] - xw[0]))
            .collect();
        Ok(Self { xs, log_ys, slopes })
    }

    /// Evaluate at `x` with AD support.
    #[inline]
    pub fn eval<T: Number>(&self, x: T) -> T {
        let i = locate(&self.xs, x.to_f64());
        let log_y_i = T::from_f64(self.log_ys[i]);
        let slope = T::from_f64(self.slopes[i]);
        let x_i = T::from_f64(self.xs[i]);
        (log_y_i + slope * (x - x_i)).exp()
    }

    /// Domain bounds.
    pub fn domain(&self) -> (f64, f64) {
        (self.xs[0], *self.xs.last().unwrap())
    }
}

// ===========================================================================
// Cubic Spline Interpolation
// ===========================================================================

/// Natural cubic spline interpolation with AD-aware evaluation.
///
/// The spline coefficients are precomputed from `f64` data.
/// Evaluation is generic: `y(x) = a + b·dx + c·dx² + d·dx³` where
/// `dx = x - x_i` is `T: Number`, carrying derivatives.
#[derive(Clone, Debug)]
pub struct CubicSplineInterp {
    xs: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    d: Vec<f64>,
}

impl CubicSplineInterp {
    /// Build a natural cubic spline from sorted (x, y) data (≥ 3 points).
    pub fn new(xs: Vec<f64>, ys: Vec<f64>) -> Result<Self, &'static str> {
        validate(&xs, &ys)?;
        let n = xs.len();
        if n < 3 {
            return Err("cubic spline requires at least 3 points");
        }

        let a = ys.clone();

        // Step sizes and divided differences
        let mut h = vec![0.0; n - 1];
        let mut alpha = vec![0.0; n];
        for i in 0..n - 1 {
            h[i] = xs[i + 1] - xs[i];
        }
        for i in 1..n - 1 {
            alpha[i] = (3.0 / h[i]) * (a[i + 1] - a[i]) - (3.0 / h[i - 1]) * (a[i] - a[i - 1]);
        }

        // Tridiagonal system (natural spline: c[0] = c[n-1] = 0)
        let mut c = vec![0.0; n];
        let mut l = vec![1.0; n];
        let mut mu = vec![0.0; n];
        let mut z = vec![0.0; n];

        for i in 1..n - 1 {
            l[i] = 2.0 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }

        for j in (0..n - 1).rev() {
            c[j] = z[j] - mu[j] * c[j + 1];
        }

        let mut b_coeffs = vec![0.0; n - 1];
        let mut d_coeffs = vec![0.0; n - 1];
        for i in 0..n - 1 {
            b_coeffs[i] = (a[i + 1] - a[i]) / h[i] - h[i] * (c[i + 1] + 2.0 * c[i]) / 3.0;
            d_coeffs[i] = (c[i + 1] - c[i]) / (3.0 * h[i]);
        }

        Ok(Self {
            xs,
            a,
            b: b_coeffs,
            c,
            d: d_coeffs,
        })
    }

    /// Evaluate at `x` with AD support.
    ///
    /// Uses Horner's form: `a + dx*(b + dx*(c + dx*d))` for numerical stability.
    #[inline]
    pub fn eval<T: Number>(&self, x: T) -> T {
        let i = locate(&self.xs, x.to_f64());
        let dx = x - T::from_f64(self.xs[i]);
        let ai = T::from_f64(self.a[i]);
        let bi = T::from_f64(self.b[i]);
        let ci = T::from_f64(self.c[i]);
        let di = T::from_f64(self.d[i]);
        // Horner: a + dx*(b + dx*(c + dx*d))
        ai + dx * (bi + dx * (ci + dx * di))
    }

    /// Evaluate the first derivative at `x` (f64).
    #[inline]
    pub fn deriv(&self, x: f64) -> f64 {
        let i = locate(&self.xs, x);
        let dx = x - self.xs[i];
        self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx * dx
    }

    /// Evaluate the second derivative at `x` (f64).
    #[inline]
    pub fn deriv2(&self, x: f64) -> f64 {
        let i = locate(&self.xs, x);
        let dx = x - self.xs[i];
        2.0 * self.c[i] + 6.0 * self.d[i] * dx
    }

    /// Domain bounds.
    pub fn domain(&self) -> (f64, f64) {
        (self.xs[0], *self.xs.last().unwrap())
    }
}

// ===========================================================================
// Monotone Convex Interpolation (Hagan-West)
// ===========================================================================

/// Monotone-preserving interpolation via the Hagan-West method.
///
/// Ensures the interpolant is monotone between data points, which is critical
/// for term structure construction (prevents negative forward rates).
#[derive(Clone, Debug)]
pub struct MonotoneConvexInterp {
    xs: Vec<f64>,
    ys: Vec<f64>,
    /// Forward rates (f_i values) at internal nodes
    fs: Vec<f64>,
}

impl MonotoneConvexInterp {
    /// Build monotone convex interpolation from sorted (x, y) data (≥ 3 points).
    pub fn new(xs: Vec<f64>, ys: Vec<f64>) -> Result<Self, &'static str> {
        validate(&xs, &ys)?;
        let n = xs.len();
        if n < 3 {
            return Err("monotone convex requires at least 3 points");
        }

        // Compute discrete forward rates at segment midpoints
        let mut fs = vec![0.0; n];

        // Interior forward rates using centered differences
        for i in 1..n - 1 {
            let h_l = xs[i] - xs[i - 1];
            let h_r = xs[i + 1] - xs[i];
            let s_l = (ys[i] - ys[i - 1]) / h_l;
            let s_r = (ys[i + 1] - ys[i]) / h_r;
            // Weighted harmonic mean to preserve monotonicity
            if s_l * s_r > 0.0 {
                let w = (2.0 * h_r + h_l) / (3.0 * (h_l + h_r));
                fs[i] = s_l * s_r / (w * s_r + (1.0 - w) * s_l);
            } else {
                fs[i] = 0.0;
            }
        }
        // Endpoint slopes
        let s0 = (ys[1] - ys[0]) / (xs[1] - xs[0]);
        let sn = (ys[n - 1] - ys[n - 2]) / (xs[n - 1] - xs[n - 2]);
        fs[0] = s0;
        fs[n - 1] = sn;

        Ok(Self { xs, ys, fs })
    }

    /// Evaluate at `x` with AD support (Hermite cubic with monotone slopes).
    #[inline]
    pub fn eval<T: Number>(&self, x: T) -> T {
        let i = locate(&self.xs, x.to_f64());
        let h = self.xs[i + 1] - self.xs[i];
        let t = (x - T::from_f64(self.xs[i])) / T::from_f64(h);
        let y0 = T::from_f64(self.ys[i]);
        let y1 = T::from_f64(self.ys[i + 1]);
        let f0 = T::from_f64(self.fs[i] * h);
        let f1 = T::from_f64(self.fs[i + 1] * h);

        // Hermite basis
        let t2 = t * t;
        let t3 = t2 * t;
        let h00 = T::from_f64(2.0) * t3 - T::from_f64(3.0) * t2 + T::one();
        let h10 = t3 - T::from_f64(2.0) * t2 + t;
        let h01 = T::from_f64(-2.0) * t3 + T::from_f64(3.0) * t2;
        let h11 = t3 - t2;

        h00 * y0 + h10 * f0 + h01 * y1 + h11 * f1
    }

    /// Domain bounds.
    pub fn domain(&self) -> (f64, f64) {
        (self.xs[0], *self.xs.last().unwrap())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn xs() -> Vec<f64> { vec![1.0, 2.0, 3.0, 4.0, 5.0] }
    fn ys() -> Vec<f64> { vec![1.0, 4.0, 9.0, 16.0, 25.0] } // x²

    // -- Linear ---------------------------------------------------------

    #[test]
    fn linear_at_nodes() {
        let interp = LinearInterp::new(xs(), ys()).unwrap();
        for (&x, &y) in xs().iter().zip(ys().iter()) {
            assert_abs_diff_eq!(interp.eval(x), y, epsilon = 1e-14);
        }
    }

    #[test]
    fn linear_midpoint() {
        let interp = LinearInterp::new(xs(), ys()).unwrap();
        // Between x=2 and x=3, slope = (9-4)/(3-2) = 5
        // y(2.5) = 4 + 5*0.5 = 6.5
        assert_abs_diff_eq!(interp.eval(2.5_f64), 6.5, epsilon = 1e-14);
    }

    #[test]
    fn linear_dual_derivative() {
        use crate::dual::Dual;
        let interp = LinearInterp::new(xs(), ys()).unwrap();
        let x = Dual::variable(2.5);
        let y = interp.eval(x);
        assert_abs_diff_eq!(y.val, 6.5, epsilon = 1e-14);
        assert_abs_diff_eq!(y.dot, 5.0, epsilon = 1e-14); // slope = 5
    }

    #[test]
    fn linear_dualvec_two_curves() {
        // Imagine we're interpolating between two curves and want
        // sensitivity to the y-values (curve pillars).
        use crate::dual_vec::DualVec;
        type D2 = DualVec<2>;

        // Two pillar points, seeded as AD inputs
        let xs = vec![0.0, 1.0];
        let y0_val = 1.0;
        let y1_val = 3.0;
        let interp = LinearInterp::new(xs.clone(), vec![y0_val, y1_val]).unwrap();

        // Eval at x = 0.5: y = y0 + (y1-y0)/(1-0) * (0.5-0) = y0 + 0.5*(y1-y0)
        // ∂y/∂y0 = 1 - 0.5 = 0.5
        // ∂y/∂y1 = 0.5
        // But since the interp is built from f64, we can't differentiate w.r.t. y-values
        // directly. The derivative w.r.t. x at x=0.5 is the slope = (y1-y0) = 2.0.
        let x = D2::variable(0.5, 0);
        let y = interp.eval(x);
        assert_abs_diff_eq!(y.val, 2.0, epsilon = 1e-14);
        assert_abs_diff_eq!(y.dot[0], 2.0, epsilon = 1e-14); // slope = 2
    }

    #[test]
    fn linear_areal() {
        use crate::tape::{with_tape, adjoint_tl, AReal};
        let interp = LinearInterp::new(xs(), ys()).unwrap();

        let (y, x_ar) = with_tape(|tape| {
            let x = tape.input(2.5);
            let y = interp.eval::<AReal>(x);
            (y, x)
        });

        let grad = adjoint_tl(y);
        assert_abs_diff_eq!(y.val, 6.5, epsilon = 1e-14);
        assert_abs_diff_eq!(grad[x_ar.idx], 5.0, epsilon = 1e-14);
    }

    // -- Log-Linear -----------------------------------------------------

    #[test]
    fn loglinear_at_nodes() {
        let ys_pos = vec![1.0, 2.0, 4.0, 8.0, 16.0]; // exponential growth
        let interp = LogLinearInterp::new(xs(), ys_pos.clone()).unwrap();
        for (&x, &y) in xs().iter().zip(ys_pos.iter()) {
            assert_abs_diff_eq!(interp.eval(x), y, epsilon = 1e-12);
        }
    }

    #[test]
    fn loglinear_dual() {
        use crate::dual::Dual;
        let ys_pos = vec![1.0, 2.0, 4.0, 8.0, 16.0];
        let interp = LogLinearInterp::new(xs(), ys_pos).unwrap();
        let x = Dual::variable(2.5);
        let y = interp.eval(x);
        // Between x=2,3: ln(y) is linear from ln(2) to ln(4)
        let expected = (0.5 * 2.0_f64.ln() + 0.5 * 4.0_f64.ln()).exp();
        assert_abs_diff_eq!(y.val, expected, epsilon = 1e-12);
        assert!(y.dot.abs() > 0.0); // positive slope
    }

    // -- Cubic Spline ---------------------------------------------------

    #[test]
    fn cubic_at_nodes() {
        let interp = CubicSplineInterp::new(xs(), ys()).unwrap();
        for (&x, &y) in xs().iter().zip(ys().iter()) {
            assert_abs_diff_eq!(interp.eval(x), y, epsilon = 1e-12);
        }
    }

    #[test]
    fn cubic_smoothness() {
        // Cubic spline should give smoother results than linear
        let interp = CubicSplineInterp::new(xs(), ys()).unwrap();
        let y_mid = interp.eval(2.5_f64);
        // For x² data, spline should be close to 6.25
        assert_abs_diff_eq!(y_mid, 6.25, epsilon = 0.2);
    }

    #[test]
    fn cubic_dual_derivative() {
        use crate::dual::Dual;
        let interp = CubicSplineInterp::new(xs(), ys()).unwrap();
        let x = Dual::variable(3.0);
        let y = interp.eval(x);
        assert_abs_diff_eq!(y.val, 9.0, epsilon = 1e-12);
        // For x², the true derivative at x=3 is 2*3 = 6.
        // The spline derivative should be close.
        assert_abs_diff_eq!(y.dot, 6.0, epsilon = 0.5);
    }

    #[test]
    fn cubic_areal() {
        use crate::tape::{with_tape, adjoint_tl, AReal};
        let interp = CubicSplineInterp::new(xs(), ys()).unwrap();

        let (y, x_ar) = with_tape(|tape| {
            let x = tape.input(3.0);
            let y = interp.eval::<AReal>(x);
            (y, x)
        });
        let grad = adjoint_tl(y);
        assert_abs_diff_eq!(y.val, 9.0, epsilon = 1e-12);
        // Spline deriv at node should be close to 2*3 = 6
        assert_abs_diff_eq!(grad[x_ar.idx], interp.deriv(3.0), epsilon = 1e-10);
    }

    #[test]
    fn cubic_deriv_vs_dual() {
        use crate::dual::Dual;
        let interp = CubicSplineInterp::new(xs(), ys()).unwrap();
        // Verify that Dual derivative matches the analytic spline derivative
        for &x in &[1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5] {
            let d = interp.eval(Dual::variable(x));
            let analytic = interp.deriv(x);
            assert_abs_diff_eq!(d.dot, analytic, epsilon = 1e-10);
        }
    }

    // -- Monotone Convex ------------------------------------------------

    #[test]
    fn monotone_at_nodes() {
        let interp = MonotoneConvexInterp::new(xs(), ys()).unwrap();
        for (&x, &y) in xs().iter().zip(ys().iter()) {
            assert_abs_diff_eq!(interp.eval(x), y, epsilon = 1e-10);
        }
    }

    #[test]
    fn monotone_preserves_monotonicity() {
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // strictly increasing
        let interp = MonotoneConvexInterp::new(xs, ys).unwrap();
        let mut prev = interp.eval(0.0_f64);
        for i in 1..40 {
            let x = i as f64 * 0.1;
            let y = interp.eval(x);
            assert!(y >= prev, "monotonicity violated at x={}: {} < {}", x, y, prev);
            prev = y;
        }
    }

    #[test]
    fn monotone_dual() {
        use crate::dual::Dual;
        let interp = MonotoneConvexInterp::new(xs(), ys()).unwrap();
        let x = Dual::variable(2.5);
        let y = interp.eval(x);
        // Should be close to 6.25 (x²) with positive derivative
        assert!(y.val > 5.0 && y.val < 8.0);
        assert!(y.dot > 0.0, "derivative should be positive for increasing data");
    }

    // -- Error handling ------------------------------------------------

    #[test]
    fn linear_rejects_short() {
        assert!(LinearInterp::new(vec![1.0], vec![1.0]).is_err());
    }

    #[test]
    fn linear_rejects_non_sorted() {
        assert!(LinearInterp::new(vec![2.0, 1.0], vec![1.0, 2.0]).is_err());
    }

    #[test]
    fn loglinear_rejects_negative() {
        assert!(LogLinearInterp::new(vec![1.0, 2.0], vec![1.0, -1.0]).is_err());
    }

    #[test]
    fn cubic_rejects_too_few() {
        assert!(CubicSplineInterp::new(vec![1.0, 2.0], vec![1.0, 2.0]).is_err());
    }
}
