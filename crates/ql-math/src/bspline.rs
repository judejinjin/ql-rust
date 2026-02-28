//! B-spline basis functions.
//!
//! Provides B-spline basis function evaluation and curve fitting.
//! B-splines of order k with knots {t_0, t_1, ..., t_n+k} form a basis
//! for the space of piecewise polynomials of degree k−1.
//!
//! Corresponds to QuantLib's `BSpline` class.

use serde::{Deserialize, Serialize};

/// B-spline basis and curve representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BSpline {
    /// Knot vector (sorted, with multiplicity).
    pub knots: Vec<f64>,
    /// Order (degree + 1). Degree 3 → order 4 (cubic).
    pub order: usize,
}

impl BSpline {
    /// Create a new B-spline with given knot vector and order.
    ///
    /// The knot vector must be non-decreasing.
    /// For `n` basis functions of order `k`, need `n + k` knots.
    pub fn new(knots: Vec<f64>, order: usize) -> Self {
        assert!(order >= 1, "order must be >= 1");
        assert!(knots.len() >= order, "need at least `order` knots");
        Self { knots, order }
    }

    /// Create a uniform B-spline with `n` basis functions and `order`.
    pub fn uniform(n: usize, order: usize, x_min: f64, x_max: f64) -> Self {
        let num_knots = n + order;
        let num_internal = num_knots as isize - 2 * order as isize;
        let mut knots = Vec::with_capacity(num_knots);

        // Clamped knots: first `order` knots at x_min, last `order` at x_max
        for _ in 0..order {
            knots.push(x_min);
        }
        if num_internal > 0 {
            let step = (x_max - x_min) / (num_internal + 1) as f64;
            for i in 1..=num_internal as usize {
                knots.push(x_min + i as f64 * step);
            }
        }
        for _ in 0..order {
            knots.push(x_max);
        }

        Self { knots, order }
    }

    /// Number of basis functions = knots.len() - order.
    pub fn num_basis(&self) -> usize {
        self.knots.len() - self.order
    }

    /// Evaluate the i-th B-spline basis function N_{i,k}(x) using Cox-de Boor recursion.
    ///
    /// i is 0-based, k = self.order.
    pub fn basis(&self, i: usize, x: f64) -> f64 {
        self.cox_de_boor(i, self.order, x)
    }

    /// Evaluate all basis functions at x, returning a vector of length `num_basis`.
    pub fn all_basis(&self, x: f64) -> Vec<f64> {
        let n = self.num_basis();
        (0..n).map(|i| self.basis(i, x)).collect()
    }

    /// Evaluate the B-spline curve at x given control points (coefficients).
    ///
    /// S(x) = Σ_i c_i N_{i,k}(x)
    pub fn evaluate(&self, x: f64, coefficients: &[f64]) -> f64 {
        assert_eq!(coefficients.len(), self.num_basis());
        let basis = self.all_basis(x);
        basis.iter().zip(coefficients.iter()).map(|(&b, &c)| b * c).sum()
    }

    /// Fit coefficients to data points using least squares.
    ///
    /// Solves min Σ_j (S(x_j) − y_j)² for the coefficients.
    pub fn fit(&self, xs: &[f64], ys: &[f64]) -> Vec<f64> {
        assert_eq!(xs.len(), ys.len());
        let n = self.num_basis();
        let m = xs.len();

        // Build basis matrix B (m × n) and normal equations B^T B c = B^T y
        let mut btb = vec![vec![0.0; n]; n];
        let mut bty = vec![0.0; n];

        for j in 0..m {
            let b = self.all_basis(xs[j]);
            for p in 0..n {
                bty[p] += b[p] * ys[j];
                for q in 0..n {
                    btb[p][q] += b[p] * b[q];
                }
            }
        }

        // Add small regularization for numerical stability
        for p in 0..n {
            btb[p][p] += 1e-12;
        }

        // Solve via Cholesky or simple Gaussian elimination
        solve_normal_equations(&btb, &bty)
    }

    // Cox-de Boor recursion for B-spline basis N_{i,k}(x).
    fn cox_de_boor(&self, i: usize, k: usize, x: f64) -> f64 {
        if k == 1 {
            // Order 1 = degree 0 (step function)
            let ti = self.knots[i];
            let ti1 = self.knots[i + 1];
            if x >= ti && x < ti1 { return 1.0; }
            // Handle right endpoint: if x == last knot and this is the last basis
            if (x - ti1).abs() < 1e-14 && i + 1 == self.knots.len() - 1 { return 1.0; }
            return 0.0;
        }

        let mut result = 0.0;

        let ti = self.knots[i];
        let ti_k_1 = self.knots[i + k - 1];
        let denom1 = ti_k_1 - ti;
        if denom1.abs() > 1e-14 {
            result += (x - ti) / denom1 * self.cox_de_boor(i, k - 1, x);
        }

        let ti1 = self.knots[i + 1];
        let ti_k = self.knots[i + k];
        let denom2 = ti_k - ti1;
        if denom2.abs() > 1e-14 {
            result += (ti_k - x) / denom2 * self.cox_de_boor(i + 1, k - 1, x);
        }

        result
    }
}

/// Solve Ax = b via Gaussian elimination with partial pivoting.
fn solve_normal_equations(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut aug: Vec<Vec<f64>> = a.iter().enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.push(b[i]);
            r
        })
        .collect();

    // Forward elimination with partial pivoting
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in col + 1..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        if pivot.abs() < 1e-20 { continue; }

        for row in col + 1..n {
            let factor = aug[row][col] / pivot;
            for j in col..=n {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in i + 1..n {
            sum -= aug[i][j] * x[j];
        }
        if aug[i][i].abs() > 1e-20 {
            x[i] = sum / aug[i][i];
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bspline_partition_of_unity() {
        // Sum of all basis functions = 1 for clamped B-spline
        let bs = BSpline::uniform(6, 4, 0.0, 1.0);
        for &x in &[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999] {
            let sum: f64 = bs.all_basis(x).iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "x={}, sum={}", x, sum);
        }
    }

    #[test]
    fn test_bspline_linear() {
        // Order 2 (linear) B-spline
        let bs = BSpline::new(vec![0.0, 0.0, 0.5, 1.0, 1.0], 2);
        assert_eq!(bs.num_basis(), 3);
        // At x=0, only first basis is nonzero
        let b0 = bs.all_basis(0.0);
        assert!((b0[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bspline_evaluate() {
        let bs = BSpline::uniform(5, 4, 0.0, 10.0);
        let coeffs = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let y = bs.evaluate(5.0, &coeffs);
        assert!(y > 0.0, "y={}", y);
    }

    #[test]
    fn test_bspline_fit_linear_data() {
        let bs = BSpline::uniform(4, 4, 0.0, 10.0);
        let xs: Vec<f64> = (0..=20).map(|i| i as f64 * 0.5).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x + 1.0).collect();

        let coeffs = bs.fit(&xs, &ys);
        // Fitted curve should match linear data
        for &x in &[1.0, 3.0, 5.0, 7.0, 9.0] {
            let y_fit = bs.evaluate(x, &coeffs);
            let y_true = 2.0 * x + 1.0;
            assert!((y_fit - y_true).abs() < 0.5, "x={}, fit={}, true={}", x, y_fit, y_true);
        }
    }

    #[test]
    fn test_bspline_non_negative() {
        let bs = BSpline::uniform(8, 4, 0.0, 5.0);
        for &x in &[0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 4.99] {
            for b in bs.all_basis(x) {
                assert!(b >= -1e-10, "negative basis at x={}: {}", x, b);
            }
        }
    }
}
