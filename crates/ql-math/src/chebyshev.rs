//! Chebyshev interpolation.
//!
//! Provides polynomial interpolation using Chebyshev nodes and the
//! barycentric formula. Chebyshev interpolation avoids Runge's phenomenon
//! and provides near-optimal polynomial approximation.
//!
//! Corresponds to QuantLib's `ChebyshevInterpolation`.

use serde::{Deserialize, Serialize};

/// Chebyshev polynomial interpolation on [a, b].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChebyshevInterpolation {
    /// Number of nodes.
    pub n: usize,
    /// Lower bound of interval.
    pub a: f64,
    /// Upper bound of interval.
    pub b: f64,
    /// Chebyshev coefficients.
    pub coefficients: Vec<f64>,
    /// Function values at Chebyshev nodes.
    pub values: Vec<f64>,
    /// Chebyshev nodes on [a, b].
    pub nodes: Vec<f64>,
}

impl ChebyshevInterpolation {
    /// Create Chebyshev interpolation of degree n−1 for function f on [a, b].
    ///
    /// Uses n Chebyshev nodes of the first kind.
    pub fn new<F: Fn(f64) -> f64>(n: usize, a: f64, b: f64, f: F) -> Self {
        assert!(n >= 1);
        assert!(b > a);

        let nodes: Vec<f64> = (0..n)
            .map(|k| {
                // Chebyshev node on [-1, 1]
                let t = ((2 * k + 1) as f64 * std::f64::consts::PI / (2 * n) as f64).cos();
                // Map to [a, b]
                0.5 * ((b - a) * t + a + b)
            })
            .collect();

        let values: Vec<f64> = nodes.iter().map(|&x| f(x)).collect();

        // Compute Chebyshev coefficients via DCT
        let coefficients = compute_chebyshev_coefficients(&values, n);

        Self { n, a, b, coefficients, values, nodes }
    }

    /// Create from pre-computed function values at Chebyshev nodes.
    pub fn from_values(n: usize, a: f64, b: f64, values: Vec<f64>) -> Self {
        assert_eq!(values.len(), n);
        let nodes: Vec<f64> = (0..n)
            .map(|k| {
                let t = ((2 * k + 1) as f64 * std::f64::consts::PI / (2 * n) as f64).cos();
                0.5 * ((b - a) * t + a + b)
            })
            .collect();
        let coefficients = compute_chebyshev_coefficients(&values, n);
        Self { n, a, b, coefficients, values, nodes }
    }

    /// Evaluate the interpolation at x using Clenshaw's algorithm.
    pub fn evaluate(&self, x: f64) -> f64 {
        // Map x from [a, b] to [-1, 1]
        let t = (2.0 * x - self.a - self.b) / (self.b - self.a);
        let t = t.clamp(-1.0, 1.0);
        clenshaw(&self.coefficients, t)
    }

    /// Evaluate using the barycentric formula (alternative method).
    pub fn evaluate_barycentric(&self, x: f64) -> f64 {
        // Check if x is very close to a node
        for (i, &node) in self.nodes.iter().enumerate() {
            if (x - node).abs() < 1e-15 {
                return self.values[i];
            }
        }

        let mut num = 0.0;
        let mut den = 0.0;
        for (i, (&node, &val)) in self.nodes.iter().zip(self.values.iter()).enumerate() {
            // Barycentric weights for Chebyshev first-kind nodes
            let w_i = if i % 2 == 0 { 1.0 } else { -1.0 }
                * ((2 * i + 1) as f64 * std::f64::consts::PI / (2 * self.n) as f64).sin();
            let term = w_i / (x - node);
            num += term * val;
            den += term;
        }
        num / den
    }

    /// Compute the integral of the interpolant over [a, b].
    pub fn integrate(&self) -> f64 {
        let half_width = (self.b - self.a) / 2.0;
        let mut integral = 0.0;
        for (k, &c) in self.coefficients.iter().enumerate() {
            if k % 2 == 0 {
                // Integral of T_k over [-1,1] = 2/(1 - k²) for even k, 0 for odd k
                if k == 0 {
                    integral += c * 2.0;
                } else {
                    integral += c * 2.0 / (1.0 - (k as f64).powi(2));
                }
            }
        }
        integral * half_width
    }

    /// Maximum degree of the polynomial.
    pub fn degree(&self) -> usize {
        self.n.saturating_sub(1)
    }
}

/// Compute Chebyshev coefficients from function values via DCT-like transform.
fn compute_chebyshev_coefficients(values: &[f64], n: usize) -> Vec<f64> {
    let mut coeffs = vec![0.0; n];
    for j in 0..n {
        let mut sum = 0.0;
        for (k, &val) in values.iter().enumerate() {
            let angle = std::f64::consts::PI * j as f64 * (2 * k + 1) as f64 / (2 * n) as f64;
            sum += val * angle.cos();
        }
        coeffs[j] = sum * 2.0 / n as f64;
    }
    coeffs[0] /= 2.0;
    coeffs
}

/// Clenshaw's algorithm for evaluating Σ c_k T_k(t).
fn clenshaw(coeffs: &[f64], t: f64) -> f64 {
    let n = coeffs.len();
    if n == 0 { return 0.0; }
    if n == 1 { return coeffs[0]; }

    let mut b_k1 = 0.0;
    let mut b_k2 = 0.0;
    for k in (1..n).rev() {
        let b_k = coeffs[k] + 2.0 * t * b_k1 - b_k2;
        b_k2 = b_k1;
        b_k1 = b_k;
    }
    coeffs[0] + t * b_k1 - b_k2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chebyshev_polynomial() {
        // Interpolate x² on [0, 1] — should be exact with n ≥ 3
        let cheb = ChebyshevInterpolation::new(5, 0.0, 1.0, |x| x * x);
        for &x in &[0.0, 0.1, 0.25, 0.5, 0.75, 1.0] {
            let y = cheb.evaluate(x);
            assert!((y - x * x).abs() < 1e-10, "x={}, y={}, expected={}", x, y, x * x);
        }
    }

    #[test]
    fn test_chebyshev_sin() {
        // Interpolate sin(x) on [0, π]
        let cheb = ChebyshevInterpolation::new(20, 0.0, std::f64::consts::PI, |x| x.sin());
        for &x in &[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] {
            let y = cheb.evaluate(x);
            let expected = x.sin();
            assert!((y - expected).abs() < 1e-8, "x={}, y={}, expected={}", x, y, expected);
        }
    }

    #[test]
    fn test_chebyshev_barycentric() {
        let cheb = ChebyshevInterpolation::new(10, -1.0, 1.0, |x| x.exp());
        for &x in &[-0.5, 0.0, 0.5] {
            let y1 = cheb.evaluate(x);
            let y2 = cheb.evaluate_barycentric(x);
            assert!((y1 - y2).abs() < 0.05, "clenshaw={}, bary={}", y1, y2);
        }
    }

    #[test]
    fn test_chebyshev_integrate() {
        // Integral of x² from 0 to 1 = 1/3
        let cheb = ChebyshevInterpolation::new(10, 0.0, 1.0, |x| x * x);
        let integral = cheb.integrate();
        assert!((integral - 1.0 / 3.0).abs() < 1e-6, "integral={}", integral);
    }

    #[test]
    fn test_chebyshev_constant() {
        let cheb = ChebyshevInterpolation::new(5, 0.0, 1.0, |_| 3.14);
        for &x in &[0.0, 0.5, 1.0] {
            let y = cheb.evaluate(x);
            assert!((y - 3.14).abs() < 1e-10, "y={}", y);
        }
    }
}
