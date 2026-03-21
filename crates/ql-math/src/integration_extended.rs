//! Extended numerical integration: Trapezoid, Segment (mid-point), 2D integral, Filon.
//!
//! **G29** — TrapezoidIntegral (composite trapezoidal rule)
//! **G30** — SegmentIntegral (mid-point rule)
//! **G31** — TwoDimensionalIntegral (iterated 1D integration)
//! **G32** — FilonIntegral (oscillatory integrands)

use crate::integration::Integrator;
use ql_core::errors::{QLError, QLResult};

// ===========================================================================
// Trapezoid Integral (G29)
// ===========================================================================

/// Composite trapezoidal rule.
///
/// Straightforward quadrature: subdivides `[a,b]` into `n` equal panels
/// and applies the trapezoidal rule. Error is O(h²).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TrapezoidIntegral {
    /// Number of intervals.
    pub intervals: usize,
}

impl TrapezoidIntegral {
    pub fn new(intervals: usize) -> Self {
        Self {
            intervals: intervals.max(1),
        }
    }
}

impl Integrator for TrapezoidIntegral {
    fn integrate<F: Fn(f64) -> f64>(&self, f: F, a: f64, b: f64) -> QLResult<f64> {
        if a >= b {
            return Err(QLError::InvalidArgument(
                "integration bounds: a must be less than b".into(),
            ));
        }
        let n = self.intervals;
        let h = (b - a) / n as f64;
        let mut sum = 0.5 * (f(a) + f(b));
        for i in 1..n {
            sum += f(a + i as f64 * h);
        }
        Ok(sum * h)
    }
}

// ===========================================================================
// Segment (Mid-point) Integral (G30)
// ===========================================================================

/// Mid-point rule (segment integral).
///
/// Evaluates the integrand at the centre of each sub-interval.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SegmentIntegral {
    /// Number of segments.
    pub intervals: usize,
}

impl SegmentIntegral {
    pub fn new(intervals: usize) -> Self {
        Self {
            intervals: intervals.max(1),
        }
    }
}

impl Integrator for SegmentIntegral {
    fn integrate<F: Fn(f64) -> f64>(&self, f: F, a: f64, b: f64) -> QLResult<f64> {
        if a >= b {
            return Err(QLError::InvalidArgument(
                "integration bounds: a must be less than b".into(),
            ));
        }
        let n = self.intervals;
        let h = (b - a) / n as f64;
        let mut sum = 0.0;
        for i in 0..n {
            sum += f(a + (i as f64 + 0.5) * h);
        }
        Ok(sum * h)
    }
}

// ===========================================================================
// Two-Dimensional Integral (G31)
// ===========================================================================

/// Iterated 1D integration over a 2D rectangular domain `[a₁,b₁]×[a₂,b₂]`.
///
/// Uses an outer `Integrator` for the first dimension and an inner one for the
/// second. This mirrors QuantLib's `TwoDimensionalIntegral`.
#[derive(Clone, Debug)]
pub struct TwoDimensionalIntegral<I: Integrator + Clone> {
    integrator: I,
}

impl<I: Integrator + Clone> TwoDimensionalIntegral<I> {
    pub fn new(integrator: I) -> Self {
        Self { integrator }
    }

    /// Integrate `f(x, y)` over `[a1, b1] × [a2, b2]`.
    pub fn integrate<F: Fn(f64, f64) -> f64>(
        &self,
        f: F,
        a1: f64,
        b1: f64,
        a2: f64,
        b2: f64,
    ) -> QLResult<f64> {
        let inner = self.integrator.clone();
        let outer = self.integrator.clone();
        outer.integrate(
            |x| {
                inner
                    .integrate(|y| f(x, y), a2, b2)
                    .unwrap_or(0.0)
            },
            a1,
            b1,
        )
    }
}

// ===========================================================================
// Filon Integral (G32)
// ===========================================================================

/// Filon's method for highly oscillatory integrands of the form
/// `∫ f(x) · sin(ωx) dx` or `∫ f(x) · cos(ωx) dx`.
///
/// Approximates `f(x)` with piecewise quadratics and integrates the product
/// with `sin`/`cos` analytically, avoiding cancellation.
///
/// Reference: Filon (1928). "On a quadrature formula for trigonometric integrals."
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FilonIntegral {
    /// Number of sub-intervals (must be even).
    pub intervals: usize,
    /// Oscillation type.
    pub kind: FilonKind,
}

/// Whether the oscillatory factor is sine or cosine.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum FilonKind {
    /// Integrate `f(x) · cos(ωx)`
    Cosine,
    /// Integrate `f(x) · sin(ωx)`
    Sine,
}

impl FilonIntegral {
    pub fn new(intervals: usize, kind: FilonKind) -> Self {
        let intervals = if intervals.is_multiple_of(2) {
            intervals.max(2)
        } else {
            (intervals + 1).max(2)
        };
        Self { intervals, kind }
    }

    /// Integrate `f(x) · {cos,sin}(ω·x)` over `[a, b]`.
    pub fn integrate<F: Fn(f64) -> f64>(
        &self,
        f: F,
        omega: f64,
        a: f64,
        b: f64,
    ) -> QLResult<f64> {
        if a >= b {
            return Err(QLError::InvalidArgument(
                "integration bounds: a must be less than b".into(),
            ));
        }
        let n = self.intervals; // must be even
        let h = (b - a) / n as f64;
        let theta = omega * h;

        if theta.abs() < 1e-12 {
            // Low-frequency limit: just do trapezoidal on f(x)
            let trap = TrapezoidIntegral::new(n);
            return trap.integrate(&f, a, b);
        }

        let s = theta.sin();
        let c = theta.cos();
        let s2 = (2.0 * theta).sin();
        let theta2 = theta * theta;
        let theta3 = theta2 * theta;

        // Filon coefficients
        let alpha = (theta2 + theta * s * c - 2.0 * s * s) / theta3;
        let beta = 2.0 * (theta * (1.0 + c * c) - s2) / theta3;
        let gamma = 4.0 * (s - theta * c) / theta3;

        let mut c_even = 0.0; // sum of f at even nodes
        let mut c_odd = 0.0; // sum of f at odd nodes

        let mut f_vals = Vec::with_capacity(n + 1);
        for i in 0..=n {
            f_vals.push(f(a + i as f64 * h));
        }

        match self.kind {
            FilonKind::Cosine => {
                let f_end = f_vals[0] * (omega * a).sin() - f_vals[n] * (omega * b).sin();

                for i in (2..n).step_by(2) {
                    c_even += f_vals[i] * (omega * (a + i as f64 * h)).cos();
                }
                for i in (1..=n - 1).step_by(2) {
                    c_odd += f_vals[i] * (omega * (a + i as f64 * h)).cos();
                }

                Ok(h * (alpha * f_end + beta * c_even + gamma * c_odd))
            }
            FilonKind::Sine => {
                let f_end = f_vals[0] * (omega * a).cos() - f_vals[n] * (omega * b).cos();

                for i in (2..n).step_by(2) {
                    c_even += f_vals[i] * (omega * (a + i as f64 * h)).sin();
                }
                for i in (1..=n - 1).step_by(2) {
                    c_odd += f_vals[i] * (omega * (a + i as f64 * h)).sin();
                }

                Ok(h * (-alpha * f_end + beta * c_even + gamma * c_odd))
            }
        }
    }
}

// ===========================================================================
// Numerical Differentiation (G60)
// ===========================================================================

/// Numerical differentiation via finite differences.
///
/// Computes first (and optionally second) derivatives using central, forward,
/// or backward difference schemes.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum DifferenceScheme {
    Central,
    Forward,
    Backward,
}

/// Numerically compute the first derivative of `f` at `x`.
pub fn numerical_derivative<F: Fn(f64) -> f64>(
    f: &F,
    x: f64,
    h: f64,
    scheme: &DifferenceScheme,
) -> f64 {
    match scheme {
        DifferenceScheme::Central => (f(x + h) - f(x - h)) / (2.0 * h),
        DifferenceScheme::Forward => (f(x + h) - f(x)) / h,
        DifferenceScheme::Backward => (f(x) - f(x - h)) / h,
    }
}

/// Numerically compute the second derivative of `f` at `x` using central differences.
pub fn numerical_second_derivative<F: Fn(f64) -> f64>(f: &F, x: f64, h: f64) -> f64 {
    (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)
}

/// Compute a vector of numerical derivatives for a function of a vector argument.
///
/// Returns ∂f/∂xᵢ for each component using central differences.
pub fn gradient<F: Fn(&[f64]) -> f64>(f: &F, x: &[f64], h: f64) -> Vec<f64> {
    let n = x.len();
    let mut grad = vec![0.0; n];
    let mut x_plus = x.to_vec();
    let mut x_minus = x.to_vec();
    for i in 0..n {
        x_plus[i] = x[i] + h;
        x_minus[i] = x[i] - h;
        grad[i] = (f(&x_plus) - f(&x_minus)) / (2.0 * h);
        x_plus[i] = x[i];
        x_minus[i] = x[i];
    }
    grad
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn trapezoid_quadratic() {
        // ∫₀¹ x² dx = 1/3
        let ti = TrapezoidIntegral::new(10000);
        let result = ti.integrate(|x| x * x, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(result, 1.0 / 3.0, epsilon = 1e-8);
    }

    #[test]
    fn trapezoid_sin() {
        // ∫₀^π sin(x) dx = 2
        let ti = TrapezoidIntegral::new(10000);
        let result = ti.integrate(f64::sin, 0.0, std::f64::consts::PI).unwrap();
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-7);
    }

    #[test]
    fn segment_integral_quadratic() {
        let si = SegmentIntegral::new(10000);
        let result = si.integrate(|x| x * x, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(result, 1.0 / 3.0, epsilon = 1e-8);
    }

    #[test]
    fn two_dimensional_integral() {
        // ∫₀¹∫₀¹ (x+y) dx dy = 1
        let si = crate::integration::SimpsonIntegral::new(100);
        let ti = TwoDimensionalIntegral::new(si);
        let result = ti.integrate(|x, y| x + y, 0.0, 1.0, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(result, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn two_dimensional_integral_product() {
        // ∫₀¹∫₀¹ x·y dx dy = 0.25
        let si = crate::integration::SimpsonIntegral::new(100);
        let ti = TwoDimensionalIntegral::new(si);
        let result = ti.integrate(|x, y| x * y, 0.0, 1.0, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(result, 0.25, epsilon = 1e-6);
    }

    #[test]
    fn filon_cosine() {
        // ∫₀^π x·cos(x) dx = -2 (by parts)
        let fi = FilonIntegral::new(100, FilonKind::Cosine);
        let result = fi
            .integrate(|x| x, 1.0, 0.0, std::f64::consts::PI)
            .unwrap();
        assert_abs_diff_eq!(result, -2.0, epsilon = 0.1);
    }

    #[test]
    fn filon_sine() {
        // ∫₀^π x·sin(x) dx = π (by parts)
        let fi = FilonIntegral::new(200, FilonKind::Sine);
        let result = fi
            .integrate(|x| x, 1.0, 0.0, std::f64::consts::PI)
            .unwrap();
        assert_abs_diff_eq!(result, std::f64::consts::PI, epsilon = 1e-3);
    }

    #[test]
    fn numerical_derivative_quadratic() {
        let f = |x: f64| x * x;
        let d = numerical_derivative(&f, 3.0, 1e-5, &DifferenceScheme::Central);
        assert_abs_diff_eq!(d, 6.0, epsilon = 1e-5);
    }

    #[test]
    fn numerical_second_derivative_cubic() {
        let f = |x: f64| x * x * x;
        // f''(x) = 6x, at x=2 → 12
        let d2 = numerical_second_derivative(&f, 2.0, 1e-4);
        assert_abs_diff_eq!(d2, 12.0, epsilon = 1e-3);
    }

    #[test]
    fn gradient_rosenbrock() {
        // f(x,y) = (1-x)^2 + 100(y-x^2)^2
        // ∂f/∂x = -2(1-x) - 400x(y-x^2)
        // ∂f/∂y = 200(y-x^2)
        let f = |v: &[f64]| (1.0 - v[0]).powi(2) + 100.0 * (v[1] - v[0] * v[0]).powi(2);
        let x = [1.0, 1.0]; // minimum, gradient should be ~0
        let g = gradient(&f, &x, 1e-6);
        assert_abs_diff_eq!(g[0], 0.0, epsilon = 1e-4);
        assert_abs_diff_eq!(g[1], 0.0, epsilon = 1e-4);
    }

    #[test]
    fn forward_and_backward_derivatives() {
        let f = |x: f64| x.exp();
        let x = 1.0;
        let h = 1e-6;
        let fwd = numerical_derivative(&f, x, h, &DifferenceScheme::Forward);
        let bwd = numerical_derivative(&f, x, h, &DifferenceScheme::Backward);
        let ctr = numerical_derivative(&f, x, h, &DifferenceScheme::Central);
        // All should be close to e^1 ≈ 2.71828
        let expected = std::f64::consts::E;
        assert_abs_diff_eq!(fwd, expected, epsilon = 1e-4);
        assert_abs_diff_eq!(bwd, expected, epsilon = 1e-4);
        assert_abs_diff_eq!(ctr, expected, epsilon = 1e-8); // central is more accurate
    }
}
