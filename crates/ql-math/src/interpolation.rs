//! Interpolation traits and implementations.
//!
//! Provides `Interpolation` trait and concrete implementations:
//! - `LinearInterpolation` — piecewise linear
//! - `LogLinearInterpolation` — piecewise linear in log-space (for discount factors)
//! - `CubicSplineInterpolation` — natural cubic spline

use ql_core::errors::{QLError, QLResult};

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// A callable interpolation built from sorted (x, y) data.
pub trait Interpolation {
    /// Interpolated value at `x`.
    fn value(&self, x: f64) -> QLResult<f64>;

    /// First derivative at `x`.
    fn derivative(&self, x: f64) -> QLResult<f64>;

    /// Primitive (integral from x_min to `x`).
    fn primitive(&self, x: f64) -> QLResult<f64>;

    /// The domain lower bound.
    fn x_min(&self) -> f64;

    /// The domain upper bound.
    fn x_max(&self) -> f64;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Pre-analysed grid metadata for O(1) lookup on uniform grids.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum GridHint {
    /// Equally-spaced x-values: index = floor((x - x0) * inv_dx)
    Uniform { x0: f64, inv_dx: f64, last: usize },
    /// Non-uniform — fall back to binary / linear search.
    General,
}

/// Detect whether `xs` is uniformly spaced (tolerance 1e-10 × dx).
fn detect_grid(xs: &[f64]) -> GridHint {
    if xs.len() < 2 {
        return GridHint::General;
    }
    let dx = xs[1] - xs[0];
    if dx <= 0.0 {
        return GridHint::General;
    }
    let tol = 1e-10 * dx.abs();
    for i in 2..xs.len() {
        if (xs[i] - xs[i - 1] - dx).abs() > tol {
            return GridHint::General;
        }
    }
    GridHint::Uniform {
        x0: xs[0],
        inv_dx: 1.0 / dx,
        last: xs.len() - 2,
    }
}

/// Locate the segment index `i` such that `xs[i] <= x < xs[i+1]`.
/// Returns `Err` if `x` is outside the domain.
#[inline(always)]
fn locate(xs: &[f64], x: f64, hint: &GridHint) -> QLResult<usize> {
    let n = xs.len();
    debug_assert!(n >= 2);

    // Fast bounds check (rarely fails on hot-path)
    if x < xs[0] - 1e-15 || x > xs[n - 1] + 1e-15 {
        return Err(QLError::InvalidArgument(format!(
            "x = {} is outside interpolation domain [{}, {}]",
            x,
            xs[0],
            xs[n - 1]
        )));
    }

    // O(1) fast-path for uniform grids
    if let GridHint::Uniform { x0, inv_dx, last } = *hint {
        let i = ((x - x0) * inv_dx) as usize;
        return Ok(i.min(last));
    }

    // Clamp to valid range
    if x <= xs[0] {
        return Ok(0);
    }
    if x >= xs[n - 2] {
        return Ok(n - 2);
    }
    // For small arrays (typical yield curves ≤ 20 points), linear scan is faster
    if n <= 20 {
        for (j, xj) in xs.iter().enumerate().skip(1) {
            if *xj > x {
                return Ok(j - 1);
            }
        }
        return Ok(n - 2);
    }
    // Binary search for larger arrays
    let pos = xs.partition_point(|&xi| xi <= x);
    Ok(if pos == 0 { 0 } else { (pos - 1).min(n - 2) })
}

fn validate_data(xs: &[f64], ys: &[f64]) -> QLResult<()> {
    if xs.len() != ys.len() {
        return Err(QLError::InvalidArgument(
            "x and y arrays must have equal length".into(),
        ));
    }
    if xs.len() < 2 {
        return Err(QLError::InvalidArgument(
            "interpolation requires at least 2 data points".into(),
        ));
    }
    for w in xs.windows(2) {
        if w[1] <= w[0] {
            return Err(QLError::InvalidArgument(
                "x values must be strictly increasing".into(),
            ));
        }
    }
    Ok(())
}

// ===========================================================================
// Linear Interpolation
// ===========================================================================

/// Piecewise linear interpolation.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LinearInterpolation {
    xs: Vec<f64>,
    ys: Vec<f64>,
    /// Precomputed slopes: `s[i] = (y[i+1]-y[i]) / (x[i+1]-x[i])`
    slopes: Vec<f64>,
    hint: GridHint,
}

impl LinearInterpolation {
    /// Build a piecewise-linear interpolation from sorted (x, y) data.
    pub fn new(xs: Vec<f64>, ys: Vec<f64>) -> QLResult<Self> {
        validate_data(&xs, &ys)?;
        let hint = detect_grid(&xs);
        let slopes: Vec<f64> = xs
            .windows(2)
            .zip(ys.windows(2))
            .map(|(xw, yw)| (yw[1] - yw[0]) / (xw[1] - xw[0]))
            .collect();
        Ok(Self { xs, ys, slopes, hint })
    }
}

impl Interpolation for LinearInterpolation {
    fn value(&self, x: f64) -> QLResult<f64> {
        let i = locate(&self.xs, x, &self.hint)?;
        Ok(self.ys[i] + self.slopes[i] * (x - self.xs[i]))
    }

    fn derivative(&self, x: f64) -> QLResult<f64> {
        let i = locate(&self.xs, x, &self.hint)?;
        Ok(self.slopes[i])
    }

    fn primitive(&self, x: f64) -> QLResult<f64> {
        let i = locate(&self.xs, x, &self.hint)?;
        // Sum of trapezoids up to segment i, then partial segment
        let mut sum = 0.0;
        for j in 0..i {
            let dx = self.xs[j + 1] - self.xs[j];
            sum += 0.5 * (self.ys[j] + self.ys[j + 1]) * dx;
        }
        let dx = x - self.xs[i];
        let y_at_x = self.ys[i] + self.slopes[i] * dx;
        sum += 0.5 * (self.ys[i] + y_at_x) * dx;
        Ok(sum)
    }

    fn x_min(&self) -> f64 {
        self.xs[0]
    }

    fn x_max(&self) -> f64 {
        self.xs[self.xs.len() - 1]
    }
}

// ===========================================================================
// LogLinear Interpolation
// ===========================================================================

/// Piecewise linear interpolation in log-space.
///
/// Interpolates `ln(y)` linearly, so `y(x) = exp(linear_interp(x))`.
/// All `y` values must be strictly positive.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LogLinearInterpolation {
    xs: Vec<f64>,
    log_ys: Vec<f64>,
    slopes: Vec<f64>,
    hint: GridHint,
}

impl LogLinearInterpolation {
    /// Build a log-linear interpolation from sorted (x, y) data.
    ///
    /// All `y` values must be strictly positive.
    pub fn new(xs: Vec<f64>, ys: Vec<f64>) -> QLResult<Self> {
        validate_data(&xs, &ys)?;
        for &y in &ys {
            if y <= 0.0 {
                return Err(QLError::InvalidArgument(
                    "LogLinear interpolation requires strictly positive y values".into(),
                ));
            }
        }
        let hint = detect_grid(&xs);
        let log_ys: Vec<f64> = ys.iter().map(|y| y.ln()).collect();
        let slopes: Vec<f64> = xs
            .windows(2)
            .zip(log_ys.windows(2))
            .map(|(xw, lw)| (lw[1] - lw[0]) / (xw[1] - xw[0]))
            .collect();
        Ok(Self {
            xs,
            log_ys,
            slopes,
            hint,
        })
    }
}

impl Interpolation for LogLinearInterpolation {
    fn value(&self, x: f64) -> QLResult<f64> {
        let i = locate(&self.xs, x, &self.hint)?;
        let log_y = self.log_ys[i] + self.slopes[i] * (x - self.xs[i]);
        Ok(log_y.exp())
    }

    fn derivative(&self, x: f64) -> QLResult<f64> {
        let i = locate(&self.xs, x, &self.hint)?;
        let log_y = self.log_ys[i] + self.slopes[i] * (x - self.xs[i]);
        Ok(self.slopes[i] * log_y.exp())
    }

    fn primitive(&self, x: f64) -> QLResult<f64> {
        let i = locate(&self.xs, x, &self.hint)?;
        let mut sum = 0.0;
        // Integrate each full segment: ∫ exp(a + b*(x-x0)) dx = (1/b)*(exp(...)-exp(a))
        for j in 0..i {
            let a = self.log_ys[j];
            let b = self.slopes[j];
            let dx = self.xs[j + 1] - self.xs[j];
            if b.abs() < 1e-15 {
                sum += a.exp() * dx;
            } else {
                sum += (1.0 / b) * ((a + b * dx).exp() - a.exp());
            }
        }
        // Partial segment i
        let a = self.log_ys[i];
        let b = self.slopes[i];
        let dx = x - self.xs[i];
        if b.abs() < 1e-15 {
            sum += a.exp() * dx;
        } else {
            sum += (1.0 / b) * ((a + b * dx).exp() - a.exp());
        }
        Ok(sum)
    }

    fn x_min(&self) -> f64 {
        self.xs[0]
    }

    fn x_max(&self) -> f64 {
        self.xs[self.xs.len() - 1]
    }
}

// ===========================================================================
// Natural Cubic Spline Interpolation
// ===========================================================================

/// Natural cubic spline interpolation (second derivative = 0 at endpoints).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CubicSplineInterpolation {
    xs: Vec<f64>,
    /// Spline coefficients: `y(x) = a[i] + b[i]*(x-x[i]) + c[i]*(x-x[i])^2 + d[i]*(x-x[i])^3`
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    d: Vec<f64>,
    hint: GridHint,
}

impl CubicSplineInterpolation {
    /// Build a natural cubic spline from sorted (x, y) data (≥ 3 points).
    pub fn new(xs: Vec<f64>, ys: Vec<f64>) -> QLResult<Self> {
        validate_data(&xs, &ys)?;
        let n = xs.len();
        if n < 3 {
            return Err(QLError::InvalidArgument(
                "cubic spline requires at least 3 data points".into(),
            ));
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

        // Tridiagonal system for natural spline (c[0] = c[n-1] = 0)
        let mut c = vec![0.0; n];
        let mut l = vec![1.0; n];
        let mut mu = vec![0.0; n];
        let mut z = vec![0.0; n];

        for i in 1..n - 1 {
            l[i] = 2.0 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }

        // Back-substitution
        for j in (0..n - 1).rev() {
            c[j] = z[j] - mu[j] * c[j + 1];
        }

        let mut b_coeffs = vec![0.0; n - 1];
        let mut d_coeffs = vec![0.0; n - 1];
        for i in 0..n - 1 {
            b_coeffs[i] = (a[i + 1] - a[i]) / h[i] - h[i] * (c[i + 1] + 2.0 * c[i]) / 3.0;
            d_coeffs[i] = (c[i + 1] - c[i]) / (3.0 * h[i]);
        }

        let hint = detect_grid(&xs);
        Ok(Self {
            xs,
            a,
            b: b_coeffs,
            c,
            d: d_coeffs,
            hint,
        })
    }
}

impl Interpolation for CubicSplineInterpolation {
    fn value(&self, x: f64) -> QLResult<f64> {
        let i = locate(&self.xs, x, &self.hint)?;
        let dx = x - self.xs[i];
        Ok(self.a[i] + self.b[i] * dx + self.c[i] * dx * dx + self.d[i] * dx * dx * dx)
    }

    fn derivative(&self, x: f64) -> QLResult<f64> {
        let i = locate(&self.xs, x, &self.hint)?;
        let dx = x - self.xs[i];
        Ok(self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx * dx)
    }

    fn primitive(&self, x: f64) -> QLResult<f64> {
        let i = locate(&self.xs, x, &self.hint)?;
        let mut sum = 0.0;
        for j in 0..i {
            let dx = self.xs[j + 1] - self.xs[j];
            sum += self.a[j] * dx
                + self.b[j] * dx * dx / 2.0
                + self.c[j] * dx * dx * dx / 3.0
                + self.d[j] * dx * dx * dx * dx / 4.0;
        }
        let dx = x - self.xs[i];
        sum += self.a[i] * dx
            + self.b[i] * dx * dx / 2.0
            + self.c[i] * dx * dx * dx / 3.0
            + self.d[i] * dx * dx * dx * dx / 4.0;
        Ok(sum)
    }

    fn x_min(&self) -> f64 {
        self.xs[0]
    }

    fn x_max(&self) -> f64 {
        self.xs[self.xs.len() - 1]
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn sample_xs() -> Vec<f64> {
        vec![1.0, 2.0, 3.0, 4.0, 5.0]
    }

    fn sample_ys() -> Vec<f64> {
        vec![1.0, 4.0, 9.0, 16.0, 25.0] // x^2
    }

    // -- Linear ---------------------------------------------------------

    #[test]
    fn linear_at_nodes() {
        let interp = LinearInterpolation::new(sample_xs(), sample_ys()).unwrap();
        for (&x, &y) in sample_xs().iter().zip(sample_ys().iter()) {
            assert_abs_diff_eq!(interp.value(x).unwrap(), y, epsilon = 1e-14);
        }
    }

    #[test]
    fn linear_midpoint() {
        let interp = LinearInterpolation::new(sample_xs(), sample_ys()).unwrap();
        // Between 2 and 3: y(2.5) = 4 + (9-4)*(0.5) = 6.5
        assert_abs_diff_eq!(interp.value(2.5).unwrap(), 6.5, epsilon = 1e-14);
    }

    #[test]
    fn linear_derivative() {
        let interp = LinearInterpolation::new(sample_xs(), sample_ys()).unwrap();
        // Slope between x=2 and x=3 is (9-4)/(3-2) = 5
        assert_abs_diff_eq!(interp.derivative(2.5).unwrap(), 5.0, epsilon = 1e-14);
    }

    #[test]
    fn linear_primitive() {
        let interp =
            LinearInterpolation::new(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 2.0]).unwrap();
        // integral of y=x from 0 to 2 is 2.0
        assert_abs_diff_eq!(interp.primitive(2.0).unwrap(), 2.0, epsilon = 1e-14);
    }

    #[test]
    fn linear_out_of_range() {
        let interp = LinearInterpolation::new(sample_xs(), sample_ys()).unwrap();
        assert!(interp.value(0.0).is_err());
        assert!(interp.value(6.0).is_err());
    }

    // -- LogLinear ------------------------------------------------------

    #[test]
    fn log_linear_at_nodes() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![1.0, 0.95, 0.90, 0.85];
        let interp = LogLinearInterpolation::new(xs.clone(), ys.clone()).unwrap();
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            assert_abs_diff_eq!(interp.value(x).unwrap(), y, epsilon = 1e-13);
        }
    }

    #[test]
    fn log_linear_midpoint() {
        let xs = vec![0.0, 1.0];
        let ys = vec![1.0, std::f64::consts::E]; // ln(y) goes from 0 to 1
        let interp = LogLinearInterpolation::new(xs, ys).unwrap();
        // At x=0.5, ln(y) = 0.5, so y = exp(0.5)
        assert_abs_diff_eq!(
            interp.value(0.5).unwrap(),
            0.5_f64.exp(),
            epsilon = 1e-13
        );
    }

    #[test]
    fn log_linear_derivative() {
        let xs = vec![0.0, 1.0];
        let ys = vec![1.0, std::f64::consts::E];
        let interp = LogLinearInterpolation::new(xs, ys).unwrap();
        // y = exp(x), dy/dx = exp(x)
        assert_abs_diff_eq!(
            interp.derivative(0.5).unwrap(),
            0.5_f64.exp(),
            epsilon = 1e-13
        );
    }

    #[test]
    fn log_linear_negative_y_rejected() {
        let xs = vec![0.0, 1.0];
        let ys = vec![1.0, -1.0];
        assert!(LogLinearInterpolation::new(xs, ys).is_err());
    }

    // -- Cubic Spline ---------------------------------------------------

    #[test]
    fn cubic_spline_at_nodes() {
        let interp =
            CubicSplineInterpolation::new(sample_xs(), sample_ys()).unwrap();
        for (&x, &y) in sample_xs().iter().zip(sample_ys().iter()) {
            assert_abs_diff_eq!(interp.value(x).unwrap(), y, epsilon = 1e-12);
        }
    }

    #[test]
    fn cubic_spline_smooth() {
        // Use sin(x) on [0, pi] — natural BCs are appropriate since sin''(0) = sin''(pi) = 0
        let n = 11;
        let xs: Vec<f64> = (0..=n).map(|i| std::f64::consts::PI * i as f64 / n as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x.sin()).collect();
        let interp = CubicSplineInterpolation::new(xs, ys).unwrap();
        // Test at non-node points in the interior
        let x = 0.5;
        assert_abs_diff_eq!(interp.value(x).unwrap(), x.sin(), epsilon = 1e-4);
        let x = 1.5;
        assert_abs_diff_eq!(interp.value(x).unwrap(), x.sin(), epsilon = 1e-4);
        let x = 2.5;
        assert_abs_diff_eq!(interp.value(x).unwrap(), x.sin(), epsilon = 1e-4);
    }

    #[test]
    fn cubic_spline_derivative() {
        // sin'(x) = cos(x), test at interior points
        let n = 20;
        let xs: Vec<f64> = (0..=n).map(|i| std::f64::consts::PI * i as f64 / n as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| x.sin()).collect();
        let interp = CubicSplineInterpolation::new(xs, ys).unwrap();
        let x = std::f64::consts::PI / 4.0;
        assert_abs_diff_eq!(interp.derivative(x).unwrap(), x.cos(), epsilon = 1e-3);
        let x = std::f64::consts::PI / 2.0;
        assert_abs_diff_eq!(interp.derivative(x).unwrap(), x.cos(), epsilon = 1e-3);
    }

    #[test]
    fn cubic_spline_continuity() {
        // Verify the spline is continuous across nodes
        let interp =
            CubicSplineInterpolation::new(sample_xs(), sample_ys()).unwrap();
        for &x in &[2.0, 3.0, 4.0] {
            let left = interp.value(x - 1e-10).unwrap();
            let right = interp.value(x + 1e-10).unwrap();
            assert_abs_diff_eq!(left, right, epsilon = 1e-6);
        }
    }

    #[test]
    fn cubic_spline_too_few_points() {
        let xs = vec![0.0, 1.0];
        let ys = vec![0.0, 1.0];
        assert!(CubicSplineInterpolation::new(xs, ys).is_err());
    }

    // -- Validation -----------------------------------------------------

    #[test]
    fn mismatched_lengths_rejected() {
        assert!(LinearInterpolation::new(vec![1.0, 2.0], vec![1.0]).is_err());
    }

    #[test]
    fn non_increasing_rejected() {
        assert!(LinearInterpolation::new(vec![1.0, 1.0], vec![1.0, 2.0]).is_err());
    }
}
