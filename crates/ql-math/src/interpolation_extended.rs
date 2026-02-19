//! Additional interpolation methods and 2D interpolation.
//!
//! - `BackwardFlatInterpolation` — step function with left-continuous values.
//! - `ForwardFlatInterpolation` — step function with right-continuous values.
//! - `BilinearInterpolation` — 2D bilinear interpolation on a rectangular grid.
//! - `BicubicSplineInterpolation` — 2D bicubic spline on a rectangular grid.

use ql_core::errors::{QLError, QLResult};

// ===========================================================================
// Backward-Flat Interpolation
// ===========================================================================

/// Backward-flat step interpolation: value at x equals the value at the
/// greatest knot ≤ x.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BackwardFlatInterpolation {
    xs: Vec<f64>,
    ys: Vec<f64>,
}

impl BackwardFlatInterpolation {
    pub fn new(xs: Vec<f64>, ys: Vec<f64>) -> QLResult<Self> {
        if xs.len() != ys.len() || xs.is_empty() {
            return Err(QLError::InvalidArgument(
                "xs and ys must be non-empty and same length".into(),
            ));
        }
        Ok(Self { xs, ys })
    }

    pub fn value(&self, x: f64) -> f64 {
        if x <= self.xs[0] {
            return self.ys[0];
        }
        if x >= *self.xs.last().unwrap() {
            return *self.ys.last().unwrap();
        }
        // Find largest index where xs[i] <= x
        match self.xs.partition_point(|&xi| xi <= x) {
            0 => self.ys[0],
            i => self.ys[i - 1],
        }
    }
}

// ===========================================================================
// Forward-Flat Interpolation
// ===========================================================================

/// Forward-flat step interpolation: value at x equals the value at the
/// smallest knot ≥ x.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ForwardFlatInterpolation {
    xs: Vec<f64>,
    ys: Vec<f64>,
}

impl ForwardFlatInterpolation {
    pub fn new(xs: Vec<f64>, ys: Vec<f64>) -> QLResult<Self> {
        if xs.len() != ys.len() || xs.is_empty() {
            return Err(QLError::InvalidArgument(
                "xs and ys must be non-empty and same length".into(),
            ));
        }
        Ok(Self { xs, ys })
    }

    pub fn value(&self, x: f64) -> f64 {
        if x <= self.xs[0] {
            return self.ys[0];
        }
        if x >= *self.xs.last().unwrap() {
            return *self.ys.last().unwrap();
        }
        // Find smallest index where xs[i] >= x
        match self.xs.partition_point(|&xi| xi < x) {
            i if i >= self.xs.len() => *self.ys.last().unwrap(),
            i => self.ys[i],
        }
    }
}

// ===========================================================================
// 2D Bilinear Interpolation
// ===========================================================================

/// Bilinear interpolation on a rectangular grid.
///
/// Grid values z\[i * ny + j\] correspond to (xs\[i\], ys\[j\]).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BilinearInterpolation {
    xs: Vec<f64>,
    ys: Vec<f64>,
    zs: Vec<f64>,
}

impl BilinearInterpolation {
    /// Create from grid axes `xs` (nx), `ys` (ny), and values `zs` (nx×ny, row-major).
    pub fn new(xs: Vec<f64>, ys: Vec<f64>, zs: Vec<f64>) -> QLResult<Self> {
        if xs.len() * ys.len() != zs.len() {
            return Err(QLError::InvalidArgument(format!(
                "zs length {} != xs.len()×ys.len() = {}",
                zs.len(),
                xs.len() * ys.len()
            )));
        }
        if xs.len() < 2 || ys.len() < 2 {
            return Err(QLError::InvalidArgument(
                "need at least 2 points per axis".into(),
            ));
        }
        Ok(Self { xs, ys, zs })
    }

    pub fn value(&self, x: f64, y: f64) -> f64 {
        let nx = self.xs.len();
        let ny = self.ys.len();

        // Clamp to grid boundaries
        let x = x.clamp(self.xs[0], self.xs[nx - 1]);
        let y = y.clamp(self.ys[0], self.ys[ny - 1]);

        let ix = find_interval(&self.xs, x);
        let iy = find_interval(&self.ys, y);

        let x0 = self.xs[ix];
        let x1 = self.xs[ix + 1];
        let y0 = self.ys[iy];
        let y1 = self.ys[iy + 1];

        let tx = if (x1 - x0).abs() > 1e-30 {
            (x - x0) / (x1 - x0)
        } else {
            0.0
        };
        let ty = if (y1 - y0).abs() > 1e-30 {
            (y - y0) / (y1 - y0)
        } else {
            0.0
        };

        let z00 = self.zs[ix * ny + iy];
        let z01 = self.zs[ix * ny + iy + 1];
        let z10 = self.zs[(ix + 1) * ny + iy];
        let z11 = self.zs[(ix + 1) * ny + iy + 1];

        (1.0 - tx) * (1.0 - ty) * z00
            + (1.0 - tx) * ty * z01
            + tx * (1.0 - ty) * z10
            + tx * ty * z11
    }
}

// ===========================================================================
// 2D Bicubic Spline Interpolation
// ===========================================================================

/// Bicubic spline interpolation on a rectangular grid.
///
/// Uses Catmull-Rom spline (a special case of Hermite interpolation).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BicubicSplineInterpolation {
    xs: Vec<f64>,
    ys: Vec<f64>,
    zs: Vec<f64>,
    ny: usize,
}

impl BicubicSplineInterpolation {
    pub fn new(xs: Vec<f64>, ys: Vec<f64>, zs: Vec<f64>) -> QLResult<Self> {
        let ny = ys.len();
        if xs.len() * ny != zs.len() {
            return Err(QLError::InvalidArgument(format!(
                "zs length {} != xs.len()×ys.len() = {}",
                zs.len(),
                xs.len() * ny
            )));
        }
        if xs.len() < 4 || ny < 4 {
            return Err(QLError::InvalidArgument(
                "bicubic spline needs at least 4 points per axis".into(),
            ));
        }
        Ok(Self { xs, ys, zs, ny })
    }

    pub fn value(&self, x: f64, y: f64) -> f64 {
        let nx = self.xs.len();

        let x = x.clamp(self.xs[0], self.xs[nx - 1]);
        let y = y.clamp(self.ys[0], self.ys[self.ny - 1]);

        let ix = find_interval(&self.xs, x).min(nx - 2);
        let iy = find_interval(&self.ys, y).min(self.ny - 2);

        let tx = if (self.xs[ix + 1] - self.xs[ix]).abs() > 1e-30 {
            (x - self.xs[ix]) / (self.xs[ix + 1] - self.xs[ix])
        } else {
            0.0
        };
        let ty = if (self.ys[iy + 1] - self.ys[iy]).abs() > 1e-30 {
            (y - self.ys[iy]) / (self.ys[iy + 1] - self.ys[iy])
        } else {
            0.0
        };

        // Bicubic Catmull-Rom: need 4×4 patch of z values
        let mut patch = [[0.0_f64; 4]; 4];
        for (di, row) in patch.iter_mut().enumerate() {
            let pi = (ix as i64 + di as i64 - 1).clamp(0, nx as i64 - 1) as usize;
            for (dj, val) in row.iter_mut().enumerate() {
                let pj = (iy as i64 + dj as i64 - 1).clamp(0, self.ny as i64 - 1) as usize;
                *val = self.zs[pi * self.ny + pj];
            }
        }

        // Interpolate 4 rows in y, then interpolate result in x
        let mut col_vals = [0.0_f64; 4];
        for (i, row) in patch.iter().enumerate() {
            col_vals[i] = catmull_rom(ty, row[0], row[1], row[2], row[3]);
        }
        catmull_rom(tx, col_vals[0], col_vals[1], col_vals[2], col_vals[3])
    }
}

/// Catmull-Rom cubic interpolation.
fn catmull_rom(t: f64, p0: f64, p1: f64, p2: f64, p3: f64) -> f64 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

/// Find the interval index i such that `xs[i] <= x < xs[i+1]`.
fn find_interval(xs: &[f64], x: f64) -> usize {
    let n = xs.len();
    if n <= 1 {
        return 0;
    }
    match xs.partition_point(|&xi| xi <= x) {
        0 => 0,
        i if i >= n => n - 2,
        i => i - 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ---- Backward-flat ----

    #[test]
    fn backward_flat_step() {
        let interp =
            BackwardFlatInterpolation::new(vec![1.0, 2.0, 3.0], vec![10.0, 20.0, 30.0]).unwrap();
        assert_abs_diff_eq!(interp.value(1.0), 10.0, epsilon = 1e-10);
        assert_abs_diff_eq!(interp.value(1.5), 10.0, epsilon = 1e-10); // takes left value
        assert_abs_diff_eq!(interp.value(2.0), 20.0, epsilon = 1e-10);
        assert_abs_diff_eq!(interp.value(2.9), 20.0, epsilon = 1e-10);
        assert_abs_diff_eq!(interp.value(3.0), 30.0, epsilon = 1e-10);
    }

    #[test]
    fn backward_flat_extrapolation() {
        let interp =
            BackwardFlatInterpolation::new(vec![1.0, 2.0], vec![10.0, 20.0]).unwrap();
        assert_abs_diff_eq!(interp.value(0.0), 10.0, epsilon = 1e-10);
        assert_abs_diff_eq!(interp.value(5.0), 20.0, epsilon = 1e-10);
    }

    // ---- Forward-flat ----

    #[test]
    fn forward_flat_step() {
        let interp =
            ForwardFlatInterpolation::new(vec![1.0, 2.0, 3.0], vec![10.0, 20.0, 30.0]).unwrap();
        assert_abs_diff_eq!(interp.value(1.0), 10.0, epsilon = 1e-10);
        assert_abs_diff_eq!(interp.value(1.5), 20.0, epsilon = 1e-10); // takes right value
        assert_abs_diff_eq!(interp.value(2.0), 20.0, epsilon = 1e-10);
    }

    #[test]
    fn forward_flat_extrapolation() {
        let interp =
            ForwardFlatInterpolation::new(vec![1.0, 2.0], vec![10.0, 20.0]).unwrap();
        assert_abs_diff_eq!(interp.value(0.0), 10.0, epsilon = 1e-10);
        assert_abs_diff_eq!(interp.value(5.0), 20.0, epsilon = 1e-10);
    }

    // ---- Bilinear 2D ----

    #[test]
    fn bilinear_corner_values() {
        // 2×2 grid: xs=[0,1], ys=[0,1], zs=[1,2,3,4]
        let interp = BilinearInterpolation::new(
            vec![0.0, 1.0],
            vec![0.0, 1.0],
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();
        assert_abs_diff_eq!(interp.value(0.0, 0.0), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(interp.value(0.0, 1.0), 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(interp.value(1.0, 0.0), 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(interp.value(1.0, 1.0), 4.0, epsilon = 1e-10);
    }

    #[test]
    fn bilinear_center() {
        let interp = BilinearInterpolation::new(
            vec![0.0, 1.0],
            vec![0.0, 1.0],
            vec![0.0, 0.0, 0.0, 4.0],
        )
        .unwrap();
        // Center: (0.5, 0.5) → 0.25 * 4 = 1.0
        assert_abs_diff_eq!(interp.value(0.5, 0.5), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn bilinear_plane() {
        // z = x + y on a 3×3 grid
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![0.0, 1.0, 2.0];
        let mut zs = Vec::new();
        for &x in &xs {
            for &y in &ys {
                zs.push(x + y);
            }
        }
        let interp = BilinearInterpolation::new(xs, ys, zs).unwrap();
        // Should exactly interpolate a plane
        assert_abs_diff_eq!(interp.value(0.5, 0.5), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(interp.value(1.5, 0.5), 2.0, epsilon = 1e-10);
    }

    // ---- Bicubic Spline 2D ----

    #[test]
    fn bicubic_on_plane() {
        // z = x + y on a 5×5 grid — bicubic should reproduce exactly
        let xs: Vec<f64> = (0..5).map(|i| i as f64).collect();
        let ys: Vec<f64> = (0..5).map(|j| j as f64).collect();
        let mut zs = Vec::new();
        for &x in &xs {
            for &y in &ys {
                zs.push(x + y);
            }
        }
        let interp = BicubicSplineInterpolation::new(xs, ys, zs).unwrap();
        assert_abs_diff_eq!(interp.value(1.5, 2.5), 4.0, epsilon = 0.1);
        assert_abs_diff_eq!(interp.value(2.0, 2.0), 4.0, epsilon = 0.1);
    }

    #[test]
    fn bicubic_smoothness() {
        // Smooth function z = sin(x) * cos(y)
        let n = 6;
        let xs: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
        let ys: Vec<f64> = (0..n).map(|j| j as f64 * 0.5).collect();
        let mut zs = Vec::new();
        for &x in &xs {
            for &y in &ys {
                zs.push(x.sin() * y.cos());
            }
        }
        let interp = BicubicSplineInterpolation::new(xs, ys, zs).unwrap();
        // Check interior point
        let x: f64 = 1.25;
        let y: f64 = 1.25;
        let exact = x.sin() * y.cos();
        let approx_val = interp.value(x, y);
        // Catmull-Rom should be reasonable
        assert!((approx_val - exact).abs() < 0.05, "bicubic error too large: {}", (approx_val - exact).abs());
    }
}
