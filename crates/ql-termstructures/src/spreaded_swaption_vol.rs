//! Spreaded swaption volatility surface.
//!
//! Wraps a base swaption volatility surface and adds a constant or
//! time-varying spread to the volatilities.
//!
//! Corresponds to QuantLib's `SpreadedSwaptionVolatility`.

use serde::{Deserialize, Serialize};

/// A swaption volatility surface that adds a spread to a base surface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpreadedSwaptionVol {
    /// Base volatilities: row-major grid of (exercise × tenor) vols.
    pub base_vols: Vec<Vec<f64>>,
    /// Exercise times (sorted).
    pub exercise_times: Vec<f64>,
    /// Swap tenor times (sorted).
    pub tenor_times: Vec<f64>,
    /// Spread to add (can be negative).
    pub spread: SpreadSpec,
}

/// Spread specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpreadSpec {
    /// Constant spread added to all vols.
    Constant(f64),
    /// Time-varying spread: one spread per exercise time.
    ByExpiry(Vec<f64>),
    /// Full grid spread: one per (exercise, tenor) point.
    Full(Vec<Vec<f64>>),
}

impl SpreadedSwaptionVol {
    /// Create with a constant spread.
    pub fn with_constant_spread(
        exercise_times: Vec<f64>,
        tenor_times: Vec<f64>,
        base_vols: Vec<Vec<f64>>,
        spread: f64,
    ) -> Self {
        Self {
            base_vols,
            exercise_times,
            tenor_times,
            spread: SpreadSpec::Constant(spread),
        }
    }

    /// Create with expiry-dependent spread.
    pub fn with_expiry_spread(
        exercise_times: Vec<f64>,
        tenor_times: Vec<f64>,
        base_vols: Vec<Vec<f64>>,
        spreads: Vec<f64>,
    ) -> Self {
        assert_eq!(exercise_times.len(), spreads.len());
        Self {
            base_vols,
            exercise_times,
            tenor_times,
            spread: SpreadSpec::ByExpiry(spreads),
        }
    }

    /// Get interpolated volatility at (exercise, tenor).
    pub fn vol(&self, exercise: f64, tenor: f64) -> f64 {
        let base = self.interpolate_base(exercise, tenor);
        let spread = self.get_spread(exercise, tenor);
        (base + spread).max(0.0)
    }

    /// Get the ATM volatility at (exercise, tenor).
    pub fn atm_vol(&self, exercise: f64, tenor: f64) -> f64 {
        self.vol(exercise, tenor)
    }

    fn interpolate_base(&self, exercise: f64, tenor: f64) -> f64 {
        bilinear_interp(&self.exercise_times, &self.tenor_times, &self.base_vols, exercise, tenor)
    }

    fn get_spread(&self, exercise: f64, _tenor: f64) -> f64 {
        match &self.spread {
            SpreadSpec::Constant(s) => *s,
            SpreadSpec::ByExpiry(spreads) => {
                linear_interp_1d(&self.exercise_times, spreads, exercise)
            }
            SpreadSpec::Full(grid) => {
                bilinear_interp(&self.exercise_times, &self.tenor_times, grid, exercise, _tenor)
            }
        }
    }
}

fn linear_interp_1d(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    if xs.is_empty() { return 0.0; }
    if x <= xs[0] { return ys[0]; }
    if x >= *xs.last().unwrap() { return *ys.last().unwrap(); }
    let idx = xs.partition_point(|&v| v < x).saturating_sub(1).min(xs.len() - 2);
    let w = (x - xs[idx]) / (xs[idx + 1] - xs[idx]);
    ys[idx] + w * (ys[idx + 1] - ys[idx])
}

fn bilinear_interp(xs: &[f64], ys: &[f64], grid: &[Vec<f64>], x: f64, y: f64) -> f64 {
    if xs.is_empty() || ys.is_empty() { return 0.0; }

    let ix = xs.partition_point(|&v| v < x).saturating_sub(1).min(xs.len().saturating_sub(2));
    let iy = ys.partition_point(|&v| v < y).saturating_sub(1).min(ys.len().saturating_sub(2));

    let ix1 = (ix + 1).min(xs.len() - 1);
    let iy1 = (iy + 1).min(ys.len() - 1);

    let wx = if ix1 > ix { (x - xs[ix]) / (xs[ix1] - xs[ix]) } else { 0.0 };
    let wy = if iy1 > iy { (y - ys[iy]) / (ys[iy1] - ys[iy]) } else { 0.0 };

    let wx = wx.clamp(0.0, 1.0);
    let wy = wy.clamp(0.0, 1.0);

    let v00 = grid[ix][iy];
    let v10 = grid[ix1][iy];
    let v01 = grid[ix][iy1];
    let v11 = grid[ix1][iy1];

    (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v10
        + (1.0 - wx) * wy * v01
        + wx * wy * v11
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_base() -> SpreadedSwaptionVol {
        let exercises = vec![1.0, 2.0, 5.0, 10.0];
        let tenors = vec![1.0, 2.0, 5.0, 10.0];
        let base_vols = vec![
            vec![0.20, 0.21, 0.22, 0.23],
            vec![0.19, 0.20, 0.21, 0.22],
            vec![0.18, 0.19, 0.20, 0.21],
            vec![0.17, 0.18, 0.19, 0.20],
        ];
        SpreadedSwaptionVol::with_constant_spread(exercises, tenors, base_vols, 0.01)
    }

    #[test]
    fn test_constant_spread() {
        let surface = make_base();
        let vol = surface.vol(1.0, 1.0);
        assert!((vol - 0.21).abs() < 1e-10, "vol={}", vol); // 0.20 + 0.01
    }

    #[test]
    fn test_zero_spread_equals_base() {
        let exercises = vec![1.0, 5.0];
        let tenors = vec![2.0, 10.0];
        let base_vols = vec![vec![0.20, 0.22], vec![0.18, 0.19]];
        let surface = SpreadedSwaptionVol::with_constant_spread(
            exercises, tenors, base_vols, 0.0,
        );
        let vol = surface.vol(1.0, 2.0);
        assert!((vol - 0.20).abs() < 1e-10, "vol={}", vol);
    }

    #[test]
    fn test_expiry_spread() {
        let exercises = vec![1.0, 5.0, 10.0];
        let tenors = vec![2.0, 5.0];
        let base_vols = vec![
            vec![0.20, 0.22],
            vec![0.19, 0.21],
            vec![0.18, 0.20],
        ];
        let spreads = vec![0.01, 0.02, 0.03];
        let surface = SpreadedSwaptionVol::with_expiry_spread(
            exercises, tenors, base_vols, spreads,
        );
        // At exercise=1, spread=0.01
        assert!((surface.vol(1.0, 2.0) - 0.21).abs() < 1e-10);
        // At exercise=10, spread=0.03
        assert!((surface.vol(10.0, 5.0) - 0.23).abs() < 1e-10);
    }

    #[test]
    fn test_interpolated_vol() {
        let surface = make_base();
        let vol = surface.vol(3.0, 3.0);
        // Should be between pillar values + spread
        assert!(vol > 0.15 && vol < 0.30, "vol={}", vol);
    }

    #[test]
    fn test_negative_spread_floor() {
        let exercises = vec![1.0, 5.0];
        let tenors = vec![2.0];
        let base_vols = vec![vec![0.05], vec![0.03]];
        let surface = SpreadedSwaptionVol::with_constant_spread(
            exercises, tenors, base_vols, -0.10,
        );
        // Vol should be floored at 0
        let vol = surface.vol(1.0, 2.0);
        assert!(vol >= 0.0, "vol={}", vol);
    }
}
