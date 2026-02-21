//! Swaption volatility term structures — matrix, cube, and SABR smile.
//!
//! These structures map `(option_expiry, swap_tenor, strike)` to a swaption
//! implied volatility, providing the rate-vol dimension analogous to
//! equity's `BlackVolTermStructure`.
//!
//! ## Structures
//!
//! - [`SwaptionVolMatrix`] — 2D ATM vol matrix (expiry × tenor), bilinear interp
//! - [`SwaptionVolCube`] — full 3D cube: ATM matrix + smile at each grid point
//! - [`SabrSwaptionVolCube`] — SABR-calibrated smile at each (expiry, tenor) point
//! - [`SwaptionConstantVol`] — trivial flat swaption vol

use ql_math::interpolation::{
    Interpolation, LinearInterpolation,
};
use crate::sabr::sabr_volatility;

// ===========================================================================
//  SwaptionVolatilityStructure trait
// ===========================================================================

/// Trait for swaption volatility term structures.
///
/// Maps (option_expiry, swap_tenor, strike) → implied vol.
pub trait SwaptionVolatilityStructure {
    /// Look up the swaption implied volatility.
    ///
    /// - `option_expiry`: time to option expiry in years
    /// - `swap_tenor`: underlying swap tenor in years
    /// - `strike`: strike rate (absolute, e.g. 0.03 = 3%)
    fn volatility(&self, option_expiry: f64, swap_tenor: f64, strike: f64) -> f64;

    /// ATM volatility (strike = forward rate).
    fn atm_volatility(&self, option_expiry: f64, swap_tenor: f64) -> f64 {
        // Default: look up at strike=0 which should return ATM vol
        self.volatility(option_expiry, swap_tenor, 0.0)
    }
}

// ===========================================================================
//  SwaptionConstantVol
// ===========================================================================

/// Constant swaption volatility (flat across all dimensions).
#[derive(Debug, Clone)]
pub struct SwaptionConstantVol {
    /// The constant vol.
    pub vol: f64,
}

impl SwaptionConstantVol {
    pub fn new(vol: f64) -> Self {
        Self { vol }
    }
}

impl SwaptionVolatilityStructure for SwaptionConstantVol {
    fn volatility(&self, _option_expiry: f64, _swap_tenor: f64, _strike: f64) -> f64 {
        self.vol
    }
}

// ===========================================================================
//  SwaptionVolMatrix (2D ATM)
// ===========================================================================

/// ATM swaption volatility matrix.
///
/// Stores ATM vols on a grid of `(option_expiry, swap_tenor)` and interpolates
/// bilinearly.
///
/// Example:
/// ```text
///              1Y swap  2Y swap  5Y swap  10Y swap
/// 1M expiry     0.45     0.42     0.38     0.35
/// 3M expiry     0.44     0.41     0.37     0.34
/// 1Y expiry     0.40     0.38     0.35     0.32
/// 5Y expiry     0.35     0.33     0.30     0.28
/// ```
#[derive(Debug, Clone)]
pub struct SwaptionVolMatrix {
    /// Option expiry axis (in years), sorted ascending.
    pub expiries: Vec<f64>,
    /// Swap tenor axis (in years), sorted ascending.
    pub tenors: Vec<f64>,
    /// Volatilities: `vols[i][j]` = vol for expiry `i`, tenor `j`.
    pub vols: Vec<Vec<f64>>,
}

impl SwaptionVolMatrix {
    /// Create a new swaption vol matrix.
    ///
    /// `vols`: 2D array indexed `[expiry_index][tenor_index]`.
    pub fn new(
        expiries: Vec<f64>,
        tenors: Vec<f64>,
        vols: Vec<Vec<f64>>,
    ) -> Self {
        debug_assert_eq!(vols.len(), expiries.len());
        for row in &vols {
            debug_assert_eq!(row.len(), tenors.len());
        }
        Self {
            expiries,
            tenors,
            vols,
        }
    }

    /// Bilinear interpolation on the 2D grid.
    fn bilinear(&self, expiry: f64, tenor: f64) -> f64 {
        let ne = self.expiries.len();
        let nt = self.tenors.len();

        // Find bracketing indices for expiry
        let (ei, ef) = bracket(&self.expiries, expiry);
        // Find bracketing indices for tenor
        let (ti, tf) = bracket(&self.tenors, tenor);

        let ei2 = (ei + 1).min(ne - 1);
        let ti2 = (ti + 1).min(nt - 1);

        let v00 = self.vols[ei][ti];
        let v01 = self.vols[ei][ti2];
        let v10 = self.vols[ei2][ti];
        let v11 = self.vols[ei2][ti2];

        // Bilinear
        let v0 = v00 + ef * (v10 - v00);
        let v1 = v01 + ef * (v11 - v01);
        v0 + tf * (v1 - v0)
    }
}

impl SwaptionVolatilityStructure for SwaptionVolMatrix {
    fn volatility(&self, option_expiry: f64, swap_tenor: f64, _strike: f64) -> f64 {
        self.bilinear(option_expiry, swap_tenor)
    }

    fn atm_volatility(&self, option_expiry: f64, swap_tenor: f64) -> f64 {
        self.bilinear(option_expiry, swap_tenor)
    }
}

// ===========================================================================
//  SwaptionVolCube (3D with smile sections)
// ===========================================================================

/// Full swaption vol cube: ATM matrix + smile spread at each grid point.
///
/// The smile is stored as a vector of `(strike_offset, vol_spread)` at each
/// `(expiry, tenor)` grid point. Non-grid queries interpolate between the
/// four nearest ATM vols and their smile sections.
#[derive(Debug, Clone)]
pub struct SwaptionVolCube {
    /// ATM volatility matrix.
    pub atm: SwaptionVolMatrix,
    /// Smile spreads at each (expiry, tenor) grid point.
    /// Indexed: `smiles[expiry_idx][tenor_idx]` = vec of `(strike_offset, vol_spread)`.
    pub smiles: Vec<Vec<Vec<(f64, f64)>>>,
}

impl SwaptionVolCube {
    /// Create a new swaption vol cube.
    pub fn new(
        atm: SwaptionVolMatrix,
        smiles: Vec<Vec<Vec<(f64, f64)>>>,
    ) -> Self {
        Self { atm, smiles }
    }

    /// Interpolate the smile spread at a given grid point.
    fn smile_spread(strikes_and_spreads: &[(f64, f64)], strike_offset: f64) -> f64 {
        if strikes_and_spreads.is_empty() {
            return 0.0;
        }
        if strikes_and_spreads.len() == 1 {
            return strikes_and_spreads[0].1;
        }
        // Linear interpolation on strike offsets
        let xs: Vec<f64> = strikes_and_spreads.iter().map(|&(k, _)| k).collect();
        let ys: Vec<f64> = strikes_and_spreads.iter().map(|&(_, v)| v).collect();
        let interp = match LinearInterpolation::new(xs, ys) {
            Ok(i) => i,
            Err(_) => return 0.0,
        };
        let clamped = strike_offset.clamp(interp.x_min(), interp.x_max());
        interp.value(clamped).unwrap_or(0.0)
    }
}

impl SwaptionVolatilityStructure for SwaptionVolCube {
    fn volatility(&self, option_expiry: f64, swap_tenor: f64, strike: f64) -> f64 {
        let atm_vol = self.atm.bilinear(option_expiry, swap_tenor);
        // For simplicity, use nearest grid point's smile
        let (ei, _) = bracket(&self.atm.expiries, option_expiry);
        let (ti, _) = bracket(&self.atm.tenors, swap_tenor);
        let smile = &self.smiles[ei][ti];
        let spread = Self::smile_spread(smile, strike);
        (atm_vol + spread).max(0.001)
    }

    fn atm_volatility(&self, option_expiry: f64, swap_tenor: f64) -> f64 {
        self.atm.bilinear(option_expiry, swap_tenor)
    }
}

// ===========================================================================
//  SabrSwaptionVolCube
// ===========================================================================

/// SABR parameters for one (expiry, tenor) point.
#[derive(Debug, Clone, Copy)]
pub struct SabrParams {
    pub alpha: f64,
    pub beta: f64,
    pub rho: f64,
    pub nu: f64,
    pub forward: f64,
}

/// SABR-based swaption vol cube.
///
/// Stores calibrated SABR parameters at each (expiry, tenor) grid point
/// and evaluates the SABR formula for arbitrary strikes.
#[derive(Debug, Clone)]
pub struct SabrSwaptionVolCube {
    /// Option expiry axis (years).
    pub expiries: Vec<f64>,
    /// Swap tenor axis (years).
    pub tenors: Vec<f64>,
    /// SABR params: `params[expiry_idx][tenor_idx]`.
    pub params: Vec<Vec<SabrParams>>,
}

impl SabrSwaptionVolCube {
    /// Create a new SABR swaption vol cube.
    pub fn new(
        expiries: Vec<f64>,
        tenors: Vec<f64>,
        params: Vec<Vec<SabrParams>>,
    ) -> Self {
        Self {
            expiries,
            tenors,
            params,
        }
    }
}

impl SwaptionVolatilityStructure for SabrSwaptionVolCube {
    fn volatility(&self, option_expiry: f64, swap_tenor: f64, strike: f64) -> f64 {
        // Find nearest grid point
        let (ei, _) = bracket(&self.expiries, option_expiry);
        let (ti, _) = bracket(&self.tenors, swap_tenor);
        let p = &self.params[ei][ti];
        sabr_volatility(strike, p.forward, option_expiry, p.alpha, p.beta, p.rho, p.nu)
    }

    fn atm_volatility(&self, option_expiry: f64, swap_tenor: f64) -> f64 {
        let (ei, _) = bracket(&self.expiries, option_expiry);
        let (ti, _) = bracket(&self.tenors, swap_tenor);
        let p = &self.params[ei][ti];
        sabr_volatility(p.forward, p.forward, option_expiry, p.alpha, p.beta, p.rho, p.nu)
    }
}

// ===========================================================================
//  Cap/Floor Volatility Structures
// ===========================================================================

/// Trait for cap/floor term volatility structures.
///
/// Maps `(expiry, strike)` → flat cap/floor vol.
pub trait CapFloorTermVolStructure {
    /// Flat cap/floor vol for a given expiry and strike.
    fn volatility(&self, expiry: f64, strike: f64) -> f64;
}

/// Constant cap/floor term vol.
#[derive(Debug, Clone)]
pub struct ConstantCapFloorTermVol {
    pub vol: f64,
}

impl ConstantCapFloorTermVol {
    pub fn new(vol: f64) -> Self {
        Self { vol }
    }
}

impl CapFloorTermVolStructure for ConstantCapFloorTermVol {
    fn volatility(&self, _expiry: f64, _strike: f64) -> f64 {
        self.vol
    }
}

/// Cap/floor vol surface: `(expiry × strike)` grid with bilinear interpolation.
#[derive(Debug, Clone)]
pub struct CapFloorTermVolSurface {
    /// Expiry tenors (years).
    pub expiries: Vec<f64>,
    /// Strike rates.
    pub strikes: Vec<f64>,
    /// Vols: `vols[expiry_idx][strike_idx]`.
    pub vols: Vec<Vec<f64>>,
}

impl CapFloorTermVolSurface {
    pub fn new(expiries: Vec<f64>, strikes: Vec<f64>, vols: Vec<Vec<f64>>) -> Self {
        Self {
            expiries,
            strikes,
            vols,
        }
    }
}

impl CapFloorTermVolStructure for CapFloorTermVolSurface {
    fn volatility(&self, expiry: f64, strike: f64) -> f64 {
        let ne = self.expiries.len();
        let ns = self.strikes.len();
        let (ei, ef) = bracket(&self.expiries, expiry);
        let (si, sf) = bracket(&self.strikes, strike);
        let ei2 = (ei + 1).min(ne - 1);
        let si2 = (si + 1).min(ns - 1);
        let v00 = self.vols[ei][si];
        let v01 = self.vols[ei][si2];
        let v10 = self.vols[ei2][si];
        let v11 = self.vols[ei2][si2];
        let v0 = v00 + ef * (v10 - v00);
        let v1 = v01 + ef * (v11 - v01);
        v0 + sf * (v1 - v0)
    }
}

// ===========================================================================
//  Helper: bracket search
// ===========================================================================

/// Find the bracketing index and interpolation fraction for `x` in sorted `xs`.
/// Returns `(lower_index, fraction)` where `fraction` ∈ [0, 1].
fn bracket(xs: &[f64], x: f64) -> (usize, f64) {
    let n = xs.len();
    if n == 0 {
        return (0, 0.0);
    }
    if x <= xs[0] {
        return (0, 0.0);
    }
    if x >= xs[n - 1] {
        return (n - 1, 0.0);
    }
    // Binary search
    let mut lo = 0;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xs[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let span = xs[hi] - xs[lo];
    let frac = if span.abs() < 1e-15 {
        0.0
    } else {
        (x - xs[lo]) / span
    };
    (lo, frac)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn constant_swaption_vol() {
        let cv = SwaptionConstantVol::new(0.25);
        assert_abs_diff_eq!(cv.volatility(1.0, 5.0, 0.03), 0.25);
        assert_abs_diff_eq!(cv.atm_volatility(2.0, 10.0), 0.25);
    }

    #[test]
    fn swaption_vol_matrix_exact_grid() {
        let matrix = SwaptionVolMatrix::new(
            vec![1.0, 5.0],
            vec![2.0, 10.0],
            vec![
                vec![0.40, 0.30],
                vec![0.35, 0.25],
            ],
        );
        // Exact grid points
        assert_abs_diff_eq!(matrix.volatility(1.0, 2.0, 0.0), 0.40);
        assert_abs_diff_eq!(matrix.volatility(5.0, 10.0, 0.0), 0.25);
    }

    #[test]
    fn swaption_vol_matrix_bilinear() {
        let matrix = SwaptionVolMatrix::new(
            vec![1.0, 5.0],
            vec![2.0, 10.0],
            vec![
                vec![0.40, 0.30],
                vec![0.30, 0.20],
            ],
        );
        // Midpoint
        let vol = matrix.volatility(3.0, 6.0, 0.0);
        // Expected: bilinear of 0.40, 0.30, 0.30, 0.20 at (0.5, 0.5) = 0.30
        assert_abs_diff_eq!(vol, 0.30, epsilon = 0.001);
    }

    #[test]
    fn sabr_swaption_vol_cube() {
        let cube = SabrSwaptionVolCube::new(
            vec![1.0],
            vec![5.0],
            vec![vec![SabrParams {
                alpha: 0.05,
                beta: 0.5,
                rho: -0.3,
                nu: 0.4,
                forward: 0.03,
            }]],
        );
        let atm = cube.volatility(1.0, 5.0, 0.03);
        let otm = cube.volatility(1.0, 5.0, 0.05);
        assert!(atm > 0.0);
        assert!(otm > 0.0);
        // Typically OTM vol differs from ATM due to smile
        assert!((atm - otm).abs() > 0.001);
    }

    #[test]
    fn cap_floor_vol_surface() {
        let surface = CapFloorTermVolSurface::new(
            vec![1.0, 5.0],
            vec![0.02, 0.05],
            vec![
                vec![0.30, 0.25],
                vec![0.28, 0.22],
            ],
        );
        assert_abs_diff_eq!(surface.volatility(1.0, 0.02), 0.30);
        assert_abs_diff_eq!(surface.volatility(5.0, 0.05), 0.22);
        // Midpoint
        let mid = surface.volatility(3.0, 0.035);
        assert!(mid > 0.0 && mid < 0.35);
    }
}
