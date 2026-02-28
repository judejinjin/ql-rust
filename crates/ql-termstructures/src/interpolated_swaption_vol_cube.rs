//! Interpolated swaption volatility cube with per-slice SABR calibration.
//!
//! Builds a full 3D volatility surface: exercise × tenor × strike.
//! For each (exercise, tenor) pair, a SABR smile is calibrated to market
//! quotes. Between slices, bilinear interpolation in (exercise, tenor) is used.
//!
//! This corresponds to QuantLib's `SwaptionVolCube1` (SABR) approach.

use serde::{Deserialize, Serialize};

/// SABR parameters for a single (exercise, tenor) slice.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SabrSlice {
    /// Expiry in years.
    pub expiry: f64,
    /// Underlying tenor in years.
    pub tenor: f64,
    /// SABR α (initial vol).
    pub alpha: f64,
    /// SABR β (backbone exponent, usually fixed).
    pub beta: f64,
    /// SABR ρ (correlation).
    pub rho: f64,
    /// SABR ν (vol-of-vol).
    pub nu: f64,
    /// ATM forward rate used for calibration.
    pub atm_forward: f64,
}

/// A fully interpolated swaption vol cube: exercise × tenor × strike.
///
/// Each (exercise, tenor) slice has SABR parameters. Between slices,
/// the implied vol is computed by bilinear interpolation of the SABR
/// vols at the target strike.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpolatedSwaptionVolCube {
    /// Unique expiry times (sorted).
    expiries: Vec<f64>,
    /// Unique tenor times (sorted).
    tenors: Vec<f64>,
    /// SABR slices, stored row-major: slices[i * n_tenors + j] for expiry i, tenor j.
    slices: Vec<SabrSlice>,
}

impl InterpolatedSwaptionVolCube {
    /// Build an interpolated cube from a flat list of calibrated SABR slices.
    ///
    /// The slices must cover the full grid `expiries × tenors`.
    pub fn new(slices: Vec<SabrSlice>) -> Self {
        let mut expiry_set: Vec<f64> = slices.iter().map(|s| s.expiry).collect();
        let mut tenor_set: Vec<f64> = slices.iter().map(|s| s.tenor).collect();
        expiry_set.sort_by(|a, b| a.partial_cmp(b).unwrap());
        expiry_set.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
        tenor_set.sort_by(|a, b| a.partial_cmp(b).unwrap());
        tenor_set.dedup_by(|a, b| (*a - *b).abs() < 1e-10);

        // Re-sort slices into row-major order
        let nt = tenor_set.len();
        let mut ordered = vec![slices[0]; expiry_set.len() * nt];
        for s in &slices {
            let ei = expiry_set.iter().position(|&e| (e - s.expiry).abs() < 1e-10).unwrap();
            let ti = tenor_set.iter().position(|&t| (t - s.tenor).abs() < 1e-10).unwrap();
            ordered[ei * nt + ti] = *s;
        }

        Self {
            expiries: expiry_set,
            tenors: tenor_set,
            slices: ordered,
        }
    }

    /// Look up (or bilinearly interpolate) the Black vol at a given
    /// expiry, tenor, and strike.
    pub fn vol(&self, expiry: f64, tenor: f64, strike: f64) -> f64 {
        let ne = self.expiries.len();
        let nt = self.tenors.len();

        // Find bracketing indices for expiry
        let (ei0, ei1, ew) = bracket(&self.expiries, expiry);
        // Find bracketing indices for tenor
        let (ti0, ti1, tw) = bracket(&self.tenors, tenor);

        // Evaluate SABR vol at each corner
        let v00 = sabr_vol_at(&self.slices[ei0 * nt + ti0], strike);
        let v01 = sabr_vol_at(&self.slices[ei0 * nt + ti1], strike);
        let v10 = sabr_vol_at(&self.slices[ei1 * nt + ti0], strike);
        let v11 = sabr_vol_at(&self.slices[ei1 * nt + ti1], strike);

        // Bilinear interpolation
        let v0 = v00 * (1.0 - tw) + v01 * tw;
        let v1 = v10 * (1.0 - tw) + v11 * tw;
        v0 * (1.0 - ew) + v1 * ew
    }

    /// ATM vol at given expiry and tenor.
    pub fn atm_vol(&self, expiry: f64, tenor: f64) -> f64 {
        let ne = self.expiries.len();
        let nt = self.tenors.len();
        let (ei0, ei1, ew) = bracket(&self.expiries, expiry);
        let (ti0, ti1, tw) = bracket(&self.tenors, tenor);

        let a00 = self.slices[ei0 * nt + ti0].alpha;
        let a01 = self.slices[ei0 * nt + ti1].alpha;
        let a10 = self.slices[ei1 * nt + ti0].alpha;
        let a11 = self.slices[ei1 * nt + ti1].alpha;

        let a0 = a00 * (1.0 - tw) + a01 * tw;
        let a1 = a10 * (1.0 - tw) + a11 * tw;
        a0 * (1.0 - ew) + a1 * ew
    }

    /// Available expiry times.
    pub fn expiries(&self) -> &[f64] { &self.expiries }
    /// Available tenor times.
    pub fn tenors(&self) -> &[f64] { &self.tenors }
    /// Number of slices.
    pub fn num_slices(&self) -> usize { self.slices.len() }
}

/// Calibrate SABR parameters from market vol quotes for a single slice.
///
/// Given an ATM forward, beta (fixed), and a set of (strike, vol) pairs,
/// calibrate α, ρ, ν via least-squares.
///
/// This uses a simplified approach: starting from alpha ≈ ATM vol,
/// then iterating a few Newton-like steps.
pub fn calibrate_sabr_slice(
    expiry: f64,
    tenor: f64,
    atm_forward: f64,
    beta: f64,
    market_strikes: &[f64],
    market_vols: &[f64],
) -> SabrSlice {
    assert_eq!(market_strikes.len(), market_vols.len());

    // Find ATM vol (closest strike to forward)
    let atm_idx = market_strikes.iter().enumerate()
        .min_by(|(_, a), (_, b)| {
            ((**a - atm_forward).abs()).partial_cmp(&((**b - atm_forward).abs())).unwrap()
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    let atm_vol = market_vols[atm_idx];

    // Initial guess
    let f_beta = atm_forward.powf(1.0 - beta);
    let mut alpha = atm_vol * f_beta;
    let mut rho = -0.3;
    let mut nu = 0.4;

    // Simple iterative calibration (gradient-free Nelder-Mead like)
    let mut best_err = f64::MAX;
    let mut best = (alpha, rho, nu);

    for _ in 0..200 {
        let err = market_strikes.iter().zip(market_vols.iter())
            .map(|(&k, &mv)| {
                let sv = sabr_implied_vol(atm_forward, k, expiry, alpha, beta, rho, nu);
                (sv - mv) * (sv - mv)
            })
            .sum::<f64>();

        if err < best_err {
            best_err = err;
            best = (alpha, rho, nu);
        }

        // Perturb and try to improve
        let da = 0.001 * alpha;
        let dr = 0.01;
        let dn = 0.01;

        // Gradient approximation
        let err_ap = calc_err(atm_forward, expiry, alpha + da, beta, rho, nu, market_strikes, market_vols);
        let err_am = calc_err(atm_forward, expiry, alpha - da, beta, rho, nu, market_strikes, market_vols);
        let err_rp = calc_err(atm_forward, expiry, alpha, beta, (rho + dr).min(0.999), nu, market_strikes, market_vols);
        let err_rm = calc_err(atm_forward, expiry, alpha, beta, (rho - dr).max(-0.999), nu, market_strikes, market_vols);
        let err_np = calc_err(atm_forward, expiry, alpha, beta, rho, (nu + dn).max(0.01), market_strikes, market_vols);
        let err_nm = calc_err(atm_forward, expiry, alpha, beta, rho, (nu - dn).max(0.01), market_strikes, market_vols);

        let ga = (err_ap - err_am) / (2.0 * da);
        let gr = (err_rp - err_rm) / (2.0 * dr);
        let gn = (err_np - err_nm) / (2.0 * dn);

        let step = 0.0001 / (ga * ga + gr * gr + gn * gn + 1e-20).sqrt();
        alpha = (alpha - step * ga).max(0.001);
        rho = (rho - step * gr).clamp(-0.999, 0.999);
        nu = (nu - step * gn).max(0.01);
    }

    SabrSlice {
        expiry,
        tenor,
        alpha: best.0,
        beta,
        rho: best.1,
        nu: best.2,
        atm_forward,
    }
}

fn calc_err(f: f64, t: f64, a: f64, b: f64, r: f64, n: f64, ks: &[f64], mvs: &[f64]) -> f64 {
    ks.iter().zip(mvs.iter())
        .map(|(&k, &mv)| {
            let sv = sabr_implied_vol(f, k, t, a, b, r, n);
            (sv - mv) * (sv - mv)
        })
        .sum()
}

// ---------------------------------------------------------------------------
// SABR implied vol (Hagan 2002)
// ---------------------------------------------------------------------------

fn sabr_implied_vol(f: f64, k: f64, t: f64, alpha: f64, beta: f64, rho: f64, nu: f64) -> f64 {
    if t <= 0.0 || alpha <= 0.0 || f <= 0.0 || k <= 0.0 {
        return alpha;
    }
    let eps = 1e-7;
    if (f - k).abs() < eps * f {
        // ATM approximation
        let f_mid = (f * k).sqrt();
        let f_beta = f_mid.powf(1.0 - beta);
        let v = alpha / f_beta;
        let correction = 1.0
            + ((1.0 - beta).powi(2) / 24.0 * alpha * alpha / f_mid.powf(2.0 * (1.0 - beta))
               + 0.25 * rho * beta * nu * alpha / f_mid.powf(1.0 - beta)
               + (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu)
            * t;
        return v * correction;
    }

    let f_mid = (f * k).sqrt();
    let f_beta = f_mid.powf(1.0 - beta);
    let log_fk = (f / k).ln();

    let z = nu / alpha * f_beta * log_fk;
    let x_z = ((1.0 - 2.0 * rho * z + z * z).sqrt() + z - rho).ln() - (1.0 - rho).ln();

    if x_z.abs() < eps {
        return alpha / f_beta;
    }

    let prefix = alpha / (f_beta * (1.0 + (1.0 - beta).powi(2) / 24.0 * log_fk * log_fk
                                    + (1.0 - beta).powi(4) / 1920.0 * log_fk.powi(4)));

    let correction = 1.0
        + ((1.0 - beta).powi(2) / 24.0 * alpha * alpha / f_mid.powf(2.0 * (1.0 - beta))
           + 0.25 * rho * beta * nu * alpha / f_mid.powf(1.0 - beta)
           + (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu)
        * t;

    prefix * z / x_z * correction
}

/// Evaluate the SABR vol for a given slice at a specific strike.
fn sabr_vol_at(slice: &SabrSlice, strike: f64) -> f64 {
    sabr_implied_vol(
        slice.atm_forward,
        strike.max(0.0001),
        slice.expiry,
        slice.alpha,
        slice.beta,
        slice.rho,
        slice.nu,
    )
}

/// Find bracketing indices and interpolation weight.
fn bracket(xs: &[f64], x: f64) -> (usize, usize, f64) {
    if xs.len() <= 1 { return (0, 0, 0.0); }
    if x <= xs[0] { return (0, 0, 0.0); }
    let n = xs.len();
    if x >= xs[n - 1] { return (n - 1, n - 1, 0.0); }
    let idx = xs.partition_point(|&v| v < x);
    let i0 = if idx > 0 { idx - 1 } else { 0 };
    let i1 = idx.min(n - 1);
    let denom = xs[i1] - xs[i0];
    let w = if denom.abs() > 1e-14 { (x - xs[i0]) / denom } else { 0.0 };
    (i0, i1, w)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn sample_cube() -> InterpolatedSwaptionVolCube {
        let mut slices = Vec::new();
        for &exp in &[1.0, 5.0, 10.0] {
            for &ten in &[2.0, 5.0, 10.0] {
                slices.push(SabrSlice {
                    expiry: exp,
                    tenor: ten,
                    alpha: 0.003 + 0.001 * exp.sqrt(),
                    beta: 0.5,
                    rho: -0.3 - 0.01 * ten,
                    nu: 0.4 + 0.01 * exp,
                    atm_forward: 0.03 + 0.001 * ten,
                });
            }
        }
        InterpolatedSwaptionVolCube::new(slices)
    }

    #[test]
    fn test_cube_at_grid_point() {
        let cube = sample_cube();
        let vol = cube.vol(1.0, 2.0, 0.03);
        assert!(vol > 0.0 && vol < 1.0, "vol={}", vol);
    }

    #[test]
    fn test_cube_interpolation() {
        let cube = sample_cube();
        // At midpoint between grid expiries
        let vol = cube.vol(3.0, 5.0, 0.035);
        assert!(vol > 0.0 && vol < 1.0, "interp vol={}", vol);
    }

    #[test]
    fn test_cube_atm_vol() {
        let cube = sample_cube();
        let atm = cube.atm_vol(1.0, 5.0);
        assert!(atm > 0.001 && atm < 0.1, "atm={}", atm);
    }

    #[test]
    fn test_sabr_implied_vol_atm() {
        let vol = sabr_implied_vol(0.03, 0.03, 1.0, 0.004, 0.5, -0.3, 0.4);
        assert!(vol > 0.0 && vol < 1.0, "sabr_atm={}", vol);
    }

    #[test]
    fn test_calibrate_sabr_slice() {
        let f = 0.03;
        let strikes = vec![0.01, 0.02, 0.03, 0.04, 0.05];
        let true_alpha = 0.004;
        let true_beta = 0.5;
        let true_rho = -0.3;
        let true_nu = 0.4;
        let vols: Vec<f64> = strikes.iter()
            .map(|&k| sabr_implied_vol(f, k, 5.0, true_alpha, true_beta, true_rho, true_nu))
            .collect();

        let slice = calibrate_sabr_slice(5.0, 10.0, f, true_beta, &strikes, &vols);
        // Calibrated alpha should be close
        assert_abs_diff_eq!(slice.alpha, true_alpha, epsilon = 0.002);
    }
}
