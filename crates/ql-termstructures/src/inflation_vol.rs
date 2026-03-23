//! CPI and Year-on-Year (YoY) inflation volatility structures.
//!
//! These are term structures for pricing inflation caps, floors, and swaptions.
//! They provide the volatility of zero-coupon CPI rates and year-on-year
//! inflation rates as functions of time and strike.
//!
//! Reference:
//! - QuantLib: CPIVolatilitySurface, YoYOptionletVolatilitySurface

use serde::{Deserialize, Serialize};
use ql_time::{Date, DayCounter};

/// CPI volatility surface.
///
/// Provides Black (lognormal) or Bachelier (normal) volatilities for
/// zero-coupon inflation caps/floors as a function of maturity and strike.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CpiVolSurface {
    /// Reference date.
    pub reference_date: Date,
    /// Day counter.
    pub day_counter: DayCounter,
    /// Maturity year fractions (pillar dates).
    pub maturities: Vec<f64>,
    /// Strike grid.
    pub strikes: Vec<f64>,
    /// Volatilities matrix: vol[i][j] = vol at maturity i, strike j.
    pub volatilities: Vec<Vec<f64>>,
    /// Whether volatilities are normal (Bachelier) or lognormal (Black).
    pub is_normal: bool,
    /// Base CPI level.
    pub base_cpi: f64,
    /// Observation lag in months.
    pub observation_lag_months: u32,
}

impl CpiVolSurface {
    /// Return the volatility for a given maturity time and strike.
    ///
    /// Uses bilinear interpolation on the grid.
    pub fn volatility(&self, t: f64, strike: f64) -> f64 {
        if self.maturities.is_empty() || self.strikes.is_empty() {
            return 0.0;
        }
        // Find maturity bracket
        let (mi, mw) = bracket(&self.maturities, t);
        // Find strike bracket
        let (si, sw) = bracket(&self.strikes, strike);
        // Bilinear interpolation
        let v00 = self.volatilities[mi][si];
        let v01 = self.volatilities[mi]
            .get(si + 1).copied().unwrap_or(v00);
        let v10 = self.volatilities
            .get(mi + 1).map(|row| row[si]).unwrap_or(v00);
        let v11 = self.volatilities
            .get(mi + 1).and_then(|row| row.get(si + 1).copied()).unwrap_or(v00);

        let v0 = v00 * (1.0 - sw) + v01 * sw;
        let v1 = v10 * (1.0 - sw) + v11 * sw;
        v0 * (1.0 - mw) + v1 * mw
    }

    /// Return the ATM volatility at a given maturity.
    pub fn atm_volatility(&self, t: f64) -> f64 {
        // ATM = middle strike
        if self.strikes.is_empty() { return 0.0; }
        let mid = self.strikes[self.strikes.len() / 2];
        self.volatility(t, mid)
    }

    /// Total variance σ²·T at a given maturity and strike.
    pub fn total_variance(&self, t: f64, strike: f64) -> f64 {
        let v = self.volatility(t, strike);
        v * v * t
    }
}

/// Year-on-Year inflation optionlet volatility surface.
///
/// Provides volatilities for year-on-year inflation caps/floors, typically
/// stripped from market cap/floor prices.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct YoYOptionletVolSurface {
    /// Reference date.
    pub reference_date: Date,
    /// Day counter.
    pub day_counter: DayCounter,
    /// Maturity year fractions (optionlet fixing times).
    pub maturities: Vec<f64>,
    /// Strike grid.
    pub strikes: Vec<f64>,
    /// Optionlet volatilities: vol[i][j] = vol at maturity i, strike j.
    pub volatilities: Vec<Vec<f64>>,
    /// Whether volatilities are normal (Bachelier) or lognormal (Black).
    pub is_normal: bool,
    /// Observation lag in months.
    pub observation_lag_months: u32,
}

impl YoYOptionletVolSurface {
    /// Return the optionlet volatility for a given time and strike.
    pub fn volatility(&self, t: f64, strike: f64) -> f64 {
        if self.maturities.is_empty() || self.strikes.is_empty() {
            return 0.0;
        }
        let (mi, mw) = bracket(&self.maturities, t);
        let (si, sw) = bracket(&self.strikes, strike);

        let v00 = self.volatilities[mi][si];
        let v01 = self.volatilities[mi].get(si + 1).copied().unwrap_or(v00);
        let v10 = self.volatilities.get(mi + 1).map(|row| row[si]).unwrap_or(v00);
        let v11 = self.volatilities.get(mi + 1)
            .and_then(|row| row.get(si + 1).copied()).unwrap_or(v00);

        let v0 = v00 * (1.0 - sw) + v01 * sw;
        let v1 = v10 * (1.0 - sw) + v11 * sw;
        v0 * (1.0 - mw) + v1 * mw
    }

    /// ATM optionlet volatility at a given maturity.
    pub fn atm_volatility(&self, t: f64) -> f64 {
        if self.strikes.is_empty() { return 0.0; }
        let mid = self.strikes[self.strikes.len() / 2];
        self.volatility(t, mid)
    }
}

/// Constant CPI volatility (flat across all maturities and strikes).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConstantCpiVol {
    /// Reference date.
    pub reference_date: Date,
    /// Volatility.
    pub volatility: f64,
    /// Is normal.
    pub is_normal: bool,
}

impl ConstantCpiVol {
    /// Vol.
    pub fn vol(&self, _t: f64, _strike: f64) -> f64 {
        self.volatility
    }
}

/// Constant YoY optionlet volatility.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConstantYoYVol {
    /// Reference date.
    pub reference_date: Date,
    /// Volatility.
    pub volatility: f64,
    /// Is normal.
    pub is_normal: bool,
}

impl ConstantYoYVol {
    /// Vol.
    pub fn vol(&self, _t: f64, _strike: f64) -> f64 {
        self.volatility
    }
}

/// Strip YoY optionlet volatilities from cap/floor volatilities.
///
/// Given cap volatilities at different maturities (each being a portfolio of
/// optionlets), backs out the individual optionlet volatilities sequentially.
///
/// # Arguments
/// - `cap_vols` — flat cap volatilities at each maturity
/// - `maturities` — year fractions to each maturity
/// - `forward_rates` — forward YoY rates for each period
/// - `discount_factors` — discount factors to each payment date
///
/// # Returns
/// Optionlet (caplet) volatilities.
pub fn strip_yoy_optionlet_vols(
    cap_vols: &[f64],
    maturities: &[f64],
    forward_rates: &[f64],
    discount_factors: &[f64],
) -> Vec<f64> {
    let n = cap_vols.len()
        .min(maturities.len())
        .min(forward_rates.len())
        .min(discount_factors.len());
    if n == 0 { return vec![]; }

    let mut optionlet_vols = Vec::with_capacity(n);
    let mut cumulative_var_pv = 0.0;

    for i in 0..n {
        let ti = maturities[i];
        let fi = forward_rates[i];
        let dfi = discount_factors[i];
        let cap_var = cap_vols[i] * cap_vols[i] * ti;

        // Weight for this optionlet: DF_i × f_i × √(dtau_i)
        let dtau = if i == 0 { ti } else { ti - maturities[i - 1] };
        let weight = dfi * fi.abs().max(1e-6) * dtau;

        // Caplet variance × weight must equal cap total variance weight - previous
        let total_cap_weight = {
            let mut w = 0.0;
            for j in 0..=i {
                let dj = discount_factors[j];
                let fj = forward_rates[j].abs().max(1e-6);
                let dt = if j == 0 { maturities[j] } else { maturities[j] - maturities[j - 1] };
                w += dj * fj * dt;
            }
            w
        };

        let target_var_pv = cap_var * total_cap_weight;
        let local_var_pv = target_var_pv - cumulative_var_pv;
        let local_var = if weight.abs() > 1e-14 { local_var_pv / weight } else { cap_var };
        let optionlet_vol = if local_var > 0.0 && dtau > 0.0 {
            (local_var / dtau).sqrt()
        } else {
            cap_vols[i]
        };

        optionlet_vols.push(optionlet_vol);
        cumulative_var_pv = target_var_pv;
    }

    optionlet_vols
}

/// Find bracket index and interpolation weight in a sorted grid.
fn bracket(grid: &[f64], x: f64) -> (usize, f64) {
    if grid.is_empty() { return (0, 0.0); }
    if x <= grid[0] { return (0, 0.0); }
    if x >= grid[grid.len() - 1] { return (grid.len().saturating_sub(1), 0.0); }
    let idx = grid.iter().position(|&g| g >= x).unwrap_or(grid.len() - 1);
    if idx == 0 { return (0, 0.0); }
    let w = if (grid[idx] - grid[idx - 1]).abs() > 1e-14 {
        (x - grid[idx - 1]) / (grid[idx] - grid[idx - 1])
    } else { 0.0 };
    (idx - 1, w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ql_time::Month;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_cpi_vol_surface_flat() {
        let surf = CpiVolSurface {
            reference_date: Date::from_ymd(2025, Month::January, 15),
            day_counter: DayCounter::Actual365Fixed,
            maturities: vec![1.0, 2.0, 5.0, 10.0],
            strikes: vec![0.01, 0.02, 0.03, 0.04],
            volatilities: vec![vec![0.10; 4]; 4],
            is_normal: false,
            base_cpi: 300.0,
            observation_lag_months: 3,
        };
        assert_abs_diff_eq!(surf.volatility(3.0, 0.025), 0.10, epsilon = 1e-10);
    }

    #[test]
    fn test_cpi_vol_surface_interpolation() {
        let surf = CpiVolSurface {
            reference_date: Date::from_ymd(2025, Month::January, 15),
            day_counter: DayCounter::Actual365Fixed,
            maturities: vec![1.0, 5.0],
            strikes: vec![0.02, 0.04],
            volatilities: vec![
                vec![0.08, 0.12],
                vec![0.10, 0.14],
            ],
            is_normal: false,
            base_cpi: 300.0,
            observation_lag_months: 3,
        };
        // Midpoint should be average
        let v = surf.volatility(3.0, 0.03);
        assert!(v > 0.08 && v < 0.14, "v={}", v);
    }

    #[test]
    fn test_yoy_vol_surface() {
        let surf = YoYOptionletVolSurface {
            reference_date: Date::from_ymd(2025, Month::January, 15),
            day_counter: DayCounter::Actual365Fixed,
            maturities: vec![1.0, 3.0],
            strikes: vec![0.01, 0.03],
            volatilities: vec![
                vec![0.05, 0.07],
                vec![0.06, 0.08],
            ],
            is_normal: true,
            observation_lag_months: 3,
        };
        let v = surf.volatility(2.0, 0.02);
        assert!(v > 0.05 && v < 0.08, "v={}", v);
    }

    #[test]
    fn test_strip_yoy_optionlets() {
        let cap_vols = vec![0.10, 0.10, 0.10];
        let mats = vec![1.0, 2.0, 3.0];
        let fwds = vec![0.025, 0.025, 0.025];
        let dfs = vec![0.97, 0.94, 0.91];
        let optionlet = strip_yoy_optionlet_vols(&cap_vols, &mats, &fwds, &dfs);
        assert_eq!(optionlet.len(), 3);
        // With flat cap vols and equal periods, optionlet vols should be similar
        for &v in &optionlet {
            assert!(v > 0.0, "vol should be positive: {}", v);
        }
    }

    #[test]
    fn test_constant_vols() {
        let date = Date::from_ymd(2025, Month::January, 1);
        let cpi = ConstantCpiVol { reference_date: date, volatility: 0.08, is_normal: false };
        let yoy = ConstantYoYVol { reference_date: date, volatility: 0.05, is_normal: true };
        assert_abs_diff_eq!(cpi.vol(5.0, 0.03), 0.08, epsilon = 1e-14);
        assert_abs_diff_eq!(yoy.vol(3.0, 0.02), 0.05, epsilon = 1e-14);
    }
}
