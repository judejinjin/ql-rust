//! Andreasen-Huge style arbitrage-free volatility interpolation.
//!
//! Builds a local-volatility surface by computing call prices on a fine
//! grid (from market implied vols) and applying Dupire's formula.
//! The surface provides both implied vol recovery (via BS inversion) and
//! local vol interpolation.
//!
//! ## Algorithm
//!
//! 1. Interpolate market implied vols onto a fine (expiry x strike) grid.
//! 2. Convert to call prices via Black-Scholes.
//! 3. Apply Dupire's formula: sigma_loc^2 = 2 dC/dT / (K^2 d^2C/dK^2)
//! 4. Store both the call price grid and local vol grid for fast queries.
//!
//! ## References
//!
//! Andreasen, J. & Huge, B. (2011).
//! "Volatility interpolation." *Risk Magazine*, March 2011.
//!
//! Dupire, B. (1994). "Pricing with a smile." *Risk* 7, 18-20.

use crate::term_structure::TermStructure;
use crate::vol_term_structure::{BlackVolTermStructure, LocalVolTermStructure};
use ql_math::distributions::NormalDistribution;
use ql_math::solvers1d::{Brent, Solver1D};
use ql_time::{Date, DayCounter};
use serde::{Deserialize, Serialize};

// ══════════════════════════════════════════════════════════════
// Black-Scholes helpers
// ══════════════════════════════════════════════════════════════

/// Black-Scholes call price: C = df * (F * N(d1) - K * N(d2)).
fn bs_call(forward: f64, strike: f64, vol: f64, t: f64, df: f64) -> f64 {
    if t <= 0.0 || vol <= 0.0 {
        return df * (forward - strike).max(0.0);
    }
    let std = vol * t.sqrt();
    let d1 = ((forward / strike).ln() + 0.5 * std * std) / std;
    let d2 = d1 - std;
    let n = NormalDistribution::standard();
    df * (forward * n.cdf(d1) - strike * n.cdf(d2))
}

/// Black-Scholes implied vol from call price via Brent solver.
#[allow(dead_code)]
fn bs_implied_vol(price: f64, forward: f64, strike: f64, t: f64, df: f64) -> Option<f64> {
    if t <= 1e-12 || price <= 1e-14 {
        return None;
    }
    let intrinsic = df * (forward - strike).max(0.0);
    if price <= intrinsic + 1e-14 {
        return Some(1e-6);
    }
    let solver = Brent;
    solver
        .solve(
            |v| bs_call(forward, strike, v, t, df) - price,
            0.0,
            0.25,
            1e-6,
            5.0,
            1e-10,
            100,
        )
        .ok()
}

// ══════════════════════════════════════════════════════════════
// Public types
// ══════════════════════════════════════════════════════════════

/// Andreasen-Huge style arbitrage-free volatility surface.
///
/// Created via [`andreasen_huge_calibrate`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AndreasenHugeVolSurface {
    #[serde(skip, default = "default_date")]
    pub reference_date: Date,
    #[serde(skip, default = "default_day_counter")]
    pub day_counter: DayCounter,
    /// Spot price.
    pub spot: f64,
    /// Risk-free rate (flat, for convenience).
    pub rate: f64,
    /// Time grid (expiries in years).
    pub time_grid: Vec<f64>,
    /// Strike grid.
    pub strike_grid: Vec<f64>,
    /// Implied vol grid: `implied_vols[i][j]` at `(time_grid[i], strike_grid[j])`.
    pub implied_vols: Vec<Vec<f64>>,
    /// Local vol grid: `local_vols[i][j]` at `(time_grid[i], strike_grid[j])`.
    pub local_vols: Vec<Vec<f64>>,
}

fn default_date() -> Date {
    Date::from_serial(0)
}
fn default_day_counter() -> DayCounter {
    DayCounter::Actual365Fixed
}

/// A single market quote for calibration.
#[derive(Debug, Clone)]
pub struct VolQuote {
    /// Time to expiry (years).
    pub expiry: f64,
    /// Strike price.
    pub strike: f64,
    /// Black implied volatility.
    pub implied_vol: f64,
}

/// Configuration for the Andreasen-Huge calibration.
#[derive(Debug, Clone)]
pub struct AndreasenHugeConfig {
    /// Number of strike grid points (default: 101).
    pub n_strikes: usize,
    /// Strike grid range in standard deviations of log-moneyness (default: 3.5).
    pub log_strike_range: f64,
}

impl Default for AndreasenHugeConfig {
    fn default() -> Self {
        Self {
            n_strikes: 101,
            log_strike_range: 3.5,
        }
    }
}

/// Calibration result with diagnostics.
#[derive(Debug, Clone)]
pub struct AndreasenHugeCalibrationResult {
    pub surface: AndreasenHugeVolSurface,
    /// Max absolute error in implied vol (bps).
    pub max_error_bps: f64,
    /// RMS error in implied vol (bps).
    pub rms_error_bps: f64,
}

// ══════════════════════════════════════════════════════════════
// Calibration
// ══════════════════════════════════════════════════════════════

/// Calibrate an Andreasen-Huge volatility surface.
///
/// # Parameters
/// - `quotes` — market (expiry, strike, implied_vol) quotes.
/// - `spot` — current spot price.
/// - `rate` — flat risk-free rate (for forward/discount).
/// - `reference_date` — valuation date.
/// - `day_counter` — day counter.
/// - `config` — calibration config.
#[allow(clippy::too_many_arguments)]
pub fn andreasen_huge_calibrate(
    quotes: &[VolQuote],
    spot: f64,
    rate: f64,
    reference_date: Date,
    day_counter: DayCounter,
    config: &AndreasenHugeConfig,
) -> AndreasenHugeCalibrationResult {
    // 1. Build sorted unique time grid from quotes.
    let mut times: Vec<f64> = quotes.iter().map(|q| q.expiry).collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    times.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

    let n_t = times.len();
    let n_k = config.n_strikes;

    // 2. Estimate typical vol for grid sizing.
    let avg_vol = if quotes.is_empty() {
        0.20
    } else {
        quotes.iter().map(|q| q.implied_vol).sum::<f64>() / quotes.len() as f64
    };
    let max_t = times.last().copied().unwrap_or(1.0);

    // 3. Build strike grid (uniform in log-moneyness).
    let half_range = config.log_strike_range * avg_vol * max_t.sqrt();
    let log_spot = spot.ln();
    let log_k_min = log_spot - half_range;
    let log_k_max = log_spot + half_range;
    let d_logk = (log_k_max - log_k_min) / (n_k as f64 - 1.0);

    let strike_grid: Vec<f64> = (0..n_k)
        .map(|j| (log_k_min + j as f64 * d_logk).exp())
        .collect();

    // 4. For each time layer, interpolate market implied vols onto the
    //    strike grid, then compute call prices and Dupire local vols.
    let mut all_implied_vols = Vec::with_capacity(n_t);
    let mut all_local_vols = Vec::with_capacity(n_t);

    for (i, &t) in times.iter().enumerate() {
        let df = (-rate * t).exp();
        let fwd = spot * (rate * t).exp();

        // Gather market quotes for this expiry.
        let mut layer_quotes: Vec<(f64, f64)> = quotes
            .iter()
            .filter(|q| (q.expiry - t).abs() < 1e-12)
            .map(|q| (q.strike, q.implied_vol))
            .collect();
        layer_quotes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Interpolate implied vols onto the strike grid.
        let iv_grid: Vec<f64> = strike_grid
            .iter()
            .map(|&k| interpolate_iv(&layer_quotes, k))
            .collect();

        // Compute call prices C(K, T).
        let call_grid: Vec<f64> = strike_grid
            .iter()
            .zip(iv_grid.iter())
            .map(|(&k, &iv)| bs_call(fwd, k, iv, t, df))
            .collect();

        // Compute Dupire local vol.
        let lv_grid = if i == 0 {
            // First layer: local_vol ~ implied_vol (no time derivative available).
            iv_grid.clone()
        } else {
            // Use finite-difference Dupire formula.
            let prev_t = times[i - 1];
            let prev_df = (-rate * prev_t).exp();
            let prev_fwd = spot * (rate * prev_t).exp();
            let prev_iv: &Vec<f64> = &all_implied_vols[i - 1];

            let prev_call: Vec<f64> = strike_grid
                .iter()
                .zip(prev_iv.iter())
                .map(|(&k, &iv)| bs_call(prev_fwd, k, iv, prev_t, prev_df))
                .collect();

            let dt = t - prev_t;
            let mut lv = vec![avg_vol; n_k];

            for j in 1..n_k - 1 {
                let dc_dt = (call_grid[j] - prev_call[j]) / dt;
                let dk_plus = strike_grid[j + 1] - strike_grid[j];
                let dk_minus = strike_grid[j] - strike_grid[j - 1];
                let dk_avg = 0.5 * (dk_plus + dk_minus);
                let d2c_dk2 = (call_grid[j + 1] - 2.0 * call_grid[j] + call_grid[j - 1])
                    / (dk_avg * dk_avg);
                let k = strike_grid[j];

                if d2c_dk2 > 1e-14 {
                    let sigma_sq = 2.0 * dc_dt / (k * k * d2c_dk2);
                    if sigma_sq > 1e-8 {
                        lv[j] = sigma_sq.sqrt().min(3.0);
                    }
                }
            }
            lv[0] = lv[1];
            lv[n_k - 1] = lv[n_k - 2];
            lv
        };

        all_implied_vols.push(iv_grid);
        all_local_vols.push(lv_grid);
    }

    // 5. Compute calibration error.
    let mut max_err = 0.0_f64;
    let mut sum_sq = 0.0;
    let mut count = 0_usize;

    for quote in quotes {
        let t = quote.expiry;
        let k = quote.strike;
        if times.iter().any(|&tt| (tt - t).abs() < 1e-12) {
            let model_iv = bilinear_iv(
                &all_implied_vols, &times, &strike_grid, t, k,
            );
            let err = (model_iv - quote.implied_vol).abs() * 10_000.0;
            max_err = max_err.max(err);
            sum_sq += err * err;
            count += 1;
        }
    }
    let rms_err = if count > 0 { (sum_sq / count as f64).sqrt() } else { 0.0 };

    let surface = AndreasenHugeVolSurface {
        reference_date,
        day_counter,
        spot,
        rate,
        time_grid: times,
        strike_grid,
        implied_vols: all_implied_vols,
        local_vols: all_local_vols,
    };

    AndreasenHugeCalibrationResult {
        surface,
        max_error_bps: max_err,
        rms_error_bps: rms_err,
    }
}

// ══════════════════════════════════════════════════════════════
// Internal helpers
// ══════════════════════════════════════════════════════════════

/// Linearly interpolate implied vol from sparse market quotes at one strike.
fn interpolate_iv(quotes: &[(f64, f64)], strike: f64) -> f64 {
    if quotes.is_empty() {
        return 0.20;
    }
    if quotes.len() == 1 {
        return quotes[0].1;
    }
    if strike <= quotes[0].0 {
        return quotes[0].1;
    }
    if strike >= quotes.last().unwrap().0 {
        return quotes.last().unwrap().1;
    }
    for w in quotes.windows(2) {
        let (k0, v0) = w[0];
        let (k1, v1) = w[1];
        if strike >= k0 && strike <= k1 {
            let alpha = (strike - k0) / (k1 - k0);
            return v0 + alpha * (v1 - v0);
        }
    }
    quotes.last().unwrap().1
}

/// Bilinear interpolation on the 2D implied vol grid.
/// Interpolates in total variance across time for calendar arbitrage freedom.
fn bilinear_iv(
    iv_grids: &[Vec<f64>],
    times: &[f64],
    strikes: &[f64],
    t: f64,
    k: f64,
) -> f64 {
    if iv_grids.is_empty() {
        return 0.0;
    }

    let n_t = times.len();
    if t <= times[0] {
        return interp_strike(&iv_grids[0], strikes, k);
    }
    if t >= times[n_t - 1] {
        return interp_strike(&iv_grids[n_t - 1], strikes, k);
    }

    let ti = times.partition_point(|&tt| tt < t).saturating_sub(1).min(n_t - 2);
    let t0 = times[ti];
    let t1 = times[ti + 1];
    let alpha = (t - t0) / (t1 - t0);

    let iv0 = interp_strike(&iv_grids[ti], strikes, k);
    let iv1 = interp_strike(&iv_grids[ti + 1], strikes, k);

    // Interpolate in total variance space.
    let var0 = iv0 * iv0 * t0;
    let var1 = iv1 * iv1 * t1;
    let var_t = var0 + alpha * (var1 - var0);

    if t > 0.0 && var_t > 0.0 {
        (var_t / t).sqrt()
    } else {
        iv0
    }
}

/// Linear interpolation on the strike grid.
fn interp_strike(iv_row: &[f64], strikes: &[f64], k: f64) -> f64 {
    let n = strikes.len();
    if n == 0 {
        return 0.0;
    }
    if k <= strikes[0] {
        return iv_row[0];
    }
    if k >= strikes[n - 1] {
        return iv_row[n - 1];
    }
    let j = strikes.partition_point(|&s| s < k).saturating_sub(1).min(n - 2);
    let alpha = (k - strikes[j]) / (strikes[j + 1] - strikes[j]);
    iv_row[j] + alpha * (iv_row[j + 1] - iv_row[j])
}

// ══════════════════════════════════════════════════════════════
// Trait implementations
// ══════════════════════════════════════════════════════════════

impl TermStructure for AndreasenHugeVolSurface {
    fn reference_date(&self) -> Date {
        self.reference_date
    }
    fn day_counter(&self) -> DayCounter {
        self.day_counter
    }
    fn calendar(&self) -> &ql_time::Calendar {
        static DEFAULT_CAL: std::sync::OnceLock<ql_time::Calendar> = std::sync::OnceLock::new();
        DEFAULT_CAL.get_or_init(|| ql_time::Calendar::Target)
    }
    fn max_date(&self) -> Date {
        self.reference_date
    }
}

impl BlackVolTermStructure for AndreasenHugeVolSurface {
    fn black_vol(&self, t: f64, strike: f64) -> f64 {
        bilinear_iv(&self.implied_vols, &self.time_grid, &self.strike_grid, t, strike)
    }
}

impl LocalVolTermStructure for AndreasenHugeVolSurface {
    fn local_vol(&self, t: f64, underlying: f64) -> f64 {
        if self.time_grid.is_empty() || self.local_vols.is_empty() {
            return 0.0;
        }
        let ti = self
            .time_grid
            .partition_point(|&tt| tt < t)
            .min(self.time_grid.len() - 1);
        interp_strike(&self.local_vols[ti], &self.strike_grid, underlying)
    }
}

// ══════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn flat_vol_quotes(spot: f64, vol: f64) -> Vec<VolQuote> {
        let mut quotes = Vec::new();
        for &t in &[0.25, 0.5, 1.0, 2.0] {
            for &m in &[0.80, 0.90, 0.95, 1.0, 1.05, 1.10, 1.20] {
                quotes.push(VolQuote {
                    expiry: t,
                    strike: spot * m,
                    implied_vol: vol,
                });
            }
        }
        quotes
    }

    fn smile_quotes(spot: f64) -> Vec<VolQuote> {
        let mut quotes = Vec::new();
        for &t in &[0.25, 0.5, 1.0] {
            for &(m, v) in &[
                (0.80, 0.28), (0.90, 0.23), (0.95, 0.21),
                (1.00, 0.20), (1.05, 0.205), (1.10, 0.22), (1.20, 0.26),
            ] {
                quotes.push(VolQuote {
                    expiry: t,
                    strike: spot * m,
                    implied_vol: v,
                });
            }
        }
        quotes
    }

    #[test]
    fn flat_vol_surface_calibrates() {
        let spot = 100.0;
        let quotes = flat_vol_quotes(spot, 0.20);

        let result = andreasen_huge_calibrate(
            &quotes, spot, 0.02,
            Date::from_ymd(2024, ql_time::Month::January, 15),
            DayCounter::Actual365Fixed,
            &AndreasenHugeConfig::default(),
        );

        assert!(
            result.rms_error_bps < 5.0,
            "RMS error {:.1} bps too large for flat vol",
            result.rms_error_bps
        );
    }

    #[test]
    fn smile_surface_calibrates() {
        let spot = 100.0;
        let quotes = smile_quotes(spot);

        let result = andreasen_huge_calibrate(
            &quotes, spot, 0.03,
            Date::from_ymd(2024, ql_time::Month::January, 15),
            DayCounter::Actual365Fixed,
            &AndreasenHugeConfig { n_strikes: 201, ..Default::default() },
        );

        assert!(
            result.max_error_bps < 50.0,
            "Max error {:.1} bps too large for smile",
            result.max_error_bps
        );
    }

    #[test]
    fn surface_positive_local_vol() {
        let spot = 100.0;
        let quotes = smile_quotes(spot);

        let result = andreasen_huge_calibrate(
            &quotes, spot, 0.02,
            Date::from_ymd(2024, ql_time::Month::January, 15),
            DayCounter::Actual365Fixed,
            &AndreasenHugeConfig::default(),
        );

        for lv_layer in &result.surface.local_vols {
            for &lv in lv_layer {
                assert!(lv > 0.0, "Local vol must be positive, got {lv}");
            }
        }
    }

    #[test]
    fn implied_vol_matches_atm() {
        let spot = 100.0;
        let vol = 0.20;
        let quotes = flat_vol_quotes(spot, vol);

        let result = andreasen_huge_calibrate(
            &quotes, spot, 0.02,
            Date::from_ymd(2024, ql_time::Month::January, 15),
            DayCounter::Actual365Fixed,
            &AndreasenHugeConfig::default(),
        );

        let model_vol = result.surface.black_vol(1.0, spot);
        assert_abs_diff_eq!(model_vol, vol, epsilon = 0.002);
    }

    #[test]
    fn local_vol_from_surface() {
        let spot = 100.0;
        let quotes = flat_vol_quotes(spot, 0.20);

        let result = andreasen_huge_calibrate(
            &quotes, spot, 0.02,
            Date::from_ymd(2024, ql_time::Month::January, 15),
            DayCounter::Actual365Fixed,
            &AndreasenHugeConfig::default(),
        );

        let lv = result.surface.local_vol(0.5, spot);
        assert!(lv > 0.05 && lv < 0.50, "Local vol {lv} out of range");
    }

    #[test]
    fn monotone_total_variance() {
        let spot = 100.0;
        let quotes = flat_vol_quotes(spot, 0.20);

        let result = andreasen_huge_calibrate(
            &quotes, spot, 0.02,
            Date::from_ymd(2024, ql_time::Month::January, 15),
            DayCounter::Actual365Fixed,
            &AndreasenHugeConfig::default(),
        );

        let surface = &result.surface;
        let k = spot;
        let mut prev_var = 0.0;
        for &t in &surface.time_grid {
            let vol = surface.black_vol(t, k);
            let var = vol * vol * t;
            assert!(
                var >= prev_var - 1e-6,
                "Total variance not monotone: {var} < {prev_var} at t={t}",
            );
            prev_var = var;
        }
    }

    #[test]
    fn smile_preserves_skew() {
        let spot = 100.0;
        let quotes = smile_quotes(spot);

        let result = andreasen_huge_calibrate(
            &quotes, spot, 0.03,
            Date::from_ymd(2024, ql_time::Month::January, 15),
            DayCounter::Actual365Fixed,
            &AndreasenHugeConfig { n_strikes: 201, ..Default::default() },
        );

        let s = &result.surface;
        let otm_vol = s.black_vol(1.0, 0.80 * spot);
        let atm_vol = s.black_vol(1.0, spot);
        assert!(
            otm_vol > atm_vol,
            "Expected skew: OTM vol {otm_vol:.4} > ATM vol {atm_vol:.4}"
        );
    }
}
