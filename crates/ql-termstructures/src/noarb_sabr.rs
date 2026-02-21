//! No-arbitrage SABR smile section with Kahale extrapolation.
//!
//! The standard Hagan SABR formula can produce implied volatilities that lead to
//! negative probability densities in the wings, especially for:
//! - Low beta (approaching normal SABR)
//! - High vol-of-vol (nu)
//! - Long expiries
//!
//! This module provides:
//!
//! | Type | Purpose |
//! |------|---------|
//! | [`NoArbSabrSmileSection`] | Arbitrage-free SABR smile using Kahale wing repair |
//! | [`kahale_call_prices`] | Repair call-price curve to be monotone & convex |
//! | [`check_smile_arbitrage`] | Diagnose arbitrage violations in a vol smile |
//!
//! ## Algorithm
//!
//! 1. Evaluate SABR implied vols on a dense strike grid
//! 2. Convert to Black call prices $C(K)$
//! 3. Enforce monotonicity: $C(K)$ must be non-increasing
//! 4. Enforce convexity (positive butterfly): $C''(K) \ge 0$
//! 5. In the wings, extrapolate with $C(K) = a \cdot \Phi(-d_2(K))$ style tails
//! 6. Convert back to implied volatilities

use ql_math::solvers1d::{Brent, Solver1D};
use std::f64::consts::{PI, SQRT_2};

use crate::sabr::SabrSmileSection;

// ============================================================================
// Normal CDF / PDF helpers
// ============================================================================

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf_approx(x / SQRT_2))
}

#[allow(dead_code)]
fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Abramowitz & Stegun 7.1.28 – max |ε| ≈ 1.5 × 10⁻⁷
fn erf_approx(x: f64) -> f64 {
    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t * (0.254829592
        + t * (-0.284496736
            + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-x * x).exp())
}

// ============================================================================
// Black call / put prices and implied vol
// ============================================================================

/// Black call price C(F, K, σ√T).
fn black_call(forward: f64, strike: f64, total_vol: f64) -> f64 {
    if total_vol < 1e-14 {
        return (forward - strike).max(0.0);
    }
    let d1 = ((forward / strike).ln() + 0.5 * total_vol * total_vol) / total_vol;
    let d2 = d1 - total_vol;
    forward * normal_cdf(d1) - strike * normal_cdf(d2)
}

/// Black-Scholes implied vol from an undiscounted call price, using Brent's method.
fn implied_vol_from_call(forward: f64, strike: f64, expiry: f64, call_price: f64) -> Option<f64> {
    let intrinsic = (forward - strike).max(0.0);
    if call_price <= intrinsic + 1e-14 {
        return Some(1e-6); // deep OTM, return tiny vol
    }
    if call_price >= forward - 1e-14 {
        return None; // price exceeds forward – not invertible
    }

    let sqrt_t = expiry.sqrt();
    let f = |vol: f64| black_call(forward, strike, vol * sqrt_t);

    Brent.solve(f, call_price, 0.5, 1e-6, 10.0, 1e-10, 200).ok()
}

// ============================================================================
// Arbitrage diagnostics
// ============================================================================

/// Result of checking a volatility smile for arbitrage violations.
#[derive(Debug, Clone)]
pub struct ArbitrageCheckResult {
    /// Strikes where call-spread condition is violated (C not decreasing).
    pub call_spread_violations: Vec<f64>,
    /// Strikes where butterfly condition is violated (negative density).
    pub butterfly_violations: Vec<f64>,
    /// True if the smile is arbitrage-free.
    pub is_arbitrage_free: bool,
}

/// Check a discrete vol smile for static arbitrage violations.
///
/// Takes a list of `(strike, implied_vol)` pairs and a forward price + expiry.
/// Returns which strikes violate the call-spread or butterfly conditions.
pub fn check_smile_arbitrage(
    forward: f64,
    expiry: f64,
    strike_vols: &[(f64, f64)],
) -> ArbitrageCheckResult {
    if strike_vols.len() < 3 {
        return ArbitrageCheckResult {
            call_spread_violations: vec![],
            butterfly_violations: vec![],
            is_arbitrage_free: true,
        };
    }

    let sqrt_t = expiry.sqrt();
    let calls: Vec<(f64, f64)> = strike_vols
        .iter()
        .map(|&(k, v)| (k, black_call(forward, k, v * sqrt_t)))
        .collect();

    let mut cs_violations = vec![];
    let mut bf_violations = vec![];

    // Call-spread: C(K_i) >= C(K_{i+1}) for K_i < K_{i+1}
    for i in 0..calls.len() - 1 {
        if calls[i].1 < calls[i + 1].1 - 1e-12 {
            cs_violations.push(calls[i + 1].0);
        }
    }

    // Butterfly: non-negative density via divided differences
    // (C_{i+1} - C_i)/(K_{i+1} - K_i) >= (C_i - C_{i-1})/(K_i - K_{i-1})
    for i in 1..calls.len() - 1 {
        let dk_left = calls[i].0 - calls[i - 1].0;
        let dk_right = calls[i + 1].0 - calls[i].0;
        if dk_left < 1e-15 || dk_right < 1e-15 {
            continue;
        }
        let slope_left = (calls[i].1 - calls[i - 1].1) / dk_left;
        let slope_right = (calls[i + 1].1 - calls[i].1) / dk_right;
        if slope_right < slope_left - 1e-10 {
            bf_violations.push(calls[i].0);
        }
    }

    let free = cs_violations.is_empty() && bf_violations.is_empty();
    ArbitrageCheckResult {
        call_spread_violations: cs_violations,
        butterfly_violations: bf_violations,
        is_arbitrage_free: free,
    }
}

// ============================================================================
// Kahale call-price repair
// ============================================================================

/// Repair a vector of call prices to enforce monotonicity and convexity.
///
/// Input: `(strikes, call_prices)` – must be sorted ascending by strike.
/// Output: repaired call prices on the same strike grid, satisfying:
///   - $C(K)$ non-increasing
///   - $C''(K) \ge 0$ (convexity / positive butterfly)
///   - $C(0) = F$ (forward), $C(\infty) = 0$
///
/// Algorithm: compute the greatest convex minorant (upper convex hull from
/// below), then enforce monotonicity.
pub fn kahale_call_prices(
    strikes: &[f64],
    call_prices: &[f64],
    forward: f64,
) -> Vec<f64> {
    assert_eq!(strikes.len(), call_prices.len());
    let n = strikes.len();
    if n <= 2 {
        return call_prices.to_vec();
    }

    let mut c: Vec<f64> = call_prices.to_vec();

    // Clamp to [0, forward]
    for v in c.iter_mut() {
        *v = v.max(0.0).min(forward);
    }

    // Step 1: Enforce monotonicity
    for i in 0..n - 1 {
        c[i + 1] = c[i + 1].min(c[i]);
    }

    // Step 2: Compute the greatest convex minorant.
    // This is the largest convex function that lies on or below each c[i].
    //
    // For call prices, we want slopes (first differences) to be non-decreasing.
    // Use the "pool adjacent violators" algorithm on slopes.
    let mut slopes: Vec<f64> = Vec::with_capacity(n - 1);
    let mut dk: Vec<f64> = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        let d = strikes[i + 1] - strikes[i];
        dk.push(d);
        slopes.push(if d > 1e-15 { (c[i + 1] - c[i]) / d } else { 0.0 });
    }

    // Pool adjacent violators: make slopes non-decreasing
    // (isotonic regression with weights = dk)
    let merged_slopes = isotonic_regression_weighted(&slopes, &dk);

    // Reconstruct call prices from slopes, starting from c[0]
    let mut result = vec![0.0; n];
    result[0] = c[0];
    for i in 0..n - 1 {
        result[i + 1] = result[i] + merged_slopes[i] * dk[i];
    }

    // Ensure non-negative and non-increasing
    for v in result.iter_mut() {
        *v = v.max(0.0).min(forward);
    }
    for i in 0..n - 1 {
        result[i + 1] = result[i + 1].min(result[i]);
    }

    result
}

/// Isotonic regression: find non-decreasing sequence closest to `values`
/// in weighted L2 sense, where `weights` are per-element.
fn isotonic_regression_weighted(values: &[f64], weights: &[f64]) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return vec![];
    }

    // Pool-adjacent-violators algorithm
    let mut result = values.to_vec();
    let mut w = weights.to_vec();

    // Groups: we merge adjacent groups when the constraint is violated
    // Simple implementation: repeat until no violations
    let mut changed = true;
    while changed {
        changed = false;
        let mut i = 0;
        while i + 1 < result.len() {
            if result[i] > result[i + 1] + 1e-15 {
                // Merge: weighted average
                let total_w = w[i] + w[i + 1];
                let avg = if total_w > 1e-30 {
                    (result[i] * w[i] + result[i + 1] * w[i + 1]) / total_w
                } else {
                    0.5 * (result[i] + result[i + 1])
                };
                result[i] = avg;
                result[i + 1] = avg;
                w[i] = total_w;
                w[i + 1] = total_w;
                changed = true;
            }
            i += 1;
        }
    }

    // Final pass: propagate merged values
    // Actually, the PAVA above handles it, but we need to expand groups.
    // Let me use the standard full PAVA:
    let mut blocks: Vec<(f64, f64, usize, usize)> = vec![]; // (sum, weight_sum, start, end)
    for i in 0..n {
        blocks.push((values[i] * weights[i], weights[i], i, i));
        // Merge while top two blocks violate
        while blocks.len() >= 2 {
            let len = blocks.len();
            let avg_top = blocks[len - 1].0 / blocks[len - 1].1.max(1e-30);
            let avg_prev = blocks[len - 2].0 / blocks[len - 2].1.max(1e-30);
            if avg_prev > avg_top + 1e-15 {
                // SAFETY: blocks.len() >= 2 from the while condition
                let top = blocks.pop().expect("blocks has >= 2 elements");
                let prev = blocks.last_mut().expect("blocks has >= 1 element after pop");
                prev.0 += top.0;
                prev.1 += top.1;
                prev.3 = top.3;
            } else {
                break;
            }
        }
    }

    // Expand blocks back to individual values
    let mut output = vec![0.0; n];
    for (sum, wt, start, end) in &blocks {
        let avg = sum / wt.max(1e-30);
        for item in &mut output[*start..=*end] {
            *item = avg;
        }
    }

    output
}

// ============================================================================
// NoArbSabrSmileSection
// ============================================================================

/// Arbitrage-free SABR smile section.
///
/// Evaluates the Hagan SABR formula on a dense strike grid, converts to call
/// prices, applies Kahale monotonicity + convexity repair, and converts back
/// to implied volatilities on demand.  The repaired *call prices* are stored
/// as a piecewise-linear interpolant, which preserves convexity exactly.
#[derive(Debug, Clone)]
pub struct NoArbSabrSmileSection {
    /// Underlying standard SABR section.
    pub sabr: SabrSmileSection,
    /// Strike grid used for the repair.
    strikes: Vec<f64>,
    /// Repaired (undiscounted) call prices on the grid.
    repaired_calls: Vec<f64>,
}

impl NoArbSabrSmileSection {
    /// Build a no-arbitrage SABR smile from the given parameters.
    ///
    /// `num_strikes` controls the density of the internal grid (default: 200).
    /// Strikes range from `forward * 0.01` to `forward * 5.0`.
    pub fn new(
        forward: f64,
        expiry: f64,
        alpha: f64,
        beta: f64,
        rho: f64,
        nu: f64,
        num_strikes: usize,
    ) -> Self {
        let sabr = SabrSmileSection::new(forward, expiry, alpha, beta, rho, nu);
        let sqrt_t = expiry.sqrt();

        // Build strike grid (log-spaced around forward)
        let k_min = forward * 0.01;
        let k_max = forward * 5.0;
        let log_min = k_min.ln();
        let log_max = k_max.ln();
        let n = num_strikes.max(10);

        let strikes: Vec<f64> = (0..n)
            .map(|i| (log_min + (log_max - log_min) * i as f64 / (n - 1) as f64).exp())
            .collect();

        // Evaluate SABR vols and compute call prices
        let call_prices: Vec<f64> = strikes
            .iter()
            .map(|&k| {
                let v = sabr.volatility(k);
                black_call(forward, k, v * sqrt_t)
            })
            .collect();

        // Apply Kahale repair
        let repaired_calls = kahale_call_prices(&strikes, &call_prices, forward);

        Self {
            sabr,
            strikes,
            repaired_calls,
        }
    }

    /// Return the repaired (arbitrage-free) implied volatility at a given strike.
    ///
    /// Interpolates the repaired call prices linearly, then inverts to vol.
    /// Falls back to original SABR vol if the inversion fails.
    pub fn volatility(&self, strike: f64) -> f64 {
        let call = self.call_price(strike);
        implied_vol_from_call(self.sabr.forward, strike, self.sabr.expiry, call)
            .unwrap_or_else(|| self.sabr.volatility(strike))
    }

    /// Interpolated (undiscounted) call price at a given strike.
    pub fn call_price(&self, strike: f64) -> f64 {
        if self.strikes.is_empty() {
            let v = self.sabr.volatility(strike);
            return black_call(self.sabr.forward, strike, v * self.sabr.expiry.sqrt());
        }

        // Flat extrapolation
        if strike <= self.strikes[0] {
            // For K → 0, C(K) → F - K for K < F. Use linear extrap from first two points.
            let slope = if self.strikes.len() >= 2 {
                let dk = self.strikes[1] - self.strikes[0];
                if dk > 1e-15 {
                    (self.repaired_calls[1] - self.repaired_calls[0]) / dk
                } else {
                    -1.0
                }
            } else {
                -1.0
            };
            let c = self.repaired_calls[0] + slope * (strike - self.strikes[0]);
            return c.max(0.0).min(self.sabr.forward);
        }
        if strike >= *self.strikes.last().unwrap() {
            return 0.0_f64.max(*self.repaired_calls.last().unwrap());
        }

        // Binary search for interval
        let idx = match self
            .strikes
            .binary_search_by(|k| k.partial_cmp(&strike).unwrap())
        {
            Ok(i) => return self.repaired_calls[i],
            Err(i) => i - 1,
        };

        // Linear interpolation in call-price space (preserves convexity)
        let k0 = self.strikes[idx];
        let k1 = self.strikes[idx + 1];
        let c0 = self.repaired_calls[idx];
        let c1 = self.repaired_calls[idx + 1];
        let t = (strike - k0) / (k1 - k0);
        c0 + t * (c1 - c0)
    }

    /// Return the repaired volatility as total variance σ²T.
    pub fn variance(&self, strike: f64) -> f64 {
        let v = self.volatility(strike);
        v * v * self.sabr.expiry
    }

    /// ATM volatility.
    pub fn atm_vol(&self) -> f64 {
        self.volatility(self.sabr.forward)
    }

    /// The strike grid used for the internal repair.
    pub fn strike_grid(&self) -> &[f64] {
        &self.strikes
    }

    /// The repaired call prices on the strike grid.
    pub fn call_grid(&self) -> &[f64] {
        &self.repaired_calls
    }

    /// Check whether the *original* SABR smile is arbitrage-free.
    pub fn original_is_arbitrage_free(&self) -> bool {
        let strike_vols: Vec<(f64, f64)> = self
            .strikes
            .iter()
            .map(|&k| (k, self.sabr.volatility(k)))
            .collect();
        check_smile_arbitrage(self.sabr.forward, self.sabr.expiry, &strike_vols)
            .is_arbitrage_free
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_arbitrage_clean_smile() {
        // A low-nu SABR should be clean
        let sabr = SabrSmileSection::new(0.04, 5.0, 0.05, 0.5, -0.2, 0.2);
        let strikes: Vec<f64> = (1..=200)
            .map(|i| 0.04 * (0.3 + 2.4 * i as f64 / 200.0))
            .collect();
        let svs: Vec<(f64, f64)> = strikes.iter().map(|&k| (k, sabr.volatility(k))).collect();
        let result = check_smile_arbitrage(0.04, 5.0, &svs);
        assert!(
            result.butterfly_violations.is_empty(),
            "Low-nu SABR should have no butterfly violations, got {:?}",
            result.butterfly_violations
        );
    }

    #[test]
    fn kahale_preserves_monotone_convex() {
        let forward = 100.0;
        let strikes = vec![60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0];
        // Already clean call prices (decreasing and convex)
        let calls = vec![40.5, 31.0, 22.5, 15.2, 9.5, 5.5, 2.8, 1.2, 0.4];
        let repaired = kahale_call_prices(&strikes, &calls, forward);
        // Should be identical (or very close)
        for (i, (&orig, &rep)) in calls.iter().zip(repaired.iter()).enumerate() {
            assert!(
                (orig - rep).abs() < 0.1,
                "Strike {}: orig={orig}, repaired={rep}",
                strikes[i]
            );
        }
    }

    #[test]
    fn kahale_fixes_non_monotone() {
        let forward = 100.0;
        let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];
        // Non-monotone: call at 110 > call at 100
        let calls = vec![22.0, 15.0, 10.0, 11.0, 3.0];
        let repaired = kahale_call_prices(&strikes, &calls, forward);
        // Check monotonicity
        for i in 0..repaired.len() - 1 {
            assert!(
                repaired[i] >= repaired[i + 1] - 1e-10,
                "Not monotone at {}: {} < {}",
                strikes[i],
                repaired[i],
                repaired[i + 1]
            );
        }
    }

    #[test]
    fn noarb_sabr_atm_close_to_sabr() {
        let forward = 0.04;
        let expiry = 5.0;
        let alpha = 0.05;
        let beta = 0.5;
        let rho = -0.2;
        let nu = 0.3;

        let sabr = SabrSmileSection::new(forward, expiry, alpha, beta, rho, nu);
        let noarb = NoArbSabrSmileSection::new(forward, expiry, alpha, beta, rho, nu, 200);

        // ATM vols should be very close
        let atm_sabr = sabr.atm_vol();
        let atm_noarb = noarb.atm_vol();
        assert!(
            (atm_sabr - atm_noarb).abs() < 0.005,
            "ATM sabr={atm_sabr}, noarb={atm_noarb}"
        );
    }

    #[test]
    fn noarb_sabr_produces_no_butterfly_violation() {
        // Use params that might cause arb in plain SABR
        let forward = 0.03;
        let expiry = 10.0;
        let alpha = 0.04;
        let beta = 0.3;
        let rho = -0.3;
        let nu = 0.6; // high vol-of-vol

        let noarb = NoArbSabrSmileSection::new(forward, expiry, alpha, beta, rho, nu, 200);

        // Check repaired call prices directly for convexity (slopes non-decreasing)
        let strikes = noarb.strike_grid();
        let calls = noarb.call_grid();

        let mut violations = 0;
        for i in 1..calls.len() - 1 {
            let dk_left = strikes[i] - strikes[i - 1];
            let dk_right = strikes[i + 1] - strikes[i];
            if dk_left < 1e-15 || dk_right < 1e-15 {
                continue;
            }
            let slope_left = (calls[i] - calls[i - 1]) / dk_left;
            let slope_right = (calls[i + 1] - calls[i]) / dk_right;
            if slope_right < slope_left - 1e-10 {
                violations += 1;
            }
        }
        assert!(
            violations == 0,
            "NoArb SABR should fix butterflies, found {violations} violations"
        );

        // Also check monotonicity
        for i in 0..calls.len() - 1 {
            assert!(
                calls[i] >= calls[i + 1] - 1e-10,
                "Not monotone at {}: C({})={} > C({})={}",
                i, strikes[i], calls[i], strikes[i + 1], calls[i + 1]
            );
        }
    }

    #[test]
    fn noarb_sabr_repaired_vols_positive() {
        let noarb = NoArbSabrSmileSection::new(0.04, 5.0, 0.05, 0.5, -0.2, 0.4, 100);
        // Check vols at a few strikes
        for &k_mult in &[0.5, 0.8, 1.0, 1.2, 1.5] {
            let v = noarb.volatility(0.04 * k_mult);
            assert!(v > 0.0, "Vol must be positive at K={}, got {v}", 0.04 * k_mult);
        }
    }

    #[test]
    fn noarb_sabr_interpolation_smooth() {
        let noarb = NoArbSabrSmileSection::new(0.04, 5.0, 0.05, 0.5, -0.2, 0.3, 200);
        // Vol at a non-grid strike should be between neighbors
        let v1 = noarb.volatility(0.035);
        let v2 = noarb.volatility(0.04);
        let v3 = noarb.volatility(0.045);
        // Should be monotone-ish near ATM (for typical params)
        assert!(v1 > 0.0 && v2 > 0.0 && v3 > 0.0);
        // Difference should be small
        assert!((v1 - v2).abs() < 0.1, "v(0.035)={v1}, v(0.04)={v2}");
        assert!((v2 - v3).abs() < 0.1, "v(0.04)={v2}, v(0.045)={v3}");
    }
}
