//! Ultimate Forward Rate (UFR) term structure.
//!
//! The UFR method (used in Solvency II and IFRS 17) extrapolates
//! yield curves beyond the last liquid point (LLP) using a Smith-Wilson
//! kernel toward a pre-specified ultimate forward rate.
//!
//! - [`UltimateForwardTermStructure`] — the UFR extrapolated curve.
//! - [`SmithWilsonParams`] — parameters for the Smith-Wilson method.

use serde::{Deserialize, Serialize};

/// Parameters for the Smith-Wilson UFR method.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SmithWilsonParams {
    /// Ultimate forward rate (annualized, continuously compounded).
    pub ufr: f64,
    /// Convergence speed (alpha parameter, typically 0.05-0.15).
    pub alpha: f64,
    /// Last liquid point (in years).
    pub last_liquid_point: f64,
    /// Convergence tolerance (typically 1bp = 0.0001).
    pub convergence_tolerance: f64,
}

impl Default for SmithWilsonParams {
    fn default() -> Self {
        Self {
            ufr: 0.036, // 3.6% (EIOPA default)
            alpha: 0.1,
            last_liquid_point: 20.0,
            convergence_tolerance: 0.0001,
        }
    }
}

/// Ultimate Forward Rate term structure.
///
/// Matches observed zero rates up to the last liquid point (LLP) and
/// extrapolates toward the UFR using the Smith-Wilson kernel function.
#[derive(Clone, Debug)]
pub struct UltimateForwardTermStructure {
    /// Smith-Wilson parameters.
    pub params: SmithWilsonParams,
    /// Calibration maturities (in years).
    pub maturities: Vec<f64>,
    /// Observed zero rates at calibration maturities.
    pub zero_rates: Vec<f64>,
    /// Calibrated kernel weights (ζ vector).
    pub weights: Vec<f64>,
}

impl UltimateForwardTermStructure {
    /// Construct and calibrate the UFR curve from observed zero rates.
    ///
    /// # Arguments
    /// - `maturities` — observed maturities (years)
    /// - `zero_rates` — observed zero rates at those maturities
    /// - `params` — Smith-Wilson parameters
    pub fn new(
        maturities: Vec<f64>,
        zero_rates: Vec<f64>,
        params: SmithWilsonParams,
    ) -> Self {
        assert_eq!(maturities.len(), zero_rates.len());
        let n = maturities.len();

        // Discount factors from UFR
        let p_ufr: Vec<f64> = maturities.iter().map(|&t| (-params.ufr * t).exp()).collect();

        // Observed discount factors
        let p_obs: Vec<f64> = maturities.iter().zip(zero_rates.iter())
            .map(|(&t, &r)| (-r * t).exp()).collect();

        // Differences: m[i] = p_obs[i] / p_ufr[i] - 1
        let m: Vec<f64> = (0..n).map(|i| p_obs[i] / p_ufr[i] - 1.0).collect();

        // Wilson kernel matrix: W[i][j] = W(t_i, t_j)
        let mut w_mat = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                w_mat[i][j] = wilson_kernel(maturities[i], maturities[j], params.alpha, params.ufr);
            }
        }

        // Solve W * ζ = m using simple Gaussian elimination
        let weights = solve_linear_system(&w_mat, &m);

        Self {
            params,
            maturities,
            zero_rates,
            weights,
        }
    }

    /// Discount factor at time t.
    pub fn discount(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }

        let p_ufr = (-self.params.ufr * t).exp();
        let mut sum = 0.0;
        for (i, &ti) in self.maturities.iter().enumerate() {
            sum += self.weights[i] * wilson_kernel(t, ti, self.params.alpha, self.params.ufr);
        }

        p_ufr * (1.0 + sum)
    }

    /// Zero rate at time t (continuously compounded).
    pub fn zero_rate(&self, t: f64) -> f64 {
        if t <= 1e-10 {
            // Short rate limit
            return self.zero_rates.first().copied().unwrap_or(self.params.ufr);
        }
        -self.discount(t).ln() / t
    }

    /// Instantaneous forward rate at time t.
    pub fn forward_rate(&self, t: f64) -> f64 {
        let dt = 0.0001;
        let df_t = self.discount(t);
        let df_tdt = self.discount(t + dt);
        -(df_tdt / df_t).ln() / dt
    }

    /// Forward rate from t1 to t2.
    pub fn forward_rate_interval(&self, t1: f64, t2: f64) -> f64 {
        if (t2 - t1).abs() < 1e-10 {
            return self.forward_rate(t1);
        }
        (self.discount(t1) / self.discount(t2)).ln() / (t2 - t1)
    }
}

/// Smith-Wilson kernel function.
///
/// W(t, u) = exp(-ufr·(t+u)) [α·min(t,u) - 0.5·exp(-α·max(t,u))·(exp(α·min(t,u)) - exp(-α·min(t,u)))]
fn wilson_kernel(t: f64, u: f64, alpha: f64, ufr: f64) -> f64 {
    let min_tu = t.min(u);
    let max_tu = t.max(u);

    let exp_neg_ufr = (-(ufr) * (t + u)).exp();

    // Wilson function: α*min(t,u) - 0.5*exp(-α*max(t,u))*(exp(α*min(t,u)) - exp(-α*min(t,u)))
    let sinh_term = (alpha * min_tu).sinh();
    let exp_max = (-alpha * max_tu).exp();

    exp_neg_ufr * (alpha * min_tu - exp_max * sinh_term)
}

/// Solve a linear system Ax = b using Gaussian elimination with partial pivoting.
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut aug: Vec<Vec<f64>> = a.iter().enumerate().map(|(i, row)| {
        let mut r = row.clone();
        r.push(b[i]);
        r
    }).collect();

    // Forward elimination
    for col in 0..n {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        #[allow(clippy::needless_range_loop)]
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        if pivot.abs() < 1e-30 {
            continue;
        }

        #[allow(clippy::needless_range_loop)]
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            #[allow(clippy::needless_range_loop)]
            for j in col..=n {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum -= aug[i][j] * x[j];
        }
        if aug[i][i].abs() > 1e-30 {
            x[i] = sum / aug[i][i];
        }
    }

    x
}

// ---------------------------------------------------------------------------
// OIS Future Rate Helper
// ---------------------------------------------------------------------------

/// A rate helper for bootstrapping from OIS (overnight index swap) futures.
///
/// This converts an OIS futures price into a zero rate observation for
/// inclusion in a bootstrap procedure.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OisFutureRateHelper {
    /// Futures price (100 - implied rate).
    pub futures_price: f64,
    /// Start of the futures reference period (year fraction).
    pub start_time: f64,
    /// End of the futures reference period (year fraction).
    pub end_time: f64,
    /// Convexity adjustment (annualized, subtracted from futures rate).
    pub convexity_adjustment: f64,
}

impl OisFutureRateHelper {
    /// Create a new OIS future rate helper.
    pub fn new(futures_price: f64, start_time: f64, end_time: f64, convexity_adjustment: f64) -> Self {
        Self { futures_price, start_time, end_time, convexity_adjustment }
    }

    /// Implied forward rate from the futures price.
    pub fn implied_rate(&self) -> f64 {
        (100.0 - self.futures_price) / 100.0
    }

    /// Adjusted forward rate (after convexity adjustment).
    pub fn adjusted_rate(&self) -> f64 {
        self.implied_rate() - self.convexity_adjustment
    }

    /// Mid-point maturity.
    pub fn pillar_time(&self) -> f64 {
        0.5 * (self.start_time + self.end_time)
    }

    /// Day count fraction.
    pub fn dcf(&self) -> f64 {
        self.end_time - self.start_time
    }

    /// Convert to a zero rate observation at the pillar time.
    ///
    /// Approximation: zero_rate ≈ adjusted_rate (for short-dated futures).
    pub fn to_zero_rate_observation(&self) -> (f64, f64) {
        (self.pillar_time(), self.adjusted_rate())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_ufr_matches_observed_rates() {
        let mats = vec![1.0, 2.0, 5.0, 10.0, 20.0];
        let rates = vec![0.03, 0.032, 0.035, 0.038, 0.040];
        let params = SmithWilsonParams {
            ufr: 0.042,
            alpha: 0.1,
            last_liquid_point: 20.0,
            convergence_tolerance: 0.0001,
        };

        let curve = UltimateForwardTermStructure::new(mats.clone(), rates.clone(), params);

        // Check that the curve reproduces observed rates
        for (i, &t) in mats.iter().enumerate() {
            let fitted = curve.zero_rate(t);
            assert_abs_diff_eq!(fitted, rates[i], epsilon = 0.002);
        }
    }

    #[test]
    fn test_ufr_convergence() {
        let mats = vec![1.0, 5.0, 10.0, 20.0];
        let rates = vec![0.03, 0.035, 0.038, 0.040];
        let ufr = 0.042;
        let params = SmithWilsonParams {
            ufr,
            alpha: 0.1,
            last_liquid_point: 20.0,
            convergence_tolerance: 0.0001,
        };

        let curve = UltimateForwardTermStructure::new(mats, rates, params);

        // Forward rate at very long maturities should converge to UFR
        let fwd_100 = curve.forward_rate(100.0);
        assert_abs_diff_eq!(fwd_100, ufr, epsilon = 0.005);
    }

    #[test]
    fn test_ufr_discount_factors_decreasing() {
        let mats = vec![1.0, 5.0, 10.0, 20.0];
        let rates = vec![0.03, 0.035, 0.038, 0.040];
        let params = SmithWilsonParams::default();

        let curve = UltimateForwardTermStructure::new(mats, rates, params);

        let df1 = curve.discount(1.0);
        let df5 = curve.discount(5.0);
        let df10 = curve.discount(10.0);
        let df30 = curve.discount(30.0);

        assert!(df1 > df5 && df5 > df10 && df10 > df30);
        assert!(df1 < 1.0 && df30 > 0.0);
    }

    #[test]
    fn test_ois_future_helper() {
        let helper = OisFutureRateHelper::new(94.75, 0.25, 0.50, 0.0001);
        assert_abs_diff_eq!(helper.implied_rate(), 0.0525, epsilon = 1e-10);
        assert_abs_diff_eq!(helper.adjusted_rate(), 0.0524, epsilon = 1e-4);
        assert_abs_diff_eq!(helper.pillar_time(), 0.375, epsilon = 1e-10);
    }
}
