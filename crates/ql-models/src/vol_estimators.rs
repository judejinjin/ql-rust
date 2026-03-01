//! Historical volatility estimators.
//!
//! Implements GARCH(1,1) parameter estimation and the Garman-Klass (1980)
//! OHLC volatility estimator, commonly used in QuantLib for calibrating
//! stochastic volatility models from historical data.
//!
//! References:
//! - Bollerslev, T. (1986), "Generalized Autoregressive Conditional Heteroscedasticity", JBES.
//! - Garman, M. & Klass, M. (1980), "On the Estimation of Security Price Volatilities
//!   from Historical Data", Journal of Business.
//! - Rogers, L. & Satchell, S. (1991), "Estimating Variance From High, Low and Closing Prices", AAPF.
//! - Yang, D. & Zhang, Q. (2000), "Drift Independent Volatility Estimation", JET.

use serde::{Deserialize, Serialize};

/// GARCH(1,1) parameters:  σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GarchParams {
    /// Long-run variance weight ω.
    pub omega: f64,
    /// News/innovation coefficient α.
    pub alpha: f64,
    /// Persistence coefficient β.
    pub beta: f64,
    /// Long-run annualised volatility = √(ω / (1 − α − β) × 252).
    pub long_run_vol: f64,
    /// Log-likelihood of the fit.
    pub log_likelihood: f64,
}

/// Fit GARCH(1,1) parameters to daily log returns using maximum-likelihood
/// estimation (closed-form MLE update).
///
/// # Arguments
/// - `log_returns` — vector of daily log returns
/// - `max_iter` — maximum EM-style iterations
/// - `tolerance` — convergence tolerance on parameter change
///
/// # Returns
/// Fitted GARCH(1,1) parameters.
pub fn garch_fit(
    log_returns: &[f64],
    max_iter: u32,
    tolerance: f64,
) -> GarchParams {
    let n = log_returns.len();
    if n < 10 {
        let var = sample_variance(log_returns);
        return GarchParams {
            omega: var * 0.1,
            alpha: 0.1,
            beta: 0.8,
            long_run_vol: (var * 252.0).sqrt(),
            log_likelihood: f64::NEG_INFINITY,
        };
    }

    // Initial guesses
    let sample_var = sample_variance(log_returns);
    let mut omega = sample_var * 0.05;
    let mut alpha = 0.10;
    let mut beta = 0.85;

    let mut best_ll = f64::NEG_INFINITY;
    let mut best = (omega, alpha, beta);

    for _iter in 0..max_iter {
        // Compute conditional variances
        let mut sigma2 = vec![0.0; n];
        sigma2[0] = sample_var;
        for t in 1..n {
            sigma2[t] = omega + alpha * log_returns[t - 1].powi(2) + beta * sigma2[t - 1];
            sigma2[t] = sigma2[t].max(1e-12);
        }

        // Log-likelihood (Gaussian)
        let ll: f64 = (1..n).map(|t| {
            -0.5 * (sigma2[t].ln() + log_returns[t].powi(2) / sigma2[t])
        }).sum();

        if ll > best_ll {
            best_ll = ll;
            best = (omega, alpha, beta);
        }

        // Simple gradient-ascent step
        let eps = 1e-6;

        // dLL/d(omega)
        let dll_domega = numerical_gradient(log_returns, omega, alpha, beta, 0, eps, &sigma2);
        // dLL/d(alpha)
        let dll_dalpha = numerical_gradient(log_returns, omega, alpha, beta, 1, eps, &sigma2);
        // dLL/d(beta)
        let dll_dbeta = numerical_gradient(log_returns, omega, alpha, beta, 2, eps, &sigma2);

        let step = 1e-5;
        let new_omega = (omega + step * dll_domega).max(1e-10);
        let new_alpha = (alpha + step * dll_dalpha).clamp(0.001, 0.49);
        let new_beta = (beta + step * dll_dbeta).clamp(0.01, 0.99);

        // Ensure stationarity: alpha + beta < 1
        let sum_ab = new_alpha + new_beta;
        let (na, nb) = if sum_ab >= 0.999 {
            (new_alpha * 0.999 / sum_ab, new_beta * 0.999 / sum_ab)
        } else {
            (new_alpha, new_beta)
        };

        let change = (new_omega - omega).abs() + (na - alpha).abs() + (nb - beta).abs();
        omega = new_omega;
        alpha = na;
        beta = nb;

        if change < tolerance { break; }
    }

    omega = best.0;
    alpha = best.1;
    beta = best.2;

    let long_run_var = if (1.0 - alpha - beta).abs() > 1e-8 {
        omega / (1.0 - alpha - beta)
    } else {
        sample_var
    };

    GarchParams {
        omega,
        alpha,
        beta,
        long_run_vol: (long_run_var * 252.0).sqrt(),
        log_likelihood: best_ll,
    }
}

/// Numerical gradient of log-likelihood for GARCH.
fn numerical_gradient(
    returns: &[f64], omega: f64, alpha: f64, beta: f64,
    param_idx: usize, eps: f64, _base_sigma2: &[f64],
) -> f64 {
    let (o_up, a_up, b_up) = match param_idx {
        0 => (omega + eps, alpha, beta),
        1 => (omega, alpha + eps, beta),
        2 => (omega, alpha, beta + eps),
        _ => unreachable!(),
    };
    let ll_up = garch_log_likelihood(returns, o_up, a_up, b_up);
    let ll_base = garch_log_likelihood(returns, omega, alpha, beta);
    (ll_up - ll_base) / eps
}

fn garch_log_likelihood(returns: &[f64], omega: f64, alpha: f64, beta: f64) -> f64 {
    let n = returns.len();
    if n < 2 { return f64::NEG_INFINITY; }
    let mut sigma2 = sample_variance(returns);
    let mut ll = 0.0;
    for t in 1..n {
        sigma2 = omega + alpha * returns[t - 1].powi(2) + beta * sigma2;
        sigma2 = sigma2.max(1e-14);
        ll += -0.5 * (sigma2.ln() + returns[t].powi(2) / sigma2);
    }
    ll
}

fn sample_variance(data: &[f64]) -> f64 {
    if data.len() < 2 { return 1e-4; }
    let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
    data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64
}

/// GARCH(1,1) forecast: predict conditional variance for `horizon` steps ahead.
///
/// Uses the recursion:
///   σ²_{t+k} = V_L + (α+β)^k · (σ²_t − V_L)
///
/// where V_L = ω/(1−α−β) is the long-run variance.
pub fn garch_forecast(params: &GarchParams, current_sigma2: f64, horizon: u32) -> Vec<f64> {
    let vl = if (1.0 - params.alpha - params.beta).abs() > 1e-8 {
        params.omega / (1.0 - params.alpha - params.beta)
    } else {
        current_sigma2
    };
    let persistence = params.alpha + params.beta;
    (1..=horizon)
        .map(|k| vl + persistence.powi(k as i32) * (current_sigma2 - vl))
        .collect()
}

/// Garman-Klass (1980) OHLC volatility estimator.
///
/// Uses open, high, low, close prices to estimate volatility more efficiently
/// than close-to-close.
///
///   σ²_GK = (1/n) Σ [ 0.5·(ln H/L)² − (2ln2−1)·(ln C/O)² ]
///
/// Returns annualised volatility (assuming 252 trading days).
///
/// # Arguments
/// - `open` — opening prices
/// - `high` — high prices
/// - `low` — low prices
/// - `close` — closing prices
pub fn garman_klass_vol(
    open: &[f64], high: &[f64], low: &[f64], close: &[f64],
) -> f64 {
    let n = open.len().min(high.len()).min(low.len()).min(close.len());
    if n == 0 { return 0.0; }

    let mut var_sum = 0.0;
    for i in 0..n {
        let ln_hl = (high[i] / low[i]).ln();
        let ln_co = (close[i] / open[i]).ln();
        var_sum += 0.5 * ln_hl * ln_hl - (2.0_f64.ln() * 2.0 - 1.0) * ln_co * ln_co;
    }
    let daily_var = var_sum / n as f64;
    (daily_var.abs() * 252.0).sqrt()
}

/// Rogers-Satchell (1991) volatility estimator.
///
/// More robust to drift than Garman-Klass:
///   σ²_RS = (1/n) Σ [ ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O) ]
///
/// Returns annualised volatility.
pub fn rogers_satchell_vol(
    open: &[f64], high: &[f64], low: &[f64], close: &[f64],
) -> f64 {
    let n = open.len().min(high.len()).min(low.len()).min(close.len());
    if n == 0 { return 0.0; }

    let mut var_sum = 0.0;
    for i in 0..n {
        let hc = (high[i] / close[i]).ln();
        let ho = (high[i] / open[i]).ln();
        let lc = (low[i] / close[i]).ln();
        let lo = (low[i] / open[i]).ln();
        var_sum += hc * ho + lc * lo;
    }
    let daily_var = var_sum / n as f64;
    (daily_var.abs() * 252.0).sqrt()
}

/// Yang-Zhang (2000) volatility estimator — combines overnight and OHLC.
///
/// Drift-independent and handles opening jumps:
///   σ²_YZ = σ²_overnight + k·σ²_close + (1−k)·σ²_RS
///
/// where k = 0.34 / (1.34 + (n+1)/(n-1)).
///
/// Returns annualised volatility.
///
/// # Arguments
/// - `open` — opening prices (length n)
/// - `high` — high prices
/// - `low` — low prices
/// - `close` — closing prices
/// - `prev_close` — previous day close prices (length n)
pub fn yang_zhang_vol(
    open: &[f64], high: &[f64], low: &[f64], close: &[f64], prev_close: &[f64],
) -> f64 {
    let n = open.len()
        .min(high.len())
        .min(low.len())
        .min(close.len())
        .min(prev_close.len());
    if n < 3 { return 0.0; }

    // Overnight returns = ln(O_t / C_{t-1})
    let overnight: Vec<f64> = (0..n).map(|i| (open[i] / prev_close[i]).ln()).collect();
    let mean_ov: f64 = overnight.iter().sum::<f64>() / n as f64;
    let var_ov: f64 = overnight.iter().map(|x| (x - mean_ov).powi(2)).sum::<f64>() / (n - 1) as f64;

    // Close-to-close returns
    let close_ret: Vec<f64> = (0..n).map(|i| (close[i] / open[i]).ln()).collect();
    let mean_cr: f64 = close_ret.iter().sum::<f64>() / n as f64;
    let var_close: f64 = close_ret.iter().map(|x| (x - mean_cr).powi(2)).sum::<f64>() / (n - 1) as f64;

    // Rogers-Satchell
    let mut var_rs = 0.0;
    for i in 0..n {
        let hc = (high[i] / close[i]).ln();
        let ho = (high[i] / open[i]).ln();
        let lc = (low[i] / close[i]).ln();
        let lo = (low[i] / open[i]).ln();
        var_rs += hc * ho + lc * lo;
    }
    var_rs /= n as f64;

    let k = 0.34 / (1.34 + (n as f64 + 1.0) / (n as f64 - 1.0));
    let daily_var = var_ov + k * var_close + (1.0 - k) * var_rs;
    (daily_var.abs() * 252.0).sqrt()
}

// ===========================================================================
// Struct-based volatility estimators (Phase 16 wrappers)
// ===========================================================================

/// Trait for historical volatility estimators.
pub trait VolatilityEstimator {
    /// Estimated annualised volatility.
    fn vol(&self) -> f64;
    /// Number of observations used.
    fn n_obs(&self) -> usize;
}

/// GARCH(1,1) volatility estimator (struct-based wrapper around [`garch_fit`]).
///
/// Fits GARCH(1,1) to a return series and provides the long-run volatility
/// estimate plus conditional volatility forecasts.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GarchEstimator {
    /// Fitted GARCH parameters.
    pub params: GarchParams,
    /// Number of observations.
    pub n_obs: usize,
    /// Last conditional variance from the fit.
    pub last_sigma2: f64,
}

impl GarchEstimator {
    /// Fit from a price series.
    pub fn from_prices(prices: &[f64]) -> Self {
        assert!(prices.len() >= 3, "Need at least 3 prices");
        let log_returns: Vec<f64> = prices.windows(2).map(|w| (w[1] / w[0]).ln()).collect();
        Self::from_returns(&log_returns)
    }

    /// Fit from a log-return series.
    pub fn from_returns(log_returns: &[f64]) -> Self {
        let params = garch_fit(log_returns, 500, 1e-8);
        let n = log_returns.len();
        // Compute last conditional variance
        let sv = sample_variance(log_returns);
        let mut sigma2 = sv;
        for t in 1..n {
            sigma2 = params.omega + params.alpha * log_returns[t - 1].powi(2) + params.beta * sigma2;
            sigma2 = sigma2.max(1e-14);
        }
        Self {
            params,
            n_obs: n,
            last_sigma2: sigma2,
        }
    }

    /// Forecast conditional volatility for `horizon` steps ahead.
    pub fn forecast(&self, horizon: u32) -> Vec<f64> {
        garch_forecast(&self.params, self.last_sigma2, horizon)
            .iter()
            .map(|&v| (v * 252.0).sqrt())
            .collect()
    }
}

impl VolatilityEstimator for GarchEstimator {
    fn vol(&self) -> f64 {
        self.params.long_run_vol
    }
    fn n_obs(&self) -> usize {
        self.n_obs
    }
}

/// Garman-Klass (1980) OHLC volatility estimator (struct wrapper).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GarmanKlassEstimator {
    /// Estimated annualised volatility.
    pub vol: f64,
    /// Number of observations.
    pub n_obs: usize,
}

impl GarmanKlassEstimator {
    /// Estimate from OHLC price arrays.
    pub fn new(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Self {
        let n = open.len().min(high.len()).min(low.len()).min(close.len());
        let vol = garman_klass_vol(open, high, low, close);
        Self { vol, n_obs: n }
    }
}

impl VolatilityEstimator for GarmanKlassEstimator {
    fn vol(&self) -> f64 {
        self.vol
    }
    fn n_obs(&self) -> usize {
        self.n_obs
    }
}

/// Rogers-Satchell (1991) volatility estimator (struct wrapper).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RogersSatchellEstimator {
    /// Estimated annualised volatility.
    pub vol: f64,
    /// Number of observations.
    pub n_obs: usize,
}

impl RogersSatchellEstimator {
    /// Estimate from OHLC price arrays.
    pub fn new(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Self {
        let n = open.len().min(high.len()).min(low.len()).min(close.len());
        let vol = rogers_satchell_vol(open, high, low, close);
        Self { vol, n_obs: n }
    }
}

impl VolatilityEstimator for RogersSatchellEstimator {
    fn vol(&self) -> f64 {
        self.vol
    }
    fn n_obs(&self) -> usize {
        self.n_obs
    }
}

/// Yang-Zhang (2000) drift-independent volatility estimator (struct wrapper).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct YangZhangEstimator {
    /// Estimated annualised volatility.
    pub vol: f64,
    /// Number of observations.
    pub n_obs: usize,
}

impl YangZhangEstimator {
    /// Estimate from OHLC + previous close.
    pub fn new(
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        prev_close: &[f64],
    ) -> Self {
        let n = open
            .len()
            .min(high.len())
            .min(low.len())
            .min(close.len())
            .min(prev_close.len());
        let vol = yang_zhang_vol(open, high, low, close, prev_close);
        Self { vol, n_obs: n }
    }
}

impl VolatilityEstimator for YangZhangEstimator {
    fn vol(&self) -> f64 {
        self.vol
    }
    fn n_obs(&self) -> usize {
        self.n_obs
    }
}

/// Simple local volatility estimator from a term structure of option
/// implied vols.
///
/// The Dupire-style local variance is:
///     σ_local²(T, K) ≈ dw/dT  (where w = σ²T is total variance)
///
/// This estimator uses finite differencing of total variance in the
/// time dimension at a given strike (ATM).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimpleLocalEstimator {
    /// Expiry times (sorted ascending).
    pub expiries: Vec<f64>,
    /// Implied Black vols corresponding to each expiry.
    pub implied_vols: Vec<f64>,
    /// Estimated local vol at each expiry midpoint.
    pub local_vols: Vec<f64>,
}

impl SimpleLocalEstimator {
    /// Construct from ATM implied vol term structure.
    ///
    /// # Arguments
    /// - `expiries` — option expiry times (years), sorted ascending
    /// - `implied_vols` — ATM implied Black vols at each expiry
    pub fn new(expiries: &[f64], implied_vols: &[f64]) -> Self {
        assert_eq!(expiries.len(), implied_vols.len());
        assert!(expiries.len() >= 2, "Need at least 2 expiries");

        let n = expiries.len();
        // Total variance: w(T) = σ²(T) × T
        let total_var: Vec<f64> = expiries
            .iter()
            .zip(implied_vols.iter())
            .map(|(&t, &v)| v * v * t)
            .collect();

        // Local variance via finite difference: σ²_loc(T_mid) ≈ Δw/ΔT
        let mut local_vols = Vec::with_capacity(n - 1);
        for i in 0..n - 1 {
            let dt = expiries[i + 1] - expiries[i];
            let dw = total_var[i + 1] - total_var[i];
            let local_var = if dt > 1e-12 { (dw / dt).max(0.0) } else { 0.0 };
            local_vols.push(local_var.sqrt());
        }

        Self {
            expiries: expiries.to_vec(),
            implied_vols: implied_vols.to_vec(),
            local_vols,
        }
    }

    /// Interpolate local vol at a given time.
    pub fn local_vol_at(&self, t: f64) -> f64 {
        if self.local_vols.is_empty() {
            return 0.0;
        }
        // Expiry midpoints
        let n = self.local_vols.len();
        let midpoints: Vec<f64> = (0..n)
            .map(|i| 0.5 * (self.expiries[i] + self.expiries[i + 1]))
            .collect();

        if t <= midpoints[0] {
            return self.local_vols[0];
        }
        if t >= midpoints[n - 1] {
            return self.local_vols[n - 1];
        }
        // Linear interpolation
        for i in 0..n - 1 {
            if t <= midpoints[i + 1] {
                let frac = (t - midpoints[i]) / (midpoints[i + 1] - midpoints[i]);
                return self.local_vols[i] + frac * (self.local_vols[i + 1] - self.local_vols[i]);
            }
        }
        self.local_vols[n - 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn synthetic_returns(n: usize, daily_vol: f64) -> Vec<f64> {
        // Deterministic "returns" with known variance
        let mut ret = Vec::with_capacity(n);
        for i in 0..n {
            // Alternating +/- with slight variation
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            ret.push(sign * daily_vol * (1.0 + 0.1 * (i as f64 / n as f64)));
        }
        ret
    }

    #[test]
    fn test_garch_fit_basic() {
        let returns = synthetic_returns(500, 0.01);
        let params = garch_fit(&returns, 200, 1e-8);
        // alpha + beta < 1 (stationarity)
        assert!(params.alpha + params.beta < 1.0,
            "alpha={}, beta={}", params.alpha, params.beta);
        assert!(params.omega > 0.0);
        assert!(params.long_run_vol > 0.0 && params.long_run_vol < 1.0,
            "lr_vol={}", params.long_run_vol);
    }

    #[test]
    fn test_garch_forecast() {
        let params = GarchParams {
            omega: 1e-6, alpha: 0.1, beta: 0.85,
            long_run_vol: 0.20, log_likelihood: 0.0,
        };
        let forecasts = garch_forecast(&params, 1.5e-4, 10);
        assert_eq!(forecasts.len(), 10);
        // Forecasts should converge toward long-run variance
        let vl = params.omega / (1.0 - params.alpha - params.beta);
        assert!((forecasts[9] - vl).abs() < (forecasts[0] - vl).abs());
    }

    #[test]
    fn test_garman_klass_vol() {
        // Synthetic OHLC with known properties
        let n = 100;
        let base = 100.0;
        let open: Vec<f64> = (0..n).map(|i| base + 0.01 * i as f64).collect();
        let high: Vec<f64> = open.iter().map(|o| o * 1.02).collect();
        let low: Vec<f64> = open.iter().map(|o| o * 0.98).collect();
        let close: Vec<f64> = open.iter().map(|o| o * 1.005).collect();
        let vol = garman_klass_vol(&open, &high, &low, &close);
        assert!(vol > 0.0 && vol < 2.0, "gk_vol={}", vol);
    }

    #[test]
    fn test_rogers_satchell_vol() {
        let n = 100;
        let base = 100.0;
        let open: Vec<f64> = (0..n).map(|i| base + 0.01 * i as f64).collect();
        let high: Vec<f64> = open.iter().map(|o| o * 1.015).collect();
        let low: Vec<f64> = open.iter().map(|o| o * 0.985).collect();
        let close: Vec<f64> = open.iter().map(|o| o * 1.003).collect();
        let vol = rogers_satchell_vol(&open, &high, &low, &close);
        assert!(vol > 0.0 && vol < 2.0, "rs_vol={}", vol);
    }

    #[test]
    fn test_yang_zhang_vol() {
        let n = 100;
        let base = 100.0;
        let open: Vec<f64> = (0..n).map(|i| base + 0.01 * i as f64).collect();
        let high: Vec<f64> = open.iter().map(|o| o * 1.02).collect();
        let low: Vec<f64> = open.iter().map(|o| o * 0.98).collect();
        let close: Vec<f64> = open.iter().map(|o| o * 1.005).collect();
        let prev_close: Vec<f64> = (0..n).map(|i| base + 0.01 * (i as f64 - 0.5)).collect();
        let vol = yang_zhang_vol(&open, &high, &low, &close, &prev_close);
        assert!(vol > 0.0 && vol < 2.0, "yz_vol={}", vol);
    }

    // ---------- Phase 16 struct estimator tests ----------

    #[test]
    fn test_garch_estimator_from_prices() {
        let prices: Vec<f64> = (0..200).map(|i| 100.0 * (1.0 + 0.001 * (i as f64).sin())).collect();
        let est = GarchEstimator::from_prices(&prices);
        assert!(est.vol() > 0.0, "garch vol={}", est.vol());
        assert_eq!(est.n_obs(), 199);
        assert!(est.last_sigma2 > 0.0);
    }

    #[test]
    fn test_garch_estimator_forecast() {
        let returns = synthetic_returns(500, 0.01);
        let est = GarchEstimator::from_returns(&returns);
        let fcast = est.forecast(5);
        assert_eq!(fcast.len(), 5);
        for v in &fcast {
            assert!(*v > 0.0 && *v < 2.0, "forecast vol={}", v);
        }
    }

    #[test]
    fn test_garman_klass_estimator() {
        let n = 100;
        let base = 100.0;
        let open: Vec<f64> = (0..n).map(|i| base + 0.01 * i as f64).collect();
        let high: Vec<f64> = open.iter().map(|o| o * 1.02).collect();
        let low: Vec<f64> = open.iter().map(|o| o * 0.98).collect();
        let close: Vec<f64> = open.iter().map(|o| o * 1.005).collect();
        let est = GarmanKlassEstimator::new(&open, &high, &low, &close);
        assert_eq!(est.n_obs(), n);
        assert_abs_diff_eq!(est.vol(), garman_klass_vol(&open, &high, &low, &close), epsilon = 1e-14);
    }

    #[test]
    fn test_rogers_satchell_estimator() {
        let n = 80;
        let base = 50.0;
        let open: Vec<f64> = (0..n).map(|i| base + 0.02 * i as f64).collect();
        let high: Vec<f64> = open.iter().map(|o| o * 1.01).collect();
        let low: Vec<f64> = open.iter().map(|o| o * 0.99).collect();
        let close: Vec<f64> = open.iter().map(|o| o * 1.002).collect();
        let est = RogersSatchellEstimator::new(&open, &high, &low, &close);
        assert_eq!(est.n_obs(), n);
        assert!(est.vol() > 0.0);
    }

    #[test]
    fn test_yang_zhang_estimator() {
        let n = 100;
        let base = 100.0;
        let open: Vec<f64> = (0..n).map(|i| base + 0.01 * i as f64).collect();
        let high: Vec<f64> = open.iter().map(|o| o * 1.02).collect();
        let low: Vec<f64> = open.iter().map(|o| o * 0.98).collect();
        let close: Vec<f64> = open.iter().map(|o| o * 1.005).collect();
        let prev_close: Vec<f64> = (0..n).map(|i| base + 0.01 * (i as f64 - 0.5)).collect();
        let est = YangZhangEstimator::new(&open, &high, &low, &close, &prev_close);
        assert_eq!(est.n_obs(), n);
        assert!(est.vol() > 0.0 && est.vol() < 2.0);
    }

    #[test]
    fn test_simple_local_estimator() {
        let expiries = vec![0.25, 0.5, 1.0, 2.0, 3.0, 5.0];
        let vols = vec![0.22, 0.21, 0.20, 0.19, 0.185, 0.18];
        let est = SimpleLocalEstimator::new(&expiries, &vols);
        assert_eq!(est.local_vols.len(), 5);
        for lv in &est.local_vols {
            assert!(*lv > 0.0 && *lv < 1.0, "local vol={}", lv);
        }
        let lv_at_1 = est.local_vol_at(1.0);
        assert!(lv_at_1 > 0.0 && lv_at_1 < 0.5, "interp local vol={}", lv_at_1);
    }

    #[test]
    fn test_vol_estimator_trait() {
        let returns = synthetic_returns(500, 0.01);
        let garch: Box<dyn VolatilityEstimator> = Box::new(GarchEstimator::from_returns(&returns));
        assert!(garch.vol() > 0.0);
        assert_eq!(garch.n_obs(), 500);
    }
}
