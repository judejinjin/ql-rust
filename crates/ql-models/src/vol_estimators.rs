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
}
