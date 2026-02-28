//! Global bootstrap for yield curves.
//!
//! Unlike the local (sequential) bootstrap which calibrates each instrument
//! one at a time, the global bootstrap simultaneously fits all instruments by
//! minimising a global objective function. This produces a smoother curve and
//! resolves ordering dependencies that local bootstrap can suffer from.
//!
//! The algorithm solves:
//!   min_θ  Σ_i  w_i · (C_i(θ) − market_i)²  +  λ · R(θ)
//!
//! where:
//! - θ = discount factors (or zero rates) at pillar dates
//! - C_i(θ) = model price of instrument i
//! - market_i = market quote (par rate, price, etc.)
//! - R(θ) = roughness penalty (second-derivative integral)
//! - λ = smoothing parameter
//!
//! Reference:
//! - QuantLib: GlobalBootstrap<> template in globalbootstrap.hpp

use serde::{Deserialize, Serialize};
use ql_time::{Date, DayCounter};

/// A single instrument for global bootstrap.
#[derive(Clone, Debug)]
pub struct GlobalBootstrapHelper {
    /// Pillar date.
    pub maturity: Date,
    /// Market quote (par rate, price, etc.).
    pub market_quote: f64,
    /// Weight in the objective function.
    pub weight: f64,
    /// Instrument type for pricing.
    pub instrument_type: GlobalHelperType,
}

/// Types of instruments supported.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum GlobalHelperType {
    /// Deposit rate helper.
    Deposit {
        /// Settlement lag in days.
        settlement_days: u32,
        /// Tenor in months.
        tenor_months: u32,
        /// Day counter.
        day_counter: DayCounter,
    },
    /// Swap rate helper.
    Swap {
        /// Fixed leg frequency (payments per year).
        fixed_freq: u32,
        /// Fixed leg day counter.
        day_counter: DayCounter,
        /// Tenor in years.
        tenor_years: u32,
    },
    /// Zero coupon bond.
    ZeroBond {
        /// Face value.
        face: f64,
    },
    /// FRA.
    Fra {
        /// Forward start in months.
        start_months: u32,
        /// Forward end in months.
        end_months: u32,
        /// Day counter.
        day_counter: DayCounter,
    },
}

/// Result from global bootstrap.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GlobalBootstrapResult {
    /// Pillar dates (year fractions from reference date).
    pub pillar_times: Vec<f64>,
    /// Calibrated discount factors.
    pub discount_factors: Vec<f64>,
    /// Calibrated zero rates (continuous compounding).
    pub zero_rates: Vec<f64>,
    /// Final objective function value (RMSE of pricing errors).
    pub rmse: f64,
    /// Number of iterations.
    pub iterations: u32,
}

/// Perform a global bootstrap of a yield curve.
///
/// # Arguments
/// - `reference_date` — valuation date
/// - `helpers` — vector of calibration instruments
/// - `day_counter` — day counter for year fraction computation
/// - `smoothing` — smoothing (roughness penalty) parameter λ
/// - `max_iter` — maximum number of Levenberg-Marquardt iterations
/// - `tolerance` — convergence tolerance for the objective
///
/// # Returns
/// Calibrated discount factors and zero rates at each pillar.
pub fn global_bootstrap(
    reference_date: Date,
    helpers: &[GlobalBootstrapHelper],
    day_counter: DayCounter,
    smoothing: f64,
    max_iter: u32,
    tolerance: f64,
) -> GlobalBootstrapResult {
    if helpers.is_empty() {
        return GlobalBootstrapResult {
            pillar_times: vec![],
            discount_factors: vec![],
            zero_rates: vec![],
            rmse: 0.0,
            iterations: 0,
        };
    }

    // Sort helpers by maturity
    let mut sorted: Vec<_> = helpers.to_vec();
    sorted.sort_by(|a, b| a.maturity.cmp(&b.maturity));

    let n = sorted.len();
    let pillar_times: Vec<f64> = sorted.iter()
        .map(|h| day_counter.year_fraction(reference_date, h.maturity))
        .collect();

    // Initial guess: flat curve at average rate
    let avg_rate: f64 = sorted.iter().map(|h| h.market_quote).sum::<f64>() / n as f64;
    let mut zero_rates: Vec<f64> = vec![avg_rate; n];

    // Levenberg-Marquardt style optimisation
    let mut lambda_lm = 0.001_f64;
    let mut best_rmse = f64::MAX;

    for iter in 0..max_iter {
        // Compute discount factors from zero rates
        let dfs: Vec<f64> = pillar_times.iter().zip(zero_rates.iter())
            .map(|(&t, &r)| (-r * t).exp())
            .collect();

        // Compute model values and residuals
        let mut residuals = Vec::with_capacity(n);
        for (i, h) in sorted.iter().enumerate() {
            let model_val = model_value(h, &pillar_times, &dfs, i);
            residuals.push((model_val - h.market_quote) * h.weight.sqrt());
        }

        // Roughness penalty: sum of squared second differences in zero rates
        let mut roughness = 0.0;
        for i in 1..n.saturating_sub(1) {
            let d2 = zero_rates[i + 1] - 2.0 * zero_rates[i] + zero_rates[i - 1];
            roughness += d2 * d2;
        }

        let obj: f64 = residuals.iter().map(|r| r * r).sum::<f64>() + smoothing * roughness;
        let rmse = (residuals.iter().map(|r| r * r).sum::<f64>() / n as f64).sqrt();

        if rmse < tolerance {
            return GlobalBootstrapResult {
                pillar_times,
                discount_factors: dfs,
                zero_rates,
                rmse,
                iterations: iter + 1,
            };
        }

        if rmse < best_rmse {
            best_rmse = rmse;
            lambda_lm *= 0.5;
        } else {
            lambda_lm *= 2.0;
        }

        // Gradient descent step on zero rates
        let eps = 1e-6;
        for j in 0..n {
            let old_r = zero_rates[j];
            zero_rates[j] = old_r + eps;
            let dfs_up: Vec<f64> = pillar_times.iter().zip(zero_rates.iter())
                .map(|(&t, &r)| (-r * t).exp())
                .collect();
            let mut obj_up = 0.0;
            for (i, h) in sorted.iter().enumerate() {
                let model_val = model_value(h, &pillar_times, &dfs_up, i);
                let r = (model_val - h.market_quote) * h.weight.sqrt();
                obj_up += r * r;
            }
            // Add roughness penalty
            let mut rough_up = 0.0;
            for i in 1..n.saturating_sub(1) {
                let d2 = zero_rates[i + 1] - 2.0 * zero_rates[i] + zero_rates[i - 1];
                rough_up += d2 * d2;
            }
            obj_up += smoothing * rough_up;

            zero_rates[j] = old_r;
            let grad = (obj_up - obj) / eps;
            let step = grad / (grad.abs() + lambda_lm);
            zero_rates[j] = old_r - 0.001 * step;
            // Keep rates positive and bounded
            zero_rates[j] = zero_rates[j].max(-0.05).min(0.50);
        }
    }

    let dfs: Vec<f64> = pillar_times.iter().zip(zero_rates.iter())
        .map(|(&t, &r)| (-r * t).exp())
        .collect();
    let rmse = {
        let mut s = 0.0;
        for (i, h) in sorted.iter().enumerate() {
            let mv = model_value(h, &pillar_times, &dfs, i);
            let r = (mv - h.market_quote) * h.weight.sqrt();
            s += r * r;
        }
        (s / n as f64).sqrt()
    };

    GlobalBootstrapResult {
        pillar_times,
        discount_factors: dfs,
        zero_rates,
        rmse,
        iterations: max_iter,
    }
}

/// Compute model value for a helper given current discount factors.
fn model_value(
    helper: &GlobalBootstrapHelper,
    pillar_times: &[f64],
    dfs: &[f64],
    index: usize,
) -> f64 {
    let t = pillar_times[index];
    let df = dfs[index];

    match &helper.instrument_type {
        GlobalHelperType::Deposit { .. } => {
            // Deposit rate = (1/DF - 1) / t
            if t > 1e-8 { (1.0 / df - 1.0) / t } else { 0.0 }
        }
        GlobalHelperType::Swap { fixed_freq, tenor_years, .. } => {
            // Par swap rate = (1 - DF_n) / Σ τ_i · DF_i
            // Simplified: equal spacing
            let n = (*fixed_freq * *tenor_years) as usize;
            if n == 0 { return 0.0; }
            let dt = t / n as f64;
            let mut annuity = 0.0;
            for k in 1..=n {
                let tk = dt * k as f64;
                let df_k = interpolate_df(pillar_times, dfs, tk);
                annuity += dt * df_k;
            }
            let df_n = interpolate_df(pillar_times, dfs, t);
            if annuity.abs() > 1e-12 { (1.0 - df_n) / annuity } else { 0.0 }
        }
        GlobalHelperType::ZeroBond { face } => {
            // Bond price = face * DF
            face * df
        }
        GlobalHelperType::Fra { start_months, end_months, .. } => {
            let t_start = *start_months as f64 / 12.0;
            let t_end = *end_months as f64 / 12.0;
            let df_start = interpolate_df(pillar_times, dfs, t_start);
            let df_end = interpolate_df(pillar_times, dfs, t_end);
            let tau = t_end - t_start;
            if tau > 1e-8 && df_end > 1e-12 {
                (df_start / df_end - 1.0) / tau
            } else { 0.0 }
        }
    }
}

/// Log-linear interpolation of discount factors.
fn interpolate_df(times: &[f64], dfs: &[f64], t: f64) -> f64 {
    if times.is_empty() { return 1.0; }
    if t <= times[0] {
        // Flat extrapolation
        let r = if times[0] > 1e-8 { -dfs[0].ln() / times[0] } else { 0.0 };
        return (-r * t).exp();
    }
    if t >= *times.last().unwrap() {
        // Flat extrapolation
        let r = {
            let tl = *times.last().unwrap();
            if tl > 1e-8 { -dfs.last().unwrap().ln() / tl } else { 0.0 }
        };
        return (-r * t).exp();
    }
    // Find interval
    let idx = times.iter().position(|&ti| ti >= t).unwrap_or(times.len() - 1);
    if idx == 0 {
        return dfs[0];
    }
    let t0 = times[idx - 1];
    let t1 = times[idx];
    let ln_df0 = dfs[idx - 1].ln();
    let ln_df1 = dfs[idx].ln();
    let w = if (t1 - t0).abs() > 1e-12 { (t - t0) / (t1 - t0) } else { 0.5 };
    ((1.0 - w) * ln_df0 + w * ln_df1).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ql_time::Month;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_global_bootstrap_flat_deposits() {
        let ref_date = Date::from_ymd(2025, Month::January, 15);
        let flat_rate = 0.05;
        let helpers: Vec<GlobalBootstrapHelper> = (1..=5).map(|y| {
            GlobalBootstrapHelper {
                maturity: ref_date + (y * 365),
                market_quote: flat_rate,
                weight: 1.0,
                instrument_type: GlobalHelperType::Deposit {
                    settlement_days: 0,
                    tenor_months: y as u32 * 12,
                    day_counter: DayCounter::Actual365Fixed,
                },
            }
        }).collect();

        let res = global_bootstrap(
            ref_date, &helpers, DayCounter::Actual365Fixed,
            0.0, 500, 1e-6,
        );
        // All zero rates should converge near 0.05
        for &r in &res.zero_rates {
            assert_abs_diff_eq!(r, flat_rate, epsilon = 0.005);
        }
        assert!(res.rmse < 0.01, "rmse={}", res.rmse);
    }

    #[test]
    fn test_global_bootstrap_empty() {
        let ref_date = Date::from_ymd(2025, Month::January, 15);
        let res = global_bootstrap(
            ref_date, &[], DayCounter::Actual365Fixed, 0.0, 100, 1e-6,
        );
        assert!(res.pillar_times.is_empty());
    }

    #[test]
    fn test_interpolate_df() {
        let times = vec![1.0, 2.0, 3.0];
        let dfs = vec![0.95, 0.90, 0.85];
        let df_15 = interpolate_df(&times, &dfs, 1.5);
        assert!(df_15 > 0.90 && df_15 < 0.95, "df_15={}", df_15);
    }
}
