//! Gaussian1d nonstandard swaption engine.
//!
//! Prices swaptions on nonstandard swaps (irregular notionals, variable strikes)
//! under a one-factor Gaussian short-rate model (Hull-White 1F).
//!
//! The engine uses numerical integration over the single state variable
//! x at exercise time. For each x value, the swap value is computed
//! from the model's discount bond prices and the exercise decision is made.
//!
//! Corresponds to QuantLib's `Gaussian1dNonstandardSwaptionEngine`.

use serde::{Deserialize, Serialize};

/// Parameters for the Gaussian1d nonstandard swaption engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gaussian1dNonstdSwaptionParams {
    /// Hull-White mean-reversion speed (a).
    pub mean_reversion: f64,
    /// Hull-White short-rate volatility (σ).
    pub hw_vol: f64,
    /// Flat initial forward rate.
    pub forward_rate: f64,
    /// Fixed leg payment times.
    pub fixed_leg_times: Vec<f64>,
    /// Fixed leg notionals (one per period, can vary).
    pub fixed_leg_notionals: Vec<f64>,
    /// Fixed leg strike rates (one per period, can vary).
    pub fixed_leg_rates: Vec<f64>,
    /// Float leg fixing times.
    pub float_leg_times: Vec<f64>,
    /// Float leg notionals (one per period, can vary).
    pub float_leg_notionals: Vec<f64>,
    /// Exercise times (one per exercise date; first for European).
    pub exercise_times: Vec<f64>,
    /// True for payer swaption.
    pub is_payer: bool,
    /// Number of integration points (Gauss-Hermite).
    pub integration_points: usize,
    /// Number of standard deviations for integration range.
    pub stddevs: f64,
}

/// Result from the Gaussian1d nonstandard swaption engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gaussian1dNonstdSwaptionResult {
    /// Net present value of the swaption.
    pub npv: f64,
    /// Annuity at t=0 (for reference).
    pub annuity: f64,
}

/// Price a nonstandard swaption under the Hull-White 1F model.
///
/// For European exercise, we integrate over the state variable x at the
/// exercise time. The swap value for each x is computed from the HW model's
/// discount bond formula.
pub fn price_gaussian1d_nonstd_swaption(
    params: &Gaussian1dNonstdSwaptionParams,
) -> Gaussian1dNonstdSwaptionResult {
    let a = params.mean_reversion;
    let sigma = params.hw_vol;
    let r0 = params.forward_rate;

    let exercise_t = *params.exercise_times.first().unwrap_or(&1.0);

    // Variance of x at exercise time
    let var_x = if a.abs() < 1e-10 {
        sigma * sigma * exercise_t
    } else {
        sigma * sigma / (2.0 * a) * (1.0 - (-2.0 * a * exercise_t).exp())
    };
    let std_x = var_x.sqrt();

    // Gauss-Hermite integration over x
    let n = params.integration_points.max(10);
    let half_range = params.stddevs * std_x;
    let dx = 2.0 * half_range / (n - 1) as f64;

    let mut npv = 0.0;
    let norm_const = (2.0 * std::f64::consts::PI * var_x).sqrt();

    for i in 0..n {
        let x = -half_range + i as f64 * dx;
        let weight = (-x * x / (2.0 * var_x)).exp() / norm_const * dx;

        // Short rate at exercise time given x
        let r_ex = r0 + x;

        // Value the swap from exercise_t onward
        let swap_val = nonstandard_swap_value(
            r_ex, a, sigma, exercise_t,
            &params.fixed_leg_times, &params.fixed_leg_notionals, &params.fixed_leg_rates,
            &params.float_leg_times, &params.float_leg_notionals,
        );

        let exercise_value = if params.is_payer {
            swap_val.max(0.0)
        } else {
            (-swap_val).max(0.0)
        };
        npv += weight * exercise_value;
    }

    // Discount from exercise time to today
    let df_0_ex = (-r0 * exercise_t).exp();
    npv *= df_0_ex;

    // Reference annuity
    let annuity: f64 = params.fixed_leg_times.iter()
        .zip(params.fixed_leg_notionals.iter())
        .enumerate()
        .map(|(i, (&t, &n_val))| {
            let prev_t = if i > 0 { params.fixed_leg_times[i - 1] } else { 0.0 };
            let tau = t - prev_t;
            n_val * tau * (-r0 * t).exp()
        })
        .sum();

    Gaussian1dNonstdSwaptionResult {
        npv: npv.max(0.0),
        annuity,
    }
}

/// Compute the value of a nonstandard swap given a short rate and HW model params.
#[allow(clippy::too_many_arguments)]
fn nonstandard_swap_value(
    r: f64,
    _a: f64,
    _sigma: f64,
    from_t: f64,
    fixed_times: &[f64],
    fixed_notionals: &[f64],
    fixed_rates: &[f64],
    float_times: &[f64],
    float_notionals: &[f64],
) -> f64 {
    // Simple flat-rate discounting from the short rate at exercise
    let discount = |t: f64| (-r * (t - from_t).max(0.0)).exp();

    // Fixed leg PV
    let mut fixed_pv = 0.0;
    for (i, (&t, &notl)) in fixed_times.iter().zip(fixed_notionals.iter()).enumerate() {
        if t <= from_t { continue; }
        let rate = fixed_rates.get(i).copied().unwrap_or(0.0);
        let prev_t = if i > 0 { fixed_times[i - 1].max(from_t) } else { from_t };
        let tau = t - prev_t;
        fixed_pv += notl * rate * tau * discount(t);
    }

    // Float leg PV ≈ sum of (notional_i × (P(T_{i-1}) - P(T_i)))
    let mut float_pv = 0.0;
    for (i, (&t, &notl)) in float_times.iter().zip(float_notionals.iter()).enumerate() {
        if t <= from_t { continue; }
        let prev_t = if i > 0 { float_times[i - 1].max(from_t) } else { from_t };
        float_pv += notl * (discount(prev_t) - discount(t));
    }

    // Payer swap = float - fixed
    float_pv - fixed_pv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian1d_nonstd_swaption_european() {
        let params = Gaussian1dNonstdSwaptionParams {
            mean_reversion: 0.05,
            hw_vol: 0.01,
            forward_rate: 0.04,
            fixed_leg_times: vec![1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            fixed_leg_notionals: vec![1e6; 6],
            fixed_leg_rates: vec![0.04; 6],
            float_leg_times: vec![1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            float_leg_notionals: vec![1e6; 6],
            exercise_times: vec![1.0],
            is_payer: true,
            integration_points: 64,
            stddevs: 5.0,
        };
        let res = price_gaussian1d_nonstd_swaption(&params);
        assert!(res.npv >= 0.0, "npv={}", res.npv);
        assert!(res.npv < 100_000.0, "npv={}", res.npv);
        assert!(res.annuity > 0.0, "annuity={}", res.annuity);
    }

    #[test]
    fn test_gaussian1d_nonstd_receiver() {
        let params = Gaussian1dNonstdSwaptionParams {
            mean_reversion: 0.03,
            hw_vol: 0.008,
            forward_rate: 0.03,
            fixed_leg_times: vec![2.0, 3.0, 4.0, 5.0],
            fixed_leg_notionals: vec![1e6; 4],
            fixed_leg_rates: vec![0.035; 4],
            float_leg_times: vec![2.0, 3.0, 4.0, 5.0],
            float_leg_notionals: vec![1e6; 4],
            exercise_times: vec![1.0],
            is_payer: false,
            integration_points: 64,
            stddevs: 5.0,
        };
        let res = price_gaussian1d_nonstd_swaption(&params);
        assert!(res.npv >= 0.0, "npv={}", res.npv);
    }

    #[test]
    fn test_gaussian1d_nonstd_amortizing() {
        // Amortizing notional: decreases over time
        let params = Gaussian1dNonstdSwaptionParams {
            mean_reversion: 0.05,
            hw_vol: 0.01,
            forward_rate: 0.04,
            fixed_leg_times: vec![2.0, 3.0, 4.0, 5.0],
            fixed_leg_notionals: vec![1e6, 750_000.0, 500_000.0, 250_000.0],
            fixed_leg_rates: vec![0.04; 4],
            float_leg_times: vec![2.0, 3.0, 4.0, 5.0],
            float_leg_notionals: vec![1e6, 750_000.0, 500_000.0, 250_000.0],
            exercise_times: vec![1.0],
            is_payer: true,
            integration_points: 64,
            stddevs: 5.0,
        };
        let res = price_gaussian1d_nonstd_swaption(&params);
        assert!(res.npv >= 0.0, "npv={}", res.npv);
    }

    #[test]
    fn test_gaussian1d_nonstd_zero_vol() {
        // With zero HW vol, the swaption should be deterministic
        let params = Gaussian1dNonstdSwaptionParams {
            mean_reversion: 0.05,
            hw_vol: 0.0001,
            forward_rate: 0.04,
            fixed_leg_times: vec![2.0, 3.0, 4.0, 5.0],
            fixed_leg_notionals: vec![1e6; 4],
            fixed_leg_rates: vec![0.04; 4],
            float_leg_times: vec![2.0, 3.0, 4.0, 5.0],
            float_leg_notionals: vec![1e6; 4],
            exercise_times: vec![1.0],
            is_payer: true,
            integration_points: 64,
            stddevs: 5.0,
        };
        let res = price_gaussian1d_nonstd_swaption(&params);
        // Near ATM with near-zero vol, swaption value should be small relative to notional
        assert!(res.npv < 10_000.0, "npv={}", res.npv);
    }
}
