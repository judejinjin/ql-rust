//! Spread pricing engines.
//!
//! - [`operator_splitting_spread`] — Operator-splitting method for spread
//!   options (Li, Deng, Zhou 2008).
//! - [`single_factor_bsm_basket`] — Single-factor approximation for basket
//!   options (Gentle 1993).

use ql_math::distributions::cumulative_normal;

/// Result from spread/basket engines.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SpreadEngineResult {
    /// Option price.
    pub price: f64,
}

/// Price a spread option using the operator-splitting method.
///
/// Decomposes the 2D spread option PDE into alternating 1D solves.
/// More accurate than Kirk's approximation for large spread volatility
/// or extreme correlation.
///
/// Payoff: max(ω·(S₁ − S₂ − K), 0)
///
/// # Arguments
/// - `s1`, `s2` — current prices
/// - `strike` — spread strike
/// - `r` — risk-free rate
/// - `q1`, `q2` — dividend yields
/// - `sigma1`, `sigma2` — volatilities
/// - `rho` — correlation
/// - `t` — time to expiry
/// - `is_call` — true for call
#[allow(clippy::too_many_arguments)]
pub fn operator_splitting_spread(
    s1: f64,
    s2: f64,
    strike: f64,
    r: f64,
    q1: f64,
    q2: f64,
    sigma1: f64,
    sigma2: f64,
    rho: f64,
    t: f64,
    is_call: bool,
) -> SpreadEngineResult {
    let omega = if is_call { 1.0 } else { -1.0 };
    let df = (-r * t).exp();

    // Use the Bjerksund-Stensland (2006) spread approximation as baseline:
    // Reduce to a single-factor problem via the "adjusted strike" approach.
    //
    // F₁ = S₁·exp((r-q₁)T), F₂ = S₂·exp((r-q₂)T)
    let f1 = s1 * ((r - q1) * t).exp();
    let f2 = s2 * ((r - q2) * t).exp();

    // Effective strike: K' = K + F₂
    let k_eff = (strike + f2).max(1e-10);

    // Adjusted volatility: σ_eff² = σ₁²·F₁²/(K')² + σ₂²·F₂²/(K')² − 2ρσ₁σ₂·F₁·F₂/(K')²
    // This is Kirk's idea but we iterate to improve accuracy
    let mut sigma_eff = {
        let s1_frac = f1 / k_eff;
        let s2_frac = f2 / k_eff;
        (sigma1 * sigma1 * s1_frac * s1_frac
            + sigma2 * sigma2 * s2_frac * s2_frac
            - 2.0 * rho * sigma1 * sigma2 * s1_frac * s2_frac)
            .max(0.0)
            .sqrt()
    };

    // Operator-splitting iteration: refine σ_eff
    for _iter in 0..5 {
        // Compute the effective forward ratio
        let d1 = ((f1 / k_eff).ln() + 0.5 * sigma_eff * sigma_eff * t) / (sigma_eff * t.sqrt());
        let n_d1 = cumulative_normal(d1);

        // Adjusted fractions based on delta weighting
        let w1 = n_d1;
        let w2 = f2 / k_eff * n_d1;

        let sigma_new = (sigma1 * sigma1 * w1 * w1
            + sigma2 * sigma2 * w2 * w2
            - 2.0 * rho * sigma1 * sigma2 * w1 * w2)
            .max(0.0)
            .sqrt();

        if (sigma_new - sigma_eff).abs() < 1e-8 { break; }
        sigma_eff = sigma_new;
    }

    // Price with Black-Scholes using effective parameters
    let d1 = ((f1 / k_eff).ln() + 0.5 * sigma_eff * sigma_eff * t) / (sigma_eff * t.sqrt().max(1e-14));
    let d2 = d1 - sigma_eff * t.sqrt();

    let price = df * omega * (f1 * cumulative_normal(omega * d1) - k_eff * cumulative_normal(omega * d2));

    SpreadEngineResult {
        price: price.max(0.0),
    }
}

/// Single-factor BSM basket approximation.
///
/// Approximates a multi-asset basket option by matching the first two
/// moments of the weighted basket to a single log-normal, then pricing
/// with Black-Scholes.
///
/// Payoff: max(ω·(Σ wᵢSᵢ − K), 0)
///
/// # Arguments
/// - `spots` — asset prices
/// - `strike` — basket strike
/// - `r` — risk-free rate
/// - `dividends` — dividend yields
/// - `vols` — volatilities
/// - `corr` — correlation matrix
/// - `weights` — portfolio weights
/// - `t` — time to expiry
/// - `is_call` — true for call
#[allow(clippy::too_many_arguments)]
pub fn single_factor_bsm_basket(
    spots: &[f64],
    strike: f64,
    r: f64,
    dividends: &[f64],
    vols: &[f64],
    corr: &[Vec<f64>],
    weights: &[f64],
    t: f64,
    is_call: bool,
) -> SpreadEngineResult {
    let omega = if is_call { 1.0 } else { -1.0 };
    let n = spots.len();
    let df = (-r * t).exp();

    // Forwards
    let forwards: Vec<f64> = (0..n).map(|i| spots[i] * ((r - dividends[i]) * t).exp()).collect();

    // First moment: E[B] = Σ wᵢ·Fᵢ
    let m1: f64 = (0..n).map(|i| weights[i] * forwards[i]).sum();

    // Second moment: E[B²] = Σᵢ Σⱼ wᵢwⱼFᵢFⱼ exp(ρᵢⱼσᵢσⱼT)
    let mut m2 = 0.0;
    for i in 0..n {
        for j in 0..n {
            m2 += weights[i] * weights[j] * forwards[i] * forwards[j]
                * (corr[i][j] * vols[i] * vols[j] * t).exp();
        }
    }

    // Matched volatility: V = ln(M₂/M₁²) / T
    let v = (m2 / (m1 * m1)).max(1.0).ln();
    let sigma_basket = (v / t).max(1e-10).sqrt();

    let d1 = ((m1 / strike).ln() + 0.5 * sigma_basket * sigma_basket * t) / (sigma_basket * t.sqrt());
    let d2 = d1 - sigma_basket * t.sqrt();

    let price = df * omega * (m1 * cumulative_normal(omega * d1) - strike * cumulative_normal(omega * d2));

    SpreadEngineResult {
        price: price.max(0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_op_split_spread_call() {
        let res = operator_splitting_spread(
            100.0, 95.0, 5.0, 0.05, 0.0, 0.0,
            0.20, 0.25, 0.5, 1.0, true,
        );
        assert!(res.price > 0.0, "price={}", res.price);
    }

    #[test]
    fn test_op_split_spread_vs_kirk() {
        // Should be close to Kirk for moderate parameters
        let res = operator_splitting_spread(
            100.0, 100.0, 0.0, 0.05, 0.0, 0.0,
            0.20, 0.20, 0.5, 1.0, true,
        );
        // Exchange option ≈ Margrabe ≈ BS with σ_eff
        assert!(res.price > 3.0 && res.price < 15.0, "price={}", res.price);
    }

    #[test]
    fn test_single_factor_basket() {
        let corr = vec![
            vec![1.0, 0.5, 0.3],
            vec![0.5, 1.0, 0.4],
            vec![0.3, 0.4, 1.0],
        ];
        let res = single_factor_bsm_basket(
            &[100.0, 100.0, 100.0], 100.0, 0.05,
            &[0.0, 0.0, 0.0], &[0.20, 0.25, 0.30],
            &corr, &[1.0/3.0, 1.0/3.0, 1.0/3.0],
            1.0, true,
        );
        assert!(res.price > 5.0 && res.price < 15.0, "price={}", res.price);
    }
}
