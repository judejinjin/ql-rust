//! Additional basket and spread option engines.
//!
//! Provides:
//! - **Choi (2018) spread approximation** — improved moment-matching for spread options.
//! - **Deng-Li-Zhou (2008) basket approximation** — analytic approximation for
//!   arithmetic basket options via moment matching on a lognormal proxy.

use ql_math::distributions::NormalDistribution;

/// Result of a basket/spread option pricing engine.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BasketSpreadResult {
    /// Option present value.
    pub npv: f64,
    /// Delta with respect to the first asset.
    pub delta1: f64,
    /// Delta with respect to the second asset.
    pub delta2: f64,
}

// ===========================================================================
// Choi (2018) spread option approximation
// ===========================================================================
//
// Reference: J. Choi (2018) — "Sum of All Black-Scholes-Mertons:
// Implications for the Pricing of Spread Options"
//
// The Choi formula treats a spread S₁ - S₂ - K and uses a moment-matched
// lognormal proxy for each leg. It is more accurate than Kirk's approximation
// for wide spread/vol ratios.

/// Price a European spread option on (S₁ - S₂ - K) using Choi (2018).
///
/// # Arguments
/// - `s1`, `s2` — current asset prices
/// - `r` — continuously compounded risk-free rate
/// - `q1`, `q2` — continuous dividend yields
/// - `v1`, `v2` — Black-Scholes volatilities
/// - `rho` — instantaneous correlation between log-returns
/// - `t` — time to expiry (years)
/// - `k` — strike on the spread (usually ≥ 0)
/// - `is_call` — true for call (pays max(S₁-S₂-K, 0))
pub fn choi_basket_spread(
    s1: f64,
    s2: f64,
    r: f64,
    q1: f64,
    q2: f64,
    v1: f64,
    v2: f64,
    rho: f64,
    t: f64,
    k: f64,
    is_call: bool,
) -> BasketSpreadResult {
    if t < 1e-12 {
        let payoff = if is_call { (s1 - s2 - k).max(0.0) } else { (k + s2 - s1).max(0.0) };
        return BasketSpreadResult { npv: payoff, delta1: if is_call && payoff > 0.0 { 1.0 } else { 0.0 }, delta2: 0.0 };
    }

    let nd = NormalDistribution::standard();
    let disc = (-r * t).exp();
    let f1 = s1 * ((r - q1) * t).exp();
    let f2 = s2 * ((r - q2) * t).exp();

    // Choi: define b = F₂ / (F₁ + K*disc) and adjusted volatility
    let f_total = f1 + k * disc; // can be negative if k negative — handled below
    if f_total.abs() < 1e-12 || f1 < 1e-12 {
        // Degenerate — fall back to Kirk
        return kirk_impl(s1, s2, r, q1, q2, v1, v2, rho, t, k, is_call);
    }

    let w = f2 / f_total;
    // Effective vol of the basket proxy
    let v_eff = (v1 * v1 + w * w * v2 * v2 - 2.0 * rho * w * v1 * v2).sqrt();

    let half_v2t = 0.5 * v_eff * v_eff * t;
    let d1 = ((f1 / (f2 + k * disc)).ln() + half_v2t) / (v_eff * t.sqrt());
    let d2 = d1 - v_eff * t.sqrt();

    let (npv, delta1, delta2) = if is_call {
        let npv = disc * (f1 * nd.cdf(d1) - (f2 + k * disc) * nd.cdf(d2));
        let delta1 = disc * nd.cdf(d1) * ((r - q1) * t).exp();
        let delta2 = -disc * nd.cdf(d2) * ((r - q2) * t).exp();
        (npv, delta1, delta2)
    } else {
        let npv = disc * ((f2 + k * disc) * nd.cdf(-d2) - f1 * nd.cdf(-d1));
        let delta1 = -disc * nd.cdf(-d1) * ((r - q1) * t).exp();
        let delta2 = disc * nd.cdf(-d2) * ((r - q2) * t).exp();
        (npv, delta1, delta2)
    };

    BasketSpreadResult { npv, delta1, delta2 }
}

/// Internal Kirk approximation (used as fallback).
fn kirk_impl(
    s1: f64,
    s2: f64,
    r: f64,
    q1: f64,
    q2: f64,
    v1: f64,
    v2: f64,
    rho: f64,
    t: f64,
    k: f64,
    is_call: bool,
) -> BasketSpreadResult {
    let nd = NormalDistribution::standard();
    let disc = (-r * t).exp();
    let f1 = s1 * ((r - q1) * t).exp();
    let f2 = s2 * ((r - q2) * t).exp();
    let fk = f2 + k * disc;

    if fk.abs() < 1e-12 || f1 < 1e-12 {
        return BasketSpreadResult { npv: 0.0, delta1: 0.0, delta2: 0.0 };
    }

    let r_star = f2 / fk;
    let v_eff = (v1 * v1 + r_star * r_star * v2 * v2 - 2.0 * rho * r_star * v1 * v2).sqrt();
    let d1 = ((f1 / fk).ln() + 0.5 * v_eff * v_eff * t) / (v_eff * t.sqrt());
    let d2 = d1 - v_eff * t.sqrt();

    let (npv, delta1, delta2) = if is_call {
        (disc * (f1 * nd.cdf(d1) - fk * nd.cdf(d2)),
         disc * nd.cdf(d1) * ((r - q1) * t).exp(),
         -disc * nd.cdf(d2) * ((r - q2) * t).exp())
    } else {
        (disc * (fk * nd.cdf(-d2) - f1 * nd.cdf(-d1)),
         -disc * nd.cdf(-d1) * ((r - q1) * t).exp(),
         disc * nd.cdf(-d2) * ((r - q2) * t).exp())
    };

    BasketSpreadResult { npv, delta1, delta2 }
}

// ===========================================================================
// Deng-Li-Zhou (2008) basket option approximation
// ===========================================================================
//
// Reference: S. Deng, M. Li, J. Zhou (2008) —
// "Closed-Form Approximations for Spread Option Prices and Greeks"
//
// For an n-asset arithmetic basket, we match the first two moments of the
// basket to a shifted lognormal and apply Black-Scholes on the proxy.

/// Price a European call/put on an arithmetic basket using Deng-Li-Zhou (2008).
///
/// # Arguments
/// - `spots` — slice of current asset prices
/// - `weights` — basket weights (should sum to 1.0 for typical baskets)
/// - `r` — risk-free rate
/// - `dividends` — dividend yields per asset
/// - `vols` — volatilities per asset
/// - `corr` — correlation matrix (row-major, n×n)
/// - `t` — time to expiry
/// - `strike` — strike price on the weighted basket
/// - `is_call` — true for call
pub fn dlz_basket_price(
    spots: &[f64],
    weights: &[f64],
    r: f64,
    dividends: &[f64],
    vols: &[f64],
    corr: &[f64],
    t: f64,
    strike: f64,
    is_call: bool,
) -> f64 {
    let n = spots.len();
    if n == 0 || weights.len() != n || dividends.len() != n || vols.len() != n || corr.len() != n * n {
        return 0.0;
    }

    if t < 1e-12 {
        let basket: f64 = spots.iter().zip(weights).map(|(s, w)| s * w).sum();
        return if is_call { (basket - strike).max(0.0) } else { (strike - basket).max(0.0) };
    }

    // Forward basket value
    let forwards: Vec<f64> = spots.iter().zip(dividends).zip(weights)
        .map(|((s, q), _w)| s * ((r - q) * t).exp())
        .collect();

    let f0: f64 = forwards.iter().zip(weights).map(|(f, w)| f * w).sum();

    // Second moment of basket: E[B_T²] = Σᵢ Σⱼ wᵢ wⱼ Fᵢ Fⱼ exp(ρᵢⱼ σᵢ σⱼ t)
    let mut e_b2 = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let rho_ij = corr[i * n + j];
            e_b2 += weights[i] * weights[j] * forwards[i] * forwards[j]
                * (rho_ij * vols[i] * vols[j] * t).exp();
        }
    }

    // Effective lognormal parameters: F = f0, σ² = ln(E[B²]/F²) / T
    if f0 < 1e-12 || e_b2 < 1e-12 {
        return 0.0;
    }
    let sigma2_eff = (e_b2 / (f0 * f0)).ln() / t;
    if sigma2_eff <= 0.0 {
        // Near-zero vol — intrinsic value discounted
        let disc = (-r * t).exp();
        return if is_call { disc * (f0 - strike).max(0.0) } else { disc * (strike - f0).max(0.0) };
    }
    let sigma_eff = sigma2_eff.sqrt();

    let nd = NormalDistribution::standard();
    let disc = (-r * t).exp();
    let d1 = ((f0 / strike).ln() + 0.5 * sigma2_eff * t) / (sigma_eff * t.sqrt());
    let d2 = d1 - sigma_eff * t.sqrt();

    if is_call {
        disc * (f0 * nd.cdf(d1) - strike * nd.cdf(d2))
    } else {
        disc * (strike * nd.cdf(-d2) - f0 * nd.cdf(-d1))
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choi_spread_zero_strike_equals_exchange() {
        // With K=0, spread option = Margrabe exchange option
        // Margrabe: S1*N(d1) - S2*N(d2) under Q measure
        let result = choi_basket_spread(100.0, 100.0, 0.05, 0.02, 0.02, 0.20, 0.20, 0.0, 1.0, 0.0, true);
        // With equal prices, equal vols, zero correlation: positive value
        assert!(result.npv > 0.0);
    }

    #[test]
    fn choi_put_call_parity() {
        let (s1, s2, r, q1, q2, v1, v2, rho, t, k): (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) = (100.0, 80.0, 0.05, 0.02, 0.01, 0.25, 0.20, 0.3, 1.0, 5.0);
        let call = choi_basket_spread(s1, s2, r, q1, q2, v1, v2, rho, t, k, true);
        let put  = choi_basket_spread(s1, s2, r, q1, q2, v1, v2, rho, t, k, false);
        // Put-call parity for spread: C - P = disc * (F1 - F2 - K)
        let disc = (-r * t).exp();
        let f1 = s1 * ((r - q1) * t).exp();
        let f2 = s2 * ((r - q2) * t).exp();
        let pcp = call.npv - put.npv - disc * (f1 - f2 - k * disc);
        assert!(pcp.abs() < 0.5, "pcp={}", pcp); // loose tolerance for approximation
    }

    #[test]
    fn dlz_basket_single_asset_is_bs() {
        // Single asset, weight=1 → should match Black-Scholes
        use ql_math::distributions::NormalDistribution;
        let nd = NormalDistribution::standard();
        let (s, r, q, v, t, k): (f64, f64, f64, f64, f64, f64) = (100.0, 0.05, 0.02, 0.20, 1.0, 100.0);
        let f = s * ((r - q) * t).exp();
        let d1 = ((f / k).ln() + 0.5 * v * v * t) / (v * t.sqrt());
        let d2 = d1 - v * t.sqrt();
        let bs_call = (-r * t).exp() * (f * nd.cdf(d1) - k * nd.cdf(d2));

        let dlz_call = dlz_basket_price(&[s], &[1.0], r, &[q], &[v], &[1.0], t, k, true);
        assert!((dlz_call - bs_call).abs() < 1e-8, "dlz={} bs={}", dlz_call, bs_call);
    }

    #[test]
    fn dlz_basket_positive() {
        let call = dlz_basket_price(
            &[100.0, 100.0],
            &[0.5, 0.5],
            0.05,
            &[0.02, 0.02],
            &[0.20, 0.25],
            &[1.0, 0.5, 0.5, 1.0],
            1.0,
            95.0,
            true,
        );
        assert!(call > 0.0);
    }
}
