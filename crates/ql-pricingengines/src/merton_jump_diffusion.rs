//! Merton (1976) jump-diffusion pricing engine.
//!
//! Prices European options under the Merton jump-diffusion model:
//!
//!   dS/S = (r − q − λk̄) dt + σ dW + J dN
//!
//! where N ~ Poisson(λ) and log(1+J) ~ N(ν, δ²).
//!
//! The analytic formula decomposes the price as a weighted sum of
//! Black-Scholes prices:
//!
//!   V = Σ_{n=0}^{∞} P(N=n) · BS(S, K, r_n, σ_n, T)
//!
//! where:
//!   P(N=n) = e^{−λ'T} (λ'T)^n / n!
//!   λ' = λ(1 + k̄)
//!   r_n = r − λk̄ + n·ln(1+k̄)/T
//!   σ_n² = σ² + n·δ²/T
//!
//! # References
//! - Merton, R.C. (1976), "Option pricing when underlying stock returns
//!   are discontinuous", *Journal of Financial Economics* 3.

use ql_math::distributions::NormalDistribution;

/// Results from the Merton jump-diffusion engine.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MertonJDResult {
    /// Net present value.
    pub npv: f64,
    /// Number of terms used in the series.
    pub num_terms: usize,
}

/// Price a European option under the Merton jump-diffusion model.
///
/// # Parameters
/// - `spot`: current spot price
/// - `strike`: option strike price
/// - `r`: risk-free rate (continuous)
/// - `q`: dividend yield (continuous)
/// - `vol`: diffusion volatility (σ)
/// - `time_to_expiry`: time to maturity in years
/// - `lambda`: jump intensity (mean jumps per year)
/// - `nu`: mean of log-jump size
/// - `delta`: standard deviation of log-jump size
/// - `is_call`: true for call, false for put
///
/// # Returns
/// Merton price computed via the series expansion (truncated when terms
/// contribute less than 1e-15 to the price).
#[allow(clippy::too_many_arguments)]
pub fn merton_jump_diffusion(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    vol: f64,
    time_to_expiry: f64,
    lambda: f64,
    nu: f64,
    delta: f64,
    is_call: bool,
) -> MertonJDResult {
    let t = time_to_expiry;

    if t <= 0.0 {
        let omega = if is_call { 1.0 } else { -1.0 };
        let intrinsic = (omega * (spot - strike)).max(0.0);
        return MertonJDResult {
            npv: intrinsic,
            num_terms: 0,
        };
    }

    // Jump compensator: k_bar = E[e^J - 1] = exp(nu + delta^2/2) - 1
    let k_bar = (nu + 0.5 * delta * delta).exp() - 1.0;

    // Modified jump intensity: lambda' = lambda * (1 + k_bar)
    let lambda_prime = lambda * (1.0 + k_bar);

    let n = NormalDistribution::standard();
    let mut price = 0.0;
    let mut log_poisson_pmf = -lambda_prime * t; // log(P(N=0))
    let max_terms = 200;

    for n_jumps in 0..max_terms {
        // Poisson weight: P(N=n) = exp(-λ'T) (λ'T)^n / n!
        let w = log_poisson_pmf.exp();

        // Modified parameters for the nth BS term
        let sigma_n_sq = vol * vol + (n_jumps as f64) * delta * delta / t;
        let sigma_n = sigma_n_sq.sqrt();
        let r_n = r - lambda * k_bar + (n_jumps as f64) * (1.0 + k_bar).ln() / t;

        // Black-Scholes with modified parameters
        let bs = black_scholes_core(spot, strike, r_n, q, sigma_n, t, is_call, &n);

        price += w * bs;

        // Update log-Poisson PMF for next term
        log_poisson_pmf += (lambda_prime * t).ln() - ((n_jumps + 1) as f64).ln();

        // Convergence check
        if n_jumps > 5 && w * bs.abs() < 1e-15 {
            return MertonJDResult {
                npv: price.max(0.0),
                num_terms: n_jumps + 1,
            };
        }
    }

    MertonJDResult {
        npv: price.max(0.0),
        num_terms: max_terms,
    }
}

/// Core Black-Scholes price computation for the Merton series.
#[allow(clippy::too_many_arguments)]
fn black_scholes_core(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    is_call: bool,
    n: &NormalDistribution,
) -> f64 {
    let omega = if is_call { 1.0 } else { -1.0 };
    let sqrt_t = t.sqrt();
    let d1 = ((spot / strike).ln() + (r - q + 0.5 * vol * vol) * t) / (vol * sqrt_t);
    let d2 = d1 - vol * sqrt_t;
    omega
        * (spot * (-q * t).exp() * n.cdf(omega * d1)
            - strike * (-r * t).exp() * n.cdf(omega * d2))
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn merton_reduces_to_bs_when_no_jumps() {
        // With lambda=0, Merton should equal Black-Scholes
        let merton = merton_jump_diffusion(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            0.0, 0.0, 0.01, // no jumps
            true,
        );
        let n = NormalDistribution::standard();
        let bs = black_scholes_core(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true, &n);
        assert_abs_diff_eq!(merton.npv, bs, epsilon = 1e-8);
    }

    #[test]
    fn merton_put_call_parity() {
        let s = 100.0;
        let k = 100.0;
        let r = 0.05;
        let q = 0.02;
        let t = 1.0;

        let call = merton_jump_diffusion(s, k, r, q, 0.20, t, 0.5, -0.1, 0.15, true).npv;
        let put = merton_jump_diffusion(s, k, r, q, 0.20, t, 0.5, -0.1, 0.15, false).npv;

        let parity = call - put - s * (-q * t).exp() + k * (-r * t).exp();
        assert_abs_diff_eq!(parity, 0.0, epsilon = 1e-8);
    }

    #[test]
    fn merton_negative_jumps_increase_put() {
        // Negative jumps should increase put prices
        let n = NormalDistribution::standard();
        let bs_put = black_scholes_core(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, false, &n);
        let merton_put = merton_jump_diffusion(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            0.5, -0.1, 0.15, // negative jumps
            false,
        ).npv;

        assert!(
            merton_put > bs_put,
            "Merton put {} should exceed BS put {} with negative jumps",
            merton_put,
            bs_put
        );
    }

    #[test]
    fn merton_positive_jumps_increase_call() {
        // Positive jumps should increase call prices
        let n = NormalDistribution::standard();
        let bs_call = black_scholes_core(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true, &n);
        let merton_call = merton_jump_diffusion(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            0.5, 0.1, 0.15, // positive jumps
            true,
        ).npv;

        assert!(
            merton_call > bs_call,
            "Merton call {} should exceed BS call {} with positive jumps",
            merton_call,
            bs_call
        );
    }

    #[test]
    fn merton_convergence() {
        // Should converge in a small number of terms
        let res = merton_jump_diffusion(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            0.5, -0.1, 0.15,
            true,
        );
        assert!(
            res.num_terms < 50,
            "Should converge in <50 terms, got {}",
            res.num_terms
        );
    }

    #[test]
    fn merton_expired_option() {
        let res = merton_jump_diffusion(100.0, 90.0, 0.05, 0.0, 0.20, 0.0, 0.5, -0.1, 0.15, true);
        assert_abs_diff_eq!(res.npv, 10.0, epsilon = 1e-10);
    }

    #[test]
    fn merton_deep_otm_put_small() {
        let res = merton_jump_diffusion(100.0, 50.0, 0.05, 0.0, 0.20, 1.0, 0.5, -0.1, 0.15, false);
        assert!(res.npv < 1.0, "Deep OTM put should be small: {}", res.npv);
    }

    #[test]
    fn merton_high_jump_intensity() {
        // High jump intensity should not cause numerical issues
        let res = merton_jump_diffusion(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            5.0, -0.02, 0.05, // many small jumps
            true,
        );
        assert!(res.npv > 0.0 && res.npv < 200.0, "Price should be reasonable: {}", res.npv);
    }

    #[test]
    fn merton_monotone_in_lambda() {
        // With symmetric jumps (nu=0), more jumps = more vol = higher prices
        let p1 = merton_jump_diffusion(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 0.1, 0.0, 0.2, true).npv;
        let p2 = merton_jump_diffusion(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 1.0, 0.0, 0.2, true).npv;
        assert!(
            p2 > p1,
            "Higher jump intensity should give higher price: {} vs {}",
            p2,
            p1
        );
    }
}
