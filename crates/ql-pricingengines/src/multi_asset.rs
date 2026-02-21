//! Multi-asset and basket option pricing.
//!
//! Provides:
//! - **MC basket engine**: Monte Carlo for baskets of correlated assets.
//! - **Stulz formula**: Analytic 2-asset European max/min option.
//! - **Kirk spread approximation**: Analytic approximation for spread options.
//! - **Margrabe formula**: Exchange option (closed-form).
//!
//! All engines use GBM dynamics with correlation via Cholesky decomposition.

use ql_math::distributions::NormalDistribution;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ===========================================================================
// Basket payoff types
// ===========================================================================

/// Basket aggregation method.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum BasketType {
    /// max(S₁, S₂, ..., Sₙ)
    MaxBasket,
    /// min(S₁, S₂, ..., Sₙ)
    MinBasket,
    /// Σ wᵢ Sᵢ
    WeightedAverage,
}

// ===========================================================================
// MC Basket Engine
// ===========================================================================

/// Results from a basket option simulation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BasketResult {
    /// Net present value.
    pub npv: f64,
    /// Standard error of the MC estimate.
    pub std_error: f64,
    /// Number of paths simulated.
    pub num_paths: usize,
}

/// Price a European basket option via Monte Carlo.
///
/// # Parameters
/// - `spots`: initial prices for each asset
/// - `weights`: basket weights (for WeightedAverage) — length must match `spots`
/// - `strike`: basket strike price
/// - `r`: risk-free rate
/// - `dividends`: dividend yields per asset
/// - `vols`: volatilities per asset
/// - `corr_matrix`: n×n correlation matrix (row-major, flat)
/// - `time_to_expiry`: time to maturity
/// - `is_call`: true for call, false for put
/// - `basket_type`: how to aggregate terminal asset values
/// - `num_paths`: number of MC paths
/// - `seed`: RNG seed
#[allow(clippy::too_many_arguments)]
pub fn mc_basket(
    spots: &[f64],
    weights: &[f64],
    strike: f64,
    r: f64,
    dividends: &[f64],
    vols: &[f64],
    corr_matrix: &[f64],
    time_to_expiry: f64,
    is_call: bool,
    basket_type: BasketType,
    num_paths: usize,
    seed: u64,
) -> BasketResult {
    let n = spots.len();
    assert_eq!(weights.len(), n);
    assert_eq!(dividends.len(), n);
    assert_eq!(vols.len(), n);
    assert_eq!(corr_matrix.len(), n * n);

    let t = time_to_expiry;
    let df = (-r * t).exp();
    let omega = if is_call { 1.0 } else { -1.0 };

    // Cholesky decomposition of correlation matrix
    let chol = cholesky_lower(n, corr_matrix);

    // Pre-compute drift for each asset: (r - q - σ²/2) * T
    let drifts: Vec<f64> = (0..n)
        .map(|i| (r - dividends[i] - 0.5 * vols[i] * vols[i]) * t)
        .collect();

    let batch_size = 5000_usize;
    let num_batches = num_paths.div_ceil(batch_size);

    #[cfg(feature = "parallel")]
    let iter = (0..num_batches).into_par_iter();
    #[cfg(not(feature = "parallel"))]
    let iter = 0..num_batches;

    let results: Vec<(f64, f64, usize)> = iter
        .map(|batch_idx| {
            let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(batch_idx as u64));
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(num_paths);
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            let count = end - start;

            for _ in start..end {
                // Generate n independent normals
                let z_indep: Vec<f64> = (0..n)
                    .map(|_| StandardNormal.sample(&mut rng))
                    .collect();

                // Apply Cholesky: z_corr = L * z_indep
                let mut terminal_values = Vec::with_capacity(n);
                for i in 0..n {
                    let mut z_corr = 0.0;
                    for j in 0..=i {
                        z_corr += chol[i * n + j] * z_indep[j];
                    }
                    let s_t = spots[i] * (drifts[i] + vols[i] * t.sqrt() * z_corr).exp();
                    terminal_values.push(s_t);
                }

                // Compute basket value
                let basket_val = match basket_type {
                    BasketType::MaxBasket => terminal_values
                        .iter()
                        .copied()
                        .fold(f64::NEG_INFINITY, f64::max),
                    BasketType::MinBasket => terminal_values
                        .iter()
                        .copied()
                        .fold(f64::INFINITY, f64::min),
                    BasketType::WeightedAverage => terminal_values
                        .iter()
                        .zip(weights.iter())
                        .map(|(s, w)| s * w)
                        .sum(),
                };

                let payoff = (omega * (basket_val - strike)).max(0.0);
                sum += payoff;
                sum_sq += payoff * payoff;
            }
            (sum, sum_sq, count)
        })
        .collect();

    let total_sum: f64 = results.iter().map(|(s, _, _)| s).sum();
    let total_sum_sq: f64 = results.iter().map(|(_, s, _)| s).sum();
    let total_count: usize = results.iter().map(|(_, _, c)| c).sum();
    let np = total_count as f64;

    let mean = total_sum / np;
    let variance = (total_sum_sq / np - mean * mean).max(0.0);
    let std_error = (variance / np).sqrt() * df;

    BasketResult {
        npv: df * mean,
        std_error,
        num_paths: total_count,
    }
}

// ===========================================================================
// Stulz formula (2-asset max/min European option)
// ===========================================================================

/// Analytic Stulz formula for European call on min(S₁, S₂) with strike K.
///
/// payoff = max(min(S₁, S₂) − K, 0)
///
/// C_min = S₁ e^{-q₁T} M₂(d₁, −y; −ρ₁)
///       + S₂ e^{-q₂T} M₂(d₂, y−σ√T; −ρ₂)
///       − K e^{-rT} M₂(d₁−σ₁√T, d₂−σ₂√T; ρ)
///
/// # References
/// - Stulz, R. (1982), "Options on the Minimum or the Maximum of Two Risky
///   Assets", *Journal of Financial Economics* 10, eq. 7.
/// - Haug, E.G. (2007), *The Complete Guide to Option Pricing Formulas*, ch 6.
#[allow(clippy::too_many_arguments)]
pub fn stulz_min_call(
    s1: f64,
    s2: f64,
    strike: f64,
    r: f64,
    q1: f64,
    q2: f64,
    vol1: f64,
    vol2: f64,
    rho: f64,
    time_to_expiry: f64,
) -> f64 {
    let t = time_to_expiry;
    let sqrt_t = t.sqrt();
    let n = NormalDistribution::standard();

    // σ = volatility of ln(S₁/S₂)
    let sigma = (vol1 * vol1 + vol2 * vol2 - 2.0 * rho * vol1 * vol2).sqrt();

    // BS d1-type for each asset vs strike
    let d1 = ((s1 / strike).ln() + (r - q1 + 0.5 * vol1 * vol1) * t) / (vol1 * sqrt_t);
    let d2 = ((s2 / strike).ln() + (r - q2 + 0.5 * vol2 * vol2) * t) / (vol2 * sqrt_t);

    // y = [ln(S₁/S₂) + (q₂ − q₁ + σ²/2)T] / (σ√T)
    let y = ((s1 / s2).ln() + (q2 - q1 + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t);

    let rho1 = (vol1 - rho * vol2) / sigma;
    let rho2 = (vol2 - rho * vol1) / sigma;

    let df = (-r * t).exp();

    let term1 = s1 * (-q1 * t).exp() * bivariate_normal_cdf(d1, -y, -rho1, &n);
    let term2 = s2 * (-q2 * t).exp()
        * bivariate_normal_cdf(d2, y - sigma * sqrt_t, -rho2, &n);
    let term3 = strike * df * bivariate_normal_cdf(d1 - vol1 * sqrt_t, d2 - vol2 * sqrt_t, rho, &n);

    (term1 + term2 - term3).max(0.0)
}

/// Stulz formula for call on max(S₁, S₂) via min-max parity:
///   C_max(K) = C_BS(S₁, K) + C_BS(S₂, K) − C_min(K)
#[allow(clippy::too_many_arguments)]
pub fn stulz_max_call(
    s1: f64,
    s2: f64,
    strike: f64,
    r: f64,
    q1: f64,
    q2: f64,
    vol1: f64,
    vol2: f64,
    rho: f64,
    time_to_expiry: f64,
) -> f64 {
    let n = NormalDistribution::standard();
    let c1 = bs_call(s1, strike, r, q1, vol1, time_to_expiry, &n);
    let c2 = bs_call(s2, strike, r, q2, vol2, time_to_expiry, &n);
    let c_min = stulz_min_call(s1, s2, strike, r, q1, q2, vol1, vol2, rho, time_to_expiry);
    (c1 + c2 - c_min).max(0.0)
}

// ===========================================================================
// Kirk Spread Approximation
// ===========================================================================

/// Kirk's approximation for a European call on the spread S₁ − S₂.
///
/// payoff = max(S₁ − S₂ − K, 0)
///
/// Kirk treats the spread as a single asset and adjusts the volatility:
///   σ_spread = sqrt(σ₁² − 2ρσ₁σ₂(S₂/(S₂+K)) + σ₂²(S₂/(S₂+K))²)
///
/// # References
/// - Kirk, E. (1995), "Correlation in the Energy Markets", *Managing
///   Energy Price Risk*.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn kirk_spread_call(
    s1: f64,
    s2: f64,
    strike: f64,
    r: f64,
    q1: f64,
    q2: f64,
    vol1: f64,
    vol2: f64,
    rho: f64,
    time_to_expiry: f64,
) -> f64 {
    let t = time_to_expiry;
    let n = NormalDistribution::standard();

    let f1 = s1 * ((r - q1) * t).exp();
    let f2 = s2 * ((r - q2) * t).exp();

    let frac = f2 / (f2 + strike);

    let sigma_spread =
        (vol1 * vol1 - 2.0 * rho * vol1 * vol2 * frac + vol2 * vol2 * frac * frac).sqrt();

    let d1 = ((f1 / (f2 + strike)).ln() + 0.5 * sigma_spread * sigma_spread * t)
        / (sigma_spread * t.sqrt());
    let d2 = d1 - sigma_spread * t.sqrt();

    let df = (-r * t).exp();
    let price = df * (f1 * n.cdf(d1) - (f2 + strike) * n.cdf(d2));
    price.max(0.0)
}

/// Kirk's approximation for a European put on the spread S₁ − S₂.
/// Uses put-call parity on the spread.
#[allow(clippy::too_many_arguments)]
pub fn kirk_spread_put(
    s1: f64,
    s2: f64,
    strike: f64,
    r: f64,
    q1: f64,
    q2: f64,
    vol1: f64,
    vol2: f64,
    rho: f64,
    time_to_expiry: f64,
) -> f64 {
    let t = time_to_expiry;
    let df = (-r * t).exp();
    let f1 = s1 * ((r - q1) * t).exp();
    let f2 = s2 * ((r - q2) * t).exp();

    let call = kirk_spread_call(s1, s2, strike, r, q1, q2, vol1, vol2, rho, t);
    let put = call - df * (f1 - f2 - strike);
    put.max(0.0)
}

// ===========================================================================
// Margrabe Formula (Exchange Option)
// ===========================================================================

/// Margrabe formula for European exchange option: right to exchange S₂ for S₁.
///
/// payoff = max(S₁ − S₂, 0)
///
/// This is equivalent to a spread call with K=0.
///
/// # References
/// - Margrabe, W. (1978), "The Value of an Option to Exchange One Asset for
///   Another", *Journal of Finance* 33.
#[allow(clippy::too_many_arguments)]
pub fn margrabe_exchange(
    s1: f64,
    s2: f64,
    q1: f64,
    q2: f64,
    vol1: f64,
    vol2: f64,
    rho: f64,
    time_to_expiry: f64,
) -> f64 {
    let t = time_to_expiry;
    let n = NormalDistribution::standard();

    let sigma = (vol1 * vol1 + vol2 * vol2 - 2.0 * rho * vol1 * vol2).sqrt();
    let sqrt_t = t.sqrt();

    let d1 = ((s1 / s2).ln() + (q2 - q1 + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;

    let price = s1 * (-q1 * t).exp() * n.cdf(d1) - s2 * (-q2 * t).exp() * n.cdf(d2);
    price.max(0.0)
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Black-Scholes call price (helper for Stulz).
fn bs_call(s: f64, k: f64, r: f64, q: f64, vol: f64, t: f64, n: &NormalDistribution) -> f64 {
    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + (r - q + 0.5 * vol * vol) * t) / (vol * sqrt_t);
    let d2 = d1 - vol * sqrt_t;
    s * (-q * t).exp() * n.cdf(d1) - k * (-r * t).exp() * n.cdf(d2)
}

/// Bivariate normal CDF using Genz (2004) algorithm.
///
/// Φ₂(x, y; ρ) = P(X ≤ x, Y ≤ y) where (X,Y) ~ BVN(0,0,1,1,ρ).
///
/// Implements Gauss-Legendre quadrature on the angle form:
///   Φ₂(a,b;ρ) = Φ(a)Φ(b) + (1/2π) ∫₀^{arcsin(ρ)} exp(-(a²+b²-2ab·sin(θ))/(2cos²(θ))) dθ
///
/// # References
/// - Genz, A. (2004), "Numerical computation of rectangular bivariate and
///   trivariate normal and t probabilities", *Statistics and Computing* 14.
fn bivariate_normal_cdf(a: f64, b: f64, rho: f64, n: &NormalDistribution) -> f64 {
    // Handle degenerate cases
    if rho.abs() < 1e-15 {
        return n.cdf(a) * n.cdf(b);
    }
    if rho >= 1.0 - 1e-15 {
        return n.cdf(a.min(b));
    }
    if rho <= -1.0 + 1e-15 {
        return (n.cdf(a) - n.cdf(-b)).max(0.0);
    }

    // Use the angle-based quadrature approach from Genz
    use std::f64::consts::PI;

    let asin_rho = rho.asin();
    let a2 = a * a;
    let b2 = b * b;
    let ab = a * b;

    // 20-point Gauss-Legendre on [-1, 1]
    let x20 = [
        -0.9931285991850949, -0.9639719272779138, -0.912_234_428_251_326,
        -0.8391169718222188, -0.7463319064601508, -0.636_053_680_726_515,
        -0.5108670019508271, -0.3737060887154195, -0.2277858511416451,
        -0.0765265211334973,
         0.0765265211334973,  0.2277858511416451,  0.3737060887154195,
         0.5108670019508271,  0.636_053_680_726_515,  0.7463319064601508,
         0.8391169718222188,  0.912_234_428_251_326,  0.9639719272779138,
         0.9931285991850949,
    ];
    let w20 = [
        0.0176140071391521, 0.0406014298003869, 0.0626720483341091,
        0.0832767415767048, 0.1019301198172404, 0.1181945319615184,
        0.1316886384491766, 0.1420961093183821, 0.1491729864726037,
        0.1527533871307259,
        0.1527533871307259, 0.1491729864726037, 0.1420961093183821,
        0.1316886384491766, 0.1181945319615184, 0.1019301198172404,
        0.0832767415767048, 0.0626720483341091, 0.0406014298003869,
        0.0176140071391521,
    ];

    let mut sum = 0.0;
    for i in 0..20 {
        // Map GL node from [-1,1] to [0, arcsin(ρ)]
        let theta = 0.5 * asin_rho * (x20[i] + 1.0);
        let sin_theta = theta.sin();
        let cos_theta_sq = 1.0 - sin_theta * sin_theta;

        if cos_theta_sq > 1e-30 {
            let exponent = -(a2 + b2 - 2.0 * ab * sin_theta) / (2.0 * cos_theta_sq);
            if exponent > -500.0 {
                sum += w20[i] * exponent.exp();
            }
        }
    }

    // Φ₂ = Φ(a)Φ(b) + (arcsin(ρ)/(4π)) * sum_quadrature
    // Factor: integral = (arcsin(ρ)/2) * Σ wᵢ f(xᵢ)  (half-range GL)
    // Then Φ₂ = Φ(a)Φ(b) + integral / (2π)
    let result = n.cdf(a) * n.cdf(b) + 0.5 * asin_rho * sum / (2.0 * PI);
    result.clamp(0.0, 1.0)
}

/// Cholesky lower-triangular decomposition of a row-major n×n matrix.
/// Returns the lower-triangular factor L such that A = L L^T, stored row-major.
fn cholesky_lower(n: usize, a: &[f64]) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum();
            if i == j {
                let val = a[i * n + i] - s;
                l[i * n + j] = if val > 0.0 { val.sqrt() } else { 1e-15 };
            } else {
                l[i * n + j] = (a[i * n + j] - s) / l[j * n + j];
            }
        }
    }
    l
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ===== MC Basket tests =====

    #[test]
    fn mc_basket_single_asset_equals_bs() {
        // With 1 asset, basket call = BS call
        let result = mc_basket(
            &[100.0],
            &[1.0],
            100.0,
            0.05,
            &[0.0],
            &[0.20],
            &[1.0], // 1x1 correlation
            1.0,
            true,
            BasketType::WeightedAverage,
            200_000,
            42,
        );
        // BS ATM call ≈ 10.45
        assert!(
            (result.npv - 10.45).abs() < 3.0 * result.std_error + 0.5,
            "Single-asset basket {} not near BS 10.45 (stderr={})",
            result.npv,
            result.std_error
        );
    }

    #[test]
    fn mc_basket_max_two_assets_positive() {
        // Max of two correlated assets
        let corr = [1.0, 0.5, 0.5, 1.0];
        let result = mc_basket(
            &[100.0, 100.0],
            &[0.5, 0.5],
            100.0,
            0.05,
            &[0.0, 0.0],
            &[0.20, 0.25],
            &corr,
            1.0,
            true,
            BasketType::MaxBasket,
            100_000,
            42,
        );
        assert!(result.npv > 0.0, "Max basket call should be positive");
        // max(S1, S2) call should exceed a single-asset call
        assert!(
            result.npv > 10.0,
            "Max basket call {} should exceed single-asset ATM",
            result.npv
        );
    }

    #[test]
    fn mc_basket_weighted_average_put() {
        let corr = [1.0, 0.3, 0.3, 1.0];
        let result = mc_basket(
            &[100.0, 100.0],
            &[0.5, 0.5],
            100.0,
            0.05,
            &[0.0, 0.0],
            &[0.20, 0.20],
            &corr,
            1.0,
            false,
            BasketType::WeightedAverage,
            100_000,
            42,
        );
        assert!(result.npv > 0.0, "Basket put should be positive");
    }

    #[test]
    fn mc_basket_5_assets() {
        // 5-asset basket — test it runs without panic
        let n = 5;
        let spots = vec![100.0; n];
        let weights = vec![0.2; n];
        let divs = vec![0.0; n];
        let vols = vec![0.20; n];
        let mut corr = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                corr[i * n + j] = if i == j { 1.0 } else { 0.3 };
            }
        }

        let result = mc_basket(
            &spots, &weights, 100.0, 0.05, &divs, &vols, &corr, 1.0,
            true, BasketType::WeightedAverage, 50_000, 42,
        );
        assert!(result.npv > 0.0, "5-asset basket call should be positive");
    }

    // ===== BVN CDF validation =====

    #[test]
    fn bvn_cdf_accuracy() {
        let n = NormalDistribution::standard();
        // Reference values from scipy.stats.multivariate_normal
        let cases = [
            (0.0, 0.0, 0.0, 0.2500000000),
            (0.0, 0.0, 0.5, 0.3333350247),
            (0.0, 0.0, -0.5, 0.1666687231),
            (1.0, 1.0, 0.5, 0.7452030974),
            (-1.0, -1.0, 0.5, 0.0625137291),
            (1.0, -1.0, 0.5, 0.1548729755),
            (2.0, 2.0, 0.3, 0.9565408868),
            (0.0, 0.0, 0.999, 0.4928835248),
            (0.0, 0.0, -0.999, 0.0071178501),
        ];
        for (a, b, rho, expected) in cases {
            let got = bivariate_normal_cdf(a, b, rho, &n);
            assert!(
                (got - expected).abs() < 0.005,
                "BVN({},{},{}) = {} expected {} (diff {})",
                a, b, rho, got, expected, (got - expected).abs()
            );
        }
    }

    // ===== Stulz tests =====

    #[test]
    fn stulz_max_call_positive() {
        let price = stulz_max_call(100.0, 100.0, 100.0, 0.05, 0.0, 0.0, 0.20, 0.20, 0.5, 1.0);
        assert!(price > 0.0, "Stulz max call should be positive: {}", price);
    }

    #[test]
    fn stulz_max_exceeds_single_asset() {
        let n = NormalDistribution::standard();
        let max_call = stulz_max_call(100.0, 100.0, 100.0, 0.05, 0.0, 0.0, 0.20, 0.20, 0.5, 1.0);
        let single_call = bs_call(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, &n);
        assert!(
            max_call > single_call,
            "Max-of-two call {} should exceed single-asset call {}",
            max_call,
            single_call
        );
    }

    #[test]
    fn stulz_max_perfect_correlation_equals_single_asset() {
        // With ρ=1 and identical assets, max(S,S) = S, so max call = single call
        let max_call =
            stulz_max_call(100.0, 100.0, 100.0, 0.05, 0.0, 0.0, 0.20, 0.20, 0.999, 1.0);
        let n = NormalDistribution::standard();
        let single_call = bs_call(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, &n);
        assert_abs_diff_eq!(max_call, single_call, epsilon = 0.5);
    }

    #[test]
    fn stulz_min_plus_max_equals_sum() {
        // min(S1,S2) + max(S1,S2) = S1 + S2
        // => Call_min(K) + Call_max(K) = Call(S1,K) + Call(S2,K)
        let min_call = stulz_min_call(100.0, 100.0, 100.0, 0.05, 0.0, 0.0, 0.20, 0.25, 0.3, 1.0);
        let max_call = stulz_max_call(100.0, 100.0, 100.0, 0.05, 0.0, 0.0, 0.20, 0.25, 0.3, 1.0);
        let n = NormalDistribution::standard();
        let c1 = bs_call(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, &n);
        let c2 = bs_call(100.0, 100.0, 0.05, 0.0, 0.25, 1.0, &n);
        assert_abs_diff_eq!(min_call + max_call, c1 + c2, epsilon = 1e-6);
    }

    #[test]
    fn stulz_mc_convergence() {
        // Stulz max call should match MC max basket
        let stulz = stulz_max_call(100.0, 100.0, 100.0, 0.05, 0.0, 0.0, 0.20, 0.25, 0.5, 1.0);
        let corr = [1.0, 0.5, 0.5, 1.0];
        let mc = mc_basket(
            &[100.0, 100.0],
            &[0.5, 0.5],
            100.0,
            0.05,
            &[0.0, 0.0],
            &[0.20, 0.25],
            &corr,
            1.0,
            true,
            BasketType::MaxBasket,
            500_000,
            42,
        );
        assert!(
            (stulz - mc.npv).abs() < 3.0 * mc.std_error + 0.5,
            "Stulz {} vs MC {}: diff {} (stderr {})",
            stulz,
            mc.npv,
            (stulz - mc.npv).abs(),
            mc.std_error
        );
    }

    // ===== Kirk spread tests =====

    #[test]
    fn kirk_spread_positive() {
        let price = kirk_spread_call(100.0, 90.0, 5.0, 0.05, 0.0, 0.0, 0.20, 0.25, 0.5, 1.0);
        assert!(price > 0.0, "Kirk spread call should be positive: {}", price);
    }

    #[test]
    fn kirk_spread_zero_strike_near_margrabe() {
        // With K=0, spread call = exchange option
        let kirk = kirk_spread_call(100.0, 100.0, 0.0, 0.05, 0.0, 0.0, 0.20, 0.25, 0.5, 1.0);
        let margrabe = margrabe_exchange(100.0, 100.0, 0.0, 0.0, 0.20, 0.25, 0.5, 1.0);
        // Kirk is an approximation, so allow some tolerance
        assert!(
            (kirk - margrabe).abs() < 1.0,
            "Kirk spread (K=0) {} should be near Margrabe {}",
            kirk,
            margrabe
        );
    }

    #[test]
    fn kirk_spread_put_call_parity() {
        let s1 = 100.0;
        let s2 = 95.0;
        let k = 3.0;
        let r = 0.05;
        let t = 1.0;
        let q1 = 0.0;
        let q2 = 0.0;

        let call = kirk_spread_call(s1, s2, k, r, q1, q2, 0.20, 0.25, 0.5, t);
        let put = kirk_spread_put(s1, s2, k, r, q1, q2, 0.20, 0.25, 0.5, t);
        let df = (-r * t).exp();
        let f1 = s1 * ((r - q1) * t).exp();
        let f2 = s2 * ((r - q2) * t).exp();

        // C - P = df * (F1 - F2 - K)
        let parity = call - put - df * (f1 - f2 - k);
        assert_abs_diff_eq!(parity, 0.0, epsilon = 1e-8);
    }

    // ===== Margrabe tests =====

    #[test]
    fn margrabe_exchange_positive() {
        let price = margrabe_exchange(110.0, 100.0, 0.0, 0.0, 0.20, 0.25, 0.5, 1.0);
        assert!(price > 0.0, "Margrabe exchange should be positive: {}", price);
    }

    #[test]
    fn margrabe_homogeneity() {
        // Scale both assets by λ => price scales by λ
        let p1 = margrabe_exchange(100.0, 100.0, 0.0, 0.0, 0.20, 0.25, 0.5, 1.0);
        let p2 = margrabe_exchange(200.0, 200.0, 0.0, 0.0, 0.20, 0.25, 0.5, 1.0);
        assert_abs_diff_eq!(p2, 2.0 * p1, epsilon = 1e-8);
    }

    #[test]
    fn margrabe_zero_vol_equals_intrinsic() {
        // With σ=0, exchange option = max(S₁ - S₂, 0) = intrinsic
        let price = margrabe_exchange(110.0, 100.0, 0.0, 0.0, 0.001, 0.001, 0.99, 1.0);
        assert_abs_diff_eq!(price, 10.0, epsilon = 0.5);
    }

    #[test]
    fn margrabe_symmetry() {
        // Exchange(S1→S2) != Exchange(S2→S1) in general, but both are positive
        let p_12 = margrabe_exchange(100.0, 100.0, 0.0, 0.0, 0.20, 0.25, 0.5, 1.0);
        let p_21 = margrabe_exchange(100.0, 100.0, 0.0, 0.0, 0.25, 0.20, 0.5, 1.0);
        assert!(p_12 > 0.0);
        assert!(p_21 > 0.0);
        // With same spot/q but different vol, prices differ
    }

    #[test]
    fn margrabe_correlation_effect() {
        // Higher correlation => lower exchange option value (less spread volatility)
        let p_low_rho = margrabe_exchange(100.0, 100.0, 0.0, 0.0, 0.20, 0.20, 0.0, 1.0);
        let p_high_rho = margrabe_exchange(100.0, 100.0, 0.0, 0.0, 0.20, 0.20, 0.8, 1.0);
        assert!(
            p_low_rho > p_high_rho,
            "Lower correlation {} should give higher exchange value than {}",
            p_low_rho,
            p_high_rho
        );
    }
}
