//! Monte Carlo Asian option engines.
//!
//! - [`mc_asian_arithmetic_price`] — MC arithmetic average price Asian
//! - [`mc_asian_arithmetic_strike`] — MC arithmetic average strike Asian
//! - [`mc_asian_geometric_price`] — MC geometric average price Asian (control variate)
//! - [`mc_asian_heston_price`] — MC arithmetic Asian under Heston dynamics

use rand::prelude::*;
use rand_distr::StandardNormal;

/// Result from MC Asian engines.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct McAsianResult {
    /// Option price.
    pub price: f64,
    /// Standard error of the estimate.
    pub std_error: f64,
}

/// Monte Carlo arithmetic average price Asian option.
///
/// Prices an Asian option where the payoff is max(ω(A - K), 0)
/// with A = (1/n) Σ S(tᵢ) the arithmetic average over fixing dates.
///
/// Uses antithetic variates for variance reduction.
///
/// # Arguments
/// - `spot`, `strike`, `r`, `q`, `sigma` — standard BS parameters
/// - `t` — total time to expiry (years)
/// - `n_fixings` — number of equally-spaced fixing dates
/// - `n_paths` — number of MC paths
/// - `is_call` — true for call, false for put
/// - `seed` — optional RNG seed
#[allow(clippy::too_many_arguments)]
pub fn mc_asian_arithmetic_price(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    n_fixings: usize,
    n_paths: usize,
    is_call: bool,
    seed: Option<u64>,
) -> McAsianResult {
    let omega = if is_call { 1.0 } else { -1.0 };
    let dt = t / n_fixings as f64;
    let drift = (r - q - 0.5 * sigma * sigma) * dt;
    let vol = sigma * dt.sqrt();
    let df = (-r * t).exp();

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::seed_from_u64(42),
    };

    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    for _ in 0..n_paths / 2 {
        let mut s1 = spot;
        let mut s2 = spot;
        let mut avg1 = 0.0;
        let mut avg2 = 0.0;

        for _ in 0..n_fixings {
            let z: f64 = rng.sample(StandardNormal);
            s1 *= (drift + vol * z).exp();
            s2 *= (drift - vol * z).exp(); // antithetic
            avg1 += s1;
            avg2 += s2;
        }

        avg1 /= n_fixings as f64;
        avg2 /= n_fixings as f64;

        let payoff1 = (omega * (avg1 - strike)).max(0.0);
        let payoff2 = (omega * (avg2 - strike)).max(0.0);
        let payoff = 0.5 * (payoff1 + payoff2);

        sum += payoff;
        sum_sq += payoff * payoff;
    }

    let n_eff = (n_paths / 2) as f64;
    let mean = sum / n_eff;
    let variance = (sum_sq / n_eff - mean * mean).max(0.0);
    let std_error = (variance / n_eff).sqrt();

    McAsianResult {
        price: df * mean,
        std_error: df * std_error,
    }
}

/// Monte Carlo arithmetic average strike Asian option.
///
/// Payoff: max(ω(S(T) - A), 0) where A = arithmetic average.
#[allow(clippy::too_many_arguments)]
pub fn mc_asian_arithmetic_strike(
    spot: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    n_fixings: usize,
    n_paths: usize,
    is_call: bool,
    seed: Option<u64>,
) -> McAsianResult {
    let omega = if is_call { 1.0 } else { -1.0 };
    let dt = t / n_fixings as f64;
    let drift = (r - q - 0.5 * sigma * sigma) * dt;
    let vol = sigma * dt.sqrt();
    let df = (-r * t).exp();

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::seed_from_u64(42),
    };

    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    for _ in 0..n_paths / 2 {
        let mut s1 = spot;
        let mut s2 = spot;
        let mut avg1 = 0.0;
        let mut avg2 = 0.0;

        for _ in 0..n_fixings {
            let z: f64 = rng.sample(StandardNormal);
            s1 *= (drift + vol * z).exp();
            s2 *= (drift - vol * z).exp();
            avg1 += s1;
            avg2 += s2;
        }

        avg1 /= n_fixings as f64;
        avg2 /= n_fixings as f64;

        let payoff1 = (omega * (s1 - avg1)).max(0.0);
        let payoff2 = (omega * (s2 - avg2)).max(0.0);
        let payoff = 0.5 * (payoff1 + payoff2);

        sum += payoff;
        sum_sq += payoff * payoff;
    }

    let n_eff = (n_paths / 2) as f64;
    let mean = sum / n_eff;
    let variance = (sum_sq / n_eff - mean * mean).max(0.0);
    let std_error = (variance / n_eff).sqrt();

    McAsianResult {
        price: df * mean,
        std_error: df * std_error,
    }
}

/// Monte Carlo geometric average price Asian option.
///
/// Payoff: max(ω(G - K), 0) where G = (Π S(tᵢ))^(1/n).
/// The geometric Asian has a closed-form solution, so this MC engine
/// can be used as a control variate for the arithmetic version.
#[allow(clippy::too_many_arguments)]
pub fn mc_asian_geometric_price(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    n_fixings: usize,
    n_paths: usize,
    is_call: bool,
    seed: Option<u64>,
) -> McAsianResult {
    let omega = if is_call { 1.0 } else { -1.0 };
    let dt = t / n_fixings as f64;
    let drift = (r - q - 0.5 * sigma * sigma) * dt;
    let vol = sigma * dt.sqrt();
    let df = (-r * t).exp();

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::seed_from_u64(42),
    };

    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    for _ in 0..n_paths / 2 {
        let mut s1 = spot;
        let mut s2 = spot;
        let mut log_avg1 = 0.0;
        let mut log_avg2 = 0.0;

        for _ in 0..n_fixings {
            let z: f64 = rng.sample(StandardNormal);
            s1 *= (drift + vol * z).exp();
            s2 *= (drift - vol * z).exp();
            log_avg1 += s1.ln();
            log_avg2 += s2.ln();
        }

        let geo1 = (log_avg1 / n_fixings as f64).exp();
        let geo2 = (log_avg2 / n_fixings as f64).exp();

        let payoff1 = (omega * (geo1 - strike)).max(0.0);
        let payoff2 = (omega * (geo2 - strike)).max(0.0);
        let payoff = 0.5 * (payoff1 + payoff2);

        sum += payoff;
        sum_sq += payoff * payoff;
    }

    let n_eff = (n_paths / 2) as f64;
    let mean = sum / n_eff;
    let variance = (sum_sq / n_eff - mean * mean).max(0.0);
    let std_error = (variance / n_eff).sqrt();

    McAsianResult {
        price: df * mean,
        std_error: df * std_error,
    }
}

/// Monte Carlo arithmetic Asian option under Heston stochastic volatility.
///
/// Simulates the Heston process:
/// ```text
/// dS = (r-q)S dt + √v S dW₁
/// dv = κ(θ-v) dt + σ √v dW₂
/// dW₁·dW₂ = ρ dt
/// ```
///
/// Uses QE (quadratic-exponential) discretization for the variance.
#[allow(clippy::too_many_arguments)]
pub fn mc_asian_heston_price(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    t: f64,
    n_fixings: usize,
    n_paths: usize,
    is_call: bool,
    seed: Option<u64>,
) -> McAsianResult {
    let omega = if is_call { 1.0 } else { -1.0 };
    let dt = t / n_fixings as f64;
    let df = (-r * t).exp();
    let rho_bar = (1.0 - rho * rho).sqrt();

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::seed_from_u64(42),
    };

    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    for _ in 0..n_paths {
        let mut s = spot;
        let mut v = v0;
        let mut avg = 0.0;

        for _ in 0..n_fixings {
            let z1: f64 = rng.sample(StandardNormal);
            let z2: f64 = rng.sample(StandardNormal);
            let w1 = z1;
            let w2 = rho * z1 + rho_bar * z2;

            // QE discretization for variance
            let m = theta + (v - theta) * (-kappa * dt).exp();
            let s2 = v * sigma * sigma * (-kappa * dt).exp() * (1.0 - (-kappa * dt).exp()) / kappa
                + theta * sigma * sigma * (1.0 - (-kappa * dt).exp()).powi(2) / (2.0 * kappa);
            let psi = s2 / (m * m).max(1e-20);

            let v_next = if psi <= 1.5 {
                // Moment matching with non-central chi-squared
                let b2 = (2.0 / psi - 1.0 + (2.0 / psi).sqrt() * (2.0 / psi - 1.0).sqrt()).max(0.0);
                let a = m / (1.0 + b2);
                let b = b2.sqrt();
                a * (b + w2).powi(2)
            } else {
                // Exponential approximation
                let p = (psi - 1.0) / (psi + 1.0);
                let beta = (1.0 - p) / m.max(1e-20);
                let u_uniform: f64 = rng.random();
                if u_uniform <= p {
                    0.0
                } else {
                    (-(1.0 - u_uniform) / ((1.0 - p) * beta).max(1e-20)).ln().abs() / beta.max(1e-20)
                }
            };

            let vol_avg = 0.5 * (v + v_next.max(0.0));
            s *= ((r - q - 0.5 * vol_avg) * dt + vol_avg.sqrt() * dt.sqrt() * w1).exp();
            v = v_next.max(0.0);
            avg += s;
        }

        avg /= n_fixings as f64;
        let payoff = (omega * (avg - strike)).max(0.0);
        sum += payoff;
        sum_sq += payoff * payoff;
    }

    let n = n_paths as f64;
    let mean = sum / n;
    let variance = (sum_sq / n - mean * mean).max(0.0);
    let std_error = (variance / n).sqrt();

    McAsianResult {
        price: df * mean,
        std_error: df * std_error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_mc_asian_arith_call() {
        let res = mc_asian_arithmetic_price(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 12, 50000, true, Some(42),
        );
        // Arithmetic Asian call should be less than European call (~10.45)
        assert!(res.price > 2.0 && res.price < 10.0, "price={}", res.price);
        assert!(res.std_error < 0.5, "se={}", res.std_error);
    }

    #[test]
    fn test_mc_asian_arith_put() {
        let res = mc_asian_arithmetic_price(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 12, 50000, false, Some(42),
        );
        assert!(res.price > 1.0 && res.price < 8.0, "price={}", res.price);
    }

    #[test]
    fn test_mc_asian_arith_strike() {
        let res = mc_asian_arithmetic_strike(
            100.0, 0.05, 0.0, 0.20, 1.0, 12, 50000, true, Some(42),
        );
        assert!(res.price > 0.0, "price={}", res.price);
    }

    #[test]
    fn test_mc_asian_geometric() {
        let res = mc_asian_geometric_price(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 12, 50000, true, Some(42),
        );
        assert!(res.price > 2.0 && res.price < 10.0, "price={}", res.price);
    }

    #[test]
    fn test_mc_asian_heston() {
        let res = mc_asian_heston_price(
            100.0, 100.0, 0.05, 0.0,
            0.04, 1.5, 0.04, 0.3, -0.7,
            1.0, 12, 20000, true, Some(42),
        );
        assert!(res.price > 1.0 && res.price < 12.0, "price={}", res.price);
    }
}
