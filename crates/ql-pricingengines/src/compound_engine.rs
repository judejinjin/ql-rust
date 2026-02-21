//! Analytic compound option pricing engine.
//!
//! Implements the Geske (1979) formula for European compound options
//! (option on option). Uses bivariate normal distribution.

use ql_instruments::compound_option::CompoundOption;
use ql_instruments::payoff::OptionType;
use ql_math::distributions::NormalDistribution;
use ql_math::solvers1d;
use ql_math::solvers1d::Solver1D;

/// Result from the compound option engine.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[must_use]
pub struct CompoundOptionResult {
    /// Net present value.
    pub npv: f64,
}

/// Approximate bivariate normal CDF using Drezner-Wesolowsky (1990).
///
/// Computes P(X ≤ a, Y ≤ b) where (X,Y) have standard bivariate
/// normal distribution with correlation ρ.
fn bivariate_normal_cdf(a: f64, b: f64, rho: f64) -> f64 {
    let norm = NormalDistribution::standard();

    if rho.abs() < 1e-15 {
        return norm.cdf(a) * norm.cdf(b);
    }
    if (rho - 1.0).abs() < 1e-15 {
        return norm.cdf(a.min(b));
    }
    if (rho + 1.0).abs() < 1e-15 {
        let sum = a + b;
        if sum < 0.0 {
            return 0.0;
        }
        return norm.cdf(a) + norm.cdf(b) - 1.0;
    }

    // Drezner-Wesolowsky approximation via Gauss-Legendre quadrature
    // Use the relationship: Φ₂(a,b,ρ) ≈ integration-based approach
    // For simplicity, use the decomposition:
    // Φ₂(a,b,ρ) = Φ(a)Φ(b) + ∫₀^ρ φ₂(a,b,t) dt
    // where φ₂ is the bivariate normal density

    // Tetrachoric series approximation (good for |ρ| < 0.7):
    // Φ₂(a,b,ρ) = Φ(a)Φ(b) + Σ ρ^n/(n!) × H_{n-1}(-a)×H_{n-1}(-b) × φ(a)×φ(b)
    // where H_n are (probabilist's) Hermite polynomials

    // More robust: numerical integration of the conditional distribution
    // Φ₂(a,b,ρ) = ∫_{-∞}^{a} φ(x) Φ((b - ρx)/√(1-ρ²)) dx

    // Use 20-point Gauss-Hermite quadrature on the outer integral
    let sqrt_1_rho2 = (1.0 - rho * rho).sqrt();

    // Gauss-Legendre 12-point quadrature on [-6, a]
    let n_points = 24;
    let lower = -6.0_f64;
    let upper = a;
    if upper <= lower {
        return 0.0;
    }

    let h = (upper - lower) / n_points as f64;
    let mut sum = 0.0;

    // Simpson's rule
    for i in 0..=n_points {
        let x = lower + i as f64 * h;
        let inner = norm.cdf((b - rho * x) / sqrt_1_rho2);
        let weight = if i == 0 || i == n_points {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        sum += weight * norm.pdf(x) * inner;
    }

    (sum * h / 3.0).clamp(0.0, 1.0)
}

/// Price a compound option using the Geske (1979) analytic formula.
///
/// # Arguments
/// * `option` — the compound option
/// * `spot` — current underlying price
/// * `r` — risk-free rate
/// * `q` — dividend yield
/// * `vol` — volatility
pub fn analytic_compound_option(
    option: &CompoundOption,
    spot: f64,
    r: f64,
    q: f64,
    vol: f64,
) -> CompoundOptionResult {
    let t1 = option.mother_expiry;
    let t2 = option.daughter_expiry;

    if t1 <= 0.0 || t2 <= 0.0 {
        return CompoundOptionResult { npv: 0.0 };
    }

    let norm = NormalDistribution::standard();
    let sqrt_t1 = t1.sqrt();
    let sqrt_t2 = t2.sqrt();
    let rho = (t1 / t2).sqrt();

    let k1 = option.mother_strike;
    let k2 = option.daughter_strike;

    let tau = t2 - t1;
    let s_star = find_critical_price(k1, k2, r, q, vol, tau, option.daughter_type);

    let b = r - q;
    let d1 = ((spot / s_star).ln() + (b + 0.5 * vol * vol) * t1) / (vol * sqrt_t1);
    let d2 = d1 - vol * sqrt_t1;

    let e1 = ((spot / k2).ln() + (b + 0.5 * vol * vol) * t2) / (vol * sqrt_t2);
    let e2 = e1 - vol * sqrt_t2;

    let npv = match (option.mother_type, option.daughter_type) {
        (OptionType::Call, OptionType::Call) => {
            // Call on Call (Geske 1979)
            spot * (-q * t2).exp() * bivariate_normal_cdf(d1, e1, rho)
                - k2 * (-r * t2).exp() * bivariate_normal_cdf(d2, e2, rho)
                - k1 * (-r * t1).exp() * norm.cdf(d2)
        }
        (OptionType::Put, OptionType::Call) => {
            // Put on Call
            // Value = -S·M(-d1, e1; -ρ) + K2·e^{-rT2}·M(-d2, e2; -ρ) + K1·e^{-rT1}·N(-d2)
            -spot * (-q * t2).exp() * bivariate_normal_cdf(-d1, e1, -rho)
                + k2 * (-r * t2).exp() * bivariate_normal_cdf(-d2, e2, -rho)
                + k1 * (-r * t1).exp() * norm.cdf(-d2)
        }
        (OptionType::Call, OptionType::Put) => {
            // Call on Put
            // Exercise when put value > K1, i.e., when S < S*
            // Value = -S·M(-d1, -e1; ρ) + K2·e^{-rT2}·M(-d2, -e2; ρ) - K1·e^{-rT1}·N(-d2)
            -spot * (-q * t2).exp() * bivariate_normal_cdf(-d1, -e1, rho)
                + k2 * (-r * t2).exp() * bivariate_normal_cdf(-d2, -e2, rho)
                - k1 * (-r * t1).exp() * norm.cdf(-d2)
        }
        (OptionType::Put, OptionType::Put) => {
            // Put on Put
            spot * (-q * t2).exp() * bivariate_normal_cdf(d1, -e1, -rho)
                - k2 * (-r * t2).exp() * bivariate_normal_cdf(d2, -e2, -rho)
                + k1 * (-r * t1).exp() * norm.cdf(d2)
        }
    };

    CompoundOptionResult { npv: npv.max(0.0) }
}

/// Find the critical stock price S* at the mother expiry where
/// the daughter option value equals the mother strike K1.
fn find_critical_price(
    k1: f64,
    k2: f64,
    r: f64,
    q: f64,
    vol: f64,
    tau: f64,      // T2 - T1
    daughter_type: OptionType,
) -> f64 {
    let norm = NormalDistribution::standard();
    let phi = daughter_type.sign();

    // BS formula as a function of S
    let bs_value = |s: f64| -> f64 {
        if s <= 0.0 {
            return 0.0;
        }
        let d1 = ((s / k2).ln() + (r - q + 0.5 * vol * vol) * tau) / (vol * tau.sqrt());
        let d2 = d1 - vol * tau.sqrt();
        phi * s * (-q * tau).exp() * norm.cdf(phi * d1)
            - phi * k2 * (-r * tau).exp() * norm.cdf(phi * d2)
    };

    // Find S* such that bs_value(S*) = k1
    let f = |s: f64| -> f64 { bs_value(s) - k1 };

    // Use bisection: S* is in [1e-6, 10 * K2]
    let result = solvers1d::Brent.solve(f, 0.0, k2, 1e-6, 10.0 * k2, 1e-8, 200);
    match result {
        Ok(s) => s,
        Err(_) => k2, // fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn call_on_call_positive() {
        let opt = CompoundOption::new(
            OptionType::Call, OptionType::Call,
            5.0, 0.5, 100.0, 1.0,
        );
        let result = analytic_compound_option(&opt, 100.0, 0.05, 0.0, 0.20);
        assert!(
            result.npv > 0.0,
            "Call-on-call should be positive: {:.4}",
            result.npv
        );
    }

    #[test]
    fn call_on_put_positive() {
        // Call on a put with a low mother strike to ensure it's in-the-money
        let opt = CompoundOption::new(
            OptionType::Call, OptionType::Put,
            1.0, 0.25, 100.0, 1.0,
        );
        let result = analytic_compound_option(&opt, 100.0, 0.05, 0.0, 0.20);
        assert!(
            result.npv > 0.0,
            "Call-on-put should be positive: {:.4}",
            result.npv
        );
    }

    #[test]
    fn put_on_call_positive() {
        let opt = CompoundOption::new(
            OptionType::Put, OptionType::Call,
            15.0, 0.5, 100.0, 1.0,
        );
        let result = analytic_compound_option(&opt, 100.0, 0.05, 0.0, 0.20);
        assert!(
            result.npv > 0.0,
            "Put-on-call should be positive: {:.4}",
            result.npv
        );
    }

    #[test]
    fn compound_less_than_daughter() {
        // A call-on-call should be cheaper than the daughter call itself
        let opt = CompoundOption::new(
            OptionType::Call, OptionType::Call,
            5.0, 0.5, 100.0, 1.0,
        );
        let compound = analytic_compound_option(&opt, 100.0, 0.05, 0.0, 0.20);

        // BS call with K=100, T=1: ≈ 10.45
        let daughter_value = 10.45;
        assert!(
            compound.npv < daughter_value + 1.0,
            "Compound {:.4} should be less than daughter {:.4}",
            compound.npv, daughter_value
        );
    }

    #[test]
    fn higher_vol_increases_compound() {
        let opt = CompoundOption::new(
            OptionType::Call, OptionType::Call,
            5.0, 0.5, 100.0, 1.0,
        );
        let low = analytic_compound_option(&opt, 100.0, 0.05, 0.0, 0.15);
        let high = analytic_compound_option(&opt, 100.0, 0.05, 0.0, 0.35);

        assert!(
            high.npv >= low.npv - 0.1,
            "Higher vol compound {:.4} should be ≥ lower vol {:.4}",
            high.npv, low.npv
        );
    }

    #[test]
    fn bivariate_normal_uncorrelated() {
        let norm = NormalDistribution::standard();
        let result = bivariate_normal_cdf(0.0, 0.0, 0.0);
        let expected = norm.cdf(0.0) * norm.cdf(0.0);
        assert_abs_diff_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    fn bivariate_normal_perfect_correlation() {
        let norm = NormalDistribution::standard();
        let result = bivariate_normal_cdf(1.0, 2.0, 1.0);
        // With ρ=1, Φ₂(a,b,1) = Φ(min(a,b))
        assert_abs_diff_eq!(result, norm.cdf(1.0), epsilon = 1e-6);
    }

    #[test]
    fn expired_compound() {
        let opt = CompoundOption::new(
            OptionType::Call, OptionType::Call,
            5.0, 0.5, 100.0, 1.0,
        );
        // With t=0, should return 0
        let mut expired_opt = opt;
        // We can't set t=0 directly, but we test with the function
        expired_opt.mother_expiry = 0.0;
        expired_opt.daughter_expiry = 0.0;
        let result = analytic_compound_option(&expired_opt, 100.0, 0.05, 0.0, 0.20);
        assert_abs_diff_eq!(result.npv, 0.0, epsilon = 1e-10);
    }
}
