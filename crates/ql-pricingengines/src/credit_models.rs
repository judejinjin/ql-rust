#![allow(clippy::too_many_arguments)]
//! Advanced credit models: CDO, nth-to-default, CDS options, copulas.
//!
//! Provides:
//! - `GaussianCopulaLHP` — Large Homogeneous Portfolio Gaussian copula CDO pricing
//! - `nth_to_default_mc` — Monte Carlo n-th to default basket pricing
//! - `cds_option_black` — Black model for CDS options
//! - `gaussian_copula_loss_distribution` — loss distribution via Gaussian copula

use ql_math::distributions::NormalDistribution;

/// Gaussian copula Large Homogeneous Portfolio (LHP) CDO tranche pricer.
///
/// Under the one-factor Gaussian copula, the conditional default probability is:
///   p(t|Z) = Φ((Φ⁻¹(p(t)) − √ρ Z) / √(1−ρ))
///
/// The LHP model assumes all names are identical.
#[derive(Debug, Clone)]
pub struct GaussianCopulaLHP {
    /// Number of names in the portfolio.
    pub n_names: usize,
    /// Uniform correlation (ρ).
    pub correlation: f64,
    /// Recovery rate (R).
    pub recovery: f64,
    /// Individual default probability p(T).
    pub default_prob: f64,
}

impl GaussianCopulaLHP {
    pub fn new(n_names: usize, correlation: f64, recovery: f64, default_prob: f64) -> Self {
        Self {
            n_names,
            correlation,
            recovery,
            default_prob,
        }
    }

    /// Expected tranche loss E[L_{A,D}] for a tranche with attachment A
    /// and detachment D, using Gauss-Hermite quadrature.
    ///
    /// The tranche loss = min(max(L − A, 0), D − A) / (D − A)
    /// where L is the portfolio loss as a fraction of notional.
    pub fn tranche_expected_loss(&self, attachment: f64, detachment: f64) -> f64 {
        if detachment <= attachment {
            return 0.0;
        }

        let n = NormalDistribution::standard();
        let phi_inv_p = inverse_normal_cdf(self.default_prob);
        let sqrt_rho = self.correlation.sqrt();
        let sqrt_1_m_rho = (1.0 - self.correlation).sqrt();
        let lgd = 1.0 - self.recovery;

        // 20-point Gauss-Hermite quadrature
        let nodes = gauss_hermite_20();
        let mut expected_tranche_loss = 0.0;

        for &(xi, wi) in &nodes {
            let z = xi * std::f64::consts::SQRT_2;
            let cond_p = n.cdf((phi_inv_p - sqrt_rho * z) / sqrt_1_m_rho);
            let cond_loss = cond_p * lgd; // expected portfolio loss fraction conditional on Z

            // Tranche loss
            let tranche_loss = cond_loss.clamp(attachment, detachment) - attachment;
            expected_tranche_loss += wi * tranche_loss;
        }

        // Normalize by √π (GH quadrature uses exp(-x²) weighting)
        expected_tranche_loss /= std::f64::consts::PI.sqrt();

        // Normalize by tranche width
        expected_tranche_loss / (detachment - attachment)
    }

    /// Compute loss distribution via Gauss-Hermite quadrature.
    /// Returns P(L ≤ x) for x = 0, dx, 2dx, ..., 1.
    pub fn loss_distribution(&self, n_points: usize) -> Vec<(f64, f64)> {
        let n = NormalDistribution::standard();
        let phi_inv_p = inverse_normal_cdf(self.default_prob);
        let sqrt_rho = self.correlation.sqrt();
        let sqrt_1_m_rho = (1.0 - self.correlation).sqrt();
        let lgd = 1.0 - self.recovery;

        let nodes = gauss_hermite_20();
        let mut result = Vec::with_capacity(n_points);

        for i in 0..n_points {
            let x = i as f64 / (n_points - 1) as f64; // loss level
            let mut prob = 0.0;

            for &(xi, wi) in &nodes {
                let z = xi * std::f64::consts::SQRT_2;
                let cond_p = n.cdf((phi_inv_p - sqrt_rho * z) / sqrt_1_m_rho);
                let cond_loss = cond_p * lgd;

                if cond_loss <= x {
                    prob += wi;
                }
            }
            prob /= std::f64::consts::PI.sqrt();
            result.push((x, prob));
        }
        result
    }
}

/// Monte Carlo n-th to default basket pricing.
///
/// Returns the expected loss of the n-th name to default.
pub fn nth_to_default_mc(
    n_names: usize,
    default_probs: &[f64], // individual default probs
    correlation: f64,
    recovery: f64,
    nth: usize,            // which default to trigger (1-indexed)
    notional: f64,
    n_paths: usize,
    seed: u64,
) -> NtdResult {
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use rand_distr::StandardNormal;

    assert_eq!(default_probs.len(), n_names);
    assert!(nth >= 1 && nth <= n_names);

    let n = NormalDistribution::standard();
    let sqrt_rho = correlation.sqrt();
    let sqrt_1_m_rho = (1.0 - correlation).sqrt();
    let lgd = (1.0 - recovery) * notional / n_names as f64;

    // Precompute Φ⁻¹(p_i)
    let thresholds: Vec<f64> = default_probs.iter().map(|&p| inverse_normal_cdf(p)).collect();

    let mut rng = SmallRng::seed_from_u64(seed);
    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    for _ in 0..n_paths {
        let z: f64 = rng.sample(StandardNormal); // common factor
        let mut n_defaults = 0usize;

        for &threshold in &thresholds {
            let eps: f64 = rng.sample(StandardNormal); // idiosyncratic
            let x = sqrt_rho * z + sqrt_1_m_rho * eps;
            if n.cdf(x) < n.cdf(threshold) {
                // Equivalent: x < threshold since Φ is monotone
                n_defaults += 1;
            }
        }

        let payoff = if n_defaults >= nth { lgd } else { 0.0 };
        sum += payoff;
        sum_sq += payoff * payoff;
    }

    let price = sum / n_paths as f64;
    let variance = sum_sq / n_paths as f64 - price * price;
    let std_error = (variance / n_paths as f64).max(0.0).sqrt();

    NtdResult {
        price,
        std_error,
        expected_defaults: sum / n_paths as f64 / lgd * lgd, // re-normalize
    }
}

/// Result of n-th to default MC.
#[derive(Debug, Clone)]
pub struct NtdResult {
    pub price: f64,
    pub std_error: f64,
    pub expected_defaults: f64,
}

/// Black model CDS option price.
///
/// A payer CDS option gives the right to buy protection at strike spread K.
/// Value = RPV01 × [F Φ(d₁) − K Φ(d₂)] for payer
///       = RPV01 × [K Φ(−d₂) − F Φ(−d₁)] for receiver
pub fn cds_option_black(
    forward_spread: f64,
    strike_spread: f64,
    vol: f64,
    expiry: f64,
    rpv01: f64, // risky PV01 (annuity)
    is_payer: bool,
) -> f64 {
    if vol <= 0.0 || expiry <= 0.0 || rpv01 <= 0.0 {
        let intrinsic = if is_payer {
            (forward_spread - strike_spread).max(0.0)
        } else {
            (strike_spread - forward_spread).max(0.0)
        };
        return rpv01 * intrinsic;
    }

    let n = NormalDistribution::standard();
    let std_dev = vol * expiry.sqrt();
    let d1 = ((forward_spread / strike_spread).ln() + 0.5 * std_dev * std_dev) / std_dev;
    let d2 = d1 - std_dev;

    if is_payer {
        rpv01 * (forward_spread * n.cdf(d1) - strike_spread * n.cdf(d2))
    } else {
        rpv01 * (strike_spread * n.cdf(-d2) - forward_spread * n.cdf(-d1))
    }
}

/// Inverse standard normal CDF (rational approximation, Beasley-Springer-Moro).
fn inverse_normal_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return -10.0;
    }
    if p >= 1.0 {
        return 10.0;
    }

    // Peter Acklam's algorithm
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383_577_518_672_69e2,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

/// 20-point Gauss-Hermite quadrature nodes and weights.
fn gauss_hermite_20() -> [(f64, f64); 20] {
    // Symmetric: 10 positive + 10 negative
    let half = [
        (0.245_340_708_300_901, 0.462_243_669_600_610),
        (0.737_473_728_545_394, 0.286_675_505_362_834),
        (1.234_076_215_395_323, 0.109_017_206_020_023),
        (1.738_537_712_116_586, 0.024_810_520_887_464),
        (2.254_974_002_089_276, 0.003_243_773_342_238),
        (2.788_806_058_428_131, 2.283_386_360_163e-4),
        (3.347_854_567_383_216, 7.802_556_478_532e-6),
        (3.944_764_040_115_198, 1.086_069_370_769e-7),
        (4.603_682_449_550_744, 4.399_340_992_273e-10),
        (5.387_480_890_011_233, 2.229_393_645_534e-13),
    ];

    let mut nodes = [(0.0, 0.0); 20];
    for (i, &(x, w)) in half.iter().enumerate() {
        nodes[2 * i] = (-x, w);
        nodes[2 * i + 1] = (x, w);
    }
    nodes
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn inverse_normal_symmetry() {
        let x1 = inverse_normal_cdf(0.05);
        let x2 = inverse_normal_cdf(0.95);
        assert_abs_diff_eq!(x1, -x2, epsilon = 1e-8);
    }

    #[test]
    fn inverse_normal_median() {
        assert_abs_diff_eq!(inverse_normal_cdf(0.5), 0.0, epsilon = 1e-8);
    }

    #[test]
    fn lhp_equity_tranche_loss_positive() {
        let model = GaussianCopulaLHP::new(125, 0.3, 0.4, 0.05);
        let el = model.tranche_expected_loss(0.0, 0.03); // equity tranche
        assert!(el > 0.0, "Equity tranche expected loss should be positive: {el}");
    }

    #[test]
    fn lhp_senior_tranche_less_than_equity() {
        let model = GaussianCopulaLHP::new(125, 0.3, 0.4, 0.05);
        let equity = model.tranche_expected_loss(0.0, 0.03);
        let senior = model.tranche_expected_loss(0.22, 1.0);
        assert!(
            senior < equity,
            "Senior tranche ({senior}) should have less loss than equity ({equity})"
        );
    }

    #[test]
    fn lhp_expected_loss_increases_with_correlation() {
        // Higher correlation → fatter tails → more equity loss
        let low = GaussianCopulaLHP::new(125, 0.1, 0.4, 0.05);
        let high = GaussianCopulaLHP::new(125, 0.5, 0.4, 0.05);
        let _el_low = low.tranche_expected_loss(0.0, 0.03);
        let _el_high = high.tranche_expected_loss(0.0, 0.03);
        // For equity, higher correlation actually reduces expected loss
        // (more tail but less middle), and vice for senior
        let sen_low = low.tranche_expected_loss(0.22, 1.0);
        let sen_high = high.tranche_expected_loss(0.22, 1.0);
        assert!(
            sen_high > sen_low,
            "Senior tranche loss should increase with correlation: low={sen_low}, high={sen_high}"
        );
    }

    #[test]
    fn lhp_loss_distribution_monotone() {
        let model = GaussianCopulaLHP::new(125, 0.3, 0.4, 0.05);
        let dist = model.loss_distribution(50);
        for i in 1..dist.len() {
            assert!(
                dist[i].1 >= dist[i - 1].1 - 1e-10,
                "CDF should be non-decreasing"
            );
        }
    }

    #[test]
    fn nth_to_default_first_most_expensive() {
        let n = 5;
        let probs = vec![0.05; n];
        let first = nth_to_default_mc(n, &probs, 0.3, 0.4, 1, 1.0, 10000, 42);
        let last = nth_to_default_mc(n, &probs, 0.3, 0.4, 5, 1.0, 10000, 42);
        assert!(
            first.price > last.price,
            "First-to-default ({}) should cost more than 5th-to-default ({})",
            first.price,
            last.price
        );
    }

    #[test]
    fn nth_to_default_positive() {
        let n = 5;
        let probs = vec![0.05; n];
        let result = nth_to_default_mc(n, &probs, 0.3, 0.4, 1, 1.0, 10000, 42);
        assert!(result.price > 0.0, "NTD price should be positive: {}", result.price);
    }

    #[test]
    fn cds_option_payer_positive() {
        let price = cds_option_black(0.01, 0.01, 0.40, 1.0, 4.0, true);
        assert!(price > 0.0, "ATM payer CDS option should be positive: {price}");
    }

    #[test]
    fn cds_option_put_call_parity() {
        let fwd = 0.012;
        let k = 0.010;
        let rpv01 = 4.0;
        let payer = cds_option_black(fwd, k, 0.40, 1.0, rpv01, true);
        let receiver = cds_option_black(fwd, k, 0.40, 1.0, rpv01, false);
        let intrinsic = rpv01 * (fwd - k);
        assert_abs_diff_eq!(payer - receiver, intrinsic, epsilon = 1e-10);
    }

    #[test]
    fn cds_option_increases_with_vol() {
        let p1 = cds_option_black(0.01, 0.01, 0.20, 1.0, 4.0, true);
        let p2 = cds_option_black(0.01, 0.01, 0.60, 1.0, 4.0, true);
        assert!(p2 > p1, "CDS option should increase with vol: {p1} vs {p2}");
    }
}
