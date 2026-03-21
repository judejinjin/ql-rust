//! Credit portfolio framework.
//!
//! Provides types and models for credit portfolio analysis:
//! - Issuers with default probability and recovery rate
//! - Credit baskets (collections of credits)
//! - Loss distribution models (Gaussian copula, semi-analytic)
//! - CDO tranche pricing beyond what's in `portfolio_credit`
//!
//! Reference:
//! - QuantLib: DefaultProbKey, Issuer, Pool, Basket (credit/)
//! - Li, D.X. (2000), "On Default Correlation"
//! - Hull & White (2004), "Valuation of CDOs"

use serde::{Deserialize, Serialize};
use ql_math::distributions::cumulative_normal;
use std::f64::consts::PI;

/// A credit issuer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Issuer {
    /// Issuer identifier / name.
    pub name: String,
    /// Notional exposure.
    pub notional: f64,
    /// Default probability to maturity.
    pub default_probability: f64,
    /// Recovery rate.
    pub recovery_rate: f64,
    /// Seniority / ranking (lower = more senior).
    pub seniority: u32,
}

/// A pool / basket of credits.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreditBasket {
    /// Collection of issuers.
    pub issuers: Vec<Issuer>,
    /// Pairwise correlation (flat for homogeneous pool).
    pub correlation: f64,
    /// Maturity in years.
    pub maturity: f64,
}

impl CreditBasket {
    /// Total notional of the basket.
    pub fn total_notional(&self) -> f64 {
        self.issuers.iter().map(|i| i.notional).sum()
    }

    /// Number of names in the basket.
    pub fn size(&self) -> usize {
        self.issuers.len()
    }

    /// Average default probability.
    pub fn average_default_prob(&self) -> f64 {
        if self.issuers.is_empty() { return 0.0; }
        self.issuers.iter().map(|i| i.default_probability).sum::<f64>() / self.issuers.len() as f64
    }

    /// Average recovery rate.
    pub fn average_recovery(&self) -> f64 {
        if self.issuers.is_empty() { return 0.0; }
        self.issuers.iter().map(|i| i.recovery_rate).sum::<f64>() / self.issuers.len() as f64
    }
}

/// CDO tranche specification.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CdoTranche {
    /// Attachment point (lower bound of losses as fraction of portfolio).
    pub attachment: f64,
    /// Detachment point.
    pub detachment: f64,
    /// Running spread (for pricing).
    pub spread: f64,
    /// Notional = (detachment − attachment) × portfolio_notional.
    pub notional: f64,
}

/// Loss distribution result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LossDistribution {
    /// Loss levels (as fraction of total notional).
    pub loss_levels: Vec<f64>,
    /// Cumulative probabilities P(Loss ≤ l).
    pub cumulative_probs: Vec<f64>,
    /// Expected loss.
    pub expected_loss: f64,
    /// Standard deviation of loss.
    pub loss_std: f64,
    /// VaR at 99% confidence.
    pub var_99: f64,
}

/// CDO tranche pricing result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CdoTrancheResult {
    /// Expected tranche loss.
    pub expected_loss: f64,
    /// Fair spread (running spread making tranche MTM = 0).
    pub fair_spread: f64,
    /// Protection leg PV.
    pub protection_leg: f64,
    /// Premium leg PV (per bp running).
    pub premium_leg: f64,
    /// Tranche delta (dPV/dCorrelation).
    pub delta: f64,
}

/// Compute the portfolio loss distribution using the Gaussian copula
/// (Large Homogeneous Portfolio approximation).
///
/// Uses the Vasicek model to compute the conditional default distribution.
pub fn loss_distribution_lhp(
    basket: &CreditBasket,
    n_points: usize,
) -> LossDistribution {
    let n = basket.size();
    if n == 0 {
        return LossDistribution {
            loss_levels: vec![], cumulative_probs: vec![],
            expected_loss: 0.0, loss_std: 0.0, var_99: 0.0,
        };
    }

    let pd = basket.average_default_prob();
    let rho = basket.correlation;
    let recovery = basket.average_recovery();
    let lgd = 1.0 - recovery;

    let n_pts = n_points.max(10);
    let mut loss_levels = Vec::with_capacity(n_pts + 1);
    let mut cum_probs = Vec::with_capacity(n_pts + 1);

    for i in 0..=n_pts {
        let loss_frac = i as f64 / n_pts as f64;
        loss_levels.push(loss_frac);

        // P(Loss ≤ l) = P(# defaults ≤ l / lgd × N)
        // Under LHP Gaussian copula:
        // P(PortfolioLoss ≤ l) = Φ((√(1−ρ)·Φ⁻¹(l/lgd) − Φ⁻¹(pd)) / √ρ)
        let default_frac = if lgd > 1e-10 { (loss_frac / lgd).min(1.0) } else { loss_frac };

        if default_frac >= 1.0 {
            cum_probs.push(1.0);
        } else if default_frac <= 0.0 {
            cum_probs.push(0.0);
        } else {
            let inv_default = inv_cumulative_normal(default_frac);
            let inv_pd = inv_cumulative_normal(pd);
            let sqrt_rho = rho.sqrt().max(1e-10);
            let sqrt_1_rho = (1.0 - rho).sqrt();

            let z = (sqrt_1_rho * inv_default - inv_pd) / sqrt_rho;
            cum_probs.push(cumulative_normal(z));
        }
    }

    // Expected loss = pd × lgd
    let expected_loss = pd * lgd;

    // VaR at 99%
    let var_99 = {
        let target = 0.99;
        let inv_target = inv_cumulative_normal(target);
        let inv_pd = inv_cumulative_normal(pd);
        let sqrt_rho = rho.sqrt().max(1e-10);
        let conditional_pd = cumulative_normal(
            (inv_pd + sqrt_rho * inv_target) / (1.0 - rho).sqrt()
        );
        conditional_pd * lgd
    };

    // Loss std dev (approximate)
    let loss_var = {
        // E[L²] - E[L]²
        // For Gaussian copula: Var(L) = pd²·lgd²·(P(2) - pd²)/pd² where P(2) is
        // the joint default probability
        let inv_pd = inv_cumulative_normal(pd);
        let p2 = bivariate_normal_cdf(inv_pd, inv_pd, rho);
        lgd * lgd * (p2 - pd * pd)
    };
    let loss_std = loss_var.abs().sqrt();

    LossDistribution {
        loss_levels,
        cumulative_probs: cum_probs,
        expected_loss,
        loss_std,
        var_99,
    }
}

/// Price a CDO tranche using the Gaussian copula LHP model.
///
/// # Arguments
/// - `basket` — credit basket
/// - `tranche` — tranche specification (attachment, detachment)
/// - `risk_free_rate` — continuous risk-free rate
/// - `n_integration` — number of Gauss-Hermite quadrature points
pub fn price_cdo_tranche(
    basket: &CreditBasket,
    tranche: &CdoTranche,
    risk_free_rate: f64,
    n_integration: usize,
) -> CdoTrancheResult {
    let pd = basket.average_default_prob();
    let rho = basket.correlation;
    let recovery = basket.average_recovery();
    let lgd = 1.0 - recovery;
    let t = basket.maturity;
    let df = (-risk_free_rate * t).exp();

    let a = tranche.attachment;
    let d = tranche.detachment;
    let width = d - a;

    if width <= 0.0 {
        return CdoTrancheResult {
            expected_loss: 0.0, fair_spread: 0.0,
            protection_leg: 0.0, premium_leg: 0.0, delta: 0.0,
        };
    }

    // Expected tranche loss = E[min(L, d) − min(L, a)] via numerical integration
    // Use Gauss-Hermite quadrature over the systematic factor
    let n_pts = n_integration.max(10);
    let mut el = 0.0;
    let _sqrt_2 = 2.0_f64.sqrt();

    for i in 0..n_pts {
        // Gauss-Hermite points (simplified: uniform grid over normal)
        let xi = -4.0 + 8.0 * (i as f64 + 0.5) / n_pts as f64;
        let wi = norm_pdf(xi) * 8.0 / n_pts as f64;

        // Conditional default probability
        let inv_pd = inv_cumulative_normal(pd.clamp(1e-10, 1.0 - 1e-10));
        let sqrt_rho = rho.sqrt().max(1e-10);
        let cond_pd = cumulative_normal(
            (inv_pd - sqrt_rho * xi) / (1.0 - rho).sqrt().max(1e-10)
        );

        // Conditional portfolio loss = cond_pd × lgd
        let cond_loss = cond_pd * lgd;

        // Tranche loss = min(max(loss - a, 0), d - a) / (d - a)
        let tranche_loss = ((cond_loss - a).max(0.0)).min(width) / width;
        el += wi * tranche_loss;
    }

    let expected_loss = el;
    let protection_leg = expected_loss * tranche.notional * df;
    let premium_leg = (1.0 - expected_loss) * tranche.notional * t * df;

    let fair_spread = if premium_leg.abs() > 1e-8 {
        protection_leg / premium_leg
    } else { 0.0 };

    // Delta: bump correlation by 1bp and reprice
    let bump = 0.0001;
    let mut basket_up = basket.clone();
    basket_up.correlation = (basket.correlation + bump).min(0.9999);
    let el_up = price_cdo_expected_loss(&basket_up, tranche, n_integration);
    let delta = (el_up - expected_loss) / bump;

    CdoTrancheResult {
        expected_loss,
        fair_spread,
        protection_leg,
        premium_leg,
        delta,
    }
}

fn price_cdo_expected_loss(basket: &CreditBasket, tranche: &CdoTranche, n_pts: usize) -> f64 {
    let pd = basket.average_default_prob();
    let rho = basket.correlation;
    let lgd = 1.0 - basket.average_recovery();
    let a = tranche.attachment;
    let d = tranche.detachment;
    let width = d - a;
    if width <= 0.0 { return 0.0; }

    let mut el = 0.0;
    for i in 0..n_pts {
        let xi = -4.0 + 8.0 * (i as f64 + 0.5) / n_pts as f64;
        let wi = norm_pdf(xi) * 8.0 / n_pts as f64;
        let inv_pd = inv_cumulative_normal(pd.clamp(1e-10, 1.0 - 1e-10));
        let cond_pd = cumulative_normal(
            (inv_pd - rho.sqrt().max(1e-10) * xi) / (1.0 - rho).sqrt().max(1e-10)
        );
        let cond_loss = cond_pd * lgd;
        let tl = ((cond_loss - a).max(0.0)).min(width) / width;
        el += wi * tl;
    }
    el
}

fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Inverse cumulative normal (rational approximation, Beasley-Springer-Moro).
fn inv_cumulative_normal(p: f64) -> f64 {
    if p <= 0.0 { return -8.0; }
    if p >= 1.0 { return 8.0; }
    if (p - 0.5).abs() < 1e-14 { return 0.0; }

    let a = [
        -3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383_577_518_672_69e2,
        -3.066479806614716e+01, 2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
        4.374664141464968e+00, 2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
    if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    }
    let q = (-2.0 * (1.0 - p).ln()).sqrt();
    -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
     ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
}

/// Bivariate standard normal CDF (Drezner-Wesolowsky approximation).
fn bivariate_normal_cdf(x: f64, y: f64, rho: f64) -> f64 {
    if rho.abs() < 1e-10 {
        return cumulative_normal(x) * cumulative_normal(y);
    }
    if (rho - 1.0).abs() < 1e-10 {
        return cumulative_normal(x.min(y));
    }
    if (rho + 1.0).abs() < 1e-10 {
        if x + y >= 0.0 { return (cumulative_normal(x) + cumulative_normal(y) - 1.0).max(0.0); }
        else { return 0.0; }
    }

    // Gauss-Legendre quadrature approximation
    let _det = (1.0 - rho * rho).sqrt();
    let mut sum = 0.0;
    let n = 20;
    for i in 0..n {
        let s = -1.0 + (2.0 * i as f64 + 1.0) / n as f64;
        let t = rho / 2.0 * (s + 1.0);
        let r = (1.0 - t * t).sqrt();
        let z = (x - t * y) / r;
        sum += norm_pdf(y) * cumulative_normal(z) * (rho / n as f64);
    }
    cumulative_normal(x) * cumulative_normal(y) + sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn sample_basket() -> CreditBasket {
        let issuers: Vec<Issuer> = (0..100).map(|i| Issuer {
            name: format!("Issuer_{}", i),
            notional: 1_000_000.0,
            default_probability: 0.02,
            recovery_rate: 0.40,
            seniority: 1,
        }).collect();
        CreditBasket {
            issuers,
            correlation: 0.30,
            maturity: 5.0,
        }
    }

    #[test]
    fn test_loss_distribution() {
        let basket = sample_basket();
        let ld = loss_distribution_lhp(&basket, 100);
        assert!(!ld.loss_levels.is_empty());
        assert!(ld.expected_loss > 0.0 && ld.expected_loss < 1.0);
        // Expected loss ≈ pd × lgd = 0.02 × 0.60 = 0.012
        assert_abs_diff_eq!(ld.expected_loss, 0.012, epsilon = 0.002);
        assert!(ld.var_99 > ld.expected_loss, "VaR > EL");
    }

    #[test]
    fn test_cdo_tranche_equity() {
        let basket = sample_basket();
        let equity = CdoTranche {
            attachment: 0.0,
            detachment: 0.03,
            spread: 0.05,
            notional: 3_000_000.0,
        };
        let res = price_cdo_tranche(&basket, &equity, 0.03, 50);
        assert!(res.expected_loss > 0.0, "el={}", res.expected_loss);
        assert!(res.fair_spread > 0.0, "spread={}", res.fair_spread);
    }

    #[test]
    fn test_cdo_tranche_senior() {
        let basket = sample_basket();
        let senior = CdoTranche {
            attachment: 0.15,
            detachment: 1.0,
            spread: 0.001,
            notional: 85_000_000.0,
        };
        let res = price_cdo_tranche(&basket, &senior, 0.03, 50);
        // Senior tranche should have very low expected loss
        assert!(res.expected_loss < 0.05, "el={}", res.expected_loss);
    }

    #[test]
    fn test_basket_properties() {
        let basket = sample_basket();
        assert_eq!(basket.size(), 100);
        assert_abs_diff_eq!(basket.total_notional(), 100_000_000.0, epsilon = 1.0);
        assert_abs_diff_eq!(basket.average_default_prob(), 0.02, epsilon = 1e-10);
        assert_abs_diff_eq!(basket.average_recovery(), 0.40, epsilon = 1e-10);
    }

    #[test]
    fn test_inv_cumulative_normal() {
        assert_abs_diff_eq!(inv_cumulative_normal(0.5), 0.0, epsilon = 1e-8);
        assert!(inv_cumulative_normal(0.975) > 1.9 && inv_cumulative_normal(0.975) < 2.0);
        assert!(inv_cumulative_normal(0.025) < -1.9 && inv_cumulative_normal(0.025) > -2.0);
    }
}
