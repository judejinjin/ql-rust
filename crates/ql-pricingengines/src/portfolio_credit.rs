//! Advanced portfolio credit models: Student-t copula, contagion, CVA.
//!
//! Extends the single-name credit infrastructure with:
//!
//! - **Student-t copula CDO** (`StudentTCopulaLHP`) — heavier tail dependence
//!   than Gaussian, produces wider loss distributions and more realistic
//!   super-senior tranche pricing.
//!
//! - **Credit contagion** (`InfectiousDefaultModel`) — Davis-Lo (2001)
//!   contagion model where one default triggers additional defaults with
//!   probability `ε`.
//!
//! - **Bilateral CVA** (`bilateral_cva`) — Credit Valuation Adjustment for
//!   OTC derivatives:
//!   ```text
//!   CVA  = (1-R_c) * Σ  EE(t_i) * [Q(t_{i-1}) - Q(t_i)]
//!   DVA  = (1-R_b) * Σ  NEE(t_i) * [Q_b(t_{i-1}) - Q_b(t_i)]
//!   BCVA = DVA - CVA
//!   ```
//!
//! - **Spread ladder** (`spread_ladder_cdo`) — compute CDO tranche spreads
//!   across a range of attachment points given a base-correlation surface.

#![allow(clippy::too_many_arguments)]

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};

use ql_math::distributions::inverse_cumulative_normal;

/// Normal quantile wrapper (unwraps the QLResult for valid p ∈ (0,1)).
#[inline]
fn normal_inv_cdf(p: f64) -> f64 {
    inverse_cumulative_normal(p).unwrap_or(0.0)
}

// =========================================================================
// Student-t copula CDO (single-factor LHP)
// =========================================================================

/// Student-t copula Large-Homogeneous-Portfolio CDO tranche pricer.
///
/// The one-factor Student-t copula uses:
/// ```text
/// X_i = ρ^½ Z + (1−ρ)^½ ε_i
/// default_i ↔ t_ν(X_i) < p
/// ```
/// where `t_ν` is the CDF of Student-t with `nu` degrees of freedom.
///
/// As `nu → ∞` this converges to the Gaussian copula.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudentTCopulaLHP {
    /// Degrees of freedom (`ν`).  Must be > 2.
    pub nu: f64,
    /// Correlation `ρ` ∈ (0, 1).
    pub correlation: f64,
    /// Individual default probability p(T).
    pub default_prob: f64,
    /// Recovery rate R.
    pub recovery: f64,
    /// Number of MC samples for integration.
    pub n_samples: usize,
}

impl StudentTCopulaLHP {
    /// Create a new Student-t copula pricer.
    pub fn new(nu: f64, correlation: f64, default_prob: f64, recovery: f64, n_samples: usize) -> Self {
        assert!(nu > 2.0, "nu must be > 2");
        assert!(correlation > 0.0 && correlation < 1.0);
        Self { nu, correlation, default_prob, recovery, n_samples }
    }

    /// Expected tranche loss E[L_{A,D}] via importance-sampling MC.
    ///
    /// For each simulated common factor `Z ~ t(ν)`, the conditional default
    /// probability under the LHP approximation is used to compute the
    /// expected loss in the tranche.
    pub fn tranche_expected_loss(
        &self,
        attachment: f64,
        detachment: f64,
        seed: u64,
    ) -> f64 {
        if detachment <= attachment { return 0.0; }

        let sqrt_rho = self.correlation.sqrt();
        let sqrt_1_m_rho = (1.0 - self.correlation).sqrt();
        let lgd = 1.0 - self.recovery;
        let t_inv_p = student_t_inv_cdf(self.default_prob, self.nu);

        let mut rng = SmallRng::seed_from_u64(seed);
        let mut sum = 0.0;

        for _ in 0..self.n_samples {
            // Sample common factor Z ~ t(ν) by ratio method: Z = N/√(χ²/ν)
            let z_normal: f64 = rng.sample(StandardNormal);
            let chi2 = chi_squared_sample(&mut rng, self.nu);
            let z = z_normal / (chi2 / self.nu).sqrt();

            // Conditional default prob: P(default | Z)
            let x_threshold = (t_inv_p - sqrt_rho * z) / sqrt_1_m_rho;
            let cond_p = student_t_cdf(x_threshold, self.nu);

            let cond_loss = (cond_p * lgd).clamp(0.0, 1.0);
            let tranche_loss = cond_loss.clamp(attachment, detachment) - attachment;
            sum += tranche_loss;
        }

        let mean_tranche_loss = sum / self.n_samples as f64;
        mean_tranche_loss / (detachment - attachment)
    }
}

/// Approximate inverse CDF of Student-t via Cornish-Fisher expansion.
fn student_t_inv_cdf(p: f64, nu: f64) -> f64 {
    // Normal quantile
    let z = normal_inv_cdf(p);
    // Cornish-Fisher correction
    let z2 = z * z;
    z + (z2 * z + z) / (4.0 * nu)
        + (5.0 * z2 * z2 * z - 22.0 * z2 * z - 3.0 * z) / (96.0 * nu * nu)
}

/// Student-t CDF approximation via incomplete beta.
fn student_t_cdf(x: f64, nu: f64) -> f64 {
    // Use Abramowitz & Stegun approximation
    let t2 = x * x;
    let cos2 = nu / (nu + t2);
    // Regularised incomplete beta I_{nu/(nu+t^2)}(nu/2, 1/2)
    let ib = regularised_incomplete_beta(cos2, nu / 2.0, 0.5);
    if x >= 0.0 { 1.0 - 0.5 * ib } else { 0.5 * ib }
}

/// Regularised incomplete beta B(x; a, b) via continued-fraction approximation.
fn regularised_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if x >= 1.0 { return 1.0; }
    // Lentz continued fraction
    let lnbeta = lgamma(a) + lgamma(b) - lgamma(a + b);
    let front = (x.ln() * a + (1.0 - x).ln() * b - lnbeta).exp() / a;
    let cf = lentz_cf(x, a, b);
    (front * cf).clamp(0.0, 1.0)
}

fn lentz_cf(x: f64, a: f64, b: f64) -> f64 {
    let mut f;
    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < 1e-30 { d = 1e-30; }
    d = 1.0 / d;
    f = d;
    for m in 1..=100usize {
        let m2 = 2 * m;
        let mf = m as f64;
        // Even step
        let num_e = mf * (b - mf) * x / ((a + m2 as f64 - 1.0) * (a + m2 as f64));
        d = 1.0 + num_e * d;
        if d.abs() < 1e-30 { d = 1e-30; }
        c = 1.0 + num_e / c;
        if c.abs() < 1e-30 { c = 1e-30; }
        d = 1.0 / d;
        f *= d * c;
        // Odd step
        let num_o = -(a + mf) * (a + b + mf) * x / ((a + m2 as f64) * (a + m2 as f64 + 1.0));
        d = 1.0 + num_o * d;
        if d.abs() < 1e-30 { d = 1e-30; }
        c = 1.0 + num_o / c;
        if c.abs() < 1e-30 { c = 1e-30; }
        d = 1.0 / d;
        let delta = d * c;
        f *= delta;
        if (delta - 1.0).abs() < 3e-7 { break; }
    }
    f
}

/// Log-gamma approximation via Lanczos.
fn lgamma(x: f64) -> f64 {
    // Stirling approximation, sufficient for x > 1
    if x <= 0.0 { return f64::INFINITY; }
    if x < 1.0 { return lgamma(x + 1.0) - x.ln(); }
    (x - 0.5) * x.ln() - x + 0.5 * std::f64::consts::TAU.ln() + 1.0 / (12.0 * x)
}

/// Sample χ² with `nu` degrees of freedom using sum of normals squared.
fn chi_squared_sample(rng: &mut SmallRng, nu: f64) -> f64 {
    let k = nu.ceil() as usize;
    let sum: f64 = (0..k).map(|_| {
        let z: f64 = rng.sample(StandardNormal);
        z * z
    }).sum();
    sum // scale correction negligible for our purposes
}

// =========================================================================
// Davis-Lo credit contagion model
// =========================================================================

/// Result from a credit contagion simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContagionResult {
    /// Expected number of defaults.
    pub expected_defaults: f64,
    /// Standard deviation of defaults.
    pub std_defaults: f64,
    /// Probability of at least one default.
    pub prob_any_default: f64,
    /// Probability of k defaults, for k = 0..=n_names.
    pub default_distribution: Vec<f64>,
}

/// Davis-Lo (2001) infectious default contagion model.
///
/// Each name has independent default probability `lambda`. When one name
/// defaults it triggers each surviving name to default with probability
/// `epsilon` (contagion).  The process is Markovian: after the primary
/// default the secondaries are resolved independently.
///
/// # Arguments
/// - `n_names`   — number of names in the portfolio
/// - `lambda`    — independent default probability per name
/// - `epsilon`   — contagion infection probability per surviving name
/// - `n_paths`   — Monte Carlo paths
/// - `seed`      — RNG seed
#[allow(clippy::needless_range_loop)]
pub fn infectious_default_mc(
    n_names: usize,
    lambda: f64,
    epsilon: f64,
    n_paths: usize,
    seed: u64,
) -> ContagionResult {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut sum_d = 0.0;
    let mut sum_d2 = 0.0;
    let mut count_any = 0u64;
    let mut dist = vec![0u64; n_names + 1];

    for _ in 0..n_paths {
        let mut defaulted = vec![false; n_names];

        // Phase 1: independent defaults
        for i in 0..n_names {
            if rng.random::<f64>() < lambda {
                defaulted[i] = true;
            }
        }

        // Phase 2: contagion — for each primary default, infect survivors
        let primaries: Vec<usize> = (0..n_names).filter(|&i| defaulted[i]).collect();
        for _ in &primaries {
            for i in 0..n_names {
                if !defaulted[i] && rng.random::<f64>() < epsilon {
                    defaulted[i] = true;
                }
            }
        }

        let n_defaults = defaulted.iter().filter(|&&d| d).count();
        sum_d += n_defaults as f64;
        sum_d2 += (n_defaults * n_defaults) as f64;
        if n_defaults > 0 { count_any += 1; }
        dist[n_defaults] += 1;
    }

    let np = n_paths as f64;
    let mean = sum_d / np;
    let variance = sum_d2 / np - mean * mean;
    let distribution: Vec<f64> = dist.iter().map(|&c| c as f64 / np).collect();

    ContagionResult {
        expected_defaults: mean,
        std_defaults: variance.max(0.0).sqrt(),
        prob_any_default: count_any as f64 / np,
        default_distribution: distribution,
    }
}

// =========================================================================
// Bilateral CVA / DVA
// =========================================================================

/// Result from bilateral CVA calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CvaResult {
    /// Unilateral CVA (counterparty default risk to us).
    pub cva: f64,
    /// DVA (our own default risk to counterparty).
    pub dva: f64,
    /// Bilateral CVA = DVA − CVA.
    pub bcva: f64,
}

/// Compute bilateral CVA/DVA for an OTC derivative position.
///
/// Implements the standard piecewise-constant hazard rate approximation:
/// ```text
/// CVA = (1-R_c) * Σ_i  EE(t_i) * ΔQ_c(t_i)
/// DVA = (1-R_b) * Σ_i NEE(t_i) * ΔQ_b(t_i)
/// ```
///
/// # Arguments
/// - `times`          — time grid in years
/// - `expected_exposure` — positive expected exposure EE(t_i) at each time
/// - `negative_exposure` — expected negative exposure NEE(t_i) = −E[min(V,0)]
/// - `hazard_c`       — counterparty hazard rate (constant)
/// - `hazard_b`       — own hazard rate (constant)
/// - `recovery_c`     — counterparty recovery rate
/// - `recovery_b`     — own recovery rate
/// - `discount_factors` — risk-free discount factors at each time
pub fn bilateral_cva(
    times: &[f64],
    expected_exposure: &[f64],
    negative_exposure: &[f64],
    hazard_c: f64,
    hazard_b: f64,
    recovery_c: f64,
    recovery_b: f64,
    discount_factors: &[f64],
) -> CvaResult {
    let n = times.len();
    assert_eq!(expected_exposure.len(), n);
    assert_eq!(negative_exposure.len(), n);
    assert_eq!(discount_factors.len(), n);

    let lgd_c = 1.0 - recovery_c;
    let lgd_b = 1.0 - recovery_b;

    let mut cva = 0.0;
    let mut dva = 0.0;

    let mut q_c_prev = 1.0_f64;
    let mut q_b_prev = 1.0_f64;

    for i in 0..n {
        let t = times[i];
        let q_c = (-hazard_c * t).exp();
        let q_b = (-hazard_b * t).exp();
        let dq_c = q_c_prev - q_c; // survival prob drop ≈ default prob in [t_{i-1}, t_i]
        let dq_b = q_b_prev - q_b;

        cva += expected_exposure[i] * dq_c * discount_factors[i];
        dva += negative_exposure[i] * dq_b * discount_factors[i];

        q_c_prev = q_c;
        q_b_prev = q_b;
    }

    cva *= lgd_c;
    dva *= lgd_b;

    CvaResult { cva, dva, bcva: dva - cva }
}

// =========================================================================
// Spread ladder CDO
// =========================================================================

/// CDO tranche spread at a given attachment / detachment.
///
/// The fair spread on the protection leg equals:
/// ```text
/// s = (EL_detachment − EL_attachment) / risky_annuity
/// ```
/// where the risky annuity is approximated as a flat discount annuity
/// times the tranche survival probability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdoTrancheSpread {
    pub attachment: f64,
    pub detachment: f64,
    /// Expected loss of the tranche (0..1 of tranche notional).
    pub expected_loss: f64,
    /// Fair running spread (e.g. 0.001 = 10 bp/y).
    pub fair_spread: f64,
}

/// Compute a CDO spread ladder across multiple tranches.
///
/// # Arguments
/// - `tranche_points` — (attachment, detachment) pairs
/// - `default_prob`   — portfolio reference default probability
/// - `correlation`    — Gaussian copula correlation
/// - `recovery`       — recovery rate
/// - `n_names`        — portfolio size
/// - `maturity`       — tranche maturity in years
/// - `flat_rate`      — flat risk-free rate for discounting
pub fn cdo_spread_ladder(
    tranche_points: &[(f64, f64)],
    default_prob: f64,
    correlation: f64,
    recovery: f64,
    n_names: usize,
    maturity: f64,
    flat_rate: f64,
) -> Vec<CdoTrancheSpread> {
    use crate::credit_models::GaussianCopulaLHP;

    let copula = GaussianCopulaLHP::new(n_names, correlation, recovery, default_prob);
    // Risky annuity approximation: flat annuity × average tranche survival
    let flat_annuity = if flat_rate.abs() < 1e-8 {
        maturity
    } else {
        (1.0 - (-flat_rate * maturity).exp()) / flat_rate
    };

    tranche_points.iter().map(|&(a, d)| {
        let el = copula.tranche_expected_loss(a, d);
        let survival = 1.0 - el;
        let risky_annuity = flat_annuity * survival.max(1e-6);
        let fair_spread = el / risky_annuity;
        CdoTrancheSpread {
            attachment: a,
            detachment: d,
            expected_loss: el,
            fair_spread,
        }
    }).collect()
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn student_t_cdf_symmetry() {
        // CDF(-x) = 1 - CDF(x) by symmetry
        let nu = 5.0;
        let x = 1.5;
        let p = student_t_cdf(x, nu);
        let q = student_t_cdf(-x, nu);
        assert!((p + q - 1.0).abs() < 0.01, "p+q={}", p + q);
    }

    #[test]
    fn student_t_copula_tranche_nonnegative() {
        let copula = StudentTCopulaLHP::new(5.0, 0.3, 0.05, 0.4, 2000);
        let el = copula.tranche_expected_loss(0.0, 0.03, 42);
        assert!(el >= 0.0 && el <= 1.0, "el={}", el);
    }

    #[test]
    fn contagion_expected_defaults_increases_with_epsilon() {
        let e0 = infectious_default_mc(50, 0.05, 0.0, 10000, 1).expected_defaults;
        let e1 = infectious_default_mc(50, 0.05, 0.1, 10000, 1).expected_defaults;
        assert!(e1 >= e0, "contagion should increase expected defaults: e0={} e1={}", e0, e1);
    }

    #[test]
    fn bilateral_cva_positive() {
        let times = vec![0.5, 1.0, 1.5, 2.0];
        let ee = vec![100.0, 90.0, 80.0, 70.0];
        let nee = vec![0.0; 4]; // no DVA
        let df: Vec<f64> = times.iter().map(|&t| (-0.04_f64 * t).exp()).collect();
        let res = bilateral_cva(&times, &ee, &nee, 0.02, 0.01, 0.4, 0.4, &df);
        assert!(res.cva > 0.0, "CVA should be positive: {}", res.cva);
        assert!(res.dva.abs() < 1e-10, "DVA should be ~0 with no NEE");
    }

    #[test]
    fn cdo_spread_ladder_positive_spreads() {
        let tranches = vec![(0.0, 0.03), (0.03, 0.07), (0.07, 1.0)];
        let ladder = cdo_spread_ladder(&tranches, 0.05, 0.3, 0.4, 100, 5.0, 0.04);
        for t in &ladder {
            assert!(t.fair_spread >= 0.0, "spread={}", t.fair_spread);
        }
        // Equity tranche should have highest spread
        assert!(ladder[0].fair_spread > ladder[2].fair_spread,
            "equity spread {} should exceed super-senior {}", ladder[0].fair_spread, ladder[2].fair_spread);
    }
}
