//! Integral CDS engine and risky bond engine.
//!
//! - **Integral CDS engine**: prices CDS using numerical integration
//!   (trapezoidal rule) over the full default probability curve, rather than
//!   the midpoint approximation used in the standard engine.
//!
//! - **Risky bond engine**: prices a fixed-rate bond accounting for issuer
//!   credit risk (default risk + recovery), producing a credit-adjusted NPV.
//!
//! Reference:
//! - QuantLib: IntegralCdsEngine, RiskyBondEngine

use serde::{Deserialize, Serialize};

/// Result from the integral CDS engine.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntegralCdsResult {
    /// Present value of the protection (default) leg.
    pub protection_leg_pv: f64,
    /// Present value of the premium (fee) leg.
    pub premium_leg_pv: f64,
    /// NPV = protection_leg − premium_leg (buyer perspective).
    pub npv: f64,
    /// Fair spread: spread making NPV = 0.
    pub fair_spread: f64,
    /// Accrued premium at default (expected).
    pub accrual_pv: f64,
    /// RPV01 (risky PV of 1 bp running).
    pub rpv01: f64,
}

/// Price a CDS using numerical integration over the default probability curve.
///
/// # Arguments
/// - `notional` — CDS notional
/// - `spread` — running CDS spread (annual, e.g. 0.01 = 100 bps)
/// - `recovery_rate` — assumed recovery rate (e.g. 0.40)
/// - `payment_times` — year fractions to premium payment dates
/// - `discount_factors` — discount factors at each payment date
/// - `survival_probs` — survival probabilities at each payment date
/// - `accrual_fractions` — accrual period fractions for each coupon
/// - `n_integration_points` — number of integration steps per period
#[allow(clippy::too_many_arguments)]
pub fn integral_cds_engine(
    notional: f64,
    spread: f64,
    recovery_rate: f64,
    payment_times: &[f64],
    discount_factors: &[f64],
    survival_probs: &[f64],
    accrual_fractions: &[f64],
    n_integration_points: usize,
) -> IntegralCdsResult {
    let n = payment_times.len()
        .min(discount_factors.len())
        .min(survival_probs.len())
        .min(accrual_fractions.len());
    let n_points = n_integration_points.max(2);

    if n == 0 {
        return IntegralCdsResult {
            protection_leg_pv: 0.0, premium_leg_pv: 0.0, npv: 0.0,
            fair_spread: 0.0, accrual_pv: 0.0, rpv01: 0.0,
        };
    }

    // Protection leg: ∫₀ᵀ (1−R) · DF(t) · (−dQ/dt) dt
    // Use trapezoidal integration over sub-intervals
    let lgd = 1.0 - recovery_rate;
    let mut protection_pv = 0.0;

    // Integrate over each period
    let mut prev_t = 0.0;
    let mut prev_df = 1.0;
    let mut prev_sp = 1.0;

    for i in 0..n {
        let t_end = payment_times[i];
        let df_end = discount_factors[i];
        let sp_end = survival_probs[i];

        let dt = (t_end - prev_t) / n_points as f64;
        for j in 1..=n_points {
            let t_j = prev_t + j as f64 * dt;
            let w = (t_j - prev_t) / (t_end - prev_t).max(1e-12);
            let _df_j = prev_df * (1.0 - w) + df_end * w; // linear interp
            let sp_j = prev_sp * (1.0 - w) + sp_end * w;

            let t_jm1 = prev_t + (j as f64 - 1.0) * dt;
            let w_prev = (t_jm1 - prev_t) / (t_end - prev_t).max(1e-12);
            let sp_jm1 = prev_sp * (1.0 - w_prev) + sp_end * w_prev;
            let df_mid = prev_df * (1.0 - (w + w_prev) / 2.0) + df_end * (w + w_prev) / 2.0;

            // Default probability in [t_{j-1}, t_j] ≈ SP(t_{j-1}) − SP(t_j)
            let dp = sp_jm1 - sp_j;
            protection_pv += lgd * df_mid * dp;
        }

        prev_t = t_end;
        prev_df = df_end;
        prev_sp = sp_end;
    }
    protection_pv *= notional;

    // Premium leg: Σ spread · τ_i · DF_i · Q(t_i)
    let mut premium_pv = 0.0;
    for i in 0..n {
        premium_pv += spread * accrual_fractions[i] * discount_factors[i] * survival_probs[i];
    }
    premium_pv *= notional;

    // Accrued premium on default (simplified: half-period accrual)
    let mut accrual_pv = 0.0;
    prev_sp = 1.0;
    for i in 0..n {
        let dp = prev_sp - survival_probs[i];
        accrual_pv += spread * 0.5 * accrual_fractions[i] * discount_factors[i] * dp;
        prev_sp = survival_probs[i];
    }
    accrual_pv *= notional;
    premium_pv += accrual_pv;

    let npv = protection_pv - premium_pv;

    // RPV01 = Σ τ_i · DF_i · Q(t_i) + accrual
    let rpv01 = premium_pv / (spread * notional).max(1e-14);

    let fair_spread = if rpv01.abs() > 1e-14 {
        protection_pv / (rpv01 * notional)
    } else { 0.0 };

    IntegralCdsResult {
        protection_leg_pv: protection_pv,
        premium_leg_pv: premium_pv,
        npv,
        fair_spread,
        accrual_pv,
        rpv01,
    }
}

/// Result from the risky bond engine.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RiskyBondResult {
    /// Risk-free NPV (ignoring default).
    pub risk_free_npv: f64,
    /// Credit-adjusted NPV (accounting for default).
    pub credit_adjusted_npv: f64,
    /// Credit spread implied by the adjustment.
    pub implied_credit_spread: f64,
    /// Default leg (expected loss from default).
    pub default_leg: f64,
    /// Survival-weighted clean price.
    pub clean_price: f64,
    /// Dirty price.
    pub dirty_price: f64,
}

/// Price a fixed-rate bond with issuer credit risk.
///
/// # Arguments
/// - `notional` — bond face value
/// - `coupon_rate` — annual coupon rate
/// - `payment_times` — year fractions to each coupon payment
/// - `discount_factors` — risk-free discount factors
/// - `survival_probs` — issuer survival probabilities at each date
/// - `recovery_rate` — recovery rate on default
/// - `accrual_fractions` — coupon accrual fractions
/// - `accrued_interest` — currently accrued interest
#[allow(clippy::too_many_arguments)]
pub fn risky_bond_engine(
    notional: f64,
    coupon_rate: f64,
    payment_times: &[f64],
    discount_factors: &[f64],
    survival_probs: &[f64],
    recovery_rate: f64,
    accrual_fractions: &[f64],
    accrued_interest: f64,
) -> RiskyBondResult {
    let n = payment_times.len()
        .min(discount_factors.len())
        .min(survival_probs.len())
        .min(accrual_fractions.len());

    if n == 0 {
        return RiskyBondResult {
            risk_free_npv: 0.0, credit_adjusted_npv: 0.0,
            implied_credit_spread: 0.0, default_leg: 0.0,
            clean_price: 0.0, dirty_price: 0.0,
        };
    }

    // Risk-free NPV: Σ coupon · DF_i + notional · DF_n
    let mut rf_npv = 0.0;
    for i in 0..n {
        rf_npv += coupon_rate * accrual_fractions[i] * discount_factors[i] * notional;
    }
    rf_npv += notional * discount_factors[n - 1]; // redemption

    // Survival-weighted coupon leg
    let mut survival_coupon = 0.0;
    for i in 0..n {
        survival_coupon += coupon_rate * accrual_fractions[i] * discount_factors[i] * survival_probs[i] * notional;
    }
    // Survival-weighted redemption
    let survival_redemption = notional * discount_factors[n - 1] * survival_probs[n - 1];

    // Default leg: expected recovery on default
    let mut recovery_value = 0.0;
    let mut prev_sp = 1.0;
    for i in 0..n {
        let dp = prev_sp - survival_probs[i];
        recovery_value += recovery_rate * notional * discount_factors[i] * dp;
        prev_sp = survival_probs[i];
    }

    let credit_adj_npv = survival_coupon + survival_redemption + recovery_value;
    let default_leg = rf_npv - credit_adj_npv + recovery_value;

    // Implied credit spread
    let duration = {
        let mut d = 0.0;
        for i in 0..n {
            d += payment_times[i] * accrual_fractions[i] * discount_factors[i] * survival_probs[i];
        }
        d + payment_times[n - 1] * discount_factors[n - 1] * survival_probs[n - 1]
    };
    let implied_spread = if duration.abs() > 1e-12 {
        (rf_npv - credit_adj_npv) / (duration * notional)
    } else { 0.0 };

    let dirty_price = credit_adj_npv / notional * 100.0;
    let clean_price = dirty_price - accrued_interest / notional * 100.0;

    RiskyBondResult {
        risk_free_npv: rf_npv,
        credit_adjusted_npv: credit_adj_npv,
        implied_credit_spread: implied_spread,
        default_leg,
        clean_price,
        dirty_price,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_integral_cds_zero_spread() {
        let times = vec![0.5, 1.0, 1.5, 2.0];
        let dfs = vec![0.975, 0.95, 0.925, 0.90];
        let sps = vec![0.99, 0.98, 0.97, 0.96];
        let taus = vec![0.5; 4];
        let res = integral_cds_engine(
            10_000_000.0, 0.0, 0.40,
            &times, &dfs, &sps, &taus, 10,
        );
        // With zero spread, premium leg is zero, NPV = protection leg
        assert!(res.protection_leg_pv > 0.0);
        assert_abs_diff_eq!(res.premium_leg_pv, 0.0, epsilon = 1.0);
    }

    #[test]
    fn test_integral_cds_fair_spread() {
        let times = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let r: f64 = 0.03;
        let h: f64 = 0.02; // hazard rate
        let dfs: Vec<f64> = times.iter().map(|&t| (-r * t).exp()).collect();
        let sps: Vec<f64> = times.iter().map(|&t| (-h * t).exp()).collect();
        let taus = vec![0.5; 6];
        let res = integral_cds_engine(
            10_000_000.0, 0.01, 0.40,
            &times, &dfs, &sps, &taus, 20,
        );
        assert!(res.fair_spread > 0.0, "fair={}", res.fair_spread);
        // Fair spread ≈ h·(1−R) ≈ 0.02 × 0.60 = 0.012 = 120 bps
        assert!(res.fair_spread > 0.005 && res.fair_spread < 0.025,
            "fair_spread={}", res.fair_spread);
    }

    #[test]
    fn test_risky_bond_no_default() {
        let times = vec![0.5, 1.0, 1.5, 2.0];
        let dfs = vec![0.975, 0.95, 0.925, 0.90];
        let sps = vec![1.0; 4]; // No default risk
        let taus = vec![0.5; 4];
        let res = risky_bond_engine(
            100.0, 0.05, &times, &dfs, &sps, 0.40, &taus, 0.0,
        );
        // With no default, credit-adjusted = risk-free
        assert_abs_diff_eq!(res.credit_adjusted_npv, res.risk_free_npv, epsilon = 0.01);
        assert_abs_diff_eq!(res.implied_credit_spread, 0.0, epsilon = 0.001);
    }

    #[test]
    fn test_risky_bond_with_default_risk() {
        let times = vec![0.5, 1.0, 1.5, 2.0];
        let r: f64 = 0.04;
        let h: f64 = 0.03;
        let dfs: Vec<f64> = times.iter().map(|&t| (-r * t).exp()).collect();
        let sps: Vec<f64> = times.iter().map(|&t| (-h * t).exp()).collect();
        let taus = vec![0.5; 4];
        let res = risky_bond_engine(
            100.0, 0.05, &times, &dfs, &sps, 0.40, &taus, 1.25,
        );
        // With credit risk, credit-adjusted < risk-free
        assert!(res.credit_adjusted_npv < res.risk_free_npv,
            "adj={}, rf={}", res.credit_adjusted_npv, res.risk_free_npv);
        assert!(res.implied_credit_spread > 0.0);
        assert!(res.clean_price > 0.0 && res.clean_price < 150.0);
    }
}
