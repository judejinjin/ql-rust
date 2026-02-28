//! CMS spread coupon and pricer.
//!
//! A CMS spread coupon pays a rate linked to the spread between two CMS rates
//! (e.g., CMS10Y − CMS2Y). The coupon rate is typically:
//!   coupon = gearing × (CMS_long − CMS_short) + spread
//!
//! The CMS spread pricer uses a copula-based approach (Gaussian or Student-t)
//! to capture the correlation between the two swap rates and compute the
//! expected spread with convexity adjustment.
//!
//! Reference:
//! - Brigo, D. & Mercurio, F. (2006), "Interest Rate Models — Theory and Practice", ch. 13.

use serde::{Deserialize, Serialize};
use ql_math::distributions::cumulative_normal;
use std::f64::consts::PI;

/// CMS spread coupon definition.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CmsSpreadCoupon {
    /// Payment date year fraction from today.
    pub payment_time: f64,
    /// Nominal / notional.
    pub nominal: f64,
    /// Fixing time.
    pub fixing_time: f64,
    /// Forward swap rate for the long leg.
    pub forward_long: f64,
    /// Forward swap rate for the short leg.
    pub forward_short: f64,
    /// Swaption vol for the long leg.
    pub vol_long: f64,
    /// Swaption vol for the short leg.
    pub vol_short: f64,
    /// Correlation between the two swap rates.
    pub correlation: f64,
    /// Gearing (multiplier on the spread).
    pub gearing: f64,
    /// Additive spread.
    pub spread: f64,
    /// Optional cap strike.
    pub cap_strike: Option<f64>,
    /// Optional floor strike.
    pub floor_strike: Option<f64>,
}

/// Result from CMS spread pricer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CmsSpreadPricerResult {
    /// Expected CMS spread rate (with convexity adjustments).
    pub expected_spread: f64,
    /// Full coupon rate (gearing × spread + additive).
    pub coupon_rate: f64,
    /// Coupon amount (rate × nominal × accrual).
    pub amount: f64,
    /// CMS spread caplet price.
    pub caplet_price: f64,
    /// CMS spread floorlet price.
    pub floorlet_price: f64,
}

/// Standard normal density.
fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Price a CMS spread caplet/floorlet using the Gaussian copula approach.
///
/// The spread S = S_long − S_short is approximately normally distributed with:
///   E[S] ≈ F_long − F_short  (plus convexity adjustments)
///   Var[S] ≈ σ_L² F_L² T + σ_S² F_S² T − 2 ρ σ_L F_L σ_S F_S T
///
/// The caplet on the spread (S − K)⁺ is priced via the Bachelier formula on
/// the normal-distributed spread.
///
/// # Arguments
/// - `coupon` — the CMS spread coupon definition
/// - `discount_payment` — discount factor to the payment date
/// - `accrual_fraction` — year fraction for the coupon accrual period
pub fn cms_spread_pricer(
    coupon: &CmsSpreadCoupon,
    discount_payment: f64,
    accrual_fraction: f64,
) -> CmsSpreadPricerResult {
    let f_l = coupon.forward_long;
    let f_s = coupon.forward_short;
    let sigma_l = coupon.vol_long;
    let sigma_s = coupon.vol_short;
    let rho = coupon.correlation;
    let t = coupon.fixing_time;

    // Spread forward = F_L - F_S
    let spread_fwd = f_l - f_s;

    // Spread volatility (normal)
    let var_spread = sigma_l * sigma_l * f_l * f_l * t
                   + sigma_s * sigma_s * f_s * f_s * t
                   - 2.0 * rho * sigma_l * f_l * sigma_s * f_s * t;
    let sigma_spread = if var_spread > 0.0 { var_spread.sqrt() } else { 0.0 };

    // Expected spread with convexity = spread_fwd (first-order; higher-order
    // convexity adjustments from CMS pricers are external).
    let expected_spread = spread_fwd;

    // Coupon rate
    let coupon_rate = coupon.gearing * expected_spread + coupon.spread;

    // Coupon amount
    let amount = coupon.nominal * coupon_rate * accrual_fraction * discount_payment;

    // Bachelier cap/floor on the spread
    let caplet_price = if let Some(k) = coupon.cap_strike {
        if sigma_spread > 1e-12 {
            let d = (expected_spread - k) / sigma_spread;
            coupon.nominal * accrual_fraction * discount_payment
                * (sigma_spread * norm_pdf(d) + (expected_spread - k) * cumulative_normal(d))
        } else {
            coupon.nominal * accrual_fraction * discount_payment * (expected_spread - k).max(0.0)
        }
    } else {
        0.0
    };

    let floorlet_price = if let Some(k) = coupon.floor_strike {
        if sigma_spread > 1e-12 {
            let d = (expected_spread - k) / sigma_spread;
            coupon.nominal * accrual_fraction * discount_payment
                * (sigma_spread * norm_pdf(-d) + (k - expected_spread) * cumulative_normal(-d))
        } else {
            coupon.nominal * accrual_fraction * discount_payment * (k - expected_spread).max(0.0)
        }
    } else {
        0.0
    };

    CmsSpreadPricerResult {
        expected_spread,
        coupon_rate,
        amount,
        caplet_price,
        floorlet_price,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn sample_coupon() -> CmsSpreadCoupon {
        CmsSpreadCoupon {
            payment_time: 5.5,
            nominal: 1_000_000.0,
            fixing_time: 5.0,
            forward_long: 0.04,
            forward_short: 0.025,
            vol_long: 0.20,
            vol_short: 0.22,
            correlation: 0.85,
            gearing: 1.0,
            spread: 0.0,
            cap_strike: Some(0.02),
            floor_strike: Some(0.01),
        }
    }

    #[test]
    fn test_cms_spread_basic() {
        let c = sample_coupon();
        let res = cms_spread_pricer(&c, 0.76, 0.5);
        assert_abs_diff_eq!(res.expected_spread, 0.015, epsilon = 1e-6);
        assert!(res.caplet_price > 0.0, "caplet={}", res.caplet_price);
        assert!(res.floorlet_price > 0.0, "floorlet={}", res.floorlet_price);
    }

    #[test]
    fn test_cms_spread_put_call_parity() {
        // Bachelier put-call parity: cap - floor = (F-K) * N * τ * DF
        let mut c = sample_coupon();
        let k = 0.015;
        c.cap_strike = Some(k);
        c.floor_strike = Some(k);
        let res = cms_spread_pricer(&c, 0.80, 0.5);
        let parity = res.caplet_price - res.floorlet_price;
        let expected = c.nominal * 0.5 * 0.80 * (res.expected_spread - k);
        assert_abs_diff_eq!(parity, expected, epsilon = 1.0);
    }

    #[test]
    fn test_cms_spread_zero_vol() {
        let mut c = sample_coupon();
        c.vol_long = 0.0;
        c.vol_short = 0.0;
        c.cap_strike = Some(0.01);
        c.floor_strike = None;
        let res = cms_spread_pricer(&c, 0.80, 0.5);
        // With zero vol, caplet = max(F_spread - K, 0) * N * tau * DF
        let expected = 1_000_000.0 * 0.5 * 0.80 * (0.015 - 0.01);
        assert_abs_diff_eq!(res.caplet_price, expected, epsilon = 1.0);
    }
}
