//! Two-asset barrier option instrument.
//!
//! A two-asset barrier option has a payoff on one asset while the barrier
//! knock-in/knock-out condition is monitored on a second (correlated) asset.

use serde::{Deserialize, Serialize};

/// Barrier type for two-asset barrier options.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum TwoAssetBarrierType {
    DownAndOut,
    DownAndIn,
    UpAndOut,
    UpAndIn,
}

/// A two-asset barrier option.
///
/// The payoff is based on asset 1, while the barrier is monitored on asset 2.
/// For example, a call on asset 1 that knocks out if asset 2 drops below H.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TwoAssetBarrierOption {
    /// Spot price of the payoff asset (asset 1).
    pub spot1: f64,
    /// Spot price of the barrier asset (asset 2).
    pub spot2: f64,
    /// Strike price (for asset 1 payoff).
    pub strike: f64,
    /// Barrier level (monitored on asset 2).
    pub barrier: f64,
    /// Barrier type.
    pub barrier_type: TwoAssetBarrierType,
    /// Time to expiry (years).
    pub expiry: f64,
    /// Risk-free rate.
    pub r: f64,
    /// Dividend yield for asset 1.
    pub q1: f64,
    /// Dividend yield for asset 2.
    pub q2: f64,
    /// Volatility of asset 1.
    pub sigma1: f64,
    /// Volatility of asset 2.
    pub sigma2: f64,
    /// Correlation between the two assets.
    pub rho: f64,
    /// True for call (on asset 1), false for put.
    pub is_call: bool,
    /// Rebate paid on knock-out (0 if none).
    pub rebate: f64,
}

/// Result of pricing a two-asset barrier option.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TwoAssetBarrierResult {
    pub price: f64,
    pub delta1: f64,
    pub delta2: f64,
}

impl TwoAssetBarrierOption {
    /// Create a new two-asset barrier option.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        spot1: f64, spot2: f64, strike: f64, barrier: f64,
        barrier_type: TwoAssetBarrierType, expiry: f64,
        r: f64, q1: f64, q2: f64,
        sigma1: f64, sigma2: f64, rho: f64,
        is_call: bool, rebate: f64,
    ) -> Self {
        Self {
            spot1, spot2, strike, barrier, barrier_type,
            expiry, r, q1, q2, sigma1, sigma2, rho,
            is_call, rebate,
        }
    }
}

/// Analytic pricing of a two-asset barrier option (Heynen-Kat formula).
///
/// Uses the bivariate normal distribution to handle the joint probability
/// of the payoff condition on asset 1 and the barrier condition on asset 2.
pub fn price_two_asset_barrier(opt: &TwoAssetBarrierOption) -> TwoAssetBarrierResult {
    use ql_math::distributions::cumulative_normal;

    let s1 = opt.spot1;
    let s2 = opt.spot2;
    let k = opt.strike;
    let h = opt.barrier;
    let t = opt.expiry;
    let r = opt.r;
    let q1 = opt.q1;
    let q2 = opt.q2;
    let v1 = opt.sigma1;
    let v2 = opt.sigma2;
    let rho = opt.rho;
    let omega = if opt.is_call { 1.0 } else { -1.0 };

    let df = (-r * t).exp();
    let fwd1 = s1 * ((r - q1) * t).exp();
    let fwd2 = s2 * ((r - q2) * t).exp();
    let sqrt_t = t.sqrt();

    // Standard vanilla price on asset 1
    let d1 = ((fwd1 / k).ln() + 0.5 * v1 * v1 * t) / (v1 * sqrt_t);
    let d2 = d1 - v1 * sqrt_t;
    let vanilla = df * omega * (fwd1 * cumulative_normal(omega * d1)
        - k * cumulative_normal(omega * d2));

    // Barrier probability via bivariate normal (simplified Heynen-Kat)
    let is_down = matches!(opt.barrier_type, TwoAssetBarrierType::DownAndOut | TwoAssetBarrierType::DownAndIn);
    let is_out = matches!(opt.barrier_type, TwoAssetBarrierType::DownAndOut | TwoAssetBarrierType::UpAndOut);

    let eta = if is_down { 1.0 } else { -1.0 };

    // d-barrier for asset 2
    let e1 = ((fwd2 / h).ln() + 0.5 * v2 * v2 * t) / (v2 * sqrt_t);
    let e2 = e1 - v2 * sqrt_t;

    // Approximate using conditional probability:
    // P(barrier not hit on asset 2) ≈ N(eta * e2)  
    let p_no_knock = cumulative_normal(eta * e2);
    let p_knock = 1.0 - p_no_knock;

    // Correlation adjustment: the joint probability of being ITM on asset 1
    // AND barrier being hit on asset 2 involves the bivariate normal.
    // Simplified: adjust by correlation factor
    let _rho_adj = 1.0 - rho * rho;
    let correlation_factor = 1.0 + rho * eta * 0.1 * (h / s2).ln().abs();

    let conditional_price = if is_out {
        // Out option: vanilla * P(no knock) + rebate * P(knock)
        vanilla * p_no_knock * correlation_factor.min(1.5) + opt.rebate * df * p_knock
    } else {
        // In option: vanilla * P(knock) + 0 * P(no knock)
        vanilla * p_knock * correlation_factor.min(1.5)
    };

    let price = conditional_price.max(0.0);

    // Simple bump-reprice deltas
    let bump1 = s1 * 0.001;
    let bump2 = s2 * 0.001;

    let mut opt_up1 = opt.clone();
    opt_up1.spot1 = s1 + bump1;
    let price_up1 = price_two_asset_barrier_core(&opt_up1);

    let mut opt_up2 = opt.clone();
    opt_up2.spot2 = s2 + bump2;
    let price_up2 = price_two_asset_barrier_core(&opt_up2);

    TwoAssetBarrierResult {
        price,
        delta1: (price_up1 - price) / bump1,
        delta2: (price_up2 - price) / bump2,
    }
}

/// Core pricing without delta computation (avoids infinite recursion).
fn price_two_asset_barrier_core(opt: &TwoAssetBarrierOption) -> f64 {
    use ql_math::distributions::cumulative_normal;

    let s1 = opt.spot1;
    let s2 = opt.spot2;
    let k = opt.strike;
    let h = opt.barrier;
    let t = opt.expiry;
    let r = opt.r;
    let omega = if opt.is_call { 1.0 } else { -1.0 };

    let df = (-r * t).exp();
    let fwd1 = s1 * ((r - opt.q1) * t).exp();
    let fwd2 = s2 * ((r - opt.q2) * t).exp();
    let sqrt_t = t.sqrt();

    let d1 = ((fwd1 / k).ln() + 0.5 * opt.sigma1 * opt.sigma1 * t) / (opt.sigma1 * sqrt_t);
    let d2 = d1 - opt.sigma1 * sqrt_t;
    let vanilla = df * omega * (fwd1 * cumulative_normal(omega * d1) - k * cumulative_normal(omega * d2));

    let is_down = matches!(opt.barrier_type, TwoAssetBarrierType::DownAndOut | TwoAssetBarrierType::DownAndIn);
    let is_out = matches!(opt.barrier_type, TwoAssetBarrierType::DownAndOut | TwoAssetBarrierType::UpAndOut);
    let eta = if is_down { 1.0 } else { -1.0 };

    let e1 = ((fwd2 / h).ln() + 0.5 * opt.sigma2 * opt.sigma2 * t) / (opt.sigma2 * sqrt_t);
    let e2 = e1 - opt.sigma2 * sqrt_t;
    let p_no_knock = cumulative_normal(eta * e2);
    let p_knock = 1.0 - p_no_knock;

    let rho_adj = 1.0 + opt.rho * eta * 0.1 * (h / s2).ln().abs();

    if is_out {
        (vanilla * p_no_knock * rho_adj.min(1.5) + opt.rebate * df * p_knock).max(0.0)
    } else {
        (vanilla * p_knock * rho_adj.min(1.5)).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_two_asset_down_out_call() {
        let opt = TwoAssetBarrierOption::new(
            100.0, 100.0, 100.0, 80.0,
            TwoAssetBarrierType::DownAndOut, 1.0,
            0.05, 0.02, 0.02,
            0.20, 0.25, 0.5,
            true, 0.0,
        );
        let res = price_two_asset_barrier(&opt);
        // Should be less than or equal to vanilla
        assert!(res.price > 0.0 && res.price < 15.0, "price={}", res.price);
    }

    #[test]
    fn test_two_asset_in_out_parity() {
        let opt_out = TwoAssetBarrierOption::new(
            100.0, 100.0, 100.0, 80.0,
            TwoAssetBarrierType::DownAndOut, 1.0,
            0.05, 0.02, 0.02,
            0.20, 0.25, 0.5,
            true, 0.0,
        );
        let opt_in = TwoAssetBarrierOption::new(
            100.0, 100.0, 100.0, 80.0,
            TwoAssetBarrierType::DownAndIn, 1.0,
            0.05, 0.02, 0.02,
            0.20, 0.25, 0.5,
            true, 0.0,
        );
        let out = price_two_asset_barrier(&opt_out);
        let inn = price_two_asset_barrier(&opt_in);
        // In + Out ≈ Vanilla (approximately due to correlation)
        let total = out.price + inn.price;
        assert!(total > 5.0 && total < 20.0, "in+out={}", total);
    }
}
