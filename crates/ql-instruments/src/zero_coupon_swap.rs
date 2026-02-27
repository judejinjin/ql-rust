//! Zero-coupon swap (fixed vs floating with no intermediate coupons).
//!
//! A zero-coupon swap exchanges a single fixed payment at maturity against
//! a compounded floating leg. Common in inflation derivatives and
//! structured products.

use serde::{Deserialize, Serialize};

/// A zero-coupon swap.
///
/// The fixed leg pays (or receives) a single compounded fixed payment at
/// maturity. The floating leg pays (or receives) the compounded floating rate.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZeroCouponSwap {
    /// Notional amount.
    pub notional: f64,
    /// Fixed rate (annualized).
    pub fixed_rate: f64,
    /// Start time (year fraction).
    pub start_time: f64,
    /// Maturity (year fraction).
    pub maturity: f64,
    /// True if paying fixed.
    pub pay_fixed: bool,
    /// Compounding frequency per year for the fixed leg.
    pub compound_freq: u32,
    /// Day count fraction for accrual (simplified).
    pub day_count_fraction: f64,
}

/// Result from zero-coupon swap pricing.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ZeroCouponSwapResult {
    /// Net present value (positive = in favour of payer).
    pub npv: f64,
    /// Fair fixed rate that makes NPV = 0.
    pub fair_rate: f64,
    /// Fixed leg PV.
    pub fixed_leg_pv: f64,
    /// Floating leg PV.
    pub floating_leg_pv: f64,
    /// DV01 (dollar value of 1bp move in floating rate).
    pub dv01: f64,
}

impl ZeroCouponSwap {
    /// Create a new zero-coupon swap.
    pub fn new(
        notional: f64,
        fixed_rate: f64,
        start_time: f64,
        maturity: f64,
        pay_fixed: bool,
        compound_freq: u32,
    ) -> Self {
        let dcf = maturity - start_time;
        Self {
            notional,
            fixed_rate,
            start_time,
            maturity,
            pay_fixed,
            compound_freq,
            day_count_fraction: dcf,
        }
    }
}

/// Price a zero-coupon swap given flat rates.
///
/// # Arguments
/// - `swap` — the swap instrument
/// - `discount_rate` — discount rate for PV
/// - `forward_rate` — floating rate forecast (flat)
pub fn price_zero_coupon_swap(
    swap: &ZeroCouponSwap,
    discount_rate: f64,
    forward_rate: f64,
) -> ZeroCouponSwapResult {
    let dcf = swap.day_count_fraction;
    let df = (-discount_rate * swap.maturity).exp();
    let n = swap.compound_freq;

    // Fixed leg: pays N * [(1 + r/n)^(n*dcf) - 1] at maturity
    let fixed_compound = if n > 0 {
        (1.0 + swap.fixed_rate / n as f64).powf(n as f64 * dcf) - 1.0
    } else {
        (swap.fixed_rate * dcf).exp() - 1.0 // continuous
    };
    let fixed_payment = swap.notional * fixed_compound;
    let fixed_leg_pv = fixed_payment * df;

    // Floating leg: pays N * [(1 + fwd/n)^(n*dcf) - 1] at maturity
    let float_compound = if n > 0 {
        (1.0 + forward_rate / n as f64).powf(n as f64 * dcf) - 1.0
    } else {
        (forward_rate * dcf).exp() - 1.0
    };
    let float_payment = swap.notional * float_compound;
    let floating_leg_pv = float_payment * df;

    let direction = if swap.pay_fixed { -1.0 } else { 1.0 };
    let npv = direction * (fixed_leg_pv - floating_leg_pv);

    // Fair rate: solve fixed_compound = float_compound
    let fair_rate = if n > 0 {
        n as f64 * ((1.0 + forward_rate / n as f64).powf(n as f64 * dcf / (n as f64 * dcf)) - 1.0)
    } else {
        forward_rate
    };

    // DV01: 1bp floating rate bump
    let bump = 0.0001;
    let float_compound_up = if n > 0 {
        (1.0 + (forward_rate + bump) / n as f64).powf(n as f64 * dcf) - 1.0
    } else {
        ((forward_rate + bump) * dcf).exp() - 1.0
    };
    let float_pv_up = swap.notional * float_compound_up * df;
    let dv01 = (float_pv_up - floating_leg_pv).abs();

    ZeroCouponSwapResult {
        npv,
        fair_rate,
        fixed_leg_pv,
        floating_leg_pv,
        dv01,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_zero_coupon_swap_at_market() {
        // When fixed rate = forward rate, NPV ≈ 0
        let swap = ZeroCouponSwap::new(
            1_000_000.0, 0.05, 0.0, 5.0, true, 1,
        );
        let res = price_zero_coupon_swap(&swap, 0.05, 0.05);
        assert_abs_diff_eq!(res.npv, 0.0, epsilon = 1.0);
    }

    #[test]
    fn test_zero_coupon_swap_pay_fixed_rates_up() {
        // Pay fixed at 5%, floating at 6%: we receive more floating → positive NPV
        let swap = ZeroCouponSwap::new(
            1_000_000.0, 0.05, 0.0, 5.0, true, 1,
        );
        let res = price_zero_coupon_swap(&swap, 0.05, 0.06);
        // pay_fixed=true, NPV = -(fixed - floating) = positive if floating > fixed
        assert!(res.npv > 0.0, "npv={}", res.npv);
    }

    #[test]
    fn test_zero_coupon_swap_dv01() {
        let swap = ZeroCouponSwap::new(
            1_000_000.0, 0.05, 0.0, 5.0, true, 1,
        );
        let res = price_zero_coupon_swap(&swap, 0.05, 0.05);
        assert!(res.dv01 > 0.0, "dv01={}", res.dv01);
    }
}
