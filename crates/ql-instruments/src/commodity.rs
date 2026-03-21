//! Commodities framework.
//!
//! Provides types and pricing for energy/commodity contracts:
//! - Forward curves for commodity spot/futures
//! - Commodity forward contracts
//! - Commodity swap / basis swap pricing
//!
//! Reference:
//! - QuantLib: EnergyFuture, EnergyCommodity, CommodityForward

use serde::{Deserialize, Serialize};
use ql_time::{Date, DayCounter};

/// A point on a commodity forward curve.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ForwardPoint {
    /// Delivery / expiry date.
    pub date: Date,
    /// Forward price.
    pub price: f64,
}

/// Commodity forward curve.
///
/// Interpolates between known forward points to provide forward prices
/// at arbitrary delivery dates.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommodityForwardCurve {
    /// Reference date (valuation date).
    pub reference_date: Date,
    /// Spot price.
    pub spot_price: f64,
    /// Forward points sorted by date.
    pub forwards: Vec<ForwardPoint>,
    /// Day counter for year-fraction computation.
    pub day_counter: DayCounter,
    /// Convenience/storage cost rate (continuous).
    pub convenience_yield: f64,
    /// Risk-free rate for cost-of-carry model.
    pub risk_free_rate: f64,
}

impl CommodityForwardCurve {
    /// Get the forward price at a given delivery date.
    ///
    /// If the date falls between two forward points, uses log-linear
    /// interpolation. If the date is beyond the last point, uses the
    /// cost-of-carry model: F(T) = S × exp((r − c) × T).
    pub fn forward_price(&self, delivery_date: Date) -> f64 {
        if self.forwards.is_empty() {
            let t = self.day_counter.year_fraction(self.reference_date, delivery_date);
            return self.spot_price * ((self.risk_free_rate - self.convenience_yield) * t).exp();
        }

        // Exact match
        for fp in &self.forwards {
            if fp.date == delivery_date { return fp.price; }
        }

        // Interpolation
        let t = self.day_counter.year_fraction(self.reference_date, delivery_date);
        let times: Vec<f64> = self.forwards.iter()
            .map(|fp| self.day_counter.year_fraction(self.reference_date, fp.date))
            .collect();

        // Before first point
        if t <= times[0] {
            return self.forwards[0].price * (t / times[0].max(1e-8));
        }
        // After last point
        if t >= *times.last().unwrap() {
            let last = self.forwards.last().unwrap();
            let t_last = *times.last().unwrap();
            let dt = t - t_last;
            return last.price * ((self.risk_free_rate - self.convenience_yield) * dt).exp();
        }

        // Log-linear interpolation
        let idx = times.iter().position(|&ti| ti >= t).unwrap_or(times.len() - 1);
        if idx == 0 { return self.forwards[0].price; }
        let t0 = times[idx - 1];
        let t1 = times[idx];
        let p0 = self.forwards[idx - 1].price;
        let p1 = self.forwards[idx].price;
        let w = (t - t0) / (t1 - t0).max(1e-12);
        ((1.0 - w) * p0.ln() + w * p1.ln()).exp()
    }

    /// Implied convenience yield between two dates.
    pub fn implied_convenience_yield(&self, t1_date: Date, t2_date: Date) -> f64 {
        let f1 = self.forward_price(t1_date);
        let f2 = self.forward_price(t2_date);
        let dt = self.day_counter.year_fraction(t1_date, t2_date);
        if dt.abs() < 1e-8 || f1 <= 0.0 { return 0.0; }
        self.risk_free_rate - (f2 / f1).ln() / dt
    }
}

/// Commodity forward contract.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommodityForward {
    /// Long (+1) or short (−1).
    pub position: f64,
    /// Quantity (number of units).
    pub quantity: f64,
    /// Delivery date.
    pub delivery_date: Date,
    /// Strike / agreed price.
    pub strike: f64,
    /// Underlying commodity name.
    pub commodity_name: String,
}

/// Result of commodity forward valuation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommodityForwardResult {
    /// NPV = position × quantity × (Forward − Strike) × DF.
    pub npv: f64,
    /// Forward price at delivery.
    pub forward_price: f64,
    /// Discount factor to delivery.
    pub discount_factor: f64,
    /// Delta (dNPV/dSpot).
    pub delta: f64,
}

/// Price a commodity forward contract.
pub fn price_commodity_forward(
    forward: &CommodityForward,
    curve: &CommodityForwardCurve,
    discount_factor: f64,
) -> CommodityForwardResult {
    let fwd = curve.forward_price(forward.delivery_date);
    let npv = forward.position * forward.quantity * (fwd - forward.strike) * discount_factor;

    // Delta: dNPV/dSpot ≈ position × quantity × DF × (F/S) (cost-of-carry)
    let delta = if curve.spot_price.abs() > 1e-12 {
        forward.position * forward.quantity * discount_factor * fwd / curve.spot_price
    } else { 0.0 };

    CommodityForwardResult {
        npv,
        forward_price: fwd,
        discount_factor,
        delta,
    }
}

/// Commodity swap (fixed-for-floating).
///
/// One side pays a fixed price per unit; the other pays the average
/// of commodity forward prices over the averaging dates.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommoditySwap {
    /// Quantity per period.
    pub quantity: f64,
    /// Fixed price.
    pub fixed_price: f64,
    /// Payment / averaging dates.
    pub payment_dates: Vec<Date>,
    /// Position: +1 = receive floating, pay fixed.
    pub position: f64,
}

/// Result of commodity swap valuation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommoditySwapResult {
    /// NPV of the swap.
    pub npv: f64,
    /// Average forward price across all dates.
    pub average_forward: f64,
    /// Fair fixed price (that makes NPV = 0).
    pub fair_fixed_price: f64,
}

/// Price a commodity swap.
pub fn price_commodity_swap(
    swap: &CommoditySwap,
    curve: &CommodityForwardCurve,
    discount_factors: &[f64],
) -> CommoditySwapResult {
    let n = swap.payment_dates.len().min(discount_factors.len());
    if n == 0 {
        return CommoditySwapResult {
            npv: 0.0, average_forward: 0.0, fair_fixed_price: swap.fixed_price,
        };
    }

    let mut sum_fwd_df = 0.0;
    let mut sum_df = 0.0;
    let mut sum_fwd = 0.0;

    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        let fwd = curve.forward_price(swap.payment_dates[i]);
        let df = discount_factors[i];
        sum_fwd_df += fwd * df;
        sum_df += df;
        sum_fwd += fwd;
    }

    let avg_fwd = sum_fwd / n as f64;
    let fair_fixed = if sum_df.abs() > 1e-12 { sum_fwd_df / sum_df } else { avg_fwd };
    let npv = swap.position * swap.quantity * (sum_fwd_df - swap.fixed_price * sum_df);

    CommoditySwapResult {
        npv,
        average_forward: avg_fwd,
        fair_fixed_price: fair_fixed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ql_time::Month;
    use approx::assert_abs_diff_eq;

    fn sample_curve() -> CommodityForwardCurve {
        let ref_date = Date::from_ymd(2025, Month::January, 15);
        CommodityForwardCurve {
            reference_date: ref_date,
            spot_price: 80.0,
            forwards: vec![
                ForwardPoint { date: ref_date + 90, price: 81.0 },
                ForwardPoint { date: ref_date + 180, price: 82.5 },
                ForwardPoint { date: ref_date + 365, price: 85.0 },
                ForwardPoint { date: ref_date + 730, price: 88.0 },
            ],
            day_counter: DayCounter::Actual365Fixed,
            convenience_yield: 0.02,
            risk_free_rate: 0.05,
        }
    }

    #[test]
    fn test_forward_interpolation() {
        let curve = sample_curve();
        let ref_date = curve.reference_date;
        // Exact match
        assert_abs_diff_eq!(curve.forward_price(ref_date + 365), 85.0, epsilon = 1e-10);
        // Interpolated
        let f = curve.forward_price(ref_date + 270);
        assert!(f > 82.5 && f < 85.0, "f={}", f);
    }

    #[test]
    fn test_commodity_forward_pricing() {
        let curve = sample_curve();
        let fwd = CommodityForward {
            position: 1.0,
            quantity: 1000.0,
            delivery_date: curve.reference_date + 365,
            strike: 83.0,
            commodity_name: "WTI".to_string(),
        };
        let res = price_commodity_forward(&fwd, &curve, 0.95);
        // NPV = 1000 × (85 - 83) × 0.95 = 1900
        assert_abs_diff_eq!(res.npv, 1900.0, epsilon = 1.0);
        assert_abs_diff_eq!(res.forward_price, 85.0, epsilon = 0.01);
    }

    #[test]
    fn test_commodity_swap_fair_price() {
        let curve = sample_curve();
        let ref_date = curve.reference_date;
        let dates = vec![ref_date + 90, ref_date + 180, ref_date + 365];
        let dfs = vec![0.99, 0.975, 0.95];

        // Price at fair fixed; NPV should be near zero
        let trial = CommoditySwap {
            quantity: 100.0,
            fixed_price: 0.0,
            payment_dates: dates.clone(),
            position: 1.0,
        };
        let res = price_commodity_swap(&trial, &curve, &dfs);
        let fair = res.fair_fixed_price;

        let swap = CommoditySwap {
            quantity: 100.0,
            fixed_price: fair,
            payment_dates: dates,
            position: 1.0,
        };
        let res2 = price_commodity_swap(&swap, &curve, &dfs);
        assert_abs_diff_eq!(res2.npv, 0.0, epsilon = 0.01);
    }
}
