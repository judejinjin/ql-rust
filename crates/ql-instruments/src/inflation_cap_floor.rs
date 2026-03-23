//! Year-on-Year and Zero-Coupon inflation cap/floor instruments.
//!
//! Provides [`YoYInflationCapFloor`] (strip of YoY inflation caplets) and
//! [`ZeroCouponInflationCapFloor`] (single CPI cap/floor on cumulative inflation).

use serde::{Deserialize, Serialize};

/// Type of an inflation cap/floor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InflationCapFloorType {
    /// Cap.
    Cap,
    /// Floor.
    Floor,
}

impl InflationCapFloorType {
    /// +1 for cap, -1 for floor (payoff convention).
    pub fn sign(&self) -> f64 {
        match self {
            InflationCapFloorType::Cap => 1.0,
            InflationCapFloorType::Floor => -1.0,
        }
    }
}

/// A single YoY inflation caplet/floorlet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YoYInflationCaplet {
    /// Fixing time (years from valuation).
    pub fixing_time: f64,
    /// Payment time (years from valuation).
    pub payment_time: f64,
    /// Year fraction for accrual.
    pub accrual_fraction: f64,
    /// Notional amount.
    pub notional: f64,
    /// Forward YoY inflation rate for this period.
    pub forward_rate: f64,
    /// Discount factor at payment date.
    pub discount: f64,
}

/// A strip of YoY inflation caplets/floorlets.
///
/// Each caplet pays `N · τ · max(ω(I(t)/I(t-1) - 1 - K), 0)` where
/// ω = +1 for cap, -1 for floor, and I(t) is the inflation index value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YoYInflationCapFloor {
    /// Cap or Floor.
    pub cap_floor_type: InflationCapFloorType,
    /// Strike (YoY inflation rate, e.g., 0.03 for 3%).
    pub strike: f64,
    /// Individual caplets/floorlets.
    pub caplets: Vec<YoYInflationCaplet>,
}

/// A zero-coupon inflation cap or floor.
///
/// Pays at maturity: `N · max(ω((I(T)/I(0))^(1/T) - 1 - K), 0) · T`
/// or equivalently on cumulative: `N · max(ω(I(T)/I(0) - (1+K)^T), 0)`.
///
/// For pricing, we model the annualized zero-inflation rate as log-normal
/// or normal and price with Black/Bachelier formulas.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroCouponInflationCapFloor {
    /// Cap or Floor.
    pub cap_floor_type: InflationCapFloorType,
    /// Strike on the annualized zero-coupon inflation rate.
    pub strike: f64,
    /// Time to maturity (years).
    pub maturity: f64,
    /// Notional.
    pub notional: f64,
    /// Forward annualized zero-coupon inflation rate.
    pub forward_rate: f64,
    /// Discount factor at maturity.
    pub discount: f64,
}

/// Build a strip of YoY inflation caplets from parameters.
///
/// Creates `n_years` annual caplets with given notional, strike,
/// forward YoY rates, and discount factors.
pub fn build_yoy_cap_floor(
    cap_floor_type: InflationCapFloorType,
    strike: f64,
    notional: f64,
    n_years: usize,
    forward_rates: &[f64],
    discount_factors: &[f64],
) -> YoYInflationCapFloor {
    assert_eq!(forward_rates.len(), n_years);
    assert_eq!(discount_factors.len(), n_years);

    let caplets: Vec<YoYInflationCaplet> = (0..n_years)
        .map(|i| {
            let t = (i + 1) as f64;
            YoYInflationCaplet {
                fixing_time: t - 0.25, // typical 3-month observation lag
                payment_time: t,
                accrual_fraction: 1.0,
                notional,
                forward_rate: forward_rates[i],
                discount: discount_factors[i],
            }
        })
        .collect();

    YoYInflationCapFloor {
        cap_floor_type,
        strike,
        caplets,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inflation_cap_floor_type_sign() {
        assert_eq!(InflationCapFloorType::Cap.sign(), 1.0);
        assert_eq!(InflationCapFloorType::Floor.sign(), -1.0);
    }

    #[test]
    fn build_yoy_cap_5y() {
        let cap = build_yoy_cap_floor(
            InflationCapFloorType::Cap,
            0.03,
            1_000_000.0,
            5,
            &[0.025, 0.026, 0.027, 0.028, 0.029],
            &[0.95, 0.90, 0.86, 0.82, 0.78],
        );
        assert_eq!(cap.caplets.len(), 5);
        assert_eq!(cap.strike, 0.03);
        assert_eq!(cap.cap_floor_type, InflationCapFloorType::Cap);
    }

    #[test]
    fn zc_inflation_cap_floor_creation() {
        let cap = ZeroCouponInflationCapFloor {
            cap_floor_type: InflationCapFloorType::Cap,
            strike: 0.03,
            maturity: 5.0,
            notional: 1_000_000.0,
            forward_rate: 0.025,
            discount: 0.78,
        };
        assert_eq!(cap.maturity, 5.0);
        assert_eq!(cap.cap_floor_type, InflationCapFloorType::Cap);
    }
}
