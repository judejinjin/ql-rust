//! Multiple-resets coupon.
//!
//! A floating-rate coupon that resets multiple times during a single
//! accrual period. The coupon rate is the average (or compounded)
//! fixings over the period, similar to how in-arrears compounding works.
//!
//! Corresponds to QuantLib's `SubPeriodsCoupon` / `MultipleResetsCoupon`.

use serde::{Deserialize, Serialize};

/// Compounding method for multiple resets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompoundingType {
    /// Simple average of fixings.
    Averaging,
    /// Compounded fixings.
    Compounding,
    /// Flat compounding (money market convention).
    FlatCompounding,
}

/// A single sub-period reset within the coupon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubPeriodReset {
    /// Fixing date/time (years from reference).
    pub fixing_time: f64,
    /// Start of sub-period.
    pub start_time: f64,
    /// End of sub-period.
    pub end_time: f64,
    /// Accrual fraction for this sub-period.
    pub accrual_fraction: f64,
    /// Observed fixing rate (set after fixing).
    pub fixing: Option<f64>,
}

/// A coupon with multiple resets during its accrual period.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultipleResetsCoupon {
    /// Notional amount.
    pub notional: f64,
    /// Payment time (years from reference date).
    pub payment_time: f64,
    /// Accrual start time.
    pub accrual_start: f64,
    /// Accrual end time.
    pub accrual_end: f64,
    /// Total accrual fraction (accrual_end - accrual_start, usually).
    pub accrual_fraction: f64,
    /// Sub-period resets.
    pub resets: Vec<SubPeriodReset>,
    /// Compounding type.
    pub compounding: CompoundingType,
    /// Spread added to each fixing.
    pub spread: f64,
    /// Gearing (multiplier on the rate).
    pub gearing: f64,
}

impl MultipleResetsCoupon {
    /// Create a new multiple-resets coupon with uniform sub-periods.
    pub fn new(
        notional: f64,
        payment_time: f64,
        accrual_start: f64,
        accrual_end: f64,
        num_resets: usize,
        compounding: CompoundingType,
        spread: f64,
        gearing: f64,
    ) -> Self {
        assert!(num_resets >= 1);
        let total_accrual = accrual_end - accrual_start;
        let sub_period = total_accrual / num_resets as f64;

        let resets: Vec<SubPeriodReset> = (0..num_resets)
            .map(|i| {
                let start = accrual_start + i as f64 * sub_period;
                let end = accrual_start + (i + 1) as f64 * sub_period;
                SubPeriodReset {
                    fixing_time: start,
                    start_time: start,
                    end_time: end,
                    accrual_fraction: sub_period,
                    fixing: None,
                }
            })
            .collect();

        Self {
            notional,
            payment_time,
            accrual_start,
            accrual_end,
            accrual_fraction: total_accrual,
            resets,
            compounding,
            spread,
            gearing,
        }
    }

    /// Set fixings for all sub-periods.
    pub fn set_fixings(&mut self, fixings: &[f64]) {
        assert_eq!(fixings.len(), self.resets.len());
        for (reset, &fixing) in self.resets.iter_mut().zip(fixings.iter()) {
            reset.fixing = Some(fixing);
        }
    }

    /// Compute the effective coupon rate given the fixings.
    ///
    /// Returns `None` if any fixing is missing.
    pub fn rate(&self) -> Option<f64> {
        let fixings: Vec<f64> = self.resets.iter()
            .map(|r| r.fixing)
            .collect::<Option<Vec<_>>>()?;

        let raw_rate = match self.compounding {
            CompoundingType::Averaging => {
                // Simple weighted average
                let total_weight: f64 = self.resets.iter().map(|r| r.accrual_fraction).sum();
                if total_weight.abs() < 1e-14 { return Some(0.0); }
                let weighted_sum: f64 = fixings.iter()
                    .zip(self.resets.iter())
                    .map(|(&fix, r)| (fix + self.spread) * r.accrual_fraction)
                    .sum();
                weighted_sum / total_weight
            }
            CompoundingType::Compounding => {
                // Compounded rate: ∏(1 + (r_i + s) × τ_i) − 1
                let product: f64 = fixings.iter()
                    .zip(self.resets.iter())
                    .map(|(&fix, r)| 1.0 + (fix + self.spread) * r.accrual_fraction)
                    .product();
                (product - 1.0) / self.accrual_fraction
            }
            CompoundingType::FlatCompounding => {
                // Flat compounding: Σ (r_i + s) × τ_i × ∏_{j<i} (1 + (r_j+s) × τ_j)
                let mut compound = 0.0;
                let mut cumulative = 1.0;
                for (i, (&fix, r)) in fixings.iter().zip(self.resets.iter()).enumerate() {
                    if i > 0 {
                        cumulative *= 1.0 + (fixings[i - 1] + self.spread) * self.resets[i - 1].accrual_fraction;
                    }
                    compound += (fix + self.spread) * r.accrual_fraction * cumulative;
                }
                compound / self.accrual_fraction
            }
        };

        Some(self.gearing * raw_rate)
    }

    /// Compute the coupon amount = notional × rate × accrual_fraction.
    pub fn amount(&self) -> Option<f64> {
        self.rate().map(|r| self.notional * r * self.accrual_fraction)
    }

    /// Number of sub-period resets.
    pub fn num_resets(&self) -> usize {
        self.resets.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_averaging_coupon() {
        let mut coupon = MultipleResetsCoupon::new(
            1_000_000.0, 1.0, 0.0, 1.0, 4,
            CompoundingType::Averaging, 0.0, 1.0,
        );
        coupon.set_fixings(&[0.05, 0.06, 0.04, 0.05]);
        let rate = coupon.rate().unwrap();
        // Average = (0.05 + 0.06 + 0.04 + 0.05) / 4 = 0.05
        assert!((rate - 0.05).abs() < 1e-10, "rate={}", rate);

        let amount = coupon.amount().unwrap();
        assert!((amount - 50_000.0).abs() < 0.01, "amount={}", amount);
    }

    #[test]
    fn test_compounding_coupon() {
        let mut coupon = MultipleResetsCoupon::new(
            1_000_000.0, 1.0, 0.0, 1.0, 2,
            CompoundingType::Compounding, 0.0, 1.0,
        );
        coupon.set_fixings(&[0.04, 0.04]);
        let rate = coupon.rate().unwrap();
        // ∏(1 + 0.04 × 0.5) − 1 = (1.02)² − 1 = 0.0404
        let expected = ((1.02_f64).powi(2) - 1.0);
        assert!((rate - expected).abs() < 1e-10, "rate={}, expected={}", rate, expected);
    }

    #[test]
    fn test_coupon_with_spread() {
        let mut coupon = MultipleResetsCoupon::new(
            1_000_000.0, 1.0, 0.0, 1.0, 1,
            CompoundingType::Averaging, 0.01, 1.0,
        );
        coupon.set_fixings(&[0.03]);
        let rate = coupon.rate().unwrap();
        // rate = 0.03 + 0.01 = 0.04
        assert!((rate - 0.04).abs() < 1e-10, "rate={}", rate);
    }

    #[test]
    fn test_coupon_with_gearing() {
        let mut coupon = MultipleResetsCoupon::new(
            1_000_000.0, 1.0, 0.0, 1.0, 1,
            CompoundingType::Averaging, 0.0, 2.0,
        );
        coupon.set_fixings(&[0.05]);
        let rate = coupon.rate().unwrap();
        // rate = 2.0 × 0.05 = 0.10
        assert!((rate - 0.10).abs() < 1e-10, "rate={}", rate);
    }

    #[test]
    fn test_missing_fixings() {
        let coupon = MultipleResetsCoupon::new(
            1_000_000.0, 1.0, 0.0, 1.0, 4,
            CompoundingType::Averaging, 0.0, 1.0,
        );
        // No fixings set
        assert!(coupon.rate().is_none());
        assert!(coupon.amount().is_none());
    }

    #[test]
    fn test_flat_compounding() {
        let mut coupon = MultipleResetsCoupon::new(
            1_000_000.0, 1.0, 0.0, 1.0, 2,
            CompoundingType::FlatCompounding, 0.0, 1.0,
        );
        coupon.set_fixings(&[0.04, 0.04]);
        let rate = coupon.rate().unwrap();
        // First sub: 0.04 × 0.5 × 1.0 = 0.02
        // Second sub: 0.04 × 0.5 × (1 + 0.04 × 0.5) = 0.04 × 0.5 × 1.02 = 0.0204
        // Total / 1.0 = 0.0404
        let expected = (0.04 * 0.5 * 1.0 + 0.04 * 0.5 * 1.02);
        assert!((rate - expected).abs() < 1e-10, "rate={}, expected={}", rate, expected);
    }
}
