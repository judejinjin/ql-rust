#![allow(clippy::too_many_arguments)]
//! Digital and capped/floored coupons.
//!
//! - `DigitalCoupon` — pays a fixed (digital) amount if a reference rate is
//!   above/below a strike at fixing.
//! - `CapFlooredCoupon` — a floating coupon with a cap, floor, or collar.
//! - `RangeAccrualCoupon` — coupon accrues only on days when the reference
//!   rate lies within a specified range.
//! - `SubPeriodCoupon` — coupon with multiple sub-period fixings, compounded
//!   or averaged.

use ql_time::{Date, DayCounter};
use crate::cashflow::CashFlow;
use crate::coupon::Coupon;

// ===========================================================================
// Digital Coupon
// ===========================================================================

/// Digital (binary) coupon: pays a fixed cash rebate if the reference rate
/// is above (call) or below (put) the strike.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DigitalCoupon {
    pub payment_date: Date,
    pub nominal: f64,
    pub accrual_start: Date,
    pub accrual_end: Date,
    pub day_counter: DayCounter,
    /// The reference rate observed at fixing.
    pub reference_rate: f64,
    /// Strike level.
    pub strike: f64,
    /// Cash rebate paid if digital is in-the-money (per unit notional, per period).
    pub cash_rate: f64,
    /// True = call digital (pays if rate > strike), false = put digital.
    pub is_call: bool,
}

impl DigitalCoupon {
    pub fn new(
        payment_date: Date,
        nominal: f64,
        accrual_start: Date,
        accrual_end: Date,
        day_counter: DayCounter,
        reference_rate: f64,
        strike: f64,
        cash_rate: f64,
        is_call: bool,
    ) -> Self {
        Self {
            payment_date,
            nominal,
            accrual_start,
            accrual_end,
            day_counter,
            reference_rate,
            strike,
            cash_rate,
            is_call,
        }
    }

    fn is_triggered(&self) -> bool {
        if self.is_call {
            self.reference_rate > self.strike
        } else {
            self.reference_rate < self.strike
        }
    }
}

impl CashFlow for DigitalCoupon {
    fn date(&self) -> Date {
        self.payment_date
    }

    fn amount(&self) -> f64 {
        if self.is_triggered() {
            self.nominal * self.cash_rate * self.accrual_period()
        } else {
            0.0
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Coupon for DigitalCoupon {
    fn nominal(&self) -> f64 {
        self.nominal
    }

    fn rate(&self) -> f64 {
        if self.is_triggered() {
            self.cash_rate
        } else {
            0.0
        }
    }

    fn accrual_start(&self) -> Date {
        self.accrual_start
    }

    fn accrual_end(&self) -> Date {
        self.accrual_end
    }

    fn accrual_period(&self) -> f64 {
        self.day_counter
            .year_fraction(self.accrual_start, self.accrual_end)
    }

    fn day_counter(&self) -> DayCounter {
        self.day_counter
    }
}

// ===========================================================================
// Cap/Floored Coupon
// ===========================================================================

/// A floating-rate coupon with an optional cap and/or floor.
///
/// Effective rate = max(floor, min(cap, gearing * reference_rate + spread)).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CapFlooredCoupon {
    /// Payment date.
    pub payment_date: Date,
    /// Notional amount.
    pub nominal: f64,
    /// Start of accrual period.
    pub accrual_start: Date,
    /// End of accrual period.
    pub accrual_end: Date,
    /// Day counter for year-fraction computation.
    pub day_counter: DayCounter,
    /// Underlying floating reference rate.
    pub reference_rate: f64,
    /// Additive spread over the reference rate.
    pub spread: f64,
    /// Multiplicative gearing factor.
    pub gearing: f64,
    /// Optional cap rate (upper bound on the effective rate).
    pub cap: Option<f64>,
    /// Optional floor rate (lower bound on the effective rate).
    pub floor: Option<f64>,
}

impl CapFlooredCoupon {
    pub fn new(
        payment_date: Date,
        nominal: f64,
        accrual_start: Date,
        accrual_end: Date,
        day_counter: DayCounter,
        reference_rate: f64,
        spread: f64,
        gearing: f64,
        cap: Option<f64>,
        floor: Option<f64>,
    ) -> Self {
        Self {
            payment_date,
            nominal,
            accrual_start,
            accrual_end,
            day_counter,
            reference_rate,
            spread,
            gearing,
            cap,
            floor,
        }
    }

    pub fn effective_rate(&self) -> f64 {
        let raw = self.gearing * self.reference_rate + self.spread;
        let floored = match self.floor {
            Some(f) => raw.max(f),
            None => raw,
        };
        match self.cap {
            Some(c) => floored.min(c),
            None => floored,
        }
    }
}

impl CashFlow for CapFlooredCoupon {
    fn date(&self) -> Date {
        self.payment_date
    }

    fn amount(&self) -> f64 {
        self.nominal * self.effective_rate() * self.accrual_period()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Coupon for CapFlooredCoupon {
    fn nominal(&self) -> f64 {
        self.nominal
    }

    fn rate(&self) -> f64 {
        self.effective_rate()
    }

    fn accrual_start(&self) -> Date {
        self.accrual_start
    }

    fn accrual_end(&self) -> Date {
        self.accrual_end
    }

    fn accrual_period(&self) -> f64 {
        self.day_counter
            .year_fraction(self.accrual_start, self.accrual_end)
    }

    fn day_counter(&self) -> DayCounter {
        self.day_counter
    }
}

// ===========================================================================
// Range Accrual Coupon
// ===========================================================================

/// Range accrual coupon: the coupon accrues proportionally to the fraction
/// of observation days where the reference rate lies within [lower, upper].
///
/// Amount = nominal * rate * accrual_period * (days_in_range / total_days).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RangeAccrualCoupon {
    /// Payment date.
    pub payment_date: Date,
    /// Notional amount.
    pub nominal: f64,
    /// Start of accrual period.
    pub accrual_start: Date,
    /// End of accrual period.
    pub accrual_end: Date,
    /// Day counter for year-fraction computation.
    pub day_counter: DayCounter,
    /// Fixed coupon rate (before range-accrual scaling).
    pub fixed_rate: f64,
    /// Lower barrier of the accrual range.
    pub lower_barrier: f64,
    /// Upper barrier of the accrual range.
    pub upper_barrier: f64,
    /// Observed rates for each observation day.
    pub observed_rates: Vec<f64>,
}

impl RangeAccrualCoupon {
    pub fn new(
        payment_date: Date,
        nominal: f64,
        accrual_start: Date,
        accrual_end: Date,
        day_counter: DayCounter,
        fixed_rate: f64,
        lower_barrier: f64,
        upper_barrier: f64,
        observed_rates: Vec<f64>,
    ) -> Self {
        Self {
            payment_date,
            nominal,
            accrual_start,
            accrual_end,
            day_counter,
            fixed_rate,
            lower_barrier,
            upper_barrier,
            observed_rates,
        }
    }

    /// Fraction of observation days in range.
    pub fn accrual_fraction(&self) -> f64 {
        if self.observed_rates.is_empty() {
            return 0.0;
        }
        let in_range = self
            .observed_rates
            .iter()
            .filter(|&&r| r >= self.lower_barrier && r <= self.upper_barrier)
            .count();
        in_range as f64 / self.observed_rates.len() as f64
    }
}

impl CashFlow for RangeAccrualCoupon {
    fn date(&self) -> Date {
        self.payment_date
    }

    fn amount(&self) -> f64 {
        self.nominal * self.fixed_rate * self.accrual_period() * self.accrual_fraction()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Coupon for RangeAccrualCoupon {
    fn nominal(&self) -> f64 {
        self.nominal
    }

    fn rate(&self) -> f64 {
        self.fixed_rate * self.accrual_fraction()
    }

    fn accrual_start(&self) -> Date {
        self.accrual_start
    }

    fn accrual_end(&self) -> Date {
        self.accrual_end
    }

    fn accrual_period(&self) -> f64 {
        self.day_counter
            .year_fraction(self.accrual_start, self.accrual_end)
    }

    fn day_counter(&self) -> DayCounter {
        self.day_counter
    }
}

// ===========================================================================
// Sub-Period Coupon
// ===========================================================================

/// Sub-period coupon: a coupon with multiple sub-period fixings.
///
/// The sub-period rates can be compounded or averaged.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SubPeriodType {
    /// Compound: (1+r₁τ₁)(1+r₂τ₂)…−1
    Compounding,
    /// Average: (r₁τ₁ + r₂τ₂ + …) / (τ₁ + τ₂ + …)
    Averaging,
}

/// A coupon composed of multiple sub-period fixings.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SubPeriodCoupon {
    /// Payment date.
    pub payment_date: Date,
    /// Notional amount.
    pub nominal: f64,
    /// Start of accrual period.
    pub accrual_start: Date,
    /// End of accrual period.
    pub accrual_end: Date,
    /// Day counter for year-fraction computation.
    pub day_counter: DayCounter,
    /// Additive spread over the compounded/averaged rate.
    pub spread: f64,
    /// Sub-period fixing rates.
    pub sub_rates: Vec<f64>,
    /// Year fractions for each sub-period.
    pub sub_year_fractions: Vec<f64>,
    /// Compounding or averaging rule for combining sub-periods.
    pub sub_period_type: SubPeriodType,
}

impl SubPeriodCoupon {
    pub fn new(
        payment_date: Date,
        nominal: f64,
        accrual_start: Date,
        accrual_end: Date,
        day_counter: DayCounter,
        spread: f64,
        sub_rates: Vec<f64>,
        sub_year_fractions: Vec<f64>,
        sub_period_type: SubPeriodType,
    ) -> Self {
        Self {
            payment_date,
            nominal,
            accrual_start,
            accrual_end,
            day_counter,
            spread,
            sub_rates,
            sub_year_fractions,
            sub_period_type,
        }
    }

    /// Compute the effective compounded or averaged rate over all sub-periods.
    pub fn effective_rate(&self) -> f64 {
        if self.sub_rates.is_empty() {
            return self.spread;
        }

        match self.sub_period_type {
            SubPeriodType::Compounding => {
                let mut compound = 1.0;
                for (r, tau) in self.sub_rates.iter().zip(self.sub_year_fractions.iter()) {
                    compound *= 1.0 + r * tau;
                }
                let total_tau: f64 = self.sub_year_fractions.iter().sum();
                if total_tau.abs() < 1e-15 {
                    return self.spread;
                }
                (compound - 1.0) / total_tau + self.spread
            }
            SubPeriodType::Averaging => {
                let weighted_sum: f64 = self
                    .sub_rates
                    .iter()
                    .zip(self.sub_year_fractions.iter())
                    .map(|(r, tau)| r * tau)
                    .sum();
                let total_tau: f64 = self.sub_year_fractions.iter().sum();
                if total_tau.abs() < 1e-15 {
                    return self.spread;
                }
                weighted_sum / total_tau + self.spread
            }
        }
    }
}

impl CashFlow for SubPeriodCoupon {
    fn date(&self) -> Date {
        self.payment_date
    }

    fn amount(&self) -> f64 {
        self.nominal * self.effective_rate() * self.accrual_period()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Coupon for SubPeriodCoupon {
    fn nominal(&self) -> f64 {
        self.nominal
    }

    fn rate(&self) -> f64 {
        self.effective_rate()
    }

    fn accrual_start(&self) -> Date {
        self.accrual_start
    }

    fn accrual_end(&self) -> Date {
        self.accrual_end
    }

    fn accrual_period(&self) -> f64 {
        self.day_counter
            .year_fraction(self.accrual_start, self.accrual_end)
    }

    fn day_counter(&self) -> DayCounter {
        self.day_counter
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_time::Month;

    // ---- Digital Coupon ----

    #[test]
    fn digital_call_triggered() {
        let dc = DigitalCoupon::new(
            Date::from_ymd(2026, Month::January, 15),
            1_000_000.0,
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
            DayCounter::Actual360,
            0.05,  // reference rate
            0.04,  // strike
            0.01,  // 1% cash rate
            true,  // call
        );
        assert!(dc.is_triggered());
        let amount = dc.amount();
        // 1M * 0.01 * ~0.5
        assert!(amount > 4_800.0 && amount < 5_200.0, "Digital call amount = {amount}");
    }

    #[test]
    fn digital_call_not_triggered() {
        let dc = DigitalCoupon::new(
            Date::from_ymd(2026, Month::January, 15),
            1_000_000.0,
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
            DayCounter::Actual360,
            0.03,  // reference rate below strike
            0.04,
            0.01,
            true,
        );
        assert!(!dc.is_triggered());
        assert_abs_diff_eq!(dc.amount(), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn digital_put_triggered() {
        let dc = DigitalCoupon::new(
            Date::from_ymd(2026, Month::January, 15),
            1_000_000.0,
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
            DayCounter::Actual360,
            0.03,
            0.04,
            0.01,
            false, // put
        );
        assert!(dc.is_triggered());
        assert!(dc.amount() > 0.0);
    }

    // ---- Cap/Floored Coupon ----

    #[test]
    fn cap_floored_collar() {
        let cf = CapFlooredCoupon::new(
            Date::from_ymd(2026, Month::January, 15),
            1_000_000.0,
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
            DayCounter::Actual360,
            0.06,  // reference rate
            0.001, // spread
            1.0,   // gearing
            Some(0.05), // cap
            Some(0.02), // floor
        );
        // 0.06 + 0.001 = 0.061 → capped at 0.05
        assert_abs_diff_eq!(cf.effective_rate(), 0.05, epsilon = 1e-15);
    }

    #[test]
    fn cap_floored_floor_active() {
        let cf = CapFlooredCoupon::new(
            Date::from_ymd(2026, Month::January, 15),
            1_000_000.0,
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
            DayCounter::Actual360,
            0.01,
            0.0,
            1.0,
            Some(0.05),
            Some(0.02),
        );
        // 0.01 → floored at 0.02
        assert_abs_diff_eq!(cf.effective_rate(), 0.02, epsilon = 1e-15);
    }

    #[test]
    fn cap_floored_no_cap_no_floor() {
        let cf = CapFlooredCoupon::new(
            Date::from_ymd(2026, Month::January, 15),
            1_000_000.0,
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
            DayCounter::Actual360,
            0.04,
            0.005,
            1.0,
            None,
            None,
        );
        assert_abs_diff_eq!(cf.effective_rate(), 0.045, epsilon = 1e-15);
    }

    // ---- Range Accrual ----

    #[test]
    fn range_accrual_all_in_range() {
        let rates = vec![0.03, 0.035, 0.04, 0.038, 0.032];
        let ra = RangeAccrualCoupon::new(
            Date::from_ymd(2026, Month::January, 15),
            1_000_000.0,
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
            DayCounter::Actual360,
            0.05,
            0.02, // lower
            0.05, // upper
            rates,
        );
        assert_abs_diff_eq!(ra.accrual_fraction(), 1.0, epsilon = 1e-15);
    }

    #[test]
    fn range_accrual_partial() {
        let rates = vec![0.03, 0.06, 0.04, 0.07, 0.035]; // 3 of 5 in range
        let ra = RangeAccrualCoupon::new(
            Date::from_ymd(2026, Month::January, 15),
            1_000_000.0,
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
            DayCounter::Actual360,
            0.05,
            0.02,
            0.05,
            rates,
        );
        assert_abs_diff_eq!(ra.accrual_fraction(), 0.6, epsilon = 1e-15);
        let full_amount = 1_000_000.0 * 0.05 * ra.accrual_period();
        assert_abs_diff_eq!(ra.amount(), full_amount * 0.6, epsilon = 1e-6);
    }

    // ---- Sub-Period Coupon ----

    #[test]
    fn sub_period_compounding() {
        // Two sub-periods: 3% for 0.25Y and 4% for 0.25Y
        let sp = SubPeriodCoupon::new(
            Date::from_ymd(2026, Month::January, 15),
            1_000_000.0,
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
            DayCounter::Actual360,
            0.0,
            vec![0.03, 0.04],
            vec![0.25, 0.25],
            SubPeriodType::Compounding,
        );
        // Compound: (1+0.03*0.25)(1+0.04*0.25) - 1 = 0.017575
        // Annualized over 0.5Y: 0.017575/0.5 = 0.03515
        let expected_rate = ((1.0 + 0.03 * 0.25) * (1.0 + 0.04 * 0.25) - 1.0) / 0.5;
        assert_abs_diff_eq!(sp.effective_rate(), expected_rate, epsilon = 1e-12);
    }

    #[test]
    fn sub_period_averaging() {
        let sp = SubPeriodCoupon::new(
            Date::from_ymd(2026, Month::January, 15),
            1_000_000.0,
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
            DayCounter::Actual360,
            0.0,
            vec![0.03, 0.04],
            vec![0.25, 0.25],
            SubPeriodType::Averaging,
        );
        // Average: (0.03*0.25 + 0.04*0.25) / 0.5 = 0.035
        assert_abs_diff_eq!(sp.effective_rate(), 0.035, epsilon = 1e-12);
    }

    #[test]
    fn sub_period_with_spread() {
        let sp = SubPeriodCoupon::new(
            Date::from_ymd(2026, Month::January, 15),
            1_000_000.0,
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
            DayCounter::Actual360,
            0.001, // 10bp spread
            vec![0.03, 0.04],
            vec![0.25, 0.25],
            SubPeriodType::Averaging,
        );
        assert_abs_diff_eq!(sp.effective_rate(), 0.036, epsilon = 1e-12);
    }
}
