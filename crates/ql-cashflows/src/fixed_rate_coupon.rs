//! Fixed-rate coupon — a coupon paying a deterministic rate.
//!
//! The coupon amount is `nominal * rate * accrual_period`.

use ql_time::{Date, DayCounter};

use crate::cashflow::CashFlow;
use crate::coupon::Coupon;

/// A fixed-rate interest coupon.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FixedRateCoupon {
    /// Payment date.
    payment_date: Date,
    /// Notional amount.
    nominal: f64,
    /// Annualized coupon rate (e.g. 0.05 = 5%).
    rate: f64,
    /// Start of accrual period.
    accrual_start: Date,
    /// End of accrual period.
    accrual_end: Date,
    /// Day counter for year fraction computation.
    day_counter: DayCounter,
}

impl FixedRateCoupon {
    /// Create a new fixed-rate coupon.
    pub fn new(
        payment_date: Date,
        nominal: f64,
        rate: f64,
        accrual_start: Date,
        accrual_end: Date,
        day_counter: DayCounter,
    ) -> Self {
        Self {
            payment_date,
            nominal,
            rate,
            accrual_start,
            accrual_end,
            day_counter,
        }
    }
}

impl CashFlow for FixedRateCoupon {
    fn date(&self) -> Date {
        self.payment_date
    }

    fn amount(&self) -> f64 {
        self.nominal * self.rate * self.accrual_period()
    }

    fn as_any(&self) -> &dyn std::any::Any { self }
}

impl Coupon for FixedRateCoupon {
    fn nominal(&self) -> f64 {
        self.nominal
    }

    fn rate(&self) -> f64 {
        self.rate
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

    fn make_coupon() -> FixedRateCoupon {
        FixedRateCoupon::new(
            Date::from_ymd(2025, Month::July, 15),
            1_000_000.0,
            0.05,
            Date::from_ymd(2025, Month::January, 15),
            Date::from_ymd(2025, Month::July, 15),
            DayCounter::Actual360,
        )
    }

    #[test]
    fn fixed_coupon_amount() {
        let c = make_coupon();
        let yf = DayCounter::Actual360.year_fraction(
            Date::from_ymd(2025, Month::January, 15),
            Date::from_ymd(2025, Month::July, 15),
        );
        let expected = 1_000_000.0 * 0.05 * yf;
        assert_abs_diff_eq!(c.amount(), expected, epsilon = 1e-6);
    }

    #[test]
    fn fixed_coupon_rate() {
        let c = make_coupon();
        assert_abs_diff_eq!(c.rate(), 0.05, epsilon = 1e-15);
    }

    #[test]
    fn fixed_coupon_nominal() {
        let c = make_coupon();
        assert_abs_diff_eq!(c.nominal(), 1_000_000.0, epsilon = 1e-10);
    }

    #[test]
    fn fixed_coupon_accrual_period() {
        let c = make_coupon();
        // Jan 15 to Jul 15 = 181 days / 360
        let yf = DayCounter::Actual360.year_fraction(
            Date::from_ymd(2025, Month::January, 15),
            Date::from_ymd(2025, Month::July, 15),
        );
        assert_abs_diff_eq!(c.accrual_period(), yf, epsilon = 1e-12);
    }

    #[test]
    fn fixed_coupon_accrued_amount() {
        let c = make_coupon();
        // Halfway through the accrual period
        let mid = Date::from_ymd(2025, Month::April, 15);
        let accrued = c.accrued_amount(mid);
        assert!(accrued > 0.0);
        assert!(accrued < c.amount());
    }

    #[test]
    fn fixed_coupon_accrued_before_start() {
        let c = make_coupon();
        let before = Date::from_ymd(2025, Month::January, 1);
        assert_abs_diff_eq!(c.accrued_amount(before), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn fixed_coupon_payment_date() {
        let c = make_coupon();
        assert_eq!(c.date(), Date::from_ymd(2025, Month::July, 15));
    }
}
