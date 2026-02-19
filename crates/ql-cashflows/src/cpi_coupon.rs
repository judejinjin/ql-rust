//! CPI-indexed (inflation-linked) coupon.
//!
//! A CPI coupon pays:
//!   N × rate × accrual_fraction × CPI(pay_date) / CPI(base_date)
//!
//! where the CPI ratio adjusts the coupon for inflation since the bond's
//! base date.

use ql_time::{Date, DayCounter};
use serde::{Deserialize, Serialize};

/// A single CPI-indexed coupon.
///
/// Pays `notional × rate × accrual × (CPI(fixing) / base_CPI)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CPICoupon {
    /// Notional principal.
    pub notional: f64,
    /// Annual coupon rate (e.g. 0.0125 for 1.25%).
    pub rate: f64,
    /// Accrual start date.
    pub accrual_start: Date,
    /// Accrual end date.
    pub accrual_end: Date,
    /// Payment date.
    pub payment_date: Date,
    /// Base (reference) CPI level at bond issuance.
    pub base_cpi: f64,
    /// Day counter for accrual fraction.
    pub day_counter: DayCounter,
}

impl CPICoupon {
    /// Create a new CPI-indexed coupon.
    pub fn new(
        notional: f64,
        rate: f64,
        accrual_start: Date,
        accrual_end: Date,
        payment_date: Date,
        base_cpi: f64,
        day_counter: DayCounter,
    ) -> Self {
        Self {
            notional,
            rate,
            accrual_start,
            accrual_end,
            payment_date,
            base_cpi,
            day_counter,
        }
    }

    /// Accrual fraction for this coupon period.
    pub fn accrual_fraction(&self) -> f64 {
        self.day_counter
            .year_fraction(self.accrual_start, self.accrual_end)
    }

    /// Inflation-adjusted coupon amount given the CPI level at the
    /// fixing date.
    ///
    /// Returns: `notional × rate × accrual_fraction × (fixing_cpi / base_cpi)`
    pub fn amount(&self, fixing_cpi: f64) -> f64 {
        let index_ratio = fixing_cpi / self.base_cpi;
        self.notional * self.rate * self.accrual_fraction() * index_ratio
    }

    /// Unadjusted (nominal) coupon amount, ignoring inflation.
    pub fn nominal_amount(&self) -> f64 {
        self.notional * self.rate * self.accrual_fraction()
    }
}

/// Generate a schedule of CPI coupons from bond parameters.
///
/// # Arguments
/// - `notional`: face value
/// - `rate`: annual coupon rate
/// - `dates`: schedule dates (from `Schedule::dates()`)
/// - `base_cpi`: CPI level at issuance
/// - `day_counter`: day count convention
pub fn generate_cpi_coupons(
    notional: f64,
    rate: f64,
    dates: &[Date],
    base_cpi: f64,
    day_counter: DayCounter,
) -> Vec<CPICoupon> {
    let mut coupons = Vec::with_capacity(dates.len().saturating_sub(1));
    for window in dates.windows(2) {
        coupons.push(CPICoupon::new(
            notional,
            rate,
            window[0],
            window[1],
            window[1],
            base_cpi,
            day_counter,
        ));
    }
    coupons
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_time::Month;

    #[test]
    fn cpi_coupon_amount_no_inflation() {
        let coupon = CPICoupon::new(
            1_000_000.0,
            0.02,
            Date::from_ymd(2025, Month::January, 15),
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2025, Month::July, 15),
            300.0,
            DayCounter::Actual365Fixed,
        );
        // If CPI unchanged → amount = N × r × accrual × 1.0
        let amount = coupon.amount(300.0);
        let nominal = coupon.nominal_amount();
        assert_abs_diff_eq!(amount, nominal, epsilon = 1e-8);
    }

    #[test]
    fn cpi_coupon_amount_with_inflation() {
        let coupon = CPICoupon::new(
            1_000_000.0,
            0.02,
            Date::from_ymd(2025, Month::January, 15),
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2025, Month::July, 15),
            300.0,
            DayCounter::Actual365Fixed,
        );
        // CPI rose 5% → ratio = 315/300 = 1.05
        let amount = coupon.amount(315.0);
        let nominal = coupon.nominal_amount();
        assert_abs_diff_eq!(amount, nominal * 1.05, epsilon = 1e-6);
    }

    #[test]
    fn generate_cpi_coupon_schedule() {
        let dates = vec![
            Date::from_ymd(2025, Month::January, 15),
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
        ];
        let coupons = generate_cpi_coupons(
            1_000_000.0,
            0.0125,
            &dates,
            300.0,
            DayCounter::Actual365Fixed,
        );
        assert_eq!(coupons.len(), 2);
        assert_eq!(coupons[0].accrual_start, dates[0]);
        assert_eq!(coupons[0].accrual_end, dates[1]);
        assert_eq!(coupons[1].accrual_start, dates[1]);
        assert_eq!(coupons[1].accrual_end, dates[2]);
    }
}
