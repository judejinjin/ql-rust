//! Convertible bond instrument.
//!
//! A convertible bond is a fixed-rate bond with an embedded option
//! allowing the holder to convert the bond into a fixed number of
//! shares of the issuer's stock.

use ql_cashflows::{fixed_leg, add_notional_exchange, Leg};
use ql_time::{Date, DayCounter, Schedule};

/// A convertible bond.
///
/// The holder can convert the bond into `conversion_ratio` shares of
/// stock at any time before maturity (American-style conversion).
#[derive(Debug)]
pub struct ConvertibleBond {
    /// Face value / notional.
    pub face_amount: f64,
    /// Settlement days.
    pub settlement_days: u32,
    /// Issue date.
    pub issue_date: Date,
    /// Maturity date.
    pub maturity_date: Date,
    /// Cash flows (coupons + final redemption).
    pub cashflows: Leg,
    /// Fixed coupon rate.
    pub coupon_rate: f64,
    /// Number of shares received upon conversion per face amount.
    pub conversion_ratio: f64,
}

impl ConvertibleBond {
    /// Create a new convertible bond.
    pub fn new(
        face_amount: f64,
        settlement_days: u32,
        schedule: &Schedule,
        coupon_rate: f64,
        day_counter: DayCounter,
        conversion_ratio: f64,
    ) -> Self {
        let dates = schedule.dates();
        let issue_date = dates[0];
        let maturity_date = *dates.last().unwrap();

        let mut cashflows = fixed_leg(schedule, &[face_amount], &[coupon_rate], day_counter);
        add_notional_exchange(&mut cashflows, maturity_date, face_amount);

        Self {
            face_amount,
            settlement_days,
            issue_date,
            maturity_date,
            cashflows,
            coupon_rate,
            conversion_ratio,
        }
    }

    /// The parity (conversion) value given the current stock price.
    ///
    /// Parity = conversion_ratio × stock_price.
    pub fn conversion_value(&self, stock_price: f64) -> f64 {
        self.conversion_ratio * stock_price
    }

    /// Whether the bond has matured.
    pub fn is_expired(&self, ref_date: Date) -> bool {
        self.maturity_date < ref_date
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ql_time::Month;

    fn make_convertible() -> ConvertibleBond {
        let schedule = Schedule::from_dates(vec![
            Date::from_ymd(2025, Month::January, 15),
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
            Date::from_ymd(2026, Month::July, 15),
            Date::from_ymd(2027, Month::January, 15),
        ]);

        ConvertibleBond::new(
            1000.0,  // $1000 face
            2,
            &schedule,
            0.03,    // 3% coupon
            DayCounter::Actual360,
            10.0,    // converts into 10 shares
        )
    }

    #[test]
    fn convertible_creation() {
        let bond = make_convertible();
        assert_eq!(bond.face_amount, 1000.0);
        assert_eq!(bond.coupon_rate, 0.03);
        assert_eq!(bond.conversion_ratio, 10.0);
        // 4 coupons + 1 redemption
        assert_eq!(bond.cashflows.len(), 5);
    }

    #[test]
    fn convertible_conversion_value() {
        let bond = make_convertible();
        // If stock is at $120, conversion value = 10 × 120 = 1200
        assert!((bond.conversion_value(120.0) - 1200.0).abs() < 1e-10);
        // If stock is at $80, conversion value = 10 × 80 = 800
        assert!((bond.conversion_value(80.0) - 800.0).abs() < 1e-10);
    }

    #[test]
    fn convertible_expired() {
        let bond = make_convertible();
        assert!(!bond.is_expired(Date::from_ymd(2025, Month::January, 1)));
        assert!(bond.is_expired(Date::from_ymd(2028, Month::January, 1)));
    }

    #[test]
    fn convertible_maturity() {
        let bond = make_convertible();
        assert_eq!(bond.maturity_date, Date::from_ymd(2027, Month::January, 15));
    }
}
