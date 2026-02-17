//! Fixed-rate and floating-rate bonds.
//!
//! A bond is a sequence of coupon cash flows plus a notional repayment.

use ql_cashflows::{fixed_leg, add_notional_exchange, Leg};
use ql_time::{Date, DayCounter, Schedule};

/// A fixed-rate bond.
#[derive(Debug)]
pub struct FixedRateBond {
    /// Face value / notional.
    pub face_amount: f64,
    /// Settlement days (business days between trade and settlement).
    pub settlement_days: u32,
    /// Issue date.
    pub issue_date: Date,
    /// Maturity date.
    pub maturity_date: Date,
    /// Cash flows (coupons + final redemption).
    pub cashflows: Leg,
    /// Fixed coupon rate.
    pub coupon_rate: f64,
}

impl FixedRateBond {
    /// Create a fixed-rate bond from a coupon schedule.
    pub fn new(
        face_amount: f64,
        settlement_days: u32,
        schedule: &Schedule,
        coupon_rate: f64,
        day_counter: DayCounter,
    ) -> Self {
        let dates = schedule.dates();
        let issue_date = dates[0];
        let maturity_date = dates[dates.len() - 1];

        let mut cashflows = fixed_leg(schedule, &[face_amount], &[coupon_rate], day_counter);
        // Add notional redemption at maturity
        add_notional_exchange(&mut cashflows, maturity_date, face_amount);

        Self {
            face_amount,
            settlement_days,
            issue_date,
            maturity_date,
            cashflows,
            coupon_rate,
        }
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

    fn make_bond() -> FixedRateBond {
        let schedule = Schedule::from_dates(vec![
            Date::from_ymd(2025, Month::January, 15),
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
            Date::from_ymd(2026, Month::July, 15),
            Date::from_ymd(2027, Month::January, 15),
        ]);
        FixedRateBond::new(100.0, 2, &schedule, 0.05, DayCounter::Actual360)
    }

    #[test]
    fn bond_cashflows_include_redemption() {
        let bond = make_bond();
        // 4 coupons + 1 redemption = 5 cash flows
        assert_eq!(bond.cashflows.len(), 5);
    }

    #[test]
    fn bond_last_cashflow_is_redemption() {
        let bond = make_bond();
        let last = bond.cashflows.last().unwrap();
        assert!((last.amount() - 100.0).abs() < 1e-10);
        assert_eq!(last.date(), Date::from_ymd(2027, Month::January, 15));
    }

    #[test]
    fn bond_not_expired() {
        let bond = make_bond();
        assert!(!bond.is_expired(Date::from_ymd(2025, Month::January, 1)));
    }

    #[test]
    fn bond_expired_after_maturity() {
        let bond = make_bond();
        assert!(bond.is_expired(Date::from_ymd(2028, Month::January, 1)));
    }

    #[test]
    fn bond_maturity_date() {
        let bond = make_bond();
        assert_eq!(bond.maturity_date, Date::from_ymd(2027, Month::January, 15));
    }
}
