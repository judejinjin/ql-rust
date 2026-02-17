//! Callable and puttable bond instruments.
//!
//! A callable bond gives the issuer the right to redeem the bond early
//! at specified call dates and prices. A puttable bond gives the holder
//! the right to sell back at specified put dates.

use ql_cashflows::{fixed_leg, add_notional_exchange, Leg};
use ql_time::{Date, DayCounter, Schedule};

/// Whether the embedded option is a call (issuer's right) or put (holder's right).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallabilityType {
    /// Issuer can redeem early (lowers bond value vs. non-callable).
    Call,
    /// Holder can sell back early (increases bond value vs. non-puttable).
    Put,
}

/// A single call/put schedule entry: the date on which the option can be
/// exercised and the price at which the bond is redeemed.
#[derive(Debug, Clone)]
pub struct CallabilityScheduleEntry {
    /// Date on which the option is exercisable.
    pub date: Date,
    /// Price (per face amount) at which the bond is called/put.
    pub price: f64,
}

/// A callable (or puttable) fixed-rate bond.
///
/// The bond has a standard coupon schedule plus an embedded option
/// that allows early redemption at dates specified in the callability
/// schedule.
#[derive(Debug)]
pub struct CallableBond {
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
    /// Type of embedded option.
    pub callability_type: CallabilityType,
    /// Schedule of dates and prices at which the bond can be called/put.
    pub callability_schedule: Vec<CallabilityScheduleEntry>,
}

impl CallableBond {
    /// Create a new callable/puttable fixed-rate bond.
    pub fn new(
        face_amount: f64,
        settlement_days: u32,
        schedule: &Schedule,
        coupon_rate: f64,
        day_counter: DayCounter,
        callability_type: CallabilityType,
        callability_schedule: Vec<CallabilityScheduleEntry>,
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
            callability_type,
            callability_schedule,
        }
    }

    /// Whether the bond has matured.
    pub fn is_expired(&self, ref_date: Date) -> bool {
        self.maturity_date < ref_date
    }

    /// Return only the call/put dates that are still exercisable
    /// (i.e., on or after `ref_date`).
    pub fn active_call_dates(&self, ref_date: Date) -> Vec<&CallabilityScheduleEntry> {
        self.callability_schedule
            .iter()
            .filter(|e| e.date >= ref_date)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ql_time::Month;

    fn make_callable_bond() -> CallableBond {
        let schedule = Schedule::from_dates(vec![
            Date::from_ymd(2025, Month::January, 15),
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
            Date::from_ymd(2026, Month::July, 15),
            Date::from_ymd(2027, Month::January, 15),
        ]);

        let call_schedule = vec![
            CallabilityScheduleEntry {
                date: Date::from_ymd(2025, Month::July, 15),
                price: 102.0,
            },
            CallabilityScheduleEntry {
                date: Date::from_ymd(2026, Month::January, 15),
                price: 101.0,
            },
            CallabilityScheduleEntry {
                date: Date::from_ymd(2026, Month::July, 15),
                price: 100.0,
            },
        ];

        CallableBond::new(
            100.0,
            2,
            &schedule,
            0.05,
            DayCounter::Actual360,
            CallabilityType::Call,
            call_schedule,
        )
    }

    #[test]
    fn callable_bond_creation() {
        let bond = make_callable_bond();
        assert_eq!(bond.face_amount, 100.0);
        assert_eq!(bond.coupon_rate, 0.05);
        assert_eq!(bond.callability_type, CallabilityType::Call);
        assert_eq!(bond.callability_schedule.len(), 3);
        // 4 coupons + 1 redemption
        assert_eq!(bond.cashflows.len(), 5);
    }

    #[test]
    fn callable_bond_active_call_dates() {
        let bond = make_callable_bond();
        let ref_date = Date::from_ymd(2026, Month::January, 1);
        let active = bond.active_call_dates(ref_date);
        // Only 2026-01-15 and 2026-07-15 are on or after ref_date
        assert_eq!(active.len(), 2);
    }

    #[test]
    fn callable_bond_expired() {
        let bond = make_callable_bond();
        assert!(!bond.is_expired(Date::from_ymd(2025, Month::January, 1)));
        assert!(bond.is_expired(Date::from_ymd(2028, Month::January, 1)));
    }

    #[test]
    fn callable_bond_maturity() {
        let bond = make_callable_bond();
        assert_eq!(bond.maturity_date, Date::from_ymd(2027, Month::January, 15));
    }

    #[test]
    fn puttable_bond_creation() {
        let schedule = Schedule::from_dates(vec![
            Date::from_ymd(2025, Month::January, 15),
            Date::from_ymd(2026, Month::January, 15),
            Date::from_ymd(2027, Month::January, 15),
        ]);
        let put_schedule = vec![CallabilityScheduleEntry {
            date: Date::from_ymd(2026, Month::January, 15),
            price: 100.0,
        }];
        let bond = CallableBond::new(
            100.0,
            2,
            &schedule,
            0.04,
            DayCounter::Actual360,
            CallabilityType::Put,
            put_schedule,
        );
        assert_eq!(bond.callability_type, CallabilityType::Put);
        assert_eq!(bond.callability_schedule.len(), 1);
    }
}
