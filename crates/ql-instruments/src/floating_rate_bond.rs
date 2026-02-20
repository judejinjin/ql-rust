//! Floating-rate bond (FRN / floater).
//!
//! A floating-rate note pays coupons linked to a reference index (e.g., EURIBOR,
//! SOFR) plus a spread, with notional repayment at maturity.

use ql_cashflows::{add_notional_exchange, ibor_leg, Leg};
use ql_indexes::IborIndex;
use ql_time::{Date, DayCounter, Schedule};

/// A floating-rate bond.
///
/// Coupons reference an IBOR index plus a spread. The constructor builds
/// the full `Leg` (IBOR coupons + notional redemption) so the bond is
/// ready for pricing via any NPV-based engine.
///
/// # Example
/// ```ignore
/// let frn = FloatingRateBond::new(
///     100.0, 2, &schedule, &index, 0.0020, day_counter,
/// );
/// ```
#[derive(Debug)]
pub struct FloatingRateBond {
    /// Face value / notional.
    pub face_amount: f64,
    /// Settlement days (business days between trade and settlement).
    pub settlement_days: u32,
    /// Issue date.
    pub issue_date: Date,
    /// Maturity date.
    pub maturity_date: Date,
    /// Cash flows (IBOR coupons + final redemption).
    pub cashflows: Leg,
    /// Spread over the index rate (e.g. 0.002 for 20bp).
    pub spread: f64,
    /// The reference IBOR index (stored for analytics).
    pub index: IborIndex,
}

impl FloatingRateBond {
    /// Create a floating-rate bond from a coupon schedule.
    ///
    /// # Parameters
    /// - `face_amount`: notional / face value
    /// - `settlement_days`: business days between trade and settlement
    /// - `schedule`: coupon payment schedule
    /// - `index`: IBOR index (e.g. `IborIndex::euribor_3m()`)
    /// - `spread`: spread over the index rate
    /// - `day_counter`: accrual day counter
    pub fn new(
        face_amount: f64,
        settlement_days: u32,
        schedule: &Schedule,
        index: &IborIndex,
        spread: f64,
        day_counter: DayCounter,
    ) -> Self {
        let dates = schedule.dates();
        let issue_date = dates[0];
        let maturity_date = dates[dates.len() - 1];

        let mut cashflows = ibor_leg(schedule, &[face_amount], index, &[spread], day_counter);
        // Add notional redemption at maturity
        add_notional_exchange(&mut cashflows, maturity_date, face_amount);

        Self {
            face_amount,
            settlement_days,
            issue_date,
            maturity_date,
            cashflows,
            spread,
            index: index.clone(),
        }
    }

    /// Create with variable notionals and spreads (amortizing / step-up).
    #[allow(clippy::too_many_arguments)]
    pub fn from_details(
        face_amount: f64,
        settlement_days: u32,
        schedule: &Schedule,
        index: &IborIndex,
        notionals: &[f64],
        spreads: &[f64],
        day_counter: DayCounter,
    ) -> Self {
        let dates = schedule.dates();
        let issue_date = dates[0];
        let maturity_date = dates[dates.len() - 1];

        let mut cashflows = ibor_leg(schedule, notionals, index, spreads, day_counter);
        add_notional_exchange(&mut cashflows, maturity_date, face_amount);

        let spread = spreads[0];
        Self {
            face_amount,
            settlement_days,
            issue_date,
            maturity_date,
            cashflows,
            spread,
            index: index.clone(),
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

    fn make_frn() -> FloatingRateBond {
        let schedule = Schedule::from_dates(vec![
            Date::from_ymd(2025, Month::January, 15),
            Date::from_ymd(2025, Month::April, 15),
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2025, Month::October, 15),
            Date::from_ymd(2026, Month::January, 15),
        ]);
        let index = IborIndex::euribor_3m();
        FloatingRateBond::new(100.0, 2, &schedule, &index, 0.002, DayCounter::Actual360)
    }

    #[test]
    fn frn_cashflows_include_redemption() {
        let frn = make_frn();
        // 4 coupons + 1 redemption = 5 cash flows
        assert_eq!(frn.cashflows.len(), 5);
    }

    #[test]
    fn frn_last_cashflow_is_redemption() {
        let frn = make_frn();
        let last = frn.cashflows.last().unwrap();
        assert!((last.amount() - 100.0).abs() < 1e-10);
        assert_eq!(last.date(), Date::from_ymd(2026, Month::January, 15));
    }

    #[test]
    fn frn_not_expired() {
        let frn = make_frn();
        assert!(!frn.is_expired(Date::from_ymd(2025, Month::January, 1)));
    }

    #[test]
    fn frn_expired_after_maturity() {
        let frn = make_frn();
        assert!(frn.is_expired(Date::from_ymd(2027, Month::January, 1)));
    }

    #[test]
    fn frn_maturity_date() {
        let frn = make_frn();
        assert_eq!(frn.maturity_date, Date::from_ymd(2026, Month::January, 15));
    }

    #[test]
    fn frn_spread_stored() {
        let frn = make_frn();
        assert!((frn.spread - 0.002).abs() < 1e-12);
    }
}
