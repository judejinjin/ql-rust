//! Zero-coupon bond.
//!
//! A bond that pays no coupons, only the face amount at maturity.
//! Typically issued at a discount to face value.

use ql_cashflows::{add_notional_exchange, Leg};
use ql_time::Date;

/// A zero-coupon bond.
#[derive(Debug)]
pub struct ZeroCouponBond {
    /// Face value / notional.
    pub face_amount: f64,
    /// Settlement days.
    pub settlement_days: u32,
    /// Issue date.
    pub issue_date: Date,
    /// Maturity date.
    pub maturity_date: Date,
    /// Cash flows (single redemption at maturity).
    pub cashflows: Leg,
}

impl ZeroCouponBond {
    /// Create a zero-coupon bond.
    pub fn new(
        face_amount: f64,
        settlement_days: u32,
        issue_date: Date,
        maturity_date: Date,
    ) -> Self {
        let mut cashflows: Leg = Vec::with_capacity(1);
        add_notional_exchange(&mut cashflows, maturity_date, face_amount);

        Self {
            face_amount,
            settlement_days,
            issue_date,
            maturity_date,
            cashflows,
        }
    }

    /// Whether the bond has matured.
    pub fn is_expired(&self, ref_date: Date) -> bool {
        self.maturity_date < ref_date
    }

    /// The dirty price given a yield-to-maturity and day count.
    ///
    /// `P = face / (1 + y)^t` where `t` is the year fraction from
    /// settlement to maturity.
    pub fn dirty_price(
        &self,
        ytm: f64,
        settlement_date: Date,
        day_counter: ql_time::DayCounter,
    ) -> f64 {
        let t = day_counter.year_fraction(settlement_date, self.maturity_date);
        self.face_amount / (1.0 + ytm).powf(t)
    }

    /// Implied yield from a dirty price.
    ///
    /// `y = (face / P)^{1/t} - 1`
    pub fn implied_yield(
        &self,
        dirty_price: f64,
        settlement_date: Date,
        day_counter: ql_time::DayCounter,
    ) -> f64 {
        let t = day_counter.year_fraction(settlement_date, self.maturity_date);
        if t.abs() < 1e-15 || dirty_price <= 0.0 {
            return 0.0;
        }
        (self.face_amount / dirty_price).powf(1.0 / t) - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_time::{DayCounter, Month};

    fn make_zcb() -> ZeroCouponBond {
        ZeroCouponBond::new(
            100.0,
            2,
            Date::from_ymd(2025, Month::January, 15),
            Date::from_ymd(2030, Month::January, 15),
        )
    }

    #[test]
    fn zcb_single_cashflow() {
        let bond = make_zcb();
        assert_eq!(bond.cashflows.len(), 1);
        assert_abs_diff_eq!(bond.cashflows[0].amount(), 100.0, epsilon = 1e-10);
        assert_eq!(
            bond.cashflows[0].date(),
            Date::from_ymd(2030, Month::January, 15)
        );
    }

    #[test]
    fn zcb_dirty_price_at_par() {
        let bond = make_zcb();
        let settlement = Date::from_ymd(2025, Month::January, 15);
        // At 0% yield, price = face
        let p = bond.dirty_price(0.0, settlement, DayCounter::Actual365Fixed);
        assert_abs_diff_eq!(p, 100.0, epsilon = 1e-10);
    }

    #[test]
    fn zcb_dirty_price_positive_yield() {
        let bond = make_zcb();
        let settlement = Date::from_ymd(2025, Month::January, 15);
        let p = bond.dirty_price(0.05, settlement, DayCounter::Actual365Fixed);
        // 100 / 1.05^5 ≈ 78.35
        assert!(p > 78.0 && p < 79.0, "price = {p}");
    }

    #[test]
    fn zcb_implied_yield_roundtrip() {
        let bond = make_zcb();
        let settlement = Date::from_ymd(2025, Month::January, 15);
        let dc = DayCounter::Actual365Fixed;
        let target_yield = 0.05;
        let price = bond.dirty_price(target_yield, settlement, dc);
        let y = bond.implied_yield(price, settlement, dc);
        assert_abs_diff_eq!(y, target_yield, epsilon = 1e-10);
    }

    #[test]
    fn zcb_not_expired() {
        let bond = make_zcb();
        assert!(!bond.is_expired(Date::from_ymd(2029, Month::December, 31)));
    }

    #[test]
    fn zcb_expired() {
        let bond = make_zcb();
        assert!(bond.is_expired(Date::from_ymd(2030, Month::January, 16)));
    }
}
