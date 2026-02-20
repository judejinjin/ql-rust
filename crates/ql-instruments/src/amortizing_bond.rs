//! Amortizing fixed-rate bond.
//!
//! A bond where the principal is repaid gradually over the life of the bond,
//! with coupon payments based on the outstanding balance at each period.

use ql_cashflows::{Leg, SimpleCashFlow};
use ql_cashflows::fixed_rate_coupon::FixedRateCoupon;
use ql_time::{Date, DayCounter, Schedule};

/// An amortizing fixed-rate bond with scheduled principal repayments.
#[derive(Debug)]
pub struct AmortizingBond {
    /// Original face amount.
    pub face_amount: f64,
    /// Settlement days.
    pub settlement_days: u32,
    /// Issue date.
    pub issue_date: Date,
    /// Maturity date.
    pub maturity_date: Date,
    /// Cash flows (coupons + amortization payments).
    pub cashflows: Leg,
    /// Fixed coupon rate.
    pub coupon_rate: f64,
    /// Amortization schedule (notional outstanding for each period).
    pub notional_schedule: Vec<f64>,
}

impl AmortizingBond {
    /// Create an amortizing bond with equal principal repayment.
    ///
    /// The total principal is divided equally across all periods. Coupons
    /// are computed on the outstanding balance at the start of each period.
    pub fn with_equal_principal(
        face_amount: f64,
        settlement_days: u32,
        schedule: &Schedule,
        coupon_rate: f64,
        day_counter: DayCounter,
    ) -> Self {
        let dates = schedule.dates();
        let n = dates.len() - 1; // number of periods
        if n == 0 {
            return Self {
                face_amount,
                settlement_days,
                issue_date: dates[0],
                maturity_date: dates[0],
                cashflows: Vec::new(),
                coupon_rate,
                notional_schedule: Vec::new(),
            };
        }

        let principal_per_period = face_amount / n as f64;
        let mut notionals = Vec::with_capacity(n);
        let mut cashflows: Leg = Vec::with_capacity(2 * n);
        let mut outstanding = face_amount;

        for i in 0..n {
            notionals.push(outstanding);
            let start = dates[i];
            let end = dates[i + 1];

            // Coupon on outstanding balance
            cashflows.push(Box::new(FixedRateCoupon::new(
                end,
                outstanding,
                coupon_rate,
                start,
                end,
                day_counter,
            )));

            // Principal repayment
            cashflows.push(Box::new(SimpleCashFlow::new(end, principal_per_period)));

            outstanding -= principal_per_period;
        }

        Self {
            face_amount,
            settlement_days,
            issue_date: dates[0],
            maturity_date: dates[dates.len() - 1],
            cashflows,
            coupon_rate,
            notional_schedule: notionals,
        }
    }

    /// Create an amortizing bond with custom notional schedule.
    ///
    /// `notionals[i]` is the outstanding principal for period `i`.
    /// Principal repayment for period `i` is `notionals[i] - notionals[i+1]`
    /// (or the full outstanding for the last period).
    pub fn with_notional_schedule(
        face_amount: f64,
        settlement_days: u32,
        schedule: &Schedule,
        coupon_rate: f64,
        notionals: &[f64],
        day_counter: DayCounter,
    ) -> Self {
        let dates = schedule.dates();
        let n = dates.len() - 1;
        if n == 0 || notionals.is_empty() {
            return Self {
                face_amount,
                settlement_days,
                issue_date: dates[0],
                maturity_date: dates[dates.len().saturating_sub(1).max(0)],
                cashflows: Vec::new(),
                coupon_rate,
                notional_schedule: Vec::new(),
            };
        }

        let mut cashflows: Leg = Vec::with_capacity(2 * n);
        let mut notional_sched = Vec::with_capacity(n);

        for i in 0..n {
            let outstanding = notionals[i.min(notionals.len() - 1)];
            notional_sched.push(outstanding);
            let start = dates[i];
            let end = dates[i + 1];

            // Coupon on outstanding balance
            cashflows.push(Box::new(FixedRateCoupon::new(
                end,
                outstanding,
                coupon_rate,
                start,
                end,
                day_counter,
            )));

            // Principal repayment
            let next_outstanding = if i + 1 < n {
                notionals[(i + 1).min(notionals.len() - 1)]
            } else {
                0.0
            };
            let principal_repayment = outstanding - next_outstanding;
            if principal_repayment.abs() > 1e-15 {
                cashflows.push(Box::new(SimpleCashFlow::new(end, principal_repayment)));
            }
        }

        Self {
            face_amount,
            settlement_days,
            issue_date: dates[0],
            maturity_date: dates[dates.len() - 1],
            cashflows,
            coupon_rate,
            notional_schedule: notional_sched,
        }
    }

    /// Whether the bond has matured.
    pub fn is_expired(&self, ref_date: Date) -> bool {
        self.maturity_date < ref_date
    }

    /// The outstanding notional at a given period index.
    pub fn outstanding_at(&self, period: usize) -> f64 {
        self.notional_schedule
            .get(period)
            .copied()
            .unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_time::Month;

    fn make_schedule() -> Schedule {
        Schedule::from_dates(vec![
            Date::from_ymd(2025, Month::January, 15),
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
            Date::from_ymd(2026, Month::July, 15),
            Date::from_ymd(2027, Month::January, 15),
        ])
    }

    #[test]
    fn equal_principal_cashflow_count() {
        let schedule = make_schedule();
        let bond = AmortizingBond::with_equal_principal(
            100.0, 2, &schedule, 0.05, DayCounter::Actual360,
        );
        // 4 periods × (1 coupon + 1 principal) = 8 cash flows
        assert_eq!(bond.cashflows.len(), 8);
    }

    #[test]
    fn equal_principal_notionals_decreasing() {
        let schedule = make_schedule();
        let bond = AmortizingBond::with_equal_principal(
            100.0, 2, &schedule, 0.05, DayCounter::Actual360,
        );
        assert_abs_diff_eq!(bond.notional_schedule[0], 100.0, epsilon = 1e-10);
        assert_abs_diff_eq!(bond.notional_schedule[1], 75.0, epsilon = 1e-10);
        assert_abs_diff_eq!(bond.notional_schedule[2], 50.0, epsilon = 1e-10);
        assert_abs_diff_eq!(bond.notional_schedule[3], 25.0, epsilon = 1e-10);
    }

    #[test]
    fn equal_principal_total_principal_equals_face() {
        let schedule = make_schedule();
        let bond = AmortizingBond::with_equal_principal(
            100.0, 2, &schedule, 0.05, DayCounter::Actual360,
        );
        // Sum of principal repayments = face
        let total_principal: f64 = bond
            .cashflows
            .iter()
            .skip(1) // skip first coupon
            .step_by(2) // every other = principal
            .map(|cf| cf.amount())
            .sum();
        assert_abs_diff_eq!(total_principal, 100.0, epsilon = 1e-10);
    }

    #[test]
    fn custom_notional_schedule() {
        let schedule = make_schedule();
        let notionals = [100.0, 80.0, 50.0, 20.0];
        let bond = AmortizingBond::with_notional_schedule(
            100.0, 2, &schedule, 0.05, &notionals, DayCounter::Actual360,
        );
        // 4 coupons + principal repayments (all nonzero)
        // Period 0: coupon + 20 principal
        // Period 1: coupon + 30 principal
        // Period 2: coupon + 30 principal
        // Period 3: coupon + 20 principal
        assert_eq!(bond.cashflows.len(), 8);
        assert_abs_diff_eq!(bond.outstanding_at(0), 100.0, epsilon = 1e-10);
        assert_abs_diff_eq!(bond.outstanding_at(1), 80.0, epsilon = 1e-10);
    }

    #[test]
    fn coupons_decrease_with_amortization() {
        let schedule = make_schedule();
        let bond = AmortizingBond::with_equal_principal(
            100.0, 2, &schedule, 0.05, DayCounter::Actual360,
        );
        // Get coupon amounts (every other starting from 0)
        let coupons: Vec<f64> = bond
            .cashflows
            .iter()
            .step_by(2)
            .map(|cf| cf.amount())
            .collect();
        // Each coupon should be less than the previous (decreasing notional)
        for w in coupons.windows(2) {
            assert!(w[0] > w[1], "coupons should decrease: {} vs {}", w[0], w[1]);
        }
    }

    #[test]
    fn amortizing_maturity() {
        let schedule = make_schedule();
        let bond = AmortizingBond::with_equal_principal(
            100.0, 2, &schedule, 0.05, DayCounter::Actual360,
        );
        assert_eq!(bond.maturity_date, Date::from_ymd(2027, Month::January, 15));
    }
}
