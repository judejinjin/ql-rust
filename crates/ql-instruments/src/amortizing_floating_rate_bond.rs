//! Amortizing floating-rate bond.
//!
//! A bond where the principal is repaid gradually over the life of the bond,
//! with coupon payments based on a floating index rate plus a spread applied
//! to the outstanding balance at each period.
//!
//! ## Amortization Types
//!
//! - **EqualPrincipal**: each period repays the same principal amount.
//! - **French**: mortgage-style equal total payment (principal + coupon).
//!
//! ## QuantLib Parity
//!
//! Corresponds to `AmortizingFloatingRateBond` in QuantLib C++
//! (ql/instruments/bonds/amortizingfloatingratebond.hpp).

use ql_cashflows::ibor_coupon::IborCoupon;
use ql_cashflows::{add_notional_exchange, Leg, SimpleCashFlow};
use ql_indexes::IborIndex;
use ql_time::{Date, DayCounter, Schedule};

/// Amortization type for determining the repayment schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmortizationType {
    /// Equal principal repayment each period.
    EqualPrincipal,
    /// French amortization: equal total payment (principal + interest).
    /// Requires an assumed coupon rate for computing the schedule.
    French,
}

/// An amortizing floating-rate bond.
///
/// Combines a declining notional schedule with IBOR-linked coupons.
/// The notional amortizes according to the chosen amortization type,
/// and each period's coupon is computed on the outstanding balance.
#[derive(Debug)]
pub struct AmortizingFloatingRateBond {
    /// Original face amount.
    pub face_amount: f64,
    /// Settlement days.
    pub settlement_days: u32,
    /// Issue date.
    pub issue_date: Date,
    /// Maturity date.
    pub maturity_date: Date,
    /// Cash flows (IBOR coupons + principal repayments).
    pub cashflows: Leg,
    /// Spread over the index rate.
    pub spread: f64,
    /// Reference IBOR index.
    pub index: IborIndex,
    /// Notional outstanding at the start of each period.
    pub notional_schedule: Vec<f64>,
    /// Amortization type.
    pub amortization_type: AmortizationType,
}

impl AmortizingFloatingRateBond {
    /// Create an amortizing floating-rate bond with equal principal repayment.
    ///
    /// Each period repays `face_amount / N` of principal, where N is the number
    /// of coupon periods. The floating coupon is computed on the outstanding
    /// balance at the start of each period.
    pub fn with_equal_principal(
        face_amount: f64,
        settlement_days: u32,
        schedule: &Schedule,
        index: &IborIndex,
        spread: f64,
        day_counter: DayCounter,
    ) -> Self {
        let dates = schedule.dates();
        let n = dates.len() - 1;
        if n == 0 {
            return Self {
                face_amount,
                settlement_days,
                issue_date: dates[0],
                maturity_date: dates[0],
                cashflows: Vec::new(),
                spread,
                index: index.clone(),
                notional_schedule: Vec::new(),
                amortization_type: AmortizationType::EqualPrincipal,
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

            // Floating coupon on outstanding balance
            let fixing_date = index
                .fixing_calendar
                .advance_business_days(start, -(index.fixing_days as i32));

            cashflows.push(Box::new(IborCoupon::new(
                end,
                outstanding,
                start,
                end,
                fixing_date,
                index.clone(),
                spread,
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
            maturity_date: dates[n],
            cashflows,
            spread,
            index: index.clone(),
            notional_schedule: notionals,
            amortization_type: AmortizationType::EqualPrincipal,
        }
    }

    /// Create an amortizing floating-rate bond with French (mortgage-style)
    /// amortization.
    ///
    /// The principal schedule is derived from a French amortization at the
    /// given assumed rate. Actual coupon payments are floating (index + spread),
    /// but the principal amortization is fixed.
    ///
    /// # Parameters
    /// * `assumed_rate` — rate used to compute the French amortization schedule
    ///   (e.g. current forward rate estimate + spread)
    pub fn with_french_amortization(
        face_amount: f64,
        settlement_days: u32,
        schedule: &Schedule,
        index: &IborIndex,
        spread: f64,
        day_counter: DayCounter,
        assumed_rate: f64,
    ) -> Self {
        let dates = schedule.dates();
        let n = dates.len() - 1;
        if n == 0 {
            return Self {
                face_amount,
                settlement_days,
                issue_date: dates[0],
                maturity_date: dates[0],
                cashflows: Vec::new(),
                spread,
                index: index.clone(),
                notional_schedule: Vec::new(),
                amortization_type: AmortizationType::French,
            };
        }

        // Compute French amortization schedule
        let mut notionals = Vec::with_capacity(n);
        let mut principal_payments = Vec::with_capacity(n);
        let mut outstanding = face_amount;

        // Equal total payment per period
        // A = P × r / (1 - (1+r)^-n)  where r is per-period rate
        let accrual_fractions: Vec<f64> = (0..n)
            .map(|i| day_counter.year_fraction(dates[i], dates[i + 1]))
            .collect();
        let avg_accrual = accrual_fractions.iter().sum::<f64>() / n as f64;
        let r = assumed_rate * avg_accrual;
        let annuity = if r.abs() < 1e-15 {
            face_amount / n as f64
        } else {
            face_amount * r / (1.0 - (1.0 + r).powi(-(n as i32)))
        };

        for i in 0..n {
            notionals.push(outstanding);
            let interest = outstanding * assumed_rate * accrual_fractions[i];
            let principal = (annuity - interest).max(0.0).min(outstanding);
            principal_payments.push(principal);
            outstanding -= principal;
        }
        // Ensure final outstanding is zero (adjust last payment for rounding)
        if outstanding.abs() > 1e-10 && !principal_payments.is_empty() {
            let last = principal_payments.len() - 1;
            principal_payments[last] += outstanding;
        }

        // Build cashflows
        let mut cashflows: Leg = Vec::with_capacity(2 * n);
        outstanding = face_amount;

        for i in 0..n {
            let start = dates[i];
            let end = dates[i + 1];

            // Floating coupon on outstanding balance
            let fixing_date = index
                .fixing_calendar
                .advance_business_days(start, -(index.fixing_days as i32));

            cashflows.push(Box::new(IborCoupon::new(
                end,
                outstanding,
                start,
                end,
                fixing_date,
                index.clone(),
                spread,
                day_counter,
            )));

            // Principal repayment
            cashflows.push(Box::new(SimpleCashFlow::new(end, principal_payments[i])));

            outstanding -= principal_payments[i];
        }

        Self {
            face_amount,
            settlement_days,
            issue_date: dates[0],
            maturity_date: dates[n],
            cashflows,
            spread,
            index: index.clone(),
            notional_schedule: notionals,
            amortization_type: AmortizationType::French,
        }
    }

    /// Create from an explicit notional schedule (fully customizable).
    pub fn from_notional_schedule(
        face_amount: f64,
        settlement_days: u32,
        schedule: &Schedule,
        index: &IborIndex,
        spread: f64,
        day_counter: DayCounter,
        notionals: &[f64],
    ) -> Self {
        let dates = schedule.dates();
        let n = dates.len() - 1;

        let notional_schedule: Vec<f64> = (0..n)
            .map(|i| notionals[i.min(notionals.len() - 1)])
            .collect();

        let mut cashflows: Leg = Vec::with_capacity(2 * n);

        for i in 0..n {
            let notional = notional_schedule[i];
            let start = dates[i];
            let end = dates[i + 1];

            let fixing_date = index
                .fixing_calendar
                .advance_business_days(start, -(index.fixing_days as i32));

            cashflows.push(Box::new(IborCoupon::new(
                end,
                notional,
                start,
                end,
                fixing_date,
                index.clone(),
                spread,
                day_counter,
            )));

            // Principal repayment: difference between this period's and next
            // period's notional
            let next_notional = if i + 1 < n {
                notional_schedule[i + 1]
            } else {
                0.0
            };
            let amort = notional - next_notional;
            if amort.abs() > 1e-12 {
                cashflows.push(Box::new(SimpleCashFlow::new(end, amort)));
            }
        }

        Self {
            face_amount,
            settlement_days,
            issue_date: dates[0],
            maturity_date: dates[n],
            cashflows,
            spread,
            index: index.clone(),
            notional_schedule,
            amortization_type: AmortizationType::EqualPrincipal,
        }
    }

    /// Whether the bond has matured relative to the given date.
    pub fn is_expired(&self, ref_date: Date) -> bool {
        self.maturity_date < ref_date
    }

    /// Outstanding notional at the start of period `i`.
    pub fn notional_at(&self, period: usize) -> f64 {
        self.notional_schedule
            .get(period)
            .copied()
            .unwrap_or(0.0)
    }

    /// Weighted average life (WAL) in years.
    ///
    /// WAL = Σ (principal_i × t_i) / total_principal
    /// where t_i is the time to the payment of principal_i.
    pub fn weighted_average_life(&self, day_counter: DayCounter, ref_date: Date) -> f64 {
        let n = self.notional_schedule.len();
        if n == 0 {
            return 0.0;
        }

        let mut wal_num = 0.0;
        let mut total_principal = 0.0;

        for i in 0..n {
            let next_notional = self.notional_schedule.get(i + 1).copied().unwrap_or(0.0);
            let principal = self.notional_schedule[i] - next_notional;
            if principal.abs() > 1e-12 {
                // Payment date: find the corresponding cashflow date
                // In our construction, principal payments share dates with coupons
                let cf_idx = 2 * i + 1; // principal payment index
                if cf_idx < self.cashflows.len() {
                    let t = day_counter.year_fraction(ref_date, self.cashflows[cf_idx].date());
                    wal_num += principal * t;
                    total_principal += principal;
                }
            }
        }

        if total_principal.abs() < 1e-12 {
            0.0
        } else {
            wal_num / total_principal
        }
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

    fn make_index() -> IborIndex {
        IborIndex::euribor_6m()
    }

    #[test]
    fn equal_principal_notionals_decrease() {
        let bond = AmortizingFloatingRateBond::with_equal_principal(
            1_000_000.0, 2, &make_schedule(), &make_index(), 0.002, DayCounter::Actual360,
        );
        assert_eq!(bond.notional_schedule.len(), 4);
        assert_abs_diff_eq!(bond.notional_schedule[0], 1_000_000.0, epsilon = 1e-6);
        assert_abs_diff_eq!(bond.notional_schedule[1], 750_000.0, epsilon = 1e-6);
        assert_abs_diff_eq!(bond.notional_schedule[2], 500_000.0, epsilon = 1e-6);
        assert_abs_diff_eq!(bond.notional_schedule[3], 250_000.0, epsilon = 1e-6);
    }

    #[test]
    fn equal_principal_cashflow_count() {
        let bond = AmortizingFloatingRateBond::with_equal_principal(
            1_000_000.0, 2, &make_schedule(), &make_index(), 0.002, DayCounter::Actual360,
        );
        // 4 IBOR coupons + 4 principal repayments = 8
        assert_eq!(bond.cashflows.len(), 8);
    }

    #[test]
    fn equal_principal_total_repaid() {
        let bond = AmortizingFloatingRateBond::with_equal_principal(
            1_000_000.0, 2, &make_schedule(), &make_index(), 0.002, DayCounter::Actual360,
        );
        // Sum of principal repayments (every other cashflow starting from index 1)
        let total: f64 = bond.cashflows.iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == 1)
            .map(|(_, cf)| cf.amount())
            .sum();
        assert_abs_diff_eq!(total, 1_000_000.0, epsilon = 1e-6);
    }

    #[test]
    fn french_amort_notionals_decrease() {
        let bond = AmortizingFloatingRateBond::with_french_amortization(
            1_000_000.0, 2, &make_schedule(), &make_index(), 0.002, DayCounter::Actual360, 0.05,
        );
        assert_eq!(bond.notional_schedule.len(), 4);
        // Each notional should be strictly smaller than the previous
        for i in 1..bond.notional_schedule.len() {
            assert!(bond.notional_schedule[i] < bond.notional_schedule[i - 1],
                "Notional should decrease: {} >= {}", bond.notional_schedule[i], bond.notional_schedule[i - 1]);
        }
    }

    #[test]
    fn french_amort_total_repaid() {
        let bond = AmortizingFloatingRateBond::with_french_amortization(
            1_000_000.0, 2, &make_schedule(), &make_index(), 0.002, DayCounter::Actual360, 0.05,
        );
        let total: f64 = bond.cashflows.iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == 1)
            .map(|(_, cf)| cf.amount())
            .sum();
        assert_abs_diff_eq!(total, 1_000_000.0, epsilon = 1.0);
    }

    #[test]
    fn from_notional_schedule_custom() {
        let notionals = &[1_000_000.0, 800_000.0, 500_000.0, 200_000.0];
        let bond = AmortizingFloatingRateBond::from_notional_schedule(
            1_000_000.0, 2, &make_schedule(), &make_index(), 0.002, DayCounter::Actual360, notionals,
        );
        assert_eq!(bond.notional_schedule.len(), 4);
        assert_abs_diff_eq!(bond.notional_schedule[0], 1_000_000.0, epsilon = 1e-6);
        assert_abs_diff_eq!(bond.notional_schedule[3], 200_000.0, epsilon = 1e-6);
    }

    #[test]
    fn wal_is_positive() {
        let bond = AmortizingFloatingRateBond::with_equal_principal(
            1_000_000.0, 2, &make_schedule(), &make_index(), 0.002, DayCounter::Actual360,
        );
        let wal = bond.weighted_average_life(
            DayCounter::Actual365Fixed,
            Date::from_ymd(2025, Month::January, 15),
        );
        assert!(wal > 0.0 && wal < 3.0, "WAL should be between 0 and 3 years, got {}", wal);
    }

    #[test]
    fn wal_equal_principal_midpoint() {
        let bond = AmortizingFloatingRateBond::with_equal_principal(
            1_000_000.0, 2, &make_schedule(), &make_index(), 0.002, DayCounter::Actual360,
        );
        let wal = bond.weighted_average_life(
            DayCounter::Actual365Fixed,
            Date::from_ymd(2025, Month::January, 15),
        );
        // For equal principal, WAL should be near the midpoint of the schedule
        assert_abs_diff_eq!(wal, 1.25, epsilon = 0.15);
    }

    #[test]
    fn not_expired_before_maturity() {
        let bond = AmortizingFloatingRateBond::with_equal_principal(
            1_000_000.0, 2, &make_schedule(), &make_index(), 0.002, DayCounter::Actual360,
        );
        assert!(!bond.is_expired(Date::from_ymd(2025, Month::January, 1)));
    }

    #[test]
    fn expired_after_maturity() {
        let bond = AmortizingFloatingRateBond::with_equal_principal(
            1_000_000.0, 2, &make_schedule(), &make_index(), 0.002, DayCounter::Actual360,
        );
        assert!(bond.is_expired(Date::from_ymd(2028, Month::January, 1)));
    }
}
