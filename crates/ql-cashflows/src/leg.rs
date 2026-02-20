//! Leg builder utilities.
//!
//! Functions to construct fixed and floating legs from a `Schedule`.
//! These produce `Leg` values (i.e. `Vec<Box<dyn CashFlow>>`) ready for
//! pricing.

use ql_indexes::IborIndex;
use ql_indexes::OvernightIndex;
use ql_time::{DayCounter, Schedule};

use crate::cashflow::Leg;
use crate::fixed_rate_coupon::FixedRateCoupon;
use crate::ibor_coupon::IborCoupon;
use crate::overnight_coupon::OvernightIndexedCoupon;
use crate::simple_cashflow::SimpleCashFlow;

/// Build a fixed-rate leg from a schedule.
///
/// Each period in the schedule generates a `FixedRateCoupon`.
/// The `notionals` and `rates` slices are extended by repeating the last
/// value if they are shorter than the number of periods.
pub fn fixed_leg(
    schedule: &Schedule,
    notionals: &[f64],
    rates: &[f64],
    day_counter: DayCounter,
) -> Leg {
    let dates = schedule.dates();
    if dates.len() < 2 || notionals.is_empty() || rates.is_empty() {
        return Vec::new();
    }

    let n = dates.len() - 1; // number of coupon periods
    let mut leg: Leg = Vec::with_capacity(n);

    for i in 0..n {
        let notional = notionals[i.min(notionals.len() - 1)];
        let rate = rates[i.min(rates.len() - 1)];
        let start = dates[i];
        let end = dates[i + 1];

        leg.push(Box::new(FixedRateCoupon::new(
            end, // payment at end of period
            notional,
            rate,
            start,
            end,
            day_counter,
        )));
    }

    leg
}

/// Build an IBOR floating-rate leg from a schedule.
///
/// Each period generates an `IborCoupon` with the given index and spread.
/// The fixing date is set to `fixing_days` business days before the
/// start of each period (as determined by the index).
///
/// Spreads slice is extended by repeating the last value if shorter.
pub fn ibor_leg(
    schedule: &Schedule,
    notionals: &[f64],
    index: &IborIndex,
    spreads: &[f64],
    day_counter: DayCounter,
) -> Leg {
    let dates = schedule.dates();
    if dates.len() < 2 || notionals.is_empty() {
        return Vec::new();
    }

    let n = dates.len() - 1;
    let default_spread = if spreads.is_empty() { 0.0 } else { spreads[0] };
    let mut leg: Leg = Vec::with_capacity(n);

    for i in 0..n {
        let notional = notionals[i.min(notionals.len() - 1)];
        let spread = if i < spreads.len() {
            spreads[i]
        } else {
            default_spread
        };
        let start = dates[i];
        let end = dates[i + 1];

        // Fixing date: fixing_days business days before the start
        let fixing_date = index
            .fixing_calendar
            .advance_business_days(start, -(index.fixing_days as i32));

        leg.push(Box::new(IborCoupon::new(
            end, // payment at end of period
            notional,
            start,
            end,
            fixing_date,
            index.clone(),
            spread,
            day_counter,
        )));
    }

    leg
}

/// Append a notional exchange (final principal repayment) to a leg.
pub fn add_notional_exchange(leg: &mut Leg, date: ql_time::Date, notional: f64) {
    leg.push(Box::new(SimpleCashFlow::new(date, notional)));
}

/// Build an overnight-indexed floating leg from a schedule.
///
/// Each period generates an `OvernightIndexedCoupon` whose rate is the
/// compounded overnight rate over the accrual period.
pub fn overnight_leg(
    schedule: &Schedule,
    notionals: &[f64],
    index: &OvernightIndex,
    spreads: &[f64],
    day_counter: DayCounter,
) -> Leg {
    let dates = schedule.dates();
    if dates.len() < 2 || notionals.is_empty() {
        return Vec::new();
    }

    let n = dates.len() - 1;
    let default_spread = if spreads.is_empty() { 0.0 } else { spreads[0] };
    let mut leg: Leg = Vec::with_capacity(n);

    for i in 0..n {
        let notional = notionals[i.min(notionals.len() - 1)];
        let spread = if i < spreads.len() {
            spreads[i]
        } else {
            default_spread
        };
        let start = dates[i];
        let end = dates[i + 1];

        leg.push(Box::new(OvernightIndexedCoupon::new(
            end,
            notional,
            start,
            end,
            index.clone(),
            spread,
            day_counter,
        )));
    }

    leg
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_time::{Date, Month, Schedule};

    fn make_semiannual_schedule() -> Schedule {
        Schedule::from_dates(vec![
            Date::from_ymd(2025, Month::January, 15),
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
            Date::from_ymd(2026, Month::July, 15),
            Date::from_ymd(2027, Month::January, 15),
        ])
    }

    #[test]
    fn fixed_leg_correct_number_of_coupons() {
        let schedule = make_semiannual_schedule();
        let leg = fixed_leg(
            &schedule,
            &[1_000_000.0],
            &[0.05],
            DayCounter::Actual360,
        );
        assert_eq!(leg.len(), 4); // 5 dates = 4 periods
    }

    #[test]
    fn fixed_leg_amounts() {
        let schedule = make_semiannual_schedule();
        let leg = fixed_leg(
            &schedule,
            &[1_000_000.0],
            &[0.05],
            DayCounter::Actual360,
        );
        // Each coupon should be nominal * rate * yf
        for cf in &leg {
            assert!(cf.amount() > 0.0, "coupon amount should be positive");
            // Roughly 50k/2 = ~25k per semiannual period at 5% on 1M
            assert!(cf.amount() > 20_000.0 && cf.amount() < 30_000.0,
                "amount = {}", cf.amount());
        }
    }

    #[test]
    fn fixed_leg_different_notionals() {
        let schedule = Schedule::from_dates(vec![
            Date::from_ymd(2025, Month::January, 15),
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
        ]);
        let leg = fixed_leg(
            &schedule,
            &[1_000_000.0, 500_000.0],
            &[0.05],
            DayCounter::Actual360,
        );
        assert_eq!(leg.len(), 2);
        assert!(leg[0].amount() > leg[1].amount()); // First has higher notional
    }

    #[test]
    fn ibor_leg_correct_number_of_coupons() {
        let schedule = make_semiannual_schedule();
        let index = IborIndex::euribor_6m();
        let leg = ibor_leg(
            &schedule,
            &[1_000_000.0],
            &index,
            &[0.001],
            DayCounter::Actual360,
        );
        assert_eq!(leg.len(), 4);
    }

    #[test]
    fn ibor_leg_payment_dates() {
        let schedule = make_semiannual_schedule();
        let dates = schedule.dates().to_vec();
        let index = IborIndex::euribor_6m();
        let leg = ibor_leg(
            &schedule,
            &[1_000_000.0],
            &index,
            &[0.0],
            DayCounter::Actual360,
        );
        // Payment dates should be the end dates of each period
        for (i, cf) in leg.iter().enumerate() {
            assert_eq!(cf.date(), dates[i + 1]);
        }
    }

    #[test]
    fn notional_exchange() {
        let mut leg: Leg = Vec::new();
        let date = Date::from_ymd(2027, Month::January, 15);
        add_notional_exchange(&mut leg, date, 1_000_000.0);
        assert_eq!(leg.len(), 1);
        assert_abs_diff_eq!(leg[0].amount(), 1_000_000.0, epsilon = 1e-10);
        assert_eq!(leg[0].date(), date);
    }
}
