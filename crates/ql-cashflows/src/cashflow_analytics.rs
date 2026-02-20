//! Cash flow analytics — NPV, BPS, accrued amount, and duration.
//!
//! These functions operate on a `Leg` and a discount curve, providing
//! the fundamental pricing analytics for fixed-income instruments.

use ql_time::Date;
use ql_termstructures::YieldTermStructure;

use crate::cashflow::Leg;
use crate::coupon::Coupon;

/// Net present value of a leg.
///
/// Sums `amount * df` for each cash flow that has not occurred as of `settle`.
pub fn npv(leg: &Leg, curve: &dyn YieldTermStructure, settle: Date) -> f64 {
    let mut total = 0.0;
    for cf in leg {
        if !cf.has_occurred(settle) {
            let df = curve.discount(cf.date());
            total += cf.amount() * df;
        }
    }
    total
}

/// Basis point sensitivity (BPS) of a leg.
///
/// The BPS is the change in NPV for a 1 basis point (0.0001) parallel
/// shift in the curve. For fixed-rate legs, this equals:
/// `sum(nominal * 0.0001 * yf * df)` for each coupon not yet occurred.
///
/// This is a first-order approximation: we compute the "DV01-like" measure
/// by computing the NPV of each coupon's year-fraction-weighted notional.
pub fn bps(leg: &Leg, curve: &dyn YieldTermStructure, settle: Date) -> f64 {
    let mut total = 0.0;
    for cf in leg {
        if !cf.has_occurred(settle) {
            if let Some(coupon) = cf.as_any().downcast_ref::<crate::fixed_rate_coupon::FixedRateCoupon>() {
                let df = curve.discount(cf.date());
                total += coupon.nominal() * 0.0001 * coupon.accrual_period() * df;
            }
        }
    }
    total
}

/// Total accrued amount on a leg at the given date.
///
/// Sums accrued interest from all coupons whose accrual period contains `date`.
pub fn accrued_amount(leg: &Leg, date: Date) -> f64 {
    let mut total = 0.0;
    for cf in leg {
        if let Some(coupon) = cf.as_any().downcast_ref::<crate::fixed_rate_coupon::FixedRateCoupon>() {
            total += coupon.accrued_amount(date);
        }
    }
    total
}

/// Net present value of a leg with separate forecast and discount curves.
///
/// For each cash flow:
/// - If it is an `IborCoupon`, forecast the rate from `forecast_curve`
/// - If it is an `OvernightIndexedCoupon`, forecast the rate from `forecast_curve`
/// - Discount the resulting amount with `discount_curve`
///
/// This is the key function for multi-curve / CSA-aware pricing.
pub fn npv_with_forecast(
    leg: &Leg,
    forecast_curve: &dyn YieldTermStructure,
    discount_curve: &dyn YieldTermStructure,
    settle: Date,
) -> f64 {
    let mut total = 0.0;
    for cf in leg {
        if !cf.has_occurred(settle) {
            let df = discount_curve.discount(cf.date());
            // Try forecasting IBOR coupons
            if let Some(ibor) = cf.as_any().downcast_ref::<crate::ibor_coupon::IborCoupon>() {
                let rate = ibor.forecast_rate(forecast_curve);
                total += ibor.nominal() * rate * ibor.accrual_period() * df;
            }
            // Try forecasting overnight coupons
            else if let Some(ois) = cf.as_any().downcast_ref::<crate::overnight_coupon::OvernightIndexedCoupon>() {
                let rate = ois.forecast_rate(forecast_curve);
                let yf = ois.accrual_period();
                total += ois.nominal() * rate * yf * df;
            }
            // For other cash flows (fixed coupons, notional exchanges), use amount()
            else {
                total += cf.amount() * df;
            }
        }
    }
    total
}

/// Macaulay duration of a leg.
///
/// Weighted average time to payment, where the weights are the present
/// values of each cash flow:
///
/// ```text
/// D = sum(t_i * cf_i * df_i) / sum(cf_i * df_i)
/// ```
pub fn duration(leg: &Leg, curve: &dyn YieldTermStructure, settle: Date) -> f64 {
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for cf in leg {
        if !cf.has_occurred(settle) {
            let t = curve.time_from_reference(cf.date());
            let df = curve.discount(cf.date());
            let pv = cf.amount() * df;
            numerator += t * pv;
            denominator += pv;
        }
    }

    if denominator.abs() < 1e-15 {
        0.0
    } else {
        numerator / denominator
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_time::{DayCounter, Month, Schedule};
    use ql_termstructures::FlatForward;
    use crate::leg::fixed_leg;

    fn make_test_leg() -> (Leg, Date) {
        let schedule = Schedule::from_dates(vec![
            Date::from_ymd(2025, Month::January, 15),
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
            Date::from_ymd(2026, Month::July, 15),
            Date::from_ymd(2027, Month::January, 15),
        ]);
        let leg = fixed_leg(&schedule, &[1_000_000.0], &[0.05], DayCounter::Actual360);
        let ref_date = Date::from_ymd(2025, Month::January, 2);
        (leg, ref_date)
    }

    #[test]
    fn npv_positive_for_fixed_leg() {
        let (leg, ref_date) = make_test_leg();
        let curve = FlatForward::new(ref_date, 0.04, DayCounter::Actual360);
        let pv = npv(&leg, &curve, ref_date);
        assert!(pv > 0.0, "NPV should be positive, got {pv}");
    }

    #[test]
    fn npv_higher_rate_lower_pv() {
        let (leg, ref_date) = make_test_leg();
        let curve_low = FlatForward::new(ref_date, 0.03, DayCounter::Actual360);
        let curve_high = FlatForward::new(ref_date, 0.06, DayCounter::Actual360);
        let pv_low = npv(&leg, &curve_low, ref_date);
        let pv_high = npv(&leg, &curve_high, ref_date);
        assert!(pv_low > pv_high, "Higher discount rate should give lower PV");
    }

    #[test]
    fn npv_excludes_past_flows() {
        let (leg, _) = make_test_leg();
        // Settle after first coupon
        let settle = Date::from_ymd(2025, Month::August, 1);
        let curve = FlatForward::new(settle, 0.04, DayCounter::Actual360);
        let pv_all = npv(&leg, &curve, Date::from_ymd(2025, Month::January, 2));
        let pv_future = npv(&leg, &curve, settle);
        assert!(pv_future < pv_all, "Should exclude past cash flows");
    }

    #[test]
    fn bps_positive() {
        let (leg, ref_date) = make_test_leg();
        let curve = FlatForward::new(ref_date, 0.04, DayCounter::Actual360);
        let b = bps(&leg, &curve, ref_date);
        assert!(b > 0.0, "BPS should be positive for a fixed leg, got {b}");
    }

    #[test]
    fn accrued_amount_mid_period() {
        let (leg, _) = make_test_leg();
        let mid = Date::from_ymd(2025, Month::April, 15);
        let acc = accrued_amount(&leg, mid);
        assert!(acc > 0.0, "Should have accrued interest mid-period, got {acc}");
    }

    #[test]
    fn accrued_amount_before_first_coupon() {
        let (leg, _) = make_test_leg();
        let before = Date::from_ymd(2025, Month::January, 1);
        let acc = accrued_amount(&leg, before);
        assert_abs_diff_eq!(acc, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn duration_positive() {
        let (leg, ref_date) = make_test_leg();
        let curve = FlatForward::new(ref_date, 0.04, DayCounter::Actual360);
        let dur = duration(&leg, &curve, ref_date);
        assert!(dur > 0.0, "Duration should be positive, got {dur}");
        // For a 2Y semiannual fixed leg, Macaulay duration ~ 1.0 years
        assert!(dur > 0.5 && dur < 2.5, "Duration = {dur}");
    }

    #[test]
    fn npv_of_5y_swap_fixed_leg() {
        // Phase 4 deliverable: generate a 5Y semiannual swap fixed leg and compute NPV
        let mut dates = vec![Date::from_ymd(2025, Month::January, 15)];
        for y in 0..5 {
            dates.push(Date::from_ymd(2025 + y, Month::July, 15));
            dates.push(Date::from_ymd(2026 + y, Month::January, 15));
        }
        let schedule = Schedule::from_dates(dates);
        let leg = fixed_leg(&schedule, &[10_000_000.0], &[0.05], DayCounter::Actual360);
        assert_eq!(leg.len(), 10); // 10 semiannual periods

        let ref_date = Date::from_ymd(2025, Month::January, 2);
        let curve = FlatForward::new(ref_date, 0.04, DayCounter::Actual360);
        let pv = npv(&leg, &curve, ref_date);

        // At 5% coupon vs 4% discount, PV of coupons should be > sum of undiscounted
        // Coupon per period ~ 10M * 5% * 0.5 = ~250k, total ~ 2.5M
        assert!(pv > 2_000_000.0 && pv < 3_000_000.0,
            "NPV of 5Y fixed leg = {pv}");
    }
}
