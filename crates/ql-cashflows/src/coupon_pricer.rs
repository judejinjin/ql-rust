//! Coupon pricer framework.
//!
//! Floating-rate coupons delegate rate computation (including convexity
//! adjustments) to a `FloatingRateCouponPricer` strategy object.
//!
//! The simplest implementation is `BlackIborCouponPricer`, which assumes
//! no convexity adjustment (suitable for vanilla IBOR coupons).

use ql_indexes::IborIndex;
use ql_termstructures::YieldTermStructure;

/// Strategy for computing floating-rate coupon rates.
///
/// Different pricers can implement convexity adjustments, timing
/// adjustments, or model-based rate computations.
pub trait FloatingRateCouponPricer: Send + Sync + std::fmt::Debug {
    /// Compute the adjusted index fixing rate for a given fixing date.
    fn adjusted_rate(
        &self,
        index: &IborIndex,
        fixing_date: ql_time::Date,
        curve: &dyn YieldTermStructure,
    ) -> f64;
}

/// Black's formula-based IBOR coupon pricer.
///
/// For vanilla IBOR coupons, this just returns the forward rate without
/// any convexity adjustment (appropriate for standard swap coupons).
#[derive(Debug, Clone)]
pub struct BlackIborCouponPricer;

impl FloatingRateCouponPricer for BlackIborCouponPricer {
    fn adjusted_rate(
        &self,
        index: &IborIndex,
        fixing_date: ql_time::Date,
        curve: &dyn YieldTermStructure,
    ) -> f64 {
        let value_date = index.value_date(fixing_date);
        let maturity_date = index.maturity_date(value_date);
        let df_start = curve.discount(value_date);
        let df_end = curve.discount(maturity_date);
        index.forecast_fixing(value_date, maturity_date, df_start, df_end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_termstructures::FlatForward;
    use ql_time::{Date, DayCounter, Month};

    #[test]
    fn black_ibor_pricer_forward_rate() {
        let pricer = BlackIborCouponPricer;
        let index = IborIndex::euribor_3m();
        let ref_date = Date::from_ymd(2025, Month::January, 2);
        let curve = FlatForward::new(ref_date, 0.04, DayCounter::Actual360);
        let fixing_date = Date::from_ymd(2025, Month::April, 15);

        let rate = pricer.adjusted_rate(&index, fixing_date, &curve);
        // Should be close to 4% for a flat curve
        assert_abs_diff_eq!(rate, 0.04, epsilon = 0.005);
    }

    #[test]
    fn pricer_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BlackIborCouponPricer>();
    }
}
