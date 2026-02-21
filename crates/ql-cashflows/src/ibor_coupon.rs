//! Floating-rate coupon trait and IBOR coupon.
//!
//! A floating-rate coupon's rate is determined by a reference index fixing.
//! The `IborCoupon` struct projects the rate from a yield curve when the
//! fixing date is in the future, or uses a stored historical fixing.

use ql_time::{Date, DayCounter};

use ql_indexes::Index;
use ql_indexes::IborIndex;
use ql_indexes::index::IndexManager;
use ql_termstructures::YieldTermStructure;

use crate::cashflow::CashFlow;
use crate::coupon::Coupon;

// ===========================================================================
// IborCoupon
// ===========================================================================

/// An IBOR-indexed floating-rate coupon.
///
/// The coupon rate is determined by the IBOR index fixing at `fixing_date`,
/// plus an optional spread.
#[derive(Debug, Clone)]
pub struct IborCoupon {
    /// Payment date.
    payment_date: Date,
    /// Notional.
    nominal: f64,
    /// Start of accrual period.
    accrual_start: Date,
    /// End of accrual period.
    accrual_end: Date,
    /// Fixing date (when the index rate is observed).
    fixing_date: Date,
    /// The IBOR index.
    index: IborIndex,
    /// Additive spread over the index rate (e.g. 0.001 = 10bp).
    spread: f64,
    /// Day counter for accrual.
    day_counter: DayCounter,
    /// Gearing (multiplicative factor on the index rate). Usually 1.0.
    gearing: f64,
    /// Cached/provided fixing rate. If None, must be forecasted.
    cached_rate: Option<f64>,
}

impl IborCoupon {
    /// Create a new IBOR coupon.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        payment_date: Date,
        nominal: f64,
        accrual_start: Date,
        accrual_end: Date,
        fixing_date: Date,
        index: IborIndex,
        spread: f64,
        day_counter: DayCounter,
    ) -> Self {
        Self {
            payment_date,
            nominal,
            accrual_start,
            accrual_end,
            fixing_date,
            index,
            spread,
            day_counter,
            gearing: 1.0,
            cached_rate: None,
        }
    }

    /// Set the gearing (multiplicative factor).
    pub fn with_gearing(mut self, gearing: f64) -> Self {
        self.gearing = gearing;
        self
    }

    /// Set a cached/known rate (bypasses forecasting).
    pub fn with_rate(mut self, rate: f64) -> Self {
        self.cached_rate = Some(rate);
        self
    }

    /// The fixing date.
    pub fn fixing_date(&self) -> Date {
        self.fixing_date
    }

    /// The IBOR index.
    pub fn index(&self) -> &IborIndex {
        &self.index
    }

    /// The spread.
    pub fn spread(&self) -> f64 {
        self.spread
    }

    /// The gearing.
    pub fn gearing(&self) -> f64 {
        self.gearing
    }

    /// Compute the index fixing by forecasting from a yield curve.
    ///
    /// rate = gearing * index_rate + spread
    pub fn forecast_rate(&self, curve: &dyn YieldTermStructure) -> f64 {
        let value_date = self.index.value_date(self.fixing_date);
        let maturity_date = self.index.maturity_date(value_date);

        let df_start = curve.discount(value_date);
        let df_end = curve.discount(maturity_date);

        let index_rate = self.index.forecast_fixing(value_date, maturity_date, df_start, df_end);
        self.gearing * index_rate + self.spread
    }

    /// Get the effective rate: cached if available, otherwise need a curve.
    pub fn effective_rate(&self) -> f64 {
        self.cached_rate
            .unwrap_or(self.gearing * self.spread) // Fallback if no rate set and no curve
    }

    /// Resolve the coupon rate using the QuantLib-style priority:
    ///
    /// 1. If a cached rate is set, use it.
    /// 2. If the fixing date is on or before `eval_date`, look up the
    ///    `IndexManager` for a historical fixing.
    /// 3. Otherwise, project from the forecast curve.
    ///
    /// This is the correct method for pricing floating legs where some
    /// fixings are already known (historical) and others are in the future.
    pub fn resolve_rate(
        &self,
        eval_date: Date,
        forecast_curve: &dyn YieldTermStructure,
    ) -> f64 {
        // 1. Cached rate wins
        if let Some(r) = self.cached_rate {
            return r;
        }
        // 2. Historical fixing
        if self.fixing_date <= eval_date {
            if let Some(fixing) = IndexManager::instance().get_fixing(
                self.index.name(),
                self.fixing_date,
            ) {
                return self.gearing * fixing + self.spread;
            }
        }
        // 3. Forecast from curve
        self.forecast_rate(forecast_curve)
    }
}

impl CashFlow for IborCoupon {
    fn date(&self) -> Date {
        self.payment_date
    }

    fn amount(&self) -> f64 {
        let rate = self.effective_rate();
        self.nominal * rate * self.accrual_period()
    }

    fn as_any(&self) -> &dyn std::any::Any { self }
}

impl Coupon for IborCoupon {
    fn nominal(&self) -> f64 {
        self.nominal
    }

    fn rate(&self) -> f64 {
        self.effective_rate()
    }

    fn accrual_start(&self) -> Date {
        self.accrual_start
    }

    fn accrual_end(&self) -> Date {
        self.accrual_end
    }

    fn accrual_period(&self) -> f64 {
        self.day_counter
            .year_fraction(self.accrual_start, self.accrual_end)
    }

    fn day_counter(&self) -> DayCounter {
        self.day_counter
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_indexes::Index;
    use ql_time::Month;
    use ql_termstructures::FlatForward;

    fn make_ibor_coupon() -> IborCoupon {
        let index = IborIndex::euribor_3m();
        IborCoupon::new(
            Date::from_ymd(2025, Month::July, 17),
            1_000_000.0,
            Date::from_ymd(2025, Month::April, 17),
            Date::from_ymd(2025, Month::July, 17),
            Date::from_ymd(2025, Month::April, 15), // fixing 2 days before
            index,
            0.001, // 10bp spread
            DayCounter::Actual360,
        )
    }

    #[test]
    fn ibor_coupon_with_cached_rate() {
        let c = make_ibor_coupon().with_rate(0.04);
        // rate = 0.04 (cached), spread already included in cached rate
        assert_abs_diff_eq!(c.rate(), 0.04, epsilon = 1e-15);
        let yf = DayCounter::Actual360.year_fraction(
            Date::from_ymd(2025, Month::April, 17),
            Date::from_ymd(2025, Month::July, 17),
        );
        let expected = 1_000_000.0 * 0.04 * yf;
        assert_abs_diff_eq!(c.amount(), expected, epsilon = 1e-6);
    }

    #[test]
    fn ibor_coupon_forecast_from_curve() {
        let c = make_ibor_coupon();
        let ref_date = Date::from_ymd(2025, Month::January, 2);
        let curve = FlatForward::new(ref_date, 0.04, DayCounter::Actual360);

        let rate = c.forecast_rate(&curve);
        // Should be close to 0.04 + 0.001 = 0.041
        assert!(rate > 0.03 && rate < 0.06, "rate = {rate}");
    }

    #[test]
    fn ibor_coupon_properties() {
        let c = make_ibor_coupon();
        assert_eq!(c.fixing_date(), Date::from_ymd(2025, Month::April, 15));
        assert_abs_diff_eq!(c.spread(), 0.001, epsilon = 1e-15);
        assert_abs_diff_eq!(c.gearing(), 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(c.nominal(), 1_000_000.0, epsilon = 1e-10);
    }

    #[test]
    fn ibor_coupon_with_gearing() {
        let c = make_ibor_coupon().with_rate(0.04).with_gearing(2.0);
        // gearing doesn't affect cached rate
        assert_abs_diff_eq!(c.rate(), 0.04, epsilon = 1e-15);
    }

    #[test]
    fn resolve_rate_uses_historical_fixing() {
        // Store a historical fixing
        let idx = IborIndex::euribor_3m();
        let fixing_date = Date::from_ymd(2025, Month::April, 15);
        IndexManager::instance().add_fixing(idx.name(), fixing_date, 0.035).unwrap();

        let c = make_ibor_coupon();
        let ref_date = Date::from_ymd(2025, Month::June, 1); // after fixing
        let curve = FlatForward::new(ref_date, 0.04, DayCounter::Actual360);

        // resolve_rate should use the stored fixing (0.035), not the curve (0.04)
        let rate = c.resolve_rate(ref_date, &curve);
        // Expected: gearing(1.0) * 0.035 + spread(0.001) = 0.036
        assert_abs_diff_eq!(rate, 0.036, epsilon = 1e-10);

        // Clean up
        IndexManager::instance().clear_fixings(idx.name());
    }

    #[test]
    fn resolve_rate_falls_back_to_curve_for_future() {
        let c = make_ibor_coupon();
        // Eval date before fixing date → should forecast from curve
        let ref_date = Date::from_ymd(2025, Month::January, 2);
        let curve = FlatForward::new(ref_date, 0.04, DayCounter::Actual360);

        let rate = c.resolve_rate(ref_date, &curve);
        // Should be close to forecast: ~0.04 + 0.001 = 0.041
        assert!(rate > 0.03 && rate < 0.06, "rate = {rate}");
    }
}
