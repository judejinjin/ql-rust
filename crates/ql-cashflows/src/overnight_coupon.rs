//! Overnight indexed coupon — compounded overnight rate over an accrual period.
//!
//! Used for SOFR, ESTR, SONIA-based instruments.
//! The coupon rate is the geometric average (compounded) of daily overnight
//! fixings over the accrual period.

use ql_time::{Date, DayCounter};

use ql_indexes::Index;
use ql_indexes::OvernightIndex;
use ql_indexes::index::IndexManager;
use ql_termstructures::YieldTermStructure;

use crate::cashflow::CashFlow;
use crate::coupon::Coupon;

/// An overnight indexed coupon (compounded overnight rate).
///
/// The rate is computed as:
/// ```text
/// rate = (product of (1 + r_i * d_i) - 1) / accrual_period
/// ```
/// where `r_i` is the overnight fixing on day `i` and `d_i` is the day
/// count fraction for that day.
///
/// When forecast from a yield curve, this simplifies to:
/// ```text
/// rate = (df_start / df_end - 1) / accrual_period
/// ```
#[derive(Debug, Clone)]
pub struct OvernightIndexedCoupon {
    /// Payment date.
    payment_date: Date,
    /// Notional.
    nominal: f64,
    /// Start of accrual period.
    accrual_start: Date,
    /// End of accrual period.
    accrual_end: Date,
    /// The overnight index.
    index: OvernightIndex,
    /// Additive spread.
    spread: f64,
    /// Day counter.
    day_counter: DayCounter,
    /// Cached rate (set after computation or from historical fixings).
    cached_rate: Option<f64>,
}

impl OvernightIndexedCoupon {
    /// Create a new overnight indexed coupon.
    pub fn new(
        payment_date: Date,
        nominal: f64,
        accrual_start: Date,
        accrual_end: Date,
        index: OvernightIndex,
        spread: f64,
        day_counter: DayCounter,
    ) -> Self {
        Self {
            payment_date,
            nominal,
            accrual_start,
            accrual_end,
            index,
            spread,
            day_counter,
            cached_rate: None,
        }
    }

    /// Set a cached rate.
    pub fn with_rate(mut self, rate: f64) -> Self {
        self.cached_rate = Some(rate);
        self
    }

    /// The overnight index.
    pub fn index(&self) -> &OvernightIndex {
        &self.index
    }

    /// The spread.
    pub fn spread(&self) -> f64 {
        self.spread
    }

    /// Forecast the compounded overnight rate from a yield curve.
    ///
    /// Uses the simple approximation: `rate = (df_start/df_end - 1) / yf`.
    pub fn forecast_rate(&self, curve: &dyn YieldTermStructure) -> f64 {
        let df_start = curve.discount(self.accrual_start);
        let df_end = curve.discount(self.accrual_end);
        let yf = self.day_counter.year_fraction(self.accrual_start, self.accrual_end);
        if yf.abs() < 1e-15 {
            return self.spread;
        }
        let compounded_rate = (df_start / df_end - 1.0) / yf;
        compounded_rate + self.spread
    }

    /// Compute the compounded rate from historical fixings.
    ///
    /// Iterates over each business day in the accrual period, compounds
    /// the overnight fixings, and returns the annualized rate.
    pub fn rate_from_fixings(&self) -> Option<f64> {
        use ql_indexes::Index;

        let calendar = &self.index.fixing_calendar;
        let mut compounded = 1.0;
        let mut d = self.accrual_start;
        let mut count = 0;

        while d < self.accrual_end {
            if calendar.is_business_day(d) {
                if let Ok(fixing) = self.index.fixing(d, false) {
                    let next_biz = calendar.advance_business_days(d, 1);
                    let dt = self.day_counter.year_fraction(d, next_biz);
                    compounded *= 1.0 + fixing * dt;
                    count += 1;
                } else {
                    return None; // Missing fixing
                }
            }
            d += 1;
        }

        if count == 0 {
            return None;
        }

        let yf = self.day_counter
            .year_fraction(self.accrual_start, self.accrual_end);
        if yf.abs() < 1e-15 {
            return Some(self.spread);
        }
        Some((compounded - 1.0) / yf + self.spread)
    }

    /// Get the effective rate.
    fn effective_rate(&self) -> f64 {
        self.cached_rate.unwrap_or(self.spread)
    }

    /// Resolve the overnight compounded rate with mixed historical + projected.
    ///
    /// For each business day in the accrual period:
    /// - If the day is before `eval_date`, use the historical fixing from `IndexManager`.
    /// - If the day is on or after `eval_date`, project from `forecast_curve`
    ///   using `df(d)/df(d+1) - 1`.
    ///
    /// Falls back to pure forecast if no fixings are available.
    pub fn resolve_rate(
        &self,
        eval_date: Date,
        forecast_curve: &dyn YieldTermStructure,
    ) -> f64 {
        if let Some(r) = self.cached_rate {
            return r;
        }

        let calendar = &self.index.fixing_calendar;
        let mut compounded = 1.0;
        let mut d = self.accrual_start;
        let mut count = 0;

        while d < self.accrual_end {
            if calendar.is_business_day(d) {
                let next_biz = calendar.advance_business_days(d, 1);
                let dt = self.day_counter.year_fraction(d, next_biz);

                let daily_rate = if d < eval_date {
                    // Historical: look up IndexManager, fall back to curve
                    IndexManager::instance()
                        .get_fixing(self.index.name(), d)
                        .unwrap_or_else(|| {
                            let df_d = forecast_curve.discount(d);
                            let df_next = forecast_curve.discount(next_biz);
                            if dt.abs() < 1e-15 { 0.0 } else { (df_d / df_next - 1.0) / dt }
                        })
                } else {
                    // Future: project from curve
                    let df_d = forecast_curve.discount(d);
                    let df_next = forecast_curve.discount(next_biz);
                    if dt.abs() < 1e-15 { 0.0 } else { (df_d / df_next - 1.0) / dt }
                };

                compounded *= 1.0 + daily_rate * dt;
                count += 1;
            }
            d += 1;
        }

        if count == 0 {
            return self.spread;
        }

        let yf = self.day_counter
            .year_fraction(self.accrual_start, self.accrual_end);
        if yf.abs() < 1e-15 {
            return self.spread;
        }
        (compounded - 1.0) / yf + self.spread
    }
}

impl CashFlow for OvernightIndexedCoupon {
    fn date(&self) -> Date {
        self.payment_date
    }

    fn amount(&self) -> f64 {
        let rate = self.effective_rate();
        self.nominal * rate * self.accrual_period()
    }

    fn as_any(&self) -> &dyn std::any::Any { self }
}

impl Coupon for OvernightIndexedCoupon {
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

    fn make_coupon() -> OvernightIndexedCoupon {
        let index = OvernightIndex::sofr();
        OvernightIndexedCoupon::new(
            Date::from_ymd(2025, Month::April, 2),
            1_000_000.0,
            Date::from_ymd(2025, Month::January, 2),
            Date::from_ymd(2025, Month::April, 2),
            index,
            0.0, // no spread
            DayCounter::Actual360,
        )
    }

    #[test]
    fn overnight_coupon_forecast_from_flat_curve() {
        let c = make_coupon();
        let ref_date = Date::from_ymd(2025, Month::January, 2);
        let curve = FlatForward::new(ref_date, 0.04, DayCounter::Actual360);

        let rate = c.forecast_rate(&curve);
        // Should be close to 4% for a flat 4% curve
        assert_abs_diff_eq!(rate, 0.04, epsilon = 0.005);
    }

    #[test]
    fn overnight_coupon_with_cached_rate() {
        let c = make_coupon().with_rate(0.045);
        assert_abs_diff_eq!(c.rate(), 0.045, epsilon = 1e-15);
        let yf = DayCounter::Actual360.year_fraction(
            Date::from_ymd(2025, Month::January, 2),
            Date::from_ymd(2025, Month::April, 2),
        );
        let expected = 1_000_000.0 * 0.045 * yf;
        assert_abs_diff_eq!(c.amount(), expected, epsilon = 1e-4);
    }

    #[test]
    fn overnight_coupon_with_spread() {
        let index = OvernightIndex::sofr();
        let c = OvernightIndexedCoupon::new(
            Date::from_ymd(2025, Month::April, 2),
            1_000_000.0,
            Date::from_ymd(2025, Month::January, 2),
            Date::from_ymd(2025, Month::April, 2),
            index,
            0.005, // 50bp spread
            DayCounter::Actual360,
        );
        let ref_date = Date::from_ymd(2025, Month::January, 2);
        let curve = FlatForward::new(ref_date, 0.04, DayCounter::Actual360);

        let rate = c.forecast_rate(&curve);
        // ~4% + 50bp = ~4.5%
        assert!(rate > 0.04 && rate < 0.05, "rate = {rate}");
    }

    #[test]
    fn overnight_coupon_properties() {
        let c = make_coupon();
        assert_abs_diff_eq!(c.nominal(), 1_000_000.0, epsilon = 1e-10);
        assert_abs_diff_eq!(c.spread(), 0.0, epsilon = 1e-15);
        assert_eq!(c.accrual_start(), Date::from_ymd(2025, Month::January, 2));
        assert_eq!(c.accrual_end(), Date::from_ymd(2025, Month::April, 2));
    }

    #[test]
    fn overnight_resolve_rate_pure_forecast() {
        let c = make_coupon();
        let ref_date = Date::from_ymd(2024, Month::December, 1); // before accrual start
        let curve = FlatForward::new(ref_date, 0.04, DayCounter::Actual360);

        let rate = c.resolve_rate(ref_date, &curve);
        // All days are future → should be close to flat 4%
        assert_abs_diff_eq!(rate, 0.04, epsilon = 0.005);
    }

    #[test]
    fn overnight_resolve_rate_hybrid() {
        // Store some historical fixings for early days, then project the rest
        let idx = OvernightIndex::sofr();
        let start = Date::from_ymd(2025, Month::January, 2);
        // Add fixings for first 5 business days
        let mut d = start;
        let cal = &idx.fixing_calendar;
        for _ in 0..5 {
            while !cal.is_business_day(d) { d += 1; }
            IndexManager::instance().add_fixing(idx.name(), d, 0.04).unwrap();
            d += 1;
        }

        let c = make_coupon();
        // Eval date in the middle of accrual: some days historical, rest projected
        let eval = Date::from_ymd(2025, Month::January, 10);
        let curve = FlatForward::new(eval, 0.04, DayCounter::Actual360);

        let rate = c.resolve_rate(eval, &curve);
        // Should be close to 4% since both fixings and curve are at 4%
        assert_abs_diff_eq!(rate, 0.04, epsilon = 0.005);

        IndexManager::instance().clear_fixings(idx.name());
    }
}
