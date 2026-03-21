//! Inflation swap rate helpers for bootstrapping inflation curves.
//!
//! Provides `RateHelper`-compatible helpers for:
//! - Zero-coupon inflation swaps (ZeroCouponInflationSwapHelper)
//! - Year-on-year inflation swaps (YoYInflationSwapHelper)
//!
//! These helpers plug into `PiecewiseZeroInflationCurve` and
//! `PiecewiseYoYInflationCurve` for inflation curve bootstrapping.
//!
//! ## QuantLib Parity
//!
//! Corresponds to:
//! - `ZeroCouponInflationSwapHelper` (ql/termstructures/inflation/inflationhelpers.hpp)
//! - `YearOnYearInflationSwapHelper`

use ql_time::{Calendar, Date, DayCounter};

use crate::term_structure::TermStructure;
use crate::inflation_term_structure::YoYInflationTermStructure;

// ===========================================================================
// ZeroCouponInflationSwapHelper
// ===========================================================================

/// Helper for bootstrapping a zero-coupon inflation curve from ZC inflation
/// swap quotes.
///
/// A zero-coupon inflation swap pays `N × [CPI(T)/CPI(0) - 1]` on the
/// inflation leg and `N × [(1+K)^T - 1]` on the fixed leg. The quoted rate K
/// is the break-even inflation rate for maturity T.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ZeroCouponInflationSwapHelper {
    /// Quoted fixed rate (break-even zero-coupon inflation rate).
    pub rate: f64,
    /// Maturity date.
    pub maturity: Date,
    /// Observation lag in months (typically 2 or 3 for CPI).
    pub observation_lag_months: u32,
    /// Day counter.
    pub day_counter: DayCounter,
    /// Calendar for business day adjustment.
    pub calendar: Calendar,
}

impl ZeroCouponInflationSwapHelper {
    /// Create a new ZC inflation swap helper.
    pub fn new(
        rate: f64,
        maturity: Date,
        observation_lag_months: u32,
        day_counter: DayCounter,
        calendar: Calendar,
    ) -> Self {
        Self {
            rate,
            maturity,
            observation_lag_months,
            day_counter,
            calendar,
        }
    }
}

// ===========================================================================
// YoYInflationSwapHelper
// ===========================================================================

/// Helper for bootstrapping a YoY inflation curve from year-on-year inflation
/// swap quotes.
///
/// A YoY inflation swap pays `N × [CPI(tᵢ)/CPI(tᵢ₋₁) - 1]` each period
/// on the inflation leg, and `N × K` on the fixed leg. The quoted rate K
/// is the break-even YoY inflation rate.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct YoYInflationSwapHelper {
    /// Quoted fixed rate (break-even YoY inflation rate).
    pub rate: f64,
    /// Maturity date.
    pub maturity: Date,
    /// Number of periods (typically annual).
    pub num_periods: usize,
    /// Observation lag in months.
    pub observation_lag_months: u32,
    /// Day counter.
    pub day_counter: DayCounter,
    /// Calendar.
    pub calendar: Calendar,
}

impl YoYInflationSwapHelper {
    /// Create a new YoY inflation swap helper.
    pub fn new(
        rate: f64,
        maturity: Date,
        num_periods: usize,
        observation_lag_months: u32,
        day_counter: DayCounter,
        calendar: Calendar,
    ) -> Self {
        Self {
            rate,
            maturity,
            num_periods,
            observation_lag_months,
            day_counter,
            calendar,
        }
    }
}

// ===========================================================================
// PiecewiseYoYInflationCurve
// ===========================================================================

/// Piecewise year-on-year inflation curve bootstrapped from YoY swap quotes.
///
/// Uses linear interpolation on YoY rates between pillar tenors.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PiecewiseYoYInflationCurve {
    reference_date: Date,
    day_counter: DayCounter,
    times: Vec<f64>,
    rates: Vec<f64>,
}

impl PiecewiseYoYInflationCurve {
    /// Bootstrap from YoY inflation swap helpers and a discount curve.
    ///
    /// `helpers`: YoY inflation swap helpers sorted by maturity.
    /// `discount_curve`: yield curve for discounting cash flows.
    pub fn bootstrap(
        reference_date: Date,
        day_counter: DayCounter,
        helpers: &[YoYInflationSwapHelper],
        discount_curve: &dyn crate::yield_term_structure::YieldTermStructure,
    ) -> Self {
        // Simple bootstrap: for a YoY swap with N annual periods at rate K,
        // the NPV is: Σᵢ₌₁ᴺ [YoY(tᵢ) - K] × df(tᵢ) = 0
        // This means: Σ YoY(tᵢ) × df(tᵢ) = K × Σ df(tᵢ)
        //
        // For the first helper (1Y), YoY(1) = K₁ directly.
        // For subsequent helpers, solve iteratively.

        let mut times = Vec::new();
        let mut rates = Vec::new();

        for helper in helpers {
            let t = day_counter.year_fraction(reference_date, helper.maturity);
            if t <= 0.0 {
                continue;
            }

            // Compute the YoY rate for this tenor
            // For simplicity, use the quoted rate as the marginal YoY rate at this tenor
            // This is exact for the first pillar and approximate for later ones
            // (a full Newton solver would be used in production)
            let yoy_rate = if rates.is_empty() {
                helper.rate
            } else {
                // The quoted rate is an average over all periods
                // Solve for the marginal rate at this tenor:
                // sum_previous_pv + new_rate × df(T) = K × sum_all_dfs
                let n = helper.num_periods;
                let dt = t / n as f64;
                let mut sum_prev_pv = 0.0;
                let mut sum_all_dfs = 0.0;

                for period in 1..=n {
                    let ti = dt * period as f64;
                    let df = discount_curve.discount_t(ti);
                    sum_all_dfs += df;

                    if period < n {
                        let interp_rate = interpolate_linear(&times, &rates, ti);
                        sum_prev_pv += interp_rate * df;
                    }
                }

                let df_last = discount_curve.discount_t(t);
                if df_last.abs() > 1e-15 {
                    (helper.rate * sum_all_dfs - sum_prev_pv) / df_last
                } else {
                    helper.rate
                }
            };

            times.push(t);
            rates.push(yoy_rate);
        }

        Self {
            reference_date,
            day_counter,
            times,
            rates,
        }
    }

    /// Get the YoY rate at time `t` via linear interpolation.
    pub fn yoy_rate_at(&self, t: f64) -> f64 {
        interpolate_linear(&self.times, &self.rates, t)
    }
}

/// Linear interpolation helper.
fn interpolate_linear(times: &[f64], values: &[f64], t: f64) -> f64 {
    if times.is_empty() {
        return 0.0;
    }
    if t <= times[0] {
        return values[0];
    }
    let n = times.len();
    if t >= times[n - 1] {
        return values[n - 1];
    }
    let idx = times.partition_point(|&ti| ti < t).min(n - 1).max(1);
    let t0 = times[idx - 1];
    let t1 = times[idx];
    let v0 = values[idx - 1];
    let v1 = values[idx];
    v0 + (v1 - v0) * (t - t0) / (t1 - t0)
}

impl TermStructure for PiecewiseYoYInflationCurve {
    fn reference_date(&self) -> Date {
        self.reference_date
    }
    fn day_counter(&self) -> DayCounter {
        self.day_counter
    }
    fn calendar(&self) -> &Calendar {
        &Calendar::NullCalendar
    }
    fn max_date(&self) -> Date {
        Date::from_serial(73050)
    }
}

impl YoYInflationTermStructure for PiecewiseYoYInflationCurve {
    fn yoy_rate(&self, t: f64) -> f64 {
        self.yoy_rate_at(t)
    }
}

// ===========================================================================
// Bootstrapping a ZC inflation curve from ZC swap helpers
// ===========================================================================

/// Bootstrap a zero-coupon inflation curve from ZC inflation swap helpers.
///
/// The ZC swap rate K at maturity T satisfies:
///   (1 + K)^T = CPI(T) / CPI(0) = (1 + i(T))^T
///
/// So the zero-coupon inflation rate is simply i(T) = K for each pillar.
pub fn bootstrap_zc_inflation_curve(
    reference_date: Date,
    day_counter: DayCounter,
    helpers: &[ZeroCouponInflationSwapHelper],
) -> crate::inflation_term_structure::PiecewiseZeroInflationCurve {
    let tenors: Vec<f64> = helpers
        .iter()
        .map(|h| day_counter.year_fraction(reference_date, h.maturity))
        .collect();
    let rates: Vec<f64> = helpers.iter().map(|h| h.rate).collect();

    crate::inflation_term_structure::PiecewiseZeroInflationCurve::new(
        reference_date,
        day_counter,
        tenors,
        rates,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_time::Month;
    use crate::inflation_term_structure::ZeroInflationTermStructure;

    #[test]
    fn zc_inflation_helper_creation() {
        let helper = ZeroCouponInflationSwapHelper::new(
            0.025,
            Date::from_ymd(2030, Month::January, 15),
            3,
            DayCounter::Actual365Fixed,
            Calendar::Target,
        );
        assert!((helper.rate - 0.025).abs() < 1e-10);
    }

    #[test]
    fn zc_inflation_curve_bootstrap() {
        let ref_date = Date::from_ymd(2025, Month::January, 15);
        let dc = DayCounter::Actual365Fixed;

        let helpers = vec![
            ZeroCouponInflationSwapHelper::new(0.020, Date::from_ymd(2026, Month::January, 15), 3, dc, Calendar::Target),
            ZeroCouponInflationSwapHelper::new(0.022, Date::from_ymd(2027, Month::January, 15), 3, dc, Calendar::Target),
            ZeroCouponInflationSwapHelper::new(0.025, Date::from_ymd(2030, Month::January, 15), 3, dc, Calendar::Target),
            ZeroCouponInflationSwapHelper::new(0.023, Date::from_ymd(2035, Month::January, 15), 3, dc, Calendar::Target),
        ];

        let curve = bootstrap_zc_inflation_curve(ref_date, dc, &helpers);

        // The 1Y pillar should give the 1Y swap rate
        let rate_1y = curve.zero_rate(1.0);
        assert_abs_diff_eq!(rate_1y, 0.020, epsilon = 0.001);

        // 5Y pillar should give the 5Y swap rate
        let rate_5y = curve.zero_rate(5.0);
        assert_abs_diff_eq!(rate_5y, 0.025, epsilon = 0.001);
    }

    #[test]
    fn yoy_inflation_helper_creation() {
        let helper = YoYInflationSwapHelper::new(
            0.023,
            Date::from_ymd(2030, Month::January, 15),
            5,
            3,
            DayCounter::Actual365Fixed,
            Calendar::Target,
        );
        assert_eq!(helper.num_periods, 5);
        assert!((helper.rate - 0.023).abs() < 1e-10);
    }

    #[test]
    fn yoy_inflation_curve_bootstrap() {
        let ref_date = Date::from_ymd(2025, Month::January, 15);
        let dc = DayCounter::Actual365Fixed;

        let flat_forward = crate::yield_curves::FlatForward::new(ref_date, 0.03, dc);

        let helpers = vec![
            YoYInflationSwapHelper::new(0.020, Date::from_ymd(2026, Month::January, 15), 1, 3, dc, Calendar::Target),
            YoYInflationSwapHelper::new(0.022, Date::from_ymd(2027, Month::January, 15), 2, 3, dc, Calendar::Target),
            YoYInflationSwapHelper::new(0.025, Date::from_ymd(2030, Month::January, 15), 5, 3, dc, Calendar::Target),
        ];

        let curve = PiecewiseYoYInflationCurve::bootstrap(ref_date, dc, &helpers, &flat_forward);

        // First pillar should match directly
        let rate_1y = curve.yoy_rate(1.0);
        assert_abs_diff_eq!(rate_1y, 0.020, epsilon = 0.001);
    }

    #[test]
    fn yoy_curve_interpolation() {
        let ref_date = Date::from_ymd(2025, Month::January, 15);
        let dc = DayCounter::Actual365Fixed;

        let flat_forward = crate::yield_curves::FlatForward::new(ref_date, 0.03, dc);

        let helpers = vec![
            YoYInflationSwapHelper::new(0.020, Date::from_ymd(2026, Month::January, 15), 1, 3, dc, Calendar::Target),
            YoYInflationSwapHelper::new(0.030, Date::from_ymd(2030, Month::January, 15), 5, 3, dc, Calendar::Target),
        ];

        let curve = PiecewiseYoYInflationCurve::bootstrap(ref_date, dc, &helpers, &flat_forward);

        // Interpolated at 3Y: the 5Y marginal rate is higher than 3% to compensate
        // for the 1Y rate at 2%, so the interpolated 3Y rate is between 2% and the 5Y marginal.
        let rate_3y = curve.yoy_rate(3.0);
        assert!(rate_3y > 0.02 && rate_3y < 0.08,
            "3Y YoY rate {} should be between 2% and 8%", rate_3y);
    }

    #[test]
    fn yoy_curve_as_term_structure() {
        let ref_date = Date::from_ymd(2025, Month::January, 15);
        let dc = DayCounter::Actual365Fixed;
        let flat_forward = crate::yield_curves::FlatForward::new(ref_date, 0.03, dc);

        let helpers = vec![
            YoYInflationSwapHelper::new(0.020, Date::from_ymd(2026, Month::January, 15), 1, 3, dc, Calendar::Target),
        ];

        let curve = PiecewiseYoYInflationCurve::bootstrap(ref_date, dc, &helpers, &flat_forward);
        assert_eq!(curve.reference_date(), ref_date);
        assert_eq!(curve.day_counter(), dc);
    }
}
