//! Concrete yield curve implementations.
//!
//! - `FlatForward` — constant forward rate for testing.
//! - `DiscountCurve` — interpolated discount factors.
//! - `ZeroCurve` — interpolated zero rates.

use ql_core::errors::{QLError, QLResult};
use ql_indexes::interest_rate::{Compounding, InterestRate};
use ql_math::interpolation::{LinearInterpolation, LogLinearInterpolation, Interpolation};
use ql_time::{Calendar, Date, DayCounter};

use crate::term_structure::TermStructure;
use crate::yield_term_structure::YieldTermStructure;

// ===========================================================================
// FlatForward
// ===========================================================================

/// A flat forward rate — constant over all maturities.
///
/// Useful as a testing workhorse and for simple discounting.
#[derive(Debug, Clone)]
pub struct FlatForward {
    reference_date: Date,
    rate: InterestRate,
    calendar: Calendar,
    max_date: Date,
}

impl FlatForward {
    /// Create a flat forward curve with continuous compounding.
    pub fn new(reference_date: Date, rate: f64, day_counter: DayCounter) -> Self {
        Self {
            reference_date,
            rate: InterestRate::new(rate, day_counter, Compounding::Continuous, 1),
            calendar: Calendar::NullCalendar,
            max_date: Date::max_date(),
        }
    }

    /// Create with a specific compounding convention.
    pub fn with_compounding(
        reference_date: Date,
        rate: f64,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: u32,
    ) -> Self {
        Self {
            reference_date,
            rate: InterestRate::new(rate, day_counter, compounding, frequency),
            calendar: Calendar::NullCalendar,
            max_date: Date::max_date(),
        }
    }

    /// The flat rate.
    pub fn rate(&self) -> f64 {
        self.rate.rate
    }
}

impl TermStructure for FlatForward {
    fn reference_date(&self) -> Date {
        self.reference_date
    }

    fn day_counter(&self) -> DayCounter {
        self.rate.day_counter
    }

    fn calendar(&self) -> Calendar {
        self.calendar
    }

    fn max_date(&self) -> Date {
        self.max_date
    }
}

impl YieldTermStructure for FlatForward {
    fn discount_impl(&self, t: f64) -> f64 {
        self.rate.discount_factor(t).unwrap_or(1.0)
    }
}

// ===========================================================================
// DiscountCurve — interpolated discount factors
// ===========================================================================

/// A yield curve defined by interpolated discount factors.
///
/// Uses log-linear interpolation on discount factors (linear in log-space),
/// which is the most common convention for discount curves.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct DiscountCurve {
    reference_date: Date,
    day_counter: DayCounter,
    calendar: Calendar,
    /// Times (year fractions from reference date).
    times: Vec<f64>,
    /// Discount factors at each time node.
    dfs: Vec<f64>,
    /// Log-linear interpolation on discount factors.
    interp: LogLinearInterpolation,
    max_date: Date,
}

impl DiscountCurve {
    /// Build a discount curve from dates and discount factors.
    ///
    /// The first date should be the reference date with df = 1.0.
    pub fn new(
        dates: Vec<Date>,
        discount_factors: Vec<f64>,
        day_counter: DayCounter,
    ) -> QLResult<Self> {
        if dates.len() != discount_factors.len() {
            return Err(QLError::InvalidArgument(
                "dates and discount_factors must have the same length".into(),
            ));
        }
        if dates.len() < 2 {
            return Err(QLError::InvalidArgument(
                "need at least 2 data points".into(),
            ));
        }

        let reference_date = dates[0];
        let times: Vec<f64> = dates
            .iter()
            .map(|&d| day_counter.year_fraction(reference_date, d))
            .collect();

        let interp = LogLinearInterpolation::new(times.clone(), discount_factors.clone())?;
        let max_date = *dates.last().unwrap();

        Ok(Self {
            reference_date,
            day_counter,
            calendar: Calendar::NullCalendar,
            times,
            dfs: discount_factors,
            interp,
            max_date,
        })
    }
}

impl TermStructure for DiscountCurve {
    fn reference_date(&self) -> Date {
        self.reference_date
    }

    fn day_counter(&self) -> DayCounter {
        self.day_counter
    }

    fn calendar(&self) -> Calendar {
        self.calendar
    }

    fn max_date(&self) -> Date {
        self.max_date
    }
}

impl YieldTermStructure for DiscountCurve {
    fn discount_impl(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }
        self.interp.value(t).unwrap_or(1.0)
    }
}

// ===========================================================================
// ZeroCurve — interpolated zero rates
// ===========================================================================

/// A yield curve defined by interpolated zero rates (continuous compounding).
///
/// Uses linear interpolation on zero rates.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ZeroCurve {
    reference_date: Date,
    day_counter: DayCounter,
    calendar: Calendar,
    /// Times (year fractions from reference date).
    times: Vec<f64>,
    /// Zero rates (continuously compounded) at each time node.
    rates: Vec<f64>,
    /// Linear interpolation on zero rates.
    interp: LinearInterpolation,
    max_date: Date,
}

impl ZeroCurve {
    /// Build a zero curve from dates and continuously-compounded zero rates.
    pub fn new(
        dates: Vec<Date>,
        rates: Vec<f64>,
        day_counter: DayCounter,
    ) -> QLResult<Self> {
        if dates.len() != rates.len() {
            return Err(QLError::InvalidArgument(
                "dates and rates must have the same length".into(),
            ));
        }
        if dates.len() < 2 {
            return Err(QLError::InvalidArgument(
                "need at least 2 data points".into(),
            ));
        }

        let reference_date = dates[0];
        let times: Vec<f64> = dates
            .iter()
            .map(|&d| day_counter.year_fraction(reference_date, d))
            .collect();

        let interp = LinearInterpolation::new(times.clone(), rates.clone())?;
        let max_date = *dates.last().unwrap();

        Ok(Self {
            reference_date,
            day_counter,
            calendar: Calendar::NullCalendar,
            times,
            rates,
            interp,
            max_date,
        })
    }
}

impl TermStructure for ZeroCurve {
    fn reference_date(&self) -> Date {
        self.reference_date
    }

    fn day_counter(&self) -> DayCounter {
        self.day_counter
    }

    fn calendar(&self) -> Calendar {
        self.calendar
    }

    fn max_date(&self) -> Date {
        self.max_date
    }
}

impl YieldTermStructure for ZeroCurve {
    fn discount_impl(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }
        let rate = self.interp.value(t).unwrap_or(0.0);
        (-rate * t).exp()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_time::Month;

    #[test]
    fn flat_forward_discount() {
        let ref_date = Date::from_ymd(2025, Month::January, 1);
        let curve = FlatForward::new(ref_date, 0.05, DayCounter::Actual365Fixed);
        let d1y = Date::from_ymd(2026, Month::January, 1);
        let t = curve.time_from_reference(d1y);
        let df = curve.discount(d1y);
        assert_abs_diff_eq!(df, (-0.05 * t).exp(), epsilon = 1e-12);
    }

    #[test]
    fn flat_forward_zero_rate() {
        let ref_date = Date::from_ymd(2025, Month::January, 1);
        let curve = FlatForward::new(ref_date, 0.05, DayCounter::Actual365Fixed);
        let d1y = Date::from_ymd(2026, Month::January, 1);
        let zr = curve
            .zero_rate(d1y, DayCounter::Actual365Fixed, Compounding::Continuous, 1)
            .unwrap();
        assert_abs_diff_eq!(zr.rate, 0.05, epsilon = 1e-10);
    }

    #[test]
    fn flat_forward_forward_rate() {
        let ref_date = Date::from_ymd(2025, Month::January, 1);
        let curve = FlatForward::new(ref_date, 0.05, DayCounter::Actual365Fixed);
        let d1 = Date::from_ymd(2026, Month::January, 1);
        let d2 = Date::from_ymd(2027, Month::January, 1);
        let fr = curve
            .forward_rate(d1, d2, DayCounter::Actual365Fixed, Compounding::Continuous, 1)
            .unwrap();
        // Flat forward: forward rate = spot rate
        assert_abs_diff_eq!(fr.rate, 0.05, epsilon = 1e-6);
    }

    #[test]
    fn flat_forward_with_simple_compounding() {
        let ref_date = Date::from_ymd(2025, Month::January, 1);
        let curve = FlatForward::with_compounding(
            ref_date,
            0.05,
            DayCounter::Actual365Fixed,
            Compounding::Simple,
            1,
        );
        let t = 1.0;
        let df = curve.discount_impl(t);
        assert_abs_diff_eq!(df, 1.0 / 1.05, epsilon = 1e-12);
    }

    #[test]
    fn discount_curve_at_nodes() {
        let ref_date = Date::from_ymd(2025, Month::January, 1);
        let dates = vec![
            ref_date,
            Date::from_ymd(2025, Month::July, 1),
            Date::from_ymd(2026, Month::January, 1),
            Date::from_ymd(2027, Month::January, 1),
        ];
        let dfs = vec![1.0, 0.975, 0.95, 0.90];
        let curve = DiscountCurve::new(dates.clone(), dfs.clone(), DayCounter::Actual365Fixed).unwrap();

        // At reference date, discount should be 1.0
        assert_abs_diff_eq!(curve.discount(ref_date), 1.0, epsilon = 1e-10);

        // At node dates, should match input discount factors
        for (i, &d) in dates.iter().enumerate().skip(1) {
            assert_abs_diff_eq!(curve.discount(d), dfs[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn discount_curve_interpolation() {
        let ref_date = Date::from_ymd(2025, Month::January, 1);
        let dates = vec![
            ref_date,
            Date::from_ymd(2026, Month::January, 1),
            Date::from_ymd(2027, Month::January, 1),
        ];
        let dfs = vec![1.0, 0.95, 0.90];
        let curve = DiscountCurve::new(dates, dfs, DayCounter::Actual365Fixed).unwrap();

        // Midpoint should be interpolated (log-linear)
        let mid = Date::from_ymd(2025, Month::July, 1);
        let df_mid = curve.discount(mid);
        assert!(df_mid > 0.90 && df_mid < 1.0, "df_mid = {df_mid}");
    }

    #[test]
    fn zero_curve_at_nodes() {
        let ref_date = Date::from_ymd(2025, Month::January, 1);
        let dates = vec![
            ref_date,
            Date::from_ymd(2026, Month::January, 1),
            Date::from_ymd(2027, Month::January, 1),
        ];
        let rates = vec![0.04, 0.05, 0.055];
        let curve = ZeroCurve::new(dates.clone(), rates.clone(), DayCounter::Actual365Fixed).unwrap();

        // At 1Y, zero rate should give correct discount
        let d1y = Date::from_ymd(2026, Month::January, 1);
        let t1 = curve.time_from_reference(d1y);
        let df1 = curve.discount(d1y);
        assert_abs_diff_eq!(df1, (-rates[1] * t1).exp(), epsilon = 1e-10);
    }

    #[test]
    fn zero_curve_interpolation() {
        let ref_date = Date::from_ymd(2025, Month::January, 1);
        let dates = vec![
            ref_date,
            Date::from_ymd(2026, Month::January, 1),
            Date::from_ymd(2027, Month::January, 1),
        ];
        let rates = vec![0.04, 0.05, 0.06];
        let curve = ZeroCurve::new(dates, rates, DayCounter::Actual365Fixed).unwrap();

        // Midpoint zero rate should be interpolated linearly
        let mid = Date::from_ymd(2025, Month::July, 1);
        let df = curve.discount(mid);
        assert!(df > 0.0 && df < 1.0, "df = {df}");
    }

    #[test]
    fn discount_curve_needs_min_points() {
        let ref_date = Date::from_ymd(2025, Month::January, 1);
        assert!(DiscountCurve::new(vec![ref_date], vec![1.0], DayCounter::Actual365Fixed).is_err());
    }

    #[test]
    fn flat_forward_instantaneous() {
        let ref_date = Date::from_ymd(2025, Month::January, 1);
        let curve = FlatForward::new(ref_date, 0.05, DayCounter::Actual365Fixed);
        let fwd = curve.forward_rate_t(2.0);
        assert_abs_diff_eq!(fwd, 0.05, epsilon = 1e-4);
    }
}
