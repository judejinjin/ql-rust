//! Yield term structure trait and interest rate helpers.
//!
//! Provides the `YieldTermStructure` trait with three interchangeable
//! representations: discount factors, zero rates, and forward rates.

use ql_core::errors::QLResult;
use ql_indexes::interest_rate::{Compounding, InterestRate};
use ql_time::{Date, DayCounter};

use crate::term_structure::TermStructure;

/// A yield term structure: maps dates to discount factors / zero rates / forward rates.
///
/// Implementors override `discount_impl(t)` and get default implementations
/// for `zero_rate` and `forward_rate`. Alternatively they can override
/// whichever representation is most natural and derive the others.
///
/// # Examples
///
/// ```
/// use ql_termstructures::{FlatForward, YieldTermStructure};
/// use ql_time::{Date, Month, DayCounter};
///
/// let today = Date::from_ymd(2025, Month::January, 15);
/// let curve = FlatForward::new(today, 0.03, DayCounter::Actual365Fixed);
///
/// // discount_t gives df at a given year fraction
/// let df_1y = curve.discount_t(1.0);
/// assert!((df_1y - (-0.03_f64).exp()).abs() < 1e-10);
/// ```
pub trait YieldTermStructure: TermStructure {
    /// Discount factor at time `t` (year fraction from reference date).
    fn discount_impl(&self, t: f64) -> f64;

    /// Discount factor at a given date.
    #[inline]
    fn discount(&self, date: Date) -> f64 {
        let t = self.time_from_reference(date);
        self.discount_impl(t)
    }

    /// Discount factor at a given time (years from reference).
    #[inline]
    fn discount_t(&self, t: f64) -> f64 {
        self.discount_impl(t)
    }

    /// Zero rate (continuously compounded) at a given date.
    fn zero_rate(
        &self,
        date: Date,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: u32,
    ) -> QLResult<InterestRate> {
        let t = day_counter.year_fraction(self.reference_date(), date);
        self.zero_rate_t(t, day_counter, compounding, frequency)
    }

    /// Zero rate (continuously compounded) at a given time.
    fn zero_rate_t(
        &self,
        t: f64,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: u32,
    ) -> QLResult<InterestRate> {
        let df = self.discount_impl(t);
        InterestRate::implied_rate(1.0 / df, day_counter, compounding, frequency, t)
    }

    /// Instantaneous forward rate at time `t` (continuous compounding).
    fn forward_rate_t(&self, t: f64) -> f64 {
        let dt = 1e-5;
        let t1 = (t - dt).max(0.0);
        let t2 = t + dt;
        let df1 = self.discount_impl(t1);
        let df2 = self.discount_impl(t2);
        if (t2 - t1).abs() < 1e-15 {
            return 0.0;
        }
        -(df2 / df1).ln() / (t2 - t1)
    }

    /// Forward rate between two dates (simple compounding).
    fn forward_rate(
        &self,
        d1: Date,
        d2: Date,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: u32,
    ) -> QLResult<InterestRate> {
        let t1 = self.time_from_reference(d1);
        let t2 = self.time_from_reference(d2);
        let df1 = self.discount_impl(t1);
        let df2 = self.discount_impl(t2);
        let compound = df1 / df2;
        let dt = t2 - t1;
        InterestRate::implied_rate(compound, day_counter, compounding, frequency, dt)
    }
}
