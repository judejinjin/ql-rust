//! Base term structure trait.
//!
//! All term structures (yield, volatility, default probability, inflation)
//! share a common base interface for reference date, day counter, and
//! time-from-reference conversion.

use ql_time::{Calendar, Date, DayCounter};

/// The base interface for all term structures.
///
/// A term structure maps dates to some financial quantity (discount factors,
/// volatilities, default probabilities, etc.). The base trait provides
/// reference date, day counter, and calendar.
pub trait TermStructure: Send + Sync {
    /// The reference date of this term structure.
    fn reference_date(&self) -> Date;

    /// The day counter used for time computation.
    fn day_counter(&self) -> DayCounter;

    /// The calendar associated with this term structure.
    fn calendar(&self) -> &Calendar;

    /// The maximum date for which the term structure is defined.
    fn max_date(&self) -> Date;

    /// Number of settlement days.
    fn settlement_days(&self) -> u32 {
        0
    }

    /// Convert a date to time (year fraction from the reference date).
    fn time_from_reference(&self, date: Date) -> f64 {
        self.day_counter()
            .year_fraction(self.reference_date(), date)
    }
}
