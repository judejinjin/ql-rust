//! Swap rate index — an index that returns par swap rates at a given tenor.
//!
//! A `SwapIndex` defines a family of vanilla interest rate swaps (e.g. "EUR 10Y
//! swap rate"). It provides the fixed/floating conventions, calendars, and day
//! counters needed to build the underlying swap for any fixing date.

use ql_core::errors::{QLError, QLResult};
use ql_currencies::Currency;
use ql_time::{
    BusinessDayConvention, Calendar, DayCounter, Date, Frequency, Period, TimeUnit,
};

use crate::ibor_index::IborIndex;
use crate::index::{Index, IndexManager};

/// A swap rate index: defines the conventions of a swap family (tenor,
/// fixed/float leg day counts, calendar, IBOR index).
///
/// Historical fixings are stored in the global `IndexManager`.
#[derive(Debug, Clone)]
pub struct SwapIndex {
    /// Human-readable name (e.g. "EurSwap10Y").
    name: String,
    /// Swap tenor.
    pub tenor: Period,
    /// Fixing days (typically 2 for EUR, 2 for USD).
    pub fixing_days: u32,
    /// Calendar for fixing / settlement.
    pub fixing_calendar: Calendar,
    /// Currency.
    pub currency: Currency,
    /// Fixed-leg payment frequency.
    pub fixed_leg_frequency: Frequency,
    /// Fixed-leg business day convention.
    pub fixed_leg_convention: BusinessDayConvention,
    /// Fixed-leg day counter.
    pub fixed_leg_day_counter: DayCounter,
    /// Floating leg IBOR index.
    pub ibor_index: IborIndex,
}

impl SwapIndex {
    /// Create a new swap index with full specification.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: &str,
        tenor: Period,
        fixing_days: u32,
        currency: Currency,
        fixing_calendar: Calendar,
        fixed_leg_frequency: Frequency,
        fixed_leg_convention: BusinessDayConvention,
        fixed_leg_day_counter: DayCounter,
        ibor_index: IborIndex,
    ) -> Self {
        Self {
            name: name.to_string(),
            tenor,
            fixing_days,
            currency,
            fixing_calendar,
            fixed_leg_frequency,
            fixed_leg_convention,
            fixed_leg_day_counter,
            ibor_index,
        }
    }

    /// Value (settlement) date from a fixing date.
    pub fn value_date(&self, fixing_date: Date) -> Date {
        self.fixing_calendar
            .advance_business_days(fixing_date, self.fixing_days as i32)
    }

    /// Maturity date of the underlying swap.
    pub fn maturity_date(&self, value_date: Date) -> Date {
        self.fixing_calendar.advance(
            value_date,
            self.tenor,
            self.fixed_leg_convention,
            false,
        )
    }
}

impl Index for SwapIndex {
    fn name(&self) -> &str {
        &self.name
    }

    fn is_valid_fixing_date(&self, date: Date) -> bool {
        self.fixing_calendar.is_business_day(date)
    }

    fn fixing(&self, date: Date, _forecast_today_fixing: bool) -> QLResult<f64> {
        if let Some(value) = IndexManager::instance().get_fixing(self.name(), date) {
            return Ok(value);
        }
        Err(QLError::InvalidArgument(format!(
            "missing fixing for {} on {}",
            self.name(),
            date
        )))
    }
}

// ---------------------------------------------------------------------------
// Concrete Swap Indexes
// ---------------------------------------------------------------------------

impl SwapIndex {
    /// EUR swap index vs Euribor 6M, annual fixed 30/360.
    pub fn eur_swap(tenor: Period) -> Self {
        Self::new(
            &format!("EurSwap{}", tenor),
            tenor,
            2,
            Currency::eur(),
            Calendar::Target,
            Frequency::Annual,
            BusinessDayConvention::ModifiedFollowing,
            DayCounter::Thirty360(ql_time::day_counter::Thirty360Convention::EurobondBasis),
            IborIndex::euribor_6m(),
        )
    }

    /// EUR 10Y swap index (convenience).
    pub fn eur_swap_10y() -> Self {
        Self::eur_swap(Period { length: 10, unit: TimeUnit::Years })
    }

    /// EUR 5Y swap index (convenience).
    pub fn eur_swap_5y() -> Self {
        Self::eur_swap(Period { length: 5, unit: TimeUnit::Years })
    }

    /// USD swap index vs USD LIBOR 3M, semi-annual fixed 30/360.
    pub fn usd_swap(tenor: Period) -> Self {
        Self::new(
            &format!("USDSwap{}", tenor),
            tenor,
            2,
            Currency::usd(),
            Calendar::UnitedStates(ql_time::calendar::USMarket::Settlement),
            Frequency::Semiannual,
            BusinessDayConvention::ModifiedFollowing,
            DayCounter::Thirty360(ql_time::day_counter::Thirty360Convention::BondBasis),
            IborIndex::usd_libor_3m(),
        )
    }

    /// USD 10Y swap index (convenience).
    pub fn usd_swap_10y() -> Self {
        Self::usd_swap(Period { length: 10, unit: TimeUnit::Years })
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ql_time::Month;

    #[test]
    fn eur_swap_10y_properties() {
        let idx = SwapIndex::eur_swap_10y();
        assert_eq!(idx.name(), "EurSwap10Y");
        assert_eq!(idx.tenor.length, 10);
        assert_eq!(idx.fixing_days, 2);
        assert_eq!(idx.currency.code, "EUR");
        assert_eq!(idx.fixed_leg_frequency, Frequency::Annual);
    }

    #[test]
    fn usd_swap_10y_properties() {
        let idx = SwapIndex::usd_swap_10y();
        assert!(idx.name().contains("USDSwap"));
        assert_eq!(idx.tenor.length, 10);
        assert_eq!(idx.currency.code, "USD");
        assert_eq!(idx.fixed_leg_frequency, Frequency::Semiannual);
    }

    #[test]
    fn swap_index_value_and_maturity() {
        let idx = SwapIndex::eur_swap_10y();
        let fixing = Date::from_ymd(2025, Month::January, 15);
        let value = idx.value_date(fixing);
        let maturity = idx.maturity_date(value);
        // Value date 2 business days after fixing
        assert!(value > fixing);
        // Maturity 10 years after value
        assert!(maturity > value);
        assert_eq!(maturity.year() - value.year(), 10);
    }

    #[test]
    fn swap_index_fixing_from_store() {
        let idx = SwapIndex::eur_swap_10y();
        let date = Date::from_ymd(2025, Month::June, 16);
        IndexManager::instance()
            .add_fixing(idx.name(), date, 0.025)
            .unwrap();
        let f = idx.fixing(date, false).unwrap();
        assert!((f - 0.025).abs() < 1e-15);
    }

    #[test]
    fn swap_index_missing_fixing() {
        let idx = SwapIndex::eur_swap_5y();
        let date = Date::from_ymd(2099, Month::December, 30);
        assert!(idx.fixing(date, false).is_err());
    }
}
