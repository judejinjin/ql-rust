//! Overnight rate indexes (SOFR, ESTR/€STR, SONIA).

use ql_core::errors::{QLError, QLResult};
use ql_currencies::Currency;
use ql_time::{
    Calendar, Date, DayCounter,
};

use crate::index::{Index, IndexManager};

/// An overnight rate index (e.g. SOFR, ESTR, SONIA).
///
/// Overnight indexes have a tenor of 1 day and typically zero fixing days
/// (fixing and value date coincide, or 1 day for some).
#[derive(Debug, Clone)]
pub struct OvernightIndex {
    /// Human-readable name.
    name: String,
    /// Number of fixing days (usually 0 for overnight indexes).
    pub fixing_days: u32,
    /// Currency.
    pub currency: Currency,
    /// Calendar for fixing date determination.
    pub fixing_calendar: Calendar,
    /// Day counter.
    pub day_counter: DayCounter,
}

impl OvernightIndex {
    /// Create a new overnight index.
    pub fn new(
        name: &str,
        fixing_days: u32,
        currency: Currency,
        fixing_calendar: Calendar,
        day_counter: DayCounter,
    ) -> Self {
        Self {
            name: name.to_string(),
            fixing_days,
            currency,
            fixing_calendar,
            day_counter,
        }
    }

    /// The value date from a fixing date.
    pub fn value_date(&self, fixing_date: Date) -> Date {
        self.fixing_calendar
            .advance_business_days(fixing_date, self.fixing_days as i32)
    }

    /// The maturity date (next business day after value date for overnight).
    pub fn maturity_date(&self, value_date: Date) -> Date {
        self.fixing_calendar
            .advance_business_days(value_date, 1)
    }
}

impl Index for OvernightIndex {
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
// Concrete Overnight Indexes
// ---------------------------------------------------------------------------

impl OvernightIndex {
    /// SOFR (Secured Overnight Financing Rate).
    pub fn sofr() -> Self {
        Self::new(
            "SOFR",
            0,
            Currency::usd(),
            Calendar::UnitedStates(ql_time::calendar::USMarket::Settlement),
            DayCounter::Actual360,
        )
    }

    /// ESTR (€STR — Euro Short-Term Rate).
    pub fn estr() -> Self {
        Self::new(
            "ESTR",
            0,
            Currency::eur(),
            Calendar::Target,
            DayCounter::Actual360,
        )
    }

    /// SONIA (Sterling Overnight Index Average).
    pub fn sonia() -> Self {
        Self::new(
            "SONIA",
            0,
            Currency::gbp(),
            Calendar::WeekendsOnly,
            DayCounter::Actual365Fixed,
        )
    }

    /// TONA (Tokyo Overnight Average Rate).
    pub fn tona() -> Self {
        Self::new(
            "TONA",
            0,
            Currency::jpy(),
            Calendar::WeekendsOnly, // simplified; full Japan calendar added separately
            DayCounter::Actual365Fixed,
        )
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
    fn sofr_properties() {
        let idx = OvernightIndex::sofr();
        assert_eq!(idx.name(), "SOFR");
        assert_eq!(idx.fixing_days, 0);
        assert_eq!(idx.currency.code, "USD");
    }

    #[test]
    fn estr_properties() {
        let idx = OvernightIndex::estr();
        assert_eq!(idx.name(), "ESTR");
        assert_eq!(idx.currency.code, "EUR");
    }

    #[test]
    fn sonia_properties() {
        let idx = OvernightIndex::sonia();
        assert_eq!(idx.name(), "SONIA");
        assert_eq!(idx.currency.code, "GBP");
    }

    #[test]
    fn overnight_value_and_maturity() {
        let idx = OvernightIndex::sofr();
        // 2025-01-15 Wednesday
        let fixing = Date::from_ymd(2025, Month::January, 15);
        let value = idx.value_date(fixing);
        let maturity = idx.maturity_date(value);
        // SOFR has 0 fixing days, so value = fixing
        assert_eq!(value, fixing);
        // Maturity = next business day
        assert!(maturity > value);
    }

    #[test]
    fn overnight_fixing_from_store() {
        let idx = OvernightIndex::sofr();
        let date = Date::from_ymd(2025, Month::January, 15);
        IndexManager::instance()
            .add_fixing("SOFR", date, 0.043)
            .unwrap();
        let f = idx.fixing(date, false).unwrap();
        assert!((f - 0.043).abs() < 1e-15);
    }

    #[test]
    fn tona_properties() {
        let idx = OvernightIndex::tona();
        assert_eq!(idx.name(), "TONA");
        assert_eq!(idx.currency.code, "JPY");
        assert_eq!(idx.fixing_days, 0);
    }

    #[test]
    fn overnight_missing_fixing() {
        let idx = OvernightIndex::estr();
        let date = Date::from_ymd(2099, Month::December, 30);
        assert!(idx.fixing(date, false).is_err());
    }
}
