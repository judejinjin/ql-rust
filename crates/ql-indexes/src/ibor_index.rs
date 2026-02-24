//! IBOR (Interbank Offered Rate) indexes.
//!
//! Includes the `IborIndex` struct and concrete instances such as Euribor.

use ql_core::errors::{QLError, QLResult};
use ql_currencies::Currency;
use ql_time::{
    BusinessDayConvention, Calendar, Date, DayCounter, Period, TimeUnit,
};

use crate::index::{Index, IndexManager};


/// An interbank offered rate index (e.g. Euribor, TIBOR).
///
/// An `IborIndex` has a tenor, fixing days, currency, and an optional
/// forecast curve handle for projecting future fixings.
#[derive(Debug, Clone)]
pub struct IborIndex {
    /// Human-readable name.
    name: String,
    /// Tenor (e.g. 3M, 6M).
    pub tenor: Period,
    /// Number of business days between fixing and value date.
    pub fixing_days: u32,
    /// Currency of the index.
    pub currency: Currency,
    /// Calendar for fixing date determination.
    pub fixing_calendar: Calendar,
    /// Business day convention for value/maturity dates.
    pub convention: BusinessDayConvention,
    /// Whether the maturity date is adjusted for end-of-month.
    pub end_of_month: bool,
    /// Day counter for the rate computation.
    pub day_counter: DayCounter,
}

impl IborIndex {
    /// Create a new IBOR index.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: &str,
        tenor: Period,
        fixing_days: u32,
        currency: Currency,
        fixing_calendar: Calendar,
        convention: BusinessDayConvention,
        end_of_month: bool,
        day_counter: DayCounter,
    ) -> Self {
        Self {
            name: name.to_string(),
            tenor,
            fixing_days,
            currency,
            fixing_calendar,
            convention,
            end_of_month,
            day_counter,
        }
    }

    /// Compute the value date from a fixing date.
    pub fn value_date(&self, fixing_date: Date) -> Date {
        self.fixing_calendar
            .advance_business_days(fixing_date, self.fixing_days as i32)
    }

    /// Compute the maturity date from a value date.
    pub fn maturity_date(&self, value_date: Date) -> Date {
        self.fixing_calendar
            .advance(value_date, self.tenor, self.convention, self.end_of_month)
    }

    /// Compute the year fraction between value date and maturity date.
    pub fn year_fraction(&self, value_date: Date, maturity_date: Date) -> f64 {
        self.day_counter.year_fraction(value_date, maturity_date)
    }

    /// Build a forecast rate given a discount factor at value date and maturity.
    pub fn forecast_fixing(
        &self,
        value_date: Date,
        maturity_date: Date,
        df_start: f64,
        df_end: f64,
    ) -> f64 {
        let t = self.year_fraction(value_date, maturity_date);
        if t.abs() < 1e-15 {
            0.0
        } else {
            (df_start / df_end - 1.0) / t
        }
    }
}

impl Index for IborIndex {
    fn name(&self) -> &str {
        &self.name
    }

    fn is_valid_fixing_date(&self, date: Date) -> bool {
        self.fixing_calendar.is_business_day(date)
    }

    fn fixing(&self, date: Date, _forecast_today_fixing: bool) -> QLResult<f64> {
        // Look up historical fixing
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
// Concrete IBOR Indexes
// ---------------------------------------------------------------------------

impl IborIndex {
    /// Euribor 3-Month.
    pub fn euribor_3m() -> Self {
        Self::new(
            "Euribor3M",
            Period {
                length: 3,
                unit: TimeUnit::Months,
            },
            2,
            Currency::eur(),
            Calendar::Target,
            BusinessDayConvention::ModifiedFollowing,
            true,
            DayCounter::Actual360,
        )
    }

    /// Euribor 6-Month.
    pub fn euribor_6m() -> Self {
        Self::new(
            "Euribor6M",
            Period {
                length: 6,
                unit: TimeUnit::Months,
            },
            2,
            Currency::eur(),
            Calendar::Target,
            BusinessDayConvention::ModifiedFollowing,
            true,
            DayCounter::Actual360,
        )
    }

    /// USD LIBOR 3-Month (legacy).
    pub fn usd_libor_3m() -> Self {
        Self::new(
            "USDLibor3M",
            Period { length: 3, unit: TimeUnit::Months },
            2,
            Currency::usd(),
            Calendar::UnitedStates(ql_time::calendar::USMarket::Settlement),
            BusinessDayConvention::ModifiedFollowing,
            true,
            DayCounter::Actual360,
        )
    }

    // -----------------------------------------------------------------------
    // Euribor additional tenors
    // -----------------------------------------------------------------------

    /// Euribor 1-Month.
    pub fn euribor_1m() -> Self {
        Self::new("Euribor1M", Period { length: 1, unit: TimeUnit::Months }, 2,
            Currency::eur(), Calendar::Target,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    /// Euribor 1-Week.
    pub fn euribor_1w() -> Self {
        Self::new("Euribor1W", Period { length: 1, unit: TimeUnit::Weeks }, 2,
            Currency::eur(), Calendar::Target,
            BusinessDayConvention::Following, false, DayCounter::Actual360)
    }

    /// Euribor 12-Month.
    pub fn euribor_12m() -> Self {
        Self::new("Euribor12M", Period { length: 12, unit: TimeUnit::Months }, 2,
            Currency::eur(), Calendar::Target,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    // -----------------------------------------------------------------------
    // GBP LIBOR (legacy)
    // -----------------------------------------------------------------------

    /// GBP LIBOR 3-Month (legacy).
    pub fn gbp_libor_3m() -> Self {
        Self::new("GBPLibor3M", Period { length: 3, unit: TimeUnit::Months }, 0,
            Currency::gbp(), Calendar::UnitedKingdom,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual365Fixed)
    }

    /// GBP LIBOR 6-Month (legacy).
    pub fn gbp_libor_6m() -> Self {
        Self::new("GBPLibor6M", Period { length: 6, unit: TimeUnit::Months }, 0,
            Currency::gbp(), Calendar::UnitedKingdom,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual365Fixed)
    }

    // -----------------------------------------------------------------------
    // CHF LIBOR (legacy)
    // -----------------------------------------------------------------------

    /// CHF LIBOR 3-Month (legacy).
    pub fn chf_libor_3m() -> Self {
        Self::new("CHFLibor3M", Period { length: 3, unit: TimeUnit::Months }, 2,
            Currency::chf(), Calendar::Switzerland,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    // -----------------------------------------------------------------------
    // JPY LIBOR (legacy)
    // -----------------------------------------------------------------------

    /// JPY LIBOR 3-Month (legacy).
    pub fn jpy_libor_3m() -> Self {
        Self::new("JPYLibor3M", Period { length: 3, unit: TimeUnit::Months }, 2,
            Currency::jpy(), Calendar::Japan,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    /// JPY LIBOR 6-Month (legacy).
    pub fn jpy_libor_6m() -> Self {
        Self::new("JPYLibor6M", Period { length: 6, unit: TimeUnit::Months }, 2,
            Currency::jpy(), Calendar::Japan,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    // -----------------------------------------------------------------------
    // BBSW (Bank Bill Swap Rate — AUD)
    // -----------------------------------------------------------------------

    /// BBSW 3-Month (Australian Bank Bill Swap Rate).
    pub fn bbsw_3m() -> Self {
        Self::new("BBSW3M", Period { length: 3, unit: TimeUnit::Months }, 1,
            Currency::aud(), Calendar::Australia,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual365Fixed)
    }

    /// BBSW 6-Month.
    pub fn bbsw_6m() -> Self {
        Self::new("BBSW6M", Period { length: 6, unit: TimeUnit::Months }, 1,
            Currency::aud(), Calendar::Australia,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual365Fixed)
    }

    // -----------------------------------------------------------------------
    // TIBOR (Tokyo IBOR — JPY)
    // -----------------------------------------------------------------------

    /// TIBOR 3-Month (Domestic Japanese Yen TIBOR).
    pub fn tibor_3m() -> Self {
        Self::new("TIBOR3M", Period { length: 3, unit: TimeUnit::Months }, 2,
            Currency::jpy(), Calendar::Japan,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual365Fixed)
    }

    /// TIBOR 6-Month.
    pub fn tibor_6m() -> Self {
        Self::new("TIBOR6M", Period { length: 6, unit: TimeUnit::Months }, 2,
            Currency::jpy(), Calendar::Japan,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual365Fixed)
    }

    // -----------------------------------------------------------------------
    // SHIBOR (Shanghai IBOR — CNY)
    // -----------------------------------------------------------------------

    /// SHIBOR 3-Month (Shanghai Interbank Offered Rate).
    pub fn shibor_3m() -> Self {
        Self::new("SHIBOR3M", Period { length: 3, unit: TimeUnit::Months }, 1,
            Currency::cny(), Calendar::China,
            BusinessDayConvention::ModifiedFollowing, false, DayCounter::Actual360)
    }

    // -----------------------------------------------------------------------
    // WIBOR (Warsaw IBOR — PLN)
    // -----------------------------------------------------------------------

    /// WIBOR 3-Month (Warsaw Interbank Offered Rate).
    pub fn wibor_3m() -> Self {
        Self::new("WIBOR3M", Period { length: 3, unit: TimeUnit::Months }, 2,
            Currency::pln(), Calendar::Poland,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual365Fixed)
    }

    /// WIBOR 6-Month.
    pub fn wibor_6m() -> Self {
        Self::new("WIBOR6M", Period { length: 6, unit: TimeUnit::Months }, 2,
            Currency::pln(), Calendar::Poland,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual365Fixed)
    }

    // -----------------------------------------------------------------------
    // PRIBOR (Prague IBOR — CZK)
    // -----------------------------------------------------------------------

    /// PRIBOR 3-Month (Prague Interbank Offered Rate).
    pub fn pribor_3m() -> Self {
        Self::new("PRIBOR3M", Period { length: 3, unit: TimeUnit::Months }, 2,
            Currency::czk(), Calendar::CzechRepublic,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    /// PRIBOR 6-Month.
    pub fn pribor_6m() -> Self {
        Self::new("PRIBOR6M", Period { length: 6, unit: TimeUnit::Months }, 2,
            Currency::czk(), Calendar::CzechRepublic,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    // -----------------------------------------------------------------------
    // BUBOR (Budapest IBOR — HUF)
    // -----------------------------------------------------------------------

    /// BUBOR 3-Month (Budapest Interbank Offered Rate).
    pub fn bubor_3m() -> Self {
        Self::new("BUBOR3M", Period { length: 3, unit: TimeUnit::Months }, 2,
            Currency::huf(), Calendar::Hungary,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    // -----------------------------------------------------------------------
    // ROBOR (Bucharest IBOR — RON)
    // -----------------------------------------------------------------------

    /// ROBOR 3-Month (Romanian Interbank Offered Rate).
    pub fn robor_3m() -> Self {
        Self::new("ROBOR3M", Period { length: 3, unit: TimeUnit::Months }, 2,
            Currency::ron(), Calendar::Romania,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    // -----------------------------------------------------------------------
    // TRLIBOR (Turkey)
    // -----------------------------------------------------------------------

    /// TRLIBOR 3-Month (Turkish Lira IBOR).
    pub fn trlibor_3m() -> Self {
        Self::new("TRLIBOR3M", Period { length: 3, unit: TimeUnit::Months }, 2,
            Currency::try_(), Calendar::Turkey,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    // -----------------------------------------------------------------------
    // USD SOFR-linked IBOR (BSBY / SOFR term rate - placeholder)
    // -----------------------------------------------------------------------

    /// USD SOFR Term 3-Month (placeholder for SOFR-linked term rate).
    pub fn sofr_term_3m() -> Self {
        Self::new("SOFRTerm3M", Period { length: 3, unit: TimeUnit::Months }, 2,
            Currency::usd(),
            Calendar::UnitedStates(ql_time::calendar::USMarket::Settlement),
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
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
    fn euribor_3m_properties() {
        let idx = IborIndex::euribor_3m();
        assert_eq!(idx.name(), "Euribor3M");
        assert_eq!(idx.fixing_days, 2);
        assert_eq!(idx.currency.code, "EUR");
    }

    #[test]
    fn euribor_6m_properties() {
        let idx = IborIndex::euribor_6m();
        assert_eq!(idx.name(), "Euribor6M");
        assert_eq!(idx.tenor.length, 6);
    }

    #[test]
    fn ibor_value_and_maturity_dates() {
        let idx = IborIndex::euribor_3m();
        // 2025-01-15 is a Wednesday (business day for TARGET)
        let fixing = Date::from_ymd(2025, Month::January, 15);
        let value = idx.value_date(fixing);
        let maturity = idx.maturity_date(value);
        // Value date should be 2 business days after fixing
        assert!(value > fixing);
        // Maturity should be 3 months after value date
        assert!(maturity > value);
    }

    #[test]
    fn ibor_fixing_from_store() {
        let idx = IborIndex::euribor_3m();
        let date = Date::from_ymd(2025, Month::January, 15);
        IndexManager::instance()
            .add_fixing("Euribor3M", date, 0.035)
            .unwrap();
        let fixing = idx.fixing(date, false).unwrap();
        assert!((fixing - 0.035).abs() < 1e-15);
    }

    #[test]
    fn ibor_missing_fixing() {
        let idx = IborIndex::euribor_3m();
        let date = Date::from_ymd(2099, Month::June, 15);
        assert!(idx.fixing(date, false).is_err());
    }

    #[test]
    fn forecast_fixing_calculation() {
        let idx = IborIndex::euribor_6m();
        let d1 = Date::from_ymd(2025, Month::January, 17);
        let d2 = Date::from_ymd(2025, Month::July, 17);
        let t = idx.year_fraction(d1, d2);
        // If df_start=1.0, df_end=0.975, rate = (1/0.975 - 1) / t
        let rate = idx.forecast_fixing(d1, d2, 1.0, 0.975);
        let expected = (1.0 / 0.975 - 1.0) / t;
        assert!((rate - expected).abs() < 1e-12);
    }
}
