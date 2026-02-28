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

    // -----------------------------------------------------------------------
    // Euribor — additional tenors
    // -----------------------------------------------------------------------

    /// Euribor 2-Week.
    pub fn euribor_2w() -> Self {
        Self::new("Euribor2W", Period { length: 2, unit: TimeUnit::Weeks }, 2,
            Currency::eur(), Calendar::Target,
            BusinessDayConvention::Following, false, DayCounter::Actual360)
    }

    /// Euribor 2-Month.
    pub fn euribor_2m() -> Self {
        Self::new("Euribor2M", Period { length: 2, unit: TimeUnit::Months }, 2,
            Currency::eur(), Calendar::Target,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    /// Euribor 4-Month.
    pub fn euribor_4m() -> Self {
        Self::new("Euribor4M", Period { length: 4, unit: TimeUnit::Months }, 2,
            Currency::eur(), Calendar::Target,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    /// Euribor 5-Month.
    pub fn euribor_5m() -> Self {
        Self::new("Euribor5M", Period { length: 5, unit: TimeUnit::Months }, 2,
            Currency::eur(), Calendar::Target,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    /// Euribor 7-Month.
    pub fn euribor_7m() -> Self {
        Self::new("Euribor7M", Period { length: 7, unit: TimeUnit::Months }, 2,
            Currency::eur(), Calendar::Target,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    /// Euribor 8-Month.
    pub fn euribor_8m() -> Self {
        Self::new("Euribor8M", Period { length: 8, unit: TimeUnit::Months }, 2,
            Currency::eur(), Calendar::Target,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    /// Euribor 9-Month.
    pub fn euribor_9m() -> Self {
        Self::new("Euribor9M", Period { length: 9, unit: TimeUnit::Months }, 2,
            Currency::eur(), Calendar::Target,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    /// Euribor 10-Month.
    pub fn euribor_10m() -> Self {
        Self::new("Euribor10M", Period { length: 10, unit: TimeUnit::Months }, 2,
            Currency::eur(), Calendar::Target,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    /// Euribor 11-Month.
    pub fn euribor_11m() -> Self {
        Self::new("Euribor11M", Period { length: 11, unit: TimeUnit::Months }, 2,
            Currency::eur(), Calendar::Target,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    // -----------------------------------------------------------------------
    // USD LIBOR — additional tenors (legacy)
    // -----------------------------------------------------------------------

    /// USD LIBOR Overnight (legacy).
    pub fn usd_libor_on() -> Self {
        Self::new("USDLiborON", Period { length: 1, unit: TimeUnit::Days }, 0,
            Currency::usd(),
            Calendar::UnitedStates(ql_time::calendar::USMarket::Settlement),
            BusinessDayConvention::Following, false, DayCounter::Actual360)
    }

    /// USD LIBOR 1-Week (legacy).
    pub fn usd_libor_1w() -> Self {
        Self::new("USDLibor1W", Period { length: 1, unit: TimeUnit::Weeks }, 2,
            Currency::usd(),
            Calendar::UnitedStates(ql_time::calendar::USMarket::Settlement),
            BusinessDayConvention::ModifiedFollowing, false, DayCounter::Actual360)
    }

    /// USD LIBOR 1-Month (legacy).
    pub fn usd_libor_1m() -> Self {
        Self::new("USDLibor1M", Period { length: 1, unit: TimeUnit::Months }, 2,
            Currency::usd(),
            Calendar::UnitedStates(ql_time::calendar::USMarket::Settlement),
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    /// USD LIBOR 2-Month (legacy).
    pub fn usd_libor_2m() -> Self {
        Self::new("USDLibor2M", Period { length: 2, unit: TimeUnit::Months }, 2,
            Currency::usd(),
            Calendar::UnitedStates(ql_time::calendar::USMarket::Settlement),
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    /// USD LIBOR 6-Month (legacy).
    pub fn usd_libor_6m() -> Self {
        Self::new("USDLibor6M", Period { length: 6, unit: TimeUnit::Months }, 2,
            Currency::usd(),
            Calendar::UnitedStates(ql_time::calendar::USMarket::Settlement),
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    /// USD LIBOR 12-Month (legacy).
    pub fn usd_libor_12m() -> Self {
        Self::new("USDLibor12M", Period { length: 12, unit: TimeUnit::Months }, 2,
            Currency::usd(),
            Calendar::UnitedStates(ql_time::calendar::USMarket::Settlement),
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    // -----------------------------------------------------------------------
    // GBP LIBOR — additional tenors (legacy)
    // -----------------------------------------------------------------------

    /// GBP LIBOR Overnight (legacy).
    pub fn gbp_libor_on() -> Self {
        Self::new("GBPLiborON", Period { length: 1, unit: TimeUnit::Days }, 0,
            Currency::gbp(), Calendar::UnitedKingdom,
            BusinessDayConvention::Following, false, DayCounter::Actual365Fixed)
    }

    /// GBP LIBOR 1-Week (legacy).
    pub fn gbp_libor_1w() -> Self {
        Self::new("GBPLibor1W", Period { length: 1, unit: TimeUnit::Weeks }, 0,
            Currency::gbp(), Calendar::UnitedKingdom,
            BusinessDayConvention::ModifiedFollowing, false, DayCounter::Actual365Fixed)
    }

    /// GBP LIBOR 1-Month (legacy).
    pub fn gbp_libor_1m() -> Self {
        Self::new("GBPLibor1M", Period { length: 1, unit: TimeUnit::Months }, 0,
            Currency::gbp(), Calendar::UnitedKingdom,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual365Fixed)
    }

    /// GBP LIBOR 2-Month (legacy).
    pub fn gbp_libor_2m() -> Self {
        Self::new("GBPLibor2M", Period { length: 2, unit: TimeUnit::Months }, 0,
            Currency::gbp(), Calendar::UnitedKingdom,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual365Fixed)
    }

    /// GBP LIBOR 12-Month (legacy).
    pub fn gbp_libor_12m() -> Self {
        Self::new("GBPLibor12M", Period { length: 12, unit: TimeUnit::Months }, 0,
            Currency::gbp(), Calendar::UnitedKingdom,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual365Fixed)
    }

    // -----------------------------------------------------------------------
    // CHF LIBOR — additional tenors (legacy)
    // -----------------------------------------------------------------------

    /// CHF LIBOR Overnight (legacy).
    pub fn chf_libor_on() -> Self {
        Self::new("CHFLiborON", Period { length: 1, unit: TimeUnit::Days }, 0,
            Currency::chf(), Calendar::Switzerland,
            BusinessDayConvention::Following, false, DayCounter::Actual360)
    }

    /// CHF LIBOR 1-Week (legacy).
    pub fn chf_libor_1w() -> Self {
        Self::new("CHFLibor1W", Period { length: 1, unit: TimeUnit::Weeks }, 2,
            Currency::chf(), Calendar::Switzerland,
            BusinessDayConvention::ModifiedFollowing, false, DayCounter::Actual360)
    }

    /// CHF LIBOR 1-Month (legacy).
    pub fn chf_libor_1m() -> Self {
        Self::new("CHFLibor1M", Period { length: 1, unit: TimeUnit::Months }, 2,
            Currency::chf(), Calendar::Switzerland,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    /// CHF LIBOR 6-Month (legacy).
    pub fn chf_libor_6m() -> Self {
        Self::new("CHFLibor6M", Period { length: 6, unit: TimeUnit::Months }, 2,
            Currency::chf(), Calendar::Switzerland,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    /// CHF LIBOR 12-Month (legacy).
    pub fn chf_libor_12m() -> Self {
        Self::new("CHFLibor12M", Period { length: 12, unit: TimeUnit::Months }, 2,
            Currency::chf(), Calendar::Switzerland,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    // -----------------------------------------------------------------------
    // JPY LIBOR — additional tenors (legacy)
    // -----------------------------------------------------------------------

    /// JPY LIBOR Overnight (legacy).
    pub fn jpy_libor_on() -> Self {
        Self::new("JPYLiborON", Period { length: 1, unit: TimeUnit::Days }, 0,
            Currency::jpy(), Calendar::Japan,
            BusinessDayConvention::Following, false, DayCounter::Actual360)
    }

    /// JPY LIBOR 1-Month (legacy).
    pub fn jpy_libor_1m() -> Self {
        Self::new("JPYLibor1M", Period { length: 1, unit: TimeUnit::Months }, 2,
            Currency::jpy(), Calendar::Japan,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    /// JPY LIBOR 12-Month (legacy).
    pub fn jpy_libor_12m() -> Self {
        Self::new("JPYLibor12M", Period { length: 12, unit: TimeUnit::Months }, 2,
            Currency::jpy(), Calendar::Japan,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    // -----------------------------------------------------------------------
    // BBSW — additional tenors
    // -----------------------------------------------------------------------

    /// BBSW 1-Month.
    pub fn bbsw_1m() -> Self {
        Self::new("BBSW1M", Period { length: 1, unit: TimeUnit::Months }, 1,
            Currency::aud(), Calendar::Australia,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual365Fixed)
    }

    /// BBSW 2-Month.
    pub fn bbsw_2m() -> Self {
        Self::new("BBSW2M", Period { length: 2, unit: TimeUnit::Months }, 1,
            Currency::aud(), Calendar::Australia,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual365Fixed)
    }

    /// BBSW 4-Month.
    pub fn bbsw_4m() -> Self {
        Self::new("BBSW4M", Period { length: 4, unit: TimeUnit::Months }, 1,
            Currency::aud(), Calendar::Australia,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual365Fixed)
    }

    /// BBSW 5-Month.
    pub fn bbsw_5m() -> Self {
        Self::new("BBSW5M", Period { length: 5, unit: TimeUnit::Months }, 1,
            Currency::aud(), Calendar::Australia,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual365Fixed)
    }

    // -----------------------------------------------------------------------
    // TIBOR — additional tenors
    // -----------------------------------------------------------------------

    /// TIBOR 1-Week.
    pub fn tibor_1w() -> Self {
        Self::new("TIBOR1W", Period { length: 1, unit: TimeUnit::Weeks }, 2,
            Currency::jpy(), Calendar::Japan,
            BusinessDayConvention::ModifiedFollowing, false, DayCounter::Actual365Fixed)
    }

    /// TIBOR 1-Month.
    pub fn tibor_1m() -> Self {
        Self::new("TIBOR1M", Period { length: 1, unit: TimeUnit::Months }, 2,
            Currency::jpy(), Calendar::Japan,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual365Fixed)
    }

    /// TIBOR 2-Month.
    pub fn tibor_2m() -> Self {
        Self::new("TIBOR2M", Period { length: 2, unit: TimeUnit::Months }, 2,
            Currency::jpy(), Calendar::Japan,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual365Fixed)
    }

    /// TIBOR 12-Month.
    pub fn tibor_12m() -> Self {
        Self::new("TIBOR12M", Period { length: 12, unit: TimeUnit::Months }, 2,
            Currency::jpy(), Calendar::Japan,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual365Fixed)
    }

    // -----------------------------------------------------------------------
    // SHIBOR — additional tenors
    // -----------------------------------------------------------------------

    /// SHIBOR Overnight.
    pub fn shibor_on() -> Self {
        Self::new("SHIBORON", Period { length: 1, unit: TimeUnit::Days }, 0,
            Currency::cny(), Calendar::China,
            BusinessDayConvention::Following, false, DayCounter::Actual360)
    }

    /// SHIBOR 1-Week.
    pub fn shibor_1w() -> Self {
        Self::new("SHIBOR1W", Period { length: 1, unit: TimeUnit::Weeks }, 1,
            Currency::cny(), Calendar::China,
            BusinessDayConvention::Following, false, DayCounter::Actual360)
    }

    /// SHIBOR 2-Week.
    pub fn shibor_2w() -> Self {
        Self::new("SHIBOR2W", Period { length: 2, unit: TimeUnit::Weeks }, 1,
            Currency::cny(), Calendar::China,
            BusinessDayConvention::Following, false, DayCounter::Actual360)
    }

    /// SHIBOR 1-Month.
    pub fn shibor_1m() -> Self {
        Self::new("SHIBOR1M", Period { length: 1, unit: TimeUnit::Months }, 1,
            Currency::cny(), Calendar::China,
            BusinessDayConvention::ModifiedFollowing, false, DayCounter::Actual360)
    }

    /// SHIBOR 6-Month.
    pub fn shibor_6m() -> Self {
        Self::new("SHIBOR6M", Period { length: 6, unit: TimeUnit::Months }, 1,
            Currency::cny(), Calendar::China,
            BusinessDayConvention::ModifiedFollowing, false, DayCounter::Actual360)
    }

    /// SHIBOR 9-Month.
    pub fn shibor_9m() -> Self {
        Self::new("SHIBOR9M", Period { length: 9, unit: TimeUnit::Months }, 1,
            Currency::cny(), Calendar::China,
            BusinessDayConvention::ModifiedFollowing, false, DayCounter::Actual360)
    }

    /// SHIBOR 1-Year.
    pub fn shibor_1y() -> Self {
        Self::new("SHIBOR1Y", Period { length: 12, unit: TimeUnit::Months }, 1,
            Currency::cny(), Calendar::China,
            BusinessDayConvention::ModifiedFollowing, false, DayCounter::Actual360)
    }

    // -----------------------------------------------------------------------
    // WIBOR — additional tenor
    // -----------------------------------------------------------------------

    /// WIBOR 1-Month.
    pub fn wibor_1m() -> Self {
        Self::new("WIBOR1M", Period { length: 1, unit: TimeUnit::Months }, 2,
            Currency::pln(), Calendar::Poland,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual365Fixed)
    }

    // -----------------------------------------------------------------------
    // BUBOR — additional tenors
    // -----------------------------------------------------------------------

    /// BUBOR 1-Month.
    pub fn bubor_1m() -> Self {
        Self::new("BUBOR1M", Period { length: 1, unit: TimeUnit::Months }, 2,
            Currency::huf(), Calendar::Hungary,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    /// BUBOR 6-Month.
    pub fn bubor_6m() -> Self {
        Self::new("BUBOR6M", Period { length: 6, unit: TimeUnit::Months }, 2,
            Currency::huf(), Calendar::Hungary,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    // -----------------------------------------------------------------------
    // ROBOR — additional tenors
    // -----------------------------------------------------------------------

    /// ROBOR 1-Month.
    pub fn robor_1m() -> Self {
        Self::new("ROBOR1M", Period { length: 1, unit: TimeUnit::Months }, 2,
            Currency::ron(), Calendar::Romania,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    /// ROBOR 6-Month.
    pub fn robor_6m() -> Self {
        Self::new("ROBOR6M", Period { length: 6, unit: TimeUnit::Months }, 2,
            Currency::ron(), Calendar::Romania,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    // -----------------------------------------------------------------------
    // CDOR (Canadian Dollar Offered Rate)
    // -----------------------------------------------------------------------

    /// CDOR 1-Month.
    pub fn cdor_1m() -> Self {
        Self::new("CDOR1M", Period { length: 1, unit: TimeUnit::Months }, 0,
            Currency::cad(), Calendar::Canada,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual365Fixed)
    }

    /// CDOR 2-Month.
    pub fn cdor_2m() -> Self {
        Self::new("CDOR2M", Period { length: 2, unit: TimeUnit::Months }, 0,
            Currency::cad(), Calendar::Canada,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual365Fixed)
    }

    /// CDOR 3-Month.
    pub fn cdor_3m() -> Self {
        Self::new("CDOR3M", Period { length: 3, unit: TimeUnit::Months }, 0,
            Currency::cad(), Calendar::Canada,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual365Fixed)
    }

    /// CDOR 6-Month.
    pub fn cdor_6m() -> Self {
        Self::new("CDOR6M", Period { length: 6, unit: TimeUnit::Months }, 0,
            Currency::cad(), Calendar::Canada,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual365Fixed)
    }

    /// CDOR 12-Month.
    pub fn cdor_12m() -> Self {
        Self::new("CDOR12M", Period { length: 12, unit: TimeUnit::Months }, 0,
            Currency::cad(), Calendar::Canada,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual365Fixed)
    }

    // -----------------------------------------------------------------------
    // JIBAR (Johannesburg Interbank Agreed Rate — ZAR)
    // -----------------------------------------------------------------------

    /// JIBAR 1-Month.
    pub fn jibar_1m() -> Self {
        Self::new("JIBAR1M", Period { length: 1, unit: TimeUnit::Months }, 0,
            Currency::zar(), Calendar::SouthAfrica,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual365Fixed)
    }

    /// JIBAR 3-Month.
    pub fn jibar_3m() -> Self {
        Self::new("JIBAR3M", Period { length: 3, unit: TimeUnit::Months }, 0,
            Currency::zar(), Calendar::SouthAfrica,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual365Fixed)
    }

    /// JIBAR 6-Month.
    pub fn jibar_6m() -> Self {
        Self::new("JIBAR6M", Period { length: 6, unit: TimeUnit::Months }, 0,
            Currency::zar(), Calendar::SouthAfrica,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual365Fixed)
    }

    /// JIBAR 12-Month.
    pub fn jibar_12m() -> Self {
        Self::new("JIBAR12M", Period { length: 12, unit: TimeUnit::Months }, 0,
            Currency::zar(), Calendar::SouthAfrica,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual365Fixed)
    }

    // -----------------------------------------------------------------------
    // BKBM (Bank Bill Benchmark Rate — NZD)
    // -----------------------------------------------------------------------

    /// BKBM 1-Month.
    pub fn bkbm_1m() -> Self {
        Self::new("BKBM1M", Period { length: 1, unit: TimeUnit::Months }, 0,
            Currency::nzd(), Calendar::NewZealand,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual365Fixed)
    }

    /// BKBM 2-Month.
    pub fn bkbm_2m() -> Self {
        Self::new("BKBM2M", Period { length: 2, unit: TimeUnit::Months }, 0,
            Currency::nzd(), Calendar::NewZealand,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual365Fixed)
    }

    /// BKBM 3-Month.
    pub fn bkbm_3m() -> Self {
        Self::new("BKBM3M", Period { length: 3, unit: TimeUnit::Months }, 0,
            Currency::nzd(), Calendar::NewZealand,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual365Fixed)
    }

    /// BKBM 6-Month.
    pub fn bkbm_6m() -> Self {
        Self::new("BKBM6M", Period { length: 6, unit: TimeUnit::Months }, 0,
            Currency::nzd(), Calendar::NewZealand,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual365Fixed)
    }

    // -----------------------------------------------------------------------
    // NIBOR (Norwegian Interbank Offered Rate — NOK)
    // -----------------------------------------------------------------------

    /// NIBOR 1-Week.
    pub fn nibor_1w() -> Self {
        Self::new("NIBOR1W", Period { length: 1, unit: TimeUnit::Weeks }, 2,
            Currency::nok(), Calendar::Norway,
            BusinessDayConvention::ModifiedFollowing, false, DayCounter::Actual360)
    }

    /// NIBOR 1-Month.
    pub fn nibor_1m() -> Self {
        Self::new("NIBOR1M", Period { length: 1, unit: TimeUnit::Months }, 2,
            Currency::nok(), Calendar::Norway,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    /// NIBOR 2-Month.
    pub fn nibor_2m() -> Self {
        Self::new("NIBOR2M", Period { length: 2, unit: TimeUnit::Months }, 2,
            Currency::nok(), Calendar::Norway,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    /// NIBOR 3-Month.
    pub fn nibor_3m() -> Self {
        Self::new("NIBOR3M", Period { length: 3, unit: TimeUnit::Months }, 2,
            Currency::nok(), Calendar::Norway,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    /// NIBOR 6-Month.
    pub fn nibor_6m() -> Self {
        Self::new("NIBOR6M", Period { length: 6, unit: TimeUnit::Months }, 2,
            Currency::nok(), Calendar::Norway,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    // -----------------------------------------------------------------------
    // STIBOR (Stockholm Interbank Offered Rate — SEK)
    // -----------------------------------------------------------------------

    /// STIBOR 1-Week.
    pub fn stibor_1w() -> Self {
        Self::new("STIBOR1W", Period { length: 1, unit: TimeUnit::Weeks }, 2,
            Currency::sek(), Calendar::Sweden,
            BusinessDayConvention::ModifiedFollowing, false, DayCounter::Actual360)
    }

    /// STIBOR 1-Month.
    pub fn stibor_1m() -> Self {
        Self::new("STIBOR1M", Period { length: 1, unit: TimeUnit::Months }, 2,
            Currency::sek(), Calendar::Sweden,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    /// STIBOR 2-Month.
    pub fn stibor_2m() -> Self {
        Self::new("STIBOR2M", Period { length: 2, unit: TimeUnit::Months }, 2,
            Currency::sek(), Calendar::Sweden,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    /// STIBOR 3-Month.
    pub fn stibor_3m() -> Self {
        Self::new("STIBOR3M", Period { length: 3, unit: TimeUnit::Months }, 2,
            Currency::sek(), Calendar::Sweden,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    /// STIBOR 6-Month.
    pub fn stibor_6m() -> Self {
        Self::new("STIBOR6M", Period { length: 6, unit: TimeUnit::Months }, 2,
            Currency::sek(), Calendar::Sweden,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    // -----------------------------------------------------------------------
    // CIBOR (Copenhagen Interbank Offered Rate — DKK)
    // -----------------------------------------------------------------------

    /// CIBOR 1-Month.
    pub fn cibor_1m() -> Self {
        Self::new("CIBOR1M", Period { length: 1, unit: TimeUnit::Months }, 2,
            Currency::dkk(), Calendar::Denmark,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    /// CIBOR 2-Month.
    pub fn cibor_2m() -> Self {
        Self::new("CIBOR2M", Period { length: 2, unit: TimeUnit::Months }, 2,
            Currency::dkk(), Calendar::Denmark,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    /// CIBOR 3-Month.
    pub fn cibor_3m() -> Self {
        Self::new("CIBOR3M", Period { length: 3, unit: TimeUnit::Months }, 2,
            Currency::dkk(), Calendar::Denmark,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    /// CIBOR 6-Month.
    pub fn cibor_6m() -> Self {
        Self::new("CIBOR6M", Period { length: 6, unit: TimeUnit::Months }, 2,
            Currency::dkk(), Calendar::Denmark,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    // -----------------------------------------------------------------------
    // MOSPRIME (Moscow Prime Offered Rate — RUB)
    // -----------------------------------------------------------------------

    /// MOSPRIME 1-Month.
    pub fn mosprime_1m() -> Self {
        Self::new("MOSPRIME1M", Period { length: 1, unit: TimeUnit::Months }, 1,
            Currency::rub(), Calendar::Russia,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    /// MOSPRIME 2-Month.
    pub fn mosprime_2m() -> Self {
        Self::new("MOSPRIME2M", Period { length: 2, unit: TimeUnit::Months }, 1,
            Currency::rub(), Calendar::Russia,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    /// MOSPRIME 3-Month.
    pub fn mosprime_3m() -> Self {
        Self::new("MOSPRIME3M", Period { length: 3, unit: TimeUnit::Months }, 1,
            Currency::rub(), Calendar::Russia,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    /// MOSPRIME 6-Month.
    pub fn mosprime_6m() -> Self {
        Self::new("MOSPRIME6M", Period { length: 6, unit: TimeUnit::Months }, 1,
            Currency::rub(), Calendar::Russia,
            BusinessDayConvention::ModifiedFollowing, true, DayCounter::Actual360)
    }

    // -----------------------------------------------------------------------
    // AMERIBOR (American Interbank Offered Rate — USD)
    // -----------------------------------------------------------------------

    /// AMERIBOR (overnight unsecured USD rate).
    pub fn ameribor() -> Self {
        Self::new("AMERIBOR", Period { length: 1, unit: TimeUnit::Days }, 0,
            Currency::usd(),
            Calendar::UnitedStates(ql_time::calendar::USMarket::Settlement),
            BusinessDayConvention::Following, false, DayCounter::Actual360)
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

    #[test]
    fn cdor_properties() {
        let idx = IborIndex::cdor_3m();
        assert_eq!(idx.name(), "CDOR3M");
        assert_eq!(idx.currency.code, "CAD");
        assert_eq!(idx.fixing_days, 0);
    }

    #[test]
    fn jibar_properties() {
        let idx = IborIndex::jibar_3m();
        assert_eq!(idx.name(), "JIBAR3M");
        assert_eq!(idx.currency.code, "ZAR");
    }

    #[test]
    fn nibor_properties() {
        let idx = IborIndex::nibor_3m();
        assert_eq!(idx.name(), "NIBOR3M");
        assert_eq!(idx.currency.code, "NOK");
    }

    #[test]
    fn stibor_properties() {
        let idx = IborIndex::stibor_3m();
        assert_eq!(idx.name(), "STIBOR3M");
        assert_eq!(idx.currency.code, "SEK");
    }

    #[test]
    fn cibor_properties() {
        let idx = IborIndex::cibor_3m();
        assert_eq!(idx.name(), "CIBOR3M");
        assert_eq!(idx.currency.code, "DKK");
    }

    #[test]
    fn bkbm_properties() {
        let idx = IborIndex::bkbm_3m();
        assert_eq!(idx.name(), "BKBM3M");
        assert_eq!(idx.currency.code, "NZD");
    }

    #[test]
    fn mosprime_properties() {
        let idx = IborIndex::mosprime_3m();
        assert_eq!(idx.name(), "MOSPRIME3M");
        assert_eq!(idx.currency.code, "RUB");
    }

    #[test]
    fn ameribor_properties() {
        let idx = IborIndex::ameribor();
        assert_eq!(idx.name(), "AMERIBOR");
        assert_eq!(idx.currency.code, "USD");
        assert_eq!(idx.fixing_days, 0);
    }

    #[test]
    fn euribor_all_tenors_exist() {
        // Verify all 14 Euribor tenors construct without panic
        let tenors = [
            IborIndex::euribor_1w(), IborIndex::euribor_2w(),
            IborIndex::euribor_1m(), IborIndex::euribor_2m(),
            IborIndex::euribor_3m(), IborIndex::euribor_4m(),
            IborIndex::euribor_5m(), IborIndex::euribor_6m(),
            IborIndex::euribor_7m(), IborIndex::euribor_8m(),
            IborIndex::euribor_9m(), IborIndex::euribor_10m(),
            IborIndex::euribor_11m(), IborIndex::euribor_12m(),
        ];
        assert_eq!(tenors.len(), 14);
        for idx in &tenors {
            assert_eq!(idx.currency.code, "EUR");
        }
    }

    #[test]
    fn usd_libor_all_tenors_exist() {
        let tenors = [
            IborIndex::usd_libor_on(), IborIndex::usd_libor_1w(),
            IborIndex::usd_libor_1m(), IborIndex::usd_libor_2m(),
            IborIndex::usd_libor_3m(), IborIndex::usd_libor_6m(),
            IborIndex::usd_libor_12m(),
        ];
        assert_eq!(tenors.len(), 7);
        for idx in &tenors {
            assert_eq!(idx.currency.code, "USD");
        }
    }
}
