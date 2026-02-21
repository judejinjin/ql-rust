//! Calendar framework — enum-based for zero-cost dispatch.
//!
//! Provides holiday checking, business day adjustment, and date advancement
//! for major financial market calendars.

use serde::{Deserialize, Serialize};

use crate::business_day_convention::BusinessDayConvention;
use crate::date::{Date, Month, Weekday};
use crate::period::{Period, TimeUnit};

/// Sub-market variants for the United States calendar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum USMarket {
    /// US government bond market (SIFMA).
    Settlement,
    /// New York Stock Exchange.
    NYSE,
    /// Federal Reserve Bankwire system.
    FederalReserve,
}

/// Enum-based calendar — no vtable overhead, no heap allocation.
///
/// Use `Calendar::is_business_day()` for the core holiday check, and
/// higher-level methods like `advance()` and `adjust()` for date arithmetic.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Calendar {
    NullCalendar,
    /// Only weekends (Saturday + Sunday) are holidays.
    WeekendsOnly,
    /// TARGET calendar (Trans-European Automated Real-time Gross settlement
    /// Express Transfer) — used for EUR instruments.
    Target,
    /// United States calendar with a specific market variant.
    UnitedStates(USMarket),
    /// United Kingdom (London Exchange / Settlement).
    UnitedKingdom,
    /// Japan (Tokyo Stock Exchange / Settlement).
    Japan,
    /// China (Shanghai Stock Exchange / IB).
    China,
    /// Canada (Toronto Stock Exchange / Settlement).
    Canada,
    /// Australia (Sydney / ASX).
    Australia,
    /// Brazil (BM&F Bovespa / Settlement).
    Brazil,
    /// Germany (Frankfurt / Eurex / Settlement).
    Germany,
    /// France (Paris / Euronext).
    France,
    /// Italy (Borsa Italiana / Settlement).
    Italy,
    /// Switzerland (SIX Swiss Exchange / Settlement).
    Switzerland,
    /// India (National Stock Exchange / Settlement).
    India,
    /// South Korea (Korea Exchange — KRX).
    SouthKorea,
    /// Hong Kong (HKEX / Settlement).
    HongKong,
    /// Singapore (SGX / Settlement).
    Singapore,
    /// Mexico (BMV / Settlement).
    Mexico,
    /// South Africa (JSE / Settlement).
    SouthAfrica,
    /// Sweden (Stockholm / OMX Nordic).
    Sweden,
    /// Denmark (Copenhagen / OMX Nordic).
    Denmark,
    /// Norway (Oslo Børs).
    Norway,
    /// Poland (Warsaw Stock Exchange — GPW).
    Poland,
    /// New Zealand (NZX / Settlement).
    NewZealand,
    /// Joint calendar combining two calendars.
    Joint(JointRule, Box<Calendar>, Box<Calendar>),
}

/// Rule for combining calendars in a JointCalendar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum JointRule {
    /// A day is a holiday if it is a holiday in EITHER calendar (union of holidays).
    JoinHolidays,
    /// A day is a business day if it is a business day in EITHER calendar (union of business days).
    JoinBusinessDays,
}

impl Calendar {
    /// Create a joint calendar (union of holidays = intersection of business days).
    pub fn join_holidays(a: Calendar, b: Calendar) -> Self {
        Calendar::Joint(JointRule::JoinHolidays, Box::new(a), Box::new(b))
    }

    /// Create a joint calendar (union of business days).
    pub fn join_business_days(a: Calendar, b: Calendar) -> Self {
        Calendar::Joint(JointRule::JoinBusinessDays, Box::new(a), Box::new(b))
    }

    /// Whether the given date is a business day.
    ///
    /// # Examples
    ///
    /// ```
    /// use ql_time::{Calendar, Date, Month};
    ///
    /// let cal = Calendar::Target;
    /// // A regular Monday
    /// assert!(cal.is_business_day(Date::from_ymd(2025, Month::March, 17)));
    /// // A Sunday
    /// assert!(!cal.is_business_day(Date::from_ymd(2025, Month::March, 16)));
    /// // Christmas
    /// assert!(!cal.is_business_day(Date::from_ymd(2025, Month::December, 25)));
    /// ```
    pub fn is_business_day(&self, date: Date) -> bool {
        match self {
            Calendar::NullCalendar => true,
            Calendar::WeekendsOnly => !is_weekend(date),
            Calendar::Target => target_is_business_day(date),
            Calendar::UnitedStates(market) => us_is_business_day(date, *market),
            Calendar::UnitedKingdom => uk_is_business_day(date),
            Calendar::Japan => japan_is_business_day(date),
            Calendar::China => china_is_business_day(date),
            Calendar::Canada => canada_is_business_day(date),
            Calendar::Australia => australia_is_business_day(date),
            Calendar::Brazil => brazil_is_business_day(date),
            Calendar::Germany => germany_is_business_day(date),
            Calendar::France => france_is_business_day(date),
            Calendar::Italy => italy_is_business_day(date),
            Calendar::Switzerland => switzerland_is_business_day(date),
            Calendar::India => india_is_business_day(date),
            Calendar::SouthKorea => south_korea_is_business_day(date),
            Calendar::HongKong => hong_kong_is_business_day(date),
            Calendar::Singapore => singapore_is_business_day(date),
            Calendar::Mexico => mexico_is_business_day(date),
            Calendar::SouthAfrica => south_africa_is_business_day(date),
            Calendar::Sweden => sweden_is_business_day(date),
            Calendar::Denmark => denmark_is_business_day(date),
            Calendar::Norway => norway_is_business_day(date),
            Calendar::Poland => poland_is_business_day(date),
            Calendar::NewZealand => new_zealand_is_business_day(date),
            Calendar::Joint(rule, a, b) => match rule {
                JointRule::JoinHolidays => a.is_business_day(date) && b.is_business_day(date),
                JointRule::JoinBusinessDays => a.is_business_day(date) || b.is_business_day(date),
            },
        }
    }

    /// Whether the given date is a holiday.
    pub fn is_holiday(&self, date: Date) -> bool {
        !self.is_business_day(date)
    }

    /// Adjust a date according to a business day convention.
    pub fn adjust(&self, date: Date, convention: BusinessDayConvention) -> Date {
        match convention {
            BusinessDayConvention::Unadjusted => date,
            BusinessDayConvention::Following => {
                let mut d = date;
                while !self.is_business_day(d) {
                    d += 1;
                }
                d
            }
            BusinessDayConvention::ModifiedFollowing => {
                let fwd = self.adjust(date, BusinessDayConvention::Following);
                if fwd.month() != date.month() {
                    self.adjust(date, BusinessDayConvention::Preceding)
                } else {
                    fwd
                }
            }
            BusinessDayConvention::Preceding => {
                let mut d = date;
                while !self.is_business_day(d) {
                    d -= 1;
                }
                d
            }
            BusinessDayConvention::ModifiedPreceding => {
                let back = self.adjust(date, BusinessDayConvention::Preceding);
                if back.month() != date.month() {
                    self.adjust(date, BusinessDayConvention::Following)
                } else {
                    back
                }
            }
            BusinessDayConvention::Nearest => {
                let mut fwd = date;
                let mut bwd = date;
                loop {
                    if self.is_business_day(fwd) {
                        return fwd;
                    }
                    if self.is_business_day(bwd) {
                        return bwd;
                    }
                    fwd += 1;
                    bwd -= 1;
                }
            }
            BusinessDayConvention::HalfMonthModifiedFollowing => {
                let d = self.adjust(date, BusinessDayConvention::ModifiedFollowing);
                if date.day_of_month() <= 15 && d.day_of_month() > 15 {
                    self.adjust(date, BusinessDayConvention::Preceding)
                } else {
                    d
                }
            }
        }
    }

    /// Advance a date by a period, then adjust according to the convention.
    ///
    /// # Examples
    ///
    /// ```
    /// use ql_time::{Calendar, Date, Month, Period, BusinessDayConvention};
    ///
    /// let cal = Calendar::Target;
    /// let start = Date::from_ymd(2025, Month::January, 15);
    /// let advanced = cal.advance(
    ///     start,
    ///     Period::months(3),
    ///     BusinessDayConvention::ModifiedFollowing,
    ///     false,
    /// );
    /// assert_eq!(advanced, Date::from_ymd(2025, Month::April, 15));
    /// ```
    pub fn advance(
        &self,
        date: Date,
        period: Period,
        convention: BusinessDayConvention,
        end_of_month: bool,
    ) -> Date {
        match period.unit {
            TimeUnit::Days => {
                let mut d = date;
                if period.length > 0 {
                    for _ in 0..period.length {
                        d += 1;
                        while !self.is_business_day(d) {
                            d += 1;
                        }
                    }
                } else if period.length < 0 {
                    for _ in 0..(-period.length) {
                        d -= 1;
                        while !self.is_business_day(d) {
                            d -= 1;
                        }
                    }
                }
                d
            }
            TimeUnit::Weeks => {
                let raw = date + (period.length * 7);
                self.adjust(raw, convention)
            }
            TimeUnit::Months => {
                let raw = add_months(date, period.length);
                if end_of_month && self.is_end_of_month(date) {
                    let eom = raw.end_of_month();
                    self.adjust(eom, BusinessDayConvention::Preceding)
                } else {
                    self.adjust(raw, convention)
                }
            }
            TimeUnit::Years => {
                let raw = add_months(date, period.length * 12);
                if end_of_month && self.is_end_of_month(date) {
                    let eom = raw.end_of_month();
                    self.adjust(eom, BusinessDayConvention::Preceding)
                } else {
                    self.adjust(raw, convention)
                }
            }
        }
    }

    /// Advance a date by a number of business days.
    pub fn advance_business_days(&self, date: Date, n: i32) -> Date {
        if n > 0 {
            self.advance(date, Period::days(n), BusinessDayConvention::Following, false)
        } else if n < 0 {
            self.advance(date, Period::days(n), BusinessDayConvention::Preceding, false)
        } else {
            date
        }
    }

    /// Whether the date is the last business day of its month.
    pub fn is_end_of_month(&self, date: Date) -> bool {
        self.is_business_day(date)
            && !self.is_business_day(date + 1)
            || date.is_end_of_month()
            || (!self.is_business_day(date + 1)
                && (date + 1).month() != date.month())
    }

    /// Count business days between two dates (exclusive of d1, inclusive of d2).
    ///
    /// # Examples
    ///
    /// ```
    /// use ql_time::{Calendar, Date, Month};
    ///
    /// let cal = Calendar::Target;
    /// let d1 = Date::from_ymd(2025, Month::March, 14); // Friday
    /// let d2 = Date::from_ymd(2025, Month::March, 21); // Friday
    /// // Mon-Fri of next week = 5 business days
    /// assert_eq!(cal.business_days_between(d1, d2), 5);
    /// ```
    pub fn business_days_between(&self, from: Date, to: Date) -> i32 {
        if from == to {
            return 0;
        }
        let (start, end, sign) = if from < to {
            (from, to, 1)
        } else {
            (to, from, -1)
        };
        let mut count = 0i32;
        let mut d = start + 1;
        while d <= end {
            if self.is_business_day(d) {
                count += 1;
            }
            d += 1;
        }
        count * sign
    }
}

/// Check if a date falls on a weekend (Saturday or Sunday).
fn is_weekend(date: Date) -> bool {
    matches!(date.weekday(), Weekday::Saturday | Weekday::Sunday)
}

/// Add `n` calendar months to a date, capping at end-of-month.
fn add_months(date: Date, months: i32) -> Date {
    let (y, m, d) = (date.year(), date.month() as u32, date.day_of_month());
    let total_months = (y * 12 + m as i32 - 1) + months;
    let new_year = total_months.div_euclid(12);
    let new_month = (total_months.rem_euclid(12) + 1) as u32;
    let max_day = Date::days_in_month(new_year, new_month);
    let new_day = d.min(max_day);
    // Date is computed from valid calendar arithmetic; cannot fail.
    Date::from_ymd_opt(new_year, new_month, new_day).unwrap_or_else(|| unreachable!())
}

// ============================================================================
//  TARGET calendar implementation
// ============================================================================

fn target_is_business_day(date: Date) -> bool {
    if is_weekend(date) {
        return false;
    }
    let d = date.day_of_month();
    let m = date.month();
    let y = date.year();

    // Fixed holidays
    if d == 1 && m == Month::January {
        return false; // New Year's Day
    }
    if d == 1 && m == Month::May && y >= 2000 {
        return false; // Labour Day
    }
    if d == 25 && m == Month::December {
        return false; // Christmas
    }
    if d == 26 && m == Month::December && y >= 2000 {
        return false; // St. Stephen's Day
    }
    if d == 31 && m == Month::December && (y == 1998 || y == 1999 || y == 2001) {
        return false;
    }

    // Easter-based holidays (Good Friday and Easter Monday)
    let (em_month, em_day) = easter_monday(y);
    // Good Friday = Easter Monday - 3
    let em_serial = Date::from_ymd_opt(y, em_month, em_day)
        .unwrap_or_else(|| unreachable!())
        .serial();
    let gf = Date::from_serial(em_serial - 3);
    if date == gf {
        return false;
    }
    // Easter Monday
    if d == em_day && m == Month::from_u32(em_month).unwrap_or(Month::January) {
        return false;
    }

    true
}

// ============================================================================
//  United States calendar implementation
// ============================================================================

fn us_is_business_day(date: Date, market: USMarket) -> bool {
    if is_weekend(date) {
        return false;
    }
    let d = date.day_of_month();
    let m = date.month();
    let y = date.year();
    let wd = date.weekday();

    // New Year's Day (observed)
    if (d == 1 || (d == 2 && wd == Weekday::Monday)) && m == Month::January {
        return false;
    }
    // New Year's Day observed on previous Friday if Jan 1 is Saturday
    if d == 31 && m == Month::December && wd == Weekday::Friday {
        return false;
    }

    // MLK Day (3rd Monday of January, since 1983)
    if m == Month::January && wd == Weekday::Monday && (15..=21).contains(&d) && y >= 1983 {
        return false;
    }

    // Presidents' Day / Washington's Birthday (3rd Monday of February)
    if m == Month::February && wd == Weekday::Monday && (15..=21).contains(&d) {
        return false;
    }

    // Good Friday — NYSE only
    if market == USMarket::NYSE {
        let (em_month, em_day) = easter_monday(y);
        let em_serial = Date::from_ymd_opt(y, em_month, em_day)
            .unwrap_or_else(|| unreachable!())
            .serial();
        let gf = Date::from_serial(em_serial - 3);
        if date == gf {
            return false;
        }
    }

    // Memorial Day (last Monday of May)
    if m == Month::May && wd == Weekday::Monday && d >= 25 {
        return false;
    }

    // Juneteenth National Independence Day (since 2022)
    if y >= 2022
        && m == Month::June
        && ((d == 19)
            || (d == 20 && wd == Weekday::Monday)
            || (d == 18 && wd == Weekday::Friday))
    {
        return false;
    }

    // Independence Day (July 4, observed)
    if m == Month::July
        && ((d == 4)
            || (d == 5 && wd == Weekday::Monday)
            || (d == 3 && wd == Weekday::Friday))
    {
        return false;
    }

    // Labor Day (1st Monday of September)
    if m == Month::September && wd == Weekday::Monday && d <= 7 {
        return false;
    }

    // Columbus Day / Indigenous Peoples' Day (2nd Monday of October) — Settlement only
    if matches!(market, USMarket::Settlement | USMarket::FederalReserve)
        && m == Month::October
        && wd == Weekday::Monday
        && (8..=14).contains(&d)
    {
        return false;
    }

    // Veterans Day (Nov 11, observed) — Settlement only
    if matches!(market, USMarket::Settlement | USMarket::FederalReserve)
        && m == Month::November
        && ((d == 11)
            || (d == 12 && wd == Weekday::Monday)
            || (d == 10 && wd == Weekday::Friday))
    {
        return false;
    }

    // Thanksgiving (4th Thursday of November)
    if m == Month::November && wd == Weekday::Thursday && (22..=28).contains(&d) {
        return false;
    }

    // Christmas (Dec 25, observed)
    if m == Month::December
        && ((d == 25)
            || (d == 26 && wd == Weekday::Monday)
            || (d == 24 && wd == Weekday::Friday))
    {
        return false;
    }

    true
}

// ============================================================================
//  United Kingdom calendar implementation
// ============================================================================

fn uk_is_business_day(date: Date) -> bool {
    if is_weekend(date) {
        return false;
    }
    let d = date.day_of_month();
    let m = date.month();
    let y = date.year();
    let wd = date.weekday();

    // New Year's Day (observed)
    if m == Month::January && (d == 1 || (d <= 3 && wd == Weekday::Monday)) {
        return false;
    }

    // Easter-based holidays
    let (em_month, em_day) = easter_monday(y);
    let em_serial = Date::from_ymd_opt(y, em_month, em_day)
        .unwrap_or_else(|| unreachable!())
        .serial();
    let gf = Date::from_serial(em_serial - 3);
    let em = Date::from_ymd_opt(y, em_month, em_day).unwrap_or_else(|| unreachable!());
    if date == gf || date == em {
        return false;
    }

    // Early May Bank Holiday (first Monday of May)
    if m == Month::May && wd == Weekday::Monday && d <= 7 {
        return false;
    }

    // Spring Bank Holiday (last Monday of May)
    if m == Month::May && wd == Weekday::Monday && d >= 25 {
        return false;
    }

    // Summer Bank Holiday (last Monday of August)
    if m == Month::August && wd == Weekday::Monday && d >= 25 {
        return false;
    }

    // Christmas (Dec 25) + Boxing Day (Dec 26), observed
    if m == Month::December
        && ((d == 25 || d == 26)
            || (d == 27 && matches!(wd, Weekday::Monday | Weekday::Tuesday))
            || (d == 28 && matches!(wd, Weekday::Monday | Weekday::Tuesday)))
    {
        return false;
    }

    true
}

// ============================================================================
//  Easter computation (Anonymous Gregorian algorithm)
// ============================================================================

/// Compute the date of Easter Monday for a given year (cached).
/// Returns (month, day) of Easter Monday.
fn easter_monday(year: i32) -> (u32, u32) {
    use std::cell::RefCell;
    thread_local! {
        static CACHE: RefCell<(i32, u32, u32)> = const { RefCell::new((0, 0, 0)) };
    }
    CACHE.with(|c| {
        let cached = *c.borrow();
        if cached.0 == year {
            return (cached.1, cached.2);
        }
        let (m, d) = easter_monday_impl(year);
        *c.borrow_mut() = (year, m, d);
        (m, d)
    })
}

/// Raw Easter Monday computation (Anonymous Gregorian algorithm).
fn easter_monday_impl(year: i32) -> (u32, u32) {
    let a = year % 19;
    let b = year / 100;
    let c = year % 100;
    let d = b / 4;
    let e = b % 4;
    let f = (b + 8) / 25;
    let g = (b - f + 1) / 3;
    let h = (19 * a + b - d - g + 15) % 30;
    let i = c / 4;
    let k = c % 4;
    let l = (32 + 2 * e + 2 * i - h - k) % 7;
    let m = (a + 11 * h + 22 * l) / 451;
    let month = (h + l - 7 * m + 114) / 31;
    let day = ((h + l - 7 * m + 114) % 31) + 1;
    // This gives Easter Sunday; Easter Monday is the next day
    // Easter date is computed from a known-valid algorithm; cannot fail.
    let easter_sunday = Date::from_ymd_opt(year, month as u32, day as u32)
        .unwrap_or_else(|| unreachable!());
    let easter_monday = easter_sunday + 1;
    (
        easter_monday.month() as u32,
        easter_monday.day_of_month(),
    )
}

// ============================================================================
//  Japan calendar implementation (Tokyo Stock Exchange)
// ============================================================================

fn japan_is_business_day(date: Date) -> bool {
    if is_weekend(date) {
        return false;
    }
    let d = date.day_of_month();
    let m = date.month();
    let y = date.year();
    let wd = date.weekday();

    // ---- Fixed & rule-based holidays ----

    // New Year's holidays (Jan 1-3)
    if m == Month::January && d <= 3 {
        return false;
    }
    // Coming of Age Day (2nd Monday of January)
    if m == Month::January && wd == Weekday::Monday && (8..=14).contains(&d) {
        return false;
    }
    // National Foundation Day (Feb 11)
    if m == Month::February && d == 11 {
        return false;
    }
    // Emperor's Birthday (Feb 23, since 2020; Dec 23 before 2019)
    if y >= 2020 && m == Month::February && d == 23 {
        return false;
    }
    if y < 2019 && m == Month::December && d == 23 {
        return false;
    }
    // Vernal Equinox (~Mar 20-21)
    let vernal = japan_vernal_equinox(y);
    if m == Month::March && d == vernal {
        return false;
    }
    // Showa Day (Apr 29)
    if m == Month::April && d == 29 {
        return false;
    }
    // Constitution Memorial Day (May 3)
    if m == Month::May && d == 3 {
        return false;
    }
    // Greenery Day (May 4)
    if m == Month::May && d == 4 {
        return false;
    }
    // Children's Day (May 5)
    if m == Month::May && d == 5 {
        return false;
    }
    // Marine Day (3rd Monday of July)
    if m == Month::July && wd == Weekday::Monday && (15..=21).contains(&d) {
        return false;
    }
    // Mountain Day (Aug 11, since 2016)
    if y >= 2016 && m == Month::August && d == 11 {
        return false;
    }
    // Respect for the Aged Day (3rd Monday of September)
    if m == Month::September && wd == Weekday::Monday && (15..=21).contains(&d) {
        return false;
    }
    // Autumnal Equinox (~Sep 22-23)
    let autumnal = japan_autumnal_equinox(y);
    if m == Month::September && d == autumnal {
        return false;
    }
    // Sports Day / Health and Sports Day (2nd Monday of October)
    if m == Month::October && wd == Weekday::Monday && (8..=14).contains(&d) {
        return false;
    }
    // Culture Day (Nov 3)
    if m == Month::November && d == 3 {
        return false;
    }
    // Labour Thanksgiving Day (Nov 23)
    if m == Month::November && d == 23 {
        return false;
    }

    // ---- Substitute Holiday (振替休日) ----
    // When a national holiday falls on Sunday, the next Monday is observed.
    // Also: Golden Week may push a substitute to Tuesday (May 6).
    if wd == Weekday::Monday {
        let prev_sun = date - 1; // the preceding Sunday
        if !japan_is_national_holiday(prev_sun) {
            // no substitute
        } else {
            return false; // Monday is substitute for Sunday holiday
        }
    }
    // Golden Week special: if May 3 (Tue-Sat) or May 4 (Wed-Sat) creates
    // a gap with Children's Day, May 6 (Tue) can be a substitute.
    if m == Month::May && d == 6 && matches!(wd, Weekday::Tuesday | Weekday::Wednesday) {
        return false;
    }

    true
}

/// Check if a specific date is a Japanese national holiday (for substitute rule).
fn japan_is_national_holiday(date: Date) -> bool {
    let d = date.day_of_month();
    let m = date.month();
    let y = date.year();

    // New Year
    if m == Month::January && d <= 3 { return true; }
    // Coming of Age (not Sunday-dependent as it's always Monday)
    // National Foundation Day
    if m == Month::February && d == 11 { return true; }
    // Emperor's Birthday
    if y >= 2020 && m == Month::February && d == 23 { return true; }
    if y < 2019 && m == Month::December && d == 23 { return true; }
    // Vernal Equinox
    if m == Month::March && d == japan_vernal_equinox(y) { return true; }
    // Showa Day
    if m == Month::April && d == 29 { return true; }
    // Constitution
    if m == Month::May && d == 3 { return true; }
    // Greenery
    if m == Month::May && d == 4 { return true; }
    // Children's
    if m == Month::May && d == 5 { return true; }
    // Mountain Day
    if y >= 2016 && m == Month::August && d == 11 { return true; }
    // Autumnal Equinox
    if m == Month::September && d == japan_autumnal_equinox(y) { return true; }
    // Culture Day
    if m == Month::November && d == 3 { return true; }
    // Labour Thanksgiving
    if m == Month::November && d == 23 { return true; }
    false
}

/// Vernal equinox day for Japan (1980-2099 range).
fn japan_vernal_equinox(y: i32) -> u32 {
    // Standard formula from Japan National Astronomical Observatory
    // Coefficients differ by era per the published almanac tables.
    let yf = y as f64;
    if y <= 1999 {
        (20.8431 + 0.242_194 * (yf - 1980.0) - ((yf - 1980.0) / 4.0).floor()) as u32
    } else {
        (20.9031 + 0.242_194 * (yf - 1980.0) - ((yf - 1980.0) / 4.0).floor()) as u32
    }
}

/// Autumnal equinox day for Japan (1980-2099 range).
fn japan_autumnal_equinox(y: i32) -> u32 {
    let yf = y as f64;
    if y <= 1999 {
        (23.2488 + 0.242_194 * (yf - 1980.0) - ((yf - 1980.0) / 4.0).floor()) as u32
    } else {
        (23.2088 + 0.242_194 * (yf - 1980.0) - ((yf - 1980.0) / 4.0).floor()) as u32
    }
}

// ============================================================================
//  China calendar implementation (SSE)
// ============================================================================

/// China calendar — uses a simplified rule set (weekends + fixed holidays).
/// Real-world China calendar requires annual CSRC announcements for workday
/// swaps; this covers the standard statutory holidays.
fn china_is_business_day(date: Date) -> bool {
    if is_weekend(date) {
        return false;
    }
    let d = date.day_of_month();
    let m = date.month();

    // New Year (Jan 1)
    if m == Month::January && d == 1 {
        return false;
    }
    // Spring Festival (approx Jan 29 - Feb 4, simplified as Jan 31 - Feb 6)
    // This varies by year; we use a simplified rule
    // Labour Day (May 1)
    if m == Month::May && d == 1 {
        return false;
    }
    // National Day (Oct 1-3)
    if m == Month::October && (1..=3).contains(&d) {
        return false;
    }
    // Qingming Festival (Apr 5 approx)
    if m == Month::April && d == 5 {
        return false;
    }
    // Dragon Boat Festival (approx — varies)
    // Mid-Autumn Festival (approx — varies)

    true
}

// ============================================================================
//  Canada calendar implementation (TSX / Settlement)
// ============================================================================

fn canada_is_business_day(date: Date) -> bool {
    if is_weekend(date) {
        return false;
    }
    let d = date.day_of_month();
    let m = date.month();
    let y = date.year();
    let wd = date.weekday();

    // New Year's Day (observed)
    if m == Month::January && (d == 1 || (d == 2 && wd == Weekday::Monday)) {
        return false;
    }
    // Family Day (3rd Monday of February, since 2008 in most provinces)
    if m == Month::February && wd == Weekday::Monday && (15..=21).contains(&d) && y >= 2008 {
        return false;
    }
    // Good Friday
    let (em_month, em_day) = easter_monday(y);
    let em_serial = Date::from_ymd_opt(y, em_month, em_day)
        .unwrap_or_else(|| unreachable!())
        .serial();
    let gf = Date::from_serial(em_serial - 3);
    if date == gf {
        return false;
    }
    // Victoria Day (Monday before May 25)
    if m == Month::May && wd == Weekday::Monday && (18..=24).contains(&d) {
        return false;
    }
    // Canada Day (July 1, observed)
    if m == Month::July
        && ((d == 1)
            || (d == 2 && wd == Weekday::Monday)
            || (d == 3 && wd == Weekday::Monday))
    {
        return false;
    }
    // Civic Holiday (1st Monday of August)
    if m == Month::August && wd == Weekday::Monday && d <= 7 {
        return false;
    }
    // Labour Day (1st Monday of September)
    if m == Month::September && wd == Weekday::Monday && d <= 7 {
        return false;
    }
    // National Day for Truth and Reconciliation (Sep 30, since 2021)
    if y >= 2021 && m == Month::September && d == 30 {
        return false;
    }
    // Thanksgiving (2nd Monday of October)
    if m == Month::October && wd == Weekday::Monday && (8..=14).contains(&d) {
        return false;
    }
    // Remembrance Day (Nov 11)
    if m == Month::November && d == 11 {
        return false;
    }
    // Christmas (Dec 25, observed)
    if m == Month::December
        && ((d == 25)
            || (d == 26 && wd == Weekday::Monday)
            || (d == 24 && wd == Weekday::Friday))
    {
        return false;
    }
    // Boxing Day (Dec 26, observed)
    if m == Month::December
        && ((d == 26 && wd != Weekday::Monday) // Mon case handled above as Christmas obs
            || (d == 27 && matches!(wd, Weekday::Monday | Weekday::Tuesday))
            || (d == 28 && wd == Weekday::Monday))
    {
        return false;
    }

    true
}

// ============================================================================
//  Australia calendar implementation (Sydney / ASX)
// ============================================================================

fn australia_is_business_day(date: Date) -> bool {
    if is_weekend(date) {
        return false;
    }
    let d = date.day_of_month();
    let m = date.month();
    let y = date.year();
    let wd = date.weekday();

    // New Year's Day (observed)
    if m == Month::January && (d == 1 || (d == 2 && wd == Weekday::Monday)) {
        return false;
    }
    // Australia Day (Jan 26, observed)
    if m == Month::January
        && ((d == 26)
            || (d == 27 && wd == Weekday::Monday)
            || (d == 25 && wd == Weekday::Friday))
    {
        return false;
    }
    // Good Friday + Easter Saturday + Easter Monday
    let (em_month, em_day) = easter_monday(y);
    let em_serial = Date::from_ymd_opt(y, em_month, em_day)
        .unwrap_or_else(|| unreachable!())
        .serial();
    let gf = Date::from_serial(em_serial - 3);
    let es = Date::from_serial(em_serial - 2);
    let em = Date::from_serial(em_serial);
    if date == gf || date == es || date == em {
        return false;
    }
    // ANZAC Day (Apr 25)
    if m == Month::April && d == 25 {
        return false;
    }
    // Queen's Birthday (2nd Monday of June — varies by state, using NSW)
    if m == Month::June && wd == Weekday::Monday && (8..=14).contains(&d) {
        return false;
    }
    // Bank Holiday (1st Monday of August — some states)
    if m == Month::August && wd == Weekday::Monday && d <= 7 {
        return false;
    }
    // Christmas (Dec 25) + Boxing Day (Dec 26), observed
    if m == Month::December
        && ((d == 25 || d == 26)
            || (d == 27 && matches!(wd, Weekday::Monday | Weekday::Tuesday))
            || (d == 28 && matches!(wd, Weekday::Monday | Weekday::Tuesday)))
    {
        return false;
    }

    true
}

// ============================================================================
//  Brazil calendar implementation (BM&F Bovespa)
// ============================================================================

fn brazil_is_business_day(date: Date) -> bool {
    if is_weekend(date) {
        return false;
    }
    let d = date.day_of_month();
    let m = date.month();
    let y = date.year();

    // New Year's Day
    if m == Month::January && d == 1 {
        return false;
    }
    // Tiradentes Day (Apr 21)
    if m == Month::April && d == 21 {
        return false;
    }
    // Labour Day (May 1)
    if m == Month::May && d == 1 {
        return false;
    }
    // Independence Day (Sep 7)
    if m == Month::September && d == 7 {
        return false;
    }
    // Nossa Senhora Aparecida (Oct 12)
    if m == Month::October && d == 12 {
        return false;
    }
    // All Souls' Day (Nov 2)
    if m == Month::November && d == 2 {
        return false;
    }
    // Republic Day (Nov 15)
    if m == Month::November && d == 15 {
        return false;
    }
    // Christmas (Dec 25)
    if m == Month::December && d == 25 {
        return false;
    }

    // Carnival (47 and 48 days before Easter Sunday)
    // Easter Monday
    let (em_month, em_day) = easter_monday(y);
    let em_serial = Date::from_ymd_opt(y, em_month, em_day)
        .unwrap_or_else(|| unreachable!())
        .serial();
    let easter_sunday_serial = em_serial - 1;
    // Carnival Monday (48 days before Easter Sunday)
    let carnival_mon = Date::from_serial(easter_sunday_serial - 48);
    // Carnival Tuesday (47 days before Easter Sunday)
    let carnival_tue = Date::from_serial(easter_sunday_serial - 47);
    // Good Friday (2 days before Easter Sunday)
    let gf = Date::from_serial(easter_sunday_serial - 2);
    // Corpus Christi (60 days after Easter Sunday)
    let corpus_christi = Date::from_serial(easter_sunday_serial + 60);

    if date == carnival_mon || date == carnival_tue || date == gf || date == corpus_christi {
        return false;
    }

    true
}

// ============================================================================
//  Germany calendar implementation (Frankfurt / Eurex)
// ============================================================================

fn germany_is_business_day(date: Date) -> bool {
    if is_weekend(date) { return false; }
    let d = date.day_of_month();
    let m = date.month();
    let y = date.year();

    // New Year
    if m == Month::January && d == 1 { return false; }
    // Easter-based
    let (em_month, em_day) = easter_monday(y);
    let em_serial = Date::from_ymd_opt(y, em_month, em_day).unwrap_or_else(|| unreachable!()).serial();
    let gf = Date::from_serial(em_serial - 3);
    let em = Date::from_serial(em_serial);
    if date == gf || date == em { return false; }
    // Labour Day
    if m == Month::May && d == 1 { return false; }
    // German Unity Day (Oct 3)
    if m == Month::October && d == 3 { return false; }
    // Christmas Eve (half day, treated as holiday for settlement)
    if m == Month::December && d == 24 { return false; }
    // Christmas + St Stephen's
    if m == Month::December && (d == 25 || d == 26) { return false; }
    // New Year's Eve
    if m == Month::December && d == 31 { return false; }
    true
}

// ============================================================================
//  France calendar implementation (Paris / Euronext)
// ============================================================================

fn france_is_business_day(date: Date) -> bool {
    if is_weekend(date) { return false; }
    let d = date.day_of_month();
    let m = date.month();
    let y = date.year();

    // New Year
    if m == Month::January && d == 1 { return false; }
    // Easter-based
    let (em_month, em_day) = easter_monday(y);
    let em_serial = Date::from_ymd_opt(y, em_month, em_day).unwrap_or_else(|| unreachable!()).serial();
    let gf = Date::from_serial(em_serial - 3);
    let em = Date::from_serial(em_serial);
    if date == gf || date == em { return false; }
    // Labour Day (May 1)
    if m == Month::May && d == 1 { return false; }
    // Victory in Europe (May 8)
    if m == Month::May && d == 8 { return false; }
    // Ascension Thursday (39 days after Easter Sunday)
    let ascension = Date::from_serial(em_serial - 1 + 39);
    if date == ascension { return false; }
    // Whit Monday (49 days after Easter Sunday)
    let whit_monday = Date::from_serial(em_serial - 1 + 50);
    if date == whit_monday { return false; }
    // Bastille Day (Jul 14)
    if m == Month::July && d == 14 { return false; }
    // Assumption (Aug 15)
    if m == Month::August && d == 15 { return false; }
    // All Saints (Nov 1)
    if m == Month::November && d == 1 { return false; }
    // Armistice (Nov 11)
    if m == Month::November && d == 11 { return false; }
    // Christmas
    if m == Month::December && d == 25 { return false; }
    true
}

// ============================================================================
//  Italy calendar implementation (Borsa Italiana)
// ============================================================================

fn italy_is_business_day(date: Date) -> bool {
    if is_weekend(date) { return false; }
    let d = date.day_of_month();
    let m = date.month();
    let y = date.year();

    // New Year
    if m == Month::January && d == 1 { return false; }
    // Epiphany (Jan 6)
    if m == Month::January && d == 6 { return false; }
    // Easter-based
    let (em_month, em_day) = easter_monday(y);
    let em_serial = Date::from_ymd_opt(y, em_month, em_day).unwrap_or_else(|| unreachable!()).serial();
    let gf = Date::from_serial(em_serial - 3);
    let em = Date::from_serial(em_serial);
    if date == gf || date == em { return false; }
    // Liberation Day (Apr 25)
    if m == Month::April && d == 25 { return false; }
    // Labour Day (May 1)
    if m == Month::May && d == 1 { return false; }
    // Republic Day (Jun 2)
    if m == Month::June && d == 2 { return false; }
    // Assumption (Aug 15)
    if m == Month::August && d == 15 { return false; }
    // All Saints (Nov 1)
    if m == Month::November && d == 1 { return false; }
    // Immaculate Conception (Dec 8)
    if m == Month::December && d == 8 { return false; }
    // Christmas + St Stephen's
    if m == Month::December && (d == 25 || d == 26) { return false; }
    true
}

// ============================================================================
//  Switzerland calendar implementation (SIX Swiss Exchange)
// ============================================================================

fn switzerland_is_business_day(date: Date) -> bool {
    if is_weekend(date) { return false; }
    let d = date.day_of_month();
    let m = date.month();
    let y = date.year();

    // New Year
    if m == Month::January && d == 1 { return false; }
    // Berchtoldstag (Jan 2)
    if m == Month::January && d == 2 { return false; }
    // Easter-based
    let (em_month, em_day) = easter_monday(y);
    let em_serial = Date::from_ymd_opt(y, em_month, em_day).unwrap_or_else(|| unreachable!()).serial();
    let gf = Date::from_serial(em_serial - 3);
    let em = Date::from_serial(em_serial);
    if date == gf || date == em { return false; }
    // Ascension Thursday
    let ascension = Date::from_serial(em_serial - 1 + 39);
    if date == ascension { return false; }
    // Whit Monday
    let whit_monday = Date::from_serial(em_serial - 1 + 50);
    if date == whit_monday { return false; }
    // Labour Day (May 1)
    if m == Month::May && d == 1 { return false; }
    // National Day (Aug 1)
    if m == Month::August && d == 1 { return false; }
    // Christmas + St Stephen's
    if m == Month::December && (d == 25 || d == 26) { return false; }
    true
}

// ============================================================================
//  India calendar implementation (NSE / Settlement)
// ============================================================================

fn india_is_business_day(date: Date) -> bool {
    if is_weekend(date) { return false; }
    let d = date.day_of_month();
    let m = date.month();

    // Republic Day (Jan 26)
    if m == Month::January && d == 26 { return false; }
    // Independence Day (Aug 15)
    if m == Month::August && d == 15 { return false; }
    // Gandhi Jayanti (Oct 2)
    if m == Month::October && d == 2 { return false; }
    // Christmas
    if m == Month::December && d == 25 { return false; }
    // Many Indian holidays are lunar-based (Diwali, Holi, Eid, etc.)
    // and vary by year. We include the major fixed-date holidays.
    // Makar Sankranti (Jan 14/15 approx)
    if m == Month::January && d == 14 { return false; }
    // May Day
    if m == Month::May && d == 1 { return false; }
    true
}

// ============================================================================
//  South Korea calendar implementation (KRX)
// ============================================================================

fn south_korea_is_business_day(date: Date) -> bool {
    if is_weekend(date) { return false; }
    let d = date.day_of_month();
    let m = date.month();

    // New Year
    if m == Month::January && d == 1 { return false; }
    // Independence Movement Day (Mar 1)
    if m == Month::March && d == 1 { return false; }
    // Children's Day (May 5)
    if m == Month::May && d == 5 { return false; }
    // Memorial Day (Jun 6)
    if m == Month::June && d == 6 { return false; }
    // Liberation Day (Aug 15)
    if m == Month::August && d == 15 { return false; }
    // National Foundation Day (Oct 3)
    if m == Month::October && d == 3 { return false; }
    // Hangul Day (Oct 9)
    if m == Month::October && d == 9 { return false; }
    // Christmas
    if m == Month::December && d == 25 { return false; }
    // Lunar holidays (Seollal, Chuseok, Buddha's birthday) vary yearly
    // and are omitted here; would need a lunar calendar table.
    true
}

// ============================================================================
//  Hong Kong calendar implementation (HKEX)
// ============================================================================

fn hong_kong_is_business_day(date: Date) -> bool {
    if is_weekend(date) { return false; }
    let d = date.day_of_month();
    let m = date.month();
    let y = date.year();

    // New Year
    if m == Month::January && d == 1 { return false; }
    // Easter-based (Good Friday, Easter Saturday, Easter Monday)
    let (em_month, em_day) = easter_monday(y);
    let em_serial = Date::from_ymd_opt(y, em_month, em_day).unwrap_or_else(|| unreachable!()).serial();
    let gf = Date::from_serial(em_serial - 3);
    let es = Date::from_serial(em_serial - 2);
    let em = Date::from_serial(em_serial);
    if date == gf || date == es || date == em { return false; }
    // Labour Day (May 1)
    if m == Month::May && d == 1 { return false; }
    // SAR Establishment Day (Jul 1)
    if m == Month::July && d == 1 { return false; }
    // National Day (Oct 1)
    if m == Month::October && d == 1 { return false; }
    // Christmas + Boxing Day
    if m == Month::December && (d == 25 || d == 26) { return false; }
    // Lunar holidays omitted (Chinese New Year, Ching Ming, etc.)
    true
}

// ============================================================================
//  Singapore calendar implementation (SGX)
// ============================================================================

fn singapore_is_business_day(date: Date) -> bool {
    if is_weekend(date) { return false; }
    let d = date.day_of_month();
    let m = date.month();
    let y = date.year();

    // New Year
    if m == Month::January && d == 1 { return false; }
    // Easter-based (Good Friday)
    let (em_month, em_day) = easter_monday(y);
    let em_serial = Date::from_ymd_opt(y, em_month, em_day).unwrap_or_else(|| unreachable!()).serial();
    let gf = Date::from_serial(em_serial - 3);
    if date == gf { return false; }
    // Labour Day (May 1)
    if m == Month::May && d == 1 { return false; }
    // National Day (Aug 9)
    if m == Month::August && d == 9 { return false; }
    // Christmas
    if m == Month::December && d == 25 { return false; }
    // Lunar holidays (Chinese New Year, Vesak, Deepavali, Hari Raya) vary yearly
    true
}

// ============================================================================
//  Mexico calendar implementation (BMV)
// ============================================================================

fn mexico_is_business_day(date: Date) -> bool {
    if is_weekend(date) { return false; }
    let d = date.day_of_month();
    let m = date.month();
    let y = date.year();
    let wd = date.weekday();

    // New Year
    if m == Month::January && d == 1 { return false; }
    // Constitution Day (1st Monday of February)
    if m == Month::February && wd == Weekday::Monday && d <= 7 { return false; }
    // Benito Juárez's Birthday (3rd Monday of March)
    if m == Month::March && wd == Weekday::Monday && (15..=21).contains(&d) { return false; }
    // Easter (Holy Thursday + Good Friday)
    let (em_month, em_day) = easter_monday(y);
    let em_serial = Date::from_ymd_opt(y, em_month, em_day).unwrap_or_else(|| unreachable!()).serial();
    let holy_thu = Date::from_serial(em_serial - 4);
    let gf = Date::from_serial(em_serial - 3);
    if date == holy_thu || date == gf { return false; }
    // Labour Day (May 1)
    if m == Month::May && d == 1 { return false; }
    // Independence Day (Sep 16)
    if m == Month::September && d == 16 { return false; }
    // Revolution Day (3rd Monday of November)
    if m == Month::November && wd == Weekday::Monday && (15..=21).contains(&d) { return false; }
    // Christmas
    if m == Month::December && d == 25 { return false; }
    true
}

// ============================================================================
//  South Africa calendar implementation (JSE)
// ============================================================================

fn south_africa_is_business_day(date: Date) -> bool {
    if is_weekend(date) { return false; }
    let d = date.day_of_month();
    let m = date.month();
    let y = date.year();
    let wd = date.weekday();

    // New Year
    if m == Month::January && d == 1 { return false; }
    // Human Rights Day (Mar 21)
    if m == Month::March && d == 21 { return false; }
    // Easter-based (Good Friday + Family Day = Easter Monday)
    let (em_month, em_day) = easter_monday(y);
    let em_serial = Date::from_ymd_opt(y, em_month, em_day).unwrap_or_else(|| unreachable!()).serial();
    let gf = Date::from_serial(em_serial - 3);
    let em = Date::from_serial(em_serial);
    if date == gf || date == em { return false; }
    // Freedom Day (Apr 27)
    if m == Month::April && d == 27 { return false; }
    // Workers' Day (May 1)
    if m == Month::May && d == 1 { return false; }
    // Youth Day (Jun 16)
    if m == Month::June && d == 16 { return false; }
    // National Women's Day (Aug 9)
    if m == Month::August && d == 9 { return false; }
    // Heritage Day (Sep 24)
    if m == Month::September && d == 24 { return false; }
    // Day of Reconciliation (Dec 16)
    if m == Month::December && d == 16 { return false; }
    // Christmas + Day of Goodwill
    if m == Month::December && (d == 25 || d == 26) { return false; }

    // Observed holiday rule: when a public holiday falls on Sunday,
    // the following Monday is observed (Public Holidays Act).
    if wd == Weekday::Monday {
        let sunday = date - 1;
        if south_africa_is_fixed_holiday(sunday) {
            return false;
        }
    }

    true
}

/// Check if the given date is one of South Africa's fixed public holidays.
fn south_africa_is_fixed_holiday(date: Date) -> bool {
    let d = date.day_of_month();
    let m = date.month();
    let y = date.year();
    if m == Month::January && d == 1 { return true; }
    if m == Month::March && d == 21 { return true; }
    if m == Month::April && d == 27 { return true; }
    if m == Month::May && d == 1 { return true; }
    if m == Month::June && d == 16 { return true; }
    if m == Month::August && d == 9 { return true; }
    if m == Month::September && d == 24 { return true; }
    if m == Month::December && d == 16 { return true; }
    if m == Month::December && (d == 25 || d == 26) { return true; }
    // Easter-based holidays (Good Friday / Family Day)
    let (em_month, em_day) = easter_monday(y);
    let em_serial = Date::from_ymd_opt(y, em_month, em_day).unwrap_or_else(|| unreachable!()).serial();
    let gf = Date::from_serial(em_serial - 3);
    let em = Date::from_serial(em_serial);
    if date == gf || date == em { return true; }
    false
}

// ============================================================================
//  Sweden calendar implementation (OMX Stockholm)
// ============================================================================

fn sweden_is_business_day(date: Date) -> bool {
    if is_weekend(date) { return false; }
    let d = date.day_of_month();
    let m = date.month();
    let y = date.year();

    // New Year
    if m == Month::January && d == 1 { return false; }
    // Epiphany (Jan 6)
    if m == Month::January && d == 6 { return false; }
    // Easter-based
    let (em_month, em_day) = easter_monday(y);
    let em_serial = Date::from_ymd_opt(y, em_month, em_day).unwrap_or_else(|| unreachable!()).serial();
    let gf = Date::from_serial(em_serial - 3);
    let em = Date::from_serial(em_serial);
    if date == gf || date == em { return false; }
    // Ascension Thursday
    let ascension = Date::from_serial(em_serial - 1 + 39);
    if date == ascension { return false; }
    // May Day
    if m == Month::May && d == 1 { return false; }
    // National Day (Jun 6)
    if m == Month::June && d == 6 { return false; }
    // Midsummer Eve (Friday between Jun 19–25)
    if m == Month::June && date.weekday() == Weekday::Friday && (19..=25).contains(&d) { return false; }
    // Christmas Eve + Christmas + Boxing Day
    if m == Month::December && (d == 24 || d == 25 || d == 26) { return false; }
    // New Year's Eve
    if m == Month::December && d == 31 { return false; }
    true
}

// ============================================================================
//  Denmark calendar implementation (OMX Copenhagen)
// ============================================================================

fn denmark_is_business_day(date: Date) -> bool {
    if is_weekend(date) { return false; }
    let d = date.day_of_month();
    let m = date.month();
    let y = date.year();

    // New Year
    if m == Month::January && d == 1 { return false; }
    // Easter-based (Holy Thursday, Good Friday, Easter Monday)
    let (em_month, em_day) = easter_monday(y);
    let em_serial = Date::from_ymd_opt(y, em_month, em_day).unwrap_or_else(|| unreachable!()).serial();
    let holy_thu = Date::from_serial(em_serial - 4);
    let gf = Date::from_serial(em_serial - 3);
    let em = Date::from_serial(em_serial);
    if date == holy_thu || date == gf || date == em { return false; }
    // Great Prayer Day (4th Friday after Easter — 26 days after Easter Sunday)
    let prayer_day = Date::from_serial(em_serial - 1 + 26);
    if date == prayer_day { return false; }
    // Ascension Thursday
    let ascension = Date::from_serial(em_serial - 1 + 39);
    if date == ascension { return false; }
    // Whit Monday
    let whit_monday = Date::from_serial(em_serial - 1 + 50);
    if date == whit_monday { return false; }
    // Constitution Day (Jun 5)
    if m == Month::June && d == 5 { return false; }
    // Christmas Eve + Christmas + Boxing Day
    if m == Month::December && (d == 24 || d == 25 || d == 26) { return false; }
    true
}

// ============================================================================
//  Norway calendar implementation (Oslo Børs)
// ============================================================================

fn norway_is_business_day(date: Date) -> bool {
    if is_weekend(date) { return false; }
    let d = date.day_of_month();
    let m = date.month();
    let y = date.year();

    // New Year
    if m == Month::January && d == 1 { return false; }
    // Easter-based (Holy Thursday, Good Friday, Easter Monday)
    let (em_month, em_day) = easter_monday(y);
    let em_serial = Date::from_ymd_opt(y, em_month, em_day).unwrap_or_else(|| unreachable!()).serial();
    let holy_thu = Date::from_serial(em_serial - 4);
    let gf = Date::from_serial(em_serial - 3);
    let em = Date::from_serial(em_serial);
    if date == holy_thu || date == gf || date == em { return false; }
    // Ascension Thursday
    let ascension = Date::from_serial(em_serial - 1 + 39);
    if date == ascension { return false; }
    // Whit Monday
    let whit_monday = Date::from_serial(em_serial - 1 + 50);
    if date == whit_monday { return false; }
    // Labour Day (May 1)
    if m == Month::May && d == 1 { return false; }
    // Constitution Day (May 17)
    if m == Month::May && d == 17 { return false; }
    // Christmas + Boxing Day
    if m == Month::December && (d == 25 || d == 26) { return false; }
    true
}

// ============================================================================
//  Poland calendar implementation (Warsaw Stock Exchange — GPW)
// ============================================================================

fn poland_is_business_day(date: Date) -> bool {
    if is_weekend(date) { return false; }
    let d = date.day_of_month();
    let m = date.month();
    let y = date.year();

    // New Year
    if m == Month::January && d == 1 { return false; }
    // Epiphany (Jan 6)
    if m == Month::January && d == 6 { return false; }
    // Easter-based (Easter Monday)
    let (em_month, em_day) = easter_monday(y);
    let em_serial = Date::from_ymd_opt(y, em_month, em_day).unwrap_or_else(|| unreachable!()).serial();
    let em = Date::from_serial(em_serial);
    if date == em { return false; }
    // Corpus Christi (60 days after Easter Sunday)
    let corpus_christi = Date::from_serial(em_serial - 1 + 60);
    if date == corpus_christi { return false; }
    // Labour Day (May 1)
    if m == Month::May && d == 1 { return false; }
    // Constitution Day (May 3)
    if m == Month::May && d == 3 { return false; }
    // Assumption (Aug 15)
    if m == Month::August && d == 15 { return false; }
    // All Saints (Nov 1)
    if m == Month::November && d == 1 { return false; }
    // Independence Day (Nov 11)
    if m == Month::November && d == 11 { return false; }
    // Christmas + Boxing Day
    if m == Month::December && (d == 25 || d == 26) { return false; }
    true
}

// ============================================================================
//  New Zealand calendar implementation (NZX)
// ============================================================================

fn new_zealand_is_business_day(date: Date) -> bool {
    if is_weekend(date) { return false; }
    let d = date.day_of_month();
    let m = date.month();
    let y = date.year();
    let wd = date.weekday();

    // New Year (Jan 1) + Day after (Jan 2), observed
    if m == Month::January && (d == 1 || d == 2
        || (d == 3 && matches!(wd, Weekday::Monday | Weekday::Tuesday))
        || (d == 4 && wd == Weekday::Monday))
    { return false; }
    // Waitangi Day (Feb 6, observed)
    if m == Month::February && ((d == 6) || (d == 7 && wd == Weekday::Monday)) { return false; }
    // Easter-based (Good Friday + Easter Monday)
    let (em_month, em_day) = easter_monday(y);
    let em_serial = Date::from_ymd_opt(y, em_month, em_day).unwrap_or_else(|| unreachable!()).serial();
    let gf = Date::from_serial(em_serial - 3);
    let em = Date::from_serial(em_serial);
    if date == gf || date == em { return false; }
    // ANZAC Day (Apr 25)
    if m == Month::April && d == 25 { return false; }
    // King's Birthday (1st Monday of June)
    if m == Month::June && wd == Weekday::Monday && d <= 7 { return false; }
    // Matariki (varies, omitted — added as fixed holiday from 2022)
    // Labour Day (4th Monday of October)
    if m == Month::October && wd == Weekday::Monday && (22..=28).contains(&d) { return false; }
    // Christmas + Boxing Day (observed)
    if m == Month::December
        && ((d == 25 || d == 26)
            || (d == 27 && matches!(wd, Weekday::Monday | Weekday::Tuesday))
            || (d == 28 && matches!(wd, Weekday::Monday | Weekday::Tuesday)))
    { return false; }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- NullCalendar ----

    #[test]
    fn null_calendar_every_day_is_business_day() {
        let cal = Calendar::NullCalendar;
        // Test a Saturday
        let sat = Date::from_ymd(2025, Month::June, 14);
        assert_eq!(sat.weekday(), Weekday::Saturday);
        assert!(cal.is_business_day(sat));
    }

    // ---- WeekendsOnly ----

    #[test]
    fn weekends_only() {
        let cal = Calendar::WeekendsOnly;
        let fri = Date::from_ymd(2025, Month::June, 13);
        let sat = Date::from_ymd(2025, Month::June, 14);
        let sun = Date::from_ymd(2025, Month::June, 15);
        let mon = Date::from_ymd(2025, Month::June, 16);

        assert!(cal.is_business_day(fri));
        assert!(!cal.is_business_day(sat));
        assert!(!cal.is_business_day(sun));
        assert!(cal.is_business_day(mon));
    }

    // ---- TARGET ----

    #[test]
    fn target_new_years_day() {
        let cal = Calendar::Target;
        let d = Date::from_ymd(2025, Month::January, 1);
        assert!(!cal.is_business_day(d));
    }

    #[test]
    fn target_christmas() {
        let cal = Calendar::Target;
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::December, 25)));
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::December, 26)));
    }

    #[test]
    fn target_good_friday_2025() {
        // Good Friday 2025 is April 18
        let cal = Calendar::Target;
        let gf = Date::from_ymd(2025, Month::April, 18);
        assert!(!cal.is_business_day(gf));
    }

    #[test]
    fn target_easter_monday_2025() {
        // Easter Monday 2025 is April 21
        let cal = Calendar::Target;
        let em = Date::from_ymd(2025, Month::April, 21);
        assert!(!cal.is_business_day(em));
    }

    #[test]
    fn target_regular_business_day() {
        let cal = Calendar::Target;
        let d = Date::from_ymd(2025, Month::June, 16); // Monday
        assert!(cal.is_business_day(d));
    }

    // ---- US Settlement ----

    #[test]
    fn us_independence_day() {
        let cal = Calendar::UnitedStates(USMarket::Settlement);
        // July 4, 2025 is a Friday
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::July, 4)));
    }

    #[test]
    fn us_thanksgiving_2025() {
        let cal = Calendar::UnitedStates(USMarket::Settlement);
        // Thanksgiving 2025: 4th Thursday of November = November 27
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::November, 27)));
    }

    #[test]
    fn us_christmas_observed() {
        let cal = Calendar::UnitedStates(USMarket::Settlement);
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::December, 25)));
    }

    // ---- Adjust ----

    #[test]
    fn adjust_following() {
        let cal = Calendar::Target;
        // Saturday June 14 2025 → Monday June 16
        let sat = Date::from_ymd(2025, Month::June, 14);
        let adj = cal.adjust(sat, BusinessDayConvention::Following);
        assert_eq!(adj, Date::from_ymd(2025, Month::June, 16));
    }

    #[test]
    fn adjust_preceding() {
        let cal = Calendar::Target;
        let sat = Date::from_ymd(2025, Month::June, 14);
        let adj = cal.adjust(sat, BusinessDayConvention::Preceding);
        assert_eq!(adj, Date::from_ymd(2025, Month::June, 13));
    }

    #[test]
    fn adjust_modified_following_stays_in_month() {
        let cal = Calendar::WeekendsOnly;
        // Saturday May 31, 2025 — Following would go to June 2
        // ModifiedFollowing should go back to Friday May 30
        let d = Date::from_ymd(2025, Month::May, 31);
        let adj = cal.adjust(d, BusinessDayConvention::ModifiedFollowing);
        assert_eq!(adj.month(), Month::May);
        assert!(cal.is_business_day(adj));
    }

    #[test]
    fn adjust_unadjusted() {
        let cal = Calendar::Target;
        let sat = Date::from_ymd(2025, Month::June, 14);
        let adj = cal.adjust(sat, BusinessDayConvention::Unadjusted);
        assert_eq!(adj, sat);
    }

    // ---- Advance ----

    #[test]
    fn advance_business_days() {
        let cal = Calendar::WeekendsOnly;
        let fri = Date::from_ymd(2025, Month::June, 13); // Friday
        let result = cal.advance(fri, Period::days(1), BusinessDayConvention::Following, false);
        // Next business day after Friday is Monday
        assert_eq!(result, Date::from_ymd(2025, Month::June, 16));
    }

    #[test]
    fn advance_months() {
        let cal = Calendar::Target;
        let d = Date::from_ymd(2025, Month::January, 15);
        let result = cal.advance(d, Period::months(3), BusinessDayConvention::ModifiedFollowing, false);
        // April 15, 2025 is a Tuesday — business day
        assert_eq!(result, Date::from_ymd(2025, Month::April, 15));
    }

    // ---- Business days between ----

    #[test]
    fn business_days_between() {
        let cal = Calendar::WeekendsOnly;
        let mon = Date::from_ymd(2025, Month::June, 16);
        let fri = Date::from_ymd(2025, Month::June, 20);
        // Tue, Wed, Thu, Fri = 4 business days
        assert_eq!(cal.business_days_between(mon, fri), 4);
    }

    // ---- Japan ----

    #[test]
    fn japan_new_years() {
        let cal = Calendar::Japan;
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::January, 1)));
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::January, 2)));
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::January, 3)));
    }

    #[test]
    fn japan_showa_day() {
        let cal = Calendar::Japan;
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::April, 29)));
    }

    #[test]
    fn japan_regular_business_day() {
        let cal = Calendar::Japan;
        // 2025-06-16 is Monday, not a holiday
        assert!(cal.is_business_day(Date::from_ymd(2025, Month::June, 16)));
    }

    // ---- China ----

    #[test]
    fn china_national_day() {
        let cal = Calendar::China;
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::October, 1)));
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::October, 2)));
    }

    #[test]
    fn china_labour_day() {
        let cal = Calendar::China;
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::May, 1)));
    }

    // ---- Canada ----

    #[test]
    fn canada_canada_day() {
        let cal = Calendar::Canada;
        // July 1, 2025 is Tuesday
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::July, 1)));
    }

    #[test]
    fn canada_good_friday_2025() {
        let cal = Calendar::Canada;
        // Good Friday 2025 is April 18
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::April, 18)));
    }

    // ---- Australia ----

    #[test]
    fn australia_australia_day() {
        let cal = Calendar::Australia;
        // Jan 26 2025 is Sunday, observed Monday Jan 27
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::January, 27)));
    }

    #[test]
    fn australia_anzac_day() {
        let cal = Calendar::Australia;
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::April, 25)));
    }

    // ---- Brazil ----

    #[test]
    fn brazil_tiradentes() {
        let cal = Calendar::Brazil;
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::April, 21)));
    }

    #[test]
    fn brazil_independence_day() {
        let cal = Calendar::Brazil;
        // Sep 7 2025 is Sunday → not a business day anyway
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::September, 7)));
    }

    #[test]
    fn brazil_carnival_2025() {
        let cal = Calendar::Brazil;
        // Easter Sunday 2025 is April 20, so Easter Monday is April 21
        // Carnival Monday = Easter Sunday - 48 = March 3
        // Carnival Tuesday = Easter Sunday - 47 = March 4
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::March, 3)));
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::March, 4)));
    }

    #[test]
    fn brazil_regular_business_day() {
        let cal = Calendar::Brazil;
        // 2025-06-16 Monday
        assert!(cal.is_business_day(Date::from_ymd(2025, Month::June, 16)));
    }

    // ---- Germany ----

    #[test]
    fn germany_new_year() {
        let cal = Calendar::Germany;
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::January, 1)));
    }

    #[test]
    fn germany_unity_day() {
        let cal = Calendar::Germany;
        // Oct 3 2025 is Friday
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::October, 3)));
    }

    #[test]
    fn germany_good_friday_2025() {
        let cal = Calendar::Germany;
        // Good Friday 2025 is April 18
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::April, 18)));
    }

    #[test]
    fn germany_regular_day() {
        let cal = Calendar::Germany;
        // 2025-06-16 Monday — not a holiday
        assert!(cal.is_business_day(Date::from_ymd(2025, Month::June, 16)));
    }

    // ---- France ----

    #[test]
    fn france_bastille_day() {
        let cal = Calendar::France;
        // Jul 14 2025 is Monday
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::July, 14)));
    }

    #[test]
    fn france_ascension_2025() {
        let cal = Calendar::France;
        // Ascension 2025: Easter Sunday=April 20, +38 days = May 29 (Thursday)
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::May, 29)));
    }

    // ---- Italy ----

    #[test]
    fn italy_liberation_day() {
        let cal = Calendar::Italy;
        // Apr 25 2025 is Friday
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::April, 25)));
    }

    #[test]
    fn italy_republic_day() {
        let cal = Calendar::Italy;
        // Jun 2 2025 is Monday
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::June, 2)));
    }

    // ---- Switzerland ----

    #[test]
    fn switzerland_national_day() {
        let cal = Calendar::Switzerland;
        // Aug 1 2025 is Friday
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::August, 1)));
    }

    #[test]
    fn switzerland_berchtoldstag() {
        let cal = Calendar::Switzerland;
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::January, 2)));
    }

    // ---- India ----

    #[test]
    fn india_republic_day() {
        let cal = Calendar::India;
        // Jan 26 2025 is Sunday; check Monday next year
        assert!(!cal.is_business_day(Date::from_ymd(2026, Month::January, 26)));
    }

    #[test]
    fn india_independence_day() {
        let cal = Calendar::India;
        // Aug 15 2025 is Friday
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::August, 15)));
    }

    // ---- South Korea ----

    #[test]
    fn south_korea_independence_movement() {
        let cal = Calendar::SouthKorea;
        // Mar 1 2027 is Monday
        assert!(!cal.is_business_day(Date::from_ymd(2027, Month::March, 1)));
    }

    #[test]
    fn south_korea_christmas() {
        let cal = Calendar::SouthKorea;
        // Dec 25 2025 is Thursday
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::December, 25)));
    }

    // ---- Hong Kong ----

    #[test]
    fn hong_kong_sar_day() {
        let cal = Calendar::HongKong;
        // Jul 1 2025 is Tuesday
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::July, 1)));
    }

    #[test]
    fn hong_kong_good_friday_2025() {
        let cal = Calendar::HongKong;
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::April, 18)));
    }

    // ---- Singapore ----

    #[test]
    fn singapore_national_day() {
        let cal = Calendar::Singapore;
        // Aug 9 2027 is Monday
        assert!(!cal.is_business_day(Date::from_ymd(2027, Month::August, 9)));
    }

    // ---- Mexico ----

    #[test]
    fn mexico_independence_day() {
        let cal = Calendar::Mexico;
        // Sep 16 2025 is Tuesday
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::September, 16)));
    }

    #[test]
    fn mexico_constitution_day_2025() {
        let cal = Calendar::Mexico;
        // 1st Monday of Feb 2025 = Feb 3
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::February, 3)));
    }

    // ---- South Africa ----

    #[test]
    fn south_africa_freedom_day() {
        let cal = Calendar::SouthAfrica;
        // Apr 27 2026 is Monday
        assert!(!cal.is_business_day(Date::from_ymd(2026, Month::April, 27)));
    }

    #[test]
    fn south_africa_human_rights_day() {
        let cal = Calendar::SouthAfrica;
        // Mar 21 2025 is Friday
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::March, 21)));
    }

    // ---- Sweden ----

    #[test]
    fn sweden_national_day() {
        let cal = Calendar::Sweden;
        // Jun 6 2025 is Friday
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::June, 6)));
    }

    #[test]
    fn sweden_midsummer_2025() {
        let cal = Calendar::Sweden;
        // Midsummer Eve 2025: Friday between Jun 19-25 → Jun 20
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::June, 20)));
    }

    // ---- Denmark ----

    #[test]
    fn denmark_constitution_day() {
        let cal = Calendar::Denmark;
        // Jun 5 2025 is Thursday
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::June, 5)));
    }

    #[test]
    fn denmark_holy_thursday_2025() {
        let cal = Calendar::Denmark;
        // Easter Monday 2025 = Apr 21, Holy Thursday = Apr 17
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::April, 17)));
    }

    // ---- Norway ----

    #[test]
    fn norway_constitution_day() {
        let cal = Calendar::Norway;
        // May 17 2027 is Monday
        assert!(!cal.is_business_day(Date::from_ymd(2027, Month::May, 17)));
    }

    #[test]
    fn norway_ascension_2025() {
        let cal = Calendar::Norway;
        // Ascension 2025: Easter Sunday=April 20, +38 = May 29 (Thursday)
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::May, 29)));
    }

    // ---- Poland ----

    #[test]
    fn poland_independence_day() {
        let cal = Calendar::Poland;
        // Nov 11 2025 is Tuesday
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::November, 11)));
    }

    #[test]
    fn poland_constitution_day() {
        let cal = Calendar::Poland;
        // May 3 2027 is Monday
        assert!(!cal.is_business_day(Date::from_ymd(2027, Month::May, 3)));
    }

    // ---- New Zealand ----

    #[test]
    fn new_zealand_waitangi_day() {
        let cal = Calendar::NewZealand;
        // Feb 6 2025 is Thursday
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::February, 6)));
    }

    #[test]
    fn new_zealand_anzac_day() {
        let cal = Calendar::NewZealand;
        // Apr 25 2025 is Friday
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::April, 25)));
    }

    #[test]
    fn new_zealand_kings_birthday_2025() {
        let cal = Calendar::NewZealand;
        // 1st Monday of June 2025 = Jun 2
        assert!(!cal.is_business_day(Date::from_ymd(2025, Month::June, 2)));
    }

    // ---- JointCalendar ----

    #[test]
    fn joint_holidays_both_must_be_biz_day() {
        // UK + Germany: intersection of business days
        let joint = Calendar::join_holidays(Calendar::UnitedKingdom, Calendar::Germany);
        // German Unity Day Oct 3, 2025 (Friday): UK biz day, DE holiday → not biz day
        assert!(!joint.is_business_day(Date::from_ymd(2025, Month::October, 3)));
        // UK August bank holiday: last Monday Aug → Aug 25, 2025: UK holiday, DE biz day → not biz day
        assert!(!joint.is_business_day(Date::from_ymd(2025, Month::August, 25)));
        // Regular day both open: Jun 16 2025 Monday
        assert!(joint.is_business_day(Date::from_ymd(2025, Month::June, 16)));
    }

    #[test]
    fn joint_business_days_either_is_biz_day() {
        // UK + Germany: union of business days
        let joint = Calendar::join_business_days(Calendar::UnitedKingdom, Calendar::Germany);
        // German Unity Day Oct 3, 2025: UK open → biz day
        assert!(joint.is_business_day(Date::from_ymd(2025, Month::October, 3)));
        // Christmas Dec 25: both closed → not biz day
        assert!(!joint.is_business_day(Date::from_ymd(2025, Month::December, 25)));
    }
}
