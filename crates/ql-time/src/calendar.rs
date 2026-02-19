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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Calendar {
    /// No holidays — every day is a business day.
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
}

impl Calendar {
    /// Whether the given date is a business day.
    pub fn is_business_day(&self, date: Date) -> bool {
        match self {
            Calendar::NullCalendar => true,
            Calendar::WeekendsOnly => !is_weekend(date),
            Calendar::Target => target_is_business_day(date),
            Calendar::UnitedStates(market) => us_is_business_day(date, *market),
            Calendar::UnitedKingdom => uk_is_business_day(date),
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
}
