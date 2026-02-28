//! Extended time utilities — G90-G92 gap closures.
//!
//! - [`asx`] (G90) — Australian Securities Exchange quarterly dates
//! - [`ecb`] (G91) — European Central Bank reserve maintenance dates
//! - [`Thirty365`](DayCounter) (G92) — 30/365 day count convention (added as DayCounter variant)

use crate::date::{Date, Month, Weekday};

// ---------------------------------------------------------------------------
// G90: ASX dates — Australian Securities Exchange quarterly dates
// ---------------------------------------------------------------------------

/// ASX quarterly months (same as IMM: March, June, September, December).
/// ASX settlement dates are the second Friday of the contract month.
pub mod asx {
    use super::*;

    const ASX_MONTHS: [Month; 4] = [
        Month::March,
        Month::June,
        Month::September,
        Month::December,
    ];

    /// Check whether a date is an ASX date (2nd Friday of Mar/Jun/Sep/Dec).
    pub fn is_asx_date(date: Date) -> bool {
        let m = date.month();
        if !matches!(
            m,
            Month::March | Month::June | Month::September | Month::December
        ) {
            return false;
        }
        let d = date.day_of_month();
        // 2nd Friday falls between 8th and 14th
        if !(8..=14).contains(&d) {
            return false;
        }
        date.weekday() == Weekday::Friday
    }

    /// Return the next ASX date on or after the given date.
    pub fn next_asx_date(date: Date) -> Date {
        let y = date.year();
        let m = date.month() as u32;

        for &asx_month in &ASX_MONTHS {
            if (asx_month as u32) >= m {
                let candidate = second_friday(y, asx_month);
                if candidate >= date {
                    return candidate;
                }
            }
        }
        // All ASX dates this year have passed
        second_friday(y + 1, Month::March)
    }

    /// Return the previous ASX date strictly before the given date.
    pub fn prev_asx_date(date: Date) -> Date {
        let y = date.year();
        let m = date.month() as u32;

        for &asx_month in ASX_MONTHS.iter().rev() {
            if (asx_month as u32) <= m {
                let candidate = second_friday(y, asx_month);
                if candidate < date {
                    return candidate;
                }
            }
        }
        second_friday(y - 1, Month::December)
    }

    /// Return a two-character ASX code for the date (e.g., "H5" for March 2025).
    pub fn asx_code(date: Date) -> Option<String> {
        if !is_asx_date(date) {
            return None;
        }
        let month_code = match date.month() {
            Month::March => 'H',
            Month::June => 'M',
            Month::September => 'U',
            Month::December => 'Z',
            _ => unreachable!(),
        };
        let year_digit = (date.year() % 10) as u8;
        Some(format!("{}{}", month_code, year_digit))
    }

    /// Compute the second Friday of a given month/year.
    fn second_friday(year: i32, month: Month) -> Date {
        let first = Date::from_ymd(year, month, 1);
        let wd = first.weekday();
        let days_to_fri = (Weekday::Friday as i32 - wd as i32 + 7) % 7;
        let first_fri = first + days_to_fri;
        // Second Friday = first Friday + 7 days
        first_fri + 7
    }
}

// ---------------------------------------------------------------------------
// G91: ECB dates — European Central Bank reserve maintenance dates
// ---------------------------------------------------------------------------

/// ECB reserve maintenance period dates.
///
/// The ECB reserve maintenance period starts on the settlement day of
/// the main refinancing operation (MRO) following the Governing Council
/// meeting where interest rate decisions are taken.
///
/// In practice, ECB dates approximate the first Wednesday on or after
/// the last day of each month (for the "new" system post-2015).
/// This module provides utilities for working with ECB dates.
pub mod ecb {
    use super::*;

    /// Known ECB meeting dates for 2024-2026 (approximate, based on published schedule).
    /// In practice these are announced annually; this provides the algorithmic fallback.
    const ECB_MONTHS: [Month; 8] = [
        Month::January,
        Month::March,
        Month::April,
        Month::June,
        Month::July,
        Month::September,
        Month::October,
        Month::December,
    ];

    /// Check whether a date is an ECB date.
    ///
    /// Uses the heuristic that ECB dates are the first Thursday of
    /// ECB meeting months (approximate for algorithmic usage).
    pub fn is_ecb_date(date: Date) -> bool {
        let m = date.month();
        if !ECB_MONTHS.contains(&m) {
            return false;
        }
        let d = date.day_of_month();
        // First Thursday falls between 1st and 7th
        if !(1..=7).contains(&d) {
            return false;
        }
        date.weekday() == Weekday::Thursday
    }

    /// Return the next ECB date on or after the given date.
    pub fn next_ecb_date(date: Date) -> Date {
        let y = date.year();
        let m = date.month() as u32;

        // Search current year
        for &ecb_month in &ECB_MONTHS {
            if (ecb_month as u32) >= m {
                let candidate = first_thursday(y, ecb_month);
                if candidate >= date {
                    return candidate;
                }
            }
        }
        // Next year
        first_thursday(y + 1, ECB_MONTHS[0])
    }

    /// Return the previous ECB date strictly before the given date.
    pub fn prev_ecb_date(date: Date) -> Date {
        let y = date.year();
        let m = date.month() as u32;

        for &ecb_month in ECB_MONTHS.iter().rev() {
            if (ecb_month as u32) <= m {
                let candidate = first_thursday(y, ecb_month);
                if candidate < date {
                    return candidate;
                }
            }
        }
        first_thursday(y - 1, *ECB_MONTHS.last().unwrap())
    }

    /// List all ECB dates in a given year.
    pub fn ecb_dates_in_year(year: i32) -> Vec<Date> {
        ECB_MONTHS
            .iter()
            .map(|&m| first_thursday(year, m))
            .collect()
    }

    /// Compute the first Thursday of a given month/year.
    fn first_thursday(year: i32, month: Month) -> Date {
        let first = Date::from_ymd(year, month, 1);
        let wd = first.weekday();
        let days_to_thu = (Weekday::Thursday as i32 - wd as i32 + 7) % 7;
        first + days_to_thu
    }
}

// ---------------------------------------------------------------------------
// G92: Thirty365 day counter — 30/365 day count convention
// ---------------------------------------------------------------------------

/// Compute the 30/365 day count between two dates.
///
/// Uses the same 30-day-month convention as 30/360 Bond Basis,
/// but divides by 365 instead of 360.
///
/// This convention is used in some Scandinavian and other markets.
pub fn thirty365_day_count(d1: Date, d2: Date) -> i32 {
    let mut dd1 = d1.day_of_month() as i32;
    let mut dd2 = d2.day_of_month() as i32;
    let mm1 = d1.month() as i32;
    let mm2 = d2.month() as i32;
    let yy1 = d1.year();
    let yy2 = d2.year();

    // Bond Basis (ISDA) rule
    if dd1 == 31 {
        dd1 = 30;
    }
    if dd2 == 31 && dd1 >= 30 {
        dd2 = 30;
    }
    360 * (yy2 - yy1) + 30 * (mm2 - mm1) + (dd2 - dd1)
}

/// Compute the 30/365 year fraction between two dates.
///
/// Year fraction = thirty365_day_count(d1, d2) / 365.0
pub fn thirty365_year_fraction(d1: Date, d2: Date) -> f64 {
    thirty365_day_count(d1, d2) as f64 / 365.0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- ASX tests ----

    #[test]
    fn test_asx_second_friday() {
        // March 2025: 1st = Saturday → first Friday = March 7 → second Friday = March 14
        let d = asx::next_asx_date(Date::from_ymd(2025, Month::March, 1));
        assert_eq!(d.year(), 2025);
        assert_eq!(d.month(), Month::March);
        assert_eq!(d.weekday(), Weekday::Friday);
        assert!(d.day_of_month() >= 8 && d.day_of_month() <= 14);
    }

    #[test]
    fn test_asx_is_date() {
        // Find the actual ASX date for March 2025 and verify
        let d = asx::next_asx_date(Date::from_ymd(2025, Month::March, 1));
        assert!(asx::is_asx_date(d));
    }

    #[test]
    fn test_asx_is_not_date() {
        let d = Date::from_ymd(2025, Month::March, 1);
        assert!(!asx::is_asx_date(d));
    }

    #[test]
    fn test_asx_next_wraps_to_next_year() {
        // After December ASX date → should jump to March next year
        let d = Date::from_ymd(2025, Month::December, 31);
        let next = asx::next_asx_date(d);
        assert_eq!(next.year(), 2026);
        assert_eq!(next.month(), Month::March);
    }

    #[test]
    fn test_asx_prev() {
        let d = Date::from_ymd(2025, Month::July, 1);
        let prev = asx::prev_asx_date(d);
        assert_eq!(prev.month(), Month::June);
        assert_eq!(prev.weekday(), Weekday::Friday);
    }

    #[test]
    fn test_asx_code() {
        let d = asx::next_asx_date(Date::from_ymd(2025, Month::June, 1));
        let code = asx::asx_code(d);
        assert!(code.is_some());
        assert!(code.unwrap().starts_with('M'));
    }

    // ---- ECB tests ----

    #[test]
    fn test_ecb_is_date() {
        // Find an ECB date and verify
        let d = ecb::next_ecb_date(Date::from_ymd(2025, Month::January, 1));
        assert!(ecb::is_ecb_date(d));
        assert_eq!(d.weekday(), Weekday::Thursday);
    }

    #[test]
    fn test_ecb_next() {
        let d = Date::from_ymd(2025, Month::February, 1);
        let next = ecb::next_ecb_date(d);
        // February is not an ECB month, so should skip to March
        assert_eq!(next.month(), Month::March);
    }

    #[test]
    fn test_ecb_prev() {
        let d = Date::from_ymd(2025, Month::May, 1);
        let prev = ecb::prev_ecb_date(d);
        assert_eq!(prev.month(), Month::April);
        assert_eq!(prev.weekday(), Weekday::Thursday);
    }

    #[test]
    fn test_ecb_dates_in_year() {
        let dates = ecb::ecb_dates_in_year(2025);
        assert_eq!(dates.len(), 8);
        for d in &dates {
            assert_eq!(d.year(), 2025);
            assert_eq!(d.weekday(), Weekday::Thursday);
            assert!(ecb::is_ecb_date(*d));
        }
    }

    // ---- Thirty365 tests ----

    #[test]
    fn test_thirty365_same_date() {
        let d = Date::from_ymd(2025, Month::January, 15);
        assert_eq!(thirty365_day_count(d, d), 0);
        assert!((thirty365_year_fraction(d, d) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_thirty365_one_month() {
        let d1 = Date::from_ymd(2025, Month::January, 15);
        let d2 = Date::from_ymd(2025, Month::February, 15);
        let days = thirty365_day_count(d1, d2);
        assert_eq!(days, 30); // 30-day month convention
        let yf = thirty365_year_fraction(d1, d2);
        assert!((yf - 30.0 / 365.0).abs() < 1e-12);
    }

    #[test]
    fn test_thirty365_six_months() {
        let d1 = Date::from_ymd(2025, Month::January, 1);
        let d2 = Date::from_ymd(2025, Month::July, 1);
        let days = thirty365_day_count(d1, d2);
        assert_eq!(days, 180); // 6 × 30 = 180
        let yf = thirty365_year_fraction(d1, d2);
        assert!((yf - 180.0 / 365.0).abs() < 1e-12);
    }

    #[test]
    fn test_thirty365_one_year() {
        let d1 = Date::from_ymd(2025, Month::January, 1);
        let d2 = Date::from_ymd(2026, Month::January, 1);
        let days = thirty365_day_count(d1, d2);
        assert_eq!(days, 360); // 12 × 30 = 360
        let yf = thirty365_year_fraction(d1, d2);
        assert!((yf - 360.0 / 365.0).abs() < 1e-12);
    }

    #[test]
    fn test_thirty365_end_of_month_adjustment() {
        // Day 31 should be adjusted to 30
        let d1 = Date::from_ymd(2025, Month::January, 31);
        let d2 = Date::from_ymd(2025, Month::March, 31);
        let days = thirty365_day_count(d1, d2);
        assert_eq!(days, 60); // Both adjusted to 30: (Mar30 - Jan30) = 60
    }
}
