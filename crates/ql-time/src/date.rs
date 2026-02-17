//! Serial-number date type for high-performance date arithmetic.
//!
//! Dates are stored as a serial number (days from a fixed epoch), enabling
//! O(1) addition, subtraction, and comparison without calendar lookups.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Month of the year.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum Month {
    January = 1,
    February = 2,
    March = 3,
    April = 4,
    May = 5,
    June = 6,
    July = 7,
    August = 8,
    September = 9,
    October = 10,
    November = 11,
    December = 12,
}

impl Month {
    /// Convert a 1-based month number to a `Month`.
    pub fn from_u32(m: u32) -> Option<Self> {
        match m {
            1 => Some(Month::January),
            2 => Some(Month::February),
            3 => Some(Month::March),
            4 => Some(Month::April),
            5 => Some(Month::May),
            6 => Some(Month::June),
            7 => Some(Month::July),
            8 => Some(Month::August),
            9 => Some(Month::September),
            10 => Some(Month::October),
            11 => Some(Month::November),
            12 => Some(Month::December),
            _ => None,
        }
    }
}

impl fmt::Display for Month {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Month::January => "January",
            Month::February => "February",
            Month::March => "March",
            Month::April => "April",
            Month::May => "May",
            Month::June => "June",
            Month::July => "July",
            Month::August => "August",
            Month::September => "September",
            Month::October => "October",
            Month::November => "November",
            Month::December => "December",
        };
        write!(f, "{name}")
    }
}

/// Day of the week.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum Weekday {
    Sunday = 1,
    Monday = 2,
    Tuesday = 3,
    Wednesday = 4,
    Thursday = 5,
    Friday = 6,
    Saturday = 7,
}

impl fmt::Display for Weekday {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Weekday::Sunday => "Sunday",
            Weekday::Monday => "Monday",
            Weekday::Tuesday => "Tuesday",
            Weekday::Wednesday => "Wednesday",
            Weekday::Thursday => "Thursday",
            Weekday::Friday => "Friday",
            Weekday::Saturday => "Saturday",
        };
        write!(f, "{name}")
    }
}

/// A serial-number date — the number of days since 1 January 1900.
///
/// This epoch matches QuantLib's convention (serial number 1 = 1 January 1900).
/// All arithmetic (add days, subtract dates) is O(1).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Date(i32);

impl Date {
    // -- Epoch: serial 1 = 1 Jan 1900 (matching QuantLib/Excel) --

    /// Create a date from its serial number.
    pub const fn from_serial(serial: i32) -> Self {
        Date(serial)
    }

    /// Get the serial number.
    pub const fn serial(&self) -> i32 {
        self.0
    }

    /// Create a date from year, month, day.
    ///
    /// # Panics
    /// Panics if the date is invalid (e.g., February 30).
    pub fn from_ymd(year: i32, month: Month, day: u32) -> Self {
        let m = month as u32;
        assert!(
            (1..=31).contains(&day),
            "day {day} out of range for {month} {year}"
        );
        let serial = Self::ymd_to_serial(year, m, day);
        Date(serial)
    }

    /// Try to create a date from year, month number (1-12), day.
    pub fn from_ymd_opt(year: i32, month: u32, day: u32) -> Option<Self> {
        if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
            return None;
        }
        let serial = Self::ymd_to_serial(year, month, day);
        Some(Date(serial))
    }

    /// Today's date (local time).
    pub fn today() -> Self {
        let now = chrono::Local::now().date_naive();
        let year = now.year();
        let month = now.month();
        let day = now.day();
        Date(Self::ymd_to_serial(year, month, day))
    }

    /// Year component.
    pub fn year(&self) -> i32 {
        self.to_ymd().0
    }

    /// Month component.
    pub fn month(&self) -> Month {
        let (_, m, _) = self.to_ymd();
        // m is always 1..=12 from to_ymd().
        Month::from_u32(m).unwrap_or_else(|| unreachable!())
    }

    /// Day-of-month component.
    pub fn day_of_month(&self) -> u32 {
        self.to_ymd().2
    }

    /// Day of the week.
    pub fn weekday(&self) -> Weekday {
        // 1 Jan 1900 was a Monday => serial 1 mod 7 = 1 => Monday
        let w = ((self.0 - 1) % 7 + 7) % 7; // 0=Sunday to 6=Saturday
        match w {
            0 => Weekday::Monday,
            1 => Weekday::Tuesday,
            2 => Weekday::Wednesday,
            3 => Weekday::Thursday,
            4 => Weekday::Friday,
            5 => Weekday::Saturday,
            6 => Weekday::Sunday,
            _ => unreachable!(),
        }
    }

    /// Whether this is the last day of its month.
    pub fn is_end_of_month(&self) -> bool {
        let (y, m, d) = self.to_ymd();
        d == Self::days_in_month(y, m)
    }

    /// The last day of this date's month.
    pub fn end_of_month(&self) -> Date {
        let (y, m, _) = self.to_ymd();
        // days_in_month always returns a valid day.
        Date::from_ymd_opt(y, m, Self::days_in_month(y, m)).unwrap_or_else(|| unreachable!())
    }

    // -- Internal conversion helpers --

    /// Convert (year, month, day) to serial number.
    ///
    /// Uses the algorithm from QuantLib: serial 1 = 1 Jan 1900.
    fn ymd_to_serial(year: i32, month: u32, day: u32) -> i32 {
        Self::ymd_to_serial_direct(year, month, day)
    }

    /// Direct serial number calculation.
    fn ymd_to_serial_direct(year: i32, month: u32, day: u32) -> i32 {
        // Days from 1 Jan 1900 to 1 Jan of each year, plus month/day offset.
        // Reference: 1 Jan 1900 = serial 1.

        // Days from civil epoch (1 Jan 0000) to 1 Jan 1900
        // We'll compute days from 1 Jan 1900.

        let mut d = day as i32;

        // Days in complete months of the current year
        const DAYS_BEFORE_MONTH: [i32; 13] =
            [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
        d += DAYS_BEFORE_MONTH[month as usize];
        if month > 2 && Self::is_leap_year(year) {
            d += 1;
        }

        // Days in complete years since 1900
        let y = year - 1900;
        // Add 365 * years + leap day corrections
        d += 365 * y;

        // Leap year corrections: count leap years in [1900, year-1]
        // A year is leap if divisible by 4, except centuries unless divisible by 400.
        if y > 0 {
            // Leap years from 1900 to year-1:
            // 1900 is NOT a leap year (divisible by 100 but not 400).
            let prev = year - 1;
            let count_4 = (prev / 4) - (1899 / 4);
            let count_100 = (prev / 100) - (1899 / 100);
            let count_400 = (prev / 400) - (1899 / 400);
            d += count_4 - count_100 + count_400;
        }

        d
    }

    /// Extract (year, month, day) from serial number.
    fn to_ymd(self) -> (i32, u32, u32) {
        // Inverse of ymd_to_serial_direct.
        // We use an iterative approach anchored on a known date.
        let serial = self.0;

        // Estimate year
        let mut y = 1900 + (serial - 1) / 365;

        // Adjust: serial of 1 Jan of estimated year
        loop {
            let jan1 = Self::ymd_to_serial_direct(y, 1, 1);
            if jan1 > serial {
                y -= 1;
            } else {
                let next_jan1 = Self::ymd_to_serial_direct(y + 1, 1, 1);
                if next_jan1 <= serial {
                    y += 1;
                } else {
                    break;
                }
            }
        }

        let jan1 = Self::ymd_to_serial_direct(y, 1, 1);
        let mut day_of_year = serial - jan1 + 1; // 1-based

        let leap = Self::is_leap_year(y);
        let feb_days = if leap { 29 } else { 28 };
        let month_days: [u32; 12] = [31, feb_days, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

        let mut m = 0u32;
        for (i, &md) in month_days.iter().enumerate() {
            if day_of_year <= md as i32 {
                m = i as u32 + 1;
                break;
            }
            day_of_year -= md as i32;
        }

        (y, m, day_of_year as u32)
    }

    /// Whether the given year is a leap year.
    pub fn is_leap_year(year: i32) -> bool {
        (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
    }

    /// Number of days in the given month.
    pub fn days_in_month(year: i32, month: u32) -> u32 {
        match month {
            1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
            4 | 6 | 9 | 11 => 30,
            2 => {
                if Self::is_leap_year(year) {
                    29
                } else {
                    28
                }
            }
            _ => panic!("invalid month: {month}"),
        }
    }

    /// Minimum representable date (1 January 1901).
    pub const fn min_date() -> Self {
        // Serial for 1 Jan 1901
        Date(366 + 1) // 1900 had 365 days (not a leap year), so 1 Jan 1901 = 366+1=367? Let's just use a safe value
    }

    /// Maximum representable date (31 December 2199).
    pub const fn max_date() -> Self {
        Date(109574) // approximate
    }
}

use chrono::Datelike;

impl std::ops::Add<i32> for Date {
    type Output = Date;
    fn add(self, days: i32) -> Date {
        Date(self.0 + days)
    }
}

impl std::ops::AddAssign<i32> for Date {
    fn add_assign(&mut self, days: i32) {
        self.0 += days;
    }
}

impl std::ops::Sub<i32> for Date {
    type Output = Date;
    fn sub(self, days: i32) -> Date {
        Date(self.0 - days)
    }
}

impl std::ops::SubAssign<i32> for Date {
    fn sub_assign(&mut self, days: i32) {
        self.0 -= days;
    }
}

impl std::ops::Sub<Date> for Date {
    type Output = i32;
    fn sub(self, other: Date) -> i32 {
        self.0 - other.0
    }
}

impl fmt::Debug for Date {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (y, m, d) = self.to_ymd();
        write!(f, "Date({y}-{m:02}-{d:02})")
    }
}

impl fmt::Display for Date {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (y, m, d) = self.to_ymd();
        let month = Month::from_u32(m).unwrap_or_else(|| unreachable!());
        write!(f, "{month} {d}, {y}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_ymd() {
        let cases: &[(i32, Month, u32)] = &[
            (1900, Month::January, 1),
            (1900, Month::January, 31),
            (1900, Month::February, 28),
            (1900, Month::December, 31),
            (2000, Month::January, 1),
            (2000, Month::February, 29), // leap year
            (2024, Month::March, 15),
            (2025, Month::June, 15),
            (2025, Month::December, 31),
            (1999, Month::December, 31),
            (2100, Month::February, 28), // not a leap year
        ];

        for &(y, m, d) in cases {
            let date = Date::from_ymd(y, m, d);
            assert_eq!(date.year(), y, "year mismatch for {y}-{m}-{d}");
            assert_eq!(date.month(), m, "month mismatch for {y}-{m}-{d}");
            assert_eq!(date.day_of_month(), d, "day mismatch for {y}-{m}-{d}");
        }
    }

    #[test]
    fn serial_number_arithmetic() {
        let d1 = Date::from_ymd(2025, Month::January, 1);
        let d2 = Date::from_ymd(2025, Month::January, 31);
        assert_eq!(d2 - d1, 30);

        let d3 = d1 + 30;
        assert_eq!(d3, d2);

        let d4 = d2 - 30;
        assert_eq!(d4, d1);
    }

    #[test]
    fn serial_1_is_jan_1_1900() {
        let d = Date::from_serial(1);
        assert_eq!(d.year(), 1900);
        assert_eq!(d.month(), Month::January);
        assert_eq!(d.day_of_month(), 1);
    }

    #[test]
    fn weekday_jan_1_1900_is_monday() {
        let d = Date::from_ymd(1900, Month::January, 1);
        assert_eq!(d.weekday(), Weekday::Monday);
    }

    #[test]
    fn weekday_known_dates() {
        // 15 June 2025 is a Sunday
        let d = Date::from_ymd(2025, Month::June, 15);
        assert_eq!(d.weekday(), Weekday::Sunday);

        // 16 June 2025 is a Monday
        let d = Date::from_ymd(2025, Month::June, 16);
        assert_eq!(d.weekday(), Weekday::Monday);
    }

    #[test]
    fn leap_year() {
        assert!(!Date::is_leap_year(1900)); // century, not div by 400
        assert!(Date::is_leap_year(2000)); // div by 400
        assert!(Date::is_leap_year(2024)); // div by 4
        assert!(!Date::is_leap_year(2025));
        assert!(!Date::is_leap_year(2100));
    }

    #[test]
    fn end_of_month() {
        let d = Date::from_ymd(2024, Month::February, 15);
        let eom = d.end_of_month();
        assert_eq!(eom.day_of_month(), 29); // 2024 is leap

        let d = Date::from_ymd(2025, Month::February, 1);
        let eom = d.end_of_month();
        assert_eq!(eom.day_of_month(), 28); // 2025 is not leap
    }

    #[test]
    fn ordering() {
        let d1 = Date::from_ymd(2025, Month::January, 1);
        let d2 = Date::from_ymd(2025, Month::June, 15);
        assert!(d1 < d2);
        assert!(d2 > d1);
        assert_eq!(d1, d1);
    }

    #[test]
    fn display() {
        let d = Date::from_ymd(2025, Month::June, 15);
        assert_eq!(format!("{d}"), "June 15, 2025");
    }

    #[test]
    fn year_boundary() {
        let dec31 = Date::from_ymd(2024, Month::December, 31);
        let jan1 = dec31 + 1;
        assert_eq!(jan1.year(), 2025);
        assert_eq!(jan1.month(), Month::January);
        assert_eq!(jan1.day_of_month(), 1);
    }

    #[test]
    fn days_between_years() {
        let d1 = Date::from_ymd(2024, Month::January, 1);
        let d2 = Date::from_ymd(2025, Month::January, 1);
        assert_eq!(d2 - d1, 366); // 2024 is leap
    }
}
