//! IMM (International Monetary Market) date utilities.
//!
//! IMM dates are the third Wednesday of March, June, September, and December.

use crate::date::{Date, Month, Weekday};

/// The four IMM months.
const IMM_MONTHS: [Month; 4] = [
    Month::March,
    Month::June,
    Month::September,
    Month::December,
];

/// Check whether a date is an IMM date (3rd Wednesday of Mar/Jun/Sep/Dec).
pub fn is_imm_date(date: Date) -> bool {
    let m = date.month();
    if !matches!(
        m,
        Month::March | Month::June | Month::September | Month::December
    ) {
        return false;
    }
    let d = date.day_of_month();
    // 3rd Wednesday falls between 15th and 21st
    if !(15..=21).contains(&d) {
        return false;
    }
    date.weekday() == Weekday::Wednesday
}

/// Return the next IMM date on or after the given date.
pub fn next_imm_date(date: Date) -> Date {
    let y = date.year();
    let m = date.month() as u32;

    for &imm_month in &IMM_MONTHS {
        if (imm_month as u32) >= m {
            let candidate = third_wednesday(y, imm_month);
            if candidate >= date {
                return candidate;
            }
        }
    }
    // All IMM dates this year are in the past — go to March next year
    third_wednesday(y + 1, Month::March)
}

/// Return the previous IMM date strictly before the given date.
pub fn prev_imm_date(date: Date) -> Date {
    let y = date.year();
    let m = date.month() as u32;

    for &imm_month in IMM_MONTHS.iter().rev() {
        if (imm_month as u32) <= m {
            let candidate = third_wednesday(y, imm_month);
            if candidate < date {
                return candidate;
            }
        }
    }
    // All IMM dates this year are in the future — go to December previous year
    third_wednesday(y - 1, Month::December)
}

/// Return a two-character IMM code for the date (e.g., "H5" for March 2025).
///
/// Returns `None` if the date is not an IMM date.
pub fn imm_code(date: Date) -> Option<String> {
    if !is_imm_date(date) {
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

/// Compute the third Wednesday of a given month/year.
fn third_wednesday(year: i32, month: Month) -> Date {
    // Start from the 1st of the month
    let first = Date::from_ymd(year, month, 1);
    let wd = first.weekday();
    // Days until first Wednesday
    let days_to_wed = (Weekday::Wednesday as i32 - wd as i32 + 7) % 7;
    // First Wednesday
    let first_wed = first + days_to_wed;
    // Third Wednesday = first Wednesday + 14 days
    first_wed + 14
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn third_wednesday_march_2025() {
        let d = third_wednesday(2025, Month::March);
        assert_eq!(d.year(), 2025);
        assert_eq!(d.month(), Month::March);
        assert_eq!(d.day_of_month(), 19);
        assert_eq!(d.weekday(), Weekday::Wednesday);
    }

    #[test]
    fn is_imm_date_positive() {
        // March 19, 2025 is the 3rd Wednesday of March
        let d = Date::from_ymd(2025, Month::March, 19);
        assert!(is_imm_date(d));
    }

    #[test]
    fn is_imm_date_negative() {
        // March 18, 2025 is a Tuesday
        let d = Date::from_ymd(2025, Month::March, 18);
        assert!(!is_imm_date(d));
        // January 15, 2025 — not an IMM month
        let d2 = Date::from_ymd(2025, Month::January, 15);
        assert!(!is_imm_date(d2));
    }

    #[test]
    fn next_imm_date_before_march() {
        let d = Date::from_ymd(2025, Month::January, 10);
        let next = next_imm_date(d);
        assert!(is_imm_date(next));
        assert_eq!(next.month(), Month::March);
        assert_eq!(next.year(), 2025);
    }

    #[test]
    fn next_imm_date_after_december() {
        let d = Date::from_ymd(2025, Month::December, 18);
        let imm_dec = third_wednesday(2025, Month::December);
        // If we're past or on the December IMM date, next should be March next year
        if d >= imm_dec {
            let next = next_imm_date(d + 1);
            assert_eq!(next.month(), Month::March);
            assert_eq!(next.year(), 2026);
        }
    }

    #[test]
    fn prev_imm_date_works() {
        let d = Date::from_ymd(2025, Month::April, 1);
        let prev = prev_imm_date(d);
        assert!(is_imm_date(prev));
        assert_eq!(prev.month(), Month::March);
        assert_eq!(prev.year(), 2025);
    }

    #[test]
    fn imm_code_march_2025() {
        let d = third_wednesday(2025, Month::March);
        assert_eq!(imm_code(d), Some("H5".to_string()));
    }

    #[test]
    fn imm_code_december_2029() {
        let d = third_wednesday(2029, Month::December);
        assert_eq!(imm_code(d), Some("Z9".to_string()));
    }

    #[test]
    fn imm_code_non_imm_date() {
        let d = Date::from_ymd(2025, Month::January, 10);
        assert_eq!(imm_code(d), None);
    }
}
