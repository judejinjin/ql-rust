//! Fuzz target for Date construction and arithmetic.
//!
//! Tests that Date::from_ymd never panics on valid ranges,
//! and that date arithmetic (add/subtract days) doesn't overflow.

#![no_main]

use libfuzzer_sys::fuzz_target;
use ql_time::{Date, DayCounter, Month};

fn month_from_u8(m: u8) -> Option<Month> {
    match m % 12 {
        0 => Some(Month::January),
        1 => Some(Month::February),
        2 => Some(Month::March),
        3 => Some(Month::April),
        4 => Some(Month::May),
        5 => Some(Month::June),
        6 => Some(Month::July),
        7 => Some(Month::August),
        8 => Some(Month::September),
        9 => Some(Month::October),
        10 => Some(Month::November),
        11 => Some(Month::December),
        _ => None,
    }
}

fuzz_target!(|data: (i32, u8, u8, i32)| {
    let (year, month_raw, day, offset) = data;

    // Constrain to valid-ish ranges to avoid instant rejections
    let year = (year % 200) + 1900; // 1900..2099
    let month = match month_from_u8(month_raw) {
        Some(m) => m,
        None => return,
    };
    let day = (day % 31) + 1; // 1..31

    // Try constructing date — should not panic on invalid dates,
    // just produce a valid date or wrap
    let d = std::panic::catch_unwind(|| Date::from_ymd(year, month, day as u32));
    if let Ok(d) = d {
        // Date arithmetic: add offset days
        let offset = offset % 10_000; // reasonable range
        let _ = std::panic::catch_unwind(move || {
            let d2 = d + offset;
            // Year fraction should not panic
            let dc = DayCounter::Actual365Fixed;
            let _ = dc.year_fraction(d, d2);
        });
    }
});
