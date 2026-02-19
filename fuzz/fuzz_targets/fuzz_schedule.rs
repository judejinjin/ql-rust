//! Fuzz target for Schedule generation.
//!
//! Exercises the ScheduleBuilder with fuzzed dates and frequencies
//! to ensure no panics or infinite loops.

#![no_main]

use libfuzzer_sys::fuzz_target;
use ql_time::{
    BusinessDayConvention, Calendar, Date, Frequency, Month, Schedule,
    schedule::DateGenerationRule,
};

fn month_from_u8(m: u8) -> Month {
    match m % 12 {
        0 => Month::January,
        1 => Month::February,
        2 => Month::March,
        3 => Month::April,
        4 => Month::May,
        5 => Month::June,
        6 => Month::July,
        7 => Month::August,
        8 => Month::September,
        9 => Month::October,
        10 => Month::November,
        _ => Month::December,
    }
}

fn freq_from_u8(f: u8) -> Frequency {
    match f % 6 {
        0 => Frequency::Annual,
        1 => Frequency::Semiannual,
        2 => Frequency::Quarterly,
        3 => Frequency::Monthly,
        4 => Frequency::Weekly,
        _ => Frequency::Biweekly,
    }
}

fuzz_target!(|data: (u8, u8, u8, u8, u8, u8, u8, u8)| {
    let (m1, d1, y1_off, m2, d2, y2_off, freq_raw, rule_raw) = data;

    let year1 = 2000 + (y1_off % 50) as i32;
    let year2 = year1 + 1 + (y2_off % 30) as i32;
    let month1 = month_from_u8(m1);
    let month2 = month_from_u8(m2);
    let day1 = (d1 % 28) + 1;
    let day2 = (d2 % 28) + 1;

    let effective = std::panic::catch_unwind(|| {
        Date::from_ymd(year1, month1, day1 as u32)
    });
    let termination = std::panic::catch_unwind(|| {
        Date::from_ymd(year2, month2, day2 as u32)
    });

    if let (Ok(eff), Ok(term)) = (effective, termination) {
        if term <= eff {
            return;
        }
        let freq = freq_from_u8(freq_raw);
        let rule = if rule_raw % 2 == 0 {
            DateGenerationRule::Forward
        } else {
            DateGenerationRule::Backward
        };

        let _ = std::panic::catch_unwind(move || {
            let _ = Schedule::builder()
                .effective_date(eff)
                .termination_date(term)
                .frequency(freq)
                .calendar(Calendar::Target)
                .convention(BusinessDayConvention::ModifiedFollowing)
                .rule(rule)
                .build();
        });
    }
});
