//! # ql-time
//!
//! Date arithmetic, calendars, day counters, schedules, and period types
//! for the ql-rust quantitative finance library.
//!
//! ## Overview
//!
//! | Module | Purpose |
//! |---|---|
//! | [`date`] | Serial-number [`Date`] type with O(1) arithmetic |
//! | [`calendar`] | Holiday calendars ([`Calendar`]: TARGET, US, UK, …) |
//! | [`day_counter`] | Day-count conventions ([`DayCounter`]: Act/360, 30/360, …) |
//! | [`schedule`] | Coupon date generation ([`Schedule`] with builder pattern) |
//! | [`period`] | Time periods ([`Period`]) and payment frequencies ([`Frequency`]) |
//! | [`business_day_convention`] | Date adjustment rules ([`BusinessDayConvention`]) |
//! | [`imm`] | IMM date utilities (third Wednesday of quarterly months) |
//!
//! ## Quick Start
//!
//! ```rust
//! use ql_time::{Date, Month, Calendar, DayCounter, Period, Frequency, Schedule,
//!              BusinessDayConvention};
//!
//! // Create dates and compute year fractions
//! let d1 = Date::from_ymd(2025, Month::January, 15);
//! let d2 = Date::from_ymd(2025, Month::July, 15);
//! let dc = DayCounter::Actual360;
//! let yf = dc.year_fraction(d1, d2);
//! assert!((yf - 181.0 / 360.0).abs() < 1e-10);
//!
//! // Adjust for holidays
//! let cal = Calendar::Target;
//! let adjusted = cal.adjust(d1, BusinessDayConvention::ModifiedFollowing);
//! assert!(cal.is_business_day(adjusted));
//!
//! // Build a schedule
//! let schedule = Schedule::builder()
//!     .effective_date(d1)
//!     .termination_date(d2)
//!     .frequency(Frequency::Quarterly)
//!     .calendar(Calendar::Target)
//!     .convention(BusinessDayConvention::ModifiedFollowing)
//!     .build()
//!     .unwrap();
//! assert!(schedule.len() >= 2);
//! ```
#![warn(missing_docs)]

pub mod business_day_convention;
pub mod calendar;
pub mod date;
pub mod day_counter;
pub mod imm;
pub mod period;
pub mod schedule;
pub mod time_extended;
pub mod cds_schedule;

pub use business_day_convention::BusinessDayConvention;
pub use calendar::{Calendar, JointRule, USMarket, UKMarket, JapanMarket, BrazilMarket};
pub use date::{Date, Month, Weekday};
pub use day_counter::DayCounter;
pub use period::{Frequency, Period, TimeUnit};
pub use schedule::Schedule;
pub use time_extended::{asx, ecb, thirty365_day_count, thirty365_year_fraction};
pub use cds_schedule::{
    Thirty360Extended, thirty360_extended_day_count, thirty360_extended_year_fraction,
    CdsDateRule, generate_cds_schedule, french_amortization,
    cds_upfront_to_spread, cds_spread_to_upfront,
};
