//! Schedule generation for coupon / cash-flow dates.
//!
//! A `Schedule` is an ordered list of dates, typically representing coupon
//! periods. It is built using the `ScheduleBuilder` (builder pattern).

use crate::business_day_convention::BusinessDayConvention;
use crate::calendar::Calendar;
use crate::date::Date;
use crate::period::{Frequency, Period, TimeUnit};
use ql_core::{QLError, QLResult};

/// Date-generation rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DateGenerationRule {
    /// Date sequence goes forward from the effective date.
    Forward,
    /// Date sequence goes backward from the maturity date.
    Backward,
    /// No intermediate dates.
    Zero,
    /// CDS standard rule (based on IMM dates).
    CDS,
}

/// An ordered schedule of dates.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(dead_code)]
pub struct Schedule {
    dates: Vec<Date>,
    calendar: Calendar,
    convention: BusinessDayConvention,
    termination_convention: BusinessDayConvention,
    frequency: Frequency,
    rule: DateGenerationRule,
    end_of_month: bool,
}

impl Schedule {
    /// Create a `ScheduleBuilder`.
    pub fn builder() -> ScheduleBuilder {
        ScheduleBuilder::new()
    }

    /// Create a schedule from an explicit list of dates.
    pub fn from_dates(dates: Vec<Date>) -> Self {
        Schedule {
            dates,
            calendar: Calendar::NullCalendar,
            convention: BusinessDayConvention::Unadjusted,
            termination_convention: BusinessDayConvention::Unadjusted,
            frequency: Frequency::NoFrequency,
            rule: DateGenerationRule::Forward,
            end_of_month: false,
        }
    }

    /// Number of dates in the schedule.
    pub fn len(&self) -> usize {
        self.dates.len()
    }

    /// Whether the schedule is empty.
    pub fn is_empty(&self) -> bool {
        self.dates.is_empty()
    }

    /// Access individual dates by index.
    pub fn date(&self, i: usize) -> Date {
        self.dates[i]
    }

    /// All dates as a slice.
    pub fn dates(&self) -> &[Date] {
        &self.dates
    }

    /// The calendar used.
    pub fn calendar(&self) -> Calendar {
        self.calendar
    }

    /// The frequency.
    pub fn frequency(&self) -> Frequency {
        self.frequency
    }

    /// Iterator over the dates.
    pub fn iter(&self) -> impl Iterator<Item = &Date> {
        self.dates.iter()
    }
}

/// Builder for constructing a `Schedule`.
pub struct ScheduleBuilder {
    effective_date: Option<Date>,
    termination_date: Option<Date>,
    frequency: Frequency,
    calendar: Calendar,
    convention: BusinessDayConvention,
    termination_convention: Option<BusinessDayConvention>,
    rule: DateGenerationRule,
    end_of_month: bool,
    first_date: Option<Date>,
    next_to_last_date: Option<Date>,
}

impl ScheduleBuilder {
    /// New builder with defaults.
    pub fn new() -> Self {
        ScheduleBuilder {
            effective_date: None,
            termination_date: None,
            frequency: Frequency::NoFrequency,
            calendar: Calendar::NullCalendar,
            convention: BusinessDayConvention::Unadjusted,
            termination_convention: None,
            rule: DateGenerationRule::Forward,
            end_of_month: false,
            first_date: None,
            next_to_last_date: None,
        }
    }

    /// Set the effective (start) date.
    pub fn effective_date(mut self, date: Date) -> Self {
        self.effective_date = Some(date);
        self
    }

    /// Set the termination (end) date.
    pub fn termination_date(mut self, date: Date) -> Self {
        self.termination_date = Some(date);
        self
    }

    /// Set the coupon frequency.
    pub fn frequency(mut self, freq: Frequency) -> Self {
        self.frequency = freq;
        self
    }

    /// Set the business-day calendar.
    pub fn calendar(mut self, cal: Calendar) -> Self {
        self.calendar = cal;
        self
    }

    /// Set the business-day convention for intermediate dates.
    pub fn convention(mut self, conv: BusinessDayConvention) -> Self {
        self.convention = conv;
        self
    }

    /// Set the business-day convention for the termination date.
    pub fn termination_convention(mut self, conv: BusinessDayConvention) -> Self {
        self.termination_convention = Some(conv);
        self
    }

    /// Set the date-generation rule (Forward, Backward, etc.).
    pub fn rule(mut self, rule: DateGenerationRule) -> Self {
        self.rule = rule;
        self
    }

    /// Whether to adjust dates to end-of-month.
    pub fn end_of_month(mut self, eom: bool) -> Self {
        self.end_of_month = eom;
        self
    }

    /// Set an explicit first coupon date (short/long front stub).
    pub fn first_date(mut self, date: Date) -> Self {
        self.first_date = Some(date);
        self
    }

    /// Set an explicit next-to-last date (short/long back stub).
    pub fn next_to_last_date(mut self, date: Date) -> Self {
        self.next_to_last_date = Some(date);
        self
    }

    /// Build the schedule.
    ///
    /// # Errors
    /// Returns `QLError::InvalidArgument` if effective_date or termination_date
    /// is not set, or if effective_date ≥ termination_date.
    pub fn build(self) -> QLResult<Schedule> {
        let effective = self
            .effective_date
            .ok_or_else(|| QLError::InvalidArgument("ScheduleBuilder: effective_date is required".into()))?;
        let termination = self
            .termination_date
            .ok_or_else(|| QLError::InvalidArgument("ScheduleBuilder: termination_date is required".into()))?;
        let term_conv = self.termination_convention.unwrap_or(self.convention);

        if effective >= termination {
            return Err(QLError::InvalidArgument(
                "effective date must be before termination date".into(),
            ));
        }

        let dates = match self.rule {
            DateGenerationRule::Zero => {
                vec![effective, termination]
            }
            DateGenerationRule::Forward => self.generate_forward(effective, termination),
            DateGenerationRule::Backward => self.generate_backward(effective, termination),
            DateGenerationRule::CDS => {
                // Simplified CDS: same as backward
                self.generate_backward(effective, termination)
            }
        };

        // Adjust dates
        let mut adjusted = Vec::with_capacity(dates.len());
        for (i, &d) in dates.iter().enumerate() {
            if i == 0 {
                adjusted.push(self.calendar.adjust(d, self.convention));
            } else if i == dates.len() - 1 {
                adjusted.push(self.calendar.adjust(d, term_conv));
            } else {
                adjusted.push(self.calendar.adjust(d, self.convention));
            }
        }

        Ok(Schedule {
            dates: adjusted,
            calendar: self.calendar,
            convention: self.convention,
            termination_convention: term_conv,
            frequency: self.frequency,
            rule: self.rule,
            end_of_month: self.end_of_month,
        })
    }

    fn generate_forward(&self, effective: Date, termination: Date) -> Vec<Date> {
        let tenor = self.frequency.to_period();
        if tenor.length == 0 {
            return vec![effective, termination];
        }

        let mut dates = vec![effective];

        // Optional first date stub
        if let Some(first) = self.first_date {
            if first > effective && first < termination {
                dates.push(first);
            }
        }

        let seed = dates[dates.len() - 1];
        let mut periods = 1;
        loop {
            let next = add_period(seed, &tenor, periods, self.end_of_month);
            if next >= termination {
                break;
            }
            // Check against next_to_last_date
            if let Some(ntl) = self.next_to_last_date {
                if next >= ntl {
                    break;
                }
            }
            dates.push(next);
            periods += 1;
        }

        if let Some(ntl) = self.next_to_last_date {
            if ntl > dates[dates.len() - 1] && ntl < termination {
                dates.push(ntl);
            }
        }

        dates.push(termination);
        dates
    }

    fn generate_backward(&self, effective: Date, termination: Date) -> Vec<Date> {
        let tenor = self.frequency.to_period();
        if tenor.length == 0 {
            return vec![effective, termination];
        }

        let mut dates = vec![termination];

        // Optional next-to-last date stub
        if let Some(ntl) = self.next_to_last_date {
            if ntl > effective && ntl < termination {
                dates.push(ntl);
            }
        }

        let seed = dates[dates.len() - 1];
        let mut periods = 1;
        loop {
            let prev = sub_period(seed, &tenor, periods, self.end_of_month);
            if prev <= effective {
                break;
            }
            if let Some(first) = self.first_date {
                if prev <= first {
                    break;
                }
            }
            dates.push(prev);
            periods += 1;
        }

        if let Some(first) = self.first_date {
            if first > effective && first < dates[dates.len() - 1] {
                dates.push(first);
            }
        }

        dates.push(effective);
        dates.reverse();
        dates
    }
}

impl Default for ScheduleBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Add `n` periods to a date.
fn add_period(date: Date, tenor: &Period, n: i32, end_of_month: bool) -> Date {
    match tenor.unit {
        TimeUnit::Days => date + (tenor.length * n),
        TimeUnit::Weeks => date + (tenor.length * n * 7),
        TimeUnit::Months => {
            let raw = add_months_raw(date, tenor.length * n);
            if end_of_month {
                raw.end_of_month()
            } else {
                raw
            }
        }
        TimeUnit::Years => {
            let raw = add_months_raw(date, tenor.length * n * 12);
            if end_of_month {
                raw.end_of_month()
            } else {
                raw
            }
        }
    }
}

/// Subtract `n` periods from a date.
fn sub_period(date: Date, tenor: &Period, n: i32, end_of_month: bool) -> Date {
    match tenor.unit {
        TimeUnit::Days => date - (tenor.length * n),
        TimeUnit::Weeks => date - (tenor.length * n * 7),
        TimeUnit::Months => {
            let raw = add_months_raw(date, -(tenor.length * n));
            if end_of_month {
                raw.end_of_month()
            } else {
                raw
            }
        }
        TimeUnit::Years => {
            let raw = add_months_raw(date, -(tenor.length * n * 12));
            if end_of_month {
                raw.end_of_month()
            } else {
                raw
            }
        }
    }
}

/// Add `n` months to a date, capping day at end of month.
fn add_months_raw(date: Date, months: i32) -> Date {
    let (y, m, d) = (date.year(), date.month() as u32, date.day_of_month());
    let total = (y * 12 + m as i32 - 1) + months;
    let new_year = total.div_euclid(12);
    let new_month = (total.rem_euclid(12) + 1) as u32;
    let max_day = Date::days_in_month(new_year, new_month);
    // Date is computed from valid calendar arithmetic.
    Date::from_ymd_opt(new_year, new_month, d.min(max_day)).unwrap_or_else(|| unreachable!())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::date::Month;

    #[test]
    fn schedule_from_explicit_dates() {
        let d1 = Date::from_ymd(2025, Month::January, 15);
        let d2 = Date::from_ymd(2025, Month::April, 15);
        let d3 = Date::from_ymd(2025, Month::July, 15);
        let s = Schedule::from_dates(vec![d1, d2, d3]);
        assert_eq!(s.len(), 3);
        assert_eq!(s.date(0), d1);
        assert_eq!(s.date(2), d3);
    }

    #[test]
    fn schedule_forward_semiannual() {
        let s = Schedule::builder()
            .effective_date(Date::from_ymd(2025, Month::January, 15))
            .termination_date(Date::from_ymd(2027, Month::January, 15))
            .frequency(Frequency::Semiannual)
            .calendar(Calendar::NullCalendar)
            .convention(BusinessDayConvention::Unadjusted)
            .rule(DateGenerationRule::Forward)
            .build()
            .unwrap();
        assert_eq!(s.len(), 5);
        assert_eq!(s.date(0), Date::from_ymd(2025, Month::January, 15));
        assert_eq!(s.date(1), Date::from_ymd(2025, Month::July, 15));
        assert_eq!(s.date(2), Date::from_ymd(2026, Month::January, 15));
        assert_eq!(s.date(3), Date::from_ymd(2026, Month::July, 15));
        assert_eq!(s.date(4), Date::from_ymd(2027, Month::January, 15));
    }

    #[test]
    fn schedule_backward_quarterly() {
        let s = Schedule::builder()
            .effective_date(Date::from_ymd(2025, Month::January, 15))
            .termination_date(Date::from_ymd(2026, Month::January, 15))
            .frequency(Frequency::Quarterly)
            .calendar(Calendar::NullCalendar)
            .convention(BusinessDayConvention::Unadjusted)
            .rule(DateGenerationRule::Backward)
            .build()
            .unwrap();
        assert_eq!(s.len(), 5);
        assert_eq!(s.date(0), Date::from_ymd(2025, Month::January, 15));
        assert_eq!(s.date(4), Date::from_ymd(2026, Month::January, 15));
    }

    #[test]
    fn schedule_with_calendar_adjustment() {
        let s = Schedule::builder()
            .effective_date(Date::from_ymd(2025, Month::January, 1))
            .termination_date(Date::from_ymd(2025, Month::July, 1))
            .frequency(Frequency::Quarterly)
            .calendar(Calendar::Target)
            .convention(BusinessDayConvention::ModifiedFollowing)
            .rule(DateGenerationRule::Forward)
            .build()
            .unwrap();
        let cal = Calendar::Target;
        for d in s.dates() {
            assert!(
                cal.is_business_day(*d),
                "date {} should be a business day",
                d
            );
        }
    }

    #[test]
    fn schedule_zero_rule() {
        let s = Schedule::builder()
            .effective_date(Date::from_ymd(2025, Month::January, 15))
            .termination_date(Date::from_ymd(2030, Month::January, 15))
            .frequency(Frequency::Annual)
            .rule(DateGenerationRule::Zero)
            .build()
            .unwrap();
        assert_eq!(s.date(0), Date::from_ymd(2025, Month::January, 15));
        assert_eq!(s.date(1), Date::from_ymd(2030, Month::January, 15));
    }

    #[test]
    fn schedule_end_of_month() {
        let s = Schedule::builder()
            .effective_date(Date::from_ymd(2025, Month::January, 31))
            .termination_date(Date::from_ymd(2025, Month::July, 31))
            .frequency(Frequency::Monthly)
            .calendar(Calendar::NullCalendar)
            .convention(BusinessDayConvention::Unadjusted)
            .rule(DateGenerationRule::Forward)
            .end_of_month(true)
            .build()
            .unwrap();
        assert_eq!(s.date(1), Date::from_ymd(2025, Month::February, 28));
        // April end of month = 30
        assert_eq!(s.date(3), Date::from_ymd(2025, Month::April, 30));
    }
}
