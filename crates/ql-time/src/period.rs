//! Time periods and frequency definitions.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Units for a time period.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TimeUnit {
    Days,
    Weeks,
    Months,
    Years,
}

impl fmt::Display for TimeUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TimeUnit::Days => write!(f, "D"),
            TimeUnit::Weeks => write!(f, "W"),
            TimeUnit::Months => write!(f, "M"),
            TimeUnit::Years => write!(f, "Y"),
        }
    }
}

/// Frequency of events (coupon payments, compounding, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Frequency {
    /// No frequency — single payment (zero coupon).
    NoFrequency,
    /// Once (same as NoFrequency for most purposes).
    Once,
    /// Once per year.
    Annual,
    /// Twice per year.
    Semiannual,
    /// Every 4 months (3 times per year).
    EveryFourthMonth,
    /// Every 3 months (4 times per year).
    Quarterly,
    /// Every 2 months (6 times per year).
    Bimonthly,
    /// Every month (12 times per year).
    Monthly,
    /// Every 4 weeks (13 times per year).
    EveryFourthWeek,
    /// Every 2 weeks (26 times per year).
    Biweekly,
    /// Every week (52 times per year).
    Weekly,
    /// Every day (365 times per year).
    Daily,
    /// Custom / other frequency.
    OtherFrequency,
}

impl Frequency {
    /// Number of events per year, or 0 for `NoFrequency` / `Once`.
    pub fn events_per_year(&self) -> u32 {
        match self {
            Frequency::NoFrequency | Frequency::Once | Frequency::OtherFrequency => 0,
            Frequency::Annual => 1,
            Frequency::Semiannual => 2,
            Frequency::EveryFourthMonth => 3,
            Frequency::Quarterly => 4,
            Frequency::Bimonthly => 6,
            Frequency::Monthly => 12,
            Frequency::EveryFourthWeek => 13,
            Frequency::Biweekly => 26,
            Frequency::Weekly => 52,
            Frequency::Daily => 365,
        }
    }

    /// Convert a frequency to a period.
    pub fn to_period(&self) -> Period {
        match self {
            Frequency::Annual => Period::new(1, TimeUnit::Years),
            Frequency::Semiannual => Period::new(6, TimeUnit::Months),
            Frequency::EveryFourthMonth => Period::new(4, TimeUnit::Months),
            Frequency::Quarterly => Period::new(3, TimeUnit::Months),
            Frequency::Bimonthly => Period::new(2, TimeUnit::Months),
            Frequency::Monthly => Period::new(1, TimeUnit::Months),
            Frequency::EveryFourthWeek => Period::new(4, TimeUnit::Weeks),
            Frequency::Biweekly => Period::new(2, TimeUnit::Weeks),
            Frequency::Weekly => Period::new(1, TimeUnit::Weeks),
            Frequency::Daily => Period::new(1, TimeUnit::Days),
            Frequency::NoFrequency | Frequency::Once | Frequency::OtherFrequency => {
                Period::new(0, TimeUnit::Days)
            }
        }
    }
}

/// A time period (length + unit).
///
/// Used for tenors (e.g., "3M", "5Y") and schedule generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Period {
    /// The number of units.
    pub length: i32,
    /// The time unit.
    pub unit: TimeUnit,
}

impl Period {
    /// Create a new period.
    pub const fn new(length: i32, unit: TimeUnit) -> Self {
        Self { length, unit }
    }

    /// Convenience: period in days.
    pub const fn days(n: i32) -> Self {
        Self::new(n, TimeUnit::Days)
    }

    /// Convenience: period in weeks.
    pub const fn weeks(n: i32) -> Self {
        Self::new(n, TimeUnit::Weeks)
    }

    /// Convenience: period in months.
    pub const fn months(n: i32) -> Self {
        Self::new(n, TimeUnit::Months)
    }

    /// Convenience: period in years.
    pub const fn years(n: i32) -> Self {
        Self::new(n, TimeUnit::Years)
    }

    /// Negate the period.
    pub fn negate(&self) -> Self {
        Self {
            length: -self.length,
            unit: self.unit,
        }
    }
}

impl fmt::Display for Period {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.length, self.unit)
    }
}

impl std::ops::Neg for Period {
    type Output = Period;
    fn neg(self) -> Self::Output {
        self.negate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn period_display() {
        assert_eq!(Period::months(3).to_string(), "3M");
        assert_eq!(Period::years(5).to_string(), "5Y");
        assert_eq!(Period::days(30).to_string(), "30D");
        assert_eq!(Period::weeks(2).to_string(), "2W");
    }

    #[test]
    fn period_negate() {
        let p = Period::months(6);
        let neg = -p;
        assert_eq!(neg.length, -6);
        assert_eq!(neg.unit, TimeUnit::Months);
    }

    #[test]
    fn frequency_to_period() {
        assert_eq!(Frequency::Semiannual.to_period(), Period::months(6));
        assert_eq!(Frequency::Quarterly.to_period(), Period::months(3));
        assert_eq!(Frequency::Annual.to_period(), Period::years(1));
        assert_eq!(Frequency::Monthly.to_period(), Period::months(1));
    }

    #[test]
    fn frequency_events_per_year() {
        assert_eq!(Frequency::Annual.events_per_year(), 1);
        assert_eq!(Frequency::Semiannual.events_per_year(), 2);
        assert_eq!(Frequency::Quarterly.events_per_year(), 4);
        assert_eq!(Frequency::Monthly.events_per_year(), 12);
        assert_eq!(Frequency::Daily.events_per_year(), 365);
    }
}
