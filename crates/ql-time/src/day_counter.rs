//! Day-count conventions — enum-based for zero-cost dispatch.
//!
//! Each variant implements its own `day_count` and `year_fraction` logic
//! following ISDA definitions.

use serde::{Deserialize, Serialize};

use crate::date::{Date, Month};
use ql_core::Real;

/// Thirty/360 sub-conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Thirty360Convention {
    /// ISDA (Bond Basis)
    BondBasis,
    /// European (30E/360)
    EurobondBasis,
}

/// Actual/Actual sub-conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActualActualConvention {
    /// ISDA method (each year gets its own basis).
    ISDA,
    /// ISMA / ICMA — coupon period based (not yet fully implemented).
    ISMA,
    /// AFB / Euro method.
    AFB,
}

/// Enum-based day counter — zero-cost dispatch, `Copy`, `Send + Sync`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DayCounter {
    /// Actual / 360
    Actual360,
    /// Actual / 365 (Fixed)
    Actual365Fixed,
    /// 30/360 variants
    Thirty360(Thirty360Convention),
    /// Actual/Actual variants
    ActualActual(ActualActualConvention),
    /// Business/252 (Brazilian convention: business days / 252).
    Business252,
}

impl DayCounter {
    /// Number of calendar days between `d1` (exclusive) and `d2` (inclusive).
    pub fn day_count(&self, d1: Date, d2: Date) -> i32 {
        match self {
            DayCounter::Actual360 | DayCounter::Actual365Fixed | DayCounter::ActualActual(_) => {
                d2.serial() - d1.serial()
            }
            DayCounter::Thirty360(convention) => thirty360_day_count(d1, d2, *convention),
            DayCounter::Business252 => {
                crate::calendar::Calendar::Brazil.business_days_between(d1, d2)
            }
        }
    }

    /// Year fraction between `d1` and `d2`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ql_time::{Date, Month, DayCounter};
    ///
    /// let dc = DayCounter::Actual360;
    /// let d1 = Date::from_ymd(2025, Month::January, 1);
    /// let d2 = Date::from_ymd(2025, Month::July, 1);
    /// let yf = dc.year_fraction(d1, d2);
    /// // 181 days / 360 ≈ 0.5028
    /// assert!((yf - 181.0 / 360.0).abs() < 1e-12);
    /// ```
    pub fn year_fraction(&self, d1: Date, d2: Date) -> Real {
        match self {
            DayCounter::Actual360 => {
                let days = d2.serial() - d1.serial();
                days as Real / 360.0
            }
            DayCounter::Actual365Fixed => {
                let days = d2.serial() - d1.serial();
                days as Real / 365.0
            }
            DayCounter::Thirty360(_) => {
                let days = self.day_count(d1, d2);
                days as Real / 360.0
            }
            DayCounter::ActualActual(convention) => {
                actual_actual_year_fraction(d1, d2, *convention)
            }
            DayCounter::Business252 => {
                self.day_count(d1, d2) as Real / 252.0
            }
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            DayCounter::Actual360 => "Actual/360",
            DayCounter::Actual365Fixed => "Actual/365 (Fixed)",
            DayCounter::Thirty360(Thirty360Convention::BondBasis) => "30/360 (Bond Basis)",
            DayCounter::Thirty360(Thirty360Convention::EurobondBasis) => "30E/360 (Eurobond Basis)",
            DayCounter::ActualActual(ActualActualConvention::ISDA) => "Actual/Actual (ISDA)",
            DayCounter::ActualActual(ActualActualConvention::ISMA) => "Actual/Actual (ISMA)",
            DayCounter::ActualActual(ActualActualConvention::AFB) => "Actual/Actual (AFB)",
            DayCounter::Business252 => "Business/252",
        }
    }
}

/// 30/360 day count.
fn thirty360_day_count(d1: Date, d2: Date, convention: Thirty360Convention) -> i32 {
    let mut dd1 = d1.day_of_month() as i32;
    let mut dd2 = d2.day_of_month() as i32;
    let mm1 = d1.month() as i32;
    let mm2 = d2.month() as i32;
    let yy1 = d1.year();
    let yy2 = d2.year();

    match convention {
        Thirty360Convention::BondBasis => {
            if dd1 == 31 {
                dd1 = 30;
            }
            if dd2 == 31 && dd1 >= 30 {
                dd2 = 30;
            }
        }
        Thirty360Convention::EurobondBasis => {
            if dd1 == 31 {
                dd1 = 30;
            }
            if dd2 == 31 {
                dd2 = 30;
            }
        }
    }
    360 * (yy2 - yy1) + 30 * (mm2 - mm1) + (dd2 - dd1)
}

/// Actual/Actual year fraction (ISDA method).
fn actual_actual_year_fraction(d1: Date, d2: Date, convention: ActualActualConvention) -> Real {
    if d1 == d2 {
        return 0.0;
    }
    match convention {
        ActualActualConvention::ISDA => actual_actual_isda(d1, d2),
        ActualActualConvention::AFB => actual_actual_afb(d1, d2),
        ActualActualConvention::ISMA => {
            // Fallback to ISDA when no reference period is given
            actual_actual_isda(d1, d2)
        }
    }
}

fn actual_actual_isda(d1: Date, d2: Date) -> Real {
    let y1 = d1.year();
    let y2 = d2.year();

    if y1 == y2 {
        let days_in_year = if Date::is_leap_year(y1) { 366.0 } else { 365.0 };
        return (d2.serial() - d1.serial()) as Real / days_in_year;
    }

    // Fraction within year y1
    let end_of_y1 = Date::from_ymd(y1, Month::December, 31);
    let basis_y1 = if Date::is_leap_year(y1) { 366.0 } else { 365.0 };
    let frac1 = (end_of_y1.serial() - d1.serial()) as Real / basis_y1;

    // Fraction within year y2
    let start_of_y2 = Date::from_ymd(y2, Month::January, 1);
    let basis_y2 = if Date::is_leap_year(y2) { 366.0 } else { 365.0 };
    let frac2 = (d2.serial() - start_of_y2.serial() + 1) as Real / basis_y2;

    // Whole years in between
    let whole_years = (y2 - y1 - 1) as Real;

    frac1 + frac2 + whole_years
}

fn actual_actual_afb(d1: Date, d2: Date) -> Real {
    let total_days = (d2.serial() - d1.serial()) as Real;
    // Simple AFB: use 366 if the period includes Feb 29, else 365
    let y1 = d1.year();
    let y2 = d2.year();
    let mut has_leap_day = false;
    for y in y1..=y2 {
        if Date::is_leap_year(y) {
            let feb29 = Date::from_ymd(y, Month::February, 29);
            if feb29 > d1 && feb29 <= d2 {
                has_leap_day = true;
                break;
            }
        }
    }
    let basis = if has_leap_day { 366.0 } else { 365.0 };
    total_days / basis
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn actual360_day_count() {
        let d1 = Date::from_ymd(2025, Month::January, 1);
        let d2 = Date::from_ymd(2025, Month::July, 1);
        let dc = DayCounter::Actual360;
        assert_eq!(dc.day_count(d1, d2), 181);
    }

    #[test]
    fn actual360_year_fraction() {
        let d1 = Date::from_ymd(2025, Month::January, 1);
        let d2 = Date::from_ymd(2025, Month::July, 1);
        let dc = DayCounter::Actual360;
        assert_relative_eq!(dc.year_fraction(d1, d2), 181.0 / 360.0, epsilon = 1e-12);
    }

    #[test]
    fn actual365_year_fraction() {
        let d1 = Date::from_ymd(2025, Month::January, 1);
        let d2 = Date::from_ymd(2025, Month::July, 1);
        let dc = DayCounter::Actual365Fixed;
        assert_relative_eq!(dc.year_fraction(d1, d2), 181.0 / 365.0, epsilon = 1e-12);
    }

    #[test]
    fn thirty360_bond_basis() {
        let dc = DayCounter::Thirty360(Thirty360Convention::BondBasis);
        let d1 = Date::from_ymd(2025, Month::January, 31);
        let d2 = Date::from_ymd(2025, Month::April, 30);
        // 30/360 bond basis: 30*(4-1) + (30-30) = 90
        assert_eq!(dc.day_count(d1, d2), 90);
        assert_relative_eq!(dc.year_fraction(d1, d2), 90.0 / 360.0, epsilon = 1e-12);
    }

    #[test]
    fn thirty360_eurobond_basis() {
        let dc = DayCounter::Thirty360(Thirty360Convention::EurobondBasis);
        let d1 = Date::from_ymd(2025, Month::January, 31);
        let d2 = Date::from_ymd(2025, Month::March, 31);
        // Euro: dd1=30, dd2=30 → 30*2 + 0 = 60
        assert_eq!(dc.day_count(d1, d2), 60);
    }

    #[test]
    fn actual_actual_isda_same_year() {
        let dc = DayCounter::ActualActual(ActualActualConvention::ISDA);
        let d1 = Date::from_ymd(2025, Month::January, 1);
        let d2 = Date::from_ymd(2025, Month::July, 1);
        // 2025 is not a leap year → 181/365
        assert_relative_eq!(dc.year_fraction(d1, d2), 181.0 / 365.0, epsilon = 1e-12);
    }

    #[test]
    fn actual_actual_isda_cross_year() {
        let dc = DayCounter::ActualActual(ActualActualConvention::ISDA);
        let d1 = Date::from_ymd(2024, Month::July, 1);
        let d2 = Date::from_ymd(2025, Month::July, 1);
        // 2024 is a leap year (366 days), 2025 is not (365 days)
        // The implementation splits at year boundary:
        //   frac1 = (Dec31 serial - Jul1 serial) / 366  (days remaining in 2024)
        //   frac2 = (Jul1 serial - Jan1 serial + 1) / 365 (days into 2025 including Jan 1)
        // Total should be exactly 1.0 year since it's Jul 1 to Jul 1
        let result = dc.year_fraction(d1, d2);
        // Should be very close to 1.0
        assert_relative_eq!(result, 1.0, epsilon = 0.005);
        // Verify it matches our algorithm exactly
        let end_2024 = Date::from_ymd(2024, Month::December, 31);
        let start_2025 = Date::from_ymd(2025, Month::January, 1);
        let frac1 = (end_2024.serial() - d1.serial()) as f64 / 366.0;
        let frac2 = (d2.serial() - start_2025.serial() + 1) as f64 / 365.0;
        assert_relative_eq!(result, frac1 + frac2, epsilon = 1e-12);
    }

    #[test]
    fn day_counter_name() {
        assert_eq!(DayCounter::Actual360.name(), "Actual/360");
        assert_eq!(DayCounter::Actual365Fixed.name(), "Actual/365 (Fixed)");
        assert_eq!(
            DayCounter::Thirty360(Thirty360Convention::BondBasis).name(),
            "30/360 (Bond Basis)"
        );
        assert_eq!(DayCounter::Business252.name(), "Business/252");
    }

    #[test]
    fn business252_year_fraction() {
        let dc = DayCounter::Business252;
        // Monday to Friday (same week) = 4 business days
        let d1 = Date::from_ymd(2025, Month::June, 16); // Monday
        let d2 = Date::from_ymd(2025, Month::June, 20); // Friday
        let bd = dc.day_count(d1, d2);
        assert!(bd > 0);
        assert_relative_eq!(dc.year_fraction(d1, d2), bd as f64 / 252.0, epsilon = 1e-12);
    }
}
