//! Inflation indexes (CPI family).
//!
//! Inflation indexes track consumer price levels. They are used for
//! pricing inflation-linked bonds (TIPS, linkers) and zero-coupon/
//! year-on-year inflation swaps.
//!
//! ## Publication lag
//!
//! CPI data is published with a delay (typically 2–3 months). The
//! `availability_lag` field captures this: a CPI observation for
//! January is typically released in March.
//!
//! ## Interpolation
//!
//! Inflation fixings can be used "flat" (no interpolation — use the
//! value for the reference month) or "linearly" interpolated between
//! consecutive months.

use ql_core::errors::{QLError, QLResult};
use ql_time::{Date, Period};

use crate::index::{Index, IndexManager};

/// Interpolation convention for inflation fixings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpiInterpolation {
    /// Use the month's CPI level with no interpolation.
    Flat,
    /// Linearly interpolate between the two surrounding months.
    Linear,
}

/// An inflation (CPI) index.
///
/// Tracks the level of a consumer-price or similar price index over time.
/// Historical fixings are stored per observation month (serial = first of month).
///
/// # Examples
///
/// ```rust
/// use ql_indexes::{InflationIndex, CpiInterpolation, Index};
/// use ql_time::{Date, Month, Period, TimeUnit};
///
/// let cpi = InflationIndex::us_cpi();
/// assert_eq!(cpi.name(), "US CPI");
/// ```
#[derive(Debug, Clone)]
pub struct InflationIndex {
    /// Human-readable name (e.g. "US CPI", "EU HICP").
    name: String,
    /// Familial name used for the IndexManager key (e.g. "CPI").
    family_name: String,
    /// Publication lag — how long after the reference month the data
    /// becomes available (e.g. Period(2, Months)).
    pub availability_lag: Period,
    /// Base date for the index (the date of the base fixing).
    pub base_date: Date,
    /// Base CPI level at the base date.
    pub base_fixing: f64,
    /// Interpolation convention.
    pub interpolation: CpiInterpolation,
    /// Whether the index is revised after first publication.
    pub revised: bool,
}

impl InflationIndex {
    /// Create a new inflation index.
    pub fn new(
        name: &str,
        family_name: &str,
        availability_lag: Period,
        base_date: Date,
        base_fixing: f64,
        interpolation: CpiInterpolation,
        revised: bool,
    ) -> Self {
        Self {
            name: name.to_string(),
            family_name: family_name.to_string(),
            availability_lag,
            base_date,
            base_fixing,
            interpolation,
            revised,
        }
    }

    /// US CPI-U (All Urban Consumers) with standard 2-month lag.
    pub fn us_cpi() -> Self {
        use ql_time::{Month, TimeUnit};
        Self::new(
            "US CPI",
            "CPI",
            Period::new(2, TimeUnit::Months),
            Date::from_ymd(2020, Month::January, 1),
            258.678, // Jan 2020 CPI-U (approximate)
            CpiInterpolation::Flat,
            false,
        )
    }

    /// Eurozone HICP (Harmonised Index of Consumer Prices) with 3-month lag.
    pub fn eu_hicp() -> Self {
        use ql_time::{Month, TimeUnit};
        Self::new(
            "EU HICP",
            "HICP",
            Period::new(3, TimeUnit::Months),
            Date::from_ymd(2020, Month::January, 1),
            105.12, // Jan 2020 HICP (approximate, 2015=100)
            CpiInterpolation::Flat,
            true,
        )
    }

    /// UK RPI (Retail Price Index) with 2-month lag.
    pub fn uk_rpi() -> Self {
        use ql_time::{Month, TimeUnit};
        Self::new(
            "UK RPI",
            "RPI",
            Period::new(2, TimeUnit::Months),
            Date::from_ymd(2020, Month::January, 1),
            290.6, // Jan 2020 RPI
            CpiInterpolation::Flat,
            false,
        )
    }

    /// The family name used as key prefix in the fixing store.
    pub fn family_name(&self) -> &str {
        &self.family_name
    }

    /// Add a fixing for a reference month. The `date` should be the first
    /// of the observation month (e.g. 2024-03-01 for March 2024 CPI).
    pub fn add_cpi_fixing(&self, date: Date, value: f64) -> QLResult<()> {
        IndexManager::instance().add_fixing(&self.name, date, value)
    }

    /// Retrieve the CPI level for a given observation date.
    ///
    /// For `Flat` interpolation, use the first-of-the-month serial.
    /// For `Linear`, interpolate between the surrounding months.
    pub fn fixing_at(&self, date: Date) -> QLResult<f64> {
        match self.interpolation {
            CpiInterpolation::Flat => {
                let ref_date = first_of_month(date);
                IndexManager::instance()
                    .get_fixing(&self.name, ref_date)
                    .ok_or(QLError::NotFound)
            }
            CpiInterpolation::Linear => {
                let d1 = first_of_month(date);
                let d2 = first_of_next_month(date);
                let v1 = IndexManager::instance()
                    .get_fixing(&self.name, d1)
                    .ok_or(QLError::NotFound)?;
                let v2 = IndexManager::instance()
                    .get_fixing(&self.name, d2)
                    .ok_or(QLError::NotFound)?;
                let day = date.day_of_month() as f64;
                let days_in_month = days_in_month_of(date) as f64;
                let w = (day - 1.0) / days_in_month;
                Ok(v1 * (1.0 - w) + v2 * w)
            }
        }
    }

    /// Compute the inflation ratio CPI(t) / CPI(base).
    pub fn ratio(&self, date: Date) -> QLResult<f64> {
        let cpi_t = self.fixing_at(date)?;
        if self.base_fixing.abs() < 1e-15 {
            return Err(QLError::InvalidArgument("Base CPI is zero".into()));
        }
        Ok(cpi_t / self.base_fixing)
    }

    /// Compute year-on-year inflation rate between two dates.
    pub fn yoy_rate(&self, date: Date, lag_months: u32) -> QLResult<f64> {
        let d1 = add_months(date, -(lag_months as i32) - 12);
        let d2 = add_months(date, -(lag_months as i32));
        let cpi1 = self.fixing_at(d1)?;
        let cpi2 = self.fixing_at(d2)?;
        if cpi1.abs() < 1e-15 {
            return Err(QLError::InvalidArgument("Previous-year CPI is zero".into()));
        }
        Ok(cpi2 / cpi1 - 1.0)
    }
}

impl Index for InflationIndex {
    fn name(&self) -> &str {
        &self.name
    }

    fn is_valid_fixing_date(&self, _date: Date) -> bool {
        // Any first-of-month is a valid observation date.
        // For simplicity, accept all dates (fixings are stored by first-of-month).
        true
    }

    fn fixing(&self, date: Date, _forecast_today_fixing: bool) -> QLResult<f64> {
        self.fixing_at(date)
    }
}

// ── Date helpers ─────────────────────────────────────────

/// Return the first day of the month containing `date`.
fn first_of_month(date: Date) -> Date {
    Date::from_ymd(date.year(), date.month(), 1)
}

/// Return the first day of the next month after `date`.
fn first_of_next_month(date: Date) -> Date {
    let y = date.year();
    let m = date.month() as u32;
    if m < 12 {
        let next = unsafe_month_from_u32(m + 1);
        Date::from_ymd(y, next, 1)
    } else {
        Date::from_ymd(y + 1, ql_time::Month::January, 1)
    }
}

/// Number of days in the month containing `date`.
fn days_in_month_of(date: Date) -> u32 {
    let start = first_of_month(date);
    let next = first_of_next_month(date);
    (next.serial() - start.serial()) as u32
}

/// Add signed months to a date (approximate: clamp day to month length).
fn add_months(date: Date, months: i32) -> Date {
    let y = date.year();
    let m = date.month() as i32;
    let d = date.day_of_month();
    let total_months = y * 12 + m - 1 + months;
    let new_y = total_months.div_euclid(12);
    let new_m = (total_months.rem_euclid(12) + 1) as u32;
    let month = unsafe_month_from_u32(new_m);

    // Clamp day to last day of target month
    let max_day = days_in_month(new_y, new_m);
    let day = d.min(max_day);
    Date::from_ymd(new_y, month, day)
}

fn days_in_month(y: i32, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) {
                29
            } else {
                28
            }
        }
        _ => 30,
    }
}

/// Convert a u32 (1-12) to a `Month` variant without panicking.
fn unsafe_month_from_u32(m: u32) -> ql_time::Month {
    use ql_time::Month;
    match m {
        1 => Month::January,
        2 => Month::February,
        3 => Month::March,
        4 => Month::April,
        5 => Month::May,
        6 => Month::June,
        7 => Month::July,
        8 => Month::August,
        9 => Month::September,
        10 => Month::October,
        11 => Month::November,
        12 => Month::December,
        _ => Month::January, // fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ql_time::Month;

    fn create_cpi_with_fixings() -> InflationIndex {
        let cpi = InflationIndex::us_cpi();
        // Add monthly fixings for 2023-2024
        let fixings = [
            (2023, Month::January, 299.170),
            (2023, Month::February, 300.840),
            (2023, Month::March, 301.836),
            (2023, Month::April, 303.363),
            (2023, Month::May, 304.127),
            (2023, Month::June, 305.109),
            (2023, Month::July, 305.691),
            (2023, Month::August, 307.026),
            (2023, Month::September, 307.789),
            (2023, Month::October, 307.671),
            (2023, Month::November, 307.051),
            (2023, Month::December, 306.746),
            (2024, Month::January, 308.417),
            (2024, Month::February, 310.326),
            (2024, Month::March, 312.332),
        ];
        for (y, m, v) in fixings {
            cpi.add_cpi_fixing(Date::from_ymd(y, m, 1), v).unwrap();
        }
        cpi
    }

    #[test]
    fn us_cpi_constructor() {
        let cpi = InflationIndex::us_cpi();
        assert_eq!(cpi.name(), "US CPI");
        assert_eq!(cpi.family_name(), "CPI");
        assert!((cpi.base_fixing - 258.678).abs() < 1e-6);
    }

    #[test]
    fn eu_hicp_constructor() {
        let hicp = InflationIndex::eu_hicp();
        assert_eq!(hicp.name(), "EU HICP");
        assert!(hicp.revised);
    }

    #[test]
    fn uk_rpi_constructor() {
        let rpi = InflationIndex::uk_rpi();
        assert_eq!(rpi.name(), "UK RPI");
    }

    #[test]
    fn flat_fixing() {
        let cpi = create_cpi_with_fixings();
        // Flat interpolation: any date in March 2024 → March 2024 fixing
        let v = cpi.fixing_at(Date::from_ymd(2024, Month::March, 15)).unwrap();
        assert!((v - 312.332).abs() < 1e-6);
    }

    #[test]
    fn linear_interpolation() {
        let mut cpi = create_cpi_with_fixings();
        cpi.interpolation = CpiInterpolation::Linear;

        // Midpoint of January 2024 (31 days): day 16 => w = 15/31 ≈ 0.4839
        let d = Date::from_ymd(2024, Month::January, 16);
        let v = cpi.fixing_at(d).unwrap();
        // Expected: 308.417 * (1 - 15/31) + 310.326 * (15/31)
        let w = 15.0 / 31.0;
        let expected = 308.417 * (1.0 - w) + 310.326 * w;
        assert!((v - expected).abs() < 1e-6);
    }

    #[test]
    fn cpi_ratio() {
        let cpi = create_cpi_with_fixings();
        let ratio = cpi.ratio(Date::from_ymd(2024, Month::January, 1)).unwrap();
        let expected = 308.417 / 258.678;
        assert!((ratio - expected).abs() < 1e-6);
    }

    #[test]
    fn yoy_rate() {
        let cpi = create_cpi_with_fixings();
        // YoY from Jan 2023 to Jan 2024 with 0-month lag
        let yoy = cpi.yoy_rate(Date::from_ymd(2024, Month::January, 1), 0).unwrap();
        let expected = 308.417 / 299.170 - 1.0;
        assert!((yoy - expected).abs() < 1e-6);
    }

    #[test]
    fn add_months_helper() {
        let d = Date::from_ymd(2024, Month::March, 15);
        let d2 = add_months(d, -13);
        assert_eq!(d2.year(), 2023);
        assert_eq!(d2.month(), Month::February);
        assert_eq!(d2.day_of_month(), 15);
    }

    #[test]
    fn first_of_month_helper() {
        let d = Date::from_ymd(2024, Month::July, 23);
        let f = first_of_month(d);
        assert_eq!(f.year(), 2024);
        assert_eq!(f.month(), Month::July);
        assert_eq!(f.day_of_month(), 1);
    }

    #[test]
    fn index_trait() {
        let cpi = create_cpi_with_fixings();
        // Index::fixing should delegate to fixing_at
        let v = cpi.fixing(Date::from_ymd(2024, Month::February, 1), false).unwrap();
        assert!((v - 310.326).abs() < 1e-6);
    }
}
