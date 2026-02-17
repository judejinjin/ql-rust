//! Index trait and global fixing store (IndexManager).

use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

use ql_core::errors::QLResult;
use ql_time::Date;

// ---------------------------------------------------------------------------
// Index Trait
// ---------------------------------------------------------------------------

/// Common interface for financial indexes (interest rates, equity, inflation).
///
/// An index can provide historical fixings (from the IndexManager store) and
/// forecast fixings (from an associated term structure).
pub trait Index: Send + Sync {
    /// Unique name (e.g. "Euribor6M Actual/360").
    fn name(&self) -> &str;

    /// Whether a given date is a valid fixing date.
    fn is_valid_fixing_date(&self, date: Date) -> bool;

    /// Return the fixing for a given date.
    ///
    /// If the date is in the past, looks up the IndexManager.
    /// If the date is today or in the future, forecasts using the term structure.
    fn fixing(&self, date: Date, forecast_today_fixing: bool) -> QLResult<f64>;

    /// Add a historical fixing.
    fn add_fixing(&self, date: Date, value: f64) -> QLResult<()> {
        IndexManager::instance().add_fixing(self.name(), date, value)
    }
}

// ---------------------------------------------------------------------------
// IndexManager — global fixing store
// ---------------------------------------------------------------------------

/// Global singleton store for historical index fixings.
///
/// In QuantLib this is implemented as a global map keyed by index name.
pub struct IndexManager {
    fixings: RwLock<HashMap<String, HashMap<i32, f64>>>,
}

static INDEX_MANAGER: OnceLock<IndexManager> = OnceLock::new();

impl IndexManager {
    /// Get the global IndexManager instance.
    pub fn instance() -> &'static IndexManager {
        INDEX_MANAGER.get_or_init(|| IndexManager {
            fixings: RwLock::new(HashMap::new()),
        })
    }

    /// Store a fixing for a given index name and date.
    pub fn add_fixing(&self, name: &str, date: Date, value: f64) -> QLResult<()> {
        let mut fixings = self.fixings.write().unwrap();
        fixings
            .entry(name.to_string())
            .or_default()
            .insert(date.serial(), value);
        Ok(())
    }

    /// Retrieve a fixing.
    pub fn get_fixing(&self, name: &str, date: Date) -> Option<f64> {
        let fixings = self.fixings.read().unwrap();
        fixings.get(name).and_then(|m| m.get(&date.serial()).copied())
    }

    /// Check whether a fixing exists for the given index and date.
    pub fn has_fixing(&self, name: &str, date: Date) -> bool {
        let fixings = self.fixings.read().unwrap();
        fixings
            .get(name)
            .is_some_and(|m| m.contains_key(&date.serial()))
    }

    /// Clear all fixings for a given index name.
    pub fn clear_fixings(&self, name: &str) {
        let mut fixings = self.fixings.write().unwrap();
        fixings.remove(name);
    }

    /// Clear all fixings for all indexes.
    pub fn clear_all_fixings(&self) {
        let mut fixings = self.fixings.write().unwrap();
        fixings.clear();
    }

    /// Return all fixing dates for a given index.
    pub fn fixing_dates(&self, name: &str) -> Vec<Date> {
        let fixings = self.fixings.read().unwrap();
        fixings
            .get(name)
            .map(|m| {
                let mut dates: Vec<Date> = m.keys().map(|&s| Date::from_serial(s)).collect();
                dates.sort();
                dates
            })
            .unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// TimeSeries — simple date→value map (used for fixings)
// ---------------------------------------------------------------------------

/// A time series of (Date, f64) values. Used for storing fixings.
#[derive(Debug, Clone)]
pub struct TimeSeries {
    data: Vec<(Date, f64)>,
}

impl TimeSeries {
    /// Create an empty time series.
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Add a data point (sorted insertion).
    pub fn insert(&mut self, date: Date, value: f64) {
        match self.data.binary_search_by_key(&date, |(d, _)| *d) {
            Ok(pos) => self.data[pos].1 = value,
            Err(pos) => self.data.insert(pos, (date, value)),
        }
    }

    /// Get the value at a date.
    pub fn get(&self, date: Date) -> Option<f64> {
        self.data
            .binary_search_by_key(&date, |(d, _)| *d)
            .ok()
            .map(|i| self.data[i].1)
    }

    /// Number of data points.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the series is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Iterate over (date, value) pairs.
    pub fn iter(&self) -> impl Iterator<Item = &(Date, f64)> {
        self.data.iter()
    }
}

impl Default for TimeSeries {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ql_time::Month;

    #[test]
    fn index_manager_add_and_get() {
        let mgr = IndexManager::instance();
        let date = Date::from_ymd(2025, Month::January, 15);
        mgr.add_fixing("TestIndex", date, 0.05).unwrap();
        assert_eq!(mgr.get_fixing("TestIndex", date), Some(0.05));
    }

    #[test]
    fn index_manager_missing_fixing() {
        let mgr = IndexManager::instance();
        let date = Date::from_ymd(2099, Month::December, 31);
        assert_eq!(mgr.get_fixing("NonexistentIndex", date), None);
    }

    #[test]
    fn time_series_basic() {
        let mut ts = TimeSeries::new();
        let d1 = Date::from_ymd(2025, Month::January, 1);
        let d2 = Date::from_ymd(2025, Month::February, 1);

        ts.insert(d1, 1.0);
        ts.insert(d2, 2.0);

        assert_eq!(ts.len(), 2);
        assert_eq!(ts.get(d1), Some(1.0));
        assert_eq!(ts.get(d2), Some(2.0));
    }

    #[test]
    fn time_series_overwrite() {
        let mut ts = TimeSeries::new();
        let d1 = Date::from_ymd(2025, Month::March, 15);
        ts.insert(d1, 1.0);
        ts.insert(d1, 2.0);
        assert_eq!(ts.len(), 1);
        assert_eq!(ts.get(d1), Some(2.0));
    }
}
