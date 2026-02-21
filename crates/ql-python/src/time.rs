//! Date, Calendar, DayCounter, Period and Schedule wrappers for Python.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use ql_time::date::Date;
use ql_time::calendar::Calendar;
use ql_time::BusinessDayConvention;
use ql_time::day_counter::DayCounter;
use ql_time::period::{Period, TimeUnit};
use ql_time::{Frequency, Schedule};
use ql_time::schedule::DateGenerationRule;

// ---------------------------------------------------------------------------
// Date
// ---------------------------------------------------------------------------

/// A serial-number date (epoch: serial 1 = 1 Jan 1900).
///
/// Construct with ``Date(year, month, day)`` where month is 1..12.
#[pyclass(name = "Date")]
#[derive(Clone)]
pub struct PyDate {
    pub(crate) inner: Date,
}

#[pymethods]
impl PyDate {
    /// Create a date from year, month (1–12), day.
    #[new]
    fn new(year: i32, month: u32, day: u32) -> PyResult<Self> {
        Date::from_ymd_opt(year, month, day)
            .map(|d| PyDate { inner: d })
            .ok_or_else(|| PyValueError::new_err(format!("invalid date: {year}-{month}-{day}")))
    }

    /// Serial number of the date.
    #[getter]
    fn serial(&self) -> i32 {
        self.inner.serial()
    }

    #[getter]
    fn year(&self) -> i32 {
        self.inner.year()
    }

    #[getter]
    fn month(&self) -> u32 {
        self.inner.month() as u32
    }

    #[getter]
    fn day(&self) -> u32 {
        self.inner.day_of_month()
    }

    /// Day of the week (1=Sun … 7=Sat).
    #[getter]
    fn weekday(&self) -> u32 {
        self.inner.weekday() as u32
    }

    /// Whether this date is the last day of its month.
    fn is_end_of_month(&self) -> bool {
        self.inner.is_end_of_month()
    }

    /// Last day of the same month.
    fn end_of_month(&self) -> PyDate {
        PyDate { inner: self.inner.end_of_month() }
    }

    /// Today's date.
    #[staticmethod]
    fn today() -> PyDate {
        PyDate { inner: Date::today() }
    }

    /// Add calendar days.
    fn add_days(&self, n: i32) -> PyDate {
        PyDate { inner: self.inner + n }
    }

    /// Difference in calendar days between two dates.
    fn days_between(&self, other: &PyDate) -> i32 {
        other.inner.serial() - self.inner.serial()
    }

    fn __repr__(&self) -> String {
        format!("Date({}, {}, {})", self.year(), self.month(), self.day())
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __richcmp__(&self, other: &PyDate, op: pyo3::basic::CompareOp) -> bool {
        match op {
            pyo3::basic::CompareOp::Lt => self.inner < other.inner,
            pyo3::basic::CompareOp::Le => self.inner <= other.inner,
            pyo3::basic::CompareOp::Eq => self.inner == other.inner,
            pyo3::basic::CompareOp::Ne => self.inner != other.inner,
            pyo3::basic::CompareOp::Gt => self.inner > other.inner,
            pyo3::basic::CompareOp::Ge => self.inner >= other.inner,
        }
    }

    fn __hash__(&self) -> u64 {
        self.inner.serial() as u64
    }
}

// ---------------------------------------------------------------------------
// Period
// ---------------------------------------------------------------------------

/// A length of time: ``Period(length, unit)`` where unit is
/// ``"D"``, ``"W"``, ``"M"``, or ``"Y"``.
#[pyclass(name = "Period")]
#[derive(Clone)]
pub struct PyPeriod {
    pub(crate) inner: Period,
}

#[pymethods]
impl PyPeriod {
    #[new]
    fn new(length: i32, unit: &str) -> PyResult<Self> {
        let tu = match unit.to_uppercase().as_str() {
            "D" | "DAYS" => TimeUnit::Days,
            "W" | "WEEKS" => TimeUnit::Weeks,
            "M" | "MONTHS" => TimeUnit::Months,
            "Y" | "YEARS" => TimeUnit::Years,
            _ => return Err(PyValueError::new_err(
                "unit must be one of: D, W, M, Y"
            )),
        };
        Ok(PyPeriod { inner: Period::new(length, tu) })
    }

    fn __repr__(&self) -> String {
        let u = match self.inner.unit {
            TimeUnit::Days => "D",
            TimeUnit::Weeks => "W",
            TimeUnit::Months => "M",
            TimeUnit::Years => "Y",
        };
        format!("Period({}, '{}')", self.inner.length, u)
    }
}

// ---------------------------------------------------------------------------
// Schedule
// ---------------------------------------------------------------------------

/// A schedule of dates built from effective/termination dates and frequency.
#[pyclass(name = "Schedule")]
#[derive(Clone)]
pub struct PySchedule {
    pub(crate) inner: Schedule,
}

#[pymethods]
impl PySchedule {
    /// Build a schedule.
    ///
    /// Parameters:
    ///   effective  — start date
    ///   termination — end date
    ///   frequency — one of: "Annual", "Semiannual", "Quarterly", "Monthly"
    ///   calendar  — one of: "TARGET", "NullCalendar", "WeekendsOnly", "US-Settlement", "US-NYSE", "UK"
    ///   convention — one of: "Unadjusted", "Following", "ModifiedFollowing", "Preceding"
    #[new]
    #[pyo3(signature = (effective, termination, frequency="Semiannual", calendar="TARGET", convention="ModifiedFollowing"))]
    fn new(
        effective: &PyDate,
        termination: &PyDate,
        frequency: &str,
        calendar: &str,
        convention: &str,
    ) -> PyResult<Self> {
        let freq = parse_frequency(frequency)?;
        let cal = parse_calendar(calendar)?;
        let conv = parse_convention(convention)?;

        let sched = Schedule::builder()
            .effective_date(effective.inner)
            .termination_date(termination.inner)
            .frequency(freq)
            .calendar(cal)
            .convention(conv)
            .termination_convention(conv)
            .rule(DateGenerationRule::Forward)
            .end_of_month(false)
            .build()
            .map_err(|e| PyValueError::new_err(format!("schedule build error: {e}")))?;

        Ok(PySchedule { inner: sched })
    }

    /// Number of dates in the schedule.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Get date at index i.
    fn date(&self, i: usize) -> PyResult<PyDate> {
        if i >= self.inner.len() {
            return Err(PyValueError::new_err(format!("index {i} out of range")));
        }
        Ok(PyDate { inner: self.inner.date(i) })
    }

    /// All dates as a list.
    fn dates(&self) -> Vec<PyDate> {
        self.inner.dates().iter().map(|&d| PyDate { inner: d }).collect()
    }

    fn __repr__(&self) -> String {
        format!("Schedule(len={})", self.inner.len())
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Check if a date is a business day in the given calendar.
///
/// Calendar names: "TARGET", "NullCalendar", "WeekendsOnly",
/// "US-Settlement", "US-NYSE", "US-FederalReserve", "UK", "Japan", etc.
#[pyfunction]
#[pyo3(signature = (date, calendar="TARGET"))]
pub fn is_business_day(date: &PyDate, calendar: &str) -> PyResult<bool> {
    let cal = parse_calendar(calendar)?;
    Ok(cal.is_business_day(date.inner))
}

/// Advance a date by a period in the given calendar.
#[pyfunction]
#[pyo3(signature = (date, period, calendar="TARGET", convention="ModifiedFollowing"))]
pub fn advance_date(
    date: &PyDate,
    period: &PyPeriod,
    calendar: &str,
    convention: &str,
) -> PyResult<PyDate> {
    let cal = parse_calendar(calendar)?;
    let conv = parse_convention(convention)?;
    let result = cal.advance(date.inner, period.inner, conv, false);
    Ok(PyDate { inner: result })
}

/// Day-count year fraction between two dates.
///
/// Day counter names: "Actual360", "Actual365Fixed", "30/360",
/// "ActualActual", "Business252".
#[pyfunction]
#[pyo3(signature = (d1, d2, day_counter="Actual365Fixed"))]
pub fn year_fraction(d1: &PyDate, d2: &PyDate, day_counter: &str) -> PyResult<f64> {
    let dc = parse_day_counter(day_counter)?;
    Ok(dc.year_fraction(d1.inner, d2.inner))
}

/// Number of business days between two dates.
#[pyfunction]
#[pyo3(signature = (d1, d2, calendar="TARGET"))]
pub fn business_days_between(d1: &PyDate, d2: &PyDate, calendar: &str) -> PyResult<i32> {
    let cal = parse_calendar(calendar)?;
    Ok(cal.business_days_between(d1.inner, d2.inner))
}

// ---------------------------------------------------------------------------
// Parsers (string → Rust enum)
// ---------------------------------------------------------------------------

pub(crate) fn parse_calendar(name: &str) -> PyResult<Calendar> {
    use ql_time::calendar::USMarket;
    match name {
        "TARGET" | "target" => Ok(Calendar::Target),
        "NullCalendar" | "null" => Ok(Calendar::NullCalendar),
        "WeekendsOnly" | "weekends" => Ok(Calendar::WeekendsOnly),
        "US-Settlement" | "US" => Ok(Calendar::UnitedStates(USMarket::Settlement)),
        "US-NYSE" | "NYSE" => Ok(Calendar::UnitedStates(USMarket::NYSE)),
        "US-FederalReserve" | "FED" => Ok(Calendar::UnitedStates(USMarket::FederalReserve)),
        "UK" => Ok(Calendar::UnitedKingdom),
        "Japan" | "JP" => Ok(Calendar::Japan),
        "China" | "CN" => Ok(Calendar::China),
        "Germany" | "DE" => Ok(Calendar::Germany),
        "France" | "FR" => Ok(Calendar::France),
        "Italy" | "IT" => Ok(Calendar::Italy),
        "Canada" | "CA" => Ok(Calendar::Canada),
        "Australia" | "AU" => Ok(Calendar::Australia),
        "Switzerland" | "CH" => Ok(Calendar::Switzerland),
        _ => Err(PyValueError::new_err(format!("unknown calendar: {name}"))),
    }
}

pub(crate) fn parse_convention(name: &str) -> PyResult<BusinessDayConvention> {
    match name {
        "Unadjusted" => Ok(BusinessDayConvention::Unadjusted),
        "Following" => Ok(BusinessDayConvention::Following),
        "ModifiedFollowing" | "MF" => Ok(BusinessDayConvention::ModifiedFollowing),
        "Preceding" => Ok(BusinessDayConvention::Preceding),
        "ModifiedPreceding" | "MP" => Ok(BusinessDayConvention::ModifiedPreceding),
        "Nearest" => Ok(BusinessDayConvention::Nearest),
        _ => Err(PyValueError::new_err(format!("unknown convention: {name}"))),
    }
}

pub(crate) fn parse_frequency(name: &str) -> PyResult<Frequency> {
    match name {
        "Annual" | "1Y" => Ok(Frequency::Annual),
        "Semiannual" | "6M" => Ok(Frequency::Semiannual),
        "Quarterly" | "3M" => Ok(Frequency::Quarterly),
        "Monthly" | "1M" => Ok(Frequency::Monthly),
        "Biweekly" | "2W" => Ok(Frequency::Biweekly),
        "Weekly" | "1W" => Ok(Frequency::Weekly),
        "Daily" | "1D" => Ok(Frequency::Daily),
        "Once" => Ok(Frequency::Once),
        _ => Err(PyValueError::new_err(format!("unknown frequency: {name}"))),
    }
}

pub(crate) fn parse_day_counter(name: &str) -> PyResult<DayCounter> {
    use ql_time::day_counter::{Thirty360Convention, ActualActualConvention};
    match name {
        "Actual360" | "ACT/360" => Ok(DayCounter::Actual360),
        "Actual365Fixed" | "ACT/365" => Ok(DayCounter::Actual365Fixed),
        "30/360" | "Thirty360" | "30/360 Bond" => {
            Ok(DayCounter::Thirty360(Thirty360Convention::BondBasis))
        }
        "30E/360" | "Eurobond" => {
            Ok(DayCounter::Thirty360(Thirty360Convention::EurobondBasis))
        }
        "ActualActual" | "ACT/ACT" | "ActualActual/ISDA" => {
            Ok(DayCounter::ActualActual(ActualActualConvention::ISDA))
        }
        "ActualActual/ISMA" => {
            Ok(DayCounter::ActualActual(ActualActualConvention::ISMA))
        }
        "Business252" => Ok(DayCounter::Business252),
        _ => Err(PyValueError::new_err(format!("unknown day counter: {name}"))),
    }
}
