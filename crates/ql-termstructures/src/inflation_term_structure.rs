//! Inflation term structures and inflation-linked instruments.
//!
//! Provides zero-coupon and year-on-year inflation term structures,
//! and a zero-coupon inflation swap instrument.

use ql_time::{Calendar, Date, DayCounter};
use crate::term_structure::TermStructure;

// =========================================================================
// ZeroInflationTermStructure trait
// =========================================================================

/// Zero-coupon inflation term structure: maps time → cumulative CPI ratio.
///
/// The "zero inflation rate" i(t) satisfies: CPI(t) = CPI(0) × (1 + i(t))^t.
pub trait ZeroInflationTermStructure: TermStructure {
    /// Zero-coupon inflation rate at time `t` (annualised).
    fn zero_rate(&self, t: f64) -> f64;

    /// Cumulative inflation factor from base to time `t`: (1 + i(t))^t.
    fn inflation_factor(&self, t: f64) -> f64 {
        let rate = self.zero_rate(t);
        (1.0 + rate).powf(t)
    }
}

// =========================================================================
// FlatZeroInflationCurve
// =========================================================================

/// Flat (constant) zero-coupon inflation rate.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FlatZeroInflationCurve {
    reference_date: Date,
    day_counter: DayCounter,
    rate: f64,
    base_cpi: f64,
}

impl FlatZeroInflationCurve {
    /// Create a flat zero-coupon inflation curve.
    pub fn new(reference_date: Date, rate: f64, base_cpi: f64, day_counter: DayCounter) -> Self {
        Self {
            reference_date,
            day_counter,
            rate,
            base_cpi,
        }
    }

    /// The base CPI level at the reference date.
    pub fn base_cpi(&self) -> f64 {
        self.base_cpi
    }

    /// Projected CPI at time t.
    pub fn cpi(&self, t: f64) -> f64 {
        self.base_cpi * (1.0 + self.rate).powf(t)
    }
}

impl TermStructure for FlatZeroInflationCurve {
    fn reference_date(&self) -> Date {
        self.reference_date
    }
    fn day_counter(&self) -> DayCounter {
        self.day_counter
    }
    fn calendar(&self) -> Calendar {
        Calendar::NullCalendar
    }
    fn max_date(&self) -> Date {
        Date::from_serial(73050)
    }
}

impl ZeroInflationTermStructure for FlatZeroInflationCurve {
    fn zero_rate(&self, _t: f64) -> f64 {
        self.rate
    }
}

// =========================================================================
// PiecewiseZeroInflationCurve
// =========================================================================

/// Piecewise-linear zero inflation curve bootstrapped from swap quotes.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PiecewiseZeroInflationCurve {
    reference_date: Date,
    day_counter: DayCounter,
    times: Vec<f64>,
    rates: Vec<f64>,
}

impl PiecewiseZeroInflationCurve {
    /// Bootstrap from zero-coupon inflation swap rates.
    ///
    /// `tenors`: year fractions for each swap maturity
    /// `rates`: quoted zero-coupon inflation swap rates
    pub fn new(
        reference_date: Date,
        day_counter: DayCounter,
        tenors: Vec<f64>,
        rates: Vec<f64>,
    ) -> Self {
        assert_eq!(tenors.len(), rates.len());
        Self {
            reference_date,
            day_counter,
            times: tenors,
            rates,
        }
    }

    /// Linearly interpolate the zero rate at time t.
    fn interpolate_rate(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return self.rates[0];
        }
        let n = self.times.len();
        if t >= self.times[n - 1] {
            return self.rates[n - 1];
        }
        let idx = self.times.partition_point(|&ti| ti < t).min(n - 1).max(1);
        let t0 = self.times[idx - 1];
        let t1 = self.times[idx];
        let r0 = self.rates[idx - 1];
        let r1 = self.rates[idx];
        r0 + (r1 - r0) * (t - t0) / (t1 - t0)
    }
}

impl TermStructure for PiecewiseZeroInflationCurve {
    fn reference_date(&self) -> Date {
        self.reference_date
    }
    fn day_counter(&self) -> DayCounter {
        self.day_counter
    }
    fn calendar(&self) -> Calendar {
        Calendar::NullCalendar
    }
    fn max_date(&self) -> Date {
        Date::from_serial(73050)
    }
}

impl ZeroInflationTermStructure for PiecewiseZeroInflationCurve {
    fn zero_rate(&self, t: f64) -> f64 {
        self.interpolate_rate(t)
    }
}

// =========================================================================
// ZeroCouponInflationSwap
// =========================================================================

/// A zero-coupon inflation swap.
///
/// At maturity, the inflation leg pays: N × [CPI(T)/CPI(0) - 1]
/// The fixed leg pays: N × [(1+K)^T - 1]
/// where K is the fixed rate.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ZeroCouponInflationSwap {
    /// Notional.
    pub notional: f64,
    /// Fixed rate (annualised).
    pub fixed_rate: f64,
    /// Maturity in years.
    pub maturity_years: f64,
    /// Base CPI at inception.
    pub base_cpi: f64,
    /// True if we pay fixed (receive inflation).
    pub pay_fixed: bool,
}

impl ZeroCouponInflationSwap {
    /// Create a new zero-coupon inflation swap.
    pub fn new(
        notional: f64,
        fixed_rate: f64,
        maturity_years: f64,
        base_cpi: f64,
        pay_fixed: bool,
    ) -> Self {
        Self {
            notional,
            fixed_rate,
            maturity_years,
            base_cpi,
            pay_fixed,
        }
    }

    /// Price the swap given an inflation curve and a discount curve.
    pub fn npv(
        &self,
        inflation_curve: &dyn ZeroInflationTermStructure,
        discount_factor: f64,
    ) -> f64 {
        let t = self.maturity_years;

        // Inflation leg: N * [CPI(T)/CPI(0) - 1] * df
        let inflation_factor = inflation_curve.inflation_factor(t);
        let inflation_leg = self.notional * (inflation_factor - 1.0) * discount_factor;

        // Fixed leg: N * [(1+K)^T - 1] * df
        let fixed_leg =
            self.notional * ((1.0 + self.fixed_rate).powf(t) - 1.0) * discount_factor;

        if self.pay_fixed {
            inflation_leg - fixed_leg // receive inflation, pay fixed
        } else {
            fixed_leg - inflation_leg // receive fixed, pay inflation
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_time::Month;

    #[test]
    fn flat_inflation_rate() {
        let ref_date = Date::from_ymd(2025, Month::January, 15);
        let curve = FlatZeroInflationCurve::new(ref_date, 0.025, 300.0, DayCounter::Actual365Fixed);
        assert_abs_diff_eq!(curve.zero_rate(5.0), 0.025, epsilon = 1e-12);
    }

    #[test]
    fn flat_inflation_cpi_projection() {
        let ref_date = Date::from_ymd(2025, Month::January, 15);
        let curve = FlatZeroInflationCurve::new(ref_date, 0.02, 300.0, DayCounter::Actual365Fixed);
        let cpi_5y = curve.cpi(5.0);
        let expected = 300.0 * 1.02_f64.powi(5);
        assert_abs_diff_eq!(cpi_5y, expected, epsilon = 1e-8);
    }

    #[test]
    fn flat_inflation_factor() {
        let ref_date = Date::from_ymd(2025, Month::January, 15);
        let curve = FlatZeroInflationCurve::new(ref_date, 0.03, 300.0, DayCounter::Actual365Fixed);
        let factor = curve.inflation_factor(10.0);
        let expected = 1.03_f64.powi(10);
        assert_abs_diff_eq!(factor, expected, epsilon = 1e-8);
    }

    #[test]
    fn piecewise_inflation_interpolation() {
        let ref_date = Date::from_ymd(2025, Month::January, 15);
        let tenors = vec![1.0, 5.0, 10.0];
        let rates = vec![0.020, 0.025, 0.030];
        let curve = PiecewiseZeroInflationCurve::new(
            ref_date,
            DayCounter::Actual365Fixed,
            tenors,
            rates,
        );

        // At nodes
        assert_abs_diff_eq!(curve.zero_rate(1.0), 0.020, epsilon = 1e-12);
        assert_abs_diff_eq!(curve.zero_rate(5.0), 0.025, epsilon = 1e-12);
        assert_abs_diff_eq!(curve.zero_rate(10.0), 0.030, epsilon = 1e-12);

        // Interpolated
        let r_3 = curve.zero_rate(3.0);
        assert!(r_3 > 0.020 && r_3 < 0.025);
    }

    #[test]
    fn zero_coupon_inflation_swap_npv_at_par() {
        // If fixed_rate == inflation_rate, NPV should be zero
        let ref_date = Date::from_ymd(2025, Month::January, 15);
        let inflation_rate = 0.025;
        let curve = FlatZeroInflationCurve::new(ref_date, inflation_rate, 300.0, DayCounter::Actual365Fixed);

        let swap = ZeroCouponInflationSwap::new(
            1_000_000.0,
            inflation_rate,
            5.0,
            300.0,
            true,
        );

        let npv = swap.npv(&curve, 0.95);
        assert_abs_diff_eq!(npv, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn zero_coupon_inflation_swap_pay_fixed_positive_when_inflation_higher() {
        let ref_date = Date::from_ymd(2025, Month::January, 15);
        let curve = FlatZeroInflationCurve::new(ref_date, 0.04, 300.0, DayCounter::Actual365Fixed);

        let swap = ZeroCouponInflationSwap::new(
            1_000_000.0,
            0.02, // fixed rate lower than inflation
            5.0,
            300.0,
            true, // pay fixed, receive inflation
        );

        let npv = swap.npv(&curve, 0.90);
        assert!(npv > 0.0, "Pay-fixed swap should be positive when inflation > fixed rate");
    }

    #[test]
    fn zero_coupon_inflation_swap_symmetry() {
        let ref_date = Date::from_ymd(2025, Month::January, 15);
        let curve = FlatZeroInflationCurve::new(ref_date, 0.03, 300.0, DayCounter::Actual365Fixed);

        let pay_fixed = ZeroCouponInflationSwap::new(1_000_000.0, 0.025, 5.0, 300.0, true);
        let receive_fixed = ZeroCouponInflationSwap::new(1_000_000.0, 0.025, 5.0, 300.0, false);

        let npv_pf = pay_fixed.npv(&curve, 0.90);
        let npv_rf = receive_fixed.npv(&curve, 0.90);

        assert_abs_diff_eq!(npv_pf + npv_rf, 0.0, epsilon = 1e-6);
    }
}
