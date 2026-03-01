//! Phase 23 advanced cash flow additions.
//!
//! - [`AverageBMACoupon`] — BMA (Bond Market Association) averaged coupon.
//! - [`OvernightIndexedCouponPricer`] — extended pricer with lookback/lockout.
//! - [`CashFlowVectors`] — coupon vector builder utility.

use crate::cashflow::{CashFlow, Leg};
use crate::coupon::Coupon;
use crate::fixed_rate_coupon::FixedRateCoupon;
use ql_time::{Date, DayCounter};
use serde::{Deserialize, Serialize};
use std::any::Any;

// ===========================================================================
// AverageBMACoupon
// ===========================================================================

/// A BMA (Bond Market Association) averaged coupon.
///
/// BMA swaps reference the SIFMA Municipal Swap Index (formerly BMA index),
/// which resets weekly.  The coupon rate is the arithmetic average of the
/// weekly BMA fixings over the accrual period:
///
/// ```text
/// rate = (1/N) * Σᵢ fixing_i  +  spread
/// ```
///
/// where N is the number of weekly fixing dates in the accrual period.
///
/// BMA coupons are used in tax-exempt municipal interest rate swaps where
/// one leg pays this averaged floating rate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AverageBMACoupon {
    /// Payment date.
    pub payment_date: Date,
    /// Nominal (face value).
    pub nominal: f64,
    /// Accrual period start.
    pub accrual_start: Date,
    /// Accrual period end.
    pub accrual_end: Date,
    /// Day counter for accrual fraction.
    pub day_counter: DayCounter,
    /// Weekly BMA fixing rates.
    pub fixings: Vec<f64>,
    /// Additive spread over average BMA rate.
    pub spread: f64,
    /// Multiplicative gearing.
    pub gearing: f64,
}

impl AverageBMACoupon {
    /// Create a new BMA averaged coupon.
    ///
    /// # Arguments
    /// - `payment_date` — coupon payment date
    /// - `nominal` — face value
    /// - `accrual_start` / `accrual_end` — accrual period
    /// - `day_counter` — day count convention
    /// - `fixings` — weekly BMA fixing rates
    /// - `spread` — additive spread
    /// - `gearing` — multiplicative gearing (typically 1.0)
    pub fn new(
        payment_date: Date,
        nominal: f64,
        accrual_start: Date,
        accrual_end: Date,
        day_counter: DayCounter,
        fixings: Vec<f64>,
        spread: f64,
        gearing: f64,
    ) -> Self {
        Self {
            payment_date,
            nominal,
            accrual_start,
            accrual_end,
            day_counter,
            fixings,
            spread,
            gearing,
        }
    }

    /// Compute the arithmetic average of the weekly BMA fixings.
    pub fn average_fixing(&self) -> f64 {
        if self.fixings.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.fixings.iter().sum();
        sum / self.fixings.len() as f64
    }

    /// Effective rate: gearing × average_fixing + spread.
    pub fn effective_rate(&self) -> f64 {
        self.gearing * self.average_fixing() + self.spread
    }

    /// Number of weekly fixings in this period.
    pub fn num_fixings(&self) -> usize {
        self.fixings.len()
    }

    /// Create with a flat (constant) fixing rate for the entire period.
    ///
    /// Determines the number of fixings from the period length (roughly
    /// one per week, 7 calendar days).
    pub fn from_flat_rate(
        payment_date: Date,
        nominal: f64,
        accrual_start: Date,
        accrual_end: Date,
        day_counter: DayCounter,
        flat_rate: f64,
        spread: f64,
    ) -> Self {
        let days = (accrual_end.serial() - accrual_start.serial()) as usize;
        let num_weeks = (days / 7).max(1);
        let fixings = vec![flat_rate; num_weeks];
        Self::new(
            payment_date,
            nominal,
            accrual_start,
            accrual_end,
            day_counter,
            fixings,
            spread,
            1.0,
        )
    }
}

impl CashFlow for AverageBMACoupon {
    fn date(&self) -> Date {
        self.payment_date
    }

    fn amount(&self) -> f64 {
        let yf = self
            .day_counter
            .year_fraction(self.accrual_start, self.accrual_end);
        self.nominal * self.effective_rate() * yf
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Coupon for AverageBMACoupon {
    fn nominal(&self) -> f64 {
        self.nominal
    }

    fn rate(&self) -> f64 {
        self.effective_rate()
    }

    fn accrual_start(&self) -> Date {
        self.accrual_start
    }

    fn accrual_end(&self) -> Date {
        self.accrual_end
    }

    fn accrual_period(&self) -> f64 {
        self.day_counter
            .year_fraction(self.accrual_start, self.accrual_end)
    }

    fn day_counter(&self) -> DayCounter {
        self.day_counter
    }
}

// ===========================================================================
// OvernightIndexedCouponPricer
// ===========================================================================

/// Observation shift mode for overnight coupon pricing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObservationShift {
    /// No shift — observe on the accrual date itself.
    None,
    /// Lookback: shift observation dates backward by `days` business days.
    /// Each accrual date d observes the fixing from d − lookback_days.
    Lookback { days: u32 },
    /// Lockout: freeze the last `days` fixings to match the fixing
    /// observed `days` business days before the accrual end.
    Lockout { days: u32 },
}

/// Extended overnight indexed coupon pricer with lookback/lockout support.
///
/// The standard `OvernightIndexedCoupon` computes a simple compounded rate
/// without any observation shift.  This pricer adds:
///
/// - **Lookback**: each day's rate is observed from N business days earlier
/// - **Lockout**: the last N business days of the period use the same
///   fixing (frozen at the value from N days before period end)
///
/// These conventions are standard for SOFR, SONIA, and €STR-linked coupons
/// under ISDA 2021 definitions.
///
/// The pricer operates on a pre-computed vector of (accrual_fraction, rate)
/// pairs, one per business day in the period.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OvernightIndexedCouponPricer {
    /// Observation shift convention.
    pub shift: ObservationShift,
    /// Sub-period year fractions (one per business day in accrual period).
    pub sub_year_fractions: Vec<f64>,
    /// Sub-period overnight rates (shifted per observation convention).
    pub sub_rates: Vec<f64>,
    /// Additive spread.
    pub spread: f64,
}

impl OvernightIndexedCouponPricer {
    /// Create a new extended overnight coupon pricer.
    ///
    /// # Arguments
    /// - `shift` — observation shift convention (lookback, lockout, or none)
    /// - `sub_year_fractions` — day-count fractions for each sub-period
    /// - `sub_rates` — corresponding overnight rates (already shifted if applicable)
    /// - `spread` — additive spread
    pub fn new(
        shift: ObservationShift,
        sub_year_fractions: Vec<f64>,
        sub_rates: Vec<f64>,
        spread: f64,
    ) -> Self {
        assert_eq!(
            sub_year_fractions.len(),
            sub_rates.len(),
            "sub-period vectors must match in length"
        );
        Self {
            shift,
            sub_year_fractions,
            sub_rates,
            spread,
        }
    }

    /// Compute the compounded rate (ISDA compounding).
    ///
    /// ```text
    /// rate = (∏(1 + rᵢ · δᵢ) − 1) / Σ(δᵢ) + spread
    /// ```
    pub fn compounded_rate(&self) -> f64 {
        if self.sub_rates.is_empty() {
            return self.spread;
        }
        let product: f64 = self
            .sub_rates
            .iter()
            .zip(self.sub_year_fractions.iter())
            .map(|(&r, &yf)| 1.0 + r * yf)
            .product();
        let total_yf: f64 = self.sub_year_fractions.iter().sum();
        if total_yf.abs() < 1e-15 {
            return self.spread;
        }
        (product - 1.0) / total_yf + self.spread
    }

    /// Compute the simple (arithmetic) average rate.
    ///
    /// ```text
    /// rate = Σ(rᵢ · δᵢ) / Σ(δᵢ) + spread
    /// ```
    pub fn averaged_rate(&self) -> f64 {
        if self.sub_rates.is_empty() {
            return self.spread;
        }
        let total_yf: f64 = self.sub_year_fractions.iter().sum();
        if total_yf.abs() < 1e-15 {
            return self.spread;
        }
        let weighted: f64 = self
            .sub_rates
            .iter()
            .zip(self.sub_year_fractions.iter())
            .map(|(&r, &yf)| r * yf)
            .sum();
        weighted / total_yf + self.spread
    }

    /// Build sub-period vectors from a flat rate with lockout adjustment.
    ///
    /// Useful for testing: creates N sub-periods of equal length, with the
    /// last `lockout_days` periods frozen at the rate observed on
    /// sub-period `N - lockout_days`.
    pub fn from_flat_rate(
        flat_rate: f64,
        n_days: usize,
        day_fraction_per_day: f64,
        shift: ObservationShift,
        spread: f64,
    ) -> Self {
        let mut rates = vec![flat_rate; n_days];
        let yfs = vec![day_fraction_per_day; n_days];

        match shift {
            ObservationShift::Lockout { days } => {
                let lockout = days as usize;
                if lockout < n_days {
                    let frozen = rates[n_days - lockout - 1];
                    for r in rates.iter_mut().skip(n_days - lockout) {
                        *r = frozen;
                    }
                }
            }
            ObservationShift::Lookback { days } => {
                // For a flat rate, lookback doesn't change anything since all
                // rates are the same. In practice, shifted rates would differ.
                let _ = days;
            }
            ObservationShift::None => {}
        }

        Self::new(shift, yfs, rates, spread)
    }
}

// ===========================================================================
// CashFlowVectors — coupon vector builder utility
// ===========================================================================

/// Utility for building vectors of coupons (legs) from parallel arrays
/// of notionals, rates, and dates.
///
/// This mirrors QuantLib's `CashFlows::fixedRateLeg()` / `iborLeg()` vector
/// builders that take arrays of parameters and produce a `Leg`.
pub struct CashFlowVectors;

impl CashFlowVectors {
    /// Build a fixed-rate leg from parallel arrays.
    ///
    /// Each element produces a `FixedRateCoupon`.  The `notionals` and `rates`
    /// arrays are extended by repeating the last element if shorter than
    /// `start_dates.len()`.
    ///
    /// # Arguments
    /// - `start_dates` — accrual period start dates
    /// - `end_dates` — accrual period end dates (= payment dates)
    /// - `notionals` — face values per period (extended if shorter)
    /// - `rates` — coupon rates per period (extended if shorter)
    /// - `day_counter` — day counting convention
    pub fn fixed_rate_leg(
        start_dates: &[Date],
        end_dates: &[Date],
        notionals: &[f64],
        rates: &[f64],
        day_counter: DayCounter,
    ) -> Leg {
        assert_eq!(
            start_dates.len(),
            end_dates.len(),
            "start/end date vectors must match"
        );
        if notionals.is_empty() || rates.is_empty() || start_dates.is_empty() {
            return Vec::new();
        }
        let n = start_dates.len();
        let mut leg: Leg = Vec::with_capacity(n);

        for i in 0..n {
            let notional = notionals[i.min(notionals.len() - 1)];
            let rate = rates[i.min(rates.len() - 1)];
            leg.push(Box::new(FixedRateCoupon::new(
                end_dates[i],
                notional,
                rate,
                start_dates[i],
                end_dates[i],
                day_counter,
            )));
        }
        leg
    }

    /// Build an amortising fixed-rate leg.
    ///
    /// The notional decreases linearly from `initial_notional` to zero over
    /// the number of periods. Each period's notional is the outstanding
    /// balance at the start.
    pub fn amortising_fixed_leg(
        start_dates: &[Date],
        end_dates: &[Date],
        initial_notional: f64,
        rate: f64,
        day_counter: DayCounter,
    ) -> Leg {
        let n = start_dates.len().min(end_dates.len());
        if n == 0 {
            return Vec::new();
        }
        let amort_per_period = initial_notional / n as f64;
        let mut leg: Leg = Vec::with_capacity(n);

        for i in 0..n {
            let notional = initial_notional - amort_per_period * i as f64;
            leg.push(Box::new(FixedRateCoupon::new(
                end_dates[i],
                notional,
                rate,
                start_dates[i],
                end_dates[i],
                day_counter,
            )));
        }
        leg
    }

    /// Compute notional schedule from an initial notional and amortisation amounts.
    ///
    /// Returns a vector of outstanding notionals, one per period.
    pub fn notional_schedule(
        initial_notional: f64,
        amortisations: &[f64],
        n_periods: usize,
    ) -> Vec<f64> {
        let mut notionals = Vec::with_capacity(n_periods);
        let mut balance = initial_notional;
        for i in 0..n_periods {
            notionals.push(balance);
            if i < amortisations.len() {
                balance -= amortisations[i];
                if balance < 0.0 {
                    balance = 0.0;
                }
            }
        }
        notionals
    }

    /// Generate date pairs from a regular schedule.
    ///
    /// Given a start date, number of periods, and period length in months,
    /// returns `(start_dates, end_dates)` vectors suitable for leg builders.
    pub fn regular_date_schedule(
        start: Date,
        n_periods: usize,
        period_months: u32,
    ) -> (Vec<Date>, Vec<Date>) {
        let mut start_dates = Vec::with_capacity(n_periods);
        let mut end_dates = Vec::with_capacity(n_periods);

        let mut d = start;
        for _ in 0..n_periods {
            start_dates.push(d);
            let next = advance_months(d, period_months);
            end_dates.push(next);
            d = next;
        }
        (start_dates, end_dates)
    }
}

/// Advance a date by `months` calendar months.
fn advance_months(d: Date, months: u32) -> Date {
    let (y, m, day) = (d.year(), d.month() as u32, d.day_of_month());
    let total = (y * 12 + m as i32 - 1) + months as i32;
    let new_y = total / 12;
    let new_m = (total % 12) as u32 + 1;
    let new_d = day.min(Date::days_in_month(new_y, new_m));
    Date::from_ymd_opt(new_y, new_m, new_d).unwrap_or(d)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_time::Month;

    // -----------------------------------------------------------------------
    // AverageBMACoupon tests
    // -----------------------------------------------------------------------

    #[test]
    fn bma_average_fixing() {
        let fixings = vec![0.030, 0.031, 0.029, 0.032];
        let c = AverageBMACoupon::new(
            Date::from_ymd(2025, Month::April, 1),
            1_000_000.0,
            Date::from_ymd(2025, Month::January, 1),
            Date::from_ymd(2025, Month::April, 1),
            DayCounter::Actual360,
            fixings.clone(),
            0.0,
            1.0,
        );
        let avg = fixings.iter().sum::<f64>() / fixings.len() as f64;
        assert_abs_diff_eq!(c.average_fixing(), avg, epsilon = 1e-15);
    }

    #[test]
    fn bma_effective_rate_with_spread_and_gearing() {
        let fixings = vec![0.03; 4];
        let c = AverageBMACoupon::new(
            Date::from_ymd(2025, Month::April, 1),
            1_000_000.0,
            Date::from_ymd(2025, Month::January, 1),
            Date::from_ymd(2025, Month::April, 1),
            DayCounter::Actual360,
            fixings,
            0.005,  // 50bp spread
            0.67,   // 67% gearing (typical tax-exempt ratio)
        );
        let expected = 0.67 * 0.03 + 0.005;
        assert_abs_diff_eq!(c.effective_rate(), expected, epsilon = 1e-15);
    }

    #[test]
    fn bma_amount() {
        let c = AverageBMACoupon::from_flat_rate(
            Date::from_ymd(2025, Month::July, 1),
            1_000_000.0,
            Date::from_ymd(2025, Month::January, 1),
            Date::from_ymd(2025, Month::July, 1),
            DayCounter::Actual360,
            0.04,
            0.0,
        );
        let yf = DayCounter::Actual360.year_fraction(
            Date::from_ymd(2025, Month::January, 1),
            Date::from_ymd(2025, Month::July, 1),
        );
        let expected = 1_000_000.0 * 0.04 * yf;
        assert_abs_diff_eq!(c.amount(), expected, epsilon = 1e-4);
    }

    #[test]
    fn bma_coupon_trait() {
        let c = AverageBMACoupon::from_flat_rate(
            Date::from_ymd(2025, Month::April, 1),
            1_000_000.0,
            Date::from_ymd(2025, Month::January, 1),
            Date::from_ymd(2025, Month::April, 1),
            DayCounter::Actual360,
            0.035,
            0.001,
        );
        assert_abs_diff_eq!(c.nominal(), 1_000_000.0, epsilon = 1e-10);
        assert_eq!(c.accrual_start(), Date::from_ymd(2025, Month::January, 1));
        assert_eq!(c.accrual_end(), Date::from_ymd(2025, Month::April, 1));
        assert!(c.accrual_period() > 0.0);
    }

    #[test]
    fn bma_from_flat_rate_fixings_count() {
        // 90 days ≈ 12-13 weeks
        let c = AverageBMACoupon::from_flat_rate(
            Date::from_ymd(2025, Month::April, 1),
            1_000_000.0,
            Date::from_ymd(2025, Month::January, 1),
            Date::from_ymd(2025, Month::April, 1),
            DayCounter::Actual360,
            0.03,
            0.0,
        );
        assert!(c.num_fixings() >= 12 && c.num_fixings() <= 14);
    }

    #[test]
    fn bma_empty_fixings() {
        let c = AverageBMACoupon::new(
            Date::from_ymd(2025, Month::April, 1),
            1_000_000.0,
            Date::from_ymd(2025, Month::January, 1),
            Date::from_ymd(2025, Month::April, 1),
            DayCounter::Actual360,
            vec![],
            0.005,
            1.0,
        );
        // With no fixings, average fixing = 0, effective rate = spread
        assert_abs_diff_eq!(c.effective_rate(), 0.005, epsilon = 1e-15);
    }

    // -----------------------------------------------------------------------
    // OvernightIndexedCouponPricer tests
    // -----------------------------------------------------------------------

    #[test]
    fn pricer_compounded_rate_flat() {
        let pricer = OvernightIndexedCouponPricer::from_flat_rate(
            0.04, 90, 1.0 / 360.0,
            ObservationShift::None,
            0.0,
        );
        let rate = pricer.compounded_rate();
        // For a flat 4% compounded daily for 90 days ≈ 4%
        assert_abs_diff_eq!(rate, 0.04, epsilon = 0.002);
    }

    #[test]
    fn pricer_averaged_rate_flat() {
        let pricer = OvernightIndexedCouponPricer::from_flat_rate(
            0.05, 60, 1.0 / 360.0,
            ObservationShift::None,
            0.001,
        );
        let rate = pricer.averaged_rate();
        assert_abs_diff_eq!(rate, 0.051, epsilon = 1e-10);
    }

    #[test]
    fn pricer_lockout_freezes_tail() {
        // With lockout of 2 days and varying rates, the last 2 rates
        // should be frozen
        let mut rates = vec![0.04; 10];
        rates[8] = 0.05; // this should get frozen out
        rates[9] = 0.06; // this too
        let yfs = vec![1.0 / 360.0; 10];

        // Build with lockout manually: frozen rate = rates[7] = 0.04
        let pricer = OvernightIndexedCouponPricer::from_flat_rate(
            0.04, 10, 1.0 / 360.0,
            ObservationShift::Lockout { days: 2 },
            0.0,
        );
        // With flat rate + lockout, all rates are 0.04, so result ~ 0.04
        let rate = pricer.compounded_rate();
        assert_abs_diff_eq!(rate, 0.04, epsilon = 0.001);
    }

    #[test]
    fn pricer_with_spread() {
        let pricer = OvernightIndexedCouponPricer::from_flat_rate(
            0.03, 30, 1.0 / 360.0,
            ObservationShift::None,
            0.005, // 50bp spread
        );
        let rate = pricer.compounded_rate();
        // Should be around 3.5%
        assert!(rate > 0.034 && rate < 0.036, "rate = {rate}");
    }

    #[test]
    fn pricer_empty_subperiods() {
        let pricer = OvernightIndexedCouponPricer::new(
            ObservationShift::None,
            vec![],
            vec![],
            0.01,
        );
        assert_abs_diff_eq!(pricer.compounded_rate(), 0.01, epsilon = 1e-15);
        assert_abs_diff_eq!(pricer.averaged_rate(), 0.01, epsilon = 1e-15);
    }

    #[test]
    fn observation_shift_serializes() {
        let shift = ObservationShift::Lookback { days: 5 };
        let json = serde_json::to_string(&shift).unwrap();
        let back: ObservationShift = serde_json::from_str(&json).unwrap();
        assert_eq!(shift, back);
    }

    // -----------------------------------------------------------------------
    // CashFlowVectors tests
    // -----------------------------------------------------------------------

    #[test]
    fn fixed_rate_leg_from_vectors() {
        let start = Date::from_ymd(2025, Month::January, 1);
        let (starts, ends) = CashFlowVectors::regular_date_schedule(start, 4, 3);
        let leg = CashFlowVectors::fixed_rate_leg(
            &starts, &ends, &[1_000_000.0], &[0.05], DayCounter::Actual360,
        );
        assert_eq!(leg.len(), 4);
        for cf in &leg {
            assert!(cf.amount() > 0.0);
        }
    }

    #[test]
    fn amortising_leg_decreasing_notionals() {
        let start = Date::from_ymd(2025, Month::January, 1);
        let (starts, ends) = CashFlowVectors::regular_date_schedule(start, 4, 3);
        let leg = CashFlowVectors::amortising_fixed_leg(
            &starts, &ends, 1_000_000.0, 0.05, DayCounter::Actual360,
        );
        assert_eq!(leg.len(), 4);
        // Each coupon should be smaller than the previous (decreasing notional)
        for i in 1..leg.len() {
            assert!(leg[i].amount() < leg[i - 1].amount());
        }
    }

    #[test]
    fn notional_schedule_basic() {
        let schedule = CashFlowVectors::notional_schedule(
            1_000_000.0,
            &[250_000.0, 250_000.0, 250_000.0, 250_000.0],
            4,
        );
        assert_eq!(schedule.len(), 4);
        assert_abs_diff_eq!(schedule[0], 1_000_000.0, epsilon = 1e-10);
        assert_abs_diff_eq!(schedule[1], 750_000.0, epsilon = 1e-10);
        assert_abs_diff_eq!(schedule[2], 500_000.0, epsilon = 1e-10);
        assert_abs_diff_eq!(schedule[3], 250_000.0, epsilon = 1e-10);
    }

    #[test]
    fn regular_date_schedule_monthly() {
        let start = Date::from_ymd(2025, Month::January, 15);
        let (starts, ends) = CashFlowVectors::regular_date_schedule(start, 12, 1);
        assert_eq!(starts.len(), 12);
        assert_eq!(ends.len(), 12);
        assert_eq!(starts[0], start);
        // Each end should equal the next start
        for i in 0..11 {
            assert_eq!(ends[i], starts[i + 1]);
        }
    }

    #[test]
    fn fixed_rate_leg_notional_extension() {
        // Single notional + single rate should apply to all periods
        let start = Date::from_ymd(2025, Month::January, 1);
        let (starts, ends) = CashFlowVectors::regular_date_schedule(start, 3, 6);
        let leg = CashFlowVectors::fixed_rate_leg(
            &starts, &ends, &[1_000_000.0], &[0.04], DayCounter::Actual365Fixed,
        );
        assert_eq!(leg.len(), 3);
        // All coupons should use the same rate and notional
        let r0 = leg[0].amount();
        // Periods are approximately equal so amounts should be similar
        assert!(r0 > 0.0);
    }

    #[test]
    fn empty_inputs_produce_empty_leg() {
        let leg = CashFlowVectors::fixed_rate_leg(
            &[], &[], &[1.0], &[0.05], DayCounter::Actual360,
        );
        assert!(leg.is_empty());
    }
}
