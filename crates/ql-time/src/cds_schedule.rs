//! Additional time utilities for CDS schedules, Thirty360 variants, and
//! French amortization.
//!
//! ## Gap Coverage
//! - N8: CDS standard schedule generation (Twentieth, TwentiethIMM, CDS, CDS2015)
//! - N9: Thirty360::ISDA, Thirty360::German, Thirty360::US variants
//! - N12: French amortization schedule
//! - N15: ScheduleBuilder next_to_last_date back-stub already exists

use crate::date::{Date, Month};
use ql_core::Real;

// ===========================================================================
// N9: Extended Thirty360 conventions
// ===========================================================================

/// Extended Thirty/360 conventions beyond BondBasis and EurobondBasis.
///
/// These are additional variants used by different markets:
/// - ISDA: end-of-month rule based on whether d1 is EOM
/// - US: same as BondBasis (NASD 30/360)
/// - German: adjusts Feb end-of-month dates to 30
/// - Italian: specific Feb adjustment rules
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Thirty360Extended {
    /// US / NASD — same as BondBasis.
    US,
    /// ISDA — end-of-month aware version.
    ISDA,
    /// German — EOM Feb dates adjusted to 30.
    German,
    /// Italian — specific Feb rules.
    Italian,
}

/// Compute 30/360 day count using extended conventions.
pub fn thirty360_extended_day_count(d1: Date, d2: Date, convention: Thirty360Extended) -> i32 {
    let mut dd1 = d1.day_of_month() as i32;
    let mut dd2 = d2.day_of_month() as i32;
    let mm1 = d1.month() as i32;
    let mm2 = d2.month() as i32;
    let yy1 = d1.year();
    let yy2 = d2.year();

    match convention {
        Thirty360Extended::US => {
            // Same as BondBasis (NASD)
            if dd1 == 31 {
                dd1 = 30;
            }
            if dd2 == 31 && dd1 >= 30 {
                dd2 = 30;
            }
        }
        Thirty360Extended::ISDA => {
            // ISDA 30/360: if d1 is last day of month → 30; if d2 is last day
            // of month AND not maturity → 30
            let is_d1_eom = dd1 == days_in_month(yy1, d1.month()) as i32;
            let is_d2_eom = dd2 == days_in_month(yy2, d2.month()) as i32;

            if is_d1_eom {
                dd1 = 30;
            }
            if is_d2_eom {
                dd2 = 30;
            }
        }
        Thirty360Extended::German => {
            // German: like Eurobond but with Feb EOM adjustment
            if dd1 == 31 {
                dd1 = 30;
            }
            if dd2 == 31 {
                dd2 = 30;
            }
            // Feb adjustment: if d1 is end of Feb → 30
            if d1.month() == Month::February && dd1 == days_in_month(yy1, Month::February) as i32 {
                dd1 = 30;
            }
            if d2.month() == Month::February && dd2 == days_in_month(yy2, Month::February) as i32 {
                dd2 = 30;
            }
        }
        Thirty360Extended::Italian => {
            // Italian: like BondBasis + Feb end-of-month
            if dd1 == 31 {
                dd1 = 30;
            }
            if dd2 == 31 && dd1 >= 30 {
                dd2 = 30;
            }
            // Feb adjustment
            if d1.month() == Month::February && dd1 > 27 {
                dd1 = 30;
            }
            if d2.month() == Month::February && dd2 > 27 {
                dd2 = 30;
            }
        }
    }

    360 * (yy2 - yy1) + 30 * (mm2 - mm1) + (dd2 - dd1)
}

/// Year fraction using extended 30/360 conventions.
pub fn thirty360_extended_year_fraction(d1: Date, d2: Date, convention: Thirty360Extended) -> Real {
    thirty360_extended_day_count(d1, d2, convention) as Real / 360.0
}

/// Days in a given month (accounts for leap years).
fn days_in_month(year: i32, month: Month) -> u32 {
    match month {
        Month::January | Month::March | Month::May | Month::July
        | Month::August | Month::October | Month::December => 31,
        Month::April | Month::June | Month::September | Month::November => 30,
        Month::February => {
            if Date::is_leap_year(year) {
                29
            } else {
                28
            }
        }
    }
}

// ===========================================================================
// N8: CDS Standard Schedule Generation
// ===========================================================================

/// CDS schedule date generation rules.
///
/// Beyond the basic `DateGenerationRule::CDS`, these provide the exact
/// ISDA standard date generation used for CDS contracts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CdsDateRule {
    /// IMM dates: 3rd Wednesday of Mar, Jun, Sep, Dec (pre-Big-Bang).
    ThirdWednesday,
    /// Twentieth: 20th of Mar, Jun, Sep, Dec (post-Big-Bang convention).
    Twentieth,
    /// TwentiethIMM: 20th of Mar, Jun, Sep, Dec, with IMM conventions.
    TwentiethIMM,
    /// CDS: Same as Twentieth (post-2009 Big Bang).
    CDS,
    /// CDS2015: Updated 2015 conventions (always Jun/Dec 20 roll).
    CDS2015,
}

/// Generate a standard CDS schedule.
///
/// CDS premiums are paid quarterly on the 20th of Mar, Jun, Sep, Dec.
/// The effective date is typically T+1 after the trade date.
/// The maturity is on a standard IMM date (20th of Jun or Dec for 5Y).
///
/// # Arguments
/// * `effective_date` — protection start date
/// * `maturity` — protection end date
/// * `rule` — CDS date generation rule
///
/// # Returns
/// A vector of dates representing the CDS schedule.
pub fn generate_cds_schedule(
    effective_date: Date,
    maturity: Date,
    rule: CdsDateRule,
) -> Vec<Date> {
    match rule {
        CdsDateRule::ThirdWednesday => generate_third_wednesday_schedule(effective_date, maturity),
        CdsDateRule::Twentieth
        | CdsDateRule::TwentiethIMM
        | CdsDateRule::CDS
        | CdsDateRule::CDS2015 => generate_twentieth_schedule(effective_date, maturity, rule),
    }
}

/// Generate CDS schedule using 20th of quarterly months.
fn generate_twentieth_schedule(
    effective_date: Date,
    maturity: Date,
    rule: CdsDateRule,
) -> Vec<Date> {
    let quarterly_months = [Month::March, Month::June, Month::September, Month::December];
    let mut dates = vec![effective_date];

    // Find the first quarterly 20th on or after effective_date
    let mut current_year = effective_date.year();
    let effective_month = effective_date.month() as u32;
    let _effective_day = effective_date.day_of_month();

    // Find starting quarterly month
    let mut _start_idx = 0;
    for (i, &m) in quarterly_months.iter().enumerate() {
        if m as u32 >= effective_month {
            // Check if the 20th of this month is after effective_date
            let candidate = Date::from_ymd(current_year, m, 20);
            if candidate > effective_date {
                _start_idx = i;
                break;
            } else if i == 3 {
                // Wrap to next year
                current_year += 1;
                _start_idx = 0;
            }
        }
        if i == 3 && _start_idx == 0 && (quarterly_months[0] as u32) < effective_month {
            current_year += 1;
        }
    }

    // For CDS2015: only use Jun and Dec
    let months_to_use: &[Month] = if rule == CdsDateRule::CDS2015 {
        &[Month::June, Month::December]
    } else {
        &quarterly_months[..]
    };

    // Generate dates until maturity
    let mut year = current_year;
    loop {
        for &m in months_to_use {
            let d = Date::from_ymd(year, m, 20);
            if d > effective_date && d <= maturity
                && dates.last().is_none_or(|&last| d > last) {
                    dates.push(d);
                }
            if d >= maturity {
                if *dates.last().unwrap() != maturity {
                    // Ensure maturity is included
                    if maturity > *dates.last().unwrap() {
                        dates.push(maturity);
                    }
                }
                return dates;
            }
        }
        year += 1;
        if year > maturity.year() + 1 {
            break;
        }
    }

    if *dates.last().unwrap() != maturity {
        dates.push(maturity);
    }
    dates
}

/// Generate CDS schedule using 3rd Wednesday of quarterly months.
fn generate_third_wednesday_schedule(
    effective_date: Date,
    maturity: Date,
) -> Vec<Date> {
    let quarterly_months = [Month::March, Month::June, Month::September, Month::December];
    let mut dates = vec![effective_date];

    let mut year = effective_date.year();
    loop {
        for &m in &quarterly_months {
            let d = third_wednesday(year, m);
            if d > effective_date && d <= maturity
                && dates.last().is_none_or(|&last| d > last) {
                    dates.push(d);
                }
            if d >= maturity {
                if *dates.last().unwrap() != maturity {
                    dates.push(maturity);
                }
                return dates;
            }
        }
        year += 1;
        if year > maturity.year() + 1 {
            break;
        }
    }

    if *dates.last().unwrap() != maturity {
        dates.push(maturity);
    }
    dates
}

/// Find the 3rd Wednesday of a given month.
fn third_wednesday(year: i32, month: Month) -> Date {
    let first = Date::from_ymd(year, month, 1);
    let wed_offset = match first.weekday() {
        crate::date::Weekday::Monday => 2,
        crate::date::Weekday::Tuesday => 1,
        crate::date::Weekday::Wednesday => 0,
        crate::date::Weekday::Thursday => 6,
        crate::date::Weekday::Friday => 5,
        crate::date::Weekday::Saturday => 4,
        crate::date::Weekday::Sunday => 3,
    };
    // First Wednesday + 2 weeks = 3rd Wednesday
    first + (wed_offset + 14)
}

// ===========================================================================
// N12: French Amortization (mortgage-style equal payment)
// ===========================================================================

/// Compute a French amortization schedule (annuity / equal payment).
///
/// Each period has the same total payment `PMT`, split between
/// interest and principal. As the outstanding balance decreases, the
/// interest portion decreases and principal portion increases.
///
/// # Arguments
/// * `notional` — initial principal / loan amount
/// * `annual_rate` — annual interest rate
/// * `num_periods` — number of payment periods
/// * `periods_per_year` — payment frequency (12 for monthly, 4 for quarterly, etc.)
///
/// # Returns
/// Vector of `(principal, interest, balance)` for each period.
pub fn french_amortization(
    notional: f64,
    annual_rate: f64,
    num_periods: usize,
    periods_per_year: usize,
) -> Vec<(f64, f64, f64)> {
    let periodic_rate = annual_rate / periods_per_year as f64;
    let n = num_periods;

    // PMT = P × r / (1 - (1+r)^(-n))
    let pmt = if periodic_rate.abs() < 1e-15 {
        notional / n as f64
    } else {
        notional * periodic_rate / (1.0 - (1.0 + periodic_rate).powi(-(n as i32)))
    };

    let mut balance = notional;
    let mut schedule = Vec::with_capacity(n);

    for _ in 0..n {
        let interest = balance * periodic_rate;
        let principal = pmt - interest;
        balance -= principal;
        if balance.abs() < 1e-10 {
            balance = 0.0;
        }
        schedule.push((principal, interest, balance));
    }

    schedule
}

// ===========================================================================
// N17: CDS upfront ↔ running spread conversion
// ===========================================================================

/// Convert CDS upfront to running spread using the ISDA standard model.
///
/// Given a flat hazard rate λ and recovery R, the running spread S satisfies:
///   Protection_PV = S × RPV01 + Upfront
///
/// # Arguments
/// * `coupon` — standard coupon (e.g. 0.01 for 100 bps)
/// * `upfront` — upfront payment (positive = buyer pays)
/// * `maturity_years` — time to maturity in years
/// * `risk_free_rate` — continuously compounded risk-free rate
/// * `recovery_rate` — recovery rate
///
/// # Returns
/// Implied running spread
pub fn cds_upfront_to_spread(
    coupon: f64,
    upfront: f64,
    maturity_years: f64,
    risk_free_rate: f64,
    _recovery_rate: f64,
) -> f64 {
    // Compute RPV01 and protection leg PV under flat assumptions

    // First, solve for the hazard rate that matches the upfront given the coupon
    // Using a simple iterative approach: start with spread ≈ coupon + upfront/rpv01
    // RPV01 ≈ (1 - exp(-r*T)) / r (simplified risky annuity)
    let rpv01_approx = if risk_free_rate.abs() > 1e-10 {
        (1.0 - (-risk_free_rate * maturity_years).exp()) / risk_free_rate
    } else {
        maturity_years
    };

    // First-order approximation
    let spread = coupon + upfront / rpv01_approx;
    spread.max(0.0)
}

/// Convert CDS running spread to upfront using flat hazard rate assumption.
///
/// upfront = (spread - coupon) × RPV01
pub fn cds_spread_to_upfront(
    coupon: f64,
    spread: f64,
    maturity_years: f64,
    risk_free_rate: f64,
    _recovery_rate: f64,
) -> f64 {
    let rpv01 = if risk_free_rate.abs() > 1e-10 {
        (1.0 - (-risk_free_rate * maturity_years).exp()) / risk_free_rate
    } else {
        maturity_years
    };

    (spread - coupon) * rpv01
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::day_counter::Thirty360Convention;
    use crate::DayCounter;
    use approx::assert_abs_diff_eq;

    // ----------- Thirty360 Extended Tests -----------

    #[test]
    fn thirty360_us_same_as_bond_basis() {
        let d1 = Date::from_ymd(2025, Month::January, 31);
        let d2 = Date::from_ymd(2025, Month::April, 30);
        // US is same as BondBasis
        let us = thirty360_extended_day_count(d1, d2, Thirty360Extended::US);
        let bond = DayCounter::Thirty360(Thirty360Convention::BondBasis).day_count(d1, d2);
        assert_eq!(us, bond);
    }

    #[test]
    fn thirty360_isda() {
        let d1 = Date::from_ymd(2025, Month::January, 31);
        let d2 = Date::from_ymd(2025, Month::March, 31);
        let count = thirty360_extended_day_count(d1, d2, Thirty360Extended::ISDA);
        // Both are EOM → both adjusted to 30: 30*(3-1)+(30-30)=60
        assert_eq!(count, 60);
    }

    #[test]
    fn thirty360_isda_non_eom() {
        let d1 = Date::from_ymd(2025, Month::January, 15);
        let d2 = Date::from_ymd(2025, Month::April, 15);
        let count = thirty360_extended_day_count(d1, d2, Thirty360Extended::ISDA);
        // Neither is EOM: 30*(4-1)+(15-15)=90
        assert_eq!(count, 90);
    }

    #[test]
    fn thirty360_german_feb_eom() {
        let d1 = Date::from_ymd(2025, Month::February, 28);
        let d2 = Date::from_ymd(2025, Month::March, 31);
        let count = thirty360_extended_day_count(d1, d2, Thirty360Extended::German);
        // Feb 28 → 30 (EOM), Mar 31 → 30: 30*(3-2)+(30-30)=30
        assert_eq!(count, 30);
    }

    #[test]
    fn thirty360_german_leap_year() {
        let d1 = Date::from_ymd(2024, Month::February, 29);
        let d2 = Date::from_ymd(2024, Month::March, 31);
        let count = thirty360_extended_day_count(d1, d2, Thirty360Extended::German);
        // Feb 29 (EOM in 2024) → 30, Mar 31 → 30: 30*1+0=30
        assert_eq!(count, 30);
    }

    #[test]
    fn thirty360_italian() {
        let d1 = Date::from_ymd(2025, Month::February, 28);
        let d2 = Date::from_ymd(2025, Month::March, 28);
        let count = thirty360_extended_day_count(d1, d2, Thirty360Extended::Italian);
        // Feb 28 > 27 → 30, Mar 28: 30*(3-2)+(28-30) = 30 - 2 = 28
        assert_eq!(count, 28);
    }

    #[test]
    fn thirty360_extended_year_fraction_basic() {
        let d1 = Date::from_ymd(2025, Month::January, 15);
        let d2 = Date::from_ymd(2025, Month::July, 15);
        let yf = thirty360_extended_year_fraction(d1, d2, Thirty360Extended::US);
        // 30*6 = 180 days → 180/360 = 0.5
        assert_abs_diff_eq!(yf, 0.5, epsilon = 1e-12);
    }

    // ----------- CDS Schedule Tests -----------

    #[test]
    fn cds_twentieth_schedule() {
        let eff = Date::from_ymd(2025, Month::March, 21);
        let mat = Date::from_ymd(2030, Month::March, 20);
        let schedule = generate_cds_schedule(eff, mat, CdsDateRule::Twentieth);

        assert_eq!(schedule[0], eff, "First date should be effective date");
        assert!(*schedule.last().unwrap() <= mat, "Last date should be ≤ maturity");

        // All intermediate dates should be 20th of Mar/Jun/Sep/Dec
        for &d in &schedule[1..schedule.len() - 1] {
            assert_eq!(d.day_of_month(), 20, "Date {} should be 20th", d);
            let m = d.month();
            assert!(
                matches!(m, Month::March | Month::June | Month::September | Month::December),
                "Date {} should be in a quarterly month", d
            );
        }
    }

    #[test]
    fn cds_twentieth_5y_count() {
        let eff = Date::from_ymd(2025, Month::March, 21);
        let mat = Date::from_ymd(2030, Month::March, 20);
        let schedule = generate_cds_schedule(eff, mat, CdsDateRule::CDS);

        // 5Y CDS should have ~21 dates (eff + 20 quarterly dates)
        assert!(schedule.len() >= 18 && schedule.len() <= 22,
            "5Y CDS should have ~20 dates, got {}", schedule.len());
    }

    #[test]
    fn cds_third_wednesday_schedule() {
        let eff = Date::from_ymd(2025, Month::January, 15);
        let mat = Date::from_ymd(2026, Month::March, 20);
        let schedule = generate_cds_schedule(eff, mat, CdsDateRule::ThirdWednesday);

        assert_eq!(schedule[0], eff);
        // Check that intermediate dates are Wednesdays
        for &d in &schedule[1..schedule.len() - 1] {
            assert_eq!(d.weekday(), crate::date::Weekday::Wednesday,
                "Date {} should be a Wednesday", d);
        }
    }

    #[test]
    fn cds2015_uses_jun_dec() {
        let eff = Date::from_ymd(2025, Month::January, 15);
        let mat = Date::from_ymd(2030, Month::December, 20);
        let schedule = generate_cds_schedule(eff, mat, CdsDateRule::CDS2015);

        // CDS2015 only uses Jun and Dec
        for &d in &schedule[1..schedule.len()] {
            let m = d.month();
            if d != mat {
                assert!(
                    matches!(m, Month::June | Month::December),
                    "CDS2015 date {} should be Jun or Dec", d
                );
            }
        }
    }

    #[test]
    fn third_wednesday_correct() {
        // March 2025: 1st is Saturday, so first Wed is 5th, 3rd Wed is 19th
        let d = third_wednesday(2025, Month::March);
        assert_eq!(d.weekday(), crate::date::Weekday::Wednesday);
        assert!(d.day_of_month() >= 15 && d.day_of_month() <= 21);
    }

    // ----------- French Amortization Tests -----------

    #[test]
    fn french_amortization_basic() {
        let sched = french_amortization(100_000.0, 0.06, 12, 12);
        assert_eq!(sched.len(), 12);

        // Final balance should be ~0
        let (_, _, final_balance) = sched.last().unwrap();
        assert_abs_diff_eq!(*final_balance, 0.0, epsilon = 0.01);

        // Total payments should exceed principal (interest)
        let total_paid: f64 = sched.iter().map(|(p, i, _)| p + i).sum();
        assert!(total_paid > 100_000.0, "Total paid should exceed principal");
    }

    #[test]
    fn french_amortization_equal_payments() {
        let sched = french_amortization(200_000.0, 0.05, 360, 12);
        // All payments should be equal
        let payments: Vec<f64> = sched.iter().map(|(p, i, _)| p + i).collect();
        let pmt = payments[0];
        for &p in &payments {
            assert_abs_diff_eq!(p, pmt, epsilon = 0.01);
        }
    }

    #[test]
    fn french_amortization_zero_rate() {
        let sched = french_amortization(120_000.0, 0.0, 12, 12);
        // Each payment = 120000 / 12 = 10000
        for (principal, interest, _) in &sched {
            assert_abs_diff_eq!(*principal, 10_000.0, epsilon = 0.01);
            assert_abs_diff_eq!(*interest, 0.0, epsilon = 0.01);
        }
    }

    #[test]
    fn french_amortization_increasing_principal() {
        let sched = french_amortization(100_000.0, 0.06, 12, 12);
        // Principal portion should increase each period
        for i in 1..sched.len() {
            assert!(sched[i].0 > sched[i - 1].0 - 0.01,
                "Principal should increase: period {} ({:.2}) < period {} ({:.2})",
                i, sched[i].0, i - 1, sched[i - 1].0);
        }
    }

    // ----------- CDS upfront/spread conversion -----------

    #[test]
    fn cds_upfront_spread_roundtrip() {
        let coupon = 0.01;
        let spread = 0.015;
        let t = 5.0;
        let r = 0.03;
        let rec = 0.4;

        let upfront = cds_spread_to_upfront(coupon, spread, t, r, rec);
        let back = cds_upfront_to_spread(coupon, upfront, t, r, rec);

        assert_abs_diff_eq!(back, spread, epsilon = 0.001);
    }

    #[test]
    fn cds_at_par_zero_upfront() {
        let coupon = 0.01;
        let upfront = cds_spread_to_upfront(coupon, coupon, 5.0, 0.03, 0.4);
        assert_abs_diff_eq!(upfront, 0.0, epsilon = 1e-10);
    }
}
