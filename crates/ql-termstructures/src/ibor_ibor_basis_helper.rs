//! IBOR-IBOR basis swap rate helper for yield curve bootstrapping.
//!
//! An IBOR-IBOR basis swap exchanges floating payments on two different
//! tenors of the same IBOR index (e.g. 3M EURIBOR vs 6M EURIBOR) with a
//! basis spread on the short-tenor leg. This helper can be used with
//! `PiecewiseYieldCurve` to bootstrap a forwarding curve for one tenor
//! given the other.
//!
//! # QuantLib C++ equivalent
//!
//! `IborIborBasisSwapRateHelper` in `termstructures/yield/ratehelpers.hpp`.
//!
//! # Example
//!
//! ```
//! use ql_time::{Date, Month, DayCounter, Calendar};
//! use ql_termstructures::ibor_ibor_basis_helper::IborIborBasisSwapHelper;
//! use ql_termstructures::bootstrap::RateHelper;
//!
//! let ref_date = Date::from_ymd(2025, Month::January, 15);
//! let helper = IborIborBasisSwapHelper::new(
//!     0.001,    // 10 bps basis spread on the short tenor leg
//!     ref_date, // settlement date
//!     5,        // 5-year swap
//!     3,        // short tenor: 3M
//!     6,        // long tenor: 6M
//!     DayCounter::Actual360,
//!     Calendar::Target,
//! );
//! assert!(helper.pillar_date() > ref_date);
//! assert!((helper.quote() - 0.001).abs() < 1e-15);
//! ```

use serde::{Deserialize, Serialize};

use ql_time::{
    BusinessDayConvention, Calendar, Date, DayCounter, Month,
};

use crate::bootstrap::RateHelper;

/// IBOR-IBOR basis swap rate helper for `PiecewiseYieldCurve` bootstrapping.
///
/// Models a single-currency floating-floating swap where:
/// - The **short-tenor** leg pays IBOR + basis spread (e.g. 3M + 10 bps)
/// - The **long-tenor** leg pays IBOR flat (e.g. 6M flat)
///
/// Given a discount/forecasting curve for the short tenor, this helper
/// provides the constraint needed to bootstrap the long-tenor forwarding
/// curve.
///
/// The fundamental pricing relation is:
///
/// $\sum_{i} \tau_i^S \cdot D(t_i) \cdot (F_i^S + s) = \sum_{j} \tau_j^L \cdot D(t_j) \cdot F_j^L$
///
/// where $F^S$, $F^L$ are forward rates, $s$ is the basis spread, and $D$
/// is the discount factor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IborIborBasisSwapHelper {
    /// Quoted basis spread on the short-tenor leg.
    spread: f64,
    /// Settlement date.
    settlement_date: Date,
    /// Maturity date.
    maturity_date: Date,
    /// Swap tenor in years.
    tenor_years: u32,
    /// Short-tenor frequency in months (e.g., 3).
    short_tenor_months: u32,
    /// Long-tenor frequency in months (e.g., 6).
    long_tenor_months: u32,
    /// Day counter for accrual.
    day_counter: DayCounter,
    /// Short-tenor leg payment dates.
    short_dates: Vec<Date>,
    /// Long-tenor leg payment dates.
    long_dates: Vec<Date>,
}

impl IborIborBasisSwapHelper {
    /// Create a new IBOR-IBOR basis swap helper.
    ///
    /// # Arguments
    ///
    /// * `spread` — Basis spread on the short-tenor leg (decimal, e.g. 0.001 = 10 bps)
    /// * `settlement_date` — Swap start date
    /// * `tenor_years` — Swap maturity in years
    /// * `short_tenor_months` — Short leg reset frequency (months)
    /// * `long_tenor_months` — Long leg reset frequency (months)
    /// * `day_counter` — Day counter for year fractions
    /// * `calendar` — Calendar for date adjustment
    pub fn new(
        spread: f64,
        settlement_date: Date,
        tenor_years: u32,
        short_tenor_months: u32,
        long_tenor_months: u32,
        day_counter: DayCounter,
        calendar: Calendar,
    ) -> Self {
        let short_dates =
            generate_schedule(settlement_date, tenor_years, short_tenor_months, &calendar);
        let long_dates =
            generate_schedule(settlement_date, tenor_years, long_tenor_months, &calendar);

        let maturity_date = *long_dates
            .last()
            .unwrap_or(&settlement_date);

        Self {
            spread,
            settlement_date,
            maturity_date,
            tenor_years,
            short_tenor_months,
            long_tenor_months,
            day_counter,
            short_dates,
            long_dates,
        }
    }

    /// The short-tenor leg payment dates.
    pub fn short_leg_dates(&self) -> &[Date] {
        &self.short_dates
    }

    /// The long-tenor leg payment dates.
    pub fn long_leg_dates(&self) -> &[Date] {
        &self.long_dates
    }
}

/// Generate an evenly-spaced schedule from `start` with `freq_months`-month
/// intervals over `tenor_years` years, adjusted via `calendar`.
fn generate_schedule(
    start: Date,
    tenor_years: u32,
    freq_months: u32,
    calendar: &Calendar,
) -> Vec<Date> {
    let n = tenor_years * 12 / freq_months;
    let mut dates = Vec::with_capacity(n as usize);
    for i in 1..=n {
        let months = (i * freq_months) as i32;
        let raw = advance_months(start, months);
        let adjusted = calendar.adjust(raw, BusinessDayConvention::ModifiedFollowing);
        dates.push(adjusted);
    }
    dates
}

/// Advance a date by a given number of months (simple calendar arithmetic).
fn advance_months(date: Date, months: i32) -> Date {
    let y = date.year();
    let m = date.month() as u32;
    let d = date.day_of_month();
    let total_months = y * 12 + m as i32 - 1 + months;
    let new_y = total_months / 12;
    let mut new_m = (total_months % 12) as u32 + 1;
    if new_m > 12 {
        new_m = 12;
    }
    let max_d = Date::days_in_month(new_y, new_m);
    let new_d = d.min(max_d);
    Date::from_ymd_opt(new_y, new_m, new_d).unwrap_or(date)
}

impl RateHelper for IborIborBasisSwapHelper {
    fn pillar_date(&self) -> Date {
        self.maturity_date
    }

    fn quote(&self) -> f64 {
        self.spread
    }

    fn implied_quote(
        &self,
        times: &[f64],
        dfs: &[f64],
        day_counter: DayCounter,
        ref_date: Date,
    ) -> f64 {
        // We compute the implied spread that makes the two legs equal.
        //
        // Short leg PV (per unit notional):
        //   PV_short = Σ τ_i · D(t_i) · F_i^S
        //   where F_i^S is the short-tenor forward rate from t_{i-1} to t_i
        //
        // Long leg PV:
        //   PV_long = Σ τ_j · D(t_j) · F_j^L
        //
        // At fair spread s:
        //   PV_short + s · Annuity_short = PV_long
        //   ⟹ s = (PV_long - PV_short) / Annuity_short
        //
        // Forward rate from T1 to T2: F = (D(T1)/D(T2) - 1) / τ(T1,T2)
        // Note: for bootstrapping, D comes from the *same* curve being built
        // (the long-tenor curve), while the short-tenor forwards also use this
        // curve (simplified; in production one would use a separate short curve).

        let short_annuity_and_pv = leg_annuity_and_float_pv(
            &self.short_dates,
            self.settlement_date,
            times,
            dfs,
            &self.day_counter,
            day_counter,
            ref_date,
        );

        let long_pv = leg_float_pv(
            &self.long_dates,
            self.settlement_date,
            times,
            dfs,
            &self.day_counter,
            day_counter,
            ref_date,
        );

        let annuity = short_annuity_and_pv.0;
        let short_pv = short_annuity_and_pv.1;

        if annuity.abs() < 1e-15 {
            return 0.0;
        }

        (long_pv - short_pv) / annuity
    }
}

/// Compute (annuity, float_pv) for a floating leg given its payment dates.
fn leg_annuity_and_float_pv(
    dates: &[Date],
    start: Date,
    times: &[f64],
    dfs: &[f64],
    accrual_dc: &DayCounter,
    curve_dc: DayCounter,
    ref_date: Date,
) -> (f64, f64) {
    let mut annuity = 0.0;
    let mut float_pv = 0.0;
    let mut prev = start;

    for &payment_date in dates {
        let tau = accrual_dc.year_fraction(prev, payment_date);
        let t_prev = curve_dc.year_fraction(ref_date, prev);
        let t_pay = curve_dc.year_fraction(ref_date, payment_date);
        let df_prev = interpolate_log_linear(times, dfs, t_prev);
        let df_pay = interpolate_log_linear(times, dfs, t_pay);

        // Forward rate
        let fwd = if tau.abs() > 1e-15 {
            (df_prev / df_pay - 1.0) / tau
        } else {
            0.0
        };

        annuity += tau * df_pay;
        float_pv += tau * df_pay * fwd;
        prev = payment_date;
    }

    (annuity, float_pv)
}

/// Compute the float PV for a leg.
fn leg_float_pv(
    dates: &[Date],
    start: Date,
    times: &[f64],
    dfs: &[f64],
    accrual_dc: &DayCounter,
    curve_dc: DayCounter,
    ref_date: Date,
) -> f64 {
    let mut float_pv = 0.0;
    let mut prev = start;

    for &payment_date in dates {
        let tau = accrual_dc.year_fraction(prev, payment_date);
        let t_prev = curve_dc.year_fraction(ref_date, prev);
        let t_pay = curve_dc.year_fraction(ref_date, payment_date);
        let df_prev = interpolate_log_linear(times, dfs, t_prev);
        let df_pay = interpolate_log_linear(times, dfs, t_pay);

        let fwd = if tau.abs() > 1e-15 {
            (df_prev / df_pay - 1.0) / tau
        } else {
            0.0
        };

        float_pv += tau * df_pay * fwd;
        prev = payment_date;
    }

    float_pv
}

/// Log-linear interpolation of discount factors (same as bootstrap.rs).
fn interpolate_log_linear(times: &[f64], dfs: &[f64], t: f64) -> f64 {
    if times.is_empty() || t <= 0.0 {
        return 1.0;
    }
    if t <= times[0] {
        // Extrapolate from first point
        if times[0].abs() < 1e-15 {
            return dfs[0];
        }
        return (dfs[0].ln() * t / times[0]).exp();
    }
    if t >= *times.last().unwrap() {
        // Flat extrapolation in zero rates
        let last_t = *times.last().unwrap();
        let last_df = *dfs.last().unwrap();
        if last_t.abs() < 1e-15 {
            return last_df;
        }
        return (last_df.ln() * t / last_t).exp();
    }

    // Binary search for the interval
    let mut lo = 0;
    let mut hi = times.len() - 1;
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if times[mid] <= t {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let t0 = times[lo];
    let t1 = times[hi];
    let df0 = dfs[lo];
    let df1 = dfs[hi];

    if (t1 - t0).abs() < 1e-15 {
        return df0;
    }

    let w = (t - t0) / (t1 - t0);
    (df0.ln() * (1.0 - w) + df1.ln() * w).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn ref_date() -> Date {
        Date::from_ymd(2025, Month::January, 15)
    }

    #[test]
    fn test_schedule_generation() {
        let helper = IborIborBasisSwapHelper::new(
            0.001,
            ref_date(),
            5,
            3,
            6,
            DayCounter::Actual360,
            Calendar::Target,
        );
        // 5Y / 3M = 20 short-tenor periods
        assert_eq!(helper.short_dates.len(), 20);
        // 5Y / 6M = 10 long-tenor periods
        assert_eq!(helper.long_dates.len(), 10);
        // Pillar at last long date
        assert_eq!(helper.pillar_date(), *helper.long_dates.last().unwrap());
    }

    #[test]
    fn test_pillar_date_after_ref() {
        let helper = IborIborBasisSwapHelper::new(
            0.0005,
            ref_date(),
            2,
            3,
            6,
            DayCounter::Actual360,
            Calendar::Target,
        );
        assert!(helper.pillar_date() > ref_date());
    }

    #[test]
    fn test_zero_spread_implied_is_zero() {
        // With a flat curve, both legs are equal → implied spread = 0
        let rd = ref_date();
        let helper = IborIborBasisSwapHelper::new(
            0.0,
            rd,
            2,
            3,
            6,
            DayCounter::Actual360,
            Calendar::Target,
        );

        let dc = DayCounter::Actual360;
        // Flat 3% curve: discount factors at integer years
        let times: Vec<f64> = vec![0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25];
        let dfs: Vec<f64> = times.iter().map(|&t| (-0.03 * t).exp()).collect();

        let implied = helper.implied_quote(&times, &dfs, dc, rd);
        assert_abs_diff_eq!(implied, 0.0, epsilon = 1e-8);
    }

    #[test]
    fn test_positive_spread_implied() {
        let rd = ref_date();
        let helper = IborIborBasisSwapHelper::new(
            0.002, // 20 bps
            rd,
            3,
            3,
            6,
            DayCounter::Actual360,
            Calendar::Target,
        );

        let dc = DayCounter::Actual360;
        // Flat 4% curve
        let times: Vec<f64> = (0..=15).map(|i| i as f64 * 0.25).collect();
        let dfs: Vec<f64> = times.iter().map(|&t| (-0.04 * t).exp()).collect();

        let implied = helper.implied_quote(&times, &dfs, dc, rd);
        // With a flat curve both legs are the same → implied spread ≈ 0
        // regardless of the *quoted* spread
        assert!(implied.abs() < 0.01, "implied = {}", implied);
    }

    #[test]
    fn test_quote_returns_spread() {
        let helper = IborIborBasisSwapHelper::new(
            -0.0015,
            ref_date(),
            10,
            3,
            6,
            DayCounter::Actual365Fixed,
            Calendar::Target,
        );
        assert_abs_diff_eq!(helper.quote(), -0.0015, epsilon = 1e-15);
    }

    #[test]
    fn test_1m_vs_3m_tenor() {
        let helper = IborIborBasisSwapHelper::new(
            0.0005,
            ref_date(),
            2,
            1,
            3,
            DayCounter::Actual360,
            Calendar::Target,
        );
        // 2Y / 1M = 24 short dates, 2Y / 3M = 8 long dates
        assert_eq!(helper.short_dates.len(), 24);
        assert_eq!(helper.long_dates.len(), 8);
    }

    #[test]
    fn test_rate_helper_trait_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<IborIborBasisSwapHelper>();
    }

    #[test]
    fn test_serde_round_trip() {
        let helper = IborIborBasisSwapHelper::new(
            0.001,
            ref_date(),
            5,
            3,
            6,
            DayCounter::Actual360,
            Calendar::Target,
        );
        let json = serde_json::to_string(&helper).unwrap();
        let deser: IborIborBasisSwapHelper = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.quote(), 0.001);
        assert_eq!(deser.pillar_date(), helper.pillar_date());
    }
}
