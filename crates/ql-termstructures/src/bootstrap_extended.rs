#![allow(clippy::too_many_arguments)]
//! Extended rate helpers for bootstrapping.
//!
//! - `OISRateHelper` — overnight index swap helper for OIS curve bootstrapping.
//! - `BondHelper` — bootstrap from clean bond prices.
//! - `FuturesRateHelper` — interest rate futures (e.g. SOFR, Eurodollar).
//! - `FRAHelper` — forward rate agreement helper.

use ql_time::{Calendar, Date, DayCounter};

use crate::bootstrap::RateHelper;

// ===========================================================================
// OISRateHelper
// ===========================================================================

/// Rate helper for bootstrapping from OIS (Overnight Index Swap) quotes.
///
/// An OIS swaps a fixed rate against a compounded overnight rate over the
/// swap tenor. The fixed side pays `rate · τ` at maturity, while the
/// floating side compounds the overnight rate daily.
///
/// The implied OIS rate from a discount curve is:
/// $$r_{\text{OIS}} = \frac{df_{\text{start}} / df_{\text{end}} - 1}{\tau}$$
/// where τ is the year fraction for the swap.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OISRateHelper {
    /// Quoted OIS rate.
    rate: f64,
    /// Start date of the swap.
    start_date: Date,
    /// End date (maturity).
    end_date: Date,
    /// Day counter for the fixed leg.
    day_counter: DayCounter,
}

impl OISRateHelper {
    /// Create an OIS rate helper.
    pub fn new(rate: f64, start_date: Date, end_date: Date, day_counter: DayCounter) -> Self {
        Self {
            rate,
            start_date,
            end_date,
            day_counter,
        }
    }

    /// Convenience: create from a tenor in months.
    pub fn from_tenor(
        rate: f64,
        start_date: Date,
        tenor_months: u32,
        day_counter: DayCounter,
        calendar: Calendar,
    ) -> Self {
        let (y, m, d) = (
            start_date.year(),
            start_date.month() as u32,
            start_date.day_of_month(),
        );
        let total_months = (y * 12 + m as i32 - 1) + tenor_months as i32;
        let new_y = total_months / 12;
        let new_m = (total_months % 12) as u32 + 1;
        let new_d = d.min(Date::days_in_month(new_y, new_m));
        let end_date = Date::from_ymd_opt(new_y, new_m, new_d).unwrap_or(start_date);
        let end_date = calendar.adjust(end_date, ql_time::BusinessDayConvention::ModifiedFollowing);

        Self {
            rate,
            start_date,
            end_date,
            day_counter,
        }
    }
}

impl RateHelper for OISRateHelper {
    fn pillar_date(&self) -> Date {
        self.end_date
    }

    fn quote(&self) -> f64 {
        self.rate
    }

    fn implied_quote(
        &self,
        times: &[f64],
        dfs: &[f64],
        day_counter: DayCounter,
        ref_date: Date,
    ) -> f64 {
        let t_start = day_counter.year_fraction(ref_date, self.start_date);
        let t_end = day_counter.year_fraction(ref_date, self.end_date);
        let yf = self.day_counter.year_fraction(self.start_date, self.end_date);

        let df_start = interpolate_log_linear(times, dfs, t_start);
        let df_end = interpolate_log_linear(times, dfs, t_end);

        if yf.abs() < 1e-15 {
            return 0.0;
        }
        (df_start / df_end - 1.0) / yf
    }
}

// ===========================================================================
// BondHelper
// ===========================================================================

/// Rate helper for bootstrapping from bond clean prices.
///
/// Given a bond with known coupon schedule and a target clean price, the
/// bootstrap finds the discount factor at maturity that matches the price.
///
/// The implied clean price from a set of discount factors is:
/// $$P = \sum_i c_i \cdot df(t_i) + \text{FV} \cdot df(T)$$
///
/// We solve for the yield that reproduces the market price, then convert
/// to an implied par rate.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BondHelper {
    /// Clean price of the bond.
    clean_price: f64,
    /// Settlement date.
    settlement_date: Date,
    /// Coupon payment dates.
    coupon_dates: Vec<Date>,
    /// Coupon amount per period (as a fraction of face value).
    coupon_rate: f64,
    /// Face value.
    face_value: f64,
    /// Day counter for the bond.
    day_counter: DayCounter,
}

impl BondHelper {
    /// Create a bond helper.
    ///
    /// # Arguments
    /// * `clean_price` — market clean price (as fraction of face, e.g. 0.98)
    /// * `settlement_date` — bond settlement date
    /// * `coupon_dates` — all future coupon and redemption dates
    /// * `coupon_rate` — annual coupon rate (e.g. 0.05 for 5%)
    /// * `face_value` — face (par) value
    /// * `day_counter` — day count convention
    pub fn new(
        clean_price: f64,
        settlement_date: Date,
        coupon_dates: Vec<Date>,
        coupon_rate: f64,
        face_value: f64,
        day_counter: DayCounter,
    ) -> Self {
        Self {
            clean_price,
            settlement_date,
            coupon_dates,
            coupon_rate,
            face_value,
            day_counter,
        }
    }
}

impl RateHelper for BondHelper {
    fn pillar_date(&self) -> Date {
        *self.coupon_dates.last().unwrap_or(&self.settlement_date)
    }

    fn quote(&self) -> f64 {
        self.clean_price
    }

    fn implied_quote(
        &self,
        times: &[f64],
        dfs: &[f64],
        day_counter: DayCounter,
        ref_date: Date,
    ) -> f64 {
        // Compute the theoretical clean price from the current discount factors
        let mut dirty_price = 0.0;
        let n = self.coupon_dates.len();
        let mut prev_date = self.settlement_date;

        for (i, &cpn_date) in self.coupon_dates.iter().enumerate() {
            let yf = self.day_counter.year_fraction(prev_date, cpn_date);
            let t = day_counter.year_fraction(ref_date, cpn_date);
            let df = interpolate_log_linear(times, dfs, t);

            // Coupon payment
            let cpn = self.coupon_rate * yf * self.face_value;
            dirty_price += cpn * df;

            // Redemption at maturity
            if i == n - 1 {
                dirty_price += self.face_value * df;
            }

            prev_date = cpn_date;
        }

        // Convert dirty to clean (approximate: assume no accrued for bootstrap)
        dirty_price / self.face_value
    }
}

// ===========================================================================
// FuturesRateHelper
// ===========================================================================

/// Rate helper for bootstrapping from interest rate futures.
///
/// Futures contracts (e.g. 3-month SOFR futures, Eurodollar futures) quote
/// a price of `100 - rate`. The implied forward rate for the contract period
/// is derived from discount factors.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FuturesRateHelper {
    /// Futures price (e.g. 95.50 implies a rate of 4.50%).
    price: f64,
    /// Start date of the futures contract period.
    start_date: Date,
    /// End date of the futures contract period (typically +3M).
    end_date: Date,
    /// Day counter.
    day_counter: DayCounter,
    /// Convexity adjustment (futures vs forward rate bias), typically small positive value.
    convexity_adjustment: f64,
}

impl FuturesRateHelper {
    /// Create a futures rate helper.
    ///
    /// The implied rate is `(100 - price) / 100 - convexity_adjustment`.
    pub fn new(
        price: f64,
        start_date: Date,
        end_date: Date,
        day_counter: DayCounter,
        convexity_adjustment: f64,
    ) -> Self {
        Self {
            price,
            start_date,
            end_date,
            day_counter,
            convexity_adjustment,
        }
    }

    /// The implied forward rate from the futures price.
    pub fn implied_rate(&self) -> f64 {
        (100.0 - self.price) / 100.0 - self.convexity_adjustment
    }
}

impl RateHelper for FuturesRateHelper {
    fn pillar_date(&self) -> Date {
        self.end_date
    }

    fn quote(&self) -> f64 {
        // The "quote" for bootstrap purposes is the adjusted forward rate
        self.implied_rate()
    }

    fn implied_quote(
        &self,
        times: &[f64],
        dfs: &[f64],
        day_counter: DayCounter,
        ref_date: Date,
    ) -> f64 {
        let t_start = day_counter.year_fraction(ref_date, self.start_date);
        let t_end = day_counter.year_fraction(ref_date, self.end_date);
        let yf = self.day_counter.year_fraction(self.start_date, self.end_date);

        let df_start = interpolate_log_linear(times, dfs, t_start);
        let df_end = interpolate_log_linear(times, dfs, t_end);

        if yf.abs() < 1e-15 {
            return 0.0;
        }

        // Simple forward rate
        (df_start / df_end - 1.0) / yf
    }
}

// ===========================================================================
// FRAHelper
// ===========================================================================

/// Rate helper for bootstrapping from Forward Rate Agreement (FRA) quotes.
///
/// An FRA settles based on the difference between the contracted rate and
/// the realised fixing rate. For bootstrap purposes, the implied rate is
/// a simple forward rate between start and end dates.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FRAHelper {
    /// Quoted FRA rate.
    rate: f64,
    /// Start date of the FRA period.
    start_date: Date,
    /// End date of the FRA period.
    end_date: Date,
    /// Day counter.
    day_counter: DayCounter,
}

impl FRAHelper {
    /// Create a FRA rate helper.
    pub fn new(rate: f64, start_date: Date, end_date: Date, day_counter: DayCounter) -> Self {
        Self {
            rate,
            start_date,
            end_date,
            day_counter,
        }
    }
}

impl RateHelper for FRAHelper {
    fn pillar_date(&self) -> Date {
        self.end_date
    }

    fn quote(&self) -> f64 {
        self.rate
    }

    fn implied_quote(
        &self,
        times: &[f64],
        dfs: &[f64],
        day_counter: DayCounter,
        ref_date: Date,
    ) -> f64 {
        let t_start = day_counter.year_fraction(ref_date, self.start_date);
        let t_end = day_counter.year_fraction(ref_date, self.end_date);
        let yf = self.day_counter.year_fraction(self.start_date, self.end_date);

        let df_start = interpolate_log_linear(times, dfs, t_start);
        let df_end = interpolate_log_linear(times, dfs, t_end);

        if yf.abs() < 1e-15 {
            return 0.0;
        }
        (df_start / df_end - 1.0) / yf
    }
}

// ===========================================================================
// Helper: log-linear interpolation
// ===========================================================================

/// Log-linear interpolation on discount factors (duplicated here for
/// independence from bootstrap.rs).
fn interpolate_log_linear(times: &[f64], dfs: &[f64], t: f64) -> f64 {
    if times.is_empty() || dfs.is_empty() {
        return 1.0;
    }
    if t <= times[0] {
        return dfs[0];
    }
    let n = times.len();
    if t >= times[n - 1] {
        if n < 2 {
            return dfs[n - 1];
        }
        let ln1 = dfs[n - 2].ln();
        let ln2 = dfs[n - 1].ln();
        let dt = times[n - 1] - times[n - 2];
        if dt.abs() < 1e-15 {
            return dfs[n - 1];
        }
        let slope = (ln2 - ln1) / dt;
        return (ln2 + slope * (t - times[n - 1])).exp();
    }

    let mut i = 0;
    for (j, tj) in times.iter().enumerate().skip(1) {
        if *tj >= t {
            i = j - 1;
            break;
        }
    }
    let t1 = times[i];
    let t2 = times[i + 1];
    let ln1 = dfs[i].ln();
    let ln2 = dfs[i + 1].ln();
    let dt = t2 - t1;
    if dt.abs() < 1e-15 {
        return dfs[i];
    }
    let frac = (t - t1) / dt;
    (ln1 + frac * (ln2 - ln1)).exp()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use crate::bootstrap::PiecewiseYieldCurve;
    use crate::yield_term_structure::YieldTermStructure;
    use ql_time::Month;

    fn ref_date() -> Date {
        Date::from_ymd(2025, Month::January, 2)
    }

    #[test]
    fn ois_helper_basic() {
        let start = ref_date();
        let end = Date::from_ymd(2025, Month::April, 2);
        let helper = OISRateHelper::new(0.04, start, end, DayCounter::Actual360);
        assert_eq!(helper.pillar_date(), end);
        assert_abs_diff_eq!(helper.quote(), 0.04, epsilon = 1e-15);
    }

    #[test]
    fn ois_bootstrap_single() {
        let start = ref_date();
        let end = Date::from_ymd(2025, Month::July, 2);
        let rate = 0.035;

        let mut helpers: Vec<Box<dyn RateHelper>> =
            vec![Box::new(OISRateHelper::new(rate, start, end, DayCounter::Actual360))];

        let curve = PiecewiseYieldCurve::new(
            ref_date(),
            &mut helpers,
            DayCounter::Actual360,
            1e-12,
        )
        .unwrap();

        // Verify: df = 1/(1 + rate * yf)
        let yf = DayCounter::Actual360.year_fraction(start, end);
        let expected_df = 1.0 / (1.0 + rate * yf);
        assert_abs_diff_eq!(curve.discount(end), expected_df, epsilon = 1e-8);
    }

    #[test]
    fn ois_helper_from_tenor() {
        let start = ref_date();
        let helper = OISRateHelper::from_tenor(
            0.04,
            start,
            6,
            DayCounter::Actual360,
            Calendar::Target,
        );
        assert!(helper.pillar_date() > start);
    }

    #[test]
    fn bond_helper_par_bond() {
        let settle = ref_date();
        // 2Y annual 5% bond at par
        let cpn_dates = vec![
            Date::from_ymd(2026, Month::January, 2),
            Date::from_ymd(2027, Month::January, 2),
        ];
        let helper = BondHelper::new(1.0, settle, cpn_dates.clone(), 0.05, 100.0, DayCounter::Actual365Fixed);
        assert_eq!(helper.pillar_date(), cpn_dates[1]);
    }

    #[test]
    fn futures_helper_implied_rate() {
        let start = Date::from_ymd(2025, Month::March, 19);
        let end = Date::from_ymd(2025, Month::June, 18);
        let helper = FuturesRateHelper::new(95.50, start, end, DayCounter::Actual360, 0.001);

        // Implied rate = (100 - 95.5) / 100 - 0.001 = 0.045 - 0.001 = 0.044
        assert_abs_diff_eq!(helper.implied_rate(), 0.044, epsilon = 1e-15);
    }

    #[test]
    fn futures_bootstrap() {
        let start = ref_date();
        let end_3m = Date::from_ymd(2025, Month::April, 2);

        let rate = 0.04;
        let price = 100.0 - rate * 100.0; // 96.0

        let mut helpers: Vec<Box<dyn RateHelper>> = vec![Box::new(FuturesRateHelper::new(
            price,
            start,
            end_3m,
            DayCounter::Actual360,
            0.0, // no convexity adjustment
        ))];

        let curve = PiecewiseYieldCurve::new(
            ref_date(),
            &mut helpers,
            DayCounter::Actual360,
            1e-12,
        )
        .unwrap();

        let yf = DayCounter::Actual360.year_fraction(start, end_3m);
        let expected_df = 1.0 / (1.0 + rate * yf);
        assert_abs_diff_eq!(curve.discount(end_3m), expected_df, epsilon = 1e-8);
    }

    #[test]
    fn fra_helper_basic() {
        let start = Date::from_ymd(2025, Month::April, 2);
        let end = Date::from_ymd(2025, Month::July, 2);
        let helper = FRAHelper::new(0.045, start, end, DayCounter::Actual360);
        assert_eq!(helper.pillar_date(), end);
        assert_abs_diff_eq!(helper.quote(), 0.045, epsilon = 1e-15);
    }

    #[test]
    fn mixed_ois_and_futures_bootstrap() {
        let start = ref_date();
        let end_3m = Date::from_ymd(2025, Month::April, 2);
        let end_6m = Date::from_ymd(2025, Month::July, 2);

        let mut helpers: Vec<Box<dyn RateHelper>> = vec![
            Box::new(OISRateHelper::new(0.04, start, end_3m, DayCounter::Actual360)),
            Box::new(FuturesRateHelper::new(
                95.5,
                start,
                end_6m,
                DayCounter::Actual360,
                0.0,
            )),
        ];

        let curve = PiecewiseYieldCurve::new(
            ref_date(),
            &mut helpers,
            DayCounter::Actual360,
            1e-12,
        )
        .unwrap();

        // Curve should have 3 nodes (ref + 2 pillars)
        assert_eq!(curve.size(), 3);

        // 3M should reproduce OIS rate
        let yf_3m = DayCounter::Actual360.year_fraction(start, end_3m);
        let expected_3m = 1.0 / (1.0 + 0.04 * yf_3m);
        assert_abs_diff_eq!(curve.discount(end_3m), expected_3m, epsilon = 1e-8);
    }
}
