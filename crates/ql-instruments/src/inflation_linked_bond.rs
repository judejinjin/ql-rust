//! Inflation-linked bond (TIPS / ILB).
//!
//! An inflation-linked bond pays inflation-adjusted coupons and an
//! inflation-adjusted principal at maturity. The canonical example is
//! US Treasury Inflation-Protected Securities (TIPS).
//!
//! # Structure
//! - Coupon: `N × rate × accrual × CPI(t)/CPI(base)`
//! - Redemption: `N × max(1, CPI(T)/CPI(base))` (deflation floor)
//!
//! # References
//! - US Treasury TIPS: <https://www.treasurydirect.gov/marketable-securities/tips/>

use ql_time::{Date, DayCounter};
use serde::{Deserialize, Serialize};

/// An inflation-linked bond (e.g. TIPS).
///
/// Coupons and principal are adjusted by the ratio CPI(t)/CPI(base).
/// The principal redemption has an optional deflation floor (par minimum).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InflationLinkedBond {
    /// Face value (notional).
    pub notional: f64,
    /// Annual coupon rate.
    pub coupon_rate: f64,
    /// Schedule dates (accrual periods).
    pub schedule_dates: Vec<Date>,
    /// Maturity date.
    pub maturity: Date,
    /// Base CPI at issuance.
    pub base_cpi: f64,
    /// Whether the principal redemption has a deflation floor at par.
    pub deflation_floor: bool,
    /// Day count convention.
    pub day_counter: DayCounter,
}

impl InflationLinkedBond {
    /// Create a new inflation-linked bond.
    pub fn new(
        notional: f64,
        coupon_rate: f64,
        schedule_dates: Vec<Date>,
        base_cpi: f64,
        deflation_floor: bool,
        day_counter: DayCounter,
    ) -> Self {
        let maturity = *schedule_dates.last().unwrap_or(&Date::from_serial(1));
        Self {
            notional,
            coupon_rate,
            schedule_dates,
            maturity,
            base_cpi,
            deflation_floor,
            day_counter,
        }
    }

    /// Compute the clean price given:
    /// - `cpi_ratios`: CPI(fixing_date)/CPI(base) for each coupon period
    /// - `discount_factors`: discount factor to each payment date
    ///
    /// `cpi_ratios` and `discount_factors` must have length = `schedule_dates.len() - 1`.
    pub fn dirty_price(
        &self,
        cpi_ratios: &[f64],
        discount_factors: &[f64],
    ) -> f64 {
        let n_periods = self.schedule_dates.len() - 1;
        assert_eq!(cpi_ratios.len(), n_periods);
        assert_eq!(discount_factors.len(), n_periods);

        let mut pv = 0.0;

        for i in 0..n_periods {
            let accrual = self.day_counter.year_fraction(
                self.schedule_dates[i],
                self.schedule_dates[i + 1],
            );
            // Inflation-adjusted coupon
            let coupon_pv =
                self.notional * self.coupon_rate * accrual * cpi_ratios[i] * discount_factors[i];
            pv += coupon_pv;
        }

        // Inflation-adjusted principal at maturity
        let final_ratio = cpi_ratios[n_periods - 1];
        let redemption_ratio = if self.deflation_floor {
            final_ratio.max(1.0) // par floor
        } else {
            final_ratio
        };
        pv += self.notional * redemption_ratio * discount_factors[n_periods - 1];

        pv
    }

    /// Price using flat inflation rate and flat discount rate.
    ///
    /// Convenience method for quick pricing.
    pub fn price_flat(
        &self,
        inflation_rate: f64,
        discount_rate: f64,
    ) -> f64 {
        let n_periods = self.schedule_dates.len() - 1;
        let mut cpi_ratios = Vec::with_capacity(n_periods);
        let mut dfs = Vec::with_capacity(n_periods);

        for i in 0..n_periods {
            let t = self.day_counter.year_fraction(
                self.schedule_dates[0],
                self.schedule_dates[i + 1],
            );
            cpi_ratios.push((1.0 + inflation_rate).powf(t));
            dfs.push((-discount_rate * t).exp());
        }

        self.dirty_price(&cpi_ratios, &dfs)
    }

    /// Break-even inflation rate: the flat inflation rate at which
    /// the bond prices at par.
    ///
    /// Uses bisection search.
    pub fn breakeven_inflation(&self, discount_rate: f64) -> f64 {
        let mut lo = -0.05_f64;
        let mut hi = 0.20_f64;
        let target = self.notional; // par

        for _ in 0..100 {
            let mid = 0.5 * (lo + hi);
            let price = self.price_flat(mid, discount_rate);
            if price > target {
                hi = mid;
            } else {
                lo = mid;
            }
            if (hi - lo) < 1e-10 {
                break;
            }
        }
        0.5 * (lo + hi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_time::Month;

    fn sample_dates() -> Vec<Date> {
        vec![
            Date::from_ymd(2025, Month::January, 15),
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
            Date::from_ymd(2026, Month::July, 15),
            Date::from_ymd(2027, Month::January, 15),
        ]
    }

    #[test]
    fn tips_price_no_inflation() {
        let bond = InflationLinkedBond::new(
            100.0, 0.02, sample_dates(), 300.0, true, DayCounter::Actual365Fixed,
        );
        // CPI unchanged → ratios all 1.0
        let n = sample_dates().len() - 1;
        let ratios = vec![1.0; n];
        let dfs = vec![1.0; n]; // no discounting
        let price = bond.dirty_price(&ratios, &dfs);
        // Sum of coupons + par = 4 × (100 × 0.02 × ~0.5) + 100 ≈ 104
        assert!(price > 103.5 && price < 104.5, "price = {}", price);
    }

    #[test]
    fn tips_deflation_floor() {
        let bond = InflationLinkedBond::new(
            100.0, 0.02, sample_dates(), 300.0, true, DayCounter::Actual365Fixed,
        );
        let n = sample_dates().len() - 1;
        // CPI dropped 10% → ratio = 0.9
        let ratios = vec![0.9; n];
        let dfs = vec![1.0; n];
        let price_floored = bond.dirty_price(&ratios, &dfs);

        let bond_no_floor = InflationLinkedBond::new(
            100.0, 0.02, sample_dates(), 300.0, false, DayCounter::Actual365Fixed,
        );
        let price_no_floor = bond_no_floor.dirty_price(&ratios, &dfs);

        // Floored bond should be worth more (redemption at par, not 90)
        assert!(
            price_floored > price_no_floor,
            "floored {} vs unfloored {}",
            price_floored,
            price_no_floor
        );
    }

    #[test]
    fn tips_price_flat_convenience() {
        let bond = InflationLinkedBond::new(
            100.0, 0.0125, sample_dates(), 300.0, true, DayCounter::Actual365Fixed,
        );
        let price = bond.price_flat(0.025, 0.04);
        // With positive inflation and positive discounting, price should be reasonable
        assert!(price > 80.0 && price < 120.0, "price = {}", price);
    }

    #[test]
    fn tips_breakeven_inflation() {
        let bond = InflationLinkedBond::new(
            100.0, 0.02, sample_dates(), 300.0, true, DayCounter::Actual365Fixed,
        );
        let bei = bond.breakeven_inflation(0.04);
        // BEI should be positive and reasonable
        assert!(bei > 0.0 && bei < 0.10, "BEI = {}", bei);
        // At the BEI, the bond should price near par
        let price = bond.price_flat(bei, 0.04);
        assert_abs_diff_eq!(price, 100.0, epsilon = 0.01);
    }
}
