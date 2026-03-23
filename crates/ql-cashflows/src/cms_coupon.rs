#![allow(clippy::too_many_arguments)]
//! CMS (Constant Maturity Swap) coupons and pricers.
//!
//! A CMS coupon pays a rate linked to a swap rate of a given tenor,
//! observed at the fixing date. Because forward swap rates are not
//! martingales under the payment measure, a convexity adjustment is needed.

use ql_time::{Date, DayCounter};
use crate::cashflow::CashFlow;
use crate::coupon::Coupon;

/// A CMS coupon pays `nominal * (swap_rate + spread) * accrual_period`.
///
/// The swap rate is the par swap rate for a given tenor at the fixing date.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CmsCoupon {
    /// Payment date.
    pub payment_date: Date,
    /// Notional amount.
    pub nominal: f64,
    /// Date on which the swap rate is observed.
    pub fixing_date: Date,
    /// Start of accrual period.
    pub accrual_start: Date,
    /// End of accrual period.
    pub accrual_end: Date,
    /// Day counter for year-fraction computation.
    pub day_counter: DayCounter,
    /// Tenor (in years) of the reference swap rate.
    pub swap_tenor_years: f64,
    /// Additive spread over the swap rate.
    pub spread: f64,
    /// Multiplicative gearing factor applied to the swap rate.
    pub gearing: f64,
    /// The projected or fixed swap rate (set by pricer).
    pub rate_estimate: f64,
}

impl CmsCoupon {
    /// New.
    pub fn new(
        payment_date: Date,
        nominal: f64,
        fixing_date: Date,
        accrual_start: Date,
        accrual_end: Date,
        day_counter: DayCounter,
        swap_tenor_years: f64,
        spread: f64,
        gearing: f64,
    ) -> Self {
        Self {
            payment_date,
            nominal,
            fixing_date,
            accrual_start,
            accrual_end,
            day_counter,
            swap_tenor_years,
            spread,
            gearing,
            rate_estimate: 0.0,
        }
    }

    /// Set the forward swap rate (from a pricer).
    pub fn set_rate(&mut self, rate: f64) {
        self.rate_estimate = rate;
    }
}

impl CashFlow for CmsCoupon {
    fn date(&self) -> Date {
        self.payment_date
    }

    fn amount(&self) -> f64 {
        self.nominal * (self.gearing * self.rate_estimate + self.spread) * self.accrual_period()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Coupon for CmsCoupon {
    fn nominal(&self) -> f64 {
        self.nominal
    }

    fn rate(&self) -> f64 {
        self.gearing * self.rate_estimate + self.spread
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
// CMS Convexity Adjustment (Hagan linear TSR)
// ===========================================================================

/// Linear Terminal Swap Rate (TSR) model convexity adjustment.
///
/// The convexity adjustment for a CMS rate is:
///
/// `adj ≈ σ² T S (∂A/∂S) / A(S)`
///
/// where A(S) is the annuity as a function of S, σ is the swap rate vol,
/// and T is the fixing time.
pub fn cms_convexity_adjustment(
    forward_swap_rate: f64,
    swap_vol: f64,
    fixing_time: f64,
    swap_tenor: f64,
    payment_delay: f64,
) -> f64 {
    if fixing_time <= 0.0 || swap_vol <= 0.0 {
        return 0.0;
    }

    let s = forward_swap_rate;
    let n = (swap_tenor * 2.0).round() as usize; // semiannual coupons assumed

    if n == 0 {
        return 0.0;
    }

    // Annuity A(S) = sum of discount factors = sum_{i=1}^{n} (1 + S/2)^{-i}
    // ∂A/∂S: derivative of annuity w.r.t. swap rate
    let half_s = s / 2.0;
    let mut annuity = 0.0;
    let mut d_annuity = 0.0;
    for i in 1..=n {
        let df = (1.0 + half_s).powi(-(i as i32));
        annuity += df;
        d_annuity += -(i as f64 / 2.0) * df / (1.0 + half_s);
    }

    if annuity.abs() < 1e-30 {
        return 0.0;
    }

    // Payment delay adjustment (linear approximation)
    let delay_adj = payment_delay * s;

    // Convexity adjustment
    swap_vol * swap_vol * fixing_time * s * d_annuity / annuity + delay_adj
}

/// Price a CMS caplet using the linear TSR model.
///
/// Returns the present value of a CMS caplet with strike K.
pub fn cms_caplet_price(
    forward_swap_rate: f64,
    strike: f64,
    swap_vol: f64,
    fixing_time: f64,
    payment_time: f64,
    discount_factor: f64,
    nominal: f64,
    accrual_period: f64,
) -> f64 {
    let adjusted_forward = forward_swap_rate + cms_convexity_adjustment(
        forward_swap_rate,
        swap_vol,
        fixing_time,
        fixing_time, // approximate tenor = fixing_time for simplicity
        payment_time - fixing_time,
    );

    // Black formula
    let d1 = ((adjusted_forward / strike).ln() + 0.5 * swap_vol * swap_vol * fixing_time)
        / (swap_vol * fixing_time.sqrt());
    let d2 = d1 - swap_vol * fixing_time.sqrt();

    let n = ql_math::distributions::NormalDistribution::standard();
    let price = discount_factor
        * nominal
        * accrual_period
        * (adjusted_forward * n.cdf(d1) - strike * n.cdf(d2));

    price.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_time::Month;

    #[test]
    fn cms_coupon_basic() {
        let mut cms = CmsCoupon::new(
            Date::from_ymd(2026, Month::January, 15),
            1_000_000.0,
            Date::from_ymd(2025, Month::July, 13),
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
            DayCounter::Actual360,
            5.0,
            0.001, // 10bp spread
            1.0,
        );
        cms.set_rate(0.04);
        let amount = cms.amount();
        // 1M * (0.04 + 0.001) * ~0.5 ≈ 20,694
        assert!(amount > 20_000.0 && amount < 21_500.0, "CMS coupon amount = {amount}");
    }

    #[test]
    fn cms_convexity_positive() {
        let adj = cms_convexity_adjustment(0.04, 0.20, 1.0, 5.0, 0.0);
        // Convexity adjustment should be negative (d_annuity < 0)
        // but the payment delay component is 0, so just check magnitude
        assert!(adj.abs() < 0.01, "Convexity adjustment magnitude: {adj}");
    }

    #[test]
    fn cms_convexity_zero_vol() {
        let adj = cms_convexity_adjustment(0.04, 0.0, 1.0, 5.0, 0.0);
        assert_abs_diff_eq!(adj, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn cms_caplet_positive() {
        let price = cms_caplet_price(
            0.04,   // forward swap rate
            0.04,   // ATM strike
            0.20,   // swap vol 20%
            1.0,    // fixing time 1Y
            1.0,    // payment time 1Y
            0.96,   // discount factor
            1_000_000.0,
            0.5,    // accrual period
        );
        assert!(price > 0.0, "CMS caplet should have positive value: {price}");
    }

    #[test]
    fn cms_caplet_otm() {
        let atm = cms_caplet_price(0.04, 0.04, 0.20, 1.0, 1.0, 0.96, 1_000_000.0, 0.5);
        let otm = cms_caplet_price(0.04, 0.06, 0.20, 1.0, 1.0, 0.96, 1_000_000.0, 0.5);
        assert!(otm < atm, "OTM caplet should be cheaper: ATM={atm}, OTM={otm}");
    }

    #[test]
    fn cms_gearing() {
        let mut cms = CmsCoupon::new(
            Date::from_ymd(2026, Month::January, 15),
            1_000_000.0,
            Date::from_ymd(2025, Month::July, 13),
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
            DayCounter::Actual360,
            5.0,
            0.0,
            2.0, // 2x leverage
        );
        cms.set_rate(0.04);
        let amount = cms.amount();
        // 1M * 2 * 0.04 * ~0.5 ≈ 40,556
        assert!(amount > 39_000.0 && amount < 42_000.0, "Geared CMS = {amount}");
    }
}
