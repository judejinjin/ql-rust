//! Extended cash flows: G76–G83.
//!
//! **G76** — YoYInflationCoupon
//! **G77** — ZeroInflationCashFlow
//! **G78** — EquityCashFlow
//! **G79** — DigitalIborCoupon
//! **G80** — DigitalCmsCoupon
//! **G81** — CapFlooredInflationCoupon
//! **G82** — CpiCouponPricer
//! **G83** — RateAveraging

use crate::cashflow::CashFlow;
use crate::coupon::Coupon;
use ql_time::{Date, DayCounter};
use std::any::Any;

// ===========================================================================
// RateAveraging (G83)
// ===========================================================================

/// Rate averaging convention for overnight coupons (G83).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RateAveraging {
    /// Compound averaging: ∏(1 + rᵢ δᵢ) − 1.
    Compound,
    /// Simple (arithmetic) averaging: Σ(rᵢ δᵢ) / Σ(δᵢ).
    Simple,
}

impl Default for RateAveraging {
    fn default() -> Self {
        Self::Compound
    }
}

/// Compute averaged rate from sub-period rates and year fractions.
pub fn compute_averaged_rate(
    rates: &[f64],
    year_fractions: &[f64],
    method: RateAveraging,
) -> f64 {
    assert_eq!(rates.len(), year_fractions.len());
    if rates.is_empty() {
        return 0.0;
    }
    match method {
        RateAveraging::Compound => {
            let total_yf: f64 = year_fractions.iter().sum();
            if total_yf.abs() < 1e-15 {
                return 0.0;
            }
            let product: f64 = rates
                .iter()
                .zip(year_fractions.iter())
                .map(|(&r, &yf)| 1.0 + r * yf)
                .product();
            (product - 1.0) / total_yf
        }
        RateAveraging::Simple => {
            let total_yf: f64 = year_fractions.iter().sum();
            if total_yf.abs() < 1e-15 {
                return 0.0;
            }
            let weighted_sum: f64 = rates
                .iter()
                .zip(year_fractions.iter())
                .map(|(&r, &yf)| r * yf)
                .sum();
            weighted_sum / total_yf
        }
    }
}

// ===========================================================================
// YoYInflationCoupon (G76)
// ===========================================================================

/// Year-on-year inflation-linked coupon (G76).
///
/// Amount = notional × (CPI_end / CPI_start − 1) × gearing + spread,
/// paid over an accrual period.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct YoYInflationCoupon {
    pub payment_date: Date,
    pub nominal: f64,
    pub accrual_start: Date,
    pub accrual_end: Date,
    pub day_counter: DayCounter,
    /// The CPI (or index) value at the start of the period.
    pub base_cpi: f64,
    /// The CPI (or index) value at the end of the period (fixing).
    pub fixing_cpi: f64,
    /// Multiplicative gearing.
    pub gearing: f64,
    /// Additive spread.
    pub spread: f64,
}

impl YoYInflationCoupon {
    pub fn new(
        payment_date: Date,
        nominal: f64,
        accrual_start: Date,
        accrual_end: Date,
        day_counter: DayCounter,
        base_cpi: f64,
        fixing_cpi: f64,
        gearing: f64,
        spread: f64,
    ) -> Self {
        Self {
            payment_date,
            nominal,
            accrual_start,
            accrual_end,
            day_counter,
            base_cpi,
            fixing_cpi,
            gearing,
            spread,
        }
    }

    /// YoY rate: CPI_end / CPI_start − 1.
    pub fn yoy_rate(&self) -> f64 {
        if self.base_cpi.abs() < 1e-15 {
            return 0.0;
        }
        self.fixing_cpi / self.base_cpi - 1.0
    }

    /// Effective rate: gearing × yoy_rate + spread.
    pub fn effective_rate(&self) -> f64 {
        self.gearing * self.yoy_rate() + self.spread
    }
}

impl CashFlow for YoYInflationCoupon {
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

impl Coupon for YoYInflationCoupon {
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
// ZeroInflationCashFlow (G77)
// ===========================================================================

/// Zero-coupon inflation cash flow (G77).
///
/// Amount = notional × (CPI_fixing / CPI_base) at maturity.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ZeroInflationCashFlow {
    pub payment_date: Date,
    pub notional: f64,
    /// CPI at the base (start) date.
    pub base_cpi: f64,
    /// CPI at the fixing date (near payment).
    pub fixing_cpi: f64,
}

impl ZeroInflationCashFlow {
    pub fn new(
        payment_date: Date,
        notional: f64,
        base_cpi: f64,
        fixing_cpi: f64,
    ) -> Self {
        Self {
            payment_date,
            notional,
            base_cpi,
            fixing_cpi,
        }
    }

    /// Index ratio: fixing / base.
    pub fn index_ratio(&self) -> f64 {
        if self.base_cpi.abs() < 1e-15 {
            return 1.0;
        }
        self.fixing_cpi / self.base_cpi
    }
}

impl CashFlow for ZeroInflationCashFlow {
    fn date(&self) -> Date {
        self.payment_date
    }

    fn amount(&self) -> f64 {
        self.notional * self.index_ratio()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ===========================================================================
// EquityCashFlow (G78)
// ===========================================================================

/// Cash flow linked to equity/index performance (G78).
///
/// Amount = notional × (S_end / S_start) at payment date.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EquityCashFlow {
    pub payment_date: Date,
    pub notional: f64,
    /// Equity index level at the start.
    pub equity_start: f64,
    /// Equity index level at the end (fixing).
    pub equity_end: f64,
    /// Participation rate (default 1.0).
    pub participation: f64,
}

impl EquityCashFlow {
    pub fn new(
        payment_date: Date,
        notional: f64,
        equity_start: f64,
        equity_end: f64,
        participation: f64,
    ) -> Self {
        Self {
            payment_date,
            notional,
            equity_start,
            equity_end,
            participation,
        }
    }

    /// Equity performance: S_end / S_start − 1.
    pub fn performance(&self) -> f64 {
        if self.equity_start.abs() < 1e-15 {
            return 0.0;
        }
        self.equity_end / self.equity_start - 1.0
    }
}

impl CashFlow for EquityCashFlow {
    fn date(&self) -> Date {
        self.payment_date
    }

    fn amount(&self) -> f64 {
        self.notional * self.participation * self.performance()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ===========================================================================
// DigitalIborCoupon (G79)
// ===========================================================================

/// IBOR coupon with a digital (binary) payoff feature (G79).
///
/// If the IBOR fixing exceeds the strike, pays a digital amount;
/// otherwise pays zero (or optionally a rebate).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DigitalIborCoupon {
    pub payment_date: Date,
    pub nominal: f64,
    pub accrual_start: Date,
    pub accrual_end: Date,
    pub day_counter: DayCounter,
    /// IBOR fixing rate.
    pub fixing: f64,
    /// Strike level for the digital feature.
    pub strike: f64,
    /// Digital payout if in-the-money.
    pub digital_amount: f64,
    /// Whether this is a call-digital (true) or put-digital (false).
    pub is_call: bool,
    /// Gearing on the underlying IBOR rate.
    pub gearing: f64,
    /// Spread on the underlying IBOR rate.
    pub spread: f64,
}

impl DigitalIborCoupon {
    pub fn new(
        payment_date: Date,
        nominal: f64,
        accrual_start: Date,
        accrual_end: Date,
        day_counter: DayCounter,
        fixing: f64,
        strike: f64,
        digital_amount: f64,
        is_call: bool,
    ) -> Self {
        Self {
            payment_date,
            nominal,
            accrual_start,
            accrual_end,
            day_counter,
            fixing,
            strike,
            digital_amount,
            is_call,
            gearing: 1.0,
            spread: 0.0,
        }
    }

    /// Effective rate considering the digital feature.
    pub fn effective_rate(&self) -> f64 {
        let underlying_rate = self.gearing * self.fixing + self.spread;
        let in_the_money = if self.is_call {
            self.fixing >= self.strike
        } else {
            self.fixing <= self.strike
        };
        if in_the_money {
            underlying_rate + self.digital_amount
        } else {
            underlying_rate
        }
    }
}

impl CashFlow for DigitalIborCoupon {
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

impl Coupon for DigitalIborCoupon {
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
// DigitalCmsCoupon (G80)
// ===========================================================================

/// CMS coupon with a digital (binary) payoff feature (G80).
///
/// If the CMS rate exceeds the strike, pays a digital amount;
/// otherwise pays the CMS rate (plus gearing/spread).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DigitalCmsCoupon {
    pub payment_date: Date,
    pub nominal: f64,
    pub accrual_start: Date,
    pub accrual_end: Date,
    pub day_counter: DayCounter,
    /// CMS rate fixing.
    pub cms_rate: f64,
    /// Strike for the digital feature.
    pub strike: f64,
    /// Digital payout if in-the-money.
    pub digital_amount: f64,
    /// Whether this is a call-digital (true) or put-digital (false).
    pub is_call: bool,
    /// Gearing.
    pub gearing: f64,
    /// Spread.
    pub spread: f64,
}

impl DigitalCmsCoupon {
    pub fn new(
        payment_date: Date,
        nominal: f64,
        accrual_start: Date,
        accrual_end: Date,
        day_counter: DayCounter,
        cms_rate: f64,
        strike: f64,
        digital_amount: f64,
        is_call: bool,
    ) -> Self {
        Self {
            payment_date,
            nominal,
            accrual_start,
            accrual_end,
            day_counter,
            cms_rate,
            strike,
            digital_amount,
            is_call,
            gearing: 1.0,
            spread: 0.0,
        }
    }

    /// Effective coupon rate.
    pub fn effective_rate(&self) -> f64 {
        let underlying = self.gearing * self.cms_rate + self.spread;
        let in_the_money = if self.is_call {
            self.cms_rate >= self.strike
        } else {
            self.cms_rate <= self.strike
        };
        if in_the_money {
            underlying + self.digital_amount
        } else {
            underlying
        }
    }
}

impl CashFlow for DigitalCmsCoupon {
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

impl Coupon for DigitalCmsCoupon {
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
// CapFlooredInflationCoupon (G81)
// ===========================================================================

/// Capped and/or floored inflation coupon (G81).
///
/// The effective inflation rate is clamped: max(floor, min(cap, rate)).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CapFlooredInflationCoupon {
    pub payment_date: Date,
    pub nominal: f64,
    pub accrual_start: Date,
    pub accrual_end: Date,
    pub day_counter: DayCounter,
    /// Inflation rate (e.g., YoY rate or zero-coupon inflation rate).
    pub inflation_rate: f64,
    /// Cap rate (None = no cap).
    pub cap: Option<f64>,
    /// Floor rate (None = no floor).
    pub floor: Option<f64>,
    /// Gearing.
    pub gearing: f64,
    /// Spread.
    pub spread: f64,
}

impl CapFlooredInflationCoupon {
    pub fn new(
        payment_date: Date,
        nominal: f64,
        accrual_start: Date,
        accrual_end: Date,
        day_counter: DayCounter,
        inflation_rate: f64,
        cap: Option<f64>,
        floor: Option<f64>,
    ) -> Self {
        Self {
            payment_date,
            nominal,
            accrual_start,
            accrual_end,
            day_counter,
            inflation_rate,
            cap,
            floor,
            gearing: 1.0,
            spread: 0.0,
        }
    }

    /// Effective rate after applying cap/floor.
    pub fn effective_rate(&self) -> f64 {
        let raw = self.gearing * self.inflation_rate + self.spread;
        let capped = if let Some(c) = self.cap {
            raw.min(c)
        } else {
            raw
        };
        if let Some(f) = self.floor {
            capped.max(f)
        } else {
            capped
        }
    }
}

impl CashFlow for CapFlooredInflationCoupon {
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

impl Coupon for CapFlooredInflationCoupon {
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
// CpiCouponPricer (G82)
// ===========================================================================

/// Pricer for CPI-linked coupons with optional convexity adjustment (G82).
///
/// Computes the inflation-adjusted coupon amount using:
///   amount = notional × (CPI_fixing / CPI_base) × year_fraction × rate
/// with optional convexity adjustment for timing mismatch.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CpiCouponPricer {
    /// Nominal coupon rate.
    pub rate: f64,
    /// Year fraction.
    pub year_fraction: f64,
    /// CPI at the base date.
    pub base_cpi: f64,
    /// CPI at the fixing date.
    pub fixing_cpi: f64,
    /// Convexity adjustment (additive, in rate terms).
    pub convexity_adjustment: f64,
}

impl CpiCouponPricer {
    pub fn new(
        rate: f64,
        year_fraction: f64,
        base_cpi: f64,
        fixing_cpi: f64,
    ) -> Self {
        Self {
            rate,
            year_fraction,
            base_cpi,
            fixing_cpi,
            convexity_adjustment: 0.0,
        }
    }

    /// Set convexity adjustment.
    pub fn with_convexity_adjustment(mut self, adj: f64) -> Self {
        self.convexity_adjustment = adj;
        self
    }

    /// Adjusted rate after convexity correction.
    pub fn adjusted_rate(&self) -> f64 {
        self.rate + self.convexity_adjustment
    }

    /// CPI index ratio.
    pub fn index_ratio(&self) -> f64 {
        if self.base_cpi.abs() < 1e-15 {
            return 1.0;
        }
        self.fixing_cpi / self.base_cpi
    }

    /// Coupon amount for a given notional.
    pub fn amount(&self, notional: f64) -> f64 {
        notional * self.index_ratio() * self.adjusted_rate() * self.year_fraction
    }

    /// Swap NPV-equivalent for a single CPI coupon.
    pub fn swap_rate(&self) -> f64 {
        self.index_ratio() * self.adjusted_rate()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_time::Month;

    fn ref_date() -> Date {
        Date::from_ymd(2025, Month::January, 15)
    }

    #[test]
    fn rate_averaging_compound() {
        let rates = vec![0.05, 0.051, 0.052];
        let yfs = vec![1.0 / 12.0; 3];
        let r = compute_averaged_rate(&rates, &yfs, RateAveraging::Compound);
        // ~5.1%
        assert!(r > 0.05 && r < 0.053);
    }

    #[test]
    fn rate_averaging_simple() {
        let rates = vec![0.05, 0.051, 0.052];
        let yfs = vec![1.0 / 12.0; 3];
        let r = compute_averaged_rate(&rates, &yfs, RateAveraging::Simple);
        assert_abs_diff_eq!(r, 0.051, epsilon = 1e-10);
    }

    #[test]
    fn yoy_inflation_coupon() {
        let cpn = YoYInflationCoupon::new(
            ref_date() + 365,
            1_000_000.0,
            ref_date(),
            ref_date() + 365,
            DayCounter::Actual365Fixed,
            100.0,
            103.0, // 3% inflation
            1.0,
            0.0,
        );
        assert_abs_diff_eq!(cpn.yoy_rate(), 0.03, epsilon = 1e-10);
        assert_abs_diff_eq!(cpn.amount(), 30_000.0, epsilon = 1.0);
    }

    #[test]
    fn zero_inflation_cashflow() {
        let cf = ZeroInflationCashFlow::new(ref_date() + 365, 1_000_000.0, 100.0, 110.0);
        assert_abs_diff_eq!(cf.index_ratio(), 1.1, epsilon = 1e-10);
        assert_abs_diff_eq!(cf.amount(), 1_100_000.0, epsilon = 1e-6);
    }

    #[test]
    fn equity_cashflow() {
        let cf = EquityCashFlow::new(ref_date() + 365, 1_000_000.0, 100.0, 120.0, 1.0);
        assert_abs_diff_eq!(cf.performance(), 0.20, epsilon = 1e-10);
        assert_abs_diff_eq!(cf.amount(), 200_000.0, epsilon = 1e-6);
    }

    #[test]
    fn digital_ibor_coupon_in_the_money() {
        let cpn = DigitalIborCoupon::new(
            ref_date() + 180,
            1_000_000.0,
            ref_date(),
            ref_date() + 180,
            DayCounter::Actual360,
            0.055, // fixing
            0.05,  // strike
            0.01,  // digital payout
            true,  // call
        );
        // ITM: fixing >= strike => pays underlying + digital
        let rate = cpn.effective_rate();
        assert_abs_diff_eq!(rate, 0.055 + 0.01, epsilon = 1e-10);
    }

    #[test]
    fn digital_ibor_coupon_out_of_the_money() {
        let cpn = DigitalIborCoupon::new(
            ref_date() + 180,
            1_000_000.0,
            ref_date(),
            ref_date() + 180,
            DayCounter::Actual360,
            0.045, // fixing < strike
            0.05,  // strike
            0.01,  // digital payout
            true,  // call
        );
        let rate = cpn.effective_rate();
        assert_abs_diff_eq!(rate, 0.045, epsilon = 1e-10);
    }

    #[test]
    fn cap_floored_inflation_coupon() {
        // Floor at 0%, cap at 5%, inflation at -1% => floored to 0%
        let cpn = CapFlooredInflationCoupon::new(
            ref_date() + 365,
            1_000_000.0,
            ref_date(),
            ref_date() + 365,
            DayCounter::Actual365Fixed,
            -0.01,      // inflation rate
            Some(0.05),  // cap
            Some(0.0),   // floor
        );
        assert_abs_diff_eq!(cpn.effective_rate(), 0.0, epsilon = 1e-10);

        // Inflation at 6% => capped to 5%
        let cpn2 = CapFlooredInflationCoupon::new(
            ref_date() + 365,
            1_000_000.0,
            ref_date(),
            ref_date() + 365,
            DayCounter::Actual365Fixed,
            0.06,
            Some(0.05),
            Some(0.0),
        );
        assert_abs_diff_eq!(cpn2.effective_rate(), 0.05, epsilon = 1e-10);
    }

    #[test]
    fn cpi_coupon_pricer() {
        let pricer = CpiCouponPricer::new(0.02, 1.0, 100.0, 105.0)
            .with_convexity_adjustment(0.001);
        assert_abs_diff_eq!(pricer.index_ratio(), 1.05, epsilon = 1e-10);
        assert_abs_diff_eq!(pricer.adjusted_rate(), 0.021, epsilon = 1e-10);
        let amt = pricer.amount(1_000_000.0);
        assert_abs_diff_eq!(amt, 1_000_000.0 * 1.05 * 0.021, epsilon = 1.0);
    }
}
