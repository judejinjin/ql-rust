//! Extended instruments: G1–G9.
//!
//! **G1** — YearOnYearInflationSwap
//! **G2** — ZeroCouponInflationSwap
//! **G3** — FloatFloatSwaption
//! **G4** — CmsRateBond
//! **G5** — AmortizingCmsRateBond
//! **G6** — BTP (Italian government bond)
//! **G7** — DividendVanillaOption
//! **G8** — DividendBarrierOption
//! **G9** — StickyRatchet

use ql_time::{Date, DayCounter};
use serde::{Deserialize, Serialize};

// ===========================================================================
// YearOnYearInflationSwap (G1)
// ===========================================================================

/// Year-on-year inflation swap (G1).
///
/// Pays (or receives) the YoY inflation rate on one leg vs a fixed rate on
/// the other. Each period's inflation rate = CPI(end)/CPI(start) − 1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YearOnYearInflationSwap {
    pub notional: f64,
    /// Schedule dates for both legs.
    pub schedule_dates: Vec<Date>,
    /// Fixed leg rate.
    pub fixed_rate: f64,
    /// Gearing on the inflation leg.
    pub gearing: f64,
    /// Spread on the inflation leg.
    pub spread: f64,
    pub day_counter: DayCounter,
    /// True = payer of fixed, receiver of inflation.
    pub pay_fixed: bool,
}

impl YearOnYearInflationSwap {
    pub fn new(
        notional: f64,
        schedule_dates: Vec<Date>,
        fixed_rate: f64,
        day_counter: DayCounter,
        pay_fixed: bool,
    ) -> Self {
        Self {
            notional,
            schedule_dates,
            fixed_rate,
            gearing: 1.0,
            spread: 0.0,
            day_counter,
            pay_fixed,
        }
    }

    /// Set gearing and spread on the inflation leg.
    pub fn with_gearing_spread(mut self, gearing: f64, spread: f64) -> Self {
        self.gearing = gearing;
        self.spread = spread;
        self
    }

    /// Compute NPV given YoY rates for each period and discount factors.
    pub fn npv(&self, yoy_rates: &[f64], discount_factors: &[f64]) -> f64 {
        let n = self.schedule_dates.len() - 1;
        assert_eq!(yoy_rates.len(), n);
        assert_eq!(discount_factors.len(), n);

        let mut fixed_pv = 0.0;
        let mut infl_pv = 0.0;

        for i in 0..n {
            let yf = self
                .day_counter
                .year_fraction(self.schedule_dates[i], self.schedule_dates[i + 1]);
            fixed_pv += self.notional * self.fixed_rate * yf * discount_factors[i];
            let infl_rate = self.gearing * yoy_rates[i] + self.spread;
            infl_pv += self.notional * infl_rate * yf * discount_factors[i];
        }

        if self.pay_fixed {
            infl_pv - fixed_pv
        } else {
            fixed_pv - infl_pv
        }
    }
}

// ===========================================================================
// ZeroCouponInflationSwap (G2)
// ===========================================================================

/// Zero-coupon inflation swap (G2).
///
/// Single exchange at maturity:
///   Inflation leg pays: notional × (CPI_end / CPI_base − 1)
///   Fixed leg pays:     notional × ((1+K)^T − 1)
/// where K is the fixed breakeven rate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroCouponInflationSwap {
    pub notional: f64,
    pub start_date: Date,
    pub maturity_date: Date,
    pub fixed_rate: f64,
    pub base_cpi: f64,
    pub day_counter: DayCounter,
    /// True = payer of fixed, receiver of inflation.
    pub pay_fixed: bool,
}

impl ZeroCouponInflationSwap {
    pub fn new(
        notional: f64,
        start_date: Date,
        maturity_date: Date,
        fixed_rate: f64,
        base_cpi: f64,
        day_counter: DayCounter,
        pay_fixed: bool,
    ) -> Self {
        Self {
            notional,
            start_date,
            maturity_date,
            fixed_rate,
            base_cpi,
            day_counter,
            pay_fixed,
        }
    }

    /// Compute NPV given the fixing CPI and a discount factor to maturity.
    pub fn npv(&self, fixing_cpi: f64, discount_factor: f64) -> f64 {
        let t = self
            .day_counter
            .year_fraction(self.start_date, self.maturity_date);
        let inflation_amount = self.notional * (fixing_cpi / self.base_cpi - 1.0);
        let fixed_amount = self.notional * ((1.0 + self.fixed_rate).powf(t) - 1.0);

        let npv = if self.pay_fixed {
            (inflation_amount - fixed_amount) * discount_factor
        } else {
            (fixed_amount - inflation_amount) * discount_factor
        };
        npv
    }

    /// Breakeven inflation rate implied by the fixing CPI.
    pub fn breakeven_rate(&self, fixing_cpi: f64) -> f64 {
        let t = self
            .day_counter
            .year_fraction(self.start_date, self.maturity_date);
        if t.abs() < 1e-15 {
            return 0.0;
        }
        (fixing_cpi / self.base_cpi).powf(1.0 / t) - 1.0
    }
}

// ===========================================================================
// FloatFloatSwaption (G3)
// ===========================================================================

/// Swaption on a float-float swap (basis swap option) (G3).
///
/// Gives the holder the right to enter a float-float swap (e.g., 3M LIBOR
/// + spread vs 6M LIBOR) at a future date.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FloatFloatSwaption {
    /// Notional amount.
    pub notional: f64,
    /// Option expiry date.
    pub expiry: Date,
    /// Swap start date (typically = expiry for physical settlement).
    pub swap_start: Date,
    /// Swap maturity date.
    pub swap_maturity: Date,
    /// Spread on the first leg.
    pub spread1: f64,
    /// Spread on the second leg.
    pub spread2: f64,
    /// Day counter.
    pub day_counter: DayCounter,
    /// Whether the holder pays leg 1 at exercise.
    pub pay_leg1: bool,
}

impl FloatFloatSwaption {
    pub fn new(
        notional: f64,
        expiry: Date,
        swap_start: Date,
        swap_maturity: Date,
        spread1: f64,
        spread2: f64,
        day_counter: DayCounter,
    ) -> Self {
        Self {
            notional,
            expiry,
            swap_start,
            swap_maturity,
            spread1,
            spread2,
            day_counter,
            pay_leg1: true,
        }
    }

    /// Time to expiry in years.
    pub fn time_to_expiry(&self, today: Date) -> f64 {
        self.day_counter.year_fraction(today, self.expiry)
    }

    /// Swap tenor in years.
    pub fn swap_tenor(&self) -> f64 {
        self.day_counter
            .year_fraction(self.swap_start, self.swap_maturity)
    }

    /// Approximate price using a Bachelier (normal) model.
    ///
    /// `forward_spread`: forward basis spread (spread1 - spread2 implied by the market)
    /// `vol`: normal volatility of the spread
    /// `df`: discount factor to expiry
    /// `today`: valuation date
    pub fn price_bachelier(
        &self,
        forward_spread: f64,
        vol: f64,
        df: f64,
        today: Date,
    ) -> f64 {
        let te = self.time_to_expiry(today);
        let swap_tenor = self.swap_tenor();
        let strike_spread = self.spread1 - self.spread2;

        if te <= 0.0 {
            // Expired: intrinsic only
            let intrinsic = (forward_spread - strike_spread).max(0.0);
            return self.notional * swap_tenor * intrinsic * df;
        }

        let total_vol = vol * te.sqrt();
        let d = if total_vol.abs() < 1e-15 {
            if forward_spread > strike_spread {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            }
        } else {
            (forward_spread - strike_spread) / total_vol
        };
        let n_d = 0.5 * (1.0 + erf_approx(d / std::f64::consts::SQRT_2));
        let n_pdf = (-0.5 * d * d).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let annuity = self.notional * swap_tenor;
        let intrinsic = (forward_spread - strike_spread) * n_d;
        let time_value = total_vol * n_pdf;
        annuity * (intrinsic + time_value) * df
    }
}

fn erf_approx(x: f64) -> f64 {
    // Abramowitz-Stegun approximation
    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-x * x).exp())
}

// ===========================================================================
// CmsRateBond (G4)
// ===========================================================================

/// Bond paying CMS-linked coupons (G4).
///
/// Coupons are tied to a CMS rate (e.g., 10Y swap rate) with optional
/// gearing and spread.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CmsRateBond {
    pub notional: f64,
    /// Schedule dates (accrual period boundaries).
    pub schedule_dates: Vec<Date>,
    /// CMS tenor in years (e.g., 10.0 for 10Y CMS).
    pub cms_tenor: f64,
    /// Gearing on the CMS rate.
    pub gearing: f64,
    /// Spread added to gearing × CMS rate.
    pub spread: f64,
    /// Optional cap on the coupon rate.
    pub cap: Option<f64>,
    /// Optional floor on the coupon rate.
    pub floor: Option<f64>,
    pub day_counter: DayCounter,
}

impl CmsRateBond {
    pub fn new(
        notional: f64,
        schedule_dates: Vec<Date>,
        cms_tenor: f64,
        gearing: f64,
        spread: f64,
        day_counter: DayCounter,
    ) -> Self {
        Self {
            notional,
            schedule_dates,
            cms_tenor,
            gearing,
            spread,
            cap: None,
            floor: None,
            day_counter,
        }
    }

    pub fn with_cap_floor(mut self, cap: Option<f64>, floor: Option<f64>) -> Self {
        self.cap = cap;
        self.floor = floor;
        self
    }

    /// Effective coupon rate for a given CMS fixing.
    pub fn effective_rate(&self, cms_fixing: f64) -> f64 {
        let mut r = self.gearing * cms_fixing + self.spread;
        if let Some(c) = self.cap {
            r = r.min(c);
        }
        if let Some(f) = self.floor {
            r = r.max(f);
        }
        r
    }

    /// Dirty price given CMS fixings and discount factors for each period.
    pub fn dirty_price(&self, cms_fixings: &[f64], discount_factors: &[f64]) -> f64 {
        let n = self.schedule_dates.len() - 1;
        assert_eq!(cms_fixings.len(), n);
        assert_eq!(discount_factors.len(), n);

        let mut pv = 0.0;
        for i in 0..n {
            let yf = self
                .day_counter
                .year_fraction(self.schedule_dates[i], self.schedule_dates[i + 1]);
            let rate = self.effective_rate(cms_fixings[i]);
            pv += self.notional * rate * yf * discount_factors[i];
        }
        // Add principal redemption
        pv += self.notional * discount_factors[n - 1];
        pv / self.notional * 100.0
    }
}

// ===========================================================================
// AmortizingCmsRateBond (G5)
// ===========================================================================

/// Amortizing bond with CMS-linked coupons (G5).
///
/// Like `CmsRateBond` but the notional reduces over time according to an
/// amortization schedule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmortizingCmsRateBond {
    pub original_notional: f64,
    /// Remaining notional for each period (length = schedule_dates.len() - 1).
    pub notionals: Vec<f64>,
    pub schedule_dates: Vec<Date>,
    pub cms_tenor: f64,
    pub gearing: f64,
    pub spread: f64,
    pub cap: Option<f64>,
    pub floor: Option<f64>,
    pub day_counter: DayCounter,
}

impl AmortizingCmsRateBond {
    pub fn new(
        original_notional: f64,
        notionals: Vec<f64>,
        schedule_dates: Vec<Date>,
        cms_tenor: f64,
        gearing: f64,
        spread: f64,
        day_counter: DayCounter,
    ) -> Self {
        Self {
            original_notional,
            notionals,
            schedule_dates,
            cms_tenor,
            gearing,
            spread,
            cap: None,
            floor: None,
            day_counter,
        }
    }

    pub fn with_cap_floor(mut self, cap: Option<f64>, floor: Option<f64>) -> Self {
        self.cap = cap;
        self.floor = floor;
        self
    }

    pub fn effective_rate(&self, cms_fixing: f64) -> f64 {
        let mut r = self.gearing * cms_fixing + self.spread;
        if let Some(c) = self.cap {
            r = r.min(c);
        }
        if let Some(f) = self.floor {
            r = r.max(f);
        }
        r
    }

    /// Dirty price given CMS fixings and discount factors.
    pub fn dirty_price(&self, cms_fixings: &[f64], discount_factors: &[f64]) -> f64 {
        let n = self.schedule_dates.len() - 1;
        assert_eq!(cms_fixings.len(), n);
        assert_eq!(discount_factors.len(), n);
        assert_eq!(self.notionals.len(), n);

        let mut pv = 0.0;
        for i in 0..n {
            let yf = self
                .day_counter
                .year_fraction(self.schedule_dates[i], self.schedule_dates[i + 1]);
            let rate = self.effective_rate(cms_fixings[i]);
            // Coupon on remaining notional
            pv += self.notionals[i] * rate * yf * discount_factors[i];
            // Amortization payment
            let amort = if i > 0 {
                (self.notionals[i - 1] - self.notionals[i]).max(0.0)
            } else {
                (self.original_notional - self.notionals[0]).max(0.0)
            };
            pv += amort * discount_factors[i];
        }
        // Final principal
        pv += self.notionals[n - 1] * discount_factors[n - 1];
        pv / self.original_notional * 100.0
    }
}

// ===========================================================================
// BTP (G6)
// ===========================================================================

/// Italian government bond (BTP, BOT, CCT, BTPS) (G6).
///
/// BTP (Buono del Tesoro Poliennale) is a fixed-rate Italian government bond.
/// BOT is a zero-coupon treasury bill. This struct handles the BTP case with
/// semi-annual coupons and Italian market conventions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Btp {
    pub notional: f64,
    /// Annual coupon rate.
    pub coupon_rate: f64,
    /// Schedule dates.
    pub schedule_dates: Vec<Date>,
    /// Day counter (typically ActualActual ISMA for BTPs).
    pub day_counter: DayCounter,
    /// Settlement days (typically 2 for BTPs on MOT).
    pub settlement_days: u32,
}

impl Btp {
    pub fn new(
        notional: f64,
        coupon_rate: f64,
        schedule_dates: Vec<Date>,
        day_counter: DayCounter,
    ) -> Self {
        Self {
            notional,
            coupon_rate,
            schedule_dates,
            day_counter,
            settlement_days: 2,
        }
    }

    /// Dirty price given discount factors for each coupon date.
    pub fn dirty_price(&self, discount_factors: &[f64]) -> f64 {
        let n = self.schedule_dates.len() - 1;
        assert_eq!(discount_factors.len(), n);

        let mut pv = 0.0;
        for i in 0..n {
            let yf = self
                .day_counter
                .year_fraction(self.schedule_dates[i], self.schedule_dates[i + 1]);
            pv += self.notional * self.coupon_rate * yf * discount_factors[i];
        }
        pv += self.notional * discount_factors[n - 1];
        pv / self.notional * 100.0
    }

    /// Accrued interest at settlement date.
    pub fn accrued_interest(&self, settlement: Date) -> f64 {
        for i in 0..(self.schedule_dates.len() - 1) {
            if settlement >= self.schedule_dates[i] && settlement < self.schedule_dates[i + 1] {
                let yf = self
                    .day_counter
                    .year_fraction(self.schedule_dates[i], settlement);
                return self.notional * self.coupon_rate * yf;
            }
        }
        0.0
    }

    /// Clean price = dirty price − accrued interest.
    pub fn clean_price(&self, discount_factors: &[f64], settlement: Date) -> f64 {
        let dirty = self.dirty_price(discount_factors);
        let accrued_pct = self.accrued_interest(settlement) / self.notional * 100.0;
        dirty - accrued_pct
    }
}

// ===========================================================================
// DividendVanillaOption (G7)
// ===========================================================================

/// Vanilla option with discrete cash or proportional dividends (G7).
///
/// The exact dividend treatment depends on the pricing engine; this struct
/// stores the dividend schedule alongside the standard option parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DividendVanillaOption {
    pub spot: f64,
    pub strike: f64,
    pub risk_free_rate: f64,
    pub volatility: f64,
    pub expiry: f64,
    pub option_type: OptionType,
    /// Discrete dividend schedule: (time to ex-date, amount or proportion).
    pub dividends: Vec<DividendEntry>,
}

/// A single discrete dividend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DividendEntry {
    /// Time to ex-dividend date (in years from valuation).
    pub time: f64,
    /// For cash dividends: fixed amount. For proportional: fraction of spot.
    pub amount: f64,
    /// Whether this is a proportional dividend.
    pub is_proportional: bool,
}

/// Option type (call/put).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptionType {
    Call,
    Put,
}

impl DividendVanillaOption {
    pub fn new(
        spot: f64,
        strike: f64,
        risk_free_rate: f64,
        volatility: f64,
        expiry: f64,
        option_type: OptionType,
        dividends: Vec<DividendEntry>,
    ) -> Self {
        Self {
            spot,
            strike,
            risk_free_rate,
            volatility,
            expiry,
            option_type,
            dividends,
        }
    }

    /// Adjust spot for discrete cash dividends (escrowed model).
    /// S_adj = S − Σ D_i × e^{−r × t_i}
    pub fn adjusted_spot(&self) -> f64 {
        let mut adj = self.spot;
        for div in &self.dividends {
            if div.time < self.expiry {
                if div.is_proportional {
                    adj *= 1.0 - div.amount;
                } else {
                    adj -= div.amount * (-self.risk_free_rate * div.time).exp();
                }
            }
        }
        adj
    }

    /// Price using Black-Scholes on the dividend-adjusted spot.
    pub fn price_bs_escrowed(&self) -> f64 {
        let s = self.adjusted_spot();
        let k = self.strike;
        let r = self.risk_free_rate;
        let v = self.volatility;
        let t = self.expiry;
        if t <= 0.0 || v <= 0.0 {
            return match self.option_type {
                OptionType::Call => (s - k).max(0.0),
                OptionType::Put => (k - s).max(0.0),
            };
        }
        let d1 = ((s / k).ln() + (r + 0.5 * v * v) * t) / (v * t.sqrt());
        let d2 = d1 - v * t.sqrt();
        let nd1 = norm_cdf(d1);
        let nd2 = norm_cdf(d2);
        match self.option_type {
            OptionType::Call => s * nd1 - k * (-r * t).exp() * nd2,
            OptionType::Put => k * (-r * t).exp() * norm_cdf(-d2) - s * norm_cdf(-d1),
        }
    }
}

fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))
}

// ===========================================================================
// DividendBarrierOption (G8)
// ===========================================================================

/// Barrier option with discrete dividends (G8).
///
/// Extends barrier option analysis with a discrete dividend schedule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DividendBarrierOption {
    pub spot: f64,
    pub strike: f64,
    pub barrier: f64,
    pub rebate: f64,
    pub barrier_type: DividendBarrierType,
    pub option_type: OptionType,
    pub risk_free_rate: f64,
    pub volatility: f64,
    pub expiry: f64,
    pub dividends: Vec<DividendEntry>,
}

/// Barrier type for dividend barrier options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DividendBarrierType {
    DownIn,
    DownOut,
    UpIn,
    UpOut,
}

impl DividendBarrierOption {
    pub fn new(
        spot: f64,
        strike: f64,
        barrier: f64,
        rebate: f64,
        barrier_type: DividendBarrierType,
        option_type: OptionType,
        risk_free_rate: f64,
        volatility: f64,
        expiry: f64,
        dividends: Vec<DividendEntry>,
    ) -> Self {
        Self {
            spot,
            strike,
            barrier,
            rebate,
            barrier_type,
            option_type,
            risk_free_rate,
            volatility,
            expiry,
            dividends,
        }
    }

    /// Adjust spot for discrete cash dividends.
    pub fn adjusted_spot(&self) -> f64 {
        let mut adj = self.spot;
        for div in &self.dividends {
            if div.time < self.expiry {
                if div.is_proportional {
                    adj *= 1.0 - div.amount;
                } else {
                    adj -= div.amount * (-self.risk_free_rate * div.time).exp();
                }
            }
        }
        adj
    }

    /// Whether the barrier has been breached at inception.
    pub fn is_knocked(&self) -> bool {
        match self.barrier_type {
            DividendBarrierType::DownIn | DividendBarrierType::DownOut => {
                self.spot <= self.barrier
            }
            DividendBarrierType::UpIn | DividendBarrierType::UpOut => {
                self.spot >= self.barrier
            }
        }
    }
}

// ===========================================================================
// StickyRatchet (G9)
// ===========================================================================

/// Sticky/ratchet structured rate coupon (G9).
///
/// A structured coupon where the rate is determined by a "sticky" or "ratchet"
/// mechanism based on the previous period's rate:
///
/// Sticky:  rate_i = min(cap, max(floor, rate_{i-1} + gearing × (index − index_prev)))
/// Ratchet: rate_i = min(cap, max(floor, rate_{i-1} + gearing × (index − strike)))
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StickyRatchet {
    /// Accrual dates (n+1 entries for n periods).
    pub schedule_dates: Vec<Date>,
    /// Day counter.
    pub day_counter: DayCounter,
    /// Initial coupon rate.
    pub initial_rate: f64,
    /// Gearing on the index change.
    pub gearing: f64,
    /// Cap on coupon rate.
    pub cap: f64,
    /// Floor on coupon rate.
    pub floor: f64,
    /// Notional.
    pub notional: f64,
    /// Mechanism type.
    pub mechanism: StickyRatchetType,
}

/// Mechanism type for sticky/ratchet coupons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StickyRatchetType {
    /// Rate adjusts based on index change from previous period.
    Sticky,
    /// Rate adjusts based on index difference from a fixed strike.
    Ratchet,
}

impl StickyRatchet {
    pub fn new(
        schedule_dates: Vec<Date>,
        day_counter: DayCounter,
        initial_rate: f64,
        gearing: f64,
        cap: f64,
        floor: f64,
        notional: f64,
        mechanism: StickyRatchetType,
    ) -> Self {
        Self {
            schedule_dates,
            day_counter,
            initial_rate,
            gearing,
            cap,
            floor,
            notional,
            mechanism,
        }
    }

    /// Compute coupon rates for each period given index fixings.
    ///
    /// For Sticky: `index_fixings` are the index levels at each fixing date.
    /// For Ratchet: the first element is the strike, remaining are index fixings.
    pub fn compute_rates(&self, index_fixings: &[f64]) -> Vec<f64> {
        let n = self.schedule_dates.len() - 1;
        let mut rates = Vec::with_capacity(n);
        let mut prev_rate = self.initial_rate;

        for i in 0..n {
            let rate = match self.mechanism {
                StickyRatchetType::Sticky => {
                    if i == 0 || index_fixings.len() <= i {
                        prev_rate
                    } else {
                        let delta = index_fixings[i] - index_fixings[i - 1];
                        (prev_rate + self.gearing * delta)
                            .min(self.cap)
                            .max(self.floor)
                    }
                }
                StickyRatchetType::Ratchet => {
                    if index_fixings.is_empty() || index_fixings.len() <= i + 1 {
                        prev_rate
                    } else {
                        let strike = index_fixings[0];
                        let delta = index_fixings[i + 1] - strike;
                        (prev_rate + self.gearing * delta)
                            .min(self.cap)
                            .max(self.floor)
                    }
                }
            };
            rates.push(rate);
            prev_rate = rate;
        }
        rates
    }

    /// Compute coupon amounts given rates and discount factors.
    pub fn coupon_amounts(&self, rates: &[f64]) -> Vec<f64> {
        let n = self.schedule_dates.len() - 1;
        assert_eq!(rates.len(), n);
        (0..n)
            .map(|i| {
                let yf = self
                    .day_counter
                    .year_fraction(self.schedule_dates[i], self.schedule_dates[i + 1]);
                self.notional * rates[i] * yf
            })
            .collect()
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

    fn annual_schedule(n_years: u32) -> Vec<Date> {
        (0..=n_years)
            .map(|y| ref_date() + (y * 365) as i32)
            .collect()
    }

    #[test]
    fn yoy_inflation_swap_npv() {
        let swap = YearOnYearInflationSwap::new(
            1_000_000.0,
            annual_schedule(3),
            0.02,
            DayCounter::Actual365Fixed,
            true,
        );
        let yoy_rates = vec![0.03, 0.025, 0.028];
        let dfs = vec![0.98, 0.96, 0.94];
        let npv = swap.npv(&yoy_rates, &dfs);
        // Inflation leg > fixed leg, so payer-of-fixed benefits
        assert!(npv > 0.0);
    }

    #[test]
    fn zero_coupon_inflation_swap() {
        let swap = ZeroCouponInflationSwap::new(
            1_000_000.0,
            ref_date(),
            ref_date() + 1825,
            0.02,
            100.0,
            DayCounter::Actual365Fixed,
            true,
        );
        // CPI goes from 100 to 112 over 5 years
        let npv = swap.npv(112.0, 0.95);
        // Inflation return = 12%, fixed = (1.02^5 - 1) ≈ 10.4%
        assert!(npv > 0.0);

        let be = swap.breakeven_rate(112.0);
        assert!(be > 0.02); // breakeven > fixed rate means inflation leg wins
    }

    #[test]
    fn cms_rate_bond_price() {
        let bond = CmsRateBond::new(
            100.0,
            annual_schedule(3),
            10.0,
            1.0,
            0.005,
            DayCounter::Actual365Fixed,
        );
        let fixings = vec![0.03, 0.032, 0.035];
        let dfs = vec![0.97, 0.94, 0.91];
        let price = bond.dirty_price(&fixings, &dfs);
        assert!(price > 80.0 && price < 120.0);
    }

    #[test]
    fn btp_dirty_price() {
        let btp = Btp::new(
            100.0,
            0.03,
            annual_schedule(5),
            DayCounter::Actual365Fixed,
        );
        let dfs: Vec<f64> = (1..=5)
            .map(|i| (-0.04 * i as f64).exp())
            .collect();
        let price = btp.dirty_price(&dfs);
        assert!(price > 90.0 && price < 110.0, "BTP price: {}", price);
    }

    #[test]
    fn dividend_vanilla_option_bs() {
        let opt = DividendVanillaOption::new(
            100.0,
            100.0,
            0.05,
            0.20,
            1.0,
            OptionType::Call,
            vec![DividendEntry {
                time: 0.5,
                amount: 2.0,
                is_proportional: false,
            }],
        );
        let adj_spot = opt.adjusted_spot();
        assert!(adj_spot < 100.0);

        let price = opt.price_bs_escrowed();
        assert!(price > 0.0 && price < 20.0, "option price: {}", price);
    }

    #[test]
    fn dividend_barrier_option_knocked() {
        let opt = DividendBarrierOption::new(
            95.0, 100.0, 95.0, 0.0,
            DividendBarrierType::DownIn,
            OptionType::Put,
            0.05, 0.20, 1.0,
            vec![],
        );
        assert!(opt.is_knocked());
    }

    #[test]
    fn sticky_ratchet_rates() {
        let sr = StickyRatchet::new(
            annual_schedule(4),
            DayCounter::Actual365Fixed,
            0.03,
            0.5,
            0.10,
            0.0,
            1_000_000.0,
            StickyRatchetType::Sticky,
        );
        let fixings = vec![0.03, 0.035, 0.04, 0.032];
        let rates = sr.compute_rates(&fixings);
        assert_eq!(rates.len(), 4);
        // First period = initial rate
        assert_abs_diff_eq!(rates[0], 0.03, epsilon = 1e-10);
        // Second period: 0.03 + 0.5 × (0.035 - 0.03) = 0.0325
        assert_abs_diff_eq!(rates[1], 0.0325, epsilon = 1e-10);
    }
}
