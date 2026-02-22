//! CPI (Consumer Price Index) / Zero-Inflation Swap.
//!
//! A CPI swap exchanges:
//! - **Inflation leg**: notional × (CPI(T)/CPI(0) − 1) paid at maturity (or
//!   as a strip of periodic CPI-linked coupons)
//! - **Fixed or floating leg**: conventional fixed-rate or Ibor + spread
//!   coupons on the same schedule
//!
//! The Rust implementation follows QuantLib's `CPISwap` but uses a simpler
//! data-oriented approach with explicit schedules (year-fractions).
//!
//! ## References
//! - Kenyon, C. & Stamm, R. (2012). *Discounting, LIBOR, CVA and Funding*.
//! - QuantLib `CPISwap` in `ql/instruments/cpiswap.hpp`.

use serde::{Deserialize, Serialize};

/// Direction of the CPI-inflation leg relative to the counterparty.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CpiSwapType {
    /// Receive CPI-linked cashflows, pay fixed/floating.
    Receiver,
    /// Pay CPI-linked cashflows, receive fixed/floating.
    Payer,
}

/// Payment convention for the inflation leg.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InflationLegType {
    /// Single terminal payment: notional × (CPI(T)/CPI(0) − 1).
    ZeroCoupon,
    /// Periodic payments: notional × τᵢ × (CPI(tᵢ)/CPI(tᵢ₋₁) − 1).
    YearOnYear,
}

/// A single cashflow on the funding (fixed or floating) leg.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingCashFlow {
    /// Accrual period in years.
    pub accrual_fraction: f64,
    /// Coupon rate (fixed) or projected Ibor rate + spread.
    pub rate: f64,
    /// Notional.
    pub notional: f64,
    /// Discount factor at payment date.
    pub discount: f64,
    /// Payment time (years from valuation).
    pub payment_time: f64,
}

impl FundingCashFlow {
    /// Present value of this cashflow.
    pub fn pv(&self) -> f64 {
        self.notional * self.accrual_fraction * self.rate * self.discount
    }
}

/// A single inflation-linked coupon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InflationCashFlow {
    /// Start-of-period CPI (or base CPI for ZC leg).
    pub cpi_start: f64,
    /// End-of-period CPI (or terminal CPI for ZC leg).
    pub cpi_end: f64,
    /// Notional.
    pub notional: f64,
    /// Accrual fraction (used for YoY leg; = T for ZC leg).
    pub accrual_fraction: f64,
    /// Discount factor at payment date.
    pub discount: f64,
    /// Payment time (years from valuation).
    pub payment_time: f64,
    /// If true, also exchange notional at maturity.
    pub notional_exchange: bool,
}

impl InflationCashFlow {
    /// CPI ratio (I(end) / I(start)).
    pub fn cpi_ratio(&self) -> f64 {
        self.cpi_end / self.cpi_start
    }

    /// Inflation-adjusted amount (before discounting).
    /// For ZC: notional × (CPI_T/CPI_0 − 1).
    /// For YoY: notional × τ × (CPI_t/CPI_{t-1} − 1).
    pub fn undiscounted_amount(&self) -> f64 {
        let ratio = self.cpi_ratio();
        if self.notional_exchange {
            // Full inflated notional payment
            self.notional * ratio
        } else {
            self.notional * self.accrual_fraction * (ratio - 1.0)
        }
    }

    /// Present value of this cashflow.
    pub fn pv(&self) -> f64 {
        self.undiscounted_amount() * self.discount
    }
}

/// A CPI (zero-inflation) swap.
///
/// Exchanges inflation-linked cashflows against a fixed or floating funding leg.
///
/// # Example — Zero-Coupon CPI swap (single exchange at maturity)
/// ```
/// use ql_instruments::cpi_swap::*;
///
/// let inflation_cf = InflationCashFlow {
///     cpi_start: 100.0,
///     cpi_end: 115.0,
///     notional: 1_000_000.0,
///     accrual_fraction: 5.0,
///     discount: 0.85,
///     payment_time: 5.0,
///     notional_exchange: false,
/// };
///
/// let funding_cf = FundingCashFlow {
///     accrual_fraction: 5.0,
///     rate: 0.03,
///     notional: 1_000_000.0,
///     discount: 0.85,
///     payment_time: 5.0,
/// };
///
/// let swap = CpiSwap {
///     swap_type: CpiSwapType::Receiver,
///     leg_type: InflationLegType::ZeroCoupon,
///     inflation_leg: vec![inflation_cf],
///     funding_leg: vec![funding_cf],
/// };
///
/// let (npv, _) = swap.npv();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpiSwap {
    /// Whether we receive or pay the inflation leg.
    pub swap_type: CpiSwapType,
    /// Periodic (YoY) or terminal (ZeroCoupon) inflation cashflows.
    pub leg_type: InflationLegType,
    /// Inflation-linked leg cashflows.
    pub inflation_leg: Vec<InflationCashFlow>,
    /// Fixed / floating funding leg cashflows.
    pub funding_leg: Vec<FundingCashFlow>,
}

impl CpiSwap {
    /// Compute NPV and break-even inflation rate.
    ///
    /// Returns `(npv, breakeven_inflation_rate)`.
    /// The break-even inflation rate is the CPI ratio implied by a zero-NPV swap.
    pub fn npv(&self) -> (f64, f64) {
        let infl_pv: f64 = self.inflation_leg.iter().map(|c| c.pv()).sum();
        let fund_pv: f64 = self.funding_leg.iter().map(|c| c.pv()).sum();

        let sign = match self.swap_type {
            CpiSwapType::Receiver => 1.0_f64,
            CpiSwapType::Payer => -1.0_f64,
        };
        let npv = sign * (infl_pv - fund_pv);

        // Break-even: find CPI ratio so that infl_pv == fund_pv.
        // For ZC swap with one cashflow: N × (r − 1) × df = fund_pv → r = 1 + fund_pv/(N×df)
        let breakeven = if let Some(cf) = self.inflation_leg.last() {
            1.0 + fund_pv / (cf.notional * cf.discount)
        } else {
            1.0
        };

        (npv, breakeven)
    }

    /// PV of the inflation leg alone.
    pub fn inflation_leg_pv(&self) -> f64 {
        self.inflation_leg.iter().map(|c| c.pv()).sum()
    }

    /// PV of the funding leg alone.
    pub fn funding_leg_pv(&self) -> f64 {
        self.funding_leg.iter().map(|c| c.pv()).sum()
    }

    /// Fair break-even inflation rate implied by zero NPV.
    ///
    /// For a ZC swap: the annualised CPI growth rate `(CPI_T/CPI_0)^(1/T) − 1`.
    pub fn fair_inflation_rate(&self) -> Option<f64> {
        if self.inflation_leg.is_empty() {
            return None;
        }
        let fund_pv = self.funding_leg_pv();
        // Sum of notional × discount on inflation leg (annuity measure)
        let ann: f64 = self.inflation_leg.iter()
            .map(|c| c.notional * c.accrual_fraction * c.discount)
            .sum();
        if ann.abs() < 1e-12 {
            return None;
        }
        Some(fund_pv / ann)
    }

    /// Duration 01: sensitivity of NPV to + 1 bp shift in funding rates.
    pub fn dv01(&self) -> f64 {
        0.0001 * self.funding_leg.iter().map(|c| c.notional * c.accrual_fraction * c.discount).sum::<f64>()
    }
}

// =========================================================================
// Convenience constructors
// =========================================================================

/// Build a zero-coupon CPI swap from market data.
///
/// # Parameters
/// - `notional`: swap notional
/// - `base_cpi`: CPI index level at inception (base date)
/// - `forward_cpi`: projected CPI index level at maturity
/// - `tenor_years`: swap maturity in years
/// - `fixed_rate`: fixed funding rate (annualised)
/// - `risk_free_df`: risk-free discount factor to maturity
/// - `swap_type`: receiver or payer
pub fn zero_coupon_cpi_swap(
    notional: f64,
    base_cpi: f64,
    forward_cpi: f64,
    tenor_years: f64,
    fixed_rate: f64,
    risk_free_df: f64,
    swap_type: CpiSwapType,
) -> CpiSwap {
    let inflation_leg = vec![InflationCashFlow {
        cpi_start: base_cpi,
        cpi_end: forward_cpi,
        notional,
        accrual_fraction: tenor_years,
        discount: risk_free_df,
        payment_time: tenor_years,
        notional_exchange: false,
    }];
    let funding_leg = vec![FundingCashFlow {
        accrual_fraction: tenor_years,
        rate: fixed_rate,
        notional,
        discount: risk_free_df,
        payment_time: tenor_years,
    }];
    CpiSwap { swap_type, leg_type: InflationLegType::ZeroCoupon, inflation_leg, funding_leg }
}

/// Build a year-on-year CPI swap from a schedule of CPI fixings.
///
/// # Parameters
/// - `notional`: swap notional
/// - `cpi_schedule`: `[(cpi_prev, cpi_curr, accrual_frac, df, t_pay)]` per period
/// - `fixed_rate`: constant fixed funding rate
/// - `swap_type`: receiver or payer
pub fn yoy_cpi_swap(
    notional: f64,
    cpi_schedule: Vec<(f64, f64, f64, f64, f64)>,
    fixed_rate: f64,
    swap_type: CpiSwapType,
) -> CpiSwap {
    let inflation_leg: Vec<InflationCashFlow> = cpi_schedule.iter().map(|&(c0, c1, tau, df, t)| {
        InflationCashFlow {
            cpi_start: c0,
            cpi_end: c1,
            notional,
            accrual_fraction: tau,
            discount: df,
            payment_time: t,
            notional_exchange: false,
        }
    }).collect();

    let funding_leg: Vec<FundingCashFlow> = cpi_schedule.iter().map(|&(_, _, tau, df, t)| {
        FundingCashFlow {
            accrual_fraction: tau,
            rate: fixed_rate,
            notional,
            discount: df,
            payment_time: t,
        }
    }).collect();

    CpiSwap { swap_type, leg_type: InflationLegType::YearOnYear, inflation_leg, funding_leg }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zc_cpi_swap_npv_sign() {
        // 5y ZC CPI swap: base CPI=100, forward CPI=115 (15% total) vs 3% fixed
        // Receiver of inflation leg
        let swap = zero_coupon_cpi_swap(1_000_000.0, 100.0, 115.0, 5.0, 0.03, 0.85, CpiSwapType::Receiver);
        let (npv, _) = swap.npv();
        // Inflation leg PV = 1e6 × 5 × (115/100 - 1) × 0.85 = 1e6 × 5 × 0.15 × 0.85 = 637_500
        // Funding leg PV   = 1e6 × 5 × 0.03 × 0.85 = 127_500
        // NPV (receiver) = 637500 - 127500 = 510000
        assert!(npv > 0.0, "receiver of high-inflation leg should be positive npv");
    }

    #[test]
    fn zc_cpi_swap_parity() {
        // Payer + Receiver = 0
        let r = zero_coupon_cpi_swap(1e6, 100.0, 110.0, 3.0, 0.02, 0.94, CpiSwapType::Receiver);
        let p = zero_coupon_cpi_swap(1e6, 100.0, 110.0, 3.0, 0.02, 0.94, CpiSwapType::Payer);
        let (npv_r, _) = r.npv();
        let (npv_p, _) = p.npv();
        assert!((npv_r + npv_p).abs() < 1e-8, "receiver + payer = 0");
    }

    #[test]
    fn breakeven_inflation_round_trip() {
        // At fair rate, NPV should be near zero
        let notional = 1_000_000.0;
        let base_cpi = 100.0;
        let fwd_cpi = 112.0;
        let tenor = 4.0;
        let df = 0.88;
        // Compute what fixed rate makes NPV = 0:
        // fund_pv = infl_pv = N × T × (ratio-1) × df = 1e6 × 4 × 0.12 × 0.88 = 422400
        // fixed rate = fund_pv / (N × T × df) = 422400 / (1e6 × 4 × 0.88) = 0.12
        let fair_rate = (fwd_cpi / base_cpi - 1.0); // = 0.12
        let swap = zero_coupon_cpi_swap(notional, base_cpi, fwd_cpi, tenor, fair_rate, df, CpiSwapType::Receiver);
        let (npv, _) = swap.npv();
        assert!(npv.abs() < 1.0, "fair swap NPV should be near zero, got {}", npv);
    }

    #[test]
    fn yoy_cpi_swap_positive_inflation() {
        // 3-year YoY swap: inflation 2% per year vs fixed 1.5%
        let schedule = vec![
            (100.0, 102.0, 1.0, 0.97, 1.0),
            (102.0, 104.04, 1.0, 0.94, 2.0),
            (104.04, 106.12, 1.0, 0.91, 3.0),
        ];
        let swap = yoy_cpi_swap(1_000_000.0, schedule, 0.015, CpiSwapType::Receiver);
        let (npv, _) = swap.npv();
        // Inflation leg PV > fixed leg PV when inflation > fixed rate
        assert!(npv > 0.0, "npv={}", npv);
    }

    #[test]
    fn dv01_positive() {
        let swap = zero_coupon_cpi_swap(1e6, 100.0, 110.0, 5.0, 0.025, 0.88, CpiSwapType::Receiver);
        assert!(swap.dv01() > 0.0);
    }
}
