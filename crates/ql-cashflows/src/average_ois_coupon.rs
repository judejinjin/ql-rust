//! Average OIS coupon — arithmetic average of daily overnight fixings.
//!
//! Distinct from [`OvernightIndexedCoupon`] (compounded) this coupon applies
//! the **arithmetic average** convention used by CME/SOFR averages and some
//! SONIA-LIBOR transition instruments.
//!
//! ## Rate Convention
//!
//! The average-rate coupon accrues interest at:
//! ```text
//! rate = (1/T) * Σ r_i * d_i  +  spread
//! ```
//! where `r_i` is the daily overnight fixing and `d_i` is the day fraction.
//!
//! When forecasted from a discount curve the formula simplifies to:
//! ```text
//! rate ≈ (df_start / df_end - 1) / accrual_period  +  spread
//! ```
//!
//! ## Lockout / Lookback
//!
//! The [`LockoutMode`] controls which fixing is used near the payment date:
//! - `None` — use the actual fixing each day
//! - `Lockout { days }` — freeze fixing `days` periods before period end
//! - `Lookback { days }` — observe fixing shifted `days` periods back

use serde::{Deserialize, Serialize};

// =========================================================================
// Lockout / Lookback convention
// =========================================================================

/// Convention for how overnight fixings near the cut-off are handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LockoutMode {
    /// No adjustment — use each day's actual fixing.
    None,
    /// Freeze the fixing `days` business days before the period ends.
    Lockout {
        /// Number of business days before period end.
        days: u32,
    },
    /// Observe the fixing shifted `days` back relative to the accrual date.
    Lookback {
        /// Number of business days to look back.
        days: u32,
    },
}

// =========================================================================
// AverageOisRate  — formula-level rate computation
// =========================================================================

/// Compute an arithmetic average OIS rate from a discount-curve representation.
///
/// In the limit of daily resets the arithmetic average converges to:
/// ```text
/// rate = (df_start / df_end - 1) / accrual_fraction
/// ```
///
/// # Arguments
/// - `df_start` — discount factor at the start of the accrual period
/// - `df_end`   — discount factor at the end of the accrual period
/// - `accrual_fraction` — year fraction of the accrual period (from day counter)
/// - `spread`   — additive spread (zero for flat OIS)
///
/// # Panics
/// Panics in debug mode if `accrual_fraction <= 0` or `df_end <= 0`.
pub fn average_ois_rate(
    df_start: f64,
    df_end: f64,
    accrual_fraction: f64,
    spread: f64,
) -> f64 {
    debug_assert!(accrual_fraction > 0.0, "accrual_fraction must be positive");
    debug_assert!(df_end > 0.0, "df_end must be positive");
    (df_start / df_end - 1.0) / accrual_fraction + spread
}

/// Compute the PV01 (basis-point sensitivity) of an average OIS coupon.
///
/// A 1bp (0.0001) upward parallel shift in the spread changes the present
/// value by:
/// ```text
/// pv01 = notional * accrual_fraction * df_payment * 1e-4
/// ```
pub fn average_ois_pv01(notional: f64, accrual_fraction: f64, df_payment: f64) -> f64 {
    notional * accrual_fraction * df_payment * 1e-4
}

// =========================================================================
// AverageOisCoupon struct
// =========================================================================

/// An arithmetic-average OIS coupon (SOFR-average style).
///
/// Stores the pre-computed inputs needed to value the coupon.  The rate is
/// evaluated lazily from discount factors via [`average_ois_rate`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AverageOisCoupon {
    /// Nominal (face value).
    pub notional: f64,
    /// Accrual period as a year fraction.
    pub accrual_fraction: f64,
    /// Discount factor at period start (e.g. from the OIS curve).
    pub df_start: f64,
    /// Discount factor at period end.
    pub df_end: f64,
    /// Discount factor to payment date (may differ from `df_end` in arrears).
    pub df_payment: f64,
    /// Additive spread over the averaged OIS rate.
    pub spread: f64,
    /// Lockout / lookback convention.
    pub lockout_mode: LockoutMode,
}

impl AverageOisCoupon {
    /// Create a new average OIS coupon.
    pub fn new(
        notional: f64,
        accrual_fraction: f64,
        df_start: f64,
        df_end: f64,
        df_payment: f64,
        spread: f64,
        lockout_mode: LockoutMode,
    ) -> Self {
        Self { notional, accrual_fraction, df_start, df_end, df_payment, spread, lockout_mode }
    }

    /// The floating rate for this period.
    pub fn rate(&self) -> f64 {
        average_ois_rate(self.df_start, self.df_end, self.accrual_fraction, self.spread)
    }

    /// The undiscounted coupon amount.
    pub fn amount(&self) -> f64 {
        self.notional * self.rate() * self.accrual_fraction
    }

    /// Present value of the coupon.
    pub fn pv(&self) -> f64 {
        self.amount() * self.df_payment
    }

    /// PV01 (DV01) sensitivity to a 1 bp parallel spread shift.
    pub fn pv01(&self) -> f64 {
        average_ois_pv01(self.notional, self.accrual_fraction, self.df_payment)
    }
}

// =========================================================================
// AverageOisLeg  — multi-period leg
// =========================================================================

/// A sequence of average OIS coupons forming one leg of a swap.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AverageOisLeg {
    /// Individual coupons.
    pub coupons: Vec<AverageOisCoupon>,
}

impl AverageOisLeg {
    /// Create from a vector of coupons.
    pub fn new(coupons: Vec<AverageOisCoupon>) -> Self {
        Self { coupons }
    }

    /// Net present value of the leg.
    pub fn npv(&self) -> f64 {
        self.coupons.iter().map(|c| c.pv()).sum()
    }

    /// Total PV01 of the leg.
    pub fn pv01(&self) -> f64 {
        self.coupons.iter().map(|c| c.pv01()).sum()
    }

    /// Par spread — the flat spread that makes `npv = 0` assuming a fixed
    /// funding leg of the same notional is priced at par.
    ///
    /// Returns `None` if total PV01 is effectively zero.
    pub fn par_spread(&self) -> Option<f64> {
        let pv01_total: f64 = self.coupons.iter().map(|c| c.pv01()).sum();
        if pv01_total.abs() < 1e-15 {
            return None;
        }
        Some(-self.npv() / (pv01_total * 1e4)) // convert from bp to decimal
    }
}

// =========================================================================
// Convenience constructor
// =========================================================================

/// Build a flat single-period average OIS coupon for a given tenor.
///
/// Useful for quick sensitivity analysis and unit testing.
///
/// # Arguments
/// - `notional`           — face value of the position
/// - `tenor_years`        — accrual horizon in years
/// - `ois_rate`           — flat OIS rate (annualised, e.g. 0.05 for 5 %)
/// - `spread`             — additive spread over OIS
/// - `df_payment`         — discount factor to the payment date
pub fn simple_average_ois_coupon(
    notional: f64,
    tenor_years: f64,
    ois_rate: f64,
    spread: f64,
    df_payment: f64,
) -> AverageOisCoupon {
    // df_start = 1, df_end derived from flat rate
    let df_end = 1.0 / (1.0 + ois_rate * tenor_years);
    AverageOisCoupon::new(
        notional,
        tenor_years,
        1.0,      // df_start = spot
        df_end,
        df_payment,
        spread,
        LockoutMode::None,
    )
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_equals_implied_ois() {
        // 5% flat OIS, 1Y coupon — rate should recover 5 %
        let c = simple_average_ois_coupon(1_000_000.0, 1.0, 0.05, 0.0, 1.0 / 1.05);
        let r = c.rate();
        assert!((r - 0.05).abs() < 1e-8, "rate={}", r);
    }

    #[test]
    fn amount_positive() {
        let c = simple_average_ois_coupon(1_000_000.0, 0.5, 0.04, 0.0, 1.0 / 1.02);
        assert!(c.amount() > 0.0);
    }

    #[test]
    fn pv01_scales_with_notional() {
        let c1 = simple_average_ois_coupon(1_000_000.0, 1.0, 0.05, 0.0, 0.95);
        let c2 = simple_average_ois_coupon(2_000_000.0, 1.0, 0.05, 0.0, 0.95);
        let ratio = c2.pv01() / c1.pv01();
        assert!((ratio - 2.0).abs() < 1e-10, "ratio={}", ratio);
    }

    #[test]
    fn leg_npv_sum_of_coupons() {
        let coupons: Vec<_> = (0..4).map(|i| {
            let t = (i + 1) as f64 * 0.25;
            simple_average_ois_coupon(1_000_000.0, 0.25, 0.05, 0.001, (-0.05 * t).exp())
        }).collect();
        let leg = AverageOisLeg::new(coupons.clone());
        let expected: f64 = coupons.iter().map(|c| c.pv()).sum();
        assert!((leg.npv() - expected).abs() < 1e-6);
    }

    #[test]
    fn lockout_mode_serialises() {
        let mode = LockoutMode::Lockout { days: 2 };
        let json = serde_json::to_string(&mode).unwrap();
        let back: LockoutMode = serde_json::from_str(&json).unwrap();
        assert_eq!(mode, back);
    }
}
