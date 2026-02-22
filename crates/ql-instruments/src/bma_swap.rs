//! BMA (Bond Market Association) swap instrument.
//!
//! A BMA swap (also known as a SIFMA swap in the US) exchanges payments
//! based on the BMA/SIFMA floating rate index for payments on a LIBOR-based
//! leg.  The BMA swap is widely used in the municipal bond market because
//! BMA rates are tax-exempt.
//!
//! ## Structure
//!
//! ```text
//! Party A pays:  BMA_t  × Notional × α_BMA  (weekly reset, actual/actual)
//! Party B pays:  LIBOR_t × θ × Notional × α_LIBOR  (3-month reset, actual/360)
//! ```
//!
//! where θ ∈ (0, 1] is the "LIBOR percentage" (typically ~0.67 for the BMA/LIBOR ratio).
//!
//! ## Pricing
//!
//! The NPV of the BMA leg equals the NPV of the LIBOR percentage × LIBOR leg when
//! fair.  The fair BMA rate given a fixed percentage q is:
//!
//! `BMA_fair = q × LIBOR_par_rate`
//!
//! ## References
//!
//! - Ametrano, F. & Bianchetti, M. (2009), *Bootstrapping the illiquidity*.
//! - Levin, A. (2002), *Interest rate model selection*, in Frank Fabozzi (ed.),
//!   *The Handbook of Fixed Income Securities*.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Instrument struct
// ---------------------------------------------------------------------------

/// A BMA (SIFMA) floating-for-floating swap.
///
/// Exchanges a BMA-indexed floating leg for a LIBOR × θ leg.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BmaSwap {
    /// Swap notional in units of currency.
    pub notional: f64,
    /// Total tenor in years.
    pub tenor_years: f64,
    /// BMA rate (annualized, e.g. 0.035 = 3.5%).
    /// This is the fixed observed BMA rate or the fair BMA rate.
    pub bma_rate: f64,
    /// LIBOR rate (annualized, e.g. 0.05 = 5%).
    pub libor_rate: f64,
    /// LIBOR percentage θ: fraction of LIBOR that equals fair BMA.
    /// Typically ~0.65–0.70 in steady state.
    pub libor_pct: f64,
    /// Day count convention for the BMA leg (e.g., Actual/Actual).
    pub bma_day_count: BmaDayCount,
    /// Day count convention for the LIBOR leg (e.g., Actual/360).
    pub libor_day_count: LiborDayCount,
    /// BMA leg reset frequency (weeks, default: weekly = 52 resets/year).
    pub bma_resets_per_year: u32,
    /// LIBOR leg reset frequency (default: quarterly = 4 resets/year).
    pub libor_resets_per_year: u32,
    /// Whether the BMA payer is position (true) or receiver (false).
    pub pay_bma: bool,
}

/// Day count convention for the BMA leg.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BmaDayCount {
    /// Actual/Actual (ISMA).
    ActualActual,
    /// Actual/365 Fixed.
    Actual365,
}

/// Day count convention for the LIBOR leg.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum LiborDayCount {
    /// Actual/360.
    Actual360,
    /// Actual/365 Fixed.
    Actual365,
}

/// Pricing result for the BMA swap.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BmaSwapResult {
    /// Net present value (positive when BMA payer has positive NPV).
    pub npv: f64,
    /// Fair BMA rate (θ × LIBOR par rate).
    pub fair_bma_rate: f64,
    /// BMA/LIBOR ratio implied by current rates.
    pub implied_ratio: f64,
    /// PV of the BMA (floating) leg.
    pub pv_bma_leg: f64,
    /// PV of the LIBOR (floating) leg.
    pub pv_libor_leg: f64,
    /// Duration-weighted BPS sensitivity.
    pub bps: f64,
}

impl BmaSwap {
    /// Create a new BMA swap with standard weekly-for-quarterly structure.
    pub fn new(
        notional: f64,
        tenor_years: f64,
        bma_rate: f64,
        libor_rate: f64,
        libor_pct: f64,
        pay_bma: bool,
    ) -> Self {
        Self {
            notional,
            tenor_years,
            bma_rate,
            libor_rate,
            libor_pct,
            bma_day_count: BmaDayCount::ActualActual,
            libor_day_count: LiborDayCount::Actual360,
            bma_resets_per_year: 52,
            libor_resets_per_year: 4,
            pay_bma,
        }
    }

    /// Price the BMA swap given a flat discount curve `discount_rate`.
    ///
    /// Uses simple annuity approximations with the given day count conventions.
    ///
    /// # Parameters
    ///
    /// - `discount_rate` — flat continuously compounded discount rate
    ///
    /// Returns a [`BmaSwapResult`].
    pub fn price(&self, discount_rate: f64) -> BmaSwapResult {
        let n = self.tenor_years;
        let r = discount_rate;

        // BMA leg: weekly floating coupons
        let bma_dcf = match self.bma_day_count {
            BmaDayCount::ActualActual | BmaDayCount::Actual365 => 1.0 / self.bma_resets_per_year as f64,
        };
        let bma_annuity = self.annuity(r, n, self.bma_resets_per_year);
        // pv_bma = notional * rate * Σ (dcf_i * df_i) = notional * rate * bma_dcf * Σ df_i
        let pv_bma = self.notional * self.bma_rate * bma_dcf * bma_annuity;

        // LIBOR leg: quarterly floating coupons
        let libor_dcf = match self.libor_day_count {
            LiborDayCount::Actual360 => 1.0 / self.libor_resets_per_year as f64
                * (365.0 / 360.0), // act/360 adjustment
            LiborDayCount::Actual365 => 1.0 / self.libor_resets_per_year as f64,
        };
        let libor_annuity = self.annuity(r, n, self.libor_resets_per_year);
        let pv_libor = self.notional * self.libor_rate * self.libor_pct
            * libor_dcf * libor_annuity;

        // NPV = PV_receive - PV_pay
        let npv = if self.pay_bma {
            pv_libor - pv_bma
        } else {
            pv_bma - pv_libor
        };

        // Fair BMA rate: equate PV of BMA leg to PV of LIBOR percentage leg.
        // Both legs float over the same tenor; the day-count difference is captured
        // by the dcf ratio.  Annuity factors divide out since both use the same
        // discount curve over the same period (and are both scaled by notional):
        //   fair_bma * bma_dcf * bma_annuity_sum = θ * libor * libor_dcf * libor_annuity_sum
        // where annuity_sum = Σ df_i (raw, unitless).
        // Normalise by converting both annuity sums to per-year units:
        //   bma_annuity_per_yr  = bma_annuity  * bma_dcf  (each term has weight dt = 1/freq)
        //   libor_annuity_per_yr = libor_annuity * libor_dcf
        let bma_ann_pv = bma_annuity * bma_dcf;      // ≈ ∫₀ᵀ df(t) dt
        let libor_ann_pv = libor_annuity * libor_dcf; // ≈ ∫₀ᵀ df(t) dt  (same, with dcf adj)
        let fair_bma_rate = if bma_ann_pv > 1e-15 {
            self.libor_pct * self.libor_rate * libor_ann_pv / bma_ann_pv
        } else {
            0.0
        };

        let implied_ratio = self.bma_rate / self.libor_rate.max(1e-10);

        // BPS of the BMA leg (PV of 1bp change in BMA rate)
        let bps = self.notional * 1e-4 * bma_dcf * bma_annuity;

        BmaSwapResult {
            npv,
            fair_bma_rate,
            implied_ratio,
            pv_bma_leg: pv_bma,
            pv_libor_leg: pv_libor,
            bps,
        }
    }

    /// Annuity factor: present value of 1 per period for n years with
    /// freq payments per year, discounted at continuous rate r.
    fn annuity(&self, r: f64, n: f64, freq: u32) -> f64 {
        let dt = 1.0 / freq as f64;
        let n_payments = (n * freq as f64).round() as usize;
        (0..n_payments)
            .map(|i| (-(i as f64 + 1.0) * dt * r).exp())
            .sum()
    }

    /// Compute the fair swap rate such that the BMA leg PV equals the LIBOR leg PV.
    pub fn fair_rate(&self, discount_rate: f64) -> f64 {
        self.price(discount_rate).fair_bma_rate
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bma_swap_fair_rate_about_67pct_libor() {
        // With libor_pct = 0.67, fair BMA ≈ 0.67 × LIBOR
        let swap = BmaSwap::new(1_000_000.0, 5.0, 0.0335, 0.05, 0.67, true);
        let result = swap.price(0.04);
        let expected = 0.67 * 0.05;
        // Allow 5% relative error due to day count differences
        assert!((result.fair_bma_rate - expected).abs() < 0.005,
            "fair_bma={} expected~={}", result.fair_bma_rate, expected);
    }

    #[test]
    fn bma_swap_at_fair_rate_zero_npv() {
        // Use a swap with bma_rate = fair_bma_rate → NPV ≈ 0
        let libor = 0.05;
        let pct = 0.67;
        // First compute fair bma_rate
        let swap0 = BmaSwap::new(1_000_000.0, 5.0, 0.03, libor, pct, true);
        let fair = swap0.fair_rate(0.04);
        // Now create swap at fair rate
        let fair_swap = BmaSwap::new(1_000_000.0, 5.0, fair, libor, pct, true);
        let r = fair_swap.price(0.04);
        assert!(r.npv.abs() < 100.0, "NPV at fair rate should be ~0, got {}", r.npv);
    }

    #[test]
    fn bma_swap_pv_legs_positive() {
        let swap = BmaSwap::new(1_000_000.0, 5.0, 0.03, 0.05, 0.67, true);
        let result = swap.price(0.04);
        assert!(result.pv_bma_leg > 0.0);
        assert!(result.pv_libor_leg > 0.0);
    }

    #[test]
    fn bma_implied_ratio() {
        let swap = BmaSwap::new(1_000_000.0, 5.0, 0.0335, 0.05, 0.67, true);
        let r = swap.price(0.04);
        assert!((r.implied_ratio - 0.67).abs() < 0.01);
    }
}
