//! BMA/SIFMA municipal floating-rate index.
//!
//! The BMA (Bond Market Association) Index — now known as the SIFMA Municipal
//! Swap Index — is a weekly floating rate used in U.S. tax-exempt interest-rate
//! swaps. Key characteristics:
//!
//! - Weekly reset (every Wednesday, effective Thursday)
//! - Tax-exempt money-market instruments basis
//! - Historically ~65–70% of USD LIBOR (due to tax exemption)
//! - Day count: Actual/360
//!
//! This module provides:
//! - [`BmaIndex`]: the rate index with fixing storage and projection
//! - [`BmaSwapFixedLeg`]: helper for the fixed leg of a BMA swap
//!
//! Reference: ISDA definitions for BMA Municipal Swap Index.

use std::collections::BTreeMap;

// =========================================================================
// BmaIndex
// =========================================================================

/// BMA/SIFMA municipal floating-rate index.
///
/// Stores weekly rate fixings and provides:
/// - Historical rate lookup by date serial key
/// - Forward rate projection (flat extrapolation of last fixing)
/// - Day-count (Act/360) year fraction for accrual
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BmaIndex {
    /// Index name (default: "BMA").
    pub name: String,
    /// Typical BMA/LIBOR ratio (historically ~0.67 for tax-exempt basis).
    pub bma_libor_ratio: f64,
    /// Day-count convention denominator (360 for Act/360).
    pub day_count_denom: f64,
    /// Stored fixings: date_serial → rate (annualised, simple).
    fixings: BTreeMap<u32, f64>,
    /// Fallback rate if no fixing is available.
    pub fallback_rate: f64,
}

impl BmaIndex {
    /// Create a new BMA index.
    ///
    /// # Parameters
    /// - `name`: index name (e.g. "BMA", "SIFMA")
    /// - `fallback_rate`: the rate to return if no fixing is registered
    pub fn new(name: impl Into<String>, fallback_rate: f64) -> Self {
        Self {
            name: name.into(),
            bma_libor_ratio: 0.67,
            day_count_denom: 360.0,
            fixings: BTreeMap::new(),
            fallback_rate,
        }
    }

    /// Standard BMA/SIFMA index with a fallback rate.
    pub fn bma(fallback_rate: f64) -> Self {
        Self::new("BMA", fallback_rate)
    }

    /// Standard SIFMA index (same as BMA, different branding).
    pub fn sifma(fallback_rate: f64) -> Self {
        Self::new("SIFMA", fallback_rate)
    }

    /// Register a weekly fixing at `date_serial` (integer date key).
    /// Rate should be annualised, e.g. 0.035 for 3.5%.
    pub fn add_fixing(&mut self, date_serial: u32, rate: f64) {
        self.fixings.insert(date_serial, rate);
    }

    /// Look up a fixing at `date_serial`. Returns `None` if not registered.
    pub fn fixing(&self, date_serial: u32) -> Option<f64> {
        self.fixings.get(&date_serial).copied()
    }

    /// Projected rate at `date_serial`:
    /// - Returns the registered fixing if present.
    /// - Otherwise returns the most recent earlier fixing.
    /// - Falls back to `fallback_rate` if no fixings exist.
    pub fn projected_rate(&self, date_serial: u32) -> f64 {
        if let Some(&r) = self.fixings.get(&date_serial) {
            return r;
        }
        // Most recent earlier fixing
        if let Some((_, &r)) = self.fixings.range(..date_serial).next_back() {
            return r;
        }
        self.fallback_rate
    }

    /// Year fraction for a period of `n_days` calendar days (Act/360).
    pub fn year_fraction(&self, n_days: u32) -> f64 {
        n_days as f64 / self.day_count_denom
    }

    /// Accrued interest for a notional of `notional` over `n_days` calendar
    /// days at the fixing rate on `date_serial`.
    pub fn accrued_interest(&self, date_serial: u32, notional: f64, n_days: u32) -> f64 {
        let rate = self.projected_rate(date_serial);
        let yf = self.year_fraction(n_days);
        notional * rate * yf
    }

    /// Estimate implied LIBOR equivalent rate: `r_bma / bma_libor_ratio`.
    pub fn equivalent_libor(&self, date_serial: u32) -> f64 {
        let r = self.projected_rate(date_serial);
        if self.bma_libor_ratio.abs() < 1e-15 { return r; }
        r / self.bma_libor_ratio
    }

    /// Set a custom BMA/LIBOR ratio (default 0.67).
    pub fn set_bma_libor_ratio(&mut self, ratio: f64) {
        assert!(ratio > 0.0 && ratio < 1.0, "BMA/LIBOR ratio should be in (0,1)");
        self.bma_libor_ratio = ratio;
    }

    /// Name of this index.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Number of registered fixings.
    pub fn n_fixings(&self) -> usize {
        self.fixings.len()
    }

    /// All fixings as a sorted vector of (date_serial, rate).
    pub fn all_fixings(&self) -> Vec<(u32, f64)> {
        self.fixings.iter().map(|(&k, &v)| (k, v)).collect()
    }
}

// =========================================================================
// BmaSwapFixedLeg — helper
// =========================================================================

/// Fixed leg of a BMA swap: pays a constant fixed rate against BMA floating.
///
/// The fixed rate is typically expressed as a spread to a benchmark
/// (e.g. "65% of LIBOR + 10 bps") or as an absolute coupon.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BmaSwapFixedLeg {
    /// Notional principal.
    pub notional: f64,
    /// Fixed coupon rate (annualised).
    pub fixed_rate: f64,
    /// Payment frequency per year.
    pub payments_per_year: u32,
    /// Tenor in years.
    pub tenor: f64,
    /// Day-count denominator (360 for muni swaps, 365 for some conventions).
    pub day_count_denom: f64,
}

impl BmaSwapFixedLeg {
    /// Create a fixed leg.
    pub fn new(notional: f64, fixed_rate: f64, payments_per_year: u32, tenor: f64) -> Self {
        Self { notional, fixed_rate, payments_per_year, tenor, day_count_denom: 360.0 }
    }

    /// PV of fixed leg given flat discount rate `r` (continuously compounded).
    pub fn pv(&self, r: f64) -> f64 {
        let n = (self.tenor * self.payments_per_year as f64).round() as usize;
        if n == 0 { return 0.0; }
        let dt = 1.0 / self.payments_per_year as f64;
        let coupon = self.notional * self.fixed_rate * dt;
        let mut pv = 0.0;
        for i in 1..=n {
            let t = i as f64 * dt;
            pv += coupon * (-r * t).exp();
        }
        // Return of principal
        pv += self.notional * (-r * self.tenor).exp();
        pv
    }

    /// Par fixed rate given flat discount rate (for comparison / calibration).
    pub fn par_rate(&self, r: f64) -> f64 {
        let n = (self.tenor * self.payments_per_year as f64).round() as usize;
        if n == 0 { return 0.0; }
        let dt = 1.0 / self.payments_per_year as f64;
        let p_tenor = (-r * self.tenor).exp();
        let annuity: f64 = (1..=n).map(|i| dt * (-r * i as f64 * dt).exp()).sum();
        if annuity < 1e-15 { return r; }
        (1.0 - p_tenor) / annuity
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bma_index_name() {
        let idx = BmaIndex::bma(0.035);
        assert_eq!(idx.name(), "BMA");
    }

    #[test]
    fn sifma_index_name() {
        let idx = BmaIndex::sifma(0.03);
        assert_eq!(idx.name(), "SIFMA");
    }

    #[test]
    fn fixing_lookup() {
        let mut idx = BmaIndex::bma(0.03);
        idx.add_fixing(1000, 0.035);
        idx.add_fixing(1007, 0.034);
        assert_eq!(idx.fixing(1000), Some(0.035));
        assert_eq!(idx.fixing(1007), Some(0.034));
        assert_eq!(idx.fixing(999), None);
    }

    #[test]
    fn projected_rate_fallback() {
        let idx = BmaIndex::bma(0.025);
        let r = idx.projected_rate(5000);
        assert_eq!(r, 0.025, "fallback rate should be used");
    }

    #[test]
    fn projected_rate_last_fixing() {
        let mut idx = BmaIndex::bma(0.025);
        idx.add_fixing(1000, 0.035);
        // Query a date after the only fixing
        let r = idx.projected_rate(1003);
        assert_eq!(r, 0.035, "should return last known fixing");
    }

    #[test]
    fn accrued_interest() {
        let mut idx = BmaIndex::bma(0.04);
        idx.add_fixing(0, 0.04);
        // ACT/360: 7 days / 360
        let accrued = idx.accrued_interest(0, 1_000_000.0, 7);
        let expected = 1_000_000.0 * 0.04 * 7.0 / 360.0;
        assert!((accrued - expected).abs() < 1e-6, "accrued = {}, expected = {}", accrued, expected);
    }

    #[test]
    fn equivalent_libor() {
        let mut idx = BmaIndex::bma(0.03);
        idx.add_fixing(1000, 0.035);
        let lib = idx.equivalent_libor(1000);
        let expected = 0.035 / 0.67;
        assert!((lib - expected).abs() < 1e-12, "equiv LIBOR = {}", lib);
    }

    #[test]
    fn fixed_leg_pv_decreasing_in_rate() {
        let leg = BmaSwapFixedLeg::new(1_000_000.0, 0.04, 4, 5.0);
        let pv1 = leg.pv(0.03);
        let pv2 = leg.pv(0.05);
        assert!(pv1 > pv2, "PV should decrease as discount rate rises");
    }

    #[test]
    fn par_rate_roundtrip() {
        let leg = BmaSwapFixedLeg::new(1_000_000.0, 0.04, 4, 5.0);
        // At the par rate, PV should equal notional
        let par = leg.par_rate(0.04);
        let leg_par = BmaSwapFixedLeg::new(1_000_000.0, par, 4, 5.0);
        let pv = leg_par.pv(0.04);
        assert!((pv - 1_000_000.0).abs() < 10.0, "par leg PV = {}", pv);
    }
}
