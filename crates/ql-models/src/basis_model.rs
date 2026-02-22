//! Basis spread models for multi-curve and cross-currency frameworks.
//!
//! This module implements two important basis conventions:
//!
//! ## FRA-OIS Basis
//!
//! The **FRA-OIS basis** is the spread between a LIBOR/EURIBOR FRA rate and
//! the market rate implied by a matching OIS swap:
//! ```text
//! basis_fra_ois = FRA_rate - OIS_implied_rate
//! ```
//! Historically ~10-15 bp in calm markets, widening significantly in stress.
//! Post-LIBOR cessation, this measures the spread between term rates (SOFR
//! Term, €STR Term) and compound overnight to the same maturity.
//!
//! ## Cross-Currency Basis
//!
//! The **xccy basis** is the spread on the non-USD leg of a floating–floating
//! cross-currency basis swap in which both legs pay O/N–compounded rates plus
//! the basis.  A negative EUR/USD xccy basis means EUR is scarce relative to
//! USD (investors pay to lend USD in exchange for EUR collateral).
//!
//! ```text
//! PV_USD(tenor) = notional * [df_USD(T) - 1 + Σ (SOFR_i + spread_xccy) * τ_i * df_USD(t_i)]
//! PV_EUR(tenor) = notional_EUR * [df_EUR(T) - 1 + Σ ESTR_i * τ_i * df_EUR(t_i)]
//! ```
//!
//! At fair value: `PV_USD = FX * PV_EUR` which constrains the xccy basis.

use serde::{Deserialize, Serialize};

// =========================================================================
// FRA-OIS Basis Model
// =========================================================================

/// A simple FRA-OIS basis model parameterised by a term structure of spreads.
///
/// The basis for each tenor is stored as a flat spread.  For a single-curve
/// approximation the same spread applies to all payment dates within the
/// tenor bucket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FraOisBasisModel {
    /// Tenor grid in years (e.g. [0.25, 0.5, 1.0, 2.0, …]).
    pub tenors: Vec<f64>,
    /// Basis spreads in units of rate (e.g. 0.001 = 10 bp) at each tenor.
    pub spreads: Vec<f64>,
}

impl FraOisBasisModel {
    /// Create a new FRA-OIS basis model.
    ///
    /// # Panics
    /// Panics if `tenors` and `spreads` have different lengths, or if `tenors`
    /// is not strictly increasing.
    pub fn new(tenors: Vec<f64>, spreads: Vec<f64>) -> Self {
        assert_eq!(tenors.len(), spreads.len(), "tenors and spreads must match");
        assert!(tenors.windows(2).all(|w| w[0] < w[1]), "tenors must be strictly increasing");
        Self { tenors, spreads }
    }

    /// Interpolate the FRA-OIS basis spread at a given tenor (in years).
    ///
    /// Uses piecewise-linear interpolation with flat extrapolation at the
    /// boundaries.
    pub fn basis_at(&self, tenor: f64) -> f64 {
        if self.tenors.is_empty() { return 0.0; }
        if tenor <= self.tenors[0] { return self.spreads[0]; }
        let n = self.tenors.len();
        if tenor >= self.tenors[n - 1] { return self.spreads[n - 1]; }
        // Linear interpolation
        for i in 0..n - 1 {
            if tenor < self.tenors[i + 1] {
                let t0 = self.tenors[i];
                let t1 = self.tenors[i + 1];
                let s0 = self.spreads[i];
                let s1 = self.spreads[i + 1];
                return s0 + (s1 - s0) * (tenor - t0) / (t1 - t0);
            }
        }
        self.spreads[n - 1]
    }

    /// Implied LIBOR/term rate = OIS rate + basis spread.
    pub fn fra_rate_from_ois(&self, ois_rate: f64, tenor: f64) -> f64 {
        ois_rate + self.basis_at(tenor)
    }

    /// OIS-implied rate = FRA rate − basis spread.
    pub fn ois_rate_from_fra(&self, fra_rate: f64, tenor: f64) -> f64 {
        fra_rate - self.basis_at(tenor)
    }

    /// Present value of the basis spread stream.
    ///
    /// Computes the PV of paying the FRA-OIS differential on a unit notional
    /// over a strip of `n_periods` quarterly payments using:
    /// ```text
    /// pv_basis = Σ basis_i * τ_i * df_i
    /// ```
    /// Assumes a flat OIS discount curve with rate `ois_flat`.
    pub fn pv_basis_strip(
        &self,
        ois_flat: f64,
        n_periods: usize,
        period_frac: f64,
    ) -> f64 {
        (0..n_periods)
            .map(|i| {
                let t = (i + 1) as f64 * period_frac;
                let df = (-ois_flat * t).exp();
                self.basis_at(t) * period_frac * df
            })
            .sum()
    }
}

// =========================================================================
// Cross-Currency Basis Model
// =========================================================================

/// Currency pair for a cross-currency basis swap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CcyPair {
    EurUsd,
    GbpUsd,
    JpyUsd,
    AudUsd,
    Other,
}

/// A cross-currency basis spread model.
///
/// The xccy basis is quoted as the spread added to the non-USD OIS leg.
/// Convention: positive = non-USD party receives extra; negative (typical
/// post-GFC) = non-USD party pays extra to receive USD.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossCurrencyBasisModel {
    /// Currency pair (non-USD / USD).
    pub pair: CcyPair,
    /// Tenor grid in years for the xccy basis term structure.
    pub tenors: Vec<f64>,
    /// xccy basis spreads (e.g. −0.0015 for −15 bp).
    pub basis_spreads: Vec<f64>,
}

impl CrossCurrencyBasisModel {
    /// Create a new cross-currency basis model.
    pub fn new(pair: CcyPair, tenors: Vec<f64>, basis_spreads: Vec<f64>) -> Self {
        assert_eq!(tenors.len(), basis_spreads.len());
        assert!(tenors.windows(2).all(|w| w[0] < w[1]));
        Self { pair, tenors, basis_spreads }
    }

    /// Interpolate xccy basis at the given tenor (flat extrapolation).
    pub fn xccy_basis_at(&self, tenor: f64) -> f64 {
        if self.tenors.is_empty() { return 0.0; }
        let n = self.tenors.len();
        if tenor <= self.tenors[0] { return self.basis_spreads[0]; }
        if tenor >= self.tenors[n - 1] { return self.basis_spreads[n - 1]; }
        for i in 0..n - 1 {
            if tenor < self.tenors[i + 1] {
                let t0 = self.tenors[i];
                let t1 = self.tenors[i + 1];
                let s0 = self.basis_spreads[i];
                let s1 = self.basis_spreads[i + 1];
                return s0 + (s1 - s0) * (tenor - t0) / (t1 - t0);
            }
        }
        self.basis_spreads[n - 1]
    }

    /// Compute the PV (in domestic/non-USD currency) of receiving the xccy
    /// basis spread on a unit notional over a flat discount curve.
    ///
    /// ```text
    /// pv = Σ xccy_basis_i * τ_i * df_dom_i
    /// ```
    pub fn pv_xccy_basis_stream(
        &self,
        dom_flat_rate: f64,
        n_periods: usize,
        period_frac: f64,
    ) -> f64 {
        (0..n_periods)
            .map(|i| {
                let t = (i + 1) as f64 * period_frac;
                let df = (-dom_flat_rate * t).exp();
                self.xccy_basis_at(t) * period_frac * df
            })
            .sum()
    }

    /// Fair-value FX forward implied by cross-currency basis.
    ///
    /// Modifies the covered-interest-parity FX forward by the basis:
    /// ```text
    /// F = S * exp((r_for - r_dom + xccy_basis(T)) * T)
    /// ```
    pub fn fx_forward_with_basis(
        &self,
        spot_fx: f64,
        r_for: f64,
        r_dom: f64,
        tenor: f64,
    ) -> f64 {
        let basis = self.xccy_basis_at(tenor);
        spot_fx * ((r_for - r_dom + basis) * tenor).exp()
    }
}

// =========================================================================
// Average OIS Basis Swap  — convenience pricer
// =========================================================================

/// Result from pricing a basis swap between two OIS rates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasisSwapResult {
    /// Fair-value basis spread (in rate units, e.g. 0.001 = 10 bp).
    pub fair_basis: f64,
    /// PV of receiving the fixed basis spread (per unit notional).
    pub pv_receive: f64,
    /// PV of paying the fixed basis spread (per unit notional).
    pub pv_pay: f64,
    /// Total DV01 of the basis swap (per unit notional).
    pub dv01: f64,
}

/// Price a simple flat basis swap between two OIS rates.
///
/// The payer pays `rate_a + spread`, the receiver pays `rate_b`.
/// At fair value `spread = rate_b − rate_a` averaged over the curve.
///
/// # Arguments
/// - `rate_a`        — OIS rate on leg A (annualised)
/// - `rate_b`        — OIS rate on leg B (annualised)
/// - `n_periods`     — number of payment periods
/// - `period_frac`   — year fraction per period
/// - `df_discount`   — discount factors at each payment date
pub fn price_basis_swap(
    rate_a: f64,
    rate_b: f64,
    n_periods: usize,
    period_frac: f64,
    df_discount: &[f64],
) -> BasisSwapResult {
    assert_eq!(df_discount.len(), n_periods, "need one df per period");

    let annuity: f64 = df_discount.iter().map(|&df| period_frac * df).sum();
    let pv_a: f64 = df_discount.iter().map(|&df| rate_a * period_frac * df).sum();
    let pv_b: f64 = df_discount.iter().map(|&df| rate_b * period_frac * df).sum();

    let fair_basis = (pv_b - pv_a) / annuity;
    let dv01 = annuity * 1e-4;

    BasisSwapResult {
        fair_basis,
        pv_receive: pv_b - pv_a,
        pv_pay: pv_a - pv_b,
        dv01,
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fra_ois_interpolation() {
        let model = FraOisBasisModel::new(
            vec![0.25, 0.5, 1.0, 2.0, 5.0],
            vec![0.0010, 0.0011, 0.0012, 0.0013, 0.0015],
        );
        // Exact nodes
        assert!((model.basis_at(1.0) - 0.0012).abs() < 1e-10);
        // Linear interpolation between 1Y and 2Y
        let mid = model.basis_at(1.5);
        assert!((mid - 0.00125).abs() < 1e-10, "mid={}", mid);
        // Extrapolation
        assert!((model.basis_at(10.0) - 0.0015).abs() < 1e-10);
    }

    #[test]
    fn fra_rate_from_ois() {
        let model = FraOisBasisModel::new(vec![1.0], vec![0.0012]);
        let ois = 0.05;
        let fra = model.fra_rate_from_ois(ois, 1.0);
        assert!((fra - 0.0512).abs() < 1e-10, "fra={}", fra);
    }

    #[test]
    fn xccy_basis_negative_typical() {
        // EUR/USD basis typically negative
        let model = CrossCurrencyBasisModel::new(
            CcyPair::EurUsd,
            vec![1.0, 2.0, 5.0, 10.0],
            vec![-0.0010, -0.0015, -0.0020, -0.0025],
        );
        assert!(model.xccy_basis_at(3.0) < 0.0);
    }

    #[test]
    fn fx_forward_with_basis_near_cip() {
        // When basis ≈ 0, forward should ≈ CIP
        let model = CrossCurrencyBasisModel::new(
            CcyPair::EurUsd,
            vec![1.0],
            vec![0.0],
        );
        let s = 1.10;
        let r_for = 0.03;
        let r_dom = 0.05;
        let t = 1.0;
        let fwd = model.fx_forward_with_basis(s, r_for, r_dom, t);
        let cip = s * ((r_for - r_dom) * t).exp();
        assert!((fwd - cip).abs() < 1e-10);
    }

    #[test]
    fn basis_swap_fair_value() {
        // When rate_a == rate_b the fair basis is zero
        let dfs: Vec<f64> = (1..=4).map(|i| (-0.04 * i as f64 * 0.25).exp()).collect();
        let res = price_basis_swap(0.04, 0.04, 4, 0.25, &dfs);
        assert!(res.fair_basis.abs() < 1e-12, "fair_basis={}", res.fair_basis);
        assert!(res.dv01 > 0.0);
    }
}
