//! Cross-currency and basis swap rate helpers for curve bootstrapping.
//!
//! These helpers allow bootstrapping of cross-currency basis curves,
//! which represent the spread between two currencies' funding rates.
//!
//! - `CrossCurrencyBasisSwapHelper` — for XCCY basis swaps where
//!   both legs pay floating + basis spread
//! - `TenorBasisSwapHelper` — for single-currency tenor basis swaps
//!   (e.g. 3M vs 6M IBOR)
//!
//! Reference:
//! - QuantLib: CrossCurrencyBasisSwapRateHelper, TenorBasisSwapRateHelper

use serde::{Deserialize, Serialize};
use ql_time::{Date, DayCounter};

/// Cross-currency basis swap rate helper.
///
/// In a XCCY basis swap, one leg pays Libor_DOM + spread,
/// the other pays Libor_FOR flat (notional exchange at start/end).
/// The helper bootstraps the foreign discount curve given the domestic
/// curve and the basis spread.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CrossCurrencyBasisSwapHelper {
    /// Tenor in years.
    pub tenor_years: u32,
    /// Basis spread in decimal (e.g., -0.002 = -20 bps).
    pub basis_spread: f64,
    /// Settlement days.
    pub settlement_days: u32,
    /// Day counter.
    pub day_counter: DayCounter,
    /// Domestic discount factor curve pillar times.
    /// (simplified: just the discount factor at the pillar maturity)
    pub domestic_df: f64,
    /// FX spot rate (units of domestic per foreign).
    pub fx_spot: f64,
    /// Foreign floating rate index frequency (payments per year).
    pub foreign_freq: u32,
    /// Domestic floating rate index frequency.
    pub domestic_freq: u32,
}

/// Tenor basis swap rate helper.
///
/// A single-currency basis swap where one leg pays a shorter tenor (e.g. 3M)
/// and the other pays a longer tenor (e.g. 6M), with a basis spread on
/// the short-tenor leg.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TenorBasisSwapHelper {
    /// Swap tenor in years.
    pub tenor_years: u32,
    /// Basis spread on the short leg (decimal).
    pub basis_spread: f64,
    /// Short tenor in months (e.g. 3).
    pub short_tenor_months: u32,
    /// Long tenor in months (e.g. 6).
    pub long_tenor_months: u32,
    /// Day counter.
    pub day_counter: DayCounter,
    /// Discount factor at maturity.
    pub discount_factor: f64,
}

/// Result of cross-currency basis curve bootstrapping.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct XccyBootstrapResult {
    /// Pillar times (year fractions).
    pub pillar_times: Vec<f64>,
    /// Foreign discount factors.
    pub foreign_dfs: Vec<f64>,
    /// Foreign zero rates (continuous compounding).
    pub foreign_zero_rates: Vec<f64>,
    /// Implied basis spreads (for verification).
    pub implied_spreads: Vec<f64>,
}

/// Bootstrap a foreign discount curve from cross-currency basis swap quotes.
///
/// Given domestic discount factors and XCCY basis spreads at various tenors,
/// computes foreign discount factors that are consistent with the basis swap
/// quotes.
///
/// The key relation for a XCCY basis swap is:
///   Σ τ_i · DF_DOM_i · (L_DOM_i + spread) = Σ τ_j · DF_FOR_j · L_FOR_j
/// with notional exchange, leading to:
///   DF_FOR_T = DF_DOM_T · (1 + spread·Annuity_DOM) / (1 + 0·Annuity_FOR)
///
/// Simplified: DF_FOR = DF_DOM · exp(basis_spread × T)
///
/// # Arguments
/// - `reference_date` — valuation date
/// - `helpers` — cross-currency basis swap quotes
/// - `day_counter` — day counter
pub fn bootstrap_xccy_curve(
    _reference_date: Date,
    helpers: &[CrossCurrencyBasisSwapHelper],
    _day_counter: DayCounter,
) -> XccyBootstrapResult {
    if helpers.is_empty() {
        return XccyBootstrapResult {
            pillar_times: vec![],
            foreign_dfs: vec![],
            foreign_zero_rates: vec![],
            implied_spreads: vec![],
        };
    }

    let mut sorted: Vec<_> = helpers.to_vec();
    sorted.sort_by_key(|h| h.tenor_years);

    let mut pillar_times = Vec::with_capacity(sorted.len());
    let mut foreign_dfs = Vec::with_capacity(sorted.len());
    let mut foreign_zero_rates = Vec::with_capacity(sorted.len());
    let mut implied_spreads = Vec::with_capacity(sorted.len());

    for h in &sorted {
        let t = h.tenor_years as f64;
        pillar_times.push(t);

        // Simplified XCCY bootstrapping:
        // DF_FOR = DF_DOM × exp(−basis_spread × T)
        // Where basis_spread is the additional cost of borrowing in the foreign currency
        let dom_df = h.domestic_df;
        let for_df = dom_df * (-h.basis_spread * t).exp();
        foreign_dfs.push(for_df);

        // Zero rate
        let zero_rate = if t > 1e-8 { -for_df.ln() / t } else { 0.0 };
        foreign_zero_rates.push(zero_rate);

        implied_spreads.push(h.basis_spread);
    }

    XccyBootstrapResult {
        pillar_times,
        foreign_dfs,
        foreign_zero_rates,
        implied_spreads,
    }
}

/// Bootstrap a basis curve from tenor basis swap quotes.
///
/// Given a short-tenor forwarding curve and tenor basis swap spreads,
/// computes the long-tenor forwarding curve.
///
/// Simplified relation:
///   (1 + F_long · τ_long) / (Π(1 + F_short · τ_short)) = 1
///   where F_long = F_short_compounded + basis_spread
///
/// # Arguments
/// - `reference_date` — valuation date
/// - `helpers` — tenor basis swap quotes
/// - `short_tenor_zero_rates` — zero rates from the short-tenor curve at each pillar
pub fn bootstrap_tenor_basis_curve(
    _reference_date: Date,
    helpers: &[TenorBasisSwapHelper],
    short_tenor_zero_rates: &[f64],
) -> XccyBootstrapResult {
    let n = helpers.len().min(short_tenor_zero_rates.len());
    if n == 0 {
        return XccyBootstrapResult {
            pillar_times: vec![],
            foreign_dfs: vec![],
            foreign_zero_rates: vec![],
            implied_spreads: vec![],
        };
    }

    let mut pillar_times = Vec::with_capacity(n);
    let mut dfs = Vec::with_capacity(n);
    let mut zero_rates = Vec::with_capacity(n);
    let mut spreads = Vec::with_capacity(n);

    for i in 0..n {
        let h = &helpers[i];
        let t = h.tenor_years as f64;
        pillar_times.push(t);

        // Long tenor zero rate = short tenor zero rate + basis spread
        let long_zero = short_tenor_zero_rates[i] + h.basis_spread;
        zero_rates.push(long_zero);
        dfs.push((-long_zero * t).exp());
        spreads.push(h.basis_spread);
    }

    XccyBootstrapResult {
        pillar_times,
        foreign_dfs: dfs,
        foreign_zero_rates: zero_rates,
        implied_spreads: spreads,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ql_time::Month;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_xccy_zero_spread() {
        let ref_date = Date::from_ymd(2025, Month::January, 15);
        let helpers = vec![
            CrossCurrencyBasisSwapHelper {
                tenor_years: 5,
                basis_spread: 0.0,
                settlement_days: 2,
                day_counter: DayCounter::Actual365Fixed,
                domestic_df: (-0.05_f64 * 5.0).exp(),
                fx_spot: 1.10,
                foreign_freq: 4,
                domestic_freq: 4,
            },
        ];
        let res = bootstrap_xccy_curve(ref_date, &helpers, DayCounter::Actual365Fixed);
        // With zero spread, foreign DF = domestic DF
        assert_abs_diff_eq!(
            res.foreign_dfs[0], (-0.05_f64 * 5.0).exp(), epsilon = 1e-10
        );
    }

    #[test]
    fn test_xccy_negative_spread() {
        let ref_date = Date::from_ymd(2025, Month::January, 15);
        let helpers = vec![
            CrossCurrencyBasisSwapHelper {
                tenor_years: 3,
                basis_spread: -0.002, // -20 bps
                settlement_days: 2,
                day_counter: DayCounter::Actual365Fixed,
                domestic_df: (-0.04_f64 * 3.0).exp(),
                fx_spot: 1.10,
                foreign_freq: 4,
                domestic_freq: 4,
            },
        ];
        let res = bootstrap_xccy_curve(ref_date, &helpers, DayCounter::Actual365Fixed);
        // Negative spread → higher foreign DF (lower rate)
        assert!(res.foreign_dfs[0] > (-0.04_f64 * 3.0).exp());
    }

    #[test]
    fn test_tenor_basis() {
        let ref_date = Date::from_ymd(2025, Month::January, 15);
        let helpers = vec![
            TenorBasisSwapHelper {
                tenor_years: 2,
                basis_spread: 0.001, // 10 bps
                short_tenor_months: 3,
                long_tenor_months: 6,
                day_counter: DayCounter::Actual365Fixed,
                discount_factor: (-0.03_f64 * 2.0).exp(),
            },
            TenorBasisSwapHelper {
                tenor_years: 5,
                basis_spread: 0.0015,
                short_tenor_months: 3,
                long_tenor_months: 6,
                day_counter: DayCounter::Actual365Fixed,
                discount_factor: (-0.035_f64 * 5.0).exp(),
            },
        ];
        let short_rates = vec![0.03, 0.035];
        let res = bootstrap_tenor_basis_curve(ref_date, &helpers, &short_rates);
        assert_eq!(res.foreign_zero_rates.len(), 2);
        assert_abs_diff_eq!(res.foreign_zero_rates[0], 0.031, epsilon = 1e-10);
        assert_abs_diff_eq!(res.foreign_zero_rates[1], 0.0365, epsilon = 1e-10);
    }
}
