//! CDS Index instrument (CDX / iTraxx).
//!
//! A CDS index is a standardized credit derivative referencing a diversified
//! portfolio of single-name CDS contracts. The buyer of protection pays a
//! fixed running coupon (100 or 500 bps for CDX IG/HY) and may pay or
//! receive an upfront amount.
//!
//! ## Key Features
//! - Standardised constituent pool (e.g. 125 names for CDX.NA.IG)
//! - Fixed coupon (100 or 500 bps)
//! - Upfront settlement (post-Big-Bang convention)
//! - Roll dates (March/September for CDX, March/September for iTraxx)
//! - Recovery rate 40% (IG) or 25-30% (HY)
//!
//! ## QuantLib Parity
//!
//! Corresponds to QuantLib's approach to pricing CDS indices as a
//! homogeneous basket with flat spread assumptions.

use crate::credit_default_swap::{CdsPremiumPeriod, CdsProtectionSide};
use ql_time::Date;
use serde::{Deserialize, Serialize};

/// CDS Index series identifier.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CdsIndexFamily {
    /// CDX.NA.IG (North America Investment Grade, 125 names).
    CdxNaIg,
    /// CDX.NA.HY (North America High Yield).
    CdxNaHy,
    /// iTraxx Europe Main (125 names).
    ITraxxEuropeMain,
    /// iTraxx Europe Crossover.
    ITraxxEuropeCrossover,
    /// iTraxx Asia ex-Japan IG.
    ITraxxAsiaIg,
    /// Custom / other index.
    Custom(String),
}

/// A CDS Index instrument.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdsIndex {
    /// Index family.
    pub family: CdsIndexFamily,
    /// Series number (e.g. 42 for CDX.NA.IG Series 42).
    pub series: u32,
    /// Version number (changes when a name defaults/is removed).
    pub version: u32,
    /// Protection side.
    pub side: CdsProtectionSide,
    /// Total notional.
    pub notional: f64,
    /// Number of constituents in the index.
    pub num_constituents: usize,
    /// Fixed running coupon (e.g. 0.01 for 100 bps).
    pub coupon: f64,
    /// Upfront fee (positive = protection buyer pays upfront).
    pub upfront: f64,
    /// Maturity date.
    pub maturity: Date,
    /// Recovery rate assumption.
    pub recovery_rate: f64,
    /// Premium payment schedule.
    pub schedule: Vec<CdsPremiumPeriod>,
}

impl CdsIndex {
    /// Create a new CDS index.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        family: CdsIndexFamily,
        series: u32,
        version: u32,
        side: CdsProtectionSide,
        notional: f64,
        num_constituents: usize,
        coupon: f64,
        upfront: f64,
        maturity: Date,
        recovery_rate: f64,
        schedule: Vec<CdsPremiumPeriod>,
    ) -> Self {
        Self {
            family,
            series,
            version,
            side,
            notional,
            num_constituents,
            coupon,
            upfront,
            maturity,
            recovery_rate,
            schedule,
        }
    }

    /// Per-name notional.
    pub fn per_name_notional(&self) -> f64 {
        if self.num_constituents > 0 {
            self.notional / self.num_constituents as f64
        } else {
            0.0
        }
    }

    /// Effective notional after `n_defaults` names have defaulted.
    pub fn outstanding_notional(&self, n_defaults: usize) -> f64 {
        let remaining = self.num_constituents.saturating_sub(n_defaults);
        self.per_name_notional() * remaining as f64
    }

    /// The implicit flat spread: coupon + upfront_annualized.
    /// This is a rough approximation assuming risky_pv01 ≈ T for short maturities.
    pub fn approximate_flat_spread(&self, time_to_maturity: f64) -> f64 {
        if time_to_maturity.abs() < 1e-15 {
            return self.coupon;
        }
        self.coupon + self.upfront / time_to_maturity
    }

    /// Whether the CDS index is a standard coupon (100 or 500 bps).
    pub fn is_standard_coupon(&self) -> bool {
        (self.coupon - 0.01).abs() < 1e-10 || (self.coupon - 0.05).abs() < 1e-10
    }

    /// Standard recovery rate for this family.
    pub fn standard_recovery(&self) -> f64 {
        match self.family {
            CdsIndexFamily::CdxNaIg | CdsIndexFamily::ITraxxEuropeMain | CdsIndexFamily::ITraxxAsiaIg => 0.40,
            CdsIndexFamily::CdxNaHy | CdsIndexFamily::ITraxxEuropeCrossover => 0.25,
            CdsIndexFamily::Custom(_) => self.recovery_rate,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ql_time::Month;

    fn make_cdx_schedule() -> Vec<CdsPremiumPeriod> {
        // 5Y quarterly schedule (20 periods)
        (0..20)
            .map(|i| {
                let year = 2025 + i / 4;
                let start_month = match i % 4 {
                    0 => Month::March,
                    1 => Month::June,
                    2 => Month::September,
                    _ => Month::December,
                };
                let end_month = match (i + 1) % 4 {
                    0 => Month::March,
                    1 => Month::June,
                    2 => Month::September,
                    _ => Month::December,
                };
                let end_year = if end_month == Month::March { year + 1 } else { year };
                CdsPremiumPeriod {
                    accrual_start: Date::from_ymd(year as i32, start_month, 20),
                    accrual_end: Date::from_ymd(end_year as i32, end_month, 20),
                    payment_date: Date::from_ymd(end_year as i32, end_month, 20),
                    accrual_fraction: 0.25,
                }
            })
            .collect()
    }

    #[test]
    fn cds_index_creation() {
        let idx = CdsIndex::new(
            CdsIndexFamily::CdxNaIg,
            42,
            1,
            CdsProtectionSide::Buyer,
            10_000_000.0,
            125,
            0.01, // 100 bps
            0.005,
            Date::from_ymd(2030, Month::March, 20),
            0.40,
            make_cdx_schedule(),
        );
        assert_eq!(idx.series, 42);
        assert_eq!(idx.num_constituents, 125);
        assert!(idx.is_standard_coupon());
        assert!((idx.per_name_notional() - 80_000.0).abs() < 1e-10);
    }

    #[test]
    fn cds_index_outstanding_notional() {
        let idx = CdsIndex::new(
            CdsIndexFamily::CdxNaIg,
            42, 1,
            CdsProtectionSide::Buyer,
            10_000_000.0,
            125,
            0.01, 0.0,
            Date::from_ymd(2030, Month::March, 20),
            0.40,
            make_cdx_schedule(),
        );
        // After 5 defaults, outstanding = 120/125 × 10M = 9,600,000
        let outstanding = idx.outstanding_notional(5);
        assert!((outstanding - 9_600_000.0).abs() < 1e-6);
    }

    #[test]
    fn cds_index_standard_recovery() {
        assert_eq!(
            CdsIndex::new(
                CdsIndexFamily::CdxNaIg, 42, 1,
                CdsProtectionSide::Buyer, 10e6, 125, 0.01, 0.0,
                Date::from_ymd(2030, Month::March, 20), 0.40,
                vec![],
            ).standard_recovery(),
            0.40
        );
        assert_eq!(
            CdsIndex::new(
                CdsIndexFamily::CdxNaHy, 42, 1,
                CdsProtectionSide::Buyer, 10e6, 100, 0.05, 0.0,
                Date::from_ymd(2030, Month::March, 20), 0.25,
                vec![],
            ).standard_recovery(),
            0.25
        );
    }

    #[test]
    fn cds_index_approximate_flat_spread() {
        let idx = CdsIndex::new(
            CdsIndexFamily::CdxNaIg,
            42, 1,
            CdsProtectionSide::Buyer,
            10_000_000.0,
            125,
            0.01,    // 100 bps coupon
            0.025,   // 2.5% upfront
            Date::from_ymd(2030, Month::March, 20),
            0.40,
            vec![],
        );
        let flat_spread = idx.approximate_flat_spread(5.0);
        // ≈ 0.01 + 0.025/5 = 0.01 + 0.005 = 0.015
        assert!((flat_spread - 0.015).abs() < 1e-10);
    }

    #[test]
    fn cds_index_hy_coupon() {
        let idx = CdsIndex::new(
            CdsIndexFamily::CdxNaHy,
            42, 1,
            CdsProtectionSide::Buyer,
            10_000_000.0,
            100,
            0.05,  // 500 bps
            0.0,
            Date::from_ymd(2030, Month::March, 20),
            0.25,
            vec![],
        );
        assert!(idx.is_standard_coupon());
    }
}
