//! Seasonality adjustments for inflation term structures.
//!
//! Inflation indexes (CPI, RPI, HICP) exhibit seasonal patterns due to
//! weather, holidays, fashion cycles, etc. This module provides multiplicative
//! and additive seasonality correction factors that can be applied to zero
//! inflation curves and year-on-year inflation curves.
//!
//! Reference:
//! - QuantLib: Seasonality, MultiplicativePriceSeasonality in seasonality.hpp

use serde::{Deserialize, Serialize};

/// Type of seasonality adjustment.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum SeasonalityType {
    /// Multiplicative: adjusted_rate = raw_rate × factor(month)
    Multiplicative,
    /// Additive: adjusted_rate = raw_rate + factor(month)
    Additive,
}

/// Monthly seasonality factors.
///
/// Factors are indexed 0..12 where index 0 = January, 11 = December.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Seasonality {
    /// Type of adjustment.
    pub seasonality_type: SeasonalityType,
    /// Monthly factors (length 12). For Multiplicative, should average ~1.0.
    /// For Additive, should average ~0.0.
    pub monthly_factors: [f64; 12],
    /// Optional frequency (e.g. 1 = monthly, 3 = quarterly).
    pub frequency: u32,
}

impl Seasonality {
    /// Create a new multiplicative seasonality from 12 monthly factors.
    ///
    /// Factors are normalised so their geometric mean is 1.0.
    pub fn multiplicative(factors: [f64; 12]) -> Self {
        let geo_mean: f64 = factors.iter().map(|f| f.ln()).sum::<f64>() / 12.0;
        let geo_mean = geo_mean.exp();
        let mut normalised = [0.0; 12];
        for i in 0..12 {
            normalised[i] = factors[i] / geo_mean;
        }
        Seasonality {
            seasonality_type: SeasonalityType::Multiplicative,
            monthly_factors: normalised,
            frequency: 1,
        }
    }

    /// Create a new additive seasonality from 12 monthly factors.
    ///
    /// Factors are centred so their arithmetic mean is 0.0.
    pub fn additive(factors: [f64; 12]) -> Self {
        let mean: f64 = factors.iter().sum::<f64>() / 12.0;
        let mut centred = [0.0; 12];
        for i in 0..12 {
            centred[i] = factors[i] - mean;
        }
        Seasonality {
            seasonality_type: SeasonalityType::Additive,
            monthly_factors: centred,
            frequency: 1,
        }
    }

    /// Get the seasonality factor for a given month (1-based: Jan=1, Dec=12).
    pub fn factor(&self, month: u32) -> f64 {
        let idx = ((month.clamp(1, 12)) - 1) as usize;
        self.monthly_factors[idx]
    }

    /// Apply seasonality adjustment to a rate.
    pub fn adjust(&self, rate: f64, month: u32) -> f64 {
        let f = self.factor(month);
        match self.seasonality_type {
            SeasonalityType::Multiplicative => rate * f,
            SeasonalityType::Additive => rate + f,
        }
    }

    /// Remove seasonality adjustment from an adjusted rate to get the raw rate.
    pub fn unadjust(&self, adjusted_rate: f64, month: u32) -> f64 {
        let f = self.factor(month);
        match self.seasonality_type {
            SeasonalityType::Multiplicative => {
                if f.abs() > 1e-14 { adjusted_rate / f } else { adjusted_rate }
            }
            SeasonalityType::Additive => adjusted_rate - f,
        }
    }

    /// Compute the seasonality correction for a zero inflation rate
    /// between two months.
    ///
    /// For multiplicative seasonality over a period from month `m1` to `m2`
    /// (going forward), the correction factor = product of monthly factors
    /// over the period, normalised by the period length.
    pub fn correction_factor(&self, start_month: u32, end_month: u32) -> f64 {
        let s = start_month.clamp(1, 12);
        let e = end_month.clamp(1, 12);

        match self.seasonality_type {
            SeasonalityType::Multiplicative => {
                let mut product = 1.0;
                let mut m = s;
                loop {
                    product *= self.monthly_factors[(m - 1) as usize];
                    if m == e { break; }
                    m = if m == 12 { 1 } else { m + 1 };
                }
                product
            }
            SeasonalityType::Additive => {
                let mut sum = 0.0;
                let mut count = 0;
                let mut m = s;
                loop {
                    sum += self.monthly_factors[(m - 1) as usize];
                    count += 1;
                    if m == e { break; }
                    m = if m == 12 { 1 } else { m + 1 };
                }
                sum / count as f64
            }
        }
    }
}

/// Estimate seasonality factors from historical monthly inflation data.
///
/// Takes a vector of (year, month, yoy_rate) observations and computes
/// the average seasonal pattern.
///
/// # Arguments
/// - `observations` — vector of (year, month [1-12], year-on-year rate)
/// - `use_multiplicative` — if true, returns multiplicative factors
///
/// # Returns
/// A `Seasonality` with estimated monthly factors.
pub fn estimate_seasonality(
    observations: &[(u32, u32, f64)],
    use_multiplicative: bool,
) -> Seasonality {
    let mut sums = [0.0_f64; 12];
    let mut counts = [0_u32; 12];

    for &(_year, month, rate) in observations {
        if (1..=12).contains(&month) {
            let idx = (month - 1) as usize;
            sums[idx] += rate;
            counts[idx] += 1;
        }
    }

    let mut avg = [0.0_f64; 12];
    let grand_mean: f64 = {
        let total: f64 = sums.iter().sum();
        let total_count: u32 = counts.iter().sum();
        if total_count > 0 { total / total_count as f64 } else { 0.0 }
    };

    for i in 0..12 {
        if counts[i] > 0 {
            avg[i] = sums[i] / counts[i] as f64;
        } else {
            avg[i] = grand_mean;
        }
    }

    if use_multiplicative {
        // Convert to multiplicative factors relative to grand mean
        let mut factors = [1.0_f64; 12];
        if grand_mean.abs() > 1e-10 {
            for i in 0..12 {
                factors[i] = avg[i] / grand_mean;
            }
        }
        Seasonality::multiplicative(factors)
    } else {
        let mut factors = [0.0_f64; 12];
        for i in 0..12 {
            factors[i] = avg[i] - grand_mean;
        }
        Seasonality::additive(factors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_multiplicative_normalisation() {
        let factors = [1.02, 0.98, 1.01, 1.0, 0.99, 1.03, 1.01, 0.97, 1.0, 1.02, 0.98, 0.99];
        let s = Seasonality::multiplicative(factors);
        let geo_mean: f64 = s.monthly_factors.iter().map(|f| f.ln()).sum::<f64>() / 12.0;
        assert_abs_diff_eq!(geo_mean.exp(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_additive_centring() {
        let factors = [0.02, -0.01, 0.01, 0.0, -0.02, 0.03, 0.01, -0.03, 0.0, 0.01, -0.01, -0.01];
        let s = Seasonality::additive(factors);
        let sum: f64 = s.monthly_factors.iter().sum();
        assert_abs_diff_eq!(sum, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_adjust_unadjust_roundtrip() {
        let factors = [1.02, 0.98, 1.01, 1.0, 0.99, 1.03, 1.01, 0.97, 1.0, 1.02, 0.98, 0.99];
        let s = Seasonality::multiplicative(factors);
        let rate = 0.025;
        for month in 1..=12 {
            let adjusted = s.adjust(rate, month);
            let back = s.unadjust(adjusted, month);
            assert_abs_diff_eq!(back, rate, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_estimate_seasonality() {
        let mut obs = Vec::new();
        let base = 0.02;
        let pattern = [0.01, -0.005, 0.005, 0.0, -0.01, 0.015, 0.01, -0.015, 0.0, 0.005, -0.005, -0.01];
        for year in 2015..=2023 {
            for month in 1..=12_u32 {
                let rate = base + pattern[(month - 1) as usize];
                obs.push((year, month, rate));
            }
        }
        let s = estimate_seasonality(&obs, false);
        // Should recover the pattern approximately
        assert!(s.monthly_factors[0] > 0.0, "Jan should be positive");
        assert!(s.monthly_factors[4] < 0.0, "May should be negative");
    }

    #[test]
    fn test_correction_factor() {
        let factors = [1.0; 12];
        let s = Seasonality::multiplicative(factors);
        let cf = s.correction_factor(3, 6);
        assert_abs_diff_eq!(cf, 1.0, epsilon = 1e-10);
    }
}
