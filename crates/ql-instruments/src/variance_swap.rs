//! Variance swap instrument.
//!
//! A variance swap is a derivative that pays the difference between
//! the realized variance of an underlying and a fixed variance strike.
//! The payoff at expiry is:
//!   Notional × (σ²_realized − K_var)
//!
//! where K_var is the variance strike.

use serde::{Deserialize, Serialize};

/// A variance swap.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarianceSwap {
    /// Notional in variance terms (vega notional / (2 × K_vol)).
    pub variance_notional: f64,
    /// Variance strike K_var (expressed as variance, not volatility).
    pub variance_strike: f64,
    /// Time to expiry in years.
    pub time_to_expiry: f64,
    /// Realized variance to date (annualized, for partially elapsed swaps).
    pub realized_variance: f64,
    /// Fraction of the total observation period that has elapsed (0 to 1).
    pub elapsed_fraction: f64,
}

impl VarianceSwap {
    /// Create a new variance swap.
    ///
    /// `vol_strike` is the volatility strike (not variance).
    /// The variance strike is vol_strike².
    pub fn from_vol_strike(
        variance_notional: f64,
        vol_strike: f64,
        time_to_expiry: f64,
    ) -> Self {
        Self {
            variance_notional,
            variance_strike: vol_strike * vol_strike,
            time_to_expiry,
            realized_variance: 0.0,
            elapsed_fraction: 0.0,
        }
    }

    /// Create a variance swap with some realized variance already observed.
    pub fn with_realized(
        variance_notional: f64,
        variance_strike: f64,
        time_to_expiry: f64,
        realized_variance: f64,
        elapsed_fraction: f64,
    ) -> Self {
        Self {
            variance_notional,
            variance_strike,
            time_to_expiry,
            realized_variance,
            elapsed_fraction,
        }
    }

    /// The volatility strike (square root of variance strike).
    pub fn vol_strike(&self) -> f64 {
        self.variance_strike.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variance_swap_creation() {
        let vs = VarianceSwap::from_vol_strike(100.0, 0.20, 1.0);
        assert_eq!(vs.variance_notional, 100.0);
        assert!((vs.variance_strike - 0.04).abs() < 1e-10);
        assert!((vs.vol_strike() - 0.20).abs() < 1e-10);
    }

    #[test]
    fn variance_swap_with_realized() {
        let vs = VarianceSwap::with_realized(100.0, 0.04, 0.5, 0.05, 0.5);
        assert_eq!(vs.elapsed_fraction, 0.5);
        assert_eq!(vs.realized_variance, 0.05);
        assert_eq!(vs.time_to_expiry, 0.5);
    }

    #[test]
    fn vol_strike_consistency() {
        let vs = VarianceSwap::from_vol_strike(100.0, 0.30, 1.0);
        assert!((vs.vol_strike() - 0.30).abs() < 1e-10);
        assert!((vs.variance_strike - 0.09).abs() < 1e-10);
    }
}
