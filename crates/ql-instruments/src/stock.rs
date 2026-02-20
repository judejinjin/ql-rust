//! Stock instrument.
//!
//! A stock is the simplest equity instrument, holding a current market price
//! (spot) and optional dividend yield for pricing purposes.

use serde::{Deserialize, Serialize};

/// A stock / equity share.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stock {
    /// Ticker or identifier.
    pub name: String,
    /// Current market price (spot).
    pub spot: f64,
    /// Continuous dividend yield (annualised).
    pub dividend_yield: f64,
}

impl Stock {
    /// Create a new stock.
    pub fn new(name: &str, spot: f64) -> Self {
        Self {
            name: name.to_string(),
            spot,
            dividend_yield: 0.0,
        }
    }

    /// Create a stock with a dividend yield.
    pub fn with_dividend(name: &str, spot: f64, dividend_yield: f64) -> Self {
        Self {
            name: name.to_string(),
            spot,
            dividend_yield,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stock_creation() {
        let s = Stock::new("AAPL", 175.0);
        assert_eq!(s.name, "AAPL");
        assert!((s.spot - 175.0).abs() < 1e-12);
        assert!((s.dividend_yield).abs() < 1e-12);
    }

    #[test]
    fn stock_with_dividend() {
        let s = Stock::with_dividend("SPY", 450.0, 0.013);
        assert!((s.dividend_yield - 0.013).abs() < 1e-12);
    }

    #[test]
    fn stock_serde_roundtrip() {
        let s = Stock::new("MSFT", 400.0);
        let json = serde_json::to_string(&s).unwrap();
        let s2: Stock = serde_json::from_str(&json).unwrap();
        assert_eq!(s.name, s2.name);
        assert!((s.spot - s2.spot).abs() < 1e-12);
    }
}
