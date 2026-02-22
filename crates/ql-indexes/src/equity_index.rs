//! Equity index: treats an equity spot price as a rate index.
//!
//! An `EquityIndex` wraps an equity instrument (e.g. S&P 500, single stock)
//! so it can be used in equity-linked swap and derivative payoffs using the
//! same `Index` interface as interest-rate indexes.
//!
//! The "fixing" is the observed closing price on the fixing date.
//! The equity "rate" is the continuous dividend yield as implied by
//! futures prices (or user-provided), which is the cost-of-carry analog.

use std::collections::BTreeMap;

// =========================================================================
// EquityFixing
// =========================================================================

/// A historical or projected equity price fixing.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct EquityFixing {
    /// Observed (or projected) equity price.
    pub price: f64,
    /// Dividend yield as of this date (annualised, continuously compounded).
    pub dividend_yield: f64,
}

// =========================================================================
// EquityIndex
// =========================================================================

/// An equity price index usable in equity-linked swap payoffs.
///
/// Stores a series of price fixings (keyed by date serial number) and
/// provides forward-price projection for missing dates via a simple
/// cost-of-carry formula:
///
///   F(T) = S₀ · exp((r - q) · T)
///
/// where `r` is the risk-free rate and `q` is the dividend yield.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EquityIndex {
    /// Human-readable name (e.g. "S&P 500", "AAPL").
    pub name: String,
    /// Spot price at the index reference date.
    pub spot: f64,
    /// Annualised continuously-compounded risk-free rate.
    pub risk_free_rate: f64,
    /// Annualised continuously-compounded dividend yield (or repo rate).
    pub dividend_yield: f64,
    /// Historical / projected fixings: day_serial → EquityFixing.
    fixings: BTreeMap<u32, EquityFixing>,
}

impl EquityIndex {
    /// Create a new equity index.
    ///
    /// # Parameters
    /// - `name`: human-readable index name
    /// - `spot`: spot price today
    /// - `risk_free_rate`: continuously compounded risk-free rate
    /// - `dividend_yield`: continuously compounded dividend yield
    pub fn new(
        name: impl Into<String>,
        spot: f64,
        risk_free_rate: f64,
        dividend_yield: f64,
    ) -> Self {
        Self {
            name: name.into(),
            spot,
            risk_free_rate,
            dividend_yield,
            fixings: BTreeMap::new(),
        }
    }

    /// Register a price fixing at `date_serial` (integer date key).
    pub fn add_fixing(&mut self, date_serial: u32, price: f64) {
        self.fixings.insert(date_serial, EquityFixing { price, dividend_yield: self.dividend_yield });
    }

    /// Look up a fixing at `date_serial`. Returns `None` if not registered.
    pub fn fixing(&self, date_serial: u32) -> Option<f64> {
        self.fixings.get(&date_serial).map(|f| f.price)
    }

    /// Forward price at time `t` years from today via cost-of-carry.
    pub fn forward_price(&self, t: f64) -> f64 {
        self.spot * ((self.risk_free_rate - self.dividend_yield) * t).exp()
    }

    /// Projected price at `date_serial`, falling back to forward pricing
    /// from the last known fixing if not directly registered.
    pub fn projected_price(&self, date_serial: u32, t_years: f64) -> f64 {
        if let Some(p) = self.fixing(date_serial) {
            return p;
        }
        // Find the closest earlier fixing
        if let Some((&last_key, last_fix)) = self.fixings.range(..date_serial).next_back() {
            let elapsed = (date_serial - last_key) as f64 / 365.0;
            let remaining = (t_years - elapsed).max(0.0);
            return last_fix.price * ((self.risk_free_rate - last_fix.dividend_yield) * remaining).exp();
        }
        // No earlier fixing: use forward from spot
        self.forward_price(t_years)
    }

    /// Total return index value at time `t` (equity + reinvested dividends).
    pub fn total_return_index(&self, t: f64) -> f64 {
        self.spot * (self.risk_free_rate * t).exp()
    }

    /// Price return (ex-dividend) index value at time `t`.
    pub fn price_return_index(&self, t: f64) -> f64 {
        self.forward_price(t)
    }

    /// Name of the index.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Number of registered fixings.
    pub fn n_fixings(&self) -> usize {
        self.fixings.len()
    }

    /// All registered fixings as a sorted vector of (date_serial, price).
    pub fn all_fixings(&self) -> Vec<(u32, f64)> {
        self.fixings.iter().map(|(&k, v)| (k, v.price)).collect()
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_price_zero_yield() {
        let idx = EquityIndex::new("TEST", 100.0, 0.05, 0.0);
        let fwd = idx.forward_price(1.0);
        assert!((fwd - 100.0 * 0.05_f64.exp()).abs() < 1e-10, "fwd = {}", fwd);
    }

    #[test]
    fn forward_price_with_dividend() {
        let idx = EquityIndex::new("SPX", 4000.0, 0.05, 0.015);
        let fwd = idx.forward_price(0.5);
        let expected = 4000.0 * (0.035 * 0.5_f64).exp();
        assert!((fwd - expected).abs() < 1e-6, "fwd = {}, expected = {}", fwd, expected);
    }

    #[test]
    fn fixing_round_trip() {
        let mut idx = EquityIndex::new("AAPL", 180.0, 0.05, 0.005);
        idx.add_fixing(1000, 175.0);
        idx.add_fixing(1001, 180.0);
        assert_eq!(idx.fixing(1000), Some(175.0));
        assert_eq!(idx.fixing(1001), Some(180.0));
        assert_eq!(idx.fixing(999), None);
    }

    #[test]
    fn projected_price_fallback() {
        let mut idx = EquityIndex::new("TEST", 100.0, 0.04, 0.02);
        idx.add_fixing(0, 100.0);
        // 6 months later (serial 182, t ≈ 0.5)
        let p = idx.projected_price(182, 0.5);
        let remaining = (0.5_f64 - 182.0_f64 / 365.0_f64).max(0.0_f64);
        let expected = 100.0_f64 * (0.02_f64 * remaining).exp();
        // Should be reasonable (cost-of-carry extrapolation)
        assert!(p > 0.0 && (p - expected).abs() < 5.0, "projected = {}", p);
    }

    #[test]
    fn total_vs_price_return() {
        let idx = EquityIndex::new("DIV", 100.0, 0.05, 0.03);
        let tr = idx.total_return_index(1.0);
        let pr = idx.price_return_index(1.0);
        assert!(tr > pr, "total return should exceed price return when r > 0");
    }
}
