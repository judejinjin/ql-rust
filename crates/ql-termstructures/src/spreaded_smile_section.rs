//! Spreaded smile section: adds an additive or multiplicative spread to a base smile.
//!
//! Wraps any base smile quote (a vol vs. strike function) and shifts or scales
//! the implied volatility by a spread amount.
//!
//! # Spread Modes
//! - **Additive**: `σ_spreaded(K,T) = σ_base(K,T) + spread`
//! - **Multiplicative**: `σ_spreaded(K,T) = σ_base(K,T) × spread_factor`

// =========================================================================
// SpreadType
// =========================================================================

/// How the spread is applied to the base volatility.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SpreadType {
    /// σ_spreaded = σ_base + spread_value
    Additive,
    /// σ_spreaded = σ_base * spread_factor
    Multiplicative,
}

// =========================================================================
// FlatVolQuote — generic base smile quote type
// =========================================================================

/// A simple flat (constant-vol) smile quote usable as a base for spreading.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FlatVolQuote {
    /// Constant implied Black volatility.
    pub vol: f64,
}

impl FlatVolQuote {
    /// Create a flat vol quote.
    pub fn new(vol: f64) -> Self {
        Self { vol }
    }

    /// Implied vol at any strike.
    pub fn implied_vol(&self, _strike: f64) -> f64 {
        self.vol
    }

    /// Black call variance at `strike` and time `t`.
    pub fn black_variance(&self, _strike: f64, t: f64) -> f64 {
        self.vol * self.vol * t
    }
}

// =========================================================================
// ArbitrarySmileQuote — tabulated smile
// =========================================================================

/// A tabulated smile: strike → implied vol via linear interpolation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TabulatedSmileQuote {
    /// Strikes (sorted ascending).
    pub strikes: Vec<f64>,
    /// Corresponding implied vols.
    pub vols: Vec<f64>,
}

impl TabulatedSmileQuote {
    /// Create a tabulated smile. `strikes` must be sorted ascending and
    /// have the same length as `vols`.
    pub fn new(strikes: Vec<f64>, vols: Vec<f64>) -> Self {
        assert_eq!(strikes.len(), vols.len());
        Self { strikes, vols }
    }

    /// Implied vol at `strike` via linear interpolation (flat extrapolation).
    pub fn implied_vol(&self, strike: f64) -> f64 {
        let n = self.strikes.len();
        if strike <= self.strikes[0] {
            return self.vols[0];
        }
        if strike >= self.strikes[n - 1] {
            return self.vols[n - 1];
        }
        // Binary search for interval
        let pos = self.strikes.partition_point(|&x| x < strike);
        let i = pos.saturating_sub(1).min(n - 2);
        let t = (strike - self.strikes[i]) / (self.strikes[i + 1] - self.strikes[i]);
        self.vols[i] + t * (self.vols[i + 1] - self.vols[i])
    }

    /// Black variance at `strike` and time `t`.
    pub fn black_variance(&self, strike: f64, t: f64) -> f64 {
        let v = self.implied_vol(strike);
        v * v * t
    }
}

// =========================================================================
// SpreadedFlatSmileSection
// =========================================================================

/// A spreaded smile section built on top of a flat base vol.
///
/// Applies an additive or multiplicative spread to a constant base volatility.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpreadedFlatSmileSection {
    /// Base flat volatility.
    pub base_vol: f64,
    /// Spread value or factor (interpretation depends on `spread_type`).
    pub spread: f64,
    /// How to apply the spread.
    pub spread_type: SpreadType,
    /// Time to expiry in years.
    pub expiry: f64,
}

impl SpreadedFlatSmileSection {
    /// Create a spreaded flat smile section.
    pub fn new(base_vol: f64, spread: f64, spread_type: SpreadType, expiry: f64) -> Self {
        Self { base_vol, spread, spread_type, expiry }
    }

    /// Spreaded implied Black volatility (strike-independent for a flat base).
    pub fn implied_vol(&self, _strike: f64) -> f64 {
        match self.spread_type {
            SpreadType::Additive => (self.base_vol + self.spread).max(0.0),
            SpreadType::Multiplicative => (self.base_vol * self.spread).max(0.0),
        }
    }

    /// Spreaded Black variance at `strike`.
    pub fn black_variance(&self, strike: f64) -> f64 {
        let v = self.implied_vol(strike);
        v * v * self.expiry
    }
}

// =========================================================================
// SpreadedSmileSection (tabulated base)
// =========================================================================

/// A spreaded smile section built on top of a tabulated smile.
///
/// Applies an additive or multiplicative vol spread across the entire
/// strike dimension.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpreadedSmileSection {
    /// Base tabulated smile.
    pub base: TabulatedSmileQuote,
    /// Spread value or factor.
    pub spread: f64,
    /// How to apply the spread.
    pub spread_type: SpreadType,
    /// Time to expiry in years.
    pub expiry: f64,
}

impl SpreadedSmileSection {
    /// Create a spreaded smile section.
    pub fn new(
        strikes: Vec<f64>,
        base_vols: Vec<f64>,
        spread: f64,
        spread_type: SpreadType,
        expiry: f64,
    ) -> Self {
        Self {
            base: TabulatedSmileQuote::new(strikes, base_vols),
            spread,
            spread_type,
            expiry,
        }
    }

    /// Spreaded implied Black volatility at `strike`.
    pub fn implied_vol(&self, strike: f64) -> f64 {
        let base_v = self.base.implied_vol(strike);
        match self.spread_type {
            SpreadType::Additive => (base_v + self.spread).max(0.0),
            SpreadType::Multiplicative => (base_v * self.spread).max(0.0),
        }
    }

    /// Spreaded Black variance at `strike`.
    pub fn black_variance(&self, strike: f64) -> f64 {
        let v = self.implied_vol(strike);
        v * v * self.expiry
    }

    /// The strikes of the base smile grid.
    pub fn strikes(&self) -> &[f64] {
        &self.base.strikes
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn additive_spread_flat() {
        let s = SpreadedFlatSmileSection::new(0.20, 0.01, SpreadType::Additive, 1.0);
        assert!((s.implied_vol(100.0) - 0.21).abs() < 1e-12);
    }

    #[test]
    fn multiplicative_spread_flat() {
        let s = SpreadedFlatSmileSection::new(0.20, 1.1, SpreadType::Multiplicative, 1.0);
        assert!((s.implied_vol(100.0) - 0.22).abs() < 1e-12);
    }

    #[test]
    fn additive_spread_tabulated() {
        let strikes = vec![90.0, 100.0, 110.0];
        let vols = vec![0.22, 0.20, 0.21];
        let s = SpreadedSmileSection::new(strikes, vols, 0.02, SpreadType::Additive, 1.0);
        assert!((s.implied_vol(100.0) - 0.22).abs() < 1e-12);
        assert!((s.implied_vol(90.0) - 0.24).abs() < 1e-12);
    }

    #[test]
    fn multiplicative_spread_tabulated() {
        let strikes = vec![90.0, 100.0, 110.0];
        let vols = vec![0.20, 0.20, 0.20];
        let s = SpreadedSmileSection::new(strikes, vols, 0.9, SpreadType::Multiplicative, 1.0);
        for &k in &[90.0, 100.0, 110.0] {
            assert!((s.implied_vol(k) - 0.18).abs() < 1e-12, "k={}", k);
        }
    }

    #[test]
    fn interpolated_smile_spread() {
        let strikes = vec![90.0, 100.0, 110.0];
        let vols = vec![0.22, 0.20, 0.21];
        let s = SpreadedSmileSection::new(strikes, vols, 0.01, SpreadType::Additive, 1.0);
        // At k=95 (midpoint), base vol = 0.21, spreaded = 0.22
        let v = s.implied_vol(95.0);
        assert!((v - 0.22).abs() < 1e-12, "interp spreaded vol: {}", v);
    }
}
