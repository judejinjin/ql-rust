//! Currency definitions following ISO 4217.
//!
//! Each currency carries its ISO alpha code, numeric code, name, and rounding
//! convention. Currencies are value types that can be compared by code.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Rounding
// ---------------------------------------------------------------------------

/// Rounding convention for monetary amounts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Rounding {
    /// No rounding.
    None,
    /// Round to nearest, half up.
    Closest(u32),
    /// Always round up (ceiling).
    Up(u32),
    /// Always round down (floor).
    Down(u32),
}

impl Rounding {
    /// Apply this rounding convention to a value.
    pub fn round(&self, value: f64) -> f64 {
        match *self {
            Rounding::None => value,
            Rounding::Closest(precision) => {
                let mult = 10f64.powi(precision as i32);
                (value * mult).round() / mult
            }
            Rounding::Up(precision) => {
                let mult = 10f64.powi(precision as i32);
                (value * mult).ceil() / mult
            }
            Rounding::Down(precision) => {
                let mult = 10f64.powi(precision as i32);
                (value * mult).floor() / mult
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Currency
// ---------------------------------------------------------------------------

/// A currency identified by its ISO 4217 code.
///
/// Currencies are immutable value types. Two currencies are equal if and only
/// if they share the same ISO alpha code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Currency {
    /// ISO 4217 three-letter code (e.g., "USD").
    pub code: &'static str,
    /// Full name (e.g., "US dollar").
    pub name: &'static str,
    /// ISO 4217 numeric code (e.g., 840 for USD).
    pub numeric_code: u16,
    /// Symbol (e.g., "$", "€").
    pub symbol: &'static str,
    /// Fractional unit name (e.g., "cent").
    pub fraction_symbol: &'static str,
    /// Number of fractional units per whole unit (e.g., 100 cents per dollar).
    pub fractions_per_unit: u32,
    /// Rounding convention.
    pub rounding: Rounding,
}

impl PartialEq for Currency {
    fn eq(&self, other: &Self) -> bool {
        self.code == other.code
    }
}

impl Eq for Currency {}

impl std::hash::Hash for Currency {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.code.hash(state);
    }
}

impl std::fmt::Display for Currency {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.code)
    }
}

// ---------------------------------------------------------------------------
// Concrete Currencies
// ---------------------------------------------------------------------------

impl Currency {
    /// US Dollar.
    pub fn usd() -> Self {
        Self {
            code: "USD",
            name: "US dollar",
            numeric_code: 840,
            symbol: "$",
            fraction_symbol: "¢",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Euro.
    pub fn eur() -> Self {
        Self {
            code: "EUR",
            name: "euro",
            numeric_code: 978,
            symbol: "€",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// British Pound Sterling.
    pub fn gbp() -> Self {
        Self {
            code: "GBP",
            name: "British pound sterling",
            numeric_code: 826,
            symbol: "£",
            fraction_symbol: "p",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Japanese Yen.
    pub fn jpy() -> Self {
        Self {
            code: "JPY",
            name: "Japanese yen",
            numeric_code: 392,
            symbol: "¥",
            fraction_symbol: "",
            fractions_per_unit: 1,
            rounding: Rounding::None,
        }
    }

    /// Swiss Franc.
    pub fn chf() -> Self {
        Self {
            code: "CHF",
            name: "Swiss franc",
            numeric_code: 756,
            symbol: "CHF",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Australian Dollar.
    pub fn aud() -> Self {
        Self {
            code: "AUD",
            name: "Australian dollar",
            numeric_code: 36,
            symbol: "A$",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Canadian Dollar.
    pub fn cad() -> Self {
        Self {
            code: "CAD",
            name: "Canadian dollar",
            numeric_code: 124,
            symbol: "C$",
            fraction_symbol: "",
            fractions_per_unit: 100,
            rounding: Rounding::Closest(2),
        }
    }

    /// Apply the currency's rounding convention to a monetary amount.
    pub fn round(&self, amount: f64) -> f64 {
        self.rounding.round(amount)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usd_properties() {
        let usd = Currency::usd();
        assert_eq!(usd.code, "USD");
        assert_eq!(usd.numeric_code, 840);
        assert_eq!(usd.fractions_per_unit, 100);
        assert_eq!(usd.to_string(), "USD");
    }

    #[test]
    fn eur_properties() {
        let eur = Currency::eur();
        assert_eq!(eur.code, "EUR");
        assert_eq!(eur.numeric_code, 978);
    }

    #[test]
    fn jpy_no_fractions() {
        let jpy = Currency::jpy();
        assert_eq!(jpy.fractions_per_unit, 1);
    }

    #[test]
    fn currency_equality() {
        assert_eq!(Currency::usd(), Currency::usd());
        assert_ne!(Currency::usd(), Currency::eur());
    }

    #[test]
    fn rounding_closest() {
        let r = Rounding::Closest(2);
        assert!((r.round(1.2345) - 1.23).abs() < 1e-15);
        assert!((r.round(1.235) - 1.24).abs() < 1e-15);
    }

    #[test]
    fn rounding_up() {
        let r = Rounding::Up(2);
        assert!((r.round(1.231) - 1.24).abs() < 1e-15);
    }

    #[test]
    fn rounding_down() {
        let r = Rounding::Down(2);
        assert!((r.round(1.239) - 1.23).abs() < 1e-15);
    }

    #[test]
    fn rounding_none() {
        let r = Rounding::None;
        let val = 1.23456789;
        assert!((r.round(val) - val).abs() < 1e-15);
    }

    #[test]
    fn currency_round_amount() {
        let usd = Currency::usd();
        assert!((usd.round(100.999) - 101.0).abs() < 1e-15);
        assert!((usd.round(100.001) - 100.0).abs() < 1e-15);
    }
}
