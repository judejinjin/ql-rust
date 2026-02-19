//! Money — an amount associated with a currency.

use std::fmt;

use ql_core::errors::{QLError, QLResult};

use crate::currency::Currency;

/// An amount of money in a specific currency.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Money {
    /// The monetary amount.
    pub amount: f64,
    /// The currency.
    pub currency: Currency,
}

impl Money {
    /// Create a new `Money` value.
    pub fn new(amount: f64, currency: Currency) -> Self {
        Self { amount, currency }
    }

    /// Round to the currency's standard precision.
    pub fn rounded(&self) -> Self {
        Self {
            amount: self.currency.round(self.amount),
            currency: self.currency.clone(),
        }
    }

    /// Add two Money values. They must share the same currency.
    pub fn add(&self, other: &Money) -> QLResult<Money> {
        if self.currency != other.currency {
            return Err(QLError::InvalidArgument(format!(
                "cannot add {} and {}",
                self.currency, other.currency
            )));
        }
        Ok(Money::new(self.amount + other.amount, self.currency.clone()))
    }

    /// Subtract another Money value. They must share the same currency.
    pub fn sub(&self, other: &Money) -> QLResult<Money> {
        if self.currency != other.currency {
            return Err(QLError::InvalidArgument(format!(
                "cannot subtract {} and {}",
                self.currency, other.currency
            )));
        }
        Ok(Money::new(self.amount - other.amount, self.currency.clone()))
    }

    /// Multiply by a scalar.
    pub fn scale(&self, factor: f64) -> Money {
        Money::new(self.amount * factor, self.currency.clone())
    }
}

impl fmt::Display for Money {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2} {}", self.amount, self.currency)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn money_creation_and_display() {
        let m = Money::new(100.50, Currency::usd());
        assert_eq!(m.to_string(), "100.50 USD");
    }

    #[test]
    fn money_add_same_currency() {
        let a = Money::new(100.0, Currency::usd());
        let b = Money::new(50.25, Currency::usd());
        let c = a.add(&b).unwrap();
        assert!((c.amount - 150.25).abs() < 1e-15);
    }

    #[test]
    fn money_add_different_currency_fails() {
        let a = Money::new(100.0, Currency::usd());
        let b = Money::new(50.0, Currency::eur());
        assert!(a.add(&b).is_err());
    }

    #[test]
    fn money_subtract() {
        let a = Money::new(100.0, Currency::eur());
        let b = Money::new(30.0, Currency::eur());
        let c = a.sub(&b).unwrap();
        assert!((c.amount - 70.0).abs() < 1e-15);
    }

    #[test]
    fn money_scale() {
        let m = Money::new(100.0, Currency::gbp());
        let scaled = m.scale(1.5);
        assert!((scaled.amount - 150.0).abs() < 1e-15);
    }

    #[test]
    fn money_rounded() {
        let m = Money::new(100.999, Currency::usd());
        let r = m.rounded();
        assert!((r.amount - 101.0).abs() < 1e-15);
    }
}
