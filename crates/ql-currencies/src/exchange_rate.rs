//! Exchange rates between currencies.

use ql_core::errors::{QLError, QLResult};

use crate::currency::Currency;
use crate::money::Money;

/// An exchange rate between two currencies.
///
/// Represents `1 source = rate × target`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ExchangeRate {
    /// The source (base) currency.
    pub source: Currency,
    /// The target (quote) currency.
    pub target: Currency,
    /// The exchange rate value.
    pub rate: f64,
}

impl ExchangeRate {
    /// Create a new exchange rate: 1 `source` = `rate` `target`.
    pub fn new(source: Currency, target: Currency, rate: f64) -> QLResult<Self> {
        if rate <= 0.0 {
            return Err(QLError::NegativeValue {
                quantity: "exchange rate",
                value: rate,
            });
        }
        Ok(Self {
            source,
            target,
            rate,
        })
    }

    /// Convert a `Money` amount from source to target currency.
    pub fn exchange(&self, money: &Money) -> QLResult<Money> {
        if money.currency == self.source {
            Ok(Money::new(money.amount * self.rate, self.target.clone()))
        } else if money.currency == self.target {
            Ok(Money::new(money.amount / self.rate, self.source.clone()))
        } else {
            Err(QLError::InvalidArgument(format!(
                "cannot exchange {} using {}/{}",
                money.currency, self.source, self.target
            )))
        }
    }

    /// Return the inverse rate (target → source).
    pub fn inverse(&self) -> Self {
        Self {
            source: self.target.clone(),
            target: self.source.clone(),
            rate: 1.0 / self.rate,
        }
    }
}

impl std::fmt::Display for ExchangeRate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{} = {:.6}", self.source, self.target, self.rate)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn exchange_rate_creation() {
        let fx = ExchangeRate::new(Currency::eur(), Currency::usd(), 1.10).unwrap();
        assert_eq!(fx.source.code, "EUR");
        assert_eq!(fx.target.code, "USD");
        assert_abs_diff_eq!(fx.rate, 1.10, epsilon = 1e-15);
    }

    #[test]
    fn exchange_rate_negative_rejected() {
        assert!(ExchangeRate::new(Currency::eur(), Currency::usd(), -1.0).is_err());
    }

    #[test]
    fn exchange_source_to_target() {
        let fx = ExchangeRate::new(Currency::eur(), Currency::usd(), 1.10).unwrap();
        let eur = Money::new(100.0, Currency::eur());
        let usd = fx.exchange(&eur).unwrap();
        assert_eq!(usd.currency.code, "USD");
        assert_abs_diff_eq!(usd.amount, 110.0, epsilon = 1e-10);
    }

    #[test]
    fn exchange_target_to_source() {
        let fx = ExchangeRate::new(Currency::eur(), Currency::usd(), 1.10).unwrap();
        let usd = Money::new(110.0, Currency::usd());
        let eur = fx.exchange(&usd).unwrap();
        assert_eq!(eur.currency.code, "EUR");
        assert_abs_diff_eq!(eur.amount, 100.0, epsilon = 1e-10);
    }

    #[test]
    fn exchange_wrong_currency_fails() {
        let fx = ExchangeRate::new(Currency::eur(), Currency::usd(), 1.10).unwrap();
        let gbp = Money::new(100.0, Currency::gbp());
        assert!(fx.exchange(&gbp).is_err());
    }

    #[test]
    fn inverse_rate() {
        let fx = ExchangeRate::new(Currency::eur(), Currency::usd(), 1.10).unwrap();
        let inv = fx.inverse();
        assert_eq!(inv.source.code, "USD");
        assert_eq!(inv.target.code, "EUR");
        assert_abs_diff_eq!(inv.rate, 1.0 / 1.10, epsilon = 1e-12);
    }

    #[test]
    fn display_format() {
        let fx = ExchangeRate::new(Currency::gbp(), Currency::usd(), 1.27).unwrap();
        assert_eq!(fx.to_string(), "GBP/USD = 1.270000");
    }
}
