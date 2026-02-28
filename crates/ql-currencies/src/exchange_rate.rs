//! Exchange rates between currencies.

use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

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

// ---------------------------------------------------------------------------
// ExchangeRateManager — global rate registry with triangulation
// ---------------------------------------------------------------------------

/// Type of exchange rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ExchangeRateType {
    /// Directly observed rate.
    Direct,
    /// Derived via triangulation through a common currency.
    Derived,
}

/// Global singleton store for exchange rates.
///
/// Rates are stored as `(source_code, target_code) → rate`. The manager can
/// look up a direct rate or automatically derive a cross rate via a common
/// intermediate currency (triangulation).
///
/// Follows the same singleton pattern as `IndexManager` in `ql-indexes`.
pub struct ExchangeRateManager {
    rates: RwLock<HashMap<(String, String), f64>>,
}

static EXCHANGE_RATE_MANAGER: OnceLock<ExchangeRateManager> = OnceLock::new();

impl ExchangeRateManager {
    /// Get the global `ExchangeRateManager` instance.
    pub fn instance() -> &'static ExchangeRateManager {
        EXCHANGE_RATE_MANAGER.get_or_init(|| ExchangeRateManager {
            rates: RwLock::new(HashMap::new()),
        })
    }

    /// Register an exchange rate (both directions are stored).
    pub fn add(&self, rate: &ExchangeRate) {
        let mut rates = self.rates.write().unwrap_or_else(|p| p.into_inner());
        rates.insert(
            (rate.source.code.to_string(), rate.target.code.to_string()),
            rate.rate,
        );
        rates.insert(
            (rate.target.code.to_string(), rate.source.code.to_string()),
            1.0 / rate.rate,
        );
    }

    /// Look up the exchange rate from `source` to `target`.
    ///
    /// First tries a direct lookup. If not found, attempts triangulation
    /// through any common intermediate currency.
    pub fn lookup(
        &self,
        source: &Currency,
        target: &Currency,
    ) -> QLResult<(f64, ExchangeRateType)> {
        if source.code == target.code {
            return Ok((1.0, ExchangeRateType::Direct));
        }

        let rates = self.rates.read().unwrap_or_else(|p| p.into_inner());

        // Direct lookup
        let key = (source.code.to_string(), target.code.to_string());
        if let Some(&r) = rates.get(&key) {
            return Ok((r, ExchangeRateType::Direct));
        }

        // Triangulation: find a currency C such that source→C and C→target exist
        // Collect all currencies reachable from source
        let source_str = source.code.to_string();
        let target_str = target.code.to_string();
        for ((from, to), &r1) in rates.iter() {
            if from == &source_str {
                // source → to exists with rate r1
                let cross_key = (to.clone(), target_str.clone());
                if let Some(&r2) = rates.get(&cross_key) {
                    return Ok((r1 * r2, ExchangeRateType::Derived));
                }
            }
        }

        Err(QLError::InvalidArgument(format!(
            "no exchange rate found for {}/{}",
            source.code, target.code,
        )))
    }

    /// Convert a `Money` amount from its currency to `target`.
    ///
    /// Uses `lookup` internally, supporting both direct and derived rates.
    pub fn convert(&self, money: &Money, target: &Currency) -> QLResult<Money> {
        let (rate, _) = self.lookup(&money.currency, target)?;
        Ok(Money::new(money.amount * rate, target.clone()))
    }

    /// Remove all stored exchange rates.
    pub fn clear(&self) {
        let mut rates = self.rates.write().unwrap_or_else(|p| p.into_inner());
        rates.clear();
    }

    /// Check whether a direct rate exists for the given currency pair.
    pub fn has_rate(&self, source: &Currency, target: &Currency) -> bool {
        let rates = self.rates.read().unwrap_or_else(|p| p.into_inner());
        rates.contains_key(&(source.code.to_string(), target.code.to_string()))
    }

    /// Return the number of stored rate entries.
    pub fn count(&self) -> usize {
        let rates = self.rates.read().unwrap_or_else(|p| p.into_inner());
        rates.len()
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

    // -----------------------------------------------------------------------
    // ExchangeRateManager tests
    // -----------------------------------------------------------------------

    #[test]
    fn manager_same_currency_returns_one() {
        let mgr = ExchangeRateManager::instance();
        let (rate, typ) = mgr.lookup(&Currency::usd(), &Currency::usd()).unwrap();
        assert_abs_diff_eq!(rate, 1.0, epsilon = 1e-15);
        assert_eq!(typ, ExchangeRateType::Direct);
    }

    #[test]
    fn manager_direct_lookup() {
        let mgr = ExchangeRateManager::instance();
        mgr.clear();
        let fx = ExchangeRate::new(Currency::eur(), Currency::usd(), 1.10).unwrap();
        mgr.add(&fx);
        let (rate, typ) = mgr.lookup(&Currency::eur(), &Currency::usd()).unwrap();
        assert_abs_diff_eq!(rate, 1.10, epsilon = 1e-12);
        assert_eq!(typ, ExchangeRateType::Direct);
    }

    #[test]
    fn manager_inverse_lookup() {
        let mgr = ExchangeRateManager::instance();
        mgr.clear();
        let fx = ExchangeRate::new(Currency::eur(), Currency::usd(), 1.10).unwrap();
        mgr.add(&fx);
        let (rate, _) = mgr.lookup(&Currency::usd(), &Currency::eur()).unwrap();
        assert_abs_diff_eq!(rate, 1.0 / 1.10, epsilon = 1e-12);
    }

    #[test]
    fn manager_triangulation() {
        let mgr = ExchangeRateManager::instance();
        mgr.clear();
        // EUR/USD = 1.10, GBP/USD = 1.30 → EUR/GBP = 1.10/1.30
        let fx1 = ExchangeRate::new(Currency::eur(), Currency::usd(), 1.10).unwrap();
        let fx2 = ExchangeRate::new(Currency::gbp(), Currency::usd(), 1.30).unwrap();
        mgr.add(&fx1);
        mgr.add(&fx2);
        let (rate, typ) = mgr.lookup(&Currency::eur(), &Currency::gbp()).unwrap();
        // EUR → USD → GBP: 1.10 * (1/1.30)
        let expected = 1.10 / 1.30;
        assert_abs_diff_eq!(rate, expected, epsilon = 1e-10);
        assert_eq!(typ, ExchangeRateType::Derived);
    }

    #[test]
    fn manager_missing_rate_fails() {
        let mgr = ExchangeRateManager::instance();
        mgr.clear();
        assert!(mgr.lookup(&Currency::inr(), &Currency::brl()).is_err());
    }

    #[test]
    fn manager_convert() {
        let mgr = ExchangeRateManager::instance();
        mgr.clear();
        let fx = ExchangeRate::new(Currency::gbp(), Currency::usd(), 1.30).unwrap();
        mgr.add(&fx);
        let gbp = Money::new(100.0, Currency::gbp());
        let usd = mgr.convert(&gbp, &Currency::usd()).unwrap();
        assert_eq!(usd.currency.code, "USD");
        assert_abs_diff_eq!(usd.amount, 130.0, epsilon = 1e-10);
    }

    #[test]
    fn manager_has_rate_and_count() {
        let mgr = ExchangeRateManager::instance();
        mgr.clear();
        assert!(!mgr.has_rate(&Currency::eur(), &Currency::jpy()));
        let fx = ExchangeRate::new(Currency::eur(), Currency::jpy(), 160.0).unwrap();
        mgr.add(&fx);
        assert!(mgr.has_rate(&Currency::eur(), &Currency::jpy()));
        assert!(mgr.has_rate(&Currency::jpy(), &Currency::eur()));
        assert_eq!(mgr.count(), 2); // both directions
    }
}
