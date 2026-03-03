//! FX Forward instrument.
//!
//! An FX forward is an agreement to exchange a specified amount of one
//! currency for another at a future date at a pre-determined exchange rate.
//!
//! ## QuantLib Parity
//!
//! Corresponds to `FxForward` in QuantLib (ql/instruments/fxforward.hpp).

use ql_currencies::currency::Currency;
use ql_time::Date;

/// FX Forward direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FxForwardType {
    /// Buy base currency, sell quote currency.
    Buy,
    /// Sell base currency, buy quote currency.
    Sell,
}

/// An FX Forward instrument.
///
/// The holder agrees to exchange `base_notional` units of `base_currency`
/// for `quote_notional` units of `quote_currency` at `maturity_date`.
///
/// The forward rate is implicitly: `quote_notional / base_notional`.
#[derive(Debug, Clone)]
pub struct FxForward {
    /// Direction: buy or sell the base currency.
    pub forward_type: FxForwardType,
    /// Base (domestic) currency.
    pub base_currency: Currency,
    /// Quote (foreign) currency.
    pub quote_currency: Currency,
    /// Notional in base currency.
    pub base_notional: f64,
    /// Notional in quote currency.
    pub quote_notional: f64,
    /// Maturity / delivery date.
    pub maturity_date: Date,
    /// Optional settlement days (default: 2).
    pub settlement_days: u32,
}

impl FxForward {
    /// Create a new FX forward.
    pub fn new(
        forward_type: FxForwardType,
        base_currency: Currency,
        quote_currency: Currency,
        base_notional: f64,
        quote_notional: f64,
        maturity_date: Date,
    ) -> Self {
        Self {
            forward_type,
            base_currency,
            quote_currency,
            base_notional,
            quote_notional,
            maturity_date,
            settlement_days: 2,
        }
    }

    /// Create from a forward rate: `fwd_rate` = quote per base.
    pub fn from_rate(
        forward_type: FxForwardType,
        base_currency: Currency,
        quote_currency: Currency,
        base_notional: f64,
        forward_rate: f64,
        maturity_date: Date,
    ) -> Self {
        Self {
            forward_type,
            base_currency,
            quote_currency,
            base_notional,
            quote_notional: base_notional * forward_rate,
            maturity_date,
            settlement_days: 2,
        }
    }

    /// Implied forward exchange rate (quote per base).
    pub fn forward_rate(&self) -> f64 {
        if self.base_notional.abs() > 1e-15 {
            self.quote_notional / self.base_notional
        } else {
            0.0
        }
    }

    /// Whether the forward has matured.
    pub fn is_expired(&self, ref_date: Date) -> bool {
        self.maturity_date < ref_date
    }

    /// Sign: +1 for buy, -1 for sell.
    pub fn sign(&self) -> f64 {
        match self.forward_type {
            FxForwardType::Buy => 1.0,
            FxForwardType::Sell => -1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ql_time::Month;

    #[test]
    fn fx_forward_creation() {
        let fwd = FxForward::new(
            FxForwardType::Buy,
            Currency::usd(),
            Currency::eur(),
            1_000_000.0,
            900_000.0,
            Date::from_ymd(2025, Month::July, 15),
        );
        assert_eq!(fwd.forward_type, FxForwardType::Buy);
        assert_eq!(fwd.base_notional, 1_000_000.0);
        assert_eq!(fwd.quote_notional, 900_000.0);
        assert!((fwd.forward_rate() - 0.9).abs() < 1e-10);
    }

    #[test]
    fn fx_forward_from_rate() {
        let fwd = FxForward::from_rate(
            FxForwardType::Sell,
            Currency::eur(),
            Currency::usd(),
            500_000.0,
            1.1111,
            Date::from_ymd(2025, Month::December, 15),
        );
        assert_eq!(fwd.forward_type, FxForwardType::Sell);
        assert!((fwd.quote_notional - 555_550.0).abs() < 1.0);
        assert!((fwd.forward_rate() - 1.1111).abs() < 1e-10);
    }

    #[test]
    fn fx_forward_expired() {
        let fwd = FxForward::new(
            FxForwardType::Buy,
            Currency::usd(),
            Currency::eur(),
            1_000_000.0,
            900_000.0,
            Date::from_ymd(2025, Month::July, 15),
        );
        assert!(!fwd.is_expired(Date::from_ymd(2025, Month::January, 1)));
        assert!(fwd.is_expired(Date::from_ymd(2025, Month::August, 1)));
    }

    #[test]
    fn fx_forward_sign() {
        assert_eq!(FxForwardType::Buy.sign(), 1.0);
        assert_eq!(FxForwardType::Sell.sign(), -1.0);
    }
}

impl FxForwardType {
    /// +1 for buy, -1 for sell.
    pub fn sign(&self) -> f64 {
        match self {
            Self::Buy => 1.0,
            Self::Sell => -1.0,
        }
    }
}
