//! Vanilla European/American option.
//!
//! A vanilla option is defined by a payoff (strike + call/put) and an
//! exercise style (European/American/Bermudan).

use crate::payoff::{Exercise, OptionType, Payoff};
use ql_time::Date;

/// A vanilla option (European, American, or Bermudan).
#[derive(Debug, Clone)]
pub struct VanillaOption {
    /// The payoff specification.
    pub payoff: Payoff,
    /// The exercise style.
    pub exercise: Exercise,
}

impl VanillaOption {
    /// Create a new vanilla option.
    pub fn new(payoff: Payoff, exercise: Exercise) -> Self {
        Self { payoff, exercise }
    }

    /// Convenience: create a European call.
    pub fn european_call(strike: f64, expiry: Date) -> Self {
        Self {
            payoff: Payoff::PlainVanilla {
                option_type: OptionType::Call,
                strike,
            },
            exercise: Exercise::European { expiry },
        }
    }

    /// Convenience: create a European put.
    pub fn european_put(strike: f64, expiry: Date) -> Self {
        Self {
            payoff: Payoff::PlainVanilla {
                option_type: OptionType::Put,
                strike,
            },
            exercise: Exercise::European { expiry },
        }
    }

    /// Whether the option has expired.
    pub fn is_expired(&self, ref_date: Date) -> bool {
        self.exercise.last_date() < ref_date
    }

    /// The strike price (convenience).
    pub fn strike(&self) -> f64 {
        self.payoff.strike()
    }

    /// The option type (convenience).
    pub fn option_type(&self) -> OptionType {
        self.payoff.option_type()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ql_time::Month;

    #[test]
    fn european_call_creation() {
        let opt = VanillaOption::european_call(100.0, Date::from_ymd(2025, Month::December, 15));
        assert_eq!(opt.option_type(), OptionType::Call);
        assert!((opt.strike() - 100.0).abs() < 1e-15);
    }

    #[test]
    fn european_put_creation() {
        let opt = VanillaOption::european_put(95.0, Date::from_ymd(2025, Month::June, 15));
        assert_eq!(opt.option_type(), OptionType::Put);
        assert!((opt.strike() - 95.0).abs() < 1e-15);
    }

    #[test]
    fn is_expired() {
        let opt = VanillaOption::european_call(100.0, Date::from_ymd(2025, Month::June, 15));
        assert!(!opt.is_expired(Date::from_ymd(2025, Month::January, 1)));
        assert!(opt.is_expired(Date::from_ymd(2025, Month::December, 31)));
    }
}
