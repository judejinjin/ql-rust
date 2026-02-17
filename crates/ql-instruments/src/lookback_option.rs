//! Lookback option instruments.
//!
//! A lookback option has a payoff that depends on the minimum or maximum
//! asset price over the life of the option.
//!
//! - **Floating-strike lookback call**: payoff = S_T - S_min
//! - **Floating-strike lookback put**: payoff = S_max - S_T
//! - **Fixed-strike lookback call**: payoff = max(S_max - K, 0)
//! - **Fixed-strike lookback put**: payoff = max(K - S_min, 0)

use crate::payoff::OptionType;

/// The type of lookback option.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LookbackType {
    /// Floating strike: the strike is the realized min (call) or max (put).
    FloatingStrike,
    /// Fixed strike: payoff depends on the realized max (call) or min (put).
    FixedStrike,
}

/// A lookback option.
#[derive(Debug, Clone)]
pub struct LookbackOption {
    /// Call or put.
    pub option_type: OptionType,
    /// Floating or fixed strike.
    pub lookback_type: LookbackType,
    /// Strike price (used only for fixed-strike lookbacks).
    pub strike: f64,
    /// Current realized minimum of the underlying price.
    pub min_so_far: f64,
    /// Current realized maximum of the underlying price.
    pub max_so_far: f64,
    /// Time to expiry in years.
    pub time_to_expiry: f64,
}

impl LookbackOption {
    /// Create a new floating-strike lookback option.
    pub fn floating_strike(
        option_type: OptionType,
        min_so_far: f64,
        max_so_far: f64,
        time_to_expiry: f64,
    ) -> Self {
        Self {
            option_type,
            lookback_type: LookbackType::FloatingStrike,
            strike: 0.0,
            min_so_far,
            max_so_far,
            time_to_expiry,
        }
    }

    /// Create a new fixed-strike lookback option.
    pub fn fixed_strike(
        option_type: OptionType,
        strike: f64,
        min_so_far: f64,
        max_so_far: f64,
        time_to_expiry: f64,
    ) -> Self {
        Self {
            option_type,
            lookback_type: LookbackType::FixedStrike,
            strike,
            min_so_far,
            max_so_far,
            time_to_expiry,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floating_strike_creation() {
        let opt = LookbackOption::floating_strike(
            OptionType::Call, 90.0, 110.0, 1.0,
        );
        assert_eq!(opt.option_type, OptionType::Call);
        assert_eq!(opt.lookback_type, LookbackType::FloatingStrike);
        assert_eq!(opt.min_so_far, 90.0);
        assert_eq!(opt.max_so_far, 110.0);
    }

    #[test]
    fn fixed_strike_creation() {
        let opt = LookbackOption::fixed_strike(
            OptionType::Put, 100.0, 90.0, 110.0, 1.0,
        );
        assert_eq!(opt.option_type, OptionType::Put);
        assert_eq!(opt.lookback_type, LookbackType::FixedStrike);
        assert_eq!(opt.strike, 100.0);
    }
}
