//! Asian option instruments.
//!
//! Defines arithmetic and geometric average price options.

use serde::{Deserialize, Serialize};

use crate::payoff::{Exercise, OptionType};

/// Averaging type for Asian options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AveragingType {
    /// Arithmetic average: (1/n) Σ Sᵢ.
    Arithmetic,
    /// Geometric average: (∏ Sᵢ)^(1/n).
    Geometric,
}

/// An Asian (average-price) option.
///
/// The payoff is max(omega * (A − K), 0) where A is the average
/// of the underlying price over the monitoring dates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsianOption {
    /// Call or put.
    pub option_type: OptionType,
    /// Strike price.
    pub strike: f64,
    /// Averaging type.
    pub averaging_type: AveragingType,
    /// Exercise style (typically European).
    pub exercise: Exercise,
}

impl AsianOption {
    /// Create a new Asian option.
    pub fn new(
        option_type: OptionType,
        strike: f64,
        averaging_type: AveragingType,
        exercise: Exercise,
    ) -> Self {
        Self {
            option_type,
            strike,
            averaging_type,
            exercise,
        }
    }

    /// Evaluate the payoff given the average price.
    pub fn payoff(&self, average: f64) -> f64 {
        (self.option_type.sign() * (average - self.strike)).max(0.0)
    }

    /// Check if the option has expired.
    pub fn is_expired(&self, ref_date: ql_time::Date) -> bool {
        self.exercise.last_date() < ref_date
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_time::{Date, Month};

    #[test]
    fn asian_call_payoff() {
        let opt = AsianOption::new(
            OptionType::Call,
            100.0,
            AveragingType::Arithmetic,
            Exercise::European {
                expiry: Date::from_ymd(2026, Month::January, 15),
            },
        );
        assert_abs_diff_eq!(opt.payoff(110.0), 10.0, epsilon = 1e-15);
        assert_abs_diff_eq!(opt.payoff(90.0), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn asian_put_payoff() {
        let opt = AsianOption::new(
            OptionType::Put,
            100.0,
            AveragingType::Geometric,
            Exercise::European {
                expiry: Date::from_ymd(2026, Month::January, 15),
            },
        );
        assert_abs_diff_eq!(opt.payoff(90.0), 10.0, epsilon = 1e-15);
        assert_abs_diff_eq!(opt.payoff(110.0), 0.0, epsilon = 1e-15);
    }
}
