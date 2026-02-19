//! Barrier option instruments.
//!
//! Defines barrier option types (up/down, in/out) and their parameters.

use serde::{Deserialize, Serialize};

use crate::payoff::{Exercise, Payoff};

#[cfg(test)]
use crate::payoff::OptionType;

/// Barrier type: whether the barrier is hit from above or below.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BarrierType {
    /// Barrier is above the current spot.
    UpIn,
    /// Barrier is above the current spot; option dies when hit.
    UpOut,
    /// Barrier is below the current spot.
    DownIn,
    /// Barrier is below the current spot; option dies when hit.
    DownOut,
}

impl BarrierType {
    /// True if this is a knock-in barrier.
    pub fn is_knock_in(self) -> bool {
        matches!(self, BarrierType::UpIn | BarrierType::DownIn)
    }

    /// True if the barrier is above the spot.
    pub fn is_up(self) -> bool {
        matches!(self, BarrierType::UpIn | BarrierType::UpOut)
    }
}

/// A barrier option.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarrierOption {
    /// The barrier type.
    pub barrier_type: BarrierType,
    /// The barrier level.
    pub barrier: f64,
    /// Rebate paid when a knock-out barrier is hit (or at expiry for knock-in not hit).
    pub rebate: f64,
    /// The underlying payoff.
    pub payoff: Payoff,
    /// The exercise style.
    pub exercise: Exercise,
}

impl BarrierOption {
    /// Create a new barrier option.
    pub fn new(
        barrier_type: BarrierType,
        barrier: f64,
        rebate: f64,
        payoff: Payoff,
        exercise: Exercise,
    ) -> Self {
        Self {
            barrier_type,
            barrier,
            rebate,
            payoff,
            exercise,
        }
    }

    /// Check if the option has expired given a reference date.
    pub fn is_expired(&self, ref_date: ql_time::Date) -> bool {
        self.exercise.last_date() < ref_date
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ql_time::{Date, Month};

    #[test]
    fn barrier_option_creation() {
        let opt = BarrierOption::new(
            BarrierType::DownOut,
            90.0,
            0.0,
            Payoff::PlainVanilla {
                option_type: OptionType::Call,
                strike: 100.0,
            },
            Exercise::European {
                expiry: Date::from_ymd(2026, Month::January, 15),
            },
        );
        assert_eq!(opt.barrier_type, BarrierType::DownOut);
        assert!(!opt.barrier_type.is_knock_in());
        assert!(!opt.barrier_type.is_up());
    }

    #[test]
    fn barrier_type_properties() {
        assert!(BarrierType::UpIn.is_knock_in());
        assert!(BarrierType::UpIn.is_up());
        assert!(!BarrierType::DownOut.is_knock_in());
        assert!(!BarrierType::DownOut.is_up());
        assert!(BarrierType::DownIn.is_knock_in());
        assert!(BarrierType::UpOut.is_up());
    }
}
