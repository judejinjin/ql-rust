//! Vanilla swing option and storage option instruments.
//!
//! Swing (take-or-pay) options are common in energy markets, allowing the
//! buyer to exercise the option multiple times over a period.

use serde::{Deserialize, Serialize};

/// A vanilla swing option (multi-exercise).
///
/// The holder has the right to exercise up to `max_exercises` times
/// (minimum `min_exercises`) at predetermined dates, purchasing/selling
/// a unit of the underlying at the strike price.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VanillaSwingOption {
    /// Strike price per exercise.
    pub strike: f64,
    /// Minimum number of exercises (take-or-pay constraint).
    pub min_exercises: usize,
    /// Maximum number of exercises.
    pub max_exercises: usize,
    /// Exercise dates as year fractions from today.
    pub exercise_dates: Vec<f64>,
    /// Notional per exercise.
    pub notional_per_exercise: f64,
    /// True for call (right to buy), false for put.
    pub is_call: bool,
}

impl VanillaSwingOption {
    /// Create a new swing option with equally spaced exercise dates.
    pub fn new_uniform(
        strike: f64,
        min_exercises: usize,
        max_exercises: usize,
        start_time: f64,
        end_time: f64,
        n_dates: usize,
        notional_per_exercise: f64,
        is_call: bool,
    ) -> Self {
        let dt = (end_time - start_time) / n_dates as f64;
        let exercise_dates = (0..n_dates)
            .map(|i| start_time + (i + 1) as f64 * dt)
            .collect();
        Self {
            strike,
            min_exercises,
            max_exercises,
            exercise_dates,
            notional_per_exercise,
            is_call,
        }
    }

    /// Number of exercise dates.
    pub fn n_dates(&self) -> usize {
        self.exercise_dates.len()
    }

    /// Total maturity.
    pub fn maturity(&self) -> f64 {
        self.exercise_dates.last().copied().unwrap_or(0.0)
    }
}

/// A vanilla storage option (gas storage, battery storage).
///
/// The holder can inject or withdraw from storage over time, subject
/// to capacity constraints, and profit from price movements.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VanillaStorageOption {
    /// Maximum storage capacity.
    pub max_capacity: f64,
    /// Minimum storage level.
    pub min_capacity: f64,
    /// Maximum injection rate per period.
    pub max_injection_rate: f64,
    /// Maximum withdrawal rate per period.
    pub max_withdrawal_rate: f64,
    /// Cost per unit injected.
    pub injection_cost: f64,
    /// Cost per unit withdrawn.
    pub withdrawal_cost: f64,
    /// Initial storage level.
    pub initial_level: f64,
    /// Decision dates as year fractions.
    pub decision_dates: Vec<f64>,
    /// Terminal value function: $/unit of stored gas at expiry.
    pub terminal_value_per_unit: f64,
}

impl VanillaStorageOption {
    /// Create a new storage option with uniform decision dates.
    #[allow(clippy::too_many_arguments)]
    pub fn new_uniform(
        max_capacity: f64,
        min_capacity: f64,
        max_injection_rate: f64,
        max_withdrawal_rate: f64,
        injection_cost: f64,
        withdrawal_cost: f64,
        initial_level: f64,
        start_time: f64,
        end_time: f64,
        n_dates: usize,
        terminal_value_per_unit: f64,
    ) -> Self {
        let dt = (end_time - start_time) / n_dates as f64;
        let decision_dates = (0..n_dates)
            .map(|i| start_time + (i + 1) as f64 * dt)
            .collect();
        Self {
            max_capacity,
            min_capacity,
            max_injection_rate,
            max_withdrawal_rate,
            injection_cost,
            withdrawal_cost,
            initial_level,
            decision_dates,
            terminal_value_per_unit,
        }
    }

    /// Number of decision dates.
    pub fn n_dates(&self) -> usize {
        self.decision_dates.len()
    }

    /// Maturity.
    pub fn maturity(&self) -> f64 {
        self.decision_dates.last().copied().unwrap_or(0.0)
    }
}

/// Result from swing option pricing.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SwingOptionPricingResult {
    /// Price.
    pub price: f64,
    /// Delta.
    pub delta: f64,
    /// Expected exercises.
    pub expected_exercises: f64,
}

/// Result from storage option pricing.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StorageOptionPricingResult {
    /// Value.
    pub value: f64,
    /// Optimal initial action.
    pub optimal_initial_action: StorageAction,
    /// Expected final level.
    pub expected_final_level: f64,
}

/// Possible storage actions.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum StorageAction {
    /// Inject.
    Inject(f64),
    /// Withdraw.
    Withdraw(f64),
    /// Do Nothing.
    DoNothing,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swing_option_creation() {
        let opt = VanillaSwingOption::new_uniform(
            50.0, 5, 20, 0.0, 1.0, 12, 10_000.0, true,
        );
        assert_eq!(opt.n_dates(), 12);
        assert_eq!(opt.max_exercises, 20);
        assert!(opt.maturity() > 0.9);
    }

    #[test]
    fn test_storage_option_creation() {
        let store = VanillaStorageOption::new_uniform(
            100_000.0, 0.0,   // capacity: 0..100k
            5_000.0, 5_000.0, // injection/withdrawal rates
            0.50, 0.30,       // costs
            50_000.0,         // start half full
            0.0, 1.0, 252,    // daily decisions for 1 year
            0.0,              // no terminal value
        );
        assert_eq!(store.n_dates(), 252);
        assert!(store.maturity() > 0.9);
    }
}
