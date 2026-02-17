//! Compound option instruments (option on option).
//!
//! A compound option is an option whose underlying is itself an option.
//! Types: call-on-call, call-on-put, put-on-call, put-on-put.

use crate::payoff::OptionType;

/// A compound option (option on an option).
#[derive(Debug, Clone)]
pub struct CompoundOption {
    /// Type of the outer (mother) option.
    pub mother_type: OptionType,
    /// Type of the inner (daughter) option.
    pub daughter_type: OptionType,
    /// Strike of the mother option (paid to exercise into the daughter).
    pub mother_strike: f64,
    /// Time to expiry of the mother option.
    pub mother_expiry: f64,
    /// Strike of the daughter option.
    pub daughter_strike: f64,
    /// Time to expiry of the daughter option (must be > mother_expiry).
    pub daughter_expiry: f64,
}

impl CompoundOption {
    /// Create a new compound option.
    ///
    /// # Panics
    /// Panics if `daughter_expiry <= mother_expiry`.
    pub fn new(
        mother_type: OptionType,
        daughter_type: OptionType,
        mother_strike: f64,
        mother_expiry: f64,
        daughter_strike: f64,
        daughter_expiry: f64,
    ) -> Self {
        assert!(
            daughter_expiry > mother_expiry,
            "Daughter expiry must be after mother expiry"
        );
        Self {
            mother_type,
            daughter_type,
            mother_strike,
            mother_expiry,
            daughter_strike,
            daughter_expiry,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compound_creation() {
        let opt = CompoundOption::new(
            OptionType::Call,
            OptionType::Call,
            5.0,   // mother strike
            0.5,   // mother expiry
            100.0, // daughter strike
            1.0,   // daughter expiry
        );
        assert_eq!(opt.mother_type, OptionType::Call);
        assert_eq!(opt.daughter_type, OptionType::Call);
        assert_eq!(opt.mother_strike, 5.0);
        assert_eq!(opt.daughter_strike, 100.0);
    }

    #[test]
    fn compound_types() {
        // Call on put
        let opt = CompoundOption::new(
            OptionType::Call, OptionType::Put,
            3.0, 0.25, 100.0, 1.0,
        );
        assert_eq!(opt.mother_type, OptionType::Call);
        assert_eq!(opt.daughter_type, OptionType::Put);
    }

    #[test]
    #[should_panic]
    fn compound_invalid_expiry() {
        CompoundOption::new(
            OptionType::Call, OptionType::Call,
            5.0, 1.0, 100.0, 0.5, // daughter before mother
        );
    }
}
