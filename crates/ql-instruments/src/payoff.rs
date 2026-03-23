//! Option payoffs and exercise types.
//!
//! Defines the payoff functions and exercise rules for vanilla and exotic options.

use ql_time::Date;
use serde::{Deserialize, Serialize};

// ===========================================================================
// OptionType
// ===========================================================================

/// Call or put.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptionType {
    /// Call.
    Call,
    /// Put.
    Put,
}

impl OptionType {
    /// +1 for call, -1 for put (the "omega" factor).
    pub fn sign(self) -> f64 {
        match self {
            OptionType::Call => 1.0,
            OptionType::Put => -1.0,
        }
    }
}

// ===========================================================================
// Payoff
// ===========================================================================

/// Payoff function for an option.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Payoff {
    /// Standard European/American payoff: max(omega*(S-K), 0).
    PlainVanilla {
        /// Field.
        option_type: OptionType,
        /// Field.
        strike: f64,
    },
    /// Digital (binary): pays a fixed cash amount if in-the-money.
    CashOrNothing {
        /// Field.
        option_type: OptionType,
        /// Field.
        strike: f64,
        /// Field.
        cash: f64,
    },
    /// Pays the asset price if in-the-money.
    AssetOrNothing {
        /// Field.
        option_type: OptionType,
        /// Field.
        strike: f64,
    },
    /// Gap payoff: pays (S - K2) if omega*(S-K1) > 0.
    Gap {
        /// Field.
        option_type: OptionType,
        /// Field.
        strike: f64,
        /// Field.
        second_strike: f64,
    },
}

impl Payoff {
    /// Evaluate the payoff at a given spot price.
    pub fn evaluate(&self, spot: f64) -> f64 {
        match self {
            Payoff::PlainVanilla { option_type, strike } => {
                (option_type.sign() * (spot - strike)).max(0.0)
            }
            Payoff::CashOrNothing {
                option_type,
                strike,
                cash,
            } => {
                if option_type.sign() * (spot - strike) > 0.0 {
                    *cash
                } else {
                    0.0
                }
            }
            Payoff::AssetOrNothing {
                option_type,
                strike,
            } => {
                if option_type.sign() * (spot - strike) > 0.0 {
                    spot
                } else {
                    0.0
                }
            }
            Payoff::Gap {
                option_type,
                strike,
                second_strike,
            } => {
                if option_type.sign() * (spot - strike) > 0.0 {
                    spot - second_strike
                } else {
                    0.0
                }
            }
        }
    }

    /// The option type of this payoff.
    pub fn option_type(&self) -> OptionType {
        match self {
            Payoff::PlainVanilla { option_type, .. }
            | Payoff::CashOrNothing { option_type, .. }
            | Payoff::AssetOrNothing { option_type, .. }
            | Payoff::Gap { option_type, .. } => *option_type,
        }
    }

    /// The strike price.
    pub fn strike(&self) -> f64 {
        match self {
            Payoff::PlainVanilla { strike, .. }
            | Payoff::CashOrNothing { strike, .. }
            | Payoff::AssetOrNothing { strike, .. }
            | Payoff::Gap { strike, .. } => *strike,
        }
    }
}

// ===========================================================================
// Exercise
// ===========================================================================

/// Exercise style of an option.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Exercise {
    /// European exercise: only at expiry.
    European {
        /// Expiry date.
        expiry: Date,
    },
    /// American exercise: any time between earliest and expiry.
    American {
        /// Earliest exercise date.
        earliest: Date,
        /// Expiry date.
        expiry: Date,
    },
    /// Bermudan exercise: on specific dates.
    Bermudan {
        /// Allowed exercise dates.
        dates: Vec<Date>,
    },
}

impl Exercise {
    /// The last possible exercise date.
    pub fn last_date(&self) -> Date {
        match self {
            Exercise::European { expiry } => *expiry,
            Exercise::American { expiry, .. } => *expiry,
            Exercise::Bermudan { dates } => dates[dates.len() - 1],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_time::Month;

    #[test]
    fn plain_vanilla_call_itm() {
        let p = Payoff::PlainVanilla {
            option_type: OptionType::Call,
            strike: 100.0,
        };
        assert_abs_diff_eq!(p.evaluate(110.0), 10.0, epsilon = 1e-15);
    }

    #[test]
    fn plain_vanilla_call_otm() {
        let p = Payoff::PlainVanilla {
            option_type: OptionType::Call,
            strike: 100.0,
        };
        assert_abs_diff_eq!(p.evaluate(90.0), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn plain_vanilla_put_itm() {
        let p = Payoff::PlainVanilla {
            option_type: OptionType::Put,
            strike: 100.0,
        };
        assert_abs_diff_eq!(p.evaluate(90.0), 10.0, epsilon = 1e-15);
    }

    #[test]
    fn cash_or_nothing_call() {
        let p = Payoff::CashOrNothing {
            option_type: OptionType::Call,
            strike: 100.0,
            cash: 50.0,
        };
        assert_abs_diff_eq!(p.evaluate(110.0), 50.0, epsilon = 1e-15);
        assert_abs_diff_eq!(p.evaluate(90.0), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn asset_or_nothing_put() {
        let p = Payoff::AssetOrNothing {
            option_type: OptionType::Put,
            strike: 100.0,
        };
        assert_abs_diff_eq!(p.evaluate(90.0), 90.0, epsilon = 1e-15);
        assert_abs_diff_eq!(p.evaluate(110.0), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn exercise_last_date() {
        let e = Exercise::European {
            expiry: Date::from_ymd(2025, Month::December, 15),
        };
        assert_eq!(e.last_date(), Date::from_ymd(2025, Month::December, 15));

        let e2 = Exercise::Bermudan {
            dates: vec![
                Date::from_ymd(2025, Month::June, 15),
                Date::from_ymd(2025, Month::December, 15),
            ],
        };
        assert_eq!(e2.last_date(), Date::from_ymd(2025, Month::December, 15));
    }

    #[test]
    fn option_type_sign() {
        assert_abs_diff_eq!(OptionType::Call.sign(), 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(OptionType::Put.sign(), -1.0, epsilon = 1e-15);
    }
}
