//! Swaption instrument.
//!
//! A swaption gives the holder the right to enter into an interest rate swap.

use serde::{Deserialize, Serialize};
use ql_time::Date;

/// Settlement type for swaptions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SettlementType {
    /// Physical delivery: the swap is entered upon exercise.
    Physical,
    /// Cash settlement: a cash payment is made.
    Cash,
}

/// Swaption type: payer or receiver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SwaptionType {
    /// Right to pay fixed (receive floating).
    Payer,
    /// Right to receive fixed (pay floating).
    Receiver,
}

impl SwaptionType {
    /// +1 for payer, -1 for receiver (consistent with payoff sign).
    pub fn sign(&self) -> f64 {
        match self {
            Self::Payer => 1.0,
            Self::Receiver => -1.0,
        }
    }
}

/// A swaption instrument.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Swaption {
    /// Payer or receiver.
    pub swaption_type: SwaptionType,
    /// Fixed rate of the underlying swap.
    pub strike: f64,
    /// Expiry date of the swaption.
    pub expiry: Date,
    /// Swap start date (= swaption expiry for European).
    pub swap_start: Date,
    /// Swap maturity date.
    pub swap_maturity: Date,
    /// Swap tenor in years (for pricing convenience).
    pub swap_tenor: f64,
    /// Annuity (DV01) of the underlying swap's fixed leg.
    pub annuity: f64,
    /// Forward swap rate.
    pub forward_rate: f64,
    /// Settlement type.
    pub settlement: SettlementType,
}

impl Swaption {
    /// Create a new European swaption.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        swaption_type: SwaptionType,
        strike: f64,
        expiry: Date,
        swap_start: Date,
        swap_maturity: Date,
        swap_tenor: f64,
        annuity: f64,
        forward_rate: f64,
        settlement: SettlementType,
    ) -> Self {
        Self {
            swaption_type,
            strike,
            expiry,
            swap_start,
            swap_maturity,
            swap_tenor,
            annuity,
            forward_rate,
            settlement,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ql_time::Month;

    #[test]
    fn swaption_creation() {
        let s = Swaption::new(
            SwaptionType::Payer,
            0.03,
            Date::from_ymd(2026, Month::January, 15),
            Date::from_ymd(2026, Month::January, 17),
            Date::from_ymd(2031, Month::January, 17),
            5.0,
            4.5,
            0.035,
            SettlementType::Physical,
        );
        assert_eq!(s.swaption_type, SwaptionType::Payer);
        assert_eq!(s.strike, 0.03);
    }

    #[test]
    fn swaption_type_sign() {
        assert_eq!(SwaptionType::Payer.sign(), 1.0);
        assert_eq!(SwaptionType::Receiver.sign(), -1.0);
    }
}
