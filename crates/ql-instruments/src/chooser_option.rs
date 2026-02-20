//! Chooser option instrument.
//!
//! A simple chooser option (as-you-like-it option) gives the holder
//! the right to choose at a future date whether the option is a call
//! or a put.  Priced via Rubinstein (1991).

use serde::{Deserialize, Serialize};

use crate::payoff::Exercise;

/// A simple chooser option.
///
/// At the choosing date the holder decides whether the option becomes
/// a European call or put with the given strike and expiry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChooserOption {
    /// Strike price for both the call and put.
    pub strike: f64,
    /// Exercise specification (European expiry = final expiry of the call/put).
    pub exercise: Exercise,
    /// Time (in years) until the holder must choose call vs put.
    /// Must satisfy `0 < choosing_time <= time_to_expiry`.
    pub choosing_time: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ql_time::{Date, Month};

    #[test]
    fn chooser_option_serde() {
        let opt = ChooserOption {
            strike: 105.0,
            exercise: Exercise::European {
                expiry: Date::from_ymd(2026, Month::June, 30),
            },
            choosing_time: 0.25,
        };
        let json = serde_json::to_string(&opt).unwrap();
        let opt2: ChooserOption = serde_json::from_str(&json).unwrap();
        assert!((opt.strike - opt2.strike).abs() < 1e-12);
        assert!((opt.choosing_time - opt2.choosing_time).abs() < 1e-12);
    }
}
