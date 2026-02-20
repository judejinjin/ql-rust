//! Cliquet (ratchet) option instrument.
//!
//! A cliquet option is a series of forward-starting at-the-money options.
//! At each reset date the strike resets to the then-current spot.
//! The final payoff is typicallly the sum of capped/floored per-period returns.

use serde::{Deserialize, Serialize};

use crate::payoff::OptionType;

/// A cliquet (ratchet) option.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliquetOption {
    /// Call or Put.
    pub option_type: OptionType,
    /// Reset dates as year-fractions from valuation
    /// (e.g. [0.25, 0.50, 0.75, 1.00] for quarterly resets over 1 year).
    pub reset_times: Vec<f64>,
    /// Local floor on each period return (e.g. −0.05 for −5%).
    pub local_floor: f64,
    /// Local cap on each period return (e.g. 0.10 for +10%).
    pub local_cap: f64,
    /// Global floor on the accumulated return (often 0.0 for principal protection).
    pub global_floor: f64,
    /// Global cap on the accumulated return (f64::INFINITY for no cap).
    pub global_cap: f64,
    /// Notional amount.
    pub notional: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cliquet_option_serde() {
        let opt = CliquetOption {
            option_type: OptionType::Call,
            reset_times: vec![0.25, 0.50, 0.75, 1.00],
            local_floor: -0.05,
            local_cap: 0.10,
            global_floor: 0.0,
            global_cap: 1e9,
            notional: 1_000_000.0,
        };
        let json = serde_json::to_string(&opt).unwrap();
        let opt2: CliquetOption = serde_json::from_str(&json).unwrap();
        assert_eq!(opt.reset_times.len(), opt2.reset_times.len());
        assert_eq!(opt.option_type, opt2.option_type);
    }
}
