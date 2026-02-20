//! Double-barrier option instrument.
//!
//! A double-barrier option has both an upper and lower barrier.
//! The option knocks in or out when **either** barrier is breached.

use serde::{Deserialize, Serialize};

use crate::payoff::{Exercise, Payoff};

/// Type of double-barrier option.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DoubleBarrierType {
    /// Knock-out: extinguished if either barrier is reached.
    KnockOut,
    /// Knock-in: activated only if either barrier is reached.
    KnockIn,
}

/// A double-barrier option with an upper and lower barrier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoubleBarrierOption {
    /// Type of double-barrier (knock-in or knock-out).
    pub barrier_type: DoubleBarrierType,
    /// Lower barrier level (must be < spot < upper).
    pub lower_barrier: f64,
    /// Upper barrier level.
    pub upper_barrier: f64,
    /// Rebate paid if the option is knocked out (for KnockOut type).
    pub rebate: f64,
    /// Option payoff (typically `PlainVanilla`).
    pub payoff: Payoff,
    /// Exercise style (typically European).
    pub exercise: Exercise,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::payoff::OptionType;
    use ql_time::{Date, Month};

    #[test]
    fn double_barrier_option_serde() {
        let opt = DoubleBarrierOption {
            barrier_type: DoubleBarrierType::KnockOut,
            lower_barrier: 80.0,
            upper_barrier: 120.0,
            rebate: 0.0,
            payoff: Payoff::PlainVanilla {
                option_type: OptionType::Call,
                strike: 100.0,
            },
            exercise: Exercise::European {
                expiry: Date::from_ymd(2025, Month::December, 31),
            },
        };
        let json = serde_json::to_string(&opt).unwrap();
        let opt2: DoubleBarrierOption = serde_json::from_str(&json).unwrap();
        assert_eq!(opt.barrier_type, opt2.barrier_type);
        assert!((opt.lower_barrier - opt2.lower_barrier).abs() < 1e-12);
        assert!((opt.upper_barrier - opt2.upper_barrier).abs() < 1e-12);
    }
}
