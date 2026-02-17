//! Cap/Floor instruments.
//!
//! A cap (floor) is a series of caplets (floorlets), each being a call (put)
//! option on an interest rate for a single accrual period.

use serde::{Deserialize, Serialize};
use ql_time::Date;

/// Cap/Floor type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapFloorType {
    Cap,
    Floor,
    Collar,
}

impl CapFloorType {
    /// +1 for cap (call on rate), -1 for floor (put on rate).
    pub fn sign(&self) -> f64 {
        match self {
            Self::Cap => 1.0,
            Self::Floor => -1.0,
            Self::Collar => 1.0, // collar is cap - floor, sign depends on leg
        }
    }
}

/// A single caplet/floorlet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Caplet {
    /// Start date of the accrual period.
    pub accrual_start: Date,
    /// End date of the accrual period.
    pub accrual_end: Date,
    /// Payment date.
    pub payment_date: Date,
    /// Accrual fraction (year fraction).
    pub accrual_fraction: f64,
    /// Notional amount.
    pub notional: f64,
    /// Forward rate for this period.
    pub forward_rate: f64,
    /// Discount factor to payment date.
    pub discount: f64,
}

/// A cap or floor instrument, consisting of a strip of caplets/floorlets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapFloor {
    /// Cap or floor.
    pub cap_floor_type: CapFloorType,
    /// Strike rate.
    pub strike: f64,
    /// The individual caplets/floorlets.
    pub caplets: Vec<Caplet>,
}

impl CapFloor {
    /// Create a new cap or floor.
    pub fn new(cap_floor_type: CapFloorType, strike: f64, caplets: Vec<Caplet>) -> Self {
        Self {
            cap_floor_type,
            strike,
            caplets,
        }
    }

    /// Total notional (from first caplet).
    pub fn notional(&self) -> f64 {
        self.caplets.first().map_or(0.0, |c| c.notional)
    }

    /// Number of caplets/floorlets.
    pub fn size(&self) -> usize {
        self.caplets.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ql_time::Month;

    fn sample_caplets() -> Vec<Caplet> {
        (0..4)
            .map(|i| Caplet {
                accrual_start: Date::from_ymd(2025 + i, Month::January, 15),
                accrual_end: Date::from_ymd(2025 + i, Month::July, 15),
                payment_date: Date::from_ymd(2025 + i, Month::July, 17),
                accrual_fraction: 0.5,
                notional: 1_000_000.0,
                forward_rate: 0.035 + 0.001 * i as f64,
                discount: 0.98_f64.powi(i as i32 + 1),
            })
            .collect()
    }

    #[test]
    fn cap_creation() {
        let cap = CapFloor::new(CapFloorType::Cap, 0.03, sample_caplets());
        assert_eq!(cap.cap_floor_type, CapFloorType::Cap);
        assert_eq!(cap.size(), 4);
        assert_eq!(cap.notional(), 1_000_000.0);
    }

    #[test]
    fn floor_creation() {
        let floor = CapFloor::new(CapFloorType::Floor, 0.04, sample_caplets());
        assert_eq!(floor.cap_floor_type, CapFloorType::Floor);
        assert_eq!(floor.strike, 0.04);
    }

    #[test]
    fn cap_floor_type_sign() {
        assert_eq!(CapFloorType::Cap.sign(), 1.0);
        assert_eq!(CapFloorType::Floor.sign(), -1.0);
    }
}
