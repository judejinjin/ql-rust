//! Credit Default Swap instrument.
//!
//! A CDS is a contract where the protection buyer pays periodic premiums
//! and receives a payment upon credit default.

use serde::{Deserialize, Serialize};
use ql_time::Date;

/// CDS protection side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CdsProtectionSide {
    /// Protection buyer (pays spread, receives on default).
    Buyer,
    /// Protection seller (receives spread, pays on default).
    Seller,
}

impl CdsProtectionSide {
    /// +1 for buyer, -1 for seller.
    pub fn sign(&self) -> f64 {
        match self {
            Self::Buyer => 1.0,
            Self::Seller => -1.0,
        }
    }
}

/// A Credit Default Swap instrument.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditDefaultSwap {
    /// Protection side (buyer or seller).
    pub side: CdsProtectionSide,
    /// Notional amount.
    pub notional: f64,
    /// Running spread (coupon rate).
    pub spread: f64,
    /// Maturity date.
    pub maturity: Date,
    /// Recovery rate assumption.
    pub recovery_rate: f64,
    /// Premium payment dates and accrual fractions.
    pub schedule: Vec<CdsPremiumPeriod>,
}

/// A single premium period of a CDS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdsPremiumPeriod {
    /// Start of accrual period.
    pub accrual_start: Date,
    /// End of accrual period.
    pub accrual_end: Date,
    /// Payment date.
    pub payment_date: Date,
    /// Year fraction for accrual.
    pub accrual_fraction: f64,
}

impl CreditDefaultSwap {
    /// Create a new CDS.
    pub fn new(
        side: CdsProtectionSide,
        notional: f64,
        spread: f64,
        maturity: Date,
        recovery_rate: f64,
        schedule: Vec<CdsPremiumPeriod>,
    ) -> Self {
        Self {
            side,
            notional,
            spread,
            maturity,
            recovery_rate,
            schedule,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ql_time::Month;

    fn make_schedule() -> Vec<CdsPremiumPeriod> {
        (0..20)
            .map(|i| {
                let year = 2025 + i / 4;
                let start_month = match i % 4 {
                    0 => Month::March,
                    1 => Month::June,
                    2 => Month::September,
                    _ => Month::December,
                };
                let end_month = match (i + 1) % 4 {
                    0 => Month::March,
                    1 => Month::June,
                    2 => Month::September,
                    _ => Month::December,
                };
                let end_year = if (i + 1) % 4 == 0 { year + 1 } else { year };
                CdsPremiumPeriod {
                    accrual_start: Date::from_ymd(year as i32, start_month, 20),
                    accrual_end: Date::from_ymd(end_year as i32, end_month, 20),
                    payment_date: Date::from_ymd(end_year as i32, end_month, 20),
                    accrual_fraction: 0.25,
                }
            })
            .collect()
    }

    #[test]
    fn cds_creation() {
        let cds = CreditDefaultSwap::new(
            CdsProtectionSide::Buyer,
            10_000_000.0,
            0.01,
            Date::from_ymd(2030, Month::March, 20),
            0.4,
            make_schedule(),
        );
        assert_eq!(cds.side, CdsProtectionSide::Buyer);
        assert_eq!(cds.notional, 10_000_000.0);
        assert_eq!(cds.schedule.len(), 20);
    }

    #[test]
    fn cds_protection_side_sign() {
        assert_eq!(CdsProtectionSide::Buyer.sign(), 1.0);
        assert_eq!(CdsProtectionSide::Seller.sign(), -1.0);
    }
}
