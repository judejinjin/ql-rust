//! Bond forward contract.
//!
//! A bond forward is an agreement to buy (long) or sell (short) a bond at a
//! specified forward settlement date at a pre-agreed forward price.
//!
//! # Valuation
//!
//! The forward price of a bond equals the spot dirty price minus the present
//! value of interim coupons, all forward-valued to the settlement date:
//!
//! $$F = (P_d - I) / d(0, T_f)$$
//!
//! where $P_d$ is the dirty price, $I$ is the PV of interim coupons, and
//! $d(0, T_f)$ is the discount factor to the forward settlement date.

use ql_time::Date;
use serde::{Deserialize, Serialize};

/// Long (buy) or short (sell) the bond.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BondForwardType {
    /// Long the bond (obligation to buy).
    Long,
    /// Short the bond (obligation to sell).
    Short,
}

impl BondForwardType {
    /// +1 for long, −1 for short.
    pub fn sign(&self) -> f64 {
        match self {
            Self::Long => 1.0,
            Self::Short => -1.0,
        }
    }
}

/// A forward contract on a bond.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BondForward {
    /// Long or short.
    pub forward_type: BondForwardType,
    /// Forward settlement date.
    pub settlement_date: Date,
    /// Agreed forward (clean) price.
    pub forward_price: f64,
    /// Notional / face of the underlying bond.
    pub face_amount: f64,
    /// Current spot dirty price of the underlying bond.
    pub spot_dirty_price: f64,
    /// Present value of interim coupon payments between now and settlement.
    pub interim_coupon_pv: f64,
    /// Discount factor from today to forward settlement date.
    pub discount_to_settlement: f64,
}

impl BondForward {
    /// Create a new bond forward.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        forward_type: BondForwardType,
        settlement_date: Date,
        forward_price: f64,
        face_amount: f64,
        spot_dirty_price: f64,
        interim_coupon_pv: f64,
        discount_to_settlement: f64,
    ) -> Self {
        Self {
            forward_type,
            settlement_date,
            forward_price,
            face_amount,
            spot_dirty_price,
            interim_coupon_pv,
            discount_to_settlement,
        }
    }

    /// Implied forward (dirty) price of the bond.
    ///
    /// $F = (P_{dirty} - I_{pv}) / d(0, T)$
    pub fn implied_forward_price(&self) -> f64 {
        if self.discount_to_settlement.abs() < 1e-15 {
            return 0.0;
        }
        (self.spot_dirty_price - self.interim_coupon_pv) / self.discount_to_settlement
    }

    /// NPV of the forward contract.
    ///
    /// For a long position: $(F_{implied} - K) \times d(0,T) \times N / 100$
    pub fn npv(&self) -> f64 {
        let implied = self.implied_forward_price();
        self.forward_type.sign()
            * (implied - self.forward_price)
            * self.discount_to_settlement
            * self.face_amount
            / 100.0
    }

    /// Whether the forward has settled.
    pub fn is_expired(&self, ref_date: Date) -> bool {
        self.settlement_date < ref_date
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ql_time::Month;

    #[test]
    fn implied_forward_price() {
        let fwd = BondForward::new(
            BondForwardType::Long,
            Date::from_ymd(2025, Month::June, 15),
            101.0,
            100_000.0,
            102.5,   // dirty price
            0.8,     // PV of interim coupons
            0.98,    // discount factor
        );
        let implied = fwd.implied_forward_price();
        // (102.5 - 0.8) / 0.98 = 101.7 / 0.98 ≈ 103.7755…
        let expected = (102.5 - 0.8) / 0.98;
        assert!((implied - expected).abs() < 1e-8);
    }

    #[test]
    fn npv_long() {
        let fwd = BondForward::new(
            BondForwardType::Long,
            Date::from_ymd(2025, Month::June, 15),
            100.0,
            100_000.0,
            102.5,
            0.5,
            0.97,
        );
        // implied = (102.5 - 0.5) / 0.97 = 105.1546…
        // NPV = (implied - 100.0) * 0.97 * 100_000 / 100
        let implied = (102.5 - 0.5) / 0.97;
        let expected = (implied - 100.0) * 0.97 * 100_000.0 / 100.0;
        assert!((fwd.npv() - expected).abs() < 1e-4);
    }

    #[test]
    fn npv_short_is_negative_of_long() {
        let base = BondForward::new(
            BondForwardType::Long,
            Date::from_ymd(2025, Month::June, 15),
            100.0, 100_000.0, 102.0, 0.3, 0.96,
        );
        let short = BondForward::new(
            BondForwardType::Short,
            Date::from_ymd(2025, Month::June, 15),
            100.0, 100_000.0, 102.0, 0.3, 0.96,
        );
        assert!((base.npv() + short.npv()).abs() < 1e-10);
    }

    #[test]
    fn is_expired() {
        let fwd = BondForward::new(
            BondForwardType::Long,
            Date::from_ymd(2025, Month::June, 15),
            100.0, 100_000.0, 102.0, 0.0, 1.0,
        );
        assert!(!fwd.is_expired(Date::from_ymd(2025, Month::January, 1)));
        assert!(fwd.is_expired(Date::from_ymd(2025, Month::July, 1)));
    }

    #[test]
    fn serde_roundtrip() {
        let fwd = BondForward::new(
            BondForwardType::Long,
            Date::from_ymd(2025, Month::June, 15),
            101.5, 1_000_000.0, 103.0, 1.2, 0.98,
        );
        let json = serde_json::to_string(&fwd).unwrap();
        let fwd2: BondForward = serde_json::from_str(&json).unwrap();
        assert!((fwd.npv() - fwd2.npv()).abs() < 1e-10);
    }
}
