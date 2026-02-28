//! Indexed cash flow.
//!
//! A cash flow whose amount is determined by multiplying a base amount
//! by an index ratio (e.g., CPI ratio for inflation-linked products,
//! or a general index factor).
//!
//! Corresponds to QuantLib's `IndexedCashFlow`.

use serde::{Deserialize, Serialize};

/// An indexed cash flow that adjusts a notional by an index ratio.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedCashFlow {
    /// Base notional amount.
    pub notional: f64,
    /// Index value at base date.
    pub base_index: f64,
    /// Index value at fixing date.
    pub fixing_index: Option<f64>,
    /// Payment time (years from reference date).
    pub payment_time: f64,
    /// Whether to pay the principal adjustment only (true) or full indexed amount (false).
    pub pays_adjustment_only: bool,
}

impl IndexedCashFlow {
    /// Create a new indexed cash flow.
    pub fn new(
        notional: f64,
        base_index: f64,
        payment_time: f64,
        pays_adjustment_only: bool,
    ) -> Self {
        Self {
            notional,
            base_index,
            fixing_index: None,
            payment_time,
            pays_adjustment_only,
        }
    }

    /// Create with known fixing.
    pub fn with_fixing(
        notional: f64,
        base_index: f64,
        fixing_index: f64,
        payment_time: f64,
        pays_adjustment_only: bool,
    ) -> Self {
        Self {
            notional,
            base_index,
            fixing_index: Some(fixing_index),
            payment_time,
            pays_adjustment_only,
        }
    }

    /// Set the fixing index value.
    pub fn set_fixing(&mut self, fixing: f64) {
        self.fixing_index = Some(fixing);
    }

    /// Index ratio = fixing_index / base_index.
    pub fn index_ratio(&self) -> Option<f64> {
        self.fixing_index.map(|f| f / self.base_index)
    }

    /// Cash flow amount.
    ///
    /// If `pays_adjustment_only`: amount = notional × (I_T/I_0 − 1)
    /// Otherwise: amount = notional × I_T/I_0
    pub fn amount(&self) -> Option<f64> {
        let ratio = self.index_ratio()?;
        if self.pays_adjustment_only {
            Some(self.notional * (ratio - 1.0))
        } else {
            Some(self.notional * ratio)
        }
    }

    /// Discounted amount given a discount factor.
    pub fn present_value(&self, discount_factor: f64) -> Option<f64> {
        self.amount().map(|a| a * discount_factor)
    }
}

/// A sequence of indexed cash flows (e.g., inflation-linked bond redemptions).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedCashFlowLeg {
    /// Individual indexed cash flows.
    pub flows: Vec<IndexedCashFlow>,
}

impl IndexedCashFlowLeg {
    /// Create a new indexed cash flow leg.
    pub fn new() -> Self {
        Self { flows: Vec::new() }
    }

    /// Add a cash flow to the leg.
    pub fn add(&mut self, flow: IndexedCashFlow) {
        self.flows.push(flow);
    }

    /// Total amount of all cash flows.
    pub fn total_amount(&self) -> Option<f64> {
        let amounts: Option<Vec<f64>> = self.flows.iter().map(|f| f.amount()).collect();
        amounts.map(|a| a.iter().sum())
    }

    /// Total present value given a flat discount rate.
    pub fn present_value(&self, rate: f64) -> Option<f64> {
        let mut pv = 0.0;
        for flow in &self.flows {
            let amt = flow.amount()?;
            let df = (-rate * flow.payment_time).exp();
            pv += amt * df;
        }
        Some(pv)
    }

    /// Number of cash flows.
    pub fn size(&self) -> usize {
        self.flows.len()
    }
}

impl Default for IndexedCashFlowLeg {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indexed_cashflow_full() {
        let cf = IndexedCashFlow::with_fixing(
            1_000_000.0, 100.0, 110.0, 1.0, false,
        );
        let amount = cf.amount().unwrap();
        // 1M × 110/100 = 1.1M
        assert!((amount - 1_100_000.0).abs() < 0.01, "amount={}", amount);
    }

    #[test]
    fn test_indexed_cashflow_adjustment_only() {
        let cf = IndexedCashFlow::with_fixing(
            1_000_000.0, 100.0, 110.0, 1.0, true,
        );
        let amount = cf.amount().unwrap();
        // 1M × (110/100 − 1) = 100,000
        assert!((amount - 100_000.0).abs() < 0.01, "amount={}", amount);
    }

    #[test]
    fn test_indexed_cashflow_no_fixing() {
        let cf = IndexedCashFlow::new(1_000_000.0, 100.0, 1.0, false);
        assert!(cf.amount().is_none());
    }

    #[test]
    fn test_indexed_cashflow_set_fixing() {
        let mut cf = IndexedCashFlow::new(1_000_000.0, 100.0, 1.0, false);
        cf.set_fixing(105.0);
        let amount = cf.amount().unwrap();
        assert!((amount - 1_050_000.0).abs() < 0.01, "amount={}", amount);
    }

    #[test]
    fn test_indexed_cashflow_pv() {
        let cf = IndexedCashFlow::with_fixing(
            1_000_000.0, 100.0, 110.0, 2.0, false,
        );
        let df = (-0.05_f64 * 2.0).exp();
        let pv = cf.present_value(df).unwrap();
        assert!((pv - 1_100_000.0 * df).abs() < 0.01, "pv={}", pv);
    }

    #[test]
    fn test_indexed_leg() {
        let mut leg = IndexedCashFlowLeg::new();
        leg.add(IndexedCashFlow::with_fixing(1e6, 100.0, 105.0, 1.0, false));
        leg.add(IndexedCashFlow::with_fixing(1e6, 100.0, 110.0, 2.0, false));

        let total = leg.total_amount().unwrap();
        let expected = 1e6 * 1.05 + 1e6 * 1.10;
        assert!((total - expected).abs() < 0.01, "total={}", total);

        let pv = leg.present_value(0.03).unwrap();
        assert!(pv > 0.0 && pv < total, "pv={}", pv);
    }

    #[test]
    fn test_deflation() {
        // Index decreases
        let cf = IndexedCashFlow::with_fixing(
            1_000_000.0, 100.0, 95.0, 1.0, true,
        );
        let amount = cf.amount().unwrap();
        // Adjustment = 1M × (0.95 − 1) = −50,000
        assert!((amount - (-50_000.0)).abs() < 0.01, "amount={}", amount);
    }
}
