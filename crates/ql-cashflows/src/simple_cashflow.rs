//! Simple cash flow — a fixed amount on a specific date.
//!
//! Used for notional exchanges, bullet payments, and similar fixed flows.

use ql_time::Date;

use crate::cashflow::CashFlow;

/// A simple cash flow: a known amount paid on a known date.
#[derive(Debug, Clone)]
pub struct SimpleCashFlow {
    /// Payment date.
    pub date: Date,
    /// Amount (positive = receipt).
    pub amount: f64,
}

impl SimpleCashFlow {
    /// Create a new simple cash flow.
    pub fn new(date: Date, amount: f64) -> Self {
        Self { date, amount }
    }
}

impl CashFlow for SimpleCashFlow {
    fn date(&self) -> Date {
        self.date
    }

    fn amount(&self) -> f64 {
        self.amount
    }

    fn as_any(&self) -> &dyn std::any::Any { self }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ql_time::Month;

    #[test]
    fn simple_cashflow_creation() {
        let cf = SimpleCashFlow::new(Date::from_ymd(2025, Month::June, 15), 1_000_000.0);
        assert!((cf.amount() - 1_000_000.0).abs() < 1e-10);
        assert_eq!(cf.date(), Date::from_ymd(2025, Month::June, 15));
    }

    #[test]
    fn simple_cashflow_has_occurred() {
        let cf = SimpleCashFlow::new(Date::from_ymd(2025, Month::January, 15), 100.0);
        assert!(cf.has_occurred(Date::from_ymd(2025, Month::February, 1)));
        assert!(!cf.has_occurred(Date::from_ymd(2025, Month::January, 1)));
    }
}
