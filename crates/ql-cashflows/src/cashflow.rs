//! Core cash flow traits and the `Leg` type alias.
//!
//! A `CashFlow` is anything that produces a known (or projected) amount on a
//! specific date. A `Leg` is a sequence of cash flows, representing one side
//! of a financial instrument.

use ql_time::Date;

// ===========================================================================
// CashFlow trait
// ===========================================================================

/// A single cash flow — an amount on a date.
pub trait CashFlow: Send + Sync + std::fmt::Debug {
    /// Payment date.
    fn date(&self) -> Date;

    /// Cash flow amount (positive = receipt, negative = payment).
    fn amount(&self) -> f64;

    /// Whether this cash flow has already occurred relative to `ref_date`.
    ///
    /// Default: the payment date is strictly before `ref_date`.
    fn has_occurred(&self, ref_date: Date) -> bool {
        self.date() < ref_date
    }

    /// Downcast support for concrete type inspection.
    fn as_any(&self) -> &dyn std::any::Any;
}

// ===========================================================================
// Leg type alias
// ===========================================================================

/// A leg is an ordered sequence of cash flows, typically representing one
/// side of a swap or bond.
pub type Leg = Vec<Box<dyn CashFlow>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct DummyCf {
        date: Date,
        amount: f64,
    }

    impl CashFlow for DummyCf {
        fn date(&self) -> Date { self.date }
        fn amount(&self) -> f64 { self.amount }
        fn as_any(&self) -> &dyn std::any::Any { self }
    }

    #[test]
    fn has_occurred_before_ref() {
        use ql_time::Month;
        let cf = DummyCf {
            date: Date::from_ymd(2025, Month::January, 15),
            amount: 100.0,
        };
        let ref_date = Date::from_ymd(2025, Month::February, 1);
        assert!(cf.has_occurred(ref_date));
    }

    #[test]
    fn has_not_occurred_after_ref() {
        use ql_time::Month;
        let cf = DummyCf {
            date: Date::from_ymd(2025, Month::June, 15),
            amount: 100.0,
        };
        let ref_date = Date::from_ymd(2025, Month::January, 1);
        assert!(!cf.has_occurred(ref_date));
    }

    #[test]
    fn leg_is_vec_of_cashflows() {
        use ql_time::Month;
        let mut leg: Leg = Vec::new();
        leg.push(Box::new(DummyCf {
            date: Date::from_ymd(2025, Month::March, 15),
            amount: 50.0,
        }));
        leg.push(Box::new(DummyCf {
            date: Date::from_ymd(2025, Month::June, 15),
            amount: 50.0,
        }));
        assert_eq!(leg.len(), 2);
        assert!((leg[0].amount() - 50.0).abs() < 1e-15);
    }
}
