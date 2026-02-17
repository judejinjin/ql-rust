//! Coupon trait — extends `CashFlow` with accrual period information.
//!
//! All interest coupons (fixed, floating, overnight) implement this trait
//! in addition to `CashFlow`.

use ql_time::{Date, DayCounter};

use crate::cashflow::CashFlow;

// ===========================================================================
// Coupon trait
// ===========================================================================

/// An interest coupon — a cash flow that accrues over a period.
pub trait Coupon: CashFlow {
    /// The notional amount on which interest accrues.
    fn nominal(&self) -> f64;

    /// The coupon rate (annualized).
    fn rate(&self) -> f64;

    /// Start of the accrual period.
    fn accrual_start(&self) -> Date;

    /// End of the accrual period.
    fn accrual_end(&self) -> Date;

    /// Accrual period as a year fraction.
    fn accrual_period(&self) -> f64;

    /// Day counter used for accrual computation.
    fn day_counter(&self) -> DayCounter;

    /// Accrued amount at the given date (pro-rata of the full coupon).
    fn accrued_amount(&self, date: Date) -> f64 {
        if date <= self.accrual_start() || date > self.accrual_end() {
            return 0.0;
        }
        let dc = self.day_counter();
        let full_period = dc.year_fraction(self.accrual_start(), self.accrual_end());
        let accrued_period = dc.year_fraction(self.accrual_start(), date);
        if full_period.abs() < 1e-15 {
            return 0.0;
        }
        self.amount() * (accrued_period / full_period)
    }
}
