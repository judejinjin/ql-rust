//! Vanilla interest rate swap.
//!
//! A fixed-for-floating swap consisting of a fixed leg and a floating
//! (IBOR or overnight) leg. The swap's NPV is the difference between
//! the two legs' present values.

use ql_cashflows::Leg;
use ql_time::{Date, DayCounter, Schedule};

/// Type of swap (payer or receiver from the fixed-leg perspective).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwapType {
    /// Pay fixed, receive floating.
    Payer,
    /// Receive fixed, pay floating.
    Receiver,
}

/// A vanilla fixed-for-floating interest rate swap.
#[derive(Debug)]
pub struct VanillaSwap {
    /// Payer or receiver (from fixed-leg perspective).
    pub swap_type: SwapType,
    /// Notional amount.
    pub nominal: f64,
    /// Fixed leg.
    pub fixed_leg: Leg,
    /// Floating leg.
    pub floating_leg: Leg,
    /// Fixed rate (annualized).
    pub fixed_rate: f64,
    /// Spread on the floating leg.
    pub spread: f64,
}

impl VanillaSwap {
    /// Create a vanilla swap from pre-built legs.
    pub fn new(
        swap_type: SwapType,
        nominal: f64,
        fixed_leg: Leg,
        floating_leg: Leg,
        fixed_rate: f64,
        spread: f64,
    ) -> Self {
        Self {
            swap_type,
            nominal,
            fixed_leg,
            floating_leg,
            fixed_rate,
            spread,
        }
    }

    /// Create a vanilla swap from schedules and market data.
    ///
    /// Builds the fixed and floating legs internally.
    #[allow(clippy::too_many_arguments)]
    pub fn from_schedules(
        swap_type: SwapType,
        nominal: f64,
        fixed_schedule: &Schedule,
        fixed_rate: f64,
        fixed_day_counter: DayCounter,
        float_schedule: &Schedule,
        index: &ql_indexes::IborIndex,
        spread: f64,
        float_day_counter: DayCounter,
    ) -> Self {
        let fixed_leg = ql_cashflows::fixed_leg(
            fixed_schedule,
            &[nominal],
            &[fixed_rate],
            fixed_day_counter,
        );
        let floating_leg = ql_cashflows::ibor_leg(
            float_schedule,
            &[nominal],
            index,
            &[spread],
            float_day_counter,
        );

        Self {
            swap_type,
            nominal,
            fixed_leg,
            floating_leg,
            fixed_rate,
            spread,
        }
    }

    /// Whether the swap has expired (all cash flows in the past).
    pub fn is_expired(&self, ref_date: Date) -> bool {
        self.fixed_leg.iter().all(|cf| cf.has_occurred(ref_date))
            && self.floating_leg.iter().all(|cf| cf.has_occurred(ref_date))
    }

    /// The maturity date (last cash flow date).
    pub fn maturity_date(&self) -> Option<Date> {
        let fixed_last = self.fixed_leg.last().map(|cf| cf.date());
        let float_last = self.floating_leg.last().map(|cf| cf.date());
        match (fixed_last, float_last) {
            (Some(f), Some(fl)) => Some(f.max(fl)),
            (Some(f), None) => Some(f),
            (None, Some(fl)) => Some(fl),
            (None, None) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ql_indexes::IborIndex;
    use ql_time::Month;

    fn make_swap() -> VanillaSwap {
        let fixed_schedule = Schedule::from_dates(vec![
            Date::from_ymd(2025, Month::January, 15),
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
        ]);
        let float_schedule = Schedule::from_dates(vec![
            Date::from_ymd(2025, Month::January, 15),
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
        ]);
        let index = IborIndex::euribor_6m();

        VanillaSwap::from_schedules(
            SwapType::Payer,
            1_000_000.0,
            &fixed_schedule,
            0.05,
            DayCounter::Actual360,
            &float_schedule,
            &index,
            0.0,
            DayCounter::Actual360,
        )
    }

    #[test]
    fn swap_legs_created() {
        let swap = make_swap();
        assert_eq!(swap.fixed_leg.len(), 2);
        assert_eq!(swap.floating_leg.len(), 2);
    }

    #[test]
    fn swap_not_expired() {
        let swap = make_swap();
        assert!(!swap.is_expired(Date::from_ymd(2025, Month::January, 1)));
    }

    #[test]
    fn swap_maturity() {
        let swap = make_swap();
        assert_eq!(
            swap.maturity_date(),
            Some(Date::from_ymd(2026, Month::January, 15))
        );
    }

    #[test]
    fn swap_type_payer() {
        let swap = make_swap();
        assert_eq!(swap.swap_type, SwapType::Payer);
    }
}
