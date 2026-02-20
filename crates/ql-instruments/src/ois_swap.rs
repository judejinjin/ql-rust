//! Overnight indexed swap (OIS).
//!
//! A fixed-for-overnight swap where the floating leg is compounded overnight
//! (SOFR, ESTR, SONIA, TONA, etc.). The fixed leg pays a fixed rate and the
//! floating leg pays the compounded overnight rate over each period.

use ql_cashflows::Leg;
use ql_indexes::OvernightIndex;
use ql_time::{Date, DayCounter, Schedule};

use crate::vanilla_swap::SwapType;

/// An overnight indexed swap (OIS).
#[derive(Debug)]
pub struct OISSwap {
    /// Payer or receiver (from fixed-leg perspective).
    pub swap_type: SwapType,
    /// Notional.
    pub nominal: f64,
    /// Fixed leg cash flows.
    pub fixed_leg: Leg,
    /// Overnight floating leg cash flows.
    pub floating_leg: Leg,
    /// Fixed rate (annualized).
    pub fixed_rate: f64,
    /// Spread on the overnight leg.
    pub spread: f64,
    /// The overnight index (e.g., SOFR).
    pub index: OvernightIndex,
}

impl OISSwap {
    /// Create an OIS from pre-built legs.
    pub fn new(
        swap_type: SwapType,
        nominal: f64,
        fixed_leg: Leg,
        floating_leg: Leg,
        fixed_rate: f64,
        spread: f64,
        index: OvernightIndex,
    ) -> Self {
        Self {
            swap_type,
            nominal,
            fixed_leg,
            floating_leg,
            fixed_rate,
            spread,
            index,
        }
    }

    /// Build an OIS from schedules and an overnight index.
    #[allow(clippy::too_many_arguments)]
    pub fn from_schedules(
        swap_type: SwapType,
        nominal: f64,
        fixed_schedule: &Schedule,
        fixed_rate: f64,
        fixed_day_counter: DayCounter,
        float_schedule: &Schedule,
        index: &OvernightIndex,
        spread: f64,
        float_day_counter: DayCounter,
    ) -> Self {
        let fixed_leg = ql_cashflows::fixed_leg(
            fixed_schedule,
            &[nominal],
            &[fixed_rate],
            fixed_day_counter,
        );
        let floating_leg = ql_cashflows::overnight_leg(
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
            index: index.clone(),
        }
    }

    /// Whether the swap has expired.
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

    /// Fair value NPV sign convention: positive = payer benefits.
    pub fn payer_sign(&self) -> f64 {
        match self.swap_type {
            SwapType::Payer => 1.0,
            SwapType::Receiver => -1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ql_time::Month;

    fn make_ois() -> OISSwap {
        let fixed_schedule = Schedule::from_dates(vec![
            Date::from_ymd(2025, Month::January, 2),
            Date::from_ymd(2025, Month::April, 2),
            Date::from_ymd(2025, Month::July, 2),
            Date::from_ymd(2025, Month::October, 2),
            Date::from_ymd(2026, Month::January, 2),
        ]);
        let float_schedule = fixed_schedule.clone();
        let index = OvernightIndex::sofr();

        OISSwap::from_schedules(
            SwapType::Payer,
            1_000_000.0,
            &fixed_schedule,
            0.04,
            DayCounter::Actual360,
            &float_schedule,
            &index,
            0.0,
            DayCounter::Actual360,
        )
    }

    #[test]
    fn ois_leg_counts() {
        let ois = make_ois();
        assert_eq!(ois.fixed_leg.len(), 4);
        assert_eq!(ois.floating_leg.len(), 4);
    }

    #[test]
    fn ois_maturity() {
        let ois = make_ois();
        assert_eq!(
            ois.maturity_date(),
            Some(Date::from_ymd(2026, Month::January, 2))
        );
    }

    #[test]
    fn ois_not_expired() {
        let ois = make_ois();
        assert!(!ois.is_expired(Date::from_ymd(2025, Month::January, 1)));
    }

    #[test]
    fn ois_payer_sign() {
        let ois = make_ois();
        assert_eq!(ois.payer_sign(), 1.0);
    }

    #[test]
    fn ois_fixed_leg_amounts_positive() {
        let ois = make_ois();
        for cf in &ois.fixed_leg {
            assert!(cf.amount() > 0.0);
        }
    }
}
