//! Basis swap — a float-for-float interest rate swap.
//!
//! Both legs are floating, each linked to a different IBOR index (or same
//! index with different tenors, e.g. 3M LIBOR vs 6M LIBOR). A spread is
//! typically applied to one leg.
//!
//! ## Example
//!
//! A 5Y USD 3M-LIBOR vs 6M-LIBOR basis swap where leg 1 pays 3M+5bp.

use ql_cashflows::Leg;
use ql_indexes::IborIndex;
use ql_time::{DayCounter, Schedule};

use crate::vanilla_swap::SwapType;

/// A basis swap (float vs float).
#[derive(Debug)]
pub struct BasisSwap {
    /// Payer (pays leg1, receives leg2) or Receiver.
    pub swap_type: SwapType,
    /// Notional principal.
    pub nominal: f64,
    /// First floating leg (e.g. 3M IBOR + spread1).
    pub leg1: Leg,
    /// Second floating leg (e.g. 6M IBOR + spread2).
    pub leg2: Leg,
    /// Spread on leg1.
    pub spread1: f64,
    /// Spread on leg2.
    pub spread2: f64,
}

impl BasisSwap {
    /// Create a basis swap from pre-built legs.
    pub fn new(
        swap_type: SwapType,
        nominal: f64,
        leg1: Leg,
        leg2: Leg,
        spread1: f64,
        spread2: f64,
    ) -> Self {
        Self {
            swap_type,
            nominal,
            leg1,
            leg2,
            spread1,
            spread2,
        }
    }

    /// Build a basis swap from schedules and two IBOR indexes.
    #[allow(clippy::too_many_arguments)]
    pub fn from_schedules(
        swap_type: SwapType,
        nominal: f64,
        schedule1: &Schedule,
        index1: &IborIndex,
        spread1: f64,
        schedule2: &Schedule,
        index2: &IborIndex,
        spread2: f64,
        day_counter: DayCounter,
    ) -> Self {
        let leg1 = ql_cashflows::ibor_leg(
            schedule1,
            &[nominal],
            index1,
            &[spread1],
            day_counter,
        );
        let leg2 = ql_cashflows::ibor_leg(
            schedule2,
            &[nominal],
            index2,
            &[spread2],
            day_counter,
        );
        Self {
            swap_type,
            nominal,
            leg1,
            leg2,
            spread1,
            spread2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ql_currencies::currency::Currency;
    use ql_time::{
        BusinessDayConvention, Calendar, Date, Month, Period, Schedule, TimeUnit,
    };

    #[test]
    fn basis_swap_construction() {
        let index3m = IborIndex::new(
            "USD-LIBOR-3M",
            Period::new(3, TimeUnit::Months),
            2,
            Currency::usd(),
            Calendar::Target,
            BusinessDayConvention::ModifiedFollowing,
            false,
            DayCounter::Actual360,
        );
        let index6m = IborIndex::new(
            "USD-LIBOR-6M",
            Period::new(6, TimeUnit::Months),
            2,
            Currency::usd(),
            Calendar::Target,
            BusinessDayConvention::ModifiedFollowing,
            false,
            DayCounter::Actual360,
        );

        let sched3m = Schedule::from_dates(vec![
            Date::from_ymd(2025, Month::January, 15),
            Date::from_ymd(2025, Month::April, 15),
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2025, Month::October, 15),
            Date::from_ymd(2026, Month::January, 15),
        ]);
        let sched6m = Schedule::from_dates(vec![
            Date::from_ymd(2025, Month::January, 15),
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
        ]);

        let swap = BasisSwap::from_schedules(
            SwapType::Payer,
            1_000_000.0,
            &sched3m,
            &index3m,
            0.0005, // 5bp spread on 3M leg
            &sched6m,
            &index6m,
            0.0,
            DayCounter::Actual360,
        );

        assert_eq!(swap.leg1.len(), 4);
        assert_eq!(swap.leg2.len(), 2);
    }
}
