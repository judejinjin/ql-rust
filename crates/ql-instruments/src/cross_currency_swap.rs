//! Cross-currency swap — exchange of cash flows in two different currencies.
//!
//! Typically one leg is a fixed or floating rate in currency A, and the other
//! is a fixed or floating rate in currency B. Notional exchange occurs at
//! inception and maturity (mark-to-market cross-currency swaps re-exchange
//! notionals at reset dates).
//!
//! ## Common Variants
//!
//! - **Fixed-for-fixed**: e.g. pay 2% USD, receive 1% EUR
//! - **Fixed-for-floating**: e.g. pay 3M EURIBOR, receive 3% USD fixed
//! - **Floating-for-floating**: e.g. pay SOFR, receive ESTR + spread

use ql_cashflows::Leg;
use ql_currencies::currency::Currency;

use crate::vanilla_swap::SwapType;

/// Cross-currency swap leg specification.
#[derive(Debug)]
pub struct XCcyLeg {
    /// Currency of the leg.
    pub currency: Currency,
    /// Cash flows for the leg.
    pub cashflows: Leg,
    /// Notional in the leg's currency.
    pub notional: f64,
}

/// A cross-currency swap.
#[derive(Debug)]
pub struct CrossCurrencySwap {
    /// Payer of leg1 / receiver of leg2.
    pub swap_type: SwapType,
    /// First leg (domestic currency).
    pub leg1: XCcyLeg,
    /// Second leg (foreign currency).
    pub leg2: XCcyLeg,
    /// Exchange notionals at start?
    pub exchange_initial_notional: bool,
    /// Exchange notionals at maturity?
    pub exchange_final_notional: bool,
    /// FX rate (units of leg1 currency per 1 unit of leg2 currency) at inception.
    pub initial_fx_rate: f64,
}

impl CrossCurrencySwap {
    /// Create a new cross-currency swap.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        swap_type: SwapType,
        leg1: XCcyLeg,
        leg2: XCcyLeg,
        exchange_initial_notional: bool,
        exchange_final_notional: bool,
        initial_fx_rate: f64,
    ) -> Self {
        Self {
            swap_type,
            leg1,
            leg2,
            exchange_initial_notional,
            exchange_final_notional,
            initial_fx_rate,
        }
    }
}

/// A float-for-float swap where both legs reference IBOR or overnight indices.
///
/// This is the same structure as a [`BasisSwap`](crate::basis_swap::BasisSwap),
/// but here we make the naming explicit for the specific pattern of two
/// floating legs under the same currency or cross-currency.
#[derive(Debug)]
pub struct FloatFloatSwap {
    /// Payer/receiver convention.
    pub swap_type: SwapType,
    /// Notional.
    pub nominal: f64,
    /// First floating leg.
    pub leg1: Leg,
    /// Second floating leg.
    pub leg2: Leg,
}

impl FloatFloatSwap {
    /// Create a new float-for-float swap from pre-built legs.
    pub fn new(swap_type: SwapType, nominal: f64, leg1: Leg, leg2: Leg) -> Self {
        Self {
            swap_type,
            nominal,
            leg1,
            leg2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ql_cashflows::leg::fixed_leg;
    use ql_time::{Date, DayCounter, Month, Schedule};

    #[test]
    fn cross_currency_swap_construction() {
        let schedule = Schedule::from_dates(vec![
            Date::from_ymd(2025, Month::January, 15),
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
        ]);

        let usd_leg = fixed_leg(&schedule, &[1_000_000.0], &[0.03], DayCounter::Actual360);
        let eur_leg = fixed_leg(&schedule, &[900_000.0], &[0.02], DayCounter::Actual360);

        let swap = CrossCurrencySwap::new(
            SwapType::Payer,
            XCcyLeg {
                currency: Currency::usd(),
                cashflows: usd_leg,
                notional: 1_000_000.0,
            },
            XCcyLeg {
                currency: Currency::eur(),
                cashflows: eur_leg,
                notional: 900_000.0,
            },
            true,
            true,
            1.1111, // 1 EUR = 1.1111 USD
        );

        assert!(swap.exchange_initial_notional);
        assert!(swap.exchange_final_notional);
    }
}
