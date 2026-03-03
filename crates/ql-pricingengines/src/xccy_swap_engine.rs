//! Cross-currency swap pricing engine.
//!
//! Discounts each leg using its own currency's yield curve and optionally
//! includes notional exchanges at inception and maturity.
//!
//! ## QuantLib Parity
//!
//! Corresponds to `DiscountingCurrencySwapEngine` in QuantLib C++.

use ql_cashflows::npv as leg_npv;
use ql_instruments::cross_currency_swap::CrossCurrencySwap;
use ql_instruments::vanilla_swap::SwapType;
use ql_termstructures::YieldTermStructure;
use ql_time::Date;

/// Results from the cross-currency swap pricing engine.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct XccySwapResult {
    /// NPV of the swap in leg1 (domestic) currency.
    pub npv: f64,
    /// Leg1 (domestic) NPV in its own currency.
    pub leg1_npv: f64,
    /// Leg2 (foreign) NPV in its own currency.
    pub leg2_npv: f64,
    /// Leg2 NPV converted to domestic currency at the spot FX rate.
    pub leg2_npv_domestic: f64,
    /// Notional exchange PV (initial + final) in domestic currency.
    pub notional_exchange_pv: f64,
    /// Fair basis spread on leg2 that makes NPV = 0 (in bps).
    pub fair_basis_spread_bps: f64,
}

/// Price a cross-currency swap by discounting each leg with its currency's curve.
///
/// # Arguments
/// * `swap` — the cross-currency swap instrument
/// * `curve1` — yield curve for leg1 (domestic) currency
/// * `curve2` — yield curve for leg2 (foreign) currency
/// * `spot_fx` — current spot FX rate: units of leg1 ccy per 1 unit of leg2 ccy
/// * `settle` — settlement / valuation date
///
/// # Returns
/// [`XccySwapResult`] with NPV in domestic (leg1) currency.
pub fn price_xccy_swap(
    swap: &CrossCurrencySwap,
    curve1: &dyn YieldTermStructure,
    curve2: &dyn YieldTermStructure,
    spot_fx: f64,
    settle: Date,
) -> XccySwapResult {
    // Discount each leg with its own curve
    let leg1_npv = leg_npv(&swap.leg1.cashflows, curve1, settle);
    let leg2_npv = leg_npv(&swap.leg2.cashflows, curve2, settle);

    // Convert leg2 to domestic currency at spot FX
    let leg2_npv_domestic = leg2_npv * spot_fx;

    // Notional exchange PV
    let mut notional_exchange_pv = 0.0;

    if swap.exchange_initial_notional {
        // At inception: receive leg2 notional × fx, pay leg1 notional
        // For a payer of leg1, we pay leg1 notional and receive leg2 notional × fx
        // Assume exchange happens at settle, so no discounting needed for past exchanges
        // For a new trade at inception, the notional exchange nets to zero
        // For a seasoned trade, the initial exchange is in the past
    }

    if swap.exchange_final_notional {
        // At maturity: pay back notionals in each currency
        // leg1 receives notional back discounted at curve1
        // leg2 receives notional back discounted at curve2
        // We need to find the maturity date from the last cashflow
        let leg1_dates: Vec<Date> = swap.leg1.cashflows.iter().map(|cf| cf.date()).collect();
        let leg2_dates: Vec<Date> = swap.leg2.cashflows.iter().map(|cf| cf.date()).collect();

        if let Some(&maturity1) = leg1_dates.last() {
            let t1 = curve1.day_counter().year_fraction(settle, maturity1);
            if t1 > 0.0 {
                let df1 = curve1.discount_t(t1);
                let df2 = if let Some(&maturity2) = leg2_dates.last() {
                    let t2 = curve2.day_counter().year_fraction(settle, maturity2);
                    curve2.discount_t(t2.max(0.0))
                } else {
                    curve2.discount_t(t1)
                };

                // Leg1 side: receive notional1 at maturity
                let leg1_notional_pv = swap.leg1.notional * df1;
                // Leg2 side: receive notional2 at maturity, convert at spot_fx
                let leg2_notional_pv = swap.leg2.notional * df2 * spot_fx;

                notional_exchange_pv = leg2_notional_pv - leg1_notional_pv;
            }
        }
    }

    let sign = match swap.swap_type {
        SwapType::Payer => -1.0,  // Pay leg1, receive leg2
        SwapType::Receiver => 1.0, // Receive leg1, pay leg2
    };

    // NPV from perspective of leg1 receiver:
    //   receive leg1 coupons - pay leg2 coupons (in domestic terms) + notional exchanges
    let coupon_npv = sign * (leg1_npv - leg2_npv_domestic);
    let npv = coupon_npv + sign * notional_exchange_pv;

    // Fair basis spread: the bps spread on leg2 that makes NPV = 0
    // Approximate: NPV ≈ 0 ⟹ spread_adj × leg2_bpv = -coupon_npv
    // leg2_bpv ≈ leg2_notional × Σ(df × Δt)
    let leg2_bpv = {
        let mut bpv = 0.0;
        let dc = curve2.day_counter();
        let dates: Vec<Date> = swap.leg2.cashflows.iter().map(|cf| cf.date()).collect();
        for i in 0..dates.len() {
            let t = dc.year_fraction(settle, dates[i]);
            if t > 0.0 {
                let df = curve2.discount_t(t);
                let dt = if i > 0 {
                    dc.year_fraction(dates[i - 1], dates[i])
                } else {
                    dc.year_fraction(settle, dates[i])
                };
                bpv += df * dt;
            }
        }
        bpv * swap.leg2.notional * spot_fx
    };

    let fair_basis_spread_bps = if leg2_bpv.abs() > 1e-15 {
        -npv / leg2_bpv * 10_000.0
    } else {
        0.0
    };

    XccySwapResult {
        npv,
        leg1_npv,
        leg2_npv,
        leg2_npv_domestic,
        notional_exchange_pv,
        fair_basis_spread_bps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_cashflows::leg::fixed_leg;
    use ql_currencies::currency::Currency;
    use ql_instruments::cross_currency_swap::XCcyLeg;
    use ql_termstructures::FlatForward;
    use ql_time::{DayCounter, Month, Schedule};

    fn make_test_swap() -> (CrossCurrencySwap, Date) {
        let settle = Date::from_ymd(2025, Month::January, 15);
        let schedule = Schedule::from_dates(vec![
            Date::from_ymd(2025, Month::January, 15),
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
            Date::from_ymd(2026, Month::July, 15),
            Date::from_ymd(2027, Month::January, 15),
        ]);

        let usd_leg = fixed_leg(&schedule, &[1_000_000.0], &[0.04], DayCounter::Actual360);
        let eur_leg = fixed_leg(&schedule, &[900_000.0], &[0.03], DayCounter::Actual360);

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
            false,
            true,
            1.1111,
        );
        (swap, settle)
    }

    #[test]
    fn xccy_swap_pricing_basic() {
        let (swap, settle) = make_test_swap();
        let usd_curve = FlatForward::new(settle, 0.04, DayCounter::Actual360);
        let eur_curve = FlatForward::new(settle, 0.03, DayCounter::Actual360);

        let result = price_xccy_swap(&swap, &usd_curve, &eur_curve, 1.1111, settle);

        // With notional exchange at maturity and different curves, NPV should be non-trivial
        assert!(result.leg1_npv > 0.0, "Leg1 NPV should be positive");
        assert!(result.leg2_npv > 0.0, "Leg2 NPV should be positive");
    }

    #[test]
    fn xccy_swap_fx_sensitivity() {
        let (swap, settle) = make_test_swap();
        let usd_curve = FlatForward::new(settle, 0.04, DayCounter::Actual360);
        let eur_curve = FlatForward::new(settle, 0.03, DayCounter::Actual360);

        let result_low = price_xccy_swap(&swap, &usd_curve, &eur_curve, 1.05, settle);
        let result_high = price_xccy_swap(&swap, &usd_curve, &eur_curve, 1.20, settle);

        // Higher FX rate means leg2 is worth more in domestic terms
        assert!(
            result_high.leg2_npv_domestic > result_low.leg2_npv_domestic,
            "Higher FX should increase leg2 domestic value"
        );
    }

    #[test]
    fn xccy_swap_same_currency_is_like_vanilla() {
        let settle = Date::from_ymd(2025, Month::January, 15);
        let schedule = Schedule::from_dates(vec![
            Date::from_ymd(2025, Month::January, 15),
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
        ]);

        let leg1_cf = fixed_leg(&schedule, &[1_000_000.0], &[0.05], DayCounter::Actual360);
        let leg2_cf = fixed_leg(&schedule, &[1_000_000.0], &[0.03], DayCounter::Actual360);

        let swap = CrossCurrencySwap::new(
            SwapType::Payer,
            XCcyLeg {
                currency: Currency::usd(),
                cashflows: leg1_cf,
                notional: 1_000_000.0,
            },
            XCcyLeg {
                currency: Currency::usd(),
                cashflows: leg2_cf,
                notional: 1_000_000.0,
            },
            false,
            false,
            1.0,
        );

        let curve = FlatForward::new(settle, 0.04, DayCounter::Actual360);
        let result = price_xccy_swap(&swap, &curve, &curve, 1.0, settle);

        // Leg1 (5%) and Leg2 (3%) both discounted at 4%
        assert!(result.leg1_npv > result.leg2_npv, "5% leg should be worth more than 3% leg");
    }

    #[test]
    fn xccy_swap_fair_basis_spread() {
        let (swap, settle) = make_test_swap();
        let usd_curve = FlatForward::new(settle, 0.04, DayCounter::Actual360);
        let eur_curve = FlatForward::new(settle, 0.03, DayCounter::Actual360);

        let result = price_xccy_swap(&swap, &usd_curve, &eur_curve, 1.1111, settle);
        // Fair basis spread should be finite
        assert!(result.fair_basis_spread_bps.is_finite());
    }

    #[test]
    fn xccy_swap_notional_exchange() {
        let settle = Date::from_ymd(2025, Month::January, 15);
        let schedule = Schedule::from_dates(vec![
            Date::from_ymd(2025, Month::January, 15),
            Date::from_ymd(2026, Month::January, 15),
        ]);

        let usd_leg1 = fixed_leg(&schedule, &[1_000_000.0], &[0.04], DayCounter::Actual360);
        let eur_leg1 = fixed_leg(&schedule, &[900_000.0], &[0.03], DayCounter::Actual360);
        let usd_leg2 = fixed_leg(&schedule, &[1_000_000.0], &[0.04], DayCounter::Actual360);
        let eur_leg2 = fixed_leg(&schedule, &[900_000.0], &[0.03], DayCounter::Actual360);

        let swap_with_exchange = CrossCurrencySwap::new(
            SwapType::Payer,
            XCcyLeg { currency: Currency::usd(), cashflows: usd_leg1, notional: 1_000_000.0 },
            XCcyLeg { currency: Currency::eur(), cashflows: eur_leg1, notional: 900_000.0 },
            false, true, 1.1111,
        );

        let swap_without_exchange = CrossCurrencySwap::new(
            SwapType::Payer,
            XCcyLeg { currency: Currency::usd(), cashflows: usd_leg2, notional: 1_000_000.0 },
            XCcyLeg { currency: Currency::eur(), cashflows: eur_leg2, notional: 900_000.0 },
            false, false, 1.1111,
        );

        let usd_curve = FlatForward::new(settle, 0.04, DayCounter::Actual360);
        let eur_curve = FlatForward::new(settle, 0.03, DayCounter::Actual360);

        let r1 = price_xccy_swap(&swap_with_exchange, &usd_curve, &eur_curve, 1.1111, settle);
        let r2 = price_xccy_swap(&swap_without_exchange, &usd_curve, &eur_curve, 1.1111, settle);

        assert_abs_diff_eq!(r2.notional_exchange_pv, 0.0, epsilon = 1e-10);
        assert!((r1.npv - r2.npv).abs() > 1.0, "Notional exchange should change NPV");
    }
}
