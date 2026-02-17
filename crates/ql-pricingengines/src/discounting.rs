//! Discounting pricing engines for swaps and bonds.
//!
//! These engines compute NPV by discounting future cash flows using a
//! yield term structure.

use ql_cashflows::npv as leg_npv;
use ql_instruments::vanilla_swap::{SwapType, VanillaSwap};
use ql_instruments::bond::FixedRateBond;
use ql_termstructures::YieldTermStructure;
use ql_time::Date;

// ===========================================================================
// DiscountingSwapEngine
// ===========================================================================

/// Results from the discounting swap engine.
#[derive(Debug, Clone)]
pub struct SwapResults {
    /// Net present value (positive = in-the-money for the payer).
    pub npv: f64,
    /// Fixed leg NPV.
    pub fixed_leg_npv: f64,
    /// Floating leg NPV.
    pub floating_leg_npv: f64,
    /// Fair rate: the fixed rate that makes NPV = 0.
    pub fair_rate: f64,
}

/// Price a vanilla swap by discounting both legs.
///
/// The floating leg's IBOR coupons are assumed to already have their rates
/// set (via `with_rate()` or from fixings). For forward-looking pricing,
/// populate the floating leg coupon rates from a forecast curve first.
pub fn price_swap(
    swap: &VanillaSwap,
    curve: &dyn YieldTermStructure,
    settle: Date,
) -> SwapResults {
    let fixed_npv = leg_npv(&swap.fixed_leg, curve, settle);
    let floating_npv = leg_npv(&swap.floating_leg, curve, settle);

    let sign = match swap.swap_type {
        SwapType::Payer => -1.0,  // Pay fixed, receive floating
        SwapType::Receiver => 1.0, // Receive fixed, pay floating
    };

    let npv = sign * fixed_npv - sign * floating_npv;

    // Fair rate: fixed rate that makes NPV = 0
    // If we change the fixed rate by dr, fixed_leg_npv changes by dr/original_rate * fixed_npv
    // So fair_rate = floating_npv / (fixed_npv / original_rate)
    let fair_rate = if fixed_npv.abs() > 1e-15 && swap.fixed_rate.abs() > 1e-15 {
        swap.fixed_rate * floating_npv / fixed_npv
    } else {
        0.0
    };

    SwapResults {
        npv,
        fixed_leg_npv: fixed_npv,
        floating_leg_npv: floating_npv,
        fair_rate,
    }
}

// ===========================================================================
// DiscountingBondEngine
// ===========================================================================

/// Results from the discounting bond engine.
#[derive(Debug, Clone)]
pub struct BondResults {
    /// Net present value (dirty price × face / 100).
    pub npv: f64,
    /// Clean price (per 100 face).
    pub clean_price: f64,
    /// Dirty price (per 100 face).
    pub dirty_price: f64,
    /// Accrued interest.
    pub accrued_interest: f64,
}

/// Price a fixed-rate bond by discounting all cash flows.
pub fn price_bond(
    bond: &FixedRateBond,
    curve: &dyn YieldTermStructure,
    settle: Date,
) -> BondResults {
    let npv = leg_npv(&bond.cashflows, curve, settle);

    // Dirty price per 100 face
    let dirty_price = if bond.face_amount.abs() > 1e-15 {
        npv / bond.face_amount * 100.0
    } else {
        0.0
    };

    // Accrued interest per 100 face
    let accrued = ql_cashflows::accrued_amount(&bond.cashflows, settle);
    let accrued_per_100 = if bond.face_amount.abs() > 1e-15 {
        accrued / bond.face_amount * 100.0
    } else {
        0.0
    };

    let clean_price = dirty_price - accrued_per_100;

    BondResults {
        npv,
        clean_price,
        dirty_price,
        accrued_interest: accrued_per_100,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_cashflows::{fixed_leg, ibor_leg, IborCoupon, Leg};
    use ql_indexes::IborIndex;
    use ql_termstructures::FlatForward;
    use ql_time::{Date, DayCounter, Month, Schedule};

    #[test]
    fn bond_price_at_par() {
        // A bond with coupon rate = discount rate should price near par
        let schedule = Schedule::from_dates(vec![
            Date::from_ymd(2025, Month::January, 15),
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
            Date::from_ymd(2026, Month::July, 15),
            Date::from_ymd(2027, Month::January, 15),
        ]);
        let bond = FixedRateBond::new(100.0, 2, &schedule, 0.05, DayCounter::Actual360);
        let ref_date = Date::from_ymd(2025, Month::January, 2);
        let curve = FlatForward::new(ref_date, 0.05, DayCounter::Actual360);

        let result = price_bond(&bond, &curve, ref_date);
        // Should be close to par (100)
        assert_abs_diff_eq!(result.dirty_price, 100.0, epsilon = 1.0);
    }

    #[test]
    fn bond_higher_rate_lower_price() {
        let schedule = Schedule::from_dates(vec![
            Date::from_ymd(2025, Month::January, 15),
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
        ]);
        let bond = FixedRateBond::new(100.0, 2, &schedule, 0.05, DayCounter::Actual360);
        let ref_date = Date::from_ymd(2025, Month::January, 2);

        let curve_low = FlatForward::new(ref_date, 0.03, DayCounter::Actual360);
        let curve_high = FlatForward::new(ref_date, 0.08, DayCounter::Actual360);

        let price_low = price_bond(&bond, &curve_low, ref_date);
        let price_high = price_bond(&bond, &curve_high, ref_date);

        assert!(
            price_low.dirty_price > price_high.dirty_price,
            "Higher yield should give lower price"
        );
    }

    #[test]
    fn swap_payer_npv_positive_when_floating_exceeds_fixed() {
        // Fixed rate 3% vs floating effectively at 5% → payer benefits
        let ref_date = Date::from_ymd(2025, Month::January, 2);
        let schedule = Schedule::from_dates(vec![
            Date::from_ymd(2025, Month::January, 15),
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
        ]);

        let fixed = fixed_leg(&schedule, &[1_000_000.0], &[0.03], DayCounter::Actual360);

        let index = IborIndex::euribor_6m();
        let mut floating = ibor_leg(
            &schedule,
            &[1_000_000.0],
            &index,
            &[0.0],
            DayCounter::Actual360,
        );
        // Set floating rates to 5% (higher than fixed 3%)
        for cf in floating.iter_mut() {
            if let Some(ibor) = cf.as_any().downcast_ref::<IborCoupon>() {
                // We can't mutate through as_any, so we rebuild
                let _ = ibor; // just checking type
            }
        }
        // Since we can't easily mutate trait objects, rebuild the floating leg with cached rates
        let mut floating_with_rates: Leg = Vec::new();
        let dates = schedule.dates();
        for i in 0..(dates.len() - 1) {
            let start = dates[i];
            let end = dates[i + 1];
            let fixing_date = index.fixing_calendar.advance_business_days(start, -(index.fixing_days as i32));
            let coupon = ql_cashflows::IborCoupon::new(
                end, 1_000_000.0, start, end, fixing_date,
                index.clone(), 0.0, DayCounter::Actual360,
            ).with_rate(0.05);
            floating_with_rates.push(Box::new(coupon));
        }

        let swap = VanillaSwap::new(
            SwapType::Payer,
            1_000_000.0,
            fixed,
            floating_with_rates,
            0.03,
            0.0,
        );

        let curve = FlatForward::new(ref_date, 0.04, DayCounter::Actual360);
        let result = price_swap(&swap, &curve, ref_date);

        // Payer: pay fixed 3%, receive floating 5% → positive NPV
        assert!(result.npv > 0.0, "Payer swap NPV should be positive, got {}", result.npv);
    }

    #[test]
    fn bond_accrued_interest() {
        let schedule = Schedule::from_dates(vec![
            Date::from_ymd(2025, Month::January, 15),
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
        ]);
        let bond = FixedRateBond::new(100.0, 2, &schedule, 0.05, DayCounter::Actual360);

        // Mid-period: should have some accrued interest
        let mid = Date::from_ymd(2025, Month::April, 15);
        let curve = FlatForward::new(Date::from_ymd(2025, Month::January, 2), 0.04, DayCounter::Actual360);
        let result = price_bond(&bond, &curve, mid);
        assert!(result.accrued_interest > 0.0, "Should have accrued interest");
        assert!(result.dirty_price > result.clean_price, "Dirty > clean");
    }
}
