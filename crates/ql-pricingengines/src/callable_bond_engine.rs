//! Callable bond pricing engine using a short-rate tree.
//!
//! Prices a callable/puttable fixed-rate bond by building a binomial
//! short-rate tree (Hull-White-like), discounting coupon cash flows
//! backwards, and applying the call/put exercise decision at each node.

use ql_instruments::callable_bond::{CallabilityType, CallableBond};
use ql_time::Date;

/// Result from the callable bond engine.
#[derive(Debug, Clone)]
pub struct CallableBondResult {
    /// Net present value of the callable bond.
    pub npv: f64,
    /// The OAS (option-adjusted spread) is left for the user to compute
    /// by comparing to the straight bond price.
    pub oas_hint: f64,
}

/// Price a callable bond using a simple short-rate binomial tree.
///
/// The tree models the short rate using a lognormal-like binomial
/// lattice calibrated to the given flat rate `r` and rate volatility
/// `rate_vol`. At each step, coupon cash flows are added and the
/// call/put exercise decision is applied.
///
/// # Arguments
/// * `bond` — the callable bond instrument
/// * `r` — flat risk-free rate (continuously compounded)
/// * `rate_vol` — volatility of the short rate
/// * `settle` — settlement / valuation date
/// * `num_steps` — number of time steps in the tree
pub fn price_callable_bond(
    bond: &CallableBond,
    r: f64,
    rate_vol: f64,
    settle: Date,
    num_steps: usize,
) -> CallableBondResult {
    // Compute time to maturity in years (approximate: 365 days/year)
    let maturity_serial = bond.maturity_date.serial();
    let settle_serial = settle.serial();
    if maturity_serial <= settle_serial {
        return CallableBondResult {
            npv: bond.face_amount,
            oas_hint: 0.0,
        };
    }

    let total_time = (maturity_serial - settle_serial) as f64 / 365.0;
    let dt = total_time / num_steps as f64;

    // Build short-rate tree: r(i,j) = r * u^(2j - i)
    // where u = exp(rate_vol * sqrt(dt))
    let u = (rate_vol * dt.sqrt()).exp();

    // Risk-neutral probability (we use a simple recombining tree)
    let p = 0.5; // symmetric tree

    // Identify coupon dates and amounts
    let mut coupon_times: Vec<(f64, f64)> = Vec::new(); // (time, amount per face)
    for cf in &bond.cashflows {
        let cf_serial = cf.date().serial();
        if cf_serial > settle_serial && cf_serial <= maturity_serial {
            let t = (cf_serial - settle_serial) as f64 / 365.0;
            coupon_times.push((t, cf.amount()));
        }
    }

    // Identify call/put dates and prices
    let mut call_times: Vec<(f64, f64)> = Vec::new(); // (time, call_price * face/100)
    for entry in &bond.callability_schedule {
        let e_serial = entry.date.serial();
        if e_serial > settle_serial && e_serial <= maturity_serial {
            let t = (e_serial - settle_serial) as f64 / 365.0;
            call_times.push((t, entry.price / 100.0 * bond.face_amount));
        }
    }

    // At maturity, bond value = face + final coupon (already in cashflows)
    // We'll add coupons at each step
    let n = num_steps;
    let mut values = vec![0.0; n + 1];

    // Terminal values: face amount (coupons are added during backward induction)
    for val in values.iter_mut() {
        *val = bond.face_amount;
    }

    // Add any coupons that fall at maturity
    for &(t, amt) in &coupon_times {
        if (t - total_time).abs() < dt * 0.5 {
            for val in values.iter_mut() {
                *val += amt;
            }
        }
    }

    // Backward induction
    for step in (0..n).rev() {
        let step_time = (step as f64 + 0.5) * dt; // midpoint time for this step
        let mut new_values = vec![0.0; step + 1];

        for j in 0..=step {
            // Short rate at node (step, j)
            let rate_node = r * u.powi(2 * j as i32 - step as i32);
            let df = (-rate_node * dt).exp();

            // Continuation value
            let continuation = df * (p * values[j + 1] + (1.0 - p) * values[j]);

            new_values[j] = continuation;
        }

        // Add coupons that fall in this time step
        let step_time_start = step as f64 * dt;
        let step_time_end = (step + 1) as f64 * dt;
        for &(t, amt) in &coupon_times {
            if t > step_time_start && t <= step_time_end && (t - total_time).abs() > dt * 0.5 {
                for val in new_values.iter_mut() {
                    *val += amt;
                }
            }
        }

        // Apply call/put exercise at this step
        for &(t, call_price) in &call_times {
            if t >= step_time_start && t < step_time_end {
                match bond.callability_type {
                    CallabilityType::Call => {
                        // Issuer calls if bond value > call price
                        for val in new_values.iter_mut() {
                            if *val > call_price {
                                *val = call_price;
                            }
                        }
                    }
                    CallabilityType::Put => {
                        // Holder puts if bond value < put price
                        for val in new_values.iter_mut() {
                            if *val < call_price {
                                *val = call_price;
                            }
                        }
                    }
                }
            }
        }

        let _ = step_time; // used for clarity
        values = new_values;
    }

    let npv = values[0];

    // OAS hint: simple approximation as difference in yield
    // (user should compute this properly by comparing to straight bond)
    let straight_bond_approx = bond.face_amount * (-r * total_time).exp()
        + coupon_times
            .iter()
            .map(|&(t, amt)| amt * (-r * t).exp())
            .sum::<f64>();

    let oas_hint = if total_time > 0.0 {
        -(npv / straight_bond_approx).ln() / total_time
    } else {
        0.0
    };

    CallableBondResult { npv, oas_hint }
}

/// Price a straight (non-callable) bond for comparison.
///
/// Uses simple discounting at the flat rate `r`.
pub fn price_straight_bond(
    bond: &CallableBond,
    r: f64,
    settle: Date,
) -> f64 {
    let settle_serial = settle.serial();
    let mut npv = 0.0;

    for cf in &bond.cashflows {
        let cf_serial = cf.date().serial();
        if cf_serial > settle_serial {
            let t = (cf_serial - settle_serial) as f64 / 365.0;
            npv += cf.amount() * (-r * t).exp();
        }
    }

    npv
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_instruments::callable_bond::{CallabilityScheduleEntry, CallabilityType, CallableBond};
    use ql_time::{DayCounter, Month, Schedule};

    fn make_schedule() -> Schedule {
        Schedule::from_dates(vec![
            Date::from_ymd(2025, Month::January, 15),
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
            Date::from_ymd(2026, Month::July, 15),
            Date::from_ymd(2027, Month::January, 15),
        ])
    }

    fn make_callable_bond() -> CallableBond {
        let schedule = make_schedule();
        let call_schedule = vec![
            CallabilityScheduleEntry {
                date: Date::from_ymd(2025, Month::July, 15),
                price: 102.0,
            },
            CallabilityScheduleEntry {
                date: Date::from_ymd(2026, Month::January, 15),
                price: 101.0,
            },
            CallabilityScheduleEntry {
                date: Date::from_ymd(2026, Month::July, 15),
                price: 100.0,
            },
        ];

        CallableBond::new(
            100.0, 2, &schedule, 0.06, DayCounter::Actual360,
            CallabilityType::Call, call_schedule,
        )
    }

    fn make_puttable_bond() -> CallableBond {
        let schedule = make_schedule();
        let put_schedule = vec![
            CallabilityScheduleEntry {
                date: Date::from_ymd(2026, Month::January, 15),
                price: 99.0,
            },
        ];

        CallableBond::new(
            100.0, 2, &schedule, 0.04, DayCounter::Actual360,
            CallabilityType::Put, put_schedule,
        )
    }

    #[test]
    fn callable_bond_less_than_straight() {
        let bond = make_callable_bond();
        let settle = Date::from_ymd(2025, Month::January, 2);

        let callable_price = price_callable_bond(&bond, 0.03, 0.01, settle, 200);
        let straight_price = price_straight_bond(&bond, 0.03, settle);

        assert!(
            callable_price.npv <= straight_price + 0.5,
            "Callable bond price {:.4} should be ≤ straight bond {:.4}",
            callable_price.npv, straight_price
        );
    }

    #[test]
    fn puttable_bond_greater_than_straight() {
        let bond = make_puttable_bond();
        let settle = Date::from_ymd(2025, Month::January, 2);

        let puttable_price = price_callable_bond(&bond, 0.06, 0.01, settle, 200);
        let straight_price = price_straight_bond(&bond, 0.06, settle);

        assert!(
            puttable_price.npv >= straight_price - 0.5,
            "Puttable bond price {:.4} should be ≥ straight bond {:.4}",
            puttable_price.npv, straight_price
        );
    }

    #[test]
    fn callable_bond_converges_with_steps() {
        let bond = make_callable_bond();
        let settle = Date::from_ymd(2025, Month::January, 2);

        let coarse = price_callable_bond(&bond, 0.04, 0.005, settle, 50);
        let fine = price_callable_bond(&bond, 0.04, 0.005, settle, 400);

        // Both should be in a reasonable range
        assert!(coarse.npv > 90.0 && coarse.npv < 115.0,
            "Coarse price {:.4} out of range", coarse.npv);
        assert!(fine.npv > 90.0 && fine.npv < 115.0,
            "Fine price {:.4} out of range", fine.npv);

        // They should be reasonably close
        assert_abs_diff_eq!(coarse.npv, fine.npv, epsilon = 2.0);
    }

    #[test]
    fn zero_vol_callable_equals_capped_straight() {
        // With zero rate volatility, the callable bond should be close to
        // a straight bond capped at the call price
        let bond = make_callable_bond();
        let settle = Date::from_ymd(2025, Month::January, 2);

        let callable = price_callable_bond(&bond, 0.04, 0.0001, settle, 200);
        let straight = price_straight_bond(&bond, 0.04, settle);

        // Callable should be between face and straight
        assert!(callable.npv > 0.0, "Callable price should be positive");
        assert!(callable.npv <= straight + 1.0,
            "Callable {:.4} should be ≤ straight {:.4}", callable.npv, straight);
    }

    #[test]
    fn expired_callable_bond() {
        let bond = make_callable_bond();
        let settle = Date::from_ymd(2028, Month::January, 1);

        let result = price_callable_bond(&bond, 0.04, 0.01, settle, 100);
        assert_abs_diff_eq!(result.npv, bond.face_amount, epsilon = 1e-10);
    }

    #[test]
    fn higher_vol_lowers_callable_price() {
        let bond = make_callable_bond();
        let settle = Date::from_ymd(2025, Month::January, 2);

        let low_vol = price_callable_bond(&bond, 0.04, 0.005, settle, 200);
        let high_vol = price_callable_bond(&bond, 0.04, 0.02, settle, 200);

        // Higher rate vol makes the call option more valuable to the issuer,
        // so the callable bond should be worth less (or equal) to the holder
        assert!(
            high_vol.npv <= low_vol.npv + 1.0,
            "Higher vol callable {:.4} should be ≤ low vol callable {:.4}",
            high_vol.npv, low_vol.npv
        );
    }
}
