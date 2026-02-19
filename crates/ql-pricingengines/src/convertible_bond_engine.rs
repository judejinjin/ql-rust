//! Convertible bond pricing engine using a binomial equity tree.
//!
//! At each node the holder decides whether to convert into shares
//! (if conversion value exceeds bond continuation value) or hold.
//! This is an American-style conversion option.

use ql_instruments::convertible_bond::ConvertibleBond;
use ql_time::Date;

/// Result from the convertible bond engine.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConvertibleBondResult {
    /// Net present value of the convertible bond.
    pub npv: f64,
    /// Delta with respect to the underlying stock price.
    pub delta: f64,
}

/// Price a convertible bond using a CRR binomial tree on the stock price.
///
/// At each node, the value is max(continuation_value + accrued_coupon, conversion_value).
/// The continuation value is discounted from the next step.
///
/// # Arguments
/// * `bond` — the convertible bond instrument
/// * `stock_price` — current stock price
/// * `r` — risk-free rate (continuously compounded)
/// * `q` — dividend yield (continuously compounded)
/// * `vol` — stock price volatility
/// * `settle` — settlement / valuation date
/// * `num_steps` — number of time steps in the tree
#[allow(clippy::too_many_arguments)]
pub fn price_convertible_bond(
    bond: &ConvertibleBond,
    stock_price: f64,
    r: f64,
    q: f64,
    vol: f64,
    settle: Date,
    num_steps: usize,
) -> ConvertibleBondResult {
    let maturity_serial = bond.maturity_date.serial();
    let settle_serial = settle.serial();
    if maturity_serial <= settle_serial {
        // Expired: worth the max of face and conversion value
        let npv = bond.face_amount.max(bond.conversion_value(stock_price));
        return ConvertibleBondResult { npv, delta: 0.0 };
    }

    let total_time = (maturity_serial - settle_serial) as f64 / 365.0;
    let dt = total_time / num_steps as f64;
    let u = (vol * dt.sqrt()).exp();
    let d = 1.0 / u;
    let growth = ((r - q) * dt).exp();
    let p = (growth - d) / (u - d);
    let df = (-r * dt).exp();

    let n = num_steps;

    // Identify coupon cash flows and their time buckets
    let mut coupon_times: Vec<(f64, f64)> = Vec::new();
    for cf in &bond.cashflows {
        let cf_serial = cf.date().serial();
        if cf_serial > settle_serial && cf_serial <= maturity_serial {
            let t = (cf_serial - settle_serial) as f64 / 365.0;
            coupon_times.push((t, cf.amount()));
        }
    }

    // Terminal values at step n
    let mut values = vec![0.0; n + 1];
    for (j, val) in values.iter_mut().enumerate() {
        let s_t = stock_price * u.powi(2 * j as i32 - n as i32);
        let bond_value = bond.face_amount; // at maturity, just face (coupons handled below)
        let conv_value = bond.conversion_ratio * s_t;
        *val = bond_value.max(conv_value);
    }

    // Add any coupon at maturity
    for &(t, amt) in &coupon_times {
        if (t - total_time).abs() < dt * 0.5 {
            for val in values.iter_mut() {
                *val += amt;
            }
        }
    }

    // Save values at step 1 for delta calculation
    let mut val_step_1 = [0.0; 2];

    // Backward induction with conversion at each node
    for step in (0..n).rev() {
        let mut new_values = vec![0.0; step + 1];

        for j in 0..=step {
            // Stock price at this node
            let s_node = stock_price * u.powi(2 * j as i32 - step as i32);

            // Continuation value (discounted expected value)
            let continuation = df * (p * values[j + 1] + (1.0 - p) * values[j]);

            // Conversion value
            let conv_value = bond.conversion_ratio * s_node;

            // Holder converts if conversion value exceeds continuation
            new_values[j] = continuation.max(conv_value);
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

        if step == 1 {
            val_step_1[0] = new_values[0];
            val_step_1[1] = new_values[1];
        }

        values = new_values;
    }

    let npv = values[0];

    // Delta from step 1
    let su = stock_price * u;
    let sd = stock_price * d;
    let delta = if n >= 1 {
        (val_step_1[1] - val_step_1[0]) / (su - sd)
    } else {
        0.0
    };

    ConvertibleBondResult { npv, delta }
}

/// Price a straight (non-convertible) bond for comparison.
pub fn price_straight_bond_for_convertible(
    bond: &ConvertibleBond,
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

    fn make_convertible() -> ConvertibleBond {
        ConvertibleBond::new(
            1000.0, 2, &make_schedule(), 0.03, DayCounter::Actual360,
            10.0, // 10 shares per $1000 face → conversion price = $100
        )
    }

    #[test]
    fn convertible_at_least_straight_bond() {
        let bond = make_convertible();
        let settle = Date::from_ymd(2025, Month::January, 2);

        let convertible_price = price_convertible_bond(
            &bond, 80.0, 0.05, 0.0, 0.25, settle, 200,
        );
        let straight_price = price_straight_bond_for_convertible(&bond, 0.05, settle);

        assert!(
            convertible_price.npv >= straight_price - 1.0,
            "Convertible {:.2} should be ≥ straight bond {:.2}",
            convertible_price.npv, straight_price
        );
    }

    #[test]
    fn convertible_at_least_conversion_value_itm() {
        let bond = make_convertible();
        let settle = Date::from_ymd(2025, Month::January, 2);
        let stock = 150.0; // conversion value = 10 × 150 = 1500 > face

        let result = price_convertible_bond(
            &bond, stock, 0.05, 0.0, 0.25, settle, 200,
        );
        let conv_value = bond.conversion_value(stock);

        assert!(
            result.npv >= conv_value - 1.0,
            "Convertible {:.2} should be ≥ conversion value {:.2}",
            result.npv, conv_value
        );
    }

    #[test]
    fn convertible_higher_vol_increases_value() {
        let bond = make_convertible();
        let settle = Date::from_ymd(2025, Month::January, 2);

        let low_vol = price_convertible_bond(
            &bond, 90.0, 0.05, 0.0, 0.15, settle, 200,
        );
        let high_vol = price_convertible_bond(
            &bond, 90.0, 0.05, 0.0, 0.40, settle, 200,
        );

        assert!(
            high_vol.npv >= low_vol.npv - 1.0,
            "Higher vol {:.2} should increase convertible value (low vol {:.2})",
            high_vol.npv, low_vol.npv
        );
    }

    #[test]
    fn convertible_deep_itm_near_conversion() {
        let bond = make_convertible();
        let settle = Date::from_ymd(2025, Month::January, 2);
        let stock = 200.0; // conv value = 2000, much > face 1000

        let result = price_convertible_bond(
            &bond, stock, 0.05, 0.0, 0.25, settle, 200,
        );
        let conv_value = bond.conversion_value(stock);

        // Deep ITM: convertible ≥ conversion value
        // (may exceed it due to coupon payments before conversion)
        assert!(
            result.npv >= conv_value * 0.95,
            "Deep ITM convertible {:.2} should be near conversion value {:.2}",
            result.npv, conv_value
        );
    }

    #[test]
    fn convertible_delta_positive() {
        let bond = make_convertible();
        let settle = Date::from_ymd(2025, Month::January, 2);

        let result = price_convertible_bond(
            &bond, 100.0, 0.05, 0.0, 0.25, settle, 200,
        );

        assert!(
            result.delta >= 0.0,
            "Convertible delta {:.4} should be non-negative",
            result.delta
        );
    }

    #[test]
    fn convertible_expired() {
        let bond = make_convertible();
        let settle = Date::from_ymd(2028, Month::January, 1);
        let stock = 120.0;

        let result = price_convertible_bond(
            &bond, stock, 0.05, 0.0, 0.25, settle, 100,
        );

        // Expired: max(face, conversion_value) = max(1000, 1200) = 1200
        assert_abs_diff_eq!(result.npv, 1200.0, epsilon = 1e-10);
    }
}
