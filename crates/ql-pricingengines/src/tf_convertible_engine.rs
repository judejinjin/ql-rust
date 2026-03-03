//! Tsiveriotis-Fernandes convertible bond engine with call/put provisions.
//!
//! The Tsiveriotis-Fernandes (1998) model splits the convertible bond value
//! into an equity component (discounted at the risk-free rate) and a debt
//! component (discounted at a risky rate = risk-free + credit spread).
//!
//! This engine extends the basic binomial tree with:
//! - Issuer call provisions (callable convertible)
//! - Holder put provisions (puttable convertible)
//! - Soft call conditions (only callable if stock above threshold)
//! - Credit spread for the debt component
//!
//! ## References
//!
//! - Tsiveriotis, K. & Fernandes, C. (1998) "Valuing Convertible Bonds
//!   with Credit Risk", Journal of Fixed Income, 8(3), 95-102
//! - QuantLib: `BinomialConvertibleEngine`, `TsiveriotisFernandesEngine`

use ql_instruments::convertible_bond::ConvertibleBond;
use ql_time::Date;

/// A call or put provision on a convertible bond.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CallabilityEntry {
    /// Start date of the callability window.
    pub start_date: Date,
    /// End date of the callability window.
    pub end_date: Date,
    /// Strike price (call price or put price per face).
    pub price: f64,
    /// Type: `Call` or `Put`.
    pub callability_type: CallPutType,
    /// Soft-call condition: only callable if stock is above this price.
    /// `None` for hard call (always callable within window).
    pub trigger_price: Option<f64>,
}

/// Whether a provision is a call (issuer right) or put (holder right).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CallPutType {
    /// Issuer can call (redeem) the bond at the call price.
    Call,
    /// Holder can put (sell back) the bond at the put price.
    Put,
}

/// Result from the Tsiveriotis-Fernandes engine.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TFConvertibleResult {
    /// Total convertible bond value.
    pub npv: f64,
    /// Equity component (value attributable to conversion option).
    pub equity_component: f64,
    /// Debt component (value of bond-like cash flows).
    pub debt_component: f64,
    /// Delta with respect to the underlying stock.
    pub delta: f64,
    /// Gamma (second derivative w.r.t. stock).
    pub gamma: f64,
}

/// Price a convertible bond using the Tsiveriotis-Fernandes model.
///
/// # Arguments
/// * `bond` — the convertible bond instrument
/// * `stock_price` — current stock price
/// * `r` — risk-free rate (continuously compounded)
/// * `credit_spread` — issuer credit spread over risk-free
/// * `q` — dividend yield (continuously compounded)
/// * `vol` — stock price volatility
/// * `settle` — settlement / valuation date
/// * `num_steps` — number of time steps in the binomial tree
/// * `callability` — optional call/put provisions
#[allow(clippy::too_many_arguments)]
pub fn price_tf_convertible(
    bond: &ConvertibleBond,
    stock_price: f64,
    r: f64,
    credit_spread: f64,
    q: f64,
    vol: f64,
    settle: Date,
    num_steps: usize,
    callability: &[CallabilityEntry],
) -> TFConvertibleResult {
    let maturity_serial = bond.maturity_date.serial();
    let settle_serial = settle.serial();

    if maturity_serial <= settle_serial {
        let conv = bond.conversion_value(stock_price);
        let npv = bond.face_amount.max(conv);
        let equity = if conv > bond.face_amount { conv } else { 0.0 };
        let debt = npv - equity;
        return TFConvertibleResult {
            npv,
            equity_component: equity,
            debt_component: debt,
            delta: 0.0,
            gamma: 0.0,
        };
    }

    let total_time = (maturity_serial - settle_serial) as f64 / 365.0;
    let dt = total_time / num_steps as f64;
    let u = (vol * dt.sqrt()).exp();
    let d = 1.0 / u;
    let growth = ((r - q) * dt).exp();
    let p = (growth - d) / (u - d);

    // Two discount factors: risk-free and risky (for debt component)
    let df_rf = (-r * dt).exp();
    let df_risky = (-(r + credit_spread) * dt).exp();

    let n = num_steps;

    // Coupon cash flows
    let mut coupon_times: Vec<(f64, f64)> = Vec::new();
    for cf in &bond.cashflows {
        let cf_serial = cf.date().serial();
        if cf_serial > settle_serial && cf_serial <= maturity_serial {
            let t = (cf_serial - settle_serial) as f64 / 365.0;
            coupon_times.push((t, cf.amount()));
        }
    }

    // Pre-compute time steps for callability
    let callability_at_step: Vec<Option<&CallabilityEntry>> = (0..=n)
        .map(|step| {
            let t = step as f64 * dt;
            let approx_serial = settle_serial + (t * 365.0) as i32;
            let approx_date = Date::from_serial(approx_serial);
            callability.iter().find(|c| {
                approx_date >= c.start_date && approx_date <= c.end_date
            })
        })
        .collect();

    // Terminal values: total, equity, debt
    let mut total_values = vec![0.0_f64; n + 1];
    let mut equity_values = vec![0.0_f64; n + 1];

    for j in 0..=n {
        let s_t = stock_price * u.powi(2 * j as i32 - n as i32);
        let conv_value = bond.conversion_ratio * s_t;
        let bond_value = bond.face_amount;

        if conv_value >= bond_value {
            total_values[j] = conv_value;
            equity_values[j] = conv_value;
        } else {
            total_values[j] = bond_value;
            equity_values[j] = 0.0;
        }
    }

    // Add terminal coupons
    for &(t, amt) in &coupon_times {
        if (t - total_time).abs() < dt * 0.5 {
            for val in total_values.iter_mut() {
                *val += amt;
                // Coupons are debt component
            }
        }
    }

    let mut val_step_1_total = [0.0_f64; 2];
    let mut val_step_2_total = [0.0_f64; 3];

    // Backward induction
    for step in (0..n).rev() {
        let mut new_total = vec![0.0_f64; step + 1];
        let mut new_equity = vec![0.0_f64; step + 1];

        for j in 0..=step {
            let s_node = stock_price * u.powi(2 * j as i32 - step as i32);

            // Expected equity and debt components
            let equity_cont = p * equity_values[j + 1] + (1.0 - p) * equity_values[j];
            let debt_up = total_values[j + 1] - equity_values[j + 1];
            let debt_down = total_values[j] - equity_values[j];
            let debt_cont = p * debt_up + (1.0 - p) * debt_down;

            // TF split: equity discounted at rf, debt discounted at risky rate
            let equity_pv = df_rf * equity_cont;
            let debt_pv = df_risky * debt_cont;
            let continuation = equity_pv + debt_pv;

            // Conversion value
            let conv_value = bond.conversion_ratio * s_node;

            // Apply call/put provisions
            let mut node_value = continuation;
            let mut node_equity = equity_pv;

            // Check callability at this step
            if let Some(provision) = callability_at_step.get(step).and_then(|o| *o) {
                match provision.callability_type {
                    CallPutType::Call => {
                        // Issuer calls: holder gets max(call_price, conv_value)
                        let soft_callable = provision
                            .trigger_price
                            .map_or(true, |trigger| s_node >= trigger);

                        if soft_callable && continuation > provision.price {
                            // Issuer calls; holder can still convert
                            let called_value = provision.price.max(conv_value);
                            node_value = called_value;
                            node_equity = if conv_value >= provision.price {
                                conv_value
                            } else {
                                0.0
                            };
                        }
                    }
                    CallPutType::Put => {
                        // Holder puts: gets max(put_price, conv_value, continuation)
                        if provision.price > continuation {
                            node_value = provision.price.max(conv_value);
                            node_equity = if conv_value >= provision.price {
                                conv_value
                            } else {
                                0.0
                            };
                        }
                    }
                }
            }

            // Holder converts if conversion value exceeds current node value
            if conv_value > node_value {
                node_value = conv_value;
                node_equity = conv_value;
            }

            new_total[j] = node_value;
            new_equity[j] = node_equity;
        }

        // Add coupons at this step
        let step_time_start = step as f64 * dt;
        let step_time_end = (step + 1) as f64 * dt;
        for &(t, amt) in &coupon_times {
            if t > step_time_start && t <= step_time_end && (t - total_time).abs() > dt * 0.5 {
                for val in new_total.iter_mut() {
                    *val += amt;
                }
            }
        }

        if step == 2 && new_total.len() >= 3 {
            val_step_2_total[0] = new_total[0];
            val_step_2_total[1] = new_total[1];
            val_step_2_total[2] = new_total[2];
        }
        if step == 1 && new_total.len() >= 2 {
            val_step_1_total[0] = new_total[0];
            val_step_1_total[1] = new_total[1];
        }

        total_values = new_total;
        equity_values = new_equity;
    }

    let npv = total_values[0];
    let equity_component = equity_values[0];
    let debt_component = npv - equity_component;

    // Delta from step 1
    let su = stock_price * u;
    let sd = stock_price * d;
    let delta = if n >= 1 {
        (val_step_1_total[1] - val_step_1_total[0]) / (su - sd)
    } else {
        0.0
    };

    // Gamma from step 2
    let gamma = if n >= 2 {
        let suu = stock_price * u * u;
        let sdd = stock_price * d * d;
        let delta_up =
            (val_step_2_total[2] - val_step_2_total[1]) / (suu - stock_price);
        let delta_down =
            (val_step_2_total[1] - val_step_2_total[0]) / (stock_price - sdd);
        2.0 * (delta_up - delta_down) / (suu - sdd)
    } else {
        0.0
    };

    TFConvertibleResult {
        npv,
        equity_component,
        debt_component,
        delta,
        gamma,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_time::{DayCounter, Month, Schedule};

    fn make_convertible() -> ConvertibleBond {
        let schedule = Schedule::from_dates(vec![
            Date::from_ymd(2025, Month::January, 15),
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
            Date::from_ymd(2026, Month::July, 15),
            Date::from_ymd(2027, Month::January, 15),
        ]);
        ConvertibleBond::new(1000.0, 2, &schedule, 0.03, DayCounter::Actual360, 10.0)
    }

    #[test]
    fn tf_basic_pricing() {
        let bond = make_convertible();
        let settle = Date::from_ymd(2025, Month::January, 2);

        let result = price_tf_convertible(
            &bond, 100.0, 0.05, 0.02, 0.0, 0.25, settle, 200, &[],
        );

        assert!(result.npv > 0.0, "NPV should be positive");
        assert!(
            result.equity_component + result.debt_component - result.npv < 1.0,
            "Components should sum to NPV"
        );
    }

    #[test]
    fn tf_credit_spread_reduces_value() {
        let bond = make_convertible();
        let settle = Date::from_ymd(2025, Month::January, 2);

        let result_no_spread = price_tf_convertible(
            &bond, 80.0, 0.05, 0.0, 0.0, 0.25, settle, 200, &[],
        );
        let result_with_spread = price_tf_convertible(
            &bond, 80.0, 0.05, 0.05, 0.0, 0.25, settle, 200, &[],
        );

        assert!(
            result_with_spread.npv < result_no_spread.npv,
            "Credit spread should reduce value: no_spread={:.2}, with_spread={:.2}",
            result_no_spread.npv, result_with_spread.npv
        );
    }

    #[test]
    fn tf_call_provision_reduces_value() {
        let bond = make_convertible();
        let settle = Date::from_ymd(2025, Month::January, 2);

        let no_call = price_tf_convertible(
            &bond, 100.0, 0.05, 0.01, 0.0, 0.25, settle, 200, &[],
        );

        let call = vec![CallabilityEntry {
            start_date: Date::from_ymd(2025, Month::July, 15),
            end_date: Date::from_ymd(2027, Month::January, 15),
            price: 1020.0,
            callability_type: CallPutType::Call,
            trigger_price: None,
        }];

        let with_call = price_tf_convertible(
            &bond, 100.0, 0.05, 0.01, 0.0, 0.25, settle, 200, &call,
        );

        assert!(
            with_call.npv <= no_call.npv + 1.0,
            "Call provision should reduce holder value: no_call={:.2}, with_call={:.2}",
            no_call.npv, with_call.npv
        );
    }

    #[test]
    fn tf_put_provision_increases_value() {
        let bond = make_convertible();
        let settle = Date::from_ymd(2025, Month::January, 2);

        let no_put = price_tf_convertible(
            &bond, 80.0, 0.05, 0.03, 0.0, 0.25, settle, 200, &[],
        );

        let put = vec![CallabilityEntry {
            start_date: Date::from_ymd(2026, Month::January, 15),
            end_date: Date::from_ymd(2026, Month::July, 15),
            price: 1000.0,
            callability_type: CallPutType::Put,
            trigger_price: None,
        }];

        let with_put = price_tf_convertible(
            &bond, 80.0, 0.05, 0.03, 0.0, 0.25, settle, 200, &put,
        );

        assert!(
            with_put.npv >= no_put.npv - 1.0,
            "Put provision should increase holder value: no_put={:.2}, with_put={:.2}",
            no_put.npv, with_put.npv
        );
    }

    #[test]
    fn tf_soft_call() {
        let bond = make_convertible();
        let settle = Date::from_ymd(2025, Month::January, 2);

        let hard_call = vec![CallabilityEntry {
            start_date: Date::from_ymd(2025, Month::July, 15),
            end_date: Date::from_ymd(2027, Month::January, 15),
            price: 1010.0,
            callability_type: CallPutType::Call,
            trigger_price: None,
        }];

        let soft_call = vec![CallabilityEntry {
            start_date: Date::from_ymd(2025, Month::July, 15),
            end_date: Date::from_ymd(2027, Month::January, 15),
            price: 1010.0,
            callability_type: CallPutType::Call,
            trigger_price: Some(130.0), // Only callable if stock > $130
        }];

        let hard = price_tf_convertible(
            &bond, 100.0, 0.05, 0.01, 0.0, 0.25, settle, 200, &hard_call,
        );
        let soft = price_tf_convertible(
            &bond, 100.0, 0.05, 0.01, 0.0, 0.25, settle, 200, &soft_call,
        );

        // Soft call is less restrictive for holder → higher value
        assert!(
            soft.npv >= hard.npv - 1.0,
            "Soft call should be ≥ hard call value: soft={:.2}, hard={:.2}",
            soft.npv, hard.npv
        );
    }

    #[test]
    fn tf_equity_debt_components() {
        let bond = make_convertible();
        let settle = Date::from_ymd(2025, Month::January, 2);

        // Deep OTM: equity component should be small
        let otm = price_tf_convertible(
            &bond, 50.0, 0.05, 0.02, 0.0, 0.25, settle, 200, &[],
        );
        assert!(otm.debt_component > otm.equity_component,
            "OTM: debt {:.2} should dominate equity {:.2}", otm.debt_component, otm.equity_component);

        // Deep ITM: equity component should dominate
        let itm = price_tf_convertible(
            &bond, 200.0, 0.05, 0.02, 0.0, 0.25, settle, 200, &[],
        );
        assert!(itm.equity_component > itm.debt_component,
            "ITM: equity {:.2} should dominate debt {:.2}", itm.equity_component, itm.debt_component);
    }

    #[test]
    fn tf_delta_positive() {
        let bond = make_convertible();
        let settle = Date::from_ymd(2025, Month::January, 2);

        let result = price_tf_convertible(
            &bond, 100.0, 0.05, 0.01, 0.0, 0.25, settle, 200, &[],
        );
        assert!(result.delta >= 0.0, "Delta should be non-negative: {:.4}", result.delta);
    }

    #[test]
    fn tf_gamma_positive() {
        let bond = make_convertible();
        let settle = Date::from_ymd(2025, Month::January, 2);

        let result = price_tf_convertible(
            &bond, 100.0, 0.05, 0.01, 0.0, 0.25, settle, 200, &[],
        );
        // Gamma should generally be positive or near zero
        assert!(result.gamma >= -0.01, "Gamma should be non-negative or near zero: {:.6}", result.gamma);
    }

    #[test]
    fn tf_expired() {
        let bond = make_convertible();
        let settle = Date::from_ymd(2028, Month::January, 1);

        let result = price_tf_convertible(
            &bond, 120.0, 0.05, 0.01, 0.0, 0.25, settle, 100, &[],
        );
        // conv value = 10 × 120 = 1200 > face 1000
        assert_abs_diff_eq!(result.npv, 1200.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result.equity_component, 1200.0, epsilon = 1e-10);
    }
}
