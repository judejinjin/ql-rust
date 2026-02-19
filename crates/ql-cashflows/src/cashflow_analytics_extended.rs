//! Extended cash flow analytics — convexity, modified duration, z-spread,
//! DV01, and time-bucketed aggregation.

use ql_time::Date;
use ql_termstructures::YieldTermStructure;

use crate::cashflow::Leg;

/// Convexity of a leg.
///
/// ```text
/// C = [sum(t_i * (t_i + dt) * cf_i * df_i)] / sum(cf_i * df_i)
/// ```
///
/// where dt is a small increment for the discrete second derivative.
pub fn convexity(leg: &Leg, curve: &dyn YieldTermStructure, settle: Date) -> f64 {
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for cf in leg {
        if !cf.has_occurred(settle) {
            let t = curve.time_from_reference(cf.date());
            let df = curve.discount(cf.date());
            let pv = cf.amount() * df;
            numerator += t * t * pv;
            denominator += pv;
        }
    }

    if denominator.abs() < 1e-15 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Modified duration: Macaulay duration / (1 + y/n).
///
/// `yield_rate` is the yield, `frequency` is payments per year (e.g. 2 for semiannual).
pub fn modified_duration(
    leg: &Leg,
    curve: &dyn YieldTermStructure,
    settle: Date,
    yield_rate: f64,
    frequency: u32,
) -> f64 {
    let mac_dur = crate::cashflow_analytics::duration(leg, curve, settle);
    mac_dur / (1.0 + yield_rate / frequency as f64)
}

/// Dollar value of a basis point (DV01).
///
/// The change in NPV for a 1bp parallel shift in the yield curve.
/// Computed via central difference: DV01 ≈ [NPV(y-1bp) - NPV(y+1bp)] / 2.
pub fn dv01(leg: &Leg, curve: &dyn YieldTermStructure, settle: Date) -> f64 {
    // Use BPS as a proxy (already computes the 1bp sensitivity)
    crate::cashflow_analytics::bps(leg, curve, settle)
}

/// Z-spread: the constant spread z added to each zero rate such that
/// the discounted cash flows equal the target price.
///
/// Solves: sum(cf_i * exp(-(r_i + z) * t_i)) = target_price
pub fn z_spread(
    leg: &Leg,
    curve: &dyn YieldTermStructure,
    settle: Date,
    target_price: f64,
    accuracy: f64,
    max_iterations: usize,
) -> Result<f64, &'static str> {
    // Bisection on z
    let mut z_lo: f64 = -0.10;
    let mut z_hi: f64 = 0.50;

    let price_at_z = |z: f64| -> f64 {
        let mut total = 0.0;
        for cf in leg {
            if !cf.has_occurred(settle) {
                let t = curve.time_from_reference(cf.date());
                let df = curve.discount(cf.date());
                // Apply additional z-spread discount
                let z_df = df * (-z * t).exp();
                total += cf.amount() * z_df;
            }
        }
        total
    };

    let mut f_lo = price_at_z(z_lo) - target_price;
    let f_hi = price_at_z(z_hi) - target_price;

    if f_lo * f_hi > 0.0 {
        return Err("z-spread: no bracket found");
    }

    for _ in 0..max_iterations {
        let z_mid = 0.5 * (z_lo + z_hi);
        let f_mid = price_at_z(z_mid) - target_price;

        if f_mid.abs() < accuracy {
            return Ok(z_mid);
        }

        if f_lo * f_mid < 0.0 {
            z_hi = z_mid;
        } else {
            z_lo = z_mid;
            f_lo = f_mid;
        }
    }

    Ok(0.5 * (z_lo + z_hi))
}

/// ATM rate: the fixed rate that makes the NPV of the leg equal to zero
/// (or a target NPV), given a discount curve.
///
/// `atm_rate = NPV / annuity` where annuity = sum(τ_i * df_i).
pub fn atm_rate(leg: &Leg, curve: &dyn YieldTermStructure, settle: Date) -> f64 {
    let annuity = crate::cashflow_analytics::bps(leg, curve, settle) / 0.0001;
    if annuity.abs() < 1e-15 {
        return 0.0;
    }
    let pv = crate::cashflow_analytics::npv(leg, curve, settle);
    pv / annuity
}

/// Time-bucketed cash flow aggregation.
///
/// Groups cash flows into time buckets (e.g. 0-1Y, 1-2Y, etc.)
/// and returns the PV contribution of each bucket.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TimeBucket {
    pub start: f64,
    pub end: f64,
    pub pv: f64,
    pub cash_flow_count: usize,
}

pub fn time_bucketed_cashflows(
    leg: &Leg,
    curve: &dyn YieldTermStructure,
    settle: Date,
    bucket_boundaries: &[f64],
) -> Vec<TimeBucket> {
    let n = bucket_boundaries.len();
    if n < 2 {
        return vec![];
    }

    let mut buckets: Vec<TimeBucket> = (0..n - 1)
        .map(|i| TimeBucket {
            start: bucket_boundaries[i],
            end: bucket_boundaries[i + 1],
            pv: 0.0,
            cash_flow_count: 0,
        })
        .collect();

    for cf in leg {
        if !cf.has_occurred(settle) {
            let t = curve.time_from_reference(cf.date());
            let df = curve.discount(cf.date());
            let pv = cf.amount() * df;

            // Find the bucket
            for bucket in &mut buckets {
                if t >= bucket.start && t < bucket.end {
                    bucket.pv += pv;
                    bucket.cash_flow_count += 1;
                    break;
                }
            }
        }
    }

    buckets
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_time::{DayCounter, Month, Schedule};
    use ql_termstructures::FlatForward;
    use crate::leg::fixed_leg;

    fn make_test_leg() -> (Leg, Date) {
        let schedule = Schedule::from_dates(vec![
            Date::from_ymd(2025, Month::January, 15),
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2026, Month::January, 15),
            Date::from_ymd(2026, Month::July, 15),
            Date::from_ymd(2027, Month::January, 15),
        ]);
        let leg = fixed_leg(&schedule, &[1_000_000.0], &[0.05], DayCounter::Actual360);
        let ref_date = Date::from_ymd(2025, Month::January, 2);
        (leg, ref_date)
    }

    #[test]
    fn convexity_positive() {
        let (leg, ref_date) = make_test_leg();
        let curve = FlatForward::new(ref_date, 0.04, DayCounter::Actual360);
        let c = convexity(&leg, &curve, ref_date);
        assert!(c > 0.0, "Convexity should be positive: {c}");
        // For a 2Y leg, convexity should be around 1-4
        assert!(c > 0.1 && c < 10.0, "Convexity = {c}");
    }

    #[test]
    fn modified_duration_less_than_macaulay() {
        let (leg, ref_date) = make_test_leg();
        let curve = FlatForward::new(ref_date, 0.04, DayCounter::Actual360);
        let mac = crate::cashflow_analytics::duration(&leg, &curve, ref_date);
        let mod_dur = modified_duration(&leg, &curve, ref_date, 0.04, 2);
        assert!(mod_dur < mac, "Modified duration {mod_dur} should be < Macaulay {mac}");
    }

    #[test]
    fn dv01_positive() {
        let (leg, ref_date) = make_test_leg();
        let curve = FlatForward::new(ref_date, 0.04, DayCounter::Actual360);
        let d = dv01(&leg, &curve, ref_date);
        assert!(d > 0.0, "DV01 should be positive: {d}");
    }

    #[test]
    fn z_spread_at_par() {
        let (leg, ref_date) = make_test_leg();
        let curve = FlatForward::new(ref_date, 0.04, DayCounter::Actual360);
        let target = crate::cashflow_analytics::npv(&leg, &curve, ref_date);
        // z-spread should be ~0 when price = NPV
        let z = z_spread(&leg, &curve, ref_date, target, 1e-10, 200).unwrap();
        assert_abs_diff_eq!(z, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn z_spread_positive_for_cheap_bond() {
        let (leg, ref_date) = make_test_leg();
        let curve = FlatForward::new(ref_date, 0.04, DayCounter::Actual360);
        let npv = crate::cashflow_analytics::npv(&leg, &curve, ref_date);
        // Bond trading at 99% of fair value → positive z-spread
        let z = z_spread(&leg, &curve, ref_date, npv * 0.99, 1e-10, 200).unwrap();
        assert!(z > 0.0, "z-spread should be positive for cheap bond: {z}");
    }

    #[test]
    fn atm_rate_close_to_coupon() {
        let (leg, ref_date) = make_test_leg();
        let curve = FlatForward::new(ref_date, 0.05, DayCounter::Actual360);
        let atm = atm_rate(&leg, &curve, ref_date);
        // ATM rate should be close to the coupon rate of 5%
        assert_abs_diff_eq!(atm, 0.05, epsilon = 0.001);
    }

    #[test]
    fn time_buckets() {
        let (leg, ref_date) = make_test_leg();
        let curve = FlatForward::new(ref_date, 0.04, DayCounter::Actual360);
        let buckets = time_bucketed_cashflows(
            &leg,
            &curve,
            ref_date,
            &[0.0, 1.0, 2.0, 3.0],
        );
        assert_eq!(buckets.len(), 3);
        let total_pv: f64 = buckets.iter().map(|b| b.pv).sum();
        let npv = crate::cashflow_analytics::npv(&leg, &curve, ref_date);
        assert_abs_diff_eq!(total_pv, npv, epsilon = 1.0);
    }
}
