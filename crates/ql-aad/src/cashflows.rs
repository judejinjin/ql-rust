//! AD-aware cashflow discounting — bond/swap NPV with curve sensitivities.
//!
//! Provides generic NPV computation over `T: Number` using
//! [`DiscountCurveAD`]. When called with `AReal`, a single adjoint pass
//! yields exact DV01 / key-rate duration / curve sensitivities for every
//! pillar — no bump-and-reprice required.
//!
//! # Example
//!
//! ```
//! use ql_aad::cashflows::{Cashflow, npv, par_rate};
//! use ql_aad::curves::DiscountCurveAD;
//!
//! let cfs = vec![
//!     Cashflow { time: 1.0, amount: 5.0 },
//!     Cashflow { time: 2.0, amount: 105.0 },
//! ];
//! let times = vec![1.0, 2.0, 5.0];
//! let rates = vec![0.03, 0.035, 0.04];
//! let curve = DiscountCurveAD::<f64>::from_zero_rates(&times, &rates);
//!
//! let value: f64 = npv(&cfs, &curve);
//! assert!(value > 0.0);
//! ```

use crate::curves::DiscountCurveAD;
use crate::number::Number;

// ===========================================================================
// Core types
// ===========================================================================

/// A single deterministic cashflow at a fixed time.
#[derive(Debug, Clone, Copy)]
pub struct Cashflow {
    /// Time in years from valuation date.
    pub time: f64,
    /// Cashflow amount (positive = receive).
    pub amount: f64,
}

// ===========================================================================
// NPV computation
// ===========================================================================

/// Compute the net present value of a series of cashflows.
///
/// `NPV = Σᵢ amount_i × DF(t_i)`
///
/// When `T = AReal`, the adjoint pass gives ∂NPV/∂(rate_pillar_j) for all j.
pub fn npv<T: Number>(cashflows: &[Cashflow], curve: &DiscountCurveAD<T>) -> T {
    let mut total = T::zero();
    for cf in cashflows {
        total += T::from_f64(cf.amount) * curve.discount(cf.time);
    }
    total
}

/// PV01: sensitivity of NPV to a parallel 1 bp shift in all rates.
///
/// Computed analytically as ∂NPV/∂r ≈ -Σᵢ t_i × amount_i × DF(t_i) × 0.0001.
///
/// For exact per-pillar PV01, use `AReal` and extract `adjoint[pillar_idx]`.
pub fn pv01<T: Number>(cashflows: &[Cashflow], curve: &DiscountCurveAD<T>) -> T {
    let mut total = T::zero();
    for cf in cashflows {
        let df = curve.discount(cf.time);
        // ∂NPV/∂r_parallel = Σ -t_i * cf_i * DF(t_i)
        total += T::from_f64(-cf.time * cf.amount * 0.0001) * df;
    }
    total
}

/// Macaulay duration: D = (Σ t_i × CF_i × DF_i) / NPV.
pub fn macaulay_duration<T: Number>(cashflows: &[Cashflow], curve: &DiscountCurveAD<T>) -> T {
    let mut weighted = T::zero();
    let mut total = T::zero();
    for cf in cashflows {
        let pv = T::from_f64(cf.amount) * curve.discount(cf.time);
        weighted += T::from_f64(cf.time) * pv;
        total += pv;
    }
    weighted / total
}

/// Modified duration: D_mod = D_mac / (1 + y), where y is the yield / compounding freq.
/// For continuous compounding, D_mod = D_mac.
pub fn modified_duration<T: Number>(cashflows: &[Cashflow], curve: &DiscountCurveAD<T>) -> T {
    // With continuous compounding, modified = Macaulay
    macaulay_duration(cashflows, curve)
}

// ===========================================================================
// Par rate / swap rate
// ===========================================================================

/// Par coupon rate for a fixed-rate bond: the coupon rate such that NPV = face.
///
/// `par = (face - face × DF(T_n)) / (face × Σ DF(t_i))`
///
/// where `t_i` are coupon dates and `T_n` is the final maturity.
pub fn par_rate<T: Number>(
    coupon_times: &[f64],
    curve: &DiscountCurveAD<T>,
    face: f64,
) -> T {
    // Annuity = Σ DF(t_i)
    let mut annuity = T::zero();
    for &t in coupon_times {
        annuity += curve.discount(t);
    }
    let t_n = *coupon_times.last().expect("need at least one coupon time");
    let df_n = curve.discount(t_n);

    // par = (1 - DF(T_n)) / annuity  (for unit face)
    (T::from_f64(face) - T::from_f64(face) * df_n) / (T::from_f64(face) * annuity)
}

/// Build a fixed-rate bond cashflow schedule.
///
/// Returns `(n_coupons + 1)` cashflows: annual coupons + final principal.
pub fn fixed_rate_bond_cashflows(
    coupon_rate: f64,
    face: f64,
    maturity_years: usize,
) -> Vec<Cashflow> {
    let mut cfs = Vec::with_capacity(maturity_years + 1);
    for y in 1..=maturity_years {
        let amount = if y == maturity_years {
            coupon_rate * face + face // final coupon + principal
        } else {
            coupon_rate * face // coupon only
        };
        cfs.push(Cashflow {
            time: y as f64,
            amount,
        });
    }
    cfs
}

/// Build a swap cashflow schedule (fixed leg only, as receiver).
///
/// Returns cashflows for a plain vanilla IRS fixed leg.
pub fn swap_fixed_leg_cashflows(
    fixed_rate: f64,
    notional: f64,
    tenor_years: usize,
) -> Vec<Cashflow> {
    let mut cfs = Vec::with_capacity(tenor_years);
    for y in 1..=tenor_years {
        cfs.push(Cashflow {
            time: y as f64,
            amount: fixed_rate * notional,
        });
    }
    cfs
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn sample_curve() -> DiscountCurveAD<f64> {
        let times = vec![1.0, 2.0, 3.0, 5.0, 10.0];
        let rates = vec![0.03, 0.032, 0.034, 0.038, 0.042];
        DiscountCurveAD::from_zero_rates(&times, &rates)
    }

    #[test]
    fn npv_single_cashflow() {
        let curve = sample_curve();
        let cfs = vec![Cashflow { time: 2.0, amount: 100.0 }];
        let value: f64 = npv(&cfs, &curve);
        let expected = 100.0 * (-0.032 * 2.0_f64).exp();
        assert_abs_diff_eq!(value, expected, epsilon = 1e-10);
    }

    #[test]
    fn npv_bond() {
        let curve = sample_curve();
        let cfs = fixed_rate_bond_cashflows(0.05, 100.0, 5);
        let value: f64 = npv(&cfs, &curve);
        // Should be close to par if coupon ≈ yield
        assert!(value > 90.0 && value < 120.0, "bond npv={}", value);
    }

    #[test]
    fn par_rate_near_yield() {
        let curve = sample_curve();
        let coupon_times: Vec<f64> = (1..=5).map(|y| y as f64).collect();
        let par: f64 = par_rate(&coupon_times, &curve, 100.0);
        // Par rate should be in the range of the curve rates
        assert!(par > 0.02 && par < 0.06, "par rate = {}", par);

        // If we create a bond with par coupon, NPV should ≈ 100
        let cfs = fixed_rate_bond_cashflows(par, 100.0, 5);
        let value: f64 = npv(&cfs, &curve);
        assert_abs_diff_eq!(value, 100.0, epsilon = 0.01);
    }

    #[test]
    fn duration_positive() {
        let curve = sample_curve();
        let cfs = fixed_rate_bond_cashflows(0.05, 100.0, 5);
        let dur: f64 = macaulay_duration(&cfs, &curve);
        assert!(dur > 0.0 && dur < 5.0, "duration = {}", dur);
        // For a 5y bond, Macaulay duration should be around 4.0-4.5
        assert!((dur - 4.5).abs() < 1.0, "duration = {}", dur);
    }

    #[test]
    fn pv01_negative() {
        let curve = sample_curve();
        let cfs = fixed_rate_bond_cashflows(0.05, 100.0, 5);
        let p01: f64 = pv01(&cfs, &curve);
        // PV01 should be negative (higher rates → lower NPV)
        assert!(p01 < 0.0, "pv01 = {}", p01);
    }

    #[test]
    fn bond_dv01_via_areal() {
        use crate::tape::{with_tape, adjoint_tl, AReal};

        let pillar_times = vec![1.0, 2.0, 3.0, 5.0, 10.0];
        let pillar_rates = vec![0.03, 0.032, 0.034, 0.038, 0.042];

        let cfs = fixed_rate_bond_cashflows(0.05, 100.0, 5);

        let (npv_val, dv01) = with_tape(|tape| {
            let rates: Vec<AReal> = pillar_rates.iter().map(|&r| tape.input(r)).collect();
            let curve = DiscountCurveAD::from_zero_rates(&pillar_times, &rates);

            let total = npv(&cfs, &curve);
            let adj = adjoint_tl(total);

            let dv01: Vec<f64> = rates.iter().map(|a| adj[a.idx] * 0.0001).collect();
            (total.val, dv01)
        });

        assert!(npv_val > 0.0);

        // All DV01s should be negative (or zero for uncoupled pillars)
        for (i, &d) in dv01.iter().enumerate() {
            assert!(d <= 0.0 + 1e-10, "dv01[{}] = {} should be <= 0", i, d);
        }
    }

    #[test]
    fn bond_dv01_vs_bump() {
        let pillar_times = vec![1.0, 2.0, 3.0, 5.0, 10.0];
        let pillar_rates = vec![0.03, 0.032, 0.034, 0.038, 0.042];
        let cfs = fixed_rate_bond_cashflows(0.05, 100.0, 5);

        // AD DV01s
        use crate::tape::{with_tape, adjoint_tl, AReal};
        let ad_dv01 = with_tape(|tape| {
            let rates: Vec<AReal> = pillar_rates.iter().map(|&r| tape.input(r)).collect();
            let curve = DiscountCurveAD::from_zero_rates(&pillar_times, &rates);
            let total = npv(&cfs, &curve);
            let adj = adjoint_tl(total);
            rates.iter().map(|a| adj[a.idx] * 0.0001).collect::<Vec<_>>()
        });

        // Bump-and-reprice DV01s
        let bump = 1e-4; // 1 bp
        for i in 0..pillar_times.len() {
            let mut rates_up = pillar_rates.clone();
            rates_up[i] += bump;
            let curve_up = DiscountCurveAD::from_zero_rates(&pillar_times, &rates_up);
            let npv_up: f64 = npv(&cfs, &curve_up);

            let mut rates_dn = pillar_rates.clone();
            rates_dn[i] -= bump;
            let curve_dn = DiscountCurveAD::from_zero_rates(&pillar_times, &rates_dn);
            let npv_dn: f64 = npv(&cfs, &curve_dn);

            let bump_dv01 = (npv_up - npv_dn) / 2.0; // central diff, already per 1bp
            assert_abs_diff_eq!(ad_dv01[i], bump_dv01, epsilon = 1e-6);
        }
    }

    #[test]
    fn swap_par_rate_areal() {
        use crate::tape::{with_tape, adjoint_tl, AReal};

        let pillar_times = vec![1.0, 2.0, 3.0, 5.0, 10.0];
        let pillar_rates = vec![0.03, 0.032, 0.034, 0.038, 0.042];
        let coupon_times: Vec<f64> = (1..=5).map(|y| y as f64).collect();

        let (par_val, sensitivities) = with_tape(|tape| {
            let rates: Vec<AReal> = pillar_rates.iter().map(|&r| tape.input(r)).collect();
            let curve = DiscountCurveAD::from_zero_rates(&pillar_times, &rates);
            let pr = par_rate(&coupon_times, &curve, 100.0);
            let adj = adjoint_tl(pr);
            let sens: Vec<f64> = rates.iter().map(|a| adj[a.idx]).collect();
            (pr.val, sens)
        });

        // Par rate should be positive and in range
        assert!(par_val > 0.02 && par_val < 0.06, "par rate = {}", par_val);

        // All sensitivities should be finite
        for (i, &s) in sensitivities.iter().enumerate() {
            assert!(s.is_finite(), "sensitivity[{}] = {} not finite", i, s);
        }
    }
}
