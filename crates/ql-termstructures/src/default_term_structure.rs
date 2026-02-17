//! Default probability term structures for credit modelling.
//!
//! Provides the `DefaultProbabilityTermStructure` trait and concrete
//! implementations including piecewise bootstrapping from CDS spreads.

use ql_time::{Calendar, Date, DayCounter};
use crate::term_structure::TermStructure;

// =========================================================================
// DefaultProbabilityTermStructure trait
// =========================================================================

/// A default probability term structure: maps time → survival/default probabilities.
pub trait DefaultProbabilityTermStructure: TermStructure {
    /// Survival probability at time `t` (years from reference date).
    fn survival_probability(&self, t: f64) -> f64;

    /// Default probability in [0, t].
    fn default_probability(&self, t: f64) -> f64 {
        1.0 - self.survival_probability(t)
    }

    /// Default probability in [t1, t2].
    fn default_probability_interval(&self, t1: f64, t2: f64) -> f64 {
        self.survival_probability(t1) - self.survival_probability(t2)
    }

    /// Default density at time `t`.
    fn default_density(&self, t: f64) -> f64 {
        let dt = 1e-4;
        let t1 = (t - dt).max(0.0);
        let t2 = t + dt;
        (self.survival_probability(t1) - self.survival_probability(t2)) / (t2 - t1)
    }

    /// Hazard rate at time `t` = -d/dt[ln S(t)].
    fn hazard_rate(&self, t: f64) -> f64 {
        let dt = 1e-4;
        let t1 = (t - dt).max(0.0);
        let t2 = t + dt;
        let s1 = self.survival_probability(t1);
        let s2 = self.survival_probability(t2);
        if s2 <= 0.0 {
            return 0.0;
        }
        -(s2.ln() - s1.ln()) / (t2 - t1)
    }
}

// =========================================================================
// FlatHazardRate
// =========================================================================

/// Constant hazard rate default probability curve.
///
/// S(t) = exp(-λt), where λ is the constant hazard rate.
#[derive(Debug, Clone)]
pub struct FlatHazardRate {
    reference_date: Date,
    day_counter: DayCounter,
    hazard_rate: f64,
}

impl FlatHazardRate {
    /// Create with a constant hazard rate.
    pub fn new(reference_date: Date, hazard_rate: f64, day_counter: DayCounter) -> Self {
        Self {
            reference_date,
            day_counter,
            hazard_rate,
        }
    }

    /// Create from a CDS spread (approximate: λ ≈ spread / (1 - recovery)).
    pub fn from_spread(
        reference_date: Date,
        spread: f64,
        recovery: f64,
        day_counter: DayCounter,
    ) -> Self {
        let hazard_rate = spread / (1.0 - recovery);
        Self::new(reference_date, hazard_rate, day_counter)
    }
}

impl TermStructure for FlatHazardRate {
    fn reference_date(&self) -> Date {
        self.reference_date
    }
    fn day_counter(&self) -> DayCounter {
        self.day_counter
    }
    fn calendar(&self) -> Calendar {
        Calendar::NullCalendar
    }
    fn max_date(&self) -> Date {
        Date::from_serial(73050)
    }
}

impl DefaultProbabilityTermStructure for FlatHazardRate {
    fn survival_probability(&self, t: f64) -> f64 {
        (-self.hazard_rate * t).exp()
    }

    fn hazard_rate(&self, _t: f64) -> f64 {
        self.hazard_rate
    }
}

// =========================================================================
// PiecewiseDefaultCurve
// =========================================================================

/// A piecewise-constant hazard rate default probability curve.
///
/// Bootstrapped from CDS spread quotes. Between knot points, the hazard rate
/// is piecewise constant.
#[derive(Debug, Clone)]
pub struct PiecewiseDefaultCurve {
    reference_date: Date,
    day_counter: DayCounter,
    /// Time knots (year fractions).
    times: Vec<f64>,
    /// Survival probabilities at each knot.
    survival_probs: Vec<f64>,
}

impl PiecewiseDefaultCurve {
    /// Bootstrap a default curve from CDS spread helpers.
    ///
    /// Each helper provides a maturity (in years) and a par CDS spread.
    /// The recovery rate is assumed constant across all tenors.
    pub fn bootstrap(
        reference_date: Date,
        day_counter: DayCounter,
        tenors: &[f64],
        spreads: &[f64],
        recovery: f64,
    ) -> Self {
        assert_eq!(tenors.len(), spreads.len());
        assert!(!tenors.is_empty());

        let mut times = vec![0.0];
        let mut survival_probs = vec![1.0];

        // Simple bootstrap: for each tenor, solve for hazard rate in the
        // interval [t_{i-1}, t_i] such that the CDS has zero NPV.
        //
        // CDS par spread: s = (1-R) * ∫₀ᵀ S(u) λ(u) du / ∫₀ᵀ S(u) du (approx)
        // With piecewise constant λ in [t_{i-1}, t_i]:
        //   s ≈ (1-R) * (1 - S(T)) / Σ_k Δt_k · S(t_k)
        //
        // We use a discrete quarterly premium leg approximation.
        let num_quarterly = |tenor: f64| -> usize { (tenor * 4.0).round() as usize };

        for i in 0..tenors.len() {
            let t_mat = tenors[i];
            let spread = spreads[i];
            let n_q = num_quarterly(t_mat).max(1);
            let dt_q = t_mat / n_q as f64;

            // Binary search for hazard rate in this interval
            let t_prev = times[times.len() - 1];
            let s_prev = survival_probs[survival_probs.len() - 1];

            let mut lambda_lo = 0.0_f64;
            let mut lambda_hi = 2.0_f64; // reasonable upper bound

            for _ in 0..100 {
                let lambda_mid = 0.5 * (lambda_lo + lambda_hi);

                // Compute S(t_mat) given hazard rate lambda_mid in [t_prev, t_mat]
                let s_mat = s_prev * (-(lambda_mid) * (t_mat - t_prev)).exp();

                // Premium leg (quarterly): Σ dt_q * S(t_k)
                let mut premium_leg = 0.0;
                for k in 1..=n_q {
                    let t_k = k as f64 * dt_q;
                    let s_k = self_survival_at(t_k, &times, &survival_probs, t_prev, s_prev, lambda_mid, t_mat);
                    premium_leg += dt_q * s_k;
                }
                premium_leg *= spread;

                // Protection leg: (1-R) * (1 - S(T)) (simplified)
                let protection_leg = (1.0 - recovery) * (1.0 - s_mat);

                if (premium_leg - protection_leg).abs() < 1e-14 {
                    break;
                }
                if premium_leg > protection_leg {
                    lambda_lo = lambda_mid;
                } else {
                    lambda_hi = lambda_mid;
                }
            }

            let lambda = 0.5 * (lambda_lo + lambda_hi);
            let s_mat = s_prev * (-lambda * (t_mat - t_prev)).exp();

            times.push(t_mat);
            survival_probs.push(s_mat);
        }

        Self {
            reference_date,
            day_counter,
            times,
            survival_probs,
        }
    }

    /// Interpolate survival probability at arbitrary time.
    fn interpolate_survival(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }

        let n = self.times.len();
        if t >= self.times[n - 1] {
            // Extrapolate with last hazard rate
            let last_lambda = -(self.survival_probs[n - 1].ln() - self.survival_probs[n - 2].ln())
                / (self.times[n - 1] - self.times[n - 2]);
            return self.survival_probs[n - 1] * (-last_lambda * (t - self.times[n - 1])).exp();
        }

        // Find bracket
        let idx = self
            .times
            .partition_point(|&ti| ti < t)
            .min(n - 1)
            .max(1);

        // Piecewise constant hazard rate in [times[idx-1], times[idx]]
        let t0 = self.times[idx - 1];
        let s0 = self.survival_probs[idx - 1];
        let t1 = self.times[idx];
        let s1 = self.survival_probs[idx];
        let lambda = -(s1.ln() - s0.ln()) / (t1 - t0);

        s0 * (-lambda * (t - t0)).exp()
    }
}

/// Helper: compute survival probability at time t given piecewise hazard rates.
fn self_survival_at(
    t: f64,
    times: &[f64],
    survival_probs: &[f64],
    t_prev: f64,
    s_prev: f64,
    lambda_current: f64,
    _t_mat: f64,
) -> f64 {
    if t <= t_prev {
        // In previously bootstrapped region: interpolate
        let n = times.len();
        if n <= 1 || t <= 0.0 {
            return 1.0;
        }
        let idx = times.partition_point(|&ti| ti < t).min(n - 1).max(1);
        let t0 = times[idx - 1];
        let s0 = survival_probs[idx - 1];
        let t1 = times[idx];
        let s1 = survival_probs[idx];
        if (t1 - t0).abs() < 1e-14 {
            return s0;
        }
        let lambda = -(s1.ln() - s0.ln()) / (t1 - t0);
        s0 * (-lambda * (t - t0)).exp()
    } else {
        // In current bootstrapping interval
        s_prev * (-lambda_current * (t - t_prev)).exp()
    }
}

impl TermStructure for PiecewiseDefaultCurve {
    fn reference_date(&self) -> Date {
        self.reference_date
    }
    fn day_counter(&self) -> DayCounter {
        self.day_counter
    }
    fn calendar(&self) -> Calendar {
        Calendar::NullCalendar
    }
    fn max_date(&self) -> Date {
        Date::from_serial(73050)
    }
}

impl DefaultProbabilityTermStructure for PiecewiseDefaultCurve {
    fn survival_probability(&self, t: f64) -> f64 {
        self.interpolate_survival(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_time::Month;

    #[test]
    fn flat_hazard_rate_survival() {
        let ref_date = Date::from_ymd(2025, Month::January, 15);
        let hr = FlatHazardRate::new(ref_date, 0.02, DayCounter::Actual365Fixed);
        assert_abs_diff_eq!(hr.survival_probability(0.0), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(hr.survival_probability(1.0), (-0.02_f64).exp(), epsilon = 1e-12);
        assert_abs_diff_eq!(hr.survival_probability(5.0), (-0.10_f64).exp(), epsilon = 1e-12);
    }

    #[test]
    fn flat_hazard_rate_default_probability() {
        let ref_date = Date::from_ymd(2025, Month::January, 15);
        let hr = FlatHazardRate::new(ref_date, 0.02, DayCounter::Actual365Fixed);
        let dp = hr.default_probability(5.0);
        assert_abs_diff_eq!(dp, 1.0 - (-0.10_f64).exp(), epsilon = 1e-12);
    }

    #[test]
    fn flat_hazard_rate_from_spread() {
        let ref_date = Date::from_ymd(2025, Month::January, 15);
        let spread = 0.01; // 100bp
        let recovery = 0.4;
        let hr = FlatHazardRate::from_spread(ref_date, spread, recovery, DayCounter::Actual365Fixed);
        let expected_lambda = 0.01 / 0.6;
        assert_abs_diff_eq!(hr.hazard_rate(1.0), expected_lambda, epsilon = 1e-12);
    }

    #[test]
    fn flat_hazard_rate_density() {
        let ref_date = Date::from_ymd(2025, Month::January, 15);
        let hr = FlatHazardRate::new(ref_date, 0.03, DayCounter::Actual365Fixed);
        // Density = λ·S(t) = λ·exp(-λt)
        let density = hr.default_density(2.0);
        let expected = 0.03 * (-0.06_f64).exp();
        assert_abs_diff_eq!(density, expected, epsilon = 1e-6);
    }

    #[test]
    fn piecewise_default_curve_survival_at_zero() {
        let ref_date = Date::from_ymd(2025, Month::January, 15);
        let tenors = vec![1.0, 3.0, 5.0];
        let spreads = vec![0.005, 0.008, 0.01];
        let recovery = 0.4;
        let curve = PiecewiseDefaultCurve::bootstrap(
            ref_date,
            DayCounter::Actual365Fixed,
            &tenors,
            &spreads,
            recovery,
        );
        assert_abs_diff_eq!(curve.survival_probability(0.0), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn piecewise_default_curve_monotone_decreasing() {
        let ref_date = Date::from_ymd(2025, Month::January, 15);
        let tenors = vec![1.0, 3.0, 5.0, 7.0, 10.0];
        let spreads = vec![0.005, 0.007, 0.009, 0.01, 0.011];
        let recovery = 0.4;
        let curve = PiecewiseDefaultCurve::bootstrap(
            ref_date,
            DayCounter::Actual365Fixed,
            &tenors,
            &spreads,
            recovery,
        );

        let mut prev_s = 1.0;
        for &t in &[0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0] {
            let s = curve.survival_probability(t);
            assert!(
                s < prev_s,
                "Survival probability should decrease: S({t})={s} >= S(prev)={prev_s}"
            );
            assert!(s > 0.0, "Survival probability should be positive at t={t}");
            prev_s = s;
        }
    }

    #[test]
    fn piecewise_default_curve_reproduced_spread() {
        // Bootstrap from a single spread and verify the par spread is recovered
        let ref_date = Date::from_ymd(2025, Month::January, 15);
        let spread = 0.01; // 100bp
        let recovery = 0.4;
        let tenor = 5.0;

        let curve = PiecewiseDefaultCurve::bootstrap(
            ref_date,
            DayCounter::Actual365Fixed,
            &[tenor],
            &[spread],
            recovery,
        );

        // Recompute par spread from bootstrapped curve
        let n_q = 20; // 5 years, quarterly
        let dt_q = tenor / n_q as f64;
        let mut premium_pv = 0.0;
        for k in 1..=n_q {
            let t_k = k as f64 * dt_q;
            premium_pv += dt_q * curve.survival_probability(t_k);
        }
        let protection_pv = (1.0 - recovery) * (1.0 - curve.survival_probability(tenor));
        let implied_spread = protection_pv / premium_pv;

        assert_abs_diff_eq!(implied_spread, spread, epsilon = 1e-6);
    }

    #[test]
    fn default_probability_interval() {
        let ref_date = Date::from_ymd(2025, Month::January, 15);
        let hr = FlatHazardRate::new(ref_date, 0.02, DayCounter::Actual365Fixed);
        let dp = hr.default_probability_interval(1.0, 3.0);
        let expected = hr.survival_probability(1.0) - hr.survival_probability(3.0);
        assert_abs_diff_eq!(dp, expected, epsilon = 1e-12);
    }
}
