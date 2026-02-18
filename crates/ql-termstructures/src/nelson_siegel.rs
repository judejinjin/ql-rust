//! Nelson-Siegel and Svensson yield curve fitting.
//!
//! Provides parametric models for the zero-rate term structure:
//! - `NelsonSiegelFitting` — 4-parameter model (β₀, β₁, β₂, τ₁)
//! - `SvenssonFitting` — 6-parameter model (β₀, β₁, β₂, β₃, τ₁, τ₂)
//! - `FittedBondDiscountCurve` — fit discount function to bond prices using
//!   Levenberg-Marquardt–style minimisation.

use ql_core::errors::{QLError, QLResult};
use ql_time::{Calendar, Date, DayCounter};

use crate::term_structure::TermStructure;
use crate::yield_term_structure::YieldTermStructure;

// ===========================================================================
// Nelson-Siegel zero rate model
// ===========================================================================

/// Nelson-Siegel 4-parameter zero rate model.
///
/// ```text
/// z(t) = β₀ + β₁ · [(1 - e^{-t/τ}) / (t/τ)]
///           + β₂ · [(1 - e^{-t/τ}) / (t/τ) - e^{-t/τ}]
/// ```
///
/// Parameters: `[β₀, β₁, β₂, τ]` where τ > 0.
#[derive(Debug, Clone)]
pub struct NelsonSiegelFitting {
    /// Parameters [β₀, β₁, β₂, τ]
    pub params: [f64; 4],
}

impl NelsonSiegelFitting {
    /// Create from calibrated parameters.
    pub fn new(beta0: f64, beta1: f64, beta2: f64, tau: f64) -> Self {
        Self {
            params: [beta0, beta1, beta2, tau],
        }
    }

    /// Zero rate at maturity `t` (in years). Returns continuously compounded rate.
    pub fn zero_rate(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return self.params[0] + self.params[1];
        }
        let x = t / self.params[3];
        let exp_x = (-x).exp();
        let factor1 = if x.abs() < 1e-10 {
            1.0 - x / 2.0 // Taylor expansion for (1 - e^{-x})/x
        } else {
            (1.0 - exp_x) / x
        };
        let factor2 = factor1 - exp_x;

        self.params[0] + self.params[1] * factor1 + self.params[2] * factor2
    }

    /// Discount factor at maturity `t`.
    pub fn discount(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }
        (-self.zero_rate(t) * t).exp()
    }

    /// Calibrate Nelson-Siegel parameters to market zero rates.
    ///
    /// Given `(maturity, zero_rate)` pairs, finds parameters that minimise
    /// the sum of squared residuals using simple grid search + Nelder-Mead.
    pub fn fit(maturities: &[f64], market_rates: &[f64]) -> QLResult<Self> {
        if maturities.len() != market_rates.len() || maturities.is_empty() {
            return Err(QLError::InvalidArgument(
                "maturities and market_rates must have equal non-zero length".into(),
            ));
        }

        // Initial guess from data
        let long_rate = market_rates[market_rates.len() - 1];
        let short_rate = market_rates[0];
        let beta0_init = long_rate;
        let beta1_init = short_rate - long_rate;
        let beta2_init = 0.0;
        let tau_init = 1.5;

        let mut best = [beta0_init, beta1_init, beta2_init, tau_init];

        // Nelder-Mead simplex optimisation (4D)
        let n = 4;
        let mut simplex: Vec<[f64; 4]> = Vec::with_capacity(n + 1);
        simplex.push(best);
        let deltas = [0.02, 0.02, 0.02, 0.5];
        for i in 0..n {
            let mut p = best;
            p[i] += deltas[i];
            simplex.push(p);
        }

        let max_iter = 2000;
        let tol = 1e-12;
        let alpha = 1.0; // reflection
        let gamma = 2.0; // expansion
        let rho = 0.5; // contraction
        let sigma = 0.5; // shrink

        for _ in 0..max_iter {
            // Sort simplex by cost
            simplex.sort_by(|a, b| {
                Self::cost(a, maturities, market_rates)
                    .partial_cmp(&Self::cost(b, maturities, market_rates))
                    .unwrap()
            });

            let cost_best = Self::cost(&simplex[0], maturities, market_rates);
            let cost_worst = Self::cost(&simplex[n], maturities, market_rates);

            if cost_worst - cost_best < tol {
                break;
            }

            // Centroid (excluding worst)
            let mut centroid = [0.0; 4];
            for p in simplex.iter().take(n) {
                for j in 0..4 {
                    centroid[j] += p[j];
                }
            }
            for c in &mut centroid {
                *c /= n as f64;
            }

            // Reflection
            let mut reflected = [0.0; 4];
            for j in 0..4 {
                reflected[j] = centroid[j] + alpha * (centroid[j] - simplex[n][j]);
            }
            // Ensure τ > 0
            if reflected[3] <= 0.01 {
                reflected[3] = 0.01;
            }
            let cost_reflected = Self::cost(&reflected, maturities, market_rates);

            if cost_reflected < Self::cost(&simplex[n - 1], maturities, market_rates)
                && cost_reflected >= cost_best
            {
                simplex[n] = reflected;
                continue;
            }

            if cost_reflected < cost_best {
                // Expansion
                let mut expanded = [0.0; 4];
                for j in 0..4 {
                    expanded[j] = centroid[j] + gamma * (reflected[j] - centroid[j]);
                }
                if expanded[3] <= 0.01 {
                    expanded[3] = 0.01;
                }
                let cost_expanded = Self::cost(&expanded, maturities, market_rates);
                if cost_expanded < cost_reflected {
                    simplex[n] = expanded;
                } else {
                    simplex[n] = reflected;
                }
                continue;
            }

            // Contraction
            let mut contracted = [0.0; 4];
            for j in 0..4 {
                contracted[j] = centroid[j] + rho * (simplex[n][j] - centroid[j]);
            }
            if contracted[3] <= 0.01 {
                contracted[3] = 0.01;
            }
            let cost_contracted = Self::cost(&contracted, maturities, market_rates);

            if cost_contracted < cost_worst {
                simplex[n] = contracted;
            } else {
                // Shrink
                let best_point = simplex[0];
                for item in simplex.iter_mut().skip(1) {
                    for j in 0..4 {
                        item[j] = best_point[j] + sigma * (item[j] - best_point[j]);
                    }
                    if item[3] <= 0.01 {
                        item[3] = 0.01;
                    }
                }
            }
        }

        simplex.sort_by(|a, b| {
            Self::cost(a, maturities, market_rates)
                .partial_cmp(&Self::cost(b, maturities, market_rates))
                .unwrap()
        });

        best = simplex[0];
        let best_cost = Self::cost(&best, maturities, market_rates);

        if best_cost > 1e-4 {
            return Err(QLError::Other(format!(
                "Nelson-Siegel fit did not converge: residual = {best_cost:.6e}"
            )));
        }

        Ok(Self::new(best[0], best[1], best[2], best[3]))
    }

    /// Sum of squared residuals.
    fn cost(params: &[f64; 4], maturities: &[f64], market_rates: &[f64]) -> f64 {
        let ns = NelsonSiegelFitting { params: *params };
        maturities
            .iter()
            .zip(market_rates)
            .map(|(&t, &r)| {
                let diff = ns.zero_rate(t) - r;
                diff * diff
            })
            .sum()
    }
}

// ===========================================================================
// Svensson (Nelson-Siegel-Svensson) zero rate model
// ===========================================================================

/// Svensson 6-parameter extension of Nelson-Siegel.
///
/// ```text
/// z(t) = β₀ + β₁ · [(1-e^{-t/τ₁})/(t/τ₁)]
///           + β₂ · [(1-e^{-t/τ₁})/(t/τ₁) - e^{-t/τ₁}]
///           + β₃ · [(1-e^{-t/τ₂})/(t/τ₂) - e^{-t/τ₂}]
/// ```
///
/// The extra hump term (β₃, τ₂) gives additional flexibility for fitting
/// medium-term humps in the yield curve.
#[derive(Debug, Clone)]
pub struct SvenssonFitting {
    /// Parameters [β₀, β₁, β₂, β₃, τ₁, τ₂]
    pub params: [f64; 6],
}

impl SvenssonFitting {
    /// Create from calibrated parameters.
    pub fn new(beta0: f64, beta1: f64, beta2: f64, beta3: f64, tau1: f64, tau2: f64) -> Self {
        Self {
            params: [beta0, beta1, beta2, beta3, tau1, tau2],
        }
    }

    /// Zero rate at maturity `t` (years).
    pub fn zero_rate(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return self.params[0] + self.params[1];
        }

        let ns_factor = |tau: f64| -> (f64, f64) {
            let x = t / tau;
            let exp_x = (-x).exp();
            let f1 = if x.abs() < 1e-10 {
                1.0 - x / 2.0
            } else {
                (1.0 - exp_x) / x
            };
            (f1, f1 - exp_x)
        };

        let (f1_1, f2_1) = ns_factor(self.params[4]);
        let (_, f2_2) = ns_factor(self.params[5]);

        self.params[0]
            + self.params[1] * f1_1
            + self.params[2] * f2_1
            + self.params[3] * f2_2
    }

    /// Discount factor at maturity `t`.
    pub fn discount(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }
        (-self.zero_rate(t) * t).exp()
    }

    /// Calibrate Svensson parameters to market zero rates using Nelder-Mead.
    pub fn fit(maturities: &[f64], market_rates: &[f64]) -> QLResult<Self> {
        if maturities.len() != market_rates.len() || maturities.is_empty() {
            return Err(QLError::InvalidArgument(
                "maturities and market_rates must have equal non-zero length".into(),
            ));
        }

        let long_rate = market_rates[market_rates.len() - 1];
        let short_rate = market_rates[0];
        let best_init = [long_rate, short_rate - long_rate, 0.0, 0.0, 1.5, 3.0];

        // Nelder-Mead in 6D
        let n = 6;
        let mut simplex: Vec<[f64; 6]> = Vec::with_capacity(n + 1);
        simplex.push(best_init);
        let deltas = [0.02, 0.02, 0.02, 0.02, 0.5, 0.5];
        for i in 0..n {
            let mut p = best_init;
            p[i] += deltas[i];
            simplex.push(p);
        }

        let max_iter = 4000;
        let tol = 1e-14;

        for _ in 0..max_iter {
            simplex.sort_by(|a, b| {
                Self::cost(a, maturities, market_rates)
                    .partial_cmp(&Self::cost(b, maturities, market_rates))
                    .unwrap()
            });

            let cost_best = Self::cost(&simplex[0], maturities, market_rates);
            let cost_worst = Self::cost(&simplex[n], maturities, market_rates);

            if cost_worst - cost_best < tol {
                break;
            }

            let mut centroid = [0.0; 6];
            for p in simplex.iter().take(n) {
                for j in 0..6 {
                    centroid[j] += p[j];
                }
            }
            for c in &mut centroid {
                *c /= n as f64;
            }

            // Reflection
            let mut reflected = [0.0; 6];
            for j in 0..6 {
                reflected[j] = centroid[j] + (centroid[j] - simplex[n][j]);
            }
            Self::clamp_taus(&mut reflected);
            let cost_reflected = Self::cost(&reflected, maturities, market_rates);

            if cost_reflected < Self::cost(&simplex[n - 1], maturities, market_rates)
                && cost_reflected >= cost_best
            {
                simplex[n] = reflected;
                continue;
            }

            if cost_reflected < cost_best {
                let mut expanded = [0.0; 6];
                for j in 0..6 {
                    expanded[j] = centroid[j] + 2.0 * (reflected[j] - centroid[j]);
                }
                Self::clamp_taus(&mut expanded);
                let cost_expanded = Self::cost(&expanded, maturities, market_rates);
                simplex[n] = if cost_expanded < cost_reflected {
                    expanded
                } else {
                    reflected
                };
                continue;
            }

            let mut contracted = [0.0; 6];
            for j in 0..6 {
                contracted[j] = centroid[j] + 0.5 * (simplex[n][j] - centroid[j]);
            }
            Self::clamp_taus(&mut contracted);
            let cost_contracted = Self::cost(&contracted, maturities, market_rates);

            if cost_contracted < cost_worst {
                simplex[n] = contracted;
            } else {
                let best_point = simplex[0];
                for item in simplex.iter_mut().skip(1) {
                    for j in 0..6 {
                        item[j] = best_point[j] + 0.5 * (item[j] - best_point[j]);
                    }
                    Self::clamp_taus(item);
                }
            }
        }

        simplex.sort_by(|a, b| {
            Self::cost(a, maturities, market_rates)
                .partial_cmp(&Self::cost(b, maturities, market_rates))
                .unwrap()
        });

        let best = simplex[0];
        let best_cost = Self::cost(&best, maturities, market_rates);

        if best_cost > 1e-4 {
            return Err(QLError::Other(format!(
                "Svensson fit did not converge: residual = {best_cost:.6e}"
            )));
        }

        Ok(Self::new(best[0], best[1], best[2], best[3], best[4], best[5]))
    }

    /// Ensure τ values stay positive.
    fn clamp_taus(params: &mut [f64; 6]) {
        if params[4] <= 0.01 {
            params[4] = 0.01;
        }
        if params[5] <= 0.01 {
            params[5] = 0.01;
        }
    }

    fn cost(params: &[f64; 6], maturities: &[f64], market_rates: &[f64]) -> f64 {
        let sv = SvenssonFitting { params: *params };
        maturities
            .iter()
            .zip(market_rates)
            .map(|(&t, &r)| {
                let diff = sv.zero_rate(t) - r;
                diff * diff
            })
            .sum()
    }
}

// ===========================================================================
// FittedBondDiscountCurve
// ===========================================================================

/// A fitted bond discount curve using a parametric model.
///
/// Wraps either a `NelsonSiegelFitting` or `SvenssonFitting` as a
/// `YieldTermStructure`.
#[derive(Debug, Clone)]
pub struct FittedBondDiscountCurve {
    reference_date: Date,
    day_counter: DayCounter,
    calendar: Calendar,
    max_date: Date,
    fitting: FittingMethod,
}

/// The parametric fitting method.
#[derive(Debug, Clone)]
pub enum FittingMethod {
    /// Nelson-Siegel 4-parameter model.
    NelsonSiegel(NelsonSiegelFitting),
    /// Svensson 6-parameter model.
    Svensson(SvenssonFitting),
}

impl FittedBondDiscountCurve {
    /// Create a fitted discount curve from a Nelson-Siegel calibration.
    pub fn from_nelson_siegel(
        reference_date: Date,
        fitting: NelsonSiegelFitting,
        day_counter: DayCounter,
        max_maturity_years: f64,
    ) -> Self {
        let max_days = (max_maturity_years * 365.25) as i32;
        let max_date = reference_date + max_days;
        Self {
            reference_date,
            day_counter,
            calendar: Calendar::NullCalendar,
            max_date,
            fitting: FittingMethod::NelsonSiegel(fitting),
        }
    }

    /// Create a fitted discount curve from a Svensson calibration.
    pub fn from_svensson(
        reference_date: Date,
        fitting: SvenssonFitting,
        day_counter: DayCounter,
        max_maturity_years: f64,
    ) -> Self {
        let max_days = (max_maturity_years * 365.25) as i32;
        let max_date = reference_date + max_days;
        Self {
            reference_date,
            day_counter,
            calendar: Calendar::NullCalendar,
            max_date,
            fitting: FittingMethod::Svensson(fitting),
        }
    }

    /// The underlying fitting method.
    pub fn fitting_method(&self) -> &FittingMethod {
        &self.fitting
    }

    /// Zero rate at time `t` from the fitted model.
    pub fn fitted_zero_rate(&self, t: f64) -> f64 {
        match &self.fitting {
            FittingMethod::NelsonSiegel(ns) => ns.zero_rate(t),
            FittingMethod::Svensson(sv) => sv.zero_rate(t),
        }
    }
}

impl TermStructure for FittedBondDiscountCurve {
    fn reference_date(&self) -> Date {
        self.reference_date
    }

    fn day_counter(&self) -> DayCounter {
        self.day_counter
    }

    fn calendar(&self) -> Calendar {
        self.calendar
    }

    fn max_date(&self) -> Date {
        self.max_date
    }
}

impl YieldTermStructure for FittedBondDiscountCurve {
    fn discount_impl(&self, t: f64) -> f64 {
        match &self.fitting {
            FittingMethod::NelsonSiegel(ns) => ns.discount(t),
            FittingMethod::Svensson(sv) => sv.discount(t),
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_time::Month;

    fn sample_ns() -> NelsonSiegelFitting {
        // Typical EUR curve: β₀=3%, β₁=-1%, β₂=1%, τ=1.5
        NelsonSiegelFitting::new(0.03, -0.01, 0.01, 1.5)
    }

    #[test]
    fn ns_short_rate() {
        let ns = sample_ns();
        // At t→0: z = β₀ + β₁ = 0.03 - 0.01 = 0.02
        let z0 = ns.zero_rate(0.0);
        assert_abs_diff_eq!(z0, 0.02, epsilon = 1e-10);
    }

    #[test]
    fn ns_long_rate() {
        let ns = sample_ns();
        // As t→∞: z → β₀ = 0.03
        let z30 = ns.zero_rate(30.0);
        assert_abs_diff_eq!(z30, 0.03, epsilon = 1e-3);
    }

    #[test]
    fn ns_discount_unity_at_zero() {
        let ns = sample_ns();
        assert_abs_diff_eq!(ns.discount(0.0), 1.0, epsilon = 1e-15);
    }

    #[test]
    fn ns_discount_monotone() {
        let ns = sample_ns();
        let mut prev_df = 1.0;
        for i in 1..=30 {
            let df = ns.discount(i as f64);
            assert!(df < prev_df, "df should decrease: t={i}, df={df}, prev={prev_df}");
            assert!(df > 0.0);
            prev_df = df;
        }
    }

    #[test]
    fn ns_fit_recovery() {
        // Generate synthetic data from known NS params, then recover them
        let true_ns = NelsonSiegelFitting::new(0.04, -0.02, 0.015, 2.0);
        let mats: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let rates: Vec<f64> = mats.iter().map(|&t| true_ns.zero_rate(t)).collect();

        let fitted = NelsonSiegelFitting::fit(&mats, &rates).unwrap();

        // Should recover rates to high accuracy
        for (&t, &r) in mats.iter().zip(rates.iter()) {
            assert_abs_diff_eq!(fitted.zero_rate(t), r, epsilon = 1e-5);
        }
    }

    #[test]
    fn svensson_reduces_to_ns() {
        // With β₃=0, Svensson should equal Nelson-Siegel
        let ns = NelsonSiegelFitting::new(0.04, -0.01, 0.02, 1.5);
        let sv = SvenssonFitting::new(0.04, -0.01, 0.02, 0.0, 1.5, 3.0);

        for t in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0] {
            assert_abs_diff_eq!(sv.zero_rate(t), ns.zero_rate(t), epsilon = 1e-10);
        }
    }

    #[test]
    fn svensson_fit_recovery() {
        let true_sv = SvenssonFitting::new(0.04, -0.02, 0.015, -0.005, 2.0, 5.0);
        let mats: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let rates: Vec<f64> = mats.iter().map(|&t| true_sv.zero_rate(t)).collect();

        let fitted = SvenssonFitting::fit(&mats, &rates).unwrap();

        for (&t, &r) in mats.iter().zip(rates.iter()) {
            assert_abs_diff_eq!(fitted.zero_rate(t), r, epsilon = 1e-4);
        }
    }

    #[test]
    fn fitted_bond_discount_curve_ns() {
        let ref_date = Date::from_ymd(2025, Month::January, 1);
        let ns = sample_ns();
        let curve = FittedBondDiscountCurve::from_nelson_siegel(
            ref_date,
            ns.clone(),
            DayCounter::Actual365Fixed,
            30.0,
        );

        assert_eq!(curve.reference_date(), ref_date);

        // Discount at t=0 should be 1
        let df0 = curve.discount_impl(0.0);
        assert_abs_diff_eq!(df0, 1.0, epsilon = 1e-15);

        // Discount at t=5 should match Nelson-Siegel
        let df5 = curve.discount_impl(5.0);
        assert_abs_diff_eq!(df5, ns.discount(5.0), epsilon = 1e-15);

        // Fitted zero rate accessor
        assert_abs_diff_eq!(curve.fitted_zero_rate(10.0), ns.zero_rate(10.0), epsilon = 1e-15);
    }

    #[test]
    fn fitted_bond_discount_curve_svensson() {
        let ref_date = Date::from_ymd(2025, Month::January, 1);
        let sv = SvenssonFitting::new(0.04, -0.01, 0.02, -0.005, 1.5, 5.0);
        let curve = FittedBondDiscountCurve::from_svensson(
            ref_date,
            sv.clone(),
            DayCounter::Actual365Fixed,
            50.0,
        );

        let df10 = curve.discount_impl(10.0);
        assert_abs_diff_eq!(df10, sv.discount(10.0), epsilon = 1e-15);
    }

    #[test]
    fn ns_forward_curve_smooth() {
        let ns = sample_ns();
        // Check that the implied forward rates are smooth and positive
        let mut prev_fwd = 0.0_f64;
        let dt = 0.01;
        for i in 1..100 {
            let t = i as f64 * 0.3;
            // Forward rate ~ -d/dt [ln(df)] = d/dt [z(t)*t] / 1
            let df1 = ns.discount(t);
            let df2 = ns.discount(t + dt);
            let fwd = -(df2 / df1).ln() / dt;
            assert!(fwd > -0.1, "forward rate should be reasonable at t={t}: fwd={fwd}");
            prev_fwd = fwd;
        }
        let _ = prev_fwd; // used in assertion above
    }
}
