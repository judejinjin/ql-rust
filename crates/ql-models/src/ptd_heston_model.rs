//! Piecewise Time-Dependent (PTD) Heston model.
//!
//! Extends the Heston stochastic volatility model by allowing the parameters
//! κ, θ, σ, ρ to be piecewise constant in time.  Within each time interval
//! [tᵢ, tᵢ₊₁) the parameters are constant; they jump at each time node.
//!
//! This is a common extension used to fit the term structure of implied
//! volatility more accurately than the flat Heston model.
//!
//! ## References
//!
//! - Benhamou, E., Gobet, E. & Miri, M. (2012), *Analytical formulas for a
//!   local volatility model with stochastic rates*, Quantitative Finance.
//! - Elices, A. (2008), *Models with time-dependent parameters using
//!   transform methods*.

use crate::calibrated_model::CalibratedModel;
use crate::parameter::{BoundaryConstraint, Parameter, PositiveConstraint};

// ---------------------------------------------------------------------------
// Time-dependent parameter slice
// ---------------------------------------------------------------------------

/// A single piecewise-constant parameter slice for the PTD Heston model.
///
/// Defines the model parameters active in the time interval [t_start, t_end).
#[derive(Clone, Debug)]
pub struct PtdHestonParamSlice {
    /// Start of the time interval (years from today).
    pub t_start: f64,
    /// End of the time interval (years from today).
    pub t_end: f64,
    /// Mean-reversion speed κ(t) (positive).
    pub kappa: f64,
    /// Long-run variance θ(t) (positive).
    pub theta: f64,
    /// Vol-of-vol σ(t) (positive).
    pub sigma: f64,
    /// Correlation ρ(t) ∈ (−1, 1).
    pub rho: f64,
}

impl PtdHestonParamSlice {
    /// Create a new parameter slice.
    pub fn new(t_start: f64, t_end: f64, kappa: f64, theta: f64, sigma: f64, rho: f64) -> Self {
        assert!(t_start < t_end, "t_start must be < t_end");
        assert!(kappa > 0.0 && theta > 0.0 && sigma > 0.0, "kappa/theta/sigma must be positive");
        assert!(rho > -1.0 && rho < 1.0, "rho must be in (-1,1)");
        Self { t_start, t_end, kappa, theta, sigma, rho }
    }

    /// Duration of this slice.
    pub fn dt(&self) -> f64 {
        self.t_end - self.t_start
    }

    /// Whether the Feller condition is satisfied for this slice: 2κθ > σ².
    pub fn feller_satisfied(&self) -> bool {
        2.0 * self.kappa * self.theta > self.sigma * self.sigma
    }
}

// ---------------------------------------------------------------------------
// PTD Heston model
// ---------------------------------------------------------------------------

/// Piecewise Time-Dependent Heston model.
///
/// The state variable is variance v(t); the spot S evolves under:
///
/// ```text
/// dS/S = (r−q) dt + √v dW_S
/// dv   = κ(t)(θ(t)−v) dt + σ(t)√v dW_v,   dW_S dW_v = ρ(t) dt
/// ```
///
/// Parameters κ(t), θ(t), σ(t), ρ(t) are piecewise constant.
pub struct PtdHestonModel {
    /// Initial spot S(0).
    pub s0: f64,
    /// Continuously compounded risk-free rate r.
    pub risk_free_rate: f64,
    /// Continuous dividend yield q.
    pub dividend_yield: f64,
    /// Initial variance v(0).
    pub v0: f64,
    /// Piecewise parameter slices, ordered by t_start.
    pub slices: Vec<PtdHestonParamSlice>,
    /// Flat parameter vector for `CalibratedModel` (v0 + 4·n_slices params).
    params: Vec<Parameter>,
}

impl PtdHestonModel {
    /// Create a new PTD Heston model.
    ///
    /// `slices` must be non-empty, sorted by `t_start`, and contiguous
    /// (i.e. each slice's `t_start == previous.t_end`).
    pub fn new(
        s0: f64,
        risk_free_rate: f64,
        dividend_yield: f64,
        v0: f64,
        slices: Vec<PtdHestonParamSlice>,
    ) -> Self {
        assert!(!slices.is_empty(), "At least one parameter slice is required");
        // Verify contiguity
        for w in slices.windows(2) {
            assert!(
                (w[1].t_start - w[0].t_end).abs() < 1e-10,
                "Slices must be contiguous: slice ends at {} but next starts at {}",
                w[0].t_end, w[1].t_start
            );
        }
        // Build flat parameter vector: v0, then per-slice (kappa, theta, sigma, rho)
        let mut params = vec![Parameter::new(v0, Box::new(PositiveConstraint))];
        for sl in &slices {
            params.push(Parameter::new(sl.kappa, Box::new(PositiveConstraint)));
            params.push(Parameter::new(sl.theta, Box::new(PositiveConstraint)));
            params.push(Parameter::new(sl.sigma, Box::new(PositiveConstraint)));
            params.push(Parameter::new(
                sl.rho,
                Box::new(BoundaryConstraint::new(vec![-0.999], vec![0.999])),
            ));
        }
        Self { s0, risk_free_rate, dividend_yield, v0, slices, params }
    }

    /// Number of time slices.
    pub fn n_slices(&self) -> usize {
        self.slices.len()
    }

    /// Get the parameter slice active at time `t`.
    ///
    /// Returns the last slice if `t` exceeds all slice end-times.
    pub fn slice_at(&self, t: f64) -> &PtdHestonParamSlice {
        for sl in &self.slices {
            if t < sl.t_end {
                return sl;
            }
        }
        self.slices.last().unwrap()
    }

    /// Compute the time-averaged κ over [0, T].
    pub fn kappa_avg(&self, t_end: f64) -> f64 {
        self.weighted_avg(t_end, |sl| sl.kappa)
    }

    /// Compute the time-averaged θ over [0, T], weighted by κ(t)·t.
    pub fn theta_avg(&self, t_end: f64) -> f64 {
        // θ_eff = (∫₀ᵀ κ(t)θ(t) dt) / (∫₀ᵀ κ(t) dt)
        let num = self.weighted_integral(t_end, |sl| sl.kappa * sl.theta);
        let den = self.weighted_integral(t_end, |sl| sl.kappa);
        if den < 1e-14 { return self.slices[0].theta; }
        num / den
    }

    /// Compute the time-averaged σ over [0, T].
    pub fn sigma_avg(&self, t_end: f64) -> f64 {
        self.weighted_avg(t_end, |sl| sl.sigma)
    }

    /// Compute the time-averaged ρ over [0, T].
    pub fn rho_avg(&self, t_end: f64) -> f64 {
        self.weighted_avg(t_end, |sl| sl.rho)
    }

    /// Build a flat constant-parameter Heston approximation valid for maturity `t`.
    ///
    /// Uses time-averaged parameters to produce a single-slice approximation
    /// that can be passed to the standard Heston pricer.
    pub fn effective_flat_params(&self, t: f64) -> (f64, f64, f64, f64) {
        (self.kappa_avg(t), self.theta_avg(t), self.sigma_avg(t), self.rho_avg(t))
    }

    // ---- internal helpers -----------------------------------------------

    fn weighted_avg(&self, t_end: f64, f: impl Fn(&PtdHestonParamSlice) -> f64) -> f64 {
        let total = self.weighted_integral(t_end, &f);
        let len = t_end.min(self.slices.last().unwrap().t_end);
        if len < 1e-14 { return f(&self.slices[0]); }
        total / len
    }

    fn weighted_integral(&self, t_end: f64, f: impl Fn(&PtdHestonParamSlice) -> f64) -> f64 {
        let mut total = 0.0;
        for sl in &self.slices {
            let t0 = sl.t_start;
            let t1 = sl.t_end.min(t_end);
            if t1 <= t0 { break; }
            total += f(sl) * (t1 - t0);
        }
        total
    }
}

impl CalibratedModel for PtdHestonModel {
    fn parameters(&self) -> &[Parameter] {
        &self.params
    }

    fn set_params(&mut self, params: &[f64]) {
        assert_eq!(params.len(), self.params.len(), "Parameter count mismatch");
        self.params[0] = Parameter::new(params[0], Box::new(PositiveConstraint));
        self.v0 = params[0];
        for (i, sl) in self.slices.iter_mut().enumerate() {
            let base = 1 + 4 * i;
            sl.kappa = params[base];
            sl.theta = params[base + 1];
            sl.sigma = params[base + 2];
            sl.rho   = params[base + 3];
            self.params[base]     = Parameter::new(params[base],     Box::new(PositiveConstraint));
            self.params[base + 1] = Parameter::new(params[base + 1], Box::new(PositiveConstraint));
            self.params[base + 2] = Parameter::new(params[base + 2], Box::new(PositiveConstraint));
            self.params[base + 3] = Parameter::new(
                params[base + 3],
                Box::new(BoundaryConstraint::new(vec![-0.999], vec![0.999])),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_model() -> PtdHestonModel {
        let slices = vec![
            PtdHestonParamSlice::new(0.0, 0.5, 1.5, 0.04, 0.3, -0.7),
            PtdHestonParamSlice::new(0.5, 1.0, 1.2, 0.05, 0.25, -0.6),
            PtdHestonParamSlice::new(1.0, 2.0, 1.0, 0.06, 0.2, -0.5),
        ];
        PtdHestonModel::new(100.0, 0.05, 0.0, 0.04, slices)
    }

    #[test]
    fn ptd_n_slices() {
        let m = make_model();
        assert_eq!(m.n_slices(), 3);
    }

    #[test]
    fn ptd_slice_at_t() {
        let m = make_model();
        assert_eq!(m.slice_at(0.25).kappa, 1.5);
        assert_eq!(m.slice_at(0.6).kappa, 1.2);
        assert_eq!(m.slice_at(1.5).kappa, 1.0);
        // After last slice
        assert_eq!(m.slice_at(5.0).kappa, 1.0);
    }

    #[test]
    fn ptd_kappa_avg_single_slice() {
        // Single-slice PTD should equal flat kappa
        let slices = vec![PtdHestonParamSlice::new(0.0, 1.0, 2.0, 0.04, 0.3, -0.5)];
        let m = PtdHestonModel::new(100.0, 0.05, 0.0, 0.04, slices);
        assert!((m.kappa_avg(1.0) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn ptd_feller_condition() {
        let m = make_model();
        for sl in &m.slices {
            // Feller: 2κθ > σ²
            let _ok = sl.feller_satisfied();
            // 2×1.5×0.04 = 0.12 > 0.09 = 0.3² → true for first slice
        }
        assert!(m.slices[0].feller_satisfied());
    }

    #[test]
    fn ptd_parameter_count() {
        let m = make_model();
        // 1 (v0) + 3 slices × 4 = 13 parameters
        assert_eq!(m.parameters().len(), 13);
    }

    #[test]
    fn ptd_effective_flat_params_range() {
        let m = make_model();
        let (kappa, theta, sigma, rho) = m.effective_flat_params(1.0);
        assert!(kappa > 0.0);
        assert!(theta > 0.0);
        assert!(sigma > 0.0);
        assert!(rho < 0.0 && rho > -1.0);
    }
}
