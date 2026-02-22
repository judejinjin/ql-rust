//! Hybrid Heston–Hull-White process (equity + stochastic rates).
//!
//! Combines the Heston stochastic-volatility equity dynamics with a
//! Hull-White one-factor short-rate model. This is the basis for the
//! `AnalyticHestonHullWhiteEngine` (H1HW / A1HW approximations).
//!
//! ## SDEs
//!
//! ```text
//! dS/S = (r − q) dt + √v dW_S
//! dv   = κ_v(θ_v − v) dt + σ_v √v dW_v,   dW_S dW_v = ρ_sv dt
//! dr   = (θ_r(t) − a r) dt + σ_r dW_r,    dW_S dW_r = ρ_sr dt
//! ```
//!
//! The equity `S`, variance `v`, and short rate `r` form a 3D joint process.

use serde::{Deserialize, Serialize};
use nalgebra::{DMatrix, DVector};
use crate::process::StochasticProcess;

/// Hybrid Heston + Hull-White stochastic process.
///
/// State vector: [ln S, v, r] (3-dimensional).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HybridHestonHullWhiteProcess {
    // --- Equity / Heston params ---
    /// Initial spot price S(0).
    pub s0: f64,
    /// Continuous dividend yield q.
    pub dividend_yield: f64,
    /// Initial variance v(0).
    pub v0: f64,
    /// Variance mean-reversion speed κ_v.
    pub kappa_v: f64,
    /// Long-run variance θ_v.
    pub theta_v: f64,
    /// Vol-of-vol σ_v.
    pub sigma_v: f64,
    /// Spot–variance correlation ρ_sv.
    pub rho_sv: f64,
    // --- Hull-White rate params ---
    /// Initial short rate r(0).
    pub r0: f64,
    /// HW mean-reversion speed a.
    pub a: f64,
    /// HW volatility σ_r.
    pub sigma_r: f64,
    /// Spot–rate correlation ρ_sr.
    pub rho_sr: f64,
}

impl HybridHestonHullWhiteProcess {
    /// Create a new hybrid process.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        s0: f64,
        dividend_yield: f64,
        v0: f64,
        kappa_v: f64,
        theta_v: f64,
        sigma_v: f64,
        rho_sv: f64,
        r0: f64,
        a: f64,
        sigma_r: f64,
        rho_sr: f64,
    ) -> Self {
        Self {
            s0,
            dividend_yield,
            v0,
            kappa_v,
            theta_v,
            sigma_v,
            rho_sv,
            r0,
            a,
            sigma_r,
            rho_sr,
        }
    }

    /// Correlation between variance and rate Brownian motions.
    ///
    /// Derived from the constraint that the full 3×3 correlation matrix
    /// must be positive semi-definite given ρ_sv and ρ_sr.
    ///
    /// Returns `None` if the implied ρ_vr is outside [−1, 1].
    pub fn rho_vr_max(&self) -> f64 {
        // Upper bound compatible with PSD: sqrt(1 − ρ_sv² − ρ_sr²)
        (1.0 - self.rho_sv * self.rho_sv - self.rho_sr * self.rho_sr)
            .max(0.0)
            .sqrt()
    }

    /// Euler-Maruyama step for the state vector [S, v, r].
    ///
    /// `z` must be a 3-element correlated normal draw obtained by
    /// Cholesky-factoring the correlation matrix Σ = [[1, ρ_sv, ρ_sr],
    /// [ρ_sv, 1, ρ_vr], [ρ_sr, ρ_vr, 1]].
    ///
    /// Returns [S(t+dt), v(t+dt), r(t+dt)].
    pub fn evolve_euler(
        &self,
        s: f64,
        v: f64,
        r: f64,
        dt: f64,
        z: &[f64; 3],
        theta_r_t: f64, // HW drift θ_r(t)
    ) -> [f64; 3] {
        let sqrt_dt = dt.sqrt();
        let vol = v.max(0.0).sqrt();
        // Equity
        let ds = s * ((r - self.dividend_yield) * dt + vol * sqrt_dt * z[0]);
        let s_new = (s + ds).max(1e-12);
        // Variance (absorbed at 0)
        let dv = self.kappa_v * (self.theta_v - v) * dt
            + self.sigma_v * vol * sqrt_dt * z[1];
        let v_new = (v + dv).max(0.0);
        // Short rate
        let dr = (theta_r_t - self.a * r) * dt + self.sigma_r * sqrt_dt * z[2];
        let r_new = r + dr;
        [s_new, v_new, r_new]
    }
}

impl StochasticProcess for HybridHestonHullWhiteProcess {
    fn size(&self) -> usize {
        3
    }

    fn factors(&self) -> usize {
        3
    }

    fn initial_values(&self) -> DVector<f64> {
        DVector::from_vec(vec![self.s0, self.v0, self.r0])
    }

    /// Approximate drift vector. Uses θ_r(t) ≈ a·r₀ (constant HW target).
    fn drift(&self, _t: f64, x: &DVector<f64>) -> DVector<f64> {
        let s = x[0].max(1e-12);
        let v = x[1].max(0.0);
        let r = x[2];
        DVector::from_vec(vec![
            s * (r - self.dividend_yield),
            self.kappa_v * (self.theta_v - v),
            self.a * (self.r0 - r),
        ])
    }

    /// Diffusion matrix: 3×3 with diagonal elements only (ignoring correlations
    /// for the default evolve; use `evolve_euler` for correlated steps).
    fn diffusion(&self, _t: f64, x: &DVector<f64>) -> DMatrix<f64> {
        let v = x[1].max(0.0);
        let vol = v.sqrt();
        let s = x[0].max(1e-12);
        let mut m = DMatrix::zeros(3, 3);
        m[(0, 0)] = s * vol;          // equity diffusion
        m[(1, 1)] = self.sigma_v * vol; // variance diffusion
        m[(2, 2)] = self.sigma_r;       // rate diffusion
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_initial_values() {
        let p = HybridHestonHullWhiteProcess::new(
            100.0, 0.02, 0.04, 1.5, 0.04, 0.3, -0.7, 0.03, 0.1, 0.01, 0.1,
        );
        let iv = p.initial_values();
        assert!((iv[0] - 100.0).abs() < 1e-12);
        assert!((iv[1] - 0.04).abs() < 1e-12);
        assert!((iv[2] - 0.03).abs() < 1e-12);
    }

    #[test]
    fn hybrid_euler_step() {
        let p = HybridHestonHullWhiteProcess::new(
            100.0, 0.0, 0.04, 1.5, 0.04, 0.3, -0.7, 0.03, 0.1, 0.01, 0.1,
        );
        let [s, v, r] = p.evolve_euler(100.0, 0.04, 0.03, 1.0 / 250.0, &[0.0, 0.0, 0.0], 0.033);
        assert!(s > 0.0);
        assert!(v >= 0.0);
        let _ = r; // rate can be any sign
    }

    #[test]
    fn hybrid_rho_vr_bound() {
        let p = HybridHestonHullWhiteProcess::new(
            100.0, 0.0, 0.04, 1.5, 0.04, 0.3, -0.7, 0.03, 0.1, 0.01, 0.1,
        );
        let bound = p.rho_vr_max();
        assert!(bound >= 0.0 && bound <= 1.0);
    }
}
