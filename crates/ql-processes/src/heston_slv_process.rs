//! Heston Stochastic Local Volatility (SLV) process.
//!
//! The SLV model combines a local volatility surface L(S,t) with
//! Heston's stochastic variance v(t):
//!
//! ```text
//! dS/S = (r−q) dt + L(S,t) √v dW_S
//! dv   = κ(θ−v) dt + σ √v dW_v,   dW_S dW_v = ρ dt
//! ```
//!
//! The leverage function L(S,t) is calibrated so that the marginal
//! distribution of S matches a given local vol surface exactly (matching
//! all market option prices), while the stochastic vol component captures
//! the vol dynamics.
//!
//! This module provides the process definition and Euler-Maruyama simulation.
//! Calibration of L(S,t) is handled by the FdHestonSLV engine in
//! `ql-pricingengines`.

use serde::{Deserialize, Serialize};
use nalgebra::{DMatrix, DVector};

use crate::process::StochasticProcess;

/// A piecewise-constant leverage function L(S, t) on a (t, S) grid.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LeverageFunction {
    /// Time grid (strictly increasing, starts at 0).
    pub times: Vec<f64>,
    /// Spot grid for each time slice (same length as `times`).
    pub spots: Vec<Vec<f64>>,
    /// Leverage values. `leverages[i][j]` = L(spots[i][j], times[i]).
    pub leverages: Vec<Vec<f64>>,
}

impl LeverageFunction {
    /// Evaluate L(spot, time) by bilinear interpolation / nearest-neighbour.
    pub fn value(&self, spot: f64, time: f64) -> f64 {
        if self.times.is_empty() {
            return 1.0;
        }
        // Find time slice
        let ti = self
            .times
            .partition_point(|&t| t <= time)
            .saturating_sub(1)
            .min(self.times.len() - 1);
        let spots = &self.spots[ti];
        let levs = &self.leverages[ti];
        if spots.is_empty() {
            return 1.0;
        }
        // Find spot index
        let si = spots
            .partition_point(|&s| s <= spot)
            .saturating_sub(1)
            .min(spots.len() - 1);
        levs[si]
    }

    /// Constant leverage = 1 (pure Heston, no local vol correction).
    pub fn flat() -> Self {
        Self {
            times: vec![0.0, 1e9],
            spots: vec![vec![0.0, 1e9], vec![0.0, 1e9]],
            leverages: vec![vec![1.0, 1.0], vec![1.0, 1.0]],
        }
    }
}

/// Heston Stochastic Local Volatility process.
///
/// State: [S, v] (2-dimensional).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HestonSLVProcess {
    /// Initial spot price.
    pub s0: f64,
    /// Risk-free rate.
    pub risk_free_rate: f64,
    /// Dividend yield.
    pub dividend_yield: f64,
    /// Initial variance v(0).
    pub v0: f64,
    /// Mean-reversion speed κ.
    pub kappa: f64,
    /// Long-run variance θ.
    pub theta: f64,
    /// Vol-of-vol σ.
    pub sigma: f64,
    /// Correlation ρ between dW_S and dW_v.
    pub rho: f64,
    /// Leverage function L(S, t).
    pub leverage: LeverageFunction,
}

impl HestonSLVProcess {
    /// Create a new SLV process with the given leverage function.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        s0: f64,
        risk_free_rate: f64,
        dividend_yield: f64,
        v0: f64,
        kappa: f64,
        theta: f64,
        sigma: f64,
        rho: f64,
        leverage: LeverageFunction,
    ) -> Self {
        Self {
            s0,
            risk_free_rate,
            dividend_yield,
            v0,
            kappa,
            theta,
            sigma,
            rho,
            leverage,
        }
    }

    /// Create a pure-Heston SLV process (leverage ≡ 1).
    pub fn pure_heston(
        s0: f64,
        risk_free_rate: f64,
        dividend_yield: f64,
        v0: f64,
        kappa: f64,
        theta: f64,
        sigma: f64,
        rho: f64,
    ) -> Self {
        Self::new(
            s0,
            risk_free_rate,
            dividend_yield,
            v0,
            kappa,
            theta,
            sigma,
            rho,
            LeverageFunction::flat(),
        )
    }

    /// Feller condition: 2κθ > σ².
    pub fn feller_satisfied(&self) -> bool {
        2.0 * self.kappa * self.theta > self.sigma * self.sigma
    }

    /// Euler-Maruyama step.
    ///
    /// `z1`, `z2` are independent standard normals; they are correlated
    /// internally using ρ.
    ///
    /// Returns [S(t+dt), v(t+dt)].
    pub fn evolve_euler(&self, t: f64, s: f64, v: f64, dt: f64, z1: f64, z2: f64) -> [f64; 2] {
        let rho = self.rho;
        let rho_bar = (1.0 - rho * rho).sqrt();
        let sqrt_dt = dt.sqrt();
        let vol = v.max(0.0).sqrt();
        let lev = self.leverage.value(s, t);
        // Correlated Brownian increments: dW_S = z1, dW_v = ρ z1 + √(1-ρ²) z2
        let dw_s = z1 * sqrt_dt;
        let dw_v = (rho * z1 + rho_bar * z2) * sqrt_dt;
        let s_new = s
            * (1.0
                + (self.risk_free_rate - self.dividend_yield) * dt
                + lev * vol * dw_s);
        let v_new = (v + self.kappa * (self.theta - v) * dt + self.sigma * vol * dw_v).max(0.0);
        [s_new.max(1e-12), v_new]
    }
}

impl StochasticProcess for HestonSLVProcess {
    fn size(&self) -> usize {
        2
    }
    fn factors(&self) -> usize {
        2
    }
    fn initial_values(&self) -> DVector<f64> {
        DVector::from_vec(vec![self.s0, self.v0])
    }
    fn drift(&self, _t: f64, x: &DVector<f64>) -> DVector<f64> {
        let s = x[0].max(1e-12);
        let v = x[1].max(0.0);
        DVector::from_vec(vec![
            s * (self.risk_free_rate - self.dividend_yield),
            self.kappa * (self.theta - v),
        ])
    }
    fn diffusion(&self, t: f64, x: &DVector<f64>) -> DMatrix<f64> {
        let s = x[0].max(1e-12);
        let v = x[1].max(0.0);
        let vol = v.sqrt();
        let lev = self.leverage.value(s, t);
        let mut m = DMatrix::zeros(2, 2);
        m[(0, 0)] = s * lev * vol;
        m[(1, 0)] = self.sigma * vol * self.rho;
        m[(1, 1)] = self.sigma * vol * (1.0 - self.rho * self.rho).max(0.0).sqrt();
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slv_initial_values() {
        let p = HestonSLVProcess::pure_heston(100.0, 0.05, 0.0, 0.04, 1.5, 0.04, 0.3, -0.7);
        let iv = p.initial_values();
        assert!((iv[0] - 100.0).abs() < 1e-12);
        assert!((iv[1] - 0.04).abs() < 1e-12);
    }

    #[test]
    fn slv_euler_positive_spot() {
        let p = HestonSLVProcess::pure_heston(100.0, 0.05, 0.0, 0.04, 1.5, 0.04, 0.3, -0.7);
        let [s, v] = p.evolve_euler(0.0, 100.0, 0.04, 1.0 / 252.0, 1.0, -1.0);
        assert!(s > 0.0);
        assert!(v >= 0.0);
    }

    #[test]
    fn slv_feller() {
        let p = HestonSLVProcess::pure_heston(100.0, 0.05, 0.0, 0.04, 1.5, 0.04, 0.3, -0.7);
        // 2 * 1.5 * 0.04 = 0.12 > 0.09 = 0.3²
        assert!(p.feller_satisfied());
    }

    #[test]
    fn leverage_function_flat() {
        let lf = LeverageFunction::flat();
        assert!((lf.value(100.0, 0.5) - 1.0).abs() < 1e-12);
    }
}
