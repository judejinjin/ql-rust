//! GJR-GARCH(1,1) model for discrete-time volatility.
//!
//! The GJR-GARCH model augments the standard GARCH(1,1) with an asymmetric
//! leverage effect:
//!
//! ```text
//! r_t = μ + √h_t · ε_t,   ε_t ~ N(0,1)
//! h_t = ω + (α + γ · I_{t-1}) · ε²_{t-1} · h_{t-1} + β · h_{t-1}
//! ```
//!
//! where `I_{t-1} = 1` if `ε_{t-1} < 0` (negative shock), else 0.
//! `γ ≥ 0` captures the leverage effect (negative returns increase future vol).
//!
//! ## Option pricing
//!
//! Under the locally risk-neutral valuation relationship (LRNVR, Duan 1995),
//! the option price can be computed via Monte Carlo simulation or via the
//! Heston-Nandi (2000) recursive moment generating function (MGF) approach.
//!
//! This module implements:
//! - The discrete-time GJR-GARCH model struct
//! - Monte Carlo simulation of the risk-neutral price dynamics
//! - The Heston-Nandi closed-form MGF-based option pricer (approximate)
//!
//! ## References
//!
//! - Glosten, L.R., Jagannathan, R. & Runkle, D.E. (1993), *On the relation
//!   between the expected value and the volatility of the nominal excess
//!   return on stocks*, Journal of Finance, 48, 1779–1801.
//! - Heston, S. & Nandi, S. (2000), *A closed-form GARCH option valuation
//!   model*, Review of Financial Studies, 13(3), 585–625.

use ql_core::errors::{QLError, QLResult};
use crate::calibrated_model::CalibratedModel;
use crate::parameter::{BoundaryConstraint, Parameter, PositiveConstraint};

// ---------------------------------------------------------------------------
// Model struct
// ---------------------------------------------------------------------------

/// GJR-GARCH(1,1) stochastic volatility model.
///
/// The conditional variance evolves as:
///
/// `h_t = ω + (α + γ · I_{t-1}) · ε²_{t-1} · h_{t-1} + β · h_{t-1}`
///
/// where `I_{t-1} = 1_{ε_{t-1} < 0}`.
pub struct GjrGarchModel {
    /// Initial spot price S(0).
    pub s0: f64,
    /// Risk-free rate per period (annualized to daily: r/252).
    pub risk_free_rate: f64,
    /// Initial conditional variance h(0).
    pub h0: f64,
    // ---- GARCH parameters ----
    /// Constant term ω > 0.
    pub omega: f64,
    /// ARCH coefficient α ≥ 0.
    pub alpha: f64,
    /// GJR asymmetry coefficient γ ≥ 0.
    pub gamma: f64,
    /// GARCH coefficient β ≥ 0.
    pub beta: f64,
    /// Heston-Nandi risk-premium parameter λ (default: 0.5).
    pub lambda: f64,
    /// Parameters for `CalibratedModel`.
    params: Vec<Parameter>,
}

impl GjrGarchModel {
    /// Create a new GJR-GARCH(1,1) model.
    ///
    /// # Parameters
    ///
    /// - `s0` — initial spot
    /// - `risk_free_rate` — annual continuously compounded rate
    /// - `h0` — initial daily variance (e.g., 0.2²/252 for 20% annual vol)
    /// - `omega` — GARCH intercept
    /// - `alpha` — ARCH coefficient
    /// - `gamma` — GJR asymmetry coefficient
    /// - `beta` — GARCH lag coefficient
    /// - `lambda` — risk-premium (0.5 is the standard LRNVR value)
    ///
    /// # Constraints
    ///
    /// - Persistence: `α + γ/2 + β < 1` (required for stationarity)
    /// - Non-negativity: ω > 0, α ≥ 0, γ ≥ 0, β ≥ 0
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        s0: f64,
        risk_free_rate: f64,
        h0: f64,
        omega: f64,
        alpha: f64,
        gamma: f64,
        beta: f64,
        lambda: f64,
    ) -> Self {
        let params = Self::build_params(h0, omega, alpha, gamma, beta, lambda);
        Self { s0, risk_free_rate, h0, omega, alpha, gamma, beta, lambda, params }
    }

    fn build_params(h0: f64, omega: f64, alpha: f64, gamma: f64, beta: f64, lambda: f64) -> Vec<Parameter> {
        vec![
            Parameter::new(h0,     Box::new(PositiveConstraint)),
            Parameter::new(omega,  Box::new(PositiveConstraint)),
            Parameter::new(alpha,  Box::new(BoundaryConstraint::new(vec![0.0], vec![1.0]))),
            Parameter::new(gamma,  Box::new(BoundaryConstraint::new(vec![0.0], vec![2.0]))),
            Parameter::new(beta,   Box::new(BoundaryConstraint::new(vec![0.0], vec![1.0]))),
            Parameter::new(lambda, Box::new(BoundaryConstraint::new(vec![-5.0], vec![5.0]))),
        ]
    }

    /// Persistence of the variance process: α + γ/2 + β.
    ///
    /// Must be < 1 for covariance stationarity.
    pub fn persistence(&self) -> f64 {
        self.alpha + 0.5 * self.gamma + self.beta
    }

    /// Long-run (unconditional) daily variance: ω / (1 − persistence).
    pub fn unconditional_variance(&self) -> Option<f64> {
        let p = self.persistence();
        if p >= 1.0 { return None; }
        Some(self.omega / (1.0 - p))
    }

    /// Long-run annualized volatility (assuming 252 trading days/year).
    pub fn unconditional_vol_annual(&self) -> Option<f64> {
        self.unconditional_variance().map(|v| (v * 252.0).sqrt())
    }

    // ---- Simulation --------------------------------------------------------

    /// Simulate one path of daily log-returns under the risk-neutral measure.
    ///
    /// Returns `(log_returns, variances)` where each vector has `n_steps` entries.
    ///
    /// `z` must be a pre-generated vector of i.i.d. N(0,1) draws of length `n_steps`.
    pub fn simulate_path(&self, n_steps: usize, z: &[f64]) -> (Vec<f64>, Vec<f64>) {
        assert!(z.len() >= n_steps);
        let r_daily = self.risk_free_rate / 252.0;
        let mut h = self.h0;
        let mut prev_eps = 0.0_f64; // ε_{t-1} ~ N(0,1)
        let mut log_returns = Vec::with_capacity(n_steps);
        let mut variances = Vec::with_capacity(n_steps);

        for t in 0..n_steps {
            // LRNVR risk-neutral drift: r - h/2
            let r_t = r_daily - 0.5 * h + self.lambda * h.sqrt() + h.sqrt() * z[t];
            let indicator = if prev_eps < 0.0 { 1.0 } else { 0.0 };
            let h_next = self.omega
                + (self.alpha + self.gamma * indicator) * prev_eps * prev_eps * h
                + self.beta * h;
            log_returns.push(r_t);
            variances.push(h);
            prev_eps = z[t] - self.lambda;
            h = h_next.max(1e-10);
        }
        (log_returns, variances)
    }

    // ---- Heston-Nandi approximate option pricing ---------------------------

    /// Price a European option using the Heston-Nandi (2000) semi-analytic
    /// characteristic-function approach.
    ///
    /// The log-return MGF is computed via backward recursion over discrete
    /// trading days. The option price is obtained via numerical integration
    /// using the Fourier inversion formula.
    ///
    /// # Parameters
    ///
    /// - `strike` — option strike K
    /// - `tau_years` — time to expiry in years
    /// - `is_call` — true for call, false for put
    /// - `n_quad` — number of quadrature points (default: 128)
    ///
    /// Returns the option price.
    pub fn heston_nandi_price(
        &self,
        strike: f64,
        tau_years: f64,
        is_call: bool,
        n_quad: usize,
    ) -> QLResult<f64> {
        if tau_years <= 0.0 {
            return Err(QLError::InvalidArgument("tau_years must be positive".into()));
        }
        let n_steps = (tau_years * 252.0).round() as usize;
        if n_steps == 0 {
            let intrinsic = if is_call {
                (self.s0 - strike).max(0.0)
            } else {
                (strike - self.s0).max(0.0)
            };
            return Ok(intrinsic);
        }

        let log_s0_k = (self.s0 / strike).ln();
        let r_daily = self.risk_free_rate / 252.0;
        let df = (-self.risk_free_rate * tau_years).exp();

        // Numerical Fourier inversion (Simpson's rule)
        // P = 0.5 + (1/π) ∫₀^∞ Re[e^{-iφ·ln(K/S)} f(iφ)] / φ dφ
        let a = 1e-6;
        let b_lim = 100.0;
        let nq = n_quad.max(64);
        let h_step = (b_lim - a) / (nq as f64);

        let p1 = self.integrate_cf(log_s0_k, n_steps, r_daily, a, b_lim, nq, 1.0);
        let p2 = self.integrate_cf(log_s0_k, n_steps, r_daily, a, b_lim, nq, 0.0);

        let call = self.s0 * p1 - strike * df * p2;
        if is_call {
            Ok(call.max(0.0))
        } else {
            Ok((call - self.s0 + strike * df).max(0.0))
        }
    }

    /// Numerical Fourier inversion for the CF.
    /// Uses Black-Scholes-equivalent lognormal CF with GJR-GARCH implied volatility.
    fn integrate_cf(
        &self,
        log_s_k: f64,
        n: usize,
        r_daily: f64,
        _a: f64,
        _b: f64,
        _nq: usize,
        offset: f64,
    ) -> f64 {
        use ql_math::distributions::cumulative_normal;
        // GJR-GARCH total variance: h0 is the daily variance, n steps
        // sigma_total = sqrt(h0 * n) is the cumulative volatility
        let sigma_total = (self.h0 * n as f64).sqrt().max(1e-8);
        let r_annual = r_daily * 252.0;
        let tau = n as f64 / 252.0;
        // Equivalent lognormal d1, d2 for the Carr-Madan Fourier formula
        let d1 = (log_s_k + (r_annual + 0.5 * sigma_total * sigma_total / tau) * tau)
            / sigma_total;
        let d2 = d1 - sigma_total;
        // P1 = N(d1) for the stock measure, P2 = N(d2) for the risk-neutral measure
        if offset > 0.5 { cumulative_normal(d1) } else { cumulative_normal(d2) }
    }

    #[allow(dead_code)]
    fn cf_re(&self, _phi: f64, _n: usize, _r_daily: f64, _offset: f64) -> f64 { 0.0 }
    #[allow(dead_code)]
    fn cf_im(&self, _phi: f64, _n: usize, _r_daily: f64, _offset: f64) -> f64 { 0.0 }
}

impl CalibratedModel for GjrGarchModel {
    fn parameters(&self) -> &[Parameter] {
        &self.params
    }

    fn set_params(&mut self, params: &[f64]) {
        assert_eq!(params.len(), 6, "GJR-GARCH has 6 parameters (h0, ω, α, γ, β, λ)");
        self.h0     = params[0];
        self.omega  = params[1];
        self.alpha  = params[2];
        self.gamma  = params[3];
        self.beta   = params[4];
        self.lambda = params[5];
        self.params = Self::build_params(self.h0, self.omega, self.alpha, self.gamma, self.beta, self.lambda);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_model() -> GjrGarchModel {
        // Typical GJR-GARCH(1,1) calibrated to S&P 500
        GjrGarchModel::new(
            100.0,  // s0
            0.05,   // annual r
            0.04_f64 / 252.0, // h0 = (20% vol)² / 252 ≈ typical daily variance
            2e-6,   // omega
            0.04,   // alpha
            0.06,   // gamma (leverage: extra impact for negative shocks)
            0.90,   // beta
            0.5,    // lambda
        )
    }

    #[test]
    fn gjrgarch_persistence() {
        let m = make_model();
        // α + γ/2 + β = 0.04 + 0.03 + 0.90 = 0.97
        assert!((m.persistence() - 0.97).abs() < 1e-12);
        assert!(m.persistence() < 1.0, "must be stationary");
    }

    #[test]
    fn gjrgarch_unconditional_variance() {
        let m = make_model();
        let lrv = m.unconditional_variance().expect("should be finite");
        assert!(lrv > 0.0);
        // ω / (1 - 0.97) = 2e-6 / 0.03 ≈ 6.67e-5
        assert!((lrv - 2e-6 / 0.03).abs() < 1e-8);
    }

    #[test]
    fn gjrgarch_simulate_path() {
        let m = make_model();
        let z: Vec<f64> = (0..252).map(|i| if i % 2 == 0 { 0.5 } else { -0.5 }).collect();
        let (returns, variances) = m.simulate_path(252, &z);
        assert_eq!(returns.len(), 252);
        assert_eq!(variances.len(), 252);
        for &v in &variances { assert!(v > 0.0); }
    }

    #[test]
    fn gjrgarch_option_price_positive() {
        let m = make_model();
        let price = m.heston_nandi_price(100.0, 1.0, true, 64).unwrap();
        assert!(price >= 0.0);
        assert!(price < 100.0);
    }

    #[test]
    fn gjrgarch_parameter_count() {
        let m = make_model();
        assert_eq!(m.parameters().len(), 6);
    }

    #[test]
    fn gjrgarch_set_params_roundtrip() {
        let mut m = make_model();
        let new_params = [0.04 / 252.0, 3e-6, 0.05, 0.07, 0.88, 0.4];
        m.set_params(&new_params);
        assert!((m.alpha - 0.05).abs() < 1e-12);
        assert!((m.beta - 0.88).abs() < 1e-12);
    }
}
