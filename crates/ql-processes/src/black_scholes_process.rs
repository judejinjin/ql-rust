//! Generalized Black-Scholes process.
//!
//! Models the spot price S under the risk-neutral measure:
//!   dS/S = (r(t) − q(t)) dt + σ(t,S) dW
//!
//! where r is the risk-free rate, q is the dividend yield, and σ is the
//! local (or constant) volatility.

use std::sync::Arc;

use ql_termstructures::YieldTermStructure;

use crate::process::StochasticProcess1D;

/// A Generalized Black-Scholes process for equity/FX spot dynamics.
///
/// Stores handles to the risk-free rate curve, dividend yield curve,
/// and a constant volatility. (Black volatility term structure deferred
/// to Phase 8.)
pub struct GeneralizedBlackScholesProcess {
    /// Initial spot price.
    spot: f64,
    /// Risk-free rate term structure.
    risk_free_rate: Arc<dyn YieldTermStructure>,
    /// Dividend yield term structure.
    dividend_yield: Arc<dyn YieldTermStructure>,
    /// Constant (flat) Black-Scholes volatility.
    black_vol: f64,
}

impl GeneralizedBlackScholesProcess {
    /// Create a new GBM process.
    pub fn new(
        spot: f64,
        risk_free_rate: Arc<dyn YieldTermStructure>,
        dividend_yield: Arc<dyn YieldTermStructure>,
        black_vol: f64,
    ) -> Self {
        Self {
            spot,
            risk_free_rate,
            dividend_yield,
            black_vol,
        }
    }

    /// The current spot price.
    pub fn spot(&self) -> f64 {
        self.spot
    }

    /// The risk-free rate curve.
    pub fn risk_free_rate(&self) -> &Arc<dyn YieldTermStructure> {
        &self.risk_free_rate
    }

    /// The dividend yield curve.
    pub fn dividend_yield(&self) -> &Arc<dyn YieldTermStructure> {
        &self.dividend_yield
    }

    /// The Black volatility.
    pub fn black_volatility(&self) -> f64 {
        self.black_vol
    }

    /// Exact evolution for GBM (log-normal).
    ///
    /// S(t+dt) = S(t) exp((r-q-σ²/2)dt + σ√dt dW)
    pub fn evolve_exact(&self, _t0: f64, s0: f64, dt: f64, dw: f64) -> f64 {
        let r = self.risk_free_rate_value();
        let q = self.dividend_yield_value();
        let v = self.black_vol;
        s0 * ((r - q - 0.5 * v * v) * dt + v * dt.sqrt() * dw).exp()
    }

    /// Helper: extract a constant risk-free rate from the curve at t=1.
    fn risk_free_rate_value(&self) -> f64 {
        let df = self.risk_free_rate.discount_t(1.0);
        -df.ln()
    }

    /// Helper: extract a constant dividend yield from the curve at t=1.
    fn dividend_yield_value(&self) -> f64 {
        let df = self.dividend_yield.discount_t(1.0);
        -df.ln()
    }
}

impl StochasticProcess1D for GeneralizedBlackScholesProcess {
    fn x0(&self) -> f64 {
        self.spot
    }

    /// Drift of GBM: (r - q - σ²/2) * S  (in log space: r - q - σ²/2)
    ///
    /// For Euler discretization on the spot: drift = (r - q) * S.
    fn drift_1d(&self, _t: f64, x: f64) -> f64 {
        let r = self.risk_free_rate_value();
        let q = self.dividend_yield_value();
        (r - q) * x
    }

    /// Diffusion: σ * S.
    fn diffusion_1d(&self, _t: f64, x: f64) -> f64 {
        self.black_vol * x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_termstructures::FlatForward;
    use ql_time::{Date, DayCounter, Month};

    fn make_gbm() -> GeneralizedBlackScholesProcess {
        let ref_date = Date::from_ymd(2025, Month::January, 2);
        let risk_free = Arc::new(FlatForward::new(ref_date, 0.05, DayCounter::Actual365Fixed));
        let div_yield = Arc::new(FlatForward::new(ref_date, 0.02, DayCounter::Actual365Fixed));
        GeneralizedBlackScholesProcess::new(100.0, risk_free, div_yield, 0.20)
    }

    #[test]
    fn gbm_initial_value() {
        let gbm = make_gbm();
        assert_abs_diff_eq!(gbm.x0(), 100.0);
    }

    #[test]
    fn gbm_drift() {
        let gbm = make_gbm();
        // drift = (r - q) * S = 0.03 * 100 = 3.0
        let d = gbm.drift_1d(0.0, 100.0);
        assert_abs_diff_eq!(d, 3.0, epsilon = 0.01);
    }

    #[test]
    fn gbm_diffusion() {
        let gbm = make_gbm();
        // diffusion = σ * S = 0.20 * 100 = 20.0
        let s = gbm.diffusion_1d(0.0, 100.0);
        assert_abs_diff_eq!(s, 20.0, epsilon = 1e-12);
    }

    #[test]
    fn gbm_exact_evolution_no_noise() {
        let gbm = make_gbm();
        let dt = 1.0;
        let dw = 0.0;
        let s1 = gbm.evolve_exact(0.0, 100.0, dt, dw);
        // S1 = 100 * exp((0.05-0.02-0.02)*1) = 100 * exp(0.01)
        let expected = 100.0 * (0.05 - 0.02 - 0.5 * 0.04_f64).exp();
        assert_abs_diff_eq!(s1, expected, epsilon = 1e-10);
    }

    #[test]
    fn gbm_euler_vs_exact_close_for_small_dt() {
        let gbm = make_gbm();
        let dt = 0.001;
        let dw = 0.5;
        let euler = gbm.evolve_1d(0.0, 100.0, dt, dw);
        let exact = gbm.evolve_exact(0.0, 100.0, dt, dw);
        // For small dt, Euler and exact should be very close
        assert_abs_diff_eq!(euler, exact, epsilon = 0.1);
    }

    #[test]
    fn gbm_martingale_property() {
        // Under risk-neutral measure, E[S(T)] = S(0) * exp((r-q)*T)
        let gbm = make_gbm();
        let dt = 1.0;
        let e = gbm.expectation_1d(0.0, 100.0, dt);
        let expected = 100.0 + (0.03) * 100.0; // Euler: S + (r-q)*S*dt
        assert_abs_diff_eq!(e, expected, epsilon = 0.1);
    }
}
