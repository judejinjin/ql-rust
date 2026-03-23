//! Extended stochastic processes: G22–G26.
//!
//! **G22** — GSRProcess (Gaussian Short Rate)
//! **G23** — MarkovFunctionalStateProcess
//! **G24** — SquareRootProcess (already exists as alias in cir_process.rs)
//! **G25** — ForwardMeasureProcess
//! **G26** — EulerDiscretization

use crate::process::StochasticProcess1D;
use serde::{Deserialize, Serialize};

// ===========================================================================
// GSRProcess (G22)
// ===========================================================================

/// Gaussian Short Rate process with time-dependent mean reversion (G22).
///
/// dr = κ(t)(θ(t) − r) dt + σ(t) dW
///
/// For the simple case, κ, θ, σ are piecewise constant.
/// The state variable x = r - φ(t) is an Ornstein-Uhlenbeck process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GsrProcess {
    /// X0.
    pub x0: f64,
    /// Piecewise-constant mean reversion (κ) values.
    pub kappa: Vec<f64>,
    /// Times at which κ changes (length = kappa.len() - 1).
    pub kappa_times: Vec<f64>,
    /// Piecewise-constant volatility (σ) values.
    pub sigma: Vec<f64>,
    /// Times at which σ changes (length = sigma.len() - 1).
    pub sigma_times: Vec<f64>,
}

impl GsrProcess {
    /// New.
    pub fn new(x0: f64, kappa: Vec<f64>, sigma: Vec<f64>) -> Self {
        Self {
            x0,
            kappa_times: Vec::new(),
            kappa,
            sigma_times: Vec::new(),
            sigma,
        }
    }

    /// Create with time-varying parameters.
    pub fn with_times(
        x0: f64,
        kappa: Vec<f64>,
        kappa_times: Vec<f64>,
        sigma: Vec<f64>,
        sigma_times: Vec<f64>,
    ) -> Self {
        Self {
            x0,
            kappa,
            kappa_times,
            sigma,
            sigma_times,
        }
    }

    /// Get κ(t).
    pub fn kappa_at(&self, t: f64) -> f64 {
        piecewise_value(&self.kappa, &self.kappa_times, t)
    }

    /// Get σ(t).
    pub fn sigma_at(&self, t: f64) -> f64 {
        piecewise_value(&self.sigma, &self.sigma_times, t)
    }
}

fn piecewise_value(values: &[f64], times: &[f64], t: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    if times.is_empty() {
        return values[0];
    }
    for (i, &ti) in times.iter().enumerate() {
        if t < ti {
            return values[i];
        }
    }
    *values.last().unwrap()
}

impl StochasticProcess1D for GsrProcess {
    fn x0(&self) -> f64 {
        self.x0
    }

    fn drift_1d(&self, t: f64, x: f64) -> f64 {
        -self.kappa_at(t) * x
    }

    fn diffusion_1d(&self, t: f64, _x: f64) -> f64 {
        self.sigma_at(t)
    }
}

// ===========================================================================
// MarkovFunctionalStateProcess (G23)
// ===========================================================================

/// State process for the Markov-functional model (G23).
///
/// dx = −κ x dt + dW  (unit volatility Ornstein-Uhlenbeck)
///
/// The short rate is then a function of the state variable:
///   r(t) = f(t, x(t))
/// where f is calibrated to market instruments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkovFunctionalStateProcess {
    /// X0.
    pub x0: f64,
    /// Mean reversion.
    pub kappa: f64,
    /// Volatility (often normalized to 1.0).
    pub vol: f64,
}

impl MarkovFunctionalStateProcess {
    /// New.
    pub fn new(kappa: f64) -> Self {
        Self {
            x0: 0.0,
            kappa,
            vol: 1.0,
        }
    }

    /// With vol.
    pub fn with_vol(mut self, vol: f64) -> Self {
        self.vol = vol;
        self
    }

    /// Conditional variance of x(t+dt) given x(t): Var = vol² × (1 - e^{-2κdt}) / (2κ).
    pub fn conditional_variance(&self, dt: f64) -> f64 {
        if self.kappa.abs() < 1e-10 {
            self.vol * self.vol * dt
        } else {
            self.vol * self.vol / (2.0 * self.kappa) * (1.0 - (-2.0 * self.kappa * dt).exp())
        }
    }

    /// Conditional expectation of x(t+dt) given x(t): E = x × e^{-κdt}.
    pub fn conditional_expectation(&self, x: f64, dt: f64) -> f64 {
        x * (-self.kappa * dt).exp()
    }
}

impl StochasticProcess1D for MarkovFunctionalStateProcess {
    fn x0(&self) -> f64 {
        self.x0
    }

    fn drift_1d(&self, _t: f64, x: f64) -> f64 {
        -self.kappa * x
    }

    fn diffusion_1d(&self, _t: f64, _x: f64) -> f64 {
        self.vol
    }
}

// ===========================================================================
// ForwardMeasureProcess (G25)
// ===========================================================================

/// Process under the T-forward measure (G25).
///
/// Given a process under the risk-neutral measure:
///   dX = μ(t,X) dt + σ(t,X) dW^Q
///
/// Under the T-forward measure:
///   dX = [μ(t,X) − σ(t,X) × σ_P(t,T)] dt + σ(t,X) dW^T
///
/// where σ_P(t,T) is the volatility of the T-maturity zero-coupon bond.
/// This wrapper applies the measure change.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForwardMeasureProcess {
    /// X0.
    pub x0: f64,
    /// Underlying drift μ (constant approximation).
    pub base_drift: f64,
    /// Underlying diffusion σ (constant approximation).
    pub base_diffusion: f64,
    /// Bond volatility σ_P(t, T) (constant approximation).
    pub bond_volatility: f64,
}

impl ForwardMeasureProcess {
    /// New.
    pub fn new(x0: f64, base_drift: f64, base_diffusion: f64, bond_volatility: f64) -> Self {
        Self {
            x0,
            base_drift,
            base_diffusion,
            bond_volatility,
        }
    }
}

impl StochasticProcess1D for ForwardMeasureProcess {
    fn x0(&self) -> f64 {
        self.x0
    }

    fn drift_1d(&self, _t: f64, _x: f64) -> f64 {
        // Forward-measure drift = μ − σ × σ_P
        self.base_drift - self.base_diffusion * self.bond_volatility
    }

    fn diffusion_1d(&self, _t: f64, _x: f64) -> f64 {
        self.base_diffusion
    }
}

// ===========================================================================
// EulerDiscretization (G26)
// ===========================================================================

/// Standalone Euler discretization schemes (G26).
///
/// Provides explicit Euler, end-of-step Euler, and predictor-corrector
/// discretizations for 1D SDEs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscretizationScheme {
    /// Standard Euler: x_{n+1} = x_n + μ(t_n, x_n)Δt + σ(t_n, x_n)√Δt ΔW.
    Euler,
    /// End-of-step Euler: uses μ(t_{n+1}, x̃_{n+1}) for drift.
    EndEuler,
    /// Predictor-corrector: average of explicit and implicit Euler.
    PredictorCorrector,
}

/// Discretize one step of a 1D SDE using the specified scheme.
///
/// * `process` — the 1D stochastic process
/// * `t` — current time
/// * `x` — current state
/// * `dt` — time step
/// * `dw` — standard normal variate × √dt
/// * `scheme` — discretization scheme
pub fn discretize_step(
    process: &dyn StochasticProcess1D,
    t: f64,
    x: f64,
    dt: f64,
    dw: f64,
    scheme: DiscretizationScheme,
) -> f64 {
    let sqrt_dt = dt.sqrt();
    match scheme {
        DiscretizationScheme::Euler => {
            let mu = process.drift_1d(t, x);
            let sigma = process.diffusion_1d(t, x);
            x + mu * dt + sigma * dw * sqrt_dt
        }
        DiscretizationScheme::EndEuler => {
            // Predictor step
            let mu0 = process.drift_1d(t, x);
            let sigma0 = process.diffusion_1d(t, x);
            let x_pred = x + mu0 * dt + sigma0 * dw * sqrt_dt;
            // Use end-of-step drift
            let mu1 = process.drift_1d(t + dt, x_pred);
            x + mu1 * dt + sigma0 * dw * sqrt_dt
        }
        DiscretizationScheme::PredictorCorrector => {
            let mu0 = process.drift_1d(t, x);
            let sigma0 = process.diffusion_1d(t, x);
            let x_pred = x + mu0 * dt + sigma0 * dw * sqrt_dt;
            let mu1 = process.drift_1d(t + dt, x_pred);
            let sigma1 = process.diffusion_1d(t + dt, x_pred);
            x + 0.5 * (mu0 + mu1) * dt + 0.5 * (sigma0 + sigma1) * dw * sqrt_dt
        }
    }
}

/// Simulate a full path using the specified discretization scheme.
///
/// * `process` — the 1D stochastic process
/// * `t_start` — start time
/// * `n_steps` — number of time steps
/// * `dt` — time step size
/// * `normal_variates` — standard normal random numbers (length = n_steps)
/// * `scheme` — discretization scheme
///
/// Returns the path values at each step (length = n_steps + 1, starting from x0).
#[allow(clippy::needless_range_loop)]
pub fn simulate_path(
    process: &dyn StochasticProcess1D,
    t_start: f64,
    n_steps: usize,
    dt: f64,
    normal_variates: &[f64],
    scheme: DiscretizationScheme,
) -> Vec<f64> {
    assert_eq!(normal_variates.len(), n_steps);
    let mut path = Vec::with_capacity(n_steps + 1);
    let mut x = process.x0();
    let mut t = t_start;
    path.push(x);

    for i in 0..n_steps {
        x = discretize_step(process, t, x, dt, normal_variates[i], scheme);
        t += dt;
        path.push(x);
    }
    path
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn gsr_constant_params() {
        let gsr = GsrProcess::new(0.0, vec![0.05], vec![0.01]);
        assert_abs_diff_eq!(gsr.kappa_at(1.0), 0.05, epsilon = 1e-15);
        assert_abs_diff_eq!(gsr.sigma_at(1.0), 0.01, epsilon = 1e-15);
        // Drift should be -κx
        assert_abs_diff_eq!(gsr.drift_1d(0.0, 0.03), -0.05 * 0.03, epsilon = 1e-15);
    }

    #[test]
    fn gsr_time_varying() {
        let gsr = GsrProcess::with_times(
            0.0,
            vec![0.05, 0.10],
            vec![2.0],
            vec![0.01, 0.015],
            vec![3.0],
        );
        assert_abs_diff_eq!(gsr.kappa_at(1.0), 0.05, epsilon = 1e-15);
        assert_abs_diff_eq!(gsr.kappa_at(5.0), 0.10, epsilon = 1e-15);
        assert_abs_diff_eq!(gsr.sigma_at(2.0), 0.01, epsilon = 1e-15);
        assert_abs_diff_eq!(gsr.sigma_at(4.0), 0.015, epsilon = 1e-15);
    }

    #[test]
    fn markov_functional_conditional() {
        let mf = MarkovFunctionalStateProcess::new(0.05).with_vol(0.01);
        let var = mf.conditional_variance(1.0);
        assert!(var > 0.0);
        let exp = mf.conditional_expectation(1.0, 1.0);
        assert!(exp < 1.0); // Mean reversion pulls toward 0
    }

    #[test]
    fn forward_measure_drift_adjustment() {
        let fmp = ForwardMeasureProcess::new(0.03, 0.05, 0.01, 0.02);
        let drift = fmp.drift_1d(0.0, 0.03);
        assert_abs_diff_eq!(drift, 0.05 - 0.01 * 0.02, epsilon = 1e-15);
    }

    #[test]
    fn euler_discretization_basic() {
        let ou = GsrProcess::new(0.0, vec![1.0], vec![0.1]);
        let x1 = discretize_step(&ou, 0.0, 0.0, 0.01, 0.0, DiscretizationScheme::Euler);
        // With dw=0, drift is -κ*x = 0 → x stays at 0
        assert_abs_diff_eq!(x1, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn predictor_corrector_refinement() {
        let ou = GsrProcess::new(0.0, vec![1.0], vec![0.1]);
        let x_euler = discretize_step(&ou, 0.0, 0.1, 0.01, 1.0, DiscretizationScheme::Euler);
        let x_pc = discretize_step(
            &ou,
            0.0,
            0.1,
            0.01,
            1.0,
            DiscretizationScheme::PredictorCorrector,
        );
        // Both should be close but predictor-corrector uses averaged drift
        assert!((x_euler - x_pc).abs() < 0.01);
    }

    #[test]
    fn simulate_path_ou() {
        let ou = GsrProcess::new(0.0, vec![1.0], vec![0.1]);
        let normals = vec![0.5, -0.3, 0.1, 0.0, -0.2];
        let path = simulate_path(&ou, 0.0, 5, 0.01, &normals, DiscretizationScheme::Euler);
        assert_eq!(path.len(), 6);
        assert_abs_diff_eq!(path[0], 0.0, epsilon = 1e-15);
    }
}
