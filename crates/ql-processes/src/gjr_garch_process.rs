//! GJR-GARCH(1,1) stochastic process.
//!
//! The Glosten-Jagannathan-Runkle GARCH model captures the **leverage effect**:
//! negative returns increase future variance more than positive returns of the
//! same magnitude.
//!
//! ## Variance dynamics
//!
//! $$h_{t+1} = \omega + (\alpha + \gamma \cdot \mathbf{1}_{\epsilon_t < 0})\epsilon_t^2 + \beta h_t$$
//!
//! where $\epsilon_t = z_t \sqrt{h_t}$, $z_t \sim N(0,1)$.
//!
//! ## Return dynamics (log-returns)
//!
//! $$\ln(S_{t+1}/S_t) = (r - q - \tfrac12 h_t)\Delta t + \sqrt{h_t \Delta t}\,z_t$$

use serde::{Deserialize, Serialize};

/// GJR-GARCH(1,1) process parameters.
///
/// # Parameters
///
/// | Symbol | Field | Constraint |
/// |--------|-------|------------|
/// | $S_0$  | `s0`  | > 0 |
/// | $r$    | `risk_free_rate` | any |
/// | $q$    | `dividend_yield` | any |
/// | $h_0$  | `h0`  | > 0 (initial variance) |
/// | $\omega$ | `omega` | > 0 (variance intercept) |
/// | $\alpha$ | `alpha` | ≥ 0 (ARCH coefficient) |
/// | $\beta$  | `beta`  | ≥ 0 (GARCH coefficient) |
/// | $\gamma$ | `gamma` | ≥ 0 (leverage / asymmetry) |
///
/// Stationarity requires $\alpha + \beta + \gamma/2 < 1$.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GjrGarchProcess {
    /// S0.
    pub s0: f64,
    /// Risk free rate.
    pub risk_free_rate: f64,
    /// Dividend yield.
    pub dividend_yield: f64,
    /// H0.
    pub h0: f64,
    /// Omega.
    pub omega: f64,
    /// Alpha.
    pub alpha: f64,
    /// Beta.
    pub beta: f64,
    /// Gamma.
    pub gamma: f64,
}

impl GjrGarchProcess {
    /// Create a new GJR-GARCH(1,1) process.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        s0: f64,
        risk_free_rate: f64,
        dividend_yield: f64,
        h0: f64,
        omega: f64,
        alpha: f64,
        beta: f64,
        gamma: f64,
    ) -> Self {
        assert!(s0 > 0.0, "s0 must be positive");
        assert!(h0 > 0.0, "h0 must be positive");
        assert!(omega > 0.0, "omega must be positive");
        assert!(alpha >= 0.0, "alpha must be non-negative");
        assert!(beta >= 0.0, "beta must be non-negative");
        assert!(gamma >= 0.0, "gamma must be non-negative");
        Self {
            s0,
            risk_free_rate,
            dividend_yield,
            h0,
            omega,
            alpha,
            beta,
            gamma,
        }
    }

    /// Check stationarity condition: α + β + γ/2 < 1.
    pub fn is_stationary(&self) -> bool {
        self.alpha + self.beta + 0.5 * self.gamma < 1.0
    }

    /// Unconditional (long-run) variance: ω / (1 − α − β − γ/2).
    ///
    /// Only valid when `is_stationary()` is true.
    pub fn unconditional_variance(&self) -> f64 {
        let denom = 1.0 - self.alpha - self.beta - 0.5 * self.gamma;
        assert!(denom > 0.0, "Process is not stationary");
        self.omega / denom
    }

    /// Unconditional annualised volatility (sqrt of unconditional variance).
    pub fn unconditional_vol(&self) -> f64 {
        self.unconditional_variance().sqrt()
    }

    /// Evolve the process one time step using Euler discretisation.
    ///
    /// Given the current spot `s`, variance `h`, time step `dt`, and standard
    /// normal draw `z`, returns `(s_next, h_next)`.
    ///
    /// $\epsilon = z \sqrt{h \cdot dt}$
    ///
    /// $\ln(S_{next}/S) = (r - q - h/2) \cdot dt + \epsilon$
    ///
    /// $h_{next} = \omega + (\alpha + \gamma \cdot \mathbf{1}_{z < 0}) \cdot \epsilon^2 / dt + \beta \cdot h$
    ///
    /// (dividing epsilon^2 by dt converts it back to variance scale)
    pub fn evolve(&self, s: f64, h: f64, dt: f64, z: f64) -> (f64, f64) {
        let h_pos = h.max(0.0);
        let epsilon = z * (h_pos * dt).sqrt();

        // Log return
        let log_return = (self.risk_free_rate - self.dividend_yield - 0.5 * h_pos) * dt + epsilon;
        let s_next = s * log_return.exp();

        // Variance update: use z^2 * h (since epsilon^2/dt = z^2 * h)
        let shock = z * z * h_pos;
        let leverage = if z < 0.0 { self.gamma } else { 0.0 };
        let h_next = self.omega + (self.alpha + leverage) * shock + self.beta * h_pos;

        (s_next, h_next.max(0.0))
    }

    /// Simulate a full path of `(spot, variance)` pairs.
    ///
    /// Returns a vector of length `num_steps + 1` starting from `(s0, h0)`.
    pub fn simulate_path(&self, dt: f64, num_steps: usize, normals: &[f64]) -> Vec<(f64, f64)> {
        assert!(
            normals.len() >= num_steps,
            "Need at least {} normals, got {}",
            num_steps,
            normals.len()
        );
        let mut path = Vec::with_capacity(num_steps + 1);
        let mut s = self.s0;
        let mut h = self.h0;
        path.push((s, h));

        for &z in &normals[..num_steps] {
            let (s_new, h_new) = self.evolve(s, h, dt, z);
            s = s_new;
            h = h_new;
            path.push((s, h));
        }
        path
    }

    /// Price a European option via Monte Carlo with the GJR-GARCH process.
    ///
    /// Returns `(npv, std_error)`. Takes pre-generated normal random draws
    /// (flat array of `num_paths * num_steps` draws).
    #[allow(clippy::too_many_arguments)]
    pub fn mc_european(
        &self,
        strike: f64,
        time_to_expiry: f64,
        is_call: bool,
        num_paths: usize,
        num_steps: usize,
        normals: &[f64],
    ) -> (f64, f64) {
        assert!(
            normals.len() >= num_paths * num_steps,
            "Need {} normals, got {}",
            num_paths * num_steps,
            normals.len()
        );

        let dt = time_to_expiry / num_steps as f64;
        let df = (-self.risk_free_rate * time_to_expiry).exp();
        let phi = if is_call { 1.0 } else { -1.0 };

        let mut sum = 0.0;
        let mut sum_sq = 0.0;

        for p in 0..num_paths {
            let mut s = self.s0;
            let mut h = self.h0;
            for step in 0..num_steps {
                let z = normals[p * num_steps + step];
                let (s_new, h_new) = self.evolve(s, h, dt, z);
                s = s_new;
                h = h_new;
            }
            let payoff = (phi * (s - strike)).max(0.0);
            sum += payoff;
            sum_sq += payoff * payoff;
        }

        let n = num_paths as f64;
        let mean = sum / n;
        let var = (sum_sq / n - mean * mean).max(0.0);
        let npv = df * mean;
        let stderr = (var / n).sqrt() * df;
        (npv, stderr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_process() -> GjrGarchProcess {
        // Typical equity GARCH params (daily variance scale, annualised)
        GjrGarchProcess::new(
            100.0, 0.05, 0.02, 0.04, // s0, r, q, h0 (vol ~20%)
            0.000005, 0.05, 0.90, 0.04, // omega, alpha, beta, gamma
        )
    }

    #[test]
    fn stationarity() {
        let p = test_process();
        // alpha + beta + gamma/2 = 0.05 + 0.90 + 0.02 = 0.97 < 1
        assert!(p.is_stationary());
    }

    #[test]
    fn unconditional_variance_positive() {
        let p = test_process();
        let uv = p.unconditional_variance();
        assert!(uv > 0.0, "Unconditional variance = {uv}");
    }

    #[test]
    fn evolve_positive_spot_and_variance() {
        let p = test_process();
        let (s, h) = p.evolve(100.0, 0.04, 1.0 / 252.0, 0.5);
        assert!(s > 0.0, "Spot must be positive: {s}");
        assert!(h > 0.0, "Variance must be positive: {h}");
    }

    #[test]
    fn leverage_effect() {
        let p = test_process();
        let dt = 1.0 / 252.0;
        // Same magnitude shock, but negative vs positive
        let (_, h_neg) = p.evolve(100.0, 0.04, dt, -1.5);
        let (_, h_pos) = p.evolve(100.0, 0.04, dt, 1.5);
        // Negative shock should produce higher variance (leverage effect)
        assert!(
            h_neg > h_pos,
            "Leverage: h_neg={h_neg} should > h_pos={h_pos}"
        );
    }

    #[test]
    fn simulate_path_length() {
        let p = test_process();
        let normals: Vec<f64> = vec![0.1; 100];
        let path = p.simulate_path(1.0 / 252.0, 100, &normals);
        assert_eq!(path.len(), 101);
        assert_eq!(path[0].0, 100.0);
        assert_eq!(path[0].1, 0.04);
    }

    #[test]
    fn mc_european_call_reasonable() {
        let p = GjrGarchProcess::new(
            100.0, 0.05, 0.0, 0.04, 0.000005, 0.05, 0.90, 0.04,
        );
        // Use a simple LCG to generate normals (Box-Muller)
        let num_paths = 20_000;
        let num_steps = 252;
        let normals = generate_normals(num_paths * num_steps, 42);
        let (npv, stderr) = p.mc_european(100.0, 1.0, true, num_paths, num_steps, &normals);
        // Should be in a reasonable range (similar to BS with ~20% vol ≈ 10.45)
        assert!(
            npv > 3.0 && npv < 25.0,
            "GARCH MC call = {npv} (stderr={stderr})"
        );
    }

    #[test]
    fn mc_european_put_call_parity() {
        let p = GjrGarchProcess::new(
            100.0, 0.05, 0.0, 0.04, 0.000005, 0.05, 0.90, 0.04,
        );
        let num_paths = 50_000;
        let num_steps = 252;
        let normals = generate_normals(num_paths * num_steps, 42);
        let (call, _) = p.mc_european(100.0, 1.0, true, num_paths, num_steps, &normals);
        let (put, _) = p.mc_european(100.0, 1.0, false, num_paths, num_steps, &normals);
        // C - P ≈ S*exp(-q*T) - K*exp(-r*T) = 100 - 100*exp(-0.05) ≈ 4.88
        let parity = 100.0 - 100.0 * (-0.05_f64).exp();
        let diff = call - put;
        assert!(
            (diff - parity).abs() < 2.0,
            "Put-call parity: C-P={diff}, expected ~{parity}"
        );
    }

    /// Simple Box-Muller normal generation (no external deps needed for tests).
    fn generate_normals(n: usize, seed: u64) -> Vec<f64> {
        let mut result = Vec::with_capacity(n);
        let mut state = seed;
        while result.len() < n {
            // LCG
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u1 = (state >> 11) as f64 / (1u64 << 53) as f64;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = (state >> 11) as f64 / (1u64 << 53) as f64;
            let u1 = u1.max(1e-15);
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            result.push(r * theta.cos());
            if result.len() < n {
                result.push(r * theta.sin());
            }
        }
        result
    }
}
