//! Cox-Ingersoll-Ross (CIR) one-factor short-rate model.
//!
//! dr = κ(θ − r) dt + σ √r dW
//!
//! Parameters: κ (mean-reversion), θ (long-run level), σ (volatility).
//! Feller condition: 2κθ > σ² ensures r > 0.
//!
//! Closed-form bond pricing via affine structure (CIR 1985).

use crate::calibrated_model::{CalibratedModel, ShortRateModel};
use crate::parameter::{Parameter, PositiveConstraint};

/// CIR short-rate model.
///
/// Bond price: P(t, T) = A(τ) exp(−B(τ) r(t))
///
/// where τ = T − t, γ = √(κ² + 2σ²), and:
///   B(τ) = 2(e^{γτ} − 1) / ((γ+κ)(e^{γτ}−1) + 2γ)
///   A(τ) = [2γ exp((κ+γ)τ/2) / ((γ+κ)(e^{γτ}−1) + 2γ)]^{2κθ/σ²}
pub struct CIRModel {
    /// Mean-reversion speed κ.
    kappa: f64,
    /// Long-run level θ.
    theta: f64,
    /// Volatility σ.
    sigma: f64,
    /// Initial short rate.
    r0: f64,
    /// Parameters: [κ, θ, σ].
    params: Vec<Parameter>,
}

impl CIRModel {
    /// Create a new CIR model.
    ///
    /// # Panics
    /// If the Feller condition 2κθ ≥ σ² is violated (warning only if close).
    pub fn new(kappa: f64, theta: f64, sigma: f64, r0: f64) -> Self {
        let params = vec![
            Parameter::new(kappa, Box::new(PositiveConstraint)),
            Parameter::new(theta, Box::new(PositiveConstraint)),
            Parameter::new(sigma, Box::new(PositiveConstraint)),
        ];
        Self {
            kappa,
            theta,
            sigma,
            r0,
            params,
        }
    }

    /// Kappa.
    pub fn kappa(&self) -> f64 {
        self.kappa
    }
    /// Theta.
    pub fn theta(&self) -> f64 {
        self.theta
    }
    /// Sigma.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }
    /// R0.
    pub fn r0(&self) -> f64 {
        self.r0
    }

    /// γ = √(κ² + 2σ²)
    fn gamma(&self) -> f64 {
        (self.kappa * self.kappa + 2.0 * self.sigma * self.sigma).sqrt()
    }

    /// B(τ) = 2(e^{γτ} − 1) / ((γ+κ)(e^{γτ}−1) + 2γ)
    pub fn bond_b(&self, tau: f64) -> f64 {
        let g = self.gamma();
        let e_gt = (g * tau).exp();
        2.0 * (e_gt - 1.0) / ((g + self.kappa) * (e_gt - 1.0) + 2.0 * g)
    }

    /// A(τ) = [2γ exp((κ+γ)τ/2) / ((γ+κ)(e^{γτ}−1) + 2γ)]^{2κθ/σ²}
    pub fn bond_a(&self, tau: f64) -> f64 {
        let g = self.gamma();
        let k = self.kappa;
        let s2 = self.sigma * self.sigma;
        let e_gt = (g * tau).exp();
        let denom = (g + k) * (e_gt - 1.0) + 2.0 * g;
        let base = 2.0 * g * ((k + g) * tau / 2.0).exp() / denom;
        let power = 2.0 * k * self.theta / s2;
        base.powf(power)
    }

    /// Zero-coupon bond price P(0, T).
    pub fn bond_price(&self, maturity: f64) -> f64 {
        self.bond_a(maturity) * (-self.bond_b(maturity) * self.r0).exp()
    }

    /// Yield R(T) = −ln P(0,T)/T.
    pub fn yield_rate(&self, maturity: f64) -> f64 {
        if maturity < 1e-15 {
            return self.r0;
        }
        -self.bond_price(maturity).ln() / maturity
    }

    /// Check the Feller condition: 2κθ ≥ σ².
    pub fn feller_satisfied(&self) -> bool {
        2.0 * self.kappa * self.theta >= self.sigma * self.sigma
    }
}

impl CalibratedModel for CIRModel {
    fn parameters(&self) -> &[Parameter] {
        &self.params
    }

    fn set_params(&mut self, vals: &[f64]) {
        assert!(vals.len() >= 3, "CIRModel requires 3 parameters");
        self.params[0].set_value(vals[0]);
        self.params[1].set_value(vals[1]);
        self.params[2].set_value(vals[2]);
        self.kappa = vals[0];
        self.theta = vals[1];
        self.sigma = vals[2];
    }
}

impl ShortRateModel for CIRModel {
    fn short_rate(&self, _t: f64, x: f64) -> f64 {
        x.max(0.0)
    }

    fn discount(&self, t: f64) -> f64 {
        self.bond_price(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_model() -> CIRModel {
        // κ=0.3, θ=0.05, σ=0.1, r0=0.05
        // Feller: 2*0.3*0.05 = 0.03 vs σ²=0.01 → satisfied
        CIRModel::new(0.3, 0.05, 0.1, 0.05)
    }

    #[test]
    fn cir_feller_condition() {
        let m = make_model();
        assert!(m.feller_satisfied());
    }

    #[test]
    fn cir_bond_price_at_zero() {
        let m = make_model();
        assert_abs_diff_eq!(m.bond_price(0.0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn cir_bond_price_positive() {
        let m = make_model();
        for t in [0.5, 1.0, 2.0, 5.0, 10.0, 30.0] {
            let p = m.bond_price(t);
            assert!(p > 0.0 && p < 1.0, "P(0,{}) = {} not in (0,1)", t, p);
        }
    }

    #[test]
    fn cir_bond_price_decreasing() {
        let m = make_model();
        let p1 = m.bond_price(1.0);
        let p5 = m.bond_price(5.0);
        let p10 = m.bond_price(10.0);
        assert!(p1 > p5);
        assert!(p5 > p10);
    }

    #[test]
    fn cir_yield_at_zero_equals_r0() {
        let m = make_model();
        let y = m.yield_rate(0.001);
        assert_abs_diff_eq!(y, 0.05, epsilon = 0.001);
    }

    #[test]
    fn cir_yield_long_run() {
        let m = make_model();
        // As T → ∞, R(∞) = 2κθ / (κ + γ)
        let g = m.gamma();
        let r_inf = 2.0 * m.kappa * m.theta / (m.kappa + g);
        let y_long = m.yield_rate(100.0);
        assert_abs_diff_eq!(y_long, r_inf, epsilon = 0.001);
    }

    #[test]
    fn cir_matches_vasicek_small_sigma() {
        // When σ → 0, CIR → deterministic, bond_price → exp(−r₀ T)
        // (since θ = r₀ and κ very large, mean-reverting to r₀ quickly)
        let m = CIRModel::new(10.0, 0.05, 0.001, 0.05);
        let p = m.bond_price(1.0);
        let p_det = (-0.05_f64).exp();
        assert_abs_diff_eq!(p, p_det, epsilon = 0.001);
    }

    #[test]
    fn cir_bond_b_analytic() {
        let m = make_model();
        let tau = 2.0;
        let g = m.gamma();
        let e_gt = (g * tau).exp();
        let expected = 2.0 * (e_gt - 1.0) / ((g + m.kappa) * (e_gt - 1.0) + 2.0 * g);
        assert_abs_diff_eq!(m.bond_b(tau), expected, epsilon = 1e-12);
    }

    #[test]
    fn cir_params() {
        let m = make_model();
        let v = m.params_as_vec();
        assert_eq!(v.len(), 3);
        assert_abs_diff_eq!(v[0], 0.3);
        assert_abs_diff_eq!(v[1], 0.05);
        assert_abs_diff_eq!(v[2], 0.1);
    }

    #[test]
    fn cir_set_params() {
        let mut m = make_model();
        m.set_params(&[0.5, 0.04, 0.08]);
        assert_abs_diff_eq!(m.kappa(), 0.5);
        assert_abs_diff_eq!(m.theta(), 0.04);
        assert_abs_diff_eq!(m.sigma(), 0.08);
    }
}
