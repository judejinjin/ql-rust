//! Vasicek one-factor short-rate model.
//!
//! dr = a(b − r) dt + σ dW
//!
//! Parameters: a (mean-reversion), b (long-run level), σ (volatility).
//! Closed-form bond pricing via affine structure.

use crate::calibrated_model::{CalibratedModel, ShortRateModel};
use crate::parameter::{Parameter, PositiveConstraint, NoConstraint};

/// Vasicek short-rate model.
///
/// Bond price: P(t, T) = A(τ) exp(−B(τ) r(t))
///
/// where τ = T − t, and:
///   B(τ) = (1 − e^{−aτ}) / a
///   A(τ) = exp{(B(τ) − τ)(a²b − σ²/2)/a² − σ²B(τ)²/(4a)}
pub struct VasicekModel {
    /// Mean-reversion speed.
    a: f64,
    /// Long-run rate level.
    b: f64,
    /// Volatility.
    sigma: f64,
    /// Initial short rate.
    r0: f64,
    /// Parameters: [a, b, σ].
    params: Vec<Parameter>,
}

impl VasicekModel {
    /// Create a new Vasicek model.
    pub fn new(a: f64, b: f64, sigma: f64, r0: f64) -> Self {
        let params = vec![
            Parameter::new(a, Box::new(PositiveConstraint)),
            Parameter::new(b, Box::new(NoConstraint)),
            Parameter::new(sigma, Box::new(PositiveConstraint)),
        ];
        Self {
            a,
            b,
            sigma,
            r0,
            params,
        }
    }

    /// A.
    pub fn a(&self) -> f64 {
        self.a
    }
    /// B.
    pub fn b(&self) -> f64 {
        self.b
    }
    /// Sigma.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }
    /// R0.
    pub fn r0(&self) -> f64 {
        self.r0
    }

    /// B(τ) = (1 − e^{−aτ}) / a
    pub fn bond_b(&self, tau: f64) -> f64 {
        if self.a.abs() < 1e-15 {
            return tau;
        }
        (1.0 - (-self.a * tau).exp()) / self.a
    }

    /// A(τ) = exp{(B(τ)−τ)(a²b − σ²/2)/a² − σ²B(τ)²/(4a)}
    pub fn bond_a(&self, tau: f64) -> f64 {
        let b_val = self.bond_b(tau);
        let a = self.a;
        let s2 = self.sigma * self.sigma;

        if a.abs() < 1e-15 {
            return (-0.5 * s2 * tau.powi(3) / 3.0).exp();
        }

        let exponent =
            (b_val - tau) * (a * a * self.b - 0.5 * s2) / (a * a) - s2 * b_val * b_val / (4.0 * a);
        exponent.exp()
    }

    /// Zero-coupon bond price P(0, T) given r(0) = r0.
    pub fn bond_price(&self, maturity: f64) -> f64 {
        self.bond_a(maturity) * (-self.bond_b(maturity) * self.r0).exp()
    }

    /// Yield R(T) = −ln P(0,T)/T for maturity T.
    pub fn yield_rate(&self, maturity: f64) -> f64 {
        if maturity < 1e-15 {
            return self.r0;
        }
        -self.bond_price(maturity).ln() / maturity
    }

    /// Zero-coupon bond option (European call/put) in the Vasicek model.
    ///
    /// Price of an option expiring at `option_expiry` on a zero-coupon bond
    /// maturing at `bond_maturity` with strike `strike`.
    ///
    /// Uses Jamshidian (1989) formula for affine models.
    pub fn bond_option(
        &self,
        option_expiry: f64,
        bond_maturity: f64,
        strike: f64,
        is_call: bool,
    ) -> f64 {
        use ql_math::distributions::NormalDistribution;

        let n = NormalDistribution::standard();
        let omega = if is_call { 1.0 } else { -1.0 };

        let tau_s = bond_maturity - option_expiry;
        let b_s = self.bond_b(tau_s);

        // Variance of r integrated from 0 to option_expiry
        let sigma_p = self.sigma
            * b_s
            * ((1.0 - (-2.0 * self.a * option_expiry).exp()) / (2.0 * self.a)).sqrt();

        let p_t = self.bond_price(bond_maturity);
        let p_s = self.bond_price(option_expiry);

        let d1 = (p_t / (p_s * strike)).ln() / sigma_p + 0.5 * sigma_p;
        let d2 = d1 - sigma_p;

        omega * (p_t * n.cdf(omega * d1) - strike * p_s * n.cdf(omega * d2))
    }
}

impl CalibratedModel for VasicekModel {
    fn parameters(&self) -> &[Parameter] {
        &self.params
    }

    fn set_params(&mut self, vals: &[f64]) {
        assert!(vals.len() >= 3, "VasicekModel requires 3 parameters");
        self.params[0].set_value(vals[0]);
        self.params[1].set_value(vals[1]);
        self.params[2].set_value(vals[2]);
        self.a = vals[0];
        self.b = vals[1];
        self.sigma = vals[2];
    }
}

impl ShortRateModel for VasicekModel {
    fn short_rate(&self, _t: f64, x: f64) -> f64 {
        x
    }

    fn discount(&self, t: f64) -> f64 {
        self.bond_price(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_model() -> VasicekModel {
        VasicekModel::new(0.1, 0.05, 0.01, 0.05)
    }

    #[test]
    fn vasicek_bond_price_at_zero() {
        let m = make_model();
        assert_abs_diff_eq!(m.bond_price(0.0), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn vasicek_bond_price_positive() {
        let m = make_model();
        for t in [0.5, 1.0, 2.0, 5.0, 10.0, 30.0] {
            let p = m.bond_price(t);
            assert!(p > 0.0 && p < 1.0, "P(0,{}) = {} not in (0,1)", t, p);
        }
    }

    #[test]
    fn vasicek_bond_price_decreasing() {
        let m = make_model();
        let p1 = m.bond_price(1.0);
        let p5 = m.bond_price(5.0);
        let p10 = m.bond_price(10.0);
        assert!(p1 > p5);
        assert!(p5 > p10);
    }

    #[test]
    fn vasicek_yield_at_zero_equals_r0() {
        let m = make_model();
        let y = m.yield_rate(0.001);
        assert_abs_diff_eq!(y, 0.05, epsilon = 0.001);
    }

    #[test]
    fn vasicek_yield_long_run() {
        let m = make_model();
        // As T → ∞, yield → b − σ²/(2a²)
        let r_inf = m.b - m.sigma * m.sigma / (2.0 * m.a * m.a);
        let y_long = m.yield_rate(100.0);
        assert_abs_diff_eq!(y_long, r_inf, epsilon = 0.001);
    }

    #[test]
    fn vasicek_bond_b_analytic() {
        let m = make_model();
        let tau = 2.0;
        let expected = (1.0 - (-0.1_f64 * tau).exp()) / 0.1;
        assert_abs_diff_eq!(m.bond_b(tau), expected, epsilon = 1e-12);
    }

    #[test]
    fn vasicek_params() {
        let m = make_model();
        let v = m.params_as_vec();
        assert_eq!(v.len(), 3);
        assert_abs_diff_eq!(v[0], 0.1);
        assert_abs_diff_eq!(v[1], 0.05);
        assert_abs_diff_eq!(v[2], 0.01);
    }

    #[test]
    fn vasicek_set_params() {
        let mut m = make_model();
        m.set_params(&[0.2, 0.04, 0.02]);
        assert_abs_diff_eq!(m.a(), 0.2);
        assert_abs_diff_eq!(m.b(), 0.04);
        assert_abs_diff_eq!(m.sigma(), 0.02);
    }

    #[test]
    fn vasicek_bond_option_call_positive() {
        let m = make_model();
        let c = m.bond_option(1.0, 5.0, 0.80, true);
        assert!(c > 0.0, "Bond call should be positive: {}", c);
    }

    #[test]
    fn vasicek_bond_option_put_call_parity() {
        let m = make_model();
        let t_opt = 1.0;
        let t_bond = 5.0;
        let k = 0.80;
        let call = m.bond_option(t_opt, t_bond, k, true);
        let put = m.bond_option(t_opt, t_bond, k, false);
        // C − P = P(0,T_bond) − K P(0,T_opt)
        let parity = call - put - m.bond_price(t_bond) + k * m.bond_price(t_opt);
        assert_abs_diff_eq!(parity, 0.0, epsilon = 1e-10);
    }
}
