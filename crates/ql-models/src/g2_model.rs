//! G2++ two-factor additive Gaussian short-rate model.
//!
//! r(t) = x(t) + y(t) + φ(t)
//!
//! dx = −a x dt + σ dW₁
//! dy = −b y dt + η dW₂
//! dW₁ dW₂ = ρ dt
//!
//! Five parameters: a, σ, b, η, ρ.
//! φ(t) is a deterministic shift that fits the initial term structure.
//!
//! Analytic bond pricing:
//!   P(t,T) = A(t,T) exp{ −B_a(T−t) x(t) − B_b(T−t) y(t) }
//! where B_a(τ) = (1−e^{−aτ})/a, B_b(τ) = (1−e^{−bτ})/b.

use crate::calibrated_model::CalibratedModel;
use crate::parameter::{BoundaryConstraint, Parameter, PositiveConstraint};

/// G2++ two-factor additive Gaussian model.
///
/// Parameters: a (x-mean-reversion), σ (x-vol), b (y-mean-reversion),
/// η (y-vol), ρ (correlation).
pub struct G2Model {
    /// Mean-reversion speed for x factor.
    a: f64,
    /// Volatility for x factor.
    sigma: f64,
    /// Mean-reversion speed for y factor.
    b: f64,
    /// Volatility for y factor.
    eta: f64,
    /// Correlation between W₁ and W₂.
    rho: f64,
    /// Flat forward rate for the φ(t) shift (constant approximation).
    forward_rate: f64,
    /// Model parameters: [a, σ, b, η, ρ].
    params: Vec<Parameter>,
}

impl G2Model {
    /// Create a new G2++ model.
    ///
    /// * `a`, `sigma` — x-factor mean-reversion and volatility
    /// * `b`, `eta` — y-factor mean-reversion and volatility
    /// * `rho` — correlation ∈ (−1, 1)
    /// * `forward_rate` — constant forward rate for term structure shift
    pub fn new(a: f64, sigma: f64, b: f64, eta: f64, rho: f64, forward_rate: f64) -> Self {
        let params = vec![
            Parameter::new(a, Box::new(PositiveConstraint)),
            Parameter::new(sigma, Box::new(PositiveConstraint)),
            Parameter::new(b, Box::new(PositiveConstraint)),
            Parameter::new(eta, Box::new(PositiveConstraint)),
            Parameter::new(rho, Box::new(BoundaryConstraint::new(vec![-1.0], vec![1.0]))),
        ];
        Self {
            a,
            sigma,
            b,
            eta,
            rho,
            forward_rate,
            params,
        }
    }

    /// Mean-reversion speed for x factor.
    pub fn a(&self) -> f64 {
        self.a
    }

    /// Volatility for x factor.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Mean-reversion speed for y factor.
    pub fn b(&self) -> f64 {
        self.b
    }

    /// Volatility for y factor.
    pub fn eta(&self) -> f64 {
        self.eta
    }

    /// Correlation between the two factors.
    pub fn rho(&self) -> f64 {
        self.rho
    }

    /// B(κ, τ) = (1 − e^{−κτ}) / κ
    fn bond_b(kappa: f64, tau: f64) -> f64 {
        if kappa.abs() < 1e-15 {
            return tau;
        }
        (1.0 - (-kappa * tau).exp()) / kappa
    }

    /// V(t, T) — variance of ln P(t,T) / P^M(t,T).
    ///
    /// V(0, T) = σ²/(2a³)(T + 2/a e^{−aT} − 1/(2a) e^{−2aT} − 3/(2a))
    ///         + η²/(2b³)(T + 2/b e^{−bT} − 1/(2b) e^{−2bT} − 3/(2b))
    ///         + 2ρση/(ab(a+b)) (T + (e^{−aT}−1)/a + (e^{−bT}−1)/b − (e^{−(a+b)T}−1)/(a+b))
    pub fn v(&self, t: f64) -> f64 {
        let a = self.a;
        let b = self.b;
        let s = self.sigma;
        let e = self.eta;
        let r = self.rho;

        let v_x = s * s / (2.0 * a * a * a)
            * (t + 2.0 / a * (-a * t).exp() - 1.0 / (2.0 * a) * (-2.0 * a * t).exp()
                - 3.0 / (2.0 * a));

        let v_y = e * e / (2.0 * b * b * b)
            * (t + 2.0 / b * (-b * t).exp() - 1.0 / (2.0 * b) * (-2.0 * b * t).exp()
                - 3.0 / (2.0 * b));

        let v_xy = 2.0 * r * s * e / (a * b * (a + b))
            * (t + ((-a * t).exp() - 1.0) / a + ((-b * t).exp() - 1.0) / b
                - ((-(a + b) * t).exp() - 1.0) / (a + b));

        v_x + v_y + v_xy
    }

    /// Zero-coupon bond price P(0, T) under the G2++ model.
    ///
    /// With x(0) = y(0) = 0 (standard convention), and constant φ = forward_rate:
    ///   P(0, T) = P^M(0,T) × exp(½ V(T))
    /// where P^M(0,T) = e^{−forward_rate × T} is the market discount factor.
    ///
    /// Actually, the correct formula is:
    ///   P(0,T) = exp{ −forward_rate × T + ½ V(T) }
    /// This holds because x(0) = y(0) = 0 and the convexity correction is V(T)/2.
    pub fn bond_price(&self, maturity: f64) -> f64 {
        if maturity <= 0.0 {
            return 1.0;
        }
        (-self.forward_rate * maturity + 0.5 * self.v(maturity)).exp()
    }

    /// Yield rate from bond price.
    pub fn yield_rate(&self, maturity: f64) -> f64 {
        if maturity <= 0.0 {
            return self.forward_rate;
        }
        let p = self.bond_price(maturity);
        -(p.ln()) / maturity
    }

    /// Price a European swaption under the G2++ model using the
    /// one-dimensional numerical integration method.
    ///
    /// The payer swaption value is:
    ///   V = Σ cᵢ P(0,Tᵢ) N(dᵢ) − ... integrated over x.
    ///
    /// For simplicity we use a Gauss-Hermite quadrature approach:
    /// condition on x at option expiry, integrate over x analytically.
    ///
    /// * `option_expiry` — swaption expiry time
    /// * `swap_tenors` — payment times [T₁, T₂, ..., Tₙ]
    /// * `fixed_rate` — fixed swap rate
    /// * `notional` — notional amount
    /// * `is_payer` — true for payer swaption
    ///
    /// Returns the swaption NPV.
    pub fn swaption_price(
        &self,
        option_expiry: f64,
        swap_tenors: &[f64],
        fixed_rate: f64,
        notional: f64,
        is_payer: bool,
    ) -> f64 {
        if swap_tenors.is_empty() {
            return 0.0;
        }

        let a = self.a;
        let b = self.b;
        let s = self.sigma;
        let e = self.eta;
        let rho = self.rho;
        let t0 = option_expiry;

        let n = swap_tenors.len();

        // Coupon amounts (including principal at last payment)
        let mut coupons = Vec::with_capacity(n);
        for i in 0..n {
            let tau = if i == 0 {
                swap_tenors[0] - t0
            } else {
                swap_tenors[i] - swap_tenors[i - 1]
            };
            let mut c = fixed_rate * tau;
            if i == n - 1 {
                c += 1.0;
            }
            coupons.push(c);
        }

        // Market zero-coupon bond prices
        let p0_t0 = (-self.forward_rate * t0).exp();
        let p0_ti: Vec<f64> = swap_tenors
            .iter()
            .map(|&t| (-self.forward_rate * t).exp())
            .collect();

        // B_a and B_b for each tenor relative to t0
        let b_a: Vec<f64> = swap_tenors
            .iter()
            .map(|&t| Self::bond_b(a, t - t0))
            .collect();
        let b_b: Vec<f64> = swap_tenors
            .iter()
            .map(|&t| Self::bond_b(b, t - t0))
            .collect();

        // Variance of x at t0
        let var_x = s * s / (2.0 * a) * (1.0 - (-2.0 * a * t0).exp());
        let std_x = var_x.sqrt();

        // Conditional variance of y given x at t0
        let var_y = e * e / (2.0 * b) * (1.0 - (-2.0 * b * t0).exp());
        let cov_xy =
            rho * s * e / (a + b) * (1.0 - (-(a + b) * t0).exp());
        let mu_y_given_x_coeff = cov_xy / var_x; // E[y|x] = mu_y_given_x_coeff * x
        let var_y_given_x = var_y - cov_xy * cov_xy / var_x;
        let std_y_given_x = var_y_given_x.max(0.0).sqrt();

        // Ln(P(t0,Ti)/P^M(t0,Ti)) correction terms
        // M_i = -B_a_i * x - B_b_i * y + 0.5 * V_i(condensed)
        // We need the V(t0,Ti) - V(0,Ti) + V(0,t0) correction
        // For simplicity, we use the standard G2++ approach:
        //   P(t0,Ti) = P^M(0,Ti)/P^M(0,t0) * exp(-B_a_i x - B_b_i y - 0.5 Σ_i)
        // where Σ_i accounts for the convexity.

        // Full conditional bond price ratios (simplified):
        // Let's compute the "lnA" correction for each coupon bond
        let ln_a: Vec<f64> = (0..n)
            .map(|i| {
                let tau_i = swap_tenors[i] - t0;
                // lnA = ln(P^M(0,Ti)/P^M(0,t0)) + 0.5*(V(Ti) - V(t0)) correction
                // Simplified: use the affine model relationship directly
                let market_ratio = (p0_ti[i] / p0_t0).ln();
                let v_corr = self.sigma_p_squared(t0, tau_i) / 2.0;
                market_ratio + v_corr
            })
            .collect();

        // Gauss-Hermite quadrature over x (32 points)
        let gh_nodes = gauss_hermite_32();
        let mut value = 0.0;

        for &(xi, wi) in &gh_nodes {
            let x = std_x * std::f64::consts::SQRT_2 * xi;
            let mu_y = mu_y_given_x_coeff * x;

            // For each x, find y* such that the swap value is zero
            // Then compute the swaption value by integrating over y
            // using the normal CDF.

            // The swap value conditioned on (x, y):
            //   SwapVal = 1 - Σ cᵢ exp(lnAᵢ - Bᵢ^a x - Bᵢ^b y)
            // This is monotonic in y, so we can find y* by Newton.

            // Find y* via Newton's method
            let mut y_star = mu_y;
            for _ in 0..50 {
                let mut f_val = -1.0;
                let mut f_deriv = 0.0;
                for i in 0..n {
                    let bond = (ln_a[i] - b_a[i] * x - b_b[i] * y_star).exp();
                    f_val += coupons[i] * bond;
                    f_deriv -= coupons[i] * b_b[i] * bond;
                }
                if f_deriv.abs() < 1e-15 {
                    break;
                }
                let step = f_val / f_deriv;
                y_star -= step;
                if step.abs() < 1e-12 {
                    break;
                }
            }

            // Now compute the swaption value given x.
            // For a payer swaption: max(1 - Σ cᵢ P(t0,Tᵢ), 0)
            // = Σ_{bonds where y > y*} cᵢ [N(dᵢ) ...] - ...
            // Use the conditional normal distribution of y.

            // For each coupon bond, compute the contribution to the
            // swaption payoff using the conditional CDF of y.
            let mut inner = 0.0;

            // P_t0 * (payer: P(t0,T0_last) N(h) - Σ cᵢ P(t0,Tᵢ) N(hᵢ))
            // With sign for payer/receiver

            // Actually: each bond contributes:
            //   cᵢ exp(lnAᵢ - Bᵢ^a x + ½ Bᵢ^b² σ²_{y|x}) N(±dᵢ)
            // where dᵢ = (y* - μ_{y|x} + Bᵢ^b σ²_{y|x}) / σ_{y|x}
            // The sign depends on payer/receiver.

            let omega = if is_payer { 1.0 } else { -1.0 };

            // Term for the "1" part of the swap:
            // This corresponds to receiving 1 at T_last in the payer leg.
            // Actually the swap value is 1 - Σ cᵢ P, so:
            // Payer = max(1 - Σ cᵢ P, 0)
            // We need: P(t0, t0) * N(omega * h0) - Σ cᵢ P(t0,Tᵢ) N(omega * hᵢ)
            // Actually this is not quite right for the standard decomposition.

            // Standard G2++ swaption formula:
            // V_payer = Σᵢ cᵢ * exp(lnAᵢ - Bᵢ^a x + ½ (Bᵢ^b)² var_{y|x})
            //           × [ −N(−(y* − μ − Bᵢ^b var_{y|x})/std) ]
            //         + N((y* − μ)/std)
            // (adjusting signs for payer)

            // Payer swaption payoff = max(1 − Σ cᵢ P(T0,Tᵢ|x,y), 0)
            // Integrating over y|x:
            // = N(d0) − Σ cᵢ P̃ᵢ(x) N(dᵢ)
            // where P̃ᵢ(x) = exp(lnAᵢ − Bᵢ^a x + ½(Bᵢ^b)²σ²_{y|x})
            //   d0 = omega * (y* − μ_{y|x}) / σ_{y|x}
            //   dᵢ = omega * (y* − μ_{y|x} − Bᵢ^b σ²_{y|x}) / σ_{y|x}

            let d0 = omega * (y_star - mu_y) / std_y_given_x;
            inner += normal_cdf(d0);

            for i in 0..n {
                let p_tilde = (ln_a[i] - b_a[i] * x
                    + 0.5 * b_b[i] * b_b[i] * var_y_given_x)
                    .exp();
                let di = omega
                    * (y_star - mu_y - b_b[i] * var_y_given_x)
                    / std_y_given_x;
                inner -= coupons[i] * p_tilde * normal_cdf(di);
            }

            value += wi * omega * inner;
        }

        // Factor: P(0,t0) * notional / sqrt(pi)
        notional * p0_t0 * value / std::f64::consts::PI.sqrt()
    }

    /// σ_P² for the period [t, t+τ] — total variance of the bond log-price.
    fn sigma_p_squared(&self, t: f64, tau: f64) -> f64 {
        let a = self.a;
        let b = self.b;
        let s = self.sigma;
        let e = self.eta;
        let rho = self.rho;

        let ba = Self::bond_b(a, tau);
        let bb = Self::bond_b(b, tau);

        let term_a = s * s * ba * ba / (2.0 * a) * (1.0 - (-2.0 * a * t).exp());
        let term_b = e * e * bb * bb / (2.0 * b) * (1.0 - (-2.0 * b * t).exp());
        let term_ab = 2.0 * rho * s * e * ba * bb / (a + b)
            * (1.0 - (-(a + b) * t).exp());

        term_a + term_b + term_ab
    }
}

impl CalibratedModel for G2Model {
    fn parameters(&self) -> &[Parameter] {
        &self.params
    }

    fn set_params(&mut self, vals: &[f64]) {
        assert!(vals.len() >= 5, "G2Model requires 5 parameters");
        for (i, &v) in vals.iter().enumerate().take(5) {
            self.params[i].set_value(v);
        }
        self.a = vals[0];
        self.sigma = vals[1];
        self.b = vals[2];
        self.eta = vals[3];
        self.rho = vals[4];
    }
}

/// Standard normal CDF (Abramowitz & Stegun approximation).
fn normal_cdf(x: f64) -> f64 {
    use ql_math::distributions::NormalDistribution;
    let n = NormalDistribution::new(0.0, 1.0).unwrap();
    n.cdf(x)
}

/// 20-point Gauss-Hermite quadrature nodes and weights.
/// Nodes xᵢ satisfy ∫ f(x) e^{-x²} dx ≈ Σ wᵢ f(xᵢ).
fn gauss_hermite_32() -> Vec<(f64, f64)> {
    // 20-point GH: 10 positive nodes + symmetry.
    // Source: Abramowitz & Stegun Table 25.10.
    let half: [(f64, f64); 10] = [
        (0.245_340_708_300_901, 0.462_243_669_600_610),
        (0.737_473_728_545_394, 0.286_675_505_362_834),
        (1.234_076_215_395_323, 0.109_017_206_020_023),
        (1.738_537_712_116_586, 0.024_810_520_887_464),
        (2.254_974_002_089_276, 0.003_243_773_342_238),
        (2.788_806_058_428_131, 0.000_228_338_636_017),
        (3.347_854_567_383_216, 0.000_007_802_556_479),
        (3.944_764_040_115_626, 0.000_000_108_606_937),
        (4.603_682_449_550_744, 0.000_000_000_439_934),
        (5.387_480_890_011_233, 0.000_000_000_000_222),
    ];

    let mut result = Vec::with_capacity(20);
    for &(x, w) in half.iter().rev() {
        result.push((-x, w));
    }
    for &(x, w) in &half {
        result.push((x, w));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_model() -> G2Model {
        // a=0.1, σ=0.01, b=0.2, η=0.015, ρ=-0.5, forward_rate=0.05
        G2Model::new(0.1, 0.01, 0.2, 0.015, -0.5, 0.05)
    }

    #[test]
    fn g2_bond_price_at_zero() {
        let m = make_model();
        let p = m.bond_price(0.0);
        assert_abs_diff_eq!(p, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn g2_bond_price_positive() {
        let m = make_model();
        for t in [0.5, 1.0, 2.0, 5.0, 10.0] {
            let p = m.bond_price(t);
            assert!(p > 0.0, "Bond price should be positive at t={t}");
        }
    }

    #[test]
    fn g2_bond_price_decreasing() {
        let m = make_model();
        let p1 = m.bond_price(1.0);
        let p5 = m.bond_price(5.0);
        assert!(p1 > p5, "Bond price should decrease with maturity");
    }

    #[test]
    fn g2_v_at_zero() {
        let m = make_model();
        let v = m.v(0.0);
        assert_abs_diff_eq!(v, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn g2_v_positive() {
        let m = make_model();
        for t in [0.5, 1.0, 5.0, 10.0] {
            let v = m.v(t);
            assert!(v > 0.0, "V(t) should be positive at t={t}: {v}");
        }
    }

    #[test]
    fn g2_params() {
        let m = make_model();
        assert_eq!(m.parameters().len(), 5);
        let v = m.params_as_vec();
        assert_abs_diff_eq!(v[0], 0.1);
        assert_abs_diff_eq!(v[1], 0.01);
        assert_abs_diff_eq!(v[2], 0.2);
        assert_abs_diff_eq!(v[3], 0.015);
        assert_abs_diff_eq!(v[4], -0.5);
    }

    #[test]
    fn g2_set_params() {
        let mut m = make_model();
        m.set_params(&[0.15, 0.02, 0.25, 0.02, -0.3]);
        assert_abs_diff_eq!(m.a(), 0.15);
        assert_abs_diff_eq!(m.sigma(), 0.02);
        assert_abs_diff_eq!(m.b(), 0.25);
        assert_abs_diff_eq!(m.eta(), 0.02);
        assert_abs_diff_eq!(m.rho(), -0.3);
    }

    #[test]
    fn g2_zero_correlation_reduces_to_sum() {
        // With ρ=0, V(T) = V_x(T) + V_y(T), no cross term.
        let m = G2Model::new(0.1, 0.01, 0.2, 0.015, 0.0, 0.05);
        let v = m.v(5.0);

        // Compute V_x and V_y independently
        let m_x_only = G2Model::new(0.1, 0.01, 0.2, 0.0, 0.0, 0.05);
        let m_y_only = G2Model::new(0.1, 0.0, 0.2, 0.015, 0.0, 0.05);
        let v_sum = m_x_only.v(5.0) + m_y_only.v(5.0);

        assert_abs_diff_eq!(v, v_sum, epsilon = 1e-12);
    }

    #[test]
    fn g2_swaption_positive() {
        let m = make_model();
        let option_expiry = 1.0;
        let swap_tenors = vec![2.0, 3.0, 4.0, 5.0];
        let fixed_rate = 0.05;

        let payer = m.swaption_price(option_expiry, &swap_tenors, fixed_rate, 1_000_000.0, true);
        assert!(payer > 0.0, "Payer swaption should be positive: {payer}");
    }

    #[test]
    fn g2_swaption_payer_receiver_parity() {
        let m = make_model();
        let option_expiry = 1.0;
        let swap_tenors = vec![2.0, 3.0, 4.0, 5.0];
        let fixed_rate = 0.05;
        let notional = 1.0;

        let payer =
            m.swaption_price(option_expiry, &swap_tenors, fixed_rate, notional, true);
        let receiver =
            m.swaption_price(option_expiry, &swap_tenors, fixed_rate, notional, false);

        // Payer - Receiver = value of underlying swap
        // = P(0,T0) - Σ cᵢ P(0,Tᵢ)
        let p_t0 = m.bond_price(option_expiry);
        let mut swap_val = p_t0;
        let n = swap_tenors.len();
        for i in 0..n {
            let tau = if i == 0 {
                swap_tenors[0] - option_expiry
            } else {
                swap_tenors[i] - swap_tenors[i - 1]
            };
            let mut c = fixed_rate * tau;
            if i == n - 1 {
                c += 1.0;
            }
            swap_val -= c * m.bond_price(swap_tenors[i]);
        }

        // Parity should hold approximately (numerical quadrature)
        let diff = payer - receiver - swap_val;
        assert_abs_diff_eq!(diff, 0.0, epsilon = 0.01);
    }
}
