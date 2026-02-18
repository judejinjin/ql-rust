//! Black-Karasinski one-factor short-rate model.
//!
//! d ln(r) = [θ(t) − a ln(r)] dt + σ dW
//!
//! The short rate is log-normal, so r > 0 by construction.
//! Unlike Vasicek/HW, no analytic bond pricing formula exists;
//! pricing is done by tree or PDE methods.
//!
//! This implementation stores constant a, σ and provides
//! tree-based bond pricing via a recombining trinomial tree.

use crate::calibrated_model::{CalibratedModel, ShortRateModel};
use crate::parameter::{Parameter, PositiveConstraint};

/// Black-Karasinski one-factor model with constant parameters.
///
/// d ln(r) = [θ(t) − a ln(r)] dt + σ dW
///
/// For constant θ(t) = a × ln(r∞), the log-rate is an OU process
/// with long-run mean ln(r∞).
pub struct BlackKarasinskiModel {
    /// Mean-reversion speed a.
    a: f64,
    /// Volatility of ln(r).
    sigma: f64,
    /// Initial short rate r(0).
    r0: f64,
    /// Long-run rate level r∞ (constant θ approximation: θ = a ln(r∞)).
    long_rate: f64,
    /// Model parameters: [a, sigma].
    params: Vec<Parameter>,
}

impl BlackKarasinskiModel {
    /// Create a new Black-Karasinski model.
    ///
    /// * `a` — mean-reversion speed
    /// * `sigma` — volatility of log-rate
    /// * `r0` — initial short rate
    /// * `long_rate` — long-run equilibrium rate
    pub fn new(a: f64, sigma: f64, r0: f64, long_rate: f64) -> Self {
        let params = vec![
            Parameter::new(a, Box::new(PositiveConstraint)),
            Parameter::new(sigma, Box::new(PositiveConstraint)),
        ];
        Self {
            a,
            sigma,
            r0,
            long_rate,
            params,
        }
    }

    /// Mean-reversion speed.
    pub fn a(&self) -> f64 {
        self.a
    }

    /// Volatility of log-rate.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Initial short rate.
    pub fn r0(&self) -> f64 {
        self.r0
    }

    /// Long-run rate level.
    pub fn long_rate(&self) -> f64 {
        self.long_rate
    }

    /// Price a zero-coupon bond P(0, T) using a trinomial tree.
    ///
    /// The log-rate x = ln(r) follows an OU process
    ///   dx = a(μ − x) dt + σ dW
    /// where μ = ln(long_rate).
    ///
    /// We build a recombining trinomial tree in x-space with `n_steps`
    /// time steps and backward-induct from P(T,T) = 1.
    pub fn bond_price_tree(&self, maturity: f64, n_steps: usize) -> f64 {
        if maturity <= 0.0 {
            return 1.0;
        }
        let dt = maturity / n_steps as f64;
        let mu = self.long_rate.ln();
        let x0 = self.r0.ln();
        let dx = self.sigma * (3.0 * dt).sqrt();

        // State grid: x0 + j * dx, j in [-j_max, j_max].
        // j_max chosen so that the tree doesn't grow beyond a certain width.
        let j_max = (0.184 / (self.a * dt)).ceil() as i64;
        let j_max = j_max.max(1);
        let width = (2 * j_max + 1) as usize;

        // Transition probabilities for node j:
        // The drift at node j: a*(mu - x_j)/dx
        // For standard trinomial tree on OU:
        //   p_up   = 1/6 + (a_j dt - a_j²dt²)/2
        //   p_mid  = 2/3 - a_j²dt²
        //   p_down = 1/6 + (−a_j dt − a_j²dt²) / 2 ... wrong sign on a_j
        // Actually: use the Hull-White style trinomial tree branching.
        //
        // At node j, ξ_j = a*(mu - (x0 + j*dx)) * dt / dx
        // p_up   = 1/6 + (ξ² + ξ)/2
        // p_mid  = 2/3 - ξ²
        // p_down = 1/6 + (ξ² - ξ)/2

        // Initialize: bond value = 1 at maturity
        let mut values = vec![1.0_f64; width];
        let idx = |j: i64| -> usize { (j + j_max) as usize };

        // Backward induction
        for _step in (0..n_steps).rev() {
            let mut new_values = vec![0.0_f64; width];

            for j in -j_max..=j_max {
                let x_j = x0 + j as f64 * dx;
                let r_j = x_j.exp(); // short rate at this node
                let discount = (-r_j * dt).exp();

                let xi = self.a * (mu - x_j) * dt / dx;

                let p_up = 1.0 / 6.0 + (xi * xi + xi) / 2.0;
                let p_mid = 2.0 / 3.0 - xi * xi;
                let p_down = 1.0 / 6.0 + (xi * xi - xi) / 2.0;

                // Determine target nodes — clamp to grid
                let j_up = (j + 1).min(j_max);
                let j_mid = j;
                let j_down = (j - 1).max(-j_max);

                let continuation = p_up * values[idx(j_up)]
                    + p_mid * values[idx(j_mid)]
                    + p_down * values[idx(j_down)];

                new_values[idx(j)] = discount * continuation;
            }
            values = new_values;
        }

        values[idx(0)]
    }

    /// Yield rate from tree-based bond price.
    pub fn yield_rate_tree(&self, maturity: f64, n_steps: usize) -> f64 {
        if maturity <= 0.0 {
            return self.r0;
        }
        let p = self.bond_price_tree(maturity, n_steps);
        -(p.ln()) / maturity
    }
}

impl CalibratedModel for BlackKarasinskiModel {
    fn parameters(&self) -> &[Parameter] {
        &self.params
    }

    fn set_params(&mut self, vals: &[f64]) {
        assert!(
            vals.len() >= 2,
            "BlackKarasinskiModel requires 2 parameters"
        );
        self.params[0].set_value(vals[0]);
        self.params[1].set_value(vals[1]);
        self.a = vals[0];
        self.sigma = vals[1];
    }
}

impl ShortRateModel for BlackKarasinskiModel {
    fn short_rate(&self, _t: f64, x: f64) -> f64 {
        // x is the log-rate; the actual rate is exp(x)
        x.exp()
    }

    fn discount(&self, t: f64) -> f64 {
        // Use the tree with a reasonable number of steps
        self.bond_price_tree(t, 100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_model() -> BlackKarasinskiModel {
        // a=0.1, σ=0.1, r0=0.05, long_rate=0.05
        BlackKarasinskiModel::new(0.1, 0.1, 0.05, 0.05)
    }

    #[test]
    fn bk_bond_price_at_zero() {
        let m = make_model();
        let p = m.bond_price_tree(0.0, 100);
        assert_abs_diff_eq!(p, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn bk_bond_price_positive() {
        let m = make_model();
        for t in [0.5, 1.0, 2.0, 5.0, 10.0] {
            let p = m.bond_price_tree(t, 200);
            assert!(p > 0.0, "Bond price should be positive at t={t}: {p}");
            assert!(p <= 1.0, "Bond price should be <= 1 at t={t}: {p}");
        }
    }

    #[test]
    fn bk_bond_price_decreasing() {
        let m = make_model();
        let p1 = m.bond_price_tree(1.0, 200);
        let p5 = m.bond_price_tree(5.0, 500);
        assert!(p1 > p5, "Bond price should decrease with maturity");
    }

    #[test]
    fn bk_short_rate_positive() {
        // BK rates are always positive (exponential)
        let m = make_model();
        assert!(m.short_rate(0.0, -3.0) > 0.0);
        assert!(m.short_rate(0.0, 0.0) > 0.0);
        assert!(m.short_rate(0.0, 3.0) > 0.0);
    }

    #[test]
    fn bk_yield_at_zero() {
        let m = make_model();
        let y = m.yield_rate_tree(0.001, 100);
        assert_abs_diff_eq!(y, 0.05, epsilon = 0.01);
    }

    #[test]
    fn bk_params() {
        let m = make_model();
        assert_eq!(m.parameters().len(), 2);
        let v = m.params_as_vec();
        assert_abs_diff_eq!(v[0], 0.1);
        assert_abs_diff_eq!(v[1], 0.1);
    }

    #[test]
    fn bk_set_params() {
        let mut m = make_model();
        m.set_params(&[0.2, 0.15]);
        assert_abs_diff_eq!(m.a(), 0.2);
        assert_abs_diff_eq!(m.sigma(), 0.15);
    }

    #[test]
    fn bk_low_vol_approaches_vasicek() {
        // With very low σ, BK behaves like exp(Vasicek), so bond price
        // should approximately match e^{-r0*T} for short maturities.
        let m = BlackKarasinskiModel::new(0.1, 0.001, 0.05, 0.05);
        let p = m.bond_price_tree(1.0, 500);
        let p_approx = (-0.05_f64 * 1.0).exp(); // roughly
        assert_abs_diff_eq!(p, p_approx, epsilon = 0.01);
    }

    #[test]
    fn bk_discount_matches_tree() {
        // discount() should use the tree internally
        let m = make_model();
        let d = m.discount(2.0);
        let p = m.bond_price_tree(2.0, 100);
        assert_abs_diff_eq!(d, p, epsilon = 1e-10);
    }
}
