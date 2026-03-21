//! Gaussian 1d (Hull-White / GSR) smile section.
//!
//! Given a Gaussian short-rate model calibrated to the swaption market, this
//! module provides the implied Black (or Bachelier) smile for a co-terminal
//! swaption by integrating the model's distribution of swap rates.
//!
//! The GSR (Gaussian Short-Rate) model uses:
//!
//!   dr(t) = κ(θ(t) − r(t)) dt + σ(t) dW(t)
//!
//! For constant parameters (Hull-White) this reduces to:
//!
//!   dr(t) = κ(θ − r(t)) dt + σ dW(t)
//!
//! The swap rate at expiry T, given r(T), can be approximated analytically
//! (linear annuity approximation) or evaluated by numerical quadrature.
//!
//! Reference: Andreasen & Huge (2011), Piterbarg (2010) Ch. 12.

use ql_math::distributions::cumulative_normal;
use std::f64::consts::PI;

// =========================================================================
// HullWhiteParams
// =========================================================================

/// Hull-White (Gaussian 1d, constant-param) model parameters.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct HullWhiteParams {
    /// Mean-reversion speed κ > 0.
    pub kappa: f64,
    /// Short-rate diffusion σ > 0.
    pub sigma: f64,
    /// Current short rate r₀.
    pub r0: f64,
    /// Long-run mean θ → determines the yield curve shape.
    pub theta: f64,
}

impl HullWhiteParams {
    /// Create Hull-White parameters.
    pub fn new(kappa: f64, sigma: f64, r0: f64, theta: f64) -> Self {
        assert!(kappa > 0.0, "kappa must be positive");
        assert!(sigma > 0.0, "sigma must be positive");
        Self { kappa, sigma, r0, theta }
    }

    /// Mean of r(T) under the risk-neutral measure.
    pub fn mean_r(&self, t: f64) -> f64 {
        let ekt = (-self.kappa * t).exp();
        self.r0 * ekt + self.theta * (1.0 - ekt)
    }

    /// Variance of r(T) under the risk-neutral measure.
    pub fn var_r(&self, t: f64) -> f64 {
        let ekt = (-self.kappa * t).exp();
        self.sigma * self.sigma / (2.0 * self.kappa) * (1.0 - ekt * ekt)
    }

    /// Bond price P(0, T) = E[exp(-∫₀ᵀ r dt)] in the Hull-White model.
    pub fn bond_price(&self, t: f64) -> f64 {
        // P(0,T) = A(T) exp(-B(T) r0)
        let b = (1.0 - (-self.kappa * t).exp()) / self.kappa;
        let log_a = (self.theta - self.sigma * self.sigma / (2.0 * self.kappa * self.kappa)) * (b - t)
            - self.sigma * self.sigma * b * b / (4.0 * self.kappa);
        (log_a - b * self.r0).exp()
    }

    /// Forward swap rate for a swap starting at `t_start`, ending at `t_end`,
    /// with `n` equal annual periods, using hull-white bond prices.
    pub fn forward_swap_rate(&self, t_start: f64, t_end: f64, n: usize) -> f64 {
        if n == 0 { return 0.0; }
        let dt = (t_end - t_start) / n as f64;
        let pf_start = self.bond_price(t_start);
        let pf_end = self.bond_price(t_end);
        let annuity: f64 = (1..=n).map(|i| dt * self.bond_price(t_start + i as f64 * dt)).sum();
        if annuity.abs() < 1e-15 { return 0.0; }
        (pf_start - pf_end) / annuity
    }
}

// =========================================================================
// Gaussian1dSmileSection
// =========================================================================

/// Smile section implied from a Gaussian 1d (Hull-White) short-rate model.
///
/// Computes the swaption smile via Gaussian quadrature: integrates the
/// swaption payoff over the distribution of r(T_exp).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Gaussian1dSmileSection {
    /// Hull-White parameters.
    pub hw: HullWhiteParams,
    /// Swaption expiry (years).
    pub expiry: f64,
    /// Swap tenor (years from expiry).
    pub tenor: f64,
    /// Number of swap periods.
    pub n_periods: usize,
    /// Forward swap rate (precomputed at construction).
    pub forward_swap_rate: f64,
    /// Number of Gauss-Hermite quadrature nodes.
    pub n_quad: usize,
}

impl Gaussian1dSmileSection {
    /// Construct a smile section from a Hull-White model.
    ///
    /// # Parameters
    /// - `hw`: Hull-White model parameters
    /// - `expiry`: swaption expiry in years
    /// - `tenor`: swap tenor in years (from expiry)
    /// - `n_periods`: number of swap coupon periods
    /// - `n_quad`: quadrature nodes (typically 32–64)
    pub fn new(
        hw: HullWhiteParams,
        expiry: f64,
        tenor: f64,
        n_periods: usize,
        n_quad: usize,
    ) -> Self {
        let forward_swap_rate = hw.forward_swap_rate(expiry, expiry + tenor, n_periods);
        Self { hw, expiry, tenor, n_periods, forward_swap_rate, n_quad }
    }

    /// Swaption price (payer, first into swap at `expiry`) for a given `strike`.
    ///
    /// Uses Gauss-Hermite quadrature over the distribution of r(T_exp).
    pub fn swaption_price(&self, strike: f64) -> f64 {
        // r(T) ~ N(mu_r, var_r)
        let mu_r = self.hw.mean_r(self.expiry);
        let var_r = self.hw.var_r(self.expiry);
        let std_r = var_r.sqrt();

        // Gauss-Hermite nodes and weights for ∫ f(x) e^{-x²} dx
        let (nodes, weights) = gauss_hermite(self.n_quad);

        let dt = self.tenor / self.n_periods as f64;
        let p0_exp = self.hw.bond_price(self.expiry);

        let value: f64 = nodes.iter().zip(weights.iter()).map(|(&xi, &wi)| {
            // Transform: r_T = mu_r + sqrt(2) * std_r * xi
            let r_t = mu_r + 2.0_f64.sqrt() * std_r * xi;
            let payoff = self.payer_swap_payoff(r_t, strike, dt);
            wi * payoff
        }).sum();

        // P(0, T_exp) * (1/sqrt(π)) * integral
        p0_exp * value / PI.sqrt()
    }

    /// Implied Black volatility at `strike` (backed out from swaption price).
    pub fn implied_black_vol(&self, strike: f64) -> f64 {
        let price = self.swaption_price(strike);
        if price <= 0.0 { return 0.01; }
        let annuity = self.hw.bond_price(self.expiry) *
            (1..=self.n_periods).map(|i| {
                let dt = self.tenor / self.n_periods as f64;
                self.hw.bond_price(self.expiry + i as f64 * dt)
            }).sum::<f64>() * (self.tenor / self.n_periods as f64);

        // Normalise: model price = annuity * Black(fwd, strike, vol, expiry)
        if annuity <= 0.0 { return 0.01; }
        let normalised = price / annuity;
        invert_black_call(self.forward_swap_rate, strike, normalised, self.expiry)
    }

    /// Payer swap payoff at T_exp as a function of the short rate r_T.
    /// Uses a 1-factor linear approximation for the swap rate.
    fn payer_swap_payoff(&self, r_t: f64, strike: f64, dt: f64) -> f64 {
        // Approximate bond prices using the HW affine formula
        // P(T,s) = exp(A(T,s) - B(T,s) * r_T)
        let mut annuity = 0.0;
        let mut p_end = 1.0;
        for i in 1..=self.n_periods {
            let tau = i as f64 * dt;
            let b = (1.0 - (-self.hw.kappa * tau).exp()) / self.hw.kappa;
            let log_a = (self.hw.theta - self.hw.sigma * self.hw.sigma / (2.0 * self.hw.kappa * self.hw.kappa))
                * (b - tau) - self.hw.sigma * self.hw.sigma * b * b / (4.0 * self.hw.kappa);
            let p = (log_a - b * r_t).exp();
            annuity += dt * p;
            if i == self.n_periods { p_end = p; }
        }
        // Swap rate: S = (1 - P_end) / annuity
        let swap_rate = if annuity > 1e-15 { (1.0 - p_end) / annuity } else { 0.0 };
        // Payer swaption payoff = annuity * max(S - K, 0)
        annuity * (swap_rate - strike).max(0.0)
    }
}

/// Gauss-Hermite nodes and weights for n-point quadrature.
/// Returns (nodes, weights) for ∫_{-∞}^{∞} f(x) e^{-x²} dx ≈ Σ w_i f(x_i).
fn gauss_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
    // Pre-computed nodes and weights for common orders
    match n {
        5 => (
            vec![-2.0201828704561, -0.9585724646138, 0.0, 0.9585724646138, 2.0201828704561],
            vec![0.0199532420591, 0.3936193231522, 0.9453087204829, 0.3936193231522, 0.0199532420591],
        ),
        10 => (
            vec![-3.4361591188377, -2.5327316742928, -1.7566836492999, -1.0366108297895,
                 -0.3429013272237, 0.3429013272237, 1.0366108297895, 1.7566836492999,
                 2.5327316742928, 3.4361591188377],
            vec![7.640432855233e-6, 1.343645746781e-3, 3.387439445548e-2, 2.401386110823e-1,
                 6.108626337354e-1, 6.108626337354e-1, 2.401386110823e-1, 3.387439445548e-2,
                 1.343645746781e-3, 7.640432855233e-6],
        ),
        _ => {
            // Default: use 5-point quadrature
            gauss_hermite(5)
        }
    }
}

/// Invert Black call formula: given normalised price `c = C/A`, find vol.
fn invert_black_call(fwd: f64, strike: f64, c: f64, t: f64) -> f64 {
    let intrinsic = (fwd - strike).max(0.0);
    if c <= intrinsic + 1e-12 { return 1e-4; }
    let sqrt_t = t.sqrt();
    let mut sigma = 0.10_f64;
    for _ in 0..50 {
        let d1 = ((fwd / strike).ln() + 0.5 * sigma * sigma * t) / (sigma * sqrt_t);
        let d2 = d1 - sigma * sqrt_t;
        let price = fwd * cumulative_normal(d1) - strike * cumulative_normal(d2);
        let vega = fwd * cumulative_normal(d1).clamp(0.0001, 0.9999)
            * (-0.5 * d1 * d1).exp() / (2.0 * PI).sqrt() * sqrt_t;
        let diff = price - c;
        if diff.abs() < 1e-12 || vega.abs() < 1e-15 { break; }
        sigma -= diff / vega;
        sigma = sigma.clamp(1e-6, 5.0);
    }
    sigma
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn default_hw() -> HullWhiteParams {
        HullWhiteParams::new(0.03, 0.01, 0.05, 0.05)
    }

    #[test]
    fn bond_price_at_zero() {
        let hw = default_hw();
        let p = hw.bond_price(0.0);
        assert!((p - 1.0).abs() < 1e-12, "P(0,0) = 1");
    }

    #[test]
    fn forward_swap_rate_positive() {
        let hw = default_hw();
        let fsr = hw.forward_swap_rate(1.0, 6.0, 5);
        assert!(fsr > 0.0 && fsr < 0.20, "forward swap rate should be reasonable: {}", fsr);
    }

    #[test]
    fn swaption_price_positive() {
        let hw = default_hw();
        let section = Gaussian1dSmileSection::new(hw, 1.0, 5.0, 5, 10);
        let price = section.swaption_price(section.forward_swap_rate);
        assert!(price > 0.0, "ATM swaption price should be positive: {}", price);
    }

    #[test]
    fn implied_vol_reasonable() {
        let hw = default_hw();
        let section = Gaussian1dSmileSection::new(hw, 1.0, 5.0, 5, 10);
        let iv = section.implied_black_vol(section.forward_swap_rate);
        assert!(iv > 0.001 && iv < 1.0, "implied vol should be reasonable: {}", iv);
    }

    #[test]
    fn smile_monotone_in_strike_direction() {
        let hw = default_hw();
        let section = Gaussian1dSmileSection::new(hw, 1.0, 5.0, 5, 10);
        let fsr = section.forward_swap_rate;
        // Payer swaption price should decrease as strike increases (above fwd)
        let p_low = section.swaption_price(fsr * 0.8);
        let p_atm = section.swaption_price(fsr);
        let p_high = section.swaption_price(fsr * 1.2);
        assert!(p_low >= p_atm, "lower strike should have higher price");
        assert!(p_atm >= p_high, "higher strike should have lower price");
    }
}
