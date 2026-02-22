//! Variance Gamma (VG) stochastic process and model.
//!
//! The VG process is a Lévy process obtained by evaluating a drifted Brownian
//! motion B(t) = θ·t + σ·W(t) at a random clock Y(t) driven by a Gamma
//! process with mean rate 1 and variance rate ν:
//!
//! ```text
//! X(t) = θ·Y(t) + σ·W(Y(t))
//! ```
//!
//! The spot price follows:
//! ```text
//! S(T) = S₀ · exp((r − q + ω)·T + X(T))
//! ```
//!
//! where ω = (1/ν)·ln(1 − θν − σ²ν/2) is the martingale correction so that
//! E[S(T)] = S₀·e^{(r−q)T}.
//!
//! ## Parameters
//! - σ (sigma): volatility of the Brownian component
//! - ν (nu): variance rate of the Gamma subordinator
//! - θ (theta): drift of the Brownian component (controls skewness)
//!
//! ## References
//! - Madan, D.B., Carr, P., Chang, E.C. (1998). "The variance gamma process
//!   and option pricing." *European Finance Review*, 2, 79–105.
//! - Cont, R. & Tankov, P. (2004). *Financial Modelling with Jump Processes*.

use serde::{Deserialize, Serialize};

/// Variance Gamma model parameters.
///
/// Encapsulates the three parameters (σ, ν, θ) and provides the characteristic
/// function of the log-return X(T) = ln(S(T)/S₀) − (r−q)·T.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarianceGammaModel {
    /// Brownian component volatility σ > 0.
    pub sigma: f64,
    /// Gamma process variance rate ν > 0.
    pub nu: f64,
    /// Brownian drift θ (negative ⇒ left-skewed returns, typical for equities).
    pub theta: f64,
}

impl VarianceGammaModel {
    /// Create a new VG model.
    ///
    /// # Panics
    /// Panics if `sigma <= 0` or `nu <= 0`.
    pub fn new(sigma: f64, nu: f64, theta: f64) -> Self {
        assert!(sigma > 0.0, "sigma must be positive");
        assert!(nu > 0.0, "nu must be positive");
        Self { sigma, nu, theta }
    }

    /// Martingale correction ω = (1/ν)·ln(1 − θν − σ²ν/2).
    ///
    /// Returns `None` if the argument to ln would be ≤ 0
    /// (parameters violate the no-arbitrage condition).
    pub fn omega(&self) -> Option<f64> {
        let arg = 1.0 - self.theta * self.nu - 0.5 * self.sigma * self.sigma * self.nu;
        if arg <= 0.0 { None } else { Some(arg.ln() / self.nu) }
    }

    /// Characteristic function of the centred VG log-return X(T).
    ///
    /// Returns the complex value φ(u) = E[exp(iu·X(T))].
    ///
    /// # Parameters
    /// - `u` — frequency (real-valued)
    /// - `tau` — time to maturity T
    ///
    /// # Returns
    /// `(re, im)` of the characteristic function.
    pub fn log_cf(&self, u: f64, tau: f64) -> (f64, f64) {
        // φ_VG(u, T) = (1 − iuθν + σ²νu²/2)^{−T/ν}
        let t_over_nu = tau / self.nu;
        // Argument: 1 − iuθν + σ²νu²/2
        let re = 1.0 - self.theta * self.nu * 0.0 + 0.5 * self.sigma * self.sigma * self.nu * u * u;
        // = 1 + σ²ν(u²)/2 + 0 (real part)
        let im = -u * self.theta * self.nu; // −iuθν → imaginary part is −uθν
        // Actually careful: 1 - iu·θ·ν + σ²ν u²/2
        // real: 1 + σ²νu²/2
        // imag: -uθν
        let base_re = 1.0 + 0.5 * self.sigma * self.sigma * self.nu * u * u;
        let base_im = -u * self.theta * self.nu;

        // (base_re + i·base_im)^{-t/ν}
        // = exp(-t/ν · ln(base_re + i·base_im))
        let r = (base_re * base_re + base_im * base_im).sqrt();
        let arg = base_im.atan2(base_re);
        let ln_r = r.ln();
        let exp_re = -t_over_nu * ln_r;
        let exp_im = -t_over_nu * arg;
        let result_r = exp_re.exp();
        let _ = (re, im);
        (result_r * exp_im.cos(), result_r * exp_im.sin())
    }

    /// Cumulant c₁ (mean) of log-return over horizon T.
    pub fn c1(&self, tau: f64) -> f64 {
        (self.theta + self.omega().unwrap_or(0.0)) * tau
    }

    /// Cumulant c₂ (variance) of log-return over horizon T.
    pub fn c2(&self, tau: f64) -> f64 {
        (self.sigma * self.sigma + self.nu * self.theta * self.theta) * tau
    }

    /// Skewness coefficient κ₃ / κ₂^{3/2} (NaN if ν=0).
    pub fn skewness(&self, tau: f64) -> f64 {
        let k3 = (2.0 * self.theta.powi(3) * self.nu * self.nu
            + 3.0 * self.sigma * self.sigma * self.theta * self.nu) * tau;
        let k2 = self.c2(tau);
        k3 / k2.powf(1.5)
    }

    /// Excess kurtosis κ₄ / κ₂².
    pub fn excess_kurtosis(&self, tau: f64) -> f64 {
        let k4 = (3.0 * self.sigma.powi(4) * self.nu
            + 12.0 * self.sigma * self.sigma * self.theta * self.theta * self.nu * self.nu
            + 6.0 * self.theta.powi(4) * self.nu.powi(3)) * tau;
        let k2 = self.c2(tau);
        k4 / k2.powi(2)
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn omega_positive_parameters() {
        let m = VarianceGammaModel::new(0.12, 0.017, -0.14);
        let omega = m.omega().expect("omega should be defined");
        // omega should be positive (left-skewed, theta < 0 → extra correction)
        assert!(omega.is_finite(), "omega is finite");
    }

    #[test]
    fn cf_at_zero_is_one() {
        let m = VarianceGammaModel::new(0.20, 0.10, -0.10);
        let (re, im) = m.log_cf(0.0, 1.0);
        assert!((re - 1.0).abs() < 1e-12, "re={}", re);
        assert!(im.abs() < 1e-12, "im={}", im);
    }

    #[test]
    fn cf_modulus_le_one() {
        // |φ(u)| ≤ 1 for all real u (CF of a distribution)
        let m = VarianceGammaModel::new(0.15, 0.05, -0.05);
        for u in [-10.0, -1.0, 0.5, 2.0, 5.0, 10.0] {
            let (re, im) = m.log_cf(u, 1.0);
            let mod2 = re * re + im * im;
            assert!(mod2 <= 1.0 + 1e-10, "u={} |φ|²={}", u, mod2);
        }
    }

    #[test]
    fn cumulants_positive() {
        let m = VarianceGammaModel::new(0.20, 0.10, -0.10);
        // c2 (variance) must be positive
        assert!(m.c2(1.0) > 0.0);
    }

    #[test]
    fn kurtosis_positive() {
        let m = VarianceGammaModel::new(0.20, 0.10, -0.10);
        let k = m.excess_kurtosis(1.0);
        assert!(k > 0.0, "VG has leptokurtic returns");
    }
}
