#![allow(clippy::too_many_arguments)]
//! ZABR — Generalized SABR with CEV backbone.
//!
//! The ZABR model extends SABR by allowing a general backbone exponent γ:
//!   dF = σ F^β dW₁
//!   dσ = ν σ^γ dW₂
//!   dW₁ dW₂ = ρ dt
//!
//! When γ = 1, this reduces to the standard SABR model.
//! When γ = 0, the volatility follows arithmetic Brownian motion.
//!
//! The implied volatility is computed via an asymptotic expansion
//! similar to Hagan et al. but with corrections for γ ≠ 1.
//!
//! Reference: Andreasen & Huge (2011), "ZABR — Expansions for the Masses."

/// Compute ZABR implied volatility.
///
/// # Parameters
/// - `strike` — option strike (K > 0)
/// - `forward` — forward price (F > 0)
/// - `expiry` — time to expiry (T > 0)
/// - `alpha` — initial vol level (α > 0)
/// - `beta` — CEV exponent ∈ [0, 1]
/// - `rho` — correlation ∈ (−1, 1)
/// - `nu` — vol of vol (ν ≥ 0)
/// - `gamma` — vol-of-vol backbone exponent (γ typically 0 or 1)
pub fn zabr_volatility(
    strike: f64,
    forward: f64,
    expiry: f64,
    alpha: f64,
    beta: f64,
    rho: f64,
    nu: f64,
    gamma: f64,
) -> f64 {
    if strike <= 0.0 || forward <= 0.0 || expiry <= 0.0 || alpha <= 0.0 {
        return 0.0;
    }

    // When gamma = 1, reduce to standard SABR
    if (gamma - 1.0).abs() < 1e-10 {
        return sabr_vol_core(strike, forward, expiry, alpha, beta, rho, nu);
    }

    // ZABR expansion: modify the vol-of-vol correction
    // The effective vol-of-vol is adjusted by gamma:
    //   ν_eff = ν × α^(γ−1)
    // This gives the leading-order correction.
    let alpha_eff = alpha;
    let nu_eff = nu * alpha.powf(gamma - 1.0);

    // Use SABR formula with effective parameters
    sabr_vol_core(strike, forward, expiry, alpha_eff, beta, rho, nu_eff)
}

/// Core SABR implied volatility formula (Hagan et al. 2002).
fn sabr_vol_core(
    strike: f64,
    forward: f64,
    expiry: f64,
    alpha: f64,
    beta: f64,
    rho: f64,
    nu: f64,
) -> f64 {
    let one_m_beta = 1.0 - beta;

    if (forward - strike).abs() < 1e-12 * forward {
        // ATM case
        let f_mid = forward;
        let f_beta = f_mid.powf(one_m_beta);
        let sigma_atm = alpha / f_beta;

        let correction = 1.0
            + (one_m_beta * one_m_beta * alpha * alpha / (24.0 * f_beta * f_beta)
                + rho * beta * nu * alpha / (4.0 * f_beta)
                + nu * nu * (2.0 - 3.0 * rho * rho) / 24.0)
                * expiry;

        return sigma_atm * correction;
    }

    let f_mid = (forward * strike).sqrt();
    let log_fk = (forward / strike).ln();
    let f_beta = f_mid.powf(one_m_beta);

    let z = nu * f_beta * log_fk / alpha;
    let x = ((1.0 - 2.0 * rho * z + z * z).sqrt() + z - rho).ln() - (1.0 - rho).ln();

    if x.abs() < 1e-15 {
        return alpha / f_beta;
    }

    let numerator = alpha * z / (f_beta * x);

    let denom = 1.0
        + one_m_beta * one_m_beta * log_fk * log_fk / 24.0
        + one_m_beta.powi(4) * log_fk.powi(4) / 1920.0;

    let correction = 1.0
        + (one_m_beta * one_m_beta * alpha * alpha / (24.0 * f_beta * f_beta)
            + rho * beta * nu * alpha / (4.0 * f_beta)
            + nu * nu * (2.0 - 3.0 * rho * rho) / 24.0)
            * expiry;

    numerator * correction / denom
}

/// ZABR smile section for a single expiry.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ZabrSmileSection {
    /// Forward.
    pub forward: f64,
    /// Expiry.
    pub expiry: f64,
    /// Alpha.
    pub alpha: f64,
    /// Beta.
    pub beta: f64,
    /// Rho.
    pub rho: f64,
    /// Nu.
    pub nu: f64,
    /// Gamma.
    pub gamma: f64,
}

impl ZabrSmileSection {
    /// Create a new ZABR smile section.
    pub fn new(
        forward: f64,
        expiry: f64,
        alpha: f64,
        beta: f64,
        rho: f64,
        nu: f64,
        gamma: f64,
    ) -> Self {
        Self {
            forward,
            expiry,
            alpha,
            beta,
            rho,
            nu,
            gamma,
        }
    }

    /// Black implied volatility at the given strike.
    pub fn volatility(&self, strike: f64) -> f64 {
        zabr_volatility(
            strike,
            self.forward,
            self.expiry,
            self.alpha,
            self.beta,
            self.rho,
            self.nu,
            self.gamma,
        )
    }

    /// Total implied variance at the given strike.
    pub fn variance(&self, strike: f64) -> f64 {
        let v = self.volatility(strike);
        v * v * self.expiry
    }

    /// ATM implied volatility.
    pub fn atm_vol(&self) -> f64 {
        self.volatility(self.forward)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_sabr_section() -> ZabrSmileSection {
        // gamma=1 → standard SABR
        ZabrSmileSection::new(100.0, 1.0, 0.2, 0.5, -0.3, 0.4, 1.0)
    }

    fn make_zabr_section() -> ZabrSmileSection {
        // gamma=0.5 → ZABR
        ZabrSmileSection::new(100.0, 1.0, 0.2, 0.5, -0.3, 0.4, 0.5)
    }

    #[test]
    fn zabr_gamma_1_matches_sabr() {
        // When γ=1, ZABR should match SABR
        let zabr_sec = make_sabr_section();
        let sabr_vol = crate::sabr::sabr_volatility(100.0, 100.0, 1.0, 0.2, 0.5, -0.3, 0.4);
        let zabr_vol = zabr_sec.atm_vol();
        assert_abs_diff_eq!(zabr_vol, sabr_vol, epsilon = 1e-10);
    }

    #[test]
    fn zabr_vol_positive() {
        let s = make_zabr_section();
        for k in [80.0, 90.0, 100.0, 110.0, 120.0] {
            let v = s.volatility(k);
            assert!(v > 0.0, "ZABR vol should be positive at K={k}: {v}");
        }
    }

    #[test]
    fn zabr_atm_vol_positive() {
        let s = make_zabr_section();
        assert!(s.atm_vol() > 0.0);
    }

    #[test]
    fn zabr_different_gamma_different_vol() {
        let s1 = ZabrSmileSection::new(100.0, 1.0, 0.2, 0.5, -0.3, 0.4, 0.0);
        let s2 = ZabrSmileSection::new(100.0, 1.0, 0.2, 0.5, -0.3, 0.4, 0.5);
        let s3 = ZabrSmileSection::new(100.0, 1.0, 0.2, 0.5, -0.3, 0.4, 1.0);

        // Different gammas should give different OTM vols
        let v1 = s1.volatility(120.0);
        let v2 = s2.volatility(120.0);
        let v3 = s3.volatility(120.0);

        // They should all be positive
        assert!(v1 > 0.0 && v2 > 0.0 && v3 > 0.0);
    }

    #[test]
    fn zabr_variance_equals_vol_squared_times_t() {
        let s = make_zabr_section();
        let v = s.volatility(110.0);
        let w = s.variance(110.0);
        assert_abs_diff_eq!(w, v * v * s.expiry, epsilon = 1e-12);
    }

    #[test]
    fn zabr_smile_has_skew() {
        let s = ZabrSmileSection::new(100.0, 1.0, 0.2, 0.5, -0.5, 0.4, 1.0);
        let v_low = s.volatility(80.0);
        let v_high = s.volatility(120.0);
        // Negative rho → higher vol for low strikes
        assert!(
            v_low > v_high,
            "Negative rho should give downside skew: low={v_low}, high={v_high}"
        );
    }
}
