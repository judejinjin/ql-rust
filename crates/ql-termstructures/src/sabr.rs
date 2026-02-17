//! SABR stochastic volatility model for smile interpolation.
//!
//! Implements the Hagan et al. (2002) SABR implied volatility formula
//! and a smile section class for a single expiry.


/// Compute the SABR implied Black volatility for a given strike.
///
/// Uses the Hagan et al. (2002) closed-form approximation.
///
/// # Parameters
/// - `strike`: option strike (must be > 0)
/// - `forward`: forward price (must be > 0)
/// - `expiry`: time to expiry in years
/// - `alpha`: SABR alpha (initial vol level)
/// - `beta`: SABR beta (0 = normal, 1 = log-normal)
/// - `rho`: SABR rho (correlation between spot and vol, in \[-1, 1\])
/// - `nu`: SABR nu (vol of vol)
///
/// # Returns
/// Black implied volatility
#[allow(clippy::too_many_arguments)]
pub fn sabr_volatility(
    strike: f64,
    forward: f64,
    expiry: f64,
    alpha: f64,
    beta: f64,
    rho: f64,
    nu: f64,
) -> f64 {
    assert!(strike > 0.0, "Strike must be positive");
    assert!(forward > 0.0, "Forward must be positive");
    assert!(expiry > 0.0, "Expiry must be positive");
    assert!(alpha > 0.0, "Alpha must be positive");
    assert!(
        (-1.0..=1.0).contains(&rho),
        "Rho must be in [-1, 1]"
    );
    assert!(nu >= 0.0, "Nu must be non-negative");

    // ATM case
    if (strike - forward).abs() < 1e-12 * forward {
        return sabr_atm_vol(forward, expiry, alpha, beta, rho, nu);
    }

    let one_minus_beta = 1.0 - beta;
    let fk = forward * strike;
    let fk_beta = fk.powf(one_minus_beta);
    let log_fk = (forward / strike).ln();

    // z = (nu / alpha) * (FK)^((1-β)/2) * ln(F/K)
    let z = (nu / alpha) * fk_beta.sqrt() * log_fk;

    // x(z) = ln[(√(1 - 2ρz + z²) + z - ρ) / (1 - ρ)]
    let x_z = if nu.abs() < 1e-12 {
        1.0
    } else {
        let sqrt_term = (1.0 - 2.0 * rho * z + z * z).sqrt();
        let numerator = sqrt_term + z - rho;
        let denominator = 1.0 - rho;
        if numerator.abs() < 1e-12 {
            1.0
        } else {
            z / (numerator / denominator).ln()
        }
    };

    // Leading term: alpha / [(FK)^((1-β)/2) * (1 + (1-β)²/24 * ln²(F/K) + (1-β)⁴/1920 * ln⁴(F/K))]
    let fk_mid = fk.powf(one_minus_beta / 2.0);
    let log2 = log_fk * log_fk;
    let log4 = log2 * log2;
    let denom = fk_mid
        * (1.0
            + one_minus_beta * one_minus_beta / 24.0 * log2
            + one_minus_beta.powi(4) / 1920.0 * log4);

    let leading = alpha / denom;

    // Correction term:
    // 1 + T * [(1-β)²/24 * α²/(FK)^(1-β) + ¼ * ρβνα/(FK)^((1-β)/2) + (2-3ρ²)/24 * ν²]
    let correction = 1.0
        + expiry
            * (one_minus_beta * one_minus_beta / 24.0 * alpha * alpha / fk_beta
                + 0.25 * rho * beta * nu * alpha / fk_mid
                + (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu);

    leading * x_z * correction
}

/// ATM SABR volatility (simplified formula when F = K).
fn sabr_atm_vol(
    forward: f64,
    expiry: f64,
    alpha: f64,
    beta: f64,
    rho: f64,
    nu: f64,
) -> f64 {
    let one_minus_beta = 1.0 - beta;
    let f_mid = forward.powf(one_minus_beta);

    let leading = alpha / f_mid;
    let correction = 1.0
        + expiry
            * (one_minus_beta * one_minus_beta / 24.0 * alpha * alpha / (f_mid * f_mid)
                + 0.25 * rho * beta * nu * alpha / f_mid
                + (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu);

    let _ = leading; // The leading term for ATM is just alpha/f_mid (no log term)
    leading * correction
}

/// A SABR smile section for a single expiry.
///
/// Given calibrated SABR parameters and a forward, provides Black vols
/// at any strike for the associated expiry.
#[derive(Debug, Clone)]
pub struct SabrSmileSection {
    /// Forward price.
    pub forward: f64,
    /// Time to expiry (years).
    pub expiry: f64,
    /// SABR alpha.
    pub alpha: f64,
    /// SABR beta.
    pub beta: f64,
    /// SABR rho.
    pub rho: f64,
    /// SABR nu (vol of vol).
    pub nu: f64,
}

impl SabrSmileSection {
    /// Create a new SABR smile section.
    pub fn new(forward: f64, expiry: f64, alpha: f64, beta: f64, rho: f64, nu: f64) -> Self {
        Self {
            forward,
            expiry,
            alpha,
            beta,
            rho,
            nu,
        }
    }

    /// Black implied volatility at a given strike.
    pub fn volatility(&self, strike: f64) -> f64 {
        sabr_volatility(
            strike,
            self.forward,
            self.expiry,
            self.alpha,
            self.beta,
            self.rho,
            self.nu,
        )
    }

    /// Variance at a given strike: σ²·T.
    pub fn variance(&self, strike: f64) -> f64 {
        let v = self.volatility(strike);
        v * v * self.expiry
    }

    /// ATM volatility.
    pub fn atm_vol(&self) -> f64 {
        self.volatility(self.forward)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn sabr_atm_equals_alpha_for_beta_1_no_volvol() {
        // For beta=1, nu=0, rho=0: σ(F,F) = alpha
        let vol = sabr_volatility(100.0, 100.0, 1.0, 0.20, 1.0, 0.0, 0.0);
        assert_abs_diff_eq!(vol, 0.20, epsilon = 1e-10);
    }

    #[test]
    fn sabr_smile_has_skew() {
        // With negative rho, OTM puts should have higher vol than OTM calls
        let forward = 100.0;
        let alpha = 0.20;
        let beta = 0.5;
        let rho = -0.3;
        let nu = 0.4;
        let expiry = 1.0;

        let vol_low = sabr_volatility(80.0, forward, expiry, alpha, beta, rho, nu);
        let vol_atm = sabr_volatility(forward, forward, expiry, alpha, beta, rho, nu);
        let vol_high = sabr_volatility(120.0, forward, expiry, alpha, beta, rho, nu);

        // Negative rho => downside skew
        assert!(vol_low > vol_atm, "OTM put vol {vol_low} should exceed ATM vol {vol_atm}");
        // Smile: wings should be higher than ATM in general
        assert!(vol_low > vol_high, "With rho<0, low strike vol should exceed high strike vol");
    }

    #[test]
    fn sabr_zero_volvol_gives_shifted_lognormal() {
        // With nu=0, SABR reduces to a shifted lognormal (CEV) model
        // Vol should be roughly alpha / F^(1-beta) for ATM
        let forward = 100.0;
        let alpha = 3.0;  // normal vol units for beta=0
        let beta = 0.0;
        let rho = 0.0;
        let nu = 0.0;
        let expiry = 1.0;

        let vol = sabr_volatility(100.0, forward, expiry, alpha, beta, rho, nu);
        // For beta=0, sigma_B ≈ alpha/F for ATM (plus small correction from alpha²/F² term)
        let expected = alpha / forward;
        assert_abs_diff_eq!(vol, expected, epsilon = 1e-4);
    }

    #[test]
    fn sabr_symmetry_for_zero_rho() {
        // With rho=0, the smile should be roughly symmetric around ATM
        let forward = 100.0;
        let alpha = 0.20;
        let beta = 1.0;
        let rho = 0.0;
        let nu = 0.4;
        let expiry = 1.0;

        let vol_90 = sabr_volatility(90.0, forward, expiry, alpha, beta, rho, nu);
        let vol_110 = sabr_volatility(110.0, forward, expiry, alpha, beta, rho, nu);

        // Not exactly symmetric due to higher-order terms, but close
        assert_abs_diff_eq!(vol_90, vol_110, epsilon = 0.005);
    }

    #[test]
    fn smile_section_atm() {
        let section = SabrSmileSection::new(100.0, 1.0, 0.20, 1.0, -0.2, 0.3);
        let atm = section.atm_vol();
        let at_forward = section.volatility(100.0);
        assert_abs_diff_eq!(atm, at_forward, epsilon = 1e-12);
    }

    #[test]
    fn smile_section_variance() {
        let section = SabrSmileSection::new(100.0, 2.0, 0.20, 0.5, 0.0, 0.3);
        let vol = section.volatility(100.0);
        let var = section.variance(100.0);
        assert_abs_diff_eq!(var, vol * vol * 2.0, epsilon = 1e-12);
    }

    #[test]
    fn sabr_vol_positive_for_various_params() {
        let cases = vec![
            (80.0, 100.0, 0.5, 0.3, 0.7, -0.5, 0.6),
            (100.0, 100.0, 1.0, 0.2, 1.0, 0.0, 0.3),
            (120.0, 100.0, 2.0, 0.1, 0.3, 0.3, 0.5),
            (95.0, 100.0, 0.25, 0.5, 0.5, -0.7, 0.8),
        ];
        for (k, f, t, alpha, beta, rho, nu) in cases {
            let vol = sabr_volatility(k, f, t, alpha, beta, rho, nu);
            assert!(
                vol > 0.0,
                "SABR vol should be positive for K={k}, F={f}, T={t}"
            );
        }
    }
}
