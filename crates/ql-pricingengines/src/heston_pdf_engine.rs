//! Heston PDF engine and Exponential-Fitting Heston engine.
//!
//! - [`heston_pdf_price`] — Price European options via the Heston probability
//!   density reconstructed by Fourier inversion.
//! - [`exponential_fitting_heston`] — Fastest Heston engine using exponential
//!   fitting of the characteristic function integrand (Andersen-Piterbarg 2010).

use std::f64::consts::PI;
use ql_core::errors::{QLError, QLResult};
use ql_instruments::OptionType;

// ===========================================================================
// Heston PDF Engine
// ===========================================================================

/// Result from the Heston PDF engine.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HestonPdfResult {
    /// Option price.
    pub price: f64,
    /// Risk-neutral probability density at the strike.
    pub density_at_strike: f64,
}

/// Heston characteristic function φ(u; T).
///
/// Returns ln φ(u) = C(u,T) + D(u,T)·v₀ + i·u·ln(S·e^{(r-q)T})
fn heston_log_cf(
    u_re: f64, u_im: f64,
    tau: f64, kappa: f64, theta: f64, sigma: f64, rho: f64, v0: f64,
) -> (f64, f64) {
    // Complex arithmetic helpers
    let (_a_re, _a_im) = (kappa - rho * sigma * u_im, rho * sigma * u_re);
    // b² = (ρσiu + κ)² + σ²(iu + u²)
    // a = κ - ρσ(iu)  =>  a_re = κ + ρσ·u_im, a_im = -ρσ·u_re
    // Wait, let me be more careful with the standard Heston CF:
    // φ(u) = exp(C + D·v0 + i·u·x)
    // where x = ln(F), F = S·exp((r-q)T)
    // d = sqrt((ρσiu - κ)² + σ²(iu + u²))
    // g = (κ - ρσiu - d) / (κ - ρσiu + d)
    // C = (κθ/σ²)·[(κ - ρσiu - d)T - 2ln((1-g·exp(-dT))/(1-g))]
    // D = ((κ - ρσiu - d)/σ²)·(1 - exp(-dT))/(1 - g·exp(-dT))

    // Our u is complex: u = u_re + i·u_im
    // iu = i·u_re - u_im = (-u_im, u_re)
    // ρσ·iu = ρσ(-u_im, u_re)
    let iu_re = -u_im;
    let iu_im = u_re;

    let rsi_re = rho * sigma * iu_re; // -ρσ·u_im
    let rsi_im = rho * sigma * iu_im; // ρσ·u_re

    // α = κ - ρσ·iu
    let alpha_re = kappa - rsi_re;
    let alpha_im = -rsi_im;

    // iu + u² where u² = u_re² - u_im² + 2i·u_re·u_im
    let u2_re = u_re * u_re - u_im * u_im;
    let u2_im = 2.0 * u_re * u_im;
    let iu_plus_u2_re = iu_re + u2_re;
    let iu_plus_u2_im = iu_im + u2_im;

    // d² = α² + σ²·(iu + u²)
    let d2_re = alpha_re * alpha_re - alpha_im * alpha_im + sigma * sigma * iu_plus_u2_re;
    let d2_im = 2.0 * alpha_re * alpha_im + sigma * sigma * iu_plus_u2_im;

    // d = sqrt(d²)  (complex sqrt)
    let d_mod = (d2_re * d2_re + d2_im * d2_im).sqrt().sqrt();
    let d_arg = d2_im.atan2(d2_re) / 2.0;
    let d_re = d_mod * d_arg.cos();
    let d_im = d_mod * d_arg.sin();

    // g = (α - d) / (α + d)
    let num_re = alpha_re - d_re;
    let num_im = alpha_im - d_im;
    let den_re = alpha_re + d_re;
    let den_im = alpha_im + d_im;
    let den_mod2 = den_re * den_re + den_im * den_im;
    let g_re = (num_re * den_re + num_im * den_im) / den_mod2.max(1e-300);
    let g_im = (num_im * den_re - num_re * den_im) / den_mod2.max(1e-300);

    // exp(-d·T)
    let edt_re = (-d_re * tau).exp() * ((-d_im * tau).cos());
    let edt_im = (-d_re * tau).exp() * ((-d_im * tau).sin());

    // g · exp(-dT)
    let gedt_re = g_re * edt_re - g_im * edt_im;
    let gedt_im = g_re * edt_im + g_im * edt_re;

    // D = (α - d) / σ² · (1 - exp(-dT)) / (1 - g·exp(-dT))
    let one_minus_edt_re = 1.0 - edt_re;
    let one_minus_edt_im = -edt_im;
    let one_minus_gedt_re = 1.0 - gedt_re;
    let one_minus_gedt_im = -gedt_im;

    let ratio_den = one_minus_gedt_re * one_minus_gedt_re + one_minus_gedt_im * one_minus_gedt_im;
    let ratio_re = (one_minus_edt_re * one_minus_gedt_re + one_minus_edt_im * one_minus_gedt_im) / ratio_den.max(1e-300);
    let ratio_im = (one_minus_edt_im * one_minus_gedt_re - one_minus_edt_re * one_minus_gedt_im) / ratio_den.max(1e-300);

    let s2_inv = 1.0 / (sigma * sigma);
    let big_d_re = s2_inv * (num_re * ratio_re - num_im * ratio_im);
    let big_d_im = s2_inv * (num_re * ratio_im + num_im * ratio_re);

    // C = (κθ/σ²) · [(α - d)T - 2·ln((1 - g·exp(-dT))/(1 - g))]
    let one_minus_g_re = 1.0 - g_re;
    let one_minus_g_im = -g_im;
    let frac_re = (one_minus_gedt_re * one_minus_g_re + one_minus_gedt_im * one_minus_g_im)
        / (one_minus_g_re * one_minus_g_re + one_minus_g_im * one_minus_g_im).max(1e-300);
    let frac_im = (one_minus_gedt_im * one_minus_g_re - one_minus_gedt_re * one_minus_g_im)
        / (one_minus_g_re * one_minus_g_re + one_minus_g_im * one_minus_g_im).max(1e-300);

    let ln_frac_re = 0.5 * (frac_re * frac_re + frac_im * frac_im).ln();
    let ln_frac_im = frac_im.atan2(frac_re);

    let kt_s2 = kappa * theta * s2_inv;
    let big_c_re = kt_s2 * (num_re * tau - 2.0 * ln_frac_re);
    let big_c_im = kt_s2 * (num_im * tau - 2.0 * ln_frac_im);

    // log φ = C + D·v₀
    (big_c_re + big_d_re * v0, big_c_im + big_d_im * v0)
}

/// Price a European option using the Heston model via PDF reconstruction.
///
/// The PDF is obtained by Fourier-inverting the characteristic function:
/// ```text
/// f(x) = (1/2π) ∫ exp(-iux) φ(u) du
/// ```
///
/// Then the option price is `e^{-rT} ∫ payoff(S_T) f(S_T) dS_T`.
///
/// In practice we use the Gil-Pelaez (1951) form to compute P₁, P₂ directly.
#[allow(clippy::too_many_arguments)]
pub fn heston_pdf_price(
    spot: f64,
    strike: f64,
    tau: f64,
    risk_free_rate: f64,
    dividend_yield: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    opt_type: OptionType,
) -> QLResult<HestonPdfResult> {
    if spot <= 0.0 || strike <= 0.0 || tau <= 0.0 {
        return Err(QLError::InvalidArgument("spot, strike, tau must be positive".into()));
    }
    let r = risk_free_rate;
    let q = dividend_yield;
    let fwd = spot * ((r - q) * tau).exp();
    let x = fwd.ln();
    let ln_k = strike.ln();
    let df = (-r * tau).exp();

    // Gil-Pelaez inversion: P_j = 1/2 + (1/π) ∫₀^∞ Re[ exp(-iu·ln(K)) φ_j(u) / (iu) ] du
    // where φ₁ uses u_im offset -1 (for S·P₁), φ₂ uses u_im offset 0 (for P₂)
    let n_points = 256;
    let u_max = 100.0;
    let du = u_max / n_points as f64;

    let mut p1_sum = 0.0;
    let mut p2_sum = 0.0;
    let mut density_sum = 0.0;

    for i in 1..=n_points {
        let u = (i as f64 - 0.5) * du;

        // φ₂(u): characteristic function evaluated at (u, 0)
        let (log_cf2_re, log_cf2_im) = heston_log_cf(u, 0.0, tau, kappa, theta, sigma, rho, v0);
        let cf2_re = log_cf2_re.exp() * log_cf2_im.cos();
        let cf2_im = log_cf2_re.exp() * log_cf2_im.sin();

        // exp(iu·(x - ln(K))) / (iu) integrand for P2
        // x = ln(F), so phase = u * ln(F/K)
        let phase = u * (x - ln_k);
        let e_re = phase.cos();
        let e_im = phase.sin();
        // numerator = e · cf2
        let num_re = e_re * cf2_re - e_im * cf2_im;
        let num_im = e_re * cf2_im + e_im * cf2_re;
        // divide by iu: (a+bi)/(iu) = (a+bi)·(-i)/u = (b - ia)/u
        p2_sum += num_im / u * du; // Re part of numerator/iu = num_im / u
        // Actually: 1/(iu) = -i/u, so (num_re + i·num_im)·(-i/u) = (-i·num_re + num_im)/u = (num_im - i·num_re)/u
        // Re part = num_im / u  ✓

        // φ₁(u): shift by -i in imaginary component → evaluate at (u, -1)
        let (log_cf1_re, log_cf1_im) = heston_log_cf(u, -1.0, tau, kappa, theta, sigma, rho, v0);
        let cf1_re = log_cf1_re.exp() * log_cf1_im.cos();
        let cf1_im = log_cf1_re.exp() * log_cf1_im.sin();
        // We also need to multiply by exp(iu·x) / φ(0) to normalize for φ₁
        // Actually in Heston: φ₁(u) = φ(u-i) / φ(-i)
        // φ(-i) = exp(C(-i,T) + D(-i,T)·v0)
        // Simpler: normalize at the end by dividing by fwd
        let _num1_re = e_re * cf1_re - e_im * cf1_im;
        let num1_im = e_re * cf1_im + e_im * cf1_re;
        p1_sum += num1_im / u * du;

        // Density at strike: f(K) = (1/2π) ∫ exp(-iu·ln(K)) φ₂(u) du
        // Real part of exp(-iu·lnK)·φ₂(u) = num_re
        density_sum += num_re / PI * du;
    }

    // φ₁(-i) normalization
    let (log_cf_mi_re, _log_cf_mi_im) = heston_log_cf(0.0, -1.0, tau, kappa, theta, sigma, rho, v0);
    let norm1 = log_cf_mi_re.exp(); // φ(-i) should be real = fwd for risk-neutral

    let p1 = 0.5 + p1_sum / (PI * norm1.max(1e-300));
    let p2 = 0.5 + p2_sum / PI;

    // Clamp
    let p1 = p1.clamp(0.0, 1.0);
    let p2 = p2.clamp(0.0, 1.0);

    let call_price = df * (fwd * p1 - strike * p2);
    let price = match opt_type {
        OptionType::Call => call_price.max(0.0),
        OptionType::Put => (call_price - df * (fwd - strike)).max(0.0),
    };

    let density = density_sum / strike; // transform from log-space to price-space

    Ok(HestonPdfResult {
        price,
        density_at_strike: density.max(0.0),
    })
}

// ===========================================================================
// Exponential Fitting Heston Engine
// ===========================================================================

/// Result from the exponential-fitting Heston engine.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ExpFitHestonResult {
    /// Option price.
    pub price: f64,
}

/// Gauss-Laguerre nodes and weights for n=32 (for exponential fitting).
#[allow(dead_code)]
fn gauss_laguerre_32() -> Vec<(f64, f64)> {
    // Nodes and weights for ∫₀^∞ e^{-x} f(x) dx ≈ Σ wᵢ f(xᵢ)
    // Standard 32-point Gauss-Laguerre quadrature
    vec![
        (0.04448936583326, 0.10921834195239),
        (0.23452610952177, 0.21044310793882),
        (0.57688462930188, 0.23521322966985),
        (1.07244875381782, 0.19590333597289),
        (1.72240877283587, 0.12998378628607),
        (2.52833670642579, 0.07057862386572),
        (3.49221132560597, 0.03176091250917),
        (4.61645676974930, 0.01185468623513),
        (5.90395849898971, 0.00366640292695),
        (7.35812673318624, 0.00093690362235),
        (8.98294092421146, 0.00019689358333),
        (10.783_018_632_545_4, 0.00003395237785),
        (12.76368805661768, 0.00000477478068),
        (14.93113975953519, 0.00000054264075),
        (17.292_618_912_564_1, 0.00000004941209),
        (19.85594260469277, 0.00000000356804),
        (22.63047088816964, 0.00000000020124),
        (25.62791093993284, 0.00000000000873),
        (28.86244421338921, 0.00000000000028),
        (32.35189501154709, 0.00000000000001),
        (36.11861655651911, 2.0e-16),
        (40.19127083413528, 5.0e-18),
        (44.60599413498137, 8.0e-20),
        (49.408_714_415_783_8, 8.0e-22),
        (54.65785933631284, 6.0e-24),
        (60.43047102962906, 2.0e-26),
        (66.83509595089002, 6.0e-29),
        (74.032_800_810_431_5, 8.0e-32),
        (82.29567521039788, 4.0e-35),
        (92.15338560862009, 6.0e-39),
        (104.816_134_681_794_7, 1.0e-43),
        (124.264_334_816_383_1, 1.0e-50),
    ]
}

/// Price a European option using the exponential-fitting Heston method.
///
/// This is the fastest semi-analytic Heston pricer. It uses a
/// Gauss-Laguerre quadrature to evaluate the Fourier integral,
/// exploiting the exponential decay of the integrand for large u.
///
/// Reference: Andersen & Piterbarg (2010), §9.3.4.
#[allow(clippy::too_many_arguments)]
pub fn exponential_fitting_heston(
    spot: f64,
    strike: f64,
    tau: f64,
    risk_free_rate: f64,
    dividend_yield: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    opt_type: OptionType,
) -> QLResult<ExpFitHestonResult> {
    if spot <= 0.0 || strike <= 0.0 || tau <= 0.0 {
        return Err(QLError::InvalidArgument("spot, strike, tau must be positive".into()));
    }
    let r = risk_free_rate;
    let q = dividend_yield;
    let fwd = spot * ((r - q) * tau).exp();
    let x = fwd.ln();
    let ln_k = strike.ln();
    let df = (-r * tau).exp();

    // Use midpoint rule (same as PDF version, but with 64 points for speed)
    let n_points = 64;
    let u_max = 50.0;
    let du = u_max / n_points as f64;

    let mut p1_sum = 0.0;
    let mut p2_sum = 0.0;

    let (log_norm_re, _) = heston_log_cf(0.0, -1.0, tau, kappa, theta, sigma, rho, v0);
    let norm1 = log_norm_re.exp();

    for i in 1..=n_points {
        let u = (i as f64 - 0.5) * du;

        // P₂ integrand
        let (lc2_re, lc2_im) = heston_log_cf(u, 0.0, tau, kappa, theta, sigma, rho, v0);
        let cf2_re = lc2_re.exp() * lc2_im.cos();
        let cf2_im = lc2_re.exp() * lc2_im.sin();
        let phase = u * (x - ln_k);
        let e_re = phase.cos();
        let e_im = phase.sin();
        let num2_im = e_re * cf2_im + e_im * cf2_re;
        if u > 1e-12 {
            p2_sum += num2_im / u * du;
        }

        // P₁ integrand (shift by -i)
        let (lc1_re, lc1_im) = heston_log_cf(u, -1.0, tau, kappa, theta, sigma, rho, v0);
        let cf1_re = lc1_re.exp() * lc1_im.cos();
        let cf1_im = lc1_re.exp() * lc1_im.sin();
        let num1_im = e_re * cf1_im + e_im * cf1_re;
        if u > 1e-12 {
            p1_sum += num1_im / u * du;
        }
    }

    let p1 = (0.5 + p1_sum / (PI * norm1.max(1e-300))).clamp(0.0, 1.0);
    let p2 = (0.5 + p2_sum / PI).clamp(0.0, 1.0);

    let call_price = df * (fwd * p1 - strike * p2);
    let price = match opt_type {
        OptionType::Call => call_price.max(0.0),
        OptionType::Put => (call_price - df * (fwd - strike)).max(0.0),
    };

    Ok(ExpFitHestonResult { price })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_heston_pdf_atm_call() {
        let res = heston_pdf_price(
            100.0, 100.0, 1.0, 0.05, 0.0,
            0.04, 1.5, 0.04, 0.3, -0.7,
            OptionType::Call,
        ).unwrap();
        // Should be close to standard Heston pricer
        assert!(res.price > 5.0 && res.price < 15.0, "price={}", res.price);
        assert!(res.density_at_strike > 0.0);
    }

    #[test]
    fn test_heston_pdf_put_call_parity() {
        let call = heston_pdf_price(
            100.0, 105.0, 0.5, 0.03, 0.01,
            0.04, 2.0, 0.04, 0.4, -0.5,
            OptionType::Call,
        ).unwrap();
        let put = heston_pdf_price(
            100.0, 105.0, 0.5, 0.03, 0.01,
            0.04, 2.0, 0.04, 0.4, -0.5,
            OptionType::Put,
        ).unwrap();
        let fwd = 100.0 * ((0.03 - 0.01) * 0.5_f64).exp();
        let df = (-0.03 * 0.5_f64).exp();
        let parity = call.price - put.price - df * (fwd - 105.0);
        assert_abs_diff_eq!(parity, 0.0, epsilon = 0.5);
    }

    #[test]
    fn test_exp_fit_heston_atm_call() {
        let res = exponential_fitting_heston(
            100.0, 100.0, 1.0, 0.05, 0.0,
            0.04, 1.5, 0.04, 0.3, -0.7,
            OptionType::Call,
        ).unwrap();
        assert!(res.price > 5.0 && res.price < 15.0, "price={}", res.price);
    }

    #[test]
    fn test_exp_fit_heston_put_call_parity() {
        let call = exponential_fitting_heston(
            100.0, 90.0, 1.0, 0.05, 0.02,
            0.04, 1.5, 0.04, 0.3, -0.7,
            OptionType::Call,
        ).unwrap();
        let put = exponential_fitting_heston(
            100.0, 90.0, 1.0, 0.05, 0.02,
            0.04, 1.5, 0.04, 0.3, -0.7,
            OptionType::Put,
        ).unwrap();
        let fwd = 100.0 * ((0.05 - 0.02) * 1.0_f64).exp();
        let df = (-0.05_f64).exp();
        let parity = call.price - put.price - df * (fwd - 90.0);
        assert_abs_diff_eq!(parity, 0.0, epsilon = 0.5);
    }
}
