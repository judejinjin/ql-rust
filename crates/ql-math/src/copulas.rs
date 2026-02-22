//! Copula functions: Gaussian, Clayton, Frank, Gumbel, Student-t.
//!
//! A copula C(u, v) connects marginal distributions to form joint distributions.
//! By Sklar's theorem: F(x,y) = C(F_X(x), F_Y(y)).

use ql_core::QLResult;

/// Gaussian copula CDF.
///
/// C(u, v; ρ) = Φ₂(Φ⁻¹(u), Φ⁻¹(v); ρ)
pub fn gaussian_copula(u: f64, v: f64, rho: f64) -> f64 {
    if u <= 0.0 || v <= 0.0 {
        return 0.0;
    }
    if u >= 1.0 {
        return v;
    }
    if v >= 1.0 {
        return u;
    }

    let x = inv_normal(u);
    let y = inv_normal(v);
    bivariate_normal_cdf(x, y, rho)
}

/// Clayton copula CDF: C(u, v; θ) = (u^{−θ} + v^{−θ} − 1)^{−1/θ}
///
/// θ > 0 gives lower tail dependence.
pub fn clayton_copula(u: f64, v: f64, theta: f64) -> f64 {
    if theta <= 0.0 || u <= 0.0 || v <= 0.0 {
        return 0.0;
    }
    if u >= 1.0 {
        return v;
    }
    if v >= 1.0 {
        return u;
    }

    let val = u.powf(-theta) + v.powf(-theta) - 1.0;
    if val <= 0.0 {
        return 0.0;
    }
    val.powf(-1.0 / theta)
}

/// Frank copula CDF: C(u, v; θ) = −(1/θ) ln(1 + (e^{−θu}−1)(e^{−θv}−1)/(e^{−θ}−1))
///
/// Symmetric dependence. θ = 0 gives independence.
pub fn frank_copula(u: f64, v: f64, theta: f64) -> f64 {
    if u <= 0.0 || v <= 0.0 {
        return 0.0;
    }
    if u >= 1.0 {
        return v;
    }
    if v >= 1.0 {
        return u;
    }

    if theta.abs() < 1e-10 {
        return u * v; // independence
    }

    let a = (-theta * u).exp_m1();
    let b = (-theta * v).exp_m1();
    let c = (-theta).exp_m1();

    -((1.0 + a * b / c).ln()) / theta
}

/// Gumbel copula CDF: C(u, v; θ) = exp(−((−ln u)^θ + (−ln v)^θ)^{1/θ})
///
/// θ ≥ 1. Upper tail dependence. θ = 1 gives independence.
pub fn gumbel_copula(u: f64, v: f64, theta: f64) -> f64 {
    if u <= 0.0 || v <= 0.0 {
        return 0.0;
    }
    if u >= 1.0 {
        return v;
    }
    if v >= 1.0 {
        return u;
    }
    if theta < 1.0 {
        return u * v; // fallback
    }

    let a = (-u.ln()).powf(theta);
    let b = (-v.ln()).powf(theta);
    (-(a + b).powf(1.0 / theta)).exp()
}

/// Bivariate standard normal CDF using Gauss-Legendre quadrature.
fn bivariate_normal_cdf(x: f64, y: f64, rho: f64) -> f64 {
    let n = crate::distributions::NormalDistribution::standard();

    if rho.abs() < 1e-12 {
        return n.cdf(x) * n.cdf(y);
    }
    if (rho - 1.0).abs() < 1e-12 {
        return n.cdf(x.min(y));
    }
    if (rho + 1.0).abs() < 1e-12 {
        return (n.cdf(x) + n.cdf(-y) - 1.0).max(0.0);
    }

    // Gauss-Legendre on arcsin(ρ) form
    let theta0 = rho.asin();
    let half_nodes = [
        (0.095_012_509_837_637_44, 0.189_450_610_455_068),
        (0.281_603_550_779_258_9, 0.182_603_415_044_924),
        (0.458_016_777_657_227_4, 0.169_156_519_395_003),
        (0.617_876_244_402_643_7, 0.149_595_988_816_577),
        (0.755_404_408_355_003, 0.124_628_971_255_534),
        (0.865_631_202_387_831_7, 0.095_158_511_682_493),
        (0.944_575_023_073_232_6, 0.062_253_523_938_648),
        (0.989_400_934_991_649_9, 0.027_152_459_411_754),
    ];

    let mut sum = 0.0;
    let half_range = theta0 / 2.0;
    let mid = theta0 / 2.0;

    for &(xi, wi) in &half_nodes {
        for &sign in &[-1.0, 1.0] {
            let theta = mid + sign * half_range * xi;
            let sin_t = theta.sin();
            let cos_t = theta.cos();
            let integrand =
                (-(x * x + y * y - 2.0 * x * y * sin_t) / (2.0 * cos_t * cos_t)).exp();
            sum += wi * integrand;
        }
    }

    sum *= half_range / (2.0 * std::f64::consts::PI);
    sum += n.cdf(x) * n.cdf(y);
    sum.clamp(0.0, 1.0)
}

/// Inverse standard normal CDF (Acklam's algorithm).
fn inv_normal(p: f64) -> f64 {
    match crate::distributions::inverse_cumulative_normal(p) {
        Ok(x) => x,
        Err(_) => {
            if p <= 0.0 {
                -10.0
            } else {
                10.0
            }
        }
    }
}

/// Kendall's tau → copula parameter conversions.
pub fn clayton_theta_from_tau(tau: f64) -> QLResult<f64> {
    if tau <= 0.0 || tau >= 1.0 {
        return Err(ql_core::QLError::InvalidArgument(
            "Kendall's tau must be in (0, 1) for Clayton".into(),
        ));
    }
    Ok(2.0 * tau / (1.0 - tau))
}

pub fn gumbel_theta_from_tau(tau: f64) -> QLResult<f64> {
    if tau <= 0.0 || tau >= 1.0 {
        return Err(ql_core::QLError::InvalidArgument(
            "Kendall's tau must be in (0, 1) for Gumbel".into(),
        ));
    }
    Ok(1.0 / (1.0 - tau))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn gaussian_copula_independence() {
        // ρ=0 → C(u,v) = u·v
        let c = gaussian_copula(0.3, 0.7, 0.0);
        assert_abs_diff_eq!(c, 0.21, epsilon = 0.01);
    }

    #[test]
    fn gaussian_copula_comonotone() {
        // ρ=1 → C(u,v) = min(u,v)
        let c = gaussian_copula(0.3, 0.7, 1.0);
        assert_abs_diff_eq!(c, 0.3, epsilon = 0.01);
    }

    #[test]
    fn gaussian_copula_bounds() {
        let c = gaussian_copula(0.5, 0.5, 0.5);
        assert!(c >= 0.0 && c <= 0.5, "Copula should be in [0, min(u,v)]: {c}");
    }

    #[test]
    fn clayton_copula_positive() {
        let c = clayton_copula(0.5, 0.5, 2.0);
        assert!(c > 0.0 && c <= 0.5);
    }

    #[test]
    fn clayton_boundary() {
        // C(u, 1) = u
        let c = clayton_copula(0.3, 1.0, 2.0);
        assert_abs_diff_eq!(c, 0.3, epsilon = 1e-10);
    }

    #[test]
    fn frank_copula_independence() {
        // θ ≈ 0 → C(u,v) ≈ u·v
        let c = frank_copula(0.3, 0.7, 0.0);
        assert_abs_diff_eq!(c, 0.21, epsilon = 1e-8);
    }

    #[test]
    fn frank_copula_positive_theta() {
        let c = frank_copula(0.5, 0.5, 5.0);
        assert!(c > 0.0 && c <= 0.5);
    }

    #[test]
    fn gumbel_copula_independence() {
        // θ=1 → C(u,v) = u·v
        let c = gumbel_copula(0.3, 0.7, 1.0);
        assert_abs_diff_eq!(c, 0.21, epsilon = 0.01);
    }

    #[test]
    fn gumbel_copula_positive() {
        let c = gumbel_copula(0.5, 0.5, 3.0);
        assert!(c > 0.0 && c <= 0.5);
    }

    #[test]
    fn copula_frechet_lower_bound() {
        // All copulas: C(u,v) ≥ max(u + v − 1, 0)
        for &rho in &[0.0, 0.3, 0.7] {
            let c = gaussian_copula(0.3, 0.4, rho);
            assert!(c >= 0.0, "Gaussian copula violated lower bound");
        }
    }

    #[test]
    fn clayton_theta_from_kendall() {
        let theta = clayton_theta_from_tau(0.5).unwrap();
        assert_abs_diff_eq!(theta, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn gumbel_theta_from_kendall() {
        let theta = gumbel_theta_from_tau(0.5).unwrap();
        assert_abs_diff_eq!(theta, 2.0, epsilon = 1e-10);
    }
}

// ===========================================================================
// Additional copulas: Independent, Min (W), Max (M), FGM, Plackett,
// Galambos, Marshall-Olkin
// ===========================================================================

/// Independence copula: C(u, v) = u · v.
pub fn independent_copula(u: f64, v: f64) -> f64 {
    u.clamp(0.0, 1.0) * v.clamp(0.0, 1.0)
}

/// Fréchet upper bound (comonotonicity copula): C(u, v) = min(u, v).
pub fn max_copula(u: f64, v: f64) -> f64 {
    u.min(v).clamp(0.0, 1.0)
}

/// Fréchet lower bound (countermonotonicity copula): C(u, v) = max(u + v − 1, 0).
pub fn min_copula(u: f64, v: f64) -> f64 {
    (u + v - 1.0).max(0.0)
}

/// Farlie-Gumbel-Morgenstern (FGM) copula.
///
/// C(u, v; θ) = u·v (1 + θ·(1−u)·(1−v)),  θ ∈ [−1, 1].
///
/// The FGM copula captures weak dependence; for stronger dependence
/// prefer Clayton/Gumbel.
pub fn fgm_copula(u: f64, v: f64, theta: f64) -> f64 {
    let u = u.clamp(0.0, 1.0);
    let v = v.clamp(0.0, 1.0);
    u * v * (1.0 + theta * (1.0 - u) * (1.0 - v))
}

/// Kendall's τ implied by FGM copula parameter θ.
///
/// τ = 2θ/9.
pub fn fgm_tau_from_theta(theta: f64) -> f64 {
    2.0 * theta / 9.0
}

/// Plackett copula CDF (solved numerically via closed form).
///
/// C(u, v; θ) where θ > 0 is the cross-product ratio (odds ratio).
/// θ = 1 gives the independence copula.
pub fn plackett_copula(u: f64, v: f64, theta: f64) -> f64 {
    let u = u.clamp(0.0, 1.0);
    let v = v.clamp(0.0, 1.0);
    if (theta - 1.0).abs() < 1e-10 {
        return u * v;
    }
    // Closed-form solution of the quadratic
    let s = u + v;
    let eta = theta - 1.0;
    let a = 1.0 + eta * s;
    let discriminant = (a * a - 4.0 * theta * eta * u * v).max(0.0);
    (a - discriminant.sqrt()) / (2.0 * eta)
}

/// Galambos copula — extreme-value copula.
///
/// C(u, v; θ) = u·v · exp(((-ln u)^{-θ} + (-ln v)^{-θ})^{-1/θ}),
/// θ ≥ 0.  θ → ∞ gives perfect dependence.
pub fn galambos_copula(u: f64, v: f64, theta: f64) -> f64 {
    if u <= 0.0 || v <= 0.0 {
        return 0.0;
    }
    if u >= 1.0 {
        return v;
    }
    if v >= 1.0 {
        return u;
    }
    if theta < 1e-14 {
        return u * v; // independence limit
    }
    let lu = -u.ln();
    let lv = -v.ln();
    let inner = (lu.powf(-theta) + lv.powf(-theta)).powf(-1.0 / theta);
    u * v * inner.exp()
}

/// Marshall-Olkin copula.
///
/// C(u, v; α, β) = min(u^{1−α}·v, u·v^{1−β}),  α,β ∈ [0,1].
///
/// Captures asymmetric tail dependence.
pub fn marshall_olkin_copula(u: f64, v: f64, alpha: f64, beta: f64) -> f64 {
    let u = u.clamp(0.0, 1.0);
    let v = v.clamp(0.0, 1.0);
    let t1 = u.powf(1.0 - alpha) * v;
    let t2 = u * v.powf(1.0 - beta);
    t1.min(t2)
}

/// Upper tail dependence coefficient λ_U for the Marshall-Olkin copula.
///
/// λ_U = min(α, β) / (α + β − α·β).
pub fn marshall_olkin_upper_tail(alpha: f64, beta: f64) -> f64 {
    let denom = alpha + beta - alpha * beta;
    if denom < 1e-15 {
        return 0.0;
    }
    alpha.min(beta) / denom
}

#[cfg(test)]
mod extra_copula_tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn independent_copula_value() {
        assert_abs_diff_eq!(independent_copula(0.5, 0.5), 0.25, epsilon = 1e-12);
        assert_abs_diff_eq!(independent_copula(0.3, 0.7), 0.21, epsilon = 1e-12);
    }

    #[test]
    fn max_copula_comonotone() {
        assert_abs_diff_eq!(max_copula(0.3, 0.7), 0.3, epsilon = 1e-12);
        assert_abs_diff_eq!(max_copula(0.8, 0.4), 0.4, epsilon = 1e-12);
    }

    #[test]
    fn min_copula_lower_bound() {
        assert_abs_diff_eq!(min_copula(0.5, 0.5), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(min_copula(0.8, 0.9), 0.7, epsilon = 1e-12);
    }

    #[test]
    fn fgm_independence_at_zero() {
        // θ=0 → independence
        assert_abs_diff_eq!(fgm_copula(0.4, 0.6, 0.0), 0.24, epsilon = 1e-12);
    }

    #[test]
    fn fgm_bounds() {
        let c = fgm_copula(0.5, 0.5, 1.0);
        assert!(c >= 0.25 && c <= 0.5, "FGM out of bounds: {c}");
    }

    #[test]
    fn plackett_independence() {
        // θ=1 → independence
        assert_abs_diff_eq!(plackett_copula(0.4, 0.5, 1.0), 0.2, epsilon = 1e-6);
    }

    #[test]
    fn galambos_independence_at_zero() {
        // θ→0+ → independence
        assert_abs_diff_eq!(galambos_copula(0.5, 0.5, 1e-10), 0.25, epsilon = 1e-6);
    }

    #[test]
    fn galambos_positive() {
        let c = galambos_copula(0.5, 0.5, 2.0);
        assert!(c > 0.25 && c <= 0.5, "Galambos out of bounds: {c}");
    }

    #[test]
    fn marshall_olkin_symmetric() {
        // With α=β, result is symmetric in u,v
        let c1 = marshall_olkin_copula(0.3, 0.7, 0.5, 0.5);
        let c2 = marshall_olkin_copula(0.7, 0.3, 0.5, 0.5);
        assert_abs_diff_eq!(c1, c2, epsilon = 1e-12);
    }

    #[test]
    fn marshall_olkin_tail_dep() {
        let lam = marshall_olkin_upper_tail(0.5, 0.5);
        // min(0.5,0.5)/(0.5+0.5−0.25) = 0.5/0.75 = 2/3
        assert_abs_diff_eq!(lam, 2.0 / 3.0, epsilon = 1e-12);
    }
}
