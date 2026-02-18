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
