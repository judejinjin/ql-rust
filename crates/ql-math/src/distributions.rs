//! Statistical distributions.
//!
//! Thin wrappers around `statrs` providing QuantLib-compatible interfaces:
//! - Normal: CDF, PDF, inverse CDF
//! - Cumulative Poisson

use ql_core::errors::{QLError, QLResult};
use statrs::distribution::{ContinuousCDF, Continuous, Discrete, DiscreteCDF};

// ===========================================================================
// Normal Distribution
// ===========================================================================

/// Standard or parametric normal distribution.
#[derive(Clone, Debug)]
pub struct NormalDistribution {
    inner: statrs::distribution::Normal,
}

impl NormalDistribution {
    /// Standard normal N(0,1).
    pub fn standard() -> Self {
        // Parameters (0.0, 1.0) are always valid for Normal::new.
        Self {
            inner: statrs::distribution::Normal::new(0.0, 1.0)
                .unwrap_or_else(|_| unreachable!()),
        }
    }

    /// Normal distribution with given mean and standard deviation.
    pub fn new(mean: f64, std_dev: f64) -> QLResult<Self> {
        statrs::distribution::Normal::new(mean, std_dev)
            .map(|inner| Self { inner })
            .map_err(|e| QLError::InvalidArgument(format!("invalid normal params: {e}")))
    }

    /// Cumulative distribution function: P(X <= x).
    pub fn cdf(&self, x: f64) -> f64 {
        ContinuousCDF::cdf(&self.inner, x)
    }

    /// Probability density function.
    pub fn pdf(&self, x: f64) -> f64 {
        self.inner.pdf(x)
    }

    /// Inverse CDF (quantile function): returns x such that CDF(x) = p.
    pub fn inverse_cdf(&self, p: f64) -> QLResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(QLError::InvalidArgument(format!(
                "probability must be in [0,1], got {p}"
            )));
        }
        if p == 0.0 {
            return Ok(f64::NEG_INFINITY);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        Ok(ContinuousCDF::inverse_cdf(&self.inner, p))
    }
}

// ===========================================================================
// Cumulative Normal (convenience)
// ===========================================================================

/// Convenience function: standard normal CDF.
pub fn cumulative_normal(x: f64) -> f64 {
    NormalDistribution::standard().cdf(x)
}

/// Convenience function: standard normal inverse CDF.
pub fn inverse_cumulative_normal(p: f64) -> QLResult<f64> {
    NormalDistribution::standard().inverse_cdf(p)
}

// ===========================================================================
// Poisson Distribution
// ===========================================================================

/// Poisson distribution.
#[derive(Clone, Debug)]
pub struct PoissonDistribution {
    inner: statrs::distribution::Poisson,
}

impl PoissonDistribution {
    /// Create a Poisson distribution with rate parameter `lambda`.
    pub fn new(lambda: f64) -> QLResult<Self> {
        statrs::distribution::Poisson::new(lambda)
            .map(|inner| Self { inner })
            .map_err(|e| QLError::InvalidArgument(format!("invalid Poisson lambda: {e}")))
    }

    /// P(X = k).
    pub fn pmf(&self, k: u64) -> f64 {
        self.inner.pmf(k)
    }

    /// P(X <= k).
    pub fn cdf(&self, k: u64) -> f64 {
        DiscreteCDF::cdf(&self.inner, k)
    }
}

// ===========================================================================
// Chi-Squared Distribution
// ===========================================================================

/// Chi-squared distribution.
#[derive(Clone, Debug)]
pub struct ChiSquaredDistribution {
    inner: statrs::distribution::ChiSquared,
}

impl ChiSquaredDistribution {
    /// Create a chi-squared distribution with `dof` degrees of freedom.
    pub fn new(dof: f64) -> QLResult<Self> {
        statrs::distribution::ChiSquared::new(dof)
            .map(|inner| Self { inner })
            .map_err(|e| QLError::InvalidArgument(format!("invalid chi-squared dof: {e}")))
    }

    /// Cumulative distribution function: P(X ≤ x).
    pub fn cdf(&self, x: f64) -> f64 {
        ContinuousCDF::cdf(&self.inner, x)
    }

    /// Probability density function.
    pub fn pdf(&self, x: f64) -> f64 {
        self.inner.pdf(x)
    }

    /// Inverse CDF (quantile function): returns x such that CDF(x) = p.
    pub fn inverse_cdf(&self, p: f64) -> QLResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(QLError::InvalidArgument(format!(
                "probability must be in [0,1], got {p}"
            )));
        }
        Ok(ContinuousCDF::inverse_cdf(&self.inner, p))
    }
}

// ===========================================================================
// Student-t Distribution
// ===========================================================================

/// Student's t-distribution.
#[derive(Clone, Debug)]
pub struct StudentTDistribution {
    inner: statrs::distribution::StudentsT,
}

impl StudentTDistribution {
    /// Create a Student-t distribution with `dof` degrees of freedom.
    pub fn new(dof: f64) -> QLResult<Self> {
        statrs::distribution::StudentsT::new(0.0, 1.0, dof)
            .map(|inner| Self { inner })
            .map_err(|e| QLError::InvalidArgument(format!("invalid Student-t dof: {e}")))
    }

    /// General Student-t with location and scale.
    pub fn with_params(location: f64, scale: f64, dof: f64) -> QLResult<Self> {
        statrs::distribution::StudentsT::new(location, scale, dof)
            .map(|inner| Self { inner })
            .map_err(|e| QLError::InvalidArgument(format!("invalid Student-t params: {e}")))
    }

    pub fn cdf(&self, x: f64) -> f64 {
        ContinuousCDF::cdf(&self.inner, x)
    }

    pub fn pdf(&self, x: f64) -> f64 {
        self.inner.pdf(x)
    }

    pub fn inverse_cdf(&self, p: f64) -> QLResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(QLError::InvalidArgument(format!(
                "probability must be in [0,1], got {p}"
            )));
        }
        Ok(ContinuousCDF::inverse_cdf(&self.inner, p))
    }
}

// ===========================================================================
// Gamma Distribution
// ===========================================================================

/// Gamma distribution with shape (α) and rate (β) parameterization.
#[derive(Clone, Debug)]
pub struct GammaDistribution {
    inner: statrs::distribution::Gamma,
}

impl GammaDistribution {
    /// Create with shape α and rate β. Mean = α/β, Var = α/β².
    pub fn new(shape: f64, rate: f64) -> QLResult<Self> {
        statrs::distribution::Gamma::new(shape, rate)
            .map(|inner| Self { inner })
            .map_err(|e| QLError::InvalidArgument(format!("invalid Gamma params: {e}")))
    }

    pub fn cdf(&self, x: f64) -> f64 {
        ContinuousCDF::cdf(&self.inner, x)
    }

    pub fn pdf(&self, x: f64) -> f64 {
        self.inner.pdf(x)
    }

    pub fn inverse_cdf(&self, p: f64) -> QLResult<f64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(QLError::InvalidArgument(format!(
                "probability must be in [0,1], got {p}"
            )));
        }
        Ok(ContinuousCDF::inverse_cdf(&self.inner, p))
    }
}

// ===========================================================================
// Binomial Distribution
// ===========================================================================

/// Binomial distribution B(n, p).
#[derive(Clone, Debug)]
pub struct BinomialDistribution {
    inner: statrs::distribution::Binomial,
}

impl BinomialDistribution {
    pub fn new(n: u64, p: f64) -> QLResult<Self> {
        statrs::distribution::Binomial::new(p, n)
            .map(|inner| Self { inner })
            .map_err(|e| QLError::InvalidArgument(format!("invalid Binomial params: {e}")))
    }

    /// P(X = k).
    pub fn pmf(&self, k: u64) -> f64 {
        self.inner.pmf(k)
    }

    /// P(X ≤ k).
    pub fn cdf(&self, k: u64) -> f64 {
        DiscreteCDF::cdf(&self.inner, k)
    }
}

// ===========================================================================
// Bivariate Normal Distribution
// ===========================================================================

/// Bivariate standard normal CDF: Φ₂(x, y; ρ).
///
/// Uses Drezner-Wesolowsky approximation (1990) for fast computation.
pub fn bivariate_normal_cdf(x: f64, y: f64, rho: f64) -> f64 {
    let n = NormalDistribution::standard();

    if rho.abs() < 1e-12 {
        return n.cdf(x) * n.cdf(y);
    }
    if (rho - 1.0).abs() < 1e-12 {
        return n.cdf(x.min(y));
    }
    if (rho + 1.0).abs() < 1e-12 {
        return (n.cdf(x) + n.cdf(-y) - 1.0).max(0.0);
    }

    // Gauss-Legendre quadrature on the arcsin(ρ) integral form
    let theta0 = rho.asin();
    let half_nodes = [
        (0.095_012_509_837_637_4, 0.189_450_610_455_068),
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

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn normal_cdf_symmetry() {
        let n = NormalDistribution::standard();
        // CDF(0) = 0.5
        assert_abs_diff_eq!(n.cdf(0.0), 0.5, epsilon = 1e-15);
        // CDF(-x) + CDF(x) = 1
        assert_abs_diff_eq!(n.cdf(-1.5) + n.cdf(1.5), 1.0, epsilon = 1e-15);
    }

    #[test]
    fn normal_cdf_known_values() {
        let n = NormalDistribution::standard();
        assert_abs_diff_eq!(n.cdf(1.0), 0.8413447460685429, epsilon = 1e-10);
        assert_abs_diff_eq!(n.cdf(2.0), 0.9772498680518208, epsilon = 1e-10);
    }

    #[test]
    fn normal_pdf_peak() {
        let n = NormalDistribution::standard();
        // PDF at 0 = 1/sqrt(2*pi)
        let expected = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        assert_abs_diff_eq!(n.pdf(0.0), expected, epsilon = 1e-15);
    }

    #[test]
    fn normal_inverse_cdf_roundtrip() {
        let n = NormalDistribution::standard();
        for &p in &[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
            let x = n.inverse_cdf(p).unwrap();
            assert_abs_diff_eq!(n.cdf(x), p, epsilon = 1e-10);
        }
    }

    #[test]
    fn normal_inverse_cdf_boundaries() {
        let n = NormalDistribution::standard();
        assert!(n.inverse_cdf(0.0).unwrap().is_infinite());
        assert!(n.inverse_cdf(1.0).unwrap().is_infinite());
        assert!(n.inverse_cdf(-0.1).is_err());
        assert!(n.inverse_cdf(1.1).is_err());
    }

    #[test]
    fn convenience_functions() {
        assert_abs_diff_eq!(cumulative_normal(0.0), 0.5, epsilon = 1e-15);
        let x = inverse_cumulative_normal(0.975).unwrap();
        assert_abs_diff_eq!(x, 1.959963984540054, epsilon = 1e-12);
    }

    #[test]
    fn custom_normal() {
        let n = NormalDistribution::new(5.0, 2.0).unwrap();
        // CDF at mean should be 0.5
        assert_abs_diff_eq!(n.cdf(5.0), 0.5, epsilon = 1e-15);
    }

    #[test]
    fn poisson_pmf() {
        let p = PoissonDistribution::new(3.0).unwrap();
        // P(X=0) = e^{-3}
        assert_abs_diff_eq!(p.pmf(0), (-3.0_f64).exp(), epsilon = 1e-14);
        // P(X=3) = e^{-3} * 3^3 / 3! = e^{-3} * 27/6
        let expected = (-3.0_f64).exp() * 27.0 / 6.0;
        assert_abs_diff_eq!(p.pmf(3), expected, epsilon = 1e-14);
    }

    #[test]
    fn poisson_cdf() {
        let p = PoissonDistribution::new(1.0).unwrap();
        // P(X <= 0) = e^{-1}
        assert_abs_diff_eq!(p.cdf(0), (-1.0_f64).exp(), epsilon = 1e-14);
    }

    #[test]
    fn chi_squared_basic() {
        let chi2 = ChiSquaredDistribution::new(2.0).unwrap();
        // CDF of chi-squared(2) at x is 1 - e^{-x/2}
        let x = 4.0;
        let expected = 1.0 - (-x / 2.0_f64).exp();
        assert_abs_diff_eq!(chi2.cdf(x), expected, epsilon = 1e-14);
    }

    #[test]
    fn student_t_symmetry() {
        let t = StudentTDistribution::new(5.0).unwrap();
        assert_abs_diff_eq!(t.cdf(0.0), 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(t.cdf(-2.0) + t.cdf(2.0), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn student_t_converges_to_normal() {
        // For large dof, Student-t → normal
        let t = StudentTDistribution::new(1000.0).unwrap();
        let n = NormalDistribution::standard();
        assert_abs_diff_eq!(t.cdf(1.96), n.cdf(1.96), epsilon = 0.005);
    }

    #[test]
    fn gamma_exponential_special_case() {
        // Gamma(1, λ) = Exponential(λ)
        let g = GammaDistribution::new(1.0, 2.0).unwrap();
        let x: f64 = 1.5;
        let expected = 1.0 - (-2.0 * x).exp();
        assert_abs_diff_eq!(g.cdf(x), expected, epsilon = 1e-10);
    }

    #[test]
    fn binomial_coin_flip() {
        let b = BinomialDistribution::new(10, 0.5).unwrap();
        // P(X = 5) = C(10,5) * 0.5^10 = 252/1024
        assert_abs_diff_eq!(b.pmf(5), 252.0 / 1024.0, epsilon = 1e-10);
    }

    #[test]
    fn bivariate_normal_independence() {
        let n = NormalDistribution::standard();
        // ρ=0 → Φ₂(x,y) = Φ(x)·Φ(y)
        let result = bivariate_normal_cdf(1.0, 1.0, 0.0);
        let expected = n.cdf(1.0) * n.cdf(1.0);
        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn bivariate_normal_comonotone() {
        let n = NormalDistribution::standard();
        // ρ=1 → Φ₂(x,y) = Φ(min(x,y))
        let result = bivariate_normal_cdf(0.5, 1.0, 1.0);
        assert_abs_diff_eq!(result, n.cdf(0.5), epsilon = 1e-10);
    }

    #[test]
    fn bivariate_normal_known_value() {
        // Φ₂(0, 0; 0.5) ≈ 0.333
        let result = bivariate_normal_cdf(0.0, 0.0, 0.5);
        assert_abs_diff_eq!(result, 0.333, epsilon = 0.01);
    }
}
