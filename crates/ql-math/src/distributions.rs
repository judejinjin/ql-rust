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
}
