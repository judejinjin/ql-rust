//! AD-aware mathematical functions: normal PDF and CDF.
//!
//! Implementations use the Abramowitz-Stegun rational approximation for the
//! normal CDF so they work generically over any `T: Number`, including `Dual`,
//! `DualVec<N>`, and plain `f64`.

use crate::number::Number;

/// Standard normal probability density function:
///     φ(x) = (1/√(2π)) · exp(-x²/2)
///
/// Generic over `T: Number` so derivatives propagate through AD types.
#[inline]
pub fn normal_pdf<T: Number>(x: T) -> T {
    let half = T::half();
    let inv_sqrt_2pi = T::from_f64(0.3989422804014327); // 1/√(2π)
    inv_sqrt_2pi * (T::zero() - half * x * x).exp()
}

/// Cumulative distribution function for the standard normal distribution.
///
/// Uses the Abramowitz-Stegun (1964, formula 26.2.17) rational approximation,
/// which is accurate to ~|eps| < 7.5e-8. The formula is:
///
/// ```text
///     Phi(x) ~ 1 - phi(x)(b1*t + b2*t^2 + b3*t^3 + b4*t^4 + b5*t^5)
/// ```
///
/// where  t = 1/(1 + 0.2316419*|x|).
///
/// This is chosen over `erfc`-based formulas because it only needs basic
/// arithmetic + `exp`, which are available on all `Number` types.
#[inline]
pub fn normal_cdf<T: Number>(x: T) -> T {
    let b1 = T::from_f64(0.319381530);
    let b2 = T::from_f64(-0.356563782);
    let b3 = T::from_f64(1.781477937);
    let b4 = T::from_f64(-1.821255978);
    let b5 = T::from_f64(1.330274429);
    let p  = T::from_f64(0.2316419);

    let ax = x.abs();
    let t = T::one() / (T::one() + p * ax);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;

    let pdf = normal_pdf(ax);
    let poly = b1 * t + b2 * t2 + b3 * t3 + b4 * t4 + b5 * t5;
    let cdf_positive = T::one() - pdf * poly;

    // For x > 0: Φ(x) = cdf_positive
    // For x < 0: Φ(x) = 1 - cdf_positive  (by symmetry)
    // Use arithmetic branch-free approach to propagate derivatives correctly.
    if x.to_f64() >= 0.0 {
        cdf_positive
    } else {
        T::one() - cdf_positive
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // Reference values from scipy.stats.norm
    #[test]
    fn pdf_at_zero() {
        let v: f64 = normal_pdf(0.0);
        assert_abs_diff_eq!(v, 0.3989422804014327, epsilon = 1e-12);
    }

    #[test]
    fn pdf_at_one() {
        let v: f64 = normal_pdf(1.0);
        assert_abs_diff_eq!(v, 0.24197072451914337, epsilon = 1e-12);
    }

    #[test]
    fn pdf_symmetry() {
        let a: f64 = normal_pdf(1.5);
        let b: f64 = normal_pdf(-1.5);
        assert_abs_diff_eq!(a, b, epsilon = 1e-15);
    }

    #[test]
    fn cdf_at_zero() {
        let v: f64 = normal_cdf(0.0);
        assert_abs_diff_eq!(v, 0.5, epsilon = 1e-7);
    }

    #[test]
    fn cdf_at_one() {
        let v: f64 = normal_cdf(1.0);
        assert_abs_diff_eq!(v, 0.8413447460685429, epsilon = 1e-7);
    }

    #[test]
    fn cdf_at_minus_one() {
        let v: f64 = normal_cdf(-1.0);
        assert_abs_diff_eq!(v, 0.15865525393145702, epsilon = 1e-7);
    }

    #[test]
    fn cdf_at_two() {
        let v: f64 = normal_cdf(2.0);
        assert_abs_diff_eq!(v, 0.9772498680518208, epsilon = 1e-7);
    }

    #[test]
    fn cdf_symmetry() {
        let a: f64 = normal_cdf(1.5);
        let b: f64 = normal_cdf(-1.5);
        assert_abs_diff_eq!(a + b, 1.0, epsilon = 1e-7);
    }

    #[test]
    fn cdf_extreme_positive() {
        let v: f64 = normal_cdf(6.0);
        assert!(v > 0.999999);
        assert!(v <= 1.0);
    }

    #[test]
    fn cdf_extreme_negative() {
        let v: f64 = normal_cdf(-6.0);
        assert!(v < 0.000001);
        assert!(v >= 0.0);
    }

    #[test]
    fn pdf_dual_derivative() {
        // d/dx φ(x) = -x φ(x)
        use crate::dual::Dual;
        let x = Dual::variable(1.0);
        let pdf = normal_pdf(x);
        let expected_derivative = -1.0 * normal_pdf(1.0_f64);
        assert_abs_diff_eq!(pdf.dot, expected_derivative, epsilon = 1e-12);
    }

    #[test]
    fn cdf_dual_derivative_is_pdf() {
        // d/dx Φ(x) = φ(x)
        use crate::dual::Dual;
        let x = Dual::variable(0.5);
        let cdf = normal_cdf(x);
        let pdf_val: f64 = normal_pdf(0.5);
        assert_abs_diff_eq!(cdf.dot, pdf_val, epsilon = 1e-6);
    }

    #[test]
    fn cdf_dual_derivative_negative() {
        // d/dx Φ(x) = φ(x) even for negative x
        use crate::dual::Dual;
        let x = Dual::variable(-1.0);
        let cdf = normal_cdf(x);
        let pdf_val: f64 = normal_pdf(-1.0);
        assert_abs_diff_eq!(cdf.dot, pdf_val, epsilon = 1e-6);
    }
}
