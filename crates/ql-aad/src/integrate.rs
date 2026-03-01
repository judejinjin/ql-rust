//! Generic numerical integration — AD-aware quadrature.
//!
//! Quadrature nodes and weights stay `f64`; only the function values and
//! accumulator are generic `T: Number`, so derivatives propagate through
//! the integrand while the mechanical parts of the quadrature are unaffected.
//!
//! # Example
//!
//! ```
//! use ql_aad::integrate::{gl_integrate, simpson_integrate};
//! use ql_aad::{Dual, Number};
//!
//! // Integrate x² from 0 to 1 = 1/3
//! let result: f64 = gl_integrate(|x: f64| x * x, 0.0, 1.0, 20);
//! assert!((result - 1.0/3.0).abs() < 1e-12);
//!
//! // With Dual: integrate f(x; a) = a*x² with ∂/∂a ∫₀¹ a·x² dx = 1/3
//! let a = Dual::variable(2.0);
//! let result: Dual = gl_integrate(|x: Dual| a * x * x, 0.0, 1.0, 20);
//! assert!((result.val - 2.0/3.0).abs() < 1e-12);
//! assert!((result.dot - 1.0/3.0).abs() < 1e-12);
//! ```

use crate::number::Number;

// ===========================================================================
// Gauss-Legendre nodes and weights
// ===========================================================================

/// 10-point Gauss-Legendre nodes and weights on [-1, 1].
fn gl10() -> &'static [(f64, f64)] {
    static NODES: [(f64, f64); 10] = [
        (-0.973906528517171720, 0.066671344308688138),
        (-0.865063366688984511, 0.149451349150580594),
        (-0.679409568299024406, 0.219086362515982044),
        (-0.433395394129247191, 0.269266719309996355),
        (-0.148874338981631211, 0.295524224714752870),
        ( 0.148874338981631211, 0.295524224714752870),
        ( 0.433395394129247191, 0.269266719309996355),
        ( 0.679409568299024406, 0.219086362515982044),
        ( 0.865063366688984511, 0.149451349150580594),
        ( 0.973906528517171720, 0.066671344308688138),
    ];
    &NODES
}

/// 20-point Gauss-Legendre nodes and weights on [-1, 1].
fn gl20() -> &'static [(f64, f64)] {
    static NODES: [(f64, f64); 20] = [
        ( 0.076526521133497334, 0.152753387130725851),
        ( 0.227785851141645078, 0.149172986472603747),
        ( 0.373706088715419561, 0.142096109318382051),
        ( 0.510867001950827098, 0.131688638449176627),
        ( 0.636053680726515025, 0.118194531961518417),
        ( 0.746331906460150793, 0.101930119817240435),
        ( 0.839116971822218823, 0.083276741576704749),
        ( 0.912234428251325906, 0.062672048334109064),
        ( 0.963971927277913791, 0.040601429800386941),
        ( 0.993128599185094925, 0.017614007139152118),
        (-0.076526521133497334, 0.152753387130725851),
        (-0.227785851141645078, 0.149172986472603747),
        (-0.373706088715419561, 0.142096109318382051),
        (-0.510867001950827098, 0.131688638449176627),
        (-0.636053680726515025, 0.118194531961518417),
        (-0.746331906460150793, 0.101930119817240435),
        (-0.839116971822218823, 0.083276741576704749),
        (-0.912234428251325906, 0.062672048334109064),
        (-0.963971927277913791, 0.040601429800386941),
        (-0.993128599185094925, 0.017614007139152118),
    ];
    &NODES
}

/// Get GL nodes/weights for a given order (10 or 20).
fn gl_nodes(order: usize) -> &'static [(f64, f64)] {
    match order {
        10 => gl10(),
        _ => gl20(), // default to 20
    }
}

// ===========================================================================
// Gauss-Legendre integration (generic)
// ===========================================================================

/// Integrate `f` over `[a, b]` using Gauss-Legendre quadrature.
///
/// The function `f` maps `T → T`; nodes/weights are f64. The quadrature
/// variable `x` is lifted to `T` via `from_f64`. Any AD-active parameters
/// captured in `f`'s closure will propagate derivatives through the result.
///
/// `order` selects 10 or 20 points (20 is default for anything else).
pub fn gl_integrate<T: Number, F: Fn(T) -> T>(f: F, a: f64, b: f64, order: usize) -> T {
    let half_range = 0.5 * (b - a);
    let mid = 0.5 * (a + b);
    let nodes = gl_nodes(order);
    let mut sum = T::zero();
    for &(xi, wi) in nodes {
        let x = T::from_f64(mid + half_range * xi);
        sum = sum + f(x) * T::from_f64(wi);
    }
    sum * T::from_f64(half_range)
}

/// Integrate `f` over `[a, b]` where the function takes f64 input but
/// returns `T: Number`. Useful when the integration variable is NOT an AD
/// input, but the integrand result carries AD information (e.g. Heston CF).
pub fn gl_integrate_f64_to_t<T: Number, F: Fn(f64) -> T>(f: F, a: f64, b: f64, order: usize) -> T {
    let half_range = 0.5 * (b - a);
    let mid = 0.5 * (a + b);
    let nodes = gl_nodes(order);
    let mut sum = T::zero();
    for &(xi, wi) in nodes {
        let x = mid + half_range * xi;
        sum = sum + f(x) * T::from_f64(wi);
    }
    sum * T::from_f64(half_range)
}

// ===========================================================================
// Simpson's rule (generic)
// ===========================================================================

/// Composite Simpson's 1/3 rule — generic over `T: Number`.
///
/// `n` must be even (rounded up if odd). The integration variable is lifted
/// to `T` at each evaluation point.
pub fn simpson_integrate<T: Number, F: Fn(T) -> T>(f: F, a: f64, b: f64, n: usize) -> T {
    let n = if n % 2 == 0 { n.max(2) } else { (n + 1).max(2) };
    let h = (b - a) / n as f64;

    let mut sum = f(T::from_f64(a)) + f(T::from_f64(b));
    for i in 1..n {
        let x = T::from_f64(a + i as f64 * h);
        let coeff = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum = sum + f(x) * T::from_f64(coeff);
    }

    sum * T::from_f64(h / 3.0)
}

// ===========================================================================
// Trapezoidal rule (generic)
// ===========================================================================

/// Composite trapezoidal rule — generic over `T: Number`.
pub fn trapezoid_integrate<T: Number, F: Fn(T) -> T>(f: F, a: f64, b: f64, n: usize) -> T {
    let n = n.max(1);
    let h = (b - a) / n as f64;
    let half = T::from_f64(0.5);

    let mut sum = (f(T::from_f64(a)) + f(T::from_f64(b))) * half;
    for i in 1..n {
        let x = T::from_f64(a + i as f64 * h);
        sum = sum + f(x);
    }

    sum * T::from_f64(h)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn gl_x_squared() {
        // ∫₀¹ x² dx = 1/3
        let r: f64 = gl_integrate(|x: f64| x * x, 0.0, 1.0, 20);
        assert_abs_diff_eq!(r, 1.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn gl_exponential() {
        // ∫₀¹ eˣ dx = e - 1
        let r: f64 = gl_integrate(|x: f64| x.exp(), 0.0, 1.0, 20);
        assert_abs_diff_eq!(r, 1.0_f64.exp() - 1.0, epsilon = 1e-12);
    }

    #[test]
    fn gl_sin() {
        // ∫₀^π sin(x) dx = 2
        let r: f64 = gl_integrate(|x: f64| x.sin(), 0.0, std::f64::consts::PI, 20);
        assert_abs_diff_eq!(r, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn gl_dual_parameter_sensitivity() {
        use crate::dual::Dual;
        // f(x; a) = a * x², ∫₀¹ a*x² dx = a/3
        // ∂/∂a (a/3) = 1/3
        let a = Dual::variable(2.0);
        let r: Dual = gl_integrate(|x: Dual| a * x * x, 0.0, 1.0, 20);
        assert_abs_diff_eq!(r.val, 2.0 / 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.dot, 1.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn gl_dual_exp_param() {
        use crate::dual::Dual;
        // ∫₀¹ exp(a*x) dx = (exp(a) - 1)/a
        // ∂/∂a = (a*exp(a) - exp(a) + 1) / a²
        let a = Dual::variable(1.0);
        let r: Dual = gl_integrate(|x: Dual| (a * x).exp(), 0.0, 1.0, 20);
        let expected = 1.0_f64.exp() - 1.0; // (e-1)
        // ∂/∂a ∫₀¹ exp(ax) dx = ∫₀¹ x·exp(ax) dx
        // = (a·exp(a) - exp(a) + 1)/a²
        // For a=1: (e - e + 1)/1 = 1.0
        let expected_d_val = 1.0 * 1.0_f64.exp() - 1.0_f64.exp() + 1.0; // = 1.0
        assert_abs_diff_eq!(r.val, expected, epsilon = 1e-10);
        assert_abs_diff_eq!(r.dot, expected_d_val, epsilon = 1e-10);
    }

    #[test]
    fn gl_areal() {
        use crate::tape::{with_tape, adjoint_tl, AReal};
        // ∫₀¹ a*x² dx = a/3, ∂/∂a = 1/3
        let (result, a_ar) = with_tape(|tape| {
            let a = tape.input(2.0);
            let r: AReal = gl_integrate(|x: AReal| a * x * x, 0.0, 1.0, 20);
            (r, a)
        });
        let grad = adjoint_tl(result);
        assert_abs_diff_eq!(result.val, 2.0 / 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(grad[a_ar.idx], 1.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn gl10_vs_gl20_accuracy() {
        // gl20 should be more accurate for higher-degree polynomials
        let exact = 1.0 / 7.0; // ∫₀¹ x⁶ dx
        let r10: f64 = gl_integrate(|x: f64| x.powi(6), 0.0, 1.0, 10);
        let r20: f64 = gl_integrate(|x: f64| x.powi(6), 0.0, 1.0, 20);
        assert!((r20 - exact).abs() <= (r10 - exact).abs() || (r10 - exact).abs() < 1e-12);
    }

    #[test]
    fn simpson_x_squared() {
        let r: f64 = simpson_integrate(|x: f64| x * x, 0.0, 1.0, 100);
        assert_abs_diff_eq!(r, 1.0 / 3.0, epsilon = 1e-8);
    }

    #[test]
    fn simpson_dual() {
        use crate::dual::Dual;
        let a = Dual::variable(3.0);
        let r: Dual = simpson_integrate(|x: Dual| a * x * x, 0.0, 1.0, 100);
        assert_abs_diff_eq!(r.val, 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(r.dot, 1.0 / 3.0, epsilon = 1e-6);
    }

    #[test]
    fn trapezoid_x_cubed() {
        let r: f64 = trapezoid_integrate(|x: f64| x * x * x, 0.0, 1.0, 1000);
        assert_abs_diff_eq!(r, 0.25, epsilon = 1e-4);
    }

    #[test]
    fn gl_f64_to_t() {
        use crate::dual::Dual;
        // Test the variant where the integration variable is f64
        let a = Dual::variable(2.0);
        let r: Dual = gl_integrate_f64_to_t(|x: f64| a * Dual::from_f64(x * x), 0.0, 1.0, 20);
        assert_abs_diff_eq!(r.val, 2.0 / 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.dot, 1.0 / 3.0, epsilon = 1e-12);
    }
}
