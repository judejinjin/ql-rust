//! AD-powered root-finding and implicit differentiation.
//!
//! # Approach
//!
//! Classical solvers like Brent and Bisection use sign-comparison logic
//! (`fa * fb > 0.0`) that is meaningless for AD types. Instead of
//! genericising those, we provide:
//!
//! 1. **`newton_ad`**: Newton's method that uses `Dual` internally — the
//!    user supplies only `f`, and the derivative comes for free.
//!
//! 2. **`newton_1d`**: Newton's method working purely in f64 but using
//!    `Dual` for the Jacobian (convenient for non-AD callers too).
//!
//! 3. **`implicit_diff`**: Given a root `x*` of `f(x, θ) = 0` (already
//!    found with any solver), compute `dx*/dθ` via the implicit function
//!    theorem so that AD can propagate through the solved quantity.
//!
//! # Example
//!
//! ```
//! use ql_aad::solvers::{newton_1d, implicit_diff};
//! use ql_aad::{Dual, Number};
//!
//! // Find sqrt(2) by solving x² - 2 = 0
//! let root = newton_1d(|x: Dual| x * x - Dual::from_f64(2.0), 1.0, 1e-14, 100).unwrap();
//! assert!((root - std::f64::consts::SQRT_2).abs() < 1e-12);
//! ```

use crate::dual::Dual;
use crate::number::Number;

/// Error returned by solver functions.
#[derive(Debug, Clone)]
pub enum SolverError {
    /// Newton failed to converge within `max_iter` iterations.
    MaxIterationsExceeded {
        /// Field.
        last_x: f64,
        /// Field.
        last_fx: f64,
        /// Field.
        iterations: usize,
    },
    /// Derivative was zero (or nearly zero), cannot proceed.
    ZeroDerivative {
        /// Field.
        x: f64,
        /// Field.
        fx: f64,
        /// Field.
        iteration: usize,
    },
}

impl std::fmt::Display for SolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolverError::MaxIterationsExceeded { last_x, last_fx, iterations } => {
                write!(f, "Newton solver: max iterations ({}) exceeded, x={}, f(x)={}", iterations, last_x, last_fx)
            }
            SolverError::ZeroDerivative { x, fx, iteration } => {
                write!(f, "Newton solver: zero derivative at x={}, f(x)={}, iteration {}", x, fx, iteration)
            }
        }
    }
}

impl std::error::Error for SolverError {}

// ===========================================================================
// Newton with Dual autodiff
// ===========================================================================

/// Newton's method using `Dual` to compute the derivative automatically.
///
/// The user provides `f: Dual → Dual` (the function to find a root of).
/// At each step, `x` is seeded as a `Dual::variable(x)` and both `f(x)` and
/// `f'(x)` are obtained in a single evaluation.
///
/// Returns the root as `f64` or an error.
pub fn newton_1d<F>(f: F, x0: f64, tol: f64, max_iter: usize) -> Result<f64, SolverError>
where
    F: Fn(Dual) -> Dual,
{
    let mut x = x0;
    for i in 0..max_iter {
        let input = Dual::variable(x);
        let output = f(input);

        if output.val.abs() < tol {
            return Ok(x);
        }

        if output.dot.abs() < 1e-30 {
            return Err(SolverError::ZeroDerivative {
                x,
                fx: output.val,
                iteration: i,
            });
        }

        x -= output.val / output.dot;
    }

    let final_output = f(Dual::variable(x));
    if final_output.val.abs() < tol {
        Ok(x)
    } else {
        Err(SolverError::MaxIterationsExceeded {
            last_x: x,
            last_fx: final_output.val,
            iterations: max_iter,
        })
    }
}

/// Newton's method generic over `T: Number`.
///
/// The function `f` maps `T → T`. At each step, we wrap `x` into a `Dual`
/// internally to get the derivative. The returned root is `f64`.
///
/// This is a convenience for callers who want the root as a plain number.
/// Use [`implicit_diff`] to propagate AD through the solved root.
pub fn newton_ad<F>(f: F, x0: f64, tol: f64, max_iter: usize) -> Result<f64, SolverError>
where
    F: Fn(Dual) -> Dual,
{
    newton_1d(f, x0, tol, max_iter)
}

// ===========================================================================
// Implicit differentiation
// ===========================================================================

/// Propagate AD through a solved root via the implicit function theorem.
///
/// Given that `g(x*, θ) = 0` defines the root `x*` as a function of
/// parameters `θ`, the implicit function theorem gives:
///
/// ```text
///   dx*/dθ = −(∂g/∂θ) / (∂g/∂x)
/// ```
///
/// This function returns `x*` as a `T: Number` carrying the correct
/// derivative with respect to any AD-active parameters in `θ` that appear
/// in the closure `g`.
///
/// # Arguments
///
/// * `root` — The solved root value `x*` (plain f64, found by any solver).
/// * `g` — The residual function `g(x, θ)` mapping `T → T`. The AD-active
///   parameters θ should be captured in the closure.
/// * `dg_dx` — The partial derivative `∂g/∂x` evaluated at `x*`, as f64.
///   Callers can obtain this with `Dual` in advance, or supply it analytically.
///
/// # How it works
///
/// We evaluate `g(from_f64(root))` which produces a `T` whose derivative
/// part is `∂g/∂θ` (since `from_f64(root)` is a constant w.r.t. θ).
/// Then we scale by `-1/dg_dx` to get `dx*/dθ`.
///
/// # Example
///
/// ```
/// use ql_aad::solvers::{newton_1d, implicit_diff};
/// use ql_aad::{Dual, Number};
///
/// // Solve x² - a = 0 for x given a = 2.
/// // x* = sqrt(a), dx*/da = 1/(2*sqrt(a))
/// let a = Dual::variable(2.0);
///
/// // First, find the root with f64 (using Dual internally for Newton):
/// let root = newton_1d(|x: Dual| x * x - Dual::from_f64(2.0), 1.0, 1e-14, 100).unwrap();
///
/// // dg/dx = 2*root
/// let dg_dx = 2.0 * root;
///
/// // Now get x* as a Dual carrying dx*/da:
/// let x_star: Dual = implicit_diff(root, |x: Dual| x * x - a, dg_dx);
///
/// assert!((x_star.val - std::f64::consts::SQRT_2).abs() < 1e-12);
/// assert!((x_star.dot - 1.0 / (2.0 * std::f64::consts::SQRT_2)).abs() < 1e-10);
/// ```
pub fn implicit_diff<T: Number, F: Fn(T) -> T>(root: f64, g: F, dg_dx: f64) -> T {
    // Evaluate g at x = root (as a constant, so only θ-derivatives flow).
    let g_at_root = g(T::from_f64(root));
    // g_at_root.val should be ~0 (root condition).
    // g_at_root.derivative part = ∂g/∂θ
    // We want dx*/dθ = -(∂g/∂θ) / (∂g/∂x)
    // = -(g_at_root - 0) / dg_dx, viewed as a T
    // But g_at_root is T with val ≈ 0 and derivatives = ∂g/∂θ.
    // So we want:  T::from_f64(root) - g_at_root / T::from_f64(dg_dx)
    // This gives val = root - 0/dg_dx = root, dot = 0 - (∂g/∂θ)/dg_dx = -(∂g/∂θ)/dg_dx ✓
    T::from_f64(root) - g_at_root / T::from_f64(dg_dx)
}

/// Convenience: find root with `newton_1d` and then propagate AD via implicit diff.
///
/// `f_dual` is used by Newton to find the root (works on `Dual`).
/// `f_generic` is evaluated at the root with the `T`-typed AD parameters.
/// `dg_dx` at the root is obtained from the final Newton iteration automatically.
pub fn solve_and_diff<T, F1, F2>(
    f_dual: F1,
    f_generic: F2,
    x0: f64,
    tol: f64,
    max_iter: usize,
) -> Result<T, SolverError>
where
    T: Number,
    F1: Fn(Dual) -> Dual,
    F2: Fn(T) -> T,
{
    // Find the root
    let root = newton_1d(&f_dual, x0, tol, max_iter)?;

    // Get dg/dx at the root
    let at_root = f_dual(Dual::variable(root));
    let dg_dx = at_root.dot;

    if dg_dx.abs() < 1e-30 {
        return Err(SolverError::ZeroDerivative {
            x: root,
            fx: at_root.val,
            iteration: max_iter,
        });
    }

    Ok(implicit_diff(root, f_generic, dg_dx))
}

// ===========================================================================
// Halley's method (cubic convergence)
// ===========================================================================

/// Halley's method using `DualVec<1>` or finite differences for the second
/// derivative. Falls back to Newton step when the second derivative is
/// negligible.
///
/// Uses `Dual` for first derivative; second derivative from a finite-
/// difference of the first derivative. This avoids requiring DualVec
/// (which carries a different type signature).
pub fn halley_1d<F>(f: F, x0: f64, tol: f64, max_iter: usize) -> Result<f64, SolverError>
where
    F: Fn(Dual) -> Dual,
{
    let mut x = x0;
    let eps = 1e-8;

    for i in 0..max_iter {
        let out = f(Dual::variable(x));
        let fx = out.val;
        let fpx = out.dot;

        if fx.abs() < tol {
            return Ok(x);
        }

        if fpx.abs() < 1e-30 {
            return Err(SolverError::ZeroDerivative {
                x,
                fx,
                iteration: i,
            });
        }

        // Second derivative via finite difference of fpx
        let out_h = f(Dual::variable(x + eps));
        let fppx = (out_h.dot - fpx) / eps;

        // Halley step: x_{n+1} = x_n - 2*f*f' / (2*f'² - f*f'')
        let denom = 2.0 * fpx * fpx - fx * fppx;
        if denom.abs() < 1e-30 {
            // Fall back to Newton
            x -= fx / fpx;
        } else {
            x -= 2.0 * fx * fpx / denom;
        }
    }

    let final_out = f(Dual::variable(x));
    if final_out.val.abs() < tol {
        Ok(x)
    } else {
        Err(SolverError::MaxIterationsExceeded {
            last_x: x,
            last_fx: final_out.val,
            iterations: max_iter,
        })
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
    fn newton_sqrt2() {
        // x² - 2 = 0 → x = √2
        let root = newton_1d(|x: Dual| x * x - Dual::from_f64(2.0), 1.0, 1e-14, 100).unwrap();
        assert_abs_diff_eq!(root, std::f64::consts::SQRT_2, epsilon = 1e-12);
    }

    #[test]
    fn newton_cube_root() {
        // x³ - 27 = 0 → x = 3
        let root = newton_1d(
            |x: Dual| x * x * x - Dual::from_f64(27.0),
            2.0, 1e-14, 100,
        ).unwrap();
        assert_abs_diff_eq!(root, 3.0, epsilon = 1e-12);
    }

    #[test]
    fn newton_exp() {
        // eˣ - 3 = 0 → x = ln(3)
        let root = newton_1d(|x: Dual| x.exp() - Dual::from_f64(3.0), 1.0, 1e-14, 100).unwrap();
        assert_abs_diff_eq!(root, 3.0_f64.ln(), epsilon = 1e-12);
    }

    #[test]
    fn newton_trig() {
        // sin(x) = 0.5 → x = π/6 (starting near π/6)
        let root = newton_1d(
            |x: Dual| x.sin() - Dual::from_f64(0.5),
            0.5, 1e-14, 100,
        ).unwrap();
        assert_abs_diff_eq!(root, std::f64::consts::FRAC_PI_6, epsilon = 1e-12);
    }

    #[test]
    fn newton_zero_derivative_error() {
        // x² at x = 0 has f'(0) = 0 and f(0) = 0 (so actually OK).
        // But x³ - looking for root at 0, if we start there f'(0) = 0 but f(0) = 0 too.
        // Let's construct: x² + 1 = 0 (no real root) starting from 0.
        // Actually f(0) = 1 ≠ 0 and f'(0) = 0.
        let result = newton_1d(|x: Dual| x * x + Dual::from_f64(1.0), 0.0, 1e-14, 5);
        assert!(result.is_err());
    }

    #[test]
    fn implicit_diff_sqrt() {
        // f(x, a) = x² - a = 0 → x* = sqrt(a)
        // dx*/da = 1 / (2*sqrt(a))
        let a_val = 4.0;
        let root = newton_1d(
            |x: Dual| x * x - Dual::from_f64(a_val),
            1.0, 1e-14, 100,
        ).unwrap();
        assert_abs_diff_eq!(root, 2.0, epsilon = 1e-12);

        // dg/dx = 2*root = 4
        let dg_dx = 2.0 * root;

        let a = Dual::variable(a_val);
        let x_star: Dual = implicit_diff(root, |x: Dual| x * x - a, dg_dx);
        assert_abs_diff_eq!(x_star.val, 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x_star.dot, 1.0 / (2.0 * 2.0), epsilon = 1e-10); // 1/(2√4) = 0.25
    }

    #[test]
    fn implicit_diff_exp_root() {
        // f(x, a) = exp(x) - a = 0 → x* = ln(a)
        // dx*/da = 1/a
        let a_val = 5.0;
        let root = newton_1d(
            |x: Dual| x.exp() - Dual::from_f64(a_val),
            1.0, 1e-14, 100,
        ).unwrap();
        assert_abs_diff_eq!(root, 5.0_f64.ln(), epsilon = 1e-12);

        let dg_dx = root.exp(); // d/dx(e^x - a) = e^x at root = a_val
        let a = Dual::variable(a_val);
        let x_star: Dual = implicit_diff(root, |x: Dual| x.exp() - a, dg_dx);
        assert_abs_diff_eq!(x_star.val, 5.0_f64.ln(), epsilon = 1e-12);
        assert_abs_diff_eq!(x_star.dot, 1.0 / 5.0, epsilon = 1e-10);
    }

    #[test]
    fn solve_and_diff_combined() {
        let a = Dual::variable(9.0);
        let x_star: Dual = solve_and_diff(
            |x: Dual| x * x - Dual::from_f64(9.0),
            |x: Dual| x * x - a,
            2.0, 1e-14, 100,
        ).unwrap();
        assert_abs_diff_eq!(x_star.val, 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x_star.dot, 1.0 / (2.0 * 3.0), epsilon = 1e-10);
    }

    #[test]
    fn solve_and_diff_areal() {
        use crate::tape::{with_tape, adjoint_tl, AReal};
        // x² - a = 0, x* = sqrt(a), dx*/da = 1/(2*sqrt(a))
        let a_val = 16.0;
        let (x_star, a_ar) = with_tape(|tape| {
            let a = tape.input(a_val);
            let x: AReal = solve_and_diff(
                |x: Dual| x * x - Dual::from_f64(a_val),
                |x: AReal| x * x - a,
                3.0, 1e-14, 100,
            ).unwrap();
            (x, a)
        });
        let grad = adjoint_tl(x_star);
        assert_abs_diff_eq!(x_star.val, 4.0, epsilon = 1e-12);
        assert_abs_diff_eq!(grad[a_ar.idx], 1.0 / (2.0 * 4.0), epsilon = 1e-10);
    }

    #[test]
    fn halley_sqrt2() {
        let root = halley_1d(|x: Dual| x * x - Dual::from_f64(2.0), 1.0, 1e-14, 100).unwrap();
        assert_abs_diff_eq!(root, std::f64::consts::SQRT_2, epsilon = 1e-12);
    }

    #[test]
    fn halley_fewer_iterations() {
        // Halley's should converge in fewer iterations than Newton for well-behaved functions.
        let mut newton_iters = 0;
        let mut x = 1.0;
        for _ in 0..100 {
            let d = Dual::variable(x);
            let out = d * d - Dual::from_f64(2.0);
            if out.val.abs() < 1e-14 { break; }
            x -= out.val / out.dot;
            newton_iters += 1;
        }

        let mut halley_iters = 0;
        x = 1.0;
        let eps = 1e-8;
        for _ in 0..100 {
            let d = Dual::variable(x);
            let out = d * d - Dual::from_f64(2.0);
            if out.val.abs() < 1e-14 { break; }
            let out_h = Dual::variable(x + eps) * Dual::variable(x + eps) - Dual::from_f64(2.0);
            let fpp = (out_h.dot - out.dot) / eps;
            let denom = 2.0 * out.dot * out.dot - out.val * fpp;
            if denom.abs() < 1e-30 { x -= out.val / out.dot; }
            else { x -= 2.0 * out.val * out.dot / denom; }
            halley_iters += 1;
        }

        assert!(halley_iters <= newton_iters);
    }
}
