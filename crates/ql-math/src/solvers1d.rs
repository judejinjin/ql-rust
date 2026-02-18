//! One-dimensional root-finding solvers.
//!
//! Provides Brent, Newton, and Bisection methods implementing the `Solver1D` trait.

use ql_core::errors::{QLError, QLResult};

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Common interface for 1-D root-finding solvers.
///
/// Finds `x` such that `f(x) = target` within the bracket `[x_min, x_max]`.
pub trait Solver1D {
    #[allow(clippy::too_many_arguments)]
    fn solve<F: Fn(f64) -> f64>(
        &self,
        f: F,
        target: f64,
        guess: f64,
        x_min: f64,
        x_max: f64,
        accuracy: f64,
        max_evaluations: usize,
    ) -> QLResult<f64>;
}

// ===========================================================================
// Brent's Method
// ===========================================================================

/// Brent's method — combines bisection, secant, and inverse quadratic interpolation.
///
/// This is the primary solver used in yield-curve bootstrapping.
#[derive(Clone, Debug, Default)]
pub struct Brent;

impl Solver1D for Brent {
    fn solve<F: Fn(f64) -> f64>(
        &self,
        f: F,
        target: f64,
        _guess: f64,
        x_min: f64,
        x_max: f64,
        accuracy: f64,
        max_evaluations: usize,
    ) -> QLResult<f64> {
        let g = |x: f64| f(x) - target;

        let mut a = x_min;
        let mut b = x_max;
        let mut fa = g(a);
        let mut fb = g(b);
        let mut evals = 2;

        if fa * fb > 0.0 {
            return Err(QLError::RootNotFound(evals));
        }

        let mut c = a;
        let mut fc = fa;
        let mut d = b - a;
        let mut e = d;

        loop {
            if fb * fc > 0.0 {
                c = a;
                fc = fa;
                d = b - a;
                e = d;
            }
            if fc.abs() < fb.abs() {
                a = b;
                b = c;
                c = a;
                fa = fb;
                fb = fc;
                fc = fa;
            }

            let tol = 2.0 * f64::EPSILON * b.abs() + 0.5 * accuracy;
            let m = 0.5 * (c - b);

            if m.abs() <= tol || fb.abs() <= accuracy {
                return Ok(b);
            }

            if evals >= max_evaluations {
                return Err(QLError::RootNotFound(evals));
            }

            if e.abs() >= tol && fa.abs() > fb.abs() {
                // Attempt inverse quadratic interpolation
                let s = fb / fa;
                let (p, q) = if (a - c).abs() < 1e-30 {
                    let p = 2.0 * m * s;
                    let q = 1.0 - s;
                    (p, q)
                } else {
                    let q0 = fa / fc;
                    let r = fb / fc;
                    let p = s * (2.0 * m * q0 * (q0 - r) - (b - a) * (r - 1.0));
                    let q = (q0 - 1.0) * (r - 1.0) * (s - 1.0);
                    (p, q)
                };

                let (p, q) = if p > 0.0 { (p, -q) } else { (-p, q) };

                let s = e;
                e = d;

                if 2.0 * p < 3.0 * m * q - (tol * q).abs() && p < (0.5 * s * q).abs() {
                    d = p / q;
                } else {
                    d = m;
                    e = m;
                }
            } else {
                d = m;
                e = m;
            }

            a = b;
            fa = fb;

            if d.abs() > tol {
                b += d;
            } else {
                b += if m > 0.0 { tol } else { -tol };
            }

            fb = g(b);
            evals += 1;
        }
    }
}

// ===========================================================================
// Newton's Method
// ===========================================================================

/// Newton-Raphson solver — requires function and derivative.
///
/// The `f` closure should return the function value; the derivative is supplied
/// via the `solve_with_derivative` method. For the `Solver1D` trait, a finite
/// difference approximation is used.
#[derive(Clone, Debug, Default)]
pub struct Newton;

impl Newton {
    /// Solve using an analytic derivative.
    pub fn solve_with_derivative<F, D>(
        &self,
        f: F,
        df: D,
        target: f64,
        guess: f64,
        accuracy: f64,
        max_evaluations: usize,
    ) -> QLResult<f64>
    where
        F: Fn(f64) -> f64,
        D: Fn(f64) -> f64,
    {
        let mut x = guess;
        for i in 0..max_evaluations {
            let fx = f(x) - target;
            if fx.abs() < accuracy {
                return Ok(x);
            }
            let dfx = df(x);
            if dfx.abs() < 1e-30 {
                return Err(QLError::RootNotFound(i + 1));
            }
            x -= fx / dfx;
        }
        Err(QLError::RootNotFound(max_evaluations))
    }
}

impl Solver1D for Newton {
    fn solve<F: Fn(f64) -> f64>(
        &self,
        f: F,
        target: f64,
        guess: f64,
        x_min: f64,
        x_max: f64,
        accuracy: f64,
        max_evaluations: usize,
    ) -> QLResult<f64> {
        let h = 1e-8;
        let mut x = guess;
        for i in 0..max_evaluations {
            let fx = f(x) - target;
            if fx.abs() < accuracy {
                return Ok(x);
            }
            let dfx = (f(x + h) - f(x - h)) / (2.0 * h);
            if dfx.abs() < 1e-30 {
                return Err(QLError::RootNotFound(i + 1));
            }
            let x_new = x - fx / dfx;
            // Clamp to bracket
            x = x_new.clamp(x_min, x_max);
        }
        Err(QLError::RootNotFound(max_evaluations))
    }
}

// ===========================================================================
// Bisection
// ===========================================================================

/// Simple bisection solver — guaranteed convergence but slow.
#[derive(Clone, Debug, Default)]
pub struct Bisection;

impl Solver1D for Bisection {
    fn solve<F: Fn(f64) -> f64>(
        &self,
        f: F,
        target: f64,
        _guess: f64,
        x_min: f64,
        x_max: f64,
        accuracy: f64,
        max_evaluations: usize,
    ) -> QLResult<f64> {
        let g = |x: f64| f(x) - target;

        let mut a = x_min;
        let mut b = x_max;
        let mut fa = g(a);
        let fb = g(b);

        if fa * fb > 0.0 {
            return Err(QLError::RootNotFound(2));
        }

        for i in 0..max_evaluations {
            let mid = 0.5 * (a + b);
            let fm = g(mid);

            if fm.abs() < accuracy || (b - a) * 0.5 < accuracy {
                return Ok(mid);
            }

            if fa * fm < 0.0 {
                b = mid;
            } else {
                a = mid;
                fa = fm;
            }

            if i + 3 >= max_evaluations {
                return Err(QLError::RootNotFound(i + 3));
            }
        }

        Ok(0.5 * (a + b))
    }
}

// ===========================================================================
// Secant Method
// ===========================================================================

/// Secant method — derivative-free, superlinear convergence.
///
/// Uses two initial evaluations and does not require a bracket.
#[derive(Clone, Debug, Default)]
pub struct Secant;

impl Solver1D for Secant {
    fn solve<F: Fn(f64) -> f64>(
        &self,
        f: F,
        target: f64,
        guess: f64,
        x_min: f64,
        x_max: f64,
        accuracy: f64,
        max_evaluations: usize,
    ) -> QLResult<f64> {
        let g = |x: f64| f(x) - target;

        let dx = guess.abs().max(1.0) * 1e-4;
        let mut x0 = guess;
        let mut x1 = guess + dx;
        x1 = x1.clamp(x_min, x_max);
        if (x1 - x0).abs() < 1e-30 {
            x0 = guess - dx;
        }

        let mut f0 = g(x0);
        let mut f1 = g(x1);
        let mut evals = 2;

        for _ in 0..max_evaluations {
            if f1.abs() < accuracy {
                return Ok(x1);
            }
            if evals >= max_evaluations {
                return Err(QLError::RootNotFound(evals));
            }

            let denom = f1 - f0;
            if denom.abs() < 1e-30 {
                return Err(QLError::RootNotFound(evals));
            }

            let x2 = x1 - f1 * (x1 - x0) / denom;
            let x2 = x2.clamp(x_min, x_max);

            x0 = x1;
            f0 = f1;
            x1 = x2;
            f1 = g(x1);
            evals += 1;
        }

        Err(QLError::RootNotFound(max_evaluations))
    }
}

// ===========================================================================
// Ridder's Method
// ===========================================================================

/// Ridder's method — bracketing method with quadratic convergence.
#[derive(Clone, Debug, Default)]
pub struct Ridder;

impl Solver1D for Ridder {
    fn solve<F: Fn(f64) -> f64>(
        &self,
        f: F,
        target: f64,
        _guess: f64,
        x_min: f64,
        x_max: f64,
        accuracy: f64,
        max_evaluations: usize,
    ) -> QLResult<f64> {
        let g = |x: f64| f(x) - target;

        let mut a = x_min;
        let mut b = x_max;
        let mut fa = g(a);
        let mut fb = g(b);
        let mut evals = 2;

        if fa * fb > 0.0 {
            return Err(QLError::RootNotFound(evals));
        }
        if fa.abs() < accuracy {
            return Ok(a);
        }
        if fb.abs() < accuracy {
            return Ok(b);
        }

        for _ in 0..max_evaluations / 2 {
            let mid = 0.5 * (a + b);
            let fm = g(mid);
            evals += 1;

            if fm.abs() < accuracy {
                return Ok(mid);
            }

            let s = (fm * fm - fa * fb).sqrt();
            if s < 1e-30 {
                return Ok(mid);
            }

            let sign = if (fa - fb) > 0.0 { 1.0 } else { -1.0 };
            let x_new = mid + (mid - a) * sign * fm / s;
            let fx = g(x_new);
            evals += 1;

            if fx.abs() < accuracy {
                return Ok(x_new);
            }

            // Update bracket
            if fm * fx < 0.0 {
                if mid < x_new {
                    a = mid;
                    fa = fm;
                    b = x_new;
                    fb = fx;
                } else {
                    a = x_new;
                    fa = fx;
                    b = mid;
                    fb = fm;
                }
            } else if fa * fx < 0.0 {
                b = x_new;
                fb = fx;
            } else {
                a = x_new;
                fa = fx;
            }

            if (b - a).abs() < accuracy {
                return Ok(0.5 * (a + b));
            }

            if evals >= max_evaluations {
                return Err(QLError::RootNotFound(evals));
            }
        }

        Ok(0.5 * (a + b))
    }
}

// ===========================================================================
// False Position (Regula Falsi)
// ===========================================================================

/// False position (regula falsi) method — bracketing with linear interpolation.
///
/// Illinois variant for improved convergence.
#[derive(Clone, Debug, Default)]
pub struct FalsePosition;

impl Solver1D for FalsePosition {
    fn solve<F: Fn(f64) -> f64>(
        &self,
        f: F,
        target: f64,
        _guess: f64,
        x_min: f64,
        x_max: f64,
        accuracy: f64,
        max_evaluations: usize,
    ) -> QLResult<f64> {
        let g = |x: f64| f(x) - target;

        let mut a = x_min;
        let mut b = x_max;
        let mut fa = g(a);
        let mut fb = g(b);
        let mut evals = 2;

        if fa * fb > 0.0 {
            return Err(QLError::RootNotFound(evals));
        }

        let mut side = 0i32; // 0=none, 1=left retained, 2=right retained

        for _ in 0..max_evaluations {
            let denom = fb - fa;
            if denom.abs() < 1e-30 {
                return Ok(0.5 * (a + b));
            }

            let c = (a * fb - b * fa) / denom;
            let fc = g(c);
            evals += 1;

            if fc.abs() < accuracy || (b - a).abs() < accuracy {
                return Ok(c);
            }

            if fa * fc < 0.0 {
                b = c;
                fb = fc;
                if side == 1 {
                    fa *= 0.5; // Illinois modification
                }
                side = 1;
            } else {
                a = c;
                fa = fc;
                if side == 2 {
                    fb *= 0.5; // Illinois modification
                }
                side = 2;
            }

            if evals >= max_evaluations {
                return Err(QLError::RootNotFound(evals));
            }
        }

        Ok(0.5 * (a + b))
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    const ACC: f64 = 1e-12;
    const MAX_EVAL: usize = 1000;
    const SQRT2: f64 = std::f64::consts::SQRT_2;

    // f(x) = x^2, root of f(x) = 2 at sqrt(2)
    fn f_quadratic(x: f64) -> f64 {
        x * x
    }

    #[test]
    fn brent_sqrt2() {
        let solver = Brent;
        let root = solver
            .solve(f_quadratic, 2.0, 1.5, 0.0, 3.0, ACC, MAX_EVAL)
            .unwrap();
        assert_abs_diff_eq!(root, SQRT2, epsilon = ACC);
    }

    #[test]
    fn brent_negative_root() {
        let solver = Brent;
        let root = solver
            .solve(f_quadratic, 2.0, -1.5, -3.0, 0.0, ACC, MAX_EVAL)
            .unwrap();
        assert_abs_diff_eq!(root, -SQRT2, epsilon = ACC);
    }

    #[test]
    fn newton_sqrt2() {
        let solver = Newton;
        let root = solver
            .solve(f_quadratic, 2.0, 1.5, 0.0, 3.0, ACC, MAX_EVAL)
            .unwrap();
        assert_abs_diff_eq!(root, SQRT2, epsilon = 1e-10);
    }

    #[test]
    fn newton_with_analytic_derivative() {
        let solver = Newton;
        let root = solver
            .solve_with_derivative(
                |x| x * x,
                |x| 2.0 * x,
                2.0,
                1.5,
                ACC,
                MAX_EVAL,
            )
            .unwrap();
        assert_abs_diff_eq!(root, SQRT2, epsilon = ACC);
    }

    #[test]
    fn bisection_sqrt2() {
        let solver = Bisection;
        let root = solver
            .solve(f_quadratic, 2.0, 1.5, 0.0, 3.0, ACC, MAX_EVAL)
            .unwrap();
        assert_abs_diff_eq!(root, SQRT2, epsilon = ACC);
    }

    #[test]
    fn brent_no_bracket() {
        let solver = Brent;
        // f(2)=4 > 2 and f(3)=9 > 2, so g(x)=x^2-2 has no sign change in [2,3]
        let result = solver.solve(f_quadratic, 2.0, 2.5, 2.0, 3.0, ACC, MAX_EVAL);
        assert!(result.is_err());
    }

    #[test]
    fn solvers_on_cosine() {
        // cos(x) = 0 => x = π/2
        let target = 0.0;
        let expected = std::f64::consts::FRAC_PI_2;

        let root = Brent
            .solve(f64::cos, target, 1.0, 0.0, 2.0, ACC, MAX_EVAL)
            .unwrap();
        assert_abs_diff_eq!(root, expected, epsilon = ACC);

        let root = Bisection
            .solve(f64::cos, target, 1.0, 0.0, 2.0, ACC, MAX_EVAL)
            .unwrap();
        assert_abs_diff_eq!(root, expected, epsilon = ACC);
    }

    #[test]
    fn brent_cubic() {
        // x^3 - 6x^2 + 11x - 6 = 0 has roots at 1, 2, 3
        let f = |x: f64| x * x * x - 6.0 * x * x + 11.0 * x - 6.0;
        let root = Brent
            .solve(f, 0.0, 2.5, 1.5, 2.5, ACC, MAX_EVAL)
            .unwrap();
        assert_abs_diff_eq!(root, 2.0, epsilon = ACC);
    }

    #[test]
    fn newton_exp() {
        // e^x = 2 => x = ln(2)
        let root = Newton
            .solve(f64::exp, 2.0, 0.5, -10.0, 10.0, ACC, MAX_EVAL)
            .unwrap();
        assert_abs_diff_eq!(root, 2.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn secant_sqrt2() {
        let solver = Secant;
        let root = solver
            .solve(f_quadratic, 2.0, 1.5, 0.0, 3.0, ACC, MAX_EVAL)
            .unwrap();
        assert_abs_diff_eq!(root, SQRT2, epsilon = 1e-10);
    }

    #[test]
    fn ridder_sqrt2() {
        let solver = Ridder;
        let root = solver
            .solve(f_quadratic, 2.0, 1.5, 0.0, 3.0, ACC, MAX_EVAL)
            .unwrap();
        assert_abs_diff_eq!(root, SQRT2, epsilon = ACC);
    }

    #[test]
    fn false_position_sqrt2() {
        let solver = FalsePosition;
        let root = solver
            .solve(f_quadratic, 2.0, 1.5, 0.0, 3.0, ACC, MAX_EVAL)
            .unwrap();
        assert_abs_diff_eq!(root, SQRT2, epsilon = ACC);
    }

    #[test]
    fn secant_cosine() {
        let root = Secant
            .solve(f64::cos, 0.0, 1.0, 0.0, 2.0, ACC, MAX_EVAL)
            .unwrap();
        assert_abs_diff_eq!(root, std::f64::consts::FRAC_PI_2, epsilon = 1e-10);
    }

    #[test]
    fn ridder_cubic() {
        let f = |x: f64| x * x * x - 6.0 * x * x + 11.0 * x - 6.0;
        let root = Ridder
            .solve(f, 0.0, 2.5, 1.5, 2.5, ACC, MAX_EVAL)
            .unwrap();
        assert_abs_diff_eq!(root, 2.0, epsilon = ACC);
    }

    #[test]
    fn false_position_exp() {
        let root = FalsePosition
            .solve(f64::exp, 2.0, 0.5, -10.0, 10.0, ACC, MAX_EVAL)
            .unwrap();
        assert_abs_diff_eq!(root, 2.0_f64.ln(), epsilon = ACC);
    }
}
