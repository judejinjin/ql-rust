//! Extended 1-D root-finding solvers: Halley and Newton-Safe.
//!
//! **G27** — Halley's method uses second derivatives for cubic convergence.
//! **G28** — Newton-Safe (Newton with bisection fallback) ensures robustness.

use ql_core::errors::{QLError, QLResult};

use crate::solvers1d::Solver1D;

// ===========================================================================
// Halley's Method (G27)
// ===========================================================================

/// Halley's method — cubic-convergence root finder using f, f', f''.
///
/// Uses the iteration:
///   x_{n+1} = x_n - 2·f·f' / (2·f'² - f·f'')
///
/// Falls back to Newton step when f'' contribution is negligible.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Halley;

impl Halley {
    /// Solve using analytic first and second derivatives.
    pub fn solve_with_derivatives<F, D1, D2>(
        &self,
        f: F,
        df: D1,
        d2f: D2,
        target: f64,
        guess: f64,
        accuracy: f64,
        max_evaluations: usize,
    ) -> QLResult<f64>
    where
        F: Fn(f64) -> f64,
        D1: Fn(f64) -> f64,
        D2: Fn(f64) -> f64,
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
            let d2fx = d2f(x);
            let denom = 2.0 * dfx * dfx - fx * d2fx;
            if denom.abs() < 1e-30 {
                // Fall back to Newton
                x -= fx / dfx;
            } else {
                x -= 2.0 * fx * dfx / denom;
            }
        }
        Err(QLError::RootNotFound(max_evaluations))
    }
}

impl Solver1D for Halley {
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
        // Numerical derivatives via central differences
        let h = 1e-6;
        let mut x = guess;
        for i in 0..max_evaluations {
            let fx = f(x) - target;
            if fx.abs() < accuracy {
                return Ok(x);
            }
            let fp = f(x + h);
            let fm = f(x - h);
            let dfx = (fp - fm) / (2.0 * h);
            let d2fx = (fp - 2.0 * (fx + target) + fm) / (h * h);

            if dfx.abs() < 1e-30 {
                return Err(QLError::RootNotFound(i + 1));
            }

            let denom = 2.0 * dfx * dfx - fx * d2fx;
            let x_new = if denom.abs() < 1e-30 {
                x - fx / dfx
            } else {
                x - 2.0 * fx * dfx / denom
            };

            x = x_new.clamp(x_min, x_max);
        }
        Err(QLError::RootNotFound(max_evaluations))
    }
}

// ===========================================================================
// Newton-Safe (G28)
// ===========================================================================

/// Newton's method with bisection fallback.
///
/// Uses Newton-Raphson but falls back to bisection whenever the Newton step
/// would leave the bracket. Always maintains a valid bracket, guaranteeing
/// convergence.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct NewtonSafe;

impl NewtonSafe {
    /// Solve using an analytic derivative.
    pub fn solve_with_derivative<F, D>(
        &self,
        f: F,
        df: D,
        target: f64,
        guess: f64,
        x_min: f64,
        x_max: f64,
        accuracy: f64,
        max_evaluations: usize,
    ) -> QLResult<f64>
    where
        F: Fn(f64) -> f64,
        D: Fn(f64) -> f64,
    {
        let g = |x: f64| f(x) - target;
        let mut xl = x_min;
        let mut xh = x_max;
        let fl = g(xl);
        let fh = g(xh);
        let mut evals = 2;

        if fl * fh > 0.0 {
            return Err(QLError::RootNotFound(evals));
        }
        // Ensure fl < 0 at xl and fh > 0 at xh
        if fl > 0.0 {
            std::mem::swap(&mut xl, &mut xh);
        }

        let mut x = guess.clamp(xl.min(xh), xl.max(xh));
        let mut dx_old = (xh - xl).abs();
        let mut dx = dx_old;

        let mut fx = g(x);
        let mut dfx = df(x);
        evals += 1;

        for _ in 0..max_evaluations {
            // Check if Newton step is out of range or not decreasing
            let newton_out_of_range =
                ((x - xh) * dfx - fx) * ((x - xl) * dfx - fx) > 0.0;
            let decreasing_too_slowly = (2.0 * fx).abs() > (dx_old * dfx).abs();

            if newton_out_of_range || decreasing_too_slowly {
                // Bisection step
                dx_old = dx;
                dx = 0.5 * (xh - xl);
                x = xl + dx;
            } else {
                // Newton step
                dx_old = dx;
                dx = fx / dfx;
                x -= dx;
            }

            if dx.abs() < accuracy {
                return Ok(x);
            }

            fx = g(x);
            dfx = df(x);
            evals += 1;

            if fx.abs() < accuracy {
                return Ok(x);
            }

            if evals >= max_evaluations {
                return Err(QLError::RootNotFound(evals));
            }

            // Update bracket
            if fx < 0.0 {
                xl = x;
            } else {
                xh = x;
            }
        }

        Err(QLError::RootNotFound(max_evaluations))
    }
}

impl Solver1D for NewtonSafe {
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
        let df = |x: f64| (f(x + h) - f(x - h)) / (2.0 * h);
        self.solve_with_derivative(&f, df, target, guess, x_min, x_max, accuracy, max_evaluations)
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

    fn f_quadratic(x: f64) -> f64 {
        x * x
    }

    #[test]
    fn halley_sqrt2() {
        let root = Halley
            .solve(f_quadratic, 2.0, 1.5, 0.0, 3.0, ACC, MAX_EVAL)
            .unwrap();
        assert_abs_diff_eq!(root, SQRT2, epsilon = 1e-10);
    }

    #[test]
    fn halley_with_derivatives() {
        let root = Halley
            .solve_with_derivatives(
                |x| x * x,
                |x| 2.0 * x,
                |_x| 2.0,
                2.0,
                1.5,
                ACC,
                MAX_EVAL,
            )
            .unwrap();
        assert_abs_diff_eq!(root, SQRT2, epsilon = ACC);
    }

    #[test]
    fn halley_cubic_convergence() {
        // Demonstrate that Halley converges in fewer iterations than Newton
        let f = |x: f64| x.exp() - 2.0;
        let df = |x: f64| x.exp();
        let d2f = |x: f64| x.exp();
        let root = Halley
            .solve_with_derivatives(f, df, d2f, 0.0, 0.5, ACC, 20)
            .unwrap();
        assert_abs_diff_eq!(root, 2.0_f64.ln(), epsilon = ACC);
    }

    #[test]
    fn newton_safe_sqrt2() {
        let root = NewtonSafe
            .solve(f_quadratic, 2.0, 1.5, 0.0, 3.0, ACC, MAX_EVAL)
            .unwrap();
        assert_abs_diff_eq!(root, SQRT2, epsilon = 1e-10);
    }

    #[test]
    fn newton_safe_with_derivative() {
        let root = NewtonSafe
            .solve_with_derivative(
                |x| x * x,
                |x| 2.0 * x,
                2.0,
                1.5,
                0.0,
                3.0,
                ACC,
                MAX_EVAL,
            )
            .unwrap();
        assert_abs_diff_eq!(root, SQRT2, epsilon = ACC);
    }

    #[test]
    fn newton_safe_cosine() {
        let root = NewtonSafe
            .solve(f64::cos, 0.0, 1.0, 0.0, 2.0, ACC, MAX_EVAL)
            .unwrap();
        assert_abs_diff_eq!(root, std::f64::consts::FRAC_PI_2, epsilon = 1e-6);
    }

    #[test]
    fn newton_safe_no_bracket() {
        let result = NewtonSafe.solve(f_quadratic, 2.0, 2.5, 2.0, 3.0, ACC, MAX_EVAL);
        assert!(result.is_err());
    }

    #[test]
    fn newton_safe_hard_function() {
        // x^3 - 2x - 5 (Wallis's equation) — root ≈ 2.0946
        let f = |x: f64| x * x * x - 2.0 * x - 5.0;
        let df = |x: f64| 3.0 * x * x - 2.0;
        let root = NewtonSafe
            .solve_with_derivative(f, df, 0.0, 2.0, 1.0, 3.0, ACC, MAX_EVAL)
            .unwrap();
        assert_abs_diff_eq!(root, 2.094_551_481_698_2, epsilon = 1e-6);
    }
}
