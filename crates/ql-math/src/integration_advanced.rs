//! Advanced numerical integration: Gauss-Kronrod and double exponential (tanh-sinh).
//!
//! ## Gauss-Kronrod G7K15
//!
//! The **G7K15** rule computes a 15-point Gauss-Kronrod integral alongside a
//! 7-point Gaussian estimate.  The difference gives an error estimate.  An
//! adaptive version subdivides the interval when the error exceeds the tolerance.
//!
//! References:
//! - Kronrod (1965). *Nodes and weights of quadrature formulas*. Consultants Bureau.
//! - Piessens et al. (1983). *QUADPACK: A Subroutine Package*. Springer.
//!
//! ## Tanh-Sinh (Double Exponential) Quadrature
//!
//! Transforms the integration variable via `t = tanh(π/2 · sinh(x))` so that
//! the integrand collapses doubly-exponentially near the endpoints.  Excels for
//! integrands with integrable singularities at the endpoints.
//!
//! Reference: Takahasi & Mori (1974). *Double exponential formulas for numerical integration.*

// =========================================================================
// G7K15 Nodes and weights  (from Piessens et al., standardised to [-1,1])
// =========================================================================

/// 15 Gauss-Kronrod nodes in [0, 1] (symmetrical about 0).
/// These are the positive nodes; zero is also a node.
const GK15_NODES: [f64; 8] = [
    0.991_455_371_120_813,
    0.949_107_912_342_758,
    0.864_864_423_359_769,
    0.741_531_185_599_394,
    0.586_087_235_467_691,
    0.405_845_151_377_397,
    0.207_784_955_007_898,
    0.0,
];

/// Weights for the 15-point Gauss-Kronrod rule (indexed as nodes above).
const GK15_WEIGHTS: [f64; 8] = [
    0.022_935_322_010_529,
    0.063_092_092_629_979,
    0.104_790_010_322_250,
    0.140_653_259_715_525,
    0.169_004_726_639_267,
    0.190_350_578_064_785,
    0.204_432_940_075_298,
    0.209_482_141_084_728,
];

/// Weights for the embedded 7-point Gauss rule (at odd-indexed nodes above + 0).
const G7_WEIGHTS: [f64; 4] = [
    // indices into GK15_NODES: 1, 3, 5, 7(=0)
    0.129_484_966_168_870, // node[1] = 0.9491...
    0.279_705_391_489_277, // node[3] = 0.7415...
    0.381_830_050_505_119, // node[5] = 0.4058...
    0.417_959_183_673_469, // node[7] = 0.0
];

// =========================================================================
// Single-step G7K15  
// =========================================================================

/// Result from a single G7K15 quadrature step.
#[derive(Debug, Clone, Copy)]
pub struct GkStepResult {
    /// 15-point Gauss-Kronrod estimate.
    pub integral: f64,
    /// Estimated absolute error (`|K15 − G7|`).
    pub error: f64,
    /// Number of function evaluations (always 15 for one step).
    pub n_evals: usize,
}

/// Evaluate the G7K15 rule over `[a, b]`.
pub fn gk15_step<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64) -> GkStepResult {
    let half = (b - a) / 2.0;
    let center = (a + b) / 2.0;

    let mut gk = 0.0;
    let mut g7 = 0.0;

    // Zero node
    let f_center = f(center);
    gk += GK15_WEIGHTS[7] * f_center;
    g7 += G7_WEIGHTS[3] * f_center;

    // Symmetric node pairs
    for k in 0..7 {
        let x = GK15_NODES[k];
        let fp = f(center + half * x);
        let fm = f(center - half * x);
        gk += GK15_WEIGHTS[k] * (fp + fm);
        // G7 uses only the odd-indexed nodes: k=1,3,5
        if k == 1 { g7 += G7_WEIGHTS[0] * (fp + fm); }
        if k == 3 { g7 += G7_WEIGHTS[1] * (fp + fm); }
        if k == 5 { g7 += G7_WEIGHTS[2] * (fp + fm); }
    }

    gk *= half;
    g7 *= half;

    GkStepResult {
        integral: gk,
        error: (gk - g7).abs(),
        n_evals: 15,
    }
}

// =========================================================================
// Adaptive Gauss-Kronrod
// =========================================================================

/// Result from adaptive Gauss-Kronrod integration.
#[derive(Debug, Clone)]
pub struct AdaptiveGkResult {
    /// Integral estimate.
    pub integral: f64,
    /// Estimated absolute error.
    pub error: f64,
    /// Total function evaluations.
    pub n_evals: usize,
    /// Whether the tolerance was achieved.
    pub converged: bool,
}

/// Adaptive Gauss-Kronrod integration using the G7K15 rule.
///
/// Subdivides the interval recursively until the error estimate falls below
/// `abs_tol + rel_tol * |integral|` or `max_depth` levels are reached.
///
/// # Arguments
/// - `f`       — integrand
/// - `a`, `b`  — integration limits (may be ±∞ not handled here; use substitution)
/// - `abs_tol` — absolute error tolerance
/// - `rel_tol` — relative error tolerance
/// - `max_evals` — maximum number of function evaluations
pub fn gauss_kronrod_adaptive<F>(
    f: F,
    a: f64,
    b: f64,
    abs_tol: f64,
    rel_tol: f64,
    max_evals: usize,
) -> AdaptiveGkResult
where
    F: Fn(f64) -> f64,
{
    // Start with one G7K15 step on the whole interval
    let initial = gk15_step(&f, a, b);
    let tol = abs_tol + rel_tol * initial.integral.abs();

    if initial.error <= tol || max_evals < 30 {
        return AdaptiveGkResult {
            integral: initial.integral,
            error: initial.error,
            n_evals: initial.n_evals,
            converged: initial.error <= tol,
        };
    }

    // Priority queue: (error, a, b, integral)
    let mut panels: Vec<(f64, f64, f64, f64)> =
        vec![(initial.error, a, b, initial.integral)];

    let mut total_integral = initial.integral;
    let mut total_error = initial.error;
    let mut total_evals = initial.n_evals;

    while total_error > abs_tol + rel_tol * total_integral.abs() {
        if total_evals + 30 > max_evals {
            break;
        }
        // Split the panel with the largest error
        panels.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
        let (worst_err, lo, hi, worst_int) = panels.pop().unwrap();

        let mid = (lo + hi) / 2.0;
        let left = gk15_step(&f, lo, mid);
        let right = gk15_step(&f, mid, hi);
        total_evals += left.n_evals + right.n_evals;

        // Update totals
        let improvement = worst_int - (left.integral + right.integral);
        let _ = improvement; // Absorbed into the sub-panel estimates
        total_integral = total_integral - worst_int + left.integral + right.integral;
        total_error = total_error - worst_err + left.error + right.error;

        panels.push((left.error, lo, mid, left.integral));
        panels.push((right.error, mid, hi, right.integral));
    }

    AdaptiveGkResult {
        integral: total_integral,
        error: total_error,
        n_evals: total_evals,
        converged: total_error <= abs_tol + rel_tol * total_integral.abs(),
    }
}

// =========================================================================
// Gauss-Kronrod on semi-infinite interval [a, ∞)
// =========================================================================

/// Integrate `f` over `[a, ∞)` using the substitution `t = a + s/(1-s)`.
///
/// Maps `[a, ∞)` → `[0, 1)` via `x = a + s/(1-s)`, `dx = 1/(1-s)² ds`.
pub fn gauss_kronrod_semi_infinite<F>(
    f: F,
    a: f64,
    abs_tol: f64,
    rel_tol: f64,
    max_evals: usize,
) -> AdaptiveGkResult
where
    F: Fn(f64) -> f64 + Clone,
{
    let g = move |s: f64| -> f64 {
        if s >= 1.0 - 1e-12 { return 0.0; }
        let x = a + s / (1.0 - s);
        let jac = 1.0 / (1.0 - s).powi(2);
        f(x) * jac
    };
    gauss_kronrod_adaptive(g, 0.0, 1.0 - 1e-8, abs_tol, rel_tol, max_evals)
}

// =========================================================================
// Tanh-Sinh (Double Exponential) Quadrature
// =========================================================================

/// Result from tanh-sinh integration.
#[derive(Debug, Clone)]
pub struct TanhSinhResult {
    /// Integral estimate.
    pub integral: f64,
    /// Estimated absolute error (difference between last two levels).
    pub error: f64,
    /// Total function evaluations.
    pub n_evals: usize,
    /// Converged to the requested tolerance.
    pub converged: bool,
}

/// Tanh-sinh (double exponential) quadrature over `[a, b]`.
///
/// Particularly effective for integrands with endpoint singularities.
///
/// # Arguments
/// - `f`     — integrand (may have integrable singularities at `a`, `b`)
/// - `a`, `b` — finite integration limits
/// - `tol`   — absolute tolerance
/// - `max_levels` — maximum refinement levels (each level doubles the nodes)
pub fn tanh_sinh<F>(
    f: F,
    a: f64,
    b: f64,
    tol: f64,
    max_levels: usize,
) -> TanhSinhResult
where
    F: Fn(f64) -> f64,
{
    // Map [a,b] to [-1,1], then apply tanh-sinh on [-1,1]
    let half = (b - a) / 2.0;
    let center = (a + b) / 2.0;
    let g = |t: f64| f(center + half * t);

    let h0 = 1.0;
    let mut h = h0;
    let mut prev_integral = 0.0;
    let mut integral = 0.0;
    let mut n_evals = 0usize;
    let mut converged = false;

    for level in 0..max_levels {
        h = h0 / 2.0_f64.powi(level as i32);
        let mut sum = 0.0;

        // Range: k = 0, ±1, ±2, … until |t| ≥ 1
        let n_max = (6.0 / h).ceil() as i64;
        for k in -n_max..=n_max {
            let x = k as f64 * h;
            let sinh_x = x.sinh();
            let cosh_x = x.cosh();
            let t = (std::f64::consts::FRAC_PI_2 * sinh_x).tanh();
            if t.abs() >= 1.0 - 1e-12 { continue; }
            let dtdx = std::f64::consts::FRAC_PI_2 * cosh_x
                / (std::f64::consts::FRAC_PI_2 * sinh_x).cosh().powi(2);
            sum += g(t) * dtdx;
            n_evals += 1;
        }
        integral = h * sum * half; // Map back from [-1,1] to [a,b]

        if level > 0 && (integral - prev_integral).abs() < tol {
            converged = true;
            break;
        }
        prev_integral = integral;
    }

    TanhSinhResult {
        integral,
        error: (integral - prev_integral).abs(),
        n_evals,
        converged,
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gk15_integrates_polynomial_exactly() {
        // ∫₀¹ (1 + x + x²) dx = 1 + 0.5 + 1/3 = 11/6
        let result = gk15_step(&|x| 1.0 + x + x * x, 0.0, 1.0);
        let exact = 11.0 / 6.0;
        assert!((result.integral - exact).abs() < 1e-12, "got {}", result.integral);
        assert!(result.error < 1e-12);
    }

    #[test]
    fn adaptive_gk_gaussian_integral() {
        // ∫₋₅⁵ exp(-x²) dx ≈ √π
        let res = gauss_kronrod_adaptive(
            |x: f64| (-x * x).exp(),
            -5.0, 5.0, 1e-10, 1e-10, 10_000,
        );
        let exact = std::f64::consts::PI.sqrt();
        assert!((res.integral - exact).abs() < 1e-8, "got {}", res.integral);
        assert!(res.converged);
    }

    #[test]
    fn adaptive_gk_trigonometric() {
        // ∫₀^π sin(x) dx = 2
        let res = gauss_kronrod_adaptive(
            |x: f64| x.sin(),
            0.0, std::f64::consts::PI, 1e-10, 1e-10, 10_000,
        );
        assert!((res.integral - 2.0).abs() < 1e-9, "got {}", res.integral);
    }

    #[test]
    fn semi_infinite_exponential_integral() {
        // ∫₀^∞ exp(-x) dx = 1
        let res = gauss_kronrod_semi_infinite(
            |x: f64| (-x).exp(),
            0.0, 1e-8, 1e-8, 50_000,
        );
        assert!((res.integral - 1.0).abs() < 1e-5, "got {}", res.integral);
    }

    #[test]
    fn tanh_sinh_smooth_integrand() {
        // ∫₀¹ x * (1-x) dx = 1/6
        let res = tanh_sinh(
            |x: f64| x * (1.0 - x),
            0.0, 1.0, 1e-10, 6,
        );
        let exact = 1.0 / 6.0;
        assert!((res.integral - exact).abs() < 1e-8, "got {}", res.integral);
    }
}
