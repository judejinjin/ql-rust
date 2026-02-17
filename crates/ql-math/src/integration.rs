//! Numerical integration methods.
//!
//! Provides:
//! - `SimpsonIntegral` — composite Simpson's rule
//! - `GaussLobattoIntegral` — adaptive Gauss-Lobatto quadrature
//! - `GaussLegendreIntegral` — Gauss-Legendre quadrature (fixed order)

use ql_core::errors::{QLError, QLResult};

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Common interface for numerical integration methods.
pub trait Integrator {
    /// Integrate `f` over `[a, b]`.
    fn integrate<F: Fn(f64) -> f64>(&self, f: F, a: f64, b: f64) -> QLResult<f64>;
}

// ===========================================================================
// Simpson's Rule
// ===========================================================================

/// Composite Simpson's 1/3 rule.
#[derive(Clone, Debug)]
pub struct SimpsonIntegral {
    /// Number of intervals (must be even; will be rounded up).
    pub intervals: usize,
}

impl SimpsonIntegral {
    /// Create a Simpson integrator with the given number of intervals (rounded up to even).
    pub fn new(intervals: usize) -> Self {
        // Ensure even
        let intervals = if intervals.is_multiple_of(2) {
            intervals.max(2)
        } else {
            (intervals + 1).max(2)
        };
        Self { intervals }
    }
}

impl Integrator for SimpsonIntegral {
    fn integrate<F: Fn(f64) -> f64>(&self, f: F, a: f64, b: f64) -> QLResult<f64> {
        if a >= b {
            return Err(QLError::InvalidArgument(
                "integration bounds: a must be less than b".into(),
            ));
        }
        let n = self.intervals;
        let h = (b - a) / n as f64;

        let mut sum = f(a) + f(b);
        for i in 1..n {
            let x = a + i as f64 * h;
            if i % 2 == 0 {
                sum += 2.0 * f(x);
            } else {
                sum += 4.0 * f(x);
            }
        }

        Ok(sum * h / 3.0)
    }
}

// ===========================================================================
// Gauss-Lobatto (Adaptive)
// ===========================================================================

/// Adaptive Gauss-Lobatto quadrature.
///
/// Uses a 4-point Lobatto rule with adaptive subdivision for accuracy.
#[derive(Clone, Debug)]
pub struct GaussLobattoIntegral {
    pub max_evaluations: usize,
    pub absolute_accuracy: f64,
}

impl GaussLobattoIntegral {
    /// Create an adaptive Gauss-Lobatto integrator.
    pub fn new(max_evaluations: usize, absolute_accuracy: f64) -> Self {
        Self {
            max_evaluations,
            absolute_accuracy,
        }
    }
}

impl Integrator for GaussLobattoIntegral {
    fn integrate<F: Fn(f64) -> f64>(&self, f: F, a: f64, b: f64) -> QLResult<f64> {
        if a >= b {
            return Err(QLError::InvalidArgument(
                "integration bounds: a must be less than b".into(),
            ));
        }

        let mut evals = 0;
        let result = lobatto_adaptive(
            &f,
            a,
            b,
            self.absolute_accuracy,
            self.max_evaluations,
            &mut evals,
        )?;
        Ok(result)
    }
}

fn lobatto_adaptive<F: Fn(f64) -> f64>(
    f: &F,
    a: f64,
    b: f64,
    tol: f64,
    max_evals: usize,
    evals: &mut usize,
) -> QLResult<f64> {
    let mid = 0.5 * (a + b);
    let h = 0.5 * (b - a);

    // 4-point Gauss-Lobatto on [-1,1]: nodes {-1, -1/√5, 1/√5, 1}
    // weights {1/6, 5/6, 5/6, 1/6}
    let alpha = 1.0 / 5.0_f64.sqrt();
    let x1 = mid - alpha * h;
    let x2 = mid + alpha * h;

    let fa = f(a);
    let fb = f(b);
    let f1 = f(x1);
    let f2 = f(x2);
    *evals += 4;

    // 4-point Lobatto estimate: h * (w0*fa + w1*f1 + w2*f2 + w3*fb)
    let i4 = h * (fa / 6.0 + 5.0 * f1 / 6.0 + 5.0 * f2 / 6.0 + fb / 6.0);

    // Simpson 3-point estimate for comparison
    let fm = f(mid);
    *evals += 1;
    let i3 = h / 3.0 * (fa + 4.0 * fm + fb);

    if (*evals >= max_evals) || (i4 - i3).abs() < tol {
        return Ok(i4);
    }

    // Subdivide
    let left = lobatto_adaptive(f, a, mid, tol / 2.0, max_evals, evals)?;
    let right = lobatto_adaptive(f, mid, b, tol / 2.0, max_evals, evals)?;
    Ok(left + right)
}

// ===========================================================================
// Gauss-Legendre (Fixed Order)
// ===========================================================================

/// Gauss-Legendre quadrature with precomputed nodes and weights.
#[derive(Clone, Debug)]
pub struct GaussLegendreIntegral {
    /// Nodes on [-1, 1].
    nodes: Vec<f64>,
    /// Weights.
    weights: Vec<f64>,
}

impl GaussLegendreIntegral {
    /// Create a Gauss-Legendre integrator of the given order (number of points).
    ///
    /// Supports orders 1 through 10 with precomputed nodes.
    pub fn new(order: usize) -> QLResult<Self> {
        let (nodes, weights) = gauss_legendre_nodes_weights(order)?;
        Ok(Self { nodes, weights })
    }
}

impl Integrator for GaussLegendreIntegral {
    fn integrate<F: Fn(f64) -> f64>(&self, f: F, a: f64, b: f64) -> QLResult<f64> {
        if a >= b {
            return Err(QLError::InvalidArgument(
                "integration bounds: a must be less than b".into(),
            ));
        }
        // Transform from [-1,1] to [a,b]: x = (b-a)/2 * t + (a+b)/2
        let half_range = 0.5 * (b - a);
        let mid = 0.5 * (a + b);

        let sum: f64 = self
            .nodes
            .iter()
            .zip(self.weights.iter())
            .map(|(&t, &w)| w * f(mid + half_range * t))
            .sum();

        Ok(half_range * sum)
    }
}

/// Precomputed Gauss-Legendre nodes and weights for orders 1–10.
fn gauss_legendre_nodes_weights(order: usize) -> QLResult<(Vec<f64>, Vec<f64>)> {
    match order {
        1 => Ok((vec![0.0], vec![2.0])),
        2 => Ok((
            vec![-0.5773502691896257, 0.5773502691896257],
            vec![1.0, 1.0],
        )),
        3 => Ok((
            vec![-0.7745966692414834, 0.0, 0.7745966692414834],
            vec![
                0.5555555555555556,
                0.8888888888888888,
                0.5555555555555556,
            ],
        )),
        4 => Ok((
            vec![
                -0.8611363115940526,
                -0.3399810435848563,
                0.3399810435848563,
                0.8611363115940526,
            ],
            vec![
                0.3478548451374538,
                0.6521451548625461,
                0.6521451548625461,
                0.3478548451374538,
            ],
        )),
        5 => Ok((
            vec![
                -0.906_179_845_938_664,
                -0.5384693101056831,
                0.0,
                0.5384693101056831,
                0.906_179_845_938_664,
            ],
            vec![
                0.2369268850561891,
                0.4786286704993665,
                0.5688888888888889,
                0.4786286704993665,
                0.2369268850561891,
            ],
        )),
        6 => Ok((
            vec![
                -0.932_469_514_203_152,
                -0.6612093864662645,
                -0.2386191860831969,
                0.2386191860831969,
                0.6612093864662645,
                0.932_469_514_203_152,
            ],
            vec![
                0.1713244923791704,
                0.3607615730481386,
                0.467_913_934_572_691,
                0.467_913_934_572_691,
                0.3607615730481386,
                0.1713244923791704,
            ],
        )),
        7 => Ok((
            vec![
                -0.9491079123427585,
                -0.7415311855993945,
                -0.4058451513773972,
                0.0,
                0.4058451513773972,
                0.7415311855993945,
                0.9491079123427585,
            ],
            vec![
                0.1294849661688697,
                0.2797053914892767,
                0.3818300505051189,
                0.4179591836734694,
                0.3818300505051189,
                0.2797053914892767,
                0.1294849661688697,
            ],
        )),
        8 => Ok((
            vec![
                -0.9602898564975363,
                -0.7966664774136267,
                -0.525_532_409_916_329,
                -0.1834346424956498,
                0.1834346424956498,
                0.525_532_409_916_329,
                0.7966664774136267,
                0.9602898564975363,
            ],
            vec![
                0.1012285362903763,
                0.2223810344533745,
                0.3137066458778873,
                0.362_683_783_378_362,
                0.362_683_783_378_362,
                0.3137066458778873,
                0.2223810344533745,
                0.1012285362903763,
            ],
        )),
        _ => {
            // For higher orders, compute using Newton's method on Legendre polynomials
            compute_gauss_legendre(order)
        }
    }
}

/// Compute Gauss-Legendre nodes and weights for arbitrary order using Newton's method.
fn compute_gauss_legendre(n: usize) -> QLResult<(Vec<f64>, Vec<f64>)> {
    if n == 0 {
        return Err(QLError::InvalidArgument(
            "Gauss-Legendre order must be >= 1".into(),
        ));
    }

    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];

    let m = n.div_ceil(2);
    for i in 0..m {
        // Initial guess
        let mut x = ((i as f64 + 0.75) / (n as f64 + 0.5) * std::f64::consts::PI).cos();

        for _ in 0..100 {
            let (p, dp) = legendre_pd(n, x);
            let dx = p / dp;
            x -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }

        let (_, dp) = legendre_pd(n, x);
        nodes[i] = -x;
        nodes[n - 1 - i] = x;
        let w = 2.0 / ((1.0 - x * x) * dp * dp);
        weights[i] = w;
        weights[n - 1 - i] = w;
    }

    Ok((nodes, weights))
}

/// Evaluate Legendre polynomial P_n(x) and its derivative P'_n(x).
fn legendre_pd(n: usize, x: f64) -> (f64, f64) {
    let mut p0 = 1.0;
    let mut p1 = x;
    for k in 2..=n {
        let pk = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
        p0 = p1;
        p1 = pk;
    }
    let dp = n as f64 * (p0 - x * p1) / (1.0 - x * x);
    (p1, dp)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn simpson_sin() {
        // ∫ sin(x) from 0 to π = 2
        let s = SimpsonIntegral::new(1000);
        let result = s.integrate(f64::sin, 0.0, std::f64::consts::PI).unwrap();
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn simpson_polynomial() {
        // ∫ x^2 from 0 to 1 = 1/3
        let s = SimpsonIntegral::new(100);
        let result = s.integrate(|x| x * x, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(result, 1.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn simpson_exact_for_cubic() {
        // Simpson's rule is exact for polynomials up to degree 3
        let s = SimpsonIntegral::new(2);
        let result = s.integrate(|x| x * x * x, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(result, 0.25, epsilon = 1e-14);
    }

    #[test]
    fn gauss_lobatto_sin() {
        let gl = GaussLobattoIntegral::new(10000, 1e-12);
        let result = gl.integrate(f64::sin, 0.0, std::f64::consts::PI).unwrap();
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn gauss_lobatto_exp() {
        // ∫ e^x from 0 to 1 = e - 1
        let gl = GaussLobattoIntegral::new(10000, 1e-12);
        let result = gl.integrate(f64::exp, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(result, std::f64::consts::E - 1.0, epsilon = 1e-10);
    }

    #[test]
    fn gauss_legendre_sin() {
        let gl = GaussLegendreIntegral::new(8).unwrap();
        let result = gl.integrate(f64::sin, 0.0, std::f64::consts::PI).unwrap();
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn gauss_legendre_polynomial() {
        // GL with n points is exact for polynomials up to degree 2n-1
        // n=5 is exact for degree 9
        let gl = GaussLegendreIntegral::new(5).unwrap();
        // ∫ x^9 from 0 to 1 = 0.1
        let result = gl.integrate(|x| x.powi(9), 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(result, 0.1, epsilon = 1e-14);
    }

    #[test]
    fn gauss_legendre_high_order() {
        // Test order 20 (computed via Newton's method)
        let gl = GaussLegendreIntegral::new(20).unwrap();
        let result = gl.integrate(f64::sin, 0.0, std::f64::consts::PI).unwrap();
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-14);
    }

    #[test]
    fn gauss_legendre_order_1() {
        // 1-point rule: midpoint
        let gl = GaussLegendreIntegral::new(1).unwrap();
        // ∫ 1 from 0 to 2 = 2
        let result = gl.integrate(|_| 1.0, 0.0, 2.0).unwrap();
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-14);
    }

    #[test]
    fn invalid_bounds() {
        let s = SimpsonIntegral::new(100);
        assert!(s.integrate(|x| x, 1.0, 0.0).is_err());
    }
}
