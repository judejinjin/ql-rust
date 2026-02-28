//! Richardson extrapolation.
//!
//! Accelerates the convergence of a sequence of approximations A(h)
//! that satisfies A(h) = A* + c₁ h^p + c₂ h^{2p} + ...
//!
//! Given A(h) and A(h/t), the extrapolated value is:
//!   A* ≈ (t^p A(h/t) − A(h)) / (t^p − 1)
//!
//! Corresponds to QuantLib's `RichardsonExtrapolation`.

use serde::{Deserialize, Serialize};

/// Richardson extrapolation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RichardsonResult {
    /// Extrapolated value.
    pub value: f64,
    /// Estimated error (difference between extrapolated and best input).
    pub error_estimate: f64,
    /// Order of extrapolation used.
    pub order: usize,
}

/// Perform single-step Richardson extrapolation.
///
/// Given two approximations:
///   - `a_h` = A(h)   (coarser approximation)
///   - `a_ht` = A(h/t) (finer approximation)
///   - `t` = step size ratio (typically 2.0)
///   - `p` = order of the leading error term
///
/// Returns (t^p × a_ht − a_h) / (t^p − 1).
pub fn richardson_extrapolate(a_h: f64, a_ht: f64, t: f64, p: f64) -> RichardsonResult {
    let tp = t.powf(p);
    let value = (tp * a_ht - a_h) / (tp - 1.0);
    let error_estimate = (value - a_ht).abs();
    RichardsonResult { value, error_estimate, order: 1 }
}

/// Perform iterated Richardson extrapolation on a sequence of approximations.
///
/// `values[i]` = A(h / t^i) for i = 0, 1, ..., n−1.
/// Each level eliminates one more error term.
///
/// This is equivalent to Romberg integration when applied to the trapezoidal rule.
pub fn richardson_table(values: &[f64], t: f64, p: f64) -> RichardsonResult {
    let n = values.len();
    if n == 0 { return RichardsonResult { value: 0.0, error_estimate: f64::INFINITY, order: 0 }; }
    if n == 1 { return RichardsonResult { value: values[0], error_estimate: f64::INFINITY, order: 0 }; }

    // Neville-like tableau
    let mut table = values.to_vec();
    let mut order = 0;

    for level in 1..n {
        let pk = p + (level - 1) as f64 * 1.0; // increasing order
        let tp = t.powf(pk);
        let new_len = table.len() - 1;
        let mut new_table = Vec::with_capacity(new_len);
        for i in 0..new_len {
            let extrapolated = (tp * table[i + 1] - table[i]) / (tp - 1.0);
            new_table.push(extrapolated);
        }
        table = new_table;
        order = level;
    }

    let value = table[0];
    let error_estimate = if n >= 2 { (value - values[n - 1]).abs() } else { f64::INFINITY };
    RichardsonResult { value, error_estimate, order }
}

/// Apply Richardson extrapolation to a function f(h) that approximates some limit.
///
/// Evaluates f at geometrically decreasing step sizes h, h/t, h/t², ...
/// and applies iterated extrapolation.
///
/// - `f` — function that takes a step size and returns an approximation
/// - `h` — initial step size
/// - `t` — step size ratio (typically 2.0)
/// - `p` — order of leading error term
/// - `levels` — number of refinement levels (2 or more)
pub fn richardson_extrapolate_fn<F: Fn(f64) -> f64>(
    f: F,
    h: f64,
    t: f64,
    p: f64,
    levels: usize,
) -> RichardsonResult {
    let values: Vec<f64> = (0..levels)
        .map(|i| f(h / t.powi(i as i32)))
        .collect();
    richardson_table(&values, t, p)
}

/// Romberg integration: Richardson extrapolation applied to the trapezoidal rule.
///
/// Integrates f over [a, b] using the trapezoidal rule with increasing subdivisions.
pub fn romberg_integrate<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, levels: usize) -> RichardsonResult {
    let trapezoidal = |n: usize| -> f64 {
        let h = (b - a) / n as f64;
        let mut sum = 0.5 * (f(a) + f(b));
        for i in 1..n {
            sum += f(a + i as f64 * h);
        }
        sum * h
    };

    let values: Vec<f64> = (0..levels)
        .map(|i| trapezoidal(1 << (i + 1))) // 2, 4, 8, 16, ...
        .collect();

    richardson_table(&values, 2.0, 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_richardson_single_step() {
        // Approximate π using inscribed polygon perimeters
        // P(n) ≈ π − c/n² for regular n-gon
        let p6 = 6.0 * (std::f64::consts::PI / 6.0).sin(); // hexagon
        let p12 = 12.0 * (std::f64::consts::PI / 12.0).sin(); // 12-gon

        let res = richardson_extrapolate(p6, p12, 2.0, 2.0);
        assert!((res.value - std::f64::consts::PI).abs() < 0.01,
            "extrapolated={}", res.value);
    }

    #[test]
    fn test_richardson_table() {
        // Four approximations with h, h/2, h/4, h/8
        let exact = std::f64::consts::PI;
        let values: Vec<f64> = (0..4)
            .map(|i| {
                let n = 6 * (1 << i); // 6, 12, 24, 48
                n as f64 * (std::f64::consts::PI / n as f64).sin()
            })
            .collect();

        let res = richardson_table(&values, 2.0, 2.0);
        assert!((res.value - exact).abs() < 1e-4, "value={}", res.value);
    }

    #[test]
    fn test_romberg_integration() {
        // Integrate x² from 0 to 1 = 1/3
        let res = romberg_integrate(|x| x * x, 0.0, 1.0, 4);
        assert!((res.value - 1.0 / 3.0).abs() < 1e-10, "value={}", res.value);
    }

    #[test]
    fn test_romberg_sin() {
        // Integrate sin(x) from 0 to π = 2
        let res = romberg_integrate(|x| x.sin(), 0.0, std::f64::consts::PI, 5);
        assert!((res.value - 2.0).abs() < 1e-6, "value={}", res.value);
    }

    #[test]
    fn test_richardson_fn() {
        // Derivative of e^x at x=1 using forward difference
        let e = std::f64::consts::E;
        let res = richardson_extrapolate_fn(
            |h| ((1.0 + h).exp() - e) / h,
            1.0, 2.0, 1.0, 5,
        );
        assert!((res.value - e).abs() < 0.1, "value={}", res.value);
    }
}
