//! Optimization framework.
//!
//! Provides `EndCriteria`, `CostFunction` trait, and concrete optimizers:
//! - `Simplex` (Nelder-Mead) — derivative-free
//! - `LevenbergMarquardt` — least-squares

use ql_core::errors::{QLError, QLResult};

// ---------------------------------------------------------------------------
// End Criteria
// ---------------------------------------------------------------------------

/// Convergence criteria for optimizers.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EndCriteria {
    pub max_iterations: usize,
    pub max_stationary_iterations: usize,
    pub root_epsilon: f64,
    pub function_epsilon: f64,
    pub gradient_epsilon: f64,
}

impl Default for EndCriteria {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            max_stationary_iterations: 100,
            root_epsilon: 1e-8,
            function_epsilon: 1e-8,
            gradient_epsilon: 1e-8,
        }
    }
}

/// Result of an optimization.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct OptimizationResult {
    /// The optimal parameter vector.
    pub parameters: Vec<f64>,
    /// The cost function value at the optimum.
    pub value: f64,
    /// Number of iterations performed.
    pub iterations: usize,
}

// ---------------------------------------------------------------------------
// Cost Function Trait
// ---------------------------------------------------------------------------

/// A function to be minimized.
pub trait CostFunction {
    /// Evaluate the cost at the given parameter vector.
    fn value(&self, params: &[f64]) -> f64;

    /// Dimensionality of the parameter space.
    fn dimension(&self) -> usize;
}

/// A least-squares cost function (for Levenberg-Marquardt).
pub trait LeastSquaresCostFunction: CostFunction {
    /// Evaluate the residuals at the given parameter vector.
    fn residuals(&self, params: &[f64]) -> Vec<f64>;
}

// ===========================================================================
// Simplex (Nelder-Mead)
// ===========================================================================

/// Nelder-Mead simplex optimizer (derivative-free).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Simplex {
    /// Initial simplex size.
    pub lambda: f64,
}

impl Simplex {
    /// Create a Nelder-Mead optimizer with the given initial simplex size.
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }

    /// Minimize a cost function starting from the given initial point.
    pub fn minimize<C: CostFunction>(
        &self,
        cost: &C,
        initial: &[f64],
        criteria: &EndCriteria,
    ) -> QLResult<OptimizationResult> {
        let n = cost.dimension();
        if initial.len() != n {
            return Err(QLError::InvalidArgument(format!(
                "initial vector length {} != dimension {}",
                initial.len(),
                n
            )));
        }

        // Build initial simplex: n+1 vertices
        let mut vertices: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
        vertices.push(initial.to_vec());
        for i in 0..n {
            let mut v = initial.to_vec();
            v[i] += self.lambda;
            vertices.push(v);
        }

        let mut values: Vec<f64> = vertices.iter().map(|v| cost.value(v)).collect();

        let alpha = 1.0; // reflection
        let gamma = 2.0; // expansion
        let rho = 0.5; // contraction
        let sigma = 0.5; // shrink

        let mut stationary_count = 0;
        let mut prev_best = f64::INFINITY;

        for iter in 0..criteria.max_iterations {
            // Sort vertices by value
            let mut order: Vec<usize> = (0..=n).collect();
            order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));

            let best_idx = order[0];
            let worst_idx = order[n];
            let second_worst_idx = order[n - 1];

            let best_val = values[best_idx];
            let worst_val = values[worst_idx];
            let second_worst_val = values[second_worst_idx];

            // Check convergence
            let range = (worst_val - best_val).abs();
            if range < criteria.function_epsilon {
                return Ok(OptimizationResult {
                    parameters: vertices[best_idx].clone(),
                    value: best_val,
                    iterations: iter,
                });
            }

            if (prev_best - best_val).abs() < criteria.function_epsilon {
                stationary_count += 1;
                if stationary_count >= criteria.max_stationary_iterations {
                    return Ok(OptimizationResult {
                        parameters: vertices[best_idx].clone(),
                        value: best_val,
                        iterations: iter,
                    });
                }
            } else {
                stationary_count = 0;
            }
            prev_best = best_val;

            // Centroid (exclude worst)
            let mut centroid = vec![0.0; n];
            for &i in &order[..n] {
                for j in 0..n {
                    centroid[j] += vertices[i][j];
                }
            }
            for c in &mut centroid {
                *c /= n as f64;
            }

            // Reflection
            let reflected: Vec<f64> = centroid
                .iter()
                .zip(vertices[worst_idx].iter())
                .map(|(&c, &w)| c + alpha * (c - w))
                .collect();
            let reflected_val = cost.value(&reflected);

            if reflected_val < best_val {
                // Try expansion
                let expanded: Vec<f64> = centroid
                    .iter()
                    .zip(reflected.iter())
                    .map(|(&c, &r)| c + gamma * (r - c))
                    .collect();
                let expanded_val = cost.value(&expanded);

                if expanded_val < reflected_val {
                    vertices[worst_idx] = expanded;
                    values[worst_idx] = expanded_val;
                } else {
                    vertices[worst_idx] = reflected;
                    values[worst_idx] = reflected_val;
                }
            } else if reflected_val < second_worst_val {
                vertices[worst_idx] = reflected;
                values[worst_idx] = reflected_val;
            } else {
                // Contraction
                let base = if reflected_val < worst_val {
                    &reflected
                } else {
                    &vertices[worst_idx]
                };
                let contracted: Vec<f64> = centroid
                    .iter()
                    .zip(base.iter())
                    .map(|(&c, &b)| c + rho * (b - c))
                    .collect();
                let contracted_val = cost.value(&contracted);

                if contracted_val < worst_val.min(reflected_val) {
                    vertices[worst_idx] = contracted;
                    values[worst_idx] = contracted_val;
                } else {
                    // Shrink
                    let best = vertices[best_idx].clone();
                    for &i in &order[1..] {
                        for j in 0..n {
                            vertices[i][j] = best[j] + sigma * (vertices[i][j] - best[j]);
                        }
                        values[i] = cost.value(&vertices[i]);
                    }
                }
            }
        }

        // Find best
        let best_idx = values
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0);

        Ok(OptimizationResult {
            parameters: vertices[best_idx].clone(),
            value: values[best_idx],
            iterations: criteria.max_iterations,
        })
    }
}

// ===========================================================================
// Levenberg-Marquardt
// ===========================================================================

/// Levenberg-Marquardt optimizer for least-squares problems.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LevenbergMarquardt {
    /// Initial damping parameter.
    pub initial_lambda: f64,
}

impl Default for LevenbergMarquardt {
    fn default() -> Self {
        Self {
            initial_lambda: 1e-3,
        }
    }
}

impl LevenbergMarquardt {
    /// Create a Levenberg-Marquardt optimizer with the given initial damping.
    pub fn new(initial_lambda: f64) -> Self {
        Self { initial_lambda }
    }

    /// Minimize a least-squares cost function starting from the given initial point.
    pub fn minimize<C: LeastSquaresCostFunction>(
        &self,
        cost: &C,
        initial: &[f64],
        criteria: &EndCriteria,
    ) -> QLResult<OptimizationResult> {
        let n = cost.dimension();
        let h = 1e-8; // finite difference step
        let mut params = initial.to_vec();
        let mut lambda = self.initial_lambda;
        let mut residuals = cost.residuals(&params);
        let mut current_cost: f64 = residuals.iter().map(|r| r * r).sum();

        for iter in 0..criteria.max_iterations {
            if current_cost.sqrt() < criteria.root_epsilon {
                return Ok(OptimizationResult {
                    parameters: params,
                    value: current_cost,
                    iterations: iter,
                });
            }

            let m = residuals.len();

            // Build Jacobian via finite differences
            let mut jacobian = vec![vec![0.0; n]; m];
            for j in 0..n {
                let mut params_h = params.clone();
                params_h[j] += h;
                let res_h = cost.residuals(&params_h);
                for i in 0..m {
                    jacobian[i][j] = (res_h[i] - residuals[i]) / h;
                }
            }

            // J^T * J
            let mut jtj = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in 0..n {
                    let s: f64 = jacobian.iter().take(m).map(|jk| jk[i] * jk[j]).sum();
                    jtj[i][j] = s;
                }
            }

            // J^T * r
            let mut jtr = vec![0.0; n];
            for i in 0..n {
                let mut s = 0.0;
                for k in 0..m {
                    s += jacobian[k][i] * residuals[k];
                }
                jtr[i] = s;
            }

            // Solve (J^T*J + lambda*I) * delta = -J^T*r using simple Gaussian elimination
            let mut augmented = jtj.clone();
            for (i, row) in augmented.iter_mut().enumerate().take(n) {
                row[i] += lambda;
            }

            let delta = solve_linear_system(&augmented, &jtr.iter().map(|v| -v).collect::<Vec<_>>());

            if let Some(delta) = delta {
                let new_params: Vec<f64> =
                    params.iter().zip(delta.iter()).map(|(p, d)| p + d).collect();
                let new_residuals = cost.residuals(&new_params);
                let new_cost: f64 = new_residuals.iter().map(|r| r * r).sum();

                if new_cost < current_cost {
                    params = new_params;
                    residuals = new_residuals;
                    let improvement = current_cost - new_cost;
                    current_cost = new_cost;
                    lambda *= 0.1;

                    if improvement < criteria.function_epsilon {
                        return Ok(OptimizationResult {
                            parameters: params,
                            value: current_cost,
                            iterations: iter,
                        });
                    }
                } else {
                    lambda *= 10.0;
                }
            } else {
                lambda *= 10.0;
            }
        }

        Ok(OptimizationResult {
            parameters: params,
            value: current_cost,
            iterations: criteria.max_iterations,
        })
    }
}

/// Simple Gaussian elimination for small systems.
#[allow(clippy::needless_range_loop)]
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.push(b[i]);
            r
        })
        .collect();

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for (row, aug_row) in aug.iter().enumerate().take(n).skip(col + 1) {
            if aug_row[col].abs() > max_val {
                max_val = aug_row[col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-30 {
            return None; // Singular
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        for row in col + 1..n {
            let factor = aug[row][col] / pivot;
            for j in col..=n {
                let val = aug[col][j];
                aug[row][j] -= factor * val;
            }
        }
    }

    // Back-substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in i + 1..n {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }

    Some(x)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
    // Minimum at (1, 1) with value 0
    struct Rosenbrock;

    impl CostFunction for Rosenbrock {
        fn value(&self, params: &[f64]) -> f64 {
            let x = params[0];
            let y = params[1];
            (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2)
        }

        fn dimension(&self) -> usize {
            2
        }
    }

    // Sphere function: f(x) = sum(x_i^2), minimum at origin
    struct Sphere(usize);

    impl CostFunction for Sphere {
        fn value(&self, params: &[f64]) -> f64 {
            params.iter().map(|x| x * x).sum()
        }

        fn dimension(&self) -> usize {
            self.0
        }
    }

    impl LeastSquaresCostFunction for Sphere {
        fn residuals(&self, params: &[f64]) -> Vec<f64> {
            params.to_vec()
        }
    }

    // Least-squares: fit a*x + b to data points
    struct LinearFit {
        xs: Vec<f64>,
        ys: Vec<f64>,
    }

    impl CostFunction for LinearFit {
        fn value(&self, params: &[f64]) -> f64 {
            self.residuals(params).iter().map(|r| r * r).sum()
        }

        fn dimension(&self) -> usize {
            2
        }
    }

    impl LeastSquaresCostFunction for LinearFit {
        fn residuals(&self, params: &[f64]) -> Vec<f64> {
            let a = params[0];
            let b = params[1];
            self.xs
                .iter()
                .zip(self.ys.iter())
                .map(|(&x, &y)| a * x + b - y)
                .collect()
        }
    }

    #[test]
    fn simplex_sphere() {
        let cost = Sphere(3);
        let simplex = Simplex::new(1.0);
        let criteria = EndCriteria {
            max_iterations: 5000,
            function_epsilon: 1e-12,
            ..EndCriteria::default()
        };
        let result = simplex
            .minimize(&cost, &[3.0, -2.0, 1.0], &criteria)
            .unwrap();
        for &p in &result.parameters {
            assert_abs_diff_eq!(p, 0.0, epsilon = 1e-4);
        }
        assert!(result.value < 1e-8);
    }

    #[test]
    fn simplex_rosenbrock() {
        let cost = Rosenbrock;
        let simplex = Simplex::new(0.5);
        let criteria = EndCriteria {
            max_iterations: 10000,
            max_stationary_iterations: 5000,
            function_epsilon: 1e-14,
            ..EndCriteria::default()
        };
        let result = simplex
            .minimize(&cost, &[-1.0, 1.0], &criteria)
            .unwrap();
        assert_abs_diff_eq!(result.parameters[0], 1.0, epsilon = 1e-3);
        assert_abs_diff_eq!(result.parameters[1], 1.0, epsilon = 1e-3);
    }

    #[test]
    fn lm_sphere() {
        let cost = Sphere(3);
        let lm = LevenbergMarquardt::default();
        let criteria = EndCriteria::default();
        let result = lm
            .minimize(&cost, &[3.0, -2.0, 1.0], &criteria)
            .unwrap();
        for &p in &result.parameters {
            assert_abs_diff_eq!(p, 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn lm_linear_fit() {
        // Fit y = 2x + 1
        let cost = LinearFit {
            xs: vec![0.0, 1.0, 2.0, 3.0, 4.0],
            ys: vec![1.0, 3.0, 5.0, 7.0, 9.0],
        };
        let lm = LevenbergMarquardt::default();
        let criteria = EndCriteria::default();
        let result = lm.minimize(&cost, &[0.0, 0.0], &criteria).unwrap();
        assert_abs_diff_eq!(result.parameters[0], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.parameters[1], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn end_criteria_default() {
        let ec = EndCriteria::default();
        assert_eq!(ec.max_iterations, 1000);
        assert_eq!(ec.max_stationary_iterations, 100);
    }

    #[test]
    fn gaussian_elimination() {
        // 2x + y = 5, x + 3y = 7
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b = vec![5.0, 7.0];
        let x = solve_linear_system(&a, &b).unwrap();
        assert_abs_diff_eq!(x[0], 1.6, epsilon = 1e-12);
        assert_abs_diff_eq!(x[1], 1.8, epsilon = 1e-12);
    }
}
