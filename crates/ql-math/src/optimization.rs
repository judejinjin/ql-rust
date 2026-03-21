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
// BFGS (Broyden-Fletcher-Goldfarb-Shanno)
// ===========================================================================

/// BFGS quasi-Newton optimizer.
///
/// Uses finite-difference gradients and a Wolfe-condition line search.
/// Suitable for smooth objective functions.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Bfgs {
    /// Finite difference step for gradient estimation.
    pub fdeps: f64,
    /// Initial Hessian scaling (identity * h0).
    pub h0: f64,
}

impl Default for Bfgs {
    fn default() -> Self {
        Self { fdeps: 1e-5, h0: 1.0 }
    }
}

impl Bfgs {
    pub fn new(fdeps: f64, h0: f64) -> Self {
        Self { fdeps, h0 }
    }

    /// Estimate gradient by central differences.
    fn gradient<C: CostFunction>(cost: &C, params: &[f64], eps: f64) -> Vec<f64> {
        let n = params.len();
        let mut grad = vec![0.0; n];
        for i in 0..n {
            let mut pf = params.to_vec();
            let mut pb = params.to_vec();
            pf[i] += eps;
            pb[i] -= eps;
            grad[i] = (cost.value(&pf) - cost.value(&pb)) / (2.0 * eps);
        }
        grad
    }

    /// Wolfe-condition backtracking line search.
    fn line_search<C: CostFunction>(
        cost: &C,
        params: &[f64],
        direction: &[f64],
        f0: f64,
        g0: &[f64],
    ) -> f64 {
        let c1 = 1e-4;
        let rho = 0.5;
        let dg = g0.iter().zip(direction).map(|(g, d)| g * d).sum::<f64>();
        let mut alpha = 1.0;
        for _ in 0..50 {
            let trial: Vec<f64> = params.iter().zip(direction).map(|(p, d)| p + alpha * d).collect();
            if cost.value(&trial) <= f0 + c1 * alpha * dg {
                return alpha;
            }
            alpha *= rho;
        }
        alpha
    }

    pub fn minimize<C: CostFunction>(
        &self,
        cost: &C,
        initial: &[f64],
        criteria: &EndCriteria,
    ) -> QLResult<OptimizationResult> {
        let n = cost.dimension();
        if initial.len() != n {
            return Err(QLError::InvalidArgument(format!(
                "initial vector length {} != dimension {}", initial.len(), n
            )));
        }

        let mut x = initial.to_vec();
        // Inverse Hessian approximation H (n×n), starts as h0*I
        let mut h: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { self.h0 } else { 0.0 }).collect())
            .collect();

        let mut g = Self::gradient(cost, &x, self.fdeps);
        let mut stationary = 0;
        let mut prev_f = cost.value(&x);

        for iter in 0..criteria.max_iterations {
            // Check convergence
            let gnorm: f64 = g.iter().map(|gi| gi * gi).sum::<f64>().sqrt();
            if gnorm < criteria.gradient_epsilon {
                return Ok(OptimizationResult { parameters: x, value: prev_f, iterations: iter });
            }

            // Compute search direction d = -H * g
            let d: Vec<f64> = (0..n)
                .map(|i| -h[i].iter().zip(&g).map(|(hij, gj)| hij * gj).sum::<f64>())
                .collect();

            let alpha = Self::line_search(cost, &x, &d, prev_f, &g);

            // Update x
            let s: Vec<f64> = d.iter().map(|di| alpha * di).collect();
            let x_new: Vec<f64> = x.iter().zip(&s).map(|(xi, si)| xi + si).collect();

            let g_new = Self::gradient(cost, &x_new, self.fdeps);
            let y: Vec<f64> = g_new.iter().zip(&g).map(|(gn, go)| gn - go).collect();

            let sy: f64 = s.iter().zip(&y).map(|(si, yi)| si * yi).sum();
            if sy.abs() > 1e-30 {
                // BFGS Hessian update
                let rho_bfgs = 1.0 / sy;
                // H_new = (I - rho*s*y^T) H (I - rho*y*s^T) + rho*s*s^T
                let mut h_new = vec![vec![0.0f64; n]; n];
                for i in 0..n {
                    for j in 0..n {
                        let delta_ij = if i == j { 1.0 } else { 0.0 };
                        let a1 = delta_ij - rho_bfgs * s[i] * y[j];
                        let a2 = delta_ij - rho_bfgs * y[i] * s[j];
                        let mut sum = 0.0;
                        #[allow(clippy::needless_range_loop)]
                        for k in 0..n {
                            for l in 0..n {
                                sum += a1 * h[k][l] * a2;
                                // Note: above is not correct for matrix, fix below
                                let _ = sum; // suppress
                            }
                        }
                        // Correct: H_new[i][j] = sum_kl (I - rho s y^T)_ik H_kl (I - rho y s^T)_lj + rho si sj
                        let lhs: f64 = (0..n).map(|k| {
                            let aik = if i == k { 1.0 } else { 0.0 } - rho_bfgs * s[i] * y[k];
                            (0..n).map(|l| {
                                let alj = if l == j { 1.0 } else { 0.0 } - rho_bfgs * y[l] * s[j];
                                aik * h[k][l] * alj
                            }).sum::<f64>()
                        }).sum();
                        h_new[i][j] = lhs + rho_bfgs * s[i] * s[j];
                    }
                }
                h = h_new;
            }

            let f_new = cost.value(&x_new);
            if (prev_f - f_new).abs() < criteria.function_epsilon {
                stationary += 1;
                if stationary >= criteria.max_stationary_iterations {
                    return Ok(OptimizationResult { parameters: x_new, value: f_new, iterations: iter });
                }
            } else {
                stationary = 0;
            }

            x = x_new;
            g = g_new;
            prev_f = f_new;
        }

        Ok(OptimizationResult { parameters: x, value: prev_f, iterations: criteria.max_iterations })
    }
}

// ===========================================================================
// Conjugate Gradient (Fletcher-Reeves / Polak-Ribière)
// ===========================================================================

/// Nonlinear Conjugate Gradient optimizer (Fletcher-Reeves variant).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ConjugateGradient {
    /// Finite difference step for gradient estimation.
    pub fdeps: f64,
}

impl Default for ConjugateGradient {
    fn default() -> Self {
        Self { fdeps: 1e-5 }
    }
}

impl ConjugateGradient {
    pub fn new(fdeps: f64) -> Self {
        Self { fdeps }
    }

    fn gradient<C: CostFunction>(cost: &C, params: &[f64], eps: f64) -> Vec<f64> {
        Bfgs::gradient(cost, params, eps)
    }

    pub fn minimize<C: CostFunction>(
        &self,
        cost: &C,
        initial: &[f64],
        criteria: &EndCriteria,
    ) -> QLResult<OptimizationResult> {
        let n = cost.dimension();
        if initial.len() != n {
            return Err(QLError::InvalidArgument(format!(
                "initial vector length {} != dimension {}", initial.len(), n
            )));
        }

        let mut x = initial.to_vec();
        let mut g = Self::gradient(cost, &x, self.fdeps);
        let mut d: Vec<f64> = g.iter().map(|gi| -gi).collect();
        let mut prev_f = cost.value(&x);
        let mut stationary = 0;

        for iter in 0..criteria.max_iterations {
            let gnorm: f64 = g.iter().map(|gi| gi * gi).sum::<f64>().sqrt();
            if gnorm < criteria.gradient_epsilon {
                return Ok(OptimizationResult { parameters: x, value: prev_f, iterations: iter });
            }

            let alpha = Bfgs::line_search(cost, &x, &d, prev_f, &g);
            let x_new: Vec<f64> = x.iter().zip(&d).map(|(xi, di)| xi + alpha * di).collect();
            let g_new = Self::gradient(cost, &x_new, self.fdeps);

            // Fletcher-Reeves beta
            let g_dot: f64 = g.iter().map(|gi| gi * gi).sum();
            let g_new_dot: f64 = g_new.iter().map(|gi| gi * gi).sum();
            let beta = if g_dot.abs() > 1e-30 { g_new_dot / g_dot } else { 0.0 };

            // Restart every n iterations (standard CG restart)
            let beta = if iter % n == n - 1 { 0.0 } else { beta };

            let d_new: Vec<f64> = g_new.iter().zip(&d).map(|(gni, di)| -gni + beta * di).collect();

            let f_new = cost.value(&x_new);
            if (prev_f - f_new).abs() < criteria.function_epsilon {
                stationary += 1;
                if stationary >= criteria.max_stationary_iterations {
                    return Ok(OptimizationResult { parameters: x_new, value: f_new, iterations: iter });
                }
            } else {
                stationary = 0;
            }

            x = x_new;
            g = g_new;
            d = d_new;
            prev_f = f_new;
        }

        Ok(OptimizationResult { parameters: x, value: prev_f, iterations: criteria.max_iterations })
    }
}

// ===========================================================================
// Simulated Annealing
// ===========================================================================

/// Simulated annealing — stochastic global optimizer.
///
/// Uses Boltzmann acceptance and exponential cooling.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SimulatedAnnealing {
    /// Initial temperature.
    pub t0: f64,
    /// Cooling rate (0 < alpha < 1).
    pub alpha: f64,
    /// Step size for random perturbations.
    pub step_size: f64,
    /// Random seed.
    pub seed: u64,
}

impl Default for SimulatedAnnealing {
    fn default() -> Self {
        Self { t0: 10.0, alpha: 0.995, step_size: 0.1, seed: 42 }
    }
}

impl SimulatedAnnealing {
    pub fn new(t0: f64, alpha: f64, step_size: f64, seed: u64) -> Self {
        Self { t0, alpha, step_size, seed }
    }

    pub fn minimize<C: CostFunction>(
        &self,
        cost: &C,
        initial: &[f64],
        criteria: &EndCriteria,
    ) -> QLResult<OptimizationResult> {
        let n = cost.dimension();
        if initial.len() != n {
            return Err(QLError::InvalidArgument(format!(
                "initial vector length {} != dimension {}", initial.len(), n
            )));
        }

        let mut x = initial.to_vec();
        let mut f = cost.value(&x);
        let mut best_x = x.clone();
        let mut best_f = f;
        let mut temp = self.t0;

        // Simple LCG pseudo-RNG for determinism
        let mut rng_state = self.seed.wrapping_add(1);
        let mut rand_f64 = move || -> f64 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng_state >> 33) as f64 / (u32::MAX as f64)
        };

        for iter in 0..criteria.max_iterations {
            // Pick random dimension and perturb
            let idx = (rand_f64() * n as f64) as usize % n;
            let mut x_new = x.clone();
            x_new[idx] += self.step_size * (2.0 * rand_f64() - 1.0);

            let f_new = cost.value(&x_new);
            let delta = f_new - f;

            // Accept or reject
            let accept = delta < 0.0 || {
                let prob = (-delta / temp).exp();
                rand_f64() < prob
            };

            if accept {
                x = x_new;
                f = f_new;
                if f < best_f {
                    best_f = f;
                    best_x = x.clone();
                }
            }

            temp *= self.alpha;

            if temp < criteria.function_epsilon || best_f < criteria.function_epsilon {
                return Ok(OptimizationResult { parameters: best_x, value: best_f, iterations: iter });
            }
        }

        Ok(OptimizationResult { parameters: best_x, value: best_f, iterations: criteria.max_iterations })
    }
}

// ===========================================================================
// Differential Evolution (DE/rand/1/bin)
// ===========================================================================

/// Differential Evolution optimizer (DE/rand/1/bin strategy).
///
/// Global stochastic optimizer. Works well for non-differentiable,
/// discontinuous, or multimodal objective functions.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DifferentialEvolution {
    /// Population size (default: 10*dim).
    pub pop_size: Option<usize>,
    /// Mutation factor F in [0,2] (default 0.8).
    pub f_factor: f64,
    /// Crossover rate CR in [0,1] (default 0.9).
    pub cr: f64,
    /// Parameter space bounds: (lower, upper) per dimension.
    pub bounds: Vec<(f64, f64)>,
    /// Random seed.
    pub seed: u64,
}

impl DifferentialEvolution {
    pub fn new(bounds: Vec<(f64, f64)>, f_factor: f64, cr: f64, seed: u64) -> Self {
        Self { pop_size: None, f_factor, cr, bounds, seed }
    }

    pub fn minimize<C: CostFunction>(
        &self,
        cost: &C,
        initial: &[f64],
        criteria: &EndCriteria,
    ) -> QLResult<OptimizationResult> {
        let n = cost.dimension();
        if initial.len() != n {
            return Err(QLError::InvalidArgument(format!(
                "initial vector length {} != dimension {}", initial.len(), n
            )));
        }
        let bounds = if self.bounds.len() == n {
            self.bounds.clone()
        } else {
            // Default: use initial ± 5 as bounds
            initial.iter().map(|&xi| (xi - 5.0, xi + 5.0)).collect()
        };

        let np = self.pop_size.unwrap_or(10 * n).max(4);
        let mut rng_state = self.seed.wrapping_add(1);
        let mut rand_f64 = move || -> f64 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng_state >> 33) as f64 / (u32::MAX as f64)
        };

        // Initialize population
        let mut pop: Vec<Vec<f64>> = (0..np)
            .map(|_| {
                bounds.iter().map(|(lo, hi)| lo + rand_f64() * (hi - lo)).collect()
            })
            .collect();
        // Put initial point in first member
        pop[0] = initial.to_vec();

        let mut fitness: Vec<f64> = pop.iter().map(|x| cost.value(x)).collect();

        let (mut best_idx, mut best_f) = fitness.iter().enumerate()
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, &v)| (i, v))
            .unwrap();

        for iter in 0..(criteria.max_iterations / np).max(1) {
            for i in 0..np {
                // Select 3 distinct random indices ≠ i
                let mut candidates: Vec<usize> = (0..np).filter(|&j| j != i).collect();
                // Shuffle first 3
                for k in 0..3 {
                    let j = k + (rand_f64() * (candidates.len() - k) as f64) as usize % (candidates.len() - k);
                    candidates.swap(k, j);
                }
                let (a, b_, c_) = (candidates[0], candidates[1], candidates[2]);

                // Mutation: v = pop[a] + F*(pop[b] - pop[c])
                let mut v: Vec<f64> = pop[a].iter().zip(&pop[b_]).zip(&pop[c_])
                    .map(|((pa, pb), pc)| pa + self.f_factor * (pb - pc))
                    .collect();

                // Clip to bounds
                for j in 0..n {
                    v[j] = v[j].clamp(bounds[j].0, bounds[j].1);
                }

                // Crossover (binomial)
                let j_rand = (rand_f64() * n as f64) as usize % n;
                let trial: Vec<f64> = v.iter().zip(&pop[i])
                    .enumerate()
                    .map(|(j, (vj, pij))| if rand_f64() < self.cr || j == j_rand { *vj } else { *pij })
                    .collect();

                // Selection
                let trial_f = cost.value(&trial);
                if trial_f <= fitness[i] {
                    pop[i] = trial;
                    fitness[i] = trial_f;
                    if trial_f < best_f {
                        best_f = trial_f;
                        best_idx = i;
                    }
                }
            }

            if best_f < criteria.function_epsilon {
                return Ok(OptimizationResult {
                    parameters: pop[best_idx].clone(),
                    value: best_f,
                    iterations: (iter + 1) * np,
                });
            }
        }

        Ok(OptimizationResult {
            parameters: pop[best_idx].clone(),
            value: best_f,
            iterations: criteria.max_iterations,
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
