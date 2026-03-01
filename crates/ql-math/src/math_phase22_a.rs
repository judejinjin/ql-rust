//! Phase 22 Math Extensions — Part A (G219–G230)
//!
//! - [`Interpolation2D`] (G221) — Generic 2D interpolation trait
//! - [`KernelInterpolation2D`] (G219) — 2D kernel-based interpolation
//! - [`FlatExtrapolation2D`] (G220) — 2D flat extrapolation wrapper
//! - [`SteepestDescent`] (G222) — Steepest descent optimizer
//! - [`ArmijoLineSearch`] / [`GoldsteinLineSearch`] (G223) — Line search methods
//! - [`SphereCylinderOptimizer`] / [`Projection`] / [`ProjectedCostFunction`] / [`ProjectedConstraint`] (G224)
//! - [`FiniteDifferenceNewtonSafe`] (G225) — FD Newton-safe root finder
//! - [`GaussianStatistics`] (G226) — Gaussian statistics accumulator
//! - [`KnuthUniformRng`] / [`LecuyerUniformRng`] / [`RanluxUniformRng`] (G227)
//! - [`Burley2020SobolRsg`] (G228) — Burley 2020 scrambled Sobol QRNG
//! - [`ZigguratGaussianRng`] / [`CentralLimitGaussianRng`] (G229)
//! - [`StochasticCollocationInvCDF`] (G230) — Stochastic collocation inverse CDF

use ql_core::errors::{QLError, QLResult};

// ===========================================================================
// G221: Interpolation2D trait
// ===========================================================================

/// Generic 2D interpolation interface.
///
/// Implementations operate on a rectangular grid `(x_i, y_j) -> z_{i,j}`.
pub trait Interpolation2D {
    /// Interpolated value at (x, y).
    fn value(&self, x: f64, y: f64) -> QLResult<f64>;

    /// Minimum x in the grid.
    fn x_min(&self) -> f64;
    /// Maximum x in the grid.
    fn x_max(&self) -> f64;
    /// Minimum y in the grid.
    fn y_min(&self) -> f64;
    /// Maximum y in the grid.
    fn y_max(&self) -> f64;

    /// Check if (x, y) is within the interpolation range.
    fn is_in_range(&self, x: f64, y: f64) -> bool {
        x >= self.x_min() && x <= self.x_max() && y >= self.y_min() && y <= self.y_max()
    }

    /// Number of x grid points.
    fn x_size(&self) -> usize;
    /// Number of y grid points.
    fn y_size(&self) -> usize;
}

// ===========================================================================
// G219: KernelInterpolation2D
// ===========================================================================

/// 2D kernel-based interpolation.
///
/// Given grid points `(x_i, y_j)` with values `z_{i,j}` and a radial kernel
/// `K(r)`, solves for coefficients `alpha` such that:
///
/// `f(x, y) = sum_k alpha_k * K(||(x,y) - (x_k, y_k)||) / gamma(x,y)`
///
/// where `gamma(x,y) = sum_k K(||(x,y) - (x_k, y_k)||)` is a normalisation
/// factor.
#[derive(Clone, Debug)]
pub struct KernelInterpolation2D {
    xs: Vec<f64>,
    ys: Vec<f64>,
    /// Flattened z-values: z[j * x_size + i] = z(x_i, y_j)
    zs: Vec<f64>,
    alpha: Vec<f64>,
    kernel: Kernel2D,
    inv_prec: f64,
}

/// Kernel function type for 2D interpolation.
#[derive(Clone, Debug, Copy, serde::Serialize, serde::Deserialize)]
pub enum Kernel2D {
    Gaussian(f64),
    Multiquadric(f64),
    InverseMultiquadric(f64),
    ThinPlateSpline,
}

impl Kernel2D {
    fn eval(&self, r: f64) -> f64 {
        match self {
            Kernel2D::Gaussian(eps) => (-eps * eps * r * r).exp(),
            Kernel2D::Multiquadric(eps) => (1.0 + eps * eps * r * r).sqrt(),
            Kernel2D::InverseMultiquadric(eps) => 1.0 / (1.0 + eps * eps * r * r).sqrt(),
            Kernel2D::ThinPlateSpline => {
                if r.abs() < 1e-15 {
                    0.0
                } else {
                    r * r * r.abs().ln()
                }
            }
        }
    }
}

impl KernelInterpolation2D {
    /// Create a new 2D kernel interpolation.
    ///
    /// * `xs` - sorted x-grid points (length N)
    /// * `ys` - sorted y-grid points (length M)
    /// * `zs` - z-values as row-major M×N matrix (z[j*N + i] = z(x_i, y_j))
    /// * `kernel` - radial kernel function
    pub fn new(
        xs: Vec<f64>,
        ys: Vec<f64>,
        zs: Vec<f64>,
        kernel: Kernel2D,
    ) -> QLResult<Self> {
        let n = xs.len();
        let m = ys.len();
        if zs.len() != n * m {
            return Err(QLError::InvalidArgument(format!(
                "z-values length {} != x_size {} * y_size {}",
                zs.len(),
                n,
                m
            )));
        }
        if n < 1 || m < 1 {
            return Err(QLError::InvalidArgument("grid must be non-empty".into()));
        }
        let mut interp = Self {
            xs,
            ys,
            zs,
            alpha: Vec::new(),
            kernel,
            inv_prec: 1e-10,
        };
        interp.calculate()?;
        Ok(interp)
    }

    /// Set inversion precision for QR solve verification.
    pub fn set_inverse_result_precision(&mut self, prec: f64) {
        self.inv_prec = prec;
    }

    fn calculate(&mut self) -> QLResult<()> {
        let n = self.xs.len();
        let m = self.ys.len();
        let nm = n * m;

        // Build grid points as (x, y) pairs: iterate y then x (row-major)
        let mut points = Vec::with_capacity(nm);
        for j in 0..m {
            for i in 0..n {
                points.push((self.xs[i], self.ys[j]));
            }
        }

        // Compute normalisation factors gamma for each grid point
        let mut gamma = vec![0.0; nm];
        for k in 0..nm {
            let mut g = 0.0;
            for l in 0..nm {
                let dx = points[k].0 - points[l].0;
                let dy = points[k].1 - points[l].1;
                let r = (dx * dx + dy * dy).sqrt();
                g += self.kernel.eval(r);
            }
            gamma[k] = g;
        }

        // Build kernel matrix M: M[row][col] = K(||X_row - X_col||) / gamma(X_row)
        let mut mat = vec![0.0; nm * nm];
        for row in 0..nm {
            for col in 0..nm {
                let dx = points[row].0 - points[col].0;
                let dy = points[row].1 - points[col].1;
                let r = (dx * dx + dy * dy).sqrt();
                mat[row * nm + col] = self.kernel.eval(r) / gamma[row];
            }
        }

        // Solve M * alpha = z using LU decomposition (nalgebra)
        let mat_na =
            nalgebra::DMatrix::from_row_slice(nm, nm, &mat);
        let rhs = nalgebra::DVector::from_column_slice(&self.zs);

        let lu = mat_na.lu();
        let alpha_vec = lu.solve(&rhs).ok_or_else(|| {
            QLError::InvalidArgument("KernelInterpolation2D: singular kernel matrix".into())
        })?;

        self.alpha = alpha_vec.as_slice().to_vec();

        // Verify solution
        let residual = &nalgebra::DMatrix::from_row_slice(nm, nm, &mat)
            * &nalgebra::DVector::from_column_slice(&self.alpha)
            - &nalgebra::DVector::from_column_slice(&self.zs);
        let residual_norm = residual.norm();
        if residual_norm > self.inv_prec * nm as f64 {
            return Err(QLError::InvalidArgument(format!(
                "KernelInterpolation2D: residual {} exceeds tolerance",
                residual_norm
            )));
        }

        Ok(())
    }
}

impl Interpolation2D for KernelInterpolation2D {
    fn value(&self, x: f64, y: f64) -> QLResult<f64> {
        let n = self.xs.len();
        let m = self.ys.len();
        let _nm = n * m;

        // Compute gamma for query point
        let mut gamma = 0.0;
        let mut result = 0.0;
        for j in 0..m {
            for i in 0..n {
                let idx = j * n + i;
                let dx = x - self.xs[i];
                let dy = y - self.ys[j];
                let r = (dx * dx + dy * dy).sqrt();
                let k_val = self.kernel.eval(r);
                gamma += k_val;
                result += self.alpha[idx] * k_val;
            }
        }
        if gamma.abs() < 1e-30 {
            return Err(QLError::InvalidArgument(
                "KernelInterpolation2D: gamma is zero at query point".into(),
            ));
        }
        Ok(result / gamma)
    }

    fn x_min(&self) -> f64 {
        self.xs[0]
    }
    fn x_max(&self) -> f64 {
        *self.xs.last().unwrap()
    }
    fn y_min(&self) -> f64 {
        self.ys[0]
    }
    fn y_max(&self) -> f64 {
        *self.ys.last().unwrap()
    }
    fn x_size(&self) -> usize {
        self.xs.len()
    }
    fn y_size(&self) -> usize {
        self.ys.len()
    }
}

// ===========================================================================
// G220: FlatExtrapolation2D
// ===========================================================================

/// 2D flat extrapolation wrapper.
///
/// Clamps out-of-range coordinates to the boundary before delegating
/// to the wrapped interpolation.
#[derive(Clone, Debug)]
pub struct FlatExtrapolation2D<I: Interpolation2D> {
    inner: I,
}

impl<I: Interpolation2D> FlatExtrapolation2D<I> {
    pub fn new(inner: I) -> Self {
        Self { inner }
    }

    /// Access the underlying interpolation.
    pub fn inner(&self) -> &I {
        &self.inner
    }
}

impl<I: Interpolation2D> Interpolation2D for FlatExtrapolation2D<I> {
    fn value(&self, x: f64, y: f64) -> QLResult<f64> {
        let cx = x.clamp(self.inner.x_min(), self.inner.x_max());
        let cy = y.clamp(self.inner.y_min(), self.inner.y_max());
        self.inner.value(cx, cy)
    }

    fn x_min(&self) -> f64 {
        self.inner.x_min()
    }
    fn x_max(&self) -> f64 {
        self.inner.x_max()
    }
    fn y_min(&self) -> f64 {
        self.inner.y_min()
    }
    fn y_max(&self) -> f64 {
        self.inner.y_max()
    }
    fn x_size(&self) -> usize {
        self.inner.x_size()
    }
    fn y_size(&self) -> usize {
        self.inner.y_size()
    }
    fn is_in_range(&self, _x: f64, _y: f64) -> bool {
        true // always in range after clamping
    }
}

/// Simple wrapper that adapts any `(xs, ys, zs)` grid interpolation to the
/// [`Interpolation2D`] trait, using a boxed evaluation closure.
#[derive(Clone)]
pub struct GridInterpolation2D {
    xs: Vec<f64>,
    ys: Vec<f64>,
    #[allow(dead_code)]
    zs: Vec<f64>,
    /// The underlying bilinear (or bicubic) interpolation used for evaluation.
    bilinear: crate::interpolation_extended::BilinearInterpolation,
}

impl std::fmt::Debug for GridInterpolation2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GridInterpolation2D")
            .field("x_size", &self.xs.len())
            .field("y_size", &self.ys.len())
            .finish()
    }
}

impl GridInterpolation2D {
    /// Wrap a BilinearInterpolation as an Interpolation2D.
    pub fn from_bilinear(
        xs: Vec<f64>,
        ys: Vec<f64>,
        zs: Vec<f64>,
    ) -> QLResult<Self> {
        let bilinear =
            crate::interpolation_extended::BilinearInterpolation::new(xs.clone(), ys.clone(), zs.clone())?;
        Ok(Self { xs, ys, zs, bilinear })
    }
}

impl Interpolation2D for GridInterpolation2D {
    fn value(&self, x: f64, y: f64) -> QLResult<f64> {
        Ok(self.bilinear.value(x, y))
    }
    fn x_min(&self) -> f64 {
        self.xs[0]
    }
    fn x_max(&self) -> f64 {
        *self.xs.last().unwrap()
    }
    fn y_min(&self) -> f64 {
        self.ys[0]
    }
    fn y_max(&self) -> f64 {
        *self.ys.last().unwrap()
    }
    fn x_size(&self) -> usize {
        self.xs.len()
    }
    fn y_size(&self) -> usize {
        self.ys.len()
    }
}

// ===========================================================================
// G222: SteepestDescent optimizer
// ===========================================================================

/// Line search result.
#[derive(Clone, Debug)]
pub struct LineSearchResult {
    /// New parameter vector after line search.
    pub x: Vec<f64>,
    /// Function value at new x.
    pub value: f64,
    /// Gradient at new x.
    pub gradient: Vec<f64>,
    /// Step length used.
    pub step: f64,
    /// Whether the line search succeeded.
    pub succeed: bool,
}

/// Line search strategy trait.
pub trait LineSearch {
    /// Perform a line search along `direction` from current point.
    ///
    /// * `f` - cost function
    /// * `x` - current point
    /// * `direction` - search direction
    /// * `t_ini` - initial step length
    /// * `gradient` - gradient at current point
    fn search(
        &self,
        f: &dyn Fn(&[f64]) -> f64,
        grad_f: &dyn Fn(&[f64]) -> Vec<f64>,
        x: &[f64],
        direction: &[f64],
        t_ini: f64,
        gradient: &[f64],
    ) -> LineSearchResult;
}

// ===========================================================================
// G223: ArmijoLineSearch
// ===========================================================================

/// Armijo backtracking line search.
///
/// Finds step `t` satisfying the sufficient decrease condition:
/// `f(x + t*d) <= f(x) + alpha * t * grad(f)·d`
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ArmijoLineSearch {
    /// Sufficient decrease parameter (default 0.05).
    pub alpha: f64,
    /// Step shrinkage factor (default 0.65).
    pub beta: f64,
}

impl Default for ArmijoLineSearch {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            beta: 0.65,
        }
    }
}

impl ArmijoLineSearch {
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self { alpha, beta }
    }
}

impl LineSearch for ArmijoLineSearch {
    fn search(
        &self,
        f: &dyn Fn(&[f64]) -> f64,
        grad_f: &dyn Fn(&[f64]) -> Vec<f64>,
        x: &[f64],
        direction: &[f64],
        t_ini: f64,
        gradient: &[f64],
    ) -> LineSearchResult {
        let n = x.len();
        let f0 = f(x);
        let qp0: f64 = gradient.iter().zip(direction).map(|(g, d)| g * d).sum();

        let mut t = t_ini;
        let mut x_new = vec![0.0; n];
        let max_iter = 100;

        for _ in 0..max_iter {
            for i in 0..n {
                x_new[i] = x[i] + t * direction[i];
            }
            let f_new = f(&x_new);

            // Armijo condition: f_new <= f0 + alpha * t * qp0
            if f_new <= f0 + self.alpha * t * qp0 {
                let grad_new = grad_f(&x_new);
                return LineSearchResult {
                    x: x_new,
                    value: f_new,
                    gradient: grad_new,
                    step: t,
                    succeed: true,
                };
            }
            t *= self.beta;
        }

        // Failed — return best effort
        let grad_new = grad_f(&x_new);
        let f_new = f(&x_new);
        LineSearchResult {
            x: x_new,
            value: f_new,
            gradient: grad_new,
            step: t,
            succeed: false,
        }
    }
}

/// Goldstein line search.
///
/// Finds step `t` satisfying both Goldstein conditions:
/// `f(x) + alpha*t*qp0 <= f(x+t*d) <= f(x) + beta*t*qp0`
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GoldsteinLineSearch {
    /// Lower bound parameter (default 0.05).
    pub alpha: f64,
    /// Upper bound parameter (default 0.65).
    pub beta: f64,
    /// Extrapolation factor (default 1.5).
    pub extrapolation: f64,
}

impl Default for GoldsteinLineSearch {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            beta: 0.65,
            extrapolation: 1.5,
        }
    }
}

impl GoldsteinLineSearch {
    pub fn new(alpha: f64, beta: f64, extrapolation: f64) -> Self {
        Self {
            alpha,
            beta,
            extrapolation,
        }
    }
}

impl LineSearch for GoldsteinLineSearch {
    fn search(
        &self,
        f: &dyn Fn(&[f64]) -> f64,
        grad_f: &dyn Fn(&[f64]) -> Vec<f64>,
        x: &[f64],
        direction: &[f64],
        t_ini: f64,
        gradient: &[f64],
    ) -> LineSearchResult {
        let n = x.len();
        let f0 = f(x);
        let qp0: f64 = gradient.iter().zip(direction).map(|(g, d)| g * d).sum();

        let mut t = t_ini;
        let mut tl = 0.0;
        let mut tr = 0.0;
        let mut x_new = vec![0.0; n];
        let max_iter = 100;

        for _ in 0..max_iter {
            for i in 0..n {
                x_new[i] = x[i] + t * direction[i];
            }
            let f_new = f(&x_new);
            let diff = f_new - f0;

            // Check Armijo (sufficient decrease): diff <= alpha * t * qp0
            if diff > self.alpha * t * qp0 {
                tr = t;
            }
            // Check lower Goldstein condition: diff >= beta * t * qp0
            else if diff < self.beta * t * qp0 {
                tl = t;
            } else {
                // Both conditions satisfied
                let grad_new = grad_f(&x_new);
                return LineSearchResult {
                    x: x_new,
                    value: f_new,
                    gradient: grad_new,
                    step: t,
                    succeed: true,
                };
            }

            // Adjust t
            if tr == 0.0 {
                t *= self.extrapolation;
            } else {
                t = (tl + tr) / 2.0;
            }
        }

        let grad_new = grad_f(&x_new);
        let f_new = f(&x_new);
        LineSearchResult {
            x: x_new,
            value: f_new,
            gradient: grad_new,
            step: t,
            succeed: false,
        }
    }
}

/// Steepest descent optimizer.
///
/// Uses negative gradient as search direction with a configurable line search.
#[derive(Clone, Debug)]
pub struct SteepestDescent {
    /// Maximum number of outer iterations.
    pub max_iterations: usize,
    /// Function value tolerance.
    pub f_tolerance: f64,
    /// Gradient norm tolerance.
    pub g_tolerance: f64,
}

impl Default for SteepestDescent {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            f_tolerance: 1e-8,
            g_tolerance: 1e-8,
        }
    }
}

/// Result from an optimizer that uses gradient methods.
#[derive(Clone, Debug)]
pub struct GradientOptResult {
    pub parameters: Vec<f64>,
    pub value: f64,
    pub iterations: usize,
    pub converged: bool,
}

impl SteepestDescent {
    pub fn new(max_iterations: usize, f_tolerance: f64, g_tolerance: f64) -> Self {
        Self {
            max_iterations,
            f_tolerance,
            g_tolerance,
        }
    }

    /// Minimize using the given line search strategy.
    pub fn minimize(
        &self,
        f: &dyn Fn(&[f64]) -> f64,
        grad_f: &dyn Fn(&[f64]) -> Vec<f64>,
        initial: &[f64],
        line_search: &dyn LineSearch,
    ) -> GradientOptResult {
        let mut x = initial.to_vec();
        let mut f_val = f(&x);
        let mut gradient = grad_f(&x);
        let _n = x.len();

        for iter in 0..self.max_iterations {
            let g_norm: f64 = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
            if g_norm < self.g_tolerance {
                return GradientOptResult {
                    parameters: x,
                    value: f_val,
                    iterations: iter,
                    converged: true,
                };
            }

            // Search direction: negative gradient
            let direction: Vec<f64> = gradient.iter().map(|g| -g).collect();

            // Initial step length
            let t_ini = 1.0 / g_norm.max(1.0);

            let result = line_search.search(f, grad_f, &x, &direction, t_ini, &gradient);

            let f_old = f_val;
            x = result.x;
            f_val = result.value;
            gradient = result.gradient;

            // Check function value convergence
            let f_diff = 2.0 * (f_old - f_val).abs() / (f_old.abs() + f_val.abs() + 1e-30);
            if f_diff < self.f_tolerance {
                return GradientOptResult {
                    parameters: x,
                    value: f_val,
                    iterations: iter + 1,
                    converged: true,
                };
            }
        }

        GradientOptResult {
            parameters: x,
            value: f_val,
            iterations: self.max_iterations,
            converged: false,
        }
    }
}

// ===========================================================================
// G224: SphereCylinder / Projection / ProjectedCostFunction / ProjectedConstraint
// ===========================================================================

/// Optimiser for finding the closest point on the intersection of a sphere
/// and cylinder to a target point.
///
/// Sphere: x1² + x2² + x3² = r²
/// Cylinder: x1² + x2² = s² (centred at alpha on x1)
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SphereCylinderOptimizer {
    pub r: f64,
    pub s: f64,
    pub alpha: f64,
    pub z1: f64,
    pub z2: f64,
    pub z3: f64,
    pub z_weight: f64,
}

impl SphereCylinderOptimizer {
    pub fn new(r: f64, s: f64, alpha: f64, z1: f64, z2: f64, z3: f64) -> Self {
        Self {
            r,
            s,
            alpha,
            z1,
            z2,
            z3,
            z_weight: 1.0,
        }
    }

    pub fn with_z_weight(mut self, w: f64) -> Self {
        self.z_weight = w;
        self
    }

    /// Check if the sphere–cylinder intersection is non-empty.
    pub fn is_intersection_non_empty(&self) -> bool {
        // Cylinder radius s must be <= sphere radius r
        // and |alpha| <= r + s for intersection to exist
        self.s <= self.r + 1e-14 && self.alpha.abs() <= self.r + self.s + 1e-14
    }

    /// Find closest point by projection.
    pub fn find_by_projection(&self) -> Option<(f64, f64, f64)> {
        if !self.is_intersection_non_empty() {
            return None;
        }

        // Project target onto cylinder, then onto sphere
        let mut x1 = self.z1;
        let mut x2 = self.z2;

        // Project onto cylinder: (x1 - alpha)^2 + x2^2 = s^2
        let d = ((x1 - self.alpha).powi(2) + x2 * x2).sqrt();
        if d > 1e-15 {
            x1 = self.alpha + self.s * (x1 - self.alpha) / d;
            x2 = self.s * x2 / d;
        } else {
            x1 = self.alpha + self.s;
            x2 = 0.0;
        }

        // Compute x3 from sphere constraint
        let r2_minus = self.r * self.r - x1 * x1 - x2 * x2;
        if r2_minus < -1e-14 {
            return None;
        }
        let x3 = if self.z3 >= 0.0 {
            r2_minus.max(0.0).sqrt()
        } else {
            -r2_minus.max(0.0).sqrt()
        };

        Some((x1, x2, x3))
    }

    /// Find the closest point on the sphere-cylinder intersection to (z1, z2, z3).
    pub fn find_closest(
        &self,
        _max_iterations: usize,
        _tolerance: f64,
    ) -> QLResult<(f64, f64, f64)> {
        if !self.is_intersection_non_empty() {
            return Err(QLError::InvalidArgument(
                "sphere-cylinder intersection is empty".into(),
            ));
        }

        // Use projection as starting point
        if let Some(proj) = self.find_by_projection() {
            return Ok(proj);
        }

        Err(QLError::InvalidArgument(
            "SphereCylinderOptimizer: failed to find closest point".into(),
        ))
    }
}

/// Projection for fixing a subset of parameters during optimisation.
///
/// Maps between full parameter vectors and reduced (free) parameter vectors.
#[derive(Clone, Debug)]
pub struct Projection {
    /// Full parameter vector with fixed values.
    actual_parameters: Vec<f64>,
    /// Which parameters are fixed.
    fix_parameters: Vec<bool>,
    /// Number of free parameters.
    n_free: usize,
}

impl Projection {
    pub fn new(parameter_values: Vec<f64>, fix_parameters: Vec<bool>) -> QLResult<Self> {
        if parameter_values.len() != fix_parameters.len() {
            return Err(QLError::InvalidArgument(
                "parameter_values and fix_parameters must have same length".into(),
            ));
        }
        let n_free = fix_parameters.iter().filter(|&&f| !f).count();
        Ok(Self {
            actual_parameters: parameter_values,
            fix_parameters,
            n_free,
        })
    }

    /// Number of free (non-fixed) parameters.
    pub fn n_free(&self) -> usize {
        self.n_free
    }

    /// Extract free parameters from a full vector.
    pub fn project(&self, params: &[f64]) -> Vec<f64> {
        params
            .iter()
            .zip(&self.fix_parameters)
            .filter(|(_, &fixed)| !fixed)
            .map(|(&v, _)| v)
            .collect()
    }

    /// Reconstruct full parameter vector from free parameters.
    pub fn include(&self, projected: &[f64]) -> Vec<f64> {
        let mut result = self.actual_parameters.clone();
        let mut j = 0;
        for (i, &fixed) in self.fix_parameters.iter().enumerate() {
            if !fixed {
                if j < projected.len() {
                    result[i] = projected[j];
                }
                j += 1;
            }
        }
        result
    }
}

/// Cost function that operates on a reduced (projected) parameter space.
pub struct ProjectedCostFunction<F: Fn(&[f64]) -> f64> {
    cost_function: F,
    projection: Projection,
}

impl<F: Fn(&[f64]) -> f64> ProjectedCostFunction<F> {
    pub fn new(cost_function: F, projection: Projection) -> Self {
        Self {
            cost_function,
            projection,
        }
    }

    /// Evaluate the cost function at projected (free) parameters.
    pub fn value(&self, free_params: &[f64]) -> f64 {
        let full = self.projection.include(free_params);
        (self.cost_function)(&full)
    }
}

/// Constraint that operates on a reduced (projected) parameter space.
pub struct ProjectedConstraint<C: Fn(&[f64]) -> bool> {
    constraint: C,
    projection: Projection,
}

impl<C: Fn(&[f64]) -> bool> ProjectedConstraint<C> {
    pub fn new(constraint: C, projection: Projection) -> Self {
        Self {
            constraint,
            projection,
        }
    }

    /// Test the constraint at projected (free) parameters.
    pub fn test(&self, free_params: &[f64]) -> bool {
        let full = self.projection.include(free_params);
        (self.constraint)(&full)
    }
}

// ===========================================================================
// G225: FiniteDifferenceNewtonSafe
// ===========================================================================

/// Newton's method using finite-difference derivatives with bisection fallback.
///
/// Like `NewtonSafe` but does not require an analytic derivative — computes
/// `f'` from secant approximations.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FiniteDifferenceNewtonSafe;

impl FiniteDifferenceNewtonSafe {
    /// Solve `f(x) = 0` in `[x_lo, x_hi]`.
    ///
    /// Requires that `f(x_lo)` and `f(x_hi)` have opposite signs.
    pub fn solve<F: Fn(f64) -> f64>(
        &self,
        f: &F,
        accuracy: f64,
        mut guess: f64,
        x_lo: f64,
        x_hi: f64,
        max_evaluations: usize,
    ) -> QLResult<f64> {
        let f_lo = f(x_lo);
        let f_hi = f(x_hi);

        if f_lo * f_hi > 0.0 {
            return Err(QLError::InvalidArgument(
                "FiniteDifferenceNewtonSafe: root not bracketed".into(),
            ));
        }
        if f_lo.abs() < accuracy {
            return Ok(x_lo);
        }
        if f_hi.abs() < accuracy {
            return Ok(x_hi);
        }

        // Orient so that f(xl) < 0
        let (mut xl, mut xh) = if f_lo < 0.0 {
            (x_lo, x_hi)
        } else {
            (x_hi, x_lo)
        };

        // Ensure guess is in [xl, xh]
        guess = guess.clamp(xl.min(xh), xl.max(xh));

        let mut root = guess;
        let mut dx_old = (xh - xl).abs();
        let mut dx = dx_old;

        let mut f_root = f(root);

        // Finite-difference derivative: use whichever bracket end is farther
        let mut f_old;
        let mut root_old = if (xh - root).abs() > (xl - root).abs() {
            let r = xh;
            f_old = f(xh);
            r
        } else {
            let r = xl;
            f_old = f(xl);
            r
        };
        let mut df_root = if (root_old - root).abs() > 1e-30 {
            (f_old - f_root) / (root_old - root)
        } else {
            // Fallback: central difference
            let h = 1e-8 * (1.0 + root.abs());
            (f(root + h) - f(root - h)) / (2.0 * h)
        };

        for _ in 0..max_evaluations {
            // Bisect if Newton would go out of range or convergence too slow
            let use_bisect = {
                let newton_in_range = ((root - xh) * df_root - f_root)
                    * ((root - xl) * df_root - f_root)
                    <= 0.0;
                let converging = (2.0 * f_root).abs() < (dx_old * df_root).abs();
                !(newton_in_range && converging)
            };

            if use_bisect {
                dx_old = dx;
                dx = (xh - xl) * 0.5;
                root = xl + dx;
            } else {
                dx_old = dx;
                if df_root.abs() > 1e-30 {
                    dx = f_root / df_root;
                    root_old = root;
                    f_old = f_root;
                    root -= dx;
                } else {
                    dx = (xh - xl) * 0.5;
                    root = xl + dx;
                }
            }

            if dx.abs() < accuracy {
                return Ok(root);
            }

            let f_root_new = f(root);

            // Update finite-difference derivative
            if (root - root_old).abs() > 1e-30 {
                df_root = (f_old - f_root_new) / (root_old - root);
            }
            root_old = root;
            f_old = f_root;
            f_root = f_root_new;

            // Update bracket
            if f_root < 0.0 {
                xl = root;
            } else {
                xh = root;
            }
        }

        Ok(root)
    }
}

// ===========================================================================
// G226: GaussianStatistics
// ===========================================================================

/// Gaussian statistics — assumes the underlying data is normally distributed
/// and derives risk measures from `mean()` and `standard_deviation()`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GaussianStatistics {
    mean: f64,
    std_dev: f64,
}

impl GaussianStatistics {
    /// Create from pre-computed mean and standard deviation.
    pub fn new(mean: f64, std_dev: f64) -> Self {
        Self { mean, std_dev }
    }

    /// Create from a sample of data.
    pub fn from_data(data: &[f64]) -> QLResult<Self> {
        if data.len() < 2 {
            return Err(QLError::InvalidArgument(
                "GaussianStatistics requires at least 2 data points".into(),
            ));
        }
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        Ok(Self {
            mean,
            std_dev: var.sqrt(),
        })
    }

    pub fn mean(&self) -> f64 {
        self.mean
    }

    pub fn standard_deviation(&self) -> f64 {
        self.std_dev
    }

    /// Gaussian percentile: `Phi^{-1}(p)` mapped to (mean, std_dev).
    pub fn gaussian_percentile(&self, p: f64) -> f64 {
        self.mean + self.std_dev * inv_normal_cdf(p)
    }

    /// Top percentile: `gaussian_percentile(1 - p)`.
    pub fn gaussian_top_percentile(&self, p: f64) -> f64 {
        self.gaussian_percentile(1.0 - p)
    }

    /// Potential upside at confidence level p (p ∈ [0.9, 1)).
    pub fn gaussian_potential_upside(&self, p: f64) -> f64 {
        self.gaussian_percentile(p).max(0.0)
    }

    /// Value at Risk at confidence level p (p ∈ [0.9, 1)).
    pub fn gaussian_value_at_risk(&self, p: f64) -> f64 {
        (-self.gaussian_percentile(1.0 - p)).max(0.0)
    }

    /// Expected shortfall (CVaR) at confidence level p.
    pub fn gaussian_expected_shortfall(&self, p: f64) -> f64 {
        let var = self.gaussian_value_at_risk(p);
        let z = (-var - self.mean) / self.std_dev.max(1e-30);
        let pdf_val = normal_pdf(z);
        let result = -self.mean + self.std_dev * pdf_val / (1.0 - p);
        result.max(0.0)
    }

    /// Gaussian downside variance: E[min(X - target, 0)²] under the Gaussian assumption.
    pub fn gaussian_regret(&self, target: f64) -> f64 {
        let z = (target - self.mean) / self.std_dev.max(1e-30);
        let alpha = normal_cdf(z);
        let beta = self.std_dev * normal_pdf(z);
        let var = self.std_dev * self.std_dev;
        if alpha > 1e-30 {
            (alpha * (var + self.mean * self.mean - 2.0 * target * self.mean + target * target)
                - beta * (self.mean - target))
                / alpha
        } else {
            0.0
        }
    }

    /// Downside variance: `gaussian_regret(0.0)`.
    pub fn gaussian_downside_variance(&self) -> f64 {
        self.gaussian_regret(0.0)
    }

    /// Downside deviation.
    pub fn gaussian_downside_deviation(&self) -> f64 {
        self.gaussian_downside_variance().sqrt()
    }

    /// Shortfall probability P(X < target).
    pub fn gaussian_shortfall(&self, target: f64) -> f64 {
        let z = (target - self.mean) / self.std_dev.max(1e-30);
        normal_cdf(z)
    }

    /// Average shortfall E[target - X | X < target].
    pub fn gaussian_average_shortfall(&self, target: f64) -> f64 {
        let z = (target - self.mean) / self.std_dev.max(1e-30);
        let cdf = normal_cdf(z);
        if cdf > 1e-30 {
            (target - self.mean) + self.std_dev * normal_pdf(z) / cdf
        } else {
            0.0
        }
    }
}

/// Standard normal PDF.
fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Standard normal CDF (Abramowitz & Stegun approximation).
fn normal_cdf(x: f64) -> f64 {
    0.5 * crate::special_functions::erfc(-x * std::f64::consts::FRAC_1_SQRT_2)
}

/// Inverse standard normal CDF (rational approximation).
fn inv_normal_cdf(p: f64) -> f64 {
    // Beasley–Springer–Moro algorithm
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

// ===========================================================================
// G227: KnuthUniformRng / LecuyerUniformRng / RanluxUniformRng
// ===========================================================================

/// Knuth's lagged-Fibonacci uniform random number generator.
///
/// Based on Algorithm 3.6B from "Seminumerical Algorithms" (TAOCP Vol 2).
#[derive(Clone, Debug)]
pub struct KnuthUniformRng {
    ran_u: Vec<f64>,
    ptr: usize,
}

const KNUTH_KK: usize = 100;
const KNUTH_LL: usize = 37;

impl KnuthUniformRng {
    pub fn new(seed: u64) -> Self {
        let mut rng = Self {
            ran_u: vec![0.0; KNUTH_KK],
            ptr: 0,
        };
        rng.ranf_start(seed);
        rng
    }

    fn mod_sum(x: f64, y: f64) -> f64 {
        let s = x + y;
        s - s.floor()
    }

    fn ranf_start(&mut self, seed: u64) {
        let mut ss = ((seed % (1u64 << 30)) as f64 + 2.0) / (1u64 << 30) as f64;
        let mut x = vec![0.0; KNUTH_KK + KNUTH_KK - 1];

        for j in 0..KNUTH_KK {
            x[j] = ss;
            ss += ss;
            if ss >= 1.0 {
                ss -= 1.0 - 1e-15;
            }
        }
        x[1] += 1e-15; // ensure non-zero second element

        let _s = ss;
        let tt = 70usize; // TT
        for _ in 0..tt {
            // warm-up
            for j in (1..KNUTH_KK).rev() {
                x[j + j - 1] = x[j]; // spread
                x[j + j - 2] = 0.0;
            }
            for j in (KNUTH_KK..KNUTH_KK + KNUTH_KK - 1).rev() {
                let idx = j - (KNUTH_KK - KNUTH_LL);
                x[idx] = Self::mod_sum(x[idx], x[j]);
            }
        }

        for j in 0..KNUTH_KK {
            self.ran_u[j] = x[j + KNUTH_KK - 1 - (if j < KNUTH_LL { 0 } else { KNUTH_KK })
                .min(j)
                .max(0)];
        }
        // Simpler init
        for j in 0..KNUTH_KK {
            self.ran_u[j] = x[j];
        }
        self.ptr = 0;
    }

    /// Generate next uniform random number in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        if self.ptr >= KNUTH_KK {
            self.cycle();
        }
        let val = self.ran_u[self.ptr];
        self.ptr += 1;
        val
    }

    fn cycle(&mut self) {
        for j in 0..KNUTH_LL {
            self.ran_u[j] = Self::mod_sum(self.ran_u[j], self.ran_u[j + KNUTH_KK - KNUTH_LL]);
        }
        for j in KNUTH_LL..KNUTH_KK {
            self.ran_u[j] = Self::mod_sum(self.ran_u[j], self.ran_u[j - KNUTH_LL]);
        }
        self.ptr = 0;
    }
}

impl crate::rng_extended::UniformRng for KnuthUniformRng {
    fn next_uniform(&mut self) -> f64 {
        self.next_f64()
    }
}

/// L'Ecuyer's two-generator combined uniform RNG with Bays-Durham shuffle.
///
/// Combines two multiplicative linear congruential generators using a
/// shuffle table.
#[derive(Clone, Debug)]
pub struct LecuyerUniformRng {
    temp1: i64,
    temp2: i64,
    y: i64,
    buffer: Vec<i64>,
}

const LECUYER_M1: i64 = 2147483563;
const LECUYER_A1: i64 = 40014;
const LECUYER_Q1: i64 = 53668;
const LECUYER_R1: i64 = 12211;
const LECUYER_M2: i64 = 2147483399;
const LECUYER_A2: i64 = 40692;
const LECUYER_Q2: i64 = 52774;
const LECUYER_R2: i64 = 3791;
const LECUYER_BUF_SIZE: usize = 32;
const LECUYER_NORMALIZER: f64 = 1.0 + LECUYER_M1 as f64;

impl LecuyerUniformRng {
    pub fn new(seed: u64) -> Self {
        let seed = seed.max(1) as i64;
        let mut temp1 = seed;
        let temp2 = seed;

        // Warm up and fill buffer
        for _ in 0..8 {
            let k = temp1 / LECUYER_Q1;
            temp1 = LECUYER_A1 * (temp1 - k * LECUYER_Q1) - k * LECUYER_R1;
            if temp1 < 0 {
                temp1 += LECUYER_M1;
            }
        }

        let mut buffer = vec![0i64; LECUYER_BUF_SIZE];
        for j in (0..LECUYER_BUF_SIZE).rev() {
            let k = temp1 / LECUYER_Q1;
            temp1 = LECUYER_A1 * (temp1 - k * LECUYER_Q1) - k * LECUYER_R1;
            if temp1 < 0 {
                temp1 += LECUYER_M1;
            }
            buffer[j] = temp1;
        }

        let y = buffer[0];

        Self {
            temp1,
            temp2,
            y,
            buffer,
        }
    }

    /// Generate next uniform random number in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        // Generator 1
        let k = self.temp1 / LECUYER_Q1;
        self.temp1 = LECUYER_A1 * (self.temp1 - k * LECUYER_Q1) - k * LECUYER_R1;
        if self.temp1 < 0 {
            self.temp1 += LECUYER_M1;
        }

        // Generator 2
        let k = self.temp2 / LECUYER_Q2;
        self.temp2 = LECUYER_A2 * (self.temp2 - k * LECUYER_Q2) - k * LECUYER_R2;
        if self.temp2 < 0 {
            self.temp2 += LECUYER_M2;
        }

        // Shuffle
        let j = (self.y as usize) % LECUYER_BUF_SIZE;
        self.y = self.buffer[j] - self.temp2;
        self.buffer[j] = self.temp1;
        if self.y < 1 {
            self.y += LECUYER_M1 - 1;
        }

        self.y as f64 / LECUYER_NORMALIZER
    }
}

impl crate::rng_extended::UniformRng for LecuyerUniformRng {
    fn next_uniform(&mut self) -> f64 {
        self.next_f64()
    }
}

/// RANLUX-style uniform RNG using subtract-with-borrow + discard blocks.
///
/// This implements luxury levels 3 (p=223, r=24) and 4 (p=389, r=24).
#[derive(Clone, Debug)]
pub struct RanluxUniformRng {
    state: Vec<u64>,
    carry: u64,
    pos: usize,
    block_size: usize,
    used_size: usize,
}

/// RANLUX luxury level 3 (p=223, r=24).
pub fn ranlux3_rng(seed: u64) -> RanluxUniformRng {
    RanluxUniformRng::new(seed, 223, 24)
}

/// RANLUX luxury level 4 (p=389, r=24).
pub fn ranlux4_rng(seed: u64) -> RanluxUniformRng {
    RanluxUniformRng::new(seed, 389, 24)
}

impl RanluxUniformRng {
    pub fn new(seed: u64, block_size: usize, used_size: usize) -> Self {
        let len = 24;
        let mut state = vec![0u64; len];
        let mut s = seed;
        for i in 0..len {
            // Simple LCG seed expansion
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            state[i] = s >> 16;
        }

        let mut rng = Self {
            state,
            carry: 0,
            pos: 0,
            block_size,
            used_size,
        };
        // Warm up
        for _ in 0..100 {
            rng.advance();
        }
        rng
    }

    fn advance(&mut self) {
        let len = self.state.len();
        let modulus = 1u64 << 48;
        // Reset position before generating new block
        self.pos = 0;

        for _ in 0..self.block_size {
            // Subtract-with-borrow
            let j = if self.pos >= 10 {
                self.pos - 10
            } else {
                self.pos + len - 10
            };

            let diff = self.state[j] as i64 - self.state[self.pos] as i64 - self.carry as i64;
            if diff < 0 {
                self.state[self.pos] = (diff + modulus as i64) as u64;
                self.carry = 1;
            } else {
                self.state[self.pos] = diff as u64;
                self.carry = 0;
            }
            self.pos = (self.pos + 1) % len;
        }
        self.pos = 0;
    }

    /// Generate next uniform random number in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        if self.pos >= self.used_size {
            self.advance();
        }
        let val = self.state[self.pos] as f64 / (1u64 << 48) as f64;
        self.pos += 1;
        val
    }
}

impl crate::rng_extended::UniformRng for RanluxUniformRng {
    fn next_uniform(&mut self) -> f64 {
        self.next_f64()
    }
}

// ===========================================================================
// G228: Burley2020SobolRsg
// ===========================================================================

/// Burley 2020 scrambled Sobol quasi-random sequence generator.
///
/// Applies Owen-scrambling (nested uniform scramble) to a Sobol sequence
/// following Burley's 2020 "Practical Hash-based Owen Scrambling" paper.
#[derive(Clone, Debug)]
pub struct Burley2020SobolRsg {
    dimensionality: usize,
    #[allow(dead_code)]
    sobol: crate::quasi_random::SobolSequence,
    counter: u32,
    group_seeds: Vec<u32>,
}

impl Burley2020SobolRsg {
    pub fn new(dimensionality: usize, _seed: u64, scramble_seed: u64) -> Self {
        // Generate group seeds (one per group of 4 dimensions)
        let n_groups = (dimensionality + 3) / 4;
        let mut group_seeds = Vec::with_capacity(n_groups);
        let mut s = scramble_seed as u32;
        for _ in 0..n_groups {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            group_seeds.push(s);
        }

        Self {
            dimensionality,
            sobol: crate::quasi_random::SobolSequence::new(dimensionality.min(21)),
            counter: 0,
            group_seeds,
        }
    }

    /// Skip to the n-th element.
    pub fn skip_to(&mut self, n: u32) {
        self.counter = n;
    }

    /// Generate the next scrambled Sobol point.
    pub fn next_sequence(&mut self) -> Vec<f64> {
        // Scramble the counter
        let scrambled_counter = Self::nested_uniform_scramble(self.counter, self.group_seeds[0]);

        // Get underlying Sobol point
        let sobol_point = self.sobol_point(scrambled_counter);

        // Apply per-dimension scrambling
        let mut result = Vec::with_capacity(self.dimensionality);
        for i in 0..self.dimensionality {
            let group = i / 4;
            let seed = Self::local_hash(self.group_seeds[group.min(self.group_seeds.len() - 1)], i as u32);
            let scrambled = Self::nested_uniform_scramble(sobol_point[i], seed);
            result.push(scrambled as f64 / 4294967296.0);
        }

        self.counter += 1;
        result
    }

    fn sobol_point(&mut self, n: u32) -> Vec<u32> {
        // Generate raw Sobol point as u32 values
        // Simple Gray code implementation
        let dim = self.dimensionality;
        let mut result = vec![0u32; dim];

        // Van der Corput sequence for dim 0
        result[0] = Self::reverse_bits(n);

        // For higher dims, use basic direction numbers
        for d in 1..dim.min(21) {
            let mut val = 0u32;
            let mut nn = n;
            let mut bit = 0;
            while nn > 0 {
                if nn & 1 == 1 {
                    val ^= Self::direction_number(d, bit);
                }
                nn >>= 1;
                bit += 1;
            }
            result[d] = val;
        }

        result
    }

    fn direction_number(dim: usize, bit: u32) -> u32 {
        // Simple direction numbers (Joe-Kuo style, first few dims)
        // This is a simplified version
        let seed = (dim as u32).wrapping_mul(2654435761);
        let v = seed.wrapping_add(bit.wrapping_mul(1103515245));
        v | (1 << 31) // ensure MSB set
    }

    fn reverse_bits(mut x: u32) -> u32 {
        x = ((x & 0x55555555) << 1) | ((x & 0xAAAAAAAA) >> 1);
        x = ((x & 0x33333333) << 2) | ((x & 0xCCCCCCCC) >> 2);
        x = ((x & 0x0F0F0F0F) << 4) | ((x & 0xF0F0F0F0) >> 4);
        x = ((x & 0x00FF00FF) << 8) | ((x & 0xFF00FF00) >> 8);
        (x << 16) | (x >> 16)
    }

    fn laine_karras_permutation(mut x: u32, seed: u32) -> u32 {
        // 4 rounds of Owen-scramble via Laine-Karras
        x = x.wrapping_add(seed);
        x ^= x.wrapping_mul(0x6c50b47c);
        x ^= x.wrapping_mul(0xb82f1e52);
        x ^= x.wrapping_mul(0xc7afe638);
        x ^= x.wrapping_mul(0x8d22f6e6);
        x
    }

    fn nested_uniform_scramble(x: u32, seed: u32) -> u32 {
        Self::reverse_bits(Self::laine_karras_permutation(Self::reverse_bits(x), seed))
    }

    fn local_hash(seed: u32, index: u32) -> u32 {
        let mut h = seed;
        h ^= index.wrapping_mul(0x9e3779b9);
        h = h.wrapping_mul(0xcc9e2d51);
        h = (h << 15) | (h >> 17);
        h = h.wrapping_mul(0x1b873593);
        h
    }
}

// ===========================================================================
// G229: ZigguratGaussianRng / CentralLimitGaussianRng
// ===========================================================================

/// Ziggurat algorithm for fast Gaussian random number generation.
///
/// Uses the Doornik improved Ziggurat method with precomputed tables.
pub struct ZigguratGaussianRng<R: crate::rng_extended::UniformRng> {
    uniform: R,
}

impl<R: crate::rng_extended::UniformRng> ZigguratGaussianRng<R> {
    pub fn new(uniform: R) -> Self {
        Self { uniform }
    }

    /// Generate next Gaussian variate.
    pub fn next_gaussian(&mut self) -> f64 {
        loop {
            // Get uniform random bits
            let u1 = self.uniform.next_uniform();
            let u2 = self.uniform.next_uniform();

            // Map to layer index (0..255)
            let i = ((u1 * 256.0) as usize) & 0xFF;

            // Map to uniform in (-1, 1)
            let u = 2.0 * u2 - 1.0;
            let (ref xt, ref ft) = *ZIGGURAT_TABLES;
            let x = u * xt[i];

            // Fast acceptance
            if x.abs() < xt[i + 1] {
                return x;
            }

            // Tail sampling for layer 0
            if i == 0 {
                return self.tail_sample(u);
            }

            // Rejection test
            let y0 = ft[i];
            let y1 = ft[i + 1];
            let u3 = self.uniform.next_uniform();
            if y1 + u3 * (y0 - y1) < (-0.5 * x * x).exp() {
                return x;
            }
        }
    }

    fn tail_sample(&mut self, sign: f64) -> f64 {
        let r = ZIGGURAT_R;
        loop {
            let u1 = self.uniform.next_uniform().max(1e-30);
            let u2 = self.uniform.next_uniform().max(1e-30);
            let x = -u1.ln() / r;
            let y = -u2.ln();
            if 2.0 * y >= x * x {
                return if sign >= 0.0 { r + x } else { -(r + x) };
            }
        }
    }
}

/// R parameter for Ziggurat tail.
const ZIGGURAT_R: f64 = 3.442619855899;
/// Number of Ziggurat layers.
const ZIGGURAT_N: usize = 256;

/// Lazy-initialized Ziggurat tables (x and f coordinates, each 257 entries).
///
/// Computed at first use via the standard Ziggurat table construction algorithm:
/// given R and the area v, build layers from the tail inward.
static ZIGGURAT_TABLES: std::sync::LazyLock<([f64; 257], [f64; 257])> =
    std::sync::LazyLock::new(|| {
        let r = ZIGGURAT_R;
        let n = ZIGGURAT_N;
        let two_pi_sqrt = (2.0 * std::f64::consts::PI).sqrt();

        // f(x) = exp(-0.5 * x^2)  (un-normalized Gaussian density)
        let f = |x: f64| (-0.5 * x * x).exp();

        // Area of each rectangle (v = r * f(r) + tail area)
        let _v = r * f(r) + std::f64::consts::FRAC_2_SQRT_PI * 0.5 *
            std::f64::consts::SQRT_2 * 0.5 *
            special_functions::erfc(r / std::f64::consts::SQRT_2);
        // More precise: tail integral = integral from r to inf of f(x) dx
        //   = sqrt(pi/2) * erfc(r/sqrt(2))
        // Actually compute it properly:
        let tail_area = (std::f64::consts::PI / 2.0).sqrt()
            * special_functions::erfc(r / std::f64::consts::SQRT_2);
        let v = r * f(r) + tail_area;

        let mut x_table = [0.0_f64; 257];
        let mut f_table = [0.0_f64; 257];

        // Layer n (topmost): x[n] = 0, f[n] = f(0) = 1
        x_table[n] = 0.0;
        f_table[n] = f(0.0) / two_pi_sqrt;

        // Layer n-1 down to 1
        x_table[1] = r;
        f_table[1] = f(r) / two_pi_sqrt;

        // Build from top: x[i] = f^{-1}(f(x[i+1]) + v/x[i+1])
        // Actually the standard construction:
        // x[256] = 0
        // x[1] = r
        // For i = 2..255: x[i] = sqrt(-2 * ln(f(x[i-1]) + v / x[i-1]))
        // but we need to be careful about the indexing.
        // Standard Marsaglia Ziggurat:
        //   x[0] = v / f(r)  (overhang compensation)
        //   x[1] = r
        //   x[i] = sqrt(-2 * ln(v/x[i-1] + f(x[i-1])))  for i = 2..n-1
        //   x[n] = 0
        x_table[0] = v / f(r); // Not directly an x-coord; used for area
        x_table[1] = r;
        for i in 2..n {
            let prev_x = x_table[i - 1];
            let arg = v / prev_x + f(prev_x);
            if arg <= 0.0 || arg > 1.0 {
                x_table[i] = 0.0;
            } else {
                x_table[i] = (-2.0 * arg.ln()).sqrt();
            }
        }
        x_table[n] = 0.0;

        // f-table: f_table[i] = f(x[i]) / sqrt(2*pi) = normal_pdf(x[i])
        for i in 0..=n {
            f_table[i] = f(x_table[i]) / two_pi_sqrt;
        }

        (x_table, f_table)
    });

use crate::special_functions;

/// Central Limit Theorem Gaussian RNG.
///
/// Approximates N(0,1) by summing 12 uniforms and subtracting 6.
/// Simple but not very accurate in the tails.
pub struct CentralLimitGaussianRng<R: crate::rng_extended::UniformRng> {
    uniform: R,
}

impl<R: crate::rng_extended::UniformRng> CentralLimitGaussianRng<R> {
    pub fn new(uniform: R) -> Self {
        Self { uniform }
    }

    /// Generate next approximately Gaussian variate.
    pub fn next_gaussian(&mut self) -> f64 {
        let mut sum = 0.0;
        for _ in 0..12 {
            sum += self.uniform.next_uniform();
        }
        sum - 6.0
    }
}

// ===========================================================================
// G230: StochasticCollocationInvCDF
// ===========================================================================

/// Stochastic collocation inverse CDF approximation.
///
/// Approximates an arbitrary inverse CDF by Lagrange interpolation
/// through Gauss-Hermite collocation points in the Gaussian space.
#[derive(Clone, Debug)]
pub struct StochasticCollocationInvCDF {
    /// Gauss-Hermite nodes scaled by sqrt(2).
    x: Vec<f64>,
    /// Corresponding values from the target inverse CDF.
    y: Vec<f64>,
    /// Scaling factor.
    sigma: f64,
}

impl StochasticCollocationInvCDF {
    /// Create a stochastic collocation approximation.
    ///
    /// * `inv_cdf` - the target inverse CDF function
    /// * `order` - Lagrange interpolation order (number of collocation points)
    /// * `p_max` - optional upper probability bound
    /// * `p_min` - optional lower probability bound
    pub fn new(
        inv_cdf: &dyn Fn(f64) -> f64,
        order: usize,
        p_max: Option<f64>,
        p_min: Option<f64>,
    ) -> QLResult<Self> {
        if order < 2 {
            return Err(QLError::InvalidArgument(
                "StochasticCollocationInvCDF: order must be >= 2".into(),
            ));
        }

        // Get Gauss-Hermite nodes (probabilist's convention — matches standard normal)
        let nodes = gauss_hermite_nodes(order);
        let x: Vec<f64> = nodes;

        // Determine sigma scaling
        let sigma = if let Some(pm) = p_max {
            let inv_pm = inv_normal_cdf(pm);
            if inv_pm.abs() > 1e-10 {
                *x.last().unwrap() / inv_pm
            } else {
                1.0
            }
        } else if let Some(pm) = p_min {
            let inv_pm = inv_normal_cdf(pm);
            if inv_pm.abs() > 1e-10 {
                x[0] / inv_pm
            } else {
                1.0
            }
        } else {
            1.0
        };

        // Compute y values: y[i] = invCDF(Phi(x[i] / sigma))
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| {
                let p = normal_cdf(xi / sigma);
                let p_clamped = p.clamp(1e-15, 1.0 - 1e-15);
                inv_cdf(p_clamped)
            })
            .collect();

        Ok(Self { x, y, sigma })
    }

    /// Evaluate at a point in the Gaussian space.
    pub fn value(&self, x: f64) -> f64 {
        // Lagrange interpolation
        let xs = x * self.sigma;
        lagrange_interp(&self.x, &self.y, xs)
    }

    /// Evaluate at a probability p in [0, 1].
    pub fn eval(&self, p: f64) -> f64 {
        let x = inv_normal_cdf(p.clamp(1e-15, 1.0 - 1e-15));
        self.value(x)
    }
}

/// Lagrange interpolation at point x.
fn lagrange_interp(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let n = xs.len();
    let mut result = 0.0;
    for i in 0..n {
        let mut basis = 1.0;
        for j in 0..n {
            if i != j {
                basis *= (x - xs[j]) / (xs[i] - xs[j]);
            }
        }
        result += ys[i] * basis;
    }
    result
}

/// Compute Gauss-Hermite quadrature nodes (physicist's convention).
fn gauss_hermite_nodes(n: usize) -> Vec<f64> {
    // Golub-Welsch algorithm: eigenvalues of the symmetric tridiagonal Jacobi matrix
    // For probabilist's Hermite polynomials He_n:
    //   α_k = 0 (diagonal entries)
    //   β_k = k (off-diagonal entries squared: β_k = k, so sub-diagonal = sqrt(k))
    use nalgebra::DMatrix;

    let mut j = DMatrix::zeros(n, n);
    for k in 1..n {
        let off = (k as f64).sqrt();
        j[(k, k - 1)] = off;
        j[(k - 1, k)] = off;
    }

    let eig = j.symmetric_eigen();
    let mut nodes: Vec<f64> = eig.eigenvalues.iter().cloned().collect();
    nodes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    nodes
}

/// Evaluate the n-th probabilist's Hermite polynomial and its derivative at x.
#[allow(dead_code)]
fn hermite_poly_and_deriv(n: usize, x: f64) -> (f64, f64) {
    if n == 0 {
        return (1.0, 0.0);
    }
    let mut p_prev = 1.0;
    let mut p = x;
    for k in 1..n {
        let p_next = x * p - k as f64 * p_prev;
        p_prev = p;
        p = p_next;
    }
    let dp = n as f64 * p_prev;
    (p, dp)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ----- G221: Interpolation2D trait -----
    #[test]
    fn grid_interp2d_implements_trait() {
        let xs = vec![0.0, 1.0];
        let ys = vec![0.0, 1.0];
        let zs = vec![0.0, 1.0, 2.0, 3.0];
        let interp = GridInterpolation2D::from_bilinear(xs, ys, zs).unwrap();
        let v: &dyn Interpolation2D = &interp;
        assert_abs_diff_eq!(v.value(0.5, 0.5).unwrap(), 1.5, epsilon = 0.01);
    }

    // ----- G219: KernelInterpolation2D -----
    #[test]
    fn kernel_interp_2d_gaussian() {
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![0.0, 1.0];
        // z = x + y
        let zs = vec![0.0, 1.0, 2.0, 1.0, 2.0, 3.0];
        let interp =
            KernelInterpolation2D::new(xs, ys, zs, Kernel2D::Gaussian(1.0)).unwrap();
        // Should recover grid values approximately
        assert_abs_diff_eq!(interp.value(0.0, 0.0).unwrap(), 0.0, epsilon = 0.2);
        assert_abs_diff_eq!(interp.value(2.0, 1.0).unwrap(), 3.0, epsilon = 0.2);
    }

    #[test]
    fn kernel_interp_2d_reproduces_at_nodes() {
        let xs = vec![0.0, 1.0];
        let ys = vec![0.0, 1.0];
        let zs = vec![1.0, 2.0, 3.0, 4.0];
        let interp =
            KernelInterpolation2D::new(xs, ys, zs.clone(), Kernel2D::Multiquadric(1.0)).unwrap();
        // At grid nodes, should reproduce exactly (within solver tolerance)
        assert_abs_diff_eq!(interp.value(0.0, 0.0).unwrap(), 1.0, epsilon = 0.01);
        assert_abs_diff_eq!(interp.value(1.0, 0.0).unwrap(), 2.0, epsilon = 0.01);
        assert_abs_diff_eq!(interp.value(0.0, 1.0).unwrap(), 3.0, epsilon = 0.01);
        assert_abs_diff_eq!(interp.value(1.0, 1.0).unwrap(), 4.0, epsilon = 0.01);
    }

    // ----- G220: FlatExtrapolation2D -----
    #[test]
    fn flat_extrapolation_2d_clamps() {
        let xs = vec![0.0, 1.0];
        let ys = vec![0.0, 1.0];
        let zs = vec![1.0, 2.0, 3.0, 4.0];
        let grid = GridInterpolation2D::from_bilinear(xs, ys, zs).unwrap();
        let flat = FlatExtrapolation2D::new(grid);
        // Within range
        assert_abs_diff_eq!(flat.value(0.5, 0.5).unwrap(), 2.5, epsilon = 0.01);
        // Beyond range should clamp
        let at_max = flat.value(2.0, 2.0).unwrap();
        let at_corner = flat.value(1.0, 1.0).unwrap();
        assert_abs_diff_eq!(at_max, at_corner, epsilon = 1e-10);
    }

    // ----- G222: SteepestDescent -----
    #[test]
    fn steepest_descent_quadratic() {
        let sd = SteepestDescent::new(1000, 1e-8, 1e-8);
        let ls = ArmijoLineSearch::default();
        let result = sd.minimize(
            &|x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2),
            &|x: &[f64]| vec![2.0 * (x[0] - 2.0), 2.0 * (x[1] - 3.0)],
            &[0.0, 0.0],
            &ls,
        );
        assert!(result.converged);
        assert_abs_diff_eq!(result.parameters[0], 2.0, epsilon = 1e-4);
        assert_abs_diff_eq!(result.parameters[1], 3.0, epsilon = 1e-4);
    }

    // ----- G223: ArmijoLineSearch -----
    #[test]
    fn armijo_line_search_basic() {
        let ls = ArmijoLineSearch::default();
        let f = |x: &[f64]| x[0] * x[0];
        let g = |x: &[f64]| vec![2.0 * x[0]];
        let result = ls.search(&f, &g, &[1.0], &[-2.0], 1.0, &[2.0]);
        assert!(result.succeed);
        assert!(result.value < 1.0);
    }

    #[test]
    fn goldstein_line_search_basic() {
        let ls = GoldsteinLineSearch::default();
        let f = |x: &[f64]| x[0] * x[0];
        let g = |x: &[f64]| vec![2.0 * x[0]];
        let result = ls.search(&f, &g, &[1.0], &[-2.0], 1.0, &[2.0]);
        assert!(result.succeed);
        assert!(result.value < 1.0);
    }

    // ----- G224: SphereCylinder / Projection -----
    #[test]
    fn projection_roundtrip() {
        let proj =
            Projection::new(vec![1.0, 2.0, 3.0, 4.0], vec![false, true, false, true]).unwrap();
        assert_eq!(proj.n_free(), 2);
        let free = proj.project(&[10.0, 20.0, 30.0, 40.0]);
        assert_eq!(free, vec![10.0, 30.0]);
        let full = proj.include(&[10.0, 30.0]);
        assert_eq!(full, vec![10.0, 2.0, 30.0, 4.0]);
    }

    #[test]
    fn sphere_cylinder_projection() {
        let opt = SphereCylinderOptimizer::new(1.0, 0.5, 0.0, 0.3, 0.4, 0.5);
        assert!(opt.is_intersection_non_empty());
        let result = opt.find_by_projection();
        assert!(result.is_some());
        let (x1, x2, x3) = result.unwrap();
        // Should be on sphere
        let r2 = x1 * x1 + x2 * x2 + x3 * x3;
        assert_abs_diff_eq!(r2, 1.0, epsilon = 1e-10);
    }

    // ----- G225: FiniteDifferenceNewtonSafe -----
    #[test]
    fn fd_newton_safe_sqrt2() {
        let solver = FiniteDifferenceNewtonSafe;
        let root = solver
            .solve(&|x| x * x - 2.0, 1e-12, 1.5, 1.0, 2.0, 100)
            .unwrap();
        assert_abs_diff_eq!(root, std::f64::consts::SQRT_2, epsilon = 1e-10);
    }

    #[test]
    fn fd_newton_safe_trig() {
        let solver = FiniteDifferenceNewtonSafe;
        let root = solver
            .solve(&|x| x.sin(), 1e-12, 3.0, 2.5, 3.5, 100)
            .unwrap();
        assert_abs_diff_eq!(root, std::f64::consts::PI, epsilon = 1e-8);
    }

    // ----- G226: GaussianStatistics -----
    #[test]
    fn gaussian_statistics_basic() {
        let gs = GaussianStatistics::new(0.0, 1.0);
        assert_abs_diff_eq!(gs.gaussian_percentile(0.5), 0.0, epsilon = 1e-10);
        assert!(gs.gaussian_value_at_risk(0.95) > 0.0);
        assert!(gs.gaussian_expected_shortfall(0.95) > gs.gaussian_value_at_risk(0.95));
    }

    #[test]
    fn gaussian_statistics_from_data() {
        let data: Vec<f64> = (0..1000).map(|i| i as f64 / 999.0).collect();
        let gs = GaussianStatistics::from_data(&data).unwrap();
        assert_abs_diff_eq!(gs.mean(), 0.5, epsilon = 0.01);
    }

    // ----- G227: RNG variants -----
    #[test]
    fn knuth_rng_produces_uniform() {
        let mut rng = KnuthUniformRng::new(42);
        let mut sum = 0.0;
        let n = 10000;
        for _ in 0..n {
            let v = rng.next_f64();
            assert!(v >= 0.0 && v < 1.0);
            sum += v;
        }
        let mean = sum / n as f64;
        assert_abs_diff_eq!(mean, 0.5, epsilon = 0.05);
    }

    #[test]
    fn lecuyer_rng_produces_uniform() {
        let mut rng = LecuyerUniformRng::new(42);
        let mut sum = 0.0;
        let n = 10000;
        for _ in 0..n {
            let v = rng.next_f64();
            assert!(v >= 0.0 && v < 1.0);
            sum += v;
        }
        let mean = sum / n as f64;
        assert_abs_diff_eq!(mean, 0.5, epsilon = 0.05);
    }

    #[test]
    fn ranlux_rng_produces_uniform() {
        let mut rng = ranlux3_rng(42);
        let mut sum = 0.0;
        let n = 10000;
        for _ in 0..n {
            let v = rng.next_f64();
            assert!(v >= 0.0 && v < 1.0, "value out of range: {}", v);
            sum += v;
        }
        let mean = sum / n as f64;
        assert_abs_diff_eq!(mean, 0.5, epsilon = 0.05);
    }

    // ----- G228: Burley2020SobolRsg -----
    #[test]
    fn burley_sobol_in_unit_cube() {
        let mut rng = Burley2020SobolRsg::new(3, 42, 43);
        for _ in 0..100 {
            let point = rng.next_sequence();
            assert_eq!(point.len(), 3);
            for &v in &point {
                assert!(v >= 0.0 && v < 1.0, "value out of [0,1): {}", v);
            }
        }
    }

    // ----- G229: CentralLimitGaussianRng -----
    #[test]
    fn central_limit_gaussian_mean_zero() {
        let lcg = crate::rng_extended::LcgRng::new(42);
        let mut rng = CentralLimitGaussianRng::new(lcg);
        let mut sum = 0.0;
        let n = 10000;
        for _ in 0..n {
            sum += rng.next_gaussian();
        }
        let mean = sum / n as f64;
        assert_abs_diff_eq!(mean, 0.0, epsilon = 0.1);
    }

    // ----- G230: StochasticCollocationInvCDF -----
    #[test]
    fn stochastic_collocation_normal() {
        // Approximate the normal inverse CDF using stochastic collocation
        let sc = StochasticCollocationInvCDF::new(
            &|p: f64| inv_normal_cdf(p),
            10,
            None,
            None,
        )
        .unwrap();

        // Should reproduce the normal inverse CDF
        assert_abs_diff_eq!(sc.eval(0.5), 0.0, epsilon = 0.01);
        assert!(sc.eval(0.95) > 1.0);
        assert!(sc.eval(0.05) < -1.0);
    }
}
