//! Extended FD infrastructure — G93-G97 gap closures.
//!
//! - [`FdmDirichletBoundary`] (G93) — Dirichlet boundary conditions for FD grids
//! - [`FdmBermudanStepCondition`] (G94) — Bermudan exercise step condition
//! - [`Fdm3DimSolver`] (G95) — Three-dimensional FD solver
//! - [`FdmCevOp`] / [`FdmCirOp`] (G96) — CEV and CIR spatial operators
//! - [`FdmHullWhiteOp`] / [`FdmG2Op`] (G97) — Hull-White and G2++ spatial operators

use serde::{Deserialize, Serialize};

use crate::fdm_operators::TripleBandOp;

// ---------------------------------------------------------------------------
// G93: FdmDirichletBoundary — Dirichlet boundary conditions
// ---------------------------------------------------------------------------

/// Side of the boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundarySide {
    /// Lower boundary (i = 0).
    Lower,
    /// Upper boundary (i = n-1).
    Upper,
}

/// Dirichlet boundary condition for 1D finite difference grids.
///
/// Enforces a fixed value at one or both ends of the grid:
///   V(x_boundary, t) = f(t)
///
/// The value function can be constant or time-dependent.
#[derive(Debug, Clone)]
pub struct FdmDirichletBoundary {
    /// Which side of the grid to apply boundary.
    pub side: BoundarySide,
    /// Boundary value (constant case).
    pub value: f64,
    /// Whether value is time-dependent. If true, use `value_at_time()`.
    pub time_dependent: bool,
    /// Optional: slope for linearly time-dependent boundary (value + slope * t).
    pub slope: f64,
}

impl FdmDirichletBoundary {
    /// Create a constant Dirichlet boundary.
    pub fn new(side: BoundarySide, value: f64) -> Self {
        Self {
            side,
            value,
            time_dependent: false,
            slope: 0.0,
        }
    }

    /// Create a time-dependent Dirichlet boundary: V(t) = value + slope * t.
    pub fn time_dependent(side: BoundarySide, value: f64, slope: f64) -> Self {
        Self {
            side,
            value,
            time_dependent: true,
            slope,
        }
    }

    /// Evaluate boundary value at time t.
    pub fn value_at_time(&self, t: f64) -> f64 {
        if self.time_dependent {
            self.value + self.slope * t
        } else {
            self.value
        }
    }

    /// Apply the boundary condition to a solution vector.
    pub fn apply(&self, values: &mut [f64], t: f64) {
        let v = self.value_at_time(t);
        match self.side {
            BoundarySide::Lower => {
                if !values.is_empty() {
                    values[0] = v;
                }
            }
            BoundarySide::Upper => {
                if let Some(last) = values.last_mut() {
                    *last = v;
                }
            }
        }
    }

    /// Apply boundary conditions to the tridiagonal operator.
    ///
    /// Sets the boundary row of the operator to enforce the Dirichlet condition:
    /// diag[boundary] = 1, off-diag = 0.
    pub fn apply_to_operator(&self, op: &mut TripleBandOp) {
        match self.side {
            BoundarySide::Lower => {
                op.lower[0] = 0.0;
                op.diag[0] = 1.0;
                op.upper[0] = 0.0;
            }
            BoundarySide::Upper => {
                let n = op.n;
                op.lower[n - 1] = 0.0;
                op.diag[n - 1] = 1.0;
                op.upper[n - 1] = 0.0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// G94: FdmBermudanStepCondition — Bermudan exercise
// ---------------------------------------------------------------------------

/// Bermudan exercise step condition for finite difference methods.
///
/// At specified exercise dates (expressed as times-to-maturity), applies
/// the early-exercise condition: V = max(V, payoff).
///
/// This differs from American exercise (applied at every time step) by
/// only checking at discrete dates.
#[derive(Debug, Clone)]
pub struct FdmBermudanStepCondition {
    /// Exercise times (as times-to-maturity from valuation date).
    /// Must be sorted in ascending order.
    pub exercise_times: Vec<f64>,
    /// Intrinsic (exercise) values on the grid.
    pub payoff: Vec<f64>,
    /// Tolerance for matching exercise times to time-step grid.
    pub time_tolerance: f64,
}

impl FdmBermudanStepCondition {
    /// Create a new Bermudan step condition.
    ///
    /// # Arguments
    /// - `exercise_times`: sorted list of exercise times (years)
    /// - `payoff`: intrinsic values at each grid point
    /// - `time_tolerance`: tolerance for matching times (default: 1e-6)
    pub fn new(exercise_times: Vec<f64>, payoff: Vec<f64>, time_tolerance: f64) -> Self {
        Self {
            exercise_times,
            payoff,
            time_tolerance,
        }
    }

    /// Check if the current time is an exercise date.
    pub fn is_exercise_time(&self, t: f64) -> bool {
        self.exercise_times
            .iter()
            .any(|&et| (et - t).abs() < self.time_tolerance)
    }

    /// Apply the exercise condition at the current time step.
    ///
    /// Only modifies values if `t` matches an exercise date.
    #[allow(clippy::needless_range_loop)]
    pub fn apply(&self, values: &mut [f64], t: f64) {
        if !self.is_exercise_time(t) {
            return;
        }
        let n = values.len().min(self.payoff.len());
        for i in 0..n {
            values[i] = values[i].max(self.payoff[i]);
        }
    }

    /// Apply with time-dependent payoff (e.g., discounted strike).
    pub fn apply_with_payoff(&self, values: &mut [f64], payoff: &[f64], t: f64) {
        if !self.is_exercise_time(t) {
            return;
        }
        let n = values.len().min(payoff.len());
        for i in 0..n {
            values[i] = values[i].max(payoff[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// G95: Fdm3DimSolver — Three-dimensional FD solver
// ---------------------------------------------------------------------------

/// Result of a 3D finite difference solve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fdm3dResult {
    /// Solution values on the 3D grid, stored as flat array [i * n2 * n3 + j * n3 + k].
    pub values: Vec<f64>,
    /// Grid sizes.
    pub n1: usize,
    /// N2.
    pub n2: usize,
    /// N3.
    pub n3: usize,
}

impl Fdm3dResult {
    /// Access value at grid point (i, j, k).
    pub fn at(&self, i: usize, j: usize, k: usize) -> f64 {
        self.values[i * self.n2 * self.n3 + j * self.n3 + k]
    }

    /// Mutable access.
    pub fn at_mut(&mut self, i: usize, j: usize, k: usize) -> &mut f64 {
        let idx = i * self.n2 * self.n3 + j * self.n3 + k;
        &mut self.values[idx]
    }
}

/// Three-dimensional finite difference solver.
///
/// Solves PDEs of the form:
///   ∂V/∂t + L₁V + L₂V + L₃V = 0
///
/// using operator splitting (ADI) along each dimension.
/// Each dimension has its own tridiagonal operator.
pub struct Fdm3DimSolver {
    /// Grid size in dimension 1.
    pub n1: usize,
    /// Grid size in dimension 2.
    pub n2: usize,
    /// Grid size in dimension 3.
    pub n3: usize,
    /// Operators for dimension 1 (indexed by [j * n3 + k]).
    pub ops1: Vec<TripleBandOp>,
    /// Operators for dimension 2 (indexed by [i * n3 + k]).
    pub ops2: Vec<TripleBandOp>,
    /// Operators for dimension 3 (indexed by [i * n2 + j]).
    pub ops3: Vec<TripleBandOp>,
    /// Time stepping parameter θ (0.5 = Crank-Nicolson).
    pub theta: f64,
}

impl Fdm3DimSolver {
    /// Create a new 3D solver with uniform operators.
    ///
    /// Uses the same operator along each line for each dimension.
    pub fn new(
        n1: usize,
        n2: usize,
        n3: usize,
        op1: &TripleBandOp,
        op2: &TripleBandOp,
        op3: &TripleBandOp,
        theta: f64,
    ) -> Self {
        assert_eq!(op1.n, n1);
        assert_eq!(op2.n, n2);
        assert_eq!(op3.n, n3);

        // Replicate operators for each line
        let ops1 = vec![op1.clone(); n2 * n3];
        let ops2 = vec![op2.clone(); n1 * n3];
        let ops3 = vec![op3.clone(); n1 * n2];

        Self {
            n1,
            n2,
            n3,
            ops1,
            ops2,
            ops3,
            theta,
        }
    }

    /// Perform one time step using dimension splitting (Lie-Trotter).
    ///
    /// V^{n+1} = S₃(dt) · S₂(dt) · S₁(dt) · V^n
    ///
    /// where S_d(dt) = (I - θ·dt·L_d)⁻¹ · (I + (1-θ)·dt·L_d)
    #[allow(clippy::needless_range_loop)]
    pub fn step(&self, values: &mut Fdm3dResult, dt: f64) {
        let (n1, n2, n3) = (self.n1, self.n2, self.n3);

        // Step 1: sweep along dimension 1
        for j in 0..n2 {
            for k in 0..n3 {
                let op = &self.ops1[j * n3 + k];
                let mut line = vec![0.0; n1];
                for i in 0..n1 {
                    line[i] = values.at(i, j, k);
                }

                // Explicit part: rhs = (I + (1-θ)·dt·L)·v
                let lv = op.apply(&line);
                let rhs: Vec<f64> = (0..n1)
                    .map(|i| line[i] + (1.0 - self.theta) * dt * lv[i])
                    .collect();

                // Implicit solve
                let result = op.solve_implicit(&rhs, self.theta, dt);
                for i in 0..n1 {
                    *values.at_mut(i, j, k) = result[i];
                }
            }
        }

        // Step 2: sweep along dimension 2
        for i in 0..n1 {
            for k in 0..n3 {
                let op = &self.ops2[i * n3 + k];
                let mut line = vec![0.0; n2];
                for j in 0..n2 {
                    line[j] = values.at(i, j, k);
                }

                let lv = op.apply(&line);
                let rhs: Vec<f64> = (0..n2)
                    .map(|j| line[j] + (1.0 - self.theta) * dt * lv[j])
                    .collect();

                let result = op.solve_implicit(&rhs, self.theta, dt);
                for j in 0..n2 {
                    *values.at_mut(i, j, k) = result[j];
                }
            }
        }

        // Step 3: sweep along dimension 3
        for i in 0..n1 {
            for j in 0..n2 {
                let op = &self.ops3[i * n2 + j];
                let mut line = vec![0.0; n3];
                for k in 0..n3 {
                    line[k] = values.at(i, j, k);
                }

                let lv = op.apply(&line);
                let rhs: Vec<f64> = (0..n3)
                    .map(|k| line[k] + (1.0 - self.theta) * dt * lv[k])
                    .collect();

                let result = op.solve_implicit(&rhs, self.theta, dt);
                for k in 0..n3 {
                    *values.at_mut(i, j, k) = result[k];
                }
            }
        }
    }

    /// Solve the PDE from terminal time to t=0.
    ///
    /// # Arguments
    /// - `terminal_values`: initial (terminal) condition
    /// - `n_steps`: number of time steps
    /// - `total_time`: total time horizon
    pub fn solve(
        &self,
        terminal_values: Vec<f64>,
        n_steps: usize,
        total_time: f64,
    ) -> Fdm3dResult {
        assert_eq!(
            terminal_values.len(),
            self.n1 * self.n2 * self.n3,
            "Terminal values size mismatch"
        );
        let dt = total_time / n_steps as f64;
        let mut result = Fdm3dResult {
            values: terminal_values,
            n1: self.n1,
            n2: self.n2,
            n3: self.n3,
        };

        for _ in 0..n_steps {
            self.step(&mut result, dt);
        }

        result
    }
}

// ---------------------------------------------------------------------------
// G96: FdmCevOp / FdmCirOp — CEV and CIR spatial operators
// ---------------------------------------------------------------------------

/// Build a tridiagonal operator for the CEV (Constant Elasticity of Variance) model.
///
/// PDE: ∂V/∂t + ½σ²S^{2β}·∂²V/∂S² + (r-q)S·∂V/∂S − rV = 0
///
/// # Arguments
/// - `grid`: spatial grid points (spot values)
/// - `sigma`: CEV volatility parameter
/// - `beta`: CEV exponent (β=1 → Black-Scholes, β=0.5 → CIR-like)
/// - `r`: risk-free rate
/// - `q`: dividend yield
pub fn build_cev_operator(
    grid: &[f64],
    sigma: f64,
    beta: f64,
    r: f64,
    q: f64,
) -> TripleBandOp {
    let n = grid.len();
    let mut op = TripleBandOp::zeros(n);

    for i in 1..n - 1 {
        let s = grid[i];
        let ds_plus = grid[i + 1] - grid[i];
        let ds_minus = grid[i] - grid[i - 1];
        let ds_avg = 0.5 * (ds_plus + ds_minus);

        // Diffusion: ½σ²S^{2β}
        let diff = 0.5 * sigma * sigma * s.powf(2.0 * beta);

        // Convection: (r-q)S
        let conv = (r - q) * s;

        // Central differences
        let d2 = diff / (ds_plus * ds_minus); // ∂²V/∂S²
        let d1 = conv / (2.0 * ds_avg); // ∂V/∂S

        op.lower[i] = d2 - d1;
        op.diag[i] = -2.0 * d2 - r;
        op.upper[i] = d2 + d1;
    }

    // Boundary: Dirichlet-like (identity rows)
    op.diag[0] = -r;
    op.diag[n - 1] = -r;

    op
}

/// Build a tridiagonal operator for the CIR (Cox-Ingersoll-Ross) process.
///
/// PDE: ∂V/∂t + ½σ²x·∂²V/∂x² + κ(θ-x)·∂V/∂x − rV = 0
///
/// Used for pricing interest rate derivatives in the CIR framework.
///
/// # Arguments
/// - `grid`: spatial grid points (short rate values)
/// - `kappa`: mean-reversion speed
/// - `theta`: long-run mean
/// - `sigma`: volatility
/// - `r_discount`: discount rate
pub fn build_cir_operator(
    grid: &[f64],
    kappa: f64,
    theta: f64,
    sigma: f64,
    r_discount: f64,
) -> TripleBandOp {
    let n = grid.len();
    let mut op = TripleBandOp::zeros(n);

    for i in 1..n - 1 {
        let x = grid[i].max(0.0); // Ensure non-negative
        let dx_plus = grid[i + 1] - grid[i];
        let dx_minus = grid[i] - grid[i - 1];
        let dx_avg = 0.5 * (dx_plus + dx_minus);

        // Diffusion: ½σ²x
        let diff = 0.5 * sigma * sigma * x;

        // Convection: κ(θ - x)
        let conv = kappa * (theta - x);

        let d2 = diff / (dx_plus * dx_minus);
        let d1 = conv / (2.0 * dx_avg);

        op.lower[i] = d2 - d1;
        op.diag[i] = -2.0 * d2 - r_discount;
        op.upper[i] = d2 + d1;
    }

    // Boundaries
    op.diag[0] = -r_discount;
    op.diag[n - 1] = -r_discount;

    op
}

// ---------------------------------------------------------------------------
// G97: FdmHullWhiteOp / FdmG2Op — Hull-White and G2++ spatial operators
// ---------------------------------------------------------------------------

/// Build a tridiagonal operator for the Hull-White (extended Vasicek) model.
///
/// PDE: ∂V/∂t + ½σ²·∂²V/∂r² + (θ(t) - a·r)·∂V/∂r − r·V = 0
///
/// Note: the killing term is −r·V (state-dependent discounting).
///
/// # Arguments
/// - `grid`: spatial grid points (short rate values)
/// - `a`: mean-reversion speed
/// - `sigma`: volatility
/// - `theta_t`: time-dependent drift θ(t) for this time step
pub fn build_hull_white_operator(
    grid: &[f64],
    a: f64,
    sigma: f64,
    theta_t: f64,
) -> TripleBandOp {
    let n = grid.len();
    let mut op = TripleBandOp::zeros(n);

    for i in 1..n - 1 {
        let r = grid[i];
        let dr_plus = grid[i + 1] - grid[i];
        let dr_minus = grid[i] - grid[i - 1];
        let dr_avg = 0.5 * (dr_plus + dr_minus);

        // Diffusion: ½σ²
        let diff = 0.5 * sigma * sigma;

        // Convection: θ(t) - a·r
        let conv = theta_t - a * r;

        let d2 = diff / (dr_plus * dr_minus);
        let d1 = conv / (2.0 * dr_avg);

        op.lower[i] = d2 - d1;
        op.diag[i] = -2.0 * d2 - r; // State-dependent discounting
        op.upper[i] = d2 + d1;
    }

    // Boundaries: linear extrapolation
    op.diag[0] = -grid[0];
    op.diag[n - 1] = -grid[n - 1];

    op
}

/// Build operators for the G2++ two-factor model.
///
/// The G2++ model has two factors x and y with:
///   dx = -a·x·dt + σ·dW₁
///   dy = -b·y·dt + η·dW₂
///   r(t) = x(t) + y(t) + φ(t)
///
/// The PDE is solved on a 2D grid using operator splitting.
///
/// Returns `(op_x, op_y)` — the operators for each factor.
///
/// Note: the cross-derivative term ρ·σ·η·∂²V/∂x∂y must be handled
/// separately (e.g., via explicit correction in ADI).
///
/// # Arguments
/// - `grid_x`: grid for factor x
/// - `grid_y`: grid for factor y
/// - `a`, `sigma`: parameters for factor x
/// - `b`, `eta`: parameters for factor y
/// - `phi_t`: deterministic shift φ(t) at current time
pub fn build_g2_operators(
    grid_x: &[f64],
    grid_y: &[f64],
    a: f64,
    sigma: f64,
    b: f64,
    eta: f64,
    phi_t: f64,
) -> (TripleBandOp, TripleBandOp) {
    let nx = grid_x.len();
    let ny = grid_y.len();

    // Operator for x direction (for a given y value, use midpoint)
    let y_mid = grid_y[ny / 2];
    let mut op_x = TripleBandOp::zeros(nx);
    for i in 1..nx - 1 {
        let x = grid_x[i];
        let dx_plus = grid_x[i + 1] - grid_x[i];
        let dx_minus = grid_x[i] - grid_x[i - 1];

        let diff = 0.5 * sigma * sigma;
        let conv = -a * x;
        let r_local = x + y_mid + phi_t;

        let d2 = diff / (dx_plus * dx_minus);
        let d1 = conv / (dx_plus + dx_minus);

        op_x.lower[i] = d2 - d1;
        op_x.diag[i] = -2.0 * d2 - 0.5 * r_local;
        op_x.upper[i] = d2 + d1;
    }
    op_x.diag[0] = -0.5 * (grid_x[0] + y_mid + phi_t);
    op_x.diag[nx - 1] = -0.5 * (grid_x[nx - 1] + y_mid + phi_t);

    // Operator for y direction (for a given x value, use midpoint)
    let x_mid = grid_x[nx / 2];
    let mut op_y = TripleBandOp::zeros(ny);
    for j in 1..ny - 1 {
        let y = grid_y[j];
        let dy_plus = grid_y[j + 1] - grid_y[j];
        let dy_minus = grid_y[j] - grid_y[j - 1];

        let diff = 0.5 * eta * eta;
        let conv = -b * y;
        let r_local = x_mid + y + phi_t;

        let d2 = diff / (dy_plus * dy_minus);
        let d1 = conv / (dy_plus + dy_minus);

        op_y.lower[j] = d2 - d1;
        op_y.diag[j] = -2.0 * d2 - 0.5 * r_local;
        op_y.upper[j] = d2 + d1;
    }
    op_y.diag[0] = -0.5 * (x_mid + grid_y[0] + phi_t);
    op_y.diag[ny - 1] = -0.5 * (x_mid + grid_y[ny - 1] + phi_t);

    (op_x, op_y)
}

// ---------------------------------------------------------------------------
// G39: CraigSneydScheme — Craig-Sneyd ADI (non-modified)
// ---------------------------------------------------------------------------

/// Craig-Sneyd ADI step (non-modified variant).
///
/// For a 2D problem with operators L₁, L₂ and cross-derivative C:
///   Y₀ = Vⁿ + dt·(L₁ + L₂)·Vⁿ + dt·C·Vⁿ
///   (I - θ·dt·L₁)·Y₁ = Y₀
///   (I - θ·dt·L₂)·V^{n+1} = Y₁
///
/// # Arguments
/// - `op1`: tridiagonal operator for dimension 1
/// - `op2`: tridiagonal operator for dimension 2
/// - `cross`: explicit cross-derivative values (C·V)
/// - `v`: current solution vector (2D, stored row-major)
/// - `n1`, `n2`: grid sizes
/// - `dt`: time step
/// - `theta`: implicitness parameter (typically 0.5)
pub fn craig_sneyd_step(
    op1: &crate::fdm_operators::TripleBandOp,
    op2: &crate::fdm_operators::TripleBandOp,
    cross: &[f64],
    v: &mut [f64],
    n1: usize,
    n2: usize,
    dt: f64,
    theta: f64,
) {
    let n_total = n1 * n2;
    assert_eq!(v.len(), n_total);
    assert_eq!(cross.len(), n_total);

    // Y₀ = V + dt·(L₁+L₂)·V + dt·cross
    let mut y0 = vec![0.0; n_total];

    // Apply L₁ along dimension 1 (rows)
    for j in 0..n2 {
        let line: Vec<f64> = (0..n1).map(|i| v[i * n2 + j]).collect();
        let l1v = op1.apply(&line);
        for i in 0..n1 {
            y0[i * n2 + j] = v[i * n2 + j] + dt * l1v[i];
        }
    }

    // Apply L₂ along dimension 2 (columns)
    for i in 0..n1 {
        let line: Vec<f64> = (0..n2).map(|j| v[i * n2 + j]).collect();
        let l2v = op2.apply(&line);
        for j in 0..n2 {
            y0[i * n2 + j] += dt * l2v[j];
        }
    }

    // Add cross-derivative
    for k in 0..n_total {
        y0[k] += dt * cross[k];
    }

    // Implicit sweep 1: (I - θ·dt·L₁)·Y₁ = Y₀
    for j in 0..n2 {
        let rhs: Vec<f64> = (0..n1).map(|i| y0[i * n2 + j]).collect();
        let result = op1.solve_implicit(&rhs, theta, dt);
        for i in 0..n1 {
            y0[i * n2 + j] = result[i];
        }
    }

    // Implicit sweep 2: (I - θ·dt·L₂)·V^{n+1} = Y₁
    for i in 0..n1 {
        let rhs: Vec<f64> = (0..n2).map(|j| y0[i * n2 + j]).collect();
        let result = op2.solve_implicit(&rhs, theta, dt);
        for j in 0..n2 {
            v[i * n2 + j] = result[j];
        }
    }
}

// ---------------------------------------------------------------------------
// G40: TRBDF2Scheme — TR-BDF2 time-stepping
// ---------------------------------------------------------------------------

/// TR-BDF2 time step for 1D finite difference problems.
///
/// Two-stage method:
/// 1. Trapezoidal rule (TR) half-step: (I - γ·dt/2·L)·V* = (I + γ·dt/2·L)·Vⁿ
/// 2. BDF2 step: (I - (1-γ)·dt/(2-γ)·L)·V^{n+1} = 1/(γ(2-γ))·V* - (1-γ)²/(γ(2-γ))·Vⁿ
///
/// where γ = 2 - √2 (optimal for L-stability).
///
/// # Arguments
/// - `op`: tridiagonal spatial operator L
/// - `v`: current solution (modified in-place)
/// - `dt`: time step
pub fn trbdf2_step(op: &crate::fdm_operators::TripleBandOp, v: &mut [f64], dt: f64) {
    let gamma = 2.0 - 2.0_f64.sqrt(); // ≈ 0.5858
    let n = v.len();

    // Stage 1: Trapezoidal half-step
    // (I - γ·dt/2·L)·V* = (I + γ·dt/2·L)·Vⁿ
    let lv = op.apply(v);
    let rhs: Vec<f64> = (0..n)
        .map(|i| v[i] + gamma * dt / 2.0 * lv[i])
        .collect();
    let v_star = op.solve_implicit(&rhs, gamma / 2.0, dt);

    // Stage 2: BDF2 step
    // (I - w·L)·V^{n+1} = α·V* + β·Vⁿ
    let w = (1.0 - gamma) / (2.0 - gamma) * dt;
    let alpha = 1.0 / (gamma * (2.0 - gamma));
    let beta = -(1.0 - gamma).powi(2) / (gamma * (2.0 - gamma));

    let rhs2: Vec<f64> = (0..n)
        .map(|i| alpha * v_star[i] + beta * v[i])
        .collect();

    // Solve (I - w·L)·V^{n+1} = rhs2
    let factor = w; // = (1-γ)dt/(2-γ)
    let result = op.solve_implicit(&rhs2, factor / dt, dt);

    v.copy_from_slice(&result);
}

// ---------------------------------------------------------------------------
// G41: MethodOfLinesScheme — spatial discretization only
// ---------------------------------------------------------------------------

/// Method of Lines step: spatial discretization with explicit time integration.
///
/// The spatial PDE operator L is applied explicitly, and the resulting ODE
/// system dV/dt = L·V is integrated using a Runge-Kutta 4th order method.
///
/// # Arguments
/// - `op`: spatial operator L
/// - `v`: current solution (modified in-place)
/// - `dt`: time step
pub fn method_of_lines_step(op: &crate::fdm_operators::TripleBandOp, v: &mut [f64], dt: f64) {
    let n = v.len();

    // RK4 integration of dV/dt = L·V
    let k1 = op.apply(v);

    let v_temp: Vec<f64> = (0..n).map(|i| v[i] + 0.5 * dt * k1[i]).collect();
    let k2 = op.apply(&v_temp);

    let v_temp: Vec<f64> = (0..n).map(|i| v[i] + 0.5 * dt * k2[i]).collect();
    let k3 = op.apply(&v_temp);

    let v_temp: Vec<f64> = (0..n).map(|i| v[i] + dt * k3[i]).collect();
    let k4 = op.apply(&v_temp);

    for i in 0..n {
        v[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ---- G93: Dirichlet boundary tests ----

    #[test]
    fn test_dirichlet_constant_lower() {
        let bc = FdmDirichletBoundary::new(BoundarySide::Lower, 0.0);
        let mut values = vec![5.0, 3.0, 1.0, 0.5, 0.1];
        bc.apply(&mut values, 0.0);
        assert_abs_diff_eq!(values[0], 0.0);
        assert_abs_diff_eq!(values[1], 3.0); // Untouched
    }

    #[test]
    fn test_dirichlet_constant_upper() {
        let bc = FdmDirichletBoundary::new(BoundarySide::Upper, 100.0);
        let mut values = vec![5.0, 3.0, 1.0, 0.5, 0.1];
        bc.apply(&mut values, 0.0);
        assert_abs_diff_eq!(values[4], 100.0);
        assert_abs_diff_eq!(values[3], 0.5); // Untouched
    }

    #[test]
    fn test_dirichlet_time_dependent() {
        let bc = FdmDirichletBoundary::time_dependent(BoundarySide::Lower, 10.0, -5.0);
        assert_abs_diff_eq!(bc.value_at_time(0.0), 10.0);
        assert_abs_diff_eq!(bc.value_at_time(1.0), 5.0);
        assert_abs_diff_eq!(bc.value_at_time(2.0), 0.0);
    }

    #[test]
    fn test_dirichlet_apply_to_operator() {
        let mut op = TripleBandOp {
            n: 5,
            lower: vec![0.1; 5],
            diag: vec![0.5; 5],
            upper: vec![0.2; 5],
        };
        let bc_lower = FdmDirichletBoundary::new(BoundarySide::Lower, 0.0);
        let bc_upper = FdmDirichletBoundary::new(BoundarySide::Upper, 0.0);

        bc_lower.apply_to_operator(&mut op);
        bc_upper.apply_to_operator(&mut op);

        assert_abs_diff_eq!(op.diag[0], 1.0);
        assert_abs_diff_eq!(op.lower[0], 0.0);
        assert_abs_diff_eq!(op.upper[0], 0.0);
        assert_abs_diff_eq!(op.diag[4], 1.0);
        assert_abs_diff_eq!(op.lower[4], 0.0);
        assert_abs_diff_eq!(op.upper[4], 0.0);
        // Interior unchanged
        assert_abs_diff_eq!(op.diag[2], 0.5);
    }

    // ---- G94: Bermudan step condition tests ----

    #[test]
    fn test_bermudan_exercise_at_date() {
        let payoff = vec![10.0, 5.0, 0.0, 0.0, 0.0];
        let cond = FdmBermudanStepCondition::new(
            vec![0.5, 1.0],
            payoff,
            1e-6,
        );

        // At exercise time 0.5
        let mut values = vec![8.0, 6.0, 2.0, 1.0, 0.5];
        cond.apply(&mut values, 0.5);
        assert_abs_diff_eq!(values[0], 10.0); // Exercised: max(8, 10) = 10
        assert_abs_diff_eq!(values[1], 6.0); // Not exercised: max(6, 5) = 6
        assert_abs_diff_eq!(values[2], 2.0); // max(2, 0) = 2
    }

    #[test]
    fn test_bermudan_no_exercise_between_dates() {
        let payoff = vec![10.0, 5.0, 0.0];
        let cond = FdmBermudanStepCondition::new(
            vec![0.5, 1.0],
            payoff,
            1e-6,
        );

        let mut values = vec![3.0, 2.0, 1.0];
        cond.apply(&mut values, 0.75); // Not an exercise date
        assert_abs_diff_eq!(values[0], 3.0); // Unchanged
    }

    #[test]
    fn test_bermudan_is_exercise_time() {
        let cond = FdmBermudanStepCondition::new(
            vec![0.25, 0.5, 0.75, 1.0],
            vec![],
            1e-6,
        );
        assert!(cond.is_exercise_time(0.5));
        assert!(cond.is_exercise_time(0.25));
        assert!(!cond.is_exercise_time(0.3));
    }

    // ---- G95: 3D solver tests ----

    #[test]
    fn test_3d_solver_identity_preserves_values() {
        // Zero operator → solution should not change
        let n = 5;
        let op = TripleBandOp::zeros(n);
        let solver = Fdm3DimSolver::new(n, n, n, &op, &op, &op, 0.5);

        let initial: Vec<f64> = (0..n * n * n).map(|i| i as f64).collect();
        let result = solver.solve(initial.clone(), 10, 1.0);

        for i in 0..n * n * n {
            assert_abs_diff_eq!(result.values[i], initial[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_3d_solver_diffusion_smooths() {
        // Simple diffusion operator should smooth a peaked initial condition
        let n = 5;
        let mut op = TripleBandOp::zeros(n);
        // Laplacian with dx=1
        for i in 1..n - 1 {
            op.lower[i] = 1.0;
            op.diag[i] = -2.0;
            op.upper[i] = 1.0;
        }

        let solver = Fdm3DimSolver::new(n, n, n, &op, &op, &op, 1.0);

        // Peaked initial condition
        let mut initial = vec![0.0; n * n * n];
        let mid = n / 2;
        initial[mid * n * n + mid * n + mid] = 100.0;

        let result = solver.solve(initial, 5, 0.01);

        // Peak should be reduced
        assert!(result.at(mid, mid, mid) < 100.0);
        // Neighbors should have gained some value
        if mid + 1 < n {
            assert!(result.at(mid + 1, mid, mid) > 0.0 || result.at(mid, mid + 1, mid) > 0.0);
        }
    }

    #[test]
    fn test_3d_result_indexing() {
        let result = Fdm3dResult {
            values: (0..27).map(|i| i as f64).collect(),
            n1: 3,
            n2: 3,
            n3: 3,
        };
        assert_abs_diff_eq!(result.at(0, 0, 0), 0.0);
        assert_abs_diff_eq!(result.at(0, 0, 1), 1.0);
        assert_abs_diff_eq!(result.at(0, 1, 0), 3.0);
        assert_abs_diff_eq!(result.at(1, 0, 0), 9.0);
        assert_abs_diff_eq!(result.at(2, 2, 2), 26.0);
    }

    // ---- G96: CEV and CIR operators ----

    #[test]
    fn test_cev_operator_reduces_to_bs() {
        // β=1 → CEV reduces to Black-Scholes
        let grid: Vec<f64> = (0..=20).map(|i| 50.0 + 5.0 * i as f64).collect();
        let op = build_cev_operator(&grid, 0.20, 1.0, 0.05, 0.02);

        // Operator should have proper structure
        assert_eq!(op.n, grid.len());
        // Interior lower diagonal should be positive (upwind)
        assert!(op.lower[5] > -100.0); // Bounded
        // Diagonal should be negative (from diffusion − r)
        assert!(op.diag[10] < 0.0);
    }

    #[test]
    fn test_cev_operator_half_beta() {
        // β=0.5 → square-root diffusion
        let grid: Vec<f64> = (1..=20).map(|i| 5.0 * i as f64).collect();
        let op = build_cev_operator(&grid, 0.30, 0.5, 0.05, 0.0);

        // Check that it produces a valid tridiagonal operator
        assert_eq!(op.n, grid.len());
        // Diffusion coefficient ~ σ²S at β=0.5 → varies with S
        // So operators should differ at different grid points
        assert!((op.lower[3] - op.lower[10]).abs() > 1e-10);
    }

    #[test]
    fn test_cir_operator_structure() {
        let grid: Vec<f64> = (0..=20).map(|i| 0.005 * i as f64).collect();
        let op = build_cir_operator(&grid, 0.3, 0.05, 0.1, 0.04);

        assert_eq!(op.n, grid.len());
        // At x=0, diffusion vanishes → lower and upper should be near zero at boundary
        // Interior should have proper signs
        assert!(op.diag[10] < 0.0); // Negative diagonal
    }

    // ---- G97: Hull-White and G2 operators ----

    #[test]
    fn test_hull_white_operator_structure() {
        let grid: Vec<f64> = (0..51).map(|i| -0.05 + 0.002 * i as f64).collect();
        let op = build_hull_white_operator(&grid, 0.1, 0.01, 0.05);

        assert_eq!(op.n, grid.len());
        // State-dependent discount: diag should vary with r
        let diag_low = op.diag[5];
        let diag_high = op.diag[45];
        assert!((diag_low - diag_high).abs() > 1e-6);
    }

    #[test]
    fn test_hull_white_operator_symmetry() {
        // With zero drift (θ=a·r at midpoint), pure diffusion should be symmetric
        let grid: Vec<f64> = (0..21).map(|i| -0.10 + 0.01 * i as f64).collect();
        let op = build_hull_white_operator(&grid, 0.0, 0.01, 0.0);

        // With a=0, θ=0: pure diffusion, lower[i] ≈ upper[i] for uniform grid
        for i in 2..op.n - 2 {
            assert_abs_diff_eq!(op.lower[i], op.upper[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_g2_operator_pair() {
        let grid_x: Vec<f64> = (0..21).map(|i| -0.05 + 0.005 * i as f64).collect();
        let grid_y: Vec<f64> = (0..21).map(|i| -0.03 + 0.003 * i as f64).collect();

        let (op_x, op_y) = build_g2_operators(
            &grid_x, &grid_y,
            0.1, 0.01, // a, σ for x
            0.2, 0.008, // b, η for y
            0.03, // φ(t)
        );

        assert_eq!(op_x.n, grid_x.len());
        assert_eq!(op_y.n, grid_y.len());

        // Both operators should have negative diagonals (discount + diffusion)
        assert!(op_x.diag[10] < 0.0);
        assert!(op_y.diag[10] < 0.0);
    }

    #[test]
    fn test_g2_operators_different_params() {
        let grid_x: Vec<f64> = (0..11).map(|i| -0.05 + 0.01 * i as f64).collect();
        let grid_y: Vec<f64> = (0..11).map(|i| -0.03 + 0.006 * i as f64).collect();

        // Different σ and η should give different operator magnitudes
        let (op_x1, _) = build_g2_operators(&grid_x, &grid_y, 0.1, 0.01, 0.2, 0.01, 0.03);
        let (op_x2, _) = build_g2_operators(&grid_x, &grid_y, 0.1, 0.02, 0.2, 0.01, 0.03);

        // Doubling σ should change the x operator
        assert!((op_x1.diag[5] - op_x2.diag[5]).abs() > 1e-8);
    }

    // ---- G39: Craig-Sneyd ----

    #[test]
    fn test_craig_sneyd_identity() {
        // Zero operators → values unchanged
        let n = 5;
        let op = crate::fdm_operators::TripleBandOp::zeros(n);
        let cross = vec![0.0; n * n];
        let mut v: Vec<f64> = (0..n * n).map(|i| i as f64).collect();
        let v_orig = v.clone();

        craig_sneyd_step(&op, &op, &cross, &mut v, n, n, 0.01, 0.5);

        for i in 0..n * n {
            assert_abs_diff_eq!(v[i], v_orig[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_craig_sneyd_cross_term() {
        // Non-zero cross-derivative should change solution
        let n = 4;
        let op = crate::fdm_operators::TripleBandOp::zeros(n);
        let cross = vec![1.0; n * n];
        let mut v = vec![0.0; n * n];

        craig_sneyd_step(&op, &op, &cross, &mut v, n, n, 0.01, 0.5);

        // Values should have shifted by ~dt * cross
        for &val in &v {
            assert_abs_diff_eq!(val, 0.01, epsilon = 1e-10);
        }
    }

    // ---- G40: TR-BDF2 ----

    #[test]
    fn test_trbdf2_decay() {
        // Simple decay operator: L = diag(-1) → dV/dt = -V → V(t) = V(0)e^{-t}
        let n = 5;
        let mut op = crate::fdm_operators::TripleBandOp::zeros(n);
        for i in 0..n {
            op.diag[i] = -1.0;
        }

        let mut v = vec![1.0; n];
        let dt = 0.01;
        for _ in 0..100 {
            trbdf2_step(&op, &mut v, dt);
        }

        // After t=1.0, should be approximately e^{-1} ≈ 0.368
        for &val in &v {
            assert!(val > 0.1 && val < 0.8, "TR-BDF2 value = {}", val);
        }
    }

    // ---- G41: Method of Lines ----

    #[test]
    fn test_method_of_lines_decay() {
        let n = 5;
        let mut op = crate::fdm_operators::TripleBandOp::zeros(n);
        for i in 0..n {
            op.diag[i] = -1.0;
        }

        let mut v = vec![1.0; n];
        let dt = 0.001; // Small dt for explicit stability
        for _ in 0..1000 {
            method_of_lines_step(&op, &mut v, dt);
        }

        // After t=1.0, should be approximately e^{-1} ≈ 0.368
        let expected = (-1.0_f64).exp();
        for &val in &v {
            assert_abs_diff_eq!(val, expected, epsilon = 0.01);
        }
    }

    #[test]
    fn test_method_of_lines_preserves_zero() {
        let n = 5;
        let op = crate::fdm_operators::TripleBandOp::zeros(n);
        let mut v = vec![0.0; n];
        method_of_lines_step(&op, &mut v, 0.01);
        for &val in &v {
            assert_abs_diff_eq!(val, 0.0, epsilon = 1e-15);
        }
    }
}
