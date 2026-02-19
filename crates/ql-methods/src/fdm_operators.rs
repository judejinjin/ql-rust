#![allow(clippy::too_many_arguments)]
//! Finite Difference operators and time-stepping schemes.
//!
//! Provides:
//! - `TripleBandLinearOp` — tridiagonal operator for 1D PDEs
//! - `build_bs_operator` — Black-Scholes 1D PDE operator
//! - `build_heston_ops` — Heston 2D PDE operators (spot and variance directions)
//! - `crank_nicolson_step` — Crank-Nicolson (θ=0.5) time step
//! - `implicit_step` — Fully implicit (θ=1) time step
//! - `douglas_adi_step` — Douglas ADI for 2D problems

/// Tridiagonal linear operator for 1D PDEs.
///
/// Represents L·u where L is a tridiagonal matrix with
/// bands lower[i], diag[i], upper[i].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TripleBandOp {
    pub n: usize,
    pub lower: Vec<f64>,
    pub diag: Vec<f64>,
    pub upper: Vec<f64>,
}

impl TripleBandOp {
    /// Create a zero operator.
    pub fn zeros(n: usize) -> Self {
        Self {
            n,
            lower: vec![0.0; n],
            diag: vec![0.0; n],
            upper: vec![0.0; n],
        }
    }

    /// Apply: y = L · x (tridiagonal multiply).
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut y = vec![0.0; n];
        y[0] = self.diag[0] * x[0] + self.upper[0] * x[1];
        for i in 1..n - 1 {
            y[i] = self.lower[i] * x[i - 1] + self.diag[i] * x[i] + self.upper[i] * x[i + 1];
        }
        y[n - 1] = self.lower[n - 1] * x[n - 2] + self.diag[n - 1] * x[n - 1];
        y
    }

    /// Solve (I − θ·dt·L)·x_new = rhs using Thomas algorithm.
    pub fn solve_implicit(&self, rhs: &[f64], theta: f64, dt: f64) -> Vec<f64> {
        let n = self.n;
        // Build (I − θ·dt·L)
        let mut a = vec![0.0; n]; // lower
        let mut b = vec![0.0; n]; // diagonal
        let mut c = vec![0.0; n]; // upper

        for i in 0..n {
            a[i] = -theta * dt * self.lower[i];
            b[i] = 1.0 - theta * dt * self.diag[i];
            c[i] = -theta * dt * self.upper[i];
        }

        // Thomas algorithm
        thomas_solve(&a, &b, &c, rhs)
    }

    /// Scale the operator by a constant.
    pub fn scale(&mut self, factor: f64) {
        for i in 0..self.n {
            self.lower[i] *= factor;
            self.diag[i] *= factor;
            self.upper[i] *= factor;
        }
    }

    /// Add another operator: self += other.
    pub fn add_assign(&mut self, other: &TripleBandOp) {
        assert_eq!(self.n, other.n);
        for i in 0..self.n {
            self.lower[i] += other.lower[i];
            self.diag[i] += other.diag[i];
            self.upper[i] += other.upper[i];
        }
    }
}

/// Thomas algorithm for tridiagonal systems.
fn thomas_solve(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
    let n = d.len();
    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];

    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];

    for i in 1..n {
        let m = b[i] - a[i] * c_prime[i - 1];
        c_prime[i] = c[i] / m;
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / m;
    }

    let mut x = vec![0.0; n];
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
    x
}

/// Build 1D Black-Scholes operator on a log-spot grid.
///
/// The BS PDE in log-spot x = ln(S):
///   ∂V/∂t + (r − q − σ²/2) ∂V/∂x + σ²/2 ∂²V/∂x² − rV = 0
///
/// Returns the spatial operator L such that ∂V/∂t = −L·V.
pub fn build_bs_operator(
    grid: &[f64],
    r: f64,
    q: f64,
    vol: f64,
) -> TripleBandOp {
    let n = grid.len();
    let mut op = TripleBandOp::zeros(n);
    let drift = r - q - 0.5 * vol * vol;
    let diffusion = 0.5 * vol * vol;

    for i in 1..n - 1 {
        let dx_plus = grid[i + 1] - grid[i];
        let dx_minus = grid[i] - grid[i - 1];
        let dx_mid = 0.5 * (dx_plus + dx_minus);

        // Second derivative: (V[i+1] − 2V[i] + V[i-1]) / dx²
        op.upper[i] = diffusion / (dx_plus * dx_mid);
        op.lower[i] = diffusion / (dx_minus * dx_mid);
        op.diag[i] = -op.upper[i] - op.lower[i] - r;

        // First derivative (central): (V[i+1] − V[i-1]) / (2dx)
        op.upper[i] += drift / (dx_plus + dx_minus);
        op.lower[i] -= drift / (dx_plus + dx_minus);
    }

    // Boundary: V(x_min) → 0 for calls, linear extrapolation
    op.diag[0] = -r;
    op.diag[n - 1] = -r;

    op
}

/// Crank-Nicolson step: advance solution from time t to t − dt (backward in time).
///
/// (I − θ·dt·L) V(t−dt) = (I + (1−θ)·dt·L) V(t)
///
/// With θ = 0.5 for Crank-Nicolson.
pub fn crank_nicolson_step(
    op: &TripleBandOp,
    v: &[f64],
    dt: f64,
    theta: f64,
) -> Vec<f64> {
    let n = v.len();
    // Explicit part: rhs = (I + (1−θ)·dt·L) V
    let lv = op.apply(v);
    let rhs: Vec<f64> = (0..n).map(|i| v[i] + (1.0 - theta) * dt * lv[i]).collect();

    // Implicit solve: (I − θ·dt·L) V_new = rhs
    op.solve_implicit(&rhs, theta, dt)
}

/// Fully implicit step (θ=1).
pub fn implicit_step(op: &TripleBandOp, v: &[f64], dt: f64) -> Vec<f64> {
    op.solve_implicit(v, 1.0, dt)
}

// ─────────────────────────────────────────────────────────────
// 2D Heston FD operators
// ─────────────────────────────────────────────────────────────

/// Heston 2D operator components.
///
/// The Heston PDE in (x = ln S, v) coordinates:
///   ∂V/∂t + (r − q − v/2) ∂V/∂x + v/2 ∂²V/∂x²
///         + κ(θ−v) ∂V/∂v + σ²v/2 ∂²V/∂v²
///         + ρσv ∂²V/∂x∂v − rV = 0
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Heston2dOps {
    /// Operator in x-direction (for each v slice).
    pub x_ops: Vec<TripleBandOp>,
    /// Operator in v-direction (for each x slice).
    pub v_ops: Vec<TripleBandOp>,
    /// Grid sizes.
    pub nx: usize,
    pub nv: usize,
    /// Correlation mixing coefficients for explicit cross derivative.
    pub cross_coeffs: Vec<f64>,
}

/// Build Heston 2D operators on (x, v) grid.
///
/// Returns operators split by direction for ADI time-stepping.
pub fn build_heston_ops(
    x_grid: &[f64],
    v_grid: &[f64],
    r: f64,
    q: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
) -> Heston2dOps {
    let nx = x_grid.len();
    let nv = v_grid.len();

    // Build x-direction operators for each v level
    let mut x_ops = Vec::with_capacity(nv);
    for &vj in v_grid {
        let v = vj.max(0.0);
        let vol = v.sqrt();
        x_ops.push(build_bs_operator(x_grid, r, q, vol));
    }

    // Build v-direction operators for each x level
    let mut v_ops = Vec::with_capacity(nx);
    for _i in 0..nx {
        let mut op = TripleBandOp::zeros(nv);
        for j in 1..nv - 1 {
            let v = v_grid[j].max(0.0);
            let dv_plus = v_grid[j + 1] - v_grid[j];
            let dv_minus = v_grid[j] - v_grid[j - 1];
            let dv_mid = 0.5 * (dv_plus + dv_minus);

            let drift = kappa * (theta - v);
            let diffusion = 0.5 * sigma * sigma * v;

            op.upper[j] = diffusion / (dv_plus * dv_mid) + drift.max(0.0) / (dv_plus + dv_minus);
            op.lower[j] = diffusion / (dv_minus * dv_mid) - drift.min(0.0) / (dv_plus + dv_minus);
            op.diag[j] = -op.upper[j] - op.lower[j];
        }
        // Boundary at v=0: drift dominates (Feller condition)
        op.diag[0] = 0.0;
        if nv > 1 {
            let dv = v_grid[1] - v_grid[0];
            op.upper[0] = kappa * theta / dv;
            op.diag[0] = -op.upper[0];
        }
        // Boundary at v_max: zero gamma
        op.diag[nv - 1] = 0.0;

        v_ops.push(op);
    }

    // Cross derivative coefficients: ρσv
    let cross_coeffs: Vec<f64> = (0..nv).map(|j| rho * sigma * v_grid[j].max(0.0)).collect();

    Heston2dOps {
        x_ops,
        v_ops,
        nx,
        nv,
        cross_coeffs,
    }
}

/// Douglas ADI step for 2D Heston problem.
///
/// Splitting: V^(n+1) = V^n + dt [L_x V^n + L_v V^n + L_cross V^n]
/// with implicit x-sweep then implicit v-sweep.
///
/// `values` is stored in row-major order: values[i * nv + j] for x[i], v[j].
pub fn douglas_adi_step(
    ops: &Heston2dOps,
    values: &mut [f64],
    dt: f64,
    theta: f64,
) {
    let nx = ops.nx;
    let nv = ops.nv;

    // Step 1: Explicit update
    let mut rhs = vec![0.0; nx * nv];
    for i in 0..nx {
        for j in 0..nv {
            let idx = i * nv + j;
            rhs[idx] = values[idx];
        }
    }

    // Add explicit operator contribution
    // x-direction
    for j in 0..nv {
        let row: Vec<f64> = (0..nx).map(|i| values[i * nv + j]).collect();
        let lx = ops.x_ops[j].apply(&row);
        for i in 0..nx {
            rhs[i * nv + j] += (1.0 - theta) * dt * lx[i];
        }
    }
    // v-direction
    for i in 0..nx {
        let col: Vec<f64> = (0..nv).map(|j| values[i * nv + j]).collect();
        let lv = ops.v_ops[i].apply(&col);
        for j in 0..nv {
            rhs[i * nv + j] += (1.0 - theta) * dt * lv[j];
        }
    }

    // Step 2: Implicit x-sweep (for each v level)
    let mut intermediate = rhs.clone();
    for j in 0..nv {
        let rhs_row: Vec<f64> = (0..nx).map(|i| rhs[i * nv + j]).collect();
        let solved = ops.x_ops[j].solve_implicit(&rhs_row, theta, dt);
        for i in 0..nx {
            intermediate[i * nv + j] = solved[i];
        }
    }

    // Step 3: Implicit v-sweep correction
    for i in 0..nx {
        // Correction: solve for the v-direction residual
        let v_old: Vec<f64> = (0..nv).map(|j| values[i * nv + j]).collect();
        let v_int: Vec<f64> = (0..nv).map(|j| intermediate[i * nv + j]).collect();
        let lv_old = ops.v_ops[i].apply(&v_old);

        // rhs for v-correction = v_int − θ·dt·L_v·v_old
        let rhs_col: Vec<f64> = (0..nv)
            .map(|j| v_int[j] - theta * dt * lv_old[j])
            .collect();
        let solved = ops.v_ops[i].solve_implicit(&rhs_col, theta, dt);
        for j in 0..nv {
            values[i * nv + j] = solved[j];
        }
    }
}

/// Apply American exercise constraint: V = max(V, payoff).
pub fn apply_american_condition(values: &mut [f64], payoff: &[f64]) {
    for i in 0..values.len() {
        values[i] = values[i].max(payoff[i]);
    }
}

// ─────────────────────────────────────────────────────────────
// 1D FD solver
// ─────────────────────────────────────────────────────────────

/// Result of a 1D FD solve.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Fd1dResult {
    /// Grid locations.
    pub grid: Vec<f64>,
    /// Solution values at grid points.
    pub values: Vec<f64>,
    /// Option price (interpolated at spot).
    pub price: f64,
    /// Delta (∂V/∂S).
    pub delta: f64,
    /// Gamma (∂²V/∂S²).
    pub gamma: f64,
    /// Theta (∂V/∂t), per year.
    pub theta: f64,
}

/// Solve 1D Black-Scholes PDE via Crank-Nicolson.
///
/// Prices a European or American option.
pub fn fd_1d_bs_solve(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    vol: f64,
    expiry: f64,
    is_call: bool,
    is_american: bool,
    n_grid: usize,
    n_time: usize,
) -> Fd1dResult {
    let x0 = spot.ln();
    let std_dev = vol * expiry.sqrt();
    let x_lo = x0 - 5.0 * std_dev;
    let x_hi = x0 + 5.0 * std_dev;
    let dx = (x_hi - x_lo) / (n_grid - 1) as f64;
    let dt = expiry / n_time as f64;

    let grid: Vec<f64> = (0..n_grid).map(|i| x_lo + i as f64 * dx).collect();

    // Terminal payoff
    let mut values: Vec<f64> = grid
        .iter()
        .map(|&x| {
            let s = x.exp();
            if is_call {
                (s - strike).max(0.0)
            } else {
                (strike - s).max(0.0)
            }
        })
        .collect();

    // Payoff for American exercise
    let payoff = values.clone();

    let op = build_bs_operator(&grid, r, q, vol);

    // Time-step backward
    let theta_cn = 0.5; // Crank-Nicolson
    for _ in 0..n_time {
        values = crank_nicolson_step(&op, &values, dt, theta_cn);

        if is_american {
            apply_american_condition(&mut values, &payoff);
        }

        // Boundary conditions
        values[0] = if is_call {
            0.0
        } else {
            (strike * (-r * expiry).exp() - grid[0].exp() * (-q * expiry).exp()).max(0.0)
        };
        let n = n_grid - 1;
        values[n] = if is_call {
            (grid[n].exp() * (-q * expiry).exp() - strike * (-r * expiry).exp()).max(0.0)
        } else {
            0.0
        };
    }

    // Interpolate at x0
    let idx = grid
        .iter()
        .position(|&x| x >= x0)
        .unwrap_or(n_grid - 1)
        .max(1)
        .min(n_grid - 2);

    let x_m = grid[idx - 1];
    let x_0 = grid[idx];
    let x_p = grid[idx + 1];
    let v_m = values[idx - 1];
    let v_0 = values[idx];
    let v_p = values[idx + 1];

    // Quadratic interpolation in x for V, then convert to S derivatives
    let t = (x0 - x_0) / (x_p - x_0);
    let price = v_0 + t * (v_p - v_m) / 2.0
        + t * t * (v_p - 2.0 * v_0 + v_m) / 2.0;

    // dV/dx via central difference
    let dv_dx = (v_p - v_m) / (x_p - x_m);
    // d²V/dx² via central difference
    let d2v_dx2 = (v_p - 2.0 * v_0 + v_m) / ((x_p - x_0) * (x_0 - x_m));

    // Convert to S-derivatives: delta = (1/S) dV/dx, gamma = (1/S²)(d²V/dx² − dV/dx)
    let delta = dv_dx / spot;
    let gamma = (d2v_dx2 - dv_dx) / (spot * spot);
    let theta_val = -(r * price - (r - q) * spot * delta - 0.5 * vol * vol * spot * spot * gamma);

    Fd1dResult {
        grid,
        values,
        price,
        delta,
        gamma,
        theta: theta_val,
    }
}

// ─────────────────────────────────────────────────────────────
// 2D Heston FD solver
// ─────────────────────────────────────────────────────────────

/// Result of 2D Heston FD solve.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HestonFdResult {
    /// Option price.
    pub price: f64,
    /// Delta.
    pub delta: f64,
    /// Gamma.
    pub gamma: f64,
    /// Vega (∂V/∂v0).
    pub vega: f64,
}

/// Solve Heston PDE via Douglas ADI on (x, v) grid.
pub fn fd_heston_solve(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    v0: f64,
    kappa: f64,
    theta_h: f64,
    sigma: f64,
    rho: f64,
    expiry: f64,
    is_call: bool,
    is_american: bool,
    nx: usize,
    nv: usize,
    n_time: usize,
) -> HestonFdResult {
    let x0 = spot.ln();
    let vol_approx = v0.sqrt();
    let std_dev = vol_approx * expiry.sqrt();
    let x_lo = x0 - 5.0 * std_dev;
    let x_hi = x0 + 5.0 * std_dev;

    let v_max = (5.0 * theta_h).max(3.0 * v0).max(1.0);

    let x_grid: Vec<f64> = (0..nx)
        .map(|i| x_lo + i as f64 * (x_hi - x_lo) / (nx - 1) as f64)
        .collect();
    let v_grid: Vec<f64> = (0..nv)
        .map(|j| j as f64 * v_max / (nv - 1) as f64)
        .collect();

    let dt = expiry / n_time as f64;

    // Terminal payoff
    let mut values = vec![0.0; nx * nv];
    let mut payoff = vec![0.0; nx * nv];
    for i in 0..nx {
        let s = x_grid[i].exp();
        let pf = if is_call {
            (s - strike).max(0.0)
        } else {
            (strike - s).max(0.0)
        };
        for j in 0..nv {
            values[i * nv + j] = pf;
            payoff[i * nv + j] = pf;
        }
    }

    let ops = build_heston_ops(&x_grid, &v_grid, r, q, kappa, theta_h, sigma, rho);

    // Time-step backward
    for _ in 0..n_time {
        douglas_adi_step(&ops, &mut values, dt, 0.5);

        if is_american {
            apply_american_condition(&mut values, &payoff);
        }

        // Boundary conditions at x boundaries
        for j in 0..nv {
            values[j] = if is_call { 0.0 } else { (strike - x_grid[0].exp()).max(0.0) };
            values[(nx - 1) * nv + j] = if is_call {
                (x_grid[nx - 1].exp() - strike * (-r * expiry).exp()).max(0.0)
            } else {
                0.0
            };
        }
    }

    // Interpolate at (x0, v0)
    let xi = x_grid
        .iter()
        .position(|&x| x >= x0)
        .unwrap_or(nx - 1)
        .max(1)
        .min(nx - 2);
    let vj = v_grid
        .iter()
        .position(|&v| v >= v0)
        .unwrap_or(nv - 1)
        .max(1)
        .min(nv - 2);

    // Bilinear interpolation
    let tx = (x0 - x_grid[xi - 1]) / (x_grid[xi] - x_grid[xi - 1]);
    let tv = (v0 - v_grid[vj - 1]) / (v_grid[vj] - v_grid[vj - 1]);

    let v00 = values[(xi - 1) * nv + (vj - 1)];
    let v10 = values[xi * nv + (vj - 1)];
    let v01 = values[(xi - 1) * nv + vj];
    let v11 = values[xi * nv + vj];

    let price = v00 * (1.0 - tx) * (1.0 - tv)
        + v10 * tx * (1.0 - tv)
        + v01 * (1.0 - tx) * tv
        + v11 * tx * tv;

    // Delta via finite differences in x
    let dx = x_grid[xi] - x_grid[xi - 1];
    let dv_dx = (v10 * (1.0 - tv) + v11 * tv - v00 * (1.0 - tv) - v01 * tv) / dx;
    let delta = dv_dx / spot;

    // Gamma
    let xi2 = xi.min(nx - 2);
    let v20 = values[(xi2 + 1) * nv + (vj - 1)];
    let v21 = values[(xi2 + 1) * nv + vj];
    let v_plus = v20 * (1.0 - tv) + v21 * tv;
    let v_mid = v10 * (1.0 - tv) + v11 * tv;
    let v_minus = v00 * (1.0 - tv) + v01 * tv;
    let dx2 = x_grid[xi2 + 1] - x_grid[xi2];
    let d2v_dx2 = (v_plus - 2.0 * v_mid + v_minus) / (dx * dx2);
    let gamma = (d2v_dx2 - dv_dx) / (spot * spot);

    // Vega via finite differences in v
    let dv = v_grid[vj] - v_grid[vj - 1];
    let vega_raw = (v01 * (1.0 - tx) + v11 * tx - v00 * (1.0 - tx) - v10 * tx) / dv;
    // Vega per unit of total vol: ∂V/∂σ = ∂V/∂v × 2√v₀
    let vega = vega_raw * 2.0 * v0.sqrt();

    HestonFdResult {
        price,
        delta,
        gamma,
        vega,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn triple_band_apply() {
        // Simple [−1, 2, −1] second-derivative stencil
        let op = TripleBandOp {
            n: 5,
            lower: vec![0.0, -1.0, -1.0, -1.0, -1.0],
            diag: vec![2.0, 2.0, 2.0, 2.0, 2.0],
            upper: vec![-1.0, -1.0, -1.0, -1.0, 0.0],
        };
        let x = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let y = op.apply(&x);
        // Interior: y[2] = −1·2 + 2·3 − 1·2 = 2
        assert_abs_diff_eq!(y[2], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn thomas_solve_identity() {
        // Solve [1,1,1] x = [1,2,3]
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 1.0, 1.0];
        let c = [0.0, 0.0, 0.0];
        let d = [1.0, 2.0, 3.0];
        let x = thomas_solve(&a, &b, &c, &d);
        assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(x[1], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(x[2], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn thomas_solve_tridiag() {
        let a = [0.0, 1.0, 1.0];
        let b = [2.0, 3.0, 2.0];
        let c = [1.0, 1.0, 0.0];
        let d = [3.0, 5.0, 3.0];
        let x = thomas_solve(&a, &b, &c, &d);
        assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(x[1], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(x[2], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn bs_operator_non_trivial() {
        let grid: Vec<f64> = (0..51).map(|i| 3.5 + i as f64 * 0.04).collect();
        let op = build_bs_operator(&grid, 0.05, 0.02, 0.20);
        // Interior diagonal should be negative (diffusion + discounting)
        assert!(op.diag[25] < 0.0);
    }

    #[test]
    fn fd_1d_european_call_vs_bs() {
        let spot = 100.0;
        let strike = 100.0;
        let r = 0.05;
        let q = 0.02;
        let vol = 0.20;
        let t = 1.0;

        let result = fd_1d_bs_solve(spot, strike, r, q, vol, t, true, false, 200, 200);

        // BS analytical
        let d1 = ((spot / strike).ln() + (r - q + 0.5 * vol * vol) * t) / (vol * t.sqrt());
        let d2 = d1 - vol * t.sqrt();
        let n = ql_math::distributions::NormalDistribution::standard();
        let bs_price = spot * (-q * t).exp() * n.cdf(d1) - strike * (-r * t).exp() * n.cdf(d2);

        assert_abs_diff_eq!(result.price, bs_price, epsilon = 0.05);
    }

    #[test]
    fn fd_1d_european_put_call_parity() {
        let spot = 100.0;
        let strike = 100.0;
        let r = 0.05;
        let q = 0.02;
        let vol = 0.25;
        let t = 0.5;

        let call = fd_1d_bs_solve(spot, strike, r, q, vol, t, true, false, 200, 200);
        let put = fd_1d_bs_solve(spot, strike, r, q, vol, t, false, false, 200, 200);

        let parity = spot * (-q * t).exp() - strike * (-r * t).exp();
        assert_abs_diff_eq!(call.price - put.price, parity, epsilon = 0.10);
    }

    #[test]
    fn fd_1d_american_put_ge_european() {
        let spot = 100.0;
        let strike = 105.0;
        let r = 0.05;
        let q = 0.0;
        let vol = 0.20;
        let t = 1.0;

        let european = fd_1d_bs_solve(spot, strike, r, q, vol, t, false, false, 200, 200);
        let american = fd_1d_bs_solve(spot, strike, r, q, vol, t, false, true, 200, 200);

        assert!(
            american.price >= european.price - 0.01,
            "American put ({}) should >= European put ({})",
            american.price,
            european.price
        );
    }

    #[test]
    fn fd_1d_delta_bounded() {
        let result = fd_1d_bs_solve(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true, false, 200, 200);
        assert!(
            result.delta > 0.0 && result.delta < 1.0,
            "Call delta should be in (0,1): {}",
            result.delta
        );
    }

    #[test]
    fn fd_heston_european_call_positive() {
        let result = fd_heston_solve(
            100.0, 100.0, 0.05, 0.0,
            0.04, 1.5, 0.04, 0.5, -0.7,
            1.0, true, false, 50, 25, 100,
        );
        assert!(result.price > 0.0, "Heston FD price should be positive: {}", result.price);
    }

    #[test]
    fn fd_heston_put_positive() {
        let result = fd_heston_solve(
            100.0, 100.0, 0.05, 0.0,
            0.04, 1.5, 0.04, 0.5, -0.7,
            1.0, false, false, 50, 25, 100,
        );
        assert!(result.price > 0.0, "Heston FD put should be positive: {}", result.price);
    }

    #[test]
    fn fd_heston_call_vs_bs_flat_vol() {
        // When σ=0 (no vol-of-vol), Heston ≈ BS with vol = √v0
        let result = fd_heston_solve(
            100.0, 100.0, 0.05, 0.0,
            0.04, 1.5, 0.04, 0.001, 0.0, // tiny sigma
            1.0, true, false, 80, 30, 100,
        );

        let vol: f64 = 0.2; // sqrt(0.04)
        let t: f64 = 1.0;
        let r: f64 = 0.05;
        let spot: f64 = 100.0;
        let strike: f64 = 100.0;
        let d1 = ((spot / strike).ln() + (r + 0.5 * vol * vol) * t) / (vol * t.sqrt());
        let d2 = d1 - vol * t.sqrt();
        let n = ql_math::distributions::NormalDistribution::standard();
        let bs_price = spot * n.cdf(d1) - strike * (-r * t).exp() * n.cdf(d2);

        assert_abs_diff_eq!(result.price, bs_price, epsilon = 1.0);
    }

    #[test]
    fn fd_heston_american_ge_european() {
        let european = fd_heston_solve(
            100.0, 105.0, 0.05, 0.0,
            0.04, 1.5, 0.04, 0.5, -0.7,
            1.0, false, false, 50, 25, 100,
        );
        let american = fd_heston_solve(
            100.0, 105.0, 0.05, 0.0,
            0.04, 1.5, 0.04, 0.5, -0.7,
            1.0, false, true, 50, 25, 100,
        );
        assert!(
            american.price >= european.price - 0.5,
            "American Heston put ({}) should >= European ({})",
            american.price,
            european.price
        );
    }

    #[test]
    fn fd_heston_delta_reasonable() {
        let result = fd_heston_solve(
            100.0, 100.0, 0.05, 0.0,
            0.04, 1.5, 0.04, 0.5, -0.7,
            1.0, true, false, 50, 25, 100,
        );
        assert!(
            result.delta > 0.0 && result.delta < 1.0,
            "Heston call delta should be in (0,1): {}",
            result.delta
        );
    }
}
