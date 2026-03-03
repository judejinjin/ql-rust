//! Generic FD/tree infrastructure for AD-compatible pricing.
//!
//! Provides `T: Number` versions of key numerical methods:
//!
//! | Item | Purpose |
//! |------|---------|
//! | [`TripleBandOpGeneric`] | Tridiagonal operator with generic coefficients |
//! | [`fd_1d_solve_generic`] | Crank-Nicolson 1D PDE solver |
//! | [`build_bs_operator_generic`] | Black-Scholes PDE operator builder |
//! | [`binomial_crr_generic`] | CRR binomial tree |
//! | [`fd_2d_solve_generic`] | 2D ADI solver (Douglas scheme) |
//!
//! Grid points (`&[f64]`) remain `f64` — they are structural, not risk factors.
//! Operator coefficients, terminal values, and solutions are in generic `T`.

use ql_core::Number;

// ===========================================================================
// INFRA-7: Generic 1D Finite Difference Infrastructure
// ===========================================================================

/// Tridiagonal linear operator with generic coefficients.
///
/// Represents L·u where L is a tridiagonal matrix:
///   `lower[i] * u[i-1] + diag[i] * u[i] + upper[i] * u[i+1]`
#[derive(Debug, Clone)]
pub struct TripleBandOpGeneric<T: Number> {
    pub n: usize,
    pub lower: Vec<T>,
    pub diag: Vec<T>,
    pub upper: Vec<T>,
}

impl<T: Number> TripleBandOpGeneric<T> {
    /// Create a zero operator of size `n`.
    pub fn zeros(n: usize) -> Self {
        Self {
            n,
            lower: vec![T::zero(); n],
            diag: vec![T::zero(); n],
            upper: vec![T::zero(); n],
        }
    }

    /// Apply: y = L · x (tridiagonal multiply).
    pub fn apply(&self, x: &[T]) -> Vec<T> {
        let n = self.n;
        assert!(n >= 2);
        let mut y = vec![T::zero(); n];
        y[0] = self.diag[0] * x[0] + self.upper[0] * x[1];
        for i in 1..n - 1 {
            y[i] = self.lower[i] * x[i - 1] + self.diag[i] * x[i] + self.upper[i] * x[i + 1];
        }
        y[n - 1] = self.lower[n - 1] * x[n - 2] + self.diag[n - 1] * x[n - 1];
        y
    }

    /// Solve `(I − θ·dt·L)·x_new = rhs` using Thomas algorithm.
    ///
    /// `theta` and `dt` are `f64` (time-stepping parameters, not risk factors).
    pub fn solve_implicit(&self, rhs: &[T], theta: f64, dt: f64) -> Vec<T> {
        let n = self.n;
        let tdt = T::from_f64(theta * dt);
        let mut a = vec![T::zero(); n];
        let mut b = vec![T::zero(); n];
        let mut c = vec![T::zero(); n];

        for i in 0..n {
            a[i] = T::zero() - tdt * self.lower[i];
            b[i] = T::one() - tdt * self.diag[i];
            c[i] = T::zero() - tdt * self.upper[i];
        }

        thomas_solve_generic(&a, &b, &c, rhs)
    }

    /// Scale the operator by a generic constant.
    pub fn scale(&mut self, factor: T) {
        for i in 0..self.n {
            self.lower[i] = self.lower[i] * factor;
            self.diag[i] = self.diag[i] * factor;
            self.upper[i] = self.upper[i] * factor;
        }
    }

    /// Add another operator: self += other.
    pub fn add_assign(&mut self, other: &TripleBandOpGeneric<T>) {
        assert_eq!(self.n, other.n);
        for i in 0..self.n {
            self.lower[i] = self.lower[i] + other.lower[i];
            self.diag[i] = self.diag[i] + other.diag[i];
            self.upper[i] = self.upper[i] + other.upper[i];
        }
    }
}

/// Thomas algorithm for tridiagonal systems, generic over `T: Number`.
fn thomas_solve_generic<T: Number>(a: &[T], b: &[T], c: &[T], d: &[T]) -> Vec<T> {
    let n = d.len();
    let mut c_prime = vec![T::zero(); n];
    let mut d_prime = vec![T::zero(); n];

    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];

    for i in 1..n {
        let m = b[i] - a[i] * c_prime[i - 1];
        c_prime[i] = c[i] / m;
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / m;
    }

    let mut x = vec![T::zero(); n];
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
    x
}

/// Build 1D Black-Scholes operator on a log-spot grid, generic over `T`.
///
/// Grid points stay `f64`; PDE coefficients (r, q, vol) are generic `T`.
///
/// BS PDE in log-spot x = ln(S):
///   ∂V/∂t + (r − q − σ²/2) ∂V/∂x + σ²/2 ∂²V/∂x² − rV = 0
pub fn build_bs_operator_generic<T: Number>(
    grid: &[f64],
    r: T,
    q: T,
    vol: T,
) -> TripleBandOpGeneric<T> {
    let n = grid.len();
    let mut op = TripleBandOpGeneric::zeros(n);
    let half = T::half();
    let mu = r - q - half * vol * vol;
    let half_sigma2 = half * vol * vol;

    for i in 1..n - 1 {
        let dx_plus = grid[i + 1] - grid[i];
        let dx_minus = grid[i] - grid[i - 1];
        let dx_avg = T::from_f64(0.5 * (dx_plus + dx_minus));

        let conv = mu / (T::from_f64(2.0) * dx_avg);
        let diff = half_sigma2 / (dx_avg * T::from_f64(dx_plus));
        let diff_m = half_sigma2 / (dx_avg * T::from_f64(dx_minus));

        op.lower[i] = diff_m - conv;
        op.upper[i] = diff + conv;
        // Corrected: diff part = -(diff + diff_m), discount = -r
        op.diag[i] = T::zero() - diff - diff_m - r;
    }

    // Boundary conditions (Dirichlet: leave as zero = absorbing)
    op.diag[0] = T::zero() - r;
    op.diag[n - 1] = T::zero() - r;

    op
}

/// Result from generic 1D FD solver.
#[derive(Debug, Clone)]
pub struct Fd1dResultGeneric<T: Number> {
    /// Solution values at grid points (at t=0).
    pub values: Vec<T>,
    /// Grid points (f64).
    pub grid: Vec<f64>,
}

/// Solve a 1D PDE using Crank-Nicolson, generic over `T: Number`.
///
/// Takes an operator `L`, terminal condition, and steps backward in time.
/// Optionally applies American exercise at each step.
///
/// Grid points are `f64`; values and operator coefficients are `T`.
pub fn fd_1d_solve_generic<T: Number>(
    op: &TripleBandOpGeneric<T>,
    grid: &[f64],
    terminal: &[T],
    n_time_steps: usize,
    total_time: f64,
    american_payoff: Option<&[T]>,
) -> Fd1dResultGeneric<T> {
    let n = grid.len();
    assert_eq!(terminal.len(), n);
    let dt = total_time / n_time_steps as f64;
    let theta = 0.5; // Crank-Nicolson

    let mut values = terminal.to_vec();

    for _ in 0..n_time_steps {
        // Explicit part: rhs = u + (1-θ)·dt·L·u
        let lu = op.apply(&values);
        let mut rhs = vec![T::zero(); n];
        let one_minus_theta_dt = T::from_f64((1.0 - theta) * dt);
        for i in 0..n {
            rhs[i] = values[i] + one_minus_theta_dt * lu[i];
        }

        // Implicit solve: (I - θ·dt·L) · u_new = rhs
        values = op.solve_implicit(&rhs, theta, dt);

        // American exercise condition
        if let Some(payoff) = american_payoff {
            for i in 0..n {
                values[i] = values[i].max(payoff[i]);
            }
        }
    }

    Fd1dResultGeneric {
        values,
        grid: grid.to_vec(),
    }
}

/// Build terminal payoff on a log-spot grid.
///
/// Grid is in log-spot space: x = ln(S). Payoff = max(ω(S - K), 0).
pub fn build_terminal_payoff<T: Number>(
    grid: &[f64],
    strike: T,
    is_call: bool,
) -> Vec<T> {
    let omega = if is_call { T::one() } else { T::zero() - T::one() };
    grid.iter()
        .map(|&x| {
            let s = T::from_f64(x.exp());
            (omega * (s - strike)).max(T::zero())
        })
        .collect()
}

/// Build a uniform log-spot grid centred at `ln(spot)`.
pub fn build_log_spot_grid(spot: f64, vol: f64, t: f64, n: usize) -> Vec<f64> {
    let center = spot.ln();
    let width = 4.0 * vol * t.sqrt(); // ±4σ√T
    let lo = center - width;
    let hi = center + width;
    let dx = (hi - lo) / (n - 1) as f64;
    (0..n).map(|i| lo + i as f64 * dx).collect()
}

// ===========================================================================
// INFRA-8: Generic 2D Finite Difference (Douglas ADI)
// ===========================================================================

/// Operators for 2D Heston PDE, generic over `T`.
#[derive(Debug, Clone)]
pub struct Heston2dOpsGeneric<T: Number> {
    /// Spot-direction operator (tridiagonal in spot, for each variance level).
    pub op_x: TripleBandOpGeneric<T>,
    /// Variance-direction operator.
    pub op_v: TripleBandOpGeneric<T>,
    /// Cross-derivative weights (n_x × n_v flat array).
    pub cross: Vec<T>,
    pub n_x: usize,
    pub n_v: usize,
}

/// Build Heston 2D operators, generic over `T`.
///
/// Grid points stay `f64`; Heston parameters are `T`.
///
/// Heston PDE:
///   ∂V/∂t + (r-q)S ∂V/∂S + ½vS² ∂²V/∂S² + κ(θ-v)∂V/∂v + ½ξ²v ∂²V/∂v² + ρξvS ∂²V/∂S∂v − rV = 0
#[allow(clippy::too_many_arguments)]
pub fn build_heston_ops_generic<T: Number>(
    grid_x: &[f64],
    grid_v: &[f64],
    r: T,
    q: T,
    kappa: T,
    theta: T,
    xi: T,
    rho: T,
) -> Heston2dOpsGeneric<T> {
    let n_x = grid_x.len();
    let n_v = grid_v.len();
    let n_total = n_x * n_v;
    let half = T::half();

    // For the ADI approach, we build averaged 1D operators.
    // The spot-direction operator uses the mid-level variance.
    let v_mid = T::from_f64(grid_v[n_v / 2]);
    let op_x = {
        let mut op = TripleBandOpGeneric::zeros(n_x);
        let mu_x = r - q - half * v_mid; // drift in log-spot
        let diff_x = half * v_mid;

        for i in 1..n_x - 1 {
            let dxp = grid_x[i + 1] - grid_x[i];
            let dxm = grid_x[i] - grid_x[i - 1];
            let dx_avg = T::from_f64(0.5 * (dxp + dxm));
            let conv = mu_x / (T::from_f64(2.0) * dx_avg);
            let d_p = diff_x / (dx_avg * T::from_f64(dxp));
            let d_m = diff_x / (dx_avg * T::from_f64(dxm));
            op.lower[i] = d_m - conv;
            op.upper[i] = d_p + conv;
            op.diag[i] = T::zero() - d_p - d_m - half * r;
        }
        op.diag[0] = T::zero() - half * r;
        op.diag[n_x - 1] = T::zero() - half * r;
        op
    };

    // Variance-direction operator (Feller mean-reversion + vol-of-vol diffusion)
    let op_v = {
        let mut op = TripleBandOpGeneric::zeros(n_v);
        for i in 1..n_v - 1 {
            let v_i = T::from_f64(grid_v[i]);
            let dvp = grid_v[i + 1] - grid_v[i];
            let dvm = grid_v[i] - grid_v[i - 1];
            let dv_avg = T::from_f64(0.5 * (dvp + dvm));
            let mu_v = kappa * (theta - v_i);
            let diff_v = half * xi * xi * v_i;
            let conv = mu_v / (T::from_f64(2.0) * dv_avg);
            let d_p = diff_v / (dv_avg * T::from_f64(dvp));
            let d_m = diff_v / (dv_avg * T::from_f64(dvm));
            op.lower[i] = d_m - conv;
            op.upper[i] = d_p + conv;
            op.diag[i] = T::zero() - d_p - d_m - half * r;
        }
        op.diag[0] = T::zero() - half * r;
        op.diag[n_v - 1] = T::zero() - half * r;
        op
    };

    // Cross-derivative weights (used for explicit cross term)
    let mut cross = vec![T::zero(); n_total];
    for j in 1..n_v - 1 {
        let v_j = T::from_f64(grid_v[j]);
        let dvp = grid_v[j + 1] - grid_v[j];
        let dvm = grid_v[j] - grid_v[j - 1];
        for i in 1..n_x - 1 {
            let dxp = grid_x[i + 1] - grid_x[i];
            let dxm = grid_x[i] - grid_x[i - 1];
            let coeff = rho * xi * v_j
                / (T::from_f64(dxp + dxm) * T::from_f64(dvp + dvm));
            cross[j * n_x + i] = coeff;
        }
    }

    Heston2dOpsGeneric {
        op_x,
        op_v,
        cross,
        n_x,
        n_v,
    }
}

/// Result from generic 2D FD solver.
#[derive(Debug, Clone)]
pub struct Fd2dResultGeneric<T: Number> {
    /// Solution on 2D grid (row-major: v-index × x-index).
    pub values: Vec<T>,
    pub grid_x: Vec<f64>,
    pub grid_v: Vec<f64>,
}

/// Solve 2D Heston PDE using Douglas ADI, generic over `T`.
///
/// Uses operator splitting: each time step is split into
/// x-direction and v-direction implicit solves with explicit cross term.
pub fn fd_2d_solve_generic<T: Number>(
    ops: &Heston2dOpsGeneric<T>,
    grid_x: &[f64],
    grid_v: &[f64],
    terminal: &[T],
    n_time_steps: usize,
    total_time: f64,
) -> Fd2dResultGeneric<T> {
    let n_x = ops.n_x;
    let n_v = ops.n_v;
    let n_total = n_x * n_v;
    assert_eq!(terminal.len(), n_total);
    let dt = total_time / n_time_steps as f64;
    let theta = 0.5;

    let mut values = terminal.to_vec();

    for _ in 0..n_time_steps {
        // Step 1: Explicit cross-derivative contribution
        let mut rhs = values.clone();
        for j in 1..n_v - 1 {
            for i in 1..n_x - 1 {
                let idx = j * n_x + i;
                let cross_term = ops.cross[idx]
                    * (values[(j + 1) * n_x + (i + 1)]
                        - values[(j + 1) * n_x + (i - 1)]
                        - values[(j - 1) * n_x + (i + 1)]
                        + values[(j - 1) * n_x + (i - 1)]);
                rhs[idx] = rhs[idx] + T::from_f64(dt) * cross_term;
            }
        }

        // Step 2: x-direction implicit sweep (for each v-level)
        let mut intermediate = rhs.clone();
        for j in 0..n_v {
            let row_start = j * n_x;
            let row_rhs: Vec<T> = rhs[row_start..row_start + n_x].to_vec();

            // Apply explicit x-part
            let lu_x = ops.op_x.apply(&row_rhs);
            let mut rhs_x = vec![T::zero(); n_x];
            let one_m_theta_dt = T::from_f64((1.0 - theta) * dt);
            for i in 0..n_x {
                rhs_x[i] = row_rhs[i] + one_m_theta_dt * lu_x[i];
            }

            let solved = ops.op_x.solve_implicit(&rhs_x, theta, dt);
            for i in 0..n_x {
                intermediate[row_start + i] = solved[i];
            }
        }

        // Step 3: v-direction implicit sweep (for each x-level)
        for i in 0..n_x {
            let col: Vec<T> = (0..n_v).map(|j| intermediate[j * n_x + i]).collect();
            let lu_v = ops.op_v.apply(&col);
            let mut rhs_v = vec![T::zero(); n_v];
            let one_m_theta_dt = T::from_f64((1.0 - theta) * dt);
            for j in 0..n_v {
                rhs_v[j] = col[j] + one_m_theta_dt * lu_v[j];
            }

            let solved = ops.op_v.solve_implicit(&rhs_v, theta, dt);
            for j in 0..n_v {
                values[j * n_x + i] = solved[j];
            }
        }
    }

    Fd2dResultGeneric {
        values,
        grid_x: grid_x.to_vec(),
        grid_v: grid_v.to_vec(),
    }
}

// ===========================================================================
// INFRA-9: Generic Binomial Tree
// ===========================================================================

/// Result from generic binomial tree.
#[derive(Debug, Clone, Copy)]
pub struct LatticeResultGeneric<T: Number> {
    pub npv: T,
    pub delta: T,
    pub gamma: T,
    pub theta: T,
}

/// CRR binomial tree, generic over `T: Number`.
///
/// All pricing parameters (spot, strike, r, q, vol) are `T`;
/// structural parameters (n_steps, is_call, is_american) stay native types.
///
/// Greeks extracted from tree nodes at steps 1 and 2.
#[allow(clippy::too_many_arguments)]
pub fn binomial_crr_generic<T: Number>(
    spot: T,
    strike: T,
    r: T,
    q: T,
    vol: T,
    time_to_expiry: T,
    is_call: bool,
    is_american: bool,
    num_steps: usize,
) -> LatticeResultGeneric<T> {
    let zero = T::zero();
    let one = T::one();
    let half = T::half();
    let dt = time_to_expiry / T::from_f64(num_steps as f64);
    let df = (zero - r * dt).exp();
    let u = (vol * dt.sqrt()).exp();
    let d = one / u;
    let growth = ((r - q) * dt).exp();
    let p = (growth - d) / (u - d);
    let omega = if is_call { one } else { zero - one };
    let n = num_steps;

    // Terminal payoffs
    let mut values: Vec<T> = (0..=n)
        .map(|j| {
            let s_t = spot * u.powf(T::from_f64(j as f64)) * d.powf(T::from_f64((n - j) as f64));
            (omega * (s_t - strike)).max(zero)
        })
        .collect();

    // Save for Greeks
    let mut val_step_1 = [T::zero(); 2];
    let mut val_step_2 = [T::zero(); 3];

    // Backward induction
    for step in (0..n).rev() {
        let mut new_values: Vec<T> = (0..=step)
            .map(|j| {
                let cont = df * (p * values[j + 1] + (one - p) * values[j]);
                if is_american {
                    let s_node = spot * u.powf(T::from_f64(j as f64))
                        * d.powf(T::from_f64((step - j) as f64));
                    let intrinsic = (omega * (s_node - strike)).max(zero);
                    cont.max(intrinsic)
                } else {
                    cont
                }
            })
            .collect();

        if step == 2 && n >= 3 {
            val_step_2 = [new_values[0], new_values[1], new_values[2]];
        }
        if step == 1 {
            val_step_1 = [new_values[0], new_values[1]];
        }

        values = new_values;
    }

    let npv = values[0];
    let su = spot * u;
    let sd = spot * d;
    let delta = if n >= 1 {
        (val_step_1[1] - val_step_1[0]) / (su - sd)
    } else {
        zero
    };

    let gamma = if n >= 3 {
        let su2 = spot * u * u;
        let sd2 = spot * d * d;
        let delta_up = (val_step_2[2] - val_step_2[1]) / (su2 - spot);
        let delta_down = (val_step_2[1] - val_step_2[0]) / (spot - sd2);
        (delta_up - delta_down) / (half * (su2 - sd2))
    } else {
        zero
    };

    let theta = if n >= 3 {
        (val_step_2[1] - npv) / (T::from_f64(2.0) * dt)
    } else {
        zero
    };

    LatticeResultGeneric {
        npv,
        delta,
        gamma,
        theta,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ql_math::generic::black_scholes_generic;

    // --- INFRA-7: 1D FD ---
    #[test]
    fn fd_1d_european_call_vs_bs() {
        let spot = 100.0_f64;
        let strike = 100.0;
        let r = 0.05;
        let q = 0.0;
        let vol = 0.20;
        let t = 1.0;

        let grid = build_log_spot_grid(spot, vol, t, 200);
        let op = build_bs_operator_generic(&grid, r, q, vol);
        let terminal = build_terminal_payoff(&grid, strike, true);

        let result = fd_1d_solve_generic(&op, &grid, &terminal, 200, t, None);

        // Interpolate at ln(spot)
        let log_s = spot.ln();
        let idx = grid.partition_point(|&x| x < log_s).min(grid.len() - 2);
        let frac = (log_s - grid[idx]) / (grid[idx + 1] - grid[idx]);
        let fd_price = result.values[idx] * (1.0 - frac) + result.values[idx + 1] * frac;

        let bs: f64 = black_scholes_generic(spot, strike, r, q, vol, t, true);
        assert!(
            (fd_price - bs).abs() < 0.5,
            "fd={fd_price} vs bs={bs}",
        );
    }

    #[test]
    fn fd_1d_american_put_geq_european() {
        let spot = 100.0_f64;
        let strike = 100.0;
        let r = 0.05;
        let q = 0.02;
        let vol = 0.20;
        let t = 1.0;

        let grid = build_log_spot_grid(spot, vol, t, 200);
        let op = build_bs_operator_generic(&grid, r, q, vol);
        let terminal = build_terminal_payoff(&grid, strike, false);

        let european = fd_1d_solve_generic(&op, &grid, &terminal, 200, t, None);
        let american = fd_1d_solve_generic(&op, &grid, &terminal, 200, t, Some(&terminal));

        let log_s = spot.ln();
        let idx = grid.partition_point(|&x| x < log_s).min(grid.len() - 2);
        let frac = (log_s - grid[idx]) / (grid[idx + 1] - grid[idx]);
        let fd_eur = european.values[idx] * (1.0 - frac) + european.values[idx + 1] * frac;
        let fd_amer = american.values[idx] * (1.0 - frac) + american.values[idx + 1] * frac;

        assert!(fd_amer >= fd_eur - 0.01, "amer={fd_amer} >= eur={fd_eur}");
    }

    #[test]
    fn fd_1d_put_call_parity() {
        let spot = 100.0_f64;
        let strike = 100.0;
        let r = 0.05;
        let q = 0.0;
        let vol = 0.20;
        let t = 1.0;

        let grid = build_log_spot_grid(spot, vol, t, 200);
        let op = build_bs_operator_generic(&grid, r, q, vol);

        let terminal_call = build_terminal_payoff(&grid, strike, true);
        let terminal_put = build_terminal_payoff(&grid, strike, false);
        let result_call = fd_1d_solve_generic(&op, &grid, &terminal_call, 200, t, None);
        let result_put = fd_1d_solve_generic(&op, &grid, &terminal_put, 200, t, None);

        let log_s = spot.ln();
        let idx = grid.partition_point(|&x| x < log_s).min(grid.len() - 2);
        let frac = (log_s - grid[idx]) / (grid[idx + 1] - grid[idx]);
        let c = result_call.values[idx] * (1.0 - frac) + result_call.values[idx + 1] * frac;
        let p = result_put.values[idx] * (1.0 - frac) + result_put.values[idx + 1] * frac;

        let parity = spot - strike * (-r * t).exp();
        assert!(
            (c - p - parity).abs() < 0.5,
            "c={c}, p={p}, parity={parity}",
        );
    }

    // --- INFRA-9: Binomial tree ---
    #[test]
    fn crr_generic_european_call_vs_bs() {
        let res: LatticeResultGeneric<f64> = binomial_crr_generic(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true, false, 500,
        );
        let bs: f64 = black_scholes_generic(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true);
        assert!(
            (res.npv - bs).abs() < 0.2,
            "crr={} vs bs={bs}",
            res.npv,
        );
    }

    #[test]
    fn crr_generic_american_put() {
        let eur: LatticeResultGeneric<f64> = binomial_crr_generic(
            100.0, 100.0, 0.05, 0.02, 0.20, 1.0, false, false, 500,
        );
        let amer: LatticeResultGeneric<f64> = binomial_crr_generic(
            100.0, 100.0, 0.05, 0.02, 0.20, 1.0, false, true, 500,
        );
        assert!(
            amer.npv >= eur.npv - 0.01,
            "amer={} >= eur={}",
            amer.npv,
            eur.npv,
        );
    }

    #[test]
    fn crr_generic_greeks_positive() {
        let res: LatticeResultGeneric<f64> = binomial_crr_generic(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0, true, false, 300,
        );
        assert!(res.delta > 0.3 && res.delta < 0.9, "delta={}", res.delta);
        assert!(res.gamma > 0.0, "gamma={}", res.gamma);
    }

    // --- INFRA-8: 2D Heston FD ---
    #[test]
    fn fd_2d_heston_european_call_positive() {
        let spot = 100.0_f64;
        let strike = 100.0;
        let r = 0.05;
        let q = 0.0;
        let v0 = 0.04;
        let kappa = 1.5;
        let theta = 0.04;
        let xi = 0.3;
        let rho = -0.7;
        let t = 1.0;

        let n_x = 50;
        let n_v = 20;

        let grid_x = build_log_spot_grid(spot, 0.20, t, n_x);
        let grid_v: Vec<f64> = (0..n_v).map(|i| i as f64 * 0.5 / (n_v - 1) as f64).collect();

        let ops = build_heston_ops_generic(&grid_x, &grid_v, r, q, kappa, theta, xi, rho);

        // Terminal payoff for call, replicated at each v-level
        let terminal: Vec<f64> = grid_v
            .iter()
            .flat_map(|_| {
                grid_x.iter().map(|&x| (x.exp() - strike).max(0.0))
            })
            .collect();

        let result = fd_2d_solve_generic(&ops, &grid_x, &grid_v, &terminal, 50, t);

        // Find price at (spot, v0)
        let log_s = spot.ln();
        let ix = grid_x.partition_point(|&x| x < log_s).min(n_x - 2);
        let iv = grid_v.partition_point(|&v| v < v0).min(n_v - 2);
        let fx = (log_s - grid_x[ix]) / (grid_x[ix + 1] - grid_x[ix]);
        let fv = (v0 - grid_v[iv]) / (grid_v[iv + 1] - grid_v[iv]);

        let v00 = result.values[iv * n_x + ix];
        let v10 = result.values[iv * n_x + ix + 1];
        let v01 = result.values[(iv + 1) * n_x + ix];
        let v11 = result.values[(iv + 1) * n_x + ix + 1];
        let price = v00 * (1.0 - fx) * (1.0 - fv) + v10 * fx * (1.0 - fv)
            + v01 * (1.0 - fx) * fv + v11 * fx * fv;

        assert!(price > 5.0 && price < 25.0, "heston_fd_2d={price}");
    }
}
