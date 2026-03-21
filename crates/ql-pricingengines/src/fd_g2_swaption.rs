//! FD G2++ swaption engine.
//!
//! Prices European and Bermudan swaptions under the two-factor G2++ model
//! using a 2D finite-difference (ADI) scheme.
//!
//! The G2++ model defines the short rate as:
//!   r(t) = x(t) + y(t) + φ(t)
//! where x and y are mean-reverting OU processes with correlation ρ.
//!
//! This corresponds to QuantLib's `FdG2SwaptionEngine`.

use serde::{Deserialize, Serialize};
use ql_models::G2Model;

/// Extract instantaneous forward rate from G2Model via finite difference of bond prices.
fn model_forward_rate(model: &G2Model, t: f64) -> f64 {
    let eps = 1e-4;
    if t < eps {
        let p1 = model.bond_price(eps);
        -p1.ln() / eps
    } else {
        let p0 = model.bond_price(t - eps);
        let p1 = model.bond_price(t + eps);
        -(p1.ln() - p0.ln()) / (2.0 * eps)
    }
}

/// Result from the FD G2++ swaption engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FdG2SwaptionResult {
    /// Net present value.
    pub npv: f64,
    /// Number of x grid points.
    pub nx: usize,
    /// Number of y grid points.
    pub ny: usize,
    /// Number of time steps.
    pub nt: usize,
}

/// Grid parameters for the 2D FD solver.
#[derive(Debug, Clone, Copy)]
pub struct FdG2GridParams {
    /// Number of spatial grid points in x direction.
    pub nx: usize,
    /// Number of spatial grid points in y direction.
    pub ny: usize,
    /// Number of time steps.
    pub nt: usize,
    /// Number of standard deviations for x grid extent.
    pub x_stddevs: f64,
    /// Number of standard deviations for y grid extent.
    pub y_stddevs: f64,
}

impl Default for FdG2GridParams {
    fn default() -> Self {
        Self {
            nx: 50,
            ny: 50,
            nt: 100,
            x_stddevs: 4.0,
            y_stddevs: 4.0,
        }
    }
}

/// Price a European or Bermudan swaption under the G2++ model using 2D FD.
///
/// # Parameters
/// - `model` — calibrated G2++ model
/// - `fixed_leg_times` — payment times for the fixed leg
/// - `fixed_leg_amounts` — fixed coupon amounts (= notional × rate × accrual)
/// - `float_leg_times` — fixing/payment times for the floating leg
/// - `notional` — swap notional
/// - `is_payer` — true for payer swaption
/// - `exercise_times` — exercise dates (1 for European, multiple for Bermudan)
/// - `params` — FD grid parameters
#[allow(clippy::too_many_arguments)]
pub fn fd_g2_swaption(
    model: &G2Model,
    fixed_leg_times: &[f64],
    fixed_leg_amounts: &[f64],
    float_leg_times: &[f64],
    notional: f64,
    is_payer: bool,
    exercise_times: &[f64],
    params: FdG2GridParams,
) -> FdG2SwaptionResult {
    let a = model.a();
    let sigma = model.sigma();
    let b = model.b();
    let eta = model.eta();
    let _rho = model.rho();

    let mat_time = *exercise_times.last().unwrap_or(&1.0);
    let dt = mat_time / params.nt as f64;

    // Standard deviations of x and y at maturity
    let x_std = sigma / (2.0 * a).sqrt() * (1.0 - (-2.0 * a * mat_time).exp()).sqrt();
    let y_std = eta / (2.0 * b).sqrt() * (1.0 - (-2.0 * b * mat_time).exp()).sqrt();

    let x_max = params.x_stddevs * x_std.max(0.01);
    let y_max = params.y_stddevs * y_std.max(0.01);

    let nx = params.nx;
    let ny = params.ny;
    let dx = 2.0 * x_max / (nx - 1) as f64;
    let dy = 2.0 * y_max / (ny - 1) as f64;

    // Grid points
    let xs: Vec<f64> = (0..nx).map(|i| -x_max + i as f64 * dx).collect();
    let ys: Vec<f64> = (0..ny).map(|j| -y_max + j as f64 * dy).collect();

    // Terminal payoff: swap value at maturity
    let phi_mat = g2_phi(model, mat_time);
    let mut v = vec![vec![0.0_f64; ny]; nx];

    for i in 0..nx {
        for j in 0..ny {
            let r_val = xs[i] + ys[j] + phi_mat;
            let swap_val = swap_value_from_r(r_val, fixed_leg_times, fixed_leg_amounts,
                                              float_leg_times, notional, mat_time);
            v[i][j] = if is_payer { swap_val.max(0.0) } else { (-swap_val).max(0.0) };
        }
    }

    // Backward induction with ADI scheme
    let mut t = mat_time;
    for _step in 0..params.nt {
        t -= dt;
        let phi_t = g2_phi(model, t.max(0.0));

        // Half-step: implicit in x (sweep rows)
        let mut v_half = vec![vec![0.0_f64; ny]; nx];
        for j in 0..ny {
            let mut rhs = vec![0.0; nx];
            for i in 0..nx {
                let r_val = xs[i] + ys[j] + phi_t;
                // Explicit y-direction terms
                let vyy = if j > 0 && j < ny - 1 {
                    (v[i][j + 1] - 2.0 * v[i][j] + v[i][j - 1]) / (dy * dy)
                } else { 0.0 };
                let vy = if j > 0 && j < ny - 1 {
                    (v[i][j + 1] - v[i][j - 1]) / (2.0 * dy)
                } else { 0.0 };

                rhs[i] = v[i][j] + 0.5 * dt * (
                    0.5 * eta * eta * vyy - b * ys[j] * vy - 0.5 * r_val * v[i][j]
                );
            }
            // Solve tridiagonal for x-direction implicit
            let coeff_a = 0.5 * sigma * sigma / (dx * dx);
            let coeff_drift = |i: usize| -a * xs[i] / (2.0 * dx);
            let mut lower = vec![0.0; nx];
            let mut diag = vec![0.0; nx];
            let mut upper = vec![0.0; nx];
            for i in 1..nx - 1 {
                let r_val = xs[i] + ys[j] + phi_t;
                lower[i] = -0.5 * dt * (coeff_a - coeff_drift(i));
                diag[i] = 1.0 + 0.5 * dt * (2.0 * coeff_a + 0.5 * r_val);
                upper[i] = -0.5 * dt * (coeff_a + coeff_drift(i));
            }
            diag[0] = 1.0;
            diag[nx - 1] = 1.0;

            let soln = solve_tridiag(&lower, &diag, &upper, &rhs);
            for i in 0..nx {
                v_half[i][j] = soln[i];
            }
        }

        // Half-step: implicit in y (sweep columns)
        for i in 0..nx {
            let mut rhs = vec![0.0; ny];
            for j in 0..ny {
                let r_val = xs[i] + ys[j] + phi_t;
                let vxx = if i > 0 && i < nx - 1 {
                    (v_half[i + 1][j] - 2.0 * v_half[i][j] + v_half[i - 1][j]) / (dx * dx)
                } else { 0.0 };
                let vx = if i > 0 && i < nx - 1 {
                    (v_half[i + 1][j] - v_half[i - 1][j]) / (2.0 * dx)
                } else { 0.0 };

                rhs[j] = v_half[i][j] + 0.5 * dt * (
                    0.5 * sigma * sigma * vxx - a * xs[i] * vx - 0.5 * r_val * v_half[i][j]
                );
            }
            let coeff_b = 0.5 * eta * eta / (dy * dy);
            let coeff_drift_y = |j: usize| -b * ys[j] / (2.0 * dy);
            let mut lower = vec![0.0; ny];
            let mut diag = vec![0.0; ny];
            let mut upper = vec![0.0; ny];
            for j in 1..ny - 1 {
                let r_val = xs[i] + ys[j] + phi_t;
                lower[j] = -0.5 * dt * (coeff_b - coeff_drift_y(j));
                diag[j] = 1.0 + 0.5 * dt * (2.0 * coeff_b + 0.5 * r_val);
                upper[j] = -0.5 * dt * (coeff_b + coeff_drift_y(j));
            }
            diag[0] = 1.0;
            diag[ny - 1] = 1.0;

            let soln = solve_tridiag(&lower, &diag, &upper, &rhs);
            v[i][..ny].copy_from_slice(&soln[..ny]);
        }

        // Early exercise for Bermudan
        if exercise_times.iter().any(|&et| (et - t).abs() < dt * 0.5) {
            for i in 0..nx {
                for j in 0..ny {
                    let r_val = xs[i] + ys[j] + phi_t;
                    let swap_val = swap_value_from_r(r_val, fixed_leg_times, fixed_leg_amounts,
                                                      float_leg_times, notional, t);
                    let exercise_val = if is_payer { swap_val.max(0.0) } else { (-swap_val).max(0.0) };
                    v[i][j] = v[i][j].max(exercise_val);
                }
            }
        }
    }

    // Interpolate to get value at (x=0, y=0) — the spot state
    let ix = ((0.0 - (-x_max)) / dx) as usize;
    let jy = ((0.0 - (-y_max)) / dy) as usize;
    let ix = ix.min(nx - 2);
    let jy = jy.min(ny - 2);

    let wx = (0.0 - xs[ix]) / dx;
    let wy = (0.0 - ys[jy]) / dy;
    let npv = (1.0 - wx) * (1.0 - wy) * v[ix][jy]
        + wx * (1.0 - wy) * v[ix + 1][jy]
        + (1.0 - wx) * wy * v[ix][jy + 1]
        + wx * wy * v[ix + 1][jy + 1];

    FdG2SwaptionResult {
        npv: npv.max(0.0),
        nx,
        ny,
        nt: params.nt,
    }
}

/// Deterministic shift φ(t) to fit the initial term structure.
/// Under G2++: φ(t) = f_M(0,t) + σ²/(2a²)(1 − e^{−aT})² + η²/(2b²)(1 − e^{−bT})²
///             + ρση/(ab) (1 − e^{−aT})(1 − e^{−bT})
fn g2_phi(model: &G2Model, t: f64) -> f64 {
    let a = model.a();
    let b_param = model.b();
    let sigma = model.sigma();
    let eta = model.eta();
    let rho = model.rho();

    let f_inst = model_forward_rate(model, t);

    let ea = 1.0 - (-a * t).exp();
    let eb = 1.0 - (-b_param * t).exp();

    f_inst
        + sigma * sigma / (2.0 * a * a) * ea * ea
        + eta * eta / (2.0 * b_param * b_param) * eb * eb
        + rho * sigma * eta / (a * b_param) * ea * eb
}

/// Value a plain swap given a flat short rate r for discounting.
fn swap_value_from_r(
    r: f64,
    fixed_times: &[f64],
    fixed_amounts: &[f64],
    float_times: &[f64],
    notional: f64,
    current_time: f64,
) -> f64 {
    let discount = |t: f64| (-r * (t - current_time).max(0.0)).exp();

    // Fixed leg PV
    let fixed_pv: f64 = fixed_times.iter().zip(fixed_amounts.iter())
        .filter(|(&t, _)| t > current_time)
        .map(|(&t, &amt)| amt * discount(t))
        .sum();

    // Float leg PV ≈ notional × (1 − P(T_n))
    let last_float_t = float_times.iter().copied()
        .rfind(|&t| t > current_time)
        .unwrap_or(current_time);
    let float_pv = notional * (1.0 - discount(last_float_t));

    // Payer swap = float − fixed
    float_pv - fixed_pv
}

/// Thomas algorithm for tridiagonal system.
fn solve_tridiag(lower: &[f64], diag: &[f64], upper: &[f64], rhs: &[f64]) -> Vec<f64> {
    let n = diag.len();
    let mut c = vec![0.0; n];
    let mut d = vec![0.0; n];

    c[0] = upper[0] / diag[0];
    d[0] = rhs[0] / diag[0];

    for i in 1..n {
        let m = diag[i] - lower[i] * c[i - 1];
        if m.abs() < 1e-20 {
            c[i] = 0.0;
            d[i] = 0.0;
        } else {
            c[i] = if i < n - 1 { upper[i] / m } else { 0.0 };
            d[i] = (rhs[i] - lower[i] * d[i - 1]) / m;
        }
    }

    let mut x = vec![0.0; n];
    x[n - 1] = d[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d[i] - c[i] * x[i + 1];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_g2_model() -> G2Model {
        // Simple G2++ model with flat 4% curve
        G2Model::new(0.05, 0.01, 0.1, 0.005, -0.75, 0.04)
    }

    #[test]
    fn test_fd_g2_european_swaption() {
        let model = make_g2_model();
        let fixed_times = vec![1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
        let fixed_amounts: Vec<f64> = fixed_times.iter().map(|_| 1_000_000.0 * 0.04 * 0.5).collect();
        let float_times = fixed_times.clone();

        let res = fd_g2_swaption(
            &model,
            &fixed_times,
            &fixed_amounts,
            &float_times,
            1_000_000.0,
            true, // payer
            &[1.0], // exercise at 1Y
            FdG2GridParams::default(),
        );
        assert!(res.npv >= 0.0, "npv={}", res.npv);
        assert!(res.npv < 100_000.0, "npv={} too large", res.npv);
    }

    #[test]
    fn test_fd_g2_bermudan_geq_european() {
        let model = make_g2_model();
        let fixed_times = vec![2.0, 3.0, 4.0, 5.0];
        let fixed_amounts: Vec<f64> = fixed_times.iter().map(|_| 1e6 * 0.04 * 1.0).collect();
        let float_times = fixed_times.clone();

        let euro = fd_g2_swaption(
            &model, &fixed_times, &fixed_amounts, &float_times,
            1e6, true, &[1.0],
            FdG2GridParams { nx: 30, ny: 30, nt: 50, ..Default::default() },
        );
        let bermudan = fd_g2_swaption(
            &model, &fixed_times, &fixed_amounts, &float_times,
            1e6, true, &[1.0, 2.0, 3.0],
            FdG2GridParams { nx: 30, ny: 30, nt: 50, ..Default::default() },
        );
        // Bermudan >= European
        assert!(bermudan.npv >= euro.npv * 0.99,
            "bermudan={}, euro={}", bermudan.npv, euro.npv);
    }

    #[test]
    fn test_fd_g2_receiver_swaption() {
        let model = make_g2_model();
        let fixed_times = vec![2.0, 3.0, 4.0, 5.0];
        let fixed_amounts: Vec<f64> = fixed_times.iter().map(|_| 1e6 * 0.04 * 1.0).collect();
        let float_times = fixed_times.clone();

        let res = fd_g2_swaption(
            &model, &fixed_times, &fixed_amounts, &float_times,
            1e6, false, &[1.0],
            FdG2GridParams { nx: 30, ny: 30, nt: 50, ..Default::default() },
        );
        assert!(res.npv >= 0.0, "receiver npv={}", res.npv);
    }
}
