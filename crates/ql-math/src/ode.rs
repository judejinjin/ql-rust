//! Ordinary Differential Equation (ODE) solvers.
//!
//! Implements adaptive Runge-Kutta methods:
//! - **RK45 (Dormand-Prince)**: 4th/5th order embedded pair with adaptive step control
//! - **RK4**: Classical fixed-step 4th-order Runge-Kutta
//!
//! Reference: Dormand & Prince (1980), *A family of embedded Runge-Kutta formulae*.

/// Result from an ODE integration.
#[derive(Debug, Clone)]
pub struct OdeResult {
    /// Solution trajectory: Vec<(t, y)>  where y is the state vector.
    pub trajectory: Vec<(f64, Vec<f64>)>,
    /// Total number of function evaluations.
    pub n_evals: usize,
    /// Total number of accepted steps.
    pub n_steps: usize,
    /// Total number of rejected steps.
    pub n_rejected: usize,
}

/// Solve the ODE system y'(t) = f(t, y) using the classical RK4 method.
///
/// # Arguments
/// - `f`    — right-hand side: `f(t, &y) -> Vec<f64>` (returns dy/dt)
/// - `t0`   — initial time
/// - `y0`   — initial state vector
/// - `t_end` — final time
/// - `dt`   — fixed time step
pub fn rk4<F>(f: F, t0: f64, y0: &[f64], t_end: f64, dt: f64) -> OdeResult
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    let n = y0.len();
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut trajectory = vec![(t, y.clone())];
    let mut n_evals = 0;
    let mut n_steps = 0;

    while t < t_end - 1e-14 {
        let h = dt.min(t_end - t);

        let k1 = f(t, &y);
        n_evals += 1;

        let y_tmp: Vec<f64> = (0..n).map(|i| y[i] + 0.5 * h * k1[i]).collect();
        let k2 = f(t + 0.5 * h, &y_tmp);
        n_evals += 1;

        let y_tmp: Vec<f64> = (0..n).map(|i| y[i] + 0.5 * h * k2[i]).collect();
        let k3 = f(t + 0.5 * h, &y_tmp);
        n_evals += 1;

        let y_tmp: Vec<f64> = (0..n).map(|i| y[i] + h * k3[i]).collect();
        let k4 = f(t + h, &y_tmp);
        n_evals += 1;

        for i in 0..n {
            y[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        t += h;
        n_steps += 1;
        trajectory.push((t, y.clone()));
    }

    OdeResult {
        trajectory,
        n_evals,
        n_steps,
        n_rejected: 0,
    }
}

/// Solve the ODE system y'(t) = f(t, y) using the adaptive Dormand-Prince RK45 method.
///
/// The step size is automatically adjusted to maintain the local truncation error
/// within the specified tolerance.
///
/// # Arguments
/// - `f`        — right-hand side: `f(t, &y) -> Vec<f64>`
/// - `t0`       — initial time
/// - `y0`       — initial state vector
/// - `t_end`    — final time
/// - `dt_init`  — initial step size guess
/// - `abs_tol`  — absolute error tolerance per component
/// - `rel_tol`  — relative error tolerance per component
/// - `max_steps` — maximum number of steps (prevents infinite loops)
pub fn rk45_adaptive<F>(
    f: F,
    t0: f64,
    y0: &[f64],
    t_end: f64,
    dt_init: f64,
    abs_tol: f64,
    rel_tol: f64,
    max_steps: usize,
) -> OdeResult
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    // Dormand-Prince coefficients
    const A2: f64 = 1.0 / 5.0;
    const A3: f64 = 3.0 / 10.0;
    const A4: f64 = 4.0 / 5.0;
    const A5: f64 = 8.0 / 9.0;

    // b21
    const B21: f64 = 1.0 / 5.0;
    // b31, b32
    const B31: f64 = 3.0 / 40.0;
    const B32: f64 = 9.0 / 40.0;
    // b41..b43
    const B41: f64 = 44.0 / 45.0;
    const B42: f64 = -56.0 / 15.0;
    const B43: f64 = 32.0 / 9.0;
    // b51..b54
    const B51: f64 = 19372.0 / 6561.0;
    const B52: f64 = -25360.0 / 2187.0;
    const B53: f64 = 64448.0 / 6561.0;
    const B54: f64 = -212.0 / 729.0;
    // b61..b65
    const B61: f64 = 9017.0 / 3168.0;
    const B62: f64 = -355.0 / 33.0;
    const B63: f64 = 46732.0 / 5247.0;
    const B64: f64 = 49.0 / 176.0;
    const B65: f64 = -5103.0 / 18656.0;

    // 5th order weights (for the solution)
    const C1: f64 = 35.0 / 384.0;
    // C2 = 0
    const C3: f64 = 500.0 / 1113.0;
    const C4: f64 = 125.0 / 192.0;
    const C5: f64 = -2187.0 / 6784.0;
    const C6: f64 = 11.0 / 84.0;

    // 4th order weights (for error estimation)
    const D1: f64 = 5179.0 / 57600.0;
    // D2 = 0
    const D3: f64 = 7571.0 / 16695.0;
    const D4: f64 = 393.0 / 640.0;
    const D5: f64 = -92097.0 / 339200.0;
    const D6: f64 = 187.0 / 2100.0;
    const D7: f64 = 1.0 / 40.0;

    let n = y0.len();
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut h = dt_init.min(t_end - t0);
    let mut trajectory = vec![(t, y.clone())];
    let mut n_evals = 0;
    let mut n_steps = 0;
    let mut n_rejected = 0;

    let h_min = 1e-14;

    for _ in 0..max_steps {
        if t >= t_end - 1e-14 {
            break;
        }
        h = h.min(t_end - t);

        // Stage 1
        let k1 = f(t, &y);
        n_evals += 1;

        // Stage 2
        let y2: Vec<f64> = (0..n).map(|i| y[i] + h * B21 * k1[i]).collect();
        let k2 = f(t + A2 * h, &y2);
        n_evals += 1;

        // Stage 3
        let y3: Vec<f64> = (0..n).map(|i| y[i] + h * (B31 * k1[i] + B32 * k2[i])).collect();
        let k3 = f(t + A3 * h, &y3);
        n_evals += 1;

        // Stage 4
        let y4: Vec<f64> = (0..n)
            .map(|i| y[i] + h * (B41 * k1[i] + B42 * k2[i] + B43 * k3[i]))
            .collect();
        let k4 = f(t + A4 * h, &y4);
        n_evals += 1;

        // Stage 5
        let y5: Vec<f64> = (0..n)
            .map(|i| y[i] + h * (B51 * k1[i] + B52 * k2[i] + B53 * k3[i] + B54 * k4[i]))
            .collect();
        let k5 = f(t + A5 * h, &y5);
        n_evals += 1;

        // Stage 6
        let y6: Vec<f64> = (0..n)
            .map(|i| {
                y[i] + h * (B61 * k1[i] + B62 * k2[i] + B63 * k3[i] + B64 * k4[i] + B65 * k5[i])
            })
            .collect();
        let k6 = f(t + h, &y6);
        n_evals += 1;

        // 5th order solution
        let y_new: Vec<f64> = (0..n)
            .map(|i| {
                y[i] + h * (C1 * k1[i] + C3 * k3[i] + C4 * k4[i] + C5 * k5[i] + C6 * k6[i])
            })
            .collect();

        // 4th order solution for error estimate (need k7 = f(t+h, y_new) for FSAL)
        let k7 = f(t + h, &y_new);
        n_evals += 1;

        // Error = y5 - y4 (difference between 5th and 4th order)
        let mut err_max = 0.0_f64;
        for i in 0..n {
            let y4_i = y[i]
                + h * (D1 * k1[i] + D3 * k3[i] + D4 * k4[i] + D5 * k5[i] + D6 * k6[i]
                    + D7 * k7[i]);
            let err_i = (y_new[i] - y4_i).abs();
            let scale = abs_tol + rel_tol * y_new[i].abs().max(y[i].abs());
            err_max = err_max.max(err_i / scale);
        }

        if err_max <= 1.0 {
            // Accept step
            t += h;
            y = y_new;
            trajectory.push((t, y.clone()));
            n_steps += 1;
        } else {
            n_rejected += 1;
        }

        // Adjust step size: h_new = h * min(5, max(0.2, 0.9 * err^(-1/5)))
        let factor = if err_max > 0.0 {
            0.9 * err_max.powf(-0.2)
        } else {
            5.0
        };
        h *= factor.clamp(0.2, 5.0);
        h = h.max(h_min);
    }

    OdeResult {
        trajectory,
        n_evals,
        n_steps,
        n_rejected,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn rk4_exponential_decay() {
        // y' = -y, y(0) = 1 → y(t) = exp(-t)
        let res = rk4(|_t, y| vec![-y[0]], 0.0, &[1.0], 1.0, 0.01);
        let (t_final, y_final) = res.trajectory.last().unwrap();
        assert_abs_diff_eq!(*t_final, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(y_final[0], (-1.0_f64).exp(), epsilon = 1e-8);
    }

    #[test]
    fn rk4_sinusoidal() {
        // y'' = -y → system: y₁' = y₂, y₂' = -y₁
        // y(0) = 0, y'(0) = 1 → y(t) = sin(t)
        let res = rk4(
            |_t, y| vec![y[1], -y[0]],
            0.0,
            &[0.0, 1.0],
            std::f64::consts::PI,
            0.001,
        );
        let (_, y_final) = res.trajectory.last().unwrap();
        // sin(π) ≈ 0
        assert_abs_diff_eq!(y_final[0], 0.0, epsilon = 1e-4);
    }

    #[test]
    fn rk45_exponential_decay() {
        let res = rk45_adaptive(
            |_t, y| vec![-y[0]],
            0.0,
            &[1.0],
            2.0,
            0.1,
            1e-10,
            1e-10,
            10_000,
        );
        let (_, y_final) = res.trajectory.last().unwrap();
        assert_abs_diff_eq!(y_final[0], (-2.0_f64).exp(), epsilon = 1e-8);
        assert!(res.n_steps < 100, "should be efficient: {} steps", res.n_steps);
    }

    #[test]
    fn rk45_harmonic_oscillator() {
        // y'' = -y: y₁' = y₂, y₂' = -y₁
        // y(0) = 1, y'(0) = 0 → y(t) = cos(t)
        let res = rk45_adaptive(
            |_t, y| vec![y[1], -y[0]],
            0.0,
            &[1.0, 0.0],
            2.0 * std::f64::consts::PI,
            0.1,
            1e-10,
            1e-10,
            10_000,
        );
        let (_, y_final) = res.trajectory.last().unwrap();
        // cos(2π) = 1, sin(2π) = 0
        assert_abs_diff_eq!(y_final[0], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(y_final[1], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn rk45_lotka_volterra() {
        // Predator-prey: x' = αx − βxy, y' = δxy − γy
        let alpha = 1.0;
        let beta = 0.1;
        let delta = 0.075;
        let gamma = 1.5;
        let res = rk45_adaptive(
            move |_t, y| {
                vec![
                    alpha * y[0] - beta * y[0] * y[1],
                    delta * y[0] * y[1] - gamma * y[1],
                ]
            },
            0.0,
            &[10.0, 5.0],
            10.0,
            0.1,
            1e-8,
            1e-8,
            50_000,
        );
        // Just check it ran without blowing up and populations stay positive
        let (_, y_final) = res.trajectory.last().unwrap();
        assert!(y_final[0] > 0.0, "prey should be positive: {}", y_final[0]);
        assert!(y_final[1] > 0.0, "predator should be positive: {}", y_final[1]);
    }

    #[test]
    fn rk45_rejects_steps_for_stiff_problem() {
        // y' = -50y, stiff relative to the initial step size
        let res = rk45_adaptive(
            |_t, y| vec![-50.0 * y[0]],
            0.0,
            &[1.0],
            0.5,
            0.5, // too large initial step
            1e-8,
            1e-8,
            10_000,
        );
        let (_, y_final) = res.trajectory.last().unwrap();
        assert_abs_diff_eq!(y_final[0], (-25.0_f64).exp(), epsilon = 1e-4);
        // Should have rejected initial step(s)
        assert!(res.n_rejected > 0, "expected step rejections for stiff ODE");
    }
}
