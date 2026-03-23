//! Finite-difference Black-Scholes barrier engine and rebate engine.
//!
//! - [`fd_bs_barrier`] — FD pricing for single-barrier options under BS dynamics
//! - [`fd_bs_rebate`] — FD pricing for barrier rebate options
//!
//! Uses Crank-Nicolson on a log-spot grid.

/// Barrier type for FD barrier engine.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum FdBsBarrierType {
    /// Down And Out.
    DownAndOut,
    /// Up And Out.
    UpAndOut,
    /// Down And In.
    DownAndIn,
    /// Up And In.
    UpAndIn,
}

/// Result from the FD BS barrier engine.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FdBsBarrierResult {
    /// Option price.
    pub price: f64,
    /// Delta.
    pub delta: f64,
    /// Gamma.
    pub gamma: f64,
}

/// Price a barrier option by solving the Black-Scholes PDE with
/// Dirichlet boundary conditions at the barrier.
///
/// Uses Crank-Nicolson time stepping on a log-spot grid.
///
/// # Arguments
/// - Standard BS option inputs
/// - `barrier` — barrier level
/// - `rebate` — cash rebate paid if barrier is hit
/// - `barrier_type` — type of barrier
/// - `n_space` — spatial grid size
/// - `n_time` — time steps
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub fn fd_bs_barrier(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    barrier: f64,
    rebate: f64,
    barrier_type: FdBsBarrierType,
    is_call: bool,
    n_space: usize,
    n_time: usize,
) -> FdBsBarrierResult {
    let omega = if is_call { 1.0 } else { -1.0 };
    let dt = t / n_time as f64;

    // Log-spot grid
    let x_barrier = barrier.ln();
    let x_spot = spot.ln();

    let (x_min, x_max) = match barrier_type {
        FdBsBarrierType::DownAndOut | FdBsBarrierType::DownAndIn => {
            (x_barrier, (spot * 5.0).ln().max(x_barrier + 5.0 * sigma * t.sqrt()))
        }
        FdBsBarrierType::UpAndOut | FdBsBarrierType::UpAndIn => {
            ((spot * 0.1).ln().min(x_barrier - 5.0 * sigma * t.sqrt()), x_barrier)
        }
    };

    let dx = (x_max - x_min) / n_space as f64;
    let n = n_space;

    // Build grid
    let mut x = vec![0.0; n + 1];
    for i in 0..=n {
        x[i] = x_min + i as f64 * dx;
    }

    // Terminal condition for knock-out
    let mut u = vec![0.0; n + 1];
    let is_knockout = matches!(barrier_type, FdBsBarrierType::DownAndOut | FdBsBarrierType::UpAndOut);

    for i in 0..=n {
        let s_i = x[i].exp();
        if is_knockout {
            u[i] = (omega * (s_i - strike)).max(0.0);
        } else {
            // Knock-in: will compute as vanilla - knock-out later
            u[i] = (omega * (s_i - strike)).max(0.0);
        }
    }

    // Crank-Nicolson coefficients for BS PDE in log-space:
    // ∂V/∂t + ½σ²∂²V/∂x² + (r-q-½σ²)∂V/∂x - rV = 0
    let mu = r - q - 0.5 * sigma * sigma;
    let alpha = 0.25 * dt * sigma * sigma / (dx * dx);
    let beta = 0.25 * dt * mu / dx;

    let mut lower = vec![0.0; n + 1];
    let mut diag_vec = vec![0.0; n + 1];
    let mut upper_vec = vec![0.0; n + 1];

    for i in 1..n {
        lower[i] = -(alpha - beta); // implicit part
        diag_vec[i] = 1.0 + 2.0 * alpha + 0.5 * dt * r;
        upper_vec[i] = -(alpha + beta);
    }
    diag_vec[0] = 1.0;
    diag_vec[n] = 1.0;

    // Time stepping
    let mut rhs_vec = vec![0.0; n + 1];

    for _step in 0..n_time {
        // Explicit part of Crank-Nicolson
        for i in 1..n {
            rhs_vec[i] = (alpha - beta) * u[i - 1]
                + (1.0 - 2.0 * alpha - 0.5 * dt * r) * u[i]
                + (alpha + beta) * u[i + 1];
        }

        // Barrier boundary condition
        match barrier_type {
            FdBsBarrierType::DownAndOut => {
                rhs_vec[0] = rebate * (-r * (_step as f64 + 1.0) * dt).exp();
                rhs_vec[n] = (omega * (x[n].exp() - strike)).max(0.0)
                    * (-r * (_step as f64 + 1.0) * dt).exp();
            }
            FdBsBarrierType::UpAndOut => {
                rhs_vec[0] = (omega * (x[0].exp() - strike)).max(0.0)
                    * (-r * (_step as f64 + 1.0) * dt).exp();
                rhs_vec[n] = rebate * (-r * (_step as f64 + 1.0) * dt).exp();
            }
            _ => {
                rhs_vec[0] = 0.0;
                rhs_vec[n] = (omega * (x[n].exp() - strike)).max(0.0)
                    * (-r * (_step as f64 + 1.0) * dt).exp();
            }
        }

        // Thomas algorithm
        let mut c = vec![0.0; n + 1];
        let mut d = vec![0.0; n + 1];
        c[0] = 0.0;
        d[0] = rhs_vec[0];
        for i in 1..=n {
            let m = diag_vec[i] - lower[i] * c[i - 1];
            c[i] = if i < n { upper_vec[i] / m } else { 0.0 };
            d[i] = (rhs_vec[i] - lower[i] * d[i - 1]) / m;
        }
        u[n] = d[n];
        for i in (0..n).rev() {
            u[i] = d[i] - c[i] * u[i + 1];
        }
    }

    // For knock-in: price = vanilla - knock-out
    if !is_knockout {
        // Compute vanilla price on the same grid
        let mut vanilla = vec![0.0; n + 1];
        for i in 0..=n {
            vanilla[i] = (omega * (x[i].exp() - strike)).max(0.0);
        }
        // We already computed KO in u, but for KI we need a separate vanilla solve
        // Instead, use in-out parity: KI = vanilla - KO
        // Run another solve for vanilla (no barrier)
        let mut v = vanilla.clone();
        for _step in 0..n_time {
            for i in 1..n {
                rhs_vec[i] = (alpha - beta) * v[i - 1]
                    + (1.0 - 2.0 * alpha - 0.5 * dt * r) * v[i]
                    + (alpha + beta) * v[i + 1];
            }
            rhs_vec[0] = (omega * (x[0].exp() - strike)).max(0.0) * (-r * (_step as f64 + 1.0) * dt).exp();
            rhs_vec[n] = (omega * (x[n].exp() - strike)).max(0.0) * (-r * (_step as f64 + 1.0) * dt).exp();
            
            let mut c = vec![0.0; n + 1];
            let mut d = vec![0.0; n + 1];
            c[0] = 0.0;
            d[0] = rhs_vec[0];
            for i in 1..=n {
                let m = diag_vec[i] - lower[i] * c[i - 1];
                c[i] = if i < n { upper_vec[i] / m } else { 0.0 };
                d[i] = (rhs_vec[i] - lower[i] * d[i - 1]) / m;
            }
            v[n] = d[n];
            for i in (0..n).rev() {
                v[i] = d[i] - c[i] * v[i + 1];
            }
        }
        // KI = vanilla - KO
        for i in 0..=n {
            u[i] = v[i] - u[i];
        }
    }

    // Interpolate at spot
    let idx = ((x_spot - x_min) / dx).floor() as i64;
    let idx = idx.clamp(0, (n as i64) - 1) as usize;
    let frac = (x_spot - x[idx]) / dx;
    let price = u[idx] * (1.0 - frac) + u[idx + 1] * frac;

    // Delta by FD
    let delta = if idx > 0 && idx < n {
        let ds = x[idx + 1].exp() - x[idx - 1].exp();
        (u[idx + 1] - u[idx - 1]) / ds
    } else {
        0.0
    };

    // Gamma
    let gamma = if idx > 0 && idx < n {
        let ds = x[idx + 1].exp() - x[idx].exp();
        let ds2 = x[idx].exp() - x[idx - 1].exp();
        let d2u = (u[idx + 1] - 2.0 * u[idx] + u[idx - 1]) / (0.5 * (ds + ds2));
        d2u / (0.5 * (ds + ds2))
    } else {
        0.0
    };

    FdBsBarrierResult {
        price: price.max(0.0),
        delta,
        gamma,
    }
}

/// Price a barrier rebate option using finite differences.
///
/// The rebate option pays a fixed amount when (and only when) the barrier
/// is crossed. This is the pure rebate component of a barrier option.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub fn fd_bs_rebate(
    spot: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    barrier: f64,
    rebate: f64,
    is_down: bool,
    n_space: usize,
    n_time: usize,
) -> FdBsBarrierResult {
    // Rebate pricing: solve BS PDE with boundary V = rebate at barrier
    // and V = 0 at the other boundary (far field)
    let dt = t / n_time as f64;
    let x_barrier = barrier.ln();
    let x_spot = spot.ln();

    let (x_min, x_max) = if is_down {
        (x_barrier, (spot * 5.0).ln().max(x_barrier + 5.0 * sigma * t.sqrt()))
    } else {
        ((spot * 0.1).ln().min(x_barrier - 5.0 * sigma * t.sqrt()), x_barrier)
    };

    let dx = (x_max - x_min) / n_space as f64;
    let n = n_space;
    let mut x = vec![0.0; n + 1];
    for i in 0..=n {
        x[i] = x_min + i as f64 * dx;
    }

    // Terminal condition: 0 (rebate paid at barrier hit, not at expiry)
    let mut u = vec![0.0; n + 1];

    let mu = r - q - 0.5 * sigma * sigma;
    let alpha = 0.25 * dt * sigma * sigma / (dx * dx);
    let beta = 0.25 * dt * mu / dx;

    let mut lower = vec![0.0; n + 1];
    let mut diag_vec = vec![0.0; n + 1];
    let mut upper_vec = vec![0.0; n + 1];

    for i in 1..n {
        lower[i] = -(alpha - beta);
        diag_vec[i] = 1.0 + 2.0 * alpha + 0.5 * dt * r;
        upper_vec[i] = -(alpha + beta);
    }
    diag_vec[0] = 1.0;
    diag_vec[n] = 1.0;

    let mut rhs_vec = vec![0.0; n + 1];

    for _step in 0..n_time {
        for i in 1..n {
            rhs_vec[i] = (alpha - beta) * u[i - 1]
                + (1.0 - 2.0 * alpha - 0.5 * dt * r) * u[i]
                + (alpha + beta) * u[i + 1];
        }

        if is_down {
            rhs_vec[0] = rebate * (-r * (_step as f64 + 1.0) * dt).exp();
            rhs_vec[n] = 0.0;
        } else {
            rhs_vec[0] = 0.0;
            rhs_vec[n] = rebate * (-r * (_step as f64 + 1.0) * dt).exp();
        }

        // Thomas algorithm
        let mut c = vec![0.0; n + 1];
        let mut d = vec![0.0; n + 1];
        c[0] = 0.0;
        d[0] = rhs_vec[0];
        for i in 1..=n {
            let m = diag_vec[i] - lower[i] * c[i - 1];
            c[i] = if i < n { upper_vec[i] / m } else { 0.0 };
            d[i] = (rhs_vec[i] - lower[i] * d[i - 1]) / m;
        }
        u[n] = d[n];
        for i in (0..n).rev() {
            u[i] = d[i] - c[i] * u[i + 1];
        }
    }

    let idx = ((x_spot - x_min) / dx).floor() as i64;
    let idx = idx.clamp(0, (n as i64) - 1) as usize;
    let frac = (x_spot - x[idx]) / dx;
    let price = u[idx] * (1.0 - frac) + u[idx + 1] * frac;

    let delta = if idx > 0 && idx < n {
        (u[idx + 1] - u[idx - 1]) / (x[idx + 1].exp() - x[idx - 1].exp())
    } else {
        0.0
    };

    FdBsBarrierResult {
        price: price.max(0.0),
        delta,
        gamma: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_fd_bs_down_and_out_call() {
        let res = fd_bs_barrier(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            80.0, 0.0, FdBsBarrierType::DownAndOut, true,
            200, 200,
        );
        // DAOut call should be less than vanilla (~10.45)
        assert!(res.price > 5.0 && res.price < 12.0, "price={}", res.price);
    }

    #[test]
    fn test_fd_bs_up_and_out_put() {
        let res = fd_bs_barrier(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            120.0, 0.0, FdBsBarrierType::UpAndOut, false,
            200, 200,
        );
        assert!(res.price > 0.0 && res.price < 8.0, "price={}", res.price);
    }

    #[test]
    fn test_fd_bs_down_and_in_call() {
        // In-out parity: DI + DO = Vanilla
        let di = fd_bs_barrier(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            80.0, 0.0, FdBsBarrierType::DownAndIn, true,
            200, 200,
        );
        let do_opt = fd_bs_barrier(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            80.0, 0.0, FdBsBarrierType::DownAndOut, true,
            200, 200,
        );
        let vanilla = di.price + do_opt.price;
        // Should be close to BS call ≈ 10.45
        assert_abs_diff_eq!(vanilla, 10.45, epsilon = 1.5);
    }

    #[test]
    fn test_fd_bs_rebate_down() {
        let res = fd_bs_rebate(
            100.0, 0.05, 0.0, 0.20, 1.0,
            80.0, 10.0, true, 200, 200,
        );
        // Rebate should be positive but less than 10
        assert!(res.price > 0.0 && res.price < 10.0, "price={}", res.price);
    }
}
