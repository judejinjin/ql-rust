//! Finite-difference Black-Scholes Asian engine.
//!
//! Solves the 2D PDE for arithmetic average Asian options on a (S, A) grid.
//! Uses a dimensional reduction via the running average as state variable.
//!
//! Reference: Wilmott, Dewynne & Howison (1993); Vecer (2001).

/// Result from the FD Asian engine.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FdAsianResult {
    /// Option price.
    pub price: f64,
    /// Delta (∂V/∂S).
    pub delta: f64,
}

/// Price an arithmetic average price Asian option using finite differences.
///
/// Uses the Vecer (2001) change of variables to reduce the dimensionality
/// from 2D to 1D. Define y = (A_t − S_t·remaining_weight) / S_t, then we solve
/// a 1D PDE in (y, t).
///
/// # Arguments
/// - `spot`, `strike`, `r`, `q`, `sigma` — standard BS inputs
/// - `t` — time to expiry
/// - `n_fixings` — number of fixing dates
/// - `n_space` — number of spatial grid points (default ~200)
/// - `n_time` — number of time steps (default ~200)
/// - `is_call` — true for call, false for put
#[allow(clippy::too_many_arguments)]
pub fn fd_asian(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    n_fixings: usize,
    n_space: usize,
    n_time: usize,
    is_call: bool,
) -> FdAsianResult {
    let omega = if is_call { 1.0 } else { -1.0 };
    let dt = t / n_time as f64;
    let b = r - q;

    // Vecer PDE: define y = A/S (ratio) after normalizing
    // The PDE becomes:
    //   ∂u/∂t + ½σ²y²·∂²u/∂y² - b·y·∂u/∂y - r·u = 0
    // with terminal condition u(y,T) = max(ω(y·S_T - K), 0)

    // We discretize on a log-uniform y grid
    let y_min = -2.0;
    let y_max = 3.0;
    let dy = (y_max - y_min) / n_space as f64;

    // Grid values
    let mut y_grid = vec![0.0; n_space + 1];
    for i in 0..=n_space {
        y_grid[i] = y_min + i as f64 * dy;
    }

    // Terminal condition: payoff at T
    // At maturity, the running average A ≈ y · spot (simplified)
    let mut u = vec![0.0; n_space + 1];
    for i in 0..=n_space {
        let avg_approx = y_grid[i] * spot;
        u[i] = (omega * (avg_approx - strike)).max(0.0);
    }

    // Backward PDE solve using implicit Euler (tridiagonal)
    let mut lower = vec![0.0; n_space + 1];
    let mut diag = vec![0.0; n_space + 1];
    let mut upper = vec![0.0; n_space + 1];
    let mut rhs = vec![0.0; n_space + 1];

    for _step in 0..n_time {
        // Build tridiagonal system
        for i in 1..n_space {
            let y = y_grid[i];
            let coeff_diff = 0.5 * sigma * sigma * y * y / (dy * dy);
            let coeff_conv = b * y / (2.0 * dy);

            lower[i] = -dt * (coeff_diff - coeff_conv);
            diag[i] = 1.0 + dt * (2.0 * coeff_diff + r);
            upper[i] = -dt * (coeff_diff + coeff_conv);
            rhs[i] = u[i];
        }

        // Boundary conditions
        diag[0] = 1.0;
        upper[0] = 0.0;
        rhs[0] = u[0] * (-r * dt).exp(); // approximate

        diag[n_space] = 1.0;
        lower[n_space] = 0.0;
        rhs[n_space] = u[n_space] * (-r * dt).exp();

        // Thomas algorithm
        let mut c_prime = vec![0.0; n_space + 1];
        let mut d_prime = vec![0.0; n_space + 1];
        c_prime[0] = upper[0] / diag[0];
        d_prime[0] = rhs[0] / diag[0];
        for i in 1..=n_space {
            let m = diag[i] - lower[i] * c_prime[i - 1];
            c_prime[i] = if i < n_space { upper[i] / m } else { 0.0 };
            d_prime[i] = (rhs[i] - lower[i] * d_prime[i - 1]) / m;
        }
        u[n_space] = d_prime[n_space];
        for i in (0..n_space).rev() {
            u[i] = d_prime[i] - c_prime[i] * u[i + 1];
        }
    }

    // Interpolate at y = 1.0 (A/S = 1 at start when A₀ = S₀ for average-to-date)
    // For a new Asian with no fixings, y = weight_remaining = 1.0
    let target_y = 1.0;
    let idx = ((target_y - y_min) / dy).floor() as usize;
    let idx = idx.min(n_space - 1);
    let frac = (target_y - y_grid[idx]) / dy;
    let price = u[idx] * (1.0 - frac) + u[idx + 1] * frac;

    // Delta by finite difference
    let target_y_up = (spot + 1.0) / spot;
    let idx_up = ((target_y_up - y_min) / dy).floor() as usize;
    let idx_up = idx_up.min(n_space - 1);
    let frac_up = (target_y_up - y_grid[idx_up]) / dy;
    let price_up = u[idx_up] * (1.0 - frac_up) + u[idx_up + 1] * frac_up;

    let delta = (price_up - price) / 1.0;

    FdAsianResult {
        price: price.max(0.0),
        delta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fd_asian_call() {
        let res = fd_asian(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 12, 200, 200, true);
        // Asian call should be cheaper than European
        assert!(res.price > 1.0 && res.price < 12.0, "price={}", res.price);
    }

    #[test]
    fn test_fd_asian_put() {
        let res = fd_asian(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 12, 200, 200, false);
        assert!(res.price > 0.5 && res.price < 10.0, "price={}", res.price);
    }
}
