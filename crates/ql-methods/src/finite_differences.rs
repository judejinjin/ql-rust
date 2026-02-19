//! Finite difference framework for option pricing.
//!
//! Implements a 1-D Crank-Nicolson PDE solver for Black-Scholes type problems.
//! Supports American exercise via a penalty/projected SOR approach.

use tracing::{info, info_span};

/// Result from a finite difference calculation.
#[derive(Debug, Clone)]
pub struct FDResult {
    /// Net present value.
    pub npv: f64,
    /// Delta (∂V/∂S).
    pub delta: f64,
    /// Gamma (∂²V/∂S²).
    pub gamma: f64,
    /// Theta (∂V/∂t), estimated from time step.
    pub theta: f64,
}

/// Price a European or American option using Crank-Nicolson finite differences.
///
/// Solves the Black-Scholes PDE:
///   ∂V/∂t + ½σ²S²∂²V/∂S² + (r-q)S·∂V/∂S − rV = 0
///
/// on a log-transformed uniform grid for numerical stability.
#[allow(clippy::too_many_arguments)]
pub fn fd_black_scholes(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    vol: f64,
    time_to_expiry: f64,
    is_call: bool,
    is_american: bool,
    num_space: usize,
    num_time: usize,
) -> FDResult {
    let _span = info_span!("fd_black_scholes", num_space, num_time, is_american).entered();
    // Transform to log-space: x = ln(S)
    let x0 = spot.ln();
    let sig2 = vol * vol;

    // Grid boundaries in log-space: ±5σ√T from ATM
    let x_spread = 5.0 * vol * time_to_expiry.sqrt();
    let x_min = x0 - x_spread.max(2.0);
    let x_max = x0 + x_spread.max(2.0);
    let dx = (x_max - x_min) / num_space as f64;
    let dt = time_to_expiry / num_time as f64;

    let n = num_space + 1; // grid points

    // Build spatial grid
    let x_grid: Vec<f64> = (0..n).map(|i| x_min + i as f64 * dx).collect();

    // Terminal condition: payoff at maturity
    let omega: f64 = if is_call { 1.0 } else { -1.0 };
    let mut v: Vec<f64> = x_grid
        .iter()
        .map(|&x| (omega * (x.exp() - strike)).max(0.0))
        .collect();

    // Coefficients for the PDE in log-space:
    //   ½σ² ∂²V/∂x² + (r - q - ½σ²) ∂V/∂x − rV + ∂V/∂t = 0
    let alpha = 0.5 * sig2;
    let beta = r - q - 0.5 * sig2;

    // Crank-Nicolson tridiagonal coefficients
    // CN: (I - ½dt·A) V^{n} = (I + ½dt·A) V^{n+1}
    // where A_{ij} discretizes the spatial operator.
    //
    // A V_i = alpha/dx² (V_{i+1} - 2V_i + V_{i-1})
    //       + beta/(2dx) (V_{i+1} - V_{i-1})
    //       - r V_i
    let a_coeff = alpha / (dx * dx); // coefficient of V_{i-1} and V_{i+1} (second deriv part)
    let b_coeff = beta / (2.0 * dx); // coefficient contribution from first derivative

    // For interior point i:
    //   lower = dt/2 * (a_coeff - b_coeff)    [coefficient of V_{i-1}]
    //   diag  = 1 + dt/2 * (2*a_coeff + r)    [coefficient of V_i on LHS]
    //   upper = dt/2 * (a_coeff + b_coeff)     [coefficient of V_{i+1} on LHS]
    //
    //   RHS uses opposite signs for the explicit half.
    let lower = 0.5 * dt * (a_coeff - b_coeff);
    let diag_lhs = 1.0 + 0.5 * dt * (2.0 * a_coeff + r);
    let upper = 0.5 * dt * (a_coeff + b_coeff);

    let diag_rhs = 1.0 - 0.5 * dt * (2.0 * a_coeff + r);

    // Intrinsic value for American exercise
    let intrinsic: Vec<f64> = x_grid
        .iter()
        .map(|&x| (omega * (x.exp() - strike)).max(0.0))
        .collect();

    // Time stepping (backwards from T to 0)
    let mut rhs = vec![0.0; n];

    for _step in 0..num_time {
        // Build RHS = (I + ½dt·A) V^{n+1}
        rhs[0] = v[0]; // boundary
        rhs[n - 1] = v[n - 1]; // boundary
        for i in 1..n - 1 {
            rhs[i] = lower * v[i - 1] + diag_rhs * v[i] + upper * v[i + 1];
        }

        // Boundary conditions at x_min and x_max
        if is_call {
            // For large S (x_max): V ≈ S·e^{-qT_rem} - K·e^{-rT_rem}
            // For small S (x_min): V ≈ 0
            rhs[0] = 0.0;
            rhs[n - 1] = (x_grid[n - 1].exp() - strike * (-r * dt).exp()).max(0.0);
        } else {
            // For large S (x_max): V ≈ 0
            // For small S (x_min): V ≈ K·e^{-rT_rem} - S
            rhs[0] = (strike * (-r * dt).exp() - x_grid[0].exp()).max(0.0);
            rhs[n - 1] = 0.0;
        }

        // Solve tridiagonal system: (I - ½dt·A) V^{n} = rhs
        // Using Thomas algorithm
        let mut cp = vec![0.0; n];
        let mut dp = vec![0.0; n];

        // Forward sweep
        cp[0] = 0.0;
        dp[0] = rhs[0];
        for i in 1..n - 1 {
            let denom = diag_lhs + lower * cp[i - 1];
            cp[i] = -upper / denom;
            dp[i] = (rhs[i] + lower * dp[i - 1]) / denom;
        }
        dp[n - 1] = rhs[n - 1];
        cp[n - 1] = 0.0;

        // Back substitution
        v[n - 1] = dp[n - 1];
        for i in (1..n - 1).rev() {
            v[i] = dp[i] - cp[i] * v[i + 1];
        }
        v[0] = dp[0];

        // American exercise constraint
        if is_american {
            for i in 0..n {
                v[i] = v[i].max(intrinsic[i]);
            }
        }
    }

    // Interpolate to find value at spot
    // Find the grid index closest to x0
    let idx = ((x0 - x_min) / dx).floor() as usize;
    let idx = idx.min(n - 3).max(1); // ensure we can compute delta/gamma

    // Quadratic interpolation for value, delta, gamma
    let x2 = x_grid[idx];
    let v1 = v[idx - 1];
    let v2 = v[idx];
    let v3 = v[idx + 1];

    // Linear interpolation between idx and idx+1
    let frac = (x0 - x2) / dx;
    let npv = v2 + frac * (v3 - v2);

    // Delta = (∂V/∂S) = (1/S) · ∂V/∂x
    let dv_dx = (v3 - v1) / (2.0 * dx);
    let delta = dv_dx / spot;

    // Gamma = (∂²V/∂S²) = (1/S²) (∂²V/∂x² − ∂V/∂x)
    let d2v_dx2 = (v3 - 2.0 * v2 + v1) / (dx * dx);
    let gamma = (d2v_dx2 - dv_dx) / (spot * spot);

    // Theta (rough estimate from the PDE)
    let theta = -(0.5 * sig2 * spot * spot * gamma
        + (r - q) * spot * delta
        - r * npv);

    let result = FDResult {
        npv,
        delta,
        gamma,
        theta,
    };
    info!(npv = result.npv, delta = result.delta, "FD solver complete");
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn fd_european_call_vs_bs() {
        // BS price for S=100, K=100, r=5%, q=0, σ=20%, T=1: ~10.45
        let result = fd_black_scholes(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0,
            true, false, 500, 500,
        );
        assert_abs_diff_eq!(result.npv, 10.45, epsilon = 0.15);
    }

    #[test]
    fn fd_european_put_vs_bs() {
        // BS put: ~5.57
        let result = fd_black_scholes(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0,
            false, false, 500, 500,
        );
        assert_abs_diff_eq!(result.npv, 5.57, epsilon = 0.15);
    }

    #[test]
    fn fd_american_put_exceeds_european() {
        let euro = fd_black_scholes(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0,
            false, false, 300, 300,
        );
        let amer = fd_black_scholes(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0,
            false, true, 300, 300,
        );
        assert!(
            amer.npv >= euro.npv - 0.01,
            "American put {} should be >= European put {}",
            amer.npv,
            euro.npv
        );
    }

    #[test]
    fn fd_deep_itm_put_near_intrinsic() {
        let result = fd_black_scholes(
            50.0, 100.0, 0.05, 0.0, 0.2, 1.0,
            false, true, 300, 300,
        );
        let intrinsic = 100.0 - 50.0;
        assert!(
            result.npv > intrinsic * 0.95,
            "Deep ITM American put {} should be near intrinsic {}",
            result.npv,
            intrinsic
        );
    }

    #[test]
    fn fd_delta_call_in_0_1() {
        let result = fd_black_scholes(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0,
            true, false, 500, 500,
        );
        assert!(
            result.delta > 0.0 && result.delta < 1.0,
            "Call delta {} should be in (0,1)",
            result.delta
        );
        // ATM call delta ≈ 0.6
        assert_abs_diff_eq!(result.delta, 0.6, epsilon = 0.1);
    }

    #[test]
    fn fd_gamma_positive() {
        let result = fd_black_scholes(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0,
            true, false, 500, 500,
        );
        assert!(result.gamma > 0.0, "Gamma should be positive, got {}", result.gamma);
    }

    #[test]
    fn fd_put_call_parity() {
        let call = fd_black_scholes(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0,
            true, false, 500, 500,
        );
        let put = fd_black_scholes(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0,
            false, false, 500, 500,
        );
        // C - P = S - K*e^{-rT}
        let lhs = call.npv - put.npv;
        let rhs = 100.0 - 100.0 * (-0.05_f64).exp();
        assert_abs_diff_eq!(lhs, rhs, epsilon = 0.3);
    }
}
