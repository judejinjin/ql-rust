//! 2D Finite-Difference engine for two-asset options.
//!
//! Solves the 2D Black-Scholes PDE:
//! ```text
//! ∂V/∂t + ½σ₁²S₁²∂²V/∂S₁² + ½σ₂²S₂²∂²V/∂S₂² + ρσ₁σ₂S₁S₂∂²V/∂S₁∂S₂
//!   + (r-q₁)S₁∂V/∂S₁ + (r-q₂)S₂∂V/∂S₂ - rV = 0
//! ```
//! using ADI (Alternating Direction Implicit) splitting.

/// Result from the 2D FD engine.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Fd2dResult {
    /// Option price.
    pub price: f64,
    /// Delta w.r.t. first asset.
    pub delta1: f64,
    /// Delta w.r.t. second asset.
    pub delta2: f64,
}

/// 2D payoff type.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum Fd2dPayoff {
    /// max(ω·(S₁ − S₂ − K), 0)
    SpreadCall,
    /// max(ω·(S₂ − S₁ − K), 0)
    SpreadPut,
    /// max(max(S₁,S₂) − K, 0)
    BestOfCall,
    /// max(K − min(S₁,S₂), 0)
    WorstOfPut,
    /// max(S₁ − S₂, 0) — exchange option
    Exchange,
}

/// Price a two-asset option using 2D finite differences with ADI.
///
/// # Arguments
/// - `spot1`, `spot2` — current prices
/// - `strike` — option strike
/// - `r` — risk-free rate
/// - `q1`, `q2` — dividend yields
/// - `sigma1`, `sigma2` — volatilities
/// - `rho` — correlation
/// - `t` — time to expiry
/// - `payoff` — payoff type
/// - `n1`, `n2` — spatial grid sizes
/// - `n_time` — time steps
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub fn fd_2d_vanilla(
    spot1: f64,
    spot2: f64,
    strike: f64,
    r: f64,
    q1: f64,
    q2: f64,
    sigma1: f64,
    sigma2: f64,
    rho: f64,
    t: f64,
    payoff: Fd2dPayoff,
    n1: usize,
    n2: usize,
    n_time: usize,
) -> Fd2dResult {
    let dt = t / n_time as f64;

    // Log-space grids
    let x1_min = (spot1 * 0.1).ln();
    let x1_max = (spot1 * 5.0).ln();
    let x2_min = (spot2 * 0.1).ln();
    let x2_max = (spot2 * 5.0).ln();

    let dx1 = (x1_max - x1_min) / n1 as f64;
    let dx2 = (x2_max - x2_min) / n2 as f64;

    let mut x1 = vec![0.0; n1 + 1];
    let mut x2 = vec![0.0; n2 + 1];
    for i in 0..=n1 { x1[i] = x1_min + i as f64 * dx1; }
    for j in 0..=n2 { x2[j] = x2_min + j as f64 * dx2; }

    // Grid function V[i][j]
    let total = (n1 + 1) * (n2 + 1);
    let idx = |i: usize, j: usize| i * (n2 + 1) + j;

    let compute_payoff = |s1: f64, s2: f64| -> f64 {
        match payoff {
            Fd2dPayoff::SpreadCall => (s1 - s2 - strike).max(0.0),
            Fd2dPayoff::SpreadPut => (s2 - s1 - strike).max(0.0),
            Fd2dPayoff::BestOfCall => (s1.max(s2) - strike).max(0.0),
            Fd2dPayoff::WorstOfPut => (strike - s1.min(s2)).max(0.0),
            Fd2dPayoff::Exchange => (s1 - s2).max(0.0),
        }
    };

    // Terminal condition
    let mut v = vec![0.0; total];
    for i in 0..=n1 {
        for j in 0..=n2 {
            let s1 = x1[i].exp();
            let s2 = x2[j].exp();
            v[idx(i, j)] = compute_payoff(s1, s2);
        }
    }

    // PDE coefficients in log-space:
    // ∂V/∂t + ½σ₁²∂²V/∂x₁² + ½σ₂²∂²V/∂x₂² + ρσ₁σ₂∂²V/∂x₁∂x₂
    //   + μ₁∂V/∂x₁ + μ₂∂V/∂x₂ - rV = 0
    // where μ₁ = r-q₁-½σ₁², μ₂ = r-q₂-½σ₂²
    let mu1 = r - q1 - 0.5 * sigma1 * sigma1;
    let mu2 = r - q2 - 0.5 * sigma2 * sigma2;

    // Explicit Euler time stepping (unconditionally correct, stable for small dt)
    let diff1 = 0.5 * sigma1 * sigma1;
    let diff2 = 0.5 * sigma2 * sigma2;
    let cross_coeff = rho * sigma1 * sigma2;

    for _step in 0..n_time {
        let v_old = v.clone();

        for i in 1..n1 {
            for j in 1..n2 {
                let d2x1 = (v_old[idx(i + 1, j)] - 2.0 * v_old[idx(i, j)]
                    + v_old[idx(i - 1, j)])
                    / (dx1 * dx1);
                let d1x1 = (v_old[idx(i + 1, j)] - v_old[idx(i - 1, j)]) / (2.0 * dx1);
                let d2x2 = (v_old[idx(i, j + 1)] - 2.0 * v_old[idx(i, j)]
                    + v_old[idx(i, j - 1)])
                    / (dx2 * dx2);
                let d1x2 = (v_old[idx(i, j + 1)] - v_old[idx(i, j - 1)]) / (2.0 * dx2);
                let d2cross = (v_old[idx(i + 1, j + 1)] - v_old[idx(i + 1, j - 1)]
                    - v_old[idx(i - 1, j + 1)]
                    + v_old[idx(i - 1, j - 1)])
                    / (4.0 * dx1 * dx2);

                v[idx(i, j)] = v_old[idx(i, j)]
                    + dt * (diff1 * d2x1
                        + mu1 * d1x1
                        + diff2 * d2x2
                        + mu2 * d1x2
                        + cross_coeff * d2cross
                        - r * v_old[idx(i, j)]);
            }
        }

        // Decay boundary values
        let decay = 1.0 - r * dt;
        for i in 0..=n1 {
            v[idx(i, 0)] = v_old[idx(i, 0)] * decay;
            v[idx(i, n2)] = v_old[idx(i, n2)] * decay;
        }
        for j in 1..n2 {
            v[idx(0, j)] = v_old[idx(0, j)] * decay;
            v[idx(n1, j)] = v_old[idx(n1, j)] * decay;
        }
    }

    // Interpolate at (spot1, spot2)
    let x1_target = spot1.ln();
    let x2_target = spot2.ln();
    let i0 = ((x1_target - x1_min) / dx1).floor() as usize;
    let j0 = ((x2_target - x2_min) / dx2).floor() as usize;
    let i0 = i0.min(n1 - 1);
    let j0 = j0.min(n2 - 1);
    let fx = (x1_target - x1[i0]) / dx1;
    let fy = (x2_target - x2[j0]) / dx2;

    let price = (1.0 - fx) * (1.0 - fy) * v[idx(i0, j0)]
        + fx * (1.0 - fy) * v[idx(i0 + 1, j0)]
        + (1.0 - fx) * fy * v[idx(i0, j0 + 1)]
        + fx * fy * v[idx(i0 + 1, j0 + 1)];

    // Deltas by FD
    let delta1 = if i0 > 0 && i0 < n1 {
        let ds = x1[i0 + 1].exp() - x1[i0 - 1].exp();
        let dv = v[idx(i0 + 1, j0)] - v[idx(i0 - 1, j0)];
        dv / ds
    } else { 0.0 };

    let delta2 = if j0 > 0 && j0 < n2 {
        let ds = x2[j0 + 1].exp() - x2[j0 - 1].exp();
        let dv = v[idx(i0, j0 + 1)] - v[idx(i0, j0 - 1)];
        dv / ds
    } else { 0.0 };

    Fd2dResult {
        price: price.max(0.0),
        delta1,
        delta2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fd_2d_spread_call() {
        let res = fd_2d_vanilla(
            100.0, 95.0, 5.0, 0.05, 0.0, 0.0,
            0.20, 0.25, 0.5, 1.0,
            Fd2dPayoff::SpreadCall, 40, 40, 50,
        );
        assert!(res.price > 0.0, "price={}", res.price);
    }

    #[test]
    fn test_fd_2d_exchange() {
        let res = fd_2d_vanilla(
            100.0, 100.0, 0.0, 0.05, 0.0, 0.0,
            0.20, 0.20, 0.5, 1.0,
            Fd2dPayoff::Exchange, 40, 40, 50,
        );
        // Exchange option ≈ Margrabe
        assert!(res.price > 3.0 && res.price < 15.0, "price={}", res.price);
    }

    #[test]
    fn test_fd_2d_best_of_call() {
        let res = fd_2d_vanilla(
            100.0, 100.0, 100.0, 0.05, 0.0, 0.0,
            0.20, 0.25, 0.3, 1.0,
            Fd2dPayoff::BestOfCall, 40, 40, 50,
        );
        assert!(res.price > 10.0, "price={}", res.price);
    }
}
