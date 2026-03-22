//! Monte Carlo pathwise Greeks via AAD.
//!
//! Each path records its computation on the thread-local tape, then the
//! adjoint pass extracts ∂payoff/∂θ for every model parameter θ. This gives
//! all first-order Greeks in a single backward sweep per path.
//!
//! Random samples (`z ~ N(0,1)`) remain `f64` — they are not AD inputs.
//! Model parameters (spot, vol, rates, etc.) are registered as tape inputs
//! and their sensitivities are accumulated across paths.
//!
//! # Example
//!
//! ```
//! use ql_aad::mc::{mc_european_aad, McEuropeanGreeks};
//! use ql_aad::OptionKind;
//!
//! let greeks = mc_european_aad(100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
//!                              OptionKind::Call, 50_000, 42);
//! assert!((greeks.npv - 10.45).abs() < 0.5); // ~BS price
//! assert!((greeks.delta - 0.637).abs() < 0.05);
//! ```

use crate::bs::OptionKind;
use crate::number::Number;
use crate::tape::{adjoint_tl, with_tape, AReal};

use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

// ===========================================================================
// Result types
// ===========================================================================

/// First-order Greeks from MC European with AAD.
#[derive(Debug, Clone)]
pub struct McEuropeanGreeks {
    /// Option price (discounted expected payoff).
    pub npv: f64,
    /// Monte Carlo standard error on NPV.
    pub std_error: f64,
    /// ∂V/∂S — delta.
    pub delta: f64,
    /// ∂V/∂σ — vega.
    pub vega: f64,
    /// ∂V/∂r — rho.
    pub rho: f64,
    /// ∂V/∂q — dividend rho.
    pub div_rho: f64,
    /// Number of paths used.
    pub num_paths: usize,
}

/// First-order Greeks from MC Heston with AAD.
#[derive(Debug, Clone)]
pub struct McHestonGreeks {
    /// Option price.
    pub npv: f64,
    /// Monte Carlo standard error.
    pub std_error: f64,
    /// ∂V/∂S₀ — spot delta.
    pub delta: f64,
    /// ∂V/∂v₀ — initial variance sensitivity (≈ vega).
    pub vega_v0: f64,
    /// ∂V/∂κ — mean reversion speed sensitivity.
    pub d_kappa: f64,
    /// ∂V/∂θ — long-run variance sensitivity.
    pub d_theta: f64,
    /// ∂V/∂σ — vol-of-vol sensitivity.
    pub d_sigma: f64,
    /// ∂V/∂ρ — correlation sensitivity.
    pub d_rho: f64,
    /// ∂V/∂r — rho (interest rate sensitivity).
    pub rho: f64,
    /// ∂V/∂q — dividend yield sensitivity.
    pub div_rho: f64,
    /// Number of paths.
    pub num_paths: usize,
}

// ===========================================================================
// European MC with AAD (reverse-mode)
// ===========================================================================

/// Price a European option via Monte Carlo with pathwise AAD Greeks.
///
/// Uses exact 1-step log-normal simulation:
///   `S_T = spot · exp((r − q − σ²/2)·T + σ·√T·Z)`
///
/// All first-order Greeks (delta, vega, rho, div_rho) are extracted from the
/// reverse-mode adjoint in a single backward pass per path.
///
/// Supports antithetic variates (each `z` generates `+z` and `−z` paths).
#[allow(clippy::too_many_arguments)]
pub fn mc_european_aad(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    vol: f64,
    time_to_expiry: f64,
    option_kind: OptionKind,
    num_paths: usize,
    seed: u64,
) -> McEuropeanGreeks {
    let mut rng = SmallRng::seed_from_u64(seed);
    let half_paths = num_paths / 2; // antithetic pairs

    // Accumulators
    let mut sum_npv = 0.0;
    let mut sum_npv_sq = 0.0;
    let mut sum_delta = 0.0;
    let mut sum_vega = 0.0;
    let mut sum_rho = 0.0;
    let mut sum_div_rho = 0.0;

    let phi = match option_kind {
        OptionKind::Call => 1.0,
        OptionKind::Put => -1.0,
    };

    for _ in 0..half_paths {
        let z: f64 = StandardNormal.sample(&mut rng);

        // Process both z and -z (antithetic)
        for &zz in &[z, -z] {
            let (pv, adj) = with_tape(|tape| {
                let s = tape.input(spot);
                let r_ad = tape.input(r);
                let q_ad = tape.input(q);
                let v = tape.input(vol);

                let tau = AReal::from_f64(time_to_expiry);
                let half = AReal::from_f64(0.5);
                let sqrt_t = AReal::from_f64(time_to_expiry.sqrt());

                // S_T = s * exp((r - q - 0.5*v²)*tau + v*sqrt_t*z)
                let drift = (r_ad - q_ad - half * v * v) * tau;
                let diffusion = v * sqrt_t * AReal::from_f64(zz);
                let st = s * (drift + diffusion).exp();

                // Payoff
                let intrinsic = AReal::from_f64(phi) * (st - AReal::from_f64(strike));
                let payoff = intrinsic.max(AReal::zero());

                // Discount
                let disc = (-r_ad * tau).exp();
                let pv = payoff * disc;

                let adj = adjoint_tl(pv);
                (pv.val, adj)
            });

            sum_npv += pv;
            sum_npv_sq += pv * pv;
            // adj indices: 0=spot, 1=r, 2=q, 3=vol
            sum_delta += adj[0];
            sum_rho += adj[1];
            sum_div_rho += adj[2];
            sum_vega += adj[3];
        }
    }

    let n = (half_paths * 2) as f64;
    let mean = sum_npv / n;
    let variance = (sum_npv_sq / n - mean * mean).max(0.0);
    let std_error = (variance / n).sqrt();

    McEuropeanGreeks {
        npv: mean,
        std_error,
        delta: sum_delta / n,
        vega: sum_vega / n,
        rho: sum_rho / n,
        div_rho: sum_div_rho / n,
        num_paths: half_paths * 2,
    }
}

// ===========================================================================
// Heston MC with AAD (reverse-mode)
// ===========================================================================

/// Price a European option under Heston stochastic volatility via Monte Carlo
/// with pathwise AAD Greeks.
///
/// Uses log-Euler for spot and Euler with full truncation for variance:
/// ```text
///   d ln(S) = (r − q − v⁺/2)·dt + √v⁺ · √dt · Z₁
///   dv = κ(θ − v⁺)·dt + σ·√v⁺ · √dt · Z₂
///   v⁺ = max(v, 0)
/// ```
///
/// where `(Z₁, Z₂)` are correlated normals with correlation `ρ`.
///
/// All 8 first-order Greeks (delta, vega_v0, d_kappa, d_theta, d_sigma,
/// d_rho, rho, div_rho) are computed via a single adjoint pass per path.
#[allow(clippy::too_many_arguments)]
pub fn mc_heston_aad(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    time_to_expiry: f64,
    option_kind: OptionKind,
    num_paths: usize,
    num_steps: usize,
    seed: u64,
) -> McHestonGreeks {
    let mut rng = SmallRng::seed_from_u64(seed);
    let dt = time_to_expiry / num_steps as f64;
    let sqrt_dt = dt.sqrt();

    let phi = match option_kind {
        OptionKind::Call => 1.0,
        OptionKind::Put => -1.0,
    };

    // Accumulators: [delta, vega_v0, d_kappa, d_theta, d_sigma, d_rho, rho, div_rho]
    let mut sum_greeks = [0.0_f64; 8];
    let mut sum_npv = 0.0;
    let mut sum_npv_sq = 0.0;

    // Pre-allocate random number buffers outside the path loop to avoid
    // per-path heap allocations.
    let mut z1_vec = Vec::with_capacity(num_steps);
    let mut z2_vec = Vec::with_capacity(num_steps);

    for _ in 0..num_paths {
        // Reuse pre-allocated buffers
        z1_vec.clear();
        z2_vec.clear();
        for _ in 0..num_steps {
            let z1: f64 = StandardNormal.sample(&mut rng);
            let z2_indep: f64 = StandardNormal.sample(&mut rng);
            z1_vec.push(z1);
            z2_vec.push(z2_indep);
        }

        let (pv, adj) = with_tape(|tape| {
            // Register 8 inputs on the tape
            let s_ad = tape.input(spot);        // idx 0
            let r_ad = tape.input(r);           // idx 1
            let q_ad = tape.input(q);           // idx 2
            let v0_ad = tape.input(v0);         // idx 3
            let kappa_ad = tape.input(kappa);   // idx 4
            let theta_ad = tape.input(theta);   // idx 5
            let sigma_ad = tape.input(sigma);   // idx 6
            let rho_ad = tape.input(rho);       // idx 7

            let dt_c = AReal::from_f64(dt);
            let sqrt_dt_c = AReal::from_f64(sqrt_dt);
            let half = AReal::from_f64(0.5);

            let mut log_s = s_ad.ln();
            let mut v = v0_ad;

            // Correlated noise factor: sqrt(1 - rho²)
            let rho_comp = (AReal::one() - rho_ad * rho_ad).sqrt();

            for step in 0..num_steps {
                let z1_c = AReal::from_f64(z1_vec[step]);
                let z2_indep_c = AReal::from_f64(z2_vec[step]);

                let v_pos = v.max(AReal::zero());
                let sqrt_v = v_pos.sqrt();

                // Correlated z2
                let z2_c = rho_ad * z1_c + rho_comp * z2_indep_c;

                // Log-Euler for spot
                log_s = log_s + (r_ad - q_ad - half * v_pos) * dt_c
                    + sqrt_v * sqrt_dt_c * z1_c;

                // Euler for variance (full truncation)
                v = v + kappa_ad * (theta_ad - v_pos) * dt_c
                    + sigma_ad * sqrt_v * sqrt_dt_c * z2_c;
                v = v.max(AReal::zero());
            }

            let st = log_s.exp();
            let intrinsic = AReal::from_f64(phi) * (st - AReal::from_f64(strike));
            let payoff = intrinsic.max(AReal::zero());
            let disc = (-r_ad * AReal::from_f64(time_to_expiry)).exp();
            let pv = payoff * disc;

            let adj = adjoint_tl(pv);
            (pv.val, adj)
        });

        sum_npv += pv;
        sum_npv_sq += pv * pv;
        // Accumulate: adj[0..8] correspond to [spot, r, q, v0, kappa, theta, sigma, rho]
        for (i, g) in sum_greeks.iter_mut().enumerate() {
            *g += adj[i];
        }
    }

    let n = num_paths as f64;
    let mean = sum_npv / n;
    let variance = (sum_npv_sq / n - mean * mean).max(0.0);
    let std_error = (variance / n).sqrt();

    McHestonGreeks {
        npv: mean,
        std_error,
        delta: sum_greeks[0] / n,
        rho: sum_greeks[1] / n,
        div_rho: sum_greeks[2] / n,
        vega_v0: sum_greeks[3] / n,
        d_kappa: sum_greeks[4] / n,
        d_theta: sum_greeks[5] / n,
        d_sigma: sum_greeks[6] / n,
        d_rho: sum_greeks[7] / n,
        num_paths,
    }
}

// ===========================================================================
// Forward-mode European MC (DualVec)
// ===========================================================================

/// European MC with forward-mode AD using `DualVec<4>`.
///
/// Seeds: [0]=spot, [1]=vol, [2]=r, [3]=q.
/// Returns `McEuropeanGreeks` with delta, vega, rho, div_rho.
///
/// This is an alternative to `mc_european_aad` — forward-mode is simpler
/// (no tape) but costs 4× per path (one seed per input). For only 4 inputs,
/// forward-mode is competitive with reverse-mode.
#[allow(clippy::too_many_arguments)]
pub fn mc_european_forward(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    vol: f64,
    time_to_expiry: f64,
    option_kind: OptionKind,
    num_paths: usize,
    seed: u64,
) -> McEuropeanGreeks {
    use crate::dual_vec::DualVec;

    let mut rng = SmallRng::seed_from_u64(seed);
    let half_paths = num_paths / 2;

    let phi = match option_kind {
        OptionKind::Call => 1.0,
        OptionKind::Put => -1.0,
    };

    let mut sum_npv = 0.0;
    let mut sum_npv_sq = 0.0;
    let mut sum_greeks = [0.0_f64; 4]; // delta, vega, rho, div_rho

    // Seed inputs
    let s: DualVec<4> = DualVec::variable(spot, 0);
    let v: DualVec<4> = DualVec::variable(vol, 1);
    let r_ad: DualVec<4> = DualVec::variable(r, 2);
    let q_ad: DualVec<4> = DualVec::variable(q, 3);

    let tau = DualVec::<4>::from_f64(time_to_expiry);
    let sqrt_t = DualVec::<4>::from_f64(time_to_expiry.sqrt());
    let half = DualVec::<4>::from_f64(0.5);

    for _ in 0..half_paths {
        let z: f64 = StandardNormal.sample(&mut rng);

        for &zz in &[z, -z] {
            let drift = (r_ad - q_ad - half * v * v) * tau;
            let diffusion = v * sqrt_t * DualVec::<4>::from_f64(zz);
            let st = s * (drift + diffusion).exp();

            let intrinsic = DualVec::<4>::from_f64(phi) * (st - DualVec::<4>::from_f64(strike));
            let payoff = intrinsic.max(DualVec::<4>::zero());
            let disc = (-r_ad * tau).exp();
            let pv = payoff * disc;

            sum_npv += pv.val;
            sum_npv_sq += pv.val * pv.val;
            sum_greeks[0] += pv.dot[0]; // delta
            sum_greeks[1] += pv.dot[1]; // vega
            sum_greeks[2] += pv.dot[2]; // rho
            sum_greeks[3] += pv.dot[3]; // div_rho
        }
    }

    let n = (half_paths * 2) as f64;
    let mean = sum_npv / n;
    let variance = (sum_npv_sq / n - mean * mean).max(0.0);
    let std_error = (variance / n).sqrt();

    McEuropeanGreeks {
        npv: mean,
        std_error,
        delta: sum_greeks[0] / n,
        vega: sum_greeks[1] / n,
        rho: sum_greeks[2] / n,
        div_rho: sum_greeks[3] / n,
        num_paths: half_paths * 2,
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
    fn european_call_npv_near_bs() {
        // Compare MC NPV to analytic BS: C ≈ 10.45
        let g = mc_european_aad(100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
                                OptionKind::Call, 100_000, 42);
        assert!((g.npv - 10.45).abs() < 0.5, "npv={}", g.npv);
    }

    #[test]
    fn european_put_npv_near_bs() {
        // BS put ≈ 5.57
        let g = mc_european_aad(100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
                                OptionKind::Put, 100_000, 42);
        assert!((g.npv - 5.57).abs() < 0.5, "npv={}", g.npv);
    }

    #[test]
    fn european_delta_near_bs() {
        // BS delta ≈ 0.6368 for ATM call
        let g = mc_european_aad(100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
                                OptionKind::Call, 200_000, 42);
        assert!((g.delta - 0.637).abs() < 0.05, "delta={}", g.delta);
    }

    #[test]
    fn european_vega_positive_call() {
        let g = mc_european_aad(100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
                                OptionKind::Call, 100_000, 42);
        assert!(g.vega > 0.0, "vega should be positive, got {}", g.vega);
        // BS vega ≈ 37.5 (per unit vol, so for σ=0.20, ∂V/∂σ ≈ 37.5)
        assert!((g.vega - 37.5).abs() < 5.0, "vega={}", g.vega);
    }

    #[test]
    fn european_rho_positive_call() {
        let g = mc_european_aad(100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
                                OptionKind::Call, 100_000, 42);
        assert!(g.rho > 0.0, "rho should be positive for call, got {}", g.rho);
    }

    #[test]
    fn european_forward_matches_reverse() {
        // Forward-mode and reverse-mode should give same Greeks
        let rev = mc_european_aad(100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
                                  OptionKind::Call, 50_000, 42);
        let fwd = mc_european_forward(100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
                                      OptionKind::Call, 50_000, 42);
        // Same seed, same RNG → should be very close
        assert_abs_diff_eq!(rev.npv, fwd.npv, epsilon = 1e-10);
        assert_abs_diff_eq!(rev.delta, fwd.delta, epsilon = 1e-10);
        assert_abs_diff_eq!(rev.vega, fwd.vega, epsilon = 1e-10);
        assert_abs_diff_eq!(rev.rho, fwd.rho, epsilon = 1e-10);
    }

    #[test]
    fn european_delta_vs_bump() {
        // Verify delta via bump-and-reprice
        let bump = 0.01;
        let g_up = mc_european_aad(100.0 + bump, 100.0, 0.05, 0.0, 0.20, 1.0,
                                   OptionKind::Call, 100_000, 42);
        let g_dn = mc_european_aad(100.0 - bump, 100.0, 0.05, 0.0, 0.20, 1.0,
                                   OptionKind::Call, 100_000, 42);
        let bump_delta = (g_up.npv - g_dn.npv) / (2.0 * bump);

        let g = mc_european_aad(100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
                                OptionKind::Call, 100_000, 42);

        // Note: different seeds vs same seed means noise, but central tendency matches
        assert!((g.delta - bump_delta).abs() < 0.05, "AAD delta={}, bump delta={}", g.delta, bump_delta);
    }

    #[test]
    fn european_put_delta_negative() {
        let g = mc_european_aad(100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
                                OptionKind::Put, 100_000, 42);
        assert!(g.delta < 0.0, "put delta should be negative, got {}", g.delta);
    }

    #[test]
    fn heston_call_npv_positive() {
        let g = mc_heston_aad(100.0, 100.0, 0.05, 0.0,
                              0.04, 2.0, 0.04, 0.3, -0.7,
                              1.0, OptionKind::Call, 20_000, 50, 42);
        assert!(g.npv > 0.0, "Heston call NPV should be positive, got {}", g.npv);
        // Should be in the ballpark of BS with vol≈0.20 (sqrt(v0)=0.20)
        assert!((g.npv - 10.0).abs() < 5.0, "Heston npv={}", g.npv);
    }

    #[test]
    fn heston_delta_positive_call() {
        let g = mc_heston_aad(100.0, 100.0, 0.05, 0.0,
                              0.04, 2.0, 0.04, 0.3, -0.7,
                              1.0, OptionKind::Call, 20_000, 50, 42);
        assert!(g.delta > 0.0 && g.delta < 1.0,
                "Heston call delta should be in (0,1), got {}", g.delta);
    }

    #[test]
    fn heston_vega_v0_positive() {
        let g = mc_heston_aad(100.0, 100.0, 0.05, 0.0,
                              0.04, 2.0, 0.04, 0.3, -0.7,
                              1.0, OptionKind::Call, 20_000, 50, 42);
        assert!(g.vega_v0 > 0.0,
                "vega_v0 should be positive for call, got {}", g.vega_v0);
    }

    #[test]
    fn heston_greeks_all_finite() {
        let g = mc_heston_aad(100.0, 100.0, 0.05, 0.0,
                              0.04, 2.0, 0.04, 0.3, -0.7,
                              1.0, OptionKind::Call, 10_000, 50, 42);
        assert!(g.delta.is_finite(), "delta not finite");
        assert!(g.vega_v0.is_finite(), "vega_v0 not finite");
        assert!(g.d_kappa.is_finite(), "d_kappa not finite");
        assert!(g.d_theta.is_finite(), "d_theta not finite");
        assert!(g.d_sigma.is_finite(), "d_sigma not finite");
        assert!(g.d_rho.is_finite(), "d_rho not finite");
        assert!(g.rho.is_finite(), "rho not finite");
        assert!(g.div_rho.is_finite(), "div_rho not finite");
    }

    #[test]
    fn heston_npv_near_analytic() {
        // Compare MC Heston against semi-analytic Heston pricer
        use crate::heston::heston_price_generic;
        let analytic: f64 = heston_price_generic(
            100.0, 100.0, 0.05, 0.0,
            1.0, 0.04, 2.0, 0.04, 0.3, -0.7,
            true,  // is_call = true
        );
        let mc = mc_heston_aad(100.0, 100.0, 0.05, 0.0,
                               0.04, 2.0, 0.04, 0.3, -0.7,
                               1.0, OptionKind::Call, 50_000, 100, 42);
        // MC should be within a few std errors of analytic
        assert!((mc.npv - analytic).abs() < 3.0 * mc.std_error + 0.5,
                "MC npv={} vs analytic={}, std_err={}", mc.npv, analytic, mc.std_error);
    }
}
