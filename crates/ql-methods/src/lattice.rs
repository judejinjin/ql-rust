//! Lattice (binomial tree) framework for option pricing.
//!
//! Implements Cox-Ross-Rubinstein (CRR) binomial trees for
//! European and American option pricing.

use tracing::info_span;

/// Result from a lattice calculation.
#[derive(Debug, Clone)]
pub struct LatticeResult {
    /// Net present value.
    pub npv: f64,
    /// Delta (∂V/∂S), estimated from the first step.
    pub delta: f64,
    /// Gamma (∂²V/∂S²), estimated from the second step.
    pub gamma: f64,
    /// Theta (∂V/∂t), estimated from time step.
    pub theta: f64,
}

/// Price a European or American option using a CRR binomial tree.
///
/// The CRR tree uses:
///   u = exp(σ√Δt),  d = 1/u
///   p = (exp((r-q)Δt) - d) / (u - d)
///
/// and backwards induction with optional early exercise.
#[allow(clippy::too_many_arguments)]
pub fn binomial_crr(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    vol: f64,
    time_to_expiry: f64,
    is_call: bool,
    is_american: bool,
    num_steps: usize,
) -> LatticeResult {
    let _span = info_span!("binomial_crr", num_steps, is_american).entered();
    let dt = time_to_expiry / num_steps as f64;
    let df = (-r * dt).exp(); // discount factor per step
    let u = (vol * dt.sqrt()).exp(); // up factor
    let d = 1.0 / u; // down factor
    let growth = ((r - q) * dt).exp();
    let p = (growth - d) / (u - d); // risk-neutral probability of up move

    let omega: f64 = if is_call { 1.0 } else { -1.0 };

    // Build terminal payoffs: at step N, node j has spot = S₀ · u^j · d^(N-j)
    let n = num_steps;
    let mut values = vec![0.0; n + 1];

    for (j, val) in values.iter_mut().enumerate().take(n + 1) {
        let s_t = spot * u.powi(j as i32) * d.powi((n - j) as i32);
        *val = (omega * (s_t - strike)).max(0.0);
    }

    // For Greeks, save values at steps n, n-1, n-2
    let mut val_step_1 = [0.0; 3]; // values at step 1 (3 nodes needed for gamma)
    let mut val_step_2 = [0.0; 3]; // values at step 2

    // Backward induction
    for step in (0..n).rev() {
        let mut new_values = vec![0.0; step + 1];
        for j in 0..=step {
            let continuation = df * (p * values[j + 1] + (1.0 - p) * values[j]);

            new_values[j] = if is_american {
                let s_node = spot * u.powi(j as i32) * d.powi((step - j) as i32);
                let intrinsic = (omega * (s_node - strike)).max(0.0);
                continuation.max(intrinsic)
            } else {
                continuation
            };
        }

        if step == 2 && n >= 3 {
            val_step_2[0] = new_values[0]; // S*d²
            val_step_2[1] = new_values[1]; // S
            val_step_2[2] = new_values[2]; // S*u²
        }
        if step == 1 {
            val_step_1[0] = new_values[0]; // S*d
            val_step_1[1] = new_values[1]; // S*u
        }

        values = new_values;
    }

    let npv = values[0];

    // Greeks from the tree
    // Delta = (V(Su) - V(Sd)) / (Su - Sd) from step 1
    let su = spot * u;
    let sd = spot * d;
    let delta = if n >= 1 {
        (val_step_1[1] - val_step_1[0]) / (su - sd)
    } else {
        0.0
    };

    // Gamma from step 2
    let gamma = if n >= 3 {
        let su2 = spot * u * u;
        let sd2 = spot * d * d;
        let s_mid = spot; // middle node at step 2

        let delta_up = (val_step_2[2] - val_step_2[1]) / (su2 - s_mid);
        let delta_down = (val_step_2[1] - val_step_2[0]) / (s_mid - sd2);
        (delta_up - delta_down) / (0.5 * (su2 - sd2))
    } else {
        0.0
    };

    // Theta: (V(t+dt) - V(t)) / dt, using the center node at step 2
    let theta = if n >= 3 {
        (val_step_2[1] - npv) / (2.0 * dt)
    } else {
        0.0
    };

    LatticeResult {
        npv,
        delta,
        gamma,
        theta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn crr_european_call_vs_bs() {
        // BS price for S=100, K=100, r=5%, q=0, σ=20%, T=1: ~10.45
        let result = binomial_crr(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, true, false, 1000);
        assert_abs_diff_eq!(result.npv, 10.45, epsilon = 0.1);
    }

    #[test]
    fn crr_european_put_vs_bs() {
        // BS put: ~5.57
        let result = binomial_crr(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, false, false, 1000);
        assert_abs_diff_eq!(result.npv, 5.57, epsilon = 0.1);
    }

    #[test]
    fn crr_american_put_exceeds_european() {
        let euro = binomial_crr(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, false, false, 500);
        let amer = binomial_crr(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, false, true, 500);
        assert!(
            amer.npv >= euro.npv - 0.01,
            "American put {} should be >= European put {}",
            amer.npv,
            euro.npv
        );
    }

    #[test]
    fn crr_american_call_no_dividend_equals_european() {
        // With no dividends, American call = European call
        let euro = binomial_crr(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, true, false, 500);
        let amer = binomial_crr(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, true, true, 500);
        assert_abs_diff_eq!(amer.npv, euro.npv, epsilon = 0.01);
    }

    #[test]
    fn crr_delta_call_in_0_1() {
        let result = binomial_crr(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, true, false, 500);
        assert!(
            result.delta > 0.0 && result.delta < 1.0,
            "Call delta {} should be in (0,1)",
            result.delta
        );
    }

    #[test]
    fn crr_gamma_positive() {
        let result = binomial_crr(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, true, false, 500);
        assert!(result.gamma > 0.0, "Gamma {} should be positive", result.gamma);
    }

    #[test]
    fn crr_put_call_parity() {
        let call = binomial_crr(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, true, false, 1000);
        let put = binomial_crr(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, false, false, 1000);
        let lhs = call.npv - put.npv;
        let rhs = 100.0 - 100.0 * (-0.05_f64).exp();
        assert_abs_diff_eq!(lhs, rhs, epsilon = 0.1);
    }

    #[test]
    fn crr_fd_american_put_agreement() {
        // CRR and FD should agree on American put price
        let crr = binomial_crr(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, false, true, 500);
        let fd = crate::finite_differences::fd_black_scholes(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, false, true, 300, 300,
        );
        assert_abs_diff_eq!(crr.npv, fd.npv, epsilon = 0.3);
    }

    #[test]
    fn crr_deep_itm_american_put() {
        let result = binomial_crr(50.0, 100.0, 0.05, 0.0, 0.2, 1.0, false, true, 500);
        let intrinsic = 100.0 - 50.0;
        assert!(
            result.npv >= intrinsic * 0.99,
            "Deep ITM American put {} should be ≥ intrinsic {}",
            result.npv,
            intrinsic
        );
    }

    #[test]
    fn crr_convergence_with_steps() {
        // Price should converge as steps increase
        let coarse = binomial_crr(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, true, false, 50);
        let fine = binomial_crr(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, true, false, 500);
        // Fine should be closer to BS 10.45
        assert!(
            (fine.npv - 10.45).abs() < (coarse.npv - 10.45).abs() + 0.1,
            "Fine {} should be closer to BS than coarse {}",
            fine.npv,
            coarse.npv
        );
    }
}
