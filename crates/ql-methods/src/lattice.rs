//! Lattice (binomial tree) framework for option pricing.
//!
//! Implements Cox-Ross-Rubinstein (CRR) binomial trees for
//! European and American option pricing.

use tracing::info_span;

/// Result from a lattice calculation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
///
/// # Examples
///
/// ```
/// use ql_methods::lattice::binomial_crr;
///
/// let res = binomial_crr(100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
///                         true, false, 500);
/// // European call converges to BS ~$10.45
/// assert!((res.npv - 10.45).abs() < 0.2);
/// ```
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

/// Price a European or American option on a CRR tree with discrete dividends.
///
/// At each tree step that coincides with a dividend date, the spot is
/// adjusted downward (cash dividends subtracted, proportional dividends
/// multiplied). The continuous yield `q` still applies for any
/// remaining continuous dividend component.
///
/// # Dividend handling
///
/// For cash dividends, each node's spot is reduced by the dividend amount.
/// For proportional dividends, each node's spot is multiplied by `(1 - rate)`.
/// This is the standard "stock price adjustment" approach.
#[allow(clippy::too_many_arguments)]
pub fn binomial_crr_discrete_dividends(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    vol: f64,
    time_to_expiry: f64,
    is_call: bool,
    is_american: bool,
    num_steps: usize,
    dividends: &ql_cashflows::DividendSchedule,
) -> LatticeResult {
    let dt = time_to_expiry / num_steps as f64;
    let df = (-r * dt).exp();
    let u = (vol * dt.sqrt()).exp();
    let d = 1.0 / u;
    let growth = ((r - q) * dt).exp();
    let p_up = (growth - d) / (u - d);
    let omega: f64 = if is_call { 1.0 } else { -1.0 };
    let n = num_steps;

    // Precompute which steps have dividends
    // Map step index → list of dividend adjustments
    let mut step_divs: Vec<Vec<(bool, f64)>> = vec![Vec::new(); n + 1];
    for div in dividends.iter() {
        let t_div = div.time();
        if t_div > 0.0 && t_div <= time_to_expiry {
            // Find the closest step (round to nearest)
            let step_f = t_div / dt;
            let step = (step_f.round() as usize).min(n).max(1);
            match div {
                ql_cashflows::Dividend::Cash { amount, .. } => {
                    step_divs[step].push((true, *amount));
                }
                ql_cashflows::Dividend::Proportional { rate, .. } => {
                    step_divs[step].push((false, *rate));
                }
            }
        }
    }

    // Build spot tree forward, storing spots at each step
    // spots[step] = Vec of spot values at each node
    let mut spots: Vec<Vec<f64>> = Vec::with_capacity(n + 1);

    // Step 0
    spots.push(vec![spot]);

    for step in 1..=n {
        let prev = &spots[step - 1];
        let mut current = Vec::with_capacity(step + 1);
        for j in 0..=step {
            // Node (step, j): from node (step-1, j-1) going up, or (step-1, j) going down
            // Spot = base * u^j * d^(step-j) with dividend adjustments
            let s = if j == 0 {
                prev[0] * d
            } else if j == step {
                prev[step - 1] * u
            } else {
                // Take average of up and down paths for consistency
                prev[j - 1] * u
            };
            current.push(s);
        }

        // Apply dividends at this step
        for &(is_cash, val) in &step_divs[step] {
            for s in current.iter_mut() {
                if is_cash {
                    *s = (*s - val).max(0.0);
                } else {
                    *s *= 1.0 - val;
                }
            }
        }

        spots.push(current);
    }

    // Build terminal payoffs
    let mut values: Vec<f64> = spots[n]
        .iter()
        .map(|&s| (omega * (s - strike)).max(0.0))
        .collect();

    let mut val_step_1 = [0.0; 3];
    let mut val_step_2 = [0.0; 3];

    // Backward induction
    for step in (0..n).rev() {
        let mut new_values = vec![0.0; step + 1];
        for j in 0..=step {
            let continuation = df * (p_up * values[j + 1] + (1.0 - p_up) * values[j]);
            new_values[j] = if is_american {
                let intrinsic = (omega * (spots[step][j] - strike)).max(0.0);
                continuation.max(intrinsic)
            } else {
                continuation
            };
        }

        if step == 2 && n >= 3 {
            val_step_2[0] = new_values[0];
            val_step_2[1] = new_values[1];
            val_step_2[2] = new_values[2];
        }
        if step == 1 {
            val_step_1[0] = new_values[0];
            val_step_1[1] = new_values[1];
        }

        values = new_values;
    }

    let npv = values[0];

    let su = spots[1].last().copied().unwrap_or(spot * u);
    let sd = spots[1].first().copied().unwrap_or(spot * d);
    let delta = if n >= 1 && (su - sd).abs() > 1e-15 {
        (val_step_1[1] - val_step_1[0]) / (su - sd)
    } else {
        0.0
    };

    let gamma = if n >= 3 {
        let s_uu = spots[2].last().copied().unwrap_or(spot * u * u);
        let s_dd = spots[2].first().copied().unwrap_or(spot * d * d);
        let h = 0.5 * (s_uu - s_dd);
        if h.abs() > 1e-15 {
            let d_up = (val_step_2[2] - val_step_2[1]) / (s_uu - spot);
            let d_dn = (val_step_2[1] - val_step_2[0]) / (spot - s_dd);
            (d_up - d_dn) / h
        } else {
            0.0
        }
    } else {
        0.0
    };

    let theta = if n >= 2 {
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

    // ── Discrete dividend tree tests ─────────────────────────

    #[test]
    fn crr_discrete_div_empty_matches_plain() {
        use ql_cashflows::DividendSchedule;
        let plain = binomial_crr(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, true, false, 500);
        let with_div = binomial_crr_discrete_dividends(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, true, false, 500,
            &DividendSchedule::empty(),
        );
        assert_abs_diff_eq!(with_div.npv, plain.npv, epsilon = 0.1);
    }

    #[test]
    fn crr_discrete_cash_div_reduces_call() {
        use ql_cashflows::{Dividend, DividendSchedule};
        let no_div = binomial_crr(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, true, false, 500);
        let divs = DividendSchedule::new(vec![Dividend::cash(0.5, 3.0)]);
        let with_div = binomial_crr_discrete_dividends(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, true, false, 500, &divs,
        );
        assert!(
            with_div.npv < no_div.npv,
            "Cash div should reduce call: {} vs {}",
            with_div.npv, no_div.npv
        );
    }

    #[test]
    fn crr_discrete_div_american_call_early_exercise() {
        use ql_cashflows::{Dividend, DividendSchedule};
        // With dividends, American call may be worth more than European
        let divs = DividendSchedule::new(vec![Dividend::cash(0.5, 5.0)]);
        let euro = binomial_crr_discrete_dividends(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, true, false, 500, &divs,
        );
        let amer = binomial_crr_discrete_dividends(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, true, true, 500, &divs,
        );
        // American call with dividends should be >= European
        assert!(
            amer.npv >= euro.npv - 0.01,
            "American call with divs ({}) should >= European ({})",
            amer.npv, euro.npv
        );
    }

    #[test]
    fn crr_discrete_proportional_div() {
        use ql_cashflows::{Dividend, DividendSchedule};
        let no_div = binomial_crr(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, true, false, 500);
        let divs = DividendSchedule::new(vec![Dividend::proportional(0.5, 0.03)]);
        let with_div = binomial_crr_discrete_dividends(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, true, false, 500, &divs,
        );
        assert!(with_div.npv < no_div.npv);
        assert!(with_div.npv > 0.0);
    }
}
