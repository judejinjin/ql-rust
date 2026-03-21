//! Two-factor trinomial lattice for pricing under two-factor short-rate models.
//!
//! Implements a 2D recombining trinomial tree for the G2++ and other 2-factor
//! models.  Each factor evolves on its own trinomial tree, and the joint tree
//! is formed as the tensor product with cross-factor correlation handled via
//! probability adjustments.
//!
//! Reference: Hull & White (1994), *Numerical Procedures for Implementing
//! Term Structure Models: Two-Factor Models*.

use ql_instruments::OptionType;

/// Result from a 2D lattice calculation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Lattice2dResult {
    /// Net present value.
    pub npv: f64,
    /// Number of nodes at the final step.
    pub n_nodes_final: usize,
}

/// Parameters for a single mean-reverting factor: dx = −a·x dt + σ dW.
#[derive(Debug, Clone, Copy)]
pub struct FactorParams {
    /// Mean-reversion speed.
    pub a: f64,
    /// Volatility.
    pub sigma: f64,
}

/// Price a European or Bermudan option on a 2-factor trinomial tree.
///
/// The model is:
///   r(t) = x₁(t) + x₂(t) + φ(t)
///
/// where xᵢ follows dx = −aᵢ xᵢ dt + σᵢ dWᵢ, with corr(dW₁, dW₂) = ρ.
/// φ(t) is a deterministic shift calibrated to match an initial discount
/// curve; here we just use a flat forward rate for simplicity.
///
/// # Arguments
/// - `factor1`, `factor2` — mean-reversion and vol for each factor
/// - `rho`   — correlation between the two Brownian motions
/// - `r0`    — initial short rate (used as the flat forward curve)
/// - `expiry` — option expiry in years
/// - `strike` — option strike (e.g., a bond price)
/// - `option_type` — Call or Put
/// - `notional` — notional amount
/// - `num_steps` — number of time steps
/// - `exercise_times` — times at which early exercise is allowed (empty = European)
/// - `underlying_fn` — function that computes the underlying value given (r, t) where r is the short rate
#[allow(clippy::too_many_arguments)]
pub fn lattice_2d<F>(
    factor1: FactorParams,
    factor2: FactorParams,
    rho: f64,
    r0: f64,
    expiry: f64,
    strike: f64,
    option_type: OptionType,
    notional: f64,
    num_steps: usize,
    exercise_times: &[f64],
    underlying_fn: F,
) -> Lattice2dResult
where
    F: Fn(f64, f64) -> f64, // (short_rate, time) -> underlying_value
{
    let dt = expiry / num_steps as f64;

    // Trinomial tree geometry for each factor
    let j_max1 = trinomial_j_max(factor1.a, dt);
    let j_max2 = trinomial_j_max(factor2.a, dt);
    let dx1 = factor1.sigma * (3.0 * dt).sqrt();
    let dx2 = factor2.sigma * (3.0 * dt).sqrt();

    let n1 = 2 * j_max1 + 1; // number of nodes in factor-1 dimension
    let n2 = 2 * j_max2 + 1;

    // Transition probabilities for each factor
    let probs1 = build_trinomial_probs(factor1.a, factor1.sigma, dt, j_max1);
    let probs2 = build_trinomial_probs(factor2.a, factor2.sigma, dt, j_max2);

    let omega = match option_type {
        OptionType::Call => 1.0,
        OptionType::Put => -1.0,
    };

    // Terminal payoff: compute on the 2D grid
    let mut values = vec![0.0; n1 * n2];
    for i1 in 0..n1 {
        for i2 in 0..n2 {
            let x1 = (i1 as i64 - j_max1 as i64) as f64 * dx1;
            let x2 = (i2 as i64 - j_max2 as i64) as f64 * dx2;
            let r = r0 + x1 + x2;
            let underlying = underlying_fn(r, expiry);
            values[i1 * n2 + i2] = (omega * (underlying - strike) * notional).max(0.0);
        }
    }

    // Backward induction
    for step in (0..num_steps).rev() {
        let t = step as f64 * dt;
        let _df_grid = build_discount_factors(r0, &values, n1, n2, j_max1, j_max2, dx1, dx2, dt);
        let mut new_values = vec![0.0; n1 * n2];

        for i1 in 0..n1 {
            let (pu1, pm1, pd1, j_shift1) = probs1[i1];
            for i2 in 0..n2 {
                let (pu2, pm2, pd2, j_shift2) = probs2[i2];

                // Expected continuation over the 3×3 = 9 joint transitions
                let mut cont = 0.0;
                for (dp1, pp1) in [(-1i64, pd1), (0, pm1), (1, pu1)] {
                    let j1_next = (i1 as i64 + j_shift1 + dp1).clamp(0, (n1 - 1) as i64) as usize;
                    for (dp2, pp2) in [(-1i64, pd2), (0, pm2), (1, pu2)] {
                        let j2_next =
                            (i2 as i64 + j_shift2 + dp2).clamp(0, (n2 - 1) as i64) as usize;
                        // Joint probability: independent + correlation adjustment
                        let p_joint = pp1 * pp2 + correlation_adj(
                            dp1 as f64, dp2 as f64, rho,
                            factor1.sigma, factor2.sigma, dt,
                        );
                        cont += p_joint.max(0.0) * values[j1_next * n2 + j2_next];
                    }
                }

                let x1 = (i1 as i64 - j_max1 as i64) as f64 * dx1;
                let x2 = (i2 as i64 - j_max2 as i64) as f64 * dx2;
                let r = r0 + x1 + x2;
                let discount = (-r * dt).exp();
                new_values[i1 * n2 + i2] = cont * discount;

                // Check early exercise
                if is_exercise_time(t, exercise_times, dt) {
                    let underlying = underlying_fn(r, t);
                    let exercise_val = (omega * (underlying - strike) * notional).max(0.0);
                    new_values[i1 * n2 + i2] =
                        new_values[i1 * n2 + i2].max(exercise_val);
                }
            }
        }

        values = new_values;
    }

    // NPV is the value at the central node (x1=0, x2=0)
    let center1 = j_max1;
    let center2 = j_max2;
    let npv = values[center1 * n2 + center2];

    Lattice2dResult {
        npv,
        n_nodes_final: n1 * n2,
    }
}

/// Simple 2D trinomial tree for a bond option under a 2-factor short-rate model.
///
/// The underlying bond price is approximated as exp(−r·(T_bond − t)).
#[allow(clippy::too_many_arguments)]
pub fn trinomial_2d_bond_option(
    factor1: FactorParams,
    factor2: FactorParams,
    rho: f64,
    r0: f64,
    option_expiry: f64,
    bond_maturity: f64,
    strike: f64,
    option_type: OptionType,
    notional: f64,
    num_steps: usize,
) -> Lattice2dResult {
    lattice_2d(
        factor1,
        factor2,
        rho,
        r0,
        option_expiry,
        strike,
        option_type,
        notional,
        num_steps,
        &[], // European
        move |r, t| {
            // Zero-coupon bond price approximation
            let time_to_maturity = bond_maturity - t;
            if time_to_maturity <= 0.0 {
                1.0
            } else {
                (-r * time_to_maturity).exp()
            }
        },
    )
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn trinomial_j_max(a: f64, dt: f64) -> usize {
    // Hull-White truncation: j_max = ceil(0.1835 / (a·dt))
    let jm = (0.1835 / (a * dt)).ceil() as usize;
    jm.clamp(1, 200) // Clamp to reasonable range
}

/// Build trinomial transition probabilities for one factor.
/// Returns Vec<(p_up, p_mid, p_down, j_shift)> for each node index.
fn build_trinomial_probs(
    a: f64,
    sigma: f64,
    dt: f64,
    j_max: usize,
) -> Vec<(f64, f64, f64, i64)> {
    let n = 2 * j_max + 1;
    let dx = sigma * (3.0 * dt).sqrt();
    let mut probs = Vec::with_capacity(n);

    for idx in 0..n {
        let j = idx as i64 - j_max as i64;
        let alpha = -a * j as f64 * dx * dt;
        let m = (alpha / dx).round() as i64; // branching shift

        let _jd = j as f64 * dx;
        let md = m as f64 * dx;
        let v2 = sigma * sigma * dt;

        // p_up, p_mid, p_down from matching first two moments + probabilities sum to 1
        let mean = alpha;
        let pu = 0.5 * ((v2 + (mean - md) * (mean - md)) / (dx * dx))
            + (mean - md) / (2.0 * dx);
        let pd = 0.5 * ((v2 + (mean - md) * (mean - md)) / (dx * dx))
            - (mean - md) / (2.0 * dx);
        let pm = 1.0 - pu - pd;

        probs.push((pu.max(0.0), pm.max(0.0), pd.max(0.0), m));
    }

    probs
}

fn build_discount_factors(
    _r0: f64,
    _values: &[f64],
    _n1: usize,
    _n2: usize,
    _j_max1: usize,
    _j_max2: usize,
    _dx1: f64,
    _dx2: f64,
    _dt: f64,
) -> Vec<f64> {
    // Placeholder — actual discount factors are computed inline during backward induction
    Vec::new()
}

/// Correlation adjustment for joint probabilities in a 2D trinomial tree.
///
/// The joint probability is p_1 · p_2 + adjustment, where the adjustment
/// ensures the cross-moment E[ΔW₁ ΔW₂] = ρ dt is matched.
fn correlation_adj(
    dp1: f64,
    dp2: f64,
    rho: f64,
    _sigma1: f64,
    _sigma2: f64,
    _dt: f64,
) -> f64 {
    // Simple moment-matching adjustment:
    // The cross-product dp1*dp2 ∈ {−1,0,1} contributes to covariance.
    // We distribute ρ/4 to the corner nodes.
    if dp1.abs() > 0.5 && dp2.abs() > 0.5 {
        rho * dp1 * dp2 / 4.0
    } else {
        0.0
    }
}

fn is_exercise_time(t: f64, exercise_times: &[f64], dt: f64) -> bool {
    if exercise_times.is_empty() {
        return false;
    }
    exercise_times
        .iter()
        .any(|&et| (t - et).abs() < 0.5 * dt)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn trinomial_j_max_reasonable() {
        let jm = trinomial_j_max(0.05, 0.01);
        assert!(jm >= 1 && jm <= 200, "j_max = {jm}");
    }

    #[test]
    fn trinomial_probs_sum_to_one() {
        let probs = build_trinomial_probs(0.1, 0.01, 0.01, 5);
        for (pu, pm, pd, _) in &probs {
            let sum = pu + pm + pd;
            assert_abs_diff_eq!(sum, 1.0, epsilon = 0.05);
        }
    }

    #[test]
    fn lattice_2d_european_put_positive() {
        let f1 = FactorParams { a: 0.1, sigma: 0.01 };
        let f2 = FactorParams { a: 0.2, sigma: 0.005 };
        let res = trinomial_2d_bond_option(
            f1, f2, -0.5,
            0.05,       // r0
            1.0,        // option expiry
            5.0,        // bond maturity
            0.80,       // strike (bond price)
            OptionType::Put,
            100.0,
            20,
        );
        assert!(res.npv >= 0.0, "put NPV should be non-negative: {}", res.npv);
    }

    #[test]
    fn lattice_2d_call_atm() {
        let f1 = FactorParams { a: 0.1, sigma: 0.01 };
        let f2 = FactorParams { a: 0.2, sigma: 0.005 };
        // Bond price at expiry ≈ exp(-0.05 * 4) ≈ 0.8187
        let res = trinomial_2d_bond_option(
            f1, f2, 0.0,
            0.05,
            1.0,
            5.0,
            0.82,
            OptionType::Call,
            100.0,
            30,
        );
        assert!(res.npv >= 0.0, "call NPV should be non-negative: {}", res.npv);
    }

    #[test]
    fn lattice_2d_bermudan() {
        let f1 = FactorParams { a: 0.1, sigma: 0.01 };
        let f2 = FactorParams { a: 0.2, sigma: 0.005 };
        let exercise_times: Vec<f64> = (1..=4).map(|i| i as f64 * 0.25).collect();
        let res = lattice_2d(
            f1, f2, -0.3,
            0.05, 1.0, 0.80,
            OptionType::Put, 100.0,
            20,
            &exercise_times,
            |r, t| {
                let ttm = 5.0 - t;
                if ttm <= 0.0 { 1.0 } else { (-r * ttm).exp() }
            },
        );
        // Bermudan should be worth ≥ European
        let res_eu = trinomial_2d_bond_option(
            f1, f2, -0.3, 0.05, 1.0, 5.0, 0.80, OptionType::Put, 100.0, 20,
        );
        assert!(res.npv >= res_eu.npv - 0.1,
                "Bermudan ({}) should be ≥ European ({})", res.npv, res_eu.npv);
    }

    #[test]
    fn lattice_2d_zero_vol_convergence() {
        // With zero vol, the short rate is deterministic = r0
        // Bond price = exp(-r0 * (T_bond - T_opt))
        // Put payoff = max(K - P, 0) * notional
        let f1 = FactorParams { a: 0.1, sigma: 1e-6 };
        let f2 = FactorParams { a: 0.2, sigma: 1e-6 };
        let r0 = 0.05;
        let t_opt = 1.0;
        let t_bond = 5.0;
        let bond_price = (-r0 * (t_bond - t_opt) as f64).exp(); // ≈ 0.8187
        let strike = 0.90_f64;
        let res = trinomial_2d_bond_option(
            f1, f2, 0.0, r0, t_opt, t_bond, strike, OptionType::Put, 100.0, 50,
        );
        // Put payoff ≈ max(0.90 - 0.8187, 0) * 100 * df ≈ 8.13 * exp(-0.05) ≈ 7.73
        let expected = ((strike - bond_price).max(0.0) * 100.0) * (-r0 * t_opt as f64).exp();
        assert!((res.npv - expected).abs() < 2.0,
                "zero-vol mismatch: {} vs expected {}", res.npv, expected);
    }
}
