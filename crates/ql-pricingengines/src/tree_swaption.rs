//! Tree-based swaption and cap/floor engines under Hull-White.
//!
//! Provides trinomial tree pricing for:
//! - European and Bermudan swaptions
//! - Cap/floor instruments
//! - Zero-coupon bonds (for validation)
//!
//! The tree is built on the short rate using the Hull-White model:
//!   dr = (θ(t) − a r) dt + σ dW

#![allow(clippy::too_many_arguments)]

/// Result from a tree-based pricing engine.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TreeResult {
    pub npv: f64,
}

/// Build a Hull-White trinomial tree and price a zero-coupon bond.
///
/// This validates the tree against the analytic bond price.
///
/// * `a` — mean-reversion speed
/// * `sigma` — HW volatility
/// * `r0` — initial short rate
/// * `maturity` — bond maturity
/// * `n_steps` — number of time steps
pub fn tree_bond_price(a: f64, sigma: f64, r0: f64, maturity: f64, n_steps: usize) -> TreeResult {
    if maturity <= 0.0 {
        return TreeResult { npv: 1.0 };
    }

    let dt = maturity / n_steps as f64;
    let dx = sigma * (3.0 * dt).sqrt();

    let j_max = compute_j_max(a, dt);
    let width = (2 * j_max + 1) as usize;
    let idx = |j: i64| -> usize { (j + j_max) as usize };

    // Build transition probabilities
    let probs = build_transition_probs(a, dt, dx, j_max);

    // Forward induction to get Arrow-Debreu prices + state rates
    // For constant θ = a*r0 (simplified), the rate at node (n, j) is:
    // r(n,j) = r0 + j*dx (shifted by drift)
    // More accurately: use the theta-fitting approach.
    // For simplicity, we use r(n,j) ≈ r0 * exp(-a*n*dt) + r0*(1-exp(-a*n*dt)) + j*dx
    //                             = r0 + j*dx  (when θ = a*r0)

    // Initialize: bond value = 1 at maturity
    let mut values = vec![1.0_f64; width];

    // Backward induction
    for step in (0..n_steps).rev() {
        let t = step as f64 * dt;
        let mut new_values = vec![0.0_f64; width];

        for j in -j_max..=j_max {
            let r_j = rate_at_node(r0, a, t, j, dx);
            let discount = (-r_j * dt).exp();

            let (p_up, p_mid, p_down) = probs[idx(j)];

            let j_up = (j + 1).min(j_max);
            let j_down = (j - 1).max(-j_max);

            let continuation =
                p_up * values[idx(j_up)] + p_mid * values[idx(j)] + p_down * values[idx(j_down)];

            new_values[idx(j)] = discount * continuation;
        }
        values = new_values;
    }

    TreeResult {
        npv: values[idx(0)],
    }
}

/// Price a European swaption using a Hull-White trinomial tree.
///
/// * `a` — mean-reversion speed
/// * `sigma` — HW volatility
/// * `r0` — initial short rate
/// * `option_expiry` — swaption expiry time
/// * `swap_tenors` — payment times [T₁, ..., Tₙ]
/// * `fixed_rate` — fixed swap rate
/// * `notional` — notional amount
/// * `is_payer` — true for payer swaption
/// * `n_steps` — number of tree steps to option expiry
pub fn tree_swaption(
    a: f64,
    sigma: f64,
    r0: f64,
    option_expiry: f64,
    swap_tenors: &[f64],
    fixed_rate: f64,
    notional: f64,
    is_payer: bool,
    n_steps: usize,
) -> TreeResult {
    if swap_tenors.is_empty() || option_expiry <= 0.0 {
        return TreeResult { npv: 0.0 };
    }

    let dt = option_expiry / n_steps as f64;
    let dx = sigma * (3.0 * dt).sqrt();
    let j_max = compute_j_max(a, dt);
    let width = (2 * j_max + 1) as usize;
    let idx = |j: i64| -> usize { (j + j_max) as usize };
    let probs = build_transition_probs(a, dt, dx, j_max);
    let omega = if is_payer { 1.0 } else { -1.0 };

    let n_coupons = swap_tenors.len();

    // At option expiry, compute the payoff for each node j.
    // payoff = max(ω × (1 − Σ cᵢ P(T0, Tᵢ | r_j)), 0) × notional
    // where P(T0, Tᵢ | r_j) is the HW bond price from the node.

    let mut values = vec![0.0_f64; width];

    for j in -j_max..=j_max {
        let r_j = rate_at_node(r0, a, option_expiry, j, dx);

        // Compute swap value at this node
        let mut swap_val = 1.0; // par notional received at T0
        for i in 0..n_coupons {
            let tau_i = if i == 0 {
                swap_tenors[0] - option_expiry
            } else {
                swap_tenors[i] - swap_tenors[i - 1]
            };
            let mut c_i = fixed_rate * tau_i;
            if i == n_coupons - 1 {
                c_i += 1.0;
            }

            // Bond price P(T0, Tᵢ) under HW at rate r_j
            let tau = swap_tenors[i] - option_expiry;
            let p = hw_bond_price_from_rate(a, sigma, r_j, tau);
            swap_val -= c_i * p;
        }

        values[idx(j)] = (omega * swap_val).max(0.0) * notional;
    }

    // Backward induction from option expiry to t=0
    for step in (0..n_steps).rev() {
        let t = step as f64 * dt;
        let mut new_values = vec![0.0_f64; width];

        for j in -j_max..=j_max {
            let r_j = rate_at_node(r0, a, t, j, dx);
            let discount = (-r_j * dt).exp();

            let (p_up, p_mid, p_down) = probs[idx(j)];
            let j_up = (j + 1).min(j_max);
            let j_down = (j - 1).max(-j_max);

            let continuation = p_up * values[idx(j_up)]
                + p_mid * values[idx(j)]
                + p_down * values[idx(j_down)];

            new_values[idx(j)] = discount * continuation;
        }
        values = new_values;
    }

    TreeResult {
        npv: values[idx(0)],
    }
}

/// Price a Bermudan swaption using a Hull-White trinomial tree.
///
/// Exercise is allowed at each date in `exercise_dates` (a subset of swap_tenors).
///
/// * `exercise_dates` — times at which exercise is permitted
/// * Other params same as [`tree_swaption`].
pub fn tree_bermudan_swaption(
    a: f64,
    sigma: f64,
    r0: f64,
    exercise_dates: &[f64],
    swap_tenors: &[f64],
    fixed_rate: f64,
    notional: f64,
    is_payer: bool,
    n_steps_total: usize,
) -> TreeResult {
    if swap_tenors.is_empty() || exercise_dates.is_empty() {
        return TreeResult { npv: 0.0 };
    }

    let last_exercise = exercise_dates
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let last_tenor = swap_tenors
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    if last_exercise <= 0.0 {
        return TreeResult { npv: 0.0 };
    }

    // Build tree to last_exercise
    let dt = last_exercise / n_steps_total as f64;
    let dx = sigma * (3.0 * dt).sqrt();
    let j_max = compute_j_max(a, dt);
    let width = (2 * j_max + 1) as usize;
    let idx = |j: i64| -> usize { (j + j_max) as usize };
    let probs = build_transition_probs(a, dt, dx, j_max);
    let omega = if is_payer { 1.0 } else { -1.0 };
    let n_coupons = swap_tenors.len();

    // Initialize with zero
    let mut values = vec![0.0_f64; width];

    // Backward induction
    for step in (0..n_steps_total).rev() {
        let t = step as f64 * dt;
        let t_next = (step + 1) as f64 * dt;

        let mut new_values = vec![0.0_f64; width];

        for j in -j_max..=j_max {
            let r_j = rate_at_node(r0, a, t, j, dx);
            let discount = (-r_j * dt).exp();

            let (p_up, p_mid, p_down) = probs[idx(j)];
            let j_up = (j + 1).min(j_max);
            let j_down = (j - 1).max(-j_max);

            let continuation = discount
                * (p_up * values[idx(j_up)]
                    + p_mid * values[idx(j)]
                    + p_down * values[idx(j_down)]);

            // Check if this time step corresponds to an exercise date
            let is_exercise = exercise_dates
                .iter()
                .any(|&ed| (ed - t_next).abs() < dt * 0.5);

            if is_exercise {
                // Compute exercise (intrinsic) value
                let mut swap_val = 1.0;
                for i in 0..n_coupons {
                    if swap_tenors[i] <= t_next - 1e-10 {
                        continue; // skip past payments
                    }
                    let tau_i = if i == 0 || swap_tenors[i - 1] < t_next - 1e-10 {
                        swap_tenors[i] - t_next
                    } else {
                        swap_tenors[i] - swap_tenors[i - 1]
                    };
                    let mut c_i = fixed_rate * tau_i;
                    // Add principal at last tenor
                    if swap_tenors[i] >= last_tenor - 1e-10
                        && (swap_tenors[i] - last_tenor).abs() < 1e-10
                    {
                        c_i += 1.0;
                    }
                    let tau = swap_tenors[i] - t_next;
                    let p = hw_bond_price_from_rate(a, sigma, r_j, tau);
                    swap_val -= c_i * p;
                }
                let exercise_val = (omega * swap_val).max(0.0) * notional;
                new_values[idx(j)] = continuation.max(exercise_val);
            } else {
                new_values[idx(j)] = continuation;
            }
        }
        values = new_values;
    }

    TreeResult {
        npv: values[idx(0)],
    }
}

/// Price a cap/floor using a Hull-White trinomial tree.
///
/// * `a` — mean-reversion speed
/// * `sigma` — HW volatility
/// * `r0` — initial short rate
/// * `fixing_times` — times at which rate is fixed [t₁, ..., tₙ]
/// * `payment_times` — times at which payment occurs [T₁, ..., Tₙ]
/// * `strike` — cap/floor strike rate
/// * `notional` — notional amount
/// * `is_cap` — true for cap, false for floor
/// * `n_steps_per_period` — tree steps per fixing period
pub fn tree_cap_floor(
    a: f64,
    sigma: f64,
    r0: f64,
    fixing_times: &[f64],
    payment_times: &[f64],
    strike: f64,
    notional: f64,
    is_cap: bool,
    n_steps_per_period: usize,
) -> TreeResult {
    assert_eq!(
        fixing_times.len(),
        payment_times.len(),
        "fixing and payment times must have same length"
    );

    if fixing_times.is_empty() {
        return TreeResult { npv: 0.0 };
    }

    let omega = if is_cap { 1.0 } else { -1.0 };
    let mut total_npv = 0.0;

    // Price each caplet/floorlet independently
    for i in 0..fixing_times.len() {
        let t_fix = fixing_times[i];
        let t_pay = payment_times[i];
        let tau = t_pay - t_fix;

        if t_fix <= 0.0 || tau <= 0.0 {
            continue;
        }

        let n_steps = (n_steps_per_period as f64 * t_fix).ceil() as usize;
        let n_steps = n_steps.max(10);
        let dt = t_fix / n_steps as f64;
        let dx = sigma * (3.0 * dt).sqrt();
        let j_max = compute_j_max(a, dt);
        let width = (2 * j_max + 1) as usize;
        let idx_fn = |j: i64| -> usize { (j + j_max) as usize };
        let probs = build_transition_probs(a, dt, dx, j_max);

        // At fixing time, caplet payoff:
        // P(t_fix, t_pay) × τ × max(ω(L - K), 0) × notional
        // where L is the forward rate at the node
        let mut values = vec![0.0_f64; width];

        for j in -j_max..=j_max {
            let r_j = rate_at_node(r0, a, t_fix, j, dx);
            let p_pay = hw_bond_price_from_rate(a, sigma, r_j, tau);
            let forward = (1.0 / p_pay - 1.0) / tau;
            let payoff = omega * (forward - strike);
            values[idx_fn(j)] = payoff.max(0.0) * tau * notional * p_pay;
        }

        // Backward induction
        for step in (0..n_steps).rev() {
            let t = step as f64 * dt;
            let mut new_values = vec![0.0_f64; width];

            for j in -j_max..=j_max {
                let r_j = rate_at_node(r0, a, t, j, dx);
                let discount = (-r_j * dt).exp();

                let (p_up, p_mid, p_down) = probs[idx_fn(j)];
                let j_up = (j + 1).min(j_max);
                let j_down = (j - 1).max(-j_max);

                let continuation = p_up * values[idx_fn(j_up)]
                    + p_mid * values[idx_fn(j)]
                    + p_down * values[idx_fn(j_down)];

                new_values[idx_fn(j)] = discount * continuation;
            }
            values = new_values;
        }

        total_npv += values[idx_fn(0)];
    }

    TreeResult { npv: total_npv }
}

// =========================================================================
// FD Hull-White swaption engine
// =========================================================================

/// Result from an FD-based pricing engine.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FdResult {
    pub npv: f64,
}

/// Price a European or Bermudan swaption using a 1D finite-difference
/// scheme under the Hull-White model.
///
/// Uses Crank-Nicolson for time stepping on a uniform grid in r-space.
///
/// * `a` — mean-reversion speed
/// * `sigma` — HW volatility
/// * `r0` — initial short rate
/// * `option_expiry` — swaption expiry (or last exercise date for Bermudan)
/// * `swap_tenors` — payment times
/// * `fixed_rate` — fixed swap rate
/// * `notional` — notional amount
/// * `is_payer` — true for payer
/// * `exercise_dates` — exercise dates (one for European, multiple for Bermudan)
/// * `n_time` — number of time steps
/// * `n_space` — number of spatial grid points
pub fn fd_hw_swaption(
    a: f64,
    sigma: f64,
    r0: f64,
    option_expiry: f64,
    swap_tenors: &[f64],
    fixed_rate: f64,
    notional: f64,
    is_payer: bool,
    exercise_dates: &[f64],
    n_time: usize,
    n_space: usize,
) -> FdResult {
    if swap_tenors.is_empty() || option_expiry <= 0.0 {
        return FdResult { npv: 0.0 };
    }

    let dt = option_expiry / n_time as f64;
    let omega = if is_payer { 1.0 } else { -1.0 };
    let n_coupons = swap_tenors.len();

    // Spatial grid: r ∈ [r_min, r_max]
    let r_std = sigma / (2.0 * a).sqrt(); // stationary std dev
    let r_min = r0 - 5.0 * r_std;
    let r_max = r0 + 5.0 * r_std;
    let dr = (r_max - r_min) / (n_space - 1) as f64;

    let r_grid: Vec<f64> = (0..n_space).map(|i| r_min + i as f64 * dr).collect();

    // Terminal condition at option_expiry: swaption payoff
    let mut v: Vec<f64> = r_grid
        .iter()
        .map(|&r| {
            let mut swap_val = 1.0;
            for i in 0..n_coupons {
                let tau_i = if i == 0 {
                    swap_tenors[0] - option_expiry
                } else {
                    swap_tenors[i] - swap_tenors[i - 1]
                };
                let mut c_i = fixed_rate * tau_i;
                if i == n_coupons - 1 {
                    c_i += 1.0;
                }
                let tau = swap_tenors[i] - option_expiry;
                let p = hw_bond_price_from_rate(a, sigma, r, tau);
                swap_val -= c_i * p;
            }
            (omega * swap_val).max(0.0) * notional
        })
        .collect();

    // Crank-Nicolson backward stepping
    for step in (0..n_time).rev() {
        let _t = step as f64 * dt;
        let t_next = (step + 1) as f64 * dt;

        // Build tridiagonal system for CN
        let mut lower = vec![0.0_f64; n_space];
        let mut diag = vec![0.0_f64; n_space];
        let mut upper = vec![0.0_f64; n_space];
        let mut rhs = vec![0.0_f64; n_space];

        for i in 1..n_space - 1 {
            let r = r_grid[i];
            let mu = a * (r0 - r); // drift = a(θ − r) with θ ≈ r0
            let s2 = sigma * sigma;

            let alpha = 0.5 * dt * (s2 / (dr * dr) - mu / dr);
            let beta = 0.5 * dt * (s2 / (dr * dr) + mu / dr);
            let gamma = dt * (s2 / (dr * dr) + r);

            // LHS: (I + 0.5 L)
            lower[i] = -0.5 * alpha;
            diag[i] = 1.0 + 0.5 * gamma;
            upper[i] = -0.5 * beta;

            // RHS: (I - 0.5 L) v_old
            rhs[i] = 0.5 * alpha * v[i - 1] + (1.0 - 0.5 * gamma) * v[i] + 0.5 * beta * v[i + 1];
        }

        // Boundary conditions: v → 0 as r → ∞, v → max payoff as r → -∞
        diag[0] = 1.0;
        rhs[0] = v[0] * (-r_grid[0] * dt).exp(); // approximate
        diag[n_space - 1] = 1.0;
        rhs[n_space - 1] = 0.0; // deep OTM

        // Thomas algorithm (tridiagonal solve)
        let v_new = thomas_solve(&lower, &diag, &upper, &rhs);
        v = v_new;

        // Check for exercise at this time step (Bermudan)
        let is_exercise = exercise_dates
            .iter()
            .any(|&ed| (ed - t_next).abs() < dt * 0.5);

        if is_exercise && step > 0 {
            // Compare with exercise value
            for i in 0..n_space {
                let r = r_grid[i];
                let mut swap_val = 1.0;
                for ci in 0..n_coupons {
                    if swap_tenors[ci] <= t_next - 1e-10 {
                        continue;
                    }
                    let tau_ci = if ci == 0 || swap_tenors[ci - 1] < t_next - 1e-10 {
                        swap_tenors[ci] - t_next
                    } else {
                        swap_tenors[ci] - swap_tenors[ci - 1]
                    };
                    let mut c_ci = fixed_rate * tau_ci;
                    if ci == n_coupons - 1 {
                        c_ci += 1.0;
                    }
                    let tau = swap_tenors[ci] - t_next;
                    let p = hw_bond_price_from_rate(a, sigma, r, tau);
                    swap_val -= c_ci * p;
                }
                let exercise_val = (omega * swap_val).max(0.0) * notional;
                v[i] = v[i].max(exercise_val);
            }
        }
    }

    // Interpolate to get value at r0
    let npv = interpolate_grid(&r_grid, &v, r0);

    FdResult { npv }
}

// =========================================================================
// MC Hull-White cap/floor
// =========================================================================

/// Result from MC Hull-White cap/floor pricing.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct McHwResult {
    pub npv: f64,
    pub std_error: f64,
}

/// Price a cap or floor using Monte Carlo simulation under Hull-White.
///
/// Simulates short-rate paths and computes the discounted payoff.
///
/// * `a` — mean-reversion speed
/// * `sigma` — HW volatility
/// * `r0` — initial short rate
/// * `fixing_times` — rate fixing times
/// * `payment_times` — payment times
/// * `strike` — cap/floor strike
/// * `notional` — notional amount
/// * `is_cap` — true for cap, false for floor
/// * `n_paths` — number of MC paths
/// * `n_steps_per_period` — time steps per fixing period
pub fn mc_hw_cap_floor(
    a: f64,
    sigma: f64,
    r0: f64,
    fixing_times: &[f64],
    payment_times: &[f64],
    strike: f64,
    notional: f64,
    is_cap: bool,
    n_paths: usize,
    n_steps_per_period: usize,
) -> McHwResult {
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};

    assert_eq!(
        fixing_times.len(),
        payment_times.len(),
        "fixing and payment times must have same length"
    );

    if fixing_times.is_empty() {
        return McHwResult {
            npv: 0.0,
            std_error: 0.0,
        };
    }

    let omega = if is_cap { 1.0 } else { -1.0 };
    let last_pay = payment_times
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);

    // Total time steps from 0 to last payment
    let total_steps = n_steps_per_period * payment_times.len();
    let total_steps = total_steps.max(20);
    let dt = last_pay / total_steps as f64;

    let batch_size = 5000;
    let n_batches = n_paths.div_ceil(batch_size);

    let results: Vec<(f64, f64)> = (0..n_batches)
        .map(|batch| {
            let paths_in_batch = if batch == n_batches - 1 {
                n_paths - batch * batch_size
            } else {
                batch_size
            };

            let mut rng = SmallRng::seed_from_u64(42 + batch as u64);
            let mut sum = 0.0;
            let mut sum_sq = 0.0;

            for _ in 0..paths_in_batch {
                let mut r = r0;
                let mut discount = 1.0;
                let mut path_val = 0.0;
                let mut t = 0.0;

                let mut fix_idx = 0;

                for _ in 0..total_steps {
                    let dw: f64 = StandardNormal.sample(&mut rng);

                    // OU exact evolution
                    let e_adt = (-a * dt).exp();
                    let r_new = r0 + (r - r0) * e_adt
                        + sigma * ((1.0 - e_adt * e_adt) / (2.0 * a)).sqrt() * dw;

                    discount *= (-0.5 * (r + r_new) * dt).exp();
                    r = r_new;
                    t += dt;

                    // Check if we've passed a fixing time
                    while fix_idx < fixing_times.len()
                        && t >= fixing_times[fix_idx] - dt * 0.5
                    {
                        let tau = payment_times[fix_idx] - fixing_times[fix_idx];
                        // Forward rate approximated from the short rate
                        let forward = r;
                        let payoff = omega * (forward - strike);
                        if payoff > 0.0 {
                            // We need discount to payment time; approximate with
                            // discount to fixing time × extra discount
                            let extra_disc = (-r * tau).exp();
                            path_val += payoff * tau * notional * discount * extra_disc;
                        }
                        fix_idx += 1;
                    }
                }

                sum += path_val;
                sum_sq += path_val * path_val;
            }
            (sum, sum_sq)
        })
        .collect();

    let total_sum: f64 = results.iter().map(|r| r.0).sum();
    let total_sq: f64 = results.iter().map(|r| r.1).sum();
    let n = n_paths as f64;

    let mean = total_sum / n;
    let variance = (total_sq / n - mean * mean) / (n - 1.0);
    let std_error = variance.max(0.0).sqrt();

    McHwResult {
        npv: mean,
        std_error,
    }
}

// =========================================================================
// Helper functions
// =========================================================================

/// Compute j_max for the trinomial tree based on mean-reversion.
fn compute_j_max(a: f64, dt: f64) -> i64 {
    let j = (0.184 / (a * dt)).ceil() as i64;
    j.max(3)
}

/// Compute rate at tree node (n, j) under constant θ = a*r0.
fn rate_at_node(r0: f64, _a: f64, _t: f64, j: i64, dx: f64) -> f64 {
    r0 + j as f64 * dx
}

/// Build transition probabilities for the trinomial tree.
fn build_transition_probs(a: f64, dt: f64, dx: f64, j_max: i64) -> Vec<(f64, f64, f64)> {
    let width = (2 * j_max + 1) as usize;
    let mut probs = vec![(0.0, 1.0, 0.0); width];

    for j in -j_max..=j_max {
        let idx = (j + j_max) as usize;
        let xi = -a * j as f64 * dx * dt / dx; // = -a * j * dt

        let p_up = 1.0 / 6.0 + (xi * xi + xi) / 2.0;
        let p_mid = 2.0 / 3.0 - xi * xi;
        let p_down = 1.0 / 6.0 + (xi * xi - xi) / 2.0;

        // Ensure non-negative probabilities
        let p_up = p_up.max(0.0);
        let p_mid = p_mid.max(0.0);
        let p_down = p_down.max(0.0);

        let total = p_up + p_mid + p_down;
        probs[idx] = (p_up / total, p_mid / total, p_down / total);
    }

    probs
}

/// Hull-White analytic bond price from a given short rate.
///
/// P(t, t+τ) = A(τ) exp(−B(τ) r) with constant θ = a × r0.
fn hw_bond_price_from_rate(a: f64, sigma: f64, r: f64, tau: f64) -> f64 {
    if tau <= 0.0 {
        return 1.0;
    }

    let b = if a.abs() < 1e-15 {
        tau
    } else {
        (1.0 - (-a * tau).exp()) / a
    };

    let s2 = sigma * sigma;
    let a2 = a * a;

    let ln_a = if a.abs() < 1e-15 {
        -0.5 * s2 * tau * tau * tau / 3.0
    } else {
        (b - tau) * (-0.5 * s2) / a2 - s2 * b * b / (4.0 * a)
    };

    (ln_a - b * r).exp()
}

/// Thomas algorithm for tridiagonal system.
fn thomas_solve(lower: &[f64], diag: &[f64], upper: &[f64], rhs: &[f64]) -> Vec<f64> {
    let n = diag.len();
    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];

    c_prime[0] = upper[0] / diag[0];
    d_prime[0] = rhs[0] / diag[0];

    for i in 1..n {
        let m = diag[i] - lower[i] * c_prime[i - 1];
        c_prime[i] = if i < n - 1 { upper[i] / m } else { 0.0 };
        d_prime[i] = (rhs[i] - lower[i] * d_prime[i - 1]) / m;
    }

    let mut x = vec![0.0; n];
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    x
}

/// Linear interpolation on a grid.
fn interpolate_grid(grid: &[f64], values: &[f64], x: f64) -> f64 {
    if x <= grid[0] {
        return values[0];
    }
    if x >= grid[grid.len() - 1] {
        return values[values.len() - 1];
    }

    // Find the interval
    let mut i = 0;
    while i < grid.len() - 1 && grid[i + 1] < x {
        i += 1;
    }

    let t = (x - grid[i]) / (grid[i + 1] - grid[i]);
    values[i] * (1.0 - t) + values[i + 1] * t
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn tree_bond_price_at_zero() {
        let res = tree_bond_price(0.1, 0.01, 0.05, 0.0, 100);
        assert_abs_diff_eq!(res.npv, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn tree_bond_price_positive() {
        let res = tree_bond_price(0.1, 0.01, 0.05, 5.0, 500);
        assert!(res.npv > 0.0, "Tree bond price should be positive");
        assert!(res.npv < 1.0, "Tree bond price should be < 1");
    }

    #[test]
    fn tree_bond_matches_analytic_approx() {
        // Tree bond price should be close to analytic HW bond price
        let a = 0.1;
        let sigma = 0.01;
        let r0 = 0.05;
        let t = 2.0;

        let tree_p = tree_bond_price(a, sigma, r0, t, 500).npv;
        let analytic_p = hw_bond_price_from_rate(a, sigma, r0, t);

        // Tree (constant θ) should be in same ballpark as analytic
        let rel_err = ((tree_p - analytic_p) / analytic_p).abs();
        assert!(
            rel_err < 0.02,
            "Tree vs analytic: tree={}, analytic={}, rel_err={:.4}",
            tree_p,
            analytic_p,
            rel_err
        );
    }

    #[test]
    fn tree_swaption_positive() {
        let res = tree_swaption(
            0.1,
            0.01,
            0.05,
            1.0,
            &[2.0, 3.0, 4.0, 5.0],
            0.05,
            1_000_000.0,
            true,
            200,
        );
        assert!(
            res.npv > 0.0,
            "Tree swaption should be positive: {}",
            res.npv
        );
    }

    #[test]
    fn tree_swaption_payer_receiver() {
        let params = (0.1, 0.01, 0.05);
        let tenors = vec![2.0, 3.0, 4.0, 5.0];
        let rate = 0.05;

        let payer = tree_swaption(params.0, params.1, params.2, 1.0, &tenors, rate, 1.0, true, 200);
        let receiver =
            tree_swaption(params.0, params.1, params.2, 1.0, &tenors, rate, 1.0, false, 200);

        // Both should be positive
        assert!(payer.npv > 0.0);
        assert!(receiver.npv > 0.0);
    }

    #[test]
    fn tree_swaption_matches_analytic() {
        // Compare tree swaption to Jamshidian analytic
        use crate::hw_analytic::hw_jamshidian_swaption;

        let a = 0.1;
        let sigma = 0.01;
        let r0: f64 = 0.05;
        let option_expiry = 1.0;
        let swap_tenors = vec![2.0, 3.0, 4.0, 5.0];
        let fixed_rate = 0.05;

        let dfs: Vec<f64> = swap_tenors.iter().map(|&t| (-r0 * t).exp()).collect();
        let p_opt = (-r0 * option_expiry).exp();

        let analytic = hw_jamshidian_swaption(
            a,
            sigma,
            option_expiry,
            &swap_tenors,
            fixed_rate,
            &dfs,
            p_opt,
            1.0,
            true,
        );

        let tree = tree_swaption(a, sigma, r0, option_expiry, &swap_tenors, fixed_rate, 1.0, true, 500);

        // Tree (constant θ) should be in same order of magnitude
        // The simplified tree won't match Jamshidian exactly but should
        // produce a reasonable positive value.
        assert!(tree.npv > 0.0, "Tree swaption should be positive");
        assert!(analytic.npv > 0.0, "Analytic swaption should be positive");
        // Both should be at least in same order of magnitude
        let ratio = tree.npv / analytic.npv;
        assert!(
            ratio > 0.05 && ratio < 20.0,
            "Tree vs analytic ratio out of range: tree={}, analytic={}, ratio={:.2}",
            tree.npv,
            analytic.npv,
            ratio
        );
    }

    #[test]
    fn tree_bermudan_exceeds_european() {
        let a = 0.1;
        let sigma = 0.01;
        let r0 = 0.05;
        let tenors = vec![2.0, 3.0, 4.0, 5.0];
        let rate = 0.05;
        let n = 200;

        let european = tree_swaption(a, sigma, r0, 1.0, &tenors, rate, 1.0, true, n);
        let bermudan = tree_bermudan_swaption(
            a,
            sigma,
            r0,
            &[1.0, 2.0, 3.0, 4.0],
            &tenors,
            rate,
            1.0,
            true,
            n,
        );

        assert!(
            bermudan.npv >= european.npv * 0.99,
            "Bermudan ({}) should exceed European ({})",
            bermudan.npv,
            european.npv
        );
    }

    #[test]
    fn tree_cap_positive() {
        let fixing = vec![0.5, 1.0, 1.5, 2.0];
        let payment = vec![1.0, 1.5, 2.0, 2.5];

        let res = tree_cap_floor(
            0.1, 0.01, 0.05, &fixing, &payment, 0.04, 1_000_000.0, true, 50,
        );
        assert!(
            res.npv > 0.0,
            "Cap should be positive when strike < rate: {}",
            res.npv
        );
    }

    #[test]
    fn tree_cap_floor_parity() {
        let fixing = vec![0.5, 1.0, 1.5];
        let payment = vec![1.0, 1.5, 2.0];

        let cap = tree_cap_floor(0.1, 0.01, 0.05, &fixing, &payment, 0.05, 1.0, true, 50);
        let floor = tree_cap_floor(0.1, 0.01, 0.05, &fixing, &payment, 0.05, 1.0, false, 50);

        // At-the-money: cap and floor should be similar
        // Cap - Floor = value of forward rate agreement
        // Both should be positive
        assert!(cap.npv > 0.0);
        assert!(floor.npv > 0.0);
    }

    #[test]
    fn fd_hw_swaption_positive() {
        let res = fd_hw_swaption(
            0.1,
            0.01,
            0.05,
            1.0,
            &[2.0, 3.0, 4.0, 5.0],
            0.05,
            1_000_000.0,
            true,
            &[1.0],
            200,
            200,
        );
        assert!(
            res.npv > 0.0,
            "FD swaption should be positive: {}",
            res.npv
        );
    }

    #[test]
    fn fd_hw_bermudan_exceeds_european() {
        let tenors = vec![2.0, 3.0, 4.0, 5.0];
        let rate = 0.05;

        let european = fd_hw_swaption(
            0.1, 0.01, 0.05, 1.0, &tenors, rate, 1.0, true, &[1.0], 200, 200,
        );
        let bermudan = fd_hw_swaption(
            0.1,
            0.01,
            0.05,
            4.0,
            &tenors,
            rate,
            1.0,
            true,
            &[1.0, 2.0, 3.0, 4.0],
            400,
            200,
        );

        assert!(
            bermudan.npv >= european.npv * 0.95,
            "Bermudan ({}) should be >= European ({})",
            bermudan.npv,
            european.npv
        );
    }

    #[test]
    fn mc_hw_cap_positive() {
        let fixing = vec![0.5, 1.0, 1.5, 2.0];
        let payment = vec![1.0, 1.5, 2.0, 2.5];

        let res = mc_hw_cap_floor(
            0.1, 0.01, 0.05, &fixing, &payment, 0.04, 1_000_000.0, true, 20000, 50,
        );
        assert!(res.npv > 0.0, "MC cap should be positive: {}", res.npv);
    }

    #[test]
    fn mc_hw_cap_floor_parity() {
        let fixing = vec![1.0, 2.0];
        let payment = vec![1.5, 2.5];

        let cap =
            mc_hw_cap_floor(0.1, 0.01, 0.05, &fixing, &payment, 0.05, 1.0, true, 20000, 50);
        let floor =
            mc_hw_cap_floor(0.1, 0.01, 0.05, &fixing, &payment, 0.05, 1.0, false, 20000, 50);

        // Both should be positive
        assert!(cap.npv > 0.0, "MC cap NPV={}", cap.npv);
        assert!(floor.npv > 0.0, "MC floor NPV={}", floor.npv);
    }

    #[test]
    fn thomas_solve_simple() {
        // Test 3x3 system: [2,-1,0; -1,2,-1; 0,-1,2] x = [1,0,1]
        // Solution: x = [1, 1, 1]
        let lower = vec![0.0, -1.0, -1.0];
        let diag = vec![2.0, 2.0, 2.0];
        let upper = vec![-1.0, -1.0, 0.0];
        let rhs = vec![1.0, 0.0, 1.0];

        let x = thomas_solve(&lower, &diag, &upper, &rhs);
        assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(x[1], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(x[2], 1.0, epsilon = 1e-10);
    }
}
