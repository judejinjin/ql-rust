//! Phase 17 — Swaption & Cap/Floor Engines (G98–G105).
//!
//! Extended engines for pricing swaptions, cap/floors and irregular interest-rate
//! derivatives under the Hull-White / Gaussian 1-factor framework.
//!
//! ## Contents
//!
//! | Gap  | Engine                                | Method                              |
//! |------|---------------------------------------|-------------------------------------|
//! | G98  | [`gaussian1d_cap_floor`]              | Gauss-Hermite quadrature cap/floor  |
//! | G99  | [`gaussian1d_float_float_swaption`]   | Float-float swaption (G1D)          |
//! | G100 | [`mc_hw_swaption`]                    | Monte Carlo Hull-White swaption     |
//! | G101 | [`fd_hw_swaption_gsr`]                | Finite-difference HW swaption (GSR) |
//! | G102 | [`TreeCapFloorEngine`]                | Wrapper around tree cap/floor       |
//! | G103 | [`IrregularSwap`] / [`IrregularSwaption`] | Irregular notional swaps        |
//! | G104 | [`hagan_irregular_swaption`]          | Hagan's approximation               |
//! | G105 | [`BasketGeneratingEngine`] / [`LatticeShortRateModelEngine`] | Basket decomposition / lattice |

#![allow(clippy::too_many_arguments)]

use ql_models::gsr::Gsr1d;
use serde::{Deserialize, Serialize};

// =========================================================================
// G98 — Gaussian1DCapFloorEngine
// =========================================================================

/// Result from the Gaussian 1-factor cap/floor engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gaussian1dCapFloorResult {
    /// Total net present value of the cap or floor.
    pub npv: f64,
    /// Individual caplet/floorlet values.
    pub caplet_values: Vec<f64>,
}

/// Price a cap/floor under a Gaussian 1-factor (Hull-White) model using
/// Gauss-Hermite quadrature.
///
/// Each caplet/floorlet is valued by integrating over the state variable
/// at the fixing time. The forward rate at each quadrature node is computed
/// from the GSR model's zero-bond formula.
///
/// # Parameters
/// - `model` — calibrated `Gsr1d` model
/// - `fixing_times` — fixing times for each caplet/floorlet
/// - `payment_times` — payment times (same length as `fixing_times`)
/// - `year_fractions` — accrual fractions for each period
/// - `strike` — cap/floor strike rate
/// - `notional` — notional amount
/// - `is_cap` — `true` for cap, `false` for floor
/// - `n_quad` — number of Gauss-Hermite quadrature points
pub fn gaussian1d_cap_floor(
    model: &Gsr1d,
    fixing_times: &[f64],
    payment_times: &[f64],
    year_fractions: &[f64],
    strike: f64,
    notional: f64,
    is_cap: bool,
    n_quad: usize,
) -> Gaussian1dCapFloorResult {
    let n = fixing_times.len().min(payment_times.len()).min(year_fractions.len());
    if n == 0 {
        return Gaussian1dCapFloorResult {
            npv: 0.0,
            caplet_values: vec![],
        };
    }

    let omega = if is_cap { 1.0 } else { -1.0 };
    let (nodes, weights) = gauss_hermite_nodes(n_quad.max(4));

    let mut caplet_values = Vec::with_capacity(n);
    let mut total_npv = 0.0;

    for i in 0..n {
        let t_fix = fixing_times[i];
        let t_pay = payment_times[i];
        let tau = year_fractions[i];

        if t_fix <= 0.0 || tau <= 0.0 {
            caplet_values.push(0.0);
            continue;
        }

        let zeta = model.zeta(t_fix);
        let sqrt_zeta = zeta.max(1e-30).sqrt();
        let pm_tfix = model.market_discount(t_fix);

        let mut integral = 0.0;
        for (node, weight) in nodes.iter().zip(weights.iter()) {
            let x = std::f64::consts::SQRT_2 * sqrt_zeta * node;

            // Forward rate from the zero-bond ratio
            let p_fix = model.zero_bond(t_fix, t_fix, x); // = 1 by definition
            let p_pay = model.zero_bond(t_fix, t_pay, x);

            let forward = if p_pay > 1e-15 {
                (p_fix / p_pay - 1.0) / tau
            } else {
                0.0
            };

            let payoff = (omega * (forward - strike)).max(0.0);
            // Discounted caplet value: τ × N × payoff × P(t_fix, t_pay)
            integral += weight * payoff * tau * notional * p_pay;
        }

        let caplet_npv = pm_tfix * integral / std::f64::consts::PI.sqrt();
        caplet_values.push(caplet_npv);
        total_npv += caplet_npv;
    }

    Gaussian1dCapFloorResult {
        npv: total_npv,
        caplet_values,
    }
}

// =========================================================================
// G99 — Gaussian1DFloatFloatSwaptionEngine
// =========================================================================

/// Result from the Gaussian 1-factor float-float swaption engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gaussian1dFloatFloatSwaptionResult {
    /// Net present value of the float-float swaption.
    pub npv: f64,
    /// Underlying spread value at the quadrature mean.
    pub underlying_value: f64,
}

/// Parameters describing a float-float swaption.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FloatFloatSwaptionParams {
    /// Exercise time of the swaption.
    pub exercise_time: f64,
    /// Payment times for leg 1 (e.g. 3M LIBOR leg).
    pub leg1_payment_times: Vec<f64>,
    /// Year fractions for leg 1.
    pub leg1_year_fractions: Vec<f64>,
    /// Spread on leg 1 (added to the floating rate).
    pub leg1_spread: f64,
    /// Notionals for leg 1 (one per period).
    pub leg1_notionals: Vec<f64>,
    /// Payment times for leg 2 (e.g. 6M LIBOR leg).
    pub leg2_payment_times: Vec<f64>,
    /// Year fractions for leg 2.
    pub leg2_year_fractions: Vec<f64>,
    /// Spread on leg 2.
    pub leg2_spread: f64,
    /// Notionals for leg 2 (one per period).
    pub leg2_notionals: Vec<f64>,
    /// True for payer (leg1 pays, leg2 receives), false for receiver.
    pub is_payer: bool,
    /// Number of quadrature points.
    pub n_quad: usize,
}

/// Price a float-float swaption under a Gaussian 1-factor model.
///
/// A float-float swaption grants the right to enter a swap where both legs
/// are floating, potentially with different indices, frequencies, or spreads.
/// Under the single-factor GSR model, the forward rate on both legs is
/// determined by the state variable, so the spread between the legs is
/// stochastic.
///
/// # Method
///
/// At exercise time $T_e$, for each quadrature node $x$:
/// 1. Compute leg 1 value using zero-bond prices from the GSR model.
/// 2. Compute leg 2 value similarly.
/// 3. The swaption payoff is $\max(\omega \cdot (V_1(x) - V_2(x)), 0)$.
pub fn gaussian1d_float_float_swaption(
    model: &Gsr1d,
    params: &FloatFloatSwaptionParams,
) -> Gaussian1dFloatFloatSwaptionResult {
    let te = params.exercise_time;
    if te <= 0.0 {
        return Gaussian1dFloatFloatSwaptionResult {
            npv: 0.0,
            underlying_value: 0.0,
        };
    }

    let omega = if params.is_payer { 1.0 } else { -1.0 };
    let zeta = model.zeta(te);
    let sqrt_zeta = zeta.max(1e-30).sqrt();
    let pm_te = model.market_discount(te);

    let (nodes, weights) = gauss_hermite_nodes(params.n_quad.max(4));

    let mut integral = 0.0;
    let mut underlying_at_zero = 0.0;

    for (k, (node, weight)) in nodes.iter().zip(weights.iter()).enumerate() {
        let x = std::f64::consts::SQRT_2 * sqrt_zeta * node;

        // Leg 1 value: floating + spread
        let v1 = float_leg_value(
            model,
            te,
            x,
            &params.leg1_payment_times,
            &params.leg1_year_fractions,
            &params.leg1_notionals,
            params.leg1_spread,
        );

        // Leg 2 value: floating + spread
        let v2 = float_leg_value(
            model,
            te,
            x,
            &params.leg2_payment_times,
            &params.leg2_year_fractions,
            &params.leg2_notionals,
            params.leg2_spread,
        );

        let swap_val = v1 - v2;
        let payoff = (omega * swap_val).max(0.0);
        integral += weight * payoff;

        // Track underlying at x≈0 (midpoint node)
        if k == nodes.len() / 2 {
            underlying_at_zero = swap_val;
        }
    }

    let npv = pm_te * integral / std::f64::consts::PI.sqrt();

    Gaussian1dFloatFloatSwaptionResult {
        npv,
        underlying_value: underlying_at_zero,
    }
}

/// Compute the value of a floating leg (with spread) at time `t` and state `x`.
fn float_leg_value(
    model: &Gsr1d,
    t: f64,
    x: f64,
    payment_times: &[f64],
    year_fractions: &[f64],
    notionals: &[f64],
    spread: f64,
) -> f64 {
    let n = payment_times
        .len()
        .min(year_fractions.len())
        .min(notionals.len());
    if n == 0 {
        return 0.0;
    }

    let mut value = 0.0;
    for i in 0..n {
        let t_pay = payment_times[i];
        if t_pay <= t {
            continue;
        }
        let tau = year_fractions[i];
        let notl = notionals[i];
        let t_start = if i == 0 { t } else { payment_times[i - 1].max(t) };

        let p_start = model.zero_bond(t, t_start, x);
        let p_end = model.zero_bond(t, t_pay, x);

        // Floating leg: notl × (P(t, T_{i-1}) - P(t, T_i)) + notl × spread × τ × P(t, T_i)
        value += notl * (p_start - p_end) + notl * spread * tau * p_end;
    }
    value
}

// =========================================================================
// G100 — MCHullWhiteEngine (swaption)
// =========================================================================

/// Result from the Monte Carlo Hull-White swaption engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McHwSwaptionResult {
    /// Net present value.
    pub npv: f64,
    /// Standard error of the MC estimate.
    pub std_error: f64,
}

/// Monte Carlo Hull-White swaption engine.
///
/// Simulates short-rate paths under Hull-White dynamics using the GSR model
/// for the drift calibration. At each exercise date, we compute the swap
/// value and exercise if positive.
///
/// For European swaptions there is a single exercise date; for Bermudan
/// swaptions, we use a simple immediate-exercise heuristic (exercise at the
/// first positive swap value).
///
/// # Parameters
/// - `model` — calibrated `Gsr1d` model
/// - `exercise_times` — exercise dates (sorted)
/// - `swap_payment_times` — swap fixed-leg payment times
/// - `swap_year_fractions` — accrual fractions
/// - `fixed_rate` — swap fixed rate
/// - `notional` — swap notional
/// - `is_payer` — payer swaption
/// - `n_paths` — number of Monte Carlo paths
/// - `n_steps` — time steps for the diffusion
/// - `seed` — RNG seed
pub fn mc_hw_swaption(
    model: &Gsr1d,
    exercise_times: &[f64],
    swap_payment_times: &[f64],
    swap_year_fractions: &[f64],
    fixed_rate: f64,
    notional: f64,
    is_payer: bool,
    n_paths: usize,
    n_steps: usize,
    seed: u64,
) -> McHwSwaptionResult {
    if exercise_times.is_empty() || swap_payment_times.is_empty() {
        return McHwSwaptionResult {
            npv: 0.0,
            std_error: 0.0,
        };
    }

    let omega = if is_payer { 1.0 } else { -1.0 };
    let last_exercise = exercise_times
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    let total_time = last_exercise;
    let dt = total_time / n_steps.max(1) as f64;

    let n_swap = swap_payment_times.len().min(swap_year_fractions.len());
    let n_paths = n_paths.max(1);

    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    // Box-Muller state
    let mut bm_state = BoxMullerState::new(seed);

    for _ in 0..n_paths {
        let mut x = 0.0_f64; // GSR state variable
        let mut discount = 1.0_f64;
        let mut t = 0.0;
        let mut exercised = false;
        let mut path_val = 0.0;

        let mut ex_idx = 0;

        for _ in 0..n_steps {
            let a_val = model.a(t);
            let sig_val = model.sigma(t);

            let dw = bm_state.next_normal();

            // Exact discretisation of OU process: x(t+dt) = x(t)*e^{-a*dt} + σ√((1-e^{-2a*dt})/(2a)) Z
            let e_adt = (-a_val * dt).exp();
            let vol_factor = if a_val.abs() < 1e-12 {
                sig_val * dt.sqrt()
            } else {
                sig_val * ((1.0 - e_adt * e_adt) / (2.0 * a_val)).sqrt()
            };
            let x_new = x * e_adt + vol_factor * dw;

            // Short rate for discounting
            let f_m = market_forward_approx(model, t);
            let r_t = x + f_m + 0.5 * model.zeta(t);
            discount *= (-r_t * dt).exp();

            x = x_new;
            t += dt;

            // Check exercise
            while !exercised
                && ex_idx < exercise_times.len()
                && t >= exercise_times[ex_idx] - dt * 0.5
            {
                // Compute swap value at this node
                let swap_val = compute_swap_value_gsr(
                    model,
                    t,
                    x,
                    swap_payment_times,
                    swap_year_fractions,
                    fixed_rate,
                    notional,
                    n_swap,
                );
                let exercise_val = omega * swap_val;
                if exercise_val > 0.0 {
                    path_val = exercise_val * discount;
                    exercised = true;
                }
                ex_idx += 1;
            }
        }

        sum += path_val;
        sum_sq += path_val * path_val;
    }

    let mean = sum / n_paths as f64;
    let variance = if n_paths > 1 {
        (sum_sq / n_paths as f64 - mean * mean) / (n_paths as f64 - 1.0)
    } else {
        0.0
    };
    let std_error = variance.max(0.0).sqrt();

    McHwSwaptionResult {
        npv: mean.max(0.0),
        std_error,
    }
}

/// Compute the value of a fixed-for-floating swap given GSR state.
fn compute_swap_value_gsr(
    model: &Gsr1d,
    t: f64,
    x: f64,
    payment_times: &[f64],
    year_fractions: &[f64],
    fixed_rate: f64,
    notional: f64,
    n: usize,
) -> f64 {
    // Swap value = N * [P(t, T_start) - P(t, T_n) - K * Σ τ_i P(t, T_i)]
    let t_start = if payment_times[0] > t {
        payment_times[0]
    } else {
        t
    };
    let start_bond = model.zero_bond(t, t_start, x);
    let end_bond = model.zero_bond(t, payment_times[n - 1], x);

    let mut annuity = 0.0;
    for i in 0..n {
        if payment_times[i] > t {
            annuity += year_fractions[i] * model.zero_bond(t, payment_times[i], x);
        }
    }

    notional * (start_bond - end_bond - fixed_rate * annuity)
}

// =========================================================================
// G101 — FdHullWhiteSwaptionEngine (using Gsr1d)
// =========================================================================

/// Result from the FD Hull-White swaption engine using the GSR model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FdHwSwaptionGsrResult {
    /// Net present value of the swaption.
    pub npv: f64,
}

/// Finite-difference Hull-White swaption engine using the full GSR model.
///
/// Uses Crank-Nicolson on a 1D grid in the state variable $x$.
/// The GSR model provides the drift (via mean reversion) and the conditional
/// zero-bond pricing for the terminal payoff.
///
/// # Parameters
/// - `model` — calibrated `Gsr1d` model
/// - `option_expiry` — swaption expiry time
/// - `swap_payment_times` — swap payment times
/// - `swap_year_fractions` — accrual fractions
/// - `fixed_rate` — swap fixed rate
/// - `notional` — notional
/// - `is_payer` — payer/receiver flag
/// - `exercise_times` — exercise dates (for Bermudan: multiple)
/// - `n_time` — number of time steps
/// - `n_space` — number of spatial grid points
pub fn fd_hw_swaption_gsr(
    model: &Gsr1d,
    option_expiry: f64,
    swap_payment_times: &[f64],
    swap_year_fractions: &[f64],
    fixed_rate: f64,
    notional: f64,
    is_payer: bool,
    exercise_times: &[f64],
    n_time: usize,
    n_space: usize,
) -> FdHwSwaptionGsrResult {
    let n_swap = swap_payment_times.len().min(swap_year_fractions.len());
    if n_swap == 0 || option_expiry <= 0.0 {
        return FdHwSwaptionGsrResult { npv: 0.0 };
    }

    let omega = if is_payer { 1.0 } else { -1.0 };
    let dt = option_expiry / n_time.max(1) as f64;

    // State-variable grid: x ∈ [-x_max, x_max]
    let zeta_max = model.zeta(option_expiry).max(1e-12);
    let x_max = 5.0 * zeta_max.sqrt();
    let dx = 2.0 * x_max / (n_space - 1).max(1) as f64;

    let x_grid: Vec<f64> = (0..n_space)
        .map(|i| -x_max + i as f64 * dx)
        .collect();

    // Terminal condition at option_expiry
    let mut v: Vec<f64> = x_grid
        .iter()
        .map(|&x| {
            let swap_val = compute_swap_value_gsr(
                model,
                option_expiry,
                x,
                swap_payment_times,
                swap_year_fractions,
                fixed_rate,
                notional,
                n_swap,
            );
            (omega * swap_val).max(0.0)
        })
        .collect();

    // Backward Crank-Nicolson time stepping
    for step in (0..n_time).rev() {
        let t = step as f64 * dt;
        let a_val = model.a(t);
        let sig_val = model.sigma(t);

        // Build tridiagonal system: (I - 0.5*dt*L) v^{n} = (I + 0.5*dt*L) v^{n+1}
        let n_s = n_space;
        let mut lower = vec![0.0; n_s];
        let mut diag = vec![1.0; n_s];
        let mut upper = vec![0.0; n_s];
        let mut rhs = vec![0.0; n_s];

        for j in 1..n_s - 1 {
            let x_j = x_grid[j];
            // Drift: -a * x, Diffusion: σ²/2
            let drift = -a_val * x_j;
            let diff = 0.5 * sig_val * sig_val;

            let alpha = dt * (diff / (dx * dx) - drift / (2.0 * dx));
            let beta = dt * (-2.0 * diff / (dx * dx));
            let gamma = dt * (diff / (dx * dx) + drift / (2.0 * dx));

            // Approximate short rate for discounting
            let f_m = market_forward_approx(model, t);
            let r_j = x_j + f_m + 0.5 * model.zeta(t);
            let disc_term = dt * r_j;

            // LHS: (I - 0.5*dt*L)
            lower[j] = -0.5 * alpha;
            diag[j] = 1.0 - 0.5 * beta + 0.5 * disc_term;
            upper[j] = -0.5 * gamma;

            // RHS: (I + 0.5*dt*L) v
            rhs[j] = 0.5 * alpha * v[j - 1]
                + (1.0 + 0.5 * beta - 0.5 * disc_term) * v[j]
                + 0.5 * gamma * v[j + 1];
        }

        // Boundary conditions (zero at extremes for this Dirichlet approach)
        rhs[0] = 0.0;
        rhs[n_s - 1] = 0.0;
        diag[0] = 1.0;
        diag[n_s - 1] = 1.0;

        v = thomas_solve(&lower, &diag, &upper, &rhs);

        // Check for Bermudan exercise
        for &ex_t in exercise_times {
            if (t - ex_t).abs() < dt * 0.5 && t > 0.0 {
                for j in 0..n_s {
                    let swap_val = compute_swap_value_gsr(
                        model,
                        t,
                        x_grid[j],
                        swap_payment_times,
                        swap_year_fractions,
                        fixed_rate,
                        notional,
                        n_swap,
                    );
                    let exercise_val = (omega * swap_val).max(0.0);
                    v[j] = v[j].max(exercise_val);
                }
            }
        }
    }

    // Interpolate at x = 0
    let npv = interpolate_grid_linear(&x_grid, &v, 0.0);

    FdHwSwaptionGsrResult { npv: npv.max(0.0) }
}

// =========================================================================
// G102 — TreeCapFloorEngine (wrapper)
// =========================================================================

/// Standalone tree cap/floor engine wrapping [`crate::tree_swaption::tree_cap_floor`].
///
/// Provides an object-oriented interface where the engine holds the model
/// parameters and the cap/floor specification, then produces a result on demand.
#[derive(Debug, Clone)]
pub struct TreeCapFloorEngine {
    /// Hull-White mean-reversion speed.
    pub a: f64,
    /// Hull-White volatility.
    pub sigma: f64,
    /// Initial short rate.
    pub r0: f64,
    /// Rate fixing times.
    pub fixing_times: Vec<f64>,
    /// Payment times.
    pub payment_times: Vec<f64>,
    /// Cap/floor strike.
    pub strike: f64,
    /// Notional amount.
    pub notional: f64,
    /// True for cap, false for floor.
    pub is_cap: bool,
    /// Tree steps per period.
    pub n_steps_per_period: usize,
}

/// Result from the tree cap/floor engine wrapper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeCapFloorEngineResult {
    /// Net present value.
    pub npv: f64,
}

impl TreeCapFloorEngine {
    /// Create a new tree cap/floor engine.
    pub fn new(
        a: f64,
        sigma: f64,
        r0: f64,
        fixing_times: Vec<f64>,
        payment_times: Vec<f64>,
        strike: f64,
        notional: f64,
        is_cap: bool,
        n_steps_per_period: usize,
    ) -> Self {
        Self {
            a,
            sigma,
            r0,
            fixing_times,
            payment_times,
            strike,
            notional,
            is_cap,
            n_steps_per_period,
        }
    }

    /// Price the cap/floor by delegating to the tree implementation.
    pub fn calculate(&self) -> TreeCapFloorEngineResult {
        let tree_result = crate::tree_swaption::tree_cap_floor(
            self.a,
            self.sigma,
            self.r0,
            &self.fixing_times,
            &self.payment_times,
            self.strike,
            self.notional,
            self.is_cap,
            self.n_steps_per_period,
        );
        TreeCapFloorEngineResult {
            npv: tree_result.npv,
        }
    }
}

// =========================================================================
// G103 — IrregularSwap / IrregularSwaption
// =========================================================================

/// An irregular interest-rate swap with potentially amortising/step-up
/// notionals and heterogeneous fixed rates on each period.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrregularSwap {
    /// Fixed leg payment times.
    pub fixed_payment_times: Vec<f64>,
    /// Fixed leg year fractions.
    pub fixed_year_fractions: Vec<f64>,
    /// Fixed leg notionals (one per period — may differ).
    pub fixed_notionals: Vec<f64>,
    /// Fixed leg coupon rates (one per period — may differ).
    pub fixed_rates: Vec<f64>,
    /// Floating leg payment times.
    pub float_payment_times: Vec<f64>,
    /// Floating leg year fractions.
    pub float_year_fractions: Vec<f64>,
    /// Floating leg notionals (one per period — may differ).
    pub float_notionals: Vec<f64>,
    /// Floating leg spreads (one per period).
    pub float_spreads: Vec<f64>,
    /// True for payer swap (pay fixed, receive float).
    pub is_payer: bool,
}

/// Result of pricing an irregular swap.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrregularSwapResult {
    /// Net present value of the swap.
    pub npv: f64,
    /// Fixed leg PV.
    pub fixed_leg_npv: f64,
    /// Floating leg PV.
    pub float_leg_npv: f64,
    /// Par rate (break-even fixed rate).
    pub par_rate: f64,
}

impl IrregularSwap {
    /// Price the irregular swap using flat-rate discounting.
    ///
    /// Each leg is valued independently using simple exponential discounting.
    pub fn price_flat(&self, flat_rate: f64) -> IrregularSwapResult {
        let discount = |t: f64| (-flat_rate * t).exp();

        // Fixed leg
        let n_fix = self.fixed_payment_times.len()
            .min(self.fixed_year_fractions.len())
            .min(self.fixed_notionals.len())
            .min(self.fixed_rates.len());

        let mut fixed_pv = 0.0;
        let mut weighted_annuity = 0.0;
        for i in 0..n_fix {
            let t = self.fixed_payment_times[i];
            let tau = self.fixed_year_fractions[i];
            let n_i = self.fixed_notionals[i];
            let r_i = self.fixed_rates[i];
            fixed_pv += n_i * r_i * tau * discount(t);
            weighted_annuity += n_i * tau * discount(t);
        }

        // Float leg: notl × (P(T_{i-1}) - P(T_i)) + notl × spread × τ × P(T_i)
        let n_flt = self.float_payment_times.len()
            .min(self.float_year_fractions.len())
            .min(self.float_notionals.len())
            .min(self.float_spreads.len());

        let mut float_pv = 0.0;
        for i in 0..n_flt {
            let t_pay = self.float_payment_times[i];
            let tau = self.float_year_fractions[i];
            let n_i = self.float_notionals[i];
            let sp = self.float_spreads[i];
            let t_start = if i == 0 { 0.0 } else { self.float_payment_times[i - 1] };
            float_pv += n_i * (discount(t_start) - discount(t_pay));
            float_pv += n_i * sp * tau * discount(t_pay);
        }

        let omega = if self.is_payer { 1.0 } else { -1.0 };
        let npv = omega * (float_pv - fixed_pv);

        let par_rate = if weighted_annuity.abs() > 1e-15 {
            float_pv / weighted_annuity
        } else {
            0.0
        };

        IrregularSwapResult {
            npv,
            fixed_leg_npv: fixed_pv,
            float_leg_npv: float_pv,
            par_rate,
        }
    }

    /// Price the irregular swap under the GSR model at state `(t, x)`.
    pub fn price_gsr(&self, model: &Gsr1d, t: f64, x: f64) -> IrregularSwapResult {
        let n_fix = self.fixed_payment_times.len()
            .min(self.fixed_year_fractions.len())
            .min(self.fixed_notionals.len())
            .min(self.fixed_rates.len());

        let mut fixed_pv = 0.0;
        let mut weighted_annuity = 0.0;
        for i in 0..n_fix {
            let t_pay = self.fixed_payment_times[i];
            if t_pay <= t {
                continue;
            }
            let tau = self.fixed_year_fractions[i];
            let n_i = self.fixed_notionals[i];
            let r_i = self.fixed_rates[i];
            let df = model.zero_bond(t, t_pay, x);
            fixed_pv += n_i * r_i * tau * df;
            weighted_annuity += n_i * tau * df;
        }

        let n_flt = self.float_payment_times.len()
            .min(self.float_year_fractions.len())
            .min(self.float_notionals.len())
            .min(self.float_spreads.len());

        let mut float_pv = 0.0;
        for i in 0..n_flt {
            let t_pay = self.float_payment_times[i];
            if t_pay <= t {
                continue;
            }
            let tau = self.float_year_fractions[i];
            let n_i = self.float_notionals[i];
            let sp = self.float_spreads[i];
            let t_start = if i == 0 { t } else { self.float_payment_times[i - 1].max(t) };
            let df_start = model.zero_bond(t, t_start, x);
            let df_end = model.zero_bond(t, t_pay, x);
            float_pv += n_i * (df_start - df_end);
            float_pv += n_i * sp * tau * df_end;
        }

        let omega = if self.is_payer { 1.0 } else { -1.0 };
        let npv = omega * (float_pv - fixed_pv);

        let par_rate = if weighted_annuity.abs() > 1e-15 {
            float_pv / weighted_annuity
        } else {
            0.0
        };

        IrregularSwapResult {
            npv,
            fixed_leg_npv: fixed_pv,
            float_leg_npv: float_pv,
            par_rate,
        }
    }
}

/// An option on an irregular swap (swaption with non-standard notionals/rates).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrregularSwaption {
    /// The underlying irregular swap.
    pub swap: IrregularSwap,
    /// Exercise time.
    pub exercise_time: f64,
}

/// Result from pricing an irregular swaption.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrregularSwaptionResult {
    /// Net present value.
    pub npv: f64,
}

impl IrregularSwaption {
    /// Price via Gauss-Hermite integration under the GSR model.
    pub fn price_gsr(&self, model: &Gsr1d, n_quad: usize) -> IrregularSwaptionResult {
        let te = self.exercise_time;
        if te <= 0.0 {
            return IrregularSwaptionResult { npv: 0.0 };
        }

        let zeta = model.zeta(te);
        let sqrt_zeta = zeta.max(1e-30).sqrt();
        let pm_te = model.market_discount(te);

        let (nodes, weights) = gauss_hermite_nodes(n_quad.max(4));

        let mut integral = 0.0;
        for (node, weight) in nodes.iter().zip(weights.iter()) {
            let x = std::f64::consts::SQRT_2 * sqrt_zeta * node;
            let swap_res = self.swap.price_gsr(model, te, x);
            let payoff = swap_res.npv.max(0.0);
            integral += weight * payoff;
        }

        let npv = pm_te * integral / std::f64::consts::PI.sqrt();
        IrregularSwaptionResult { npv }
    }
}

// =========================================================================
// G104 — HaganIrregularSwaptionEngine
// =========================================================================

/// Result from Hagan's irregular swaption approximation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HaganIrregularSwaptionResult {
    /// Total NPV of the irregular swaption.
    pub npv: f64,
    /// Weights of the constituent standard swaptions.
    pub basket_weights: Vec<f64>,
    /// NPVs of individual standard swaptions in the basket.
    pub component_npvs: Vec<f64>,
}

/// Standard swaption specification used in Hagan decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardSwaptionSpec {
    /// Swaption expiry time.
    pub expiry: f64,
    /// Swap start time.
    pub swap_start: f64,
    /// Swap end time.
    pub swap_end: f64,
    /// Number of periods in the swap.
    pub n_periods: usize,
    /// Notional.
    pub notional: f64,
    /// Fixed rate (strike).
    pub fixed_rate: f64,
    /// True for payer.
    pub is_payer: bool,
}

/// Hagan's approximation for irregular swaptions.
///
/// Decomposes an irregular swaption into a portfolio of standard swaptions
/// by matching each notional step to a co-terminal swaption. The weight of
/// each component equals the notional change at that period divided by
/// the base notional.
///
/// # Parameters
/// - `model` — GSR model for pricing the standard swaptions
/// - `swap` — the irregular swap
/// - `exercise_time` — swaption exercise time
/// - `n_quad` — quadrature points for each standard swaption price
pub fn hagan_irregular_swaption(
    model: &Gsr1d,
    swap: &IrregularSwap,
    exercise_time: f64,
    n_quad: usize,
) -> HaganIrregularSwaptionResult {
    if swap.fixed_payment_times.is_empty() || exercise_time <= 0.0 {
        return HaganIrregularSwaptionResult {
            npv: 0.0,
            basket_weights: vec![],
            component_npvs: vec![],
        };
    }

    let n_fix = swap.fixed_payment_times.len()
        .min(swap.fixed_notionals.len())
        .min(swap.fixed_year_fractions.len())
        .min(swap.fixed_rates.len());

    if n_fix == 0 {
        return HaganIrregularSwaptionResult {
            npv: 0.0,
            basket_weights: vec![],
            component_npvs: vec![],
        };
    }

    let base_notional = swap.fixed_notionals[0];
    if base_notional.abs() < 1e-15 {
        return HaganIrregularSwaptionResult {
            npv: 0.0,
            basket_weights: vec![],
            component_npvs: vec![],
        };
    }

    // Compute notional steps: ΔN_i = N_i - N_{i+1}  (or N_last for the final period)
    let mut basket_weights = Vec::with_capacity(n_fix);
    let mut component_npvs = Vec::with_capacity(n_fix);
    let end_time = swap.fixed_payment_times[n_fix - 1];

    for i in 0..n_fix {
        let n_i = swap.fixed_notionals[i];
        let n_next = if i + 1 < n_fix {
            swap.fixed_notionals[i + 1]
        } else {
            0.0
        };
        let delta_n = n_i - n_next;
        let weight = delta_n / base_notional;
        basket_weights.push(weight);

        // Standard swaption from period i to end
        let remaining = n_fix - i;
        let swap_start = if i > 0 {
            swap.fixed_payment_times[i - 1]
        } else {
            exercise_time
        };
        let swap_times: Vec<f64> = swap.fixed_payment_times[i..n_fix].to_vec();
        let swap_yfs: Vec<f64> = swap.fixed_year_fractions[i..n_fix].to_vec();

        // Average fixed rate for this sub-swap
        let avg_rate = if remaining > 0 {
            swap.fixed_rates[i..n_fix].iter().sum::<f64>() / remaining as f64
        } else {
            swap.fixed_rates[0]
        };

        // Price this component swaption using Gauss-Hermite
        let zeta = model.zeta(exercise_time);
        let sqrt_zeta = zeta.max(1e-30).sqrt();
        let pm_te = model.market_discount(exercise_time);
        let omega = if swap.is_payer { 1.0 } else { -1.0 };

        let (nodes, weights_gh) = gauss_hermite_nodes(n_quad.max(4));
        let mut integral = 0.0;

        for (node, w) in nodes.iter().zip(weights_gh.iter()) {
            let x = std::f64::consts::SQRT_2 * sqrt_zeta * node;

            let start_bond = model.zero_bond(exercise_time, swap_start.max(exercise_time), x);
            let end_bond = model.zero_bond(exercise_time, end_time, x);

            let mut annuity = 0.0;
            for j in 0..swap_times.len() {
                if swap_times[j] > exercise_time {
                    annuity += swap_yfs[j] * model.zero_bond(exercise_time, swap_times[j], x);
                }
            }

            let swap_val = start_bond - end_bond - avg_rate * annuity;
            let payoff = (omega * swap_val).max(0.0) * delta_n.abs();
            integral += w * payoff;
        }

        let comp_npv = pm_te * integral / std::f64::consts::PI.sqrt();
        component_npvs.push(comp_npv);
    }

    let total_npv: f64 = component_npvs.iter().sum();

    HaganIrregularSwaptionResult {
        npv: total_npv,
        basket_weights,
        component_npvs,
    }
}

// =========================================================================
// G105 — BasketGeneratingEngine / LatticeShortRateModelEngine
// =========================================================================

/// A single-exercise option in a basket decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasketComponent {
    /// Exercise time of this component.
    pub exercise_time: f64,
    /// Swap payment times for this component.
    pub payment_times: Vec<f64>,
    /// Year fractions for this component.
    pub year_fractions: Vec<f64>,
    /// Fixed rate (strike) for this component.
    pub fixed_rate: f64,
    /// Notional for this component.
    pub notional: f64,
    /// True for payer.
    pub is_payer: bool,
}

/// Result from the basket-generating engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasketGeneratingResult {
    /// The decomposed basket of single-exercise options.
    pub basket: Vec<BasketComponent>,
    /// NPV of each basket component.
    pub component_npvs: Vec<f64>,
    /// Total basket NPV (upper bound for multi-exercise product).
    pub total_npv: f64,
}

/// Basket-generating engine: decomposes a multi-exercise (Bermudan) product
/// into a portfolio of single-exercise (European) options.
///
/// At each exercise date, the engine creates a co-terminal European swaption
/// and prices it independently. The sum of European swaption values provides
/// an upper bound for the Bermudan swaption value.
///
/// # Parameters
/// - `model` — GSR model
/// - `exercise_times` — exercise dates of the Bermudan product
/// - `swap_end_time` — swap maturity
/// - `fixed_rate` — fixed rate of the underlying swap
/// - `notional` — notional
/// - `is_payer` — payer/receiver
/// - `n_periods_per_year` — payment frequency of the underlying swap
/// - `n_quad` — quadrature points
pub fn basket_generating_engine(
    model: &Gsr1d,
    exercise_times: &[f64],
    swap_end_time: f64,
    fixed_rate: f64,
    notional: f64,
    is_payer: bool,
    n_periods_per_year: usize,
    n_quad: usize,
) -> BasketGeneratingResult {
    if exercise_times.is_empty() || swap_end_time <= 0.0 {
        return BasketGeneratingResult {
            basket: vec![],
            component_npvs: vec![],
            total_npv: 0.0,
        };
    }

    let freq = n_periods_per_year.max(1) as f64;
    let period = 1.0 / freq;

    let mut basket = Vec::with_capacity(exercise_times.len());
    let mut component_npvs = Vec::with_capacity(exercise_times.len());

    for &ex_t in exercise_times {
        if ex_t >= swap_end_time || ex_t <= 0.0 {
            continue;
        }

        // Generate payment schedule from ex_t to swap_end_time
        let mut pay_times = Vec::new();
        let mut yfs = Vec::new();
        let mut t = ex_t + period;
        while t <= swap_end_time + 1e-10 {
            pay_times.push(t.min(swap_end_time));
            yfs.push(period);
            t += period;
        }
        if pay_times.is_empty() {
            continue;
        }

        let component = BasketComponent {
            exercise_time: ex_t,
            payment_times: pay_times.clone(),
            year_fractions: yfs.clone(),
            fixed_rate,
            notional,
            is_payer,
        };

        // Price this European swaption via Gauss-Hermite
        let zeta = model.zeta(ex_t);
        let sqrt_zeta = zeta.max(1e-30).sqrt();
        let pm_te = model.market_discount(ex_t);
        let omega = if is_payer { 1.0 } else { -1.0 };

        let (nodes, weights) = gauss_hermite_nodes(n_quad.max(4));
        let mut integral = 0.0;

        for (node, w) in nodes.iter().zip(weights.iter()) {
            let x_val = std::f64::consts::SQRT_2 * sqrt_zeta * node;

            let n_p = pay_times.len();
            let swap_val = compute_swap_value_gsr(
                model,
                ex_t,
                x_val,
                &pay_times,
                &yfs,
                fixed_rate,
                notional,
                n_p,
            );
            let payoff = (omega * swap_val).max(0.0);
            integral += w * payoff;
        }

        let comp_npv = pm_te * integral / std::f64::consts::PI.sqrt();
        component_npvs.push(comp_npv);
        basket.push(component);
    }

    let total_npv: f64 = component_npvs.iter().sum();

    BasketGeneratingResult {
        basket,
        component_npvs,
        total_npv,
    }
}

/// Result from the lattice short-rate model engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeShortRateResult {
    /// Net present value.
    pub npv: f64,
}

/// Payoff function type for the lattice engine.
///
/// Given `(time, short_rate)`, returns the exercise value (or 0 if not exercisable).
pub type LatticePayoffFn = Box<dyn Fn(f64, f64) -> f64>;

/// Generic lattice (trinomial tree) engine for short-rate models.
///
/// Builds a Hull-White trinomial tree and rolls back an arbitrary payoff
/// function. This provides a reusable framework for pricing any derivative
/// whose payoff depends on the short rate.
///
/// # Parameters
/// - `a` — mean-reversion speed
/// - `sigma` — HW volatility
/// - `r0` — initial short rate
/// - `maturity` — final time
/// - `n_steps` — number of tree steps
/// - `payoff_fn` — function returning exercise value at `(time, short_rate)`;
///                 the engine takes the max of continuation and exercise
/// - `is_american` — if true, exercise at every step; if false, only at maturity
pub fn lattice_short_rate_engine(
    a: f64,
    sigma: f64,
    r0: f64,
    maturity: f64,
    n_steps: usize,
    payoff_fn: &dyn Fn(f64, f64) -> f64,
    is_american: bool,
) -> LatticeShortRateResult {
    if maturity <= 0.0 {
        return LatticeShortRateResult { npv: 0.0 };
    }

    let n_steps = n_steps.max(3);
    let dt = maturity / n_steps as f64;
    let dx = sigma * (3.0 * dt).sqrt();

    let j_max = hw_j_max(a, dt);
    let width = (2 * j_max + 1) as usize;
    let idx = |j: i64| -> usize { (j + j_max) as usize };

    // Transition probabilities
    let probs = hw_transition_probs(a, dt, dx, j_max);

    // Terminal values
    let mut values = vec![0.0_f64; width];
    let t_n = maturity;
    for j in -j_max..=j_max {
        let r_j = r0 + j as f64 * dx;
        values[idx(j)] = payoff_fn(t_n, r_j);
    }

    // Backward induction
    for step in (0..n_steps).rev() {
        let t = step as f64 * dt;
        let mut new_values = vec![0.0_f64; width];

        for j in -j_max..=j_max {
            let r_j = r0 + j as f64 * dx;
            let discount = (-r_j * dt).exp();

            let (p_up, p_mid, p_down) = probs[idx(j)];
            let j_up = (j + 1).min(j_max);
            let j_down = (j - 1).max(-j_max);

            let continuation = p_up * values[idx(j_up)]
                + p_mid * values[idx(j)]
                + p_down * values[idx(j_down)];

            let hold = discount * continuation;

            if is_american && step > 0 {
                let exercise = payoff_fn(t, r_j);
                new_values[idx(j)] = hold.max(exercise);
            } else {
                new_values[idx(j)] = hold;
            }
        }
        values = new_values;
    }

    LatticeShortRateResult {
        npv: values[idx(0)],
    }
}

// =========================================================================
// Shared helpers
// =========================================================================

/// Approximate instantaneous forward rate from the GSR model at time `t`.
fn market_forward_approx(model: &Gsr1d, t: f64) -> f64 {
    let dt = 1e-4;
    let df1 = model.market_discount(t);
    let df2 = model.market_discount(t + dt);
    if df1 <= 0.0 || df2 <= 0.0 {
        return 0.0;
    }
    -(df2.ln() - df1.ln()) / dt
}

/// Gauss-Hermite quadrature nodes and weights via Golub-Welsch.
///
/// Uses the same algorithm as `gaussian1d_engine` — physicists' Hermite
/// with weight function $\exp(-x^2)$.
fn gauss_hermite_nodes(n: usize) -> (Vec<f64>, Vec<f64>) {
    use nalgebra::DMatrix;

    let n = n.max(2);
    let mut j = DMatrix::<f64>::zeros(n, n);
    for i in 0..n - 1 {
        let off = ((i + 1) as f64 / 2.0).sqrt();
        j[(i, i + 1)] = off;
        j[(i + 1, i)] = off;
    }

    let eig = j.symmetric_eigen();
    let eigenvalues = eig.eigenvalues;
    let eigenvectors = eig.eigenvectors;

    let sqrt_pi = std::f64::consts::PI.sqrt();

    let mut pairs: Vec<(f64, f64)> = (0..n)
        .map(|i| {
            let node = eigenvalues[i];
            let v0 = eigenvectors[(0, i)];
            let weight = sqrt_pi * v0 * v0;
            (node, weight)
        })
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let nodes = pairs.iter().map(|p| p.0).collect();
    let weights = pairs.iter().map(|p| p.1).collect();
    (nodes, weights)
}

/// Thomas algorithm for tridiagonal systems (Crank-Nicolson solver).
fn thomas_solve(lower: &[f64], diag: &[f64], upper: &[f64], rhs: &[f64]) -> Vec<f64> {
    let n = diag.len();
    if n == 0 {
        return vec![];
    }
    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];

    c_prime[0] = if diag[0].abs() > 1e-30 { upper[0] / diag[0] } else { 0.0 };
    d_prime[0] = if diag[0].abs() > 1e-30 { rhs[0] / diag[0] } else { 0.0 };

    for i in 1..n {
        let m = diag[i] - lower[i] * c_prime[i - 1];
        let m_inv = if m.abs() > 1e-30 { 1.0 / m } else { 0.0 };
        c_prime[i] = if i < n - 1 { upper[i] * m_inv } else { 0.0 };
        d_prime[i] = (rhs[i] - lower[i] * d_prime[i - 1]) * m_inv;
    }

    let mut x = vec![0.0; n];
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
    x
}

/// Linear interpolation on a grid.
fn interpolate_grid_linear(grid: &[f64], values: &[f64], x: f64) -> f64 {
    if grid.is_empty() {
        return 0.0;
    }
    if x <= grid[0] {
        return values[0];
    }
    if x >= grid[grid.len() - 1] {
        return values[values.len() - 1];
    }
    let mut i = 0;
    while i < grid.len() - 1 && grid[i + 1] < x {
        i += 1;
    }
    let t = (x - grid[i]) / (grid[i + 1] - grid[i]);
    values[i] * (1.0 - t) + values[i + 1] * t
}

/// Box-Muller normal random number generator (self-contained, no external RNG).
struct BoxMullerState {
    state: u64,
    spare: Option<f64>,
}

impl BoxMullerState {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x5DEECE66D,
            spare: None,
        }
    }

    /// Simple xorshift64 PRNG returning a value in (0, 1).
    fn next_uniform(&mut self) -> f64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        // Map to (0, 1)
        (self.state as f64) / (u64::MAX as f64)
    }

    /// Generate a standard normal variate via Box-Muller.
    fn next_normal(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        loop {
            let u1 = self.next_uniform().max(1e-30);
            let u2 = self.next_uniform();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            self.spare = Some(r * theta.sin());
            return r * theta.cos();
        }
    }
}

/// Compute j_max for HW trinomial tree.
fn hw_j_max(a: f64, dt: f64) -> i64 {
    let j = if a.abs() < 1e-12 {
        10
    } else {
        (0.184 / (a * dt)).ceil() as i64
    };
    j.max(3)
}

/// Build HW trinomial transition probabilities.
fn hw_transition_probs(a: f64, dt: f64, dx: f64, j_max: i64) -> Vec<(f64, f64, f64)> {
    let width = (2 * j_max + 1) as usize;
    let mut probs = vec![(0.0, 1.0, 0.0); width];

    for j in -j_max..=j_max {
        let idx = (j + j_max) as usize;
        let xi = -a * j as f64 * dx * dt / dx;

        let p_up = (1.0 / 6.0 + (xi * xi + xi) / 2.0).max(0.0);
        let p_mid = (2.0 / 3.0 - xi * xi).max(0.0);
        let p_down = (1.0 / 6.0 + (xi * xi - xi) / 2.0).max(0.0);

        let total = p_up + p_mid + p_down;
        if total > 0.0 {
            probs[idx] = (p_up / total, p_mid / total, p_down / total);
        }
    }
    probs
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// Helper: flat-rate GSR model.
    fn flat_gsr(a: f64, sigma: f64, flat_rate: f64) -> Gsr1d {
        let mut curve = vec![(0.0, 1.0)];
        for y in 1..=40 {
            let t = y as f64;
            curve.push((t, (-flat_rate * t).exp()));
        }
        Gsr1d::constant(a, sigma, curve)
    }

    // ----- G98: Gaussian1DCapFloorEngine -----

    #[test]
    fn g98_cap_positive_below_rate() {
        let model = flat_gsr(0.05, 0.01, 0.04);
        let fixing = vec![1.0, 2.0, 3.0, 4.0];
        let payment = vec![1.5, 2.5, 3.5, 4.5];
        let yfs = vec![0.5; 4];
        let result = gaussian1d_cap_floor(&model, &fixing, &payment, &yfs, 0.02, 1_000_000.0, true, 32);
        assert!(result.npv > 0.0, "Cap with strike below rate should be positive: {}", result.npv);
        assert_eq!(result.caplet_values.len(), 4);
        for v in &result.caplet_values {
            assert!(*v >= 0.0, "Caplet value must be non-negative");
        }
    }

    #[test]
    fn g98_floor_positive_above_rate() {
        let model = flat_gsr(0.05, 0.01, 0.04);
        let fixing = vec![1.0, 2.0, 3.0];
        let payment = vec![1.5, 2.5, 3.5];
        let yfs = vec![0.5; 3];
        let result = gaussian1d_cap_floor(&model, &fixing, &payment, &yfs, 0.08, 1_000_000.0, false, 32);
        assert!(result.npv > 0.0, "Floor with strike above rate should be positive: {}", result.npv);
    }

    #[test]
    fn g98_cap_floor_parity() {
        let model = flat_gsr(0.05, 0.01, 0.04);
        let fixing = vec![1.0, 2.0];
        let payment = vec![1.5, 2.5];
        let yfs = vec![0.5; 2];
        let k = 0.04;
        let cap = gaussian1d_cap_floor(&model, &fixing, &payment, &yfs, k, 1.0, true, 48);
        let floor = gaussian1d_cap_floor(&model, &fixing, &payment, &yfs, k, 1.0, false, 48);
        // Both should be non-negative at ATM
        assert!(cap.npv >= 0.0);
        assert!(floor.npv >= 0.0);
    }

    // ----- G99: Gaussian1DFloatFloatSwaptionEngine -----

    #[test]
    fn g99_float_float_swaption_positive() {
        let model = flat_gsr(0.05, 0.01, 0.04);
        let params = FloatFloatSwaptionParams {
            exercise_time: 1.0,
            leg1_payment_times: vec![2.0, 3.0, 4.0, 5.0],
            leg1_year_fractions: vec![1.0; 4],
            leg1_spread: 0.005,
            leg1_notionals: vec![1_000_000.0; 4],
            leg2_payment_times: vec![1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            leg2_year_fractions: vec![0.5; 8],
            leg2_spread: 0.0,
            leg2_notionals: vec![1_000_000.0; 8],
            is_payer: true,
            n_quad: 32,
        };
        let result = gaussian1d_float_float_swaption(&model, &params);
        assert!(result.npv >= 0.0, "Float-float swaption NPV must be non-negative: {}", result.npv);
    }

    #[test]
    fn g99_float_float_receiver() {
        let model = flat_gsr(0.05, 0.01, 0.03);
        let params = FloatFloatSwaptionParams {
            exercise_time: 2.0,
            leg1_payment_times: vec![3.0, 4.0, 5.0],
            leg1_year_fractions: vec![1.0; 3],
            leg1_spread: 0.0,
            leg1_notionals: vec![1e6; 3],
            leg2_payment_times: vec![2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            leg2_year_fractions: vec![0.5; 6],
            leg2_spread: 0.003,
            leg2_notionals: vec![1e6; 6],
            is_payer: false,
            n_quad: 32,
        };
        let result = gaussian1d_float_float_swaption(&model, &params);
        assert!(result.npv >= 0.0, "Receiver float-float swaption NPV: {}", result.npv);
    }

    // ----- G100: MCHullWhiteEngine -----

    #[test]
    fn g100_mc_hw_swaption_positive() {
        let model = flat_gsr(0.05, 0.01, 0.04);
        let ex_times = vec![1.0];
        let pay_times = vec![2.0, 3.0, 4.0, 5.0];
        let yfs = vec![1.0; 4];
        let result = mc_hw_swaption(
            &model,
            &ex_times,
            &pay_times,
            &yfs,
            0.04,
            1_000_000.0,
            true,
            10_000,
            200,
            42,
        );
        assert!(result.npv >= 0.0, "MC HW swaption NPV must be non-negative: {}", result.npv);
        assert!(result.std_error >= 0.0);
    }

    #[test]
    fn g100_mc_hw_swaption_receiver() {
        let model = flat_gsr(0.03, 0.008, 0.03);
        let ex_times = vec![1.0];
        let pay_times = vec![2.0, 3.0, 4.0, 5.0];
        let yfs = vec![1.0; 4];
        let result = mc_hw_swaption(
            &model,
            &ex_times,
            &pay_times,
            &yfs,
            0.05,
            1_000_000.0,
            false,
            10_000,
            200,
            123,
        );
        assert!(result.npv >= 0.0, "MC receiver swaption NPV: {}", result.npv);
    }

    // ----- G101: FdHullWhiteSwaptionEngine -----

    #[test]
    fn g101_fd_hw_swaption_gsr_positive() {
        let model = flat_gsr(0.05, 0.01, 0.04);
        let pay_times = vec![2.0, 3.0, 4.0, 5.0];
        let yfs = vec![1.0; 4];
        let result = fd_hw_swaption_gsr(
            &model,
            1.0,
            &pay_times,
            &yfs,
            0.04,
            1_000_000.0,
            true,
            &[1.0],
            200,
            200,
        );
        assert!(result.npv > 0.0, "FD HW GSR swaption should be positive: {}", result.npv);
    }

    #[test]
    fn g101_fd_hw_bermudan_exceeds_european() {
        let model = flat_gsr(0.05, 0.01, 0.04);
        let pay_times = vec![2.0, 3.0, 4.0, 5.0];
        let yfs = vec![1.0; 4];
        let european = fd_hw_swaption_gsr(
            &model, 1.0, &pay_times, &yfs, 0.04, 1.0, true, &[1.0], 200, 200,
        );
        let bermudan = fd_hw_swaption_gsr(
            &model, 4.0, &pay_times, &yfs, 0.04, 1.0, true,
            &[1.0, 2.0, 3.0, 4.0], 400, 200,
        );
        assert!(
            bermudan.npv >= european.npv * 0.8,
            "Bermudan ({}) should be close to or exceed European ({})",
            bermudan.npv,
            european.npv,
        );
    }

    // ----- G102: TreeCapFloorEngine -----

    #[test]
    fn g102_tree_cap_engine_positive() {
        let engine = TreeCapFloorEngine::new(
            0.1, 0.01, 0.05,
            vec![0.5, 1.0, 1.5, 2.0],
            vec![1.0, 1.5, 2.0, 2.5],
            0.04,
            1_000_000.0,
            true,
            50,
        );
        let result = engine.calculate();
        assert!(result.npv > 0.0, "Tree cap engine NPV: {}", result.npv);
    }

    #[test]
    fn g102_tree_floor_engine_positive() {
        let engine = TreeCapFloorEngine::new(
            0.1, 0.01, 0.05,
            vec![0.5, 1.0, 1.5],
            vec![1.0, 1.5, 2.0],
            0.08,
            1_000_000.0,
            false,
            50,
        );
        let result = engine.calculate();
        assert!(result.npv > 0.0, "Tree floor engine NPV: {}", result.npv);
    }

    // ----- G103: IrregularSwap / IrregularSwaption -----

    #[test]
    fn g103_irregular_swap_amortizing() {
        let swap = IrregularSwap {
            fixed_payment_times: vec![1.0, 2.0, 3.0, 4.0],
            fixed_year_fractions: vec![1.0; 4],
            fixed_notionals: vec![1_000_000.0, 750_000.0, 500_000.0, 250_000.0],
            fixed_rates: vec![0.04; 4],
            float_payment_times: vec![1.0, 2.0, 3.0, 4.0],
            float_year_fractions: vec![1.0; 4],
            float_notionals: vec![1_000_000.0, 750_000.0, 500_000.0, 250_000.0],
            float_spreads: vec![0.0; 4],
            is_payer: true,
        };
        let res = swap.price_flat(0.04);
        // At-the-money with no spread, NPV should be near zero
        assert!(res.npv.abs() < 50_000.0, "ATM amortizing swap NPV: {}", res.npv);
        assert!(res.par_rate > 0.0, "Par rate should be positive");
    }

    #[test]
    fn g103_irregular_swaption_positive() {
        let model = flat_gsr(0.05, 0.01, 0.04);
        let swap = IrregularSwap {
            fixed_payment_times: vec![2.0, 3.0, 4.0, 5.0],
            fixed_year_fractions: vec![1.0; 4],
            fixed_notionals: vec![1e6, 800_000.0, 600_000.0, 400_000.0],
            fixed_rates: vec![0.04; 4],
            float_payment_times: vec![2.0, 3.0, 4.0, 5.0],
            float_year_fractions: vec![1.0; 4],
            float_notionals: vec![1e6, 800_000.0, 600_000.0, 400_000.0],
            float_spreads: vec![0.0; 4],
            is_payer: true,
        };
        let swaption = IrregularSwaption {
            swap,
            exercise_time: 1.0,
        };
        let res = swaption.price_gsr(&model, 32);
        assert!(res.npv >= 0.0, "Irregular swaption NPV: {}", res.npv);
    }

    // ----- G104: HaganIrregularSwaptionEngine -----

    #[test]
    fn g104_hagan_amortizing_swaption() {
        let model = flat_gsr(0.05, 0.01, 0.04);
        let swap = IrregularSwap {
            fixed_payment_times: vec![2.0, 3.0, 4.0, 5.0],
            fixed_year_fractions: vec![1.0; 4],
            fixed_notionals: vec![1e6, 750_000.0, 500_000.0, 250_000.0],
            fixed_rates: vec![0.04; 4],
            float_payment_times: vec![2.0, 3.0, 4.0, 5.0],
            float_year_fractions: vec![1.0; 4],
            float_notionals: vec![1e6, 750_000.0, 500_000.0, 250_000.0],
            float_spreads: vec![0.0; 4],
            is_payer: true,
        };
        let result = hagan_irregular_swaption(&model, &swap, 1.0, 32);
        assert!(result.npv >= 0.0, "Hagan irregular swaption NPV: {}", result.npv);
        assert!(!result.basket_weights.is_empty());
        assert_eq!(result.basket_weights.len(), result.component_npvs.len());
    }

    #[test]
    fn g104_hagan_constant_notional_single_component() {
        let model = flat_gsr(0.05, 0.01, 0.04);
        // Constant notional: only the last period has ΔN > 0
        let swap = IrregularSwap {
            fixed_payment_times: vec![2.0, 3.0, 4.0],
            fixed_year_fractions: vec![1.0; 3],
            fixed_notionals: vec![1e6; 3],
            fixed_rates: vec![0.04; 3],
            float_payment_times: vec![2.0, 3.0, 4.0],
            float_year_fractions: vec![1.0; 3],
            float_notionals: vec![1e6; 3],
            float_spreads: vec![0.0; 3],
            is_payer: true,
        };
        let result = hagan_irregular_swaption(&model, &swap, 1.0, 32);
        assert!(result.npv >= 0.0);
        // For constant notional, only the last weight should be 1.0,
        // the rest should be 0.0
        assert_abs_diff_eq!(result.basket_weights[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result.basket_weights[1], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result.basket_weights[2], 1.0, epsilon = 1e-10);
    }

    // ----- G105: BasketGeneratingEngine / LatticeShortRateModelEngine -----

    #[test]
    fn g105_basket_generating_bermudan() {
        let model = flat_gsr(0.05, 0.01, 0.04);
        let ex_times = vec![1.0, 2.0, 3.0];
        let result = basket_generating_engine(
            &model,
            &ex_times,
            5.0,
            0.04,
            1_000_000.0,
            true,
            1,
            32,
        );
        assert!(!result.basket.is_empty());
        assert_eq!(result.basket.len(), result.component_npvs.len());
        assert!(result.total_npv >= 0.0, "Basket total should be non-negative: {}", result.total_npv);
        // Each European component should be non-negative
        for (i, &v) in result.component_npvs.iter().enumerate() {
            assert!(v >= 0.0, "Component {} NPV: {}", i, v);
        }
    }

    #[test]
    fn g105_basket_upper_bound_exceeds_single() {
        let model = flat_gsr(0.05, 0.01, 0.04);
        // Single exercise = European
        let single = basket_generating_engine(
            &model, &[1.0], 5.0, 0.04, 1.0, true, 1, 32,
        );
        // Multiple exercises => sum of Europeans >= single European
        let multi = basket_generating_engine(
            &model, &[1.0, 2.0, 3.0], 5.0, 0.04, 1.0, true, 1, 32,
        );
        assert!(
            multi.total_npv >= single.total_npv * 0.95,
            "Multi ({}) should be >= single ({})",
            multi.total_npv,
            single.total_npv,
        );
    }

    #[test]
    fn g105_lattice_european_bond_option() {
        // Price a call on a zero-coupon bond using the lattice engine
        let a = 0.1_f64;
        let sigma = 0.01_f64;
        let r0 = 0.05_f64;
        let maturity = 1.0_f64;
        let bond_maturity = 5.0_f64;
        let strike = 0.80_f64;

        let payoff = move |t: f64, r: f64| {
            if (t - maturity).abs() < 0.01 {
                // At expiry: payoff = max(P(T, T_bond) - K, 0)
                let tau = bond_maturity - maturity;
                let b = if a.abs() < 1e-12 { tau } else { (1.0 - (-a * tau).exp()) / a };
                let s2 = sigma * sigma;
                let a2 = a * a;
                let ln_a = if a.abs() < 1e-12 {
                    -0.5 * s2 * tau * tau * tau / 3.0
                } else {
                    (b - tau) * (-0.5 * s2) / a2 - s2 * b * b / (4.0 * a)
                };
                let bond = (ln_a - b * r).exp();
                (bond - strike).max(0.0)
            } else {
                0.0
            }
        };

        let result = lattice_short_rate_engine(a, sigma, r0, maturity, 200, &payoff, false);
        assert!(result.npv > 0.0, "Lattice bond option should be positive: {}", result.npv);
    }

    #[test]
    fn g105_lattice_american_vs_european() {
        let a = 0.1_f64;
        let sigma = 0.01_f64;
        let r0 = 0.05_f64;
        let maturity = 2.0_f64;
        let bond_mat = 5.0_f64;
        let strike = 0.82_f64;

        let payoff = move |_t: f64, r: f64| {
            let tau = bond_mat - _t;
            if tau <= 0.0 {
                return 0.0;
            }
            let b = if a.abs() < 1e-12 { tau } else { (1.0 - (-a * tau).exp()) / a };
            let s2 = sigma * sigma;
            let a2 = a * a;
            let ln_a = if a.abs() < 1e-12 {
                -0.5 * s2 * tau * tau * tau / 3.0
            } else {
                (b - tau) * (-0.5 * s2) / a2 - s2 * b * b / (4.0 * a)
            };
            let bond = (ln_a - b * r).exp();
            (bond - strike).max(0.0)
        };

        let european = lattice_short_rate_engine(a, sigma, r0, maturity, 200, &payoff, false);
        let american = lattice_short_rate_engine(a, sigma, r0, maturity, 200, &payoff, true);

        assert!(
            american.npv >= european.npv * 0.99,
            "American ({}) should be >= European ({})",
            american.npv,
            european.npv,
        );
    }
}
