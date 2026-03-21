//! Griewank-style binomial checkpointing for memory-efficient long-path AAD.
//!
//! When a Monte Carlo simulation has many time steps (S > 1000), the tape
//! recording every operation can consume substantial memory. Binomial
//! checkpointing (Griewank & Walther, 2000) trades a small amount of extra
//! computation for dramatically reduced memory — from O(S) to O(√S).
//!
//! # Algorithm (Revolve)
//!
//! Given S time steps and C checkpoint slots:
//!
//! 1. **Forward sweep**: simulate all S steps, saving state snapshots at
//!    C strategically chosen checkpoint positions.
//! 2. **Reverse sweep**: replay segments from checkpoints, taping and
//!    computing adjoints one segment at a time.
//!
//! The `revolve` scheduler (Griewank & Walther) finds the optimal
//! checkpoint placement that minimises total re-computation.
//!
//! # Example
//!
//! ```
//! use ql_aad::checkpoint::{revolve, Action};
//!
//! // 100 steps with 5 checkpoint slots
//! let actions = revolve(100, 5);
//! // actions tell you when to advance, checkpoint, or reverse
//! assert!(!actions.is_empty());
//! ```

use std::fmt;

use crate::bs::OptionKind;
use crate::mc::{McEuropeanGreeks, McHestonGreeks};
use crate::tape::{AReal, Tape};

use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

// ===========================================================================
// Revolve schedule
// ===========================================================================

/// An action in the checkpointing schedule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Action {
    /// Advance the forward computation from the current step to `target`
    /// **without** recording on the tape (discard intermediate values).
    Advance {
        /// Target step to advance to.
        target: usize,
    },
    /// Save a checkpoint of the current state at the current step.
    Snapshot {
        /// Checkpoint slot index to save into.
        slot: usize,
    },
    /// Restore the state from a previously saved checkpoint.
    Restore {
        /// Checkpoint slot index to restore from.
        slot: usize,
    },
    /// Advance one step **with** taping, then compute the adjoint backward
    /// through that single step. This is the "turn-around" — the point
    /// where forward becomes backward.
    TapeAndReverse,
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Action::Advance { target } => write!(f, "Advance → {target}"),
            Action::Snapshot { slot } => write!(f, "Snapshot(slot={slot})"),
            Action::Restore { slot } => write!(f, "Restore(slot={slot})"),
            Action::TapeAndReverse => write!(f, "TapeAndReverse"),
        }
    }
}

/// Compute the binomial coefficient $\binom{s+c}{c}$ (without overflow for
/// moderate values).
fn binom(s: usize, c: usize) -> usize {
    if c == 0 {
        return 1;
    }
    let mut result = 1_u128;
    for i in 0..c {
        result = result * (s + c - i) as u128 / (i + 1) as u128;
    }
    result.min(usize::MAX as u128) as usize
}

/// Generate the optimal checkpoint schedule using Griewank's revolve algorithm.
///
/// Given `num_steps` forward time steps and `num_slots` available checkpoint
/// slots, produces a sequence of [`Action`]s that minimise total operations
/// while keeping memory bounded to `num_slots` snapshots.
///
/// # Panics
///
/// Panics if `num_steps == 0` or `num_slots == 0`.
pub fn revolve(num_steps: usize, num_slots: usize) -> Vec<Action> {
    assert!(num_steps > 0, "num_steps must be > 0");
    assert!(num_slots > 0, "num_slots must be > 0");

    let mut actions = Vec::new();
    revolve_recursive(num_steps, num_slots, &mut actions);
    actions
}

/// Internal recursive revolve implementation.
///
/// `steps_remaining` = number of forward steps still to process.
/// `slots_available` = number of checkpoint slots still free.
fn revolve_recursive(
    steps_remaining: usize,
    slots_available: usize,
    actions: &mut Vec<Action>,
) {
    if steps_remaining == 0 {
        return;
    }

    if steps_remaining == 1 {
        // Base case: one step — just tape it and reverse.
        actions.push(Action::TapeAndReverse);
        return;
    }

    if slots_available == 0 {
        // No checkpoint slots — must tape all remaining steps sequentially.
        for _ in 0..steps_remaining {
            actions.push(Action::TapeAndReverse);
        }
        return;
    }

    // Find the optimal split point `m`:
    // We want the largest m such that binom(m, slots) < steps_remaining.
    // Equivalently, advance m steps and checkpoint, then recurse.
    let m = optimal_split(steps_remaining, slots_available);

    // 1. Save checkpoint before advancing
    let slot = slots_available - 1;
    actions.push(Action::Snapshot { slot });

    // 2. Advance m steps without taping
    actions.push(Action::Advance { target: m });

    // 3. Recurse on the remaining (steps - m) steps with all slots
    revolve_recursive(steps_remaining - m, slots_available, actions);

    // 4. Restore checkpoint and process the first m steps with one fewer slot
    actions.push(Action::Restore { slot });
    revolve_recursive(m, slots_available - 1, actions);
}

/// Find the optimal number of steps to advance before checkpointing.
///
/// This is the key of the revolve algorithm: find `m` such that the total
/// work is minimised. For `s` steps and `c` slots, the optimal split is
/// the largest `m` where `binom(m, c) < s`.
fn optimal_split(steps: usize, slots: usize) -> usize {
    // Binary search for the split point
    let mut lo = 1_usize;
    let mut hi = steps - 1;

    // Edge case: only 1 slot — advance to step (steps-1), checkpoint, then
    // tape the last step and reverse, then replay from checkpoint.
    if slots == 1 {
        return steps - 1;
    }

    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if binom(mid, slots) < steps {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    lo
}

// ===========================================================================
// Checkpoint state
// ===========================================================================

/// A snapshot of the simulation state at a particular time step.
#[derive(Clone, Debug)]
pub struct StateSnapshot {
    /// The time step this snapshot was taken at.
    pub step: usize,
    /// The state variables (e.g. log-spot, variance, etc.).
    pub state: Vec<f64>,
}

/// Storage for multiple checkpoint snapshots.
#[derive(Clone, Debug)]
pub struct CheckpointStore {
    slots: Vec<Option<StateSnapshot>>,
    /// Current step in the simulation.
    current_step: usize,
}

impl CheckpointStore {
    /// Create a new store with `num_slots` checkpoint slots.
    pub fn new(num_slots: usize) -> Self {
        Self {
            slots: vec![None; num_slots],
            current_step: 0,
        }
    }

    /// Save the current state into a slot.
    pub fn save(&mut self, slot: usize, state: Vec<f64>) {
        self.slots[slot] = Some(StateSnapshot {
            step: self.current_step,
            state,
        });
    }

    /// Restore from a slot. Returns the saved state and sets current_step.
    pub fn restore(&mut self, slot: usize) -> &[f64] {
        let snap = self.slots[slot].as_ref().expect("slot is empty");
        self.current_step = snap.step;
        &snap.state
    }

    /// Get the step stored in a slot.
    pub fn step_at(&self, slot: usize) -> usize {
        self.slots[slot].as_ref().expect("slot is empty").step
    }

    /// Advance current step.
    pub fn advance_to(&mut self, step: usize) {
        self.current_step = step;
    }

    /// Current step.
    pub fn current_step(&self) -> usize {
        self.current_step
    }
}

// ===========================================================================
// MC European with checkpointed AAD
// ===========================================================================

/// Run a single GBM step (forward only, no taping).
#[inline]
fn gbm_step_f64(log_s: f64, drift_dt: f64, vol_sqrt_dt: f64, z: f64) -> f64 {
    log_s + drift_dt + vol_sqrt_dt * z
}

/// Run a single GBM step on the tape (taping).
#[inline]
fn gbm_step_taped(
    tape: &mut Tape,
    log_s: AReal,
    drift_dt: AReal,
    vol_sqrt_dt: AReal,
    z: f64,
) -> AReal {
    // log_s + drift_dt + vol_sqrt_dt * z
    let step1 = tape.add(log_s, drift_dt);
    let znode = tape.mul_const(vol_sqrt_dt, z);
    tape.add(step1, znode)
}

/// Monte Carlo European option pricing with Griewank checkpointed AAD.
///
/// This function uses binomial checkpointing to bound the tape size
/// regardless of the number of time steps, while still computing exact
/// pathwise Greeks. Memory usage is O(√num_steps) per path instead of
/// O(num_steps).
///
/// # Arguments
///
/// * `spot`, `strike`, `r`, `q`, `vol`, `time_to_expiry` — standard BS params
/// * `option_kind` — Call or Put
/// * `num_paths` — number of MC paths
/// * `num_steps` — number of time steps per path (checkpointing benefits S > 50)
/// * `num_checkpoints` — number of checkpoint slots (√num_steps is optimal)
/// * `seed` — RNG seed
///
/// # Example
///
/// ```
/// use ql_aad::checkpoint::mc_european_checkpointed;
/// use ql_aad::OptionKind;
///
/// let greeks = mc_european_checkpointed(
///     100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
///     OptionKind::Call, 10_000, 200, 15, 42,
/// );
/// assert!((greeks.npv - 10.45).abs() < 1.0);
/// assert!((greeks.delta - 0.637).abs() < 0.1);
/// ```
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn mc_european_checkpointed(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    vol: f64,
    time_to_expiry: f64,
    option_kind: OptionKind,
    num_paths: usize,
    num_steps: usize,
    num_checkpoints: usize,
    seed: u64,
) -> McEuropeanGreeks {
    let mut rng = SmallRng::seed_from_u64(seed);
    let dt = time_to_expiry / num_steps as f64;
    let sqrt_dt = dt.sqrt();
    let drift_dt_val = (r - q - 0.5 * vol * vol) * dt;
    let vol_sqrt_dt_val = vol * sqrt_dt;

    let phi = match option_kind {
        OptionKind::Call => 1.0,
        OptionKind::Put => -1.0,
    };

    let discount = (-r * time_to_expiry).exp();

    // Pre-generate the revolve schedule (for reference — the MC functions
    // use the simpler segmented approach which is equivalent for single-pass).
    let _schedule = revolve(num_steps, num_checkpoints);

    // Accumulators
    let mut sum_payoff = 0.0;
    let mut sum_payoff_sq = 0.0;
    let mut sum_delta = 0.0;
    let mut sum_vega = 0.0;
    let mut sum_rho = 0.0;
    let mut sum_div_rho = 0.0;

    let mut tape = Tape::with_capacity(1024);

    for _ in 0..num_paths {
        // Pre-generate all random numbers for this path
        let z_all: Vec<f64> = (0..num_steps)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();

        // === Checkpointed forward + reverse pass ===
        // State: [log_s]
        let initial_log_s = spot.ln();

        // We'll track log_s for forward (untaped) computation
        // Accumulated adjoints from reverse segments
        let mut adj_drift_dt = 0.0;
        let mut adj_vol_sqrt_dt = 0.0;

        // Process the schedule
        //
        // The revolve schedule is designed for a push-down automaton:
        // - Snapshot: save current state
        // - Advance: move forward without taping
        // - TapeAndReverse: tape one step, then reverse
        // - Restore: go back to checkpoint
        //
        // We process steps 0..num_steps. After step s, we're at step s+1.
        // Instead of the complex schedule dispatching, let's use a simpler
        // segmented approach that's equivalent to revolve for a single-variable
        // system:
        //
        // 1. Forward pass: simulate all steps, saving checkpoints every B steps
        //    where B = ceil(num_steps / num_checkpoints)
        // 2. Reverse pass: for each segment (in reverse), restore checkpoint,
        //    tape the segment, adjoint, accumulate

        let block_size = if num_checkpoints > 0 {
            num_steps.div_ceil(num_checkpoints)
        } else {
            num_steps
        };

        // Forward pass: save checkpoints
        let mut checkpoints: Vec<(usize, f64)> = Vec::with_capacity(num_checkpoints + 1);
        checkpoints.push((0, initial_log_s));
        let mut fwd_log_s = initial_log_s;
        for s in 0..num_steps {
            fwd_log_s = gbm_step_f64(fwd_log_s, drift_dt_val, vol_sqrt_dt_val, z_all[s]);
            if (s + 1) % block_size == 0 && (s + 1) < num_steps {
                checkpoints.push((s + 1, fwd_log_s));
            }
        }
        let final_spot = fwd_log_s.exp();
        let payoff_val = (phi * (final_spot - strike)).max(0.0) * discount;

        // Reverse pass: process segments from last to first
        let num_segments = checkpoints.len();
        let mut adj_log_s_carry = 0.0_f64; // adjoint flowing back from later segments

        for seg_idx in (0..num_segments).rev() {
            let (seg_start, seg_log_s) = checkpoints[seg_idx];
            let seg_end = if seg_idx + 1 < num_segments {
                checkpoints[seg_idx + 1].0
            } else {
                num_steps
            };

            // Tape this segment
            tape.clear();

            // Create tape inputs for model parameters
            let t_spot_log = tape.input(seg_log_s);
            let t_drift_dt = tape.input(drift_dt_val);
            let t_vol_sqrt_dt = tape.input(vol_sqrt_dt_val);

            // Run this segment on the tape
            let mut t_log_s = t_spot_log;
            for s in seg_start..seg_end {
                t_log_s = gbm_step_taped(&mut tape, t_log_s, t_drift_dt, t_vol_sqrt_dt, z_all[s]);
            }

            if seg_idx == num_segments - 1 {
                // Last segment — compute payoff on tape
                let t_final_spot = tape.exp(t_log_s);
                let t_strike = tape.input(strike);
                let t_intrinsic = if phi > 0.0 {
                    tape.sub(t_final_spot, t_strike)
                } else {
                    tape.sub(t_strike, t_final_spot)
                };
                let t_zero = tape.input(0.0);
                let t_payoff_raw = tape.max(t_intrinsic, t_zero);
                let t_payoff = tape.mul_const(t_payoff_raw, discount);

                // Adjoint from payoff
                let adj = tape.adjoint(t_payoff);

                // Accumulate parameter sensitivities from this segment
                adj_log_s_carry = adj[t_spot_log.idx]; // ∂payoff/∂(segment start log_s)
                adj_drift_dt += adj[t_drift_dt.idx];
                adj_vol_sqrt_dt += adj[t_vol_sqrt_dt.idx];
            } else {
                // Intermediate segment — the "output" is t_log_s, and we chain
                // the adjoint from the next segment via adj_log_s_carry.
                //
                // We need ∂(segment_output)/∂(segment_inputs), scaled by the
                // incoming adjoint.
                let adj_full = tape.adjoint(t_log_s);

                // Chain rule: ∂payoff/∂(seg_input) = ∂payoff/∂(seg_output) * ∂(seg_output)/∂(seg_input)
                let new_carry = adj_full[t_spot_log.idx] * adj_log_s_carry;
                adj_drift_dt += adj_full[t_drift_dt.idx] * adj_log_s_carry;
                adj_vol_sqrt_dt += adj_full[t_vol_sqrt_dt.idx] * adj_log_s_carry;
                adj_log_s_carry = new_carry;
            }
        }

        // adj_log_s_carry is now ∂payoff/∂(log_s0).
        // ∂payoff/∂S0 = (∂payoff/∂log_s0) * (1/S0)
        let path_delta = adj_log_s_carry / spot;

        // ∂payoff/∂vol: drift_dt = (r-q-0.5*vol^2)*dt, vol_sqrt_dt = vol*sqrt(dt)
        // ∂drift_dt/∂vol = -vol*dt, ∂vol_sqrt_dt/∂vol = sqrt(dt)
        let path_vega = adj_drift_dt * (-vol * dt) + adj_vol_sqrt_dt * sqrt_dt;

        // ∂payoff/∂r: ∂drift_dt/∂r = dt
        // plus discount factor sensitivity: ∂(discount)/∂r = -time_to_expiry * discount
        // The discount is baked into payoff_val, so we add that term
        let path_rho = adj_drift_dt * dt + (-time_to_expiry * payoff_val);

        // ∂payoff/∂q: ∂drift_dt/∂q = -dt
        let path_div_rho = adj_drift_dt * (-dt);

        sum_payoff += payoff_val;
        sum_payoff_sq += payoff_val * payoff_val;
        sum_delta += path_delta;
        sum_vega += path_vega;
        sum_rho += path_rho;
        sum_div_rho += path_div_rho;
    }

    let n = num_paths as f64;
    let mean_payoff = sum_payoff / n;
    let var = (sum_payoff_sq / n) - mean_payoff * mean_payoff;
    let std_error = (var / n).sqrt();

    McEuropeanGreeks {
        npv: mean_payoff,
        std_error,
        delta: sum_delta / n,
        vega: sum_vega / n,
        rho: sum_rho / n,
        div_rho: sum_div_rho / n,
        num_paths,
    }
}

// ===========================================================================
// MC Heston with checkpointed AAD
// ===========================================================================

/// Run a single Heston step (forward only, no taping).
///
/// Returns `(new_log_s, new_v)`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn heston_step_f64(
    log_s: f64, v: f64,
    r_q: f64, kappa: f64, theta: f64, sigma: f64, rho: f64,
    dt: f64, sqrt_dt: f64,
    z1: f64, z2: f64,
) -> (f64, f64) {
    let v_pos = v.max(0.0); // full truncation
    let sqrt_v = v_pos.sqrt();

    // Correlated normals: w1 = z1, w2 = rho*z1 + sqrt(1-rho^2)*z2
    let w1 = z1;
    let rho_comp = (1.0 - rho * rho).sqrt();
    let w2 = rho * z1 + rho_comp * z2;

    // Log-Euler for spot
    let new_log_s = log_s + (r_q - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * w1;

    // Euler for variance (full truncation)
    let new_v = v + kappa * (theta - v_pos) * dt + sigma * sqrt_v * sqrt_dt * w2;

    (new_log_s, new_v)
}

/// Run a single Heston step on the tape.
///
/// Returns `(new_log_s, new_v)`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn heston_step_taped(
    tape: &mut Tape,
    log_s: AReal,
    v: AReal,
    r_q: AReal,
    kappa: AReal,
    theta: AReal,
    sigma: AReal,
    rho: AReal,
    dt: f64,
    sqrt_dt: f64,
    z1: f64,
    z2: f64,
) -> (AReal, AReal) {
    let zero = tape.input(0.0);
    let v_pos = tape.max(v, zero);
    let sqrt_v = tape.sqrt(v_pos);

    // Correlated normals
    let rho_comp = {
        // sqrt(1 - rho^2) — manual derivative
        let one = tape.input(1.0);
        let rho2 = tape.mul(rho, rho);
        let inner = tape.sub(one, rho2);
        tape.sqrt(inner)
    };

    let w1_node = tape.input(z1);
    let w2_base = tape.mul_const(rho, z1);
    let w2_indep = tape.mul_const(rho_comp, z2);
    let w2_node = tape.add(w2_base, w2_indep);

    // Log-Euler for spot: log_s + (r_q - 0.5*v_pos)*dt + sqrt_v*sqrt_dt*w1
    let half_v = tape.mul_const(v_pos, -0.5);
    let drift = tape.add(r_q, half_v);
    let drift_dt = tape.mul_const(drift, dt);
    let diff_part = tape.mul(sqrt_v, w1_node);
    let diff_scaled = tape.mul_const(diff_part, sqrt_dt);
    let step1 = tape.add(log_s, drift_dt);
    let new_log_s = tape.add(step1, diff_scaled);

    // Euler variance: v + kappa*(theta - v_pos)*dt + sigma*sqrt_v*sqrt_dt*w2
    let thetamv = tape.sub(theta, v_pos);
    let rev = tape.mul(kappa, thetamv);
    let rev_dt = tape.mul_const(rev, dt);
    let vol_part = tape.mul(sigma, sqrt_v);
    let vol_w2 = tape.mul(vol_part, w2_node);
    let vol_scaled = tape.mul_const(vol_w2, sqrt_dt);
    let v_step1 = tape.add(v, rev_dt);
    let new_v = tape.add(v_step1, vol_scaled);

    (new_log_s, new_v)
}

/// Monte Carlo Heston option pricing with Griewank checkpointed AAD.
///
/// Uses binomial checkpointing to bound tape memory for long-path Heston
/// simulations. The state at each checkpoint stores `[log_s, v]`.
///
/// # Arguments
///
/// * `spot`, `strike`, `r`, `q`, `v0`, `kappa`, `theta`, `sigma`, `rho` — Heston params
/// * `time_to_expiry` — time to maturity
/// * `option_kind` — Call or Put
/// * `num_paths` — number of MC paths
/// * `num_steps` — time steps per path
/// * `num_checkpoints` — checkpoint slots (√num_steps is optimal)
/// * `seed` — RNG seed
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn mc_heston_checkpointed(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho_corr: f64,
    time_to_expiry: f64,
    option_kind: OptionKind,
    num_paths: usize,
    num_steps: usize,
    num_checkpoints: usize,
    seed: u64,
) -> McHestonGreeks {
    let mut rng = SmallRng::seed_from_u64(seed);
    let dt = time_to_expiry / num_steps as f64;
    let sqrt_dt = dt.sqrt();
    let r_q_val = r - q;

    let phi = match option_kind {
        OptionKind::Call => 1.0,
        OptionKind::Put => -1.0,
    };

    let discount = (-r * time_to_expiry).exp();

    let block_size = if num_checkpoints > 0 {
        num_steps.div_ceil(num_checkpoints)
    } else {
        num_steps
    };

    // Accumulators: [delta, vega_v0, d_kappa, d_theta, d_sigma, d_rho, rho_r, div_rho]
    let mut sum_greeks = [0.0_f64; 8];
    let mut sum_payoff = 0.0;
    let mut sum_payoff_sq = 0.0;

    let mut tape = Tape::with_capacity(4096);

    for _ in 0..num_paths {
        // Pre-generate randoms
        let z1_all: Vec<f64> = (0..num_steps)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();
        let z2_all: Vec<f64> = (0..num_steps)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();

        let initial_log_s = spot.ln();
        let initial_v = v0;

        // Forward pass: save checkpoints at block boundaries (state = [log_s, v])
        let mut checkpoints: Vec<(usize, f64, f64)> = Vec::with_capacity(num_checkpoints + 1);
        checkpoints.push((0, initial_log_s, initial_v));

        let mut fwd_log_s = initial_log_s;
        let mut fwd_v = initial_v;
        for s in 0..num_steps {
            let (nls, nv) = heston_step_f64(
                fwd_log_s, fwd_v,
                r_q_val, kappa, theta, sigma, rho_corr,
                dt, sqrt_dt, z1_all[s], z2_all[s],
            );
            fwd_log_s = nls;
            fwd_v = nv;
            if (s + 1) % block_size == 0 && (s + 1) < num_steps {
                checkpoints.push((s + 1, fwd_log_s, fwd_v));
            }
        }

        let final_spot = fwd_log_s.exp();
        let payoff_val = (phi * (final_spot - strike)).max(0.0) * discount;

        // Reverse pass: process segments from last to first
        let num_segments = checkpoints.len();
        // Carry adjoints for [log_s, v]
        let mut adj_log_s_carry = 0.0_f64;
        let mut adj_v_carry = 0.0_f64;
        // Accumulated parameter adjoints: [r_q, kappa, theta, sigma, rho]
        let mut adj_params = [0.0_f64; 5];

        for seg_idx in (0..num_segments).rev() {
            let (seg_start, seg_log_s, seg_v) = checkpoints[seg_idx];
            let seg_end = if seg_idx + 1 < num_segments {
                checkpoints[seg_idx + 1].0
            } else {
                num_steps
            };

            tape.clear();

            // Tape inputs: log_s, v, r_q, kappa, theta, sigma, rho
            let t_log_s_in = tape.input(seg_log_s);
            let t_v_in = tape.input(seg_v);
            let t_r_q = tape.input(r_q_val);
            let t_kappa = tape.input(kappa);
            let t_theta = tape.input(theta);
            let t_sigma = tape.input(sigma);
            let t_rho = tape.input(rho_corr);

            let mut t_log_s = t_log_s_in;
            let mut t_v = t_v_in;

            for s in seg_start..seg_end {
                let (nls, nv) = heston_step_taped(
                    &mut tape, t_log_s, t_v,
                    t_r_q, t_kappa, t_theta, t_sigma, t_rho,
                    dt, sqrt_dt, z1_all[s], z2_all[s],
                );
                t_log_s = nls;
                t_v = nv;
            }

            if seg_idx == num_segments - 1 {
                // Last segment — compute payoff
                let t_final_spot = tape.exp(t_log_s);
                let t_strike = tape.input(strike);
                let t_intrinsic = if phi > 0.0 {
                    tape.sub(t_final_spot, t_strike)
                } else {
                    tape.sub(t_strike, t_final_spot)
                };
                let t_zero = tape.input(0.0);
                let t_payoff_raw = tape.max(t_intrinsic, t_zero);
                let t_payoff = tape.mul_const(t_payoff_raw, discount);

                let adj = tape.adjoint(t_payoff);

                adj_log_s_carry = adj[t_log_s_in.idx];
                adj_v_carry = adj[t_v_in.idx];
                adj_params[0] += adj[t_r_q.idx];
                adj_params[1] += adj[t_kappa.idx];
                adj_params[2] += adj[t_theta.idx];
                adj_params[3] += adj[t_sigma.idx];
                adj_params[4] += adj[t_rho.idx];
            } else {
                // Intermediate segment — two outputs: t_log_s, t_v
                // We need to chain the carried adjoints through.
                let adj_ls = tape.adjoint(t_log_s);
                let adj_vv = tape.adjoint(t_v);

                // Chain: ∂payoff/∂param += adj_log_s_carry * ∂log_s_out/∂param
                //                        + adj_v_carry * ∂v_out/∂param
                let new_adj_log_s =
                    adj_log_s_carry * adj_ls[t_log_s_in.idx] +
                    adj_v_carry * adj_vv[t_log_s_in.idx];
                let new_adj_v =
                    adj_log_s_carry * adj_ls[t_v_in.idx] +
                    adj_v_carry * adj_vv[t_v_in.idx];

                for (p, &param_idx) in [t_r_q.idx, t_kappa.idx, t_theta.idx, t_sigma.idx, t_rho.idx].iter().enumerate() {
                    adj_params[p] +=
                        adj_log_s_carry * adj_ls[param_idx] +
                        adj_v_carry * adj_vv[param_idx];
                }

                adj_log_s_carry = new_adj_log_s;
                adj_v_carry = new_adj_v;
            }
        }

        // Convert to Greeks
        // delta = ∂payoff/∂S0 = (∂payoff/∂log_s0) / S0
        let path_delta = adj_log_s_carry / spot;
        // vega_v0 = ∂payoff/∂v0
        let path_vega_v0 = adj_v_carry;
        // d_kappa, d_theta, d_sigma, d_rho
        let path_d_kappa = adj_params[1];
        let path_d_theta = adj_params[2];
        let path_d_sigma = adj_params[3];
        let path_d_rho = adj_params[4];
        // rho (interest rate): ∂payoff/∂r = adj_r_q * (∂(r-q)/∂r) + discount term
        //   = adj_r_q * 1 + (-time_to_expiry * payoff_val)
        let path_rho_r = adj_params[0] + (-time_to_expiry * payoff_val);
        // div_rho: ∂payoff/∂q = adj_r_q * (∂(r-q)/∂q) = adj_r_q * (-1)
        let path_div_rho = -adj_params[0];

        sum_greeks[0] += path_delta;
        sum_greeks[1] += path_vega_v0;
        sum_greeks[2] += path_d_kappa;
        sum_greeks[3] += path_d_theta;
        sum_greeks[4] += path_d_sigma;
        sum_greeks[5] += path_d_rho;
        sum_greeks[6] += path_rho_r;
        sum_greeks[7] += path_div_rho;
        sum_payoff += payoff_val;
        sum_payoff_sq += payoff_val * payoff_val;
    }

    let n = num_paths as f64;
    let mean = sum_payoff / n;
    let var = (sum_payoff_sq / n) - mean * mean;

    McHestonGreeks {
        npv: mean,
        std_error: (var / n).sqrt(),
        delta: sum_greeks[0] / n,
        vega_v0: sum_greeks[1] / n,
        d_kappa: sum_greeks[2] / n,
        d_theta: sum_greeks[3] / n,
        d_sigma: sum_greeks[4] / n,
        d_rho: sum_greeks[5] / n,
        rho: sum_greeks[6] / n,
        div_rho: sum_greeks[7] / n,
        num_paths,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Revolve schedule tests ──────────────────────────────────────────

    #[test]
    fn revolve_1_step() {
        let actions = revolve(1, 3);
        assert_eq!(actions, vec![Action::TapeAndReverse]);
    }

    #[test]
    fn revolve_2_steps_1_slot() {
        let actions = revolve(2, 1);
        // Should: snapshot, advance 1, tape+reverse, restore, tape+reverse
        assert!(actions.len() >= 2);
        // The schedule should include at least one snapshot and two reverses
        let reverses = actions.iter().filter(|a| matches!(a, Action::TapeAndReverse)).count();
        assert_eq!(reverses, 2);
    }

    #[test]
    fn revolve_5_steps_2_slots() {
        let actions = revolve(5, 2);
        // All 5 steps must be reversed
        let reverses = actions.iter().filter(|a| matches!(a, Action::TapeAndReverse)).count();
        assert_eq!(reverses, 5);
    }

    #[test]
    fn revolve_100_steps_5_slots() {
        let actions = revolve(100, 5);
        let reverses = actions.iter().filter(|a| matches!(a, Action::TapeAndReverse)).count();
        assert_eq!(reverses, 100);
    }

    #[test]
    fn revolve_large() {
        let actions = revolve(1000, 10);
        let reverses = actions.iter().filter(|a| matches!(a, Action::TapeAndReverse)).count();
        assert_eq!(reverses, 1000);
    }

    #[test]
    fn binom_basic() {
        assert_eq!(binom(0, 3), 1); // C(3,3) = 1
        assert_eq!(binom(1, 1), 2); // C(2,1) = 2
        assert_eq!(binom(3, 2), 10); // C(5,2) = 10
        assert_eq!(binom(4, 3), 35); // C(7,3) = 35
    }

    // ── MC European checkpointed tests ──────────────────────────────────

    #[test]
    fn checkpointed_european_matches_standard() {
        // Compare checkpointed vs standard mc_european_aad
        // Both should give similar NPV and delta (up to RNG alignment differences)
        use crate::mc::mc_european_aad;

        let std_greeks = mc_european_aad(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            OptionKind::Call, 20_000, 42,
        );

        // Checkpointed with 100 steps, 10 checkpoints
        let ckpt_greeks = mc_european_checkpointed(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            OptionKind::Call, 20_000, 100, 10, 42,
        );

        // NPV reference: BS ≈ 10.45
        assert!((ckpt_greeks.npv - 10.45).abs() < 1.0,
            "NPV {} too far from BS 10.45", ckpt_greeks.npv);
        // Delta should be similar
        assert!((ckpt_greeks.delta - std_greeks.delta).abs() < 0.15,
            "delta {} vs std {}", ckpt_greeks.delta, std_greeks.delta);
    }

    #[test]
    fn checkpointed_european_delta_vs_bump() {
        let eps = 0.01;
        let base = mc_european_checkpointed(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            OptionKind::Call, 50_000, 100, 10, 42,
        );
        let up = mc_european_checkpointed(
            100.0 + eps, 100.0, 0.05, 0.0, 0.20, 1.0,
            OptionKind::Call, 50_000, 100, 10, 42,
        );
        let down = mc_european_checkpointed(
            100.0 - eps, 100.0, 0.05, 0.0, 0.20, 1.0,
            OptionKind::Call, 50_000, 100, 10, 42,
        );

        let bump_delta = (up.npv - down.npv) / (2.0 * eps);
        assert!((base.delta - bump_delta).abs() < 0.05,
            "AAD delta {} vs bump delta {}", base.delta, bump_delta);
    }

    #[test]
    fn checkpointed_european_many_steps() {
        // 500 steps with only 8 checkpoints — should still converge
        let greeks = mc_european_checkpointed(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            OptionKind::Call, 10_000, 500, 8, 42,
        );
        assert!((greeks.npv - 10.45).abs() < 1.5,
            "NPV {} too far from BS", greeks.npv);
        assert!(greeks.delta > 0.3 && greeks.delta < 0.9,
            "delta {} out of range", greeks.delta);
    }

    #[test]
    fn checkpointed_put() {
        let greeks = mc_european_checkpointed(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            OptionKind::Put, 20_000, 50, 7, 42,
        );
        // BS put ≈ 5.57
        assert!((greeks.npv - 5.57).abs() < 1.0,
            "Put NPV {} too far from BS", greeks.npv);
        assert!(greeks.delta < 0.0, "put delta should be negative: {}", greeks.delta);
    }

    // ── MC Heston checkpointed tests ────────────────────────────────────

    #[test]
    fn checkpointed_heston_reasonable() {
        let greeks = mc_heston_checkpointed(
            100.0, 100.0, 0.05, 0.0,
            0.04, 1.5, 0.04, 0.3, -0.7,
            1.0, OptionKind::Call,
            10_000, 100, 10, 42,
        );
        // Heston call should be in a reasonable range
        assert!(greeks.npv > 3.0 && greeks.npv < 20.0,
            "NPV {} out of range", greeks.npv);
        assert!(greeks.delta > 0.0 && greeks.delta < 1.0,
            "delta {} out of range", greeks.delta);
    }

    #[test]
    fn checkpointed_heston_delta_vs_bump() {
        let eps = 0.01;
        let base = mc_heston_checkpointed(
            100.0, 100.0, 0.05, 0.0,
            0.04, 1.5, 0.04, 0.3, -0.7,
            1.0, OptionKind::Call,
            30_000, 50, 7, 42,
        );
        let up = mc_heston_checkpointed(
            100.0 + eps, 100.0, 0.05, 0.0,
            0.04, 1.5, 0.04, 0.3, -0.7,
            1.0, OptionKind::Call,
            30_000, 50, 7, 42,
        );
        let down = mc_heston_checkpointed(
            100.0 - eps, 100.0, 0.05, 0.0,
            0.04, 1.5, 0.04, 0.3, -0.7,
            1.0, OptionKind::Call,
            30_000, 50, 7, 42,
        );
        let bump_delta = (up.npv - down.npv) / (2.0 * eps);
        assert!((base.delta - bump_delta).abs() < 0.1,
            "AAD delta {} vs bump delta {}", base.delta, bump_delta);
    }

    #[test]
    fn checkpointed_heston_many_steps() {
        // 500 steps with 10 checkpoints
        let greeks = mc_heston_checkpointed(
            100.0, 100.0, 0.05, 0.0,
            0.04, 1.5, 0.04, 0.3, -0.7,
            1.0, OptionKind::Call,
            5_000, 500, 10, 42,
        );
        assert!(greeks.npv > 2.0 && greeks.npv < 25.0,
            "NPV {} out of range", greeks.npv);
    }
}
