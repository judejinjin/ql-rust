//! FDM time-stepping schemes and step conditions.
//!
//! Organises the ADI schemes (Douglas, Hundsdorfer-Verwer, Craig-Sneyd) into a
//! single ergonomic API and provides reusable **step conditions** that can be
//! applied after each time step, such as:
//! - [`BarrierCondition`]: zero out values beyond the barrier
//! - [`AmericanExerciseCondition`]: early-exercise for American options
//! - [`AveragingCondition`]: running sum for Asian options
//! - [`RunningExtremeCondition`]: running max/min for lookbacks
//!
//! ## References
//!
//! - Iacus, S.M. (2011). *Option Pricing and Estimation of Financial Models
//!   with R*. Wiley.
//! - Hout, K. & Welfert, B. (2007). "Stability of ADI schemes applied to
//!   convection-diffusion equations with mixed derivative terms." *App. Num. Math.*

use serde::{Deserialize, Serialize};

use crate::fdm_operators::{TripleBandOp, crank_nicolson_step, hundsdorfer_verwer_step,
    modified_craig_sneyd_step, Heston2dOps, douglas_adi_step, apply_american_condition,
    AdiScheme, fd_heston_solve_adi, HestonFdResult};
use crate::fdm_meshers::Mesher1d;

// =========================================================================
// FDM scheme selector
// =========================================================================

/// Available 1D time-stepping schemes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FdmScheme1d {
    /// Crank-Nicolson (θ = ½): second-order in time and space. Good general choice.
    CrankNicolson,
    /// Fully implicit: first-order in time, unconditionally stable.
    FullyImplicit,
}

/// Available 2D / multi-dimensional ADI time-stepping schemes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FdmScheme2d {
    /// Douglas (1962): first-order in mixed derivatives, simple.
    Douglas,
    /// Hundsdorfer-Verwer (2003): second-order, handles cross-derivative well.
    HundsdorferVerwer,
    /// Modified Craig-Sneyd (2008): improved L-stable variant.
    ModifiedCraigSneyd,
}

// =========================================================================
// Step conditions
// =========================================================================

/// Direction for barrier or extremum conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BarrierDirection {
    /// Condition applies below/at the lower boundary.
    Down,
    /// Condition applies above/at the upper boundary.
    Up,
}

/// Knock-in / knock-out barrier step condition for 1D grids.
///
/// After each time step, zeros out the option value at grid points that
/// have crossed the barrier (knock-out) or sets them to the payoff value
/// for knock-in contracts.
#[derive(Debug, Clone)]
pub struct BarrierCondition {
    /// Barrier level in spot space.
    pub barrier_level: f64,
    /// Down or up barrier.
    pub direction: BarrierDirection,
    /// If true this is a knock-out: zero beyond barrier. Otherwise knock-in (no-op on grid).
    pub is_out: bool,
}

impl BarrierCondition {
    /// Apply to a grid of values (in log-spot space).
    ///
    /// `locs` — grid locations (log-spot or spot, depending on mesher).
    /// `values` — option values to modify in-place.
    /// `use_log_scale` — true if `locs` are log-spot (ln S), false if spot.
    pub fn apply(&self, locs: &[f64], values: &mut [f64], use_log_scale: bool) {
        if !self.is_out { return; } // knock-in: no action on underlying grid
        let barrier_log = if use_log_scale { self.barrier_level.ln() } else { self.barrier_level };
        for (i, &loc) in locs.iter().enumerate() {
            let crossed = match self.direction {
                BarrierDirection::Down => loc <= barrier_log,
                BarrierDirection::Up  => loc >= barrier_log,
            };
            if crossed {
                values[i] = 0.0;
            }
        }
    }
}

/// Early-exercise (American) step condition.
///
/// After each time step, takes the maximum of the current value and the
/// intrinsic value at each grid point.
pub struct AmericanExerciseCondition {
    /// Intrinsic (exercise) value at each grid point.
    pub intrinsic: Vec<f64>,
}

impl AmericanExerciseCondition {
    /// Apply the early-exercise condition in-place.
    pub fn apply(&self, values: &mut [f64]) {
        apply_american_condition(values, &self.intrinsic);
    }
}

/// Running-sum step condition for arithmetic Asian options.
///
/// Maintains a running sum of spot prices (or discounted values) across
/// time steps to compute the average. Holds one running-sum grid per
/// spatial grid point.
#[derive(Debug, Clone)]
pub struct AveragingCondition {
    /// Number of averaging dates remaining.
    pub remaining_dates: usize,
    /// Dates visited so far.
    pub dates_visited: usize,
    /// Running sum of averages (same length as spatial grid).
    pub running_sum: Vec<f64>,
}

impl AveragingCondition {
    /// Create a new averaging condition with zero running sums.
    pub fn new(n_grid: usize, total_dates: usize) -> Self {
        Self {
            remaining_dates: total_dates,
            dates_visited: 0,
            running_sum: vec![0.0; n_grid],
        }
    }

    /// Record one averaging date: add the current spot at each grid point.
    ///
    /// `spot_grid` — current spot price at each grid point (not discounted).
    pub fn record(&mut self, spot_grid: &[f64]) {
        assert_eq!(spot_grid.len(), self.running_sum.len());
        for (s, v) in self.running_sum.iter_mut().zip(spot_grid.iter()) {
            *s += *v;
        }
        self.dates_visited += 1;
        if self.remaining_dates > 0 { self.remaining_dates -= 1; }
    }

    /// Current running average at each grid point.
    pub fn average(&self) -> Vec<f64> {
        let n = self.dates_visited.max(1) as f64;
        self.running_sum.iter().map(|&s| s / n).collect()
    }
}

/// Running extreme (max/min) condition for lookback options.
#[derive(Debug, Clone)]
pub struct RunningExtremeCondition {
    /// Track maximum (true) or minimum (false).
    pub is_max: bool,
    /// Current running max/min at each grid point.
    pub extremum: Vec<f64>,
}

impl RunningExtremeCondition {
    /// Create with initial extreme values.
    pub fn new_max(n_grid: usize, initial: f64) -> Self {
        Self { is_max: true, extremum: vec![initial; n_grid] }
    }
    pub fn new_min(n_grid: usize, initial: f64) -> Self {
        Self { is_max: false, extremum: vec![initial; n_grid] }
    }

    /// Update extreme at each grid point against `spot_grid`.
    pub fn update(&mut self, spot_grid: &[f64]) {
        for (e, &s) in self.extremum.iter_mut().zip(spot_grid.iter()) {
            if self.is_max { *e = e.max(s); } else { *e = e.min(s); }
        }
    }
}

// =========================================================================
// 1D FDM solver with step conditions
// =========================================================================

/// Result from a 1D FDM solve with step conditions.
#[derive(Debug, Clone)]
pub struct Fd1dStepResult {
    /// Option values at each grid point at t=0.
    pub values: Vec<f64>,
    /// Time step count used.
    pub n_steps: usize,
}

/// 1D FDM PDE solver that applies a sequence of step conditions at each step.
///
/// Solves backward from T to 0 with the chosen scheme and applies
/// the provided step conditions at every time step.
///
/// # Type parameters
/// None – conditions are trait objects.
pub struct Fd1dSolver<'a> {
    /// The spatial grid.
    pub mesher: &'a Mesher1d,
    /// BS operator (time-independent portion).
    pub op: &'a TripleBandOp,
    /// Time step size Δt.
    pub dt: f64,
    /// Number of time steps.
    pub n_steps: usize,
    /// Time-stepping scheme.
    pub scheme: FdmScheme1d,
    /// Barrier condition (optional).
    pub barrier: Option<BarrierCondition>,
    /// American early-exercise condition (optional).
    pub american: Option<AmericanExerciseCondition>,
}

impl<'a> Fd1dSolver<'a> {
    /// Solve backward from terminal values, returning values at t=0.
    pub fn solve(&self, terminal_values: Vec<f64>) -> Fd1dStepResult {
        let mut v = terminal_values;
        let locs = &self.mesher.locations;

        for _ in 0..self.n_steps {
            // Apply time step
            v = match self.scheme {
                FdmScheme1d::CrankNicolson => crank_nicolson_step(self.op, &v, self.dt, 0.5),
                FdmScheme1d::FullyImplicit => {
                    use crate::fdm_operators::implicit_step;
                    implicit_step(self.op, &v, self.dt)
                }
            };

            // Apply barrier condition
            if let Some(bc) = &self.barrier {
                bc.apply(locs, &mut v, true); // assume log-scale mesher
            }

            // Apply American condition
            if let Some(am) = &self.american {
                am.apply(&mut v);
            }
        }

        Fd1dStepResult { values: v, n_steps: self.n_steps }
    }
}

// =========================================================================
// 2D Heston solver with scheme selection
// =========================================================================

/// Price a European/American option under Heston using a selectable ADI scheme.
///
/// Thin ergonomic wrapper around `fd_heston_solve_adi` that accepts the
/// `FdmScheme2d` enum instead of the lower-level `AdiScheme`.
///
/// # Arguments
/// - `spot`, `strike`, `r`, `q` — market parameters
/// - `v0`, `kappa`, `theta_h`, `sigma`, `rho` — Heston parameters
/// - `expiry` — time to expiry in years
/// - `is_call`, `is_american` — option flags
/// - `nx`, `nv`, `n_time` — grid dimensions
/// - `scheme` — ADI scheme to use
#[allow(clippy::too_many_arguments)]
pub fn heston_adi_solve(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    v0: f64,
    kappa: f64,
    theta_h: f64,
    sigma: f64,
    rho: f64,
    expiry: f64,
    is_call: bool,
    is_american: bool,
    nx: usize,
    nv: usize,
    n_time: usize,
    scheme: FdmScheme2d,
) -> HestonFdResult {
    let adi_scheme = match scheme {
        FdmScheme2d::Douglas => AdiScheme::Douglas,
        FdmScheme2d::HundsdorferVerwer => AdiScheme::HundsdorferVerwer,
        FdmScheme2d::ModifiedCraigSneyd => AdiScheme::ModifiedCraigSneyd,
    };
    fd_heston_solve_adi(
        spot, strike, r, q, v0, kappa, theta_h, sigma, rho,
        expiry, is_call, is_american, nx, nv, n_time, adi_scheme,
    )
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fdm_meshers::*;
    use crate::fdm_operators::build_bs_operator;

    #[test]
    fn barrier_condition_down_out() {
        let locs: Vec<f64> = (0..10).map(|i| (80.0 + i as f64 * 2.0).ln()).collect(); // ln 80..98
        let mut values: Vec<f64> = vec![5.0; 10];
        let bc = BarrierCondition { barrier_level: 85.0, direction: BarrierDirection::Down, is_out: true };
        bc.apply(&locs, &mut values, true);
        // Points with loc ≤ ln(85) should be zeroed
        let ln85 = 85.0_f64.ln();
        for (i, &loc) in locs.iter().enumerate() {
            if loc <= ln85 {
                assert_eq!(values[i], 0.0, "i={}", i);
            } else {
                assert_eq!(values[i], 5.0, "i={}", i);
            }
        }
    }

    #[test]
    fn averaging_condition_accumulates() {
        let mut avg = AveragingCondition::new(3, 3);
        avg.record(&[100.0, 105.0, 110.0]);
        avg.record(&[102.0, 107.0, 112.0]);
        let a = avg.average();
        assert!((a[0] - 101.0).abs() < 1e-10, "avg[0]={}", a[0]);
    }

    #[test]
    fn running_max_condition() {
        let mut rm = RunningExtremeCondition::new_max(3, 100.0);
        rm.update(&[95.0, 110.0, 90.0]);
        assert_eq!(rm.extremum[0], 100.0); // no update, 95 < 100
        assert_eq!(rm.extremum[1], 110.0); // updated
    }

    #[test]
    fn fd1d_crank_nicolson_call_positive() {
        // Build a simple Bs mesher and operator, solve a European call
        let s_min = 20.0_f64;
        let s_max = 300.0_f64;
        let ns = 50;
        // uniform log-space mesher
        let mesher = uniform_1d_mesher(s_min.ln(), s_max.ln(), ns);
        let r = 0.05;
        let sigma = 0.20;
        let tau = 1.0;
        let k = 100.0;
        let dt = tau / 200.0;
        let n_steps = 200;

        let op = build_bs_operator(&mesher.locations, r, 0.0, sigma);

        // Terminal payoff: call
        let terminal: Vec<f64> = mesher.locations.iter().map(|&x| {
            let s = x.exp();
            (s - k).max(0.0)
        }).collect();

        let solver = Fd1dSolver {
            mesher: &mesher,
            op: &op,
            dt,
            n_steps,
            scheme: FdmScheme1d::CrankNicolson,
            barrier: None,
            american: None,
        };
        let result = solver.solve(terminal);

        // Find ATM value (interpolate near x ≈ ln(100))
        let ln100 = 100.0_f64.ln();
        let idx = mesher.lower_index(ln100);
        let v = result.values[idx];
        assert!(v > 5.0 && v < 20.0, "ATM call={}", v);
    }

    #[test]
    fn fd1d_barrier_reduces_price() {
        let s_min = 20.0_f64;
        let s_max = 300.0_f64;
        let ns = 50;
        let mesher = uniform_1d_mesher(s_min.ln(), s_max.ln(), ns);
        let r = 0.05;
        let sigma = 0.20;
        let k = 100.0;
        let dt = 1.0 / 200.0;
        let n_steps = 200;
        let op = build_bs_operator(&mesher.locations, r, 0.0, sigma);
        let terminal: Vec<f64> = mesher.locations.iter().map(|&x| (x.exp() - k).max(0.0)).collect();

        // Without barrier
        let solver_no_barrier = Fd1dSolver {
            mesher: &mesher, op: &op, dt, n_steps,
            scheme: FdmScheme1d::CrankNicolson,
            barrier: None, american: None,
        };
        let r_no = solver_no_barrier.solve(terminal.clone());

        // With down-and-out barrier at 80
        let solver_barrier = Fd1dSolver {
            mesher: &mesher, op: &op, dt, n_steps,
            scheme: FdmScheme1d::CrankNicolson,
            barrier: Some(BarrierCondition { barrier_level: 80.0, direction: BarrierDirection::Down, is_out: true }),
            american: None,
        };
        let r_bar = solver_barrier.solve(terminal);

        let ln100 = 100.0_f64.ln();
        let idx = mesher.lower_index(ln100);
        assert!(r_bar.values[idx] < r_no.values[idx], "barrier reduces price");
    }
}
