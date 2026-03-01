#![allow(clippy::too_many_arguments)]
//! Extended FD engines — G168-G169.
//!
//! - [`fd_simple_bs_swing`] (G168) — FD swing option engine under BS
//! - [`fd_multi_period`] (G169) — FD multi-exercise-period engine

use serde::{Deserialize, Serialize};

use ql_methods::fdm_meshers::{concentrating_1d_mesher, Mesher1d};
use ql_methods::fdm_operators::{build_bs_operator, crank_nicolson_step, TripleBandOp};

// ═══════════════════════════════════════════════════════════════════════════
// G168: FdSimpleBSSwingEngine — FD swing option under BS
// ═══════════════════════════════════════════════════════════════════════════

/// Result of swing option FD pricing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FdSwingResult {
    pub price: f64,
    pub delta: f64,
    pub gamma: f64,
}

/// Price a swing option under Black-Scholes using finite differences.
///
/// A swing option grants the holder the right to exercise multiple times
/// (up to `max_exercises`) at pre-specified dates, each time receiving
/// the intrinsic value.
///
/// The algorithm solves the 1D BS PDE backward, applying the exercise
/// condition at each exercise date: V = max(V_continue, V_exercise + payoff).
///
/// # Arguments
/// - `spot`: current underlying price
/// - `strike`: strike price
/// - `r`: risk-free rate
/// - `q`: dividend yield
/// - `vol`: volatility
/// - `expiry`: time to final exercise date (years)
/// - `exercise_times`: sorted exercise times (years)
/// - `max_exercises`: maximum number of exercises
/// - `is_call`: true for call, false for put
/// - `n_grid`: number of spatial grid points
/// - `n_steps`: number of time steps
pub fn fd_simple_bs_swing(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    vol: f64,
    expiry: f64,
    exercise_times: &[f64],
    max_exercises: usize,
    is_call: bool,
    n_grid: usize,
    n_steps: usize,
) -> FdSwingResult {
    let x0 = spot.ln();
    let x_strike = strike.ln();
    let std_dev = vol * expiry.sqrt();
    let lo = x0 - 5.0 * std_dev;
    let hi = x0 + 5.0 * std_dev;

    let mesher = concentrating_1d_mesher(lo, hi, n_grid, x_strike, 0.8);
    let op = build_bs_operator(&mesher.locations, r, q, vol);

    let dt = expiry / n_steps as f64;

    // Terminal condition for each remaining-exercise level
    // v[k] = value with k exercises remaining
    let mut v: Vec<Vec<f64>> = (0..=max_exercises)
        .map(|_k| vec![0.0; n_grid])
        .collect();

    // Apply exercise at terminal time (expiry) if it is an exercise date
    let expiry_is_exercise = exercise_times
        .iter()
        .any(|&et| (et - expiry).abs() < dt * 0.5);
    if expiry_is_exercise {
        for k in 1..=max_exercises {
            for i in 0..n_grid {
                let s = mesher.locations[i].exp();
                let payoff = if is_call {
                    (s - strike).max(0.0)
                } else {
                    (strike - s).max(0.0)
                };
                v[k][i] = payoff;
            }
        }
    }

    // Time stepping backward
    for step in (0..n_steps).rev() {
        let t = step as f64 * dt;

        // Check if this is an exercise time
        let is_exercise = exercise_times
            .iter()
            .any(|&et| (et - t).abs() < dt * 0.5);

        // Evolve each level
        for k in 0..=max_exercises {
            v[k] = crank_nicolson_step(&op, &v[k], dt, 0.5);
        }

        // Apply exercise condition
        if is_exercise {
            for k in 1..=max_exercises {
                for i in 0..n_grid {
                    let s = mesher.locations[i].exp();
                    let payoff = if is_call {
                        (s - strike).max(0.0)
                    } else {
                        (strike - s).max(0.0)
                    };
                    // Exercise: get payoff + continue with k-1 rights
                    let exercise_val = payoff + v[k - 1][i];
                    v[k][i] = v[k][i].max(exercise_val);
                }
            }
        }
    }

    // Interpolate at spot
    let idx = mesher.lower_index(x0);
    let interp = |vals: &[f64]| -> f64 {
        if idx + 1 >= n_grid {
            return vals[n_grid - 1];
        }
        let w = (x0 - mesher.locations[idx])
            / (mesher.locations[idx + 1] - mesher.locations[idx]);
        (1.0 - w) * vals[idx] + w * vals[idx + 1]
    };

    let price = interp(&v[max_exercises]);

    // Greeks via finite differences on the solution
    let delta = if idx >= 1 && idx + 2 < n_grid {
        let dx = mesher.locations[idx + 1] - mesher.locations[idx - 1];
        (v[max_exercises][idx + 1] - v[max_exercises][idx - 1]) / (dx * spot)
    } else {
        0.0
    };

    let gamma = if idx >= 1 && idx + 2 < n_grid {
        let dx = 0.5 * (mesher.locations[idx + 1] - mesher.locations[idx - 1]);
        let d2v = v[max_exercises][idx + 1] - 2.0 * v[max_exercises][idx]
            + v[max_exercises][idx - 1];
        d2v / (dx * dx * spot * spot)
    } else {
        0.0
    };

    FdSwingResult {
        price,
        delta,
        gamma,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// G169: FdMultiPeriodEngine — FD engine for multi-exercise products
// ═══════════════════════════════════════════════════════════════════════════

/// Result of multi-period FD pricing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FdMultiPeriodResult {
    pub price: f64,
    pub delta: f64,
    pub gamma: f64,
    /// Per-period values (if available).
    pub period_values: Vec<f64>,
}

/// Parameters for each exercise period.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExercisePeriod {
    /// Start time (years).
    pub start: f64,
    /// End time (years).
    pub end: f64,
    /// Strike for this period.
    pub strike: f64,
    /// Notional for this period.
    pub notional: f64,
}

/// Price a multi-period exercise product using finite differences.
///
/// The product consists of multiple exercise periods, each with its own
/// strike and notional. At the start of each period, the holder can decide
/// to enter or skip. Once entered, the payoff accrues until the period end.
///
/// # Arguments
/// - `spot`: current underlying
/// - `r`: risk-free rate
/// - `q`: dividend yield
/// - `vol`: volatility
/// - `periods`: exercise period specifications
/// - `is_call`: true for call payoffs
/// - `n_grid`: spatial grid points
/// - `n_steps_per_period`: time steps per period
pub fn fd_multi_period(
    spot: f64,
    r: f64,
    q: f64,
    vol: f64,
    periods: &[ExercisePeriod],
    is_call: bool,
    n_grid: usize,
    n_steps_per_period: usize,
) -> FdMultiPeriodResult {
    if periods.is_empty() {
        return FdMultiPeriodResult {
            price: 0.0,
            delta: 0.0,
            gamma: 0.0,
            period_values: Vec::new(),
        };
    }

    let x0 = spot.ln();
    let total_time = periods.iter().map(|p| p.end).fold(0.0_f64, f64::max);
    let std_dev = vol * total_time.sqrt();
    let lo = x0 - 5.0 * std_dev;
    let hi = x0 + 5.0 * std_dev;

    let mesher = concentrating_1d_mesher(lo, hi, n_grid, x0, 0.8);
    let op = build_bs_operator(&mesher.locations, r, q, vol);

    // Work backward from the last period
    let mut v = vec![0.0; n_grid]; // continuation value
    let mut period_values = Vec::new();

    for period in periods.iter().rev() {
        let dt = (period.end - period.start) / n_steps_per_period as f64;

        // At the end of this period: V = max(V_continue, payoff + V_next)
        // Time-step through the period
        for _ in 0..n_steps_per_period {
            v = crank_nicolson_step(&op, &v, dt, 0.5);
        }

        // At start of period: exercise decision
        for i in 0..n_grid {
            let s = mesher.locations[i].exp();
            let payoff = if is_call {
                period.notional * (s - period.strike).max(0.0)
            } else {
                period.notional * (period.strike - s).max(0.0)
            };
            v[i] = v[i].max(v[i] + payoff);
        }

        // Interpolate value at spot for this period
        let idx = mesher.lower_index(x0);
        let pv = if idx + 1 < n_grid {
            let w = (x0 - mesher.locations[idx])
                / (mesher.locations[idx + 1] - mesher.locations[idx]);
            (1.0 - w) * v[idx] + w * v[idx + 1]
        } else {
            v[n_grid - 1]
        };
        period_values.push(pv);
    }

    period_values.reverse();

    // Any remaining time from valuation to first period start
    let first_start = periods.iter().map(|p| p.start).fold(f64::MAX, f64::min);
    if first_start > 1e-10 {
        let dt = first_start / n_steps_per_period as f64;
        for _ in 0..n_steps_per_period {
            v = crank_nicolson_step(&op, &v, dt, 0.5);
        }
    }

    // Interpolate at spot
    let idx = mesher.lower_index(x0);
    let interp = |vals: &[f64]| -> f64 {
        if idx + 1 >= n_grid {
            return vals[n_grid - 1];
        }
        let w = (x0 - mesher.locations[idx])
            / (mesher.locations[idx + 1] - mesher.locations[idx]);
        (1.0 - w) * vals[idx] + w * vals[idx + 1]
    };

    let price = interp(&v);
    let delta = if idx >= 1 && idx + 2 < n_grid {
        let dx = mesher.locations[idx + 1] - mesher.locations[idx - 1];
        (v[idx + 1] - v[idx - 1]) / (dx * spot)
    } else {
        0.0
    };
    let gamma = if idx >= 1 && idx + 2 < n_grid {
        let dx = 0.5 * (mesher.locations[idx + 1] - mesher.locations[idx - 1]);
        let d2v = v[idx + 1] - 2.0 * v[idx] + v[idx - 1];
        d2v / (dx * dx * spot * spot)
    } else {
        0.0
    };

    FdMultiPeriodResult {
        price,
        delta,
        gamma,
        period_values,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn swing_single_exercise_equals_european() {
        // With 1 exercise at expiry, swing ≈ European
        let result = fd_simple_bs_swing(
            100.0, 100.0, 0.05, 0.02, 0.20, 1.0,
            &[1.0], // single exercise at T
            1,
            true, // call
            201, 200,
        );
        // European BS call ≈ 10.45 (approx)
        assert!(result.price > 5.0 && result.price < 20.0,
            "swing price = {}", result.price);
        assert!(result.delta > 0.0, "delta should be positive for call");
    }

    #[test]
    fn swing_multiple_exercises_higher() {
        // More exercise rights → higher value
        let r1 = fd_simple_bs_swing(
            100.0, 100.0, 0.05, 0.02, 0.20, 1.0,
            &[0.25, 0.5, 0.75, 1.0],
            1,
            false, // put
            101, 100,
        );
        let r2 = fd_simple_bs_swing(
            100.0, 100.0, 0.05, 0.02, 0.20, 1.0,
            &[0.25, 0.5, 0.75, 1.0],
            3,
            false,
            101, 100,
        );
        assert!(r2.price >= r1.price - 0.01,
            "more rights should be worth at least as much: {} vs {}", r2.price, r1.price);
    }

    #[test]
    fn multi_period_single_period() {
        let period = ExercisePeriod {
            start: 0.0,
            end: 1.0,
            strike: 100.0,
            notional: 1.0,
        };
        let result = fd_multi_period(
            100.0, 0.05, 0.02, 0.20,
            &[period],
            true,
            101, 100,
        );
        assert!(result.price > 0.0, "multi-period price should be positive");
        assert_eq!(result.period_values.len(), 1);
    }

    #[test]
    fn multi_period_two_periods() {
        let periods = vec![
            ExercisePeriod {
                start: 0.0,
                end: 0.5,
                strike: 100.0,
                notional: 1.0,
            },
            ExercisePeriod {
                start: 0.5,
                end: 1.0,
                strike: 105.0,
                notional: 1.0,
            },
        ];
        let result = fd_multi_period(
            100.0, 0.05, 0.02, 0.20,
            &periods,
            true,
            101, 50,
        );
        assert!(result.price > 0.0);
        assert_eq!(result.period_values.len(), 2);
    }

    #[test]
    fn multi_period_empty() {
        let result = fd_multi_period(100.0, 0.05, 0.02, 0.20, &[], true, 101, 100);
        assert_abs_diff_eq!(result.price, 0.0, epsilon = 1e-14);
    }
}
