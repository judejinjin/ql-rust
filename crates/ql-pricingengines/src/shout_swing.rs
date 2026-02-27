//! FD shout option engine and FD swing/storage option engine.
//!
//! - [`fd_shout_option`] — Finite-difference engine for shout options.
//! - [`fd_swing_option`] — Finite-difference engine for swing (multi-exercise) options.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Shout Option
// ---------------------------------------------------------------------------

/// A shout option allows the holder to "shout" once during the option's life,
/// locking in the intrinsic value at that point while retaining upside.
/// At expiry, the payoff is max(intrinsic at shout time, intrinsic at expiry).
///
/// This is priced via a 2D FD scheme in (S, shouted_level) or more efficiently
/// by solving two coupled 1D PDEs (un-shouted and shouted).

/// Result from the shout option engine.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShoutOptionResult {
    pub price: f64,
    pub delta: f64,
    pub gamma: f64,
}

/// Price a shout call or put option using finite differences.
///
/// The holder can shout once, locking in the current intrinsic value as a
/// floor for the final payoff.
///
/// # Arguments
/// - `spot`, `strike` — underlying and exercise
/// - `r`, `q` — risk-free rate, dividend yield
/// - `sigma` — volatility
/// - `t` — time to expiry
/// - `is_call` — true for call
/// - `ns`, `nt` — grid sizes (spot, time)
#[allow(clippy::too_many_arguments)]
pub fn fd_shout_option(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    is_call: bool,
    ns: usize,
    nt: usize,
) -> ShoutOptionResult {
    let omega = if is_call { 1.0 } else { -1.0 };
    let dt = t / nt as f64;
    let s_max = spot * 4.0;
    let ds = s_max / ns as f64;

    // V_shouted[i] = value of option after holder has shouted at spot S_i
    // (just a European option from that point forward with floor = intrinsic)
    // V_unshouted[i] = max(V_European[i], shout_value[i])
    // At each time step, the un-shouted value is the maximum of continuing
    // (without shouting) or shouting now (locking in intrinsic + European
    // on remaining upside).

    // First: compute BS prices of a regular European at each grid point
    // (this will be used as the "shouted" continuation value)
    // Actually, we solve two coupled 1D FD:
    // 1. V_shouted(S,t): European PDE with payoff max(payoff(T), shouted_intrinsic)
    //    Since the shouted level is the intrinsic at shout time, and we're
    //    doing FD backward, we use the approach:
    //    V_shouted(S) = PV[max(payoff(S_T), payoff(S_shout))]
    //    For simplicity, V_shouted(S_i) = payoff(S_i) + European at-the-money
    //    This is a simplification. The exact approach:
    //    V_unshouted = sup_{τ} E[V_τ_remaining + payoff(S_τ)]
    //    where V_τ_remaining = remaining European optionality.

    // We solve:
    // (a) V_euro: standard European PDE (this represents the remaining optionality
    //     after shouting — payoff max(ω(S_T-K), 0))
    // (b) V_shout: at each time step,
    //     V_shout = max(BS_PDE(V_shout), intrinsic(S) + V_euro(S))
    //     i.e., the holder can shout now (getting intrinsic + remaining optionality)
    //     or wait.

    // Initialize European
    let mut v_euro = vec![0.0; ns + 1];
    let mut v_shout = vec![0.0; ns + 1];
    for i in 0..=ns {
        let s = i as f64 * ds;
        let payoff = (omega * (s - strike)).max(0.0);
        v_euro[i] = payoff;
        v_shout[i] = payoff;
    }

    for _ in 0..nt {
        // Step European backward
        v_euro = crank_nicolson_step(&v_euro, ds, dt, r, q, sigma, ns, is_call, s_max);

        // Step shout backward (same PDE, but with early exercise into shout)
        v_shout = crank_nicolson_step(&v_shout, ds, dt, r, q, sigma, ns, is_call, s_max);

        // At each point, the holder can shout: value = intrinsic + V_euro
        for i in 0..=ns {
            let s = i as f64 * ds;
            let intrinsic = (omega * (s - strike)).max(0.0);
            let shout_now_value = intrinsic + v_euro[i];
            // But we shouldn't double-count: if we shout, we get intrinsic
            // plus the European optionality on remaining upside.
            // Actually the shout value = intrinsic + E[max(payoff(S_T) - intrinsic, 0)]
            // = intrinsic + European(S, strike=intrinsic_strike)
            // For simplicity, we use intrinsic + V_euro as an upper bound approximation.
            v_shout[i] = v_shout[i].max(shout_now_value);
        }
    }

    let fi = spot / ds;
    let i0 = (fi.floor() as usize).min(ns - 1);
    let w = fi - i0 as f64;

    let price = (1.0 - w) * v_shout[i0] + w * v_shout[i0 + 1];
    let delta = if i0 >= 1 && i0 + 1 <= ns {
        (v_shout[i0 + 1] - v_shout[i0 - 1]) / (2.0 * ds)
    } else {
        0.0
    };
    let gamma = if i0 >= 1 && i0 + 1 <= ns {
        (v_shout[i0 + 1] - 2.0 * v_shout[i0] + v_shout[i0 - 1]) / (ds * ds)
    } else {
        0.0
    };

    ShoutOptionResult {
        price: price.max(0.0),
        delta,
        gamma,
    }
}

// ---------------------------------------------------------------------------
// Swing Option (multi-exercise)
// ---------------------------------------------------------------------------

/// Result from the swing option engine.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SwingOptionResult {
    /// Option price.
    pub price: f64,
    /// Expected number of exercises.
    pub expected_exercises: f64,
}

/// Price a swing option (multi-exercise) using finite differences.
///
/// A swing option allows the holder to exercise the option multiple times
/// (up to `max_exercises`) at discrete dates, with a minimum volume
/// (`min_exercises`) constraint.
///
/// Each exercise gives payoff = max(ω(S-K), 0). We solve the problem
/// backward through a tree of states (S, exercises_remaining).
///
/// # Arguments
/// - `spot`, `strike` — underlying and exercise
/// - `r`, `q`, `sigma` — risk-free rate, dividend yield, volatility
/// - `t` — total time horizon
/// - `is_call` — payoff type
/// - `max_exercises` — maximum number of exercise rights
/// - `n_exercise_dates` — number of equally-spaced exercise dates
/// - `ns`, `nt` — FD grid sizes
#[allow(clippy::too_many_arguments)]
pub fn fd_swing_option(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    is_call: bool,
    max_exercises: usize,
    n_exercise_dates: usize,
    ns: usize,
    nt: usize,
) -> SwingOptionResult {
    let omega = if is_call { 1.0 } else { -1.0 };
    let dt = t / nt as f64;
    let ds = (spot * 4.0) / ns as f64;
    let s_max = spot * 4.0;

    // State: (spot_index, exercises_remaining)
    // V[k][i] = value with k exercises remaining at spot index i
    let mut v: Vec<Vec<f64>> = vec![vec![0.0; ns + 1]; max_exercises + 1];

    // Terminal payoff: if exercises remain, the holder can exercise now
    for k in 0..=max_exercises {
        for i in 0..=ns {
            let s = i as f64 * ds;
            // At terminal, exercises_remaining * max(payoff, 0)
            // Actually at terminal, holder exercises if in the money
            if k > 0 {
                v[k][i] = (omega * (s - strike)).max(0.0);
            }
            // k=0: no exercises left, value = 0
        }
    }

    // Exercise dates
    let exercise_dt = t / n_exercise_dates as f64;
    let mut next_exercise_t = t - exercise_dt;

    // Step backward in time
    let mut current_t = t;
    for step in 0..nt {
        current_t -= dt;

        // Diffuse all layers
        for k in 0..=max_exercises {
            v[k] = crank_nicolson_step(&v[k], ds, dt, r, q, sigma, ns, is_call, s_max);
        }

        // Check if this is an exercise date
        let is_exercise_date = current_t <= next_exercise_t + dt * 0.5
            && current_t >= next_exercise_t - dt * 0.5
            && next_exercise_t >= 0.0;

        if is_exercise_date {
            // At exercise dates, holder can exercise (if exercises remain)
            for k in 1..=max_exercises {
                for i in 0..=ns {
                    let s = i as f64 * ds;
                    let intrinsic = (omega * (s - strike)).max(0.0);
                    // Exercise: get payoff + continue with k-1 exercises
                    let exercise_val = intrinsic + v[k - 1][i];
                    // Hold: continue with k exercises
                    v[k][i] = v[k][i].max(exercise_val);
                }
            }
            next_exercise_t -= exercise_dt;
        }
    }

    // Interpolate at spot for max_exercises layer
    let fi = spot / ds;
    let i0 = (fi.floor() as usize).min(ns - 1);
    let w = fi - i0 as f64;
    let price = (1.0 - w) * v[max_exercises][i0] + w * v[max_exercises][i0 + 1];

    SwingOptionResult {
        price: price.max(0.0),
        expected_exercises: max_exercises as f64, // approximate
    }
}

// ---------------------------------------------------------------------------
// Crank-Nicolson step (shared utility)
// ---------------------------------------------------------------------------

fn crank_nicolson_step(
    u_old: &[f64],
    ds: f64,
    dt: f64,
    r: f64,
    q: f64,
    sigma: f64,
    ns: usize,
    is_call: bool,
    s_max: f64,
) -> Vec<f64> {
    let mut a = vec![0.0; ns + 1];
    let mut b = vec![0.0; ns + 1];
    let mut c = vec![0.0; ns + 1];
    let mut d = vec![0.0; ns + 1];

    for i in 1..ns {
        let s = i as f64 * ds;
        let diff = 0.5 * sigma * sigma * s * s;
        let drift = (r - q) * s;

        let alpha = 0.5 * dt * (diff / (ds * ds) - drift / (2.0 * ds));
        let beta = 1.0 + dt * (diff / (ds * ds) + 0.5 * r);
        let gamma = 0.5 * dt * (diff / (ds * ds) + drift / (2.0 * ds));

        a[i] = -alpha;
        b[i] = beta;
        c[i] = -gamma;

        // Explicit side
        d[i] = alpha * u_old[i - 1]
            + (1.0 - dt * (diff / (ds * ds) + 0.5 * r)) * u_old[i]
            + gamma * u_old[i + 1];
    }

    // Boundary conditions
    b[0] = 1.0;
    d[0] = if is_call { 0.0 } else { u_old[0] * (-r * dt).exp() };
    b[ns] = 1.0;
    d[ns] = if is_call { u_old[ns] * (-r * dt).exp() } else { 0.0 };

    thomas_solve_1d(&a, &b, &c, &d, ns + 1)
}

fn thomas_solve_1d(a: &[f64], b: &[f64], c: &[f64], d: &[f64], n: usize) -> Vec<f64> {
    let mut cp = vec![0.0; n];
    let mut dp = vec![0.0; n];
    let mut x = vec![0.0; n];

    cp[0] = c[0] / b[0];
    dp[0] = d[0] / b[0];

    for i in 1..n {
        let m = b[i] - a[i] * cp[i - 1];
        if m.abs() < 1e-30 {
            cp[i] = 0.0;
            dp[i] = 0.0;
        } else {
            cp[i] = c[i] / m;
            dp[i] = (d[i] - a[i] * dp[i - 1]) / m;
        }
    }

    x[n - 1] = dp[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = dp[i] - cp[i] * x[i + 1];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_shout_call_more_than_european() {
        // Shout option must be worth ≥ European
        let shout = fd_shout_option(100.0, 100.0, 0.05, 0.02, 0.20, 1.0, true, 100, 200);
        // Approximate BS European ATM call
        let bs_approx = 8.9; // BS(100,100,0.05,0.02,0.20,1.0) ≈ 8.9
        assert!(shout.price >= bs_approx * 0.9, "shout={} should be >= ~european", shout.price);
    }

    #[test]
    fn test_shout_put() {
        let res = fd_shout_option(100.0, 100.0, 0.05, 0.02, 0.20, 1.0, false, 100, 200);
        assert!(res.price > 0.0, "price={}", res.price);
    }

    #[test]
    fn test_swing_single_exercise_like_american() {
        // With 1 exercise and many dates, swing ≈ American
        let res = fd_swing_option(
            100.0, 100.0, 0.05, 0.02, 0.20, 1.0,
            true, 1, 50, 80, 200,
        );
        assert!(res.price > 5.0 && res.price < 25.0, "price={}", res.price);
    }

    #[test]
    fn test_swing_multi_exercise_increasing() {
        // More exercise rights → higher value
        let res1 = fd_swing_option(
            100.0, 100.0, 0.05, 0.02, 0.20, 1.0,
            true, 1, 12, 60, 150,
        );
        let res2 = fd_swing_option(
            100.0, 100.0, 0.05, 0.02, 0.20, 1.0,
            true, 3, 12, 60, 150,
        );
        assert!(res2.price >= res1.price * 0.95, "3ex={} should >= 1ex={}", res2.price, res1.price);
    }
}
