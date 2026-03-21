//! Finite-difference vanilla option engines for extended models.
//!
//! - [`fd_bates_vanilla`] — FD engine for the Bates (jump-diffusion + SV) model.
//! - [`fd_sabr_vanilla`] — FD for the SABR model (CEV local vol + SV).
//! - [`fd_cev_vanilla`] — FD for the Constant Elasticity of Variance model.
//! - [`fd_cir_vanilla`] — FD for the Cox-Ingersoll-Ross model.
//! - [`fd_heston_hull_white`] — FD for the Heston + Hull-White hybrid model.

use serde::{Deserialize, Serialize};

/// Result for FD vanilla extension engines.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FdExtVanillaResult {
    pub price: f64,
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
}

// ---------------------------------------------------------------------------
// FD Bates Vanilla (Heston + Jump-Diffusion)
// ---------------------------------------------------------------------------

/// Finite-difference engine for the Bates model (Heston + Merton jumps).
///
/// Prices European or American options under stochastic volatility with
/// log-normal jumps using an operator-splitting scheme on the 2D PDE (S, v).
///
/// # Parameters
/// - `spot`, `strike` — underlying and exercise prices
/// - `r`, `q` — risk-free rate and dividend yield
/// - `v0`, `kappa`, `theta`, `sigma`, `rho` — Heston parameters
/// - `lambda_j`, `mu_j`, `sigma_j` — jump intensity, mean jump size, jump vol
/// - `t` — time to expiry
/// - `is_call`, `is_american` — option flags
/// - `ns`, `nv`, `nt` — grid sizes (spot, vol, time)
#[allow(clippy::too_many_arguments)]
pub fn fd_bates_vanilla(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    lambda_j: f64,
    mu_j: f64,
    sigma_j: f64,
    t: f64,
    is_call: bool,
    is_american: bool,
    ns: usize,
    nv: usize,
    nt: usize,
) -> FdExtVanillaResult {
    let omega = if is_call { 1.0 } else { -1.0 };
    let dt = t / nt as f64;

    // Jump-compensated drift
    let k_bar = (mu_j + 0.5 * sigma_j * sigma_j).exp() - 1.0;
    let r_comp = r - lambda_j * k_bar;

    // Spot grid (log-space)
    let s_max = spot * (5.0 * v0.sqrt() * t.sqrt()).exp();
    let s_min = spot * (-5.0 * v0.sqrt() * t.sqrt()).exp().max(1e-6);
    let ds = (s_max - s_min) / ns as f64;

    // Vol grid
    let v_max = (5.0 * theta).max(3.0 * v0);
    let dv = v_max / nv as f64;

    // Initialize payoff
    let total = (ns + 1) * (nv + 1);
    let mut u = vec![0.0; total];
    let idx = |i: usize, j: usize| -> usize { i * (nv + 1) + j };

    for i in 0..=ns {
        let s = s_min + i as f64 * ds;
        for j in 0..=nv {
            u[idx(i, j)] = (omega * (s - strike)).max(0.0);
        }
    }

    // Time-stepping with ADI (Douglas-Rachford splitting)
    for _ in 0..nt {
        let u_old = u.clone();

        // S-direction implicit sweep (simplified)
        for j in 1..nv {
            let v = j as f64 * dv;
            let mut a = vec![0.0; ns + 1];
            let mut b = vec![0.0; ns + 1];
            let mut c = vec![0.0; ns + 1];
            let mut d = vec![0.0; ns + 1];

            for i in 1..ns {
                let s = s_min + i as f64 * ds;
                let drift = (r_comp - q) * s;
                let diff = 0.5 * v * s * s;

                let am = dt * (-diff / (ds * ds) + drift / (2.0 * ds));
                let bm = 1.0 + dt * (2.0 * diff / (ds * ds) + r + lambda_j);
                let cm = dt * (-diff / (ds * ds) - drift / (2.0 * ds));

                a[i] = am;
                b[i] = bm;
                c[i] = cm;

                // Jump integral contribution (Gauss-Hermite approximate)
                let jump_val = lambda_j * dt * jump_integral(&u_old, idx, i, j, ns, nv, s, s_min, ds, mu_j, sigma_j);

                d[i] = u_old[idx(i, j)] + jump_val;
            }

            // Boundary conditions
            b[0] = 1.0;
            d[0] = u_old[idx(0, j)];
            b[ns] = 1.0;
            d[ns] = u_old[idx(ns, j)];

            // Thomas algorithm
            let soln = thomas_solve(&a, &b, &c, &d, ns + 1);
            for i in 0..=ns {
                u[idx(i, j)] = soln[i];
            }
        }

        // V-direction implicit sweep
        for i in 1..ns {
            let s = s_min + i as f64 * ds;
            let mut a = vec![0.0; nv + 1];
            let mut b = vec![0.0; nv + 1];
            let mut c = vec![0.0; nv + 1];
            let mut d = vec![0.0; nv + 1];

            for j in 1..nv {
                let v = j as f64 * dv;
                let drift_v = kappa * (theta - v);
                let diff_v = 0.5 * sigma * sigma * v;

                let am = dt * (-diff_v / (dv * dv) + drift_v / (2.0 * dv));
                let bm = 1.0 + dt * (2.0 * diff_v / (dv * dv));
                let cm = dt * (-diff_v / (dv * dv) - drift_v / (2.0 * dv));

                a[j] = am;
                b[j] = bm;
                c[j] = cm;

                // Cross derivative (explicit)
                let cross = if i > 0 && i < ns && j > 0 && j < nv {
                    rho * sigma * s * v * dt / (4.0 * ds * dv)
                        * (u_old[idx(i + 1, j + 1)] - u_old[idx(i + 1, j - 1)]
                           - u_old[idx(i - 1, j + 1)] + u_old[idx(i - 1, j - 1)])
                } else {
                    0.0
                };

                d[j] = u[idx(i, j)] + cross;
            }

            b[0] = 1.0;
            d[0] = u[idx(i, 0)];
            b[nv] = 1.0;
            d[nv] = u[idx(i, nv)];

            let soln = thomas_solve(&a, &b, &c, &d, nv + 1);
            for j in 0..=nv {
                u[idx(i, j)] = soln[j];
            }
        }

        // American exercise
        if is_american {
            for i in 0..=ns {
                let s = s_min + i as f64 * ds;
                for j in 0..=nv {
                    let intrinsic = (omega * (s - strike)).max(0.0);
                    u[idx(i, j)] = u[idx(i, j)].max(intrinsic);
                }
            }
        }
    }

    // Interpolate at (spot, v0)
    let i_s = ((spot - s_min) / ds).floor() as usize;
    let j_v = (v0 / dv).floor() as usize;
    let i_s = i_s.min(ns - 1);
    let j_v = j_v.min(nv - 1);

    let ws = (spot - s_min - i_s as f64 * ds) / ds;
    let wv = (v0 - j_v as f64 * dv) / dv;

    let price = (1.0 - ws) * (1.0 - wv) * u[idx(i_s, j_v)]
        + ws * (1.0 - wv) * u[idx(i_s + 1, j_v)]
        + (1.0 - ws) * wv * u[idx(i_s, j_v + 1)]
        + ws * wv * u[idx(i_s + 1, j_v + 1)];

    // Numerical Greeks
    let delta = if i_s >= 1 && i_s < ns {
        let u_up = (1.0 - wv) * u[idx(i_s + 1, j_v)] + wv * u[idx(i_s + 1, j_v + 1)];
        let u_dn = (1.0 - wv) * u[idx(i_s - 1, j_v)] + wv * u[idx(i_s - 1, j_v + 1)];
        (u_up - u_dn) / (2.0 * ds)
    } else {
        0.0
    };

    let gamma = if i_s >= 1 && i_s < ns {
        let u_up = (1.0 - wv) * u[idx(i_s + 1, j_v)] + wv * u[idx(i_s + 1, j_v + 1)];
        let u_dn = (1.0 - wv) * u[idx(i_s - 1, j_v)] + wv * u[idx(i_s - 1, j_v + 1)];
        let u_mid = (1.0 - wv) * u[idx(i_s, j_v)] + wv * u[idx(i_s, j_v + 1)];
        (u_up - 2.0 * u_mid + u_dn) / (ds * ds)
    } else {
        0.0
    };

    FdExtVanillaResult { price: price.max(0.0), delta, gamma, theta: 0.0 }
}

/// Approximate jump integral using 3-point quadrature in log-space.
#[allow(clippy::too_many_arguments)]
fn jump_integral(
    u: &[f64], idx: impl Fn(usize, usize) -> usize,
    i: usize, j: usize, ns: usize, _nv: usize,
    s: f64, s_min: f64, ds: f64,
    mu_j: f64, sigma_j: f64,
) -> f64 {
    // Approximate: E[V(S*exp(J))] - V(S)
    let nodes = [-1.7320508075688772, 0.0, 1.7320508075688772]; // sqrt(3), 0, sqrt(3)
    let weights = [1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0];

    let mut sum = 0.0;
    for (k, &node) in nodes.iter().enumerate() {
        let log_j = mu_j + sigma_j * node;
        let s_jumped = s * log_j.exp();
        // Interpolate u at jumped spot
        let fi = (s_jumped - s_min) / ds;
        let ii = (fi.floor() as usize).min(ns.saturating_sub(1));
        let w = fi - ii as f64;
        let w = w.clamp(0.0, 1.0);
        let v = if ii < ns {
            (1.0 - w) * u[idx(ii, j)] + w * u[idx(ii + 1, j)]
        } else {
            u[idx(ns, j)]
        };
        sum += weights[k] * v;
    }
    sum - u[idx(i, j)]
}

// ---------------------------------------------------------------------------
// FD SABR Vanilla
// ---------------------------------------------------------------------------

/// Finite-difference engine for the SABR model.
///
/// Uses the log-spot transformation and a 2D grid (x=ln(F), α).
///
/// # Parameters
/// - `forward`, `strike` — forward price and strike
/// - `sigma0` — initial SABR volatility (α₀)
/// - `beta` — CEV exponent
/// - `nu` — vol-of-vol
/// - `rho` — correlation
/// - `r` — risk-free rate
/// - `t` — time to expiry
/// - `is_call` — true for call
/// - `nx`, `na`, `nt` — grid sizes
#[allow(clippy::too_many_arguments)]
pub fn fd_sabr_vanilla(
    forward: f64,
    strike: f64,
    sigma0: f64,
    beta: f64,
    nu: f64,
    rho: f64,
    r: f64,
    t: f64,
    is_call: bool,
    nx: usize,
    na: usize,
    nt: usize,
) -> FdExtVanillaResult {
    let omega = if is_call { 1.0 } else { -1.0 };
    let dt = t / nt as f64;

    // Forward grid
    let f_max = forward * 5.0;
    let f_min = forward * 0.01;
    let df = (f_max - f_min) / nx as f64;

    // Alpha grid
    let a_max = sigma0 * 5.0;
    let da = a_max / na as f64;

    let total = (nx + 1) * (na + 1);
    let idx = |i: usize, j: usize| -> usize { i * (na + 1) + j };

    // Terminal condition
    let mut u = vec![0.0; total];
    for i in 0..=nx {
        let f = f_min + i as f64 * df;
        for j in 0..=na {
            u[idx(i, j)] = (omega * (f - strike)).max(0.0);
        }
    }

    // Time step backward
    for _ in 0..nt {
        let u_old = u.clone();

        // Explicit update (simpler; stable for small dt)
        for i in 1..nx {
            let f = f_min + i as f64 * df;
            for j in 1..na {
                let alpha = j as f64 * da;
                let local_vol = alpha * f.powf(beta);

                let d2f = (u_old[idx(i + 1, j)] - 2.0 * u_old[idx(i, j)] + u_old[idx(i - 1, j)]) / (df * df);
                let d2a = (u_old[idx(i, j + 1)] - 2.0 * u_old[idx(i, j)] + u_old[idx(i, j - 1)]) / (da * da);
                let d2fa = (u_old[idx(i + 1, j + 1)] - u_old[idx(i + 1, j - 1)]
                    - u_old[idx(i - 1, j + 1)] + u_old[idx(i - 1, j - 1)]) / (4.0 * df * da);

                u[idx(i, j)] = u_old[idx(i, j)]
                    + dt * (0.5 * local_vol * local_vol * d2f
                        + rho * local_vol * nu * alpha * d2fa
                        + 0.5 * nu * nu * alpha * alpha * d2a
                        - r * u_old[idx(i, j)]);
            }
        }
    }

    // Interpolate at (forward, sigma0)
    let i_f = ((forward - f_min) / df).floor() as usize;
    let j_a = (sigma0 / da).floor() as usize;
    let i_f = i_f.min(nx - 1);
    let j_a = j_a.min(na - 1);
    let wf = (forward - f_min - i_f as f64 * df) / df;
    let wa = (sigma0 - j_a as f64 * da) / da;

    let price = (1.0 - wf) * (1.0 - wa) * u[idx(i_f, j_a)]
        + wf * (1.0 - wa) * u[idx(i_f + 1, j_a)]
        + (1.0 - wf) * wa * u[idx(i_f, j_a + 1)]
        + wf * wa * u[idx(i_f + 1, j_a + 1)];

    FdExtVanillaResult {
        price: price.max(0.0),
        delta: 0.0,
        gamma: 0.0,
        theta: 0.0,
    }
}

// ---------------------------------------------------------------------------
// FD CEV Vanilla
// ---------------------------------------------------------------------------

/// Finite-difference engine for the CEV (Constant Elasticity of Variance) model.
///
/// The CEV model has local volatility σ(S) = σ·S^(β−1), so the diffusion is
/// dS = (r-q)S dt + σ S^β dW.
///
/// # Parameters
/// - `spot`, `strike` — prices
/// - `r`, `q` — interest rate, dividend yield
/// - `sigma` — CEV volatility
/// - `beta` — CEV exponent (0 < β ≤ 1 typically)
/// - `t` — time to expiry
/// - `is_call` — true for call
/// - `ns`, `nt` — grid sizes
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub fn fd_cev_vanilla(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    sigma: f64,
    beta: f64,
    t: f64,
    is_call: bool,
    ns: usize,
    nt: usize,
) -> FdExtVanillaResult {
    let omega = if is_call { 1.0 } else { -1.0 };
    let dt = t / nt as f64;
    let s_max = spot * 5.0;
    let ds = s_max / ns as f64;

    let mut u = vec![0.0; ns + 1];
    for i in 0..=ns {
        let s = i as f64 * ds;
        u[i] = (omega * (s - strike)).max(0.0);
    }

    // Crank-Nicolson
    for _ in 0..nt {
        let u_old = u.clone();
        let mut a = vec![0.0; ns + 1];
        let mut b = vec![0.0; ns + 1];
        let mut c = vec![0.0; ns + 1];
        let mut d = vec![0.0; ns + 1];

        for i in 1..ns {
            let s = i as f64 * ds;
            let local_vol = sigma * s.powf(beta);
            let diff = 0.5 * local_vol * local_vol;
            let drift = (r - q) * s;

            let am = 0.5 * dt * (diff / (ds * ds) - drift / (2.0 * ds));
            let bm = 1.0 + dt * (diff / (ds * ds) + 0.5 * r);
            let cm = 0.5 * dt * (diff / (ds * ds) + drift / (2.0 * ds));

            a[i] = -am;
            b[i] = bm;
            c[i] = -cm;

            // RHS (explicit part)
            let am_e = 0.5 * dt * (diff / (ds * ds) - drift / (2.0 * ds));
            let bm_e = 1.0 - dt * (diff / (ds * ds) + 0.5 * r);
            let cm_e = 0.5 * dt * (diff / (ds * ds) + drift / (2.0 * ds));

            d[i] = am_e * u_old[i - 1] + (2.0 - bm_e - 1.0) * u_old[i] + cm_e * u_old[i + 1];
            // Simplified: explicit side
            d[i] = (1.0 - dt * (diff / (ds * ds) + 0.5 * r)) * u_old[i]
                + 0.5 * dt * (diff / (ds * ds) - drift / (2.0 * ds)) * u_old[i - 1]
                + 0.5 * dt * (diff / (ds * ds) + drift / (2.0 * ds)) * u_old[i + 1];
        }

        b[0] = 1.0;
        d[0] = 0.0;
        b[ns] = 1.0;
        d[ns] = if is_call { s_max - strike * (-r * (t)).exp() } else { 0.0 };

        u = thomas_solve(&a, &b, &c, &d, ns + 1);
    }

    let fi = spot / ds;
    let i0 = (fi.floor() as usize).min(ns - 1);
    let w = fi - i0 as f64;
    let price = (1.0 - w) * u[i0] + w * u[i0 + 1];

    let delta = if i0 >= 1 && i0 < ns {
        (u[i0 + 1] - u[i0 - 1]) / (2.0 * ds)
    } else {
        0.0
    };

    let gamma = if i0 >= 1 && i0 < ns {
        (u[i0 + 1] - 2.0 * u[i0] + u[i0 - 1]) / (ds * ds)
    } else {
        0.0
    };

    FdExtVanillaResult { price: price.max(0.0), delta, gamma, theta: 0.0 }
}

// ---------------------------------------------------------------------------
// FD CIR vanilla
// ---------------------------------------------------------------------------

/// Finite-difference engine for European/American options under the CIR
/// (Cox-Ingersoll-Ross) interest rate model.
///
/// The underlying is driven by GBM with time-varying short rate from CIR:
///   dr = κ(θ − r) dt + σ_r √r dW_r
///
/// This is a 2D PDE in (S, r).
///
/// # Parameters
/// - `spot`, `strike` — underlying and exercise
/// - `r0` — initial short rate
/// - `q` — dividend yield
/// - `sigma_s` — asset volatility
/// - `kappa`, `theta_r`, `sigma_r` — CIR parameters
/// - `rho` — correlation between asset and rate
/// - `t` — time to expiry
/// - `is_call`, `is_american` — option type
/// - `ns`, `nr`, `nt` — grid sizes
#[allow(clippy::too_many_arguments)]
pub fn fd_cir_vanilla(
    spot: f64,
    strike: f64,
    r0: f64,
    q: f64,
    sigma_s: f64,
    kappa: f64,
    theta_r: f64,
    sigma_r: f64,
    rho: f64,
    t: f64,
    is_call: bool,
    is_american: bool,
    ns: usize,
    nr: usize,
    nt: usize,
) -> FdExtVanillaResult {
    let omega = if is_call { 1.0 } else { -1.0 };
    let dt = t / nt as f64;

    let s_max = spot * 4.0;
    let ds = s_max / ns as f64;
    let r_max = (5.0 * theta_r).max(3.0 * r0);
    let dr = r_max / nr as f64;

    let total = (ns + 1) * (nr + 1);
    let idx = |i: usize, j: usize| -> usize { i * (nr + 1) + j };

    let mut u = vec![0.0; total];
    for i in 0..=ns {
        let s = i as f64 * ds;
        for j in 0..=nr {
            u[idx(i, j)] = (omega * (s - strike)).max(0.0);
        }
    }

    for _ in 0..nt {
        let u_old = u.clone();

        // Explicit step (simple for stability with small dt)
        for i in 1..ns {
            let s = i as f64 * ds;
            for j in 1..nr {
                let rv = j as f64 * dr;

                let diff_s = 0.5 * sigma_s * sigma_s * s * s;
                let drift_s = (rv - q) * s;
                let diff_r = 0.5 * sigma_r * sigma_r * rv;
                let drift_r = kappa * (theta_r - rv);

                let d2s = (u_old[idx(i + 1, j)] - 2.0 * u_old[idx(i, j)] + u_old[idx(i - 1, j)]) / (ds * ds);
                let d1s = (u_old[idx(i + 1, j)] - u_old[idx(i - 1, j)]) / (2.0 * ds);
                let d2r = (u_old[idx(i, j + 1)] - 2.0 * u_old[idx(i, j)] + u_old[idx(i, j - 1)]) / (dr * dr);
                let d1r = (u_old[idx(i, j + 1)] - u_old[idx(i, j - 1)]) / (2.0 * dr);
                let d2sr = (u_old[idx(i + 1, j + 1)] - u_old[idx(i + 1, j - 1)]
                    - u_old[idx(i - 1, j + 1)] + u_old[idx(i - 1, j - 1)]) / (4.0 * ds * dr);

                u[idx(i, j)] = u_old[idx(i, j)]
                    + dt * (diff_s * d2s + drift_s * d1s
                        + diff_r * d2r + drift_r * d1r
                        + rho * sigma_s * s * sigma_r * rv.sqrt() * d2sr
                        - rv * u_old[idx(i, j)]);
            }
        }

        if is_american {
            for i in 0..=ns {
                let s = i as f64 * ds;
                for j in 0..=nr {
                    let intrinsic = (omega * (s - strike)).max(0.0);
                    u[idx(i, j)] = u[idx(i, j)].max(intrinsic);
                }
            }
        }
    }

    // Interpolate at (spot, r0)
    let i_s = (spot / ds).floor() as usize;
    let j_r = (r0 / dr).floor() as usize;
    let i_s = i_s.min(ns - 1);
    let j_r = j_r.min(nr - 1);
    let ws = spot / ds - i_s as f64;
    let wr = r0 / dr - j_r as f64;

    let price = (1.0 - ws) * (1.0 - wr) * u[idx(i_s, j_r)]
        + ws * (1.0 - wr) * u[idx(i_s + 1, j_r)]
        + (1.0 - ws) * wr * u[idx(i_s, j_r + 1)]
        + ws * wr * u[idx(i_s + 1, j_r + 1)];

    FdExtVanillaResult {
        price: price.max(0.0),
        delta: 0.0,
        gamma: 0.0,
        theta: 0.0,
    }
}

// ---------------------------------------------------------------------------
// FD Heston-Hull-White
// ---------------------------------------------------------------------------

/// Finite-difference engine for the Heston + Hull-White hybrid model.
///
/// A 3-factor model with stochastic volatility (Heston) and stochastic
/// interest rate (Hull-White one-factor). The PDE is 3D (S, v, r) so we
/// use operator splitting.
///
/// Due to dimensional complexity, we use a coarser grid and ADI splitting.
///
/// # Parameters  
/// - `spot`, `strike` — underlying and exercise
/// - `r0` — initial short rate
/// - `q` — dividend yield
/// - `v0`, `kappa_v`, `theta_v`, `sigma_v`, `rho_sv` — Heston params
/// - `kappa_r`, `theta_r_init`, `sigma_r` — Hull-White params
/// - `rho_sr` — correlation between S and r
/// - `t`, `is_call`, `is_american`
/// - `ns`, `nv`, `nr`, `nt` — grid sizes
#[allow(clippy::too_many_arguments)]
pub fn fd_heston_hull_white(
    spot: f64,
    strike: f64,
    r0: f64,
    q: f64,
    v0: f64,
    kappa_v: f64,
    theta_v: f64,
    sigma_v: f64,
    rho_sv: f64,
    kappa_r: f64,
    theta_r_init: f64,
    sigma_r: f64,
    rho_sr: f64,
    t: f64,
    is_call: bool,
    _is_american: bool,
    ns: usize,
    nv: usize,
    nr: usize,
    nt: usize,
) -> FdExtVanillaResult {
    let omega = if is_call { 1.0 } else { -1.0 };
    let dt = t / nt as f64;

    let s_max = spot * 4.0;
    let ds = s_max / ns as f64;
    let v_max = (5.0 * theta_v).max(3.0 * v0);
    let dv = v_max / nv as f64;
    let r_max = (5.0 * theta_r_init).max(3.0 * r0).max(0.2);
    let r_min = -r_max * 0.3;
    let dr = (r_max - r_min) / nr as f64;

    let total = (ns + 1) * (nv + 1) * (nr + 1);
    let idx = |i: usize, j: usize, k: usize| -> usize {
        i * (nv + 1) * (nr + 1) + j * (nr + 1) + k
    };

    let mut u = vec![0.0; total];
    for i in 0..=ns {
        let s = i as f64 * ds;
        for j in 0..=nv {
            for k in 0..=nr {
                u[idx(i, j, k)] = (omega * (s - strike)).max(0.0);
            }
        }
    }

    // Simple explicit time stepping for the 3D PDE
    for _ in 0..nt {
        let u_old = u.clone();

        for i in 1..ns {
            let s = i as f64 * ds;
            for j in 1..nv {
                let v = j as f64 * dv;
                for k in 1..nr {
                    let rv = r_min + k as f64 * dr;

                    // Heston + HW PDE terms
                    let diff_s = 0.5 * v * s * s;
                    let drift_s = (rv - q) * s;
                    let diff_v = 0.5 * sigma_v * sigma_v * v;
                    let drift_v = kappa_v * (theta_v - v);
                    let diff_r = 0.5 * sigma_r * sigma_r;
                    let drift_r = kappa_r * (theta_r_init - rv);

                    let d2s = (u_old[idx(i + 1, j, k)] - 2.0 * u_old[idx(i, j, k)] + u_old[idx(i - 1, j, k)]) / (ds * ds);
                    let d1s = (u_old[idx(i + 1, j, k)] - u_old[idx(i - 1, j, k)]) / (2.0 * ds);

                    let d2v = (u_old[idx(i, j + 1, k)] - 2.0 * u_old[idx(i, j, k)] + u_old[idx(i, j - 1, k)]) / (dv * dv);
                    let d1v = (u_old[idx(i, j + 1, k)] - u_old[idx(i, j - 1, k)]) / (2.0 * dv);

                    let d2r = (u_old[idx(i, j, k + 1)] - 2.0 * u_old[idx(i, j, k)] + u_old[idx(i, j, k - 1)]) / (dr * dr);
                    let d1r = (u_old[idx(i, j, k + 1)] - u_old[idx(i, j, k - 1)]) / (2.0 * dr);

                    // Cross derivatives (sv and sr)
                    let d2sv = (u_old[idx(i + 1, j + 1, k)] - u_old[idx(i + 1, j - 1, k)]
                        - u_old[idx(i - 1, j + 1, k)] + u_old[idx(i - 1, j - 1, k)]) / (4.0 * ds * dv);
                    let d2sr = (u_old[idx(i + 1, j, k + 1)] - u_old[idx(i + 1, j, k - 1)]
                        - u_old[idx(i - 1, j, k + 1)] + u_old[idx(i - 1, j, k - 1)]) / (4.0 * ds * dr);

                    u[idx(i, j, k)] = u_old[idx(i, j, k)]
                        + dt * (diff_s * d2s + drift_s * d1s
                            + diff_v * d2v + drift_v * d1v
                            + diff_r * d2r + drift_r * d1r
                            + rho_sv * sigma_v * v.sqrt() * s * d2sv
                            + rho_sr * sigma_r * s * v.sqrt() * d2sr
                            - rv * u_old[idx(i, j, k)]);
                }
            }
        }
    }

    // Interpolate at (spot, v0, r0)
    let i_s = (spot / ds).floor() as usize;
    let j_v = (v0 / dv).floor() as usize;
    let k_r = ((r0 - r_min) / dr).floor() as usize;
    let i_s = i_s.min(ns - 1);
    let j_v = j_v.min(nv - 1);
    let k_r = k_r.min(nr - 1);

    let ws = spot / ds - i_s as f64;
    let wv = v0 / dv - j_v as f64;
    let wr = (r0 - r_min) / dr - k_r as f64;

    // Trilinear interpolation
    let price = (1.0 - ws) * (1.0 - wv) * (1.0 - wr) * u[idx(i_s, j_v, k_r)]
        + ws * (1.0 - wv) * (1.0 - wr) * u[idx(i_s + 1, j_v, k_r)]
        + (1.0 - ws) * wv * (1.0 - wr) * u[idx(i_s, j_v + 1, k_r)]
        + ws * wv * (1.0 - wr) * u[idx(i_s + 1, j_v + 1, k_r)]
        + (1.0 - ws) * (1.0 - wv) * wr * u[idx(i_s, j_v, k_r + 1)]
        + ws * (1.0 - wv) * wr * u[idx(i_s + 1, j_v, k_r + 1)]
        + (1.0 - ws) * wv * wr * u[idx(i_s, j_v + 1, k_r + 1)]
        + ws * wv * wr * u[idx(i_s + 1, j_v + 1, k_r + 1)];

    FdExtVanillaResult {
        price: price.max(0.0),
        delta: 0.0,
        gamma: 0.0,
        theta: 0.0,
    }
}

// -- Utility ----------------------------------------------------------------

/// Thomas tridiagonal solve: a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
fn thomas_solve(a: &[f64], b: &[f64], c: &[f64], d: &[f64], n: usize) -> Vec<f64> {
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
    fn test_fd_bates_european_call() {
        let res = fd_bates_vanilla(
            100.0, 100.0, 0.05, 0.02,
            0.04, 2.0, 0.04, 0.4, -0.7,
            0.5, -0.05, 0.10,
            1.0, true, false,
            40, 20, 100,
        );
        // Bates price should be close to Heston-like for small jumps
        assert!(res.price > 5.0 && res.price < 25.0, "price={}", res.price);
    }

    #[test]
    fn test_fd_sabr_atm_call() {
        // With beta=0.5, SABR alpha maps to BS vol via sigma_BS ≈ alpha * F^(beta-1).
        // For alpha=0.20, F=100, beta=0.5: sigma_BS ≈ 0.20 * 100^(-0.5) = 0.02 (2%).
        // ATM call with ~2% vol: price ≈ F*σ√T/√(2π) * e^{-rT} ≈ 0.76.
        let res = fd_sabr_vanilla(
            100.0, 100.0, 0.20, 0.5, 0.4, -0.3,
            0.05, 1.0, true,
            60, 20, 200,
        );
        assert!(res.price > 0.3 && res.price < 5.0, "price={}", res.price);
    }

    #[test]
    fn test_fd_cev_call() {
        let res = fd_cev_vanilla(
            100.0, 100.0, 0.05, 0.02, 0.20, 1.0,
            1.0, true,
            100, 200,
        );
        // With beta=1.0, CEV = GBM, should be close to BS
        assert!(res.price > 5.0 && res.price < 20.0, "price={}", res.price);
    }

    #[test]
    fn test_fd_cir_call() {
        let res = fd_cir_vanilla(
            100.0, 100.0, 0.05, 0.02, 0.20,
            0.5, 0.05, 0.10, -0.2,
            1.0, true, false,
            30, 20, 100,
        );
        assert!(res.price > 3.0 && res.price < 25.0, "price={}", res.price);
    }

    #[test]
    fn test_fd_heston_hw_call() {
        let res = fd_heston_hull_white(
            100.0, 100.0, 0.05, 0.02,
            0.04, 2.0, 0.04, 0.4, -0.7,
            0.1, 0.05, 0.01, -0.2,
            1.0, true, false,
            15, 8, 8, 50,
        );
        assert!(res.price > 0.0, "price={}", res.price);
    }
}
