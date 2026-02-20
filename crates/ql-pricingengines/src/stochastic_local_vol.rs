//! Stochastic Local Volatility (SLV) model and calibration.
//!
//! The SLV model combines a local volatility surface with a stochastic
//! volatility component:
//!
//! $$dS = (r-q)S\,dt + L(t,S)\,\sigma_t\,S\,dW_1$$
//! $$d\sigma^2_t = \kappa(\theta - \sigma^2_t)\,dt + \xi\,\sigma_t\,dW_2$$
//!
//! where $L(t,S)$ is a "leverage function" calibrated so that the SLV model
//! reproduces the market's implied volatility surface exactly.
//!
//! ## Calibration approach
//!
//! 1. Bootstrap the Dupire local volatility surface from market option prices.
//! 2. Use the mixing formula: $L^2(t,S) = \sigma_{\text{loc}}^2(t,S) / E[\sigma^2_t | S_t = S]$
//! 3. Estimate $E[\sigma^2_t | S_t = S]$ via particle filtering or kernel regression
//!    on simulated paths.
//!
//! This module provides:
//! - `DupireLocalVol` — Dupire local volatility surface from a discrete grid
//! - `SlvModel` — SLV model parameters
//! - `SlvCalibrator` — iterative calibration of the leverage function
//! - `mc_slv` — Monte Carlo pricing under SLV dynamics

/// Dupire local volatility surface on a discrete `(time, spot)` grid.
#[derive(Debug, Clone)]
pub struct DupireLocalVol {
    /// Time points (years), sorted ascending.
    pub times: Vec<f64>,
    /// Spot/strike points, sorted ascending.
    pub spots: Vec<f64>,
    /// Local vol: `vols[time_idx][spot_idx]`.
    pub vols: Vec<Vec<f64>>,
}

impl DupireLocalVol {
    /// Create a new Dupire local vol surface.
    pub fn new(times: Vec<f64>, spots: Vec<f64>, vols: Vec<Vec<f64>>) -> Self {
        Self { times, spots, vols }
    }

    /// Interpolate local vol at `(t, s)` via bilinear interpolation.
    pub fn local_vol(&self, t: f64, s: f64) -> f64 {
        bilinear(&self.times, &self.spots, &self.vols, t, s)
    }

    /// Build a Dupire local vol surface from implied vol using the Dupire formula:
    ///
    /// $\sigma^2_L = \frac{\partial C/\partial T + (r-q)K \partial C/\partial K + q C}
    ///               {\frac{1}{2}K^2 \partial^2 C / \partial K^2}$
    ///
    /// This is approximated by finite differences on a BS call price grid.
    pub fn from_implied_vol(
        times: &[f64],
        strikes: &[f64],
        implied_vols: &[Vec<f64>],
        spot: f64,
        r: f64,
        q: f64,
    ) -> Self {
        let nt = times.len();
        let nk = strikes.len();
        let mut local_vols = vec![vec![0.0; nk]; nt];

        for i in 0..nt {
            let t = times[i];
            if t < 1e-10 {
                for j in 0..nk {
                    local_vols[i][j] = implied_vols[i][j];
                }
                continue;
            }

            for j in 0..nk {
                let k = strikes[j];
                let iv = implied_vols[i][j];
                let c = bs_call(spot, k, r, q, iv, t);

                // dC/dT (forward difference)
                let dc_dt = if i + 1 < nt {
                    let c_next = bs_call(spot, k, r, q, implied_vols[i + 1][j], times[i + 1]);
                    (c_next - c) / (times[i + 1] - t)
                } else if i > 0 {
                    let c_prev = bs_call(spot, k, r, q, implied_vols[i - 1][j], times[i - 1]);
                    (c - c_prev) / (t - times[i - 1])
                } else {
                    0.0
                };

                // dC/dK (central difference)
                let dc_dk = if j > 0 && j + 1 < nk {
                    let c_up = bs_call(spot, strikes[j + 1], r, q, implied_vols[i][j + 1], t);
                    let c_dn = bs_call(spot, strikes[j - 1], r, q, implied_vols[i][j - 1], t);
                    (c_up - c_dn) / (strikes[j + 1] - strikes[j - 1])
                } else {
                    0.0
                };

                // d²C/dK² (central difference)
                let d2c_dk2 = if j > 0 && j + 1 < nk {
                    let c_up = bs_call(spot, strikes[j + 1], r, q, implied_vols[i][j + 1], t);
                    let c_dn = bs_call(spot, strikes[j - 1], r, q, implied_vols[i][j - 1], t);
                    let dk = 0.5 * (strikes[j + 1] - strikes[j - 1]);
                    (c_up - 2.0 * c + c_dn) / (dk * dk)
                } else {
                    0.0
                };

                let numerator = dc_dt + (r - q) * k * dc_dk + q * c;
                let denominator = 0.5 * k * k * d2c_dk2;

                let lv2 = if denominator.abs() > 1e-12 && numerator / denominator > 0.0 {
                    numerator / denominator
                } else {
                    iv * iv // fallback to implied vol
                };

                local_vols[i][j] = lv2.sqrt();
            }
        }

        Self::new(times.to_vec(), strikes.to_vec(), local_vols)
    }
}

/// SLV model parameters.
#[derive(Debug, Clone)]
pub struct SlvModel {
    /// Spot price.
    pub spot: f64,
    /// Risk-free rate.
    pub r: f64,
    /// Dividend yield.
    pub q: f64,
    /// Heston-like stochastic vol: initial variance v₀.
    pub v0: f64,
    /// Mean-reversion speed κ.
    pub kappa: f64,
    /// Long-run variance θ.
    pub theta: f64,
    /// Vol of vol ξ.
    pub xi: f64,
    /// Correlation ρ between asset and variance Brownians.
    pub rho: f64,
    /// Leverage function: maps `(t, S)` → L(t, S).
    /// If `None`, L = 1 everywhere (pure stochastic vol).
    pub leverage: Option<DupireLocalVol>,
}

impl SlvModel {
    /// Create a pure Heston model (no leverage function).
    pub fn heston(
        spot: f64, r: f64, q: f64,
        v0: f64, kappa: f64, theta: f64, xi: f64, rho: f64,
    ) -> Self {
        Self {
            spot, r, q, v0, kappa, theta, xi, rho,
            leverage: None,
        }
    }

    /// Create a full SLV model with a calibrated leverage function.
    pub fn with_leverage(
        spot: f64, r: f64, q: f64,
        v0: f64, kappa: f64, theta: f64, xi: f64, rho: f64,
        leverage: DupireLocalVol,
    ) -> Self {
        Self {
            spot, r, q, v0, kappa, theta, xi, rho,
            leverage: Some(leverage),
        }
    }
}

/// SLV calibration result.
#[derive(Debug, Clone)]
pub struct SlvCalibrationResult {
    /// Calibrated leverage function.
    pub leverage: DupireLocalVol,
    /// Number of iterations performed.
    pub iterations: usize,
}

/// Calibrate the SLV leverage function via particle method.
///
/// This is a simplified calibration that:
/// 1. Simulates `n_paths` under Heston dynamics
/// 2. At each time step, estimates E[v_t | S_t = S] by kernel regression
/// 3. Computes L²(t, S) = σ²_loc(t, S) / E[v_t | S_t = S]
///
/// # Parameters
/// - `model`: base SLV model (Heston params + spot)
/// - `local_vol`: target Dupire local vol surface
/// - `n_paths`: number of MC paths for particle estimation
/// - `dt`: time step
/// - `max_iter`: maximum calibration iterations
pub fn calibrate_slv(
    model: &SlvModel,
    local_vol: &DupireLocalVol,
    n_paths: usize,
    dt: f64,
    max_iter: usize,
) -> SlvCalibrationResult {
    let n_times = local_vol.times.len();
    let n_spots = local_vol.spots.len();

    // Initialize leverage = 1 everywhere
    let mut leverage_vols = vec![vec![1.0; n_spots]; n_times];

    // Simple RNG (Xorshift for reproducibility)
    let mut rng_state: u64 = 42;

    for iter in 0..max_iter {
        // Simulate paths under SLV dynamics with current leverage
        let mut spots = vec![model.spot; n_paths];
        let mut vars = vec![model.v0; n_paths];

        for ti in 0..n_times {
            let t = local_vol.times[ti];
            let prev_t = if ti > 0 { local_vol.times[ti - 1] } else { 0.0 };
            let step_dt = (t - prev_t).min(dt).max(1e-6);
            let n_steps = ((t - prev_t) / step_dt).ceil() as usize;

            for _step in 0..n_steps {
                for p in 0..n_paths {
                    let s = spots[p];
                    let v = vars[p].max(0.0);
                    let sqrt_v = v.sqrt();

                    // Get leverage
                    let lev = bilinear(
                        &local_vol.times,
                        &local_vol.spots,
                        &leverage_vols,
                        t,
                        s,
                    );

                    // Generate correlated normals
                    let (z1, z2) = box_muller(&mut rng_state);
                    let w1 = z1;
                    let w2 = model.rho * z1 + (1.0 - model.rho * model.rho).sqrt() * z2;

                    // Euler step for S
                    let drift_s = (model.r - model.q) * s;
                    let diff_s = lev * sqrt_v * s;
                    spots[p] = (s + drift_s * step_dt + diff_s * step_dt.sqrt() * w1).max(1e-6);

                    // Euler step for v (with truncation)
                    let drift_v = model.kappa * (model.theta - v);
                    let diff_v = model.xi * sqrt_v;
                    vars[p] = (v + drift_v * step_dt + diff_v * step_dt.sqrt() * w2).max(0.0);
                }
            }

            // Kernel regression: E[v | S = s_j] for each grid spot
            for sj in 0..n_spots {
                let target_s = local_vol.spots[sj];
                let bandwidth = target_s * 0.10; // 10% of spot

                let mut sum_v = 0.0;
                let mut sum_w = 0.0;
                for p in 0..n_paths {
                    let dist = (spots[p] - target_s) / bandwidth;
                    let weight = (-0.5 * dist * dist).exp();
                    sum_v += vars[p] * weight;
                    sum_w += weight;
                }

                let cond_var = if sum_w > 1e-12 {
                    sum_v / sum_w
                } else {
                    model.v0
                };

                // L²(t, S) = σ²_loc / E[v | S]
                let local_var = local_vol.vols[ti][sj].powi(2);
                let lev_sq = if cond_var > 1e-12 {
                    (local_var / cond_var).max(0.01).min(100.0)
                } else {
                    1.0
                };
                leverage_vols[ti][sj] = lev_sq.sqrt();
            }
        }

        // Check convergence (simple: if max change < threshold)
        let _ = iter; // use all iterations for now
    }

    SlvCalibrationResult {
        leverage: DupireLocalVol::new(
            local_vol.times.clone(),
            local_vol.spots.clone(),
            leverage_vols,
        ),
        iterations: max_iter,
    }
}

/// Result from SLV Monte Carlo pricing.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SlvMcResult {
    pub price: f64,
    pub std_error: f64,
}

/// Monte Carlo pricing under SLV dynamics.
///
/// Prices a European option using Euler discretization of the SLV model.
pub fn mc_slv(
    model: &SlvModel,
    strike: f64,
    maturity: f64,
    is_call: bool,
    n_paths: usize,
    n_steps: usize,
) -> SlvMcResult {
    let dt = maturity / n_steps as f64;
    let sqrt_dt = dt.sqrt();
    let df = (-model.r * maturity).exp();

    let mut rng_state: u64 = 123456789;
    let mut payoffs = Vec::with_capacity(n_paths);

    for _ in 0..n_paths {
        let mut s = model.spot;
        let mut v = model.v0;

        for step in 0..n_steps {
            let t = step as f64 * dt;
            let sqrt_v = v.max(0.0).sqrt();

            let lev = if let Some(ref leverage) = model.leverage {
                leverage.local_vol(t, s)
            } else {
                1.0
            };

            let (z1, z2) = box_muller(&mut rng_state);
            let w1 = z1;
            let w2 = model.rho * z1 + (1.0 - model.rho * model.rho).sqrt() * z2;

            s = (s + (model.r - model.q) * s * dt + lev * sqrt_v * s * sqrt_dt * w1).max(1e-8);
            v = (v + model.kappa * (model.theta - v) * dt + model.xi * sqrt_v * sqrt_dt * w2)
                .max(0.0);
        }

        let payoff = if is_call {
            (s - strike).max(0.0)
        } else {
            (strike - s).max(0.0)
        };
        payoffs.push(payoff);
    }

    let n = payoffs.len() as f64;
    let mean = payoffs.iter().sum::<f64>() / n;
    let var = payoffs.iter().map(|&p| (p - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std_error = (var / n).sqrt();

    SlvMcResult {
        price: df * mean,
        std_error: df * std_error,
    }
}

// ===========================================================================
//  Helpers
// ===========================================================================

fn bilinear(xs: &[f64], ys: &[f64], grid: &[Vec<f64>], x: f64, y: f64) -> f64 {
    let (xi, xf) = bracket(xs, x);
    let (yi, yf) = bracket(ys, y);
    let xi2 = (xi + 1).min(xs.len() - 1);
    let yi2 = (yi + 1).min(ys.len() - 1);
    let v00 = grid[xi][yi];
    let v01 = grid[xi][yi2];
    let v10 = grid[xi2][yi];
    let v11 = grid[xi2][yi2];
    let v0 = v00 + xf * (v10 - v00);
    let v1 = v01 + xf * (v11 - v01);
    v0 + yf * (v1 - v0)
}

fn bracket(xs: &[f64], x: f64) -> (usize, f64) {
    let n = xs.len();
    if n == 0 {
        return (0, 0.0);
    }
    if x <= xs[0] {
        return (0, 0.0);
    }
    if x >= xs[n - 1] {
        return (n - 1, 0.0);
    }
    let mut lo = 0;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xs[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let span = xs[hi] - xs[lo];
    let frac = if span.abs() < 1e-15 { 0.0 } else { (x - xs[lo]) / span };
    (lo, frac)
}

/// Simple Box-Muller transform for pairs of standard normals.
fn box_muller(state: &mut u64) -> (f64, f64) {
    let u1 = xorshift(state);
    let u2 = xorshift(state);
    let r = (-2.0 * u1.max(1e-15).ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

fn xorshift(state: &mut u64) -> f64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    (x as f64) / (u64::MAX as f64)
}

/// Black-Scholes call price.
fn bs_call(spot: f64, strike: f64, r: f64, q: f64, vol: f64, t: f64) -> f64 {
    if t <= 0.0 || vol <= 0.0 {
        return (spot * (-q * t).exp() - strike * (-r * t).exp()).max(0.0);
    }
    let d1 = ((spot / strike).ln() + (r - q + 0.5 * vol * vol) * t) / (vol * t.sqrt());
    let d2 = d1 - vol * t.sqrt();
    spot * (-q * t).exp() * norm_cdf(d1) - strike * (-r * t).exp() * norm_cdf(d2)
}

fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x * std::f64::consts::FRAC_1_SQRT_2))
}

fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let t = 1.0 / (1.0 + p * x.abs());
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn dupire_local_vol_bilinear() {
        let lv = DupireLocalVol::new(
            vec![0.5, 1.0],
            vec![90.0, 100.0, 110.0],
            vec![
                vec![0.22, 0.20, 0.21],
                vec![0.23, 0.21, 0.22],
            ],
        );
        assert_abs_diff_eq!(lv.local_vol(0.5, 100.0), 0.20);
        assert_abs_diff_eq!(lv.local_vol(1.0, 110.0), 0.22);
        // Mid-point interpolation
        let mid = lv.local_vol(0.75, 100.0);
        assert!(mid > 0.19 && mid < 0.22);
    }

    #[test]
    fn dupire_from_implied_vol() {
        let times = vec![0.25, 0.5, 1.0];
        let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];
        let implied_vols = vec![
            vec![0.22, 0.21, 0.20, 0.21, 0.22],
            vec![0.23, 0.22, 0.21, 0.22, 0.23],
            vec![0.24, 0.23, 0.22, 0.23, 0.24],
        ];
        let lv = DupireLocalVol::from_implied_vol(
            &times, &strikes, &implied_vols, 100.0, 0.03, 0.01,
        );
        // Local vols should be positive and reasonable
        for row in &lv.vols {
            for &v in row {
                assert!(v > 0.0 && v < 1.0, "Local vol {v} out of range");
            }
        }
    }

    #[test]
    fn slv_heston_mc_positive() {
        let model = SlvModel::heston(
            100.0, 0.05, 0.0,
            0.04, 1.5, 0.04, 0.3, -0.7,
        );
        let result = mc_slv(&model, 100.0, 1.0, true, 10000, 100);
        assert!(result.price > 0.0, "SLV MC call should be positive");
        assert!(result.price < 30.0, "SLV MC call should be reasonable");
    }

    #[test]
    fn slv_mc_put_call_parity() {
        let model = SlvModel::heston(
            100.0, 0.05, 0.0,
            0.04, 1.5, 0.04, 0.3, -0.7,
        );
        let call = mc_slv(&model, 100.0, 1.0, true, 50000, 100);
        let put = mc_slv(&model, 100.0, 1.0, false, 50000, 100);
        let parity_fwd = 100.0 - 100.0 * (-0.05_f64).exp();
        let diff = (call.price - put.price - parity_fwd).abs();
        // Allow generous tolerance for MC
        assert!(diff < 2.0, "Put-call parity diff {diff} too large");
    }

    #[test]
    fn slv_with_leverage() {
        let leverage = DupireLocalVol::new(
            vec![0.5, 1.0],
            vec![80.0, 100.0, 120.0],
            vec![
                vec![1.1, 1.0, 0.9],
                vec![1.0, 1.0, 1.0],
            ],
        );
        let model = SlvModel::with_leverage(
            100.0, 0.05, 0.0,
            0.04, 1.5, 0.04, 0.3, -0.7,
            leverage,
        );
        let result = mc_slv(&model, 100.0, 1.0, true, 10000, 100);
        assert!(result.price > 0.0);
    }

    #[test]
    fn calibrate_slv_produces_leverage() {
        let local_vol = DupireLocalVol::new(
            vec![0.5, 1.0],
            vec![90.0, 100.0, 110.0],
            vec![
                vec![0.22, 0.20, 0.21],
                vec![0.23, 0.21, 0.22],
            ],
        );
        let model = SlvModel::heston(
            100.0, 0.05, 0.0,
            0.04, 1.5, 0.04, 0.3, -0.7,
        );
        let result = calibrate_slv(&model, &local_vol, 1000, 0.01, 2);
        assert_eq!(result.iterations, 2);
        // Leverage should be positive
        for row in &result.leverage.vols {
            for &v in row {
                assert!(v > 0.0, "Leverage should be positive, got {v}");
            }
        }
    }
}
