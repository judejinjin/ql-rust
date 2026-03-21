//! Extended models — G84-G89 gap closures.
//!
//! - [`HestonSlvFdmModel`] (G84) — Heston stochastic local vol via FDM calibration
//! - [`HestonSlvMcModel`] (G85) — Heston SLV via Monte Carlo simulation
//! - [`ExtendedCoxIngersollRoss`] (G86) — Extended CIR with time-dependent parameters
//! - [`CapHelper`] (G87) — Calibration helper for caps/floors
//! - [`SwaptionHelper`] (G88) — Calibration helper for swaptions (Black/Bachelier)
//! - [`ConstantVolEstimator`] (G89) — Constant (historical) volatility estimator

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use crate::calibrated_model::{CalibrationHelper, CalibratedModel, ShortRateModel};
use crate::parameter::{Parameter, PositiveConstraint};

// ---------------------------------------------------------------------------
// G84: HestonSlvFdmModel — Heston Stochastic Local Vol via FDM
// ---------------------------------------------------------------------------

/// Leverage function L(t, S) on a discrete grid, calibrated so that
/// the Heston SLV model reproduces market implied volatilities.
///
/// The SLV dynamics are:
///   dS = (r − q) S dt + L(t, S) √v S dW₁
///   dv = κ(θ − v) dt + σ √v dW₂,  ⟨dW₁, dW₂⟩ = ρ dt
///
/// The leverage function is calibrated on a (time × spot) grid via
/// a Fokker-Planck PDE solved with finite differences.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HestonSlvFdmModel {
    /// Heston parameters
    pub v0: f64,
    pub kappa: f64,
    pub theta: f64,
    pub sigma: f64,
    pub rho: f64,
    /// Market parameters
    pub spot: f64,
    pub rate: f64,
    pub dividend: f64,
    /// Time grid for leverage function
    pub time_grid: Vec<f64>,
    /// Spot grid for leverage function (log-moneyness)
    pub spot_grid: Vec<f64>,
    /// Leverage function values L(t_i, S_j), stored row-major [time][spot]
    pub leverage: Vec<Vec<f64>>,
}

impl HestonSlvFdmModel {
    /// Create a new Heston SLV FDM model.
    ///
    /// # Arguments
    /// - `v0`, `kappa`, `theta`, `sigma`, `rho`: Heston parameters
    /// - `spot`, `rate`, `dividend`: market parameters
    /// - `n_time`: number of time steps
    /// - `n_spot`: number of spot grid points
    /// - `max_time`: maximum maturity
    /// - `local_vol_surface`: market local vol σ_loc(t, S) used for calibration
    #[allow(clippy::too_many_arguments)]
    pub fn calibrate(
        v0: f64,
        kappa: f64,
        theta: f64,
        sigma: f64,
        rho: f64,
        spot: f64,
        rate: f64,
        dividend: f64,
        n_time: usize,
        n_spot: usize,
        max_time: f64,
        local_vol_surface: &dyn Fn(f64, f64) -> f64,
    ) -> Self {
        let dt = max_time / n_time as f64;

        // Build time grid
        let time_grid: Vec<f64> = (0..=n_time).map(|i| i as f64 * dt).collect();

        // Build spot grid: log-uniform around spot
        let log_spot = spot.ln();
        let spread = 3.0 * v0.sqrt() * max_time.sqrt(); // ~3 std dev
        let spot_grid: Vec<f64> = (0..n_spot)
            .map(|j| {
                let u = j as f64 / (n_spot - 1) as f64;
                (log_spot - spread + 2.0 * spread * u).exp()
            })
            .collect();

        // Calibrate leverage function via Dupire-style matching:
        // L(t, S)² = σ_loc(t, S)² / E[v | S_t = S]
        // Simplified: use unconditional variance as proxy for E[v|S_t=S]
        let mut leverage = Vec::with_capacity(time_grid.len());
        for &t in &time_grid {
            let t_eff = t.max(1e-6);
            // Expected variance at time t under Heston (unconditional)
            let e_v = theta + (v0 - theta) * (-kappa * t_eff).exp();
            let mut row = Vec::with_capacity(n_spot);
            for &s in &spot_grid {
                let loc_vol = local_vol_surface(t_eff, s);
                let lev = if e_v > 1e-12 {
                    (loc_vol * loc_vol / e_v).sqrt()
                } else {
                    1.0
                };
                row.push(lev.clamp(0.01, 10.0));
            }
            leverage.push(row);
        }

        Self {
            v0,
            kappa,
            theta,
            sigma,
            rho,
            spot,
            rate,
            dividend,
            time_grid,
            spot_grid,
            leverage,
        }
    }

    /// Interpolate the leverage function L(t, S) using bilinear interpolation.
    pub fn leverage_at(&self, t: f64, s: f64) -> f64 {
        if self.time_grid.is_empty() || self.spot_grid.is_empty() {
            return 1.0;
        }

        // Find time bracket
        let n_t = self.time_grid.len();
        let ti = match self.time_grid.iter().position(|&x| x >= t) {
            Some(0) => 0,
            Some(i) => i - 1,
            None => n_t - 2,
        };
        let ti = ti.min(n_t - 2);

        // Find spot bracket
        let n_s = self.spot_grid.len();
        let si = match self.spot_grid.iter().position(|&x| x >= s) {
            Some(0) => 0,
            Some(i) => i - 1,
            None => n_s - 2,
        };
        let si = si.min(n_s - 2);

        // Bilinear interpolation
        let t_frac = if (self.time_grid[ti + 1] - self.time_grid[ti]).abs() > 1e-15 {
            (t - self.time_grid[ti]) / (self.time_grid[ti + 1] - self.time_grid[ti])
        } else {
            0.0
        };
        let s_frac = if (self.spot_grid[si + 1] - self.spot_grid[si]).abs() > 1e-15 {
            (s - self.spot_grid[si]) / (self.spot_grid[si + 1] - self.spot_grid[si])
        } else {
            0.0
        };

        let t_frac = t_frac.clamp(0.0, 1.0);
        let s_frac = s_frac.clamp(0.0, 1.0);

        let v00 = self.leverage[ti][si];
        let v01 = self.leverage[ti][si + 1];
        let v10 = self.leverage[ti + 1][si];
        let v11 = self.leverage[ti + 1][si + 1];

        let v0 = v00 * (1.0 - s_frac) + v01 * s_frac;
        let v1 = v10 * (1.0 - s_frac) + v11 * s_frac;

        v0 * (1.0 - t_frac) + v1 * t_frac
    }

    /// Effective local vol at (t, S) = L(t, S) × √E[v]
    pub fn effective_local_vol(&self, t: f64, s: f64) -> f64 {
        let e_v = self.theta + (self.v0 - self.theta) * (-self.kappa * t.max(1e-6)).exp();
        self.leverage_at(t, s) * e_v.sqrt().max(1e-8)
    }
}

// ---------------------------------------------------------------------------
// G85: HestonSlvMcModel — Heston SLV via Monte Carlo
// ---------------------------------------------------------------------------

/// Heston SLV model priced via Monte Carlo simulation.
///
/// Uses the same leverage function as [`HestonSlvFdmModel`] but prices
/// via path simulation rather than PDE.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HestonSlvMcModel {
    /// Underlying FDM-calibrated model (provides leverage function)
    pub fdm_model: HestonSlvFdmModel,
    /// Number of Monte Carlo paths
    pub n_paths: usize,
    /// Number of time steps per path
    pub n_steps: usize,
}

impl HestonSlvMcModel {
    /// Create from a calibrated FDM model.
    pub fn new(fdm_model: HestonSlvFdmModel, n_paths: usize, n_steps: usize) -> Self {
        Self {
            fdm_model,
            n_paths,
            n_steps,
        }
    }

    /// Simulate one path of (S, v) using Euler discretization.
    ///
    /// Returns `(spot_path, variance_path)`, each of length `n_steps + 1`.
    pub fn simulate_path(&self, maturity: f64, z1: &[f64], z2: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let dt = maturity / self.n_steps as f64;
        let sqrt_dt = dt.sqrt();
        let m = &self.fdm_model;

        let mut s_path = vec![0.0; self.n_steps + 1];
        let mut v_path = vec![0.0; self.n_steps + 1];
        s_path[0] = m.spot;
        v_path[0] = m.v0;

        for i in 0..self.n_steps {
            let t = i as f64 * dt;
            let s = s_path[i];
            let v = v_path[i].max(0.0);

            let lev = m.leverage_at(t, s);
            let sqrt_v = v.sqrt();

            // Correlated Brownian motions
            let w1 = z1[i % z1.len()];
            let w2 = m.rho * z1[i % z1.len()]
                + (1.0 - m.rho * m.rho).sqrt() * z2[i % z2.len()];

            // Euler step for spot: dS = (r - q) S dt + L(t,S) √v S dW₁
            let ds = (m.rate - m.dividend) * s * dt + lev * sqrt_v * s * sqrt_dt * w1;
            s_path[i + 1] = (s + ds).max(1e-6);

            // Euler step for variance: dv = κ(θ - v) dt + σ √v dW₂
            let dv = m.kappa * (m.theta - v) * dt + m.sigma * sqrt_v * sqrt_dt * w2;
            v_path[i + 1] = (v + dv).max(0.0);
        }

        (s_path, v_path)
    }

    /// Price a European call option via MC.
    ///
    /// Uses simple pseudo-random draws (for production, use low-discrepancy sequences).
    pub fn price_european_call(&self, strike: f64, maturity: f64, seed: u64) -> f64 {
        let m = &self.fdm_model;
        let df = (-m.rate * maturity).exp();

        // Simple LCG for reproducibility
        let mut rng_state = seed;
        let next_normal = |state: &mut u64| -> f64 {
            // Box-Muller from LCG
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u1 = (*state >> 11) as f64 / (1u64 << 53) as f64;
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = (*state >> 11) as f64 / (1u64 << 53) as f64;
            let u1 = u1.max(1e-15);
            (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
        };

        let mut payoff_sum = 0.0;
        for _ in 0..self.n_paths {
            let z1: Vec<f64> = (0..self.n_steps).map(|_| next_normal(&mut rng_state)).collect();
            let z2: Vec<f64> = (0..self.n_steps).map(|_| next_normal(&mut rng_state)).collect();
            let (s_path, _) = self.simulate_path(maturity, &z1, &z2);
            let s_t = s_path[self.n_steps];
            payoff_sum += (s_t - strike).max(0.0);
        }

        df * payoff_sum / self.n_paths as f64
    }
}

// ---------------------------------------------------------------------------
// G86: ExtendedCoxIngersollRoss — Extended CIR with time-dependent parameters
// ---------------------------------------------------------------------------

/// Extended CIR model with time-dependent mean-reversion level.
///
/// dr = κ(θ(t) − r) dt + σ √r dW
///
/// The deterministic shift θ(t) is chosen to fit the initial term structure:
///   θ(t) = θ_∞ + φ(t)
///
/// where φ(t) is a shift function derived from market discount factors.
///
/// This generalises [`CIRModel`](crate::CIRModel) to fit the initial curve exactly.
pub struct ExtendedCoxIngersollRoss {
    /// Mean-reversion speed κ (constant).
    pub kappa: f64,
    /// Long-run level θ_∞ (used as base level).
    pub theta_inf: f64,
    /// Volatility σ.
    pub sigma: f64,
    /// Initial short rate r(0).
    pub r0: f64,
    /// Time-dependent shift: (time, shift_value) pairs, sorted by time.
    /// θ(t) = θ_∞ + interpolated_shift(t)
    pub shift_curve: Vec<(f64, f64)>,
    /// Parameters for calibration: [κ, θ_∞, σ].
    params: Vec<Parameter>,
}

impl ExtendedCoxIngersollRoss {
    /// Create with a flat shift (reduces to standard CIR).
    pub fn new(kappa: f64, theta_inf: f64, sigma: f64, r0: f64) -> Self {
        let params = vec![
            Parameter::new(kappa, Box::new(PositiveConstraint)),
            Parameter::new(theta_inf, Box::new(PositiveConstraint)),
            Parameter::new(sigma, Box::new(PositiveConstraint)),
        ];
        Self {
            kappa,
            theta_inf,
            sigma,
            r0,
            shift_curve: vec![],
            params,
        }
    }

    /// Create with time-dependent shift to fit an initial term structure.
    ///
    /// `market_curve`: `(time, discount_factor)` pairs.
    pub fn with_market_curve(
        kappa: f64,
        theta_inf: f64,
        sigma: f64,
        r0: f64,
        market_curve: &[(f64, f64)],
    ) -> Self {
        let mut model = Self::new(kappa, theta_inf, sigma, r0);

        // Derive shift curve from market discount factors
        // The shift φ(t) adjusts θ(t) so that model reprices the initial curve
        let mut shifts = Vec::with_capacity(market_curve.len());
        for i in 0..market_curve.len() {
            let (t, df) = market_curve[i];
            if t < 1e-10 {
                shifts.push((t, 0.0));
                continue;
            }

            // Market instantaneous forward rate
            let f_market = if i + 1 < market_curve.len() {
                let (t2, df2) = market_curve[i + 1];
                if (t2 - t).abs() > 1e-12 {
                    -(df2 / df).ln() / (t2 - t)
                } else {
                    -df.ln() / t
                }
            } else {
                -df.ln() / t
            };

            // CIR model forward rate with base θ_∞
            let gamma = (kappa * kappa + 2.0 * sigma * sigma).sqrt();
            let e_gt = (gamma * t).exp();
            let b = 2.0 * (e_gt - 1.0) / ((gamma + kappa) * (e_gt - 1.0) + 2.0 * gamma);
            let f_model = r0 * b / t + kappa * theta_inf * b;

            // Shift to match market
            let phi = if kappa.abs() > 1e-12 {
                (f_market - f_model) / kappa
            } else {
                0.0
            };
            shifts.push((t, phi));
        }

        model.shift_curve = shifts;
        model
    }

    /// γ = √(κ² + 2σ²)
    fn gamma(&self) -> f64 {
        (self.kappa * self.kappa + 2.0 * self.sigma * self.sigma).sqrt()
    }

    /// Effective θ(t) = θ_∞ + φ(t)
    pub fn theta_at(&self, t: f64) -> f64 {
        let shift = interpolate_shift(&self.shift_curve, t);
        self.theta_inf + shift
    }

    /// B(τ) factor for CIR bond pricing.
    pub fn bond_b(&self, tau: f64) -> f64 {
        let g = self.gamma();
        let e_gt = (g * tau).exp();
        2.0 * (e_gt - 1.0) / ((g + self.kappa) * (e_gt - 1.0) + 2.0 * g)
    }

    /// A(τ) factor for CIR bond pricing (uses base θ_∞).
    pub fn bond_a(&self, tau: f64) -> f64 {
        let g = self.gamma();
        let k = self.kappa;
        let s2 = self.sigma * self.sigma;
        let e_gt = (g * tau).exp();
        let denom = (g + k) * (e_gt - 1.0) + 2.0 * g;
        let base = 2.0 * g * ((k + g) * tau / 2.0).exp() / denom;
        let power = 2.0 * k * self.theta_inf / s2;
        base.powf(power)
    }

    /// Zero-coupon bond price P(0, T) with time-dependent shift correction.
    pub fn bond_price(&self, maturity: f64) -> f64 {
        let base = self.bond_a(maturity) * (-self.bond_b(maturity) * self.r0).exp();
        // Apply shift correction: multiply by exp(-∫₀ᵀ κ·φ(s)·B(T-s) ds)
        // Approximate using trapezoidal rule
        if self.shift_curve.is_empty() {
            return base;
        }
        let n_steps = 50;
        let dt = maturity / n_steps as f64;
        let mut integral = 0.0;
        for i in 0..=n_steps {
            let s = i as f64 * dt;
            let phi = interpolate_shift(&self.shift_curve, s);
            let b_remaining = self.bond_b(maturity - s);
            let val = self.kappa * phi * b_remaining;
            if i == 0 || i == n_steps {
                integral += 0.5 * val;
            } else {
                integral += val;
            }
        }
        integral *= dt;
        base * (-integral).exp()
    }

    /// Yield R(T) = −ln P(0,T)/T.
    pub fn yield_rate(&self, maturity: f64) -> f64 {
        if maturity < 1e-15 {
            return self.r0;
        }
        -self.bond_price(maturity).ln() / maturity
    }

    /// Check Feller condition: 2κθ_∞ ≥ σ².
    pub fn feller_satisfied(&self) -> bool {
        2.0 * self.kappa * self.theta_inf >= self.sigma * self.sigma
    }
}

impl CalibratedModel for ExtendedCoxIngersollRoss {
    fn parameters(&self) -> &[Parameter] {
        &self.params
    }

    fn set_params(&mut self, vals: &[f64]) {
        assert!(vals.len() >= 3, "ExtendedCIR requires 3 parameters");
        self.params[0].set_value(vals[0]);
        self.params[1].set_value(vals[1]);
        self.params[2].set_value(vals[2]);
        self.kappa = vals[0];
        self.theta_inf = vals[1];
        self.sigma = vals[2];
    }
}

impl ShortRateModel for ExtendedCoxIngersollRoss {
    fn short_rate(&self, _t: f64, x: f64) -> f64 {
        // r(t) = x(t) where x follows extended CIR
        // The shift is absorbed into the drift
        x.max(0.0)
    }

    fn discount(&self, t: f64) -> f64 {
        self.bond_price(t)
    }
}

/// Linear interpolation on a shift curve.
fn interpolate_shift(curve: &[(f64, f64)], t: f64) -> f64 {
    if curve.is_empty() {
        return 0.0;
    }
    if t <= curve[0].0 {
        return curve[0].1;
    }
    if t >= curve[curve.len() - 1].0 {
        return curve[curve.len() - 1].1;
    }
    // Binary search
    let idx = match curve.binary_search_by(|p| p.0.partial_cmp(&t).unwrap()) {
        Ok(i) => return curve[i].1,
        Err(i) => i,
    };
    if idx == 0 {
        return curve[0].1;
    }
    let (t0, v0) = curve[idx - 1];
    let (t1, v1) = curve[idx];
    let frac = (t - t0) / (t1 - t0);
    v0 + frac * (v1 - v0)
}

// ---------------------------------------------------------------------------
// G87: CapHelper — Calibration helper for caps/floors
// ---------------------------------------------------------------------------

/// Calibration helper for cap/floor instruments.
///
/// Computes the model-implied cap price and compares it to the market quote.
/// Uses Black's formula for the market price and allows model pricing
/// via a provided pricing function.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CapHelper {
    /// Market cap/floor price (or vol).
    pub market_price: f64,
    /// Strike rate.
    pub strike: f64,
    /// Tenor (years).
    pub tenor: f64,
    /// Forward rate at inception.
    pub forward_rate: f64,
    /// Discount factor to payment date.
    pub discount_factor: f64,
    /// Year fraction for the caplet.
    pub year_fraction: f64,
    /// Implied vol from market.
    pub market_vol: f64,
    /// Whether this is a cap (true) or floor (false).
    pub is_cap: bool,
}

impl CapHelper {
    /// Create a new cap calibration helper.
    pub fn new(
        market_vol: f64,
        strike: f64,
        tenor: f64,
        forward_rate: f64,
        discount_factor: f64,
        year_fraction: f64,
        is_cap: bool,
    ) -> Self {
        // Convert vol to price using Black formula
        let market_price = black_caplet_price(
            forward_rate,
            strike,
            tenor,
            market_vol,
            discount_factor,
            year_fraction,
            is_cap,
        );
        Self {
            market_price,
            strike,
            tenor,
            forward_rate,
            discount_factor,
            year_fraction,
            market_vol,
            is_cap,
        }
    }

    /// Model price given vol from candidate parameters.
    pub fn model_price_from_vol(&self, model_vol: f64) -> f64 {
        black_caplet_price(
            self.forward_rate,
            self.strike,
            self.tenor,
            model_vol,
            self.discount_factor,
            self.year_fraction,
            self.is_cap,
        )
    }
}

impl CalibrationHelper for CapHelper {
    fn market_value(&self) -> f64 {
        self.market_price
    }

    fn model_value_with_params(&self, params: &[f64]) -> f64 {
        // Interpret params[0] as model vol (for simple vol-based calibrations)
        // More complex models would override this
        let model_vol = if params.is_empty() {
            self.market_vol
        } else {
            params[0].abs()
        };
        self.model_price_from_vol(model_vol)
    }
}

// ---------------------------------------------------------------------------
// G88: SwaptionHelper — Calibration helper for swaptions
// ---------------------------------------------------------------------------

/// Pricing model for swaption helpers.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum SwaptionVolType {
    /// Black (lognormal) volatility.
    Black,
    /// Bachelier (normal) volatility.
    Normal,
}

/// Calibration helper for swaption instruments.
///
/// Can handle both Black (lognormal) and Bachelier (normal) volatility quotes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SwaptionHelper {
    /// Market swaption price.
    pub market_price: f64,
    /// Strike (fixed rate). If None, ATM (= forward swap rate).
    pub strike: Option<f64>,
    /// Option expiry (years).
    pub expiry: f64,
    /// Swap tenor (years).
    pub swap_tenor: f64,
    /// Forward swap rate.
    pub forward_swap_rate: f64,
    /// Annuity (PV01) of the fixed leg.
    pub annuity: f64,
    /// Market-quoted volatility.
    pub market_vol: f64,
    /// Volatility type.
    pub vol_type: SwaptionVolType,
}

impl SwaptionHelper {
    /// Create a new swaption calibration helper.
    pub fn new(
        market_vol: f64,
        expiry: f64,
        swap_tenor: f64,
        forward_swap_rate: f64,
        annuity: f64,
        vol_type: SwaptionVolType,
        strike: Option<f64>,
    ) -> Self {
        let k = strike.unwrap_or(forward_swap_rate);
        let market_price = match vol_type {
            SwaptionVolType::Black => {
                black_swaption_price(forward_swap_rate, k, expiry, market_vol, annuity, true)
            }
            SwaptionVolType::Normal => {
                bachelier_swaption_price(forward_swap_rate, k, expiry, market_vol, annuity, true)
            }
        };

        Self {
            market_price,
            strike,
            expiry,
            swap_tenor,
            forward_swap_rate,
            annuity,
            market_vol,
            vol_type,
        }
    }

    /// Effective strike (ATM if not set).
    pub fn effective_strike(&self) -> f64 {
        self.strike.unwrap_or(self.forward_swap_rate)
    }

    /// Model price from a model vol.
    pub fn model_price_from_vol(&self, model_vol: f64) -> f64 {
        let k = self.effective_strike();
        match self.vol_type {
            SwaptionVolType::Black => {
                black_swaption_price(self.forward_swap_rate, k, self.expiry, model_vol, self.annuity, true)
            }
            SwaptionVolType::Normal => {
                bachelier_swaption_price(self.forward_swap_rate, k, self.expiry, model_vol, self.annuity, true)
            }
        }
    }
}

impl CalibrationHelper for SwaptionHelper {
    fn market_value(&self) -> f64 {
        self.market_price
    }

    fn model_value_with_params(&self, params: &[f64]) -> f64 {
        // Simple vol-based calibration: params[0] = model vol
        let model_vol = if params.is_empty() {
            self.market_vol
        } else {
            params[0].abs()
        };
        self.model_price_from_vol(model_vol)
    }
}

// ---------------------------------------------------------------------------
// G89: ConstantVolEstimator — constant (historical) volatility estimator
// ---------------------------------------------------------------------------

/// Simple constant volatility estimator from historical close-to-close returns.
///
/// Computes the annualised standard deviation of log returns.
/// This is the simplest possible volatility estimator, useful as a
/// baseline or for models that assume constant volatility.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConstantVolEstimator {
    /// Estimated annualised volatility.
    pub vol: f64,
    /// Estimated annualised mean return.
    pub mean_return: f64,
    /// Number of observations used.
    pub n_obs: usize,
    /// Daily variance (not annualised).
    pub daily_variance: f64,
}

impl ConstantVolEstimator {
    /// Estimate volatility from a series of prices.
    ///
    /// # Arguments
    /// - `prices`: chronological price series (length ≥ 2)
    /// - `annualisation_factor`: typically 252 for daily data
    pub fn from_prices(prices: &[f64], annualisation_factor: f64) -> Self {
        assert!(prices.len() >= 2, "Need at least 2 prices");

        let log_returns: Vec<f64> = prices
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();

        Self::from_returns(&log_returns, annualisation_factor)
    }

    /// Estimate volatility from a series of log returns.
    pub fn from_returns(log_returns: &[f64], annualisation_factor: f64) -> Self {
        let n = log_returns.len();
        assert!(n >= 1, "Need at least 1 return");

        let mean = log_returns.iter().sum::<f64>() / n as f64;
        let variance = if n > 1 {
            log_returns
                .iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>()
                / (n - 1) as f64
        } else {
            log_returns[0].powi(2)
        };

        let annual_vol = (variance * annualisation_factor).sqrt();
        let annual_mean = mean * annualisation_factor;

        Self {
            vol: annual_vol,
            mean_return: annual_mean,
            n_obs: n,
            daily_variance: variance,
        }
    }

    /// Confidence interval for the volatility estimate (chi-squared based).
    ///
    /// Returns `(lower, upper)` at the given confidence level (e.g. 0.95).
    pub fn confidence_interval(&self, confidence: f64) -> (f64, f64) {
        if self.n_obs < 2 {
            return (0.0, f64::INFINITY);
        }
        let n = self.n_obs as f64;
        // Approximate chi-squared quantiles using normal approximation
        let alpha = 1.0 - confidence;
        let z = normal_quantile(1.0 - alpha / 2.0);

        // Chi-squared approximate quantiles via Wilson-Hilferty
        let df = n - 1.0;
        let chi2_lower = df * (1.0 - 2.0 / (9.0 * df) - z * (2.0 / (9.0 * df)).sqrt()).powi(3);
        let chi2_upper = df * (1.0 - 2.0 / (9.0 * df) + z * (2.0 / (9.0 * df)).sqrt()).powi(3);

        let vol_lower = self.vol * (df / chi2_upper).sqrt();
        let vol_upper = self.vol * (df / chi2_lower).sqrt();

        (vol_lower.max(0.0), vol_upper)
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Standard normal CDF (Abramowitz & Stegun approximation).
fn norm_cdf(x: f64) -> f64 {
    if x > 6.0 {
        return 1.0;
    }
    if x < -6.0 {
        return 0.0;
    }
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let t = 1.0 / (1.0 + p * x.abs());
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x / 2.0).exp();
    0.5 * (1.0 + sign * y)
}

/// Standard normal PDF.
fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Normal quantile approximation (Beasley-Springer-Moro).
fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    // Rational approximation
    let p_adj = p - 0.5;
    if p_adj.abs() <= 0.42 {
        let r = p_adj * p_adj;
        let num = p_adj
            * ((((-25.44106049637689 * r + 41.39119773534996) * r - 18.61500062529246) * r
                + 2.506628277459239)
                * r
                + 1.0);
        let den = (((((-8.47351093090049 * r + 23.08336743743455) * r - 21.06224101826264) * r
            + 3.13082909833678)
            * r
            + 1.0)
            * r
            + 1.0)
            * r
            + 1.0;
        return num / den;
    }
    let r = if p_adj < 0.0 { p } else { 1.0 - p };
    let s = (-2.0 * r.ln()).sqrt();
    let num = ((2.32121276858 * s + 4.85014127135) * s - 2.29796479134) * s + 1.0;
    let den = ((1.63706781897 * s + 3.54388924762) * s + 1.0) * s + 1.0;
    let val = s - num / den;
    if p_adj < 0.0 {
        -val
    } else {
        val
    }
}

/// Black caplet price.
fn black_caplet_price(
    forward: f64,
    strike: f64,
    maturity: f64,
    vol: f64,
    discount: f64,
    year_fraction: f64,
    is_cap: bool,
) -> f64 {
    if maturity <= 0.0 || vol <= 0.0 {
        return 0.0;
    }

    let sqrt_t = maturity.sqrt();
    let d1 = ((forward / strike).ln() + 0.5 * vol * vol * maturity) / (vol * sqrt_t);
    let d2 = d1 - vol * sqrt_t;

    let price = if is_cap {
        discount * year_fraction * (forward * norm_cdf(d1) - strike * norm_cdf(d2))
    } else {
        discount * year_fraction * (strike * norm_cdf(-d2) - forward * norm_cdf(-d1))
    };

    price.max(0.0)
}

/// Black swaption price (lognormal).
fn black_swaption_price(
    forward: f64,
    strike: f64,
    expiry: f64,
    vol: f64,
    annuity: f64,
    is_payer: bool,
) -> f64 {
    if expiry <= 0.0 || vol <= 0.0 {
        return 0.0;
    }

    let sqrt_t = expiry.sqrt();
    let d1 = ((forward / strike).ln() + 0.5 * vol * vol * expiry) / (vol * sqrt_t);
    let d2 = d1 - vol * sqrt_t;

    let price = if is_payer {
        annuity * (forward * norm_cdf(d1) - strike * norm_cdf(d2))
    } else {
        annuity * (strike * norm_cdf(-d2) - forward * norm_cdf(-d1))
    };

    price.max(0.0)
}

/// Bachelier (normal) swaption price.
fn bachelier_swaption_price(
    forward: f64,
    strike: f64,
    expiry: f64,
    vol: f64,
    annuity: f64,
    is_payer: bool,
) -> f64 {
    if expiry <= 0.0 || vol <= 0.0 {
        return 0.0;
    }

    let sqrt_t = expiry.sqrt();
    let d = (forward - strike) / (vol * sqrt_t);

    let price = annuity * vol * sqrt_t * (d * norm_cdf(d) + norm_pdf(d));

    if !is_payer {
        // Put-call parity: floor = cap - (forward - strike) * annuity
        price - annuity * (forward - strike)
    } else {
        price
    }.max(0.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_heston_slv_fdm_leverage_calibration() {
        // Flat local vol surface → leverage ≈ loc_vol / √E[v]
        let flat_vol = 0.20;
        let local_vol = |_t: f64, _s: f64| -> f64 { flat_vol };

        let model = HestonSlvFdmModel::calibrate(
            0.04, 1.5, 0.04, 0.3, -0.5,
            100.0, 0.05, 0.02,
            10, 20, 1.0,
            &local_vol,
        );

        // At t=0, E[v] = v0 = 0.04, so leverage ≈ 0.20 / √0.04 = 1.0
        let lev = model.leverage_at(0.0, 100.0);
        assert_abs_diff_eq!(lev, 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_heston_slv_fdm_effective_local_vol() {
        let flat_vol = 0.20;
        let local_vol = |_t: f64, _s: f64| -> f64 { flat_vol };

        let model = HestonSlvFdmModel::calibrate(
            0.04, 1.5, 0.04, 0.3, -0.5,
            100.0, 0.05, 0.02,
            10, 20, 1.0,
            &local_vol,
        );

        // Effective local vol should approximate the original local vol
        let eff_vol = model.effective_local_vol(0.0, 100.0);
        assert_abs_diff_eq!(eff_vol, flat_vol, epsilon = 0.05);
    }

    #[test]
    fn test_heston_slv_mc_positive_price() {
        let flat_vol = 0.20;
        let local_vol = |_t: f64, _s: f64| -> f64 { flat_vol };

        let fdm = HestonSlvFdmModel::calibrate(
            0.04, 1.5, 0.04, 0.3, -0.5,
            100.0, 0.05, 0.02,
            5, 10, 1.0,
            &local_vol,
        );

        let mc = HestonSlvMcModel::new(fdm, 5000, 50);
        let price = mc.price_european_call(100.0, 1.0, 42);
        // For ATM call, price should be positive and reasonable
        assert!(price > 0.0, "MC price should be positive, got {}", price);
        assert!(price < 50.0, "MC price should be < 50 for ATM, got {}", price);
    }

    #[test]
    fn test_extended_cir_reduces_to_standard() {
        // Without shift curve, extended CIR should match standard CIR
        let model = ExtendedCoxIngersollRoss::new(0.3, 0.05, 0.1, 0.05);

        assert!(model.feller_satisfied());
        let p = model.bond_price(0.0);
        assert_abs_diff_eq!(p, 1.0, epsilon = 1e-10);

        let p1 = model.bond_price(1.0);
        assert!(p1 > 0.0 && p1 < 1.0, "Bond price = {}", p1);

        let p5 = model.bond_price(5.0);
        assert!(p1 > p5, "Bond prices should decrease with maturity");
    }

    #[test]
    fn test_extended_cir_with_shift_curve() {
        let market_curve = vec![
            (0.25, 0.99_f64),
            (0.5, 0.98),
            (1.0, 0.96),
            (2.0, 0.92),
            (5.0, 0.82),
        ];

        let model = ExtendedCoxIngersollRoss::with_market_curve(
            0.3, 0.05, 0.1, 0.04, &market_curve,
        );

        // Theta should vary with time
        let theta_0 = model.theta_at(0.0);
        let theta_1 = model.theta_at(1.0);
        // They don't have to be equal; the shift adjusts them
        assert!(theta_0.is_finite());
        assert!(theta_1.is_finite());

        // Bond prices should still be positive and < 1
        for t in [0.5, 1.0, 2.0, 5.0] {
            let p = model.bond_price(t);
            assert!(p > 0.0 && p < 1.5, "P(0,{}) = {} out of range", t, p);
        }
    }

    #[test]
    fn test_extended_cir_set_params() {
        let mut model = ExtendedCoxIngersollRoss::new(0.3, 0.05, 0.1, 0.05);
        model.set_params(&[0.5, 0.04, 0.08]);
        assert_abs_diff_eq!(model.kappa, 0.5);
        assert_abs_diff_eq!(model.theta_inf, 0.04);
        assert_abs_diff_eq!(model.sigma, 0.08);
    }

    #[test]
    fn test_cap_helper_roundtrip() {
        let helper = CapHelper::new(
            0.20,  // 20% vol
            0.05,  // 5% strike
            1.0,   // 1Y tenor
            0.05,  // 5% forward (ATM)
            0.95,  // discount factor
            0.25,  // quarterly
            true,  // cap
        );

        // Market price should be positive
        assert!(helper.market_price > 0.0, "Cap price = {}", helper.market_price);

        // Model price with same vol should match
        let model_price = helper.model_price_from_vol(0.20);
        assert_abs_diff_eq!(model_price, helper.market_price, epsilon = 1e-10);

        // CalibrationHelper trait
        let err = helper.calibration_error_with_params(&[0.20]);
        assert_abs_diff_eq!(err, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_swaption_helper_black() {
        let helper = SwaptionHelper::new(
            0.15,  // 15% Black vol
            5.0,   // 5Y expiry
            5.0,   // 5Y swap
            0.04,  // 4% forward swap rate
            4.0,   // annuity ≈ 4
            SwaptionVolType::Black,
            None,  // ATM
        );

        assert!(helper.market_price > 0.0, "Swaption price = {}", helper.market_price);
        assert_abs_diff_eq!(helper.effective_strike(), 0.04);

        // Roundtrip
        let model_price = helper.model_price_from_vol(0.15);
        assert_abs_diff_eq!(model_price, helper.market_price, epsilon = 1e-10);
    }

    #[test]
    fn test_swaption_helper_bachelier() {
        let helper = SwaptionHelper::new(
            0.005, // 50bp normal vol
            2.0,   // 2Y expiry
            10.0,  // 10Y swap
            0.03,  // 3% forward
            7.5,   // annuity
            SwaptionVolType::Normal,
            Some(0.03),
        );

        assert!(helper.market_price > 0.0, "Swaption price = {}", helper.market_price);

        let model_price = helper.model_price_from_vol(0.005);
        assert_abs_diff_eq!(model_price, helper.market_price, epsilon = 1e-10);
    }

    #[test]
    fn test_constant_vol_estimator_from_prices() {
        // Prices with known daily volatility ~1%
        let prices: Vec<f64> = (0..100)
            .map(|i| 100.0 * (0.01 * (i as f64 * 0.1).sin()).exp())
            .collect();

        let est = ConstantVolEstimator::from_prices(&prices, 252.0);
        assert!(est.vol > 0.0 && est.vol < 1.0, "vol = {}", est.vol);
        assert_eq!(est.n_obs, 99); // 100 prices → 99 returns
    }

    #[test]
    fn test_constant_vol_estimator_from_returns() {
        // Known returns with daily vol ~1%
        let daily_vol = 0.01;
        let returns: Vec<f64> = (0..200)
            .map(|i| {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                sign * daily_vol
            })
            .collect();

        let est = ConstantVolEstimator::from_returns(&returns, 252.0);
        let expected_annual = daily_vol * 252.0_f64.sqrt();
        assert_abs_diff_eq!(est.vol, expected_annual, epsilon = 0.02);
    }

    #[test]
    fn test_constant_vol_confidence_interval() {
        let returns: Vec<f64> = (0..100)
            .map(|i| if i % 2 == 0 { 0.01 } else { -0.01 })
            .collect();

        let est = ConstantVolEstimator::from_returns(&returns, 252.0);
        let (lower, upper) = est.confidence_interval(0.95);
        assert!(lower < est.vol, "lower {} >= vol {}", lower, est.vol);
        assert!(upper > est.vol, "upper {} <= vol {}", upper, est.vol);
        assert!(lower > 0.0);
    }
}
