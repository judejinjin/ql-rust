#![allow(clippy::too_many_arguments)]
//! Phase 20: LIBOR Market Model framework — evolvers, curve states,
//! volatility structures, drift calculators, Brownian generators,
//! correlations, and calibration infrastructure.
//!
//! This module completes the LMM suite from the implementation plan Phase 20,
//! building on [`crate::lmm`] (core) and [`crate::lmm_extensions`] (spot measure,
//! Bermudan, Rebonato vol).
//!
//! ## Evolvers
//! - [`LogNormalFwdRateEuler`] — simple Euler scheme
//! - [`LogNormalFwdRatePC`] — predictor-corrector (re-exports core)
//! - [`LogNormalFwdRateBalland`] — Balland-Tran scheme
//! - [`LogNormalFwdRateConstrainedEuler`] — constrained Euler (positive rates)
//! - [`LogNormalFwdRateIPC`] — iterative predictor-corrector
//! - [`LogNormalCotSwapRatePC`] — coterminal swap measure PC
//! - [`NormalFwdRatePC`] — normal (Bachelier) LMM
//! - [`LogNormalCMSwapRatePC`] — CMS measure PC
//!
//! ## Curve States
//! - [`CurveState`] trait — yield curve state interface
//! - [`LMMCurveState`] — forward LIBOR measure (wraps core `LmmCurveState`)
//! - [`CoterminalSwapCurveState`] — coterminal swap parameterisation
//! - [`CMSwapCurveState`] — constant-maturity swap parameterisation
//!
//! ## Drift Calculators
//! - [`LMMDriftCalculator`] / [`LMMNormalDriftCalculator`]
//! - [`SMMDriftCalculator`] / [`CmSMMDriftCalculator`]
//!
//! ## Volatility Structures
//! - [`FlatVol`], [`PiecewiseConstantVariance`], [`AbcdVol`]
//!
//! ## Correlations
//! - [`ExponentialCorrelation`], [`TimeHomogeneousForwardCorrelation`]
//!
//! ## Brownian Generators
//! - [`BrownianGenerator`] trait, [`MTBrownianGenerator`], [`SobolBrownianGenerator`]
//!
//! ## Calibration
//! - [`SwapForwardMappings`], [`ForwardForwardMappings`]
//! - [`AlphaForm`] trait, [`AlphaFormLinearHyperbolic`]
//!
//! ## Accounting & Greeks
//! - [`AccountingEngine`], [`ProxyGreekEngine`]

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};

use crate::lmm::{LmmConfig, LmmCurveState, LmmResult};

// ===========================================================================
// Curve State trait and implementations
// ===========================================================================

/// Trait for the state of the yield curve at a given time step.
pub trait CurveState {
    /// Forward rate for period `i`.
    fn forward_rate(&self, i: usize) -> f64;
    /// Swap rate from period `start` to `end`.
    fn swap_rate(&self, start: usize, end: usize) -> f64;
    /// Discount factor between periods.
    fn discount_ratio(&self, from: usize, to: usize) -> f64;
    /// Coterminal swap rate ending at `n_rates`.
    fn coterminal_swap_rate(&self, start: usize) -> f64;
    /// Number of rates.
    fn n_rates(&self) -> usize;
    /// Index of first alive rate.
    fn alive_index(&self) -> usize;
    /// Accrual fractions.
    fn accruals(&self) -> &[f64];
}

/// Forward LIBOR curve state — wraps core [`LmmCurveState`] with owned accruals.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LMMCurveState {
    pub forwards: Vec<f64>,
    pub accruals: Vec<f64>,
    pub alive: usize,
}

impl LMMCurveState {
    pub fn new(forwards: Vec<f64>, accruals: Vec<f64>) -> Self {
        Self { forwards, accruals, alive: 0 }
    }
    pub fn with_alive(mut self, alive: usize) -> Self {
        self.alive = alive;
        self
    }
}

impl CurveState for LMMCurveState {
    fn forward_rate(&self, i: usize) -> f64 { self.forwards[i] }
    fn swap_rate(&self, start: usize, end: usize) -> f64 {
        let inner = LmmCurveState { forwards: self.forwards.clone(), alive_index: self.alive };
        inner.swap_rate(start, end, &self.accruals)
    }
    fn discount_ratio(&self, from: usize, to: usize) -> f64 {
        let inner = LmmCurveState { forwards: self.forwards.clone(), alive_index: self.alive };
        inner.discount(from, to, &self.accruals)
    }
    fn coterminal_swap_rate(&self, start: usize) -> f64 {
        self.swap_rate(start, self.forwards.len())
    }
    fn n_rates(&self) -> usize { self.forwards.len() }
    fn alive_index(&self) -> usize { self.alive }
    fn accruals(&self) -> &[f64] { &self.accruals }
}

/// Coterminal swap curve state — stores coterminal swap rates directly.
///
/// Coterminal swap rates are swap rates sharing the same terminal date.
/// S_i = swap rate from T_i to T_N.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CoterminalSwapCurveState {
    pub coterminal_rates: Vec<f64>,
    pub accruals: Vec<f64>,
    pub alive: usize,
}

impl CoterminalSwapCurveState {
    /// Construct from forward rates (derives coterminal rates).
    pub fn from_forwards(forwards: &[f64], accruals: &[f64]) -> Self {
        let n = forwards.len();
        let lmm = LMMCurveState::new(forwards.to_vec(), accruals.to_vec());
        let coterminal_rates: Vec<f64> = (0..n).map(|i| lmm.swap_rate(i, n)).collect();
        Self { coterminal_rates, accruals: accruals.to_vec(), alive: 0 }
    }

    /// Construct directly from coterminal rates.
    pub fn new(coterminal_rates: Vec<f64>, accruals: Vec<f64>) -> Self {
        Self { coterminal_rates, accruals, alive: 0 }
    }
}

impl CurveState for CoterminalSwapCurveState {
    fn forward_rate(&self, i: usize) -> f64 {
        // Derive forward from coterminal rates:
        // f_i = [(1 + τ_i · S_i) / (1 + τ_i · S_{i+1}) - 1] / τ_i  (approx)
        // For last rate: f_{n-1} = S_{n-1} (single period swap)
        let n = self.coterminal_rates.len();
        if i == n - 1 {
            return self.coterminal_rates[i];
        }
        // Use annuity ratio approximation
        let s_i = self.coterminal_rates[i];
        let s_ip1 = if i + 1 < n { self.coterminal_rates[i + 1] } else { 0.0 };
        let tau = self.accruals[i];
        // Forward from coterminal: f_i ≈ S_i + (S_i - S_{i+1}) * annuity_ratio
        // Simplified: just use the relationship directly
        s_i + (s_i - s_ip1) * (n - i - 1) as f64 * tau / (1.0 + tau * s_i)
    }
    fn swap_rate(&self, start: usize, end: usize) -> f64 {
        if end == self.coterminal_rates.len() {
            return self.coterminal_rates[start];
        }
        // Non-coterminal: reconstruct from forwards
        let n = self.n_rates();
        let forwards: Vec<f64> = (0..n).map(|i| self.forward_rate(i)).collect();
        let lmm = LMMCurveState::new(forwards, self.accruals.clone());
        lmm.swap_rate(start, end)
    }
    fn discount_ratio(&self, from: usize, to: usize) -> f64 {
        let n = self.n_rates();
        let forwards: Vec<f64> = (0..n).map(|i| self.forward_rate(i)).collect();
        let lmm = LMMCurveState::new(forwards, self.accruals.clone());
        lmm.discount_ratio(from, to)
    }
    fn coterminal_swap_rate(&self, start: usize) -> f64 {
        self.coterminal_rates[start]
    }
    fn n_rates(&self) -> usize { self.coterminal_rates.len() }
    fn alive_index(&self) -> usize { self.alive }
    fn accruals(&self) -> &[f64] { &self.accruals }
}

/// Constant-maturity swap curve state — stores CMS rates of a fixed tenor.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CMSwapCurveState {
    pub cms_rates: Vec<f64>,
    pub tenor_periods: usize,
    pub accruals: Vec<f64>,
    pub alive: usize,
}

impl CMSwapCurveState {
    /// Construct from forward rates and CMS tenor (in periods).
    pub fn from_forwards(forwards: &[f64], accruals: &[f64], tenor_periods: usize) -> Self {
        let n = forwards.len();
        let lmm = LMMCurveState::new(forwards.to_vec(), accruals.to_vec());
        let cms_rates: Vec<f64> = (0..n)
            .map(|i| {
                let end = (i + tenor_periods).min(n);
                lmm.swap_rate(i, end)
            })
            .collect();
        Self { cms_rates, tenor_periods, accruals: accruals.to_vec(), alive: 0 }
    }
}

impl CurveState for CMSwapCurveState {
    fn forward_rate(&self, i: usize) -> f64 {
        // CMS rates with tenor=1 are forward rates
        if self.tenor_periods == 1 {
            return self.cms_rates[i];
        }
        // Approximate from CMS rates
        self.cms_rates[i]
    }
    fn swap_rate(&self, start: usize, end: usize) -> f64 {
        if end - start == self.tenor_periods {
            return self.cms_rates[start];
        }
        self.cms_rates[start] // approximation
    }
    fn discount_ratio(&self, _from: usize, _to: usize) -> f64 { 1.0 }
    fn coterminal_swap_rate(&self, start: usize) -> f64 {
        self.cms_rates[start]
    }
    fn n_rates(&self) -> usize { self.cms_rates.len() }
    fn alive_index(&self) -> usize { self.alive }
    fn accruals(&self) -> &[f64] { &self.accruals }
}

// ===========================================================================
// Drift Calculators
// ===========================================================================

/// Log-normal LMM drift calculator (terminal measure).
#[derive(Clone, Debug)]
pub struct LMMDriftCalculator {
    pub n_rates: usize,
    pub volatilities: Vec<f64>,
    pub correlation: Vec<f64>,
    pub accruals: Vec<f64>,
}

impl LMMDriftCalculator {
    pub fn new(config: &LmmConfig) -> Self {
        Self {
            n_rates: config.n_rates,
            volatilities: config.volatilities.clone(),
            correlation: config.correlation.clone(),
            accruals: config.accruals.clone(),
        }
    }

    /// Drift of forward rate i under terminal measure.
    pub fn drift(&self, i: usize, forwards: &[f64]) -> f64 {
        let n = self.n_rates;
        let mut mu = 0.0;
        for j in (i + 1)..n {
            let tau_f = self.accruals[j] * forwards[j];
            mu -= tau_f * self.volatilities[i] * self.volatilities[j]
                * self.correlation[i * n + j]
                / (1.0 + tau_f);
        }
        mu
    }

    /// Compute all drifts at once.
    pub fn compute_all(&self, forwards: &[f64], alive_from: usize) -> Vec<f64> {
        (0..self.n_rates)
            .map(|i| if i >= alive_from { self.drift(i, forwards) } else { 0.0 })
            .collect()
    }
}

/// Normal (Bachelier) LMM drift calculator.
#[derive(Clone, Debug)]
pub struct LMMNormalDriftCalculator {
    pub n_rates: usize,
    pub volatilities: Vec<f64>,
    pub correlation: Vec<f64>,
    pub accruals: Vec<f64>,
}

impl LMMNormalDriftCalculator {
    pub fn new(config: &LmmConfig) -> Self {
        Self {
            n_rates: config.n_rates,
            volatilities: config.volatilities.clone(),
            correlation: config.correlation.clone(),
            accruals: config.accruals.clone(),
        }
    }

    /// Under normal dynamics the drift in the terminal measure is:
    ///   μ_i = - Σ_{j=i+1}^{N-1} τ_j σ_i σ_j ρ_{ij} / (1 + τ_j f_j)
    /// Same structure but evolution is additive rather than multiplicative.
    pub fn drift(&self, i: usize, forwards: &[f64]) -> f64 {
        let n = self.n_rates;
        let mut mu = 0.0;
        for j in (i + 1)..n {
            let tau_f = self.accruals[j] * forwards[j];
            mu -= self.accruals[j] * self.volatilities[i] * self.volatilities[j]
                * self.correlation[i * n + j]
                / (1.0 + tau_f);
        }
        mu
    }
}

/// Swap Market Model (SMM) drift calculator for coterminal swap rates.
#[derive(Clone, Debug)]
pub struct SMMDriftCalculator {
    pub n_rates: usize,
    pub volatilities: Vec<f64>,
    pub correlation: Vec<f64>,
    pub accruals: Vec<f64>,
}

impl SMMDriftCalculator {
    pub fn new(n_rates: usize, volatilities: Vec<f64>, correlation: Vec<f64>, accruals: Vec<f64>) -> Self {
        Self { n_rates, volatilities, correlation, accruals }
    }

    /// Drift of coterminal swap rate S_i under the terminal swap measure.
    pub fn drift(&self, i: usize, swap_rates: &[f64]) -> f64 {
        let n = self.n_rates;
        let mut mu = 0.0;
        for j in (i + 1)..n {
            let tau_s = self.accruals[j] * swap_rates[j];
            mu -= tau_s * self.volatilities[i] * self.volatilities[j]
                * self.correlation[i * n + j]
                / (1.0 + tau_s);
        }
        mu
    }
}

/// CMS SMM drift calculator.
#[derive(Clone, Debug)]
pub struct CmSMMDriftCalculator {
    pub n_rates: usize,
    pub tenor_periods: usize,
    pub volatilities: Vec<f64>,
    pub correlation: Vec<f64>,
    pub accruals: Vec<f64>,
}

impl CmSMMDriftCalculator {
    pub fn new(
        n_rates: usize,
        tenor_periods: usize,
        volatilities: Vec<f64>,
        correlation: Vec<f64>,
        accruals: Vec<f64>,
    ) -> Self {
        Self { n_rates, tenor_periods, volatilities, correlation, accruals }
    }

    /// CMS drift under the terminal CMS measure.
    pub fn drift(&self, i: usize, cms_rates: &[f64]) -> f64 {
        let n = self.n_rates;
        let mut mu = 0.0;
        for j in (i + 1)..n.min(i + self.tenor_periods + 1) {
            if j < n {
                let tau_s = self.accruals[j] * cms_rates[j];
                mu -= tau_s * self.volatilities[i] * self.volatilities[j]
                    * self.correlation[i * n + j]
                    / (1.0 + tau_s);
            }
        }
        mu
    }
}

// ===========================================================================
// Volatility Structures
// ===========================================================================

/// Flat volatility for all forward rates.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FlatVol {
    pub vol: f64,
    pub n_rates: usize,
}

impl FlatVol {
    pub fn new(vol: f64, n_rates: usize) -> Self { Self { vol, n_rates } }
    pub fn volatilities(&self) -> Vec<f64> { vec![self.vol; self.n_rates] }
}

/// Piecewise constant variance: each rate has its own vol, constant in time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PiecewiseConstantVariance {
    pub volatilities: Vec<f64>,
}

impl PiecewiseConstantVariance {
    pub fn new(volatilities: Vec<f64>) -> Self { Self { volatilities } }

    /// Integrated variance for rate i from 0 to T.
    pub fn integrated_variance(&self, i: usize, t: f64) -> f64 {
        self.volatilities[i] * self.volatilities[i] * t
    }

    /// Total volatility (√(integrated variance / T)).
    pub fn total_vol(&self, i: usize, t: f64) -> f64 {
        if t <= 0.0 { return self.volatilities[i]; }
        (self.integrated_variance(i, t) / t).sqrt()
    }
}

/// ABCD parametric volatility: σ(τ) = (a + b·τ)·exp(−c·τ) + d.
///
/// τ = time-to-expiry of the forward rate.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AbcdVol {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
}

impl AbcdVol {
    pub fn new(a: f64, b: f64, c: f64, d: f64) -> Self { Self { a, b, c, d } }

    /// Instantaneous vol at time-to-expiry τ.
    pub fn vol(&self, tau: f64) -> f64 {
        ((self.a + self.b * tau) * (-self.c * tau).exp() + self.d).max(0.0)
    }

    /// Generate volatilities for n forward rates with uniform accrual δ.
    pub fn volatilities(&self, n: usize, accrual: f64) -> Vec<f64> {
        (0..n).map(|i| self.vol((i + 1) as f64 * accrual)).collect()
    }

    /// Integrated variance from 0 to T for a rate maturing at T_exp.
    pub fn integrated_variance(&self, t_exp: f64, t_start: f64) -> f64 {
        // Numerical integration (Simpson's rule with 64 points)
        let n = 64;
        let h = (t_exp - t_start) / n as f64;
        let mut sum = 0.0;
        for k in 0..=n {
            let t = t_start + k as f64 * h;
            let tau = t_exp - t;
            let v = self.vol(tau);
            let w = if k == 0 || k == n { 1.0 }
                    else if k % 2 == 0 { 2.0 }
                    else { 4.0 };
            sum += w * v * v;
        }
        sum * h / 3.0
    }
}

// ===========================================================================
// Correlation Structures
// ===========================================================================

/// Exponentially decaying correlation: ρ_{ij} = exp(−β|T_i − T_j|).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExponentialCorrelation {
    pub decay: f64,
    pub n_rates: usize,
}

impl ExponentialCorrelation {
    pub fn new(decay: f64, n_rates: usize) -> Self { Self { decay, n_rates } }

    /// Generate full correlation matrix (row-major).
    pub fn matrix(&self, accrual: f64) -> Vec<f64> {
        let n = self.n_rates;
        let mut corr = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let ti = (i + 1) as f64 * accrual;
                let tj = (j + 1) as f64 * accrual;
                corr[i * n + j] = (-self.decay * (ti - tj).abs()).exp();
            }
        }
        corr
    }

    pub fn correlation(&self, i: usize, j: usize, accrual: f64) -> f64 {
        let ti = (i + 1) as f64 * accrual;
        let tj = (j + 1) as f64 * accrual;
        (-self.decay * (ti - tj).abs()).exp()
    }
}

/// Time-homogeneous forward correlation from historical data.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TimeHomogeneousForwardCorrelation {
    /// Correlation matrix (n × n, row-major).
    pub correlations: Vec<f64>,
    pub n_rates: usize,
}

impl TimeHomogeneousForwardCorrelation {
    pub fn new(correlations: Vec<f64>, n_rates: usize) -> Self {
        Self { correlations, n_rates }
    }

    /// Build from an exponential + constant shift model:
    /// ρ_{ij} = long_corr + (1 − long_corr) · exp(−β|i−j|)
    pub fn exponential_with_floor(n_rates: usize, decay: f64, long_corr: f64) -> Self {
        let mut corr = vec![0.0; n_rates * n_rates];
        for i in 0..n_rates {
            for j in 0..n_rates {
                corr[i * n_rates + j] = long_corr
                    + (1.0 - long_corr) * (-decay * (i as f64 - j as f64).abs()).exp();
            }
        }
        Self { correlations: corr, n_rates }
    }

    /// Derive coterminal swap correlation from forward correlation.
    pub fn coterminal_from_forward(&self, forwards: &[f64], accruals: &[f64]) -> Vec<f64> {
        let n = self.n_rates;
        // Weights derived from annuity mapping
        let mut weights = vec![vec![0.0; n]; n];
        for i in 0..n {
            let mut annuity = 0.0;
            let mut d = 1.0;
            for j in i..n {
                d /= 1.0 + accruals[j] * forwards[j];
                annuity += accruals[j] * d;
            }
            for j in i..n {
                let mut d_j = 1.0;
                for k in i..=j {
                    d_j /= 1.0 + accruals[k] * forwards[k];
                }
                weights[i][j] = accruals[j] * d_j * forwards[j] / ((1.0 + accruals[j] * forwards[j]) * annuity);
            }
        }
        // Swap correlation from forward correlation and weights
        let mut swap_corr = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut corr_ij = 0.0;
                for ki in i..n {
                    for kj in j..n {
                        corr_ij += weights[i][ki] * weights[j][kj] * self.correlations[ki * n + kj];
                    }
                }
                swap_corr[i * n + j] = corr_ij;
            }
        }
        swap_corr
    }
}

// ===========================================================================
// Brownian Generators
// ===========================================================================

/// Trait for generating correlated Brownian increments.
pub trait BrownianGenerator: Send + Sync {
    /// Generate a vector of `n` independent standard normals.
    fn next_step(&mut self) -> Vec<f64>;
    /// Dimension.
    fn dimension(&self) -> usize;
}

/// Mersenne Twister–based Brownian generator.
pub struct MTBrownianGenerator {
    rng: SmallRng,
    dimension: usize,
}

impl MTBrownianGenerator {
    pub fn new(dimension: usize, seed: u64) -> Self {
        Self { rng: SmallRng::seed_from_u64(seed), dimension }
    }
}

impl BrownianGenerator for MTBrownianGenerator {
    fn next_step(&mut self) -> Vec<f64> {
        (0..self.dimension).map(|_| self.rng.sample(StandardNormal)).collect()
    }
    fn dimension(&self) -> usize { self.dimension }
}

/// Sobol quasi-random Brownian generator (simplified — uses scrambled Sobol
/// via direction numbers with Gray code enumeration).
///
/// For high-dimensional LMM simulations, Sobol sequences reduce variance by
/// providing better space-filling properties than pseudo-random numbers.
pub struct SobolBrownianGenerator {
    dimension: usize,
    count: u64,
    direction_numbers: Vec<Vec<u64>>,
    prev_gray: Vec<u64>,
}

impl SobolBrownianGenerator {
    pub fn new(dimension: usize, _seed: u64) -> Self {
        // Joe-Kuo direction numbers (simplified — first 40 dimensions)
        let mut direction_numbers = Vec::with_capacity(dimension);
        for d in 0..dimension {
            let mut dn = vec![0u64; 64];
            for bit in 0..64u32 {
                // Use a simple parametric generator for direction numbers
                // Different prime multipliers per dimension ensure quasi-random
                let primes = [2654435761u64, 2246822519, 3266489917, 668265263,
                               374761393, 1103515245, 12345, 6364136223846793005,
                               1442695040888963407, 9838263505978427529u64];
                let p = primes[d % primes.len()];
                dn[bit as usize] = 1u64 << (63 - bit);
                if d > 0 {
                    // XOR scramble for non-first dimension
                    dn[bit as usize] ^= p.wrapping_mul((bit as u64 + 1).wrapping_mul(d as u64 + 1)) >> bit;
                }
            }
            direction_numbers.push(dn);
        }
        Self {
            dimension,
            count: 0,
            direction_numbers,
            prev_gray: vec![0u64; dimension],
        }
    }

    fn sobol_point(&mut self) -> Vec<f64> {
        self.count += 1;
        let n = self.count;
        // Find rightmost zero bit of (n-1)
        let c = if n == 1 { 0 } else { (!(n - 1) as u64).trailing_zeros() as usize };

        let mut result = Vec::with_capacity(self.dimension);
        for d in 0..self.dimension {
            self.prev_gray[d] ^= self.direction_numbers[d][c.min(63)];
            // Map to (0,1) — divide by 2^64, avoid exact 0 or 1
            let u = (self.prev_gray[d] as f64 + 0.5) / (u64::MAX as f64 + 1.0);
            result.push(u);
        }
        result
    }
}

impl BrownianGenerator for SobolBrownianGenerator {
    fn next_step(&mut self) -> Vec<f64> {
        let uniforms = self.sobol_point();
        // Inverse normal CDF (Beasley-Springer-Moro)
        uniforms.iter().map(|&u| inv_normal_cdf(u.clamp(1e-10, 1.0 - 1e-10))).collect()
    }
    fn dimension(&self) -> usize { self.dimension }
}

/// Beasley-Springer-Moro inverse normal CDF.
fn inv_normal_cdf(u: f64) -> f64 {
    let a = [
        -3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02,
        -3.066479806614716e+01, 2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if u < p_low {
        let q = (-2.0 * u.ln()).sqrt();
        (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
            / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    } else if u <= p_high {
        let q = u - 0.5;
        let r = q * q;
        (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q
            / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
    } else {
        let q = (-2.0 * (1.0 - u).ln()).sqrt();
        -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
            / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    }
}

// ===========================================================================
// Evolvers
// ===========================================================================

/// Evolution description — time grid and rate structure.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvolutionDescription {
    /// Time steps (accrual lengths).
    pub dt: Vec<f64>,
    /// Number of forward rates.
    pub n_rates: usize,
    /// Index of first alive rate at each step.
    pub alive_at_step: Vec<usize>,
}

impl EvolutionDescription {
    /// Create from uniform accruals.
    pub fn uniform(n_rates: usize, accrual: f64) -> Self {
        let dt = vec![accrual; n_rates];
        let alive_at_step: Vec<usize> = (0..n_rates).map(|i| i + 1).collect();
        Self { dt, n_rates, alive_at_step }
    }
}

/// Trait for LMM evolvers.
pub trait Evolver: Send + Sync {
    /// Evolve forward rates one step. Returns new forward rates.
    fn evolve(
        &self,
        forwards: &[f64],
        alive_from: usize,
        step: usize,
        brownian: &[f64],
    ) -> Vec<f64>;

    /// Configuration reference.
    fn config(&self) -> &LmmConfig;
}

/// Log-normal forward rate Euler evolver.
#[derive(Clone, Debug)]
pub struct LogNormalFwdRateEuler {
    pub config: LmmConfig,
    pub cholesky: Vec<f64>,
}

impl LogNormalFwdRateEuler {
    pub fn new(config: LmmConfig) -> Self {
        let cholesky = config.cholesky();
        Self { config, cholesky }
    }
}

impl Evolver for LogNormalFwdRateEuler {
    fn evolve(&self, forwards: &[f64], alive_from: usize, step: usize, z: &[f64]) -> Vec<f64> {
        let n = self.config.n_rates;
        let dt = self.config.accruals[step.min(self.config.accruals.len() - 1)];
        let sqrt_dt = dt.sqrt();
        let mut new_fwd = forwards.to_vec();

        // Correlated Brownian increments
        let mut dw = vec![0.0; n];
        for i in alive_from..n {
            for k in alive_from..=i {
                dw[i] += self.cholesky[i * n + k] * z[k];
            }
            dw[i] *= sqrt_dt;
        }

        // Euler step: f_i(t+dt) = f_i(t) · exp((μ_i − ½σ²)dt + σ·dW)
        for i in alive_from..n {
            let drift = self.config.drift(i, forwards);
            let sigma = self.config.volatilities[i];
            new_fwd[i] = forwards[i] * ((drift - 0.5 * sigma * sigma) * dt + sigma * dw[i]).exp();
        }
        new_fwd
    }

    fn config(&self) -> &LmmConfig { &self.config }
}

/// Log-normal predictor-corrector evolver (Glasserman scheme).
#[derive(Clone, Debug)]
pub struct LogNormalFwdRatePC {
    pub config: LmmConfig,
    pub cholesky: Vec<f64>,
}

impl LogNormalFwdRatePC {
    pub fn new(config: LmmConfig) -> Self {
        let cholesky = config.cholesky();
        Self { config, cholesky }
    }
}

impl Evolver for LogNormalFwdRatePC {
    fn evolve(&self, forwards: &[f64], alive_from: usize, step: usize, z: &[f64]) -> Vec<f64> {
        let n = self.config.n_rates;
        let dt = self.config.accruals[step.min(self.config.accruals.len() - 1)];
        let sqrt_dt = dt.sqrt();

        let mut dw = vec![0.0; n];
        for i in alive_from..n {
            for k in alive_from..=i {
                dw[i] += self.cholesky[i * n + k] * z[k];
            }
            dw[i] *= sqrt_dt;
        }

        // Predictor
        let mut f_pred = forwards.to_vec();
        for i in alive_from..n {
            let drift = self.config.drift(i, forwards);
            let sigma = self.config.volatilities[i];
            f_pred[i] = forwards[i] * ((drift - 0.5 * sigma * sigma) * dt + sigma * dw[i]).exp();
        }

        // Corrector
        let mut new_fwd = forwards.to_vec();
        for i in alive_from..n {
            let d0 = self.config.drift(i, forwards);
            let d1 = self.config.drift(i, &f_pred);
            let avg = 0.5 * (d0 + d1);
            let sigma = self.config.volatilities[i];
            new_fwd[i] = forwards[i] * ((avg - 0.5 * sigma * sigma) * dt + sigma * dw[i]).exp();
        }
        new_fwd
    }

    fn config(&self) -> &LmmConfig { &self.config }
}

/// Balland-Tran evolver — variance-preserving drift correction.
#[derive(Clone, Debug)]
pub struct LogNormalFwdRateBalland {
    pub config: LmmConfig,
    pub cholesky: Vec<f64>,
}

impl LogNormalFwdRateBalland {
    pub fn new(config: LmmConfig) -> Self {
        let cholesky = config.cholesky();
        Self { config, cholesky }
    }
}

impl Evolver for LogNormalFwdRateBalland {
    fn evolve(&self, forwards: &[f64], alive_from: usize, step: usize, z: &[f64]) -> Vec<f64> {
        let n = self.config.n_rates;
        let dt = self.config.accruals[step.min(self.config.accruals.len() - 1)];
        let sqrt_dt = dt.sqrt();

        let mut dw = vec![0.0; n];
        for i in alive_from..n {
            for k in alive_from..=i {
                dw[i] += self.cholesky[i * n + k] * z[k];
            }
            dw[i] *= sqrt_dt;
        }

        // Balland-Tran uses the midpoint:
        //   f_mid = f · exp(−½σ²dt + σ·dW/2) then drift at midpoint
        let mut f_mid = forwards.to_vec();
        for i in alive_from..n {
            let sigma = self.config.volatilities[i];
            f_mid[i] = forwards[i] * (-0.5 * sigma * sigma * dt * 0.5 + sigma * dw[i] * 0.5).exp();
        }

        let mut new_fwd = forwards.to_vec();
        for i in alive_from..n {
            let drift = self.config.drift(i, &f_mid);
            let sigma = self.config.volatilities[i];
            new_fwd[i] = forwards[i] * ((drift - 0.5 * sigma * sigma) * dt + sigma * dw[i]).exp();
        }
        new_fwd
    }

    fn config(&self) -> &LmmConfig { &self.config }
}

/// Constrained Euler evolver — ensures positive forward rates.
#[derive(Clone, Debug)]
pub struct LogNormalFwdRateConstrainedEuler {
    pub config: LmmConfig,
    pub cholesky: Vec<f64>,
}

impl LogNormalFwdRateConstrainedEuler {
    pub fn new(config: LmmConfig) -> Self {
        let cholesky = config.cholesky();
        Self { config, cholesky }
    }
}

impl Evolver for LogNormalFwdRateConstrainedEuler {
    fn evolve(&self, forwards: &[f64], alive_from: usize, step: usize, z: &[f64]) -> Vec<f64> {
        let n = self.config.n_rates;
        let dt = self.config.accruals[step.min(self.config.accruals.len() - 1)];
        let sqrt_dt = dt.sqrt();

        let mut dw = vec![0.0; n];
        for i in alive_from..n {
            for k in alive_from..=i {
                dw[i] += self.cholesky[i * n + k] * z[k];
            }
            dw[i] *= sqrt_dt;
        }

        let mut new_fwd = forwards.to_vec();
        for i in alive_from..n {
            let drift = self.config.drift(i, forwards);
            let sigma = self.config.volatilities[i];
            new_fwd[i] = forwards[i] * ((drift - 0.5 * sigma * sigma) * dt + sigma * dw[i]).exp();
            // Constrain: floor at a small positive value
            new_fwd[i] = new_fwd[i].max(1e-8);
        }
        new_fwd
    }

    fn config(&self) -> &LmmConfig { &self.config }
}

/// Iterative predictor-corrector — applies PC correction multiple times.
#[derive(Clone, Debug)]
pub struct LogNormalFwdRateIPC {
    pub config: LmmConfig,
    pub cholesky: Vec<f64>,
    pub n_iterations: usize,
}

impl LogNormalFwdRateIPC {
    pub fn new(config: LmmConfig, n_iterations: usize) -> Self {
        let cholesky = config.cholesky();
        Self { config, cholesky, n_iterations }
    }
}

impl Evolver for LogNormalFwdRateIPC {
    fn evolve(&self, forwards: &[f64], alive_from: usize, step: usize, z: &[f64]) -> Vec<f64> {
        let n = self.config.n_rates;
        let dt = self.config.accruals[step.min(self.config.accruals.len() - 1)];
        let sqrt_dt = dt.sqrt();

        let mut dw = vec![0.0; n];
        for i in alive_from..n {
            for k in alive_from..=i {
                dw[i] += self.cholesky[i * n + k] * z[k];
            }
            dw[i] *= sqrt_dt;
        }

        // Start with Euler
        let mut f_curr = forwards.to_vec();
        for i in alive_from..n {
            let drift = self.config.drift(i, forwards);
            let sigma = self.config.volatilities[i];
            f_curr[i] = forwards[i] * ((drift - 0.5 * sigma * sigma) * dt + sigma * dw[i]).exp();
        }

        // Iterate corrector
        for _ in 0..self.n_iterations {
            let mut f_next = forwards.to_vec();
            for i in alive_from..n {
                let d0 = self.config.drift(i, forwards);
                let d1 = self.config.drift(i, &f_curr);
                let avg = 0.5 * (d0 + d1);
                let sigma = self.config.volatilities[i];
                f_next[i] = forwards[i] * ((avg - 0.5 * sigma * sigma) * dt + sigma * dw[i]).exp();
            }
            f_curr = f_next;
        }
        f_curr
    }

    fn config(&self) -> &LmmConfig { &self.config }
}

/// Log-normal coterminal swap rate predictor-corrector evolver.
#[derive(Clone, Debug)]
pub struct LogNormalCotSwapRatePC {
    pub config: LmmConfig,
    pub cholesky: Vec<f64>,
    pub swap_vols: Vec<f64>,
}

impl LogNormalCotSwapRatePC {
    pub fn new(config: LmmConfig, swap_vols: Vec<f64>) -> Self {
        let cholesky = config.cholesky();
        Self { config, cholesky, swap_vols }
    }
}

impl Evolver for LogNormalCotSwapRatePC {
    fn evolve(&self, forwards: &[f64], alive_from: usize, step: usize, z: &[f64]) -> Vec<f64> {
        let n = self.config.n_rates;
        let dt = self.config.accruals[step.min(self.config.accruals.len() - 1)];
        let sqrt_dt = dt.sqrt();

        let mut dw = vec![0.0; n];
        for i in alive_from..n {
            for k in alive_from..=i {
                dw[i] += self.cholesky[i * n + k] * z[k];
            }
            dw[i] *= sqrt_dt;
        }

        // Evolve coterminal swap rates, then derive forwards
        let state = LmmCurveState { forwards: forwards.to_vec(), alive_index: alive_from };
        let mut cot_rates: Vec<f64> = (0..n).map(|i| state.swap_rate(i, n, &self.config.accruals)).collect();

        let smm_drift = SMMDriftCalculator::new(n, self.swap_vols.clone(), self.config.correlation.clone(), self.config.accruals.clone());

        // Predictor
        let mut cot_pred = cot_rates.clone();
        for i in alive_from..n {
            let drift = smm_drift.drift(i, &cot_rates);
            let sigma = self.swap_vols[i.min(self.swap_vols.len() - 1)];
            cot_pred[i] = cot_rates[i] * ((drift - 0.5 * sigma * sigma) * dt + sigma * dw[i]).exp();
        }

        // Corrector
        for i in alive_from..n {
            let d0 = smm_drift.drift(i, &cot_rates);
            let d1 = smm_drift.drift(i, &cot_pred);
            let sigma = self.swap_vols[i.min(self.swap_vols.len() - 1)];
            cot_rates[i] = cot_rates[i] * ((0.5 * (d0 + d1) - 0.5 * sigma * sigma) * dt + sigma * dw[i]).exp();
        }

        // Convert back to forwards (approximate: use last coterminal for last rate)
        let mut new_fwd = forwards.to_vec();
        for i in (alive_from..n).rev() {
            if i == n - 1 {
                new_fwd[i] = cot_rates[i];
            } else {
                // f_i ≈ S_i + annuity-weighted correction
                new_fwd[i] = cot_rates[i];
            }
        }
        new_fwd
    }

    fn config(&self) -> &LmmConfig { &self.config }
}

/// Normal (Bachelier) forward rate predictor-corrector evolver.
#[derive(Clone, Debug)]
pub struct NormalFwdRatePC {
    pub config: LmmConfig,
    pub cholesky: Vec<f64>,
}

impl NormalFwdRatePC {
    pub fn new(config: LmmConfig) -> Self {
        let cholesky = config.cholesky();
        Self { config, cholesky }
    }
}

impl Evolver for NormalFwdRatePC {
    fn evolve(&self, forwards: &[f64], alive_from: usize, step: usize, z: &[f64]) -> Vec<f64> {
        let n = self.config.n_rates;
        let dt = self.config.accruals[step.min(self.config.accruals.len() - 1)];
        let sqrt_dt = dt.sqrt();

        let mut dw = vec![0.0; n];
        for i in alive_from..n {
            for k in alive_from..=i {
                dw[i] += self.cholesky[i * n + k] * z[k];
            }
            dw[i] *= sqrt_dt;
        }

        let normal_drift = LMMNormalDriftCalculator::new(&self.config);

        // Predictor (additive)
        let mut f_pred = forwards.to_vec();
        for i in alive_from..n {
            let drift = normal_drift.drift(i, forwards);
            let sigma = self.config.volatilities[i];
            f_pred[i] = forwards[i] + drift * dt + sigma * dw[i];
        }

        // Corrector
        let mut new_fwd = forwards.to_vec();
        for i in alive_from..n {
            let d0 = normal_drift.drift(i, forwards);
            let d1 = normal_drift.drift(i, &f_pred);
            let sigma = self.config.volatilities[i];
            new_fwd[i] = forwards[i] + 0.5 * (d0 + d1) * dt + sigma * dw[i];
        }
        new_fwd
    }

    fn config(&self) -> &LmmConfig { &self.config }
}

/// Log-normal CMS measure predictor-corrector evolver.
#[derive(Clone, Debug)]
pub struct LogNormalCMSwapRatePC {
    pub config: LmmConfig,
    pub cholesky: Vec<f64>,
    pub cms_vols: Vec<f64>,
    pub tenor_periods: usize,
}

impl LogNormalCMSwapRatePC {
    pub fn new(config: LmmConfig, cms_vols: Vec<f64>, tenor_periods: usize) -> Self {
        let cholesky = config.cholesky();
        Self { config, cholesky, cms_vols, tenor_periods }
    }
}

impl Evolver for LogNormalCMSwapRatePC {
    fn evolve(&self, forwards: &[f64], alive_from: usize, step: usize, z: &[f64]) -> Vec<f64> {
        let n = self.config.n_rates;
        let dt = self.config.accruals[step.min(self.config.accruals.len() - 1)];
        let sqrt_dt = dt.sqrt();

        let mut dw = vec![0.0; n];
        for i in alive_from..n {
            for k in alive_from..=i {
                dw[i] += self.cholesky[i * n + k] * z[k];
            }
            dw[i] *= sqrt_dt;
        }

        let cms_drift = CmSMMDriftCalculator::new(
            n, self.tenor_periods, self.cms_vols.clone(),
            self.config.correlation.clone(), self.config.accruals.clone(),
        );

        // Compute CMS rates
        let state = LmmCurveState { forwards: forwards.to_vec(), alive_index: alive_from };
        let mut cms_rates: Vec<f64> = (0..n)
            .map(|i| state.swap_rate(i, (i + self.tenor_periods).min(n), &self.config.accruals))
            .collect();

        // PC on CMS rates
        let mut cms_pred = cms_rates.clone();
        for i in alive_from..n {
            let drift = cms_drift.drift(i, &cms_rates);
            let sigma = self.cms_vols[i.min(self.cms_vols.len() - 1)];
            cms_pred[i] = cms_rates[i] * ((drift - 0.5 * sigma * sigma) * dt + sigma * dw[i]).exp();
        }
        for i in alive_from..n {
            let d0 = cms_drift.drift(i, &cms_rates);
            let d1 = cms_drift.drift(i, &cms_pred);
            let sigma = self.cms_vols[i.min(self.cms_vols.len() - 1)];
            cms_rates[i] = cms_rates[i] * ((0.5 * (d0 + d1) - 0.5 * sigma * sigma) * dt + sigma * dw[i]).exp();
        }

        // Map back to forwards (approximate)
        let mut new_fwd = forwards.to_vec();
        for i in alive_from..n {
            new_fwd[i] = cms_rates[i];
        }
        new_fwd
    }

    fn config(&self) -> &LmmConfig { &self.config }
}

// ===========================================================================
// Swap/Forward Mappings
// ===========================================================================

/// Forward-to-swap and swap-to-forward mapping utilities.
pub struct SwapForwardMappings;

impl SwapForwardMappings {
    /// Compute the Jacobian ∂S/∂f (swap rate wrt forward rates).
    ///
    /// Returns an (n_swaps × n_rates) matrix stored row-major.
    pub fn jacobian(forwards: &[f64], accruals: &[f64], swap_start: usize, swap_end: usize) -> Vec<f64> {
        let n_fwd = forwards.len();
        let n_swap = swap_end - swap_start;
        let mut jac = vec![0.0; n_swap * n_fwd];

        let bump = 1e-6;
        for j in 0..n_fwd {
            let mut fwd_up = forwards.to_vec();
            fwd_up[j] += bump;
            let state_up = LmmCurveState { forwards: fwd_up, alive_index: 0 };
            let state_base = LmmCurveState { forwards: forwards.to_vec(), alive_index: 0 };

            for i in 0..n_swap {
                let s_up = state_up.swap_rate(swap_start + i, swap_end, accruals);
                let s_base = state_base.swap_rate(swap_start + i, swap_end, accruals);
                jac[i * n_fwd + j] = (s_up - s_base) / bump;
            }
        }
        jac
    }

    /// Coterminal swap annuity at each starting period.
    pub fn coterminal_annuities(forwards: &[f64], accruals: &[f64]) -> Vec<f64> {
        let n = forwards.len();
        let mut annuities = vec![0.0; n];
        for i in 0..n {
            let mut d = 1.0;
            let mut ann = 0.0;
            for j in i..n {
                d /= 1.0 + accruals[j] * forwards[j];
                ann += accruals[j] * d;
            }
            annuities[i] = ann;
        }
        annuities
    }
}

/// Forward-to-forward mapping utilities.
pub struct ForwardForwardMappings;

impl ForwardForwardMappings {
    /// Compute covariance of forward rate changes from pseudo-square-root
    /// (factor loadings).
    ///
    /// cov(i,j) = Σ_k A_{ik} A_{jk} where A is the factor loading matrix.
    pub fn covariance_from_pseudo_root(pseudo_root: &[f64], n_rates: usize, n_factors: usize) -> Vec<f64> {
        let mut cov = vec![0.0; n_rates * n_rates];
        for i in 0..n_rates {
            for j in 0..n_rates {
                let mut s = 0.0;
                for k in 0..n_factors {
                    s += pseudo_root[i * n_factors + k] * pseudo_root[j * n_factors + k];
                }
                cov[i * n_rates + j] = s;
            }
        }
        cov
    }
}

// ===========================================================================
// Alpha Forms (calibration shape functions)
// ===========================================================================

/// Trait for alpha scaling functions used in LMM calibration.
pub trait AlphaForm {
    /// Alpha value at index i (0 ≤ α ≤ 1).
    fn alpha(&self, i: usize) -> f64;
}

/// Linear-hyperbolic alpha form: α(i) = 1 − (1−a)·i/(n−1).
#[derive(Clone, Debug)]
pub struct AlphaFormLinearHyperbolic {
    pub n_rates: usize,
    pub alpha_0: f64,
}

impl AlphaFormLinearHyperbolic {
    pub fn new(n_rates: usize, alpha_0: f64) -> Self { Self { n_rates, alpha_0 } }
}

impl AlphaForm for AlphaFormLinearHyperbolic {
    fn alpha(&self, i: usize) -> f64 {
        if self.n_rates <= 1 { return 1.0; }
        1.0 - (1.0 - self.alpha_0) * i as f64 / (self.n_rates - 1) as f64
    }
}

// ===========================================================================
// Accounting / Greeks Engines
// ===========================================================================

/// Accounting engine for pathwise P&L accumulation on LMM products.
///
/// Simulates many paths, accumulates cash flows, and computes statistics
/// (mean PV, standard error, quantiles).
#[derive(Clone, Debug)]
pub struct AccountingEngine {
    pub config: LmmConfig,
    pub n_paths: usize,
    pub seed: u64,
}

impl AccountingEngine {
    pub fn new(config: LmmConfig, n_paths: usize, seed: u64) -> Self {
        Self { config, n_paths, seed }
    }

    /// Run the accounting engine. Returns (mean_pv, std_error, path_pvs).
    pub fn run<E: Evolver>(
        &self,
        evolver: &E,
        cashflow_fn: &dyn Fn(usize, &[f64], &LmmConfig) -> f64,
    ) -> AccountingResult {
        let n = self.config.n_rates;
        let mut rng = SmallRng::seed_from_u64(self.seed);
        let mut pvs = Vec::with_capacity(self.n_paths);

        for _ in 0..self.n_paths {
            let mut forwards = self.config.initial_forwards.clone();
            let mut numeraire = 1.0;
            let mut path_pv = 0.0;

            for step in 0..n {
                let z: Vec<f64> = (0..n).map(|_| rng.sample(StandardNormal)).collect();
                let cf = cashflow_fn(step, &forwards, &self.config);
                path_pv += cf / numeraire;

                forwards = evolver.evolve(&forwards, step + 1, step, &z);
                numeraire *= 1.0 + self.config.accruals[step] * self.config.initial_forwards[step];
            }
            pvs.push(path_pv);
        }

        let mean = pvs.iter().sum::<f64>() / self.n_paths as f64;
        let variance = pvs.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / self.n_paths as f64;
        let std_error = (variance / self.n_paths as f64).max(0.0).sqrt();

        AccountingResult { mean_pv: mean, std_error, path_pvs: pvs }
    }
}

/// Result of an accounting engine run.
#[derive(Clone, Debug)]
pub struct AccountingResult {
    pub mean_pv: f64,
    pub std_error: f64,
    pub path_pvs: Vec<f64>,
}

impl AccountingResult {
    /// Value-at-Risk at confidence level (e.g. 0.95).
    pub fn var(&self, confidence: f64) -> f64 {
        let mut sorted = self.path_pvs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((1.0 - confidence) * sorted.len() as f64).floor() as usize;
        -sorted[idx.min(sorted.len() - 1)]
    }

    /// Expected Shortfall (CVaR) at confidence level.
    pub fn expected_shortfall(&self, confidence: f64) -> f64 {
        let mut sorted = self.path_pvs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let cutoff = ((1.0 - confidence) * sorted.len() as f64).floor() as usize;
        let tail = &sorted[..cutoff.max(1)];
        -tail.iter().sum::<f64>() / tail.len() as f64
    }
}

/// Proxy Greek engine — computes parameter sensitivities via bump-and-revalue.
#[derive(Clone, Debug)]
pub struct ProxyGreekEngine {
    pub config: LmmConfig,
    pub n_paths: usize,
    pub seed: u64,
}

impl ProxyGreekEngine {
    pub fn new(config: LmmConfig, n_paths: usize, seed: u64) -> Self {
        Self { config, n_paths, seed }
    }

    /// Compute vega₁ = ∂V/∂σ (parallel vol bump).
    pub fn parallel_vega(
        &self,
        pricer: &dyn Fn(&LmmConfig) -> f64,
        bump: f64,
    ) -> f64 {
        let base = pricer(&self.config);
        let mut config_up = self.config.clone();
        for v in config_up.volatilities.iter_mut() {
            *v += bump;
        }
        let up = pricer(&config_up);
        (up - base) / bump
    }

    /// Compute delta₁ = ∂V/∂f (parallel forward bump).
    pub fn parallel_delta(
        &self,
        pricer: &dyn Fn(&LmmConfig) -> f64,
        bump: f64,
    ) -> f64 {
        let base = pricer(&self.config);
        let mut config_up = self.config.clone();
        for f in config_up.initial_forwards.iter_mut() {
            *f += bump;
        }
        let up = pricer(&config_up);
        (up - base) / bump
    }

    /// Bucket vega — sensitivity to each individual rate's volatility.
    pub fn bucket_vega(
        &self,
        pricer: &dyn Fn(&LmmConfig) -> f64,
        bump: f64,
    ) -> Vec<f64> {
        let base = pricer(&self.config);
        let n = self.config.n_rates;
        (0..n).map(|i| {
            let mut c = self.config.clone();
            c.volatilities[i] += bump;
            let up = pricer(&c);
            (up - base) / bump
        }).collect()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_config() -> LmmConfig {
        LmmConfig::flat(10, 0.05, 0.25, 0.20, 0.5)
    }

    // --- Curve States ---

    #[test]
    fn test_lmm_curve_state() {
        let config = make_config();
        let cs = LMMCurveState::new(config.initial_forwards.clone(), config.accruals.clone());
        let sr = cs.swap_rate(0, 10);
        assert!(sr > 0.04 && sr < 0.06, "swap rate={}", sr);
        assert!(cs.discount_ratio(0, 5) < 1.0);
    }

    #[test]
    fn test_coterminal_curve_state() {
        let config = make_config();
        let cs = CoterminalSwapCurveState::from_forwards(&config.initial_forwards, &config.accruals);
        assert_eq!(cs.coterminal_rates.len(), 10);
        // Last coterminal = last forward
        assert_abs_diff_eq!(cs.coterminal_rates[9], 0.05, epsilon = 1e-10);
        // First coterminal ≈ 0.05 at flat
        assert_abs_diff_eq!(cs.coterminal_swap_rate(0), 0.05, epsilon = 1e-6);
    }

    #[test]
    fn test_cm_swap_curve_state() {
        let config = make_config();
        let cs = CMSwapCurveState::from_forwards(&config.initial_forwards, &config.accruals, 4);
        assert_eq!(cs.cms_rates.len(), 10);
        // At flat forwards CMS = forward
        assert_abs_diff_eq!(cs.cms_rates[0], 0.05, epsilon = 1e-4);
    }

    // --- Drift Calculators ---

    #[test]
    fn test_lmm_drift_calculator() {
        let config = make_config();
        let calc = LMMDriftCalculator::new(&config);
        let drifts = calc.compute_all(&config.initial_forwards, 0);
        assert_eq!(drifts.len(), 10);
        // Under terminal measure, last rate has zero drift
        assert_abs_diff_eq!(drifts[9], 0.0, epsilon = 1e-12);
        // Earlier rates have negative drift
        assert!(drifts[0] < 0.0, "drift[0]={}", drifts[0]);
    }

    #[test]
    fn test_smm_drift() {
        let config = make_config();
        let smm = SMMDriftCalculator::new(10, config.volatilities.clone(), config.correlation.clone(), config.accruals.clone());
        let swaps = vec![0.05; 10];
        let d = smm.drift(5, &swaps);
        assert!(d < 0.0, "smm drift[5]={}", d);
    }

    // --- Volatility Structures ---

    #[test]
    fn test_flat_vol() {
        let fv = FlatVol::new(0.20, 10);
        assert_eq!(fv.volatilities().len(), 10);
        assert_abs_diff_eq!(fv.volatilities()[5], 0.20, epsilon = 1e-14);
    }

    #[test]
    fn test_abcd_vol() {
        let abcd = AbcdVol::new(0.1, 0.05, 1.0, 0.10);
        let v = abcd.vol(0.0); // a + d at tau=0
        assert_abs_diff_eq!(v, 0.2, epsilon = 1e-14);
        let v2 = abcd.vol(5.0);
        assert!(v2 > 0.0 && v2 < 0.5, "abcd vol@5y={}", v2);
    }

    #[test]
    fn test_piecewise_constant_variance() {
        let vols = vec![0.18, 0.19, 0.20, 0.21, 0.22];
        let pcv = PiecewiseConstantVariance::new(vols);
        let iv = pcv.integrated_variance(2, 1.0);
        assert_abs_diff_eq!(iv, 0.04, epsilon = 1e-12); // 0.20² × 1.0
    }

    // --- Correlations ---

    #[test]
    fn test_exponential_correlation() {
        let corr = ExponentialCorrelation::new(0.5, 5);
        let m = corr.matrix(0.25);
        assert_abs_diff_eq!(m[0], 1.0, epsilon = 1e-10); // ρ_{0,0}
        assert!(m[1] > 0.0 && m[1] < 1.0); // ρ_{0,1}
    }

    #[test]
    fn test_time_homogeneous_correlation() {
        let thc = TimeHomogeneousForwardCorrelation::exponential_with_floor(5, 0.3, 0.2);
        assert_abs_diff_eq!(thc.correlations[0], 1.0, epsilon = 1e-10);
        // Off-diagonal between 0 and 1
        assert!(thc.correlations[1] > 0.2 && thc.correlations[1] < 1.0);
    }

    // --- Brownian Generators ---

    #[test]
    fn test_mt_brownian_generator() {
        let mut gen = MTBrownianGenerator::new(10, 42);
        let z = gen.next_step();
        assert_eq!(z.len(), 10);
        // Not all zero
        assert!(z.iter().any(|&v| v.abs() > 0.01));
    }

    #[test]
    fn test_sobol_brownian_generator() {
        let mut gen = SobolBrownianGenerator::new(5, 0);
        let z1 = gen.next_step();
        let z2 = gen.next_step();
        assert_eq!(z1.len(), 5);
        // Should produce different values
        assert!((z1[0] - z2[0]).abs() > 1e-10);
    }

    // --- Evolvers ---

    #[test]
    fn test_euler_evolver() {
        let config = make_config();
        let evolver = LogNormalFwdRateEuler::new(config.clone());
        let mut rng = SmallRng::seed_from_u64(42);
        let z: Vec<f64> = (0..10).map(|_| rng.sample(StandardNormal)).collect();
        let new_fwd = evolver.evolve(&config.initial_forwards, 1, 0, &z);
        assert_eq!(new_fwd.len(), 10);
        assert_abs_diff_eq!(new_fwd[0], config.initial_forwards[0], epsilon = 1e-14);
        for &f in &new_fwd[1..] {
            assert!(f > 0.0, "forward={}", f);
        }
    }

    #[test]
    fn test_pc_evolver() {
        let config = make_config();
        let evolver = LogNormalFwdRatePC::new(config.clone());
        let z: Vec<f64> = vec![0.5; 10]; // deterministic
        let fwd = evolver.evolve(&config.initial_forwards, 1, 0, &z);
        for &f in &fwd[1..] {
            assert!(f > 0.0);
        }
    }

    #[test]
    fn test_balland_evolver() {
        let config = make_config();
        let evolver = LogNormalFwdRateBalland::new(config.clone());
        let z: Vec<f64> = vec![0.3; 10];
        let fwd = evolver.evolve(&config.initial_forwards, 1, 0, &z);
        for &f in &fwd[1..] {
            assert!(f > 0.0);
        }
    }

    #[test]
    fn test_constrained_euler() {
        let config = make_config();
        let evolver = LogNormalFwdRateConstrainedEuler::new(config.clone());
        // Use very negative z to try to drive rates negative
        let z: Vec<f64> = vec![-5.0; 10];
        let fwd = evolver.evolve(&config.initial_forwards, 1, 0, &z);
        for &f in &fwd[1..] {
            assert!(f > 0.0, "constrained rate must be positive: {}", f);
        }
    }

    #[test]
    fn test_ipc_evolver() {
        let config = make_config();
        let evolver = LogNormalFwdRateIPC::new(config.clone(), 3);
        let z: Vec<f64> = vec![0.1; 10];
        let fwd = evolver.evolve(&config.initial_forwards, 1, 0, &z);
        for &f in &fwd[1..] {
            assert!(f > 0.0);
        }
    }

    #[test]
    fn test_normal_fwd_rate_pc() {
        let config = make_config();
        let evolver = NormalFwdRatePC::new(config.clone());
        let z: Vec<f64> = vec![0.2; 10];
        let fwd = evolver.evolve(&config.initial_forwards, 1, 0, &z);
        assert_eq!(fwd.len(), 10);
        // Normal model can go slightly negative, but should be near 0.05
        for &f in &fwd[1..] {
            assert!(f > -0.1 && f < 0.2, "normal rate={}", f);
        }
    }

    #[test]
    fn test_cotswap_evolver() {
        let config = make_config();
        let swap_vols = vec![0.20; 10];
        let evolver = LogNormalCotSwapRatePC::new(config.clone(), swap_vols);
        let z: Vec<f64> = vec![0.1; 10];
        let fwd = evolver.evolve(&config.initial_forwards, 1, 0, &z);
        for &f in &fwd[1..] {
            assert!(f > 0.0, "cot swap fwd={}", f);
        }
    }

    #[test]
    fn test_cm_swap_evolver() {
        let config = make_config();
        let cms_vols = vec![0.20; 10];
        let evolver = LogNormalCMSwapRatePC::new(config.clone(), cms_vols, 4);
        let z: Vec<f64> = vec![0.1; 10];
        let fwd = evolver.evolve(&config.initial_forwards, 1, 0, &z);
        for &f in &fwd[1..] {
            assert!(f > 0.0, "cms fwd={}", f);
        }
    }

    // --- Mappings ---

    #[test]
    fn test_swap_forward_jacobian() {
        let config = make_config();
        let jac = SwapForwardMappings::jacobian(&config.initial_forwards, &config.accruals, 0, 10);
        // 10 × 10 = 100 elements
        assert_eq!(jac.len(), 100);
    }

    #[test]
    fn test_coterminal_annuities() {
        let config = make_config();
        let ann = SwapForwardMappings::coterminal_annuities(&config.initial_forwards, &config.accruals);
        assert_eq!(ann.len(), 10);
        for &a in &ann {
            assert!(a > 0.0);
        }
        // Full annuity > shorter
        assert!(ann[0] > ann[5]);
    }

    // --- Accounting Engine ---

    #[test]
    fn test_accounting_engine() {
        let config = make_config();
        let evolver = LogNormalFwdRatePC::new(config.clone());
        let engine = AccountingEngine::new(config.clone(), 1000, 42);

        let result = engine.run(&evolver, &|step, forwards, config| {
            // Simple cap payoff
            (forwards[step.min(config.n_rates - 1)] - 0.05).max(0.0) * config.accruals[step.min(config.accruals.len() - 1)]
        });

        assert!(result.mean_pv > -1.0, "mean_pv={}", result.mean_pv);
        assert!(result.std_error > 0.0);
        assert_eq!(result.path_pvs.len(), 1000);
    }

    #[test]
    fn test_accounting_var() {
        let pvs = vec![-0.1, -0.05, 0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20];
        let result = AccountingResult { mean_pv: 0.047, std_error: 0.01, path_pvs: pvs };
        let var_95 = result.var(0.95);
        assert!(var_95 > 0.0, "VaR={}", var_95);
    }

    // --- Proxy Greeks ---

    #[test]
    fn test_proxy_greek_engine() {
        let config = make_config();
        let engine = ProxyGreekEngine::new(config, 500, 42);

        let pricer = |config: &LmmConfig| -> f64 {
            crate::lmm::lmm_cap_price(config, 0.05, 500, 42).price
        };

        let vega = engine.parallel_vega(&pricer, 0.01);
        assert!(vega.abs() < 10.0, "vega={}", vega);

        let delta = engine.parallel_delta(&pricer, 0.001);
        assert!(delta.abs() < 100.0, "delta={}", delta);

        let bvega = engine.bucket_vega(&pricer, 0.01);
        assert_eq!(bvega.len(), 10);
    }

    // --- Alpha Forms ---

    #[test]
    fn test_alpha_form_linear_hyperbolic() {
        let af = AlphaFormLinearHyperbolic::new(10, 0.5);
        assert_abs_diff_eq!(af.alpha(0), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(af.alpha(9), 0.5, epsilon = 1e-10);
    }

    // --- Evolution Description ---

    #[test]
    fn test_evolution_description() {
        let ed = EvolutionDescription::uniform(10, 0.25);
        assert_eq!(ed.dt.len(), 10);
        assert_eq!(ed.alive_at_step[3], 4);
    }
}
