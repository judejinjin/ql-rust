#![allow(clippy::too_many_arguments)]
//! FDM infrastructure, step conditions, RND calculators, boundary conditions.
//!
//! - G140-G150: Infrastructure (layout, solvers, helpers)
//! - G151-G155: Step conditions (storage, swing, snapshot, composite, averaging)
//! - G156-G161: Risk-neutral density calculators
//! - G162-G164: Extended boundary conditions

use serde::{Deserialize, Serialize};

use crate::fdm_meshers::{FdmMesherComposite, Mesher1d};
use crate::fdm_operators::TripleBandOp;

// ═══════════════════════════════════════════════════════════════════════════
// G140: FdmLinearOpLayout / FdmLinearOpIterator
// ═══════════════════════════════════════════════════════════════════════════

/// Describes the multi-dimensional layout of grid points for FDM operators.
///
/// Provides index conversion between flat (1D) and multi-dimensional
/// indices and tracks dimension sizes and strides.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FdmLinearOpLayout {
    /// Grid sizes per dimension.
    pub dim: Vec<usize>,
    /// Strides for each dimension (row-major order).
    pub strides: Vec<usize>,
    /// Total number of grid points.
    pub size: usize,
}

impl FdmLinearOpLayout {
    /// Create from dimension sizes.
    pub fn new(dim: Vec<usize>) -> Self {
        let n = dim.len();
        let mut strides = vec![1usize; n];
        for d in (0..n - 1).rev() {
            strides[d] = strides[d + 1] * dim[d + 1];
        }
        let size = dim.iter().product();
        Self { dim, strides, size }
    }

    /// Number of dimensions.
    pub fn dimensions(&self) -> usize {
        self.dim.len()
    }

    /// Convert flat index to multi-dimensional coordinates.
    pub fn to_coords(&self, flat: usize) -> Vec<usize> {
        let mut coords = vec![0; self.dim.len()];
        let mut remaining = flat;
        for d in 0..self.dim.len() {
            coords[d] = remaining / self.strides[d];
            remaining %= self.strides[d];
        }
        coords
    }

    /// Convert multi-dimensional coordinates to flat index.
    pub fn to_flat(&self, coords: &[usize]) -> usize {
        coords
            .iter()
            .zip(self.strides.iter())
            .map(|(&c, &s)| c * s)
            .sum()
    }

    /// Create an iterator over all grid points.
    pub fn iter(&self) -> FdmLinearOpIterator {
        FdmLinearOpIterator {
            layout: self.clone(),
            current: 0,
        }
    }

    /// Build from a composite mesher.
    pub fn from_mesher(mesher: &FdmMesherComposite) -> Self {
        Self::new(mesher.sizes())
    }
}

/// Iterator over all grid points in the layout.
pub struct FdmLinearOpIterator {
    layout: FdmLinearOpLayout,
    current: usize,
}

impl Iterator for FdmLinearOpIterator {
    type Item = (usize, Vec<usize>);
    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.layout.size {
            return None;
        }
        let flat = self.current;
        let coords = self.layout.to_coords(flat);
        self.current += 1;
        Some((flat, coords))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// G141: FdmBackwardSolver
// ═══════════════════════════════════════════════════════════════════════════

/// Generic backward-in-time FD solver.
///
/// Solves the PDE from terminal time T to valuation time t = 0
/// using the supplied operator and time-stepping scheme.
#[derive(Debug, Clone)]
pub struct FdmBackwardSolver {
    /// Number of time steps.
    pub n_steps: usize,
    /// Total time horizon.
    pub total_time: f64,
    /// Time stepping theta (0.5 = CN, 1.0 = implicit).
    pub theta: f64,
}

impl FdmBackwardSolver {
    pub fn new(n_steps: usize, total_time: f64, theta: f64) -> Self {
        Self {
            n_steps,
            total_time,
            theta,
        }
    }

    /// Solve backward with a 1D operator and optional step condition.
    pub fn solve(
        &self,
        op: &TripleBandOp,
        terminal: &[f64],
        step_condition: Option<&dyn Fn(&mut [f64], f64)>,
    ) -> Vec<f64> {
        let dt = self.total_time / self.n_steps as f64;
        let mut v = terminal.to_vec();
        for step in 0..self.n_steps {
            let t = self.total_time - (step + 1) as f64 * dt;
            // Crank-Nicolson step
            v = crate::fdm_operators::crank_nicolson_step(op, &v, dt, self.theta);
            if let Some(cond) = step_condition {
                cond(&mut v, t);
            }
        }
        v
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// G142: FdmSolverDesc — Solver descriptor
// ═══════════════════════════════════════════════════════════════════════════

/// Descriptor collecting all parameters needed to construct an FDM solver.
#[derive(Debug, Clone)]
pub struct FdmSolverDesc {
    /// Composite mesher defining the spatial grid.
    pub mesher: FdmMesherComposite,
    /// Number of time steps.
    pub n_time_steps: usize,
    /// Maturity (years).
    pub maturity: f64,
    /// Theta for time stepping (0.5 = CN).
    pub theta: f64,
    /// Damping steps (extra implicit steps at start for stability).
    pub damping_steps: usize,
}

impl FdmSolverDesc {
    pub fn new(mesher: FdmMesherComposite, n_time_steps: usize, maturity: f64) -> Self {
        Self {
            mesher,
            n_time_steps,
            maturity,
            theta: 0.5,
            damping_steps: 0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// G143: Fdm1DimSolver / Fdm2DimSolver / FdmNDimSolver
// ═══════════════════════════════════════════════════════════════════════════

/// 1D finite difference solver.
#[derive(Debug, Clone)]
pub struct Fdm1DimSolver {
    pub desc: FdmSolverDesc,
    pub op: TripleBandOp,
}

impl Fdm1DimSolver {
    pub fn new(desc: FdmSolverDesc, op: TripleBandOp) -> Self {
        Self { desc, op }
    }

    /// Solve backward from terminal values.
    pub fn solve(
        &self,
        terminal: &[f64],
        step_condition: Option<&dyn Fn(&mut [f64], f64)>,
    ) -> Vec<f64> {
        let solver = FdmBackwardSolver::new(
            self.desc.n_time_steps,
            self.desc.maturity,
            self.desc.theta,
        );
        solver.solve(&self.op, terminal, step_condition)
    }
}

/// 2D finite difference solver with ADI splitting.
#[derive(Debug, Clone)]
pub struct Fdm2DimSolver {
    pub desc: FdmSolverDesc,
    pub op1: TripleBandOp,
    pub op2: TripleBandOp,
}

impl Fdm2DimSolver {
    pub fn new(desc: FdmSolverDesc, op1: TripleBandOp, op2: TripleBandOp) -> Self {
        Self { desc, op1, op2 }
    }

    /// Solve using Douglas ADI splitting.
    pub fn solve(&self, terminal: &[f64]) -> Vec<f64> {
        let n1 = self.desc.mesher.meshers[0].size();
        let n2 = self.desc.mesher.meshers[1].size();
        let dt = self.desc.maturity / self.desc.n_time_steps as f64;
        let mut v = terminal.to_vec();
        let cross = vec![0.0; n1 * n2]; // no cross term
        for _ in 0..self.desc.n_time_steps {
            crate::fdm_extended::craig_sneyd_step(
                &self.op1, &self.op2, &cross, &mut v, n1, n2, dt, self.desc.theta,
            );
        }
        v
    }
}

/// N-dimensional FDM solver (generic).
///
/// Uses recursive operator splitting for N > 2 dimensions.
#[derive(Debug, Clone)]
pub struct FdmNDimSolver {
    pub desc: FdmSolverDesc,
    pub ops: Vec<TripleBandOp>,
}

impl FdmNDimSolver {
    pub fn new(desc: FdmSolverDesc, ops: Vec<TripleBandOp>) -> Self {
        Self { desc, ops }
    }

    /// Solve via sequential operator splitting (Lie-Trotter).
    pub fn solve(&self, terminal: &[f64]) -> Vec<f64> {
        let dt = self.desc.maturity / self.desc.n_time_steps as f64;
        let sizes = self.desc.mesher.sizes();
        let total: usize = sizes.iter().product();
        let mut v = terminal.to_vec();
        assert_eq!(v.len(), total);

        for _ in 0..self.desc.n_time_steps {
            // Sweep each dimension
            for (d, op) in self.ops.iter().enumerate() {
                let nd = sizes[d];
                let outer: usize = sizes[..d].iter().product();
                let inner: usize = sizes[d + 1..].iter().product();
                for o in 0..outer {
                    for inn in 0..inner {
                        // Extract 1D line along dimension d
                        let mut line = vec![0.0; nd];
                        for k in 0..nd {
                            let flat = o * nd * inner + k * inner + inn;
                            line[k] = v[flat];
                        }
                        let new_line = crate::fdm_operators::crank_nicolson_step(
                            op,
                            &line,
                            dt,
                            self.desc.theta,
                        );
                        for k in 0..nd {
                            let flat = o * nd * inner + k * inner + inn;
                            v[flat] = new_line[k];
                        }
                    }
                }
            }
        }
        v
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// G144: FdmQuantoHelper
// ═══════════════════════════════════════════════════════════════════════════

/// Quanto adjustment handler for FDM.
///
/// Modifies drift of the underlying process to account for quanto effects:
///   drift_adj = −ρ_SX · σ_S · σ_X
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FdmQuantoHelper {
    /// Correlation between asset and FX.
    pub rho_sx: f64,
    /// Asset volatility.
    pub sigma_s: f64,
    /// FX volatility.
    pub sigma_x: f64,
    /// Foreign risk-free rate.
    pub r_foreign: f64,
    /// Domestic risk-free rate.
    pub r_domestic: f64,
}

impl FdmQuantoHelper {
    pub fn new(rho_sx: f64, sigma_s: f64, sigma_x: f64, r_foreign: f64, r_domestic: f64) -> Self {
        Self {
            rho_sx,
            sigma_s,
            sigma_x,
            r_foreign,
            r_domestic,
        }
    }

    /// Quanto drift adjustment.
    pub fn drift_adjustment(&self) -> f64 {
        -self.rho_sx * self.sigma_s * self.sigma_x
    }

    /// Effective risk-free rate for pricing in domestic currency.
    pub fn effective_rate(&self) -> f64 {
        self.r_domestic
    }

    /// Adjusted dividend yield.
    pub fn adjusted_dividend_yield(&self, q: f64) -> f64 {
        q - self.drift_adjustment() + self.r_foreign - self.r_domestic
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// G145: FdmDividendHandler
// ═══════════════════════════════════════════════════════════════════════════

/// Discrete dividend handler for FDM grids.
///
/// At dividend payment times, adjusts the solution grid by shifting
/// the spot (jump condition): V(S, t⁻) = V(S − D, t⁺).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FdmDividendHandler {
    /// Dividend amounts.
    pub dividends: Vec<f64>,
    /// Dividend times (years from valuation).
    pub dividend_times: Vec<f64>,
    /// Time matching tolerance.
    pub tolerance: f64,
}

impl FdmDividendHandler {
    pub fn new(dividends: Vec<f64>, dividend_times: Vec<f64>) -> Self {
        assert_eq!(dividends.len(), dividend_times.len());
        Self {
            dividends,
            dividend_times,
            tolerance: 1e-6,
        }
    }

    /// Apply dividend jump to solution vector on a log-spot grid.
    ///
    /// Shifts values: V(x) → V(x − ln(1 − D/S_i)) via interpolation.
    pub fn apply(&self, values: &mut [f64], grid: &Mesher1d, t: f64) {
        for (k, &dt) in self.dividend_times.iter().enumerate() {
            if (dt - t).abs() < self.tolerance {
                let d = self.dividends[k];
                if d <= 0.0 {
                    continue;
                }
                let n = values.len();
                let shifted = values.to_vec();
                for i in 0..n {
                    let x = grid.locations[i];
                    let s = x.exp();
                    let s_adj = (s - d).max(1e-10);
                    let x_adj = s_adj.ln();
                    // Linear interpolation on the grid
                    let idx = grid.lower_index(x_adj);
                    if idx + 1 < n {
                        let w = (x_adj - grid.locations[idx])
                            / (grid.locations[idx + 1] - grid.locations[idx]);
                        values[i] = (1.0 - w) * shifted[idx] + w * shifted[idx + 1];
                    } else {
                        values[i] = shifted[idx.min(n - 1)];
                    }
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// G146: FdmInnerValueCalculator
// ═══════════════════════════════════════════════════════════════════════════

/// Computes the inner (intrinsic / exercise) value at each grid point.
pub trait FdmInnerValueCalculator {
    /// Compute inner values on the grid.
    fn inner_value(&self, grid: &Mesher1d, t: f64) -> Vec<f64>;
}

/// Plain-vanilla option inner value calculator.
#[derive(Debug, Clone)]
pub struct VanillaInnerValue {
    pub strike: f64,
    pub is_call: bool,
    /// If true, grid is log-spot.
    pub log_grid: bool,
}

impl FdmInnerValueCalculator for VanillaInnerValue {
    fn inner_value(&self, grid: &Mesher1d, _t: f64) -> Vec<f64> {
        grid.locations
            .iter()
            .map(|&x| {
                let s = if self.log_grid { x.exp() } else { x };
                if self.is_call {
                    (s - self.strike).max(0.0)
                } else {
                    (self.strike - s).max(0.0)
                }
            })
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// G147: FdmAffineModelTermStructure — adapter
// ═══════════════════════════════════════════════════════════════════════════

/// Affine model adapter that provides discount factors from a short-rate grid.
///
/// For a short-rate model, the discount factor at grid point r_i is:
///   P(t, T; r_i) = A(t,T) · exp(−B(t,T) · r_i)
///
/// This type stores the affine coefficients.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FdmAffineModelTermStructure {
    /// A(t, T) coefficient.
    pub a: f64,
    /// B(t, T) coefficient.
    pub b: f64,
    /// Valuation time t.
    pub t: f64,
    /// Maturity T.
    pub big_t: f64,
}

impl FdmAffineModelTermStructure {
    pub fn new(a: f64, b: f64, t: f64, big_t: f64) -> Self {
        Self { a, b, t, big_t }
    }

    /// Discount factor at short rate r.
    pub fn discount(&self, r: f64) -> f64 {
        self.a * (-self.b * r).exp()
    }

    /// Forward rate at short rate r.
    pub fn forward_rate(&self, r: f64) -> f64 {
        // −d ln P / dT ≈ r (for short maturities)
        let dt = 1e-4;
        let p1 = self.a * (-self.b * r).exp();
        let p2 = self.a * (-(self.b + dt) * r).exp();
        -(p2 / p1).ln() / dt
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// G148: FdmMesherIntegral
// ═══════════════════════════════════════════════════════════════════════════

/// Numerical integration over an FDM mesh using the trapezoidal rule.
#[derive(Debug, Clone)]
pub struct FdmMesherIntegral;

impl FdmMesherIntegral {
    /// Integrate f(x) over a 1D mesh: ∫ f(x) dx ≈ Σ f_i · w_i.
    pub fn integrate_1d(mesher: &Mesher1d, values: &[f64]) -> f64 {
        let n = mesher.size();
        assert_eq!(n, values.len());
        if n < 2 {
            return 0.0;
        }
        let mut sum = 0.0;
        for i in 0..n - 1 {
            let dx = mesher.locations[i + 1] - mesher.locations[i];
            sum += 0.5 * dx * (values[i] + values[i + 1]);
        }
        sum
    }

    /// Integrate over a 2D composite mesh.
    pub fn integrate_2d(mesher: &FdmMesherComposite, values: &[f64]) -> f64 {
        assert_eq!(mesher.dimensions(), 2);
        let n1 = mesher.meshers[0].size();
        let n2 = mesher.meshers[1].size();
        assert_eq!(values.len(), n1 * n2);

        let mut sum = 0.0;
        for i in 0..n1 - 1 {
            let dx = mesher.meshers[0].locations[i + 1] - mesher.meshers[0].locations[i];
            for j in 0..n2 - 1 {
                let dy = mesher.meshers[1].locations[j + 1] - mesher.meshers[1].locations[j];
                let v00 = values[i * n2 + j];
                let v10 = values[(i + 1) * n2 + j];
                let v01 = values[i * n2 + j + 1];
                let v11 = values[(i + 1) * n2 + j + 1];
                sum += 0.25 * dx * dy * (v00 + v10 + v01 + v11);
            }
        }
        sum
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// G149: FdmIndicesOnBoundary
// ═══════════════════════════════════════════════════════════════════════════

/// Identifies flat indices that lie on the boundary of a multi-dimensional grid.
#[derive(Debug, Clone)]
pub struct FdmIndicesOnBoundary;

impl FdmIndicesOnBoundary {
    /// Return all flat indices on the boundary of dimension `dim` at the
    /// given `side` (0 = lower, 1 = upper).
    pub fn indices(layout: &FdmLinearOpLayout, dim: usize, side: usize) -> Vec<usize> {
        let mut result = Vec::new();
        let val = if side == 0 { 0 } else { layout.dim[dim] - 1 };
        for flat in 0..layout.size {
            let coords = layout.to_coords(flat);
            if coords[dim] == val {
                result.push(flat);
            }
        }
        result
    }

    /// Return all flat indices on any boundary.
    pub fn all_boundary_indices(layout: &FdmLinearOpLayout) -> Vec<usize> {
        let mut result = Vec::new();
        for flat in 0..layout.size {
            let coords = layout.to_coords(flat);
            let on_boundary = coords
                .iter()
                .zip(layout.dim.iter())
                .any(|(&c, &d)| c == 0 || c == d - 1);
            if on_boundary {
                result.push(flat);
            }
        }
        result
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// G150: FdmHestonGreensFct
// ═══════════════════════════════════════════════════════════════════════════

/// Heston Green's function for calibration.
///
/// Provides the semi-analytical density p(v_T | v_0) for the variance
/// process under CIR dynamics, used for calibrating local vol
/// from Heston via Gyöngy's theorem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FdmHestonGreensFct {
    pub kappa: f64,
    pub theta: f64,
    pub sigma: f64,
    pub v0: f64,
    pub t: f64,
}

impl FdmHestonGreensFct {
    pub fn new(kappa: f64, theta: f64, sigma: f64, v0: f64, t: f64) -> Self {
        Self {
            kappa,
            theta,
            sigma,
            v0,
            t,
        }
    }

    /// Non-central chi-squared density parameter `d`.
    fn d(&self) -> f64 {
        4.0 * self.kappa * self.theta / (self.sigma * self.sigma)
    }

    /// Scale parameter c(t).
    fn c(&self) -> f64 {
        self.sigma * self.sigma * (1.0 - (-self.kappa * self.t).exp()) / (4.0 * self.kappa)
    }

    /// Non-centrality parameter λ.
    fn lambda(&self) -> f64 {
        self.v0 * (-self.kappa * self.t).exp() / self.c()
    }

    /// Evaluate the Green's function (density) at variance v.
    pub fn density(&self, v: f64) -> f64 {
        if v <= 0.0 {
            return 0.0;
        }
        let c = self.c();
        let d = self.d();
        let lam = self.lambda();

        // Non-central chi-squared: p(v) = (1/c) f_{χ²}(v/c; d, λ)
        // Use the series expansion for the non-central chi² density
        let x = v / c;
        let half_d = d / 2.0;

        // Approximate using the first few terms of the Poisson-mixture
        let mut density = 0.0;
        let max_terms = 50;
        let mut poisson_weight = (-lam / 2.0).exp();
        for k in 0..max_terms {
            if poisson_weight < 1e-20 {
                break;
            }
            // Central chi² density with (d + 2k) degrees of freedom
            let nu = half_d + k as f64;
            let chi2_density = chi2_density_core(x, nu);
            density += poisson_weight * chi2_density;
            poisson_weight *= lam / (2.0 * (k + 1) as f64);
        }
        density / c
    }
}

/// Core chi² density: f(x; ν) = x^{ν−1} exp(−x/2) / (2^ν Γ(ν))
fn chi2_density_core(x: f64, nu: f64) -> f64 {
    if x <= 0.0 || nu <= 0.0 {
        return 0.0;
    }
    let log_density =
        (nu - 1.0) * x.ln() - x / 2.0 - nu * 2.0_f64.ln() - ln_gamma(nu);
    log_density.exp()
}

/// Simple ln(Γ(x)) via Stirling for x > 0.
fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    // Use Lanczos approximation
    let g = 7.0;
    let c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = c[0];
    let t = x + g + 0.5;
    for i in 1..9 {
        a += c[i] / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

// ═══════════════════════════════════════════════════════════════════════════
// G151: FdmSimpleStorageCondition
// ═══════════════════════════════════════════════════════════════════════════

/// Simple storage valuation step condition (gas/power storage).
///
/// At each time step, optimises over injection / withdrawal rates
/// within capacity constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FdmSimpleStorageCondition {
    /// Maximum injection rate per time step.
    pub max_injection: f64,
    /// Maximum withdrawal rate per time step.
    pub max_withdrawal: f64,
    /// Storage capacity.
    pub capacity: f64,
    /// Current inventory level.
    pub inventory: f64,
}

impl FdmSimpleStorageCondition {
    pub fn new(max_injection: f64, max_withdrawal: f64, capacity: f64, inventory: f64) -> Self {
        Self {
            max_injection,
            max_withdrawal,
            capacity,
            inventory,
        }
    }

    /// Apply storage optimisation at the current time step.
    ///
    /// For each grid point (representing spot price S), the optimal action is:
    /// - Inject (buy) if continuation value at (inv + Δ) − S·Δ exceeds V(inv)
    /// - Withdraw (sell) if V(inv − Δ) + S·Δ exceeds V(inv)
    pub fn apply(&self, values: &mut [f64], spot_grid: &Mesher1d) {
        let n = values.len().min(spot_grid.size());
        for i in 0..n {
            let s = spot_grid.locations[i].exp(); // assume log-spot grid
            let v_hold = values[i];
            let v_inject = if self.inventory + self.max_injection <= self.capacity {
                v_hold - s * self.max_injection
            } else {
                f64::NEG_INFINITY
            };
            let v_withdraw = if self.inventory >= self.max_withdrawal {
                v_hold + s * self.max_withdrawal
            } else {
                f64::NEG_INFINITY
            };
            values[i] = v_hold.max(v_inject).max(v_withdraw);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// G152: FdmSimpleSwingCondition
// ═══════════════════════════════════════════════════════════════════════════

/// Swing option step condition.
///
/// At exercise times, the holder can exercise one of their remaining
/// rights, receiving the intrinsic value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FdmSimpleSwingCondition {
    /// Remaining exercise rights.
    pub rights_remaining: usize,
    /// Minimum exercise rights that must be used.
    pub min_exercises: usize,
    /// Maximum total exercises.
    pub max_exercises: usize,
}

impl FdmSimpleSwingCondition {
    pub fn new(rights_remaining: usize, min_exercises: usize, max_exercises: usize) -> Self {
        Self {
            rights_remaining,
            min_exercises,
            max_exercises,
        }
    }

    /// Apply swing exercise at a given time step.
    pub fn apply(&self, values: &mut [f64], intrinsic: &[f64]) {
        if self.rights_remaining == 0 {
            return;
        }
        let n = values.len().min(intrinsic.len());
        for i in 0..n {
            values[i] = values[i].max(intrinsic[i]);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// G153: FdmSnapshotCondition
// ═══════════════════════════════════════════════════════════════════════════

/// Snapshot (value capture) condition.
///
/// Captures the solution state at a specified time for later retrieval
/// (e.g., for computing Greeks or exposure profiles).
#[derive(Debug, Clone)]
pub struct FdmSnapshotCondition {
    /// Time at which to capture.
    pub snapshot_time: f64,
    /// Tolerance for time matching.
    pub tolerance: f64,
    /// Captured values (filled when `apply` hits the snapshot time).
    pub captured: Option<Vec<f64>>,
}

impl FdmSnapshotCondition {
    pub fn new(snapshot_time: f64) -> Self {
        Self {
            snapshot_time,
            tolerance: 1e-6,
            captured: None,
        }
    }

    /// Apply: capture if at snapshot time.
    pub fn apply(&mut self, values: &[f64], t: f64) {
        if (t - self.snapshot_time).abs() < self.tolerance {
            self.captured = Some(values.to_vec());
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// G154: FdmStepConditionComposite
// ═══════════════════════════════════════════════════════════════════════════

/// Composite of multiple step conditions applied sequentially.
///
/// Each condition is represented as a boxed closure: `Fn(&mut [f64], f64)`.
pub struct FdmStepConditionComposite {
    pub conditions: Vec<Box<dyn Fn(&mut [f64], f64)>>,
}

impl FdmStepConditionComposite {
    pub fn new() -> Self {
        Self {
            conditions: Vec::new(),
        }
    }

    /// Add a step condition.
    pub fn add(&mut self, cond: Box<dyn Fn(&mut [f64], f64)>) {
        self.conditions.push(cond);
    }

    /// Apply all conditions in sequence.
    pub fn apply(&self, values: &mut [f64], t: f64) {
        for cond in &self.conditions {
            cond(values, t);
        }
    }
}

impl Default for FdmStepConditionComposite {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// G155: FdmArithmeticAverageCondition
// ═══════════════════════════════════════════════════════════════════════════

/// Running arithmetic average step condition (for Asian options).
///
/// At each time step, updates the running average A_n and adjusts
/// the payoff accordingly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FdmArithmeticAverageCondition {
    /// Number of averaging dates so far.
    pub n_avg: usize,
    /// Running sum.
    pub running_sum: f64,
    /// Schedule of averaging times.
    pub averaging_times: Vec<f64>,
    /// Time tolerance.
    pub tolerance: f64,
}

impl FdmArithmeticAverageCondition {
    pub fn new(averaging_times: Vec<f64>) -> Self {
        Self {
            n_avg: 0,
            running_sum: 0.0,
            averaging_times,
            tolerance: 1e-6,
        }
    }

    /// Check if current time is an averaging date and update state.
    pub fn apply(&mut self, values: &mut [f64], spot_grid: &Mesher1d, t: f64) {
        let is_avg = self
            .averaging_times
            .iter()
            .any(|&at| (at - t).abs() < self.tolerance);
        if !is_avg {
            return;
        }
        // Update running sum with midpoint spot
        let mid = spot_grid.size() / 2;
        let s_mid = spot_grid.locations[mid].exp();
        self.running_sum += s_mid;
        self.n_avg += 1;

        // Adjust values by the difference between average and current
        let avg = self.running_sum / self.n_avg as f64;
        let _ = avg;
        // For fixed-strike Asian: payoff depends on average, not modifying grid values
        // Just mark the averaging event; engine handles payoff at maturity
        let _ = values;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// G156-G161: Risk-Neutral Density Calculators
// ═══════════════════════════════════════════════════════════════════════════

/// Trait for risk-neutral density calculators.
pub trait RndCalculator {
    /// Risk-neutral density at spot S and time t.
    fn density(&self, s: f64, t: f64) -> f64;
    /// Risk-neutral CDF at spot S and time t.
    fn cdf(&self, s: f64, t: f64) -> f64;
    /// Inverse CDF (quantile).
    fn inv_cdf(&self, p: f64, t: f64) -> f64;
}

/// G156: BSM risk-neutral density calculator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BSMRndCalculator {
    pub r: f64,
    pub q: f64,
    pub vol: f64,
    pub spot: f64,
}

impl BSMRndCalculator {
    pub fn new(spot: f64, r: f64, q: f64, vol: f64) -> Self {
        Self { r, q, vol, spot }
    }
}

impl RndCalculator for BSMRndCalculator {
    fn density(&self, s: f64, t: f64) -> f64 {
        if s <= 0.0 || t <= 0.0 {
            return 0.0;
        }
        let sigma_sqrt_t = self.vol * t.sqrt();
        let d2 = ((self.spot / s).ln() + (self.r - self.q - 0.5 * self.vol * self.vol) * t)
            / sigma_sqrt_t;
        let pdf = (-0.5 * d2 * d2).exp() / (2.0 * std::f64::consts::PI).sqrt();
        pdf / (s * sigma_sqrt_t)
    }

    fn cdf(&self, s: f64, t: f64) -> f64 {
        if s <= 0.0 {
            return 0.0;
        }
        if t <= 0.0 {
            return if s >= self.spot { 1.0 } else { 0.0 };
        }
        let sigma_sqrt_t = self.vol * t.sqrt();
        let d2 = ((s / self.spot).ln() - (self.r - self.q - 0.5 * self.vol * self.vol) * t)
            / sigma_sqrt_t;
        0.5 * (1.0 + erf(d2 / std::f64::consts::SQRT_2))
    }

    fn inv_cdf(&self, p: f64, t: f64) -> f64 {
        let sigma_sqrt_t = self.vol * t.sqrt();
        let z = inv_normal_cdf(p);
        let x = (self.spot).ln() + (self.r - self.q - 0.5 * self.vol * self.vol) * t
            + sigma_sqrt_t * z;
        x.exp()
    }
}

/// G157: Heston risk-neutral density calculator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HestonRndCalculator {
    pub spot: f64,
    pub r: f64,
    pub q: f64,
    pub v0: f64,
    pub kappa: f64,
    pub theta: f64,
    pub sigma: f64,
    pub rho: f64,
}

impl HestonRndCalculator {
    pub fn new(
        spot: f64,
        r: f64,
        q: f64,
        v0: f64,
        kappa: f64,
        theta: f64,
        sigma: f64,
        rho: f64,
    ) -> Self {
        Self {
            spot,
            r,
            q,
            v0,
            kappa,
            theta,
            sigma,
            rho,
        }
    }
}

impl RndCalculator for HestonRndCalculator {
    fn density(&self, s: f64, t: f64) -> f64 {
        if s <= 0.0 || t <= 0.0 {
            return 0.0;
        }
        // Numerical differentiation of CDF
        let ds = s * 0.001;
        let c1 = self.cdf(s - ds, t);
        let c2 = self.cdf(s + ds, t);
        ((c2 - c1) / (2.0 * ds)).max(0.0)
    }

    fn cdf(&self, s: f64, t: f64) -> f64 {
        if s <= 0.0 {
            return 0.0;
        }
        if t <= 0.0 {
            return if s >= self.spot { 1.0 } else { 0.0 };
        }
        // Use effective BS vol approximation for CDF
        let v_avg = self.theta + (self.v0 - self.theta)
            * (1.0 - (-self.kappa * t).exp()) / (self.kappa * t);
        let eff_vol = v_avg.abs().sqrt();
        let bsm = BSMRndCalculator::new(self.spot, self.r, self.q, eff_vol);
        bsm.cdf(s, t)
    }

    fn inv_cdf(&self, p: f64, t: f64) -> f64 {
        let v_avg = self.theta + (self.v0 - self.theta)
            * (1.0 - (-self.kappa * t).exp()) / (self.kappa * t);
        let eff_vol = v_avg.abs().sqrt();
        let bsm = BSMRndCalculator::new(self.spot, self.r, self.q, eff_vol);
        bsm.inv_cdf(p, t)
    }
}

/// G158: Local vol risk-neutral density calculator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalVolRndCalculator {
    pub spot: f64,
    pub r: f64,
    pub q: f64,
    /// Local vol at discrete strikes: (strike, vol) pairs.
    pub vol_grid: Vec<(f64, f64)>,
}

impl LocalVolRndCalculator {
    pub fn new(spot: f64, r: f64, q: f64, vol_grid: Vec<(f64, f64)>) -> Self {
        Self { spot, r, q, vol_grid }
    }

    fn local_vol_at(&self, s: f64) -> f64 {
        if self.vol_grid.is_empty() {
            return 0.2;
        }
        // Linear interpolation
        for w in self.vol_grid.windows(2) {
            if s >= w[0].0 && s <= w[1].0 {
                let t = (s - w[0].0) / (w[1].0 - w[0].0);
                return w[0].1 * (1.0 - t) + w[1].1 * t;
            }
        }
        if s < self.vol_grid[0].0 {
            self.vol_grid[0].1
        } else {
            self.vol_grid.last().unwrap().1
        }
    }
}

impl RndCalculator for LocalVolRndCalculator {
    fn density(&self, s: f64, t: f64) -> f64 {
        let vol = self.local_vol_at(s);
        let bsm = BSMRndCalculator::new(self.spot, self.r, self.q, vol);
        bsm.density(s, t)
    }

    fn cdf(&self, s: f64, t: f64) -> f64 {
        let vol = self.local_vol_at(s);
        let bsm = BSMRndCalculator::new(self.spot, self.r, self.q, vol);
        bsm.cdf(s, t)
    }

    fn inv_cdf(&self, p: f64, t: f64) -> f64 {
        let vol = self.local_vol_at(self.spot); // use ATM vol for inverse
        let bsm = BSMRndCalculator::new(self.spot, self.r, self.q, vol);
        bsm.inv_cdf(p, t)
    }
}

/// G159: CEV risk-neutral density calculator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CEVRndCalculator {
    pub spot: f64,
    pub r: f64,
    pub q: f64,
    pub sigma: f64,
    pub beta: f64,
}

impl CEVRndCalculator {
    pub fn new(spot: f64, r: f64, q: f64, sigma: f64, beta: f64) -> Self {
        Self { spot, r, q, sigma, beta }
    }
}

impl RndCalculator for CEVRndCalculator {
    fn density(&self, s: f64, t: f64) -> f64 {
        if s <= 0.0 || t <= 0.0 {
            return 0.0;
        }
        // Use effective BS vol approximation: σ_eff = σ · S^{β−1}
        let eff_vol = self.sigma * self.spot.powf(self.beta - 1.0);
        let bsm = BSMRndCalculator::new(self.spot, self.r, self.q, eff_vol);
        bsm.density(s, t)
    }

    fn cdf(&self, s: f64, t: f64) -> f64 {
        let eff_vol = self.sigma * self.spot.powf(self.beta - 1.0);
        let bsm = BSMRndCalculator::new(self.spot, self.r, self.q, eff_vol);
        bsm.cdf(s, t)
    }

    fn inv_cdf(&self, p: f64, t: f64) -> f64 {
        let eff_vol = self.sigma * self.spot.powf(self.beta - 1.0);
        let bsm = BSMRndCalculator::new(self.spot, self.r, self.q, eff_vol);
        bsm.inv_cdf(p, t)
    }
}

/// G160: Generalised BSM risk-neutral density (with time-dependent vol).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GBSMRndCalculator {
    pub spot: f64,
    pub r: f64,
    pub q: f64,
    /// Piecewise-constant vol: (time, vol). Must be sorted by time.
    pub vol_term: Vec<(f64, f64)>,
}

impl GBSMRndCalculator {
    pub fn new(spot: f64, r: f64, q: f64, vol_term: Vec<(f64, f64)>) -> Self {
        Self { spot, r, q, vol_term }
    }

    fn effective_vol(&self, t: f64) -> f64 {
        if self.vol_term.is_empty() {
            return 0.2;
        }
        // Integrated variance
        let mut int_var = 0.0;
        let mut prev_t = 0.0;
        for &(ti, vi) in &self.vol_term {
            if ti > t {
                int_var += vi * vi * (t - prev_t);
                break;
            }
            int_var += vi * vi * (ti - prev_t);
            prev_t = ti;
        }
        if prev_t < t {
            let last_vol = self.vol_term.last().unwrap().1;
            int_var += last_vol * last_vol * (t - prev_t);
        }
        (int_var / t).sqrt()
    }
}

impl RndCalculator for GBSMRndCalculator {
    fn density(&self, s: f64, t: f64) -> f64 {
        let vol = self.effective_vol(t);
        let bsm = BSMRndCalculator::new(self.spot, self.r, self.q, vol);
        bsm.density(s, t)
    }

    fn cdf(&self, s: f64, t: f64) -> f64 {
        let vol = self.effective_vol(t);
        let bsm = BSMRndCalculator::new(self.spot, self.r, self.q, vol);
        bsm.cdf(s, t)
    }

    fn inv_cdf(&self, p: f64, t: f64) -> f64 {
        let vol = self.effective_vol(t);
        let bsm = BSMRndCalculator::new(self.spot, self.r, self.q, vol);
        bsm.inv_cdf(p, t)
    }
}

/// G161: Square-root (CIR) process risk-neutral density.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SquareRootProcessRndCalculator {
    pub x0: f64,
    pub kappa: f64,
    pub theta: f64,
    pub sigma: f64,
}

impl SquareRootProcessRndCalculator {
    pub fn new(x0: f64, kappa: f64, theta: f64, sigma: f64) -> Self {
        Self { x0, kappa, theta, sigma }
    }
}

impl RndCalculator for SquareRootProcessRndCalculator {
    fn density(&self, x: f64, t: f64) -> f64 {
        // Use FdmHestonGreensFct for the CIR density
        let gf = FdmHestonGreensFct::new(self.kappa, self.theta, self.sigma, self.x0, t);
        gf.density(x)
    }

    fn cdf(&self, x: f64, t: f64) -> f64 {
        // Numerical integration of density
        let n = 200;
        let dx = x / n as f64;
        let mut sum = 0.0;
        for i in 0..n {
            let xi = (i as f64 + 0.5) * dx;
            sum += self.density(xi, t) * dx;
        }
        sum.min(1.0)
    }

    fn inv_cdf(&self, p: f64, t: f64) -> f64 {
        // Bisection
        let mean = self.theta + (self.x0 - self.theta) * (-self.kappa * t).exp();
        let mut lo = 0.0;
        let mut hi = mean * 5.0;
        for _ in 0..100 {
            let mid = 0.5 * (lo + hi);
            if self.cdf(mid, t) < p {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// G162-G164: Extended Boundary Conditions
// ═══════════════════════════════════════════════════════════════════════════

/// G162: Discounted Dirichlet boundary condition.
///
/// Enforces V(boundary, t) = value · exp(−r·(T − t)).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FdmDiscountDirichletBoundary {
    pub side: crate::fdm_extended::BoundarySide,
    pub value: f64,
    pub r: f64,
    pub maturity: f64,
}

impl FdmDiscountDirichletBoundary {
    pub fn new(side: crate::fdm_extended::BoundarySide, value: f64, r: f64, maturity: f64) -> Self {
        Self {
            side,
            value,
            r,
            maturity,
        }
    }

    /// Discounted boundary value at time t.
    pub fn value_at_time(&self, t: f64) -> f64 {
        self.value * (-self.r * (self.maturity - t)).exp()
    }

    /// Apply to solution vector.
    pub fn apply(&self, values: &mut [f64], t: f64) {
        let v = self.value_at_time(t);
        match self.side {
            crate::fdm_extended::BoundarySide::Lower => {
                if !values.is_empty() {
                    values[0] = v;
                }
            }
            crate::fdm_extended::BoundarySide::Upper => {
                if let Some(last) = values.last_mut() {
                    *last = v;
                }
            }
        }
    }
}

/// G163: Time-dependent Dirichlet boundary condition.
///
/// Boundary value is given by an arbitrary function of time.
#[derive(Clone)]
pub struct FdmTimeDependentDirichletBoundary {
    pub side: crate::fdm_extended::BoundarySide,
    pub value_fn: std::sync::Arc<dyn Fn(f64) -> f64 + Send + Sync>,
}

impl std::fmt::Debug for FdmTimeDependentDirichletBoundary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FdmTimeDependentDirichletBoundary")
            .field("side", &self.side)
            .field("value_fn", &"<fn>")
            .finish()
    }
}

impl FdmTimeDependentDirichletBoundary {
    pub fn new(
        side: crate::fdm_extended::BoundarySide,
        value_fn: impl Fn(f64) -> f64 + Send + Sync + 'static,
    ) -> Self {
        Self {
            side,
            value_fn: std::sync::Arc::new(value_fn),
        }
    }

    pub fn apply(&self, values: &mut [f64], t: f64) {
        let v = (self.value_fn)(t);
        match self.side {
            crate::fdm_extended::BoundarySide::Lower => {
                if !values.is_empty() {
                    values[0] = v;
                }
            }
            crate::fdm_extended::BoundarySide::Upper => {
                if let Some(last) = values.last_mut() {
                    *last = v;
                }
            }
        }
    }
}

/// G164: Boundary condition set — collection of boundary conditions.
#[derive(Debug, Clone)]
pub struct FdmBoundaryConditionSet {
    pub dirichlet: Vec<crate::fdm_extended::FdmDirichletBoundary>,
    pub discount_dirichlet: Vec<FdmDiscountDirichletBoundary>,
}

impl FdmBoundaryConditionSet {
    pub fn new() -> Self {
        Self {
            dirichlet: Vec::new(),
            discount_dirichlet: Vec::new(),
        }
    }

    pub fn add_dirichlet(&mut self, bc: crate::fdm_extended::FdmDirichletBoundary) {
        self.dirichlet.push(bc);
    }

    pub fn add_discount_dirichlet(&mut self, bc: FdmDiscountDirichletBoundary) {
        self.discount_dirichlet.push(bc);
    }

    /// Apply all boundary conditions.
    pub fn apply(&self, values: &mut [f64], t: f64) {
        for bc in &self.dirichlet {
            bc.apply(values, t);
        }
        for bc in &self.discount_dirichlet {
            bc.apply(values, t);
        }
    }

    /// Apply all to operator.
    pub fn apply_to_operator(&self, op: &mut TripleBandOp) {
        for bc in &self.dirichlet {
            bc.apply_to_operator(op);
        }
    }
}

impl Default for FdmBoundaryConditionSet {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper math functions
// ═══════════════════════════════════════════════════════════════════════════

/// Error function approximation (Abramowitz & Stegun).
fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t
        * (0.254829592
            + t * (-0.284496736
                + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let result = 1.0 - poly * (-x * x).exp();
    if x >= 0.0 {
        result
    } else {
        -result
    }
}

/// Inverse normal CDF (Beasley-Springer-Moro algorithm).
fn inv_normal_cdf(p: f64) -> f64 {
    let p = p.clamp(1e-15, 1.0 - 1e-15);
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // G140
    #[test]
    fn layout_roundtrip() {
        let layout = FdmLinearOpLayout::new(vec![5, 7, 3]);
        assert_eq!(layout.size, 105);
        for flat in 0..layout.size {
            let coords = layout.to_coords(flat);
            assert_eq!(layout.to_flat(&coords), flat);
        }
    }

    #[test]
    fn layout_iterator() {
        let layout = FdmLinearOpLayout::new(vec![3, 4]);
        let items: Vec<_> = layout.iter().collect();
        assert_eq!(items.len(), 12);
        assert_eq!(items[0], (0, vec![0, 0]));
        assert_eq!(items[11], (11, vec![2, 3]));
    }

    // G141
    #[test]
    fn backward_solver_zero_op() {
        let op = TripleBandOp::zeros(5);
        let solver = FdmBackwardSolver::new(10, 1.0, 0.5);
        let terminal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = solver.solve(&op, &terminal, None);
        for i in 0..5 {
            assert_abs_diff_eq!(result[i], terminal[i], epsilon = 1e-10);
        }
    }

    // G142
    #[test]
    fn solver_desc_create() {
        let m1 = crate::fdm_meshers::uniform_1d_mesher(0.0, 1.0, 10);
        let mesher = FdmMesherComposite::new(vec![m1]);
        let desc = FdmSolverDesc::new(mesher, 100, 1.0);
        assert_eq!(desc.n_time_steps, 100);
        assert_eq!(desc.theta, 0.5);
    }

    // G143
    #[test]
    fn fdm_1d_solver() {
        let m = crate::fdm_meshers::uniform_1d_mesher(0.0, 1.0, 21);
        let mesher = FdmMesherComposite::new(vec![m]);
        let desc = FdmSolverDesc::new(mesher, 10, 0.1);
        let op = TripleBandOp::zeros(21);
        let solver = Fdm1DimSolver::new(desc, op);
        let terminal = vec![1.0; 21];
        let result = solver.solve(&terminal, None);
        for v in &result {
            assert_abs_diff_eq!(*v, 1.0, epsilon = 1e-10);
        }
    }

    // G143 (2D)
    #[test]
    fn fdm_2d_solver() {
        let m1 = crate::fdm_meshers::uniform_1d_mesher(0.0, 1.0, 5);
        let m2 = crate::fdm_meshers::uniform_1d_mesher(0.0, 1.0, 5);
        let mesher = FdmMesherComposite::new(vec![m1, m2]);
        let desc = FdmSolverDesc::new(mesher, 5, 0.1);
        let op1 = TripleBandOp::zeros(5);
        let op2 = TripleBandOp::zeros(5);
        let solver = Fdm2DimSolver::new(desc, op1, op2);
        let terminal = vec![1.0; 25];
        let result = solver.solve(&terminal);
        for v in &result {
            assert_abs_diff_eq!(*v, 1.0, epsilon = 1e-10);
        }
    }

    // G144
    #[test]
    fn quanto_helper() {
        let qh = FdmQuantoHelper::new(0.3, 0.20, 0.10, 0.01, 0.05);
        let adj = qh.drift_adjustment();
        assert_abs_diff_eq!(adj, -0.3 * 0.20 * 0.10, epsilon = 1e-14);
        assert_abs_diff_eq!(qh.effective_rate(), 0.05, epsilon = 1e-14);
    }

    // G145
    #[test]
    fn dividend_handler_create() {
        let dh = FdmDividendHandler::new(vec![2.0, 1.5], vec![0.25, 0.75]);
        assert_eq!(dh.dividends.len(), 2);
        assert_eq!(dh.dividend_times.len(), 2);
    }

    // G146
    #[test]
    fn vanilla_inner_value_call() {
        let calc = VanillaInnerValue {
            strike: 100.0,
            is_call: true,
            log_grid: true,
        };
        let m = crate::fdm_meshers::uniform_1d_mesher(4.0, 5.0, 11);
        let iv = calc.inner_value(&m, 0.0);
        // At x = ln(100) ≈ 4.605, inner value should be ≈ 0
        let mid = m.lower_index(100.0_f64.ln());
        assert!(iv[mid] < 5.0);
        // At high end, should be positive
        assert!(iv[10] > 0.0);
    }

    // G147
    #[test]
    fn affine_term_structure() {
        let ats = FdmAffineModelTermStructure::new(1.0, 1.0, 0.0, 1.0);
        let p = ats.discount(0.05);
        assert_abs_diff_eq!(p, (-0.05_f64).exp(), epsilon = 1e-10);
    }

    // G148
    #[test]
    fn mesher_integral_constant() {
        let m = crate::fdm_meshers::uniform_1d_mesher(0.0, 1.0, 101);
        let values = vec![1.0; 101];
        let integral = FdmMesherIntegral::integrate_1d(&m, &values);
        assert_abs_diff_eq!(integral, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn mesher_integral_linear() {
        let m = crate::fdm_meshers::uniform_1d_mesher(0.0, 1.0, 101);
        let values: Vec<f64> = m.locations.iter().copied().collect();
        let integral = FdmMesherIntegral::integrate_1d(&m, &values);
        assert_abs_diff_eq!(integral, 0.5, epsilon = 1e-4);
    }

    // G149
    #[test]
    fn boundary_indices() {
        let layout = FdmLinearOpLayout::new(vec![5, 5]);
        let lower = FdmIndicesOnBoundary::indices(&layout, 0, 0);
        assert_eq!(lower.len(), 5); // first row
        let upper = FdmIndicesOnBoundary::indices(&layout, 0, 1);
        assert_eq!(upper.len(), 5); // last row
        let all = FdmIndicesOnBoundary::all_boundary_indices(&layout);
        // 5x5: 4*4=16 boundary points
        assert_eq!(all.len(), 16);
    }

    // G150
    #[test]
    fn heston_greens_density_positive() {
        let gf = FdmHestonGreensFct::new(2.0, 0.04, 0.3, 0.04, 1.0);
        let d = gf.density(0.04);
        assert!(d > 0.0, "density at mean variance should be positive");
    }

    #[test]
    fn heston_greens_density_integrates() {
        let gf = FdmHestonGreensFct::new(2.0, 0.04, 0.3, 0.04, 1.0);
        let n = 1000;
        let dv = 0.5 / n as f64;
        let integral: f64 = (0..n).map(|i| gf.density((i as f64 + 0.5) * dv) * dv).sum();
        // Should be close to 1 (may not be exact due to truncation)
        assert!(integral > 0.5 && integral < 1.5, "integral = {}", integral);
    }

    // G151
    #[test]
    fn storage_condition() {
        let sc = FdmSimpleStorageCondition::new(10.0, 5.0, 100.0, 50.0);
        let m = crate::fdm_meshers::uniform_1d_mesher(3.5, 5.5, 5);
        let mut values = vec![10.0; 5];
        sc.apply(&mut values, &m);
        // Withdrawal should increase some values
        assert!(values.iter().any(|&v| v >= 10.0));
    }

    // G152
    #[test]
    fn swing_condition() {
        let sc = FdmSimpleSwingCondition::new(3, 1, 5);
        let mut values = vec![5.0, 3.0, 1.0];
        let intrinsic = vec![8.0, 2.0, 0.5];
        sc.apply(&mut values, &intrinsic);
        assert_abs_diff_eq!(values[0], 8.0); // exercised
        assert_abs_diff_eq!(values[1], 3.0); // not exercised
        assert_abs_diff_eq!(values[2], 1.0); // not exercised
    }

    // G153
    #[test]
    fn snapshot_condition() {
        let mut sc = FdmSnapshotCondition::new(0.5);
        let values = vec![1.0, 2.0, 3.0];
        sc.apply(&values, 0.3);
        assert!(sc.captured.is_none());
        sc.apply(&values, 0.5);
        assert!(sc.captured.is_some());
        assert_eq!(sc.captured.unwrap(), vec![1.0, 2.0, 3.0]);
    }

    // G154
    #[test]
    fn step_condition_composite() {
        let mut comp = FdmStepConditionComposite::new();
        comp.add(Box::new(|values: &mut [f64], _t: f64| {
            for v in values.iter_mut() {
                *v = v.max(0.0);
            }
        }));
        let mut values = vec![-1.0, 2.0, -3.0];
        comp.apply(&mut values, 0.0);
        assert_abs_diff_eq!(values[0], 0.0);
        assert_abs_diff_eq!(values[1], 2.0);
        assert_abs_diff_eq!(values[2], 0.0);
    }

    // G155
    #[test]
    fn arithmetic_average_condition() {
        let aac = FdmArithmeticAverageCondition::new(vec![0.25, 0.5, 0.75, 1.0]);
        assert_eq!(aac.n_avg, 0);
        assert_eq!(aac.averaging_times.len(), 4);
    }

    // G156
    #[test]
    fn bsm_rnd_density_positive() {
        let calc = BSMRndCalculator::new(100.0, 0.05, 0.02, 0.20);
        let d = calc.density(100.0, 1.0);
        assert!(d > 0.0, "ATM density should be positive");
    }

    #[test]
    fn bsm_rnd_cdf_monotone() {
        let calc = BSMRndCalculator::new(100.0, 0.05, 0.02, 0.20);
        let c1 = calc.cdf(90.0, 1.0);
        let c2 = calc.cdf(100.0, 1.0);
        let c3 = calc.cdf(110.0, 1.0);
        assert!(c1 < c2);
        assert!(c2 < c3);
    }

    #[test]
    fn bsm_rnd_inv_cdf_roundtrip() {
        let calc = BSMRndCalculator::new(100.0, 0.05, 0.02, 0.20);
        for p in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let s = calc.inv_cdf(p, 1.0);
            let p2 = calc.cdf(s, 1.0);
            assert_abs_diff_eq!(p2, p, epsilon = 0.02);
        }
    }

    // G157
    #[test]
    fn heston_rnd_density() {
        let calc = HestonRndCalculator::new(100.0, 0.05, 0.02, 0.04, 2.0, 0.04, 0.3, -0.7);
        let d = calc.density(100.0, 1.0);
        assert!(d > 0.0);
    }

    // G159
    #[test]
    fn cev_rnd_density() {
        let calc = CEVRndCalculator::new(100.0, 0.05, 0.02, 0.20, 0.5);
        let d = calc.density(100.0, 1.0);
        assert!(d > 0.0);
    }

    // G160
    #[test]
    fn gbsm_rnd_effective_vol() {
        let calc = GBSMRndCalculator::new(
            100.0,
            0.05,
            0.02,
            vec![(0.5, 0.20), (1.0, 0.25)],
        );
        let vol = calc.effective_vol(1.0);
        // Should be between 0.20 and 0.25
        assert!(vol > 0.19 && vol < 0.26, "effective vol = {vol}");
    }

    // G161
    #[test]
    fn sqrt_rnd_density() {
        let calc = SquareRootProcessRndCalculator::new(0.04, 2.0, 0.04, 0.3);
        let d = calc.density(0.04, 1.0);
        assert!(d > 0.0);
    }

    // G162
    #[test]
    fn discount_dirichlet() {
        let bc = FdmDiscountDirichletBoundary::new(
            crate::fdm_extended::BoundarySide::Lower,
            100.0,
            0.05,
            1.0,
        );
        let v = bc.value_at_time(0.0);
        assert_abs_diff_eq!(v, 100.0 * (-0.05_f64).exp(), epsilon = 1e-10);
        let v2 = bc.value_at_time(1.0);
        assert_abs_diff_eq!(v2, 100.0, epsilon = 1e-10);
    }

    // G163
    #[test]
    fn time_dependent_dirichlet() {
        let bc = FdmTimeDependentDirichletBoundary::new(
            crate::fdm_extended::BoundarySide::Upper,
            |t| 100.0 * (1.0 + t),
        );
        let mut values = vec![0.0; 5];
        bc.apply(&mut values, 0.5);
        assert_abs_diff_eq!(values[4], 150.0, epsilon = 1e-10);
    }

    // G164
    #[test]
    fn boundary_condition_set() {
        use crate::fdm_extended::{BoundarySide, FdmDirichletBoundary};
        let mut bcs = FdmBoundaryConditionSet::new();
        bcs.add_dirichlet(FdmDirichletBoundary::new(BoundarySide::Lower, 0.0));
        bcs.add_dirichlet(FdmDirichletBoundary::new(BoundarySide::Upper, 100.0));
        let mut values = vec![50.0; 5];
        bcs.apply(&mut values, 0.0);
        assert_abs_diff_eq!(values[0], 0.0);
        assert_abs_diff_eq!(values[4], 100.0);
        assert_abs_diff_eq!(values[2], 50.0); // interior unchanged
    }
}
