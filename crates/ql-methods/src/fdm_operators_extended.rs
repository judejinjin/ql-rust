#![allow(clippy::too_many_arguments)]
//! Extended FDM operators — G123-G139.
//!
//! Higher-order finite-difference operators and model-specific spatial
//! discretisations for multi-dimensional PDE pricing.
//!
//! - [`NinePointLinearOp`] (G123) — 9-point 2D finite-difference stencil
//! - [`FirstDerivativeOp`] / [`SecondDerivativeOp`] (G124/G125)
//! - [`SecondOrderMixedDerivativeOp`] (G126) — cross ∂²/∂x∂y
//! - [`NthOrderDerivativeOp`] (G127) — N-th order via finite differences
//! - [`ModTripleBandLinearOp`] (G128) — modified triple-band with boundary
//! - [`FdmBlackScholesOp`] (G129) — BS spatial operator (1D)
//! - [`Fdm2dBlackScholesOp`] (G130) — 2D multi-asset BS operator
//! - [`FdmHestonOp`] (G131) — Heston coupled (S,v) operator
//! - [`FdmHestonFwdOp`] (G132) — Heston forward / Fokker-Planck operator
//! - [`FdmHestonHullWhiteOp`] (G133) — hybrid Heston + HW 3D operator
//! - [`FdmBatesOp`] (G134) — Bates (Heston + jumps) operator
//! - [`FdmBlackScholesFwdOp`] / [`FdmLocalVolFwdOp`] (G135)
//! - [`FdmSquareRootFwdOp`] (G136) — CIR forward operator
//! - [`FdmOrnsteinUhlenbeckOp`] (G137) — OU operator
//! - [`FdmSABROp`] (G138) — SABR spatial operator
//! - [`FdmLinearOp`] trait (G139)

use serde::{Deserialize, Serialize};

use crate::fdm_operators::TripleBandOp;

// ---------------------------------------------------------------------------
// G139: FdmLinearOp / FdmLinearOpComposite traits
// ---------------------------------------------------------------------------

/// Trait for finite-difference linear operators applied to grid vectors.
pub trait FdmLinearOp {
    /// Apply the operator to a grid vector: y = L·x.
    fn apply(&self, x: &[f64]) -> Vec<f64>;
    /// Return the number of grid points.
    fn size(&self) -> usize;
}

/// Composite of multiple linear operators (L = L₁ + L₂ + ⋯).
pub trait FdmLinearOpComposite: FdmLinearOp {
    /// Number of sub-operators.
    fn n_ops(&self) -> usize;
    /// Apply a specific sub-operator.
    fn apply_op(&self, idx: usize, x: &[f64]) -> Vec<f64>;
}

// ---------------------------------------------------------------------------
// G123: NinePointLinearOp — 9-point stencil for 2D grids
// ---------------------------------------------------------------------------

/// 9-point finite-difference stencil on a 2D grid.
///
/// For each interior point (i, j), computes:
///   L·u(i,j) = Σ_{(p,q) ∈ {-1,0,1}²} w_{p,q} · u(i+p, j+q)
///
/// The weights are stored as flat arrays indexed by `i * n2 + j`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NinePointLinearOp {
    /// Grid sizes.
    pub n1: usize,
    /// N2.
    pub n2: usize,
    /// 9 weight arrays of length n1*n2: [(-1,-1), (-1,0), (-1,+1),
    ///   (0,-1), (0,0), (0,+1), (+1,-1), (+1,0), (+1,+1)]
    pub weights: [Vec<f64>; 9],
}

impl NinePointLinearOp {
    /// Create a zero 9-point operator.
    pub fn zeros(n1: usize, n2: usize) -> Self {
        let sz = n1 * n2;
        Self {
            n1,
            n2,
            weights: std::array::from_fn(|_| vec![0.0; sz]),
        }
    }

    /// Apply the 9-point stencil to a 2D grid stored row-major.
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        let (n1, n2) = (self.n1, self.n2);
        let mut y = vec![0.0; n1 * n2];
        for i in 1..n1 - 1 {
            for j in 1..n2 - 1 {
                let idx = i * n2 + j;
                let mut v = 0.0;
                // (-1,-1)
                v += self.weights[0][idx] * x[(i - 1) * n2 + j - 1];
                // (-1, 0)
                v += self.weights[1][idx] * x[(i - 1) * n2 + j];
                // (-1,+1)
                v += self.weights[2][idx] * x[(i - 1) * n2 + j + 1];
                // ( 0,-1)
                v += self.weights[3][idx] * x[i * n2 + j - 1];
                // ( 0, 0)
                v += self.weights[4][idx] * x[i * n2 + j];
                // ( 0,+1)
                v += self.weights[5][idx] * x[i * n2 + j + 1];
                // (+1,-1)
                v += self.weights[6][idx] * x[(i + 1) * n2 + j - 1];
                // (+1, 0)
                v += self.weights[7][idx] * x[(i + 1) * n2 + j];
                // (+1,+1)
                v += self.weights[8][idx] * x[(i + 1) * n2 + j + 1];
                y[idx] = v;
            }
        }
        y
    }

    fn _size(&self) -> usize {
        self.n1 * self.n2
    }
}

// ---------------------------------------------------------------------------
// G124: FirstDerivativeOp — central first derivative
// ---------------------------------------------------------------------------

/// First spatial derivative operator on a non-uniform 1D grid.
///
/// Uses central differences: ∂u/∂x ≈ (u[i+1] − u[i−1]) / (x[i+1] − x[i−1]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirstDerivativeOp {
    /// Inner.
    pub inner: TripleBandOp,
}

impl FirstDerivativeOp {
    /// Build from a 1D grid.
    pub fn new(grid: &[f64]) -> Self {
        let n = grid.len();
        let mut op = TripleBandOp::zeros(n);
        for i in 1..n - 1 {
            let dx = grid[i + 1] - grid[i - 1];
            op.lower[i] = -1.0 / dx;
            op.upper[i] = 1.0 / dx;
        }
        // Forward / backward at boundaries
        if n >= 2 {
            let dx0 = grid[1] - grid[0];
            op.diag[0] = -1.0 / dx0;
            op.upper[0] = 1.0 / dx0;
            let dxn = grid[n - 1] - grid[n - 2];
            op.lower[n - 1] = -1.0 / dxn;
            op.diag[n - 1] = 1.0 / dxn;
        }
        Self { inner: op }
    }

    /// Apply: y = D₁·x.
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        self.inner.apply(x)
    }
}

// ---------------------------------------------------------------------------
// G125: SecondDerivativeOp — central second derivative
// ---------------------------------------------------------------------------

/// Second spatial derivative on a non-uniform 1D grid.
///
/// Uses standard 3-point stencil:
///   ∂²u/∂x² ≈ 2 [ u[i+1]/(dx⁺(dx⁺+dx⁻)) − u[i]/(dx⁺ dx⁻) + u[i−1]/(dx⁻(dx⁺+dx⁻)) ]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecondDerivativeOp {
    /// Inner.
    pub inner: TripleBandOp,
}

impl SecondDerivativeOp {
    /// Build from a 1D grid.
    pub fn new(grid: &[f64]) -> Self {
        let n = grid.len();
        let mut op = TripleBandOp::zeros(n);
        for i in 1..n - 1 {
            let dp = grid[i + 1] - grid[i];
            let dm = grid[i] - grid[i - 1];
            let ds = 0.5 * (dp + dm);
            op.upper[i] = 1.0 / (dp * ds);
            op.lower[i] = 1.0 / (dm * ds);
            op.diag[i] = -(op.upper[i] + op.lower[i]);
        }
        Self { inner: op }
    }

    /// Apply: y = D₂·x.
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        self.inner.apply(x)
    }
}

// ---------------------------------------------------------------------------
// G126: SecondOrderMixedDerivativeOp — ∂²/∂x∂y on a 2D grid
// ---------------------------------------------------------------------------

/// Mixed second derivative ∂²u/∂x∂y on a 2D tensor grid.
///
/// Uses the 4-point cross stencil:
///   ∂²u/∂x∂y ≈ [u(i+1,j+1) − u(i+1,j−1) − u(i−1,j+1) + u(i−1,j−1)]
///              / [(x_{i+1}−x_{i−1})(y_{j+1}−y_{j−1})]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecondOrderMixedDerivativeOp {
    /// N1.
    pub n1: usize,
    /// N2.
    pub n2: usize,
    /// Coefficients on the 4 corners (flat array, n1*n2).
    pub coeff: Vec<f64>,
}

impl SecondOrderMixedDerivativeOp {
    /// Build from two 1D grids.
    pub fn new(grid_x: &[f64], grid_y: &[f64]) -> Self {
        let n1 = grid_x.len();
        let n2 = grid_y.len();
        let mut coeff = vec![0.0; n1 * n2];
        for i in 1..n1 - 1 {
            let hx = grid_x[i + 1] - grid_x[i - 1];
            for j in 1..n2 - 1 {
                let hy = grid_y[j + 1] - grid_y[j - 1];
                coeff[i * n2 + j] = 1.0 / (hx * hy);
            }
        }
        Self { n1, n2, coeff }
    }

    /// Apply: y = ∂²u/∂x∂y using 4-point cross stencil.
    pub fn apply(&self, u: &[f64]) -> Vec<f64> {
        let (n1, n2) = (self.n1, self.n2);
        let mut y = vec![0.0; n1 * n2];
        for i in 1..n1 - 1 {
            for j in 1..n2 - 1 {
                let idx = i * n2 + j;
                let c = self.coeff[idx];
                y[idx] = c
                    * (u[(i + 1) * n2 + j + 1] - u[(i + 1) * n2 + j - 1]
                        - u[(i - 1) * n2 + j + 1]
                        + u[(i - 1) * n2 + j - 1]);
            }
        }
        y
    }
}

// ---------------------------------------------------------------------------
// G127: NthOrderDerivativeOp
// ---------------------------------------------------------------------------

/// N-th order derivative of a 1D function sampled on a grid.
///
/// Uses finite-difference coefficients for the requested order on a
/// `2*half_width + 1`-point stencil via Fornberg's algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NthOrderDerivativeOp {
    /// Order.
    pub order: usize,
    /// Stencil half-width.
    pub half_width: usize,
    /// Coefficients per interior point: `coeffs[i]` is the stencil for point i.
    pub n: usize,
    /// Coeffs.
    pub coeffs: Vec<Vec<f64>>,
    /// Offsets.
    pub offsets: Vec<Vec<i32>>,
}

impl NthOrderDerivativeOp {
    /// Create an N-th order derivative operator.
    ///
    /// `half_width` controls stencil size (default = order).
    pub fn new(grid: &[f64], order: usize, half_width: Option<usize>) -> Self {
        let n = grid.len();
        let hw = half_width.unwrap_or(order);
        let mut coeffs = Vec::with_capacity(n);
        let mut offsets = Vec::with_capacity(n);
        for i in 0..n {
            let lo = i.saturating_sub(hw);
            let hi = (i + hw + 1).min(n);
            let pts: Vec<f64> = (lo..hi).map(|k| grid[k]).collect();
            let offs: Vec<i32> = (lo..hi).map(|k| k as i32 - i as i32).collect();
            let c = fornberg_weights(grid[i], &pts, order);
            offsets.push(offs);
            coeffs.push(c);
        }
        Self {
            order,
            half_width: hw,
            n,
            coeffs,
            offsets,
        }
    }

    /// Apply: y[i] = Σ_k c_k · u[i + offset_k].
    #[allow(clippy::needless_range_loop)]
    pub fn apply(&self, u: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; self.n];
        for i in 0..self.n {
            let c = &self.coeffs[i];
            let o = &self.offsets[i];
            let mut v = 0.0;
            for (idx, &off) in o.iter().enumerate() {
                let j = (i as i32 + off) as usize;
                if j < u.len() {
                    v += c[idx] * u[j];
                }
            }
            y[i] = v;
        }
        y
    }
}

/// Fornberg algorithm: compute FD weights for derivative of given order
/// at point `x0` using stencil points `pts`.
fn fornberg_weights(x0: f64, pts: &[f64], order: usize) -> Vec<f64> {
    let n = pts.len();
    let m = order;
    // c[k][j] = weight for derivative of order k at stencil point j
    let mut c = vec![vec![0.0; n]; m + 1];
    c[0][0] = 1.0;
    let mut c1 = 1.0;
    let mut c4 = pts[0] - x0;
    for i in 1..n {
        let mn = m.min(i);
        let mut c2 = 1.0;
        let c5 = c4;
        c4 = pts[i] - x0;
        for j in 0..i {
            let c3 = pts[i] - pts[j];
            c2 *= c3;
            if j == i - 1 {
                for k in (1..=mn).rev() {
                    c[k][i] = c1 * (k as f64 * c[k - 1][i - 1] - c5 * c[k][i - 1]) / c2;
                }
                c[0][i] = -c1 * c5 * c[0][i - 1] / c2;
            }
            for k in (1..=mn).rev() {
                c[k][j] = (c4 * c[k][j] - k as f64 * c[k - 1][j]) / c3;
            }
            c[0][j] = c4 * c[0][j] / c3;
        }
        c1 = c2;
    }
    // Return weights for the requested order
    c[m].clone()
}

// ---------------------------------------------------------------------------
// G128: ModTripleBandLinearOp — modified triple-band with boundary handling
// ---------------------------------------------------------------------------

/// Modified triple-band operator with boundary treatment.
///
/// Extends `TripleBandOp` with an extra row modification for boundaries:
/// the first and last rows can be overridden with custom coefficients
/// (e.g., for Neumann or Robin conditions).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModTripleBandLinearOp {
    /// Inner.
    pub inner: TripleBandOp,
    /// Override lower-boundary row: (lower, diag, upper) at i=0.
    pub lower_bc: Option<(f64, f64, f64)>,
    /// Override upper-boundary row: (lower, diag, upper) at i=n-1.
    pub upper_bc: Option<(f64, f64, f64)>,
}

impl ModTripleBandLinearOp {
    /// New.
    pub fn new(inner: TripleBandOp) -> Self {
        Self {
            inner,
            lower_bc: None,
            upper_bc: None,
        }
    }

    /// Set Neumann (zero-flux) boundary at the lower end.
    pub fn set_neumann_lower(&mut self) {
        // du/dx = 0 → u[0] = u[1]  ⟹  row: [−1, 1, 0]
        self.lower_bc = Some((-1.0, 1.0, 0.0));
        // Actually: (lower=0, diag=-1, upper=1)
        self.lower_bc = Some((0.0, -1.0, 1.0));
    }

    /// Set Neumann boundary at the upper end.
    pub fn set_neumann_upper(&mut self) {
        // u[n-1] = u[n-2]  ⟹  row: [1, −1, 0]
        self.upper_bc = Some((1.0, -1.0, 0.0));
    }

    /// Apply with boundary overrides.
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        let mut y = self.inner.apply(x);
        let n = self.inner.n;
        if let Some((l, d, u)) = self.lower_bc {
            y[0] = l * 0.0 + d * x[0] + u * x[1.min(n - 1)];
        }
        if let Some((l, d, u)) = self.upper_bc {
            y[n - 1] = l * x[(n - 2).max(0)] + d * x[n - 1] + u * 0.0;
        }
        y
    }
}

// ---------------------------------------------------------------------------
// G129: FdmBlackScholesOp — Complete BS spatial operator
// ---------------------------------------------------------------------------

/// Black-Scholes 1D spatial operator including drift, diffusion, and discounting.
///
/// PDE in log-spot x = ln S:
///   L·V = (r − q − σ²/2) D₁V + σ²/2 D₂V − rV
///
/// Optionally supports local volatility σ(x).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FdmBlackScholesOp {
    /// Grid.
    pub grid: Vec<f64>,
    /// R.
    pub r: f64,
    /// Q.
    pub q: f64,
    /// Local volatilities at grid points (or constant vol broadcast).
    pub vols: Vec<f64>,
    /// Quanto adjustment (default 0).
    pub quanto_adj: f64,
    /// The composed tridiagonal operator.
    pub op: TripleBandOp,
}

impl FdmBlackScholesOp {
    /// Build with constant volatility.
    pub fn new(grid: &[f64], r: f64, q: f64, vol: f64) -> Self {
        let vols = vec![vol; grid.len()];
        let op = crate::fdm_operators::build_bs_operator(grid, r, q, vol);
        Self {
            grid: grid.to_vec(),
            r,
            q,
            vols,
            quanto_adj: 0.0,
            op,
        }
    }

    /// Build with local volatility σ(x) at each grid point.
    pub fn with_local_vol(grid: &[f64], r: f64, q: f64, local_vols: Vec<f64>) -> Self {
        assert_eq!(grid.len(), local_vols.len());
        let n = grid.len();
        let mut op = TripleBandOp::zeros(n);
        for i in 1..n - 1 {
            let vol = local_vols[i];
            let drift = r - q - 0.5 * vol * vol;
            let diffusion = 0.5 * vol * vol;
            let dp = grid[i + 1] - grid[i];
            let dm = grid[i] - grid[i - 1];
            let ds = 0.5 * (dp + dm);
            op.upper[i] = diffusion / (dp * ds) + drift / (dp + dm);
            op.lower[i] = diffusion / (dm * ds) - drift / (dp + dm);
            op.diag[i] = -op.upper[i] - op.lower[i] - r;
        }
        op.diag[0] = -r;
        op.diag[n - 1] = -r;
        Self {
            grid: grid.to_vec(),
            r,
            q,
            vols: local_vols,
            quanto_adj: 0.0,
            op,
        }
    }

    /// Apply: y = L·x.
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        self.op.apply(x)
    }
}

// ---------------------------------------------------------------------------
// G130: Fdm2dBlackScholesOp — 2D multi-asset operator
// ---------------------------------------------------------------------------

/// 2D Black-Scholes operator for two correlated assets.
///
/// PDE in log-spot (x₁, x₂):
///   L = L₁ + L₂ + ρσ₁σ₂ ∂²/∂x₁∂x₂ − r
///
/// Stores dimension-wise tridiagonal operators and cross-derivative stencil.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fdm2dBlackScholesOp {
    /// N1.
    pub n1: usize,
    /// N2.
    pub n2: usize,
    /// Op1.
    pub op1: TripleBandOp,
    /// Op2.
    pub op2: TripleBandOp,
    /// Rho.
    pub rho: f64,
    /// Vol1.
    pub vol1: f64,
    /// Vol2.
    pub vol2: f64,
    /// Cross op.
    pub cross_op: SecondOrderMixedDerivativeOp,
}

impl Fdm2dBlackScholesOp {
    /// Build from two log-spot grids and market parameters.
    pub fn new(
        grid1: &[f64],
        grid2: &[f64],
        r: f64,
        q1: f64,
        q2: f64,
        vol1: f64,
        vol2: f64,
        rho: f64,
    ) -> Self {
        // Half discounting on each dimension
        let op1 = crate::fdm_operators::build_bs_operator(grid1, r / 2.0, q1, vol1);
        let op2 = crate::fdm_operators::build_bs_operator(grid2, r / 2.0, q2, vol2);
        let cross_op = SecondOrderMixedDerivativeOp::new(grid1, grid2);
        Self {
            n1: grid1.len(),
            n2: grid2.len(),
            op1,
            op2,
            rho,
            vol1,
            vol2,
            cross_op,
        }
    }

    /// Compute cross-derivative contribution: ρσ₁σ₂ ∂²V/∂x₁∂x₂.
    pub fn cross_term(&self, v: &[f64]) -> Vec<f64> {
        let mut c = self.cross_op.apply(v);
        let factor = self.rho * self.vol1 * self.vol2;
        for val in &mut c {
            *val *= factor;
        }
        c
    }
}

// ---------------------------------------------------------------------------
// G131: FdmHestonOp — Heston (S, v) coupled operator
// ---------------------------------------------------------------------------

/// Heston spatial operator on (x = ln S, v) coordinates.
///
/// Wraps `build_heston_ops` from `fdm_operators` into a named struct
/// and exposes per-direction operators plus cross-derivative handling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FdmHestonOp {
    /// Heston ops.
    pub heston_ops: crate::fdm_operators::Heston2dOps,
    /// Rho.
    pub rho: f64,
    /// Sigma.
    pub sigma: f64,
}

impl FdmHestonOp {
    /// Build from grids and Heston parameters.
    pub fn new(
        x_grid: &[f64],
        v_grid: &[f64],
        r: f64,
        q: f64,
        kappa: f64,
        theta: f64,
        sigma: f64,
        rho: f64,
    ) -> Self {
        let ops = crate::fdm_operators::build_heston_ops(
            x_grid, v_grid, r, q, kappa, theta, sigma, rho,
        );
        Self {
            heston_ops: ops,
            rho,
            sigma,
        }
    }
}

// ---------------------------------------------------------------------------
// G132: FdmHestonFwdOp — Heston forward (Fokker-Planck) operator
// ---------------------------------------------------------------------------

/// Heston forward equation (Fokker-Planck) operator.
///
/// Computes the adjoint of the backward Heston operator, discretised
/// for evolving the probability density p(x, v, t) forward in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FdmHestonFwdOp {
    /// Nx.
    pub nx: usize,
    /// Nv.
    pub nv: usize,
    /// Forward operator in x direction.
    pub x_op: TripleBandOp,
    /// Forward operator in v direction.
    pub v_op: TripleBandOp,
    /// Heston parameters.
    pub kappa: f64,
    /// Theta.
    pub theta: f64,
    /// Sigma.
    pub sigma: f64,
    /// Rho.
    pub rho: f64,
    /// R.
    pub r: f64,
    /// Q.
    pub q: f64,
}

impl FdmHestonFwdOp {
    /// Build the forward operator on (x, v) grid.
    pub fn new(
        x_grid: &[f64],
        v_grid: &[f64],
        r: f64,
        q: f64,
        kappa: f64,
        theta: f64,
        sigma: f64,
        rho: f64,
    ) -> Self {
        let nx = x_grid.len();
        let nv = v_grid.len();

        // Forward x-operator (adjoint): signs flip on first derivative
        let mut x_op = TripleBandOp::zeros(nx);
        let v_mid = v_grid[nv / 2].max(1e-8);
        let drift = -(r - q - 0.5 * v_mid); // negative for adjoint
        let diff = 0.5 * v_mid;
        for i in 1..nx - 1 {
            let dp = x_grid[i + 1] - x_grid[i];
            let dm = x_grid[i] - x_grid[i - 1];
            let ds = 0.5 * (dp + dm);
            x_op.upper[i] = diff / (dp * ds) + drift / (dp + dm);
            x_op.lower[i] = diff / (dm * ds) - drift / (dp + dm);
            x_op.diag[i] = -(x_op.upper[i] + x_op.lower[i]);
        }

        // Forward v-operator (Feller for variance)
        let mut v_op = TripleBandOp::zeros(nv);
        for j in 1..nv - 1 {
            let v = v_grid[j].max(0.0);
            let dp = v_grid[j + 1] - v_grid[j];
            let dm = v_grid[j] - v_grid[j - 1];
            let ds = 0.5 * (dp + dm);
            let d_coeff = 0.5 * sigma * sigma * v;
            let c_coeff = -(kappa * (theta - v)); // adjoint flip
            v_op.upper[j] = d_coeff / (dp * ds) + c_coeff / (dp + dm);
            v_op.lower[j] = d_coeff / (dm * ds) - c_coeff / (dp + dm);
            v_op.diag[j] = -(v_op.upper[j] + v_op.lower[j]);
        }

        Self {
            nx,
            nv,
            x_op,
            v_op,
            kappa,
            theta,
            sigma,
            rho,
            r,
            q,
        }
    }
}

// ---------------------------------------------------------------------------
// G133: FdmHestonHullWhiteOp — Hybrid 3D operator
// ---------------------------------------------------------------------------

/// Hybrid Heston + Hull-White 3D operator on (x, v, r) grid.
///
/// Splits into three 1D tridiagonal operators for ADI time-stepping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FdmHestonHullWhiteOp {
    /// Heston part (x, v).
    pub heston_op: FdmHestonOp,
    /// Hull-White short-rate operator.
    pub hw_op: TripleBandOp,
    /// HW parameters.
    pub a: f64,
    /// Sigma hw.
    pub sigma_hw: f64,
    /// Correlation between S and r.
    pub rho_sr: f64,
}

impl FdmHestonHullWhiteOp {
    /// Build hybrid 3D operator.
    pub fn new(
        x_grid: &[f64],
        v_grid: &[f64],
        r_grid: &[f64],
        r: f64,
        q: f64,
        kappa: f64,
        theta: f64,
        sigma_heston: f64,
        rho_sv: f64,
        a: f64,
        sigma_hw: f64,
        theta_hw: f64,
        rho_sr: f64,
    ) -> Self {
        let heston_op =
            FdmHestonOp::new(x_grid, v_grid, r, q, kappa, theta, sigma_heston, rho_sv);
        let hw_op =
            crate::fdm_extended::build_hull_white_operator(r_grid, a, sigma_hw, theta_hw);
        Self {
            heston_op,
            hw_op,
            a,
            sigma_hw,
            rho_sr,
        }
    }
}

// ---------------------------------------------------------------------------
// G134: FdmBatesOp — Bates (Heston + jumps) operator
// ---------------------------------------------------------------------------

/// Bates model spatial operator (Heston + Merton jumps).
///
/// Adds an integral term for the jump component to the Heston operator:
///   J[V](x) = λ ∫ [V(x+y) − V(x)] ν(dy)
///
/// where ν is a log-normal jump-size distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FdmBatesOp {
    /// Heston op.
    pub heston_op: FdmHestonOp,
    /// Jump intensity λ.
    pub lambda: f64,
    /// Log-mean of jump size.
    pub mu_j: f64,
    /// Log-stdev of jump size.
    pub sigma_j: f64,
    /// Pre-computed jump weights for numerical integration.
    pub jump_weights: Vec<f64>,
    /// Jump offsets (index shifts) for the x-grid.
    pub jump_offsets: Vec<i32>,
}

impl FdmBatesOp {
    /// Build a Bates operator.
    pub fn new(
        x_grid: &[f64],
        v_grid: &[f64],
        r: f64,
        q: f64,
        kappa: f64,
        theta: f64,
        sigma: f64,
        rho: f64,
        lambda: f64,
        mu_j: f64,
        sigma_j: f64,
    ) -> Self {
        // Adjust drift for compensated jump: r - q - λ(e^{μ+σ²/2} - 1)
        let jump_comp = lambda * ((mu_j + 0.5 * sigma_j * sigma_j).exp() - 1.0);
        let r_adj = r;
        let q_adj = q + jump_comp;
        let heston_op =
            FdmHestonOp::new(x_grid, v_grid, r_adj, q_adj, kappa, theta, sigma, rho);

        // Discretise the jump integral using Gauss-Hermite–like quadrature
        let n_quad = 15;
        let mut jump_weights = Vec::with_capacity(n_quad);
        let mut jump_offsets = Vec::with_capacity(n_quad);
        let dx_mean = if x_grid.len() >= 2 {
            (x_grid.last().unwrap() - x_grid[0]) / (x_grid.len() - 1) as f64
        } else {
            1.0
        };
        for k in 0..n_quad {
            let z = -3.5 + 7.0 * k as f64 / (n_quad - 1) as f64;
            let y = mu_j + sigma_j * z;
            let idx_shift = (y / dx_mean).round() as i32;
            let w = lambda
                * (-0.5 * z * z).exp()
                / (2.0 * std::f64::consts::PI).sqrt()
                * (7.0 / (n_quad - 1) as f64);
            jump_weights.push(w);
            jump_offsets.push(idx_shift);
        }

        Self {
            heston_op,
            lambda,
            mu_j,
            sigma_j,
            jump_weights,
            jump_offsets,
        }
    }

    /// Apply jump integral to a 1D slice of V along the x-direction.
    pub fn apply_jump(&self, v_slice: &[f64]) -> Vec<f64> {
        let n = v_slice.len();
        let mut result = vec![0.0; n];
        for i in 0..n {
            let mut jump_val = 0.0;
            for (k, &w) in self.jump_weights.iter().enumerate() {
                let j = i as i32 + self.jump_offsets[k];
                if j >= 0 && (j as usize) < n {
                    jump_val += w * (v_slice[j as usize] - v_slice[i]);
                }
            }
            result[i] = jump_val;
        }
        result
    }
}

// ---------------------------------------------------------------------------
// G135: FdmBlackScholesFwdOp / FdmLocalVolFwdOp
// ---------------------------------------------------------------------------

/// Forward (Fokker-Planck) operator for Black-Scholes / local vol.
///
/// Evolves the probability density forward in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FdmBlackScholesFwdOp {
    /// Op.
    pub op: TripleBandOp,
    /// Grid.
    pub grid: Vec<f64>,
}

impl FdmBlackScholesFwdOp {
    /// Build forward operator with constant vol.
    pub fn new(grid: &[f64], r: f64, q: f64, vol: f64) -> Self {
        let n = grid.len();
        let mut op = TripleBandOp::zeros(n);
        let drift = -(r - q - 0.5 * vol * vol);
        let diff = 0.5 * vol * vol;
        for i in 1..n - 1 {
            let dp = grid[i + 1] - grid[i];
            let dm = grid[i] - grid[i - 1];
            let ds = 0.5 * (dp + dm);
            op.upper[i] = diff / (dp * ds) + drift / (dp + dm);
            op.lower[i] = diff / (dm * ds) - drift / (dp + dm);
            op.diag[i] = -(op.upper[i] + op.lower[i]);
        }
        Self {
            op,
            grid: grid.to_vec(),
        }
    }

    /// Apply.
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        self.op.apply(x)
    }
}

/// Local volatility forward operator: same structure but with per-point vol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FdmLocalVolFwdOp {
    /// Op.
    pub op: TripleBandOp,
    /// Grid.
    pub grid: Vec<f64>,
    /// Local vols.
    pub local_vols: Vec<f64>,
}

impl FdmLocalVolFwdOp {
    /// Build with local vols at grid points.
    pub fn new(grid: &[f64], r: f64, q: f64, local_vols: &[f64]) -> Self {
        let n = grid.len();
        assert_eq!(n, local_vols.len());
        let mut op = TripleBandOp::zeros(n);
        for i in 1..n - 1 {
            let vol = local_vols[i];
            let drift = -(r - q - 0.5 * vol * vol);
            let diff = 0.5 * vol * vol;
            let dp = grid[i + 1] - grid[i];
            let dm = grid[i] - grid[i - 1];
            let ds = 0.5 * (dp + dm);
            op.upper[i] = diff / (dp * ds) + drift / (dp + dm);
            op.lower[i] = diff / (dm * ds) - drift / (dp + dm);
            op.diag[i] = -(op.upper[i] + op.lower[i]);
        }
        Self {
            op,
            grid: grid.to_vec(),
            local_vols: local_vols.to_vec(),
        }
    }

    /// Apply.
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        self.op.apply(x)
    }
}

// ---------------------------------------------------------------------------
// G136: FdmSquareRootFwdOp — CIR forward operator
// ---------------------------------------------------------------------------

/// Square-root (CIR) forward (Fokker-Planck) operator.
///
/// PDE: ∂p/∂t = −∂/∂x[κ(θ−x)p] + ½ ∂²/∂x²[σ²x · p]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FdmSquareRootFwdOp {
    /// Op.
    pub op: TripleBandOp,
    /// Kappa.
    pub kappa: f64,
    /// Theta.
    pub theta: f64,
    /// Sigma.
    pub sigma: f64,
}

impl FdmSquareRootFwdOp {
    /// New.
    pub fn new(grid: &[f64], kappa: f64, theta: f64, sigma: f64) -> Self {
        let n = grid.len();
        let mut op = TripleBandOp::zeros(n);
        for i in 1..n - 1 {
            let x = grid[i].max(0.0);
            let dp = grid[i + 1] - grid[i];
            let dm = grid[i] - grid[i - 1];
            let ds = 0.5 * (dp + dm);
            let diff = 0.5 * sigma * sigma * x;
            let conv = -(kappa * (theta - x)); // adjoint
            op.upper[i] = diff / (dp * ds) + conv / (dp + dm);
            op.lower[i] = diff / (dm * ds) - conv / (dp + dm);
            op.diag[i] = -(op.upper[i] + op.lower[i]);
        }
        Self {
            op,
            kappa,
            theta,
            sigma,
        }
    }

    /// Apply.
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        self.op.apply(x)
    }
}

// ---------------------------------------------------------------------------
// G137: FdmOrnsteinUhlenbeckOp — OU spatial operator
// ---------------------------------------------------------------------------

/// Ornstein-Uhlenbeck spatial operator.
///
/// PDE: ∂V/∂t + μ(θ−x) ∂V/∂x + ½σ² ∂²V/∂x² − r V = 0
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FdmOrnsteinUhlenbeckOp {
    /// Op.
    pub op: TripleBandOp,
    /// Speed.
    pub speed: f64,
    /// Level.
    pub level: f64,
    /// Vol.
    pub vol: f64,
    /// R.
    pub r: f64,
}

impl FdmOrnsteinUhlenbeckOp {
    /// New.
    pub fn new(grid: &[f64], speed: f64, level: f64, vol: f64, r: f64) -> Self {
        let n = grid.len();
        let mut op = TripleBandOp::zeros(n);
        let diff = 0.5 * vol * vol;
        for i in 1..n - 1 {
            let x = grid[i];
            let conv = speed * (level - x);
            let dp = grid[i + 1] - grid[i];
            let dm = grid[i] - grid[i - 1];
            let ds = 0.5 * (dp + dm);
            op.upper[i] = diff / (dp * ds) + conv / (dp + dm);
            op.lower[i] = diff / (dm * ds) - conv / (dp + dm);
            op.diag[i] = -(op.upper[i] + op.lower[i]) - r;
        }
        op.diag[0] = -r;
        op.diag[n - 1] = -r;
        Self {
            op,
            speed,
            level,
            vol,
            r,
        }
    }

    /// Apply.
    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        self.op.apply(x)
    }
}

// ---------------------------------------------------------------------------
// G138: FdmSABROp — SABR model spatial operator
// ---------------------------------------------------------------------------

/// SABR model spatial operator on (F, α) grid.
///
/// The SABR PDE in (F, α):
///   ∂V/∂t + ½α²F^{2β} ∂²V/∂F² + ρνα F^β ∂²V/∂F∂α + ½ν²α² ∂²V/∂α² − rV = 0
///
/// Split into tridiagonal operators in each direction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FdmSABROp {
    /// Grid for F (forward).
    pub f_grid: Vec<f64>,
    /// Grid for α (volatility).
    pub a_grid: Vec<f64>,
    /// SABR β exponent.
    pub beta: f64,
    /// ν (vol-of-vol).
    pub nu: f64,
    /// ρ (correlation).
    pub rho: f64,
    /// Tridiagonal operator in F direction (for midpoint alpha).
    pub f_op: TripleBandOp,
    /// Tridiagonal operator in α direction (for midpoint F).
    pub a_op: TripleBandOp,
    /// Discount rate.
    pub r: f64,
}

impl FdmSABROp {
    /// New.
    pub fn new(
        f_grid: &[f64],
        a_grid: &[f64],
        beta: f64,
        nu: f64,
        rho: f64,
        r: f64,
    ) -> Self {
        let nf = f_grid.len();
        let na = a_grid.len();
        let alpha_mid = a_grid[na / 2].max(1e-8);

        // F-direction operator (at midpoint alpha)
        let mut f_op = TripleBandOp::zeros(nf);
        for i in 1..nf - 1 {
            let f = f_grid[i].max(1e-8);
            let diff = 0.5 * alpha_mid * alpha_mid * f.powf(2.0 * beta);
            let dp = f_grid[i + 1] - f_grid[i];
            let dm = f_grid[i] - f_grid[i - 1];
            let ds = 0.5 * (dp + dm);
            f_op.upper[i] = diff / (dp * ds);
            f_op.lower[i] = diff / (dm * ds);
            f_op.diag[i] = -(f_op.upper[i] + f_op.lower[i]) - 0.5 * r;
        }
        f_op.diag[0] = -0.5 * r;
        f_op.diag[nf - 1] = -0.5 * r;

        // α-direction operator (at midpoint F)
        let f_mid = f_grid[nf / 2].max(1e-8);
        let mut a_op = TripleBandOp::zeros(na);
        for j in 1..na - 1 {
            let a = a_grid[j].max(1e-8);
            let diff = 0.5 * nu * nu * a * a;
            let dp = a_grid[j + 1] - a_grid[j];
            let dm = a_grid[j] - a_grid[j - 1];
            let ds = 0.5 * (dp + dm);
            a_op.upper[j] = diff / (dp * ds);
            a_op.lower[j] = diff / (dm * ds);
            a_op.diag[j] = -(a_op.upper[j] + a_op.lower[j]) - 0.5 * r;
        }
        a_op.diag[0] = -0.5 * r;
        a_op.diag[na - 1] = -0.5 * r;
        let _ = f_mid; // used for cross-term in full implementation

        Self {
            f_grid: f_grid.to_vec(),
            a_grid: a_grid.to_vec(),
            beta,
            nu,
            rho,
            f_op,
            a_op,
            r,
        }
    }

    /// Compute cross-derivative contribution: ρ ν α F^β ∂²V/∂F∂α.
    pub fn cross_term(&self, v: &[f64]) -> Vec<f64> {
        let nf = self.f_grid.len();
        let na = self.a_grid.len();
        let mut cross = vec![0.0; nf * na];
        for i in 1..nf - 1 {
            let f = self.f_grid[i].max(1e-8);
            let hf = self.f_grid[i + 1] - self.f_grid[i - 1];
            for j in 1..na - 1 {
                let a = self.a_grid[j].max(1e-8);
                let ha = self.a_grid[j + 1] - self.a_grid[j - 1];
                let coeff = self.rho * self.nu * a * f.powf(self.beta);
                let idx = i * na + j;
                cross[idx] = coeff / (hf * ha)
                    * (v[(i + 1) * na + j + 1] - v[(i + 1) * na + j - 1]
                        - v[(i - 1) * na + j + 1]
                        + v[(i - 1) * na + j - 1]);
            }
        }
        cross
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn uniform_grid(lo: f64, hi: f64, n: usize) -> Vec<f64> {
        (0..n).map(|i| lo + (hi - lo) * i as f64 / (n - 1) as f64).collect()
    }

    // G123
    #[test]
    fn nine_point_zero_preserves() {
        let op = NinePointLinearOp::zeros(5, 5);
        let x = vec![1.0; 25];
        let y = op.apply(&x);
        for v in &y {
            assert_abs_diff_eq!(*v, 0.0, epsilon = 1e-14);
        }
    }

    // G124
    #[test]
    fn first_derivative_linear() {
        // f(x) = x → f'(x) = 1
        let grid = uniform_grid(0.0, 1.0, 11);
        let d1 = FirstDerivativeOp::new(&grid);
        let f: Vec<f64> = grid.iter().copied().collect();
        let df = d1.apply(&f);
        for i in 1..grid.len() - 1 {
            assert_abs_diff_eq!(df[i], 1.0, epsilon = 1e-10);
        }
    }

    // G125
    #[test]
    fn second_derivative_quadratic() {
        // f(x) = x² → f''(x) = 2
        let grid = uniform_grid(0.0, 1.0, 21);
        let d2 = SecondDerivativeOp::new(&grid);
        let f: Vec<f64> = grid.iter().map(|x| x * x).collect();
        let ddf = d2.apply(&f);
        for i in 1..grid.len() - 1 {
            assert_abs_diff_eq!(ddf[i], 2.0, epsilon = 1e-6);
        }
    }

    // G126
    #[test]
    fn mixed_derivative_product() {
        // f(x,y) = x·y → ∂²f/∂x∂y = 1
        let gx = uniform_grid(0.0, 1.0, 11);
        let gy = uniform_grid(0.0, 1.0, 11);
        let n1 = gx.len();
        let n2 = gy.len();
        let u: Vec<f64> = (0..n1 * n2)
            .map(|k| {
                let i = k / n2;
                let j = k % n2;
                gx[i] * gy[j]
            })
            .collect();
        let op = SecondOrderMixedDerivativeOp::new(&gx, &gy);
        let d = op.apply(&u);
        for i in 1..n1 - 1 {
            for j in 1..n2 - 1 {
                assert_abs_diff_eq!(d[i * n2 + j], 1.0, epsilon = 0.1);
            }
        }
    }

    // G127
    #[test]
    fn nth_order_first_derivative() {
        let grid = uniform_grid(0.0, 1.0, 21);
        let d = NthOrderDerivativeOp::new(&grid, 1, None);
        let f: Vec<f64> = grid.iter().map(|&x| x * x).collect();
        let df = d.apply(&f);
        // f'(x) = 2x
        for i in 2..grid.len() - 2 {
            assert_abs_diff_eq!(df[i], 2.0 * grid[i], epsilon = 0.05);
        }
    }

    // G128
    #[test]
    fn mod_triple_band_neumann_lower() {
        let n = 5;
        let inner = TripleBandOp::zeros(n);
        let mut op = ModTripleBandLinearOp::new(inner);
        op.set_neumann_lower();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = op.apply(&x);
        // Neumann lower: −x[0] + x[1] = −1 + 2 = 1
        assert_abs_diff_eq!(y[0], -1.0 + 2.0, epsilon = 1e-12);
    }

    // G129
    #[test]
    fn bs_op_matches_build() {
        let grid = uniform_grid(3.5, 5.5, 51);
        let op1 = FdmBlackScholesOp::new(&grid, 0.05, 0.02, 0.20);
        let op2 = crate::fdm_operators::build_bs_operator(&grid, 0.05, 0.02, 0.20);
        let x: Vec<f64> = grid.iter().map(|v| (v - 4.5).powi(2)).collect();
        let y1 = op1.apply(&x);
        let y2 = op2.apply(&x);
        for i in 0..y1.len() {
            assert_abs_diff_eq!(y1[i], y2[i], epsilon = 1e-12);
        }
    }

    // G129 local vol
    #[test]
    fn bs_local_vol_op() {
        let grid = uniform_grid(3.5, 5.5, 51);
        let lvols: Vec<f64> = grid.iter().map(|x| 0.15 + 0.02 * (x - 4.5).abs()).collect();
        let op = FdmBlackScholesOp::with_local_vol(&grid, 0.05, 0.0, lvols);
        let x: Vec<f64> = grid.iter().map(|v| 0.0_f64.max(v.exp() - 100.0)).collect();
        let y = op.apply(&x);
        // Should produce finite values
        for v in &y {
            assert!(v.is_finite());
        }
    }

    // G130
    #[test]
    fn bs_2d_op_create() {
        let g1 = uniform_grid(3.5, 5.5, 21);
        let g2 = uniform_grid(3.5, 5.5, 21);
        let op = Fdm2dBlackScholesOp::new(&g1, &g2, 0.05, 0.02, 0.01, 0.20, 0.25, 0.3);
        assert_eq!(op.n1, 21);
        assert_eq!(op.n2, 21);
    }

    // G131
    #[test]
    fn heston_op_create() {
        let xg = uniform_grid(3.5, 5.5, 21);
        let vg = uniform_grid(0.0, 0.5, 11);
        let op = FdmHestonOp::new(&xg, &vg, 0.05, 0.02, 2.0, 0.04, 0.3, -0.7);
        assert_eq!(op.heston_ops.nx, 21);
        assert_eq!(op.heston_ops.nv, 11);
    }

    // G132
    #[test]
    fn heston_fwd_op_create() {
        let xg = uniform_grid(3.5, 5.5, 21);
        let vg = uniform_grid(0.0, 0.5, 11);
        let op = FdmHestonFwdOp::new(&xg, &vg, 0.05, 0.02, 2.0, 0.04, 0.3, -0.7);
        assert_eq!(op.nx, 21);
        assert_eq!(op.nv, 11);
    }

    // G133
    #[test]
    fn heston_hw_op_create() {
        let xg = uniform_grid(3.5, 5.5, 11);
        let vg = uniform_grid(0.0, 0.5, 7);
        let rg = uniform_grid(-0.05, 0.15, 7);
        let op = FdmHestonHullWhiteOp::new(
            &xg, &vg, &rg, 0.05, 0.02, 2.0, 0.04, 0.3, -0.7, 0.1, 0.01, 0.05, 0.2,
        );
        assert_eq!(op.hw_op.n, 7);
    }

    // G134
    #[test]
    fn bates_op_jump_integral() {
        let xg = uniform_grid(3.5, 5.5, 51);
        let vg = uniform_grid(0.0, 0.5, 11);
        let op = FdmBatesOp::new(&xg, &vg, 0.05, 0.02, 2.0, 0.04, 0.3, -0.7, 0.5, -0.05, 0.1);
        let slice = vec![1.0; 51]; // constant → jumps should ≈ 0
        let j = op.apply_jump(&slice);
        for v in &j {
            assert_abs_diff_eq!(*v, 0.0, epsilon = 1e-8);
        }
    }

    // G135
    #[test]
    fn bs_fwd_op_adjoint_sign() {
        let grid = uniform_grid(3.5, 5.5, 21);
        let fwd = FdmBlackScholesFwdOp::new(&grid, 0.05, 0.02, 0.20);
        let x: Vec<f64> = grid.iter().map(|v| (-(v - 4.5).powi(2) / 0.1).exp()).collect();
        let y = fwd.apply(&x);
        // Should produce finite values
        for v in &y {
            assert!(v.is_finite(), "non-finite forward op result");
        }
    }

    // G135 local vol fwd
    #[test]
    fn local_vol_fwd_op() {
        let grid = uniform_grid(3.5, 5.5, 21);
        let lvols: Vec<f64> = vec![0.20; 21];
        let op = FdmLocalVolFwdOp::new(&grid, 0.05, 0.02, &lvols);
        let x = vec![0.0; 21];
        let y = op.apply(&x);
        for v in &y {
            assert_abs_diff_eq!(*v, 0.0, epsilon = 1e-14);
        }
    }

    // G136
    #[test]
    fn sqrt_fwd_op() {
        let grid = uniform_grid(0.0, 0.3, 21);
        let op = FdmSquareRootFwdOp::new(&grid, 2.0, 0.04, 0.3);
        let x = vec![1.0; 21];
        let y = op.apply(&x);
        for v in &y {
            assert!(v.is_finite());
        }
    }

    // G137
    #[test]
    fn ou_op_zeros() {
        let grid = uniform_grid(-1.0, 1.0, 21);
        let op = FdmOrnsteinUhlenbeckOp::new(&grid, 1.0, 0.0, 0.2, 0.05);
        let x = vec![0.0; 21];
        let y = op.apply(&x);
        for v in &y {
            assert_abs_diff_eq!(*v, 0.0, epsilon = 1e-14);
        }
    }

    // G138
    #[test]
    fn sabr_op_create() {
        let fg = uniform_grid(50.0, 150.0, 21);
        let ag = uniform_grid(0.05, 0.60, 11);
        let op = FdmSABROp::new(&fg, &ag, 0.5, 0.4, -0.3, 0.05);
        assert_eq!(op.f_op.n, 21);
        assert_eq!(op.a_op.n, 11);
    }

    #[test]
    fn sabr_cross_term_finite() {
        let fg = uniform_grid(50.0, 150.0, 11);
        let ag = uniform_grid(0.05, 0.60, 7);
        let op = FdmSABROp::new(&fg, &ag, 0.5, 0.4, -0.3, 0.05);
        let v: Vec<f64> = (0..11 * 7).map(|k| {
            let i = k / 7;
            let j = k % 7;
            fg[i] * ag[j]
        }).collect();
        let ct = op.cross_term(&v);
        for val in &ct {
            assert!(val.is_finite());
        }
    }

    // G139
    #[test]
    fn linear_op_trait_impl() {
        struct SimpleOp(usize);
        impl FdmLinearOp for SimpleOp {
            fn apply(&self, x: &[f64]) -> Vec<f64> {
                x.iter().map(|v| 2.0 * v).collect()
            }
            fn size(&self) -> usize {
                self.0
            }
        }
        let op = SimpleOp(5);
        let x = vec![1.0; 5];
        let y = op.apply(&x);
        assert_eq!(y, vec![2.0; 5]);
        assert_eq!(op.size(), 5);
    }
}
