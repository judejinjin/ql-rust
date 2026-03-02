//! Vectorized (SIMD-width) tape evaluation for Monte Carlo batches.
//!
//! Instead of processing one MC path at a time through the scalar
//! [`Tape`](crate::tape::Tape), this module processes **N paths
//! simultaneously** using `[f64; N]` arrays ([`Lanes<N>`]).
//!
//! The compiler auto-vectorizes element-wise operations on these arrays
//! into SIMD instructions (AVX2 for N=4, AVX-512 for N=8) when the
//! target supports them.
//!
//! All paths in a batch share the same tape structure (identical
//! computational graph) but each lane carries its own values and partial
//! derivatives. This amortises tape traversal overhead and improves
//! cache utilisation vs. the scalar tape.
//!
//! # Example
//!
//! ```
//! use ql_aad::simd::{Lanes, SimdTape};
//!
//! let mut tape = SimdTape::<4>::new();
//! let x = tape.input_scalar(3.0);
//! let y = tape.input_scalar(5.0);
//! let z = tape.mul(x, y);               // z = 15 across all 4 lanes
//! let adj = tape.adjoint(z);
//! assert!((adj[x.idx].data[0] - 5.0).abs() < 1e-14); // ∂z/∂x = y
//! ```

use smallvec::{smallvec, SmallVec};

use crate::bs::OptionKind;
use crate::mc::{McEuropeanGreeks, McHestonGreeks};

use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

// ===========================================================================
// Lanes<N> — SIMD-friendly f64 array
// ===========================================================================

/// A batch of N `f64` values, intended for SIMD-width tape evaluation.
///
/// The compiler will auto-vectorize element-wise operations on these
/// fixed-size arrays into SIMD instructions when the target supports
/// them (e.g. AVX2 `ymm` registers for N=4).
#[derive(Clone, Copy, Debug)]
pub struct Lanes<const N: usize> {
    /// Per-lane values.
    pub data: [f64; N],
}

impl<const N: usize> Lanes<N> {
    /// All lanes set to `v`.
    #[inline]
    pub fn splat(v: f64) -> Self {
        Self { data: [v; N] }
    }

    /// All zeros.
    #[inline]
    pub fn zero() -> Self { Self::splat(0.0) }

    /// All ones.
    #[inline]
    pub fn one() -> Self { Self::splat(1.0) }

    /// Create from a raw array.
    #[inline]
    pub fn from_array(data: [f64; N]) -> Self { Self { data } }

    /// Horizontal sum across all lanes.
    #[inline]
    pub fn hsum(self) -> f64 {
        let mut s = 0.0;
        for k in 0..N { s += self.data[k]; }
        s
    }

    /// Horizontal sum of squares.
    #[inline]
    pub fn hsum_sq(self) -> f64 {
        let mut s = 0.0;
        for k in 0..N { s += self.data[k] * self.data[k]; }
        s
    }

    /// True if every lane is exactly zero.
    #[inline]
    pub fn all_zero(self) -> bool {
        for k in 0..N {
            if self.data[k] != 0.0 { return false; }
        }
        true
    }

    // --- Element-wise transcendentals ---

    /// Element-wise `exp`.
    #[inline]
    pub fn exp(self) -> Self {
        let mut out = [0.0; N];
        for k in 0..N { out[k] = self.data[k].exp(); }
        Self { data: out }
    }

    /// Element-wise `ln`.
    #[inline]
    pub fn ln(self) -> Self {
        let mut out = [0.0; N];
        for k in 0..N { out[k] = self.data[k].ln(); }
        Self { data: out }
    }

    /// Element-wise `sqrt`.
    #[inline]
    pub fn sqrt(self) -> Self {
        let mut out = [0.0; N];
        for k in 0..N { out[k] = self.data[k].sqrt(); }
        Self { data: out }
    }

    /// Element-wise `abs`.
    #[inline]
    pub fn abs(self) -> Self {
        let mut out = [0.0; N];
        for k in 0..N { out[k] = self.data[k].abs(); }
        Self { data: out }
    }

    /// Element-wise `max`.
    #[inline]
    pub fn max(self, other: Self) -> Self {
        let mut out = [0.0; N];
        for k in 0..N { out[k] = self.data[k].max(other.data[k]); }
        Self { data: out }
    }

    /// Element-wise `min`.
    #[inline]
    pub fn min(self, other: Self) -> Self {
        let mut out = [0.0; N];
        for k in 0..N { out[k] = self.data[k].min(other.data[k]); }
        Self { data: out }
    }

    /// Element-wise `sin`.
    #[inline]
    pub fn sin(self) -> Self {
        let mut out = [0.0; N];
        for k in 0..N { out[k] = self.data[k].sin(); }
        Self { data: out }
    }

    /// Element-wise `cos`.
    #[inline]
    pub fn cos(self) -> Self {
        let mut out = [0.0; N];
        for k in 0..N { out[k] = self.data[k].cos(); }
        Self { data: out }
    }

    /// Element-wise fused multiply-add: `self * a + b`.
    #[inline]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        let mut out = [0.0; N];
        for k in 0..N { out[k] = self.data[k].mul_add(a.data[k], b.data[k]); }
        Self { data: out }
    }
}

// --- Arithmetic operator impls for Lanes<N> ---

use std::ops::{Add, Sub, Mul, Div, Neg, AddAssign};

impl<const N: usize> Add for Lanes<N> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut out = [0.0; N];
        for k in 0..N { out[k] = self.data[k] + rhs.data[k]; }
        Self { data: out }
    }
}

impl<const N: usize> Sub for Lanes<N> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut out = [0.0; N];
        for k in 0..N { out[k] = self.data[k] - rhs.data[k]; }
        Self { data: out }
    }
}

impl<const N: usize> Mul for Lanes<N> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let mut out = [0.0; N];
        for k in 0..N { out[k] = self.data[k] * rhs.data[k]; }
        Self { data: out }
    }
}

impl<const N: usize> Div for Lanes<N> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let mut out = [0.0; N];
        for k in 0..N { out[k] = self.data[k] / rhs.data[k]; }
        Self { data: out }
    }
}

impl<const N: usize> Neg for Lanes<N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let mut out = [0.0; N];
        for k in 0..N { out[k] = -self.data[k]; }
        Self { data: out }
    }
}

impl<const N: usize> AddAssign for Lanes<N> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for k in 0..N { self.data[k] += rhs.data[k]; }
    }
}

// ===========================================================================
// SimdTape<N> — Vectorized computation tape
// ===========================================================================

/// A node in the vectorized tape.
struct SimdNode<const N: usize> {
    /// Forward values for each of the N lanes.
    #[allow(dead_code)]
    values: Lanes<N>,
    /// Partial derivatives w.r.t. children: `(child_index, per-lane partials)`.
    partials: SmallVec<[(usize, Lanes<N>); 2]>,
}

/// Vectorized computation tape — processes N Monte Carlo paths simultaneously.
///
/// All N paths share the same computation graph (same sequence of operations).
/// Each node stores N values (one per lane) and N partial derivatives per child.
/// The adjoint pass computes per-lane gradients in a single traversal.
pub struct SimdTape<const N: usize> {
    nodes: Vec<SimdNode<N>>,
}

/// Active real on a [`SimdTape`] — carries N lane values and a tape index.
#[derive(Clone, Copy, Debug)]
pub struct SimdReal<const N: usize> {
    /// Index on the tape.
    pub idx: usize,
    /// Per-lane forward values.
    pub val: Lanes<N>,
}

impl<const N: usize> SimdTape<N> {
    /// Create a new empty tape.
    pub fn new() -> Self {
        Self { nodes: Vec::with_capacity(4096) }
    }

    /// Create a tape with pre-allocated capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self { nodes: Vec::with_capacity(cap) }
    }

    /// Clear the tape for reuse (keeps allocation).
    pub fn clear(&mut self) {
        self.nodes.clear();
    }

    /// Number of nodes on the tape.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the tape is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Push a node with value and partials.
    fn push(
        &mut self,
        values: Lanes<N>,
        partials: SmallVec<[(usize, Lanes<N>); 2]>,
    ) -> SimdReal<N> {
        let idx = self.nodes.len();
        self.nodes.push(SimdNode { values, partials });
        SimdReal { idx, val: values }
    }

    // --- Inputs and constants ---

    /// Register an AD input with per-lane values.
    pub fn input(&mut self, val: Lanes<N>) -> SimdReal<N> {
        self.push(val, SmallVec::new())
    }

    /// Register an AD input with the same value across all lanes.
    pub fn input_scalar(&mut self, val: f64) -> SimdReal<N> {
        self.push(Lanes::splat(val), SmallVec::new())
    }

    /// Create a per-lane constant (not differentiated).
    pub fn constant(&mut self, val: Lanes<N>) -> SimdReal<N> {
        self.push(val, SmallVec::new())
    }

    /// Create a scalar constant (same on all lanes, not differentiated).
    pub fn constant_scalar(&mut self, val: f64) -> SimdReal<N> {
        self.push(Lanes::splat(val), SmallVec::new())
    }

    // --- Unary operations ---

    /// Negate: `−a`.
    pub fn neg(&mut self, a: SimdReal<N>) -> SimdReal<N> {
        self.push(-a.val, smallvec![(a.idx, Lanes::splat(-1.0))])
    }

    /// Absolute value (sub-gradient).
    pub fn abs(&mut self, a: SimdReal<N>) -> SimdReal<N> {
        let mut d = [0.0; N];
        let mut val = [0.0; N];
        for k in 0..N {
            val[k] = a.val.data[k].abs();
            d[k] = if a.val.data[k] >= 0.0 { 1.0 } else { -1.0 };
        }
        self.push(Lanes::from_array(val), smallvec![(a.idx, Lanes::from_array(d))])
    }

    /// Exponential: `exp(a)`.
    pub fn exp(&mut self, a: SimdReal<N>) -> SimdReal<N> {
        let e = a.val.exp();
        self.push(e, smallvec![(a.idx, e)])
    }

    /// Natural logarithm: `ln(a)`.
    pub fn ln(&mut self, a: SimdReal<N>) -> SimdReal<N> {
        let mut d = [0.0; N];
        for k in 0..N { d[k] = 1.0 / a.val.data[k]; }
        self.push(a.val.ln(), smallvec![(a.idx, Lanes::from_array(d))])
    }

    /// Square root: `√a`.
    ///
    /// Uses subgradient 0 at a=0 to avoid Inf partials (important for
    /// Heston variance truncation where `max(v,0)` can be exactly zero).
    pub fn sqrt(&mut self, a: SimdReal<N>) -> SimdReal<N> {
        let s = a.val.sqrt();
        let mut d = [0.0; N];
        for k in 0..N {
            d[k] = if s.data[k] > 0.0 { 0.5 / s.data[k] } else { 0.0 };
        }
        self.push(s, smallvec![(a.idx, Lanes::from_array(d))])
    }

    /// Reciprocal: `1/a`.
    pub fn recip(&mut self, a: SimdReal<N>) -> SimdReal<N> {
        let mut val = [0.0; N];
        let mut d = [0.0; N];
        for k in 0..N {
            let inv = 1.0 / a.val.data[k];
            val[k] = inv;
            d[k] = -inv * inv;
        }
        self.push(Lanes::from_array(val), smallvec![(a.idx, Lanes::from_array(d))])
    }

    /// Sine: `sin(a)`.
    pub fn sin(&mut self, a: SimdReal<N>) -> SimdReal<N> {
        self.push(a.val.sin(), smallvec![(a.idx, a.val.cos())])
    }

    /// Cosine: `cos(a)`.
    pub fn cos(&mut self, a: SimdReal<N>) -> SimdReal<N> {
        self.push(a.val.cos(), smallvec![(a.idx, -a.val.sin())])
    }

    /// Integer power: `a^n`.
    pub fn powi(&mut self, a: SimdReal<N>, n: i32) -> SimdReal<N> {
        let mut val = [0.0; N];
        let mut d = [0.0; N];
        for k in 0..N {
            val[k] = a.val.data[k].powi(n);
            d[k] = n as f64 * a.val.data[k].powi(n - 1);
        }
        self.push(Lanes::from_array(val), smallvec![(a.idx, Lanes::from_array(d))])
    }

    // --- Binary operations ---

    /// Addition: `a + b`.
    pub fn add(&mut self, a: SimdReal<N>, b: SimdReal<N>) -> SimdReal<N> {
        self.push(
            a.val + b.val,
            SmallVec::from_buf([(a.idx, Lanes::one()), (b.idx, Lanes::one())]),
        )
    }

    /// Subtraction: `a − b`.
    pub fn sub(&mut self, a: SimdReal<N>, b: SimdReal<N>) -> SimdReal<N> {
        self.push(
            a.val - b.val,
            SmallVec::from_buf([(a.idx, Lanes::one()), (b.idx, Lanes::splat(-1.0))]),
        )
    }

    /// Multiplication: `a × b`.
    pub fn mul(&mut self, a: SimdReal<N>, b: SimdReal<N>) -> SimdReal<N> {
        self.push(
            a.val * b.val,
            SmallVec::from_buf([(a.idx, b.val), (b.idx, a.val)]),
        )
    }

    /// Division: `a / b`.
    pub fn div(&mut self, a: SimdReal<N>, b: SimdReal<N>) -> SimdReal<N> {
        let mut inv_b = [0.0; N];
        for k in 0..N { inv_b[k] = 1.0 / b.val.data[k]; }
        let inv = Lanes::from_array(inv_b);
        let val = a.val * inv;
        let mut db = [0.0; N];
        for k in 0..N { db[k] = -a.val.data[k] * inv_b[k] * inv_b[k]; }
        self.push(val, SmallVec::from_buf([(a.idx, inv), (b.idx, Lanes::from_array(db))]))
    }

    /// Element-wise max with per-lane sub-gradients.
    ///
    /// Unlike the scalar tape (which stores only the winning child), the
    /// vectorized tape always stores both children with per-lane masks,
    /// since different lanes may select different operands.
    pub fn max(&mut self, a: SimdReal<N>, b: SimdReal<N>) -> SimdReal<N> {
        let mut val = [0.0; N];
        let mut pa = [0.0; N];
        let mut pb = [0.0; N];
        for k in 0..N {
            if a.val.data[k] >= b.val.data[k] {
                val[k] = a.val.data[k];
                pa[k] = 1.0;
            } else {
                val[k] = b.val.data[k];
                pb[k] = 1.0;
            }
        }
        self.push(
            Lanes::from_array(val),
            SmallVec::from_buf([
                (a.idx, Lanes::from_array(pa)),
                (b.idx, Lanes::from_array(pb)),
            ]),
        )
    }

    /// Element-wise min with per-lane sub-gradients.
    pub fn min(&mut self, a: SimdReal<N>, b: SimdReal<N>) -> SimdReal<N> {
        let mut val = [0.0; N];
        let mut pa = [0.0; N];
        let mut pb = [0.0; N];
        for k in 0..N {
            if a.val.data[k] <= b.val.data[k] {
                val[k] = a.val.data[k];
                pa[k] = 1.0;
            } else {
                val[k] = b.val.data[k];
                pb[k] = 1.0;
            }
        }
        self.push(
            Lanes::from_array(val),
            SmallVec::from_buf([
                (a.idx, Lanes::from_array(pa)),
                (b.idx, Lanes::from_array(pb)),
            ]),
        )
    }

    /// Power: `a^b`.
    pub fn powf(&mut self, a: SimdReal<N>, b: SimdReal<N>) -> SimdReal<N> {
        let mut val = [0.0; N];
        let mut da = [0.0; N];
        let mut db = [0.0; N];
        for k in 0..N {
            val[k] = a.val.data[k].powf(b.val.data[k]);
            da[k] = b.val.data[k] * a.val.data[k].powf(b.val.data[k] - 1.0);
            db[k] = val[k] * a.val.data[k].ln();
        }
        self.push(
            Lanes::from_array(val),
            SmallVec::from_buf([
                (a.idx, Lanes::from_array(da)),
                (b.idx, Lanes::from_array(db)),
            ]),
        )
    }

    /// Multiply by a scalar constant: `a × c`.
    pub fn mul_const(&mut self, a: SimdReal<N>, c: f64) -> SimdReal<N> {
        let c_lanes = Lanes::splat(c);
        self.push(a.val * c_lanes, smallvec![(a.idx, c_lanes)])
    }

    /// Add a scalar constant: `a + c`.
    pub fn add_const(&mut self, a: SimdReal<N>, c: f64) -> SimdReal<N> {
        self.push(a.val + Lanes::splat(c), smallvec![(a.idx, Lanes::one())])
    }

    // --- Adjoint computation ---

    /// Compute per-lane adjoints from the given output node.
    ///
    /// Returns `adj` where `adj[i].data[k]` = ∂output_k / ∂node_i for
    /// lane k. For input nodes, this gives the gradient w.r.t. each input.
    pub fn adjoint(&self, output: SimdReal<N>) -> Vec<Lanes<N>> {
        let n = self.nodes.len();
        let mut adj = vec![Lanes::<N>::zero(); n];
        adj[output.idx] = Lanes::one();

        for i in (0..=output.idx).rev() {
            let a_i = adj[i];
            if a_i.all_zero() { continue; }
            for &(child_idx, partial) in &self.nodes[i].partials {
                adj[child_idx] = adj[child_idx] + a_i * partial;
            }
        }

        adj
    }

    /// Compute per-lane gradients w.r.t. specific input nodes.
    pub fn gradient(
        &self,
        output: SimdReal<N>,
        inputs: &[SimdReal<N>],
    ) -> Vec<Lanes<N>> {
        let adj = self.adjoint(output);
        inputs.iter().map(|inp| adj[inp.idx]).collect()
    }
}

impl<const N: usize> Default for SimdTape<N> {
    fn default() -> Self { Self::new() }
}

// ===========================================================================
// MC European with vectorised tape
// ===========================================================================

/// European MC with SIMD-width AAD — processes N paths per tape traversal.
///
/// Antithetic variates: within each N-lane batch, half the lanes use `+z`
/// and the other half use `−z` (requires N to be even and ≥ 2).
///
/// Returns the same [`McEuropeanGreeks`] as the scalar
/// [`mc_european_aad`](crate::mc::mc_european_aad).
#[allow(clippy::too_many_arguments)]
pub fn mc_european_simd<const N: usize>(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    vol: f64,
    time_to_expiry: f64,
    option_kind: OptionKind,
    num_paths: usize,
    seed: u64,
) -> McEuropeanGreeks {
    assert!(N >= 2 && N % 2 == 0, "N must be even and >= 2");

    let mut rng = SmallRng::seed_from_u64(seed);
    let num_batches = num_paths / N;
    let actual_paths = num_batches * N;
    let mut tape = SimdTape::<N>::new();

    let phi = match option_kind {
        OptionKind::Call => 1.0,
        OptionKind::Put => -1.0,
    };

    let mut sum_npv = 0.0;
    let mut sum_npv_sq = 0.0;
    let mut sum_delta = 0.0;
    let mut sum_vega = 0.0;
    let mut sum_rho = 0.0;
    let mut sum_div_rho = 0.0;

    for _ in 0..num_batches {
        tape.clear();

        // Generate N/2 z-values + antithetic mirrors
        let mut z_data = [0.0; N];
        for k in 0..N / 2 {
            let z: f64 = StandardNormal.sample(&mut rng);
            z_data[2 * k] = z;
            z_data[2 * k + 1] = -z;
        }

        // Inputs (shared model parameters, same across all lanes)
        let s = tape.input_scalar(spot);       // idx 0
        let r_ad = tape.input_scalar(r);       // idx 1
        let q_ad = tape.input_scalar(q);       // idx 2
        let v = tape.input_scalar(vol);        // idx 3

        // Constants
        let z_c = tape.constant(Lanes::from_array(z_data));
        let tau = tape.constant_scalar(time_to_expiry);
        let sqrt_t = tape.constant_scalar(time_to_expiry.sqrt());
        let half = tape.constant_scalar(0.5);

        // S_T = s · exp((r − q − σ²/2)·τ + σ·√τ·z)
        let v_sq = tape.mul(v, v);
        let half_v_sq = tape.mul(half, v_sq);
        let r_minus_q = tape.sub(r_ad, q_ad);
        let drift_rate = tape.sub(r_minus_q, half_v_sq);
        let drift = tape.mul(drift_rate, tau);
        let vol_sqrt_t = tape.mul(v, sqrt_t);
        let diffusion = tape.mul(vol_sqrt_t, z_c);
        let exponent = tape.add(drift, diffusion);
        let exp_val = tape.exp(exponent);
        let st = tape.mul(s, exp_val);

        // payoff = max(φ·(S_T − K), 0)
        let strike_c = tape.constant_scalar(strike);
        let raw = tape.sub(st, strike_c);
        let phi_c = tape.constant_scalar(phi);
        let intrinsic = tape.mul(phi_c, raw);
        let zero = tape.constant_scalar(0.0);
        let payoff = tape.max(intrinsic, zero);

        // PV = payoff · exp(−r·τ)
        let neg_r = tape.neg(r_ad);
        let neg_r_tau = tape.mul(neg_r, tau);
        let disc = tape.exp(neg_r_tau);
        let pv = tape.mul(payoff, disc);

        // Adjoint
        let adj = tape.adjoint(pv);

        // Accumulate across lanes
        for k in 0..N {
            let pv_k = pv.val.data[k];
            sum_npv += pv_k;
            sum_npv_sq += pv_k * pv_k;
            sum_delta += adj[s.idx].data[k];
            sum_rho += adj[r_ad.idx].data[k];
            sum_div_rho += adj[q_ad.idx].data[k];
            sum_vega += adj[v.idx].data[k];
        }
    }

    let n = actual_paths as f64;
    let mean = sum_npv / n;
    let variance = (sum_npv_sq / n - mean * mean).max(0.0);
    let std_error = (variance / n).sqrt();

    McEuropeanGreeks {
        npv: mean,
        std_error,
        delta: sum_delta / n,
        vega: sum_vega / n,
        rho: sum_rho / n,
        div_rho: sum_div_rho / n,
        num_paths: actual_paths,
    }
}

// ===========================================================================
// MC Heston with vectorised tape
// ===========================================================================

/// Heston MC with SIMD-width AAD — processes N paths per tape traversal.
///
/// Log-Euler discretisation for spot + Euler with full truncation for
/// variance. Antithetic variates with paired lanes.
///
/// Returns the same [`McHestonGreeks`] as the scalar
/// [`mc_heston_aad`](crate::mc::mc_heston_aad).
#[allow(clippy::too_many_arguments)]
pub fn mc_heston_simd<const N: usize>(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    time_to_expiry: f64,
    option_kind: OptionKind,
    num_paths: usize,
    num_steps: usize,
    seed: u64,
) -> McHestonGreeks {
    assert!(N >= 2 && N % 2 == 0, "N must be even and >= 2");

    let mut rng = SmallRng::seed_from_u64(seed);
    let dt = time_to_expiry / num_steps as f64;
    let sqrt_dt = dt.sqrt();
    let num_batches = num_paths / N;
    let actual_paths = num_batches * N;

    let phi = match option_kind {
        OptionKind::Call => 1.0,
        OptionKind::Put => -1.0,
    };

    let mut sum_npv = 0.0;
    let mut sum_npv_sq = 0.0;
    let mut sum_greeks = [0.0_f64; 8]; // spot, r, q, v0, kappa, theta, sigma, rho

    let mut tape = SimdTape::<N>::new();

    for _ in 0..num_batches {
        // Pre-generate all random numbers for this batch
        let mut z1_batch: Vec<Lanes<N>> = Vec::with_capacity(num_steps);
        let mut z2_batch: Vec<Lanes<N>> = Vec::with_capacity(num_steps);
        for _ in 0..num_steps {
            let mut z1_data = [0.0; N];
            let mut z2_data = [0.0; N];
            for k in 0..N / 2 {
                let z1: f64 = StandardNormal.sample(&mut rng);
                let z2: f64 = StandardNormal.sample(&mut rng);
                z1_data[2 * k] = z1;
                z1_data[2 * k + 1] = -z1;
                z2_data[2 * k] = z2;
                z2_data[2 * k + 1] = -z2;
            }
            z1_batch.push(Lanes::from_array(z1_data));
            z2_batch.push(Lanes::from_array(z2_data));
        }

        tape.clear();

        // Register 8 inputs
        let s_ad = tape.input_scalar(spot);         // 0
        let r_ad = tape.input_scalar(r);            // 1
        let q_ad = tape.input_scalar(q);            // 2
        let v0_ad = tape.input_scalar(v0);          // 3
        let kappa_ad = tape.input_scalar(kappa);    // 4
        let theta_ad = tape.input_scalar(theta);    // 5
        let sigma_ad = tape.input_scalar(sigma);    // 6
        let rho_ad = tape.input_scalar(rho);        // 7

        let dt_c = tape.constant_scalar(dt);
        let sqrt_dt_c = tape.constant_scalar(sqrt_dt);
        let half = tape.constant_scalar(0.5);
        let zero = tape.constant_scalar(0.0);
        let one = tape.constant_scalar(1.0);

        // log_s = ln(spot), v = v0
        let mut log_s = tape.ln(s_ad);
        let mut v = v0_ad;

        // ρ_comp = √(1 − ρ²)
        let rho_sq = tape.mul(rho_ad, rho_ad);
        let one_minus_rho_sq = tape.sub(one, rho_sq);
        let rho_comp = tape.sqrt(one_minus_rho_sq);

        // Pre-compute r − q (constant across time steps)
        let r_minus_q = tape.sub(r_ad, q_ad);

        for step in 0..num_steps {
            let z1_c = tape.constant(z1_batch[step]);
            let z2_indep_c = tape.constant(z2_batch[step]);

            // v⁺ = max(v, 0)
            let v_pos = tape.max(v, zero);
            let sqrt_v = tape.sqrt(v_pos);

            // Correlated z₂ = ρ·z₁ + √(1−ρ²)·z₂_indep
            let rho_z1 = tape.mul(rho_ad, z1_c);
            let rho_comp_z2 = tape.mul(rho_comp, z2_indep_c);
            let z2_c = tape.add(rho_z1, rho_comp_z2);

            // Log-Euler for spot:
            //   log_s += (r − q − ½·v⁺)·dt + √v⁺·√dt·z₁
            let half_v = tape.mul(half, v_pos);
            let drift_rate = tape.sub(r_minus_q, half_v);
            let drift = tape.mul(drift_rate, dt_c);
            let vol_term = tape.mul(sqrt_v, sqrt_dt_c);
            let diffusion = tape.mul(vol_term, z1_c);
            let step_log = tape.add(drift, diffusion);
            log_s = tape.add(log_s, step_log);

            // Euler for variance (full truncation):
            //   v += κ·(θ − v⁺)·dt + σ·√v⁺·√dt·z₂
            let theta_minus_v = tape.sub(theta_ad, v_pos);
            let mean_rev = tape.mul(kappa_ad, theta_minus_v);
            let var_drift = tape.mul(mean_rev, dt_c);
            let vol_vol = tape.mul(sigma_ad, sqrt_v);
            let var_diff_1 = tape.mul(vol_vol, sqrt_dt_c);
            let var_diff = tape.mul(var_diff_1, z2_c);
            let var_change = tape.add(var_drift, var_diff);
            let v_raw = tape.add(v, var_change);
            v = tape.max(v_raw, zero);
        }

        // S_T = exp(log_s)
        let st = tape.exp(log_s);

        // payoff = max(φ·(S_T − K), 0)
        let strike_c = tape.constant_scalar(strike);
        let raw = tape.sub(st, strike_c);
        let phi_c = tape.constant_scalar(phi);
        let intrinsic = tape.mul(phi_c, raw);
        let payoff = tape.max(intrinsic, zero);

        // PV = payoff · exp(−r·τ)
        let neg_r = tape.neg(r_ad);
        let tau_c = tape.constant_scalar(time_to_expiry);
        let neg_r_tau = tape.mul(neg_r, tau_c);
        let disc = tape.exp(neg_r_tau);
        let pv = tape.mul(payoff, disc);

        // Adjoint
        let adj = tape.adjoint(pv);

        // Accumulate across lanes
        for k in 0..N {
            let pv_k = pv.val.data[k];
            sum_npv += pv_k;
            sum_npv_sq += pv_k * pv_k;
            sum_greeks[0] += adj[s_ad.idx].data[k];
            sum_greeks[1] += adj[r_ad.idx].data[k];
            sum_greeks[2] += adj[q_ad.idx].data[k];
            sum_greeks[3] += adj[v0_ad.idx].data[k];
            sum_greeks[4] += adj[kappa_ad.idx].data[k];
            sum_greeks[5] += adj[theta_ad.idx].data[k];
            sum_greeks[6] += adj[sigma_ad.idx].data[k];
            sum_greeks[7] += adj[rho_ad.idx].data[k];
        }
    }

    let n = actual_paths as f64;
    let mean = sum_npv / n;
    let variance = (sum_npv_sq / n - mean * mean).max(0.0);
    let std_error = (variance / n).sqrt();

    McHestonGreeks {
        npv: mean,
        std_error,
        delta: sum_greeks[0] / n,
        rho: sum_greeks[1] / n,
        div_rho: sum_greeks[2] / n,
        vega_v0: sum_greeks[3] / n,
        d_kappa: sum_greeks[4] / n,
        d_theta: sum_greeks[5] / n,
        d_sigma: sum_greeks[6] / n,
        d_rho: sum_greeks[7] / n,
        num_paths: actual_paths,
    }
}

// ===========================================================================
// Convenience wrappers (N = 4, optimal for AVX2)
// ===========================================================================

/// European MC with 4-wide SIMD lanes (optimal for AVX2).
///
/// Equivalent to `mc_european_simd::<4>(...)`.
#[allow(clippy::too_many_arguments)]
pub fn mc_european_simd4(
    spot: f64, strike: f64, r: f64, q: f64, vol: f64,
    time_to_expiry: f64, option_kind: OptionKind,
    num_paths: usize, seed: u64,
) -> McEuropeanGreeks {
    mc_european_simd::<4>(
        spot, strike, r, q, vol, time_to_expiry, option_kind, num_paths, seed,
    )
}

/// Heston MC with 4-wide SIMD lanes (optimal for AVX2).
///
/// Equivalent to `mc_heston_simd::<4>(...)`.
#[allow(clippy::too_many_arguments)]
pub fn mc_heston_simd4(
    spot: f64, strike: f64, r: f64, q: f64,
    v0: f64, kappa: f64, theta: f64, sigma: f64, rho: f64,
    time_to_expiry: f64, option_kind: OptionKind,
    num_paths: usize, num_steps: usize, seed: u64,
) -> McHestonGreeks {
    mc_heston_simd::<4>(
        spot, strike, r, q, v0, kappa, theta, sigma, rho,
        time_to_expiry, option_kind, num_paths, num_steps, seed,
    )
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // -----------------------------------------------------------------------
    // Lanes<N> unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn lanes_arithmetic() {
        let a = Lanes::<4>::from_array([1.0, 2.0, 3.0, 4.0]);
        let b = Lanes::<4>::from_array([5.0, 6.0, 7.0, 8.0]);
        let c = a + b;
        assert_eq!(c.data, [6.0, 8.0, 10.0, 12.0]);
        let d = a * b;
        assert_eq!(d.data, [5.0, 12.0, 21.0, 32.0]);
        let e = b - a;
        assert_eq!(e.data, [4.0, 4.0, 4.0, 4.0]);
        let f = -a;
        assert_eq!(f.data, [-1.0, -2.0, -3.0, -4.0]);
    }

    #[test]
    fn lanes_transcendentals() {
        let a = Lanes::<4>::splat(1.0);
        let e = a.exp();
        for k in 0..4 {
            assert_abs_diff_eq!(e.data[k], 1.0_f64.exp(), epsilon = 1e-14);
        }
        let l = e.ln();
        for k in 0..4 {
            assert_abs_diff_eq!(l.data[k], 1.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn lanes_horizontal_sum() {
        let a = Lanes::<4>::from_array([1.0, 2.0, 3.0, 4.0]);
        assert_abs_diff_eq!(a.hsum(), 10.0, epsilon = 1e-14);
        assert_abs_diff_eq!(a.hsum_sq(), 30.0, epsilon = 1e-14);
    }

    #[test]
    fn lanes_all_zero() {
        assert!(Lanes::<4>::zero().all_zero());
        assert!(!Lanes::<4>::from_array([0.0, 0.0, 1.0, 0.0]).all_zero());
    }

    // -----------------------------------------------------------------------
    // SimdTape unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn tape_mul_adjoint() {
        let mut tape = SimdTape::<4>::new();
        let x = tape.input_scalar(3.0);
        let y = tape.input_scalar(5.0);
        let z = tape.mul(x, y);
        assert_abs_diff_eq!(z.val.data[0], 15.0, epsilon = 1e-14);

        let adj = tape.adjoint(z);
        for k in 0..4 {
            assert_abs_diff_eq!(adj[x.idx].data[k], 5.0, epsilon = 1e-14);
            assert_abs_diff_eq!(adj[y.idx].data[k], 3.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn tape_exp_chain() {
        // f(x) = exp(x²), f'(x) = 2x·exp(x²)
        let mut tape = SimdTape::<4>::new();
        let x = tape.input_scalar(1.5);
        let x2 = tape.mul(x, x);
        let e = tape.exp(x2);
        let adj = tape.adjoint(e);
        let expected = 2.0 * 1.5 * (1.5_f64 * 1.5).exp();
        for k in 0..4 {
            assert_abs_diff_eq!(adj[x.idx].data[k], expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn tape_max_per_lane_divergence() {
        // Different lanes take different branches in max
        let mut tape = SimdTape::<4>::new();
        let a = tape.input(Lanes::from_array([5.0, -1.0, 3.0, -2.0]));
        let b = tape.input(Lanes::from_array([2.0, 4.0, 3.0, 1.0]));
        let m = tape.max(a, b);

        assert_eq!(m.val.data, [5.0, 4.0, 3.0, 1.0]);

        let adj = tape.adjoint(m);
        // Lane 0: a chosen → ∂/∂a = 1, ∂/∂b = 0
        assert_abs_diff_eq!(adj[a.idx].data[0], 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(adj[b.idx].data[0], 0.0, epsilon = 1e-15);
        // Lane 1: b chosen → ∂/∂a = 0, ∂/∂b = 1
        assert_abs_diff_eq!(adj[a.idx].data[1], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(adj[b.idx].data[1], 1.0, epsilon = 1e-15);
        // Lane 2: tie (a >= b) → ∂/∂a = 1
        assert_abs_diff_eq!(adj[a.idx].data[2], 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(adj[b.idx].data[2], 0.0, epsilon = 1e-15);
        // Lane 3: b chosen
        assert_abs_diff_eq!(adj[a.idx].data[3], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(adj[b.idx].data[3], 1.0, epsilon = 1e-15);
    }

    #[test]
    fn tape_per_lane_inputs() {
        // f(x) = x² with different x per lane
        let mut tape = SimdTape::<4>::new();
        let x = tape.input(Lanes::from_array([1.0, 2.0, 3.0, 4.0]));
        let x2 = tape.mul(x, x);
        let adj = tape.adjoint(x2);
        // ∂(x²)/∂x = 2x per lane
        assert_abs_diff_eq!(adj[x.idx].data[0], 2.0, epsilon = 1e-14);
        assert_abs_diff_eq!(adj[x.idx].data[1], 4.0, epsilon = 1e-14);
        assert_abs_diff_eq!(adj[x.idx].data[2], 6.0, epsilon = 1e-14);
        assert_abs_diff_eq!(adj[x.idx].data[3], 8.0, epsilon = 1e-14);
    }

    #[test]
    fn tape_gradient_convenience() {
        let mut tape = SimdTape::<4>::new();
        let x = tape.input_scalar(2.0);
        let y = tape.input_scalar(3.0);
        let xy = tape.mul(x, y);
        let z = tape.add(xy, x); // z = x*y + x
        let g = tape.gradient(z, &[x, y]);
        for k in 0..4 {
            assert_abs_diff_eq!(g[0].data[k], 4.0, epsilon = 1e-14); // ∂z/∂x = y+1
            assert_abs_diff_eq!(g[1].data[k], 2.0, epsilon = 1e-14); // ∂z/∂y = x
        }
    }

    #[test]
    fn tape_clear_and_reuse() {
        let mut tape = SimdTape::<4>::new();
        let x = tape.input_scalar(2.0);
        let _y = tape.mul(x, x);
        assert_eq!(tape.len(), 2);

        tape.clear();
        assert_eq!(tape.len(), 0);

        let a = tape.input_scalar(5.0);
        let b = tape.exp(a);
        let adj = tape.adjoint(b);
        for k in 0..4 {
            assert_abs_diff_eq!(adj[a.idx].data[k], 5.0_f64.exp(), epsilon = 1e-13);
        }
    }

    #[test]
    fn tape_ln_sqrt_chain() {
        // f(x) = sqrt(ln(x)), f'(x) = 1 / (2x sqrt(ln(x)))
        let mut tape = SimdTape::<4>::new();
        let x = tape.input_scalar(3.0);
        let lnx = tape.ln(x);
        let f = tape.sqrt(lnx);
        let adj = tape.adjoint(f);
        let expected = 1.0 / (2.0 * 3.0 * 3.0_f64.ln().sqrt());
        for k in 0..4 {
            assert_abs_diff_eq!(adj[x.idx].data[k], expected, epsilon = 1e-13);
        }
    }

    // -----------------------------------------------------------------------
    // MC European SIMD tests
    // -----------------------------------------------------------------------

    #[test]
    fn european_simd_npv_near_bs() {
        let g = mc_european_simd::<4>(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            OptionKind::Call, 100_000, 42,
        );
        // BS price ≈ 10.45
        assert!((g.npv - 10.45).abs() < 0.5, "npv={}", g.npv);
    }

    #[test]
    fn european_simd_delta_near_bs() {
        let g = mc_european_simd::<4>(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            OptionKind::Call, 200_000, 42,
        );
        // BS delta ≈ 0.637
        assert!((g.delta - 0.637).abs() < 0.05, "delta={}", g.delta);
    }

    #[test]
    fn european_simd_vega_positive() {
        let g = mc_european_simd::<4>(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            OptionKind::Call, 100_000, 42,
        );
        assert!(g.vega > 0.0, "vega should be positive, got {}", g.vega);
    }

    #[test]
    fn european_simd_rho_positive() {
        let g = mc_european_simd::<4>(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            OptionKind::Call, 100_000, 42,
        );
        assert!(g.rho > 0.0, "rho should be positive, got {}", g.rho);
    }

    #[test]
    fn european_simd_put_delta_negative() {
        let g = mc_european_simd::<4>(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            OptionKind::Put, 100_000, 42,
        );
        assert!(g.delta < 0.0, "put delta should be negative, got {}", g.delta);
    }

    #[test]
    fn simd_vs_scalar_european_consistent() {
        // Both methods should give statistically similar results (not
        // identical due to different RNG draw patterns)
        let scalar = crate::mc::mc_european_aad(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            OptionKind::Call, 100_000, 42,
        );
        let simd = mc_european_simd::<4>(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            OptionKind::Call, 100_000, 42,
        );
        assert!((scalar.npv - simd.npv).abs() < 0.5,
                "scalar npv={} vs simd npv={}", scalar.npv, simd.npv);
        assert!((scalar.delta - simd.delta).abs() < 0.05,
                "scalar delta={} vs simd delta={}", scalar.delta, simd.delta);
    }

    #[test]
    fn european_simd_n8_works() {
        let g = mc_european_simd::<8>(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            OptionKind::Call, 80_000, 42,
        );
        assert!((g.npv - 10.45).abs() < 0.5, "npv={}", g.npv);
    }

    #[test]
    fn european_simd4_wrapper() {
        let g = mc_european_simd4(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            OptionKind::Call, 100_000, 42,
        );
        assert!((g.npv - 10.45).abs() < 0.5, "npv={}", g.npv);
    }

    // -----------------------------------------------------------------------
    // MC Heston SIMD tests
    // -----------------------------------------------------------------------

    #[test]
    fn heston_simd_npv_positive() {
        let g = mc_heston_simd::<4>(
            100.0, 100.0, 0.05, 0.0,
            0.04, 2.0, 0.04, 0.3, -0.7,
            1.0, OptionKind::Call, 20_000, 50, 42,
        );
        assert!(g.npv > 0.0, "Heston call NPV should be positive, got {}", g.npv);
        assert!((g.npv - 10.0).abs() < 5.0, "npv={}", g.npv);
    }

    #[test]
    fn heston_simd_delta_in_range() {
        let g = mc_heston_simd::<4>(
            100.0, 100.0, 0.05, 0.0,
            0.04, 2.0, 0.04, 0.3, -0.7,
            1.0, OptionKind::Call, 20_000, 50, 42,
        );
        assert!(g.delta > 0.0 && g.delta < 1.0,
                "delta should be in (0,1), got {}", g.delta);
    }

    #[test]
    fn heston_simd_vega_v0_positive() {
        let g = mc_heston_simd::<4>(
            100.0, 100.0, 0.05, 0.0,
            0.04, 2.0, 0.04, 0.3, -0.7,
            1.0, OptionKind::Call, 20_000, 50, 42,
        );
        assert!(g.vega_v0 > 0.0, "vega_v0 should be positive, got {}", g.vega_v0);
    }

    #[test]
    fn heston_simd_greeks_finite() {
        let g = mc_heston_simd::<4>(
            100.0, 100.0, 0.05, 0.0,
            0.04, 2.0, 0.04, 0.3, -0.7,
            1.0, OptionKind::Call, 10_000, 50, 42,
        );
        assert!(g.delta.is_finite());
        assert!(g.vega_v0.is_finite());
        assert!(g.d_kappa.is_finite());
        assert!(g.d_theta.is_finite());
        assert!(g.d_sigma.is_finite());
        assert!(g.d_rho.is_finite());
        assert!(g.rho.is_finite());
        assert!(g.div_rho.is_finite());
    }

    #[test]
    fn heston_simd_npv_near_analytic() {
        use crate::heston::heston_price_generic;
        let analytic: f64 = heston_price_generic(
            100.0, 100.0, 0.05, 0.0,
            1.0, 0.04, 2.0, 0.04, 0.3, -0.7,
            true,
        );
        let mc = mc_heston_simd::<4>(
            100.0, 100.0, 0.05, 0.0,
            0.04, 2.0, 0.04, 0.3, -0.7,
            1.0, OptionKind::Call, 50_000, 100, 42,
        );
        assert!((mc.npv - analytic).abs() < 3.0 * mc.std_error + 0.5,
                "MC npv={} vs analytic={}, std_err={}", mc.npv, analytic, mc.std_error);
    }

    #[test]
    fn heston_simd4_wrapper() {
        let g = mc_heston_simd4(
            100.0, 100.0, 0.05, 0.0,
            0.04, 2.0, 0.04, 0.3, -0.7,
            1.0, OptionKind::Call, 20_000, 50, 42,
        );
        assert!(g.npv > 0.0);
        assert!(g.delta > 0.0 && g.delta < 1.0);
    }
}
