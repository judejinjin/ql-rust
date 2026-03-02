//! SIMD + JIT composition: vectorised forward evaluation with JIT-compiled
//! backward pass processing N Monte Carlo paths simultaneously.
//!
//! The scalar [`JitTape`](crate::jit::JitTape) records once and replays
//! via `forward_eval`, but processes only one path per adjoint call.
//! The [`SimdTape`](crate::simd::SimdTape) processes N paths per tape
//! traversal, but re-records the entire graph each batch.
//!
//! This module combines both: record once on a `JitTape`, then replay
//! the forward pass with N-lane ([`Lanes<N>`](crate::simd::Lanes)) values
//! and execute a JIT-compiled backward pass that processes all N lanes
//! per call.
//!
//! **Benefits over scalar JIT:**
//! - N× fewer forward\_eval + backward calls
//! - Better cache utilisation (N adjacent values per node)
//!
//! **Benefits over SIMD tape:**
//! - Record once, replay many (no per-batch tape construction)
//! - JIT-compiled backward pass (no interpretation overhead)
//!
//! # Example
//!
//! ```
//! use ql_aad::jit::JitTape;
//! use ql_aad::simd::Lanes;
//! use ql_aad::simd_jit::SimdJitContext;
//!
//! let mut tape = JitTape::new();
//! let x = tape.input(2.0);
//! let y = tape.input(3.0);
//! let z = tape.mul(x, y);
//!
//! let mut ctx = SimdJitContext::<4>::new(&tape, z);
//! let inputs = [Lanes::splat(2.0), Lanes::splat(3.0)];
//! ctx.forward_eval(&tape, &inputs, &[]);
//! ctx.fill_partials(&tape, z);
//! let adj = ctx.adjoint_interpreted(&tape, z);
//! // Each lane: ∂z/∂x = y = 3
//! assert!((adj[x.idx as usize].data[0] - 3.0).abs() < 1e-14);
//! ```

use crate::bs::OptionKind;
use crate::jit::{JitReal, JitTape, Op};
use crate::mc::{McEuropeanGreeks, McHestonGreeks};
use crate::simd::Lanes;

use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

// ===========================================================================
// Lane-wise helper functions for operations not on Lanes<N>
// ===========================================================================

#[inline]
fn lanes_tan<const N: usize>(x: Lanes<N>) -> Lanes<N> {
    let mut r = [0.0; N];
    for i in 0..N {
        r[i] = x.data[i].tan();
    }
    Lanes::from_array(r)
}

#[inline]
fn lanes_asin<const N: usize>(x: Lanes<N>) -> Lanes<N> {
    let mut r = [0.0; N];
    for i in 0..N {
        r[i] = x.data[i].asin();
    }
    Lanes::from_array(r)
}

#[inline]
fn lanes_acos<const N: usize>(x: Lanes<N>) -> Lanes<N> {
    let mut r = [0.0; N];
    for i in 0..N {
        r[i] = x.data[i].acos();
    }
    Lanes::from_array(r)
}

#[inline]
fn lanes_atan<const N: usize>(x: Lanes<N>) -> Lanes<N> {
    let mut r = [0.0; N];
    for i in 0..N {
        r[i] = x.data[i].atan();
    }
    Lanes::from_array(r)
}

#[inline]
fn lanes_sinh<const N: usize>(x: Lanes<N>) -> Lanes<N> {
    let mut r = [0.0; N];
    for i in 0..N {
        r[i] = x.data[i].sinh();
    }
    Lanes::from_array(r)
}

#[inline]
fn lanes_cosh<const N: usize>(x: Lanes<N>) -> Lanes<N> {
    let mut r = [0.0; N];
    for i in 0..N {
        r[i] = x.data[i].cosh();
    }
    Lanes::from_array(r)
}

#[inline]
fn lanes_tanh<const N: usize>(x: Lanes<N>) -> Lanes<N> {
    let mut r = [0.0; N];
    for i in 0..N {
        r[i] = x.data[i].tanh();
    }
    Lanes::from_array(r)
}

#[inline]
fn lanes_log2<const N: usize>(x: Lanes<N>) -> Lanes<N> {
    let mut r = [0.0; N];
    for i in 0..N {
        r[i] = x.data[i].log2();
    }
    Lanes::from_array(r)
}

#[inline]
fn lanes_log10<const N: usize>(x: Lanes<N>) -> Lanes<N> {
    let mut r = [0.0; N];
    for i in 0..N {
        r[i] = x.data[i].log10();
    }
    Lanes::from_array(r)
}

#[inline]
fn lanes_floor<const N: usize>(x: Lanes<N>) -> Lanes<N> {
    let mut r = [0.0; N];
    for i in 0..N {
        r[i] = x.data[i].floor();
    }
    Lanes::from_array(r)
}

#[inline]
fn lanes_ceil<const N: usize>(x: Lanes<N>) -> Lanes<N> {
    let mut r = [0.0; N];
    for i in 0..N {
        r[i] = x.data[i].ceil();
    }
    Lanes::from_array(r)
}

#[inline]
fn lanes_powi<const N: usize>(x: Lanes<N>, n: i32) -> Lanes<N> {
    let mut r = [0.0; N];
    for i in 0..N {
        r[i] = x.data[i].powi(n);
    }
    Lanes::from_array(r)
}

#[inline]
fn lanes_powf<const N: usize>(a: Lanes<N>, b: Lanes<N>) -> Lanes<N> {
    let mut r = [0.0; N];
    for i in 0..N {
        r[i] = a.data[i].powf(b.data[i]);
    }
    Lanes::from_array(r)
}

#[inline]
fn lanes_atan2<const N: usize>(a: Lanes<N>, b: Lanes<N>) -> Lanes<N> {
    let mut r = [0.0; N];
    for i in 0..N {
        r[i] = a.data[i].atan2(b.data[i]);
    }
    Lanes::from_array(r)
}

#[inline]
fn lanes_recip<const N: usize>(x: Lanes<N>) -> Lanes<N> {
    Lanes::splat(1.0) / x
}

// ===========================================================================
// SimdJitContext<N>
// ===========================================================================

/// Context for SIMD-width forward evaluation and partial computation on a
/// [`JitTape`].
///
/// Holds N-lane values, partials, and adjoint buffers that are reused
/// across MC batches. Created once for a given tape size, then used
/// repeatedly with different input values.
pub struct SimdJitContext<const N: usize> {
    /// Per-node values for N lanes (same indexing as JitTape nodes).
    pub(crate) values: Vec<Lanes<N>>,
    /// Flat partials buffer: `total_partials * N` f64 values.
    /// Layout: for scalar partial index `p`, lane `l` →
    ///   flat index `p * N + l`, i.e. N lanes are interleaved per partial.
    pub(crate) partials: Vec<f64>,
    /// Flat adjoint buffer: `num_nodes * N` f64 values.
    /// Layout: for node `i`, lane `l` → flat index `i * N + l`.
    pub(crate) adj: Vec<f64>,
    /// Number of scalar partial entries.
    total_partials: usize,
}

impl<const N: usize> SimdJitContext<N> {
    /// Create a context pre-allocated for the given tape and output node.
    pub fn new(tape: &JitTape, output: JitReal) -> Self {
        let num_nodes = output.idx as usize + 1;
        let total_partials = tape.total_partials(output);
        Self {
            values: vec![Lanes::zero(); tape.len()],
            partials: vec![0.0; total_partials * N],
            adj: vec![0.0; num_nodes * N],
            total_partials,
        }
    }

    /// Forward-evaluate the tape with N-lane inputs.
    ///
    /// `inputs` — one `Lanes<N>` per `Op::Input` node (in registration order).
    /// `ext_inputs` — one `Lanes<N>` per `Op::ExtInput` node.
    pub fn forward_eval(
        &mut self,
        tape: &JitTape,
        inputs: &[Lanes<N>],
        ext_inputs: &[Lanes<N>],
    ) {
        assert_eq!(inputs.len(), tape.input_indices.len());
        assert_eq!(ext_inputs.len(), tape.ext_input_indices.len());

        // Inject new input values
        for (i, &idx) in tape.input_indices.iter().enumerate() {
            self.values[idx] = inputs[i];
        }
        for (i, &idx) in tape.ext_input_indices.iter().enumerate() {
            self.values[idx] = ext_inputs[i];
        }

        // Replay forward
        let n = tape.ops.len();
        for i in 0..n {
            let ch = tape.children[i];
            let va = || self.values[ch[0] as usize];
            let vb = || self.values[ch[1] as usize];

            self.values[i] = match tape.ops[i] {
                Op::Input | Op::ExtInput => continue, // already set
                Op::Const(c) => Lanes::splat(c),
                Op::Neg => -va(),
                Op::Abs => va().abs(),
                Op::Exp => va().exp(),
                Op::Ln => va().ln(),
                Op::Sqrt => va().sqrt(),
                Op::Recip => lanes_recip(va()),
                Op::Sin => va().sin(),
                Op::Cos => va().cos(),
                Op::Tan => lanes_tan(va()),
                Op::Asin => lanes_asin(va()),
                Op::Acos => lanes_acos(va()),
                Op::Atan => lanes_atan(va()),
                Op::Sinh => lanes_sinh(va()),
                Op::Cosh => lanes_cosh(va()),
                Op::Tanh => lanes_tanh(va()),
                Op::Log2 => lanes_log2(va()),
                Op::Log10 => lanes_log10(va()),
                Op::Floor => lanes_floor(va()),
                Op::Ceil => lanes_ceil(va()),
                Op::Powi(n) => lanes_powi(va(), n),
                Op::MulConst(c) => va() * Lanes::splat(c),
                Op::AddConst(c) => va() + Lanes::splat(c),
                Op::Add => va() + vb(),
                Op::Sub => va() - vb(),
                Op::Mul => va() * vb(),
                Op::Div => va() / vb(),
                Op::Powf => lanes_powf(va(), vb()),
                Op::Max => va().max(vb()),
                Op::Min => va().min(vb()),
                Op::Atan2 => lanes_atan2(va(), vb()),
            };
        }
    }

    /// Fill the interleaved partials buffer for all nodes up to `output`.
    ///
    /// The partials are laid out so that scalar partial index `p`, lane `l`
    /// is at flat index `p * N + l`. This matches the layout expected by
    /// the JIT-compiled backward pass.
    pub fn fill_partials(&mut self, tape: &JitTape, output: JitReal) {
        let output_idx = output.idx as usize;
        let mut p_scalar = 0usize; // scalar partial offset

        for i in (0..=output_idx).rev() {
            let ch = tape.children[i];
            let vi = self.values[i];
            let va = || self.values[ch[0] as usize];
            let vb = || self.values[ch[1] as usize];

            match tape.ops[i] {
                // Leaf: no partials
                Op::Input | Op::ExtInput | Op::Const(_) => {}

                // ---- Unary (1 partial × N lanes) ----
                Op::Neg => {
                    for l in 0..N {
                        self.partials[p_scalar * N + l] = -1.0;
                    }
                    p_scalar += 1;
                }
                Op::Abs => {
                    let a = va();
                    for l in 0..N {
                        self.partials[p_scalar * N + l] =
                            if a.data[l] >= 0.0 { 1.0 } else { -1.0 };
                    }
                    p_scalar += 1;
                }
                Op::Exp => {
                    // ∂exp(a)/∂a = exp(a) = vi
                    for l in 0..N {
                        self.partials[p_scalar * N + l] = vi.data[l];
                    }
                    p_scalar += 1;
                }
                Op::Ln => {
                    let a = va();
                    for l in 0..N {
                        self.partials[p_scalar * N + l] = 1.0 / a.data[l];
                    }
                    p_scalar += 1;
                }
                Op::Sqrt => {
                    for l in 0..N {
                        self.partials[p_scalar * N + l] =
                            if vi.data[l] > 0.0 { 0.5 / vi.data[l] } else { 0.0 };
                    }
                    p_scalar += 1;
                }
                Op::Recip => {
                    // ∂(1/a)/∂a = -(1/a)² = -vi²
                    for l in 0..N {
                        self.partials[p_scalar * N + l] = -vi.data[l] * vi.data[l];
                    }
                    p_scalar += 1;
                }
                Op::Sin => {
                    let a = va();
                    for l in 0..N {
                        self.partials[p_scalar * N + l] = a.data[l].cos();
                    }
                    p_scalar += 1;
                }
                Op::Cos => {
                    let a = va();
                    for l in 0..N {
                        self.partials[p_scalar * N + l] = -a.data[l].sin();
                    }
                    p_scalar += 1;
                }
                Op::Tan => {
                    let a = va();
                    for l in 0..N {
                        let c = a.data[l].cos();
                        self.partials[p_scalar * N + l] = 1.0 / (c * c);
                    }
                    p_scalar += 1;
                }
                Op::Asin => {
                    let a = va();
                    for l in 0..N {
                        let v = a.data[l];
                        self.partials[p_scalar * N + l] =
                            1.0 / (1.0 - v * v).sqrt();
                    }
                    p_scalar += 1;
                }
                Op::Acos => {
                    let a = va();
                    for l in 0..N {
                        let v = a.data[l];
                        self.partials[p_scalar * N + l] =
                            -1.0 / (1.0 - v * v).sqrt();
                    }
                    p_scalar += 1;
                }
                Op::Atan => {
                    let a = va();
                    for l in 0..N {
                        let v = a.data[l];
                        self.partials[p_scalar * N + l] = 1.0 / (1.0 + v * v);
                    }
                    p_scalar += 1;
                }
                Op::Sinh => {
                    let a = va();
                    for l in 0..N {
                        self.partials[p_scalar * N + l] = a.data[l].cosh();
                    }
                    p_scalar += 1;
                }
                Op::Cosh => {
                    let a = va();
                    for l in 0..N {
                        self.partials[p_scalar * N + l] = a.data[l].sinh();
                    }
                    p_scalar += 1;
                }
                Op::Tanh => {
                    for l in 0..N {
                        self.partials[p_scalar * N + l] =
                            1.0 - vi.data[l] * vi.data[l];
                    }
                    p_scalar += 1;
                }
                Op::Log2 => {
                    let a = va();
                    for l in 0..N {
                        self.partials[p_scalar * N + l] =
                            1.0 / (a.data[l] * std::f64::consts::LN_2);
                    }
                    p_scalar += 1;
                }
                Op::Log10 => {
                    let a = va();
                    for l in 0..N {
                        self.partials[p_scalar * N + l] =
                            1.0 / (a.data[l] * std::f64::consts::LN_10);
                    }
                    p_scalar += 1;
                }
                Op::Floor | Op::Ceil => {
                    for l in 0..N {
                        self.partials[p_scalar * N + l] = 0.0;
                    }
                    p_scalar += 1;
                }
                Op::Powi(n) => {
                    let a = va();
                    for l in 0..N {
                        self.partials[p_scalar * N + l] =
                            n as f64 * a.data[l].powi(n - 1);
                    }
                    p_scalar += 1;
                }
                Op::MulConst(c) => {
                    for l in 0..N {
                        self.partials[p_scalar * N + l] = c;
                    }
                    p_scalar += 1;
                }
                Op::AddConst(_) => {
                    for l in 0..N {
                        self.partials[p_scalar * N + l] = 1.0;
                    }
                    p_scalar += 1;
                }

                // ---- Binary (2 partials × N lanes) ----
                Op::Add => {
                    for l in 0..N {
                        self.partials[p_scalar * N + l] = 1.0;
                        self.partials[(p_scalar + 1) * N + l] = 1.0;
                    }
                    p_scalar += 2;
                }
                Op::Sub => {
                    for l in 0..N {
                        self.partials[p_scalar * N + l] = 1.0;
                        self.partials[(p_scalar + 1) * N + l] = -1.0;
                    }
                    p_scalar += 2;
                }
                Op::Mul => {
                    let a = va();
                    let b = vb();
                    for l in 0..N {
                        self.partials[p_scalar * N + l] = b.data[l]; // ∂(ab)/∂a = b
                        self.partials[(p_scalar + 1) * N + l] = a.data[l]; // ∂(ab)/∂b = a
                    }
                    p_scalar += 2;
                }
                Op::Div => {
                    let a = va();
                    let b = vb();
                    for l in 0..N {
                        let inv_b = 1.0 / b.data[l];
                        self.partials[p_scalar * N + l] = inv_b;
                        self.partials[(p_scalar + 1) * N + l] =
                            -a.data[l] * inv_b * inv_b;
                    }
                    p_scalar += 2;
                }
                Op::Powf => {
                    let a = va();
                    let b = vb();
                    for l in 0..N {
                        self.partials[p_scalar * N + l] =
                            b.data[l] * a.data[l].powf(b.data[l] - 1.0);
                        self.partials[(p_scalar + 1) * N + l] =
                            vi.data[l] * a.data[l].ln();
                    }
                    p_scalar += 2;
                }
                Op::Max => {
                    let a = va();
                    let b = vb();
                    for l in 0..N {
                        let winner_a = a.data[l] >= b.data[l];
                        self.partials[p_scalar * N + l] =
                            if winner_a { 1.0 } else { 0.0 };
                        self.partials[(p_scalar + 1) * N + l] =
                            if winner_a { 0.0 } else { 1.0 };
                    }
                    p_scalar += 2;
                }
                Op::Min => {
                    let a = va();
                    let b = vb();
                    for l in 0..N {
                        let winner_a = a.data[l] <= b.data[l];
                        self.partials[p_scalar * N + l] =
                            if winner_a { 1.0 } else { 0.0 };
                        self.partials[(p_scalar + 1) * N + l] =
                            if winner_a { 0.0 } else { 1.0 };
                    }
                    p_scalar += 2;
                }
                Op::Atan2 => {
                    let a = va();
                    let b = vb();
                    for l in 0..N {
                        let denom = a.data[l] * a.data[l] + b.data[l] * b.data[l];
                        self.partials[p_scalar * N + l] = b.data[l] / denom;
                        self.partials[(p_scalar + 1) * N + l] =
                            -a.data[l] / denom;
                    }
                    p_scalar += 2;
                }
            }
        }
        debug_assert_eq!(p_scalar, self.total_partials);
    }

    /// Interpreted N-lane adjoint using the tape for topology info.
    ///
    /// Uses the partials buffer (must call [`fill_partials`](Self::fill_partials)
    /// first) to compute the adjoint for all N lanes simultaneously.
    /// Returns one `Lanes<N>` per node.
    pub fn adjoint_interpreted(
        &self,
        tape: &JitTape,
        output: JitReal,
    ) -> Vec<Lanes<N>> {
        let output_idx = output.idx as usize;
        let num_nodes = output_idx + 1;
        let mut adj_flat = vec![0.0_f64; num_nodes * N];

        // adj[output] = 1.0 for all lanes
        for l in 0..N {
            adj_flat[output_idx * N + l] = 1.0;
        }

        let mut p_scalar = 0usize;
        for i in (0..=output_idx).rev() {
            let nc = tape.ops[i].num_children();
            if nc == 0 {
                continue;
            }

            for l in 0..N {
                let a_i = adj_flat[i * N + l];
                if a_i == 0.0 {
                    continue;
                }
                for j in 0..nc {
                    let child = tape.children[i][j] as usize;
                    let p = self.partials[(p_scalar + j) * N + l];
                    adj_flat[child * N + l] += a_i * p;
                }
            }
            p_scalar += nc;
        }

        // Convert flat f64 buffer to Vec<Lanes<N>>
        let mut result = vec![Lanes::zero(); num_nodes];
        for i in 0..num_nodes {
            let mut data = [0.0; N];
            for l in 0..N {
                data[l] = adj_flat[i * N + l];
            }
            result[i] = Lanes::from_array(data);
        }
        result
    }

    /// Get the N-lane value of a node after forward evaluation.
    #[inline]
    pub fn value(&self, node: JitReal) -> Lanes<N> {
        self.values[node.idx as usize]
    }
}

// ===========================================================================
// CompiledAdjointSimd<N> — JIT-compiled N-lane backward pass
// ===========================================================================

/// A Cranelift JIT-compiled adjoint function that processes N lanes
/// (paths) simultaneously.
///
/// The compiled function operates on interleaved data: for node `i`,
/// lane `l`, the flat index is `i * N + l`. Both the partials and
/// adjoint arrays use this layout.
pub struct CompiledAdjointSimd<const N: usize> {
    func: unsafe extern "C" fn(*const f64, *mut f64),
    num_nodes: usize,
    output_idx: usize,
    total_partials: usize,
    // Must outlive `func` — JIT code lives in module memory.
    _module: JITModule,
}

impl<const N: usize> CompiledAdjointSimd<N> {
    /// Execute the compiled adjoint into the context's adjoint buffer.
    ///
    /// `ctx.partials` must have been filled by
    /// [`fill_partials`](SimdJitContext::fill_partials) first.
    pub fn execute_into_ctx(&self, ctx: &mut SimdJitContext<N>) {
        assert!(ctx.adj.len() >= self.num_nodes * N);
        assert!(ctx.partials.len() >= self.total_partials * N);
        ctx.adj[..self.num_nodes * N].fill(0.0);
        unsafe {
            (self.func)(ctx.partials.as_ptr(), ctx.adj.as_mut_ptr());
        }
    }

    /// Execute the compiled adjoint. Returns adjoint as `Vec<Lanes<N>>`.
    pub fn execute_with_ctx(&self, ctx: &mut SimdJitContext<N>) -> Vec<Lanes<N>> {
        self.execute_into_ctx(ctx);
        let num_nodes = self.num_nodes;
        let mut result = vec![Lanes::zero(); num_nodes];
        for i in 0..num_nodes {
            let mut data = [0.0; N];
            for l in 0..N {
                data[l] = ctx.adj[i * N + l];
            }
            result[i] = Lanes::from_array(data);
        }
        result
    }

    /// Read the adjoint for a specific node from the raw adjoint buffer.
    /// Must be called after `execute_into_ctx`.
    #[inline]
    pub fn read_adj(&self, ctx: &SimdJitContext<N>, node: JitReal) -> Lanes<N> {
        let idx = node.idx as usize;
        let mut data = [0.0; N];
        for l in 0..N {
            data[l] = ctx.adj[idx * N + l];
        }
        Lanes::from_array(data)
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Output node index.
    pub fn output_idx(&self) -> usize {
        self.output_idx
    }
}

// Safety: the JIT module and function pointer are thread-safe once finalized.
unsafe impl<const N: usize> Send for CompiledAdjointSimd<N> {}
unsafe impl<const N: usize> Sync for CompiledAdjointSimd<N> {}

// ===========================================================================
// JIT compilation of N-lane backward pass
// ===========================================================================

/// Compile an N-lane adjoint backward pass for the given tape and output.
///
/// The compiled function processes N paths simultaneously using an
/// unrolled per-lane approach: for each node, the load–fmul–fadd–store
/// sequence is emitted N times (once per lane) with baked-in byte offsets.
///
/// Data layout: node `i`, lane `l` → flat index `i * N + l`.
///
/// # Panics
///
/// Panics if the tape is too large for byte offsets to fit in `i32`.
pub fn compile_adjoint_simd<const N: usize>(
    tape: &JitTape,
    output: JitReal,
) -> CompiledAdjointSimd<N> {
    let output_idx = output.idx as usize;
    let num_nodes = output_idx + 1;
    let total_partials = tape.total_partials(output);

    // Ensure byte offsets fit in i32
    let max_adj_bytes = num_nodes * N * 8;
    let max_partial_bytes = total_partials * N * 8;
    assert!(
        max_adj_bytes <= i32::MAX as usize && max_partial_bytes <= i32::MAX as usize,
        "tape too large for SIMD JIT (adj={}, partials={} bytes)",
        max_adj_bytes,
        max_partial_bytes
    );

    // ---- Cranelift setup ----
    let mut flag_builder = settings::builder();
    flag_builder.set("opt_level", "speed").unwrap();
    flag_builder.set("is_pic", "false").unwrap();
    let isa_builder =
        cranelift_native::builder().expect("failed to create native ISA builder");
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .expect("failed to finish ISA");

    let builder =
        JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    let mut module = JITModule::new(builder);

    // Function signature: fn(partials: *const f64, adj: *mut f64)
    let ptr_type = module.target_config().pointer_type();
    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(ptr_type)); // partials
    sig.params.push(AbiParam::new(ptr_type)); // adj

    let func_id = module
        .declare_function("jit_adjoint_simd", Linkage::Local, &sig)
        .unwrap();

    let mut ctx = module.make_context();
    ctx.func.signature = sig;

    let mut builder_ctx = FunctionBuilderContext::new();
    {
        let mut fb = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);
        let block = fb.create_block();
        fb.append_block_params_for_function_params(block);
        fb.switch_to_block(block);
        fb.seal_block(block);

        let partials_ptr = fb.block_params(block)[0];
        let adj_ptr = fb.block_params(block)[1];

        // adj[output_idx * N + l] = 1.0 for all lanes
        let one = fb.ins().f64const(1.0);
        for l in 0..N {
            let byte_off = ((output_idx * N + l) * 8) as i32;
            fb.ins().store(MemFlags::new(), one, adj_ptr, byte_off);
        }

        // Emit adjoint accumulations: unrolled per lane
        let mut p_scalar: usize = 0;

        for i in (0..=output_idx).rev() {
            let nc = tape.ops[i].num_children();
            if nc == 0 {
                continue;
            }

            // For each lane, load adj[i], accumulate into children
            for l in 0..N {
                let adj_i_byte = ((i * N + l) * 8) as i32;
                let adj_i = fb.ins().load(
                    types::F64,
                    MemFlags::new(),
                    adj_ptr,
                    adj_i_byte,
                );

                for j in 0..nc {
                    let child = tape.children[i][j] as usize;
                    let child_byte = ((child * N + l) * 8) as i32;

                    // Partial for scalar offset (p_scalar + j), lane l
                    let p_byte = (((p_scalar + j) * N + l) * 8) as i32;
                    let p = fb.ins().load(
                        types::F64,
                        MemFlags::new(),
                        partials_ptr,
                        p_byte,
                    );

                    let product = fb.ins().fmul(adj_i, p);

                    let old = fb.ins().load(
                        types::F64,
                        MemFlags::new(),
                        adj_ptr,
                        child_byte,
                    );
                    let new_val = fb.ins().fadd(old, product);
                    fb.ins()
                        .store(MemFlags::new(), new_val, adj_ptr, child_byte);
                }
            }

            p_scalar += nc;
        }

        fb.ins().return_(&[]);
        fb.finalize();
    }

    module.define_function(func_id, &mut ctx).unwrap();
    module.clear_context(&mut ctx);
    module.finalize_definitions().unwrap();

    let code_ptr = module.get_finalized_function(func_id);
    let func: unsafe extern "C" fn(*const f64, *mut f64) =
        unsafe { std::mem::transmute(code_ptr) };

    CompiledAdjointSimd {
        func,
        num_nodes,
        output_idx,
        total_partials,
        _module: module,
    }
}

// ===========================================================================
// MC European with SIMD + JIT
// ===========================================================================

/// Price a European option via Monte Carlo with SIMD + JIT composition.
///
/// Records one path on a [`JitTape`], compiles an N-lane adjoint backward
/// pass, then processes N paths per iteration: vectorised forward evaluation
/// + JIT-compiled N-lane backward pass.
///
/// **Throughput**: N× fewer calls than scalar JIT, no per-batch tape
/// reconstruction unlike [`mc_european_simd`](crate::simd::mc_european_simd).
///
/// Uses antithetic variates: within each N-lane batch, lanes `2k` and
/// `2k+1` use `+z` and `−z` respectively.
#[allow(clippy::too_many_arguments)]
pub fn mc_european_simd_jit<const N: usize>(
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

    let phi = match option_kind {
        OptionKind::Call => 1.0,
        OptionKind::Put => -1.0,
    };

    // ---- Step 1: Record one path on JitTape ----
    let z0: f64 = StandardNormal.sample(&mut rng);
    let mut tape = JitTape::new();

    let s_in = tape.input(spot);   // input 0
    let r_in = tape.input(r);     // input 1
    let q_in = tape.input(q);     // input 2
    let v_in = tape.input(vol);   // input 3
    let z_in = tape.ext_input(z0); // ext 0

    let tau_c = tape.constant(time_to_expiry);
    let sqrt_t_c = tape.constant(time_to_expiry.sqrt());
    let half_c = tape.constant(0.5);
    let phi_c = tape.constant(phi);
    let strike_c = tape.constant(strike);
    let zero_c = tape.constant(0.0);

    // drift = (r - q - 0.5*v²) * tau
    let v_sq = tape.mul(v_in, v_in);
    let half_v_sq = tape.mul(half_c, v_sq);
    let rq = tape.sub(r_in, q_in);
    let drift_rate = tape.sub(rq, half_v_sq);
    let drift = tape.mul(drift_rate, tau_c);

    // diffusion = v * sqrt(tau) * z
    let v_sqrt_t = tape.mul(v_in, sqrt_t_c);
    let diffusion = tape.mul(v_sqrt_t, z_in);

    // S_T = spot * exp(drift + diffusion)
    let exponent = tape.add(drift, diffusion);
    let growth = tape.exp(exponent);
    let st = tape.mul(s_in, growth);

    // payoff = max(phi * (S_T - K), 0)
    let diff = tape.sub(st, strike_c);
    let intrinsic = tape.mul(phi_c, diff);
    let payoff = tape.max(intrinsic, zero_c);

    // discount = exp(-r * tau)
    let neg_r = tape.neg(r_in);
    let neg_r_tau = tape.mul(neg_r, tau_c);
    let disc = tape.exp(neg_r_tau);

    // npv = payoff * discount
    let npv_node = tape.mul(payoff, disc);

    // ---- Step 2: Compile N-lane adjoint ----
    let compiled = compile_adjoint_simd::<N>(&tape, npv_node);

    // ---- Step 3: Create context and accumulators ----
    let mut ctx = SimdJitContext::<N>::new(&tape, npv_node);

    let mut sum_npv = 0.0;
    let mut sum_npv_sq = 0.0;
    let mut sum_delta = 0.0;
    let mut sum_vega = 0.0;
    let mut sum_rho = 0.0;
    let mut sum_div_rho = 0.0;

    // Inputs: model parameters are shared across lanes
    let s_lanes = Lanes::splat(spot);
    let r_lanes = Lanes::splat(r);
    let q_lanes = Lanes::splat(q);
    let v_lanes = Lanes::splat(vol);

    // ---- Step 4: Process first batch (z0 already sampled) ----
    {
        let mut z_data = [0.0; N];
        z_data[0] = z0;
        z_data[1] = -z0;
        for k in 1..N / 2 {
            let z: f64 = StandardNormal.sample(&mut rng);
            z_data[2 * k] = z;
            z_data[2 * k + 1] = -z;
        }
        let z_lanes = Lanes::from_array(z_data);

        let inputs = [s_lanes, r_lanes, q_lanes, v_lanes];
        let ext_inputs = [z_lanes];
        ctx.forward_eval(&tape, &inputs, &ext_inputs);
        ctx.fill_partials(&tape, npv_node);
        compiled.execute_into_ctx(&mut ctx);

        let pv = ctx.value(npv_node);
        let adj_s = compiled.read_adj(&ctx, s_in);
        let adj_r = compiled.read_adj(&ctx, r_in);
        let adj_q = compiled.read_adj(&ctx, q_in);
        let adj_v = compiled.read_adj(&ctx, v_in);

        for l in 0..N {
            sum_npv += pv.data[l];
            sum_npv_sq += pv.data[l] * pv.data[l];
            sum_delta += adj_s.data[l];
            sum_rho += adj_r.data[l];
            sum_div_rho += adj_q.data[l];
            sum_vega += adj_v.data[l];
        }
    }

    // ---- Step 5: Remaining batches ----
    for _ in 1..num_batches {
        let mut z_data = [0.0; N];
        for k in 0..N / 2 {
            let z: f64 = StandardNormal.sample(&mut rng);
            z_data[2 * k] = z;
            z_data[2 * k + 1] = -z;
        }
        let z_lanes = Lanes::from_array(z_data);

        let inputs = [s_lanes, r_lanes, q_lanes, v_lanes];
        let ext_inputs = [z_lanes];
        ctx.forward_eval(&tape, &inputs, &ext_inputs);
        ctx.fill_partials(&tape, npv_node);
        compiled.execute_into_ctx(&mut ctx);

        let pv = ctx.value(npv_node);
        let adj_s = compiled.read_adj(&ctx, s_in);
        let adj_r = compiled.read_adj(&ctx, r_in);
        let adj_q = compiled.read_adj(&ctx, q_in);
        let adj_v = compiled.read_adj(&ctx, v_in);

        for l in 0..N {
            sum_npv += pv.data[l];
            sum_npv_sq += pv.data[l] * pv.data[l];
            sum_delta += adj_s.data[l];
            sum_rho += adj_r.data[l];
            sum_div_rho += adj_q.data[l];
            sum_vega += adj_v.data[l];
        }
    }

    let n = actual_paths as f64;
    let mean = sum_npv / n;
    let variance = (sum_npv_sq / n - mean * mean).max(0.0);

    McEuropeanGreeks {
        npv: mean,
        std_error: (variance / n).sqrt(),
        delta: sum_delta / n,
        vega: sum_vega / n,
        rho: sum_rho / n,
        div_rho: sum_div_rho / n,
        num_paths: actual_paths,
    }
}

// ===========================================================================
// MC Heston with SIMD + JIT
// ===========================================================================

/// Price a European option under Heston via Monte Carlo with SIMD + JIT.
///
/// Records one path's full simulation on a [`JitTape`], compiles an N-lane
/// adjoint, then replays N paths per iteration with vectorised forward
/// evaluation + JIT-compiled backward pass.
///
/// Antithetic pairing within lanes: lanes `2k` and `2k+1` share the same
/// random numbers with opposite sign.
#[allow(clippy::too_many_arguments)]
pub fn mc_heston_simd_jit<const N: usize>(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho_corr: f64,
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

    // ---- Step 1: Record one path on JitTape ----
    // Pre-generate random numbers for the first path
    let mut z1_first = Vec::with_capacity(num_steps);
    let mut z2_first = Vec::with_capacity(num_steps);
    for _ in 0..num_steps {
        z1_first.push(StandardNormal.sample(&mut rng));
        z2_first.push(StandardNormal.sample(&mut rng));
    }

    let mut tape = JitTape::new();

    // AD inputs (8)
    let s_in = tape.input(spot);
    let r_in = tape.input(r);
    let q_in = tape.input(q);
    let v0_in = tape.input(v0);
    let kappa_in = tape.input(kappa);
    let theta_in = tape.input(theta);
    let sigma_in = tape.input(sigma);
    let rho_in = tape.input(rho_corr);

    // External inputs: z1 and z2 per step
    let mut z1_ext = Vec::with_capacity(num_steps);
    let mut z2_ext = Vec::with_capacity(num_steps);
    for step in 0..num_steps {
        z1_ext.push(tape.ext_input(z1_first[step]));
        z2_ext.push(tape.ext_input(z2_first[step]));
    }

    // Constants
    let dt_c = tape.constant(dt);
    let sqrt_dt_c = tape.constant(sqrt_dt);
    let half_c = tape.constant(0.5);
    let one_c = tape.constant(1.0);
    let zero_c = tape.constant(0.0);
    let phi_c = tape.constant(phi);
    let strike_c = tape.constant(strike);

    // Simulation: log-Euler for spot, Euler + full truncation for variance
    let mut log_s = tape.ln(s_in);
    let mut v_cur = v0_in;

    for step in 0..num_steps {
        // v_plus = max(v, 0)
        let v_plus = tape.max(v_cur, zero_c);
        let sqrt_v = tape.sqrt(v_plus);

        // Correlated normals: w1 = z1, w2 = rho*z1 + sqrt(1-rho²)*z2
        let z1 = z1_ext[step];
        let z2 = z2_ext[step];
        let rho_z1 = tape.mul(rho_in, z1);
        let one_minus_rho2 = {
            let rho2 = tape.mul(rho_in, rho_in);
            tape.sub(one_c, rho2)
        };
        let sqrt_1m_rho2 = tape.sqrt(one_minus_rho2);
        let comp = tape.mul(sqrt_1m_rho2, z2);
        let w2 = tape.add(rho_z1, comp);

        // Log-spot: log_s += (r - q - v/2)*dt + sqrt(v)*sqrt(dt)*z1
        let half_v = tape.mul(half_c, v_plus);
        let r_minus_q = tape.sub(r_in, q_in);
        let drift_rate = tape.sub(r_minus_q, half_v);
        let drift = tape.mul(drift_rate, dt_c);

        let vol_part = tape.mul(sqrt_v, sqrt_dt_c);
        let diffusion = tape.mul(vol_part, z1);

        let log_s_new = tape.add(log_s, drift);
        log_s = tape.add(log_s_new, diffusion);

        // Variance: v += kappa*(theta - v)*dt + sigma*sqrt(v)*sqrt(dt)*w2
        let mean_rev = tape.sub(theta_in, v_cur);
        let kappa_mr = tape.mul(kappa_in, mean_rev);
        let v_drift = tape.mul(kappa_mr, dt_c);

        let sigma_sqrt_v = tape.mul(sigma_in, sqrt_v);
        let v_vol = tape.mul(sigma_sqrt_v, sqrt_dt_c);
        let v_diffusion = tape.mul(v_vol, w2);

        let v_new = tape.add(v_cur, v_drift);
        v_cur = tape.add(v_new, v_diffusion);
    }

    // Terminal payoff
    let s_t = tape.exp(log_s);
    let diff = tape.sub(s_t, strike_c);
    let intrinsic = tape.mul(phi_c, diff);
    let payoff = tape.max(intrinsic, zero_c);

    // Discount
    let neg_r = tape.neg(r_in);
    let tau_c = tape.constant(time_to_expiry);
    let neg_r_tau = tape.mul(neg_r, tau_c);
    let disc = tape.exp(neg_r_tau);
    let npv_node = tape.mul(payoff, disc);

    // ---- Step 2: Compile N-lane adjoint ----
    let compiled = compile_adjoint_simd::<N>(&tape, npv_node);

    // ---- Step 3: Create context and accumulators ----
    let mut ctx = SimdJitContext::<N>::new(&tape, npv_node);

    let mut sum_npv = 0.0;
    let mut sum_npv_sq = 0.0;
    let mut sum_greeks = [0.0_f64; 8];

    let input_nodes = [s_in, r_in, q_in, v0_in, kappa_in, theta_in, sigma_in, rho_in];

    // ---- Step 4: Process first batch ----
    {
        let inputs: Vec<Lanes<N>> = [spot, r, q, v0, kappa, theta, sigma, rho_corr]
            .iter()
            .map(|&v| Lanes::splat(v))
            .collect();

        // Build ext_inputs: z1 and z2 per step, with antithetic pairing
        let mut ext_inputs: Vec<Lanes<N>> = Vec::with_capacity(2 * num_steps);
        for step in 0..num_steps {
            // z1 for this step
            let mut z1_data = [0.0; N];
            z1_data[0] = z1_first[step];
            z1_data[1] = -z1_first[step];
            for k in 1..N / 2 {
                let z: f64 = StandardNormal.sample(&mut rng);
                z1_data[2 * k] = z;
                z1_data[2 * k + 1] = -z;
            }
            ext_inputs.push(Lanes::from_array(z1_data));

            // z2 for this step
            let mut z2_data = [0.0; N];
            z2_data[0] = z2_first[step];
            z2_data[1] = -z2_first[step];
            for k in 1..N / 2 {
                let z: f64 = StandardNormal.sample(&mut rng);
                z2_data[2 * k] = z;
                z2_data[2 * k + 1] = -z;
            }
            ext_inputs.push(Lanes::from_array(z2_data));
        }

        ctx.forward_eval(&tape, &inputs, &ext_inputs);
        ctx.fill_partials(&tape, npv_node);
        compiled.execute_into_ctx(&mut ctx);

        let pv = ctx.value(npv_node);
        for l in 0..N {
            sum_npv += pv.data[l];
            sum_npv_sq += pv.data[l] * pv.data[l];
        }
        for (g, &node) in sum_greeks.iter_mut().zip(input_nodes.iter()) {
            let adj = compiled.read_adj(&ctx, node);
            for l in 0..N {
                *g += adj.data[l];
            }
        }
    }

    // ---- Step 5: Remaining batches ----
    for _ in 1..num_batches {
        let inputs: Vec<Lanes<N>> = [spot, r, q, v0, kappa, theta, sigma, rho_corr]
            .iter()
            .map(|&v| Lanes::splat(v))
            .collect();

        let mut ext_inputs: Vec<Lanes<N>> = Vec::with_capacity(2 * num_steps);
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
            ext_inputs.push(Lanes::from_array(z1_data));
            ext_inputs.push(Lanes::from_array(z2_data));
        }

        ctx.forward_eval(&tape, &inputs, &ext_inputs);
        ctx.fill_partials(&tape, npv_node);
        compiled.execute_into_ctx(&mut ctx);

        let pv = ctx.value(npv_node);
        for l in 0..N {
            sum_npv += pv.data[l];
            sum_npv_sq += pv.data[l] * pv.data[l];
        }
        for (g, &node) in sum_greeks.iter_mut().zip(input_nodes.iter()) {
            let adj = compiled.read_adj(&ctx, node);
            for l in 0..N {
                *g += adj.data[l];
            }
        }
    }

    let n = actual_paths as f64;
    let mean = sum_npv / n;
    let variance = (sum_npv_sq / n - mean * mean).max(0.0);

    McHestonGreeks {
        npv: mean,
        std_error: (variance / n).sqrt(),
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

/// Convenience wrapper: 4-lane (AVX2-width) European MC with SIMD + JIT.
#[allow(clippy::too_many_arguments)]
pub fn mc_european_simd_jit4(
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
    mc_european_simd_jit::<4>(spot, strike, r, q, vol, time_to_expiry, option_kind, num_paths, seed)
}

/// Convenience wrapper: 4-lane (AVX2-width) Heston MC with SIMD + JIT.
#[allow(clippy::too_many_arguments)]
pub fn mc_heston_simd_jit4(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho_corr: f64,
    time_to_expiry: f64,
    option_kind: OptionKind,
    num_paths: usize,
    num_steps: usize,
    seed: u64,
) -> McHestonGreeks {
    mc_heston_simd_jit::<4>(
        spot, strike, r, q, v0, kappa, theta, sigma, rho_corr,
        time_to_expiry, option_kind, num_paths, num_steps, seed,
    )
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Basic tape operations ──────────────────────────────────────────

    #[test]
    fn simple_mul_adjoint_matches_scalar() {
        let mut tape = JitTape::new();
        let x = tape.input(3.0);
        let y = tape.input(5.0);
        let z = tape.mul(x, y);

        // Scalar JIT reference
        let adj_ref = tape.adjoint(z);
        assert!((adj_ref[x.idx as usize] - 5.0).abs() < 1e-14);
        assert!((adj_ref[y.idx as usize] - 3.0).abs() < 1e-14);

        // SIMD JIT with N=4, all lanes identical
        let mut ctx = SimdJitContext::<4>::new(&tape, z);
        let inputs = [Lanes::splat(3.0), Lanes::splat(5.0)];
        ctx.forward_eval(&tape, &inputs, &[]);
        ctx.fill_partials(&tape, z);

        let adj_simd = ctx.adjoint_interpreted(&tape, z);
        for l in 0..4 {
            assert!(
                (adj_simd[x.idx as usize].data[l] - 5.0).abs() < 1e-14,
                "lane {l}: dz/dx"
            );
            assert!(
                (adj_simd[y.idx as usize].data[l] - 3.0).abs() < 1e-14,
                "lane {l}: dz/dy"
            );
        }
    }

    #[test]
    fn jit_compiled_simd_matches_interpreted() {
        let mut tape = JitTape::new();
        let x = tape.input(2.0);
        let y = tape.input(3.0);
        let xy = tape.mul(x, y);
        let z = tape.exp(xy);

        let mut ctx = SimdJitContext::<4>::new(&tape, z);
        let inputs = [Lanes::splat(2.0), Lanes::splat(3.0)];
        ctx.forward_eval(&tape, &inputs, &[]);
        ctx.fill_partials(&tape, z);

        // Interpreted reference
        let adj_interp = ctx.adjoint_interpreted(&tape, z);

        // JIT compiled
        let compiled = compile_adjoint_simd::<4>(&tape, z);
        compiled.execute_into_ctx(&mut ctx);
        let adj_jit_x = compiled.read_adj(&ctx, x);
        let adj_jit_y = compiled.read_adj(&ctx, y);

        for l in 0..4 {
            assert!(
                (adj_jit_x.data[l] - adj_interp[x.idx as usize].data[l]).abs() < 1e-12,
                "lane {l}: dz/dx JIT vs interp"
            );
            assert!(
                (adj_jit_y.data[l] - adj_interp[y.idx as usize].data[l]).abs() < 1e-12,
                "lane {l}: dz/dy JIT vs interp"
            );
        }
    }

    #[test]
    fn divergent_lanes_max() {
        // Different lanes take different branches in max
        let mut tape = JitTape::new();
        let x = tape.input(1.0);
        let k = tape.constant(0.5);
        let diff = tape.sub(x, k);
        let zero = tape.constant(0.0);
        let payoff = tape.max(diff, zero);

        let mut ctx = SimdJitContext::<4>::new(&tape, payoff);
        // Lane 0: x=1.0 → payoff = 0.5, lane 2: x=0.3 → payoff = 0
        let x_lanes = Lanes::from_array([1.0, 0.8, 0.3, 0.1]);
        ctx.forward_eval(&tape, &[x_lanes], &[]);
        ctx.fill_partials(&tape, payoff);

        let compiled = compile_adjoint_simd::<4>(&tape, payoff);
        compiled.execute_into_ctx(&mut ctx);
        let adj_x = compiled.read_adj(&ctx, x);

        // Lanes 0,1 are ITM → delta = 1.0; Lanes 2,3 are OTM → delta = 0.0
        assert!((adj_x.data[0] - 1.0).abs() < 1e-14, "lane 0 ITM");
        assert!((adj_x.data[1] - 1.0).abs() < 1e-14, "lane 1 ITM");
        assert!((adj_x.data[2]).abs() < 1e-14, "lane 2 OTM");
        assert!((adj_x.data[3]).abs() < 1e-14, "lane 3 OTM");
    }

    #[test]
    fn forward_eval_different_lane_values() {
        let mut tape = JitTape::new();
        let x = tape.input(1.0);
        let z = tape.ext_input(0.5);
        let xz = tape.mul(x, z);
        let out = tape.exp(xz);

        let mut ctx = SimdJitContext::<2>::new(&tape, out);
        let x_lanes = Lanes::from_array([2.0, 3.0]);
        let z_lanes = Lanes::from_array([0.1, 0.2]);
        ctx.forward_eval(&tape, &[x_lanes], &[z_lanes]);

        let val = ctx.value(out);
        assert!((val.data[0] - (2.0 * 0.1_f64).exp()).abs() < 1e-12);
        assert!((val.data[1] - (3.0 * 0.2_f64).exp()).abs() < 1e-12);
    }

    // ── MC European SIMD + JIT ──────────────────────────────────────────

    #[test]
    fn mc_european_simd_jit_call_near_bs() {
        let bs = crate::bs::bs_price_f64(100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
                                      OptionKind::Call);
        let mc = mc_european_simd_jit::<4>(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            OptionKind::Call, 200_000, 42,
        );
        assert!(
            (mc.npv - bs).abs() < 0.15,
            "MC SIMD-JIT call {:.4} vs BS {:.4}",
            mc.npv, bs
        );
    }

    #[test]
    fn mc_european_simd_jit_put_near_bs() {
        let bs = crate::bs::bs_price_f64(100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
                                      OptionKind::Put);
        let mc = mc_european_simd_jit::<4>(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            OptionKind::Put, 200_000, 42,
        );
        assert!(
            (mc.npv - bs).abs() < 0.15,
            "MC SIMD-JIT put {:.4} vs BS {:.4}",
            mc.npv, bs
        );
    }

    #[test]
    fn mc_european_simd_jit_delta_positive_call() {
        let mc = mc_european_simd_jit::<4>(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            OptionKind::Call, 200_000, 42,
        );
        assert!(mc.delta > 0.0, "call delta should be positive: {}", mc.delta);
    }

    #[test]
    fn mc_european_simd_jit_delta_near_scalar_jit() {
        let scalar = crate::jit::mc_european_jit(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            OptionKind::Call, 200_000, 42,
        );
        let simd = mc_european_simd_jit::<4>(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            OptionKind::Call, 200_000, 42,
        );
        // Different RNG streams so allow wider tolerance
        assert!(
            (simd.delta - scalar.delta).abs() < 0.02,
            "SIMD-JIT delta {:.5} vs scalar {:.5}",
            simd.delta, scalar.delta
        );
    }

    #[test]
    fn mc_european_simd_jit_vega_positive_call() {
        let mc = mc_european_simd_jit::<4>(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            OptionKind::Call, 200_000, 42,
        );
        assert!(mc.vega > 0.0, "call vega should be positive: {}", mc.vega);
    }

    // ── MC Heston SIMD + JIT ──────────────────────────────────────────

    #[test]
    fn mc_heston_simd_jit_call_positive_npv() {
        let mc = mc_heston_simd_jit::<4>(
            100.0, 100.0, 0.05, 0.0,
            0.04, 2.0, 0.04, 0.3, -0.7,
            1.0, OptionKind::Call, 40_000, 50, 42,
        );
        assert!(mc.npv > 0.0, "Heston call NPV should be positive: {}", mc.npv);
    }

    #[test]
    fn mc_heston_simd_jit_delta_positive_call() {
        let mc = mc_heston_simd_jit::<4>(
            100.0, 100.0, 0.05, 0.0,
            0.04, 2.0, 0.04, 0.3, -0.7,
            1.0, OptionKind::Call, 40_000, 50, 42,
        );
        assert!(
            mc.delta > 0.0,
            "Heston call delta should be positive: {}",
            mc.delta
        );
    }

    #[test]
    fn mc_heston_simd_jit_near_scalar() {
        let scalar = crate::jit::mc_heston_jit(
            100.0, 100.0, 0.05, 0.0,
            0.04, 2.0, 0.04, 0.3, -0.7,
            1.0, OptionKind::Call, 40_000, 50, 42,
        );
        let simd = mc_heston_simd_jit::<4>(
            100.0, 100.0, 0.05, 0.0,
            0.04, 2.0, 0.04, 0.3, -0.7,
            1.0, OptionKind::Call, 40_000, 50, 42,
        );
        // Different RNG streams — use wider tolerance
        assert!(
            (simd.npv - scalar.npv).abs() < 1.0,
            "SIMD-JIT Heston NPV {:.4} vs scalar {:.4}",
            simd.npv, scalar.npv
        );
    }

    // ── N=2 (minimal) ──────────────────────────────────────────────────

    #[test]
    fn mc_european_simd_jit_n2() {
        let bs = crate::bs::bs_price_f64(100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
                                      OptionKind::Call);
        let mc = mc_european_simd_jit::<2>(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            OptionKind::Call, 200_000, 42,
        );
        assert!(
            (mc.npv - bs).abs() < 0.15,
            "N=2 MC SIMD-JIT call {:.4} vs BS {:.4}",
            mc.npv, bs
        );
    }

    // ── Convenience wrappers ───────────────────────────────────────────

    #[test]
    fn convenience_wrappers_compile() {
        let mc = mc_european_simd_jit4(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            OptionKind::Call, 10_000, 42,
        );
        assert!(mc.npv > 0.0);

        let mc_h = mc_heston_simd_jit4(
            100.0, 100.0, 0.05, 0.0,
            0.04, 2.0, 0.04, 0.3, -0.7,
            1.0, OptionKind::Call, 10_000, 20, 42,
        );
        assert!(mc_h.npv > 0.0);
    }
}
