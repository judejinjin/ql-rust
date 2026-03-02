//! Cranelift JIT compilation of the adjoint backward pass.
//!
//! This module provides [`JitTape`], a computation tape that records operations
//! with explicit op-codes. After one forward recording the adjoint pass can be
//! compiled to native code via Cranelift — sub-millisecond compilation, then
//! thousands of re-evaluations at full native speed with zero interpretation
//! overhead.
//!
//! # Architecture
//!
//! ```text
//!  Record once    →   Compile adjoint   →   Per-path loop:
//!  (JitTape)          (Cranelift JIT)        forward_eval → fill_partials → execute
//! ```
//!
//! The compiled function is a straight-line sequence of load–fmul–fadd–store
//! instructions with all topology (child indices) resolved at compile time.
//! No loops, no branches, no bounds checks, no `SmallVec` pointer chasing.
//!
//! # Example
//!
//! ```
//! use ql_aad::jit::JitTape;
//!
//! let mut tape = JitTape::new();
//! let x = tape.input(3.0);
//! let y = tape.input(5.0);
//! let z = tape.mul(x, y);    // z = 15
//!
//! // Interpreted adjoint (reference)
//! let adj_ref = tape.adjoint(z);
//! assert!((adj_ref[x.idx as usize] - 5.0).abs() < 1e-14);
//!
//! // JIT-compiled adjoint
//! let compiled = tape.compile_adjoint(z);
//! let mut partials = vec![0.0; tape.total_partials(z)];
//! tape.fill_partials_into(z, &mut partials);
//! let adj_jit = compiled.execute(&partials);
//! assert!((adj_jit[x.idx as usize] - 5.0).abs() < 1e-14);
//! ```

use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use crate::bs::OptionKind;
use crate::mc::{McEuropeanGreeks, McHestonGreeks};

use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

/// Sentinel for "no child" in the children array.
const NO_CHILD: u32 = u32::MAX;

// ===========================================================================
// Op enum
// ===========================================================================

/// Operation recorded on the [`JitTape`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    // ---- Leaf (0 children) ----
    /// AD input — contributes to gradients.
    Input,
    /// External input — no gradient (e.g. random numbers).
    ExtInput,
    /// Compile-time constant.
    Const(f64),

    // ---- Unary (1 child: children\[0\]) ----
    Neg,
    Abs,
    Exp,
    Ln,
    Sqrt,
    Recip,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Log2,
    Log10,
    Floor,
    Ceil,
    /// Integer power with exponent baked in.
    Powi(i32),
    /// Multiply by a compile-time constant.
    MulConst(f64),
    /// Add a compile-time constant.
    AddConst(f64),

    // ---- Binary (2 children: children\[0\], children\[1\]) ----
    Add,
    Sub,
    Mul,
    Div,
    Powf,
    /// `max(a, b)` — always records both children (fixed topology).
    Max,
    /// `min(a, b)` — always records both children (fixed topology).
    Min,
    Atan2,
}

impl Op {
    /// Number of children this operation depends on.
    #[inline]
    pub(crate) fn num_children(self) -> usize {
        match self {
            Op::Input | Op::ExtInput | Op::Const(_) => 0,
            Op::Neg | Op::Abs | Op::Exp | Op::Ln | Op::Sqrt | Op::Recip
            | Op::Sin | Op::Cos | Op::Tan | Op::Asin | Op::Acos | Op::Atan
            | Op::Sinh | Op::Cosh | Op::Tanh | Op::Log2 | Op::Log10
            | Op::Floor | Op::Ceil | Op::Powi(_) | Op::MulConst(_) | Op::AddConst(_) => 1,
            Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Powf
            | Op::Max | Op::Min | Op::Atan2 => 2,
        }
    }
}

// ===========================================================================
// JitReal
// ===========================================================================

/// Handle to a node on the [`JitTape`].
#[derive(Clone, Copy, Debug)]
pub struct JitReal {
    /// Index on the tape.
    pub idx: u32,
    /// Cached forward value.
    pub val: f64,
}

// ===========================================================================
// JitTape
// ===========================================================================

/// A computation tape with explicit op-codes, designed for Cranelift JIT
/// compilation of the adjoint backward pass.
///
/// Unlike [`Tape`](crate::tape::Tape) which stores pre-computed partial
/// derivatives in a `SmallVec`, `JitTape` stores typed [`Op`] codes and
/// computes partials during [`fill_partials_into`](Self::fill_partials_into).
/// This enables topology-fixed replay: the same tape structure is re-evaluated
/// with different input values (e.g. per MC path).
pub struct JitTape {
    pub(crate) ops: Vec<Op>,
    pub(crate) children: Vec<[u32; 2]>,
    /// Forward values, populated during recording and `forward_eval`.
    pub(crate) values: Vec<f64>,
    /// Indices of `Op::Input` nodes.
    pub(crate) input_indices: Vec<usize>,
    /// Indices of `Op::ExtInput` nodes.
    pub(crate) ext_input_indices: Vec<usize>,
}

impl JitTape {
    /// Create an empty tape.
    pub fn new() -> Self {
        Self {
            ops: Vec::with_capacity(256),
            children: Vec::with_capacity(256),
            values: Vec::with_capacity(256),
            input_indices: Vec::new(),
            ext_input_indices: Vec::new(),
        }
    }

    /// Number of nodes on the tape.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Whether the tape is empty.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    // ---- internal push ----

    fn push(&mut self, op: Op, ch: [u32; 2], val: f64) -> JitReal {
        let idx = self.ops.len() as u32;
        self.ops.push(op);
        self.children.push(ch);
        self.values.push(val);
        JitReal { idx, val }
    }

    // ---- leaf constructors ----

    /// Register an AD input (gradient will be computed for this node).
    pub fn input(&mut self, val: f64) -> JitReal {
        let r = self.push(Op::Input, [NO_CHILD; 2], val);
        self.input_indices.push(r.idx as usize);
        r
    }

    /// Register an external input (no gradient). Used for random numbers
    /// that change per MC path but are not differentiated.
    pub fn ext_input(&mut self, val: f64) -> JitReal {
        let r = self.push(Op::ExtInput, [NO_CHILD; 2], val);
        self.ext_input_indices.push(r.idx as usize);
        r
    }

    /// Record a compile-time constant.
    pub fn constant(&mut self, val: f64) -> JitReal {
        self.push(Op::Const(val), [NO_CHILD; 2], val)
    }

    // ---- unary operations ----

    /// Negate: `-a`.
    pub fn neg(&mut self, a: JitReal) -> JitReal {
        self.push(Op::Neg, [a.idx, NO_CHILD], -a.val)
    }

    /// Absolute value: `|a|`.
    pub fn abs(&mut self, a: JitReal) -> JitReal {
        self.push(Op::Abs, [a.idx, NO_CHILD], a.val.abs())
    }

    /// Exponential: `exp(a)`.
    pub fn exp(&mut self, a: JitReal) -> JitReal {
        self.push(Op::Exp, [a.idx, NO_CHILD], a.val.exp())
    }

    /// Natural log: `ln(a)`.
    pub fn ln(&mut self, a: JitReal) -> JitReal {
        self.push(Op::Ln, [a.idx, NO_CHILD], a.val.ln())
    }

    /// Square root: `√a`.
    pub fn sqrt(&mut self, a: JitReal) -> JitReal {
        self.push(Op::Sqrt, [a.idx, NO_CHILD], a.val.sqrt())
    }

    /// Reciprocal: `1/a`.
    pub fn recip(&mut self, a: JitReal) -> JitReal {
        self.push(Op::Recip, [a.idx, NO_CHILD], 1.0 / a.val)
    }

    /// Sine.
    pub fn sin(&mut self, a: JitReal) -> JitReal {
        self.push(Op::Sin, [a.idx, NO_CHILD], a.val.sin())
    }

    /// Cosine.
    pub fn cos(&mut self, a: JitReal) -> JitReal {
        self.push(Op::Cos, [a.idx, NO_CHILD], a.val.cos())
    }

    /// Tangent.
    pub fn tan(&mut self, a: JitReal) -> JitReal {
        self.push(Op::Tan, [a.idx, NO_CHILD], a.val.tan())
    }

    /// Arc sine.
    pub fn asin(&mut self, a: JitReal) -> JitReal {
        self.push(Op::Asin, [a.idx, NO_CHILD], a.val.asin())
    }

    /// Arc cosine.
    pub fn acos(&mut self, a: JitReal) -> JitReal {
        self.push(Op::Acos, [a.idx, NO_CHILD], a.val.acos())
    }

    /// Arc tangent.
    pub fn atan(&mut self, a: JitReal) -> JitReal {
        self.push(Op::Atan, [a.idx, NO_CHILD], a.val.atan())
    }

    /// Hyperbolic sine.
    pub fn sinh(&mut self, a: JitReal) -> JitReal {
        self.push(Op::Sinh, [a.idx, NO_CHILD], a.val.sinh())
    }

    /// Hyperbolic cosine.
    pub fn cosh(&mut self, a: JitReal) -> JitReal {
        self.push(Op::Cosh, [a.idx, NO_CHILD], a.val.cosh())
    }

    /// Hyperbolic tangent.
    pub fn tanh(&mut self, a: JitReal) -> JitReal {
        self.push(Op::Tanh, [a.idx, NO_CHILD], a.val.tanh())
    }

    /// Log base 2.
    pub fn log2(&mut self, a: JitReal) -> JitReal {
        self.push(Op::Log2, [a.idx, NO_CHILD], a.val.log2())
    }

    /// Log base 10.
    pub fn log10(&mut self, a: JitReal) -> JitReal {
        self.push(Op::Log10, [a.idx, NO_CHILD], a.val.log10())
    }

    /// Floor.
    pub fn floor(&mut self, a: JitReal) -> JitReal {
        self.push(Op::Floor, [a.idx, NO_CHILD], a.val.floor())
    }

    /// Ceil.
    pub fn ceil(&mut self, a: JitReal) -> JitReal {
        self.push(Op::Ceil, [a.idx, NO_CHILD], a.val.ceil())
    }

    /// Integer power: `a^n`.
    pub fn powi(&mut self, a: JitReal, n: i32) -> JitReal {
        self.push(Op::Powi(n), [a.idx, NO_CHILD], a.val.powi(n))
    }

    /// Multiply by constant: `a * c`.
    pub fn mul_const(&mut self, a: JitReal, c: f64) -> JitReal {
        self.push(Op::MulConst(c), [a.idx, NO_CHILD], a.val * c)
    }

    /// Add constant: `a + c`.
    pub fn add_const(&mut self, a: JitReal, c: f64) -> JitReal {
        self.push(Op::AddConst(c), [a.idx, NO_CHILD], a.val + c)
    }

    // ---- binary operations ----

    /// Addition: `a + b`.
    pub fn add(&mut self, a: JitReal, b: JitReal) -> JitReal {
        self.push(Op::Add, [a.idx, b.idx], a.val + b.val)
    }

    /// Subtraction: `a - b`.
    pub fn sub(&mut self, a: JitReal, b: JitReal) -> JitReal {
        self.push(Op::Sub, [a.idx, b.idx], a.val - b.val)
    }

    /// Multiplication: `a * b`.
    pub fn mul(&mut self, a: JitReal, b: JitReal) -> JitReal {
        self.push(Op::Mul, [a.idx, b.idx], a.val * b.val)
    }

    /// Division: `a / b`.
    pub fn div(&mut self, a: JitReal, b: JitReal) -> JitReal {
        self.push(Op::Div, [a.idx, b.idx], a.val / b.val)
    }

    /// Power: `a^b`.
    pub fn powf(&mut self, a: JitReal, b: JitReal) -> JitReal {
        self.push(Op::Powf, [a.idx, b.idx], a.val.powf(b.val))
    }

    /// Max with fixed topology (both children always recorded).
    pub fn max(&mut self, a: JitReal, b: JitReal) -> JitReal {
        self.push(Op::Max, [a.idx, b.idx], a.val.max(b.val))
    }

    /// Min with fixed topology (both children always recorded).
    pub fn min(&mut self, a: JitReal, b: JitReal) -> JitReal {
        self.push(Op::Min, [a.idx, b.idx], a.val.min(b.val))
    }

    /// `atan2(a, b)`.
    pub fn atan2(&mut self, a: JitReal, b: JitReal) -> JitReal {
        self.push(Op::Atan2, [a.idx, b.idx], a.val.atan2(b.val))
    }

    // =====================================================================
    // Forward evaluation
    // =====================================================================

    /// Replay the tape operations with new input values.
    ///
    /// `inputs` — values for `Op::Input` nodes (in registration order).
    /// `ext_inputs` — values for `Op::ExtInput` nodes (in registration order).
    pub fn forward_eval(&mut self, inputs: &[f64], ext_inputs: &[f64]) {
        assert_eq!(inputs.len(), self.input_indices.len());
        assert_eq!(ext_inputs.len(), self.ext_input_indices.len());

        // Inject new input values
        for (i, &idx) in self.input_indices.iter().enumerate() {
            self.values[idx] = inputs[i];
        }
        for (i, &idx) in self.ext_input_indices.iter().enumerate() {
            self.values[idx] = ext_inputs[i];
        }

        // Replay forward
        let n = self.ops.len();
        for i in 0..n {
            let ch = self.children[i];
            let va = || self.values[ch[0] as usize];
            let vb = || self.values[ch[1] as usize];

            self.values[i] = match self.ops[i] {
                Op::Input | Op::ExtInput => continue, // already set
                Op::Const(c) => c,
                Op::Neg => -va(),
                Op::Abs => va().abs(),
                Op::Exp => va().exp(),
                Op::Ln => va().ln(),
                Op::Sqrt => va().sqrt(),
                Op::Recip => 1.0 / va(),
                Op::Sin => va().sin(),
                Op::Cos => va().cos(),
                Op::Tan => va().tan(),
                Op::Asin => va().asin(),
                Op::Acos => va().acos(),
                Op::Atan => va().atan(),
                Op::Sinh => va().sinh(),
                Op::Cosh => va().cosh(),
                Op::Tanh => va().tanh(),
                Op::Log2 => va().log2(),
                Op::Log10 => va().log10(),
                Op::Floor => va().floor(),
                Op::Ceil => va().ceil(),
                Op::Powi(n) => va().powi(n),
                Op::MulConst(c) => va() * c,
                Op::AddConst(c) => va() + c,
                Op::Add => va() + vb(),
                Op::Sub => va() - vb(),
                Op::Mul => va() * vb(),
                Op::Div => va() / vb(),
                Op::Powf => va().powf(vb()),
                Op::Max => va().max(vb()),
                Op::Min => va().min(vb()),
                Op::Atan2 => va().atan2(vb()),
            };
        }
    }

    // =====================================================================
    // Partial derivatives (flat array)
    // =====================================================================

    /// Total number of partial derivative entries for the backward pass
    /// from `output` to the beginning of the tape.
    pub fn total_partials(&self, output: JitReal) -> usize {
        (0..=output.idx as usize)
            .map(|i| self.ops[i].num_children())
            .sum()
    }

    /// Fill a pre-allocated buffer with partial derivatives in the order
    /// expected by the compiled adjoint function.
    ///
    /// Walk nodes in **reverse** order (matching the backward pass), and for
    /// each node append the partial(s) w.r.t. its children.
    pub fn fill_partials_into(&self, output: JitReal, buf: &mut [f64]) {
        let output_idx = output.idx as usize;
        let mut offset = 0usize;

        for i in (0..=output_idx).rev() {
            let ch = self.children[i];
            let vi = self.values[i];
            let va = || self.values[ch[0] as usize];
            let vb = || self.values[ch[1] as usize];

            match self.ops[i] {
                // Leaf: no partials
                Op::Input | Op::ExtInput | Op::Const(_) => {}

                // ---- Unary (1 partial) ----
                Op::Neg => {
                    buf[offset] = -1.0;
                    offset += 1;
                }
                Op::Abs => {
                    buf[offset] = if va() >= 0.0 { 1.0 } else { -1.0 };
                    offset += 1;
                }
                Op::Exp => {
                    buf[offset] = vi; // exp(a) = values[i]
                    offset += 1;
                }
                Op::Ln => {
                    buf[offset] = 1.0 / va();
                    offset += 1;
                }
                Op::Sqrt => {
                    buf[offset] = if vi > 0.0 { 0.5 / vi } else { 0.0 };
                    offset += 1;
                }
                Op::Recip => {
                    buf[offset] = -vi * vi; // -(1/a)^2
                    offset += 1;
                }
                Op::Sin => {
                    buf[offset] = va().cos();
                    offset += 1;
                }
                Op::Cos => {
                    buf[offset] = -va().sin();
                    offset += 1;
                }
                Op::Tan => {
                    let c = va().cos();
                    buf[offset] = 1.0 / (c * c);
                    offset += 1;
                }
                Op::Asin => {
                    let a = va();
                    buf[offset] = 1.0 / (1.0 - a * a).sqrt();
                    offset += 1;
                }
                Op::Acos => {
                    let a = va();
                    buf[offset] = -1.0 / (1.0 - a * a).sqrt();
                    offset += 1;
                }
                Op::Atan => {
                    let a = va();
                    buf[offset] = 1.0 / (1.0 + a * a);
                    offset += 1;
                }
                Op::Sinh => {
                    buf[offset] = va().cosh();
                    offset += 1;
                }
                Op::Cosh => {
                    buf[offset] = va().sinh();
                    offset += 1;
                }
                Op::Tanh => {
                    buf[offset] = 1.0 - vi * vi;
                    offset += 1;
                }
                Op::Log2 => {
                    buf[offset] = 1.0 / (va() * std::f64::consts::LN_2);
                    offset += 1;
                }
                Op::Log10 => {
                    buf[offset] = 1.0 / (va() * std::f64::consts::LN_10);
                    offset += 1;
                }
                Op::Floor | Op::Ceil => {
                    buf[offset] = 0.0; // not differentiable
                    offset += 1;
                }
                Op::Powi(n) => {
                    buf[offset] = n as f64 * va().powi(n - 1);
                    offset += 1;
                }
                Op::MulConst(c) => {
                    buf[offset] = c;
                    offset += 1;
                }
                Op::AddConst(_) => {
                    buf[offset] = 1.0;
                    offset += 1;
                }

                // ---- Binary (2 partials) ----
                Op::Add => {
                    buf[offset] = 1.0;
                    buf[offset + 1] = 1.0;
                    offset += 2;
                }
                Op::Sub => {
                    buf[offset] = 1.0;
                    buf[offset + 1] = -1.0;
                    offset += 2;
                }
                Op::Mul => {
                    buf[offset] = vb();     // ∂(a*b)/∂a = b
                    buf[offset + 1] = va(); // ∂(a*b)/∂b = a
                    offset += 2;
                }
                Op::Div => {
                    let inv_b = 1.0 / vb();
                    buf[offset] = inv_b;                         // ∂(a/b)/∂a
                    buf[offset + 1] = -va() * inv_b * inv_b;     // ∂(a/b)/∂b
                    offset += 2;
                }
                Op::Powf => {
                    let a = va();
                    let b = vb();
                    buf[offset] = b * a.powf(b - 1.0);           // ∂(a^b)/∂a
                    buf[offset + 1] = vi * a.ln();                // ∂(a^b)/∂b
                    offset += 2;
                }
                Op::Max => {
                    let winner_a = va() >= vb();
                    buf[offset] = if winner_a { 1.0 } else { 0.0 };
                    buf[offset + 1] = if winner_a { 0.0 } else { 1.0 };
                    offset += 2;
                }
                Op::Min => {
                    let winner_a = va() <= vb();
                    buf[offset] = if winner_a { 1.0 } else { 0.0 };
                    buf[offset + 1] = if winner_a { 0.0 } else { 1.0 };
                    offset += 2;
                }
                Op::Atan2 => {
                    let a = va();
                    let b = vb();
                    let denom = a * a + b * b;
                    buf[offset] = b / denom;       // ∂atan2/∂y
                    buf[offset + 1] = -a / denom;  // ∂atan2/∂x
                    offset += 2;
                }
            }
        }
        debug_assert_eq!(offset, buf.len());
    }

    /// Convenience: allocate and fill partials.
    pub fn fill_partials(&self, output: JitReal) -> Vec<f64> {
        let n = self.total_partials(output);
        let mut buf = vec![0.0; n];
        self.fill_partials_into(output, &mut buf);
        buf
    }

    // =====================================================================
    // Interpreted adjoint (reference / fallback)
    // =====================================================================

    /// Compute the adjoint using the precomputed partials — interpreted
    /// reference implementation.
    pub fn adjoint(&self, output: JitReal) -> Vec<f64> {
        let partials = self.fill_partials(output);
        let output_idx = output.idx as usize;
        let n = output_idx + 1;
        let mut adj = vec![0.0; n];
        adj[output_idx] = 1.0;

        let mut p_offset = 0usize;
        for i in (0..=output_idx).rev() {
            let a_i = adj[i];
            if a_i == 0.0 {
                p_offset += self.ops[i].num_children();
                continue;
            }
            let nc = self.ops[i].num_children();
            for j in 0..nc {
                let child = self.children[i][j] as usize;
                adj[child] += a_i * partials[p_offset];
                p_offset += 1;
            }
        }
        adj
    }

    // =====================================================================
    // JIT compilation via Cranelift
    // =====================================================================

    /// Compile the adjoint backward pass into a native function.
    ///
    /// The returned [`CompiledAdjoint`] can be called repeatedly with
    /// different partial-derivative arrays (from [`fill_partials_into`]).
    ///
    /// # Panics
    ///
    /// Panics if the tape has more than ~268 million nodes (byte offset would
    /// exceed `i32::MAX`).
    pub fn compile_adjoint(&self, output: JitReal) -> CompiledAdjoint {
        let output_idx = output.idx as usize;
        let num_nodes = output_idx + 1;

        // Ensure byte offsets fit in i32 (Cranelift Offset32)
        assert!(
            num_nodes * 8 <= i32::MAX as usize,
            "tape too large for JIT (>{} nodes)",
            i32::MAX as usize / 8
        );

        // ---- Cranelift setup ----
        let mut flag_builder = settings::builder();
        flag_builder.set("opt_level", "speed").unwrap();
        // On some platforms we need these:
        flag_builder.set("is_pic", "false").unwrap();
        let isa_builder = cranelift_native::builder()
            .expect("failed to create native ISA builder");
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .expect("failed to finish ISA");

        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);

        // Function signature: fn(partials: *const f64, adj: *mut f64)
        let ptr_type = module.target_config().pointer_type();
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(ptr_type)); // partials
        sig.params.push(AbiParam::new(ptr_type)); // adj

        let func_id = module
            .declare_function("jit_adjoint", Linkage::Local, &sig)
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

            // adj[output_idx] = 1.0
            let one = fb.ins().f64const(1.0);
            fb.ins()
                .store(MemFlags::new(), one, adj_ptr, (output_idx * 8) as i32);

            // Emit adjoint accumulations in reverse node order
            let mut p_byte_offset: i32 = 0;

            for i in (0..=output_idx).rev() {
                let nc = self.ops[i].num_children();
                if nc == 0 {
                    continue;
                }

                // Load adj[i]
                let adj_i =
                    fb.ins()
                        .load(types::F64, MemFlags::new(), adj_ptr, (i * 8) as i32);

                for j in 0..nc {
                    let child = self.children[i][j] as usize;
                    let child_byte_off = (child * 8) as i32;

                    // Load partial from flat array
                    let p = fb.ins().load(
                        types::F64,
                        MemFlags::new(),
                        partials_ptr,
                        p_byte_offset,
                    );

                    // product = adj_i * partial
                    let product = fb.ins().fmul(adj_i, p);

                    // adj[child] += product
                    let old =
                        fb.ins()
                            .load(types::F64, MemFlags::new(), adj_ptr, child_byte_off);
                    let new_val = fb.ins().fadd(old, product);
                    fb.ins()
                        .store(MemFlags::new(), new_val, adj_ptr, child_byte_off);

                    p_byte_offset += 8; // next partial (f64 = 8 bytes)
                }
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

        CompiledAdjoint {
            func,
            num_nodes,
            output_idx,
            _module: module,
        }
    }
}

impl Default for JitTape {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// CompiledAdjoint
// ===========================================================================

/// A Cranelift JIT-compiled adjoint function.
///
/// Holds the compiled native code and can be called repeatedly with
/// different partial derivative arrays. The `_module` field keeps the
/// JIT memory alive.
pub struct CompiledAdjoint {
    func: unsafe extern "C" fn(*const f64, *mut f64),
    num_nodes: usize,
    output_idx: usize,
    // Must outlive `func` — JIT code lives in module memory.
    _module: JITModule,
}

impl CompiledAdjoint {
    /// Execute the compiled adjoint. Returns a freshly allocated gradient
    /// vector of length `num_nodes`.
    pub fn execute(&self, partials: &[f64]) -> Vec<f64> {
        let mut adj = vec![0.0; self.num_nodes];
        self.execute_into(partials, &mut adj);
        adj
    }

    /// Execute the compiled adjoint into a pre-allocated buffer.
    /// `adj` must have length ≥ `num_nodes` and is zeroed before execution.
    pub fn execute_into(&self, partials: &[f64], adj: &mut [f64]) {
        assert!(adj.len() >= self.num_nodes);
        // Zero out and let the JIT function set adj[output_idx] = 1.0
        adj[..self.num_nodes].fill(0.0);
        unsafe {
            (self.func)(partials.as_ptr(), adj.as_mut_ptr());
        }
    }

    /// Number of nodes the compiled function expects.
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// The output node index.
    pub fn output_idx(&self) -> usize {
        self.output_idx
    }
}

// Safety: the JIT module and function pointer are thread-safe once finalized.
unsafe impl Send for CompiledAdjoint {}
unsafe impl Sync for CompiledAdjoint {}

// ===========================================================================
// MC European with JIT-compiled adjoint
// ===========================================================================

/// Price a European option via Monte Carlo with JIT-compiled pathwise Greeks.
///
/// Records one path on a [`JitTape`], compiles the adjoint backward pass via
/// Cranelift, then replays the forward pass and executes the compiled adjoint
/// for each MC path. This eliminates all interpretation overhead from the
/// adjoint pass.
///
/// Uses antithetic variates (each `z` → `+z` and `−z`).
#[allow(clippy::too_many_arguments)]
pub fn mc_european_jit(
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
    let mut rng = SmallRng::seed_from_u64(seed);
    let half_paths = num_paths / 2;

    let phi = match option_kind {
        OptionKind::Call => 1.0,
        OptionKind::Put => -1.0,
    };

    // ---- Step 1: Record one path and compile ----
    let z0: f64 = StandardNormal.sample(&mut rng);
    let mut tape = JitTape::new();

    // AD inputs
    let s_in = tape.input(spot);        // input 0
    let r_in = tape.input(r);           // input 1
    let q_in = tape.input(q);           // input 2
    let v_in = tape.input(vol);         // input 3
    // External input (random normal)
    let z_in = tape.ext_input(z0);      // ext 0

    // Build computation graph
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

    // Compile the adjoint
    let compiled = compiled_adjoint_from_tape(&tape, npv_node);

    // Pre-allocate buffers
    let n_partials = tape.total_partials(npv_node);
    let n_nodes = npv_node.idx as usize + 1;
    let mut partials_buf = vec![0.0; n_partials];
    let mut adj_buf = vec![0.0; n_nodes];

    // Accumulators
    let mut sum_npv = 0.0;
    let mut sum_npv_sq = 0.0;
    let mut sum_delta = 0.0;
    let mut sum_vega = 0.0;
    let mut sum_rho = 0.0;
    let mut sum_div_rho = 0.0;

    let inputs = [spot, r, q, vol];

    // ---- Step 2: Process first path pair (z0, -z0) ----
    // z0 forward was already computed during recording
    {
        tape.fill_partials_into(npv_node, &mut partials_buf);
        compiled.execute_into(&partials_buf, &mut adj_buf);
        let pv = tape.values[npv_node.idx as usize];
        sum_npv += pv;
        sum_npv_sq += pv * pv;
        sum_delta += adj_buf[s_in.idx as usize];
        sum_rho += adj_buf[r_in.idx as usize];
        sum_div_rho += adj_buf[q_in.idx as usize];
        sum_vega += adj_buf[v_in.idx as usize];
    }
    // Antithetic: -z0
    {
        tape.forward_eval(&inputs, &[-z0]);
        tape.fill_partials_into(npv_node, &mut partials_buf);
        compiled.execute_into(&partials_buf, &mut adj_buf);
        let pv = tape.values[npv_node.idx as usize];
        sum_npv += pv;
        sum_npv_sq += pv * pv;
        sum_delta += adj_buf[s_in.idx as usize];
        sum_rho += adj_buf[r_in.idx as usize];
        sum_div_rho += adj_buf[q_in.idx as usize];
        sum_vega += adj_buf[v_in.idx as usize];
    }

    // ---- Step 3: Remaining path pairs ----
    for _ in 1..half_paths {
        let z: f64 = StandardNormal.sample(&mut rng);

        for &zz in &[z, -z] {
            tape.forward_eval(&inputs, &[zz]);
            tape.fill_partials_into(npv_node, &mut partials_buf);
            compiled.execute_into(&partials_buf, &mut adj_buf);

            let pv = tape.values[npv_node.idx as usize];
            sum_npv += pv;
            sum_npv_sq += pv * pv;
            sum_delta += adj_buf[s_in.idx as usize];
            sum_rho += adj_buf[r_in.idx as usize];
            sum_div_rho += adj_buf[q_in.idx as usize];
            sum_vega += adj_buf[v_in.idx as usize];
        }
    }

    let n = (half_paths * 2) as f64;
    let mean = sum_npv / n;
    let variance = (sum_npv_sq / n - mean * mean).max(0.0);

    McEuropeanGreeks {
        npv: mean,
        std_error: (variance / n).sqrt(),
        delta: sum_delta / n,
        vega: sum_vega / n,
        rho: sum_rho / n,
        div_rho: sum_div_rho / n,
        num_paths: half_paths * 2,
    }
}

// ===========================================================================
// MC Heston with JIT-compiled adjoint
// ===========================================================================

/// Price a European option under Heston stochastic volatility via Monte Carlo
/// with JIT-compiled pathwise Greeks.
///
/// Same discretisation as [`mc_heston_aad`](crate::mc::mc_heston_aad):
/// log-Euler for spot, Euler with full truncation for variance.
///
/// Records one path's full simulation on a [`JitTape`], compiles the adjoint,
/// then replays each path with `forward_eval` + compiled backward pass.
///
/// No antithetic variates for Heston (matches `mc_heston_aad` behaviour).
#[allow(clippy::too_many_arguments)]
pub fn mc_heston_jit(
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
    let mut rng = SmallRng::seed_from_u64(seed);
    let dt = time_to_expiry / num_steps as f64;
    let sqrt_dt = dt.sqrt();

    let phi = match option_kind {
        OptionKind::Call => 1.0,
        OptionKind::Put => -1.0,
    };

    // ---- Step 1: Record one path and compile ----
    // Pre-generate random numbers for the first path
    let mut z1_first = Vec::with_capacity(num_steps);
    let mut z2_first = Vec::with_capacity(num_steps);
    for _ in 0..num_steps {
        z1_first.push(StandardNormal.sample(&mut rng));
        z2_first.push(StandardNormal.sample(&mut rng));
    }

    let mut tape = JitTape::new();

    // AD inputs (8 inputs with gradients)
    let s_in = tape.input(spot);         // input 0
    let r_in = tape.input(r);            // input 1
    let q_in = tape.input(q);            // input 2
    let v0_in = tape.input(v0);          // input 3
    let kappa_in = tape.input(kappa);    // input 4
    let theta_in = tape.input(theta);    // input 5
    let sigma_in = tape.input(sigma);    // input 6
    let rho_in = tape.input(rho_corr);   // input 7

    // External inputs: 2 * num_steps normals
    let mut z1_nodes = Vec::with_capacity(num_steps);
    let mut z2_nodes = Vec::with_capacity(num_steps);
    for step in 0..num_steps {
        z1_nodes.push(tape.ext_input(z1_first[step]));
        z2_nodes.push(tape.ext_input(z2_first[step]));
    }

    // Constants
    let dt_c = tape.constant(dt);
    let sqrt_dt_c = tape.constant(sqrt_dt);
    let half_c = tape.constant(0.5);
    let one_c = tape.constant(1.0);
    let zero_c = tape.constant(0.0);
    let phi_c = tape.constant(phi);
    let strike_c = tape.constant(strike);

    // Correlated noise factor: sqrt(1 - rho²)
    let rho_sq = tape.mul(rho_in, rho_in);
    let one_minus_rho_sq = tape.sub(one_c, rho_sq);
    let rho_comp = tape.sqrt(one_minus_rho_sq);

    let mut log_s = tape.ln(s_in);
    let mut v = v0_in;

    for step in 0..num_steps {
        let z1_c = z1_nodes[step];
        let z2_indep_c = z2_nodes[step];

        // v_pos = max(v, 0)
        let v_pos = tape.max(v, zero_c);
        let sqrt_v = tape.sqrt(v_pos);

        // Correlated z2 = rho * z1 + sqrt(1-rho²) * z2_indep
        let rho_z1 = tape.mul(rho_in, z1_c);
        let comp_z2 = tape.mul(rho_comp, z2_indep_c);
        let z2_c = tape.add(rho_z1, comp_z2);

        // Log-Euler for spot:
        // log_s += (r - q - 0.5*v_pos)*dt + sqrt_v*sqrt_dt*z1
        let half_v = tape.mul(half_c, v_pos);
        let rq = tape.sub(r_in, q_in);
        let drift_rate = tape.sub(rq, half_v);
        let drift_term = tape.mul(drift_rate, dt_c);
        let vol_term_1 = tape.mul(sqrt_v, sqrt_dt_c);
        let vol_term = tape.mul(vol_term_1, z1_c);
        let step_total = tape.add(drift_term, vol_term);
        log_s = tape.add(log_s, step_total);

        // Euler for variance:
        // v = v + kappa*(theta - v_pos)*dt + sigma*sqrt_v*sqrt_dt*z2
        let theta_diff = tape.sub(theta_in, v_pos);
        let kappa_theta = tape.mul(kappa_in, theta_diff);
        let var_drift = tape.mul(kappa_theta, dt_c);
        let sig_sqrt_v = tape.mul(sigma_in, sqrt_v);
        let var_vol_1 = tape.mul(sig_sqrt_v, sqrt_dt_c);
        let var_vol = tape.mul(var_vol_1, z2_c);
        let v_next_1 = tape.add(v, var_drift);
        let v_next = tape.add(v_next_1, var_vol);
        // Full truncation
        v = tape.max(v_next, zero_c);
    }

    // S_T = exp(log_s)
    let st = tape.exp(log_s);

    // Payoff
    let diff = tape.sub(st, strike_c);
    let intrinsic = tape.mul(phi_c, diff);
    let payoff = tape.max(intrinsic, zero_c);

    // Discount
    let neg_r = tape.neg(r_in);
    let tau_disc = tape.constant(time_to_expiry);
    let neg_r_tau = tape.mul(neg_r, tau_disc);
    let disc = tape.exp(neg_r_tau);
    let npv_node = tape.mul(payoff, disc);

    // Compile
    let compiled = compiled_adjoint_from_tape(&tape, npv_node);

    // Pre-allocate
    let n_partials = tape.total_partials(npv_node);
    let n_nodes = npv_node.idx as usize + 1;
    let mut partials_buf = vec![0.0; n_partials];
    let mut adj_buf = vec![0.0; n_nodes];

    // Accumulators: [delta, rho, div_rho, vega_v0, d_kappa, d_theta, d_sigma, d_rho]
    let mut sum_greeks = [0.0_f64; 8];
    let mut sum_npv = 0.0;
    let mut sum_npv_sq = 0.0;

    let ad_inputs = [spot, r, q, v0, kappa, theta, sigma, rho_corr];
    let input_ids = [
        s_in.idx as usize,
        r_in.idx as usize,
        q_in.idx as usize,
        v0_in.idx as usize,
        kappa_in.idx as usize,
        theta_in.idx as usize,
        sigma_in.idx as usize,
        rho_in.idx as usize,
    ];

    // ---- Step 2: Process first path (already recorded) ----
    {
        tape.fill_partials_into(npv_node, &mut partials_buf);
        compiled.execute_into(&partials_buf, &mut adj_buf);
        let pv = tape.values[npv_node.idx as usize];
        sum_npv += pv;
        sum_npv_sq += pv * pv;
        for (g, &id) in sum_greeks.iter_mut().zip(input_ids.iter()) {
            *g += adj_buf[id];
        }
    }

    // ---- Step 3: Remaining paths ----
    let mut ext_inputs = vec![0.0f64; 2 * num_steps];
    for _ in 1..num_paths {
        for step in 0..num_steps {
            ext_inputs[2 * step] = StandardNormal.sample(&mut rng);
            ext_inputs[2 * step + 1] = StandardNormal.sample(&mut rng);
        }
        tape.forward_eval(&ad_inputs, &ext_inputs);
        tape.fill_partials_into(npv_node, &mut partials_buf);
        compiled.execute_into(&partials_buf, &mut adj_buf);

        let pv = tape.values[npv_node.idx as usize];
        sum_npv += pv;
        sum_npv_sq += pv * pv;
        for (g, &id) in sum_greeks.iter_mut().zip(input_ids.iter()) {
            *g += adj_buf[id];
        }
    }

    let n = num_paths as f64;
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
        num_paths,
    }
}

// ===========================================================================
// Helper: compile adjoint (separated for reuse)
// ===========================================================================

/// Compile the adjoint backward pass from a recorded `JitTape`.
fn compiled_adjoint_from_tape(tape: &JitTape, output: JitReal) -> CompiledAdjoint {
    tape.compile_adjoint(output)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ---- JitTape recording + interpreted adjoint ----

    #[test]
    fn jit_tape_add() {
        let mut tape = JitTape::new();
        let x = tape.input(3.0);
        let y = tape.input(5.0);
        let z = tape.add(x, y);
        assert_abs_diff_eq!(tape.values[z.idx as usize], 8.0, epsilon = 1e-15);
        let adj = tape.adjoint(z);
        assert_abs_diff_eq!(adj[x.idx as usize], 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(adj[y.idx as usize], 1.0, epsilon = 1e-15);
    }

    #[test]
    fn jit_tape_mul() {
        let mut tape = JitTape::new();
        let x = tape.input(3.0);
        let y = tape.input(5.0);
        let z = tape.mul(x, y);
        assert_abs_diff_eq!(tape.values[z.idx as usize], 15.0, epsilon = 1e-15);
        let adj = tape.adjoint(z);
        assert_abs_diff_eq!(adj[x.idx as usize], 5.0, epsilon = 1e-15);
        assert_abs_diff_eq!(adj[y.idx as usize], 3.0, epsilon = 1e-15);
    }

    #[test]
    fn jit_tape_chain() {
        // f(x) = exp(x²), f'(x) = 2x·exp(x²)
        let mut tape = JitTape::new();
        let x = tape.input(1.5);
        let x2 = tape.mul(x, x);
        let e = tape.exp(x2);
        let adj = tape.adjoint(e);
        let expected = 2.0 * 1.5 * (1.5 * 1.5_f64).exp();
        assert_abs_diff_eq!(adj[x.idx as usize], expected, epsilon = 1e-10);
    }

    #[test]
    fn jit_tape_div() {
        let mut tape = JitTape::new();
        let x = tape.input(6.0);
        let y = tape.input(3.0);
        let z = tape.div(x, y);
        assert_abs_diff_eq!(tape.values[z.idx as usize], 2.0, epsilon = 1e-15);
        let adj = tape.adjoint(z);
        assert_abs_diff_eq!(adj[x.idx as usize], 1.0 / 3.0, epsilon = 1e-14);
        assert_abs_diff_eq!(adj[y.idx as usize], -2.0 / 3.0, epsilon = 1e-14);
    }

    #[test]
    fn jit_tape_max_subgradient() {
        // max(x, 0) where x > 0 → gradient flows to x
        let mut tape = JitTape::new();
        let x = tape.input(3.0);
        let zero = tape.constant(0.0);
        let m = tape.max(x, zero);
        let adj = tape.adjoint(m);
        assert_abs_diff_eq!(adj[x.idx as usize], 1.0, epsilon = 1e-15);

        // max(x, 0) where x < 0 → gradient does not flow to x
        let mut tape2 = JitTape::new();
        let x2 = tape2.input(-2.0);
        let zero2 = tape2.constant(0.0);
        let m2 = tape2.max(x2, zero2);
        let adj2 = tape2.adjoint(m2);
        assert_abs_diff_eq!(adj2[x2.idx as usize], 0.0, epsilon = 1e-15);
    }

    #[test]
    fn jit_tape_sqrt_safe() {
        // sqrt(0) should produce partial = 0 (safe subgradient)
        let mut tape = JitTape::new();
        let x = tape.input(0.0);
        let s = tape.sqrt(x);
        let adj = tape.adjoint(s);
        assert!(adj[x.idx as usize].is_finite());
        assert_abs_diff_eq!(adj[x.idx as usize], 0.0, epsilon = 1e-15);
    }

    #[test]
    fn jit_tape_forward_eval() {
        let mut tape = JitTape::new();
        let x = tape.input(2.0);
        let y = tape.input(3.0);
        let z = tape.mul(x, y);
        assert_abs_diff_eq!(tape.values[z.idx as usize], 6.0, epsilon = 1e-15);

        // Re-evaluate with new inputs
        tape.forward_eval(&[4.0, 5.0], &[]);
        assert_abs_diff_eq!(tape.values[z.idx as usize], 20.0, epsilon = 1e-15);

        // Adjoint should reflect new values
        let adj = tape.adjoint(z);
        assert_abs_diff_eq!(adj[x.idx as usize], 5.0, epsilon = 1e-15);
        assert_abs_diff_eq!(adj[y.idx as usize], 4.0, epsilon = 1e-15);
    }

    #[test]
    fn jit_tape_ext_input() {
        let mut tape = JitTape::new();
        let x = tape.input(2.0);
        let z = tape.ext_input(10.0);
        let out = tape.mul(x, z);
        let adj = tape.adjoint(out);
        // ∂(x*z)/∂x = z = 10
        assert_abs_diff_eq!(adj[x.idx as usize], 10.0, epsilon = 1e-14);
        // z is ExtInput → its gradient exists in adj but is not an AD input
        assert_abs_diff_eq!(adj[z.idx as usize], 2.0, epsilon = 1e-14);

        // Re-eval with different external
        tape.forward_eval(&[2.0], &[20.0]);
        let adj2 = tape.adjoint(out);
        assert_abs_diff_eq!(adj2[x.idx as usize], 20.0, epsilon = 1e-14);
    }

    // ---- JIT compiled vs interpreted adjoint ----

    #[test]
    fn compiled_matches_interpreted_add() {
        let mut tape = JitTape::new();
        let x = tape.input(3.0);
        let y = tape.input(5.0);
        let z = tape.add(x, y);

        let adj_interp = tape.adjoint(z);
        let compiled = tape.compile_adjoint(z);
        let partials = tape.fill_partials(z);
        let adj_jit = compiled.execute(&partials);

        assert_abs_diff_eq!(adj_interp[x.idx as usize], adj_jit[x.idx as usize], epsilon = 1e-15);
        assert_abs_diff_eq!(adj_interp[y.idx as usize], adj_jit[y.idx as usize], epsilon = 1e-15);
    }

    #[test]
    fn compiled_matches_interpreted_mul() {
        let mut tape = JitTape::new();
        let x = tape.input(3.0);
        let y = tape.input(5.0);
        let z = tape.mul(x, y);

        let adj_interp = tape.adjoint(z);
        let compiled = tape.compile_adjoint(z);
        let partials = tape.fill_partials(z);
        let adj_jit = compiled.execute(&partials);

        assert_abs_diff_eq!(adj_interp[x.idx as usize], adj_jit[x.idx as usize], epsilon = 1e-15);
        assert_abs_diff_eq!(adj_interp[y.idx as usize], adj_jit[y.idx as usize], epsilon = 1e-15);
    }

    #[test]
    fn compiled_matches_interpreted_chain() {
        // f(x) = exp(x²)
        let mut tape = JitTape::new();
        let x = tape.input(1.5);
        let x2 = tape.mul(x, x);
        let e = tape.exp(x2);

        let adj_interp = tape.adjoint(e);
        let compiled = tape.compile_adjoint(e);
        let partials = tape.fill_partials(e);
        let adj_jit = compiled.execute(&partials);

        assert_abs_diff_eq!(adj_interp[x.idx as usize], adj_jit[x.idx as usize], epsilon = 1e-12);
    }

    #[test]
    fn compiled_matches_interpreted_complex() {
        // f(x,y) = (x*y + exp(x)) / y
        let mut tape = JitTape::new();
        let x = tape.input(2.0);
        let y = tape.input(3.0);
        let xy = tape.mul(x, y);
        let ex = tape.exp(x);
        let sum = tape.add(xy, ex);
        let out = tape.div(sum, y);

        let adj_interp = tape.adjoint(out);
        let compiled = tape.compile_adjoint(out);
        let partials = tape.fill_partials(out);
        let adj_jit = compiled.execute(&partials);

        for i in 0..tape.len() {
            assert_abs_diff_eq!(adj_interp[i], adj_jit[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn compiled_matches_interpreted_max() {
        // f(x) = max(x - 100, 0) where x = 105
        let mut tape = JitTape::new();
        let x = tape.input(105.0);
        let k = tape.constant(100.0);
        let diff = tape.sub(x, k);
        let zero = tape.constant(0.0);
        let out = tape.max(diff, zero);

        let adj_interp = tape.adjoint(out);
        let compiled = tape.compile_adjoint(out);
        let partials = tape.fill_partials(out);
        let adj_jit = compiled.execute(&partials);

        assert_abs_diff_eq!(adj_interp[x.idx as usize], adj_jit[x.idx as usize], epsilon = 1e-15);
        assert_abs_diff_eq!(adj_jit[x.idx as usize], 1.0, epsilon = 1e-15);

        // OTM: x = 95
        tape.forward_eval(&[95.0], &[]);
        let adj_interp2 = tape.adjoint(out);
        let partials2 = tape.fill_partials(out);
        let adj_jit2 = compiled.execute(&partials2);

        assert_abs_diff_eq!(adj_interp2[x.idx as usize], adj_jit2[x.idx as usize], epsilon = 1e-15);
        assert_abs_diff_eq!(adj_jit2[x.idx as usize], 0.0, epsilon = 1e-15);
    }

    #[test]
    fn compiled_replayed_with_new_values() {
        // Compile once, evaluate with different inputs
        let mut tape = JitTape::new();
        let x = tape.input(3.0);
        let y = tape.input(5.0);
        let z = tape.mul(x, y);

        let compiled = tape.compile_adjoint(z);
        let mut partials_buf = vec![0.0; tape.total_partials(z)];
        let mut adj_buf = vec![0.0; z.idx as usize + 1];

        // First evaluation (original values)
        tape.fill_partials_into(z, &mut partials_buf);
        compiled.execute_into(&partials_buf, &mut adj_buf);
        assert_abs_diff_eq!(adj_buf[x.idx as usize], 5.0, epsilon = 1e-15);
        assert_abs_diff_eq!(adj_buf[y.idx as usize], 3.0, epsilon = 1e-15);

        // Second evaluation (new values)
        tape.forward_eval(&[10.0, 20.0], &[]);
        tape.fill_partials_into(z, &mut partials_buf);
        compiled.execute_into(&partials_buf, &mut adj_buf);
        assert_abs_diff_eq!(adj_buf[x.idx as usize], 20.0, epsilon = 1e-15);
        assert_abs_diff_eq!(adj_buf[y.idx as usize], 10.0, epsilon = 1e-15);
    }

    #[test]
    fn compiled_bs_call_delta() {
        // Build a simple BS call on the JitTape, verify delta ≈ BS delta
        let spot = 100.0;
        let strike = 100.0;
        let r = 0.05;
        let q = 0.0;
        let vol = 0.20;
        let tau = 1.0;

        let mut tape = JitTape::new();
        let s = tape.input(spot);
        let v = tape.input(vol);

        let tau_c = tape.constant(tau);
        let sqrt_t = tape.constant(tau.sqrt());
        let half = tape.constant(0.5);
        let strike_c = tape.constant(strike);
        let zero_c = tape.constant(0.0);
        let r_c = tape.constant(r);
        let q_c = tape.constant(q);

        // S_T = s * exp((r - q - 0.5*v²)*tau + v*sqrt(tau)*0)
        // For deterministic test, set z=0 (then S_T = s * exp(drift))
        let z_c = tape.constant(0.5); // a fixed "z" value

        let vsq = tape.mul(v, v);
        let hvsq = tape.mul(half, vsq);
        let drift_rate_inner = tape.sub(r_c, q_c);
        let drift_rate = tape.sub(drift_rate_inner, hvsq);
        let drift = tape.mul(drift_rate, tau_c);
        let v_sqrt_t = tape.mul(v, sqrt_t);
        let diffusion = tape.mul(v_sqrt_t, z_c);
        let exponent = tape.add(drift, diffusion);
        let growth = tape.exp(exponent);
        let st = tape.mul(s, growth);
        let diff = tape.sub(st, strike_c);
        let payoff = tape.max(diff, zero_c);

        let neg_r = tape.neg(r_c);
        let neg_r_tau = tape.mul(neg_r, tau_c);
        let disc = tape.exp(neg_r_tau);
        let npv = tape.mul(payoff, disc);

        // Interpreted
        let adj_interp = tape.adjoint(npv);

        // JIT
        let compiled = tape.compile_adjoint(npv);
        let partials = tape.fill_partials(npv);
        let adj_jit = compiled.execute(&partials);

        // Delta should match between interpreted and JIT
        assert_abs_diff_eq!(
            adj_interp[s.idx as usize],
            adj_jit[s.idx as usize],
            epsilon = 1e-12
        );

        // Delta should be positive and finite
        assert!(adj_jit[s.idx as usize] > 0.0);
        assert!(adj_jit[s.idx as usize].is_finite());

        // Finite-difference validation
        let bump = 1e-6;
        tape.forward_eval(&[spot + bump, vol], &[]);
        let npv_up = tape.values[npv.idx as usize];
        tape.forward_eval(&[spot - bump, vol], &[]);
        let npv_dn = tape.values[npv.idx as usize];
        let fd_delta = (npv_up - npv_dn) / (2.0 * bump);
        assert_abs_diff_eq!(adj_jit[s.idx as usize], fd_delta, epsilon = 1e-6);
    }

    // ---- MC European JIT ----

    #[test]
    fn jit_european_call_npv() {
        let g = mc_european_jit(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            OptionKind::Call, 100_000, 42,
        );
        // BS call ≈ 10.45
        assert!((g.npv - 10.45).abs() < 0.5, "npv={}", g.npv);
    }

    #[test]
    fn jit_european_delta() {
        let g = mc_european_jit(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            OptionKind::Call, 200_000, 42,
        );
        // BS delta ≈ 0.637
        assert!((g.delta - 0.637).abs() < 0.05, "delta={}", g.delta);
    }

    #[test]
    fn jit_european_vega_positive() {
        let g = mc_european_jit(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            OptionKind::Call, 100_000, 42,
        );
        assert!(g.vega > 0.0, "vega={}", g.vega);
        assert!((g.vega - 37.5).abs() < 5.0, "vega={}", g.vega);
    }

    #[test]
    fn jit_european_put_delta_negative() {
        let g = mc_european_jit(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            OptionKind::Put, 100_000, 42,
        );
        assert!(g.delta < 0.0, "put delta={}", g.delta);
    }

    #[test]
    fn jit_european_matches_aad() {
        // Same seed, same RNG, same antithetic pairing → should match closely
        let aad = crate::mc::mc_european_aad(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            OptionKind::Call, 50_000, 42,
        );
        let jit = mc_european_jit(
            100.0, 100.0, 0.05, 0.0, 0.20, 1.0,
            OptionKind::Call, 50_000, 42,
        );
        // NPVs should be very close (same paths, same arithmetic)
        assert_abs_diff_eq!(aad.npv, jit.npv, epsilon = 0.2);
        assert_abs_diff_eq!(aad.delta, jit.delta, epsilon = 0.02);
        assert_abs_diff_eq!(aad.vega, jit.vega, epsilon = 2.0);
    }

    // ---- MC Heston JIT ----

    #[test]
    fn jit_heston_call_npv() {
        let g = mc_heston_jit(
            100.0, 100.0, 0.05, 0.0,
            0.04, 2.0, 0.04, 0.3, -0.7,
            1.0, OptionKind::Call, 20_000, 50, 42,
        );
        assert!(g.npv > 0.0, "npv={}", g.npv);
        assert!((g.npv - 10.0).abs() < 5.0, "npv={}", g.npv);
    }

    #[test]
    fn jit_heston_delta_positive() {
        let g = mc_heston_jit(
            100.0, 100.0, 0.05, 0.0,
            0.04, 2.0, 0.04, 0.3, -0.7,
            1.0, OptionKind::Call, 20_000, 50, 42,
        );
        assert!(g.delta > 0.0 && g.delta < 1.0, "delta={}", g.delta);
    }

    #[test]
    fn jit_heston_greeks_finite() {
        let g = mc_heston_jit(
            100.0, 100.0, 0.05, 0.0,
            0.04, 2.0, 0.04, 0.3, -0.7,
            1.0, OptionKind::Call, 10_000, 50, 42,
        );
        assert!(g.delta.is_finite(), "delta={}", g.delta);
        assert!(g.vega_v0.is_finite(), "vega_v0={}", g.vega_v0);
        assert!(g.d_kappa.is_finite(), "d_kappa={}", g.d_kappa);
        assert!(g.d_theta.is_finite(), "d_theta={}", g.d_theta);
        assert!(g.d_sigma.is_finite(), "d_sigma={}", g.d_sigma);
        assert!(g.d_rho.is_finite(), "d_rho={}", g.d_rho);
        assert!(g.rho.is_finite(), "rho={}", g.rho);
        assert!(g.div_rho.is_finite(), "div_rho={}", g.div_rho);
    }

    #[test]
    fn jit_heston_vega_v0_positive() {
        let g = mc_heston_jit(
            100.0, 100.0, 0.05, 0.0,
            0.04, 2.0, 0.04, 0.3, -0.7,
            1.0, OptionKind::Call, 20_000, 50, 42,
        );
        assert!(g.vega_v0 > 0.0, "vega_v0={}", g.vega_v0);
    }

    #[test]
    fn jit_heston_matches_aad() {
        // Both should produce similar NPV (same seed)
        let aad = crate::mc::mc_heston_aad(
            100.0, 100.0, 0.05, 0.0,
            0.04, 2.0, 0.04, 0.3, -0.7,
            1.0, OptionKind::Call, 10_000, 50, 42,
        );
        let jit = mc_heston_jit(
            100.0, 100.0, 0.05, 0.0,
            0.04, 2.0, 0.04, 0.3, -0.7,
            1.0, OptionKind::Call, 10_000, 50, 42,
        );
        // NPVs should be similar (not identical due to different tape structures,
        // but statistically in the same ballpark given same seed+RNG)
        assert!(
            (aad.npv - jit.npv).abs() < 3.0 * aad.std_error.max(jit.std_error) + 1.0,
            "aad.npv={}, jit.npv={}", aad.npv, jit.npv
        );
    }

    #[test]
    fn jit_heston_near_analytic() {
        use crate::heston::heston_price_generic;
        let analytic: f64 = heston_price_generic(
            100.0, 100.0, 0.05, 0.0,
            1.0, 0.04, 2.0, 0.04, 0.3, -0.7,
            true,
        );
        let mc = mc_heston_jit(
            100.0, 100.0, 0.05, 0.0,
            0.04, 2.0, 0.04, 0.3, -0.7,
            1.0, OptionKind::Call, 50_000, 100, 42,
        );
        assert!(
            (mc.npv - analytic).abs() < 3.0 * mc.std_error + 0.5,
            "MC npv={} vs analytic={}", mc.npv, analytic
        );
    }

    // ---- Ops coverage ----

    #[test]
    fn jit_compiled_neg_sub() {
        let mut tape = JitTape::new();
        let x = tape.input(5.0);
        let y = tape.input(3.0);
        let neg_x = tape.neg(x);
        let z = tape.sub(neg_x, y); // z = -x - y = -8

        let adj_interp = tape.adjoint(z);
        let compiled = tape.compile_adjoint(z);
        let partials = tape.fill_partials(z);
        let adj_jit = compiled.execute(&partials);

        assert_abs_diff_eq!(adj_interp[x.idx as usize], adj_jit[x.idx as usize], epsilon = 1e-15);
        assert_abs_diff_eq!(adj_jit[x.idx as usize], -1.0, epsilon = 1e-15); // ∂z/∂x = -1
        assert_abs_diff_eq!(adj_jit[y.idx as usize], -1.0, epsilon = 1e-15); // ∂z/∂y = -1
    }

    #[test]
    fn jit_compiled_ln_sqrt() {
        // f(x) = ln(sqrt(x)) = 0.5*ln(x), f'(x) = 0.5/x
        let mut tape = JitTape::new();
        let x = tape.input(4.0);
        let s = tape.sqrt(x);
        let out = tape.ln(s);

        let adj_interp = tape.adjoint(out);
        let compiled = tape.compile_adjoint(out);
        let partials = tape.fill_partials(out);
        let adj_jit = compiled.execute(&partials);

        assert_abs_diff_eq!(adj_interp[x.idx as usize], adj_jit[x.idx as usize], epsilon = 1e-14);
        assert_abs_diff_eq!(adj_jit[x.idx as usize], 0.5 / 4.0, epsilon = 1e-14);
    }

    #[test]
    fn jit_compiled_mul_const_add_const() {
        // f(x) = 3*x + 7, f'(x) = 3
        let mut tape = JitTape::new();
        let x = tape.input(5.0);
        let y = tape.mul_const(x, 3.0);
        let out = tape.add_const(y, 7.0);

        let adj_interp = tape.adjoint(out);
        let compiled = tape.compile_adjoint(out);
        let partials = tape.fill_partials(out);
        let adj_jit = compiled.execute(&partials);

        assert_abs_diff_eq!(adj_jit[x.idx as usize], 3.0, epsilon = 1e-15);
        assert_abs_diff_eq!(adj_interp[x.idx as usize], adj_jit[x.idx as usize], epsilon = 1e-15);
    }

    #[test]
    fn jit_compiled_powi() {
        // f(x) = x³, f'(x) = 3x²
        let mut tape = JitTape::new();
        let x = tape.input(2.0);
        let out = tape.powi(x, 3);

        let adj_interp = tape.adjoint(out);
        let compiled = tape.compile_adjoint(out);
        let partials = tape.fill_partials(out);
        let adj_jit = compiled.execute(&partials);

        assert_abs_diff_eq!(adj_jit[x.idx as usize], 12.0, epsilon = 1e-12);
        assert_abs_diff_eq!(adj_interp[x.idx as usize], adj_jit[x.idx as usize], epsilon = 1e-12);
    }

    #[test]
    fn jit_compiled_min() {
        let mut tape = JitTape::new();
        let x = tape.input(2.0);
        let y = tape.input(5.0);
        let out = tape.min(x, y); // min(2,5) = 2 → gradient to x

        let adj_interp = tape.adjoint(out);
        let compiled = tape.compile_adjoint(out);
        let partials = tape.fill_partials(out);
        let adj_jit = compiled.execute(&partials);

        assert_abs_diff_eq!(adj_jit[x.idx as usize], 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(adj_jit[y.idx as usize], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(adj_interp[x.idx as usize], adj_jit[x.idx as usize], epsilon = 1e-15);
    }

    #[test]
    fn jit_compiled_recip() {
        // f(x) = 1/x, f'(x) = -1/x²
        let mut tape = JitTape::new();
        let x = tape.input(2.0);
        let out = tape.recip(x);

        let compiled = tape.compile_adjoint(out);
        let partials = tape.fill_partials(out);
        let adj_jit = compiled.execute(&partials);

        assert_abs_diff_eq!(adj_jit[x.idx as usize], -0.25, epsilon = 1e-14);
    }

    #[test]
    fn jit_compiled_abs() {
        // |x| at x=3 → ∂/∂x = 1
        let mut tape = JitTape::new();
        let x = tape.input(3.0);
        let out = tape.abs(x);
        let compiled = tape.compile_adjoint(out);
        let partials = tape.fill_partials(out);
        let adj = compiled.execute(&partials);
        assert_abs_diff_eq!(adj[x.idx as usize], 1.0, epsilon = 1e-15);

        // |x| at x=-3 → ∂/∂x = -1
        tape.forward_eval(&[-3.0], &[]);
        let partials2 = tape.fill_partials(out);
        let adj2 = compiled.execute(&partials2);
        assert_abs_diff_eq!(adj2[x.idx as usize], -1.0, epsilon = 1e-15);
    }

    #[test]
    fn jit_compiled_sin_cos() {
        // f(x) = sin(x), f'(x) = cos(x)
        let mut tape = JitTape::new();
        let x = tape.input(1.0);
        let out = tape.sin(x);
        let compiled = tape.compile_adjoint(out);
        let partials = tape.fill_partials(out);
        let adj = compiled.execute(&partials);
        assert_abs_diff_eq!(adj[x.idx as usize], 1.0_f64.cos(), epsilon = 1e-14);

        // f(x) = cos(x), f'(x) = -sin(x)
        let mut tape2 = JitTape::new();
        let x2 = tape2.input(1.0);
        let out2 = tape2.cos(x2);
        let compiled2 = tape2.compile_adjoint(out2);
        let partials2 = tape2.fill_partials(out2);
        let adj2 = compiled2.execute(&partials2);
        assert_abs_diff_eq!(adj2[x2.idx as usize], -1.0_f64.sin(), epsilon = 1e-14);
    }

    #[test]
    fn jit_compiled_tanh() {
        // f(x) = tanh(x), f'(x) = 1 - tanh²(x)
        let mut tape = JitTape::new();
        let x = tape.input(0.5);
        let out = tape.tanh(x);
        let compiled = tape.compile_adjoint(out);
        let partials = tape.fill_partials(out);
        let adj = compiled.execute(&partials);
        let t = 0.5_f64.tanh();
        assert_abs_diff_eq!(adj[x.idx as usize], 1.0 - t * t, epsilon = 1e-14);
    }

    #[test]
    fn jit_compiled_many_inputs() {
        // f(x₁..x₁₀) = Σ xᵢ², ∂f/∂xᵢ = 2xᵢ
        let mut tape = JitTape::new();
        let inputs: Vec<JitReal> = (1..=10).map(|i| tape.input(i as f64)).collect();

        let mut sum = tape.mul(inputs[0], inputs[0]);
        for &xi in &inputs[1..] {
            let xi2 = tape.mul(xi, xi);
            sum = tape.add(sum, xi2);
        }

        let adj_interp = tape.adjoint(sum);
        let compiled = tape.compile_adjoint(sum);
        let partials = tape.fill_partials(sum);
        let adj_jit = compiled.execute(&partials);

        for (i, inp) in inputs.iter().enumerate() {
            let expected = 2.0 * (i + 1) as f64;
            assert_abs_diff_eq!(adj_jit[inp.idx as usize], expected, epsilon = 1e-12);
            assert_abs_diff_eq!(adj_interp[inp.idx as usize], adj_jit[inp.idx as usize], epsilon = 1e-12);
        }
    }
}
