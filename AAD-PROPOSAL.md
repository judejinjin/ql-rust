# Adjoint Algorithmic Differentiation (AAD) for ql-rust

**Date:** 2026-02-28  
**Author:** auto-generated proposal  
**Status:** Draft

---

## 1. Motivation

Computing risk sensitivities (Greeks) is the dominant cost in derivatives pricing.
Today ql-rust supports three approaches, each with significant limitations:

| Method | Where used | Limitation |
|--------|-----------|------------|
| **Analytic closed-form** | Black-Scholes only | Restricted to models with known partial derivatives |
| **Bump-and-reprice** | `equity_risk_ladder`, `curve_sensitivities`, `key_rate_durations` | Cost scales as **O(N)** — one full reprice per risk factor |
| **Grid extraction** | FD and lattice solvers | Only delta/gamma/theta; model-specific |

For a typical interest-rate exotic with 50+ curve pillar inputs, bump-and-reprice
requires **100+ reprices** (central differences) to fill a risk report. Monte Carlo
Heston Greeks are not computed at all — users must bump externally.

**AAD computes _all_ first-order sensitivities in a single backward pass at a cost
of 2–5× one forward price evaluation**, regardless of the number of risk factors.
This is the single highest-impact performance improvement available to a quant library.

### Industry context

Every major sell-side quant library has adopted AAD:

- **QuantLib** does _not_ have native AAD (a known gap acknowledged by its maintainers)
- **ORE/Quaternion** bolt AAD onto QuantLib via `CppAD` / `ADOL-C` / XAD
- **Murex, Calypso, Bloomberg DLIB** all run adjoint-mode AD in production
- **Antoine Savine's "Modern Computational Finance"** (Wiley, 2018) is the canonical reference

Implementing AAD natively would make ql-rust **the first open-source Rust quant library
with built-in adjoint Greeks**, a meaningful differentiator.

---

## 2. Background: Forward vs Reverse Mode

### 2.1 Forward mode (dual numbers)

Augment each scalar with a derivative seed:

$$\tilde{x} = (x,\; \dot{x}) \qquad \text{where } \dot{x} = \frac{\partial x}{\partial x_i}$$

Arithmetic propagates derivatives forward through the computation:

$$\tilde{a} \cdot \tilde{b} = (ab,\; a\dot{b} + \dot{a}b)$$

**Cost:** One forward sweep computes derivatives w.r.t. **one** input. For $N$ inputs,
need $N$ sweeps → **O(N)** — same scaling as bump-and-reprice (but with smaller constants
and no finite-difference truncation error).

**Advantage:** No tape, trivial to implement, exact derivatives, works with control flow.

**Best for:** Small parameter count ($N \le 5$), e.g. BS Greeks w.r.t. $(S, \sigma, r, q, T)$.

### 2.2 Reverse mode (adjoint / backpropagation)

Record the forward computation on a **tape** (Wengert list), then propagate adjoints
backward:

$$\bar{x}_i = \frac{\partial y}{\partial x_i}$$

**Cost:** One forward sweep + one backward sweep → **O(1)** w.r.t. number of inputs.
Computes _all_ $\partial y / \partial x_i$ simultaneously.

**Memory:** Must store the full tape (intermediate values). Tape size is proportional to
the number of operations in the forward pass.

**Best for:** Many inputs, one output — the classic derivatives pricing scenario
(one price, many risk factors).

### 2.3 Recommendation

**Implement both**, with reverse-mode as the primary target:

| Mode | Use case in ql-rust |
|------|---------------------|
| **Forward (dual numbers)** | Quick wins — BS Greeks, simple analytic models, unit testing, validation against known closed-form Greeks |
| **Reverse (tape-based)** | Production path — MC Greeks, Heston/Bates Greeks, curve sensitivities, XVA, portfolio-level risk |

---

## 3. Architecture

### 3.1 The `Number` trait — generic scalar abstraction

The foundational change is introducing a trait that abstracts over `f64` and AD-enabled
types. This follows the pattern used by `nalgebra::RealField` and Antoine Savine's
`Number` concept:

```rust
// crates/ql-aad/src/number.rs

/// Marker + arithmetic trait for scalars that can be used in pricing computations.
/// Implemented by f64 (no-op), Dual (forward-mode), and AReal (reverse-mode).
pub trait Number:
    Copy
    + Clone
    + Send
    + Sync
    + Default
    + PartialOrd
    + fmt::Debug
    + fmt::Display
    + From<f64>
    + Into<f64>           // extract value (drops derivative info)
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
{
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;

    // Transcendental functions — must be overridden for AD types
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
    fn powf(self, n: Self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;

    // Constants
    fn zero() -> Self;
    fn one() -> Self;
    fn pi() -> Self;
    fn epsilon() -> Self;

    // Comparison (needed because PartialOrd on AD types compares values only)
    fn is_positive(self) -> bool;
    fn is_negative(self) -> bool;
}

impl Number for f64 { /* delegate to std f64 methods — zero overhead */ }
```

### 3.2 Forward-mode: `Dual`

```rust
// crates/ql-aad/src/dual.rs

/// Forward-mode dual number: value + one directional derivative.
#[derive(Clone, Copy, Debug)]
pub struct Dual {
    pub val: f64,
    pub dot: f64,   // ∂val/∂(seeded input)
}

impl Number for Dual { /* propagate dot through arithmetic and transcendentals */ }
```

For multiple seeds simultaneously, a `DualVec<const N: usize>` variant carries an
array of $N$ derivatives (multi-directional forward mode):

```rust
#[derive(Clone, Copy, Debug)]
pub struct DualVec<const N: usize> {
    pub val: f64,
    pub dot: [f64; N],
}
```

This is ideal for BS-style engines where $N = 5$ (spot, vol, rate, div yield, time).

### 3.3 Reverse-mode: tape-based `AReal`

```rust
// crates/ql-aad/src/tape.rs

/// Thread-local computation tape (Wengert list).
pub struct Tape {
    nodes: Vec<Node>,
}

struct Node {
    /// Index of this node
    idx: usize,
    /// Partial derivatives w.r.t. children
    partials: SmallVec<[(usize, f64); 2]>,  // (child_idx, ∂self/∂child)
    /// Cached value (for backward pass inspection)
    value: f64,
}

/// Active scalar on the tape — records operations during forward pass.
#[derive(Clone, Copy)]
pub struct AReal {
    idx: usize,     // position on the tape
    val: f64,       // forward value
}

impl Number for AReal { /* push nodes to thread-local tape */ }

impl Tape {
    /// Run the backward pass from output node, returns gradient w.r.t. all inputs.
    pub fn adjoint(&self, output: AReal) -> Vec<f64>;

    /// Reset tape for next computation (reuse allocation).
    pub fn clear(&mut self);

    /// Mark an AReal as an input (for selective gradient extraction).
    pub fn input(&mut self, val: f64) -> AReal;
}
```

**Thread-local tape** avoids contention — each pricing thread gets its own tape.
This matches the rayon-parallel MC architecture: each batch records its own tape,
computes its own adjoints, and results are aggregated.

### 3.4 New crate: `ql-aad`

```
crates/ql-aad/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── number.rs          # Number trait + f64 impl
    ├── dual.rs            # Forward-mode Dual / DualVec<N>
    ├── tape.rs            # Tape, Node, AReal
    ├── math.rs            # AD-aware normal CDF, erf, special functions
    └── tests/
        ├── dual_tests.rs
        ├── tape_tests.rs
        └── validation.rs  # compare AD Greeks vs analytic / finite-diff
```

**Dependencies:** minimal — only `std`, `smallvec`. No external AD framework.
The `Number` trait is re-exported from `ql-core` so downstream crates can be generic.

### 3.5 Crate dependency graph (additions in bold)

```
ql-aad  ←  ql-core  ←  ql-math  ←  ql-termstructures  ←  ql-pricingengines
                                 ←  ql-processes
                                 ←  ql-instruments
```

`ql-aad` has zero dependencies on other ql-rust crates. `ql-core` re-exports
`Number`, `Dual`, `AReal`, `Tape`.

---

## 4. Incremental rollout strategy

The codebase is 100% `f64`-monomorphic today. A big-bang conversion would be
destabilizing. Instead, we adopt a **four-phase** plan where each phase delivers
standalone value and all existing `f64` code paths remain unchanged.

### Phase 1: Foundation (ql-aad crate + forward-mode quick wins)

**Scope:** ~2,000 LOC · ~1 week

1. Create `ql-aad` crate with `Number` trait, `Dual`, `DualVec<N>`, `impl Number for f64`
2. Implement all transcendentals for `Dual` with unit tests
3. Add AD-aware `normal_cdf<T: Number>()` and `normal_pdf<T: Number>()`
4. Create `bs_price_generic<T: Number>(...)` — a single generic BS pricer that
   works for `f64`, `Dual`, and (later) `AReal`. Validate against existing
   `AnalyticEuropeanResults` closed-form Greeks.
5. Expose `bs_greeks_forward_ad(spot, strike, r, q, vol, t) -> AnalyticEuropeanResults`
   that seeds `DualVec<5>` and computes all Greeks in one pass.

**Deliverable:** Forward-mode BS Greeks matching closed-form to machine epsilon.
Zero impact on existing code — purely additive.

**Validation:**
```rust
#[test]
fn ad_greeks_match_analytic() {
    let analytic = price_european(spot, strike, r, q, vol, t, Call);
    let ad = bs_greeks_forward_ad(spot, strike, r, q, vol, t, Call);
    assert!((analytic.delta - ad.delta).abs() < 1e-14);
    assert!((analytic.gamma - ad.gamma).abs() < 1e-14);
    // ...
}
```

### Phase 2: Reverse-mode tape + Heston/Bates Greeks

**Scope:** ~3,000 LOC · ~2 weeks

1. Implement `Tape` and `AReal` with full `Number` trait
2. Tape memory management: `clear()`, arena-style reuse, configurable initial capacity
3. Create `heston_price_generic<T: Number>(...)` — lift the Heston semi-analytic
   engine to generic scalar. Requires generic Gauss-Lobatto/Gauss-Laguerre
   integration (only the integrand, not the quadrature weights which stay `f64`).
4. `heston_greeks_aad(params) -> HestonGreeks` — adjoint pass gives
   $\partial V / \partial \{S, v_0, \kappa, \theta, \sigma, \rho, r, q\}$ in one shot.
5. Same for Bates (Heston + Merton jumps).

**Deliverable:** First-ever Heston/Bates adjoint Greeks in an open-source Rust library.
These are currently unavailable in ql-rust (engines return only NPV).

**Benchmark target:** Adjoint Heston Greeks in < 3× the cost of a single price.

### Phase 3: Generic math layer (selective)

**Scope:** ~4,000 LOC · ~2 weeks

Not all math needs to be made generic — only the functions that appear on the
"hot path" of pricing computations. Prioritize by usage:

| Module | Action | Rationale |
|--------|--------|-----------|
| `distributions` | `normal_cdf<T>`, `normal_pdf<T>`, `inv_normal_cdf` (f64-only OK) | CDF/PDF appear in every BS-family model |
| `interpolation` | `LinearInterpolation<T>`, `CubicSpline<T>` | Curve lookups appear in term structure bootstrapping |
| `solvers1d` | `brent<T>`, `newton<T>` | Implied vol, root-finding during calibration |
| `integration` | Integrand generic, weights `f64` | Heston characteristic function |
| `matrix` | Delegate to nalgebra's generic `RealField` | Cholesky, eigendecomposition for correlation |
| `optimization` | **Skip** — calibration inputs/outputs stay `f64` | Calibration finds parameters; AD of calibration is a second-order concern |
| `statistics` | **Skip** — post-pricing analytics | Running mean, VaR etc. don't need AD |
| `copulas` | **Skip** | Rarely on the pricing hot path |
| `fft` | **Skip** | COS method can use AD through the characteristic function directly |

The key insight: **only ~40% of the math library needs generalization**. Functions
that are "setup" or "post-processing" can remain `f64`.

**Backward compatibility:** Every generic function retains a non-generic convenience
wrapper:

```rust
// Generic version
pub fn linear_interp<T: Number>(xs: &[f64], ys: &[T], x: f64) -> T { ... }

// Convenience wrapper — existing call sites unchanged
pub fn linear_interp_f64(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    linear_interp::<f64>(xs, ys, x)
}
```

### Phase 4: MC pathwise Greeks + production integration

**Scope:** ~3,000 LOC · ~2 weeks

1. **Pathwise MC Greeks:** For each simulated path, the payoff is a differentiable
   function of the simulated asset prices, which are differentiable functions of
   the model parameters. The tape records the entire path + payoff, and the
   adjoint pass produces $\partial \text{payoff}_i / \partial \theta$ for every
   path $i$.

   ```rust
   pub fn mc_european_aad<T: Number>(
       spot: T, strike: f64, r: T, q: T, vol: T,
       t: f64, option_type: OptionType,
       num_paths: usize, seed: u64,
   ) -> MCAADResult {
       // Each path:
       //   1. tape.clear()
       //   2. simulate path with T arithmetic (records on tape)
       //   3. compute payoff
       //   4. tape.adjoint(payoff) → path-level Greeks
       //   5. accumulate
   }
   ```

   **Note on non-differentiable payoffs:** Digital/barrier payoffs have
   discontinuities. For these, combine AAD with **likelihood ratio method** (LRM)
   or **Malliavin calculus** smoothing — pathwise AAD handles the continuous parts,
   LRM handles the indicator function.

2. **Curve-level AAD:** Record the adjoint through the **discount factor lookup**
   in a swap/bond pricer. The tape captures the interpolation through the curve,
   giving exact DV01/KRD per pillar without any bumping.

3. **Portfolio-level adjoint aggregation:** For a portfolio of $M$ instruments
   sharing $N$ market inputs, run one forward pass pricing all instruments
   (or batch them), then one backward pass from the total PnL to get
   $\partial \text{PnL} / \partial x_i$ for all $i$.

4. **Integration with `ReactivePortfolio`:** When any market input changes, the
   reactive system reprices affected instruments. With AAD, it _also_ efficiently
   recomputes all Greeks without separate bumping.

---

## 5. Performance expectations

Based on published benchmarks (Savine 2018, Capriotti 2011, Henrard 2014):

| Engine | Current cost (bump) | AAD cost | Speedup |
|--------|-------------------|----------|---------|
| BS European (5 Greeks) | 10× (5 central diffs) | 3–4× (forward `DualVec<5>`) | **2.5–3×** |
| Heston (8 params) | 16× (N/A today — would need bumping) | 3–5× (reverse tape) | **3–5×** |
| MC European (5 Greeks, 100K paths) | 10× | 3–5× | **2–3×** |
| MC Heston (8 Greeks, 100K paths) | 16× | 3–5× | **3–5×** |
| Swap DV01 (20 pillars) | 40× | 3–5× | **8–13×** |
| Portfolio (50 instruments, 200 risk factors) | 400× | 5–10× | **40–80×** |

The portfolio-level case is where AAD delivers transformative speedup: computing
all sensitivities of a portfolio to all market inputs goes from O(N·M) to O(M)
where M is the total forward-pass cost.

**Memory overhead:** The tape stores ~40 bytes per elementary operation (two `usize`
indices + two `f64` partials). A Heston price evaluation involves ~10K operations
→ ~400 KB tape. An MC simulation with 10K paths × 252 steps → ~100M operations
→ ~4 GB tape if naïve. **Checkpointing** (re-compute sub-intervals instead of
storing everything) reduces this to ~50 MB — a standard technique.

---

## 6. Key design decisions

### 6.1 Build a tape from scratch vs. use an existing crate

| Option | Pros | Cons |
|--------|------|------|
| **Custom tape** | Full control over memory layout, checkpointing, thread-local storage; matches Savine's proven design; no external dependency risk | More code to write and maintain |
| **`num-dual` crate** | Mature forward-mode dual numbers | No reverse-mode; limited to forward |
| **`autodiff` crate** | Reverse-mode support | Experimental; proc-macro based; limited control over tape |
| **Enzyme (LLVM plugin)** | Zero-overhead AD at LLVM IR level | Requires nightly Rust; fragile toolchain dependency; opaque |

**Recommendation: custom tape.** The quant AD problem is well-understood (Savine's
book provides a complete blueprint), and a custom implementation gives us:
- Thread-local tapes matching rayon's work-stealing model
- Checkpointing for MC path memory management
- Arena allocation tuned for our access patterns
- No nightly-only or proc-macro dependencies
- Full debuggability

### 6.2 Generics vs. operator overloading only

Two schools exist:

**(A) Make everything generic** — `fn price<T: Number>(spot: T, ...) -> T`

- **Pro:** Single source of truth; optimizer monomorphizes to zero-overhead f64 code
- **Con:** Requires touching many files; generic bounds proliferate; some functions
  (e.g. involving `statrs`) can't easily be made generic

**(B) Operator overloading only** — use `AReal` that looks like `f64` but records

- **Pro:** Less refactoring; just change `f64` → `AReal` at call sites
- **Con:** Loses type safety; can't have both f64 and AD paths coexist in one function;
  no monomorphization benefit

**Recommendation: (A) generics**, applied incrementally per Phase 3's prioritization.
Rust's monomorphization guarantees that `price::<f64>(...)` compiles to identical code
as today's `price(...)` — zero performance regression. The generic bound `T: Number`
is erased at compile time for `f64`.

### 6.3 Thread-local vs. shared tape

**Thread-local.** Each rayon worker thread gets its own `Tape` via `thread_local!`.
This avoids all synchronization overhead and matches the existing MC parallelism
model where each batch is independent. Aggregation of per-path Greeks happens after
the adjoint pass, same as NPV aggregation today.

### 6.4 Handling non-differentiable operations

| Operation | Occurs in | Strategy |
|-----------|-----------|----------|
| `max(x, 0)` (call payoff) | Vanilla options | Subgradient: $\partial \max(x,0)/\partial x = \mathbb{1}_{x>0}$ — standard, produces correct pathwise delta |
| `if x > barrier` | Barrier options | Likelihood ratio method for the indicator; AAD for the rest |
| `sort`, `median` | LSM regression basis | Detach from tape — regression coefficients are treated as constants in the backward pass |
| `integer indexing` | Interpolation bracket lookup | Index is non-differentiable but the interpolated value is — record the interpolation, not the search |
| Early exercise decision | American MC (LSM) | The exercise boundary is non-differentiable; use "backward differentiation through optimal stopping" (Leclerc et al., 2009) or treat the boundary as frozen |

### 6.5 Checkpointing for MC

For a Monte Carlo simulation with $P$ paths and $S$ time steps, storing the full tape
for all paths is infeasible. Standard solution:

1. **Per-path taping:** Clear the tape between paths. Each path's adjoint is computed
   independently and Greeks are accumulated.
2. **Binomial checkpointing (Griewank):** For very long paths ($S > 1000$), store
   checkpoints every $\sqrt{S}$ steps. Recompute from the nearest checkpoint during
   the backward pass. Trades $O(\sqrt{S})$ memory for $O(S \log S)$ compute.

Option 1 is sufficient for typical equity/rates MC (252 steps) and is simple
to implement. Option 2 is a future optimization for long-dated exotics.

---

## 7. Testing strategy

### 7.1 Validation hierarchy

Each AD-computed Greek is validated against at least two independent methods:

```
AD Greek  ←→  Analytic formula (where available)
          ←→  Central finite difference (universal fallback)
          ←→  Forward-mode dual (cross-validate reverse-mode)
```

### 7.2 Test categories

| Category | Count | Description |
|----------|-------|-------------|
| **Unit: Dual arithmetic** | ~30 | `+`, `−`, `×`, `÷`, `exp`, `ln`, `sqrt`, chain rule |
| **Unit: Tape operations** | ~30 | Record, adjoint, clear, reuse, nested tapes |
| **Integration: BS Greeks** | ~10 | All 5 Greeks for calls/puts, ATM/ITM/OTM |
| **Integration: Heston Greeks** | ~10 | 8 sensitivities vs finite-diff |
| **Integration: MC Greeks** | ~10 | Pathwise delta/vega/rho vs bump-and-reprice |
| **Regression: f64 parity** | ~20 | Verify `price::<f64>(...)` produces identical results to existing non-generic code |
| **Performance: no regression** | ~5 | Criterion benchmarks ensuring f64 path is unchanged |
| **Stress: large tape** | ~5 | Memory usage, tape clear/reuse, checkpointing |

### 7.3 Criterion benchmarks

```rust
// Benchmark: f64 path must not regress
bench_group.bench_function("bs_european_f64",        |b| b.iter(|| bs_price::<f64>(...)));
bench_group.bench_function("bs_european_dual5",      |b| b.iter(|| bs_price::<DualVec<5>>(...)));
bench_group.bench_function("bs_european_areal",      |b| b.iter(|| bs_price::<AReal>(...)));
bench_group.bench_function("heston_price_f64",       |b| b.iter(|| heston_price::<f64>(...)));
bench_group.bench_function("heston_greeks_adjoint",  |b| b.iter(|| heston_greeks_aad(...)));
bench_group.bench_function("mc_european_adjoint_50k", |b| b.iter(|| mc_european_aad::<AReal>(...)));
```

---

## 8. Risks and mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **Generic proliferation** — `T: Number` bounds spread to every function | Code complexity | High | Phase 3 limits generics to hot-path functions only; convenience wrappers hide generics from users |
| **Compile time increase** — monomorphization doubles code generation | Developer experience | Medium | Feature-gate AD types behind `cfg(feature = "aad")`; f64-only builds unaffected |
| **`statrs` incompatibility** — `statrs` only works with `f64` | Blocks generic distributions | High | Implement our own `normal_cdf<T>`, `normal_pdf<T>` using Abramowitz-Stegun or rational approximations. We already have `NormalDistribution` wrapper — replace internals. |
| **`nalgebra` generics** — nalgebra supports `RealField` but our usage is f64 | Blocks generic matrix ops | Low | nalgebra already supports generic scalars; implement `RealField` for `AReal`/`Dual` |
| **MC tape memory** — 100K paths × 252 steps = large tape | OOM on big simulations | Medium | Per-path taping (clear between paths); checkpointing for very long paths |
| **Non-differentiable payoffs** — barriers, digitals | Incorrect Greeks | Medium | Hybrid AAD + likelihood ratio; document which payoffs support pathwise vs LRM |
| **f64 performance regression** — generic code slower than hardcoded f64 | Breaks existing benchmarks | Low | Rust monomorphization eliminates abstraction cost; verify with criterion |

---

## 9. Estimated effort

| Phase | Scope | LOC | Duration | Depends on |
|-------|-------|-----|----------|------------|
| **Phase 1** | `ql-aad` crate, `Number`, `Dual`, forward-mode BS Greeks | ~2,000 | 1 week | — |
| **Phase 2** | Reverse-mode tape, `AReal`, Heston/Bates adjoint Greeks | ~3,000 | 2 weeks | Phase 1 |
| **Phase 3** | Generic math layer (distributions, interpolation, solvers) | ~4,000 | 2 weeks | Phase 1 |
| **Phase 4** | MC pathwise Greeks, curve-level AAD, portfolio integration | ~3,000 | 2 weeks | Phases 2 + 3 |
| **Phase 5** | Vectorized (SIMD) tape evaluation for MC batches | ~1,500 | 1 week | Phase 2 |
| **Phase 6** | Cranelift JIT compilation of backward pass (optional) | ~2,500 | 2 weeks | Phase 2 |
| **Total** | | ~16,000 | ~10 weeks | |

Phases 2 and 3 can proceed in parallel after Phase 1 is complete.
Phases 5 and 6 are independent optimizations — Phase 5 (SIMD) is recommended first.

---

## 10. Success criteria

1. **Correctness:** All AD Greeks agree with analytic formulas (where available) or
   central finite differences to within $10^{-8}$ relative tolerance.

2. **Zero regression:** `bs_price::<f64>()` produces bit-identical results to today's
   `price_european()` and runs within 5% of current benchmark times.

3. **Performance:** Adjoint Heston Greeks (8 sensitivities) computed in < 5× the cost
   of a single forward price.

4. **Usability:** Computing Greeks requires ≤ 3 lines of additional code vs. a plain
   price call:
   ```rust
   let mut tape = Tape::new();
   let spot = tape.input(100.0);
   let price = heston_price(&tape, spot, ...);
   let greeks = tape.adjoint(price);  // greeks[spot.idx] == delta
   ```

5. **Scope:** At least BS, Heston, Bates, and one MC engine have AD-enabled Greek
   computation by end of Phase 4.

---

## 11. JIT compilation of the tape (future optimization)

### 11.1 The interpretation overhead problem

The tape backward pass in a scalar interpreter looks like:

```rust
fn adjoint(&self, output_idx: usize) -> Vec<f64> {
    let mut adj = vec![0.0; self.nodes.len()];
    adj[output_idx] = 1.0;
    for i in (0..=output_idx).rev() {
        let node = &self.nodes[i];
        for &(child_idx, partial) in &node.partials {
            adj[child_idx] += adj[i] * partial;
        }
    }
    adj
}
```

The actual useful work per node is a single `fma`. The overhead — indirect jumps,
`SmallVec` pointer chasing, bounds checks, branch misprediction on varying node
arity — is typically **3–10×** the useful arithmetic (Griewank & Walther 2008).
For large tapes (>10K nodes), this overhead becomes the bottleneck.

Two complementary techniques can eliminate it:

### 11.2 Vectorized tape evaluation (recommended Phase 5)

Instead of processing one MC path at a time through the tape, process **N paths
simultaneously** using SIMD. Each node applies its operation to a lane of values:

```rust
use std::simd::f64x4;

fn eval_tape_simd(tape: &Tape, inputs: &[f64x4]) -> Vec<f64x4> {
    // Same tape structure, 4 paths evaluated in parallel per node
}
```

This is the approach described by Savine (*Modern Computational Finance*, Ch. 14).
In MC simulation all paths share identical tape structure, so one tape traversal
computes adjoints for 4/8 paths at once:

| ISA | SIMD width | Paths per traversal | Expected throughput gain |
|-----|------------|---------------------|------------------------|
| SSE2 | 128-bit | 2 `f64` | ~1.8× |
| AVX2 | 256-bit | 4 `f64` | ~3.5× |
| AVX-512 | 512-bit | 8 `f64` | ~6× |

**Implementation:** Use Rust's `std::simd` (portable SIMD, stabilizing) or
`core::arch` intrinsics. The tape `Node` struct is unchanged; only the evaluation
loop and value storage become SIMD-typed.

**Recommendation: implement this first (Phase 5).** Lower complexity than JIT,
significant throughput gain, no external dependencies, easily debuggable.

### 11.3 JIT-compiling the backward pass (optional Phase 6)

After recording a tape, **compile the adjoint traversal into native code** that
eliminates all interpretation overhead:

```
 Record tape    →   Analyze/optimize   →   JIT-compile    →   Execute ×10⁶
 (Vec<Node>)        (CSE, DCE, fold)       (Cranelift)        (native fn)
```

The compiled function is a straight-line sequence of `fma` and load/store
instructions with all structure resolved at compile time — no branches, no
indirection, no bounds checks.

#### 11.3.1 Open-source JIT frameworks for Rust

**Cranelift** (recommended):

| Aspect | Detail |
|--------|--------|
| Crates | `cranelift-codegen` / `cranelift-frontend` / `cranelift-jit` / `cranelift-module` / `cranelift-native` (v0.129+) |
| Nature | Pure Rust, no system dependencies |
| Compilation speed | ~0.5ms for a 10K-op tape |
| Code quality | ~70% of LLVM `-O2` |
| Arithmetic ops | `fadd`, `fsub`, `fmul`, `fdiv`, `fneg`, `fabs`, `sqrt`, `fma`, `fmax`, `fmin` — all native |
| Transcendentals | `exp`, `ln`, `sin`, `cos` — **not built-in**, must emit calls to `libm` |
| Maturity | Production (powers Wasmtime, `rustc_codegen_cranelift`) |

Cranelift is ideal because:
- Sub-millisecond compilation makes JIT practical even for moderate reuse (>100 evals)
- The adjoint backward pass is mostly `fma` + arithmetic — Cranelift handles this well
- Pure Rust = builds anywhere, no CI headaches

```rust
// Conceptual: compile tape backward pass to native
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::Module;

fn compile_adjoint(tape: &Tape) -> CompiledAdjoint {
    let mut module = JITModule::new(/* ... */);
    // fn adjoint_fn(values: *const f64, adjoints: *mut f64)
    let mut builder = FunctionBuilder::new(/* ... */);

    // Emit straight-line code for each tape node (reverse order):
    for node in tape.nodes.iter().rev() {
        // adj[child] += adj[node.idx] * partial
        // → load, fmul/fma, store — all direct, no branches
    }

    module.finalize_definitions().unwrap();
    // Return callable function pointer
}
```

**LLVM via inkwell** (alternative, feature-gated):

| Aspect | Detail |
|--------|--------|
| Crate | `inkwell` v0.8+ (safe wrapper over `llvm-sys`) |
| Nature | Requires system LLVM 11–21 installation (~100MB+) |
| Compilation speed | ~20ms for a 10K-op tape at `-O2` |
| Code quality | Best-in-class; auto-vectorization, SLP, GVN |
| Transcendentals | Native LLVM intrinsics (`llvm.exp.f64`, `llvm.log.f64`, etc.) |
| Maturity | Powers `rustc`, Clang, Julia, virtually everything |

LLVM generates ~20–30% faster code than Cranelift for arithmetic-heavy workloads,
but requires a heavy system dependency. Recommended only behind
`feature = "jit-llvm"` for users who need maximum throughput on long-running MC.

**Other options considered and rejected:**

| Option | Reason to skip |
|--------|---------------|
| `dynasm-rs` | Assembler-level: hand-write x86-64/AArch64 asm per tape op, manual register allocation — maintenance nightmare |
| GNU Lightning (`lightning-sys`) | Abandoned, negligible adoption |
| Enzyme / `#[autodiff]` | LLVM-plugin, nightly-only, compile-time AD (not runtime tape) — fundamentally different. Worth watching for stabilization but not applicable to tape JIT |

#### 11.3.2 When JIT is worth it

| Scenario | Tape reuse | JIT overhead | Worthwhile? |
|----------|------------|-------------|-------------|
| MC pricing (10K+ paths) | Same structure, all paths | ~0.5ms (Cranelift) amortized over seconds of MC | **Yes, strongly** |
| Calibration (100+ iterations) | Same model structure | ~0.5ms amortized over 100+ evals | **Yes** |
| Bumped repricing (N risk factors) | Identical tape | ~0.5ms amortized over N evals | **Yes** |
| One-off analytic price | Used once | Compilation > interpretation | **No** |
| Heterogeneous portfolio | Different tape per instrument | One compilation each, 1 eval each | **No** |

#### 11.3.3 Reported speedups

| Source | Speedup from compiling tape | Context |
|--------|---------------------------|---------|
| CppADCodeGen (open source C++) | 2–10× over CppAD interpreted tape | Large tapes (>10K nodes) |
| JAX `jit(grad(f))` vs eager | 5–50× | Python → XLA compilation |
| Matlogica AADC (commercial) | 10–100× over bump-and-reprice | Includes AAD benefit + compilation |
| Griewank & Walther (theoretical) | 3–10× interpretation overhead eliminated | Scalar operations |

Realistic expectation for ql-rust: **2–5× over interpreted tape** for the adjoint
pass alone (arithmetic-heavy, few transcendentals). Combined with vectorization,
**JIT + SIMD could yield 8–20×** over naïve scalar tape interpretation.

#### 11.3.4 Composing JIT with vectorized evaluation

The optimal endpoint is JIT-compiling a **vectorized** backward pass — the compiled
function processes 4/8 paths per invocation using SIMD registers directly, with all
tape structure resolved at compile time. This is essentially what XLA does for JAX
on CPUs. Cranelift supports SIMD types (`f64x2`, `f64x4` via `128` and `256`-bit
vector types), making this feasible.

### 11.4 Feature-gating strategy

```toml
# crates/ql-aad/Cargo.toml
[features]
default = []
simd = []                 # vectorized tape evaluation (std::simd)
jit = [                   # Cranelift JIT compilation
    "cranelift-codegen",
    "cranelift-frontend",
    "cranelift-jit",
    "cranelift-module",
    "cranelift-native",
]
jit-llvm = ["inkwell"]    # LLVM JIT (requires system LLVM)
```

The scalar tape interpreter is always available (no features required). Vectorized
evaluation and JIT are opt-in performance tiers.

### 11.5 Existing art: who does JIT + AD?

| Project | Language | Approach | Open source? |
|---------|----------|----------|-------------|
| **JAX** | Python → XLA | Trace jaxpr → XLA HLO → LLVM native | Yes (Apache 2.0) |
| **PyTorch Inductor** | Python → Triton/C++ | FX graph → TorchInductor backend | Yes (BSD) |
| **CppADCodeGen** | C++ | CppAD tape → C source → GCC/Clang → dlopen | Yes (EPL 2.0) |
| **Matlogica AADC** | C++ | Proprietary vectorized tape + optional compilation | No (commercial) |
| **XAD** | C++ | Tape-based, no JIT | Yes (AGPL 3.0) |
| **Enzyme** | LLVM IR | Compile-time AD of LLVM IR (not tape JIT) | Yes (Apache 2.0) |
| **Rust ecosystem** | — | **No existing Rust project combines JIT + AD** | — |

Implementing Cranelift-based tape JIT would be **novel in the Rust ecosystem**.

---

## 12. Instrument-Level AD Gap Analysis

**Date of audit:** 2026-03-03

The `ql-aad` crate provides a mature AD infrastructure (forward-mode `Dual`/`DualVec`,
reverse-mode `AReal` with tape, SIMD vectorization, Griewank checkpointing, LRM for
non-smooth payoffs, Cranelift JIT compilation). However, coverage is concentrated on a
narrow set of instruments. This section catalogues every instrument type in
`ql-instruments`, its pricing engine(s) in `ql-pricingengines`, and its current AD status.

### 12.1 Current AD coverage

| Instrument | Engine / AD Module | AD Mode | Greeks |
|------------|-------------------|---------|--------|
| European vanilla (BS) | `bs_price_generic<T>` | Forward `DualVec<5>` | Δ, Γ, ν, θ, ρ |
| European vanilla (Heston) | `heston_price_generic<T>` | Forward `DualVec<8>` / Reverse | Δ, Γ, ν₀, ∂κ, ∂θ, ∂σ, ∂ρ, ρ_rate |
| European vanilla (Bates) | `bates_price_generic<T>` | Forward `DualVec<11>` / Reverse | All Heston + ∂λ, ∂ν, ∂δⱼ |
| European vanilla (MC GBM) | `mc_european_aad` | Reverse (tape) | Δ, ν, ρ, div_ρ |
| European vanilla (MC Heston) | `mc_heston_aad` | Reverse (tape) | 8 first-order Greeks |
| Forward-start option (MC) | `mc_european_forward` | Forward `DualVec<4>` | Δ, ν, ρ, div_ρ |
| Digital option | `mc_digital_lrm` | LRM | Δ, ν, ρ, div_ρ |
| Barrier (DO/UO, MC) | `mc_barrier_do_lrm`, `mc_barrier_uo_lrm` | LRM | Δ, ν, ρ, div_ρ |
| Barrier + vanilla (hybrid) | `mc_barrier_vanilla_hybrid` | AAD + LRM | Δ, ν, ρ, div_ρ |
| Fixed-rate bond (cashflows) | `cashflows::npv<T>` + `DiscountCurveAD` | Reverse | ∂NPV/∂pillar |
| Vanilla swap (fixed leg) | `cashflows::npv<T>` + `DiscountCurveAD` | Reverse | ∂NPV/∂pillar |
| Portfolio (bonds/swaps) | `Portfolio::compute()` | Reverse | All KRDs in one pass |

**Total: 12 instrument/engine combinations** out of **60+ instrument types** and
**100+ pricing engines**.

### 12.2 Instruments with NO AD support — full gap list

The gaps are organized by difficulty tier, which determines the recommended
conversion order.

#### Tier 1 — Easy (pure arithmetic, no distribution dependency)

These engines use only basic arithmetic (`+`, `−`, `×`, `÷`, `exp`, `sqrt`, `powf`)
already available in the `Number` trait. Can be converted immediately.

| ID | Instrument | Engine | LOC | Blocker | Priority |
|----|-----------|--------|-----|---------|----------|
| **AD-1** | Variance swap | `price_variance_swap` | ~60 | None | **High** |
| **AD-2** | Quanto adjustment | `quanto_adjustment` | ~40 | None | **High** |
| **AD-3** | BSM + Hull-White | `price_bsm_hull_white` | ~100 | Needs generic `cumulative_normal` (inline A-S approx available) | High |
| **AD-4** | Digital American | `digital_american` | ~200 | Needs generic `cumulative_normal` | High |

#### Tier 2 — Medium (need generic `cumulative_normal<T>` / `cumulative_normal_pdf<T>`)

All are analytic closed-form engines whose only blocker is that `NormalDistribution::cdf`
wraps `statrs` (f64-only). Once a generic Abramowitz-Stegun or rational-approximation
`cumulative_normal<T: Number>` exists, conversion is mechanical.

| ID | Instrument | Engine | LOC | Notes | Priority |
|----|-----------|--------|-----|-------|----------|
| **AD-5** | European vanilla (main engine) | `price_european` | ~150 | Duplicates `bs_price_generic` but produces `AnalyticEuropeanResults` with all Greeks | High |
| **AD-6** | Swaption (Black) | `black_swaption` | ~100 | Pure BS-style formula | **High** |
| **AD-7** | Swaption (Bachelier) | `bachelier_swaption` | ~80 | Normal-model formula | High |
| **AD-8** | Cap/floor (Black) | `black_cap_floor` | ~120 | Loop over caplets, each is BS formula | **High** |
| **AD-9** | Cap/floor (Bachelier) | `bachelier_cap_floor` | ~100 | Loop over caplets, normal-model | High |
| **AD-10** | Asian (geometric, continuous) | `asian_geometric_continuous_avg_price` | ~60 | Kemna-Vorst closed-form | Medium |
| **AD-11** | Asian (geometric, discrete) | `asian_geometric_discrete_avg_price` | ~80 | Closed-form | Medium |
| **AD-12** | Asian (Turnbull-Wakeman) | `asian_turnbull_wakeman` | ~80 | Closed-form approximation | Medium |
| **AD-13** | Asian (Levy) | `asian_levy` | ~60 | Log-normal approximation | Medium |
| **AD-14** | Lookback (floating/fixed) | `analytic_lookback` | ~200 | Goldman-Sosin-Gatto; minor `b ≈ 0` branch | Medium |
| **AD-15** | Chooser option | `chooser_price` | ~60 | Rubinstein (1991) closed-form | Medium |
| **AD-16** | Spread option (Kirk) | `kirk_spread_call/put` | ~80 | BS-style with effective vol | Medium |
| **AD-17** | Spread option (operator-splitting) | `operator_splitting_spread` | ~120 | Fixed 5-iteration loop (AD-friendly) | Medium |
| **AD-18** | Exchange option (Margrabe) | `margrabe_exchange` | ~60 | Closed-form BS variant | Medium |
| **AD-19** | Max/min call (Stulz) | `stulz_max_call`, `stulz_min_call` | ~120 | Bivariate normal CDF needed | Medium |
| **AD-20** | Double barrier | `double_barrier_knockout/knockin` | ~150 | Eigenfunction series + `sin`/`PI` | Medium |
| **AD-21** | Binary barrier (Reiner-Rubinstein) | `analytic_binary_barrier` | ~200 | CDF + `powf` heavy | Medium |
| **AD-22** | Double binary barrier | `double_binary_barrier` | ~100 | CDF + `powf` | Medium |
| **AD-23** | Partial-time barrier | `partial_time_barrier` | ~120 | Bivariate normal CDF | Medium |
| **AD-24** | Vanna-Volga barrier | `vanna_volga_barrier` | ~150 | Internal BS helpers + CDF | Medium |
| **AD-25** | HW bond option | `hw_bond_option` | ~80 | CDF | Medium |
| **AD-26** | HW caplet/floorlet | `hw_caplet`, `hw_floorlet` | ~60 | CDF | Medium |
| **AD-27** | HW Jamshidian swaption | `hw_jamshidian_swaption` | ~120 | CDF + root-find for critical rate | Medium–Hard |
| **AD-28** | Merton jump-diffusion | `merton_jump_diffusion` | ~100 | Truncated series (fast convergence) | Medium |
| **AD-29** | Inflation cap/floor (Black) | `black_yoy_inflation_cap_floor`, `black_zc_inflation_cap_floor` | ~120 | BS formula; instrument struct is f64 | Medium |
| **AD-30** | Inflation cap/floor (Bachelier) | `bachelier_yoy_inflation_cap_floor`, `bachelier_zc_inflation_cap_floor` | ~100 | Normal-model | Medium |
| **AD-31** | Quanto European | `quanto_european` | ~80 | CDF + quanto adjustment | Medium |
| **AD-32** | Power option | `power_option` | ~60 | `powf(alpha)` (in `Number`) + CDF | Medium |
| **AD-33** | Forward-start option (analytic) | `forward_start_option` | ~60 | BS-style + CDF | Medium |
| **AD-34** | Digital barrier | `digital_barrier` | ~80 | CDF + `powf` | Medium |
| **AD-35** | GJR-GARCH option | `gjr_garch_option` | ~100 | Recursive variance loop + CDF | Medium |
| **AD-36** | Vasicek bond option | `vasicek_bond_option` | ~80 | Analytic CDF | Medium |
| **AD-37** | Vasicek European equity | `vasicek_european_equity` | ~60 | Analytic CDF | Medium |
| **AD-38** | CDS option (Black) | `cds_option_black` | ~60 | BS-style on CDS spread | Medium |
| **AD-39** | CDO tranche (Gaussian copula LHP) | `price_cdo_tranche` / `loss_distribution_lhp` | ~200 | Gauss-Hermite integration + `cumulative_normal` | Medium |
| **AD-40** | Cliquet option | `cliquet_price` | ~100 | Calls `analytic_european` per period | Medium |
| **AD-41** | Replicating variance swap | `replicating_variance_swap` | ~100 | Numerical integration over strike grid | Medium |
| **AD-42** | Holder/writer extensible | `holder_extensible`, `writer_extensible` | ~80 | CDF | Medium |
| **AD-43** | Soft barrier | `price_soft_barrier` | ~80 | CDF | Medium |

#### Tier 3 — Hard (structural barriers)

These engines have fundamental obstacles: iterative root-finding with value-dependent
termination, RNG in MC paths, `Arc<dyn Trait>` term-structure objects, FD grids,
or non-differentiable algorithms (sorting, regression, early-exercise decisions).

| ID | Instrument | Engine | Blocker | Strategy |
|----|-----------|--------|---------|----------|
| **AD-44** | American (BAW) | `barone_adesi_whaley` | Newton iteration for critical S* | Implicit-function-theorem adjoint (IFT): differentiate through converged root without taping the iterations |
| **AD-45** | American (Bjerksund-Stensland) | `bjerksund_stensland` | Conditional branching on exercise regions | Forward-mode `Dual` feasible; reverse needs IFT |
| **AD-46** | American (QD+/QDFP) | `qd_plus_american`, `qdfp_american` | Fixed-point iteration | IFT at convergence; forward `Dual` for small param sets |
| **AD-47** | American (LSM MC) | `mc_american_longstaff_schwartz` | MC + regression + backward exercise | Pathwise: freeze regression coefficients, AD through payoff conditional on frozen boundary (Leclerc et al. 2009) |
| **AD-48** | Compound option | `analytic_compound_option` | Root-find for critical underlying price + bivariate normal | IFT for root; generic bivariate normal needed |
| **AD-49** | Vanilla swap (engine-level) | `price_swap`, `price_swap_multicurve` | `&dyn YieldTermStructure` returns f64 | Build `GenericYieldTermStructure<T>` trait or adapter around `DiscountCurveAD` |
| **AD-50** | OIS swap | `price_ois` | `&dyn YieldTermStructure` | Same as AD-49 |
| **AD-51** | Fixed-rate bond (engine) | `price_bond` | `&dyn YieldTermStructure` + `Date` scheduling | Same as AD-49; `Date` ops produce f64 year fractions (OK — those aren't risk factors) |
| **AD-52** | Floating-rate bond | `price_floating_bond` | `&dyn YieldTermStructure` × 2 (discount + forecast) | Same as AD-49 + forecast curve dependency |
| **AD-53** | Zero-coupon bond | | `&dyn YieldTermStructure` | Same |
| **AD-54** | Amortizing bond/FRN | | Scheduling + `&dyn YieldTermStructure` | Same |
| **AD-55** | Inflation-linked bond | | Inflation term structure (`f64`) | Needs generic inflation curve |
| **AD-56** | Callable bond | `price_callable_bond` | Short-rate tree + exercise decisions | Treat call schedule as non-differentiable; AD through tree node values |
| **AD-57** | Convertible bond | `price_convertible_bond` | Binomial tree + conversion/call/put exercise | Same as callable + equity dependency |
| **AD-58** | CDS (mid-point & ISDA) | `midpoint_cds_engine`, `isda_cds_engine` | `Arc<dyn DefaultProbabilityTermStructure>` + `Arc<dyn YieldTermStructure>` | Generic curve traits; IFT for ISDA settlement-lag adjustment |
| **AD-59** | CDS index | `price_cds_index` | `Arc<dyn>` curves + Date scheduling | Same as AD-58 |
| **AD-60** | Cross-currency swap | `price_xccy_swap` | Two yield curves + FX spot + `leg_npv` | Generic dual-curve framework |
| **AD-61** | FX forward | `price_fx_forward` | Two yield curves + `Date` | Generic yield curves |
| **AD-62** | BMA swap | `price_bma_swap` | `&dyn YieldTermStructure` | Same |
| **AD-63** | Zero-coupon swap | `price_zero_coupon_swap` | `&dyn YieldTermStructure` | Same |
| **AD-64** | Basis swap | `price_basis_swap` | Dual curves | Same |
| **AD-65** | CPI swap | `price_cpi_swap` | Inflation curve | Generic inflation curve |
| **AD-66** | FRA | `price_fra` | `&dyn YieldTermStructure` | Same |
| **AD-67** | Cat bond | `price_cat_bond` | Catastrophe model + `&dyn YieldTermStructure` | Partial: curve part can be AD; cat model is external |
| **AD-68** | Bond forward | `price_bond_forward` | `&dyn YieldTermStructure` | Same |
| **AD-69** | Asian (MC arithmetic) | `mc_asian_arithmetic_*` | RNG + path averaging | Pathwise AD through GBM path; arithmetic mean is smooth |
| **AD-70** | Asian (MC Heston) | `mc_asian_heston_price` | RNG + 2-factor SDE | Pathwise AD; same architecture as `mc_heston_aad` |
| **AD-71** | Basket (MC) | `mc_basket`, `mc_european_basket`, `mc_american_basket` | RNG + Cholesky + multi-asset | Multi-asset pathwise AD; Cholesky is differentiable |
| **AD-72** | Quanto barrier | `price_quanto_barrier` | Barrier + quanto adjustment | Combine AD-2 (quanto) + LRM (barrier) |
| **AD-73** | Two-asset barrier/correlation | `two_asset_correlation` | Bivariate normal + multi-asset | Generic bivariate normal; 2-asset `Number` |
| **AD-74** | Swing/storage option | `fd_swing_option`, `fd_simple_bs_swing` | FD grid + multi-exercise | Generic FD infrastructure |
| **AD-75** | Shout option | `fd_shout_option` | FD grid + exercise decision | Same |
| **AD-76** | Mountain range (Himalaya etc.) | `mc_mountain_range` | MC + sorting/selection (non-diff.) | LRM for selection; pathwise for remainder |
| **AD-77** | Heston barrier (FD) | `fd_heston_barrier` | 2D FD grid | Generic 2D PDE solver |
| **AD-78** | COS-method Heston | `cos_heston_price` | Hardcoded complex `C` struct | Replace with `Complex<T: Number>` (already in `ql-aad`) |
| **AD-79** | Stochastic local vol (MC) | `mc_slv` | Calibration + MC SDE + kernel regression | Freeze local-vol surface, AD through MC paths only |
| **AD-80** | MC Heston + Hull-White | `price_mc_heston_hw` | 3-factor RNG + SDE | Extend `mc_heston_aad` to 3 factors |
| **AD-81** | Binomial barrier | `binomial_barrier` | Tree + knockout logic | Forward-mode through tree; LRM for barrier |
| **AD-82** | MC barrier | `mc_barrier` (with Brownian bridge) | RNG + barrier monitoring | Pathwise AD + LRM (extends existing `lrm.rs`) |
| **AD-83** | MC variance swap | `mc_variance_swap` | RNG + log-returns | Pathwise AD; log-return is differentiable |
| **AD-84** | MC digital | `mc_digital` | RNG + indicator | LRM (already exists for 1-step; extend to multi-step) |
| **AD-85** | nth-to-default (MC) | `nth_to_default_mc` | RNG + copula + sorting | LRM for default ordering |
| **AD-86** | CDO (integral engine) | `integral_cdo_engine`, `midpoint_cdo_engine` | Loss model abstraction + copula integration | Generic copula + integration |
| **AD-87** | FD American | Various FD engines | FD grid | Generic 1D/2D FD solver |
| **AD-88** | Tree swaption/cap | `tree_swaption`, `tree_cap_floor` | Short-rate tree + exercise | Forward-mode through tree; IFT for exercise |
| **AD-89** | Gaussian1d engines | `gaussian1d_swaption`, `gaussian1d_cap_floor` | Short-rate model + numerical integral | Generic numerical integration (already in `ql-aad`) |
| **AD-90** | FD G2 swaption | `fd_g2_swaption` | 2D FD grid (2-factor) | Generic 2D FD |
| **AD-91** | LMM products (MC) | `lmm_product_mc` | High-dimensional MC + calibration | Pathwise AD through LMM SDE |
| **AD-92** | Variance Gamma (COS) | `vg_cos_price` | COS method with VG char. function | Replace inline complex with `Complex<T>` |
| **AD-93** | Commodity forward/swap | `price_commodity_forward`, `price_commodity_swap` | Commodity curves (f64) | Generic commodity curve |
| **AD-94** | Asset swap / equity TRS | `price_asset_swap` | Bond + swap + `&dyn YieldTermStructure` | Same as AD-49 |

### 12.3 Cross-cutting infrastructure gaps

Before the Tier 2 and Tier 3 gaps can be addressed, these shared infrastructure
pieces must be built:

| ID | Component | Status | Impact |
|----|-----------|--------|--------|
| **INFRA-1** | `cumulative_normal<T: Number>(x: T) -> T` | **Missing** from `ql-pricingengines` (exists as `ql_aad::math::normal_cdf` but not re-exported to engines) | Blocks AD-5 through AD-43 (39 gaps) |
| **INFRA-2** | `bivariate_normal_cdf<T: Number>(x: T, y: T, rho: f64) -> T` | **Missing** | Blocks AD-19, AD-23, AD-42, AD-48, AD-73 |
| **INFRA-3** | `GenericYieldTermStructure<T: Number>` (or adapter from `DiscountCurveAD<T>` to `YieldTermStructure` trait) | **Missing** in `ql-termstructures` | Blocks all Tier 3 discounting engines (AD-49 through AD-68) |
| **INFRA-4** | `GenericDefaultProbabilityTermStructure<T>` | **Missing** | Blocks AD-58, AD-59, AD-85, AD-86 |
| **INFRA-5** | `leg_npv<T: Number>()` — generic cashflow valuation in `ql-cashflows` | **Missing** (current `ql_cashflows::npv` is f64; `ql_aad::cashflows::npv<T>` exists but is not wired into `ql-cashflows`) | Blocks AD-49 through AD-68 |
| **INFRA-6** | `Complex<T: Number>` integration with `ql-pricingengines` | **Exists** in `ql-aad` but not used by `cos_heston` or other COS-method engines | Blocks AD-78, AD-92 |
| **INFRA-7** | Generic 1D FD grid / Crank-Nicolson step | **Missing** (all in `ql-methods` as f64) | Blocks AD-74, AD-75, AD-87 |
| **INFRA-8** | Generic 2D FD grid / ADI step | **Missing** | Blocks AD-77, AD-90 |
| **INFRA-9** | Generic short-rate tree builder | **Missing** | Blocks AD-56, AD-57, AD-88 |
| **INFRA-10** | IFT adjoint for iterative solvers (`solve_and_diff` integration with engines) | **Exists** in `ql-aad::solvers` but not wired into American engines or HW Jamshidian | Blocks AD-44 through AD-48 |

### 12.4 Recommended conversion roadmap

```
Phase A: Infrastructure (INFRA-1, INFRA-2)
  └─ Wire `ql_aad::math::normal_cdf` into `ql-pricingengines` via ql-core re-export
  └─ Implement generic bivariate normal CDF

Phase B: Tier 1 (AD-1 to AD-4)                          ~4 engines, ~400 LOC
  └─ Immediate wins, no blockers

Phase C: Tier 2 — IR derivatives (AD-6 to AD-9)          ~4 engines, ~400 LOC
  └─ Black/Bachelier swaptions & cap/floors — highest business value

Phase D: Tier 2 — Exotics (AD-10 to AD-43)              ~34 engines, ~3,000 LOC
  └─ All closed-form analytic exotic engines

Phase E: INFRA-3, INFRA-5 (generic curves + cashflows)
  └─ Build GenericYieldTermStructure<T> adapter

Phase F: Tier 3 — Discounting engines (AD-49 to AD-68)   ~20 engines, ~2,000 LOC
  └─ All swap/bond engines using generic curves

Phase G: Tier 3 — American + iterative (AD-44 to AD-48)  ~5 engines, ~500 LOC
  └─ Wire IFT adjoint solvers into American approximations

Phase H: Tier 3 — MC exotics (AD-69 to AD-85)            ~17 engines, ~3,000 LOC
  └─ Pathwise AD through MC SDEs (extends existing mc.rs / lrm.rs)

Phase I: INFRA-7, INFRA-8, INFRA-9 (generic FD/tree)
  └─ Build generic 1D/2D FD and tree infrastructure

Phase J: Tier 3 — FD/tree engines (AD-56, AD-57, AD-74, AD-75, AD-77, AD-87–AD-90)
  └─ All FD and tree engines using generic grid/tree

Phase K: Tier 3 — Remaining (AD-86, AD-91–AD-94)
  └─ Credit portfolio, LMM, commodity, asset swap
```

### 12.5 Coverage summary

| Category | Total instruments | AD-enabled | Gap count | Coverage |
|----------|------------------|------------|-----------|----------|
| European vanilla (equity) | 5 engines | 5 | 0 | **100%** |
| Exotic options (analytic) | ~20 engines | 0 | 20 | **0%** |
| Exotic options (MC/FD) | ~15 engines | 3 (barrier LRM, digital LRM, fwd-start) | 12 | **20%** |
| IR derivatives (linear) | ~12 instruments | 2 (bond/swap via cashflows) | 10 | **17%** |
| IR derivatives (non-linear) | ~8 instruments | 0 | 8 | **0%** |
| Credit | ~5 instruments | 0 | 5 | **0%** |
| FX / commodity | ~4 instruments | 0 | 4 | **0%** |
| Multi-asset | ~6 engines | 0 | 6 | **0%** |
| **Overall** | **~60+ instruments** | **~10** | **94** | **~14%** |

The `ql-aad` infrastructure is production-quality — the gap is in **lifting the
pricing engines** to be generic over `T: Number`. The biggest leverage points are:

1. **INFRA-1** (generic normal CDF) — unblocks 39 analytic engines in one shot
2. **AD-6/AD-8** (swaptions, cap/floors) — highest business value IR derivatives
3. **INFRA-3** (generic yield curves) — unblocks all 20 discounting engines
4. **AD-44–AD-46** (American approximations) — the most-requested missing Greeks

---

## 13. References

1. A. Savine, *Modern Computational Finance: AAD and Parallel Simulations*, Wiley, 2018
2. L. Capriotti, "Fast Greeks by Algorithmic Differentiation", *J. Computational Finance*, 2011
3. M. Henrard, *Algorithmic Differentiation in Finance Explained*, Palgrave, 2017
4. A. Griewank & A. Walther, *Evaluating Derivatives*, SIAM, 2008
5. M. Leclerc, Q. Liang, I. Schneider, "Fast Monte Carlo Bermudan Greeks", *Risk*, 2009
6. Savine & Huge, "Differential Machine Learning", *Risk*, 2020
7. B. Bell, "CppAD: A Package for C++ Algorithmic Differentiation", *Comp. Infrastructure for Operations Research*, 2012
8. Baydin et al., "Automatic Differentiation in Machine Learning: a Survey", *JMLR*, 2018
9. Giles & Glasserman, "Smoking Adjoints: fast computation of Greeks in Monte Carlo", *Risk*, 2006
10. Capriotti & Giles, "Algorithmic Differentiation: Adjoint Greeks Made Easy", *SSRN*, 2012
