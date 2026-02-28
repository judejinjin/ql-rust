# ql-rust vs QuantLib C++ — Gap Analysis

**Date:** 2026-02-28  
**ql-rust commit:** `3a27f8b` → current (1,927 tests passing)  
**QuantLib C++ reference:** `/mnt/c/cplusplus/quantlib/`

## Current Coverage Summary

| Metric | ql-rust | QuantLib C++ | Coverage |
|--------|---------|--------------|----------|
| Source files | 300+ `.rs` | ~1,312 `.hpp` | — |
| SLOC (approx) | ~95K | ~300K+ | — |
| Crates / modules | 14 crates | 16 directories | — |
| Instrument types | 40+ | ~80+ | ~50% |
| Pricing engines | 53+ modules | ~150+ | ~35% |
| Models | 17 | ~15 core + 125 LMM | ~80% core |
| Stochastic processes | 12 | 21 | ~57% |
| Term structures (yield) | 25+ | 26 | ~96% |
| Vol surfaces | 20+ | 40+ | ~50% |
| Calendars | 44 | 45 | ~98% |
| Day counters | 8 variants | 12 | ~67% |
| Optimization methods | 6 | 13 | ~46% |
| Math/interpolation | 15+ | 21 | ~71% |
| Tests | 1,904 | — | — |

**Overall estimated coverage: ~58%** of QuantLib C++ production features.

---

## HIGH Priority Gaps

| # | Gap | Description | Target Crate |
|---|-----|-------------|--------------|
| H1 | **Piecewise spread yield curve** | `PiecewiseZeroSpreadedTermStructure`, `PiecewiseForwardSpreadedTermStructure` — bootstrap a spread curve on top of a base curve | `ql-termstructures` |
| H2 | **Interpolated swaption vol cube** | Full `InterpolatedSwaptionVolatilityCube` — 3D interpolation over exercise × tenor × strike with per-expiry SABR calibration | `ql-termstructures` |
| H3 | **FD G2++ solver & swaption engine** | `FdG2SwaptionEngine` — finite-difference pricing of Bermudan swaptions under the two-factor G2++ model | `ql-pricingengines` |
| H4 | **Missing day counters** | `Actual366`, `Actual364`, `Actual365_25`, `SimpleDayCounter`, `One` — used in Nordic, commodity, and regulatory contexts | `ql-time` |

---

## MEDIUM Priority Gaps

### Engines & Instruments

| # | Gap | Description | Target Crate |
|---|-----|-------------|--------------|
| M1 | **Soft barrier option** | Barrier with smooth payoff transition (exponential softening) | `ql-pricingengines` |
| M7 | **AnalyticBSMHullWhiteEngine** | Hybrid BS + Hull-White for equity options with stochastic rates | `ql-pricingengines` |
| M12 | **Gaussian1d nonstandard swaption** | `Gaussian1dNonstandardSwaptionEngine` for amortizing/step-up Bermudan swaptions | `ql-pricingengines` |
| M13 | **MC Heston–Hull-White** | Monte Carlo pricing of equity options under Heston + stochastic rates | `ql-pricingengines` |

### Term Structures & Volatility

| # | Gap | Description | Target Crate |
|---|-----|-------------|--------------|
| M16 | **Caplet variance curve** | `CapletVarianceCurve` — piecewise-constant caplet implied vols | `ql-termstructures` |
| M17 | **Spread swaption vol** | `SpreadedSwaptionVolatility` — shift an existing swaption vol surface by a spread | `ql-termstructures` |

### Math & Numerical Methods

| # | Gap | Description | Target Crate |
|---|-----|-------------|--------------|
| M27 | **B-splines** | B-spline basis functions (Cox-de Boor) for curve/surface fitting | `ql-math` |
| M28 | **Chebyshev interpolation** | Chebyshev polynomial interpolation with Clenshaw evaluation | `ql-math` |
| M30 | **Richardson extrapolation** | Convergence acceleration for numerical methods | `ql-math` |
| M31 | **Brownian bridge** | Path construction via Brownian bridge for improved QMC convergence | `ql-math` |
| M34 | **Halton sequence** | Alternative quasi-random number generator | `ql-math` |

### Cash Flows

| # | Gap | Description | Target Crate |
|---|-----|-------------|--------------|
| M44 | **Multiple-resets coupon** | Compounded RFR coupon with multiple intra-period fixings | `ql-cashflows` |
| M45 | **Indexed cash flow** | Cash flow linked to an arbitrary index fixing | `ql-cashflows` |

---

## LOW Priority Gaps

| # | Gap | Description | Status |
|---|-----|-------------|--------|
| L1 | CAT bonds | Catastrophe bonds with event-triggered losses | ✅ `ql-instruments/src/cat_bond.rs` |
| L3 | Himalaya/Everest/Pagoda options | Mountain-range multi-asset exotics | ✅ `ql-pricingengines/src/mountain_range.rs` |
| L7 | FX `BlackDeltaCalculator` | FX delta conventions (spot, forward, premium-adjusted) | ✅ `ql-math/src/black_delta.rs` |
| L10 | 2 missing copulas | Ali-Mikhail-Haq, Husler-Reiss (6 others already existed) | ✅ `ql-math/src/copulas.rs` |
| L13 | Exp-sinh quadrature | Doubly-exponential quadrature for semi-infinite integrals | ✅ `ql-math/src/integration_advanced.rs` |
| L20 | Sparse matrix / ILU preconditioner | CSR storage + ILU(0) factorization | ✅ `ql-math/src/sparse.rs` |
| L21 | BiCGStab / GMRES solvers | Iterative sparse linear system solvers | ✅ `ql-math/src/sparse.rs` |
| L23 | SVD decomposition | Singular value decomposition convenience wrapper | ✅ `ql-math/src/matrix.rs` |
| L24 | Adaptive Runge-Kutta ODE solver | RK4 + Dormand-Prince RK45 adaptive | ✅ `ql-math/src/ode.rs` |
| L25 | Lattice2d / trinomial trees | 2-factor trinomial lattice with correlation | ✅ `ql-methods/src/lattice_2d.rs` |
| L31 | Actual/366, Actual/364, One | Specialized day counter variants (done in H4) | ✅ done |
| L32–L37 | Composite/derived/delta quotes | Extended quote types | ✅ already existed |
| L38–L41 | Missing indexes/currencies | ~70 IBOR indexes, 35 currencies, ExchangeRateManager | ✅ `ql-currencies`, `ql-indexes` |

---

## ql-rust Advantages (features QuantLib C++ lacks)

| # | Feature | Description |
|---|---------|-------------|
| R1 | SecDB/Beacon persistence | Full `ql-persistence` crate with ACID trade store |
| R2 | Native Python bindings | PyO3-based `ql-python` crate (QuantLib uses SWIG) |
| R3 | CLI tool | `ql-cli` command-line pricing |
| R4 | Reactive portfolio | `ReactivePortfolio`, `FeedDrivenQuote`, streaming architecture |
| R5 | Portfolio analytics | Equity risk ladders, scenario analysis, curve sensitivities |
| R6 | Global bootstrap | Simultaneous multi-instrument curve fitting |
| R7 | Cross-currency basis bootstrap | Dedicated xccy and tenor basis helpers |
| R8 | Smith-Wilson UFR curve | Solvency II ultimate forward rate |
| R9 | Serde serialization | All instruments/curves are `Serialize`/`Deserialize` |
| R10 | Thread-safe design | `Send + Sync`, `rayon` parallelism for MC/FD |
| R11 | MC control variates | Geometric Asian closed-form CV |

---

## Implementation Plan

**Phase 1 — HIGH priority (H1–H4):** 4 items ✅ done  
**Phase 2 — MEDIUM engines/instruments (M1, M7, M12, M13):** 4 items ✅ done  
**Phase 3 — MEDIUM term structures (M16, M17):** 2 items ✅ done  
**Phase 4 — MEDIUM math (M27, M28, M30, M31, M34):** 5 items ✅ done  
**Phase 5 — MEDIUM cashflows (M44, M45):** 2 items ✅ done  
**Phase 6 — LOW priority (L1–L25):** 10 items ✅ done  
**Phase 7 — LOW indexes/currencies (L38–L41):** 4 items ✅ done  

**Total: 31 modules implemented** across 4 sessions.  
**Remaining:** none — all identified gaps closed.
