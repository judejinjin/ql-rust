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

## Remaining Gaps — Full QuantLib Parity

The following gaps were identified by exhaustive comparison of every `.hpp` file
in QuantLib C++ against ql-rust. Items marked with ✅ are closed; items marked
with — are open. C++ infrastructure patterns (base classes, CRTP traits, builder
helpers, `Discretized*` classes) are excluded as they are not applicable to Rust's
trait-based architecture.

### Instruments (G1–G9)

| # | Gap | Description | Target Crate |
|---|-----|-------------|--------------|
| G1 | YearOnYearInflationSwap | YoY inflation swap (separate from CPI swap) | `ql-instruments` |
| G2 | ZeroCouponInflationSwap | Zero-coupon inflation swap | `ql-instruments` |
| G3 | FloatFloatSwaption | Swaption on float-float swap | `ql-instruments` |
| G4 | CmsRateBond | Bond paying CMS-linked coupons | `ql-instruments` |
| G5 | AmortizingCmsRateBond | Amortizing bond with CMS coupons | `ql-instruments` |
| G6 | BTP | Italian government bond (CCT, BTP, BTPS, BOT) | `ql-instruments` |
| G7 | DividendVanillaOption | Vanilla option with discrete cash/proportional dividends | `ql-instruments` |
| G8 | DividendBarrierOption | Barrier option with discrete dividends | `ql-instruments` |
| G9 | StickyRatchet | Sticky/ratchet structured rates coupon | `ql-instruments` |

### Pricing Engines (G10–G21)

| # | Gap | Description | Target Crate |
|---|-----|-------------|--------------|
| G10 | AnalyticDividendEuropeanEngine | European option with discrete dividends (Escrowed model) | `ql-pricingengines` |
| G11 | G2SwaptionEngine (analytic) | Analytic G2++ swaption pricing (Jamshidian-like decomposition) | `ql-pricingengines` |
| G12 | TreeSwapEngine | Swap pricing on short-rate lattice | `ql-pricingengines` |
| G13 | AnalyticPerformanceEngine | Analytic cliquet performance option | `ql-pricingengines` |
| G14 | MCPerformanceEngine | MC cliquet performance option | `ql-pricingengines` |
| G15 | AnalyticBinaryBarrierEngine | Binary barrier (cash-or-nothing at barrier) | `ql-pricingengines` |
| G16 | AnalyticDoubleBarrierBinaryEngine | Double-barrier binary options | `ql-pricingengines` |
| G17 | ForwardPerformanceEngine | Forward-start performance option | `ql-pricingengines` |
| G18 | BondFunctions | Clean/dirty price, yield, duration, convexity, BPS utilities | `ql-pricingengines` |
| G19 | BlackCalculator | Reusable Black-76 / BS formula calculator struct | `ql-pricingengines` |
| G20 | InflationCapFloorEngine | Pricing engine for YoY and zero-coupon inflation caps/floors | `ql-pricingengines` |
| G21 | FdHestonDoubleBarrierEngine | FD Heston double-barrier pricing | `ql-pricingengines` |

### Stochastic Processes (G22–G26)

| # | Gap | Description | Target Crate |
|---|-----|-------------|--------------|
| G22 | GSRProcess | Gaussian Short Rate process (time-dependent mean reversion) | `ql-processes` |
| G23 | MarkovFunctionalStateProcess | State process for Markov-functional model | `ql-processes` |
| G24 | SquareRootProcess | Generic CIR-style square-root diffusion | `ql-processes` |
| G25 | ForwardMeasureProcess | Process under forward measure (T-forward numeraire) | `ql-processes` |
| G26 | EulerDiscretization | Standalone Euler / End-Euler / Predictor-Corrector discretization | `ql-processes` |

### Math — Solvers, Integration, Interpolation (G27–G41)

| # | Gap | Description | Target Crate |
|---|-----|-------------|--------------|
| G27 | Halley solver | Cubic-convergence root finder using second derivatives | `ql-math` |
| G28 | NewtonSafe solver | Newton with bisection fallback for robustness | `ql-math` |
| G29 | TrapezoidIntegral | Basic trapezoidal quadrature | `ql-math` |
| G30 | SegmentIntegral | Mid-point rule integration | `ql-math` |
| G31 | TwoDimensionalIntegral | Iterated 1D integration over 2D domain | `ql-math` |
| G32 | FilonIntegral | Filon quadrature for highly oscillatory integrands | `ql-math` |
| G33 | GaussianOrthogonalPolynomial | Gauss-Laguerre, Gauss-Hermite, Gauss-Jacobi, Gauss-Hyperbolic | `ql-math` |
| G34 | XABRInterpolation | Generalized XABR interpolation framework | `ql-math` |
| G35 | KernelInterpolation | Kernel (RBF) interpolation 1D and 2D | `ql-math` |
| G36 | MultiCubicSpline | N-dimensional cubic spline | `ql-math` |
| G37 | ABCDInterpolation | ABCD-parametric interpolation | `ql-math` |
| G38 | BackwardFlatLinearInterpolation | 2D interpolation: backward-flat × linear | `ql-math` |
| G39 | CraigSneydScheme | Craig-Sneyd ADI scheme (non-modified variant) | `ql-math` |
| G40 | TRBDF2Scheme | TR-BDF2 time-stepping for stiff FD problems | `ql-math` |
| G41 | MethodOfLinesScheme | Method of Lines for FD (spatial discretization only) | `ql-math` |

### Math — Linear Algebra & Special Functions (G42–G53)

| # | Gap | Description | Target Crate |
|---|-----|-------------|--------------|
| G42 | QR decomposition | Householder QR factorization | `ql-math` |
| G43 | PseudoSqrt | Pseudo square root of correlation/covariance matrices | `ql-math` |
| G44 | MatrixExponential | Matrix exponential via Padé approximation | `ql-math` |
| G45 | TQR eigendecomposition | Tridiagonal QR for symmetric eigenproblems | `ql-math` |
| G46 | Factorial | Factorial, log-factorial, double factorial | `ql-math` |
| G47 | IncompleteGamma | Upper/lower incomplete gamma function | `ql-math` |
| G48 | ModifiedBessel | Modified Bessel functions I₀, I₁, K₀, K₁ | `ql-math` |
| G49 | ErrorFunction | erf / erfc / inverseErf (standalone, AD-ready) | `ql-math` |
| G50 | BernsteinPolynomial | Bernstein basis polynomials | `ql-math` |
| G51 | Rounding | Up/down/closest/floor/ceiling rounding with precision | `ql-math` |
| G52 | GeneralLinearLeastSquares | Weighted least squares with arbitrary basis functions | `ql-math` |
| G53 | AutoCovariance | Autocovariance / autocorrelation of time series | `ql-math` |

### Math — Random Numbers & Statistics (G54–G60)

| # | Gap | Description | Target Crate |
|---|-----|-------------|--------------|
| G54 | BoxMullerGaussianRng | Box-Muller normal variate generator | `ql-math` |
| G55 | InverseCumulativeRng | Inverse-CDF normal/generic variate generator | `ql-math` |
| G56 | SobolBrownianBridgeRsg | Sobol + Brownian bridge for QMC path generation | `ql-math` |
| G57 | BivariateStudentT | Bivariate Student-t distribution | `ql-math` |
| G58 | DiscrepancyStatistics | Star-discrepancy measure for QMC sequences | `ql-math` |
| G59 | Histogram | Histogram with automatic/fixed binning | `ql-math` |
| G60 | NumericalDifferentiation | Central/forward/backward finite-difference derivatives | `ql-math` |

### Term Structures (G61–G75)

| # | Gap | Description | Target Crate |
|---|-----|-------------|--------------|
| G61 | LocalConstantVol | Constant local volatility surface | `ql-termstructures` |
| G62 | FixedLocalVolSurface | Pre-computed local vol on a fixed grid | `ql-termstructures` |
| G63 | GridModelLocalVolSurface | Local vol from a calibrated model on a grid | `ql-termstructures` |
| G64 | ImpliedVolTermStructure | Implied vol term structure (shifted reference) | `ql-termstructures` |
| G65 | DefaultDensityStructure | Default density term structure base | `ql-termstructures` |
| G66 | HazardRateStructure | Hazard rate term structure base | `ql-termstructures` |
| G67 | InterpolatedHazardRateCurve | Interpolated hazard rate curve | `ql-termstructures` |
| G68 | InterpolatedSurvivalProbCurve | Interpolated survival probability curve | `ql-termstructures` |
| G69 | CapFloorTermVolCurve | At-the-money cap/floor vol curve (1D) | `ql-termstructures` |
| G70 | OptionletVolatilityStructure | Optionlet vol surface base + constant variant | `ql-termstructures` |
| G71 | SpreadedOptionletVol | Optionlet vol surface with additive spread | `ql-termstructures` |
| G72 | InterpolatedZeroInflationCurve | Interpolated zero-coupon inflation curve | `ql-termstructures` |
| G73 | PiecewiseYoYInflationCurve | Bootstrapped YoY inflation curve | `ql-termstructures` |
| G74 | Gaussian1dSwaptionVol | Swaption vol from Gaussian 1D model | `ql-termstructures` |
| G75 | VolatilityType | ShiftedLognormal / Normal vol type enum | `ql-termstructures` |

### Cash Flows (G76–G83)

| # | Gap | Description | Target Crate |
|---|-----|-------------|--------------|
| G76 | YoYInflationCoupon | Year-on-year inflation-linked coupon | `ql-cashflows` |
| G77 | ZeroInflationCashFlow | Zero-coupon inflation cash flow (CPI ratio × notional) | `ql-cashflows` |
| G78 | EquityCashFlow | Cash flow linked to equity index performance | `ql-cashflows` |
| G79 | DigitalIborCoupon | IBOR coupon with digital (binary) feature | `ql-cashflows` |
| G80 | DigitalCmsCoupon | CMS coupon with digital (binary) feature | `ql-cashflows` |
| G81 | CapFlooredInflationCoupon | Capped/floored inflation-linked coupon | `ql-cashflows` |
| G82 | CpiCouponPricer | Pricer for CPI-linked coupons (convexity adjustment) | `ql-cashflows` |
| G83 | RateAveraging | Compound vs simple averaging enum + logic | `ql-cashflows` |

### Models (G84–G89)

| # | Gap | Description | Target Crate |
|---|-----|-------------|--------------|
| G84 | HestonSLVFdmModel | Heston stochastic local vol via FDM calibration | `ql-models` |
| G85 | HestonSLVMCModel | Heston SLV via MC simulation | `ql-models` |
| G86 | ExtendedCoxIngersollRoss | Extended CIR with time-dependent parameters | `ql-models` |
| G87 | CapHelper | Calibration helper for caps/floors | `ql-models` |
| G88 | SwaptionHelper | Calibration helper for swaptions (Black/Bachelier) | `ql-models` |
| G89 | ConstantVolEstimator | Constant (historical) volatility estimator | `ql-models` |

### Time (G90–G92)

| # | Gap | Description | Target Crate |
|---|-----|-------------|--------------|
| G90 | ASX dates | Australian Securities Exchange quarterly dates | `ql-time` |
| G91 | ECB dates | European Central Bank reserve maintenance dates | `ql-time` |
| G92 | Thirty365 day counter | 30/365 day count convention | `ql-time` |

### FD Infrastructure (G93–G97)

| # | Gap | Description | Target Crate |
|---|-----|-------------|--------------|
| G93 | FdmDirichletBoundary | Dirichlet boundary conditions for FD grids | `ql-methods` |
| G94 | FdmBermudanStepCondition | Bermudan exercise step condition for FD | `ql-methods` |
| G95 | Fdm3DimSolver | Three-dimensional FD solver | `ql-methods` |
| G96 | FdmCEVOp / FdmCIROp | CEV and CIR spatial operators | `ql-methods` |
| G97 | FdmHullWhiteOp / FdmG2Op | Hull-White and G2++ spatial operators | `ql-methods` |

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

**Phase 8 — Full parity (G1–G97):** 97 items — in progress  

**Total previously implemented: 31 modules** across 4 sessions.  
**Remaining: 97 items** for 100% QuantLib parity.
