# Changelog

All notable changes to the ql-rust project are documented in this file.

## [0.3.1] — 2025-06-19

### Code Quality: Clippy-Clean Workspace + Further Deduplication

#### Clippy audit — zero warnings with `-D warnings`
- Fixed 240+ clippy warnings across all 17 crates
- Categories resolved: float precision (106), assignment ops (16), boolean simplification (15), loop indexing (30+), manual clamp (10+), deprecated `gen()` → `random()`, dead code, doc formatting, type complexity, manual slice copy, too-many-arguments
- Strategy: auto-fix pass first (`cargo clippy --fix`), then manual fixes for remaining issues
- Numerical code loops preserved with `#[allow(clippy::needless_range_loop)]` where indexed form is clearer

#### Additional f64 → generic deduplication
- `margrabe_exchange` now delegates to `margrabe_exchange_generic::<f64>`
- `bs_vega` now delegates to `bs_vega_generic::<f64>`
- `sabr_volatility` now delegates to `sabr_vol_generic::<f64>` (keeps input assertions)
- Total deduplicated: 9 functions (6 prior + 3 new)

## [0.3.0] — 2026-03-03

### AAD-Complete: 94 Generic Engines + Full AD Integration

#### Generic `T: Number` pricing engines (INFRA-1 through INFRA-10)
- `ql_pricingengines::generic` — 100+ generic engines parameterised over `T: Number`, covering all 94 AD gaps (AD-1 through AD-94) from the AAD proposal
- Engines span: Black-Scholes, Black-76, Bachelier, BAW American, Merton jump-diffusion, chooser, Kirk spread, Margrabe exchange, lookback, Asian geometric, variance swap, quanto, CDS midpoint, compound, power, Stulz max-call, HW bond option, forward-start, and many more
- `ql_math::generic` — AD-compatible `normal_cdf`, `normal_pdf`, `bivariate_normal_cdf`, `black_scholes_generic`, `discount_factor`
- `ql_termstructures::generic` — `GenericYieldCurve<T>` trait, `FlatCurve<T>`, `InterpDiscountCurve<T>`, `InterpZeroCurve<T>`, `nelson_siegel_discount`, `svensson_discount`
- `ql_methods::generic` — generic finite-difference and tree methods

#### f64 → generic deduplication
- `price_european` and `black_scholes_price` now delegate to `bs_european_generic::<f64>`
- `barone_adesi_whaley` delegates to `barone_adesi_whaley_generic::<f64>`
- `merton_jump_diffusion` delegates to `merton_jd_generic::<f64>`
- `chooser_price` delegates to `chooser_generic::<f64>`
- `binomial_crr` delegates to `binomial_crr_generic::<f64>`
- Eliminates code duplication; bug fixes in generic engines benefit both paths

#### AD integration tests (33 tests)
- Forward-mode `Dual` tests: BS delta/vega, Black-76, Bachelier, BAW, caplet, Kirk spread, Margrabe, Asian geometric, lookback, HW bond option, power option, Stulz max-call, forward-start, CDS midpoint yield sensitivity
- Multi-seed `DualVec<N>` tests: BS all-Greeks-in-one-pass (5 partials), Black swaption, Merton jump-diffusion, variance swap
- Reverse-mode `AReal` tests: BS all-Greeks via tape adjoint, chooser, bond PV rate sensitivity, swap PV discount sensitivity, Nelson-Siegel curve parameter sensitivity
- Higher-order Greeks (FD-on-AD): gamma, vanna, volga, charm — validated against BS closed-form and pure second-order FD
- BAW gamma FD-on-AD vs pure FD; DualVec & AReal gamma consistency
- All derivatives validated against central finite differences (bump=1e-5, tol≤1e-3)
- `f64` parity test: `bs_european_generic::<f64>` matches known BS analytical values

#### AD benchmarks (Criterion)
- `ad_bs_european`: f64 vs Dual (1.7×) vs DualVec<5> (3.7×) vs AReal (35×)
- `ad_baw_american`: f64 vs Dual (1.5×) vs DualVec<5> (2.4×) vs AReal (26×)
- `ad_merton_jd`: f64 vs Dual (1.8×) vs DualVec<5> (3.3×) vs AReal (32×)
- `ad_chooser`: f64 vs Dual (1.7×) vs AReal (50×)

#### Facade re-exports
- `ql_rust::generic` module re-exports all generic engines, math, term structures, and methods
- `ql_rust::Number` re-exports the core `Number` trait

## [0.2.1] — 2026-02-23

### Phase 25: Reactive Pricing Infrastructure

#### Observer / LazyObject wiring (item 4)
- `HasNpv` trait — extract a scalar NPV from any pricing result struct
- `NpvProvider` trait (`Observer + Observable + Send + Sync`) — reactive lazy instrument abstraction
- Blanket `impl NpvProvider for LazyInstrument<I, R> where R: HasNpv`
- `ReactivePortfolio` — aggregates `NpvProvider` entries; lazy-cached `total_npv()` with `AtomicBool` valid flag; self-invalidates and notifies downstream on any entry change
- `wire_entry()` free function — adds entry to portfolio and registers portfolio as its observer in one call
- 8 unit tests in `ql-core::portfolio`

#### Real-time market data trait (item 5)
- `MarketDataFeed` trait (dyn-compatible) — `subscribe(&str, …)`, `unsubscribe`, `publish`, `active_tickers`, `subscription_count`
- `FeedEvent` struct with `new`, `with_bbo`, `full` constructors
- `FeedField` enum — `Bid`, `Ask`, `Mid` (default), `Last`
- `SubscriptionId` opaque newtype
- `FeedCallback = Arc<dyn Fn(FeedEvent) + Send + Sync>`
- `InMemoryFeed` — thread-safe mock feed (RwLock-protected subscriber map)
- `FeedDrivenQuote` — bridges feed ticks into `SimpleQuote`; auto-unsubscribes on `Drop`
- 14 unit tests + doctests in `ql-core::market_data`

#### Integration
- `ql-rust` facade re-exports all new types (`MarketDataFeed`, `InMemoryFeed`, `FeedDrivenQuote`, `FeedEvent`, `FeedField`, `SubscriptionId`, `FeedCallback`, `HasNpv`, `NpvProvider`, `ReactivePortfolio`, `wire_entry`, `Observable`, `Observer`)
- `reactive_pricing` example: `SimpleQuote → LazyInstrument → ReactivePortfolio` + `InMemoryFeed → FeedDrivenQuote → instrument`
- 15 integration tests in `ql-rust::test_reactive_integration` (full observer chain, feed → portfolio, 3-level chain, Drop cleanup)
- 6 new Criterion benchmarks: lazy cache-hit, lazy cache-miss, portfolio cached, portfolio after invalidation, feed publish 1 subscriber, feed publish 10 subscribers

## [0.2.0] — 2026-02-18

### Phase 13: Advanced American Option Engines
- Barone-Adesi-Whaley (BAW) analytic approximation
- Bjerksund-Stensland 2002 two-point boundary
- QD+ high-precision American pricing (iteration until convergence)
- `AmericanApproxResult` with NPV, early-exercise premium, critical price

### Phase 14: Bates Model & Jump-Diffusion
- `BatesModel` (Heston + Merton jumps: λ, μ_J, σ_J)
- `BatesProcess` stochastic process
- `bates_price` / `bates_price_flat` Fourier integration
- `merton_jump_diffusion` closed-form pricing
- `mc_bates` Monte Carlo for Bates SDE

### Phase 15: Multi-Asset & Basket Options
- Stulz max/min-on-two-assets closed-form
- Kirk's spread option approximation
- Margrabe exchange option formula
- `mc_basket` Monte Carlo for N-asset baskets (max, min, spread)
- Cholesky-correlated multi-asset path generation

### Phase 16: Short-Rate Models — Full Coverage
- `VasicekModel`, `CIRModel` with analytic bond pricing
- `HullWhiteModel` with calibration, mean-reversion, term structure fitting
- `BlackKarasinskiModel` with lognormal short rate
- `G2Model` (two-factor Gaussian) with analytic swaption pricing
- `OrnsteinUhlenbeckProcess`, `HullWhiteProcess`, `CoxIngersollRossProcess`

### Phase 17: Advanced Swaption & Cap/Floor Engines
- Hull-White analytic bond options, caplets, floorlets
- Jamshidian swaption decomposition
- Trinomial tree pricing (bonds, swaptions, Bermudans, caps/floors)
- Finite-difference Hull-White swaption solver
- Monte Carlo Hull-White cap/floor pricing

### Phase 18: Advanced Volatility Surfaces & Smile Models
- SVI (Stochastic Volatility Inspired) smile parameterisation with 5-param calibration
- ZABR extension of SABR (variable γ backbone)
- Optionlet stripping from cap volatilities
- `BlackVarianceCurve` and `SmileSectionSurface` interpolation

### Phase 19: Advanced Finite Difference Framework
- `Mesher1d`, `FdmMesherComposite` mesh generation
- Concentrating and log-spot meshers for non-uniform grids
- `TripleBandOp`, `Heston2dOps` operator discretisation
- Crank-Nicolson, implicit, and Douglas ADI time stepping
- 1D BS FD solver and 2D Heston FD solver with American exercise

### Phase 20: LIBOR Market Model Framework
- `LmmConfig` with forward rate and volatility structure
- `LmmCurveState` for forward rate evolution
- `lmm_cap_price` — caplet pricing via log-normal displaced diffusion
- `lmm_swaption_price` — Bermudan/European swaption via MC

### Phase 21: Advanced Credit Models
- `GaussianCopulaLHP` — large homogeneous portfolio CDO tranche pricing
- `nth_to_default_mc` — Monte Carlo N-th to default basket
- `cds_option_black` — CDS option pricing via Black's formula
- ISDA standard CDS model coupon dates

### Phase 22: Math Library Extensions
- **Copulas:** Gaussian, Clayton, Frank, Gumbel with Kendall's τ conversion
- **Statistics:** General, Incremental (Welford), Risk (VaR/CVaR), Convergence
- **Distributions:** Student's t, Gamma, Binomial, bivariate normal CDF
- **Interpolation:** backward/forward flat, bilinear 2D, bicubic Catmull-Rom 2D
- **Quasi-random:** Halton (40-dim), Sobol (21-dim), Brownian bridge
- **FFT:** Radix-2 Cooley-Tukey, Carr-Madan option pricing
- **Solvers:** Secant, Ridder, false-position (Illinois)

### Phase 23: Advanced Cash Flows & Coupons
- `CmsCoupon` with linear TSR convexity adjustment and Black caplet pricing
- `DigitalCoupon` (binary payoff), `CapFlooredCoupon` (cap/floor/collar)
- `RangeAccrualCoupon` (fraction of days in range)
- `SubPeriodCoupon` (compounding and averaging sub-periods)
- Extended analytics: convexity, modified duration, DV01, Z-spread, ATM rate
- Time-bucketed cash flow analysis

### Phase 24: Advanced Yield Curve & Fitting
- `NelsonSiegelFitting` (4-parameter) with Nelder-Mead optimisation
- `SvenssonFitting` (6-parameter) extension
- `FittedBondDiscountCurve` as `YieldTermStructure`
- `CompositeZeroYieldStructure` (additive combination of two curves)
- `ImpliedTermStructure` (forward-starting derived curve)
- `ForwardCurve` (interpolated instantaneous forward rates)
- `UltimateForwardTermStructure` (Smith-Wilson / Solvency II UFR extrapolation)
- `SpreadedTermStructure` (base + constant zero-rate spread)
- `OISRateHelper`, `BondHelper`, `FuturesRateHelper`, `FRAHelper`

### Quality & Cross-Cutting
- Updated facade crate (ql-rust) with all Phase 13-24 re-exports
- New integration tests: American pricing pipeline, multi-asset pipeline,
  short-rate model pipeline, yield curve fitting pipeline, advanced cashflows pipeline
- New benchmarks: American approximation, Nelson-Siegel fit, FD Heston 2D, MC basket,
  Vasicek bond, G2 swaption, FFT, Cholesky, CMS caplet, LMM cap, CDO tranche, CDS option
- Golden cross-validation tests for American, spread, Nelson-Siegel, short-rate,
  FD advanced, credit models, LMM, CMS, and advanced curves (65 total)
- Heston/Bates quadrature optimised (Gauss-Legendre 128-point, replacing adaptive)
- LSM regression flat-array optimisation (cache-friendly row-major layout)
- Calendar Easter-Monday cache and interpolation small-array linear scan
- 8 fuzz targets (BS pricing, date, interpolation, SABR, schedule,
  serde instruments, serde term structures, Heston/Bates)
- 963 tests total, zero clippy warnings

## [0.1.1] — 2026-02-17

### Quality — Definition of Done Completion
- Removed all 67 `unwrap()`/`expect()` calls from production code (20 files)
- Added `///` doc comments to 31 remaining public functions
- 13 golden cross-validation tests (calendar, yield curve, Black-Scholes)
- Heston calibration integration test (synthetic vol surface, params ± 1e-4)
- 4 new benchmarks: Heston analytic, Heston calibration, calendar advance, interpolation
- 3 standalone examples: `price_european_option`, `bootstrap_yield_curve`, `calibrate_heston`
- Updated README and CHANGELOG

### Test Summary
- 542+ tests total (502 unit + 16 integration + 11 property + 13 golden)
- Zero clippy warnings, zero doc warnings

## [0.1.0] — 2025-07-19

### Phase 0 — Workspace & Core Foundation
- Created Cargo workspace with `resolver = "2"`
- `ql-core`: `QLError` enum (math, time, instrument, term-structure, persistence variants), `QLResult<T>` alias, `SimpleQuote`

### Phase 1 — Date & Time Framework (`ql-time`)
- `Date` newtype (serial-number based, epoch 1 Jan 1900)
- `Calendar` with US/UK/TARGET holiday calendars
- `DayCounter`: Actual/360, Actual/365 Fixed, 30/360, Actual/Actual
- `Schedule` builder with forward/backward generation
- `BusinessDayConvention` adjustments (Following, ModifiedFollowing, Preceding, etc.)
- `Period`, `Month`, `Weekday`, `Frequency`, `TimeUnit`

### Phase 2 — Mathematics Toolkit (`ql-math`)
- Interpolation: linear, log-linear, cubic spline (natural & clamped)
- Root-finding: Brent, Newton, bisection, secant
- 1-D integration: Simpson, trapezoidal, Gauss-Legendre
- Optimization: Levenberg-Marquardt
- Matrix operations via nalgebra: Cholesky, SVD, eigenvalue
- Normal/cumulative-normal distribution functions

### Phase 3 — Currencies & Indexes
- `ql-currencies`: 30+ ISO 4217 currencies (USD, EUR, GBP, JPY, CHF, etc.)
- `ql-indexes`: `IborIndex` with factory methods (`euribor_3m`, `euribor_6m`, `usd_libor_3m`)
- `InterestRate` with `Compounding` enum (Simple, Compounded, Continuous)
- Rate conversion and equivalence

### Phase 4 — Term Structures (`ql-termstructures`)
- `YieldTermStructure` trait with `discount`, `discount_t`, `zero_rate`, `forward_rate`
- `FlatForward` constant-rate curve
- `PiecewiseYieldCurve` bootstrapping from deposit + swap helpers
- `DepositRateHelper`, `SwapRateHelper` implementing `RateHelper` trait
- `ZeroCurve`, `DiscountCurve` (node-based)

### Phase 5 — Cash Flows & Legs (`ql-cashflows`)
- `CashFlow` trait (date, amount, has_occurred)
- `FixedRateCoupon`, `FloatingRateCoupon`
- `fixed_leg()`, `ibor_leg()` builders
- Analytics: `npv()`, `bps()`, `accrued_amount()`, `duration()`

### Phase 6 — Instruments (`ql-instruments`)
- `VanillaOption` (European/American call/put)
- `VanillaSwap` with `from_schedules()` constructor
- `FixedRateBond`
- `BarrierOption` (Up/Down × In/Out)
- `Swaption`, `CapFloor`
- `CreditDefaultSwap`

### Phase 7 — Stochastic Processes & Models
- `ql-processes`: `GeneralizedBlackScholesProcess`, `HestonProcess`
- `ql-models`: `HestonModel` with 5-parameter calibration

### Phase 8 — Analytic Pricing Engines (`ql-pricingengines`)
- Black-Scholes European pricing with all Greeks (delta, gamma, vega, theta, rho)
- `implied_volatility()` via Newton's method
- Heston semi-analytic pricing via Fourier integration
- `price_swap()`, `price_bond()` with full result structs
- Black and Bachelier swaption/cap-floor pricing
- CDS midpoint engine
- Callable bond, convertible bond, lookback, compound option, variance swap engines

### Phase 9 — Volatility Surfaces
- `BlackConstantVol`, `BlackVarianceSurface` (strike × expiry grid)
- `LocalVolSurface` (Dupire's formula)
- `SabrSmileSection` with SABR volatility formula
- `sabr_volatility()` helper function

### Phase 10 — Numerical Methods (`ql-methods`)
- Monte Carlo: `mc_european`, `mc_barrier`, `mc_asian`, `mc_heston`
  - Parallel path generation via Rayon
  - Antithetic variates for variance reduction
- Finite Differences: `fd_black_scholes` (Crank-Nicolson scheme)
  - European and American exercise
- Lattice: `binomial_crr` (Cox-Ross-Rubinstein)
  - European and American exercise
  - Delta, gamma, theta from lattice

### Phase 11 — Persistence (`ql-persistence`)
- `Trade` struct with full trade lifecycle fields
- `ObjectStore` trait with `put_trade`, `get_trade`, `query_trades`
- `EmbeddedStore` backed by redb (embedded key-value database)
- `InMemoryStore` for testing
- `LifecycleEvent` and event sourcing (`append_event`, `replay_events`)
- `TradeFilter` with counterparty/book/instrument-type/status filtering
- Versioning support

### Phase 12 — CLI & Exotic Extensions (`ql-cli`)
- Clap-based CLI with subcommands: `price`, `curve`, `trade`, `list`, `risk`
- Exotic instruments: `AsianOption`, `LookbackOption`, `CompoundOption`, `VarianceSwap`
- `CallableBond`, `ConvertibleBond`
- Zero-inflation term structures and swaps

### Quality & Hardening
- 529 unit/integration/property tests at time of release
- Integration test suites: yield curve pipeline, swap pricing pipeline, option pricing pipeline, persistence round-trip
- Property-based tests (proptest): put-call parity, price bounds, Greeks ranges, discount factor monotonicity, American ≥ European
- 14 Criterion benchmarks covering all major operations
- Zero clippy warnings
- Comprehensive README with architecture diagram, code examples, and CLI usage
