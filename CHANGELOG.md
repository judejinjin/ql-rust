# Changelog

All notable changes to the ql-rust project are documented in this file.

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
- 529 tests total (502 unit + 16 integration + 11 property-based)
- Integration test suites: yield curve pipeline, swap pricing pipeline, option pricing pipeline, persistence round-trip
- Property-based tests (proptest): put-call parity, price bounds, Greeks ranges, discount factor monotonicity, American ≥ European
- 14 Criterion benchmarks covering all major operations
- Zero clippy warnings
- Comprehensive README with architecture diagram, code examples, and CLI usage
