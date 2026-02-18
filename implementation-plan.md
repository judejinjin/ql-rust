# ql-rust — Detailed Implementation Plan

> A phased plan to re-implement QuantLib in Rust with modern crates, SecDB/Beacon-style persistence, and production-grade engineering practices. Derived from [project.md](project.md) and [secdb-beacon-summary.md](secdb-beacon-summary.md).

---

## Table of Contents

1. [Project Overview & Goals](#1-project-overview--goals)
2. [Workspace Bootstrap](#2-workspace-bootstrap)
3. [Phase 1: Foundations (ql-core + ql-time)](#3-phase-1-foundations-ql-core--ql-time)
4. [Phase 2: Math Library (ql-math)](#4-phase-2-math-library-ql-math)
5. [Phase 3: Rates Infrastructure (ql-currencies, ql-indexes, ql-termstructures)](#5-phase-3-rates-infrastructure)
6. [Phase 4: Cash Flows (ql-cashflows)](#6-phase-4-cash-flows-ql-cashflows)
7. [Phase 5: Instruments & Pricing Engines](#7-phase-5-instruments--pricing-engines)
8. [Phase 6: Stochastic Processes & Models](#8-phase-6-stochastic-processes--models)
9. [Phase 7: Advanced Numerical Methods](#9-phase-7-advanced-numerical-methods)
10. [Phase 8: Volatility Surfaces](#10-phase-8-volatility-surfaces)
11. [Phase 9: Credit & Inflation](#11-phase-9-credit--inflation)
12. [Phase 10: Experimental & Exotics](#12-phase-10-experimental--exotics)
13. [Phase 11: Persistence Layer](#13-phase-11-persistence-layer)
14. [Phase 12: Integration, CLI & API Shell](#14-phase-12-integration-cli--api-shell)
15. [Phase 13: Advanced American Option Engines](#15-phase-13-advanced-american-option-engines)
16. [Phase 14: Bates Model & Jump-Diffusion](#16-phase-14-bates-model--jump-diffusion)
17. [Phase 15: Multi-Asset & Basket Options](#17-phase-15-multi-asset--basket-options)
18. [Phase 16: Short-Rate Models — Full Coverage](#18-phase-16-short-rate-models--full-coverage)
19. [Phase 17: Advanced Swaption & Cap/Floor Engines](#19-phase-17-advanced-swaption--capfloor-engines)
20. [Phase 18: Advanced Volatility Surfaces & Smile Models](#20-phase-18-advanced-volatility-surfaces--smile-models)
21. [Phase 19: Advanced Finite Difference Framework](#21-phase-19-advanced-finite-difference-framework)
22. [Phase 20: LIBOR Market Model Framework](#22-phase-20-libor-market-model-framework)
23. [Phase 21: Advanced Credit Models](#23-phase-21-advanced-credit-models)
24. [Phase 22: Math Library Extensions](#24-phase-22-math-library-extensions)
25. [Phase 23: Advanced Cash Flows & Coupons](#25-phase-23-advanced-cash-flows--coupons)
26. [Phase 24: Advanced Yield Curve & Fitting](#26-phase-24-advanced-yield-curve--fitting)
27. [Cross-Cutting Concerns](#27-cross-cutting-concerns)
28. [Dependency Graph Between Phases](#28-dependency-graph-between-phases)
29. [Testing & Validation Strategy](#29-testing--validation-strategy)
30. [Performance Benchmarking Plan](#30-performance-benchmarking-plan)
31. [Risk Register & Mitigations](#31-risk-register--mitigations)
32. [Estimated Timeline](#32-estimated-timeline)
33. [Definition of Done — Per-Phase Checklist](#33-definition-of-done--per-phase-checklist)

---

## 1. Project Overview & Goals

### 1.1 What We Are Building

A **complete quantitative finance library in Rust** that:

- Faithfully re-implements QuantLib's instrument/engine/term-structure object model.
- Exploits Rust's ownership, enum dispatch, and `rayon` parallelism for zero-cost abstractions and fearless concurrency.
- Provides SecDB/Beacon-style persistence with event sourcing, bitemporal versioning, and DAG-aware storage.
- Ships as a multi-crate Cargo workspace (`ql-rust`) publishable to crates.io.

### 1.2 Non-Goals (Out of Scope)

- GUI / desktop trading application.
- Real-time market data feed adapters (will define trait interfaces only).
- Regulatory reporting generators.
- 1:1 API compatibility with QuantLib C++ (idiomatic Rust API is preferred).

### 1.3 Success Criteria

| Criterion | Target |
|---|---|
| Vanilla European option (BS analytic) matches QuantLib to 1e-10 | Phase 5 |
| Yield curve bootstrap from deposits + swaps within 1bp of QuantLib | Phase 3 |
| Heston calibration converges to same params as QuantLib ± 1e-6 | Phase 6 |
| Monte Carlo European 100k paths within 2× QuantLib speed | Phase 7 |
| Full `cargo test` pass with zero `unsafe` outside FFI (if any) | All phases |
| `cargo clippy -- -D warnings` clean | All phases |

---

## 2. Workspace Bootstrap

**Goal:** Set up the Cargo workspace skeleton, CI, linting, and documentation infrastructure before writing any domain code.

### 2.1 Tasks

| # | Task | Output |
|---|---|---|
| 2.1 | Create workspace root `Cargo.toml` with `[workspace]` members list | `ql-rust/Cargo.toml` |
| 2.2 | Scaffold all 12 crate directories with `cargo init --lib` | `crates/ql-{core,time,math,currencies,indexes,termstructures,cashflows,instruments,processes,models,pricingengines,methods}/` |
| 2.3 | Add a `crates/ql-persistence/` crate for the storage layer | `crates/ql-persistence/` |
| 2.4 | Configure shared `[workspace.dependencies]` for common crates | Workspace-level dependency dedup |
| 2.5 | Add `rustfmt.toml` with project style rules | Consistent formatting |
| 2.6 | Add `clippy.toml` / `.cargo/config.toml` with strict lint settings | `#![deny(clippy::all, clippy::pedantic)]` |
| 2.7 | Create `.github/workflows/ci.yml` — `cargo check`, `cargo test`, `cargo clippy`, `cargo doc` | Green CI on every push |
| 2.8 | Add `README.md` with crate map, build instructions, license (Apache 2.0 / MIT dual) | Developer onboarding |
| 2.9 | Add `examples/` directory with placeholder `price_european_option.rs` | `cargo run --example price_european_option` scaffold |
| 2.10 | Add `benches/` directory with placeholder Criterion benchmark | `cargo bench` scaffold |

### 2.2 Workspace `Cargo.toml` Structure

```toml
[workspace]
resolver = "2"
members = [
    "crates/ql-core",
    "crates/ql-time",
    "crates/ql-math",
    "crates/ql-currencies",
    "crates/ql-indexes",
    "crates/ql-termstructures",
    "crates/ql-cashflows",
    "crates/ql-instruments",
    "crates/ql-processes",
    "crates/ql-models",
    "crates/ql-pricingengines",
    "crates/ql-methods",
    "crates/ql-persistence",
]

[workspace.dependencies]
thiserror = "2"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tracing = "0.1"
approx = "0.5"
chrono = { version = "0.4", features = ["serde"] }
nalgebra = "0.33"
rand = "0.9"
rand_distr = "0.5"
rayon = "1.10"
statrs = "0.18"
argmin = "0.10"
criterion = { version = "0.5", features = ["html_reports"] }
bincode = "1"
```

### 2.3 Acceptance Criteria

- [ ] `cargo check --workspace` succeeds (empty libs compile).
- [ ] `cargo test --workspace` succeeds (no tests yet, but no errors).
- [ ] `cargo clippy --workspace -- -D warnings` clean.
- [ ] `cargo doc --workspace --no-deps` generates docs.
- [ ] CI workflow passes on GitHub.

---

## 3. Phase 1: Foundations (`ql-core` + `ql-time`)

**Duration estimate:** 2–3 weeks
**Depends on:** Phase 0 (workspace bootstrap)
**Milestone:** *Can generate a coupon schedule for a 5Y semiannual swap.*

### 3.1 `ql-core` — Foundational Types & Patterns

#### 3.1.1 `types.rs` — Type Aliases

```
pub type Real = f64;
pub type Rate = f64;
pub type Spread = f64;
pub type DiscountFactor = f64;
pub type Volatility = f64;
pub type Time = f64;
pub type Natural = u32;
pub type Integer = i32;
pub type Size = usize;
```

#### 3.1.2 `errors.rs` — Error Types

| Error Variant | When |
|---|---|
| `NullEngine` | `npv()` called without setting a pricing engine |
| `EmptyHandle` | Dereferencing an unlinked Handle |
| `MissingResult { field }` | A requested result (NPV, delta, etc.) was not computed |
| `DateOutOfRange(Date)` | Date is beyond term structure range |
| `NegativeValue { quantity, value }` | Negative volatility, rate, or notional |
| `CalibrationFailure(String)` | Optimizer did not converge |
| `RootNotFound(usize)` | Root-finding solver exceeded max iterations |
| `InvalidArgument(String)` | Catch-all for bad inputs |
| `NotFound` | Object not found in persistence store |

Implement with `thiserror`. Define `pub type QLResult<T> = Result<T, QLError>;`.

#### 3.1.3 `observable.rs` — Observer / Observable

Implement the callback-based observer pattern with `Arc<Weak<dyn Observer>>` as described in project.md §17.4.1. Key types:

- `trait Observable` — `register_observer`, `unregister_observer`, `notify_observers`.
- `trait Observer: Send + Sync` — `fn update(&self)`.
- `struct ObservableState` — internal bookkeeping with `Weak` refs to prevent cycles.

Test: Register two observers on a `SimpleQuote`, change the quote, verify both observers received `update()`.

#### 3.1.4 `lazy.rs` — LazyObject / CachedValue

Implement `LazyCache` (Cell-based dirty flag) and `Cached<T>` (RefCell-based cached value) as in project.md §17.4.2.

Test: Create a `Cached<f64>`, verify computation runs only once, invalidate, verify recomputation.

#### 3.1.5 `handle.rs` — Handle / RelinkableHandle

Implement `Handle<T>` (shared `Arc<RwLock<Link<T>>>`) and `RelinkableHandle<T>` as in project.md §17.4.3.

Test:
1. Create `RelinkableHandle<dyn YieldTermStructure>` → clone `Handle` → verify both see the same object.
2. Relink → verify the cloned Handle now sees the new object.

#### 3.1.6 `settings.rs` — Global Settings Singleton

Implement with `OnceLock<Settings>` as in project.md §17.4.4.

- `evaluation_date() -> Date`
- `set_evaluation_date(Date)` (with observer notification)
- `include_reference_date_events`, `include_todays_cashflows`

Test: Set evaluation date, verify `Settings::instance().evaluation_date()` returns it.

#### 3.1.7 `quote.rs` — Quote Trait + SimpleQuote

```rust
pub trait Quote: Observable + Send + Sync {
    fn value(&self) -> QLResult<f64>;
    fn is_valid(&self) -> bool;
}

pub struct SimpleQuote { /* value: Cell<Option<f64>>, observable_state */ }
```

Test: Create SimpleQuote, set value, verify observers notified.

---

### 3.2 `ql-time` — Dates, Calendars, Day Counters, Schedules

#### 3.2.1 `date.rs` — Serial-Number Date

- Internal representation: `i32` serial number (days from epoch).
- Conversion to/from `(year, month, day)` via lookup table or formula.
- Implement `Add<i32>`, `Sub<Date>`, `Ord`, `Display`, `Serialize/Deserialize`.
- `Date::today()` using `chrono::Local::now()`.

Test: Round-trip `from_ymd` ↔ `year/month/day`, verify serial number arithmetic, `weekday()`.

#### 3.2.2 `period.rs` — Period, TimeUnit, Frequency

```rust
pub enum TimeUnit { Days, Weeks, Months, Years }
pub struct Period { pub length: i32, pub unit: TimeUnit }
pub enum Frequency { Once, Annual, Semiannual, Quarterly, Monthly, EveryFourthWeek, Biweekly, Weekly, Daily, OtherFrequency }
```

Test: `Period::new(6, Months)`, arithmetic, conversion.

#### 3.2.3 `calendar.rs` — Calendar Enum

Enum-based (not trait-object) for zero-cost dispatch. Start with:

| Calendar | Priority |
|---|---|
| `NullCalendar` | Immediate (needed for tests) |
| `TARGET` | Immediate (EUR rates) |
| `UnitedStates::Settlement` | Immediate (USD rates) |
| `UnitedStates::NYSE` | Phase 5 (equity options) |
| `UnitedKingdom::Exchange` | Phase 3 (GBP rates) |
| `Japan` | Phase 9 |
| `WeekendsOnly` | Immediate |
| `JointCalendar` | Phase 3 |

Methods: `is_business_day(Date)`, `is_holiday(Date)`, `advance(Date, Period, Convention)`, `adjust(Date, Convention)`, `business_days_between(Date, Date)`.

Test: Verify TARGET holidays for 2024-2025 against QuantLib reference. Verify `advance()` with `ModifiedFollowing`.

#### 3.2.4 `day_counter.rs` — DayCounter Enum

Start with:

| Day Counter | Priority |
|---|---|
| `Actual360` | Immediate |
| `Actual365Fixed` | Immediate |
| `Thirty360::BondBasis` | Phase 4 (bonds) |
| `Thirty360::EuroBondBasis` | Phase 4 |
| `ActualActual::ISDA` | Phase 4 (bonds) |
| `ActualActual::ISMA` | Phase 4 |
| `Business252` | Phase 9 (BRL) |

Methods: `day_count(d1, d2) -> i32`, `year_fraction(d1, d2) -> f64`.

Test: Verify `year_fraction` for known date pairs against QuantLib.

#### 3.2.5 `schedule.rs` — Schedule Builder

Builder pattern for generating coupon dates:

```rust
let schedule = Schedule::builder()
    .from(effective_date)
    .to(maturity_date)
    .with_frequency(Semiannual)
    .with_calendar(Calendar::Target)
    .with_convention(ModifiedFollowing)
    .with_termination_convention(ModifiedFollowing)
    .with_rule(DateGeneration::Backward)
    .build()?;
```

Test: Generate semiannual schedule for a 5Y swap, verify dates match QuantLib.

#### 3.2.6 `business_day_convention.rs`

Enum: `Following`, `ModifiedFollowing`, `Preceding`, `ModifiedPreceding`, `Unadjusted`, `HalfMonthModifiedFollowing`, `Nearest`.

#### 3.2.7 `imm.rs` — IMM Dates

`is_imm_date()`, `next_date()`, `imm_code()`.

### 3.3 Phase 1 Deliverables

- [ ] `cargo test -p ql-core` — all unit tests pass.
- [ ] `cargo test -p ql-time` — all unit tests pass, including date/calendar/schedule tests cross-validated against QuantLib.
- [ ] Example: generate and print a 5Y semiannual swap schedule.
- [ ] Documentation: `cargo doc -p ql-core -p ql-time`.

---

## 4. Phase 2: Math Library (`ql-math`)

**Duration estimate:** 2–3 weeks
**Depends on:** Phase 1 (needs `ql-core` types and errors)
**Milestone:** *Can bootstrap a yield curve from market rates using Brent root-finding and linear interpolation.*

### 4.1 Interpolation (`interpolation.rs` + `interpolations/`)

#### 4.1.1 Traits

```rust
pub trait Interpolation {
    fn value(&self, x: f64) -> f64;
    fn derivative(&self, x: f64) -> f64;
    fn primitive(&self, x: f64) -> f64;  // integral
}

pub trait Interpolator: Clone {
    type Interp: Interpolation;
    fn interpolate(&self, x: &[f64], y: &[f64]) -> Self::Interp;
}
```

#### 4.1.2 Concrete Implementations (Priority Order)

| Interpolation | Phase Needed | Notes |
|---|---|---|
| `LinearInterpolation` | Phase 2 | Simplest; needed for basic curve construction |
| `LogLinearInterpolation` | Phase 2 | Standard for discount factors |
| `CubicInterpolation` (natural/clamped) | Phase 3 | Smooth curves |
| `MonotoneCubicInterpolation` | Phase 3 | Hyman filter for monotone discount factors |
| `SABRInterpolation` | Phase 8 | Smile interpolation |

Test: Interpolate known data points, verify values + derivatives against analytical results.

### 4.2 Root Finding (`solvers/`)

| Solver | Priority |
|---|---|
| `Brent` | Phase 2 — primary solver for bootstrapping |
| `Newton` | Phase 2 — used in implied vol |
| `Bisection` | Phase 2 — fallback |
| `Ridder` | Phase 6 — alternative for calibration |

Interface:

```rust
pub trait Solver1D {
    fn solve<F: Fn(f64) -> f64>(
        &self, f: F, target: f64, guess: f64,
        x_min: f64, x_max: f64, accuracy: f64,
        max_evaluations: usize,
    ) -> QLResult<f64>;
}
```

Test: Solve `x^2 - 2 = 0` with each solver, verify `x ≈ √2` to 1e-12.

### 4.3 Distributions (`distributions.rs`)

Wrap `statrs` crate:

| Distribution | Usage |
|---|---|
| Normal (CDF, PDF, inverse CDF) | Black-Scholes, everywhere |
| Chi-squared | Heston engine |
| Poisson | Jump diffusion |
| Non-central chi-squared | CIR process |

Test: Verify normal CDF/inverse CDF round-trip, boundary values.

### 4.4 Optimization (`optimization.rs`)

Wrap `argmin` crate:

| Optimizer | Usage |
|---|---|
| `LevenbergMarquardt` | Heston calibration, curve fitting |
| `NelderMead` (Simplex) | General-purpose, derivative-free |
| `LBFGS` | Gradient-based, Nelson-Siegel fitting |

Define:

```rust
pub struct EndCriteria {
    pub max_iterations: usize,
    pub max_stationary_iterations: usize,
    pub root_epsilon: f64,
    pub function_epsilon: f64,
    pub gradient_epsilon: f64,
}
```

Test: Minimize Rosenbrock function with each optimizer, verify convergence.

### 4.5 Integration (`integration.rs`)

| Method | Usage |
|---|---|
| `GaussLobattoIntegral` | Analytic Heston engine |
| `SimpsonIntegral` | General quadrature |
| `GaussLegendreIntegral` | High-accuracy quadrature |

Test: Integrate `sin(x)` from 0 to π, verify result = 2.0 to 1e-12.

### 4.6 Random Number Generation (`rng.rs`, `sobol.rs`)

| Generator | Usage |
|---|---|
| Mersenne Twister (via `rand`) | Standard MC simulation |
| Sobol sequences | Quasi-Monte Carlo variance reduction |
| Inverse cumulative normal | Transform uniform → normal variates |
| Box-Muller | Alternative normal variate generation |

Test: Generate 1M normal variates, verify mean ≈ 0 and stddev ≈ 1. Verify Sobol uniformity.

### 4.7 Matrix Utilities (`matrix.rs`)

Thin re-exports / wrappers from `nalgebra`:

```rust
pub type Vector = nalgebra::DVector<f64>;
pub type Matrix = nalgebra::DMatrix<f64>;
```

Add: Cholesky decomposition helper for correlated path generation.

### 4.8 Phase 2 Deliverables

- [ ] `cargo test -p ql-math` — all unit tests pass.
- [ ] `Brent::solve` can find implied discount factor given a swap rate.
- [ ] Linear + LogLinear interpolation fully functional.
- [ ] Normal distribution CDF/PDF/Inverse match QuantLib to 1e-15.

---

## 5. Phase 3: Rates Infrastructure

**Duration estimate:** 3–4 weeks
**Depends on:** Phase 1 + Phase 2
**Milestone:** *Can bootstrap a yield curve from deposit rates + swap rates and compute discount factors, zero rates, and forward rates.*

**Crates:** `ql-currencies`, `ql-indexes`, `ql-termstructures`

### 5.1 `ql-currencies`

Implement `Currency` struct (ISO code, name, numeric code, rounding rules):

| Currency | Priority |
|---|---|
| USD, EUR, GBP, JPY, CHF | Phase 3 |
| AUD, CAD, NZD, SEK, NOK | Phase 5 |
| BRL, CNY, INR, KRW, MXN | Phase 9 |

Also: `Money` (amount + currency), `ExchangeRate`, `ExchangeRateManager` singleton.

### 5.2 `ql-indexes`

#### 5.2.1 Index Trait

```rust
pub trait Index: Observable + Send + Sync {
    fn name(&self) -> &str;
    fn fixing_calendar(&self) -> Calendar;
    fn is_valid_fixing_date(&self, date: Date) -> bool;
    fn fixing(&self, date: Date, forecast_today_fixing: bool) -> QLResult<f64>;
}
```

#### 5.2.2 IndexManager Singleton

Global store for historical fixings: `HashMap<String, TimeSeries<f64>>`.

#### 5.2.3 Concrete Indexes (Priority)

| Index | Type | Priority |
|---|---|---|
| SOFR | `OvernightIndex` | Phase 3 |
| ESTR (€STR) | `OvernightIndex` | Phase 3 |
| Euribor 3M/6M | `IborIndex` | Phase 3 |
| USD SOFR Swap Index | `SwapIndex` | Phase 5 |
| SONIA | `OvernightIndex` | Phase 5 |
| US CPI | `ZeroInflationIndex` | Phase 9 |

#### 5.2.4 InterestRateIndex Base

Contains: `tenor`, `fixing_days`, `currency`, `day_counter`, `Handle<dyn YieldTermStructure>` for forecasting.

### 5.3 `ql-termstructures`

#### 5.3.1 Base Term Structure Trait

As designed in project.md §17.5.3: `reference_date()`, `day_counter()`, `calendar()`, `max_date()`, `time_from_reference()`, `settlement_days()`.

#### 5.3.2 YieldTermStructure Trait

Three interchangeable representations (override one, get the other two):
- `discount_impl(t) -> f64`
- `zero_rate_impl(t) -> f64`
- `forward_rate_impl(t) -> f64`

Default implementations derive the other two from whichever one is overridden.

#### 5.3.3 Concrete Yield Curves (Priority)

| Curve | Priority | Notes |
|---|---|---|
| `FlatForward` | Phase 3 | Testing workhorse |
| `DiscountCurve` | Phase 3 | Interpolated discount factors |
| `ZeroCurve` | Phase 3 | Interpolated zero rates |
| `PiecewiseYieldCurve<Traits, Interp>` | Phase 3 | Bootstrapped — the workhorse |
| `ZeroSpreadedTermStructure` | Phase 5 | Spread over base curve |
| `ForwardSpreadedTermStructure` | Phase 5 | Spread over base forward |
| `FittedBondDiscountCurve` | Phase 9 | Nelson-Siegel, Svensson |

#### 5.3.4 Bootstrapping Framework

Implement iterative bootstrap:

1. **RateHelper trait** — `pillar_date()`, `quote_value()`, `implied_quote(curve)`, `set_term_structure(Handle)`.
2. **Concrete helpers:**

| Helper | Phase | Market Input |
|---|---|---|
| `DepositRateHelper` | Phase 3 | Deposit rates |
| `FraRateHelper` | Phase 3 | Forward rate agreements |
| `SwapRateHelper` | Phase 3 | Swap par rates |
| `OISRateHelper` | Phase 3 | OIS swap rates |
| `FuturesRateHelper` | Phase 5 | Interest rate futures prices |
| `BondHelper` | Phase 9 | Bond prices |

3. **IterativeBootstrap** — for each helper in pillar-date order, use `Brent` solver to find the interpolation node that makes `implied_quote == market_quote`.

#### 5.3.5 Tasks & Tests

- [ ] Build a `FlatForward` curve, verify `discount(1Y) = exp(-r)`.
- [ ] Bootstrap from 3 deposits + 5 swaps, verify discount factors match QuantLib to 1e-8.
- [ ] Verify `zero_rate` and `forward_rate` consistency with `discount`.
- [ ] Test `ZeroSpreadedTermStructure` adds spread correctly.

### 5.4 Phase 3 Deliverables

- [ ] `cargo test -p ql-currencies -p ql-indexes -p ql-termstructures` — all pass.
- [ ] Example: `bootstrap_yield_curve.rs` — bootstrap USD SOFR curve from deposits + OIS swaps, print discount factors.
- [ ] Cross-validate against QuantLib's `PiecewiseYieldCurve` with same inputs.

---

## 6. Phase 4: Cash Flows (`ql-cashflows`)

**Duration estimate:** 2 weeks
**Depends on:** Phase 1 + Phase 3 (indexes, term structures for floating coupons)
**Milestone:** *Can generate and value both legs of a fixed/float interest rate swap.*

### 6.1 Core Traits

```rust
pub trait CashFlow: Send + Sync {
    fn date(&self) -> Date;
    fn amount(&self) -> f64;
    fn has_occurred(&self, ref_date: Date) -> bool;
}

pub trait Coupon: CashFlow {
    fn nominal(&self) -> f64;
    fn rate(&self) -> f64;
    fn accrual_start(&self) -> Date;
    fn accrual_end(&self) -> Date;
    fn accrual_period(&self) -> f64;
    fn day_counter(&self) -> DayCounter;
}

pub type Leg = Vec<Box<dyn CashFlow>>;
```

### 6.2 Concrete Cash Flows (Priority)

| Type | Priority | Notes |
|---|---|---|
| `SimpleCashFlow` | Phase 4 | Fixed amount on a date (notional exchange) |
| `FixedRateCoupon` | Phase 4 | Fixed-rate coupon |
| `IborCoupon` | Phase 4 | IBOR-indexed floating coupon |
| `OvernightIndexedCoupon` | Phase 4 | Compounded overnight (SOFR/ESTR) |
| `CappedFlooredCoupon` | Phase 8 | Cap/floor features |
| `CMSCoupon` | Phase 8 | CMS-rate coupon |
| `InflationCoupon` | Phase 9 | CPI/YoY linked |

### 6.3 Coupon Pricers

`FloatingRateCouponPricer` trait — strategy for convexity adjustment:

```rust
pub trait FloatingRateCouponPricer: Send + Sync {
    fn initialize(&mut self, coupon: &dyn FloatingRateCoupon);
    fn adjusted_fixing(&self) -> f64;
    fn swap_rate(&self) -> f64;  // for CMS coupons
}
```

Start with `BlackIborCouponPricer` for vanilla IBOR coupons.

### 6.4 Leg Builder Utilities

Functions to construct legs from schedules:

```rust
pub fn fixed_leg(
    schedule: &Schedule, notionals: &[f64],
    rates: &[f64], day_counter: DayCounter,
) -> Leg;

pub fn ibor_leg(
    schedule: &Schedule, notionals: &[f64],
    index: Arc<dyn IborIndex>, spreads: &[f64],
    day_counter: DayCounter,
) -> Leg;
```

### 6.5 Cash Flow Analytics

Utility functions on `Leg`:

```rust
pub fn npv(leg: &Leg, curve: &dyn YieldTermStructure, settle: Date) -> f64;
pub fn bps(leg: &Leg, curve: &dyn YieldTermStructure, settle: Date) -> f64;
pub fn accrued_amount(leg: &Leg, settle: Date) -> f64;
pub fn duration(leg: &Leg, rate: InterestRate, settle: Date) -> f64;
```

### 6.6 Phase 4 Deliverables

- [ ] Generate fixed + floating legs for a 5Y semiannual swap.
- [ ] NPV of fixed leg matches QuantLib's `CashFlows::npv()`.
- [ ] `OvernightIndexedCoupon` computes compounded SOFR rate correctly.

---

## 7. Phase 5: Instruments & Pricing Engines

**Duration estimate:** 3–4 weeks
**Depends on:** Phases 1–4
**Milestone:** *Can price vanilla European options (BS analytic), interest rate swaps (discounting engine), and fixed-rate bonds.*

### 7.1 `ql-instruments` — Instrument Trait

As designed in project.md §17.5.4:

```rust
pub trait Instrument: Sized {
    type Arguments;
    type Results: Into<InstrumentResults>;
    fn is_expired(&self) -> bool;
    fn build_arguments(&self) -> Self::Arguments;
}
```

Plus `PricedInstrument<I>` wrapper with lazy caching + engine dispatch.

### 7.2 Payoffs & Exercises

```rust
pub enum OptionType { Call, Put }

pub enum Payoff {
    PlainVanilla { option_type: OptionType, strike: f64 },
    CashOrNothing { option_type: OptionType, strike: f64, cash: f64 },
    AssetOrNothing { option_type: OptionType, strike: f64 },
    Gap { option_type: OptionType, strike: f64, second_strike: f64 },
}

pub enum Exercise {
    European { expiry: Date },
    American { earliest: Date, expiry: Date },
    Bermudan { dates: Vec<Date> },
}
```

### 7.3 Concrete Instruments (Priority)

| Instrument | Phase | Engine(s) |
|---|---|---|
| `VanillaOption` | Phase 5 | `AnalyticEuropeanEngine`, MC, FD |
| `VanillaSwap` | Phase 5 | `DiscountingSwapEngine` |
| `FixedRateBond` | Phase 5 | `DiscountingBondEngine` |
| `FloatingRateBond` | Phase 5 | `DiscountingBondEngine` |
| `ForwardRateAgreement` | Phase 5 | Analytic |
| `BarrierOption` | Phase 7 | Analytic, MC, FD |
| `AsianOption` | Phase 10 | MC |
| `Swaption` | Phase 8 | Black, Bachelier, HW tree |
| `CapFloor` | Phase 8 | Black, Bachelier |
| `CreditDefaultSwap` | Phase 9 | `MidPointCdsEngine` |

### 7.4 `ql-pricingengines` — Concrete Engines

#### Phase 5 Engines

| Engine | Instrument | Method |
|---|---|---|
| `AnalyticEuropeanEngine` | `VanillaOption` (European) | Black-Scholes closed-form (project.md §17.5.4) |
| `DiscountingSwapEngine` | `VanillaSwap` | NPV = Σ fixed leg − Σ float leg, discounted |
| `DiscountingBondEngine` | `Bond` | NPV = Σ discounted cash flows |

Implement `AnalyticEuropeanEngine` exactly as in project.md §17.5.4 — compute d1, d2, NPV, delta, gamma, vega.

### 7.5 Implied Volatility

Add `implied_volatility()` method on `VanillaOption`:
- Use `Brent` solver to find vol that makes BS price = target price.

### 7.6 Phase 5 Deliverables

- [ ] BS European call/put price matches QuantLib to 1e-10.
- [ ] All Greeks (delta, gamma, vega, theta, rho) match QuantLib to 1e-8.
- [ ] Swap NPV matches QuantLib to 1e-6 (basis point precision).
- [ ] Bond dirty/clean price matches QuantLib.
- [ ] `implied_volatility()` converges for ITM/ATM/OTM options.
- [ ] Example: `price_european_option.rs` — full working example.

---

## 8. Phase 6: Stochastic Processes & Models

**Duration estimate:** 3–4 weeks
**Depends on:** Phase 5 (needs instruments for calibration helpers)
**Milestone:** *Can calibrate a Heston model to market vol surface and price with `AnalyticHestonEngine`.*

### 8.1 `ql-processes`

#### 8.1.1 Process Traits

As in project.md §17.5.5: `StochasticProcess` (multi-D) and `StochasticProcess1D` (scalar).

#### 8.1.2 Concrete Processes (Priority)

| Process | Phase | State Variables |
|---|---|---|
| `GeneralizedBlackScholesProcess` | Phase 6 | Spot |
| `HestonProcess` | Phase 6 | Spot + variance |
| `HullWhiteProcess` | Phase 6 | Short rate |
| `OrnsteinUhlenbeckProcess` | Phase 6 | Mean-reverting |
| `BatesProcess` | Phase 10 | Heston + jumps |
| `G2Process` | Phase 10 | Two-factor Gaussian |

#### 8.1.3 Discretization Strategies

- Euler discretization (default)
- Milstein (for Heston)
- Exact (for GBM, OU)
- Quadratic exponential (QE scheme for Heston)

### 8.2 `ql-models`

#### 8.2.1 CalibratedModel Trait

As in project.md §17.5.6:
- `Parameter` with constraints and transformations.
- `calibrate(helpers, optimizer, end_criteria)`.
- `params()` / `set_params()`.

#### 8.2.2 Concrete Models (Priority)

| Model | Phase | Parameters |
|---|---|---|
| `HestonModel` | Phase 6 | v0, κ, θ, σ, ρ |
| `HullWhiteModel` | Phase 6 | a (mean reversion), σ |
| `BlackKarasinskiModel` | Phase 8 | a, σ |
| `Vasicek` | Phase 8 | a, b, σ |
| `BatesModel` | Phase 10 | Heston + λ, ν, δ |
| `G2Model` | Phase 10 | a, σ, b, η, ρ |

#### 8.2.3 Calibration Helpers

- `VanillaOptionHelper` — calibrate equity models to option prices.
- `SwaptionHelper` — calibrate short-rate models to swaption vols.
- `CapFloorHelper` — calibrate to cap/floor vols.

### 8.3 Analytic Heston Engine

Implement the Heston characteristic function + Gauss-Lobatto integration for European option pricing. This is a major deliverable — validates process, model, calibration, and engine infrastructure.

### 8.4 Phase 6 Deliverables

- [ ] `GeneralizedBlackScholesProcess` produces correct drift/diffusion.
- [ ] `HestonModel` calibrates to a synthetic vol surface (5 strikes × 4 expiries).
- [ ] `AnalyticHestonEngine` price matches QuantLib to 1e-6.
- [ ] `HullWhiteModel` calibrates to swaption grid.
- [ ] Example: `calibrate_heston.rs`.

---

## 9. Phase 7: Advanced Numerical Methods

**Duration estimate:** 4–5 weeks
**Depends on:** Phase 6
**Milestone:** *Can price American options with finite differences and exotic options with Monte Carlo.*

### 9.1 `ql-methods` — Monte Carlo Framework

#### 9.1.1 Components

- **PathGenerator** — generates a single sample path using a `StochasticProcess` + RNG.
- **MultiPathGenerator** — correlated multi-dimensional paths (Cholesky decomposition).
- **PathPricer** — values a payoff along a path.
- **MonteCarloModel** — orchestrates path generation + pricing + statistics.

#### 9.1.2 Parallel MC with `rayon`

As in project.md §17.6 — use `into_par_iter()` for embarrassingly parallel path generation. Per-thread RNG via `thread_rng()`.

#### 9.1.3 MC Engines (Priority)

| Engine | Instrument | Notes |
|---|---|---|
| `MCEuropeanEngine` | `VanillaOption` | Validates MC framework against analytic |
| `MCBarrierEngine` | `BarrierOption` | Path-dependent |
| `MCAmericanEngine` (Longstaff-Schwartz) | `VanillaOption` (American) | Regression-based early exercise |
| `MCAsianEngine` | `AsianOption` | Arithmetic/geometric averaging |
| `MCHestonEngine` | `VanillaOption` | Multi-dimensional path |

#### 9.1.4 Variance Reduction

- Antithetic variates
- Control variates (analytic European as control)
- Quasi-Monte Carlo (Sobol)

### 9.2 Finite Difference Framework

#### 9.2.1 Components

- **FdmMesher** — defines the spatial grid.
- **FdmLinearOp** — discretized PDE operator.
- **FdmStepCondition** — American exercise, barrier, etc.
- **FdmSolver** (Crank-Nicolson / Douglas scheme) — time-stepping.

#### 9.2.2 FD Engines (Priority)

| Engine | Instrument | Notes |
|---|---|---|
| `FdBlackScholesVanillaEngine` | `VanillaOption` (American/European) | 1D PDE |
| `FdHestonVanillaEngine` | `VanillaOption` | 2D PDE (spot × variance) |
| `FdBlackScholesBarrierEngine` | `BarrierOption` | 1D PDE with boundary |

### 9.3 Lattice Framework

#### 9.3.1 Components

- **TreeLattice** — binomial/trinomial tree.
- **BinomialTree** — CRR, Jarrow-Rudd, Tian, Leisen-Reimer.
- **TrinomialTree** — for short-rate models.

#### 9.3.2 Lattice Engines

| Engine | Instrument | Notes |
|---|---|---|
| `BinomialVanillaEngine` | `VanillaOption` (American/European) | Validates against BS analytic |
| `TreeSwaptionEngine` | `Swaption` | Hull-White tree |

### 9.4 Phase 7 Deliverables

- [ ] MC European price converges to BS analytic ± 3 std errors.
- [ ] American put price > European put price (early exercise premium).
- [ ] FD American option matches Binomial tree to 1e-4.
- [ ] MC Asian option matches QuantLib.
- [ ] Benchmark: MC 100k paths within 2× QuantLib runtime.
- [ ] Benchmark: FD 200×200 grid within 2× QuantLib runtime.

---

## 10. Phase 8: Volatility Surfaces

**Duration estimate:** 3 weeks
**Depends on:** Phase 6 + Phase 7
**Milestone:** *Can construct a complete Black vol surface with SABR smile interpolation and price swaptions / cap-floors.*

### 10.1 Volatility Term Structures

| Trait / Type | Description |
|---|---|
| `BlackVolTermStructure` | Base trait: `black_vol(t, strike)`, `black_variance(t, strike)` |
| `LocalVolTermStructure` | `local_vol(t, underlying)` |
| `BlackConstantVol` | Flat vol surface |
| `BlackVarianceSurface` | Interpolated vol grid (strike × expiry) |
| `LocalVolSurface` (Dupire) | Derived from Black vol via Dupire formula |

### 10.2 Smile Interpolation

| Method | Notes |
|---|---|
| SABR | `sabr_volatility(strike, forward, t, alpha, beta, rho, nu)` |
| `SabrSmileSection` | SABR for a single expiry |
| `KahaleSmileSection` | Arbitrage-free extrapolation |

### 10.3 Swaption & Cap/Floor Instruments

| Instrument | Engine |
|---|---|
| `Swaption` | `BlackSwaptionEngine`, `BachelierSwaptionEngine`, `TreeSwaptionEngine` |
| `CapFloor` | `BlackCapFloorEngine`, `BachelierCapFloorEngine` |

### 10.4 Phase 8 Deliverables

- [ ] SABR formula matches QuantLib's `sabrVolatility()` to 1e-10.
- [ ] `BlackVarianceSurface` interpolates correctly.
- [ ] Swaption price matches QuantLib's Black engine.
- [ ] `SwaptionVolCube` can be constructed and queried.

---

## 11. Phase 9: Credit & Inflation

**Duration estimate:** 3 weeks
**Depends on:** Phase 5 + Phase 3
**Milestone:** *Can price CDS, bootstrap default probability curves, and value inflation-linked bonds.*

### 11.1 Credit

- `DefaultProbabilityTermStructure` trait: `survival_probability(t)`, `default_density(t)`, `hazard_rate(t)`.
- `PiecewiseDefaultCurve` — bootstrap from CDS spreads.
- `CreditDefaultSwap` instrument + `MidPointCdsEngine`.
- `CdsHelper` — rate helper for default curve bootstrapping.

### 11.2 Inflation

- `ZeroInflationTermStructure`, `YoYInflationTermStructure`.
- `ZeroInflationIndex` (e.g., US CPI), `YoYInflationIndex`.
- `ZeroCouponInflationSwap`, `YearOnYearInflationSwap`.

### 11.3 Phase 9 Deliverables

- [ ] CDS par spread reproduced from bootstrapped default curve.
- [ ] CDS upfront/NPV matches QuantLib to 1e-6.
- [ ] Inflation swap NPV matches QuantLib.

---

## 12. Phase 10: Experimental & Exotics

**Duration estimate:** 4–6 weeks
**Depends on:** Phases 6–9
**Milestone:** *Can price callable bonds, barrier options (analytic/MC/FD), Asian options, lookback options, cliquet options, variance swaps, and exotic chooser/compound options.*

### 12.1 Callable Bonds

- `CallableBond` instrument with `CallabilitySchedule` (call/put dates + prices).
- `TreeCallableBondEngine` — Hull-White trinomial tree pricing.
- `BlackCallableBondEngine` — Black's model for European callable.
- `CallableBondConstantVol`, `CallableBondVolStructure`.
- `DiscretizedCallableFixedRateBond` — lattice discretization.

### 12.2 Advanced Barrier Options

- `DoubleBarrierOption` (knock-in/knock-out with two barriers) + `AnalyticDoubleBarrierEngine`.
- `PartialTimeBarrierOption` (barrier active for part of option life) + `AnalyticPartialTimeBarrierEngine`.
- `SoftBarrierOption` (softened payoff) + `AnalyticSoftBarrierEngine`.
- `TwoAssetBarrierOption` + `AnalyticTwoAssetBarrierEngine`.
- `QuantoBarrierOption` — quanto-adjusted barrier.
- `FdHestonBarrierEngine` — 2D FD barrier pricing under Heston.
- `FdHestonDoubleBarrierEngine`, `FdHestonRebateEngine`.
- `BinomialBarrierEngine` — lattice barrier pricing.

### 12.3 Advanced Asian Options

- `AnalyticContinuousGeometricAveragePriceAsianEngine` — closed-form geometric average.
- `AnalyticDiscreteGeometricAveragePriceAsianEngine`, `AnalyticDiscreteGeometricAverageStrikeAsianEngine`.
- `TurnbullWakemanAsianEngine` — analytic approximation for arithmetic average.
- `ContinuousArithmeticAsianLevyEngine` — Levy approximation.
- `ChoiAsianEngine` — Choi et al. expansion.
- `MCDiscreteArithmeticAveragePriceHestonEngine` — MC Asian under Heston.
- `FdBlackScholesAsianEngine` — FD Asian pricing.

### 12.4 Lookback Options

- `LookbackOption` instrument (fixed/floating strike, continuous/discrete).
- `AnalyticContinuousFixedLookbackEngine`.
- `AnalyticContinuousFloatingLookbackEngine`.
- `AnalyticContinuousPartialFixedLookbackEngine`.
- `AnalyticContinuousPartialFloatingLookbackEngine`.
- `MCLookbackEngine` — MC for path-dependent lookbacks.

### 12.5 Cliquet & Forward-Start Options

- `CliquetOption` instrument + `AnalyticCliquetEngine`.
- `AnalyticPerformanceEngine` — performance option pricing.
- `MCPerformanceEngine` — MC performance option.
- `ForwardVanillaOption` + `ForwardEngine`, `ForwardPerformanceEngine`.
- `MCForwardEuropeanBSEngine`, `MCForwardEuropeanHestonEngine`.

### 12.6 Exotic & Chooser Options

- `SimpleChooserOption` + `AnalyticSimpleChooserEngine`.
- `ComplexChooserOption` + `AnalyticComplexChooserEngine`.
- `CompoundOption` + `AnalyticCompoundOptionEngine`.
- `HolderExtensibleOption` + `AnalyticHolderExtensibleOptionEngine`.
- `WriterExtensibleOption` + `AnalyticWriterExtensibleOptionEngine`.
- `MargrabeOption` + `AnalyticEuropeanMargrabeEngine`, `AnalyticAmericanMargrabeEngine`.
- `TwoAssetCorrelationOption` + `AnalyticTwoAssetCorrelationEngine`.
- `EverestOption` + `MCEverestEngine` — rainbow option on worst performer.
- `HimalayaOption` + `MCHimalayaEngine` — mountain-range option.
- `PagodaOption` + `MCPagodaEngine`.

### 12.7 Variance & Volatility Instruments

- `VarianceSwap` instrument + `ReplicatingVarianceSwapEngine` + `MCVarianceSwapEngine`.
- `VanillaStorageOption`, `VanillaSwingOption` — energy/commodity storage.

### 12.8 Convertible Bonds

- `ConvertibleBond` (zero-coupon, fixed-rate, floating-rate variants).
- `BinomialConvertibleEngine` — tree-based pricing.
- `RiskyBondEngine` — credit-adjusted bond pricing.

### 12.9 Other Instruments

- `BondForward`, `FixedRateBondForward`.
- `CompositeInstrument` — portfolio of instruments.
- `Stock` — simple equity instrument.
- `AssetSwap` — bond + swap combination.
- `OvernightIndexedSwap` (OIS), `BMASwap`.
- `FloatFloatSwap`, `FloatFloatSwaption`, `NonstandardSwap`, `NonstandardSwaption`.
- `ZeroCouponSwap`, `ZeroCouponBond` instrument.
- `EquityTotalReturnSwap`.
- `PerpetualFutures` + `DiscountingPerpetualFuturesEngine`.
- `OvernightIndexFuture`.

### 12.10 Advanced Bond Types

- `AmortizingFixedRateBond`, `AmortizingFloatingRateBond`.
- `AmortizingCmsRateBond`, `CmsRateBond`.
- `CPIBond` — CPI-linked bond.
- `BTP` — Italian government bonds.

### 12.11 Inflation Instruments (Extended)

- `CPICapFloor` + `InflationCapFloorEngine`.
- `CPISwap`.
- `InflationCapFloor` + `YoYInflationCapFloorEngine`.
- `YearOnYearInflationSwap`, `ZeroCouponInflationSwap` (extended).

### 12.12 Quanto Options

- `QuantoVanillaOption`, `QuantoForwardVanillaOption`.
- `QuantoEngine` — generic quanto wrapper for any vanilla engine.
- `QuantoTermStructure` — quanto-adjusted yield curve.

### 12.13 Cat Bonds

- `CatBond` instrument + `MonteCarloCatBondEngine`.
- `CatRisk` — catastrophe risk models (BetaRisk, EventSetRisk).
- `RiskyNotional` — notional at risk.

### 12.14 Phase 10 Deliverables

- [ ] Callable bond price matches QuantLib's `TreeCallableBondEngine` to 1e-4.
- [ ] Double barrier option matches analytic formula to 1e-8.
- [ ] Lookback option matches analytic to 1e-8.
- [ ] Asian option (geometric) matches closed-form to 1e-10.
- [ ] Variance swap replication matches QuantLib.
- [ ] Cliquet, chooser, compound options match analytic formulas.
- [ ] Convertible bond prices within 1% of QuantLib tree engine.
- [ ] At least 5 golden cross-validation tests against QuantLib C++.

---

## 13. Phase 11: Persistence Layer

**Duration estimate:** 3–4 weeks
**Depends on:** Phase 1 (ql-core types) — can start in parallel with Phases 3–5
**Milestone:** *Can save/load instruments, trades, curves, and lifecycle events with bitemporal versioning.*

### 13.1 `ql-persistence` Crate

#### 13.1.1 Abstract Trait — `ObjectStore`

As designed in project.md §18.7:

```rust
#[async_trait]
pub trait ObjectStore: Send + Sync {
    async fn get<T: Persistable>(&self, id: &ObjectId) -> QLResult<T>;
    async fn get_as_of<T: Persistable>(&self, id: &ObjectId, as_of: DateTime<Utc>) -> QLResult<T>;
    async fn put<T: Persistable>(&self, id: &ObjectId, obj: &T, user: &str) -> QLResult<u64>;
    async fn append_event(&self, trade_id: &ObjectId, event: &LifecycleEvent, user: &str) -> QLResult<ObjectId>;
    async fn replay_events(&self, trade_id: &ObjectId) -> QLResult<Vec<LifecycleEvent>>;
    async fn query_trades(&self, filter: &TradeFilter) -> QLResult<Vec<Trade>>;
    async fn save_snapshot(&self, snapshot: &MarketSnapshot) -> QLResult<ObjectId>;
    async fn load_snapshot(&self, date: Date, snapshot_type: SnapshotType) -> QLResult<MarketSnapshot>;
}
```

#### 13.1.2 `Persistable` Marker Trait

```rust
pub trait Persistable: Serialize + for<'de> Deserialize<'de> + Send + Sync {
    fn object_type() -> &'static str;
}
```

Derive `Persistable` for all instruments, trades, curves, market data snapshots.

#### 13.1.3 Domain Objects

| Object | Key Fields |
|---|---|
| `Trade` | trade_id, instrument, counterparty, book, notional, direction, status, trade_date, settlement_date |
| `LifecycleEvent` | event_type (Executed, Amended, Novated, Terminated, Matured, CashSettled), event_date, payload |
| `MarketSnapshot` | snapshot_date, type (EOD/Intraday), quotes, curves, vol surfaces |
| `TradeFilter` | status, counterparty, book, instrument_type, date_range |

### 13.2 Backend Implementations (Layered)

#### 13.2.1 Phase 11a: `EmbeddedStore` (redb)

- Zero infrastructure, compiled into binary.
- `redb` tables: `trades`, `events`, `snapshots`.
- Serialization: `bincode` for speed, `serde_json` for debugging.
- Feature-gated: `ql-persistence = { features = ["redb"] }`.

#### 13.2.2 Phase 11b: `PostgresStore` (sqlx + JSONB)

- Schema as designed in project.md §18.2.
- Bitemporal with `valid_from` / `valid_to` columns.
- GIN indexes on JSONB for fast queries.
- Feature-gated: `ql-persistence = { features = ["postgres"] }`.

#### 13.2.3 Phase 11c: Analytics Export (DuckDB + Parquet)

- Nightly export of trade + risk data to Parquet files.
- DuckDB queries for portfolio analytics.
- Feature-gated: `ql-persistence = { features = ["analytics"] }`.

### 13.3 Phase 11 Deliverables

- [ ] `EmbeddedStore` can round-trip a `VanillaSwap` trade.
- [ ] Event sourcing: append 5 lifecycle events, replay, verify trade state.
- [ ] Bitemporal query: retrieve trade "as of" a past timestamp.
- [ ] `PostgresStore` integration test with a test container.
- [ ] `ObjectStore` trait allows swapping backends without changing business logic.

---

## 14. Phase 12: Integration, CLI & API Shell

**Duration estimate:** 2–3 weeks
**Depends on:** Phases 5–11
**Milestone:** *End-to-end workflow: load market data → bootstrap curves → price portfolio → persist results.*

### 14.1 Convenience Crate — `ql-rust`

Top-level "facade" crate that re-exports the most common types:

```rust
pub use ql_core::*;
pub use ql_time::*;
pub use ql_instruments::*;
pub use ql_pricingengines::*;
pub use ql_termstructures::*;
```

### 14.2 CLI Tool

Binary crate `ql-cli`:

```
ql-cli price --instrument vanilla-option --spot 100 --strike 105 --vol 0.2 --rate 0.05 --expiry 1Y
ql-cli curve bootstrap --input market_data.json --output curve.json
ql-cli trade book --type irs --notional 50M --tenor 5Y --fixed-rate 3.5%
ql-cli trade lifecycle --trade-id TRD-001 --events
```

### 14.3 End-to-End Example

```rust
// 1. Load market data
let spot = SimpleQuote::new(100.0);
let rate_handle = RelinkableHandle::new(Arc::new(FlatForward::new(today, 0.05, Actual365Fixed)));
let vol_handle = RelinkableHandle::new(Arc::new(BlackConstantVol::new(today, 0.20, Actual365Fixed)));

// 2. Bootstrap a yield curve
let helpers = vec![
    DepositRateHelper::new(0.045, Period::new(3, Months), ...),
    SwapRateHelper::new(0.05, Period::new(5, Years), ...),
];
let curve = PiecewiseYieldCurve::<ZeroYield, LogLinear>::new(today, helpers, Actual365Fixed);

// 3. Price an option
let process = BlackScholesProcess::new(spot.handle(), rate_handle.handle(), vol_handle.handle());
let engine = AnalyticEuropeanEngine::new(Arc::new(process));
let option = PricedInstrument::new(
    VanillaOption::european(Call, 105.0, today + Period::new(1, Years)),
    Box::new(engine),
);
println!("NPV = {}", option.npv());

// 4. Persist
let store = EmbeddedStore::open("trading.redb")?;
store.put(&ObjectId("OPT-001".into()), &option_trade, "jude").await?;
```

### 14.4 Phase 12 Deliverables

- [ ] End-to-end example compiles and runs.
- [ ] CLI can price a European option from command-line arguments.
- [ ] CLI can bootstrap a curve from a JSON file.

---

## 15. Phase 13: Advanced American Option Engines

**Duration estimate:** 3–4 weeks
**Depends on:** Phase 5 (vanilla options), Phase 7 (MC/FD/lattice)
**Milestone:** *Can price American options with analytic approximations (BAW, Bjerksund-Stensland) and Longstaff-Schwartz Monte Carlo.*

### 15.1 Analytic Approximation Engines

| Engine | Method | QuantLib Source |
|---|---|---|
| `BaroneAdesiWhaleyEngine` | BAW quadratic approximation for American options | `baroneadesiwhaleyengine.hpp` |
| `BjerksundStenslandEngine` | Bjerksund-Stensland 1993/2002 approximation | `bjerksundstenslandengine.hpp` |
| `JuQuadraticEngine` | Ju quadratic approximation | `juquadraticengine.hpp` |
| `QdPlusAmericanEngine` | QD+ method (high accuracy) | `qdplusamericanengine.hpp` |
| `QdFpAmericanEngine` | QD fixed-point iteration | `qdfpamericanengine.hpp` |

### 15.2 Longstaff-Schwartz MC Engine

- `MCAmericanEngine` — Longstaff-Schwartz regression-based early exercise.
- `LongstaffSchwartzPathPricer` — regression on basis functions.
- `LSMBasisSystem` — polynomial basis (monomial, Laguerre, Hermite, hyperbolic, Chebyshev).
- `GenericLSRegression` — generic least-squares regression.
- `EarlyExercisePathPricer` trait — interface for early exercise decisions.
- `BrownianBridge` — bridge construction for path generation.

### 15.3 Dividend Handling

- `AnalyticDividendEuropeanEngine` — European with discrete dividends (escrowed model).
- `CashDividendEuropeanEngine` — cash dividend model.
- `FdBlackScholesShoutEngine` — FD shout option.
- `DividendSchedule` — schedule of dividend payments.

### 15.4 Digital American Options

- `AnalyticDigitalAmericanEngine` — American digital (cash-or-nothing, asset-or-nothing).
- `MCDigitalEngine` — MC digital option pricing.

### 15.5 Phase 13 Deliverables

- [ ] BAW price within 0.5% of FD benchmark for ATM American put.
- [ ] Bjerksund-Stensland price within 0.3% of FD benchmark.
- [ ] QD+ price within 0.01% of FD benchmark (high accuracy).
- [ ] Longstaff-Schwartz converges to FD price ± 3 standard errors at 100k paths.
- [ ] Early exercise boundary is monotonic and converges with time steps.
- [ ] Benchmark: BAW < 1 μs, LS-MC (10k paths) < 100 ms.

---

## 16. Phase 14: Bates Model & Jump-Diffusion

**Duration estimate:** 2–3 weeks
**Depends on:** Phase 6 (Heston model), Phase 7 (MC/FD)
**Milestone:** *Can calibrate a Bates model (Heston + jumps) and price options with jump-diffusion dynamics.*

### 16.1 Bates Model & Process

- `BatesModel` — extends `HestonModel` with jump parameters (λ, ν, δ).
  - Parameters: v0, κ, θ, σ, ρ (Heston) + λ (jump intensity), ν (jump mean), δ (jump vol).
- `BatesProcess` — extends `HestonProcess` with Merton-style jumps.
  - Compound Poisson process with log-normal jump sizes.
- `BatesDetJumpModel`, `BatesDoubleExpModel`, `BatesDoubleExpDetJumpModel` — variants.

### 16.2 Bates Engines

| Engine | Method | Notes |
|---|---|---|
| `BatesEngine` | Semi-analytic (characteristic function) | Extends `AnalyticHestonEngine` |
| `FdBatesVanillaEngine` | 2D FD + jump integral | `fdbatesvanillaengine.hpp` |
| `MCEuropeanHestonEngine` | MC with Heston dynamics | Path-based |
| `MCHestonHullWhiteEngine` | MC hybrid Heston-HW | Correlated rate + equity |

### 16.3 Merton Jump-Diffusion

- `Merton76Process` — GBM + compound Poisson jumps.
- `JumpDiffusionEngine` — analytic Merton jump-diffusion pricing.

### 16.4 GJR-GARCH Model

- `GjrGarchModel` — asymmetric GARCH volatility model.
- `GjrGarchProcess` — GJR-GARCH stochastic process.
- `AnalyticGjrGarchEngine` — analytic engine.
- `MCEuropeanGjrGarchEngine` — MC engine.

### 16.5 Phase 14 Deliverables

- [ ] Bates characteristic function matches Heston when jump parameters are zero.
- [ ] Bates calibration recovers known parameters from synthetic surface.
- [ ] Bates engine price matches QuantLib to 1e-6.
- [ ] FD Bates matches semi-analytic to 1e-4.
- [ ] Merton jump-diffusion matches analytic formula to 1e-10.
- [ ] GJR-GARCH calibration to historical returns converges.
- [ ] Benchmark: Bates analytic < 500 μs, FD Bates < 5 s.

---

## 17. Phase 15: Multi-Asset & Basket Options

**Duration estimate:** 3–4 weeks
**Depends on:** Phase 6 (processes), Phase 7 (MC)
**Milestone:** *Can price basket options, spread options, and multi-asset correlation products.*

### 17.1 Multi-Asset Infrastructure

- `MultiAssetOption` — base for multi-asset payoffs.
- `BasketOption` — option on a basket of underlyings.
- `StochasticProcessArray` — correlated multi-dimensional process array.
- `MultiPathGenerator` — correlated path generation via Cholesky.
- `BasketPayoff` — MinBasket, MaxBasket, AverageBasket payoff types.

### 17.2 Basket Engines

| Engine | Method | Notes |
|---|---|---|
| `MCEuropeanBasketEngine` | MC simulation | General multi-asset |
| `MCAmericanBasketEngine` | MC + Longstaff-Schwartz | American basket |
| `StulzEngine` | Analytic 2-asset max/min | Stulz formula |
| `SingleFactorBSMBasketEngine` | Moment-matching approximation | Fast approximate |
| `DengLiZhouBasketEngine` | Deng-Li-Zhou expansion | Higher-order approximation |
| `ChoiBasketEngine` | Choi et al. expansion | Accurate for many assets |
| `Fd2dBlackScholesVanillaEngine` | 2D FD grid | Two correlated underlyings |
| `FdNDimBlackScholesVanillaEngine` | N-dimensional FD | General N-asset |

### 17.3 Spread Options

| Engine | Spread Type | Notes |
|---|---|---|
| `KirkEngine` | Kirk approximation | 2-asset spread |
| `BjerksundStenslandSpreadEngine` | Bjerksund-Stensland spread | Improved approximation |
| `OperatorSplittingSpreadEngine` | Operator splitting | FD-based |
| `SpreadBlackScholesVanillaEngine` | BS spread | Closed-form for special cases |

### 17.4 Exchange Options

- `MargrabeOption` — exchange one asset for another.
- `AnalyticEuropeanMargrabeEngine` — Margrabe formula (closed-form).
- `AnalyticAmericanMargrabeEngine` — American exercise approximation.

### 17.5 Phase 15 Deliverables

- [ ] 2-asset basket MC price converges to Stulz formula for max-option.
- [ ] Spread option Kirk price matches QuantLib to 1e-6.
- [ ] Margrabe formula matches Black-Scholes at special parameters.
- [ ] 5-asset basket MC with antithetic variates: std error < 0.1% at 100k paths.
- [ ] 2D FD spread matches MC to 1e-3.
- [ ] Benchmark: 5-asset MC basket (100k paths) < 2 s.

---

## 18. Phase 16: Short-Rate Models — Full Coverage

**Duration estimate:** 4–5 weeks
**Depends on:** Phase 6 (HullWhite base), Phase 7 (lattice/FD)
**Milestone:** *Complete short-rate model suite: CIR, Vasicek, Black-Karasinski, GSR, G2, Markov Functional.*

### 18.1 One-Factor Affine Models

| Model | Parameters | Notes |
|---|---|---|
| `Vasicek` | a (mean reversion), b (long-run), σ | Classic Vasicek |
| `CoxIngersollRoss` | θ (speed), μ (level), σ | CIR square-root |
| `ExtendedCoxIngersollRoss` | Time-dependent CIR | Calibrated to curve |
| `BlackKarasinski` | a(t), σ(t) | Log-normal short rate |

### 18.2 Gaussian 1D Framework

- `Gaussian1DModel` trait — base for one-factor Gaussian models.
- `GSR` — Gaussian Short Rate model with piecewise constant volatilities.
- `GSRProcess`, `GSRProcessCore` — associated stochastic processes.
- `MarkovFunctional` — Markov functional model (calibrated to swaptions/caplets).
- `MfStateProcess` — state process for Markov functional.

### 18.3 Two-Factor Models

- `G2Model` — two-factor additive Gaussian model.
  - Parameters: a, σ, b, η, ρ (5 parameters).
- `G2Process` — correlated two-factor Gaussian process.
- `TwoFactorModel` trait — base trait for two-factor short-rate models.

### 18.4 CIR Process

- `CoxIngersollRossProcess` — square-root diffusion process.
- `SquareRootProcess` — general square-root process.

### 18.5 Volatility Estimation

- `ConstantEstimator` — constant vol from historical data.
- `GarchEstimator` — GARCH(1,1) vol estimation.
- `GarmanKlass` — Garman-Klass range-based estimator.
- `SimpleLocalEstimator` — simple local vol estimator.

### 18.6 Calibration Helpers (Extended)

- `SwaptionHelper` — calibrate short-rate models to swaption market.
- `CapHelper` — calibrate to cap/floor market.

### 18.7 Phase 16 Deliverables

- [ ] Vasicek bond price matches analytic formula to 1e-10.
- [ ] CIR bond price matches analytic to 1e-8.
- [ ] Black-Karasinski calibrated to swaption grid, reprices within 1 bp.
- [ ] GSR calibrated model reprices caplets within 0.5 bp.
- [ ] G2 model calibrated to swaption matrix, reprices within 2 bp.
- [ ] Markov functional calibrated to coterminal swaptions.
- [ ] Short-rate tree discount factors match curve-implied to 1e-6.
- [ ] Benchmark: Vasicek bond price < 100 ns, G2 swaption < 10 ms.

---

## 19. Phase 17: Advanced Swaption & Cap/Floor Engines

**Duration estimate:** 3–4 weeks
**Depends on:** Phase 16 (short-rate models), Phase 8 (vol surfaces)
**Milestone:** *Full swaption/cap-floor engine suite: Gaussian 1D, Jamshidian, FD Hull-White/G2, tree-based.*

### 19.1 Gaussian 1D Engines

| Engine | Instrument | Notes |
|---|---|---|
| `Gaussian1DSwaptionEngine` | Swaption | Numerical integration |
| `Gaussian1DJamshidianSwaptionEngine` | Swaption | Jamshidian decomposition |
| `Gaussian1DNonstandardSwaptionEngine` | Nonstandard swaption | Bermudan + amortizing |
| `Gaussian1DFloatFloatSwaptionEngine` | Float-float swaption | CMS spread |
| `Gaussian1DCapFloorEngine` | Cap/Floor | Numerical integration |

### 19.2 Analytic & Semi-Analytic Engines

- `JamshidianSwaptionEngine` — Jamshidian trick for coupon-bond options.
- `G2SwaptionEngine` — analytic G2 swaption pricing.
- `AnalyticCapFloorEngine` — Hull-White analytic cap/floor.
- `BachelierSwaptionEngine` — normal (Bachelier) model swaption pricing.
- `BachelierCapFloorEngine` — normal model cap/floor pricing.
- `BasketGeneratingEngine` — generates calibration baskets for Bermudan swaptions.

### 19.3 FD Swaption Engines

- `FdHullWhiteSwaptionEngine` — 1D FD under Hull-White.
- `FdG2SwaptionEngine` — 2D FD under G2 two-factor model.

### 19.4 Tree Engines (Extended)

- `TreeSwaptionEngine` — generic short-rate tree swaption.
- `TreeCapFloorEngine` — tree-based cap/floor pricing.
- `TreeSwapEngine` — tree-based swap pricing.
- `MCHullWhiteEngine` — MC simulation under Hull-White for cap/floor.
- `LatticeShortRateModelEngine` — generic lattice engine for short-rate models.

### 19.5 Nonstandard & Irregular Instruments

- `NonstandardSwap`, `NonstandardSwaption` — amortizing/step-up notional.
- `IrregularSwap`, `IrregularSwaption` — irregular schedules.
- `HaganIrregularSwaptionEngine` — Hagan's method for irregular swaptions.

### 19.6 Phase 17 Deliverables

- [ ] Gaussian1D swaption matches Black swaption at flat vol to 1e-6.
- [ ] Jamshidian decomposition matches tree engine to 1e-4.
- [ ] FD Hull-White Bermudan swaption matches tree to 1e-3.
- [ ] G2 swaption analytic matches FD to 1e-4.
- [ ] Bachelier cap/floor matches Black at zero-rate limit.
- [ ] MC Hull-White cap converges to analytic ± 3 std errors.
- [ ] Nonstandard swaption with amortizing notional prices correctly.
- [ ] Benchmark: Jamshidian swaption < 1 ms, FD HW Bermudan < 500 ms.

---

## 20. Phase 18: Advanced Volatility Surfaces & Smile Models

**Duration estimate:** 4–5 weeks
**Depends on:** Phase 8 (SABR base), Phase 6 (Heston)
**Milestone:** *Complete vol surface infrastructure: SVI, ZABR, NoArb-SABR, Andreasen-Huge, Dupire local vol, full swaption vol cube.*

### 20.1 Equity/FX Volatility Surfaces

| Type | Description | QuantLib Source |
|---|---|---|
| `BlackVarianceSurface` | Interpolated strike × expiry grid | `blackvariancesurface.hpp` |
| `BlackVarianceCurve` | ATM vol term structure | `blackvariancecurve.hpp` |
| `ImpliedVolTermStructure` | Implied from another surface | `impliedvoltermstructure.hpp` |
| `HestonBlackVolSurface` | Black vol derived from Heston params | `hestonblackvolsurface.hpp` |
| `LocalVolSurface` | Dupire local vol from Black vol | `localvolsurface.hpp` |
| `LocalVolCurve` | 1D local vol | `localvolcurve.hpp` |
| `LocalConstantVol` | Constant local vol | `localconstantvol.hpp` |
| `FixedLocalVolSurface` | Parameterized local vol | `fixedlocalvolsurface.hpp` |
| `GridModelLocalVolSurface` | Grid-based local vol | `gridmodellocalvolsurface.hpp` |
| `NoExceptLocalVolSurface` | Exception-safe local vol wrapper | `noexceptlocalvolsurface.hpp` |

### 20.2 Advanced Smile Models

| Model | Description | QuantLib Source |
|---|---|---|
| SVI | Stochastic Volatility Inspired parameterization | `sviinterpolation.hpp` |
| `SviSmileSection`, `SviInterpolatedSmileSection` | SVI per-expiry smile | `svismilesection.hpp` |
| ZABR | Generalized SABR with CEV backbone | `zabr.hpp`, `zabrinterpolation.hpp` |
| `ZabrSmileSection`, `ZabrInterpolatedSmileSection` | ZABR smile sections | `zabrsmilesection.hpp` |
| NoArb-SABR | Arbitrage-free SABR (Hagan et al.) | `noarbsabr.hpp` |
| `NoArbSabrSmileSection`, `NoArbSabrInterpolatedSmileSection` | NoArb-SABR smile | `noarbsabrsmilesection.hpp` |
| `KahaleSmileSection` | Kahale monotonic convex interpolation | `kahalesmilesection.hpp` |
| `InterpolatedSmileSection` | Generic interpolated smile | `interpolatedsmilesection.hpp` |
| `SpreadedSmileSection` | Smile with additive spread | `spreadedsmilesection.hpp` |
| `AtmAdjustedSmileSection`, `AtmSmileSection` | ATM-normalized smile | `atmadjustedsmilesection.hpp` |
| `FlatSmileSection` | Constant vol smile section | `flatsmilesection.hpp` |

### 20.3 Andreasen-Huge Volatility Interpolation

- `AndersenHugeVolatilityInterpl` — Andreasen-Huge piecewise-linear local vol.
- `AndersenHugeVolatilityAdapter` — wraps interpolation as Black vol surface.
- `AndersenHugeLocalVolAdapter` — wraps as local vol surface.
- Produces arbitrage-free interpolation from discrete option prices.

### 20.4 Swaption Volatility Cube

| Type | Description |
|---|---|
| `SwaptionVolMatrix` | ATM swaption vol (option tenor × swap tenor) |
| `SwaptionVolCube` | Full cube: ATM matrix + smile per point |
| `InterpolatedSwaptionVolatilityCube` | Smile via SABR interpolation |
| `SabrSwaptionVolatilityCube` | SABR-calibrated per-expiry smile |
| `NoArbSabrSwaptionVolatilityCube` | Arbitrage-free SABR cube |
| `SwaptionConstantVol` | Constant swaption vol |
| `SpreadedSwaptionVol` | Spreaded swaption vol |
| `Gaussian1DSwaptionVolatility` | Swaption vol from Gaussian 1D model |
| `SwaptionVolDiscrete` | Discrete swaption vol data |

### 20.5 Cap/Floor Volatility Structures

- `CapFloorTermVolatilityStructure` — base trait.
- `CapFloorTermVolCurve` — flat vol per tenor.
- `CapFloorTermVolSurface` — strike × tenor surface.
- `ConstantCapFloorTermVol` — constant cap/floor vol.
- `OptionletVolatilityStructure` — base for optionlet vols.
- `ConstantOptionletVol`, `SpreadedOptionletVol`.
- `OptionletStripper1`, `OptionletStripper2` — strip caps into optionlets.
- `StrippedOptionlet`, `StrippedOptionletAdapter`, `StrippedOptionletBase`.
- `CapletVarianceCurve` — caplet variance term structure.

### 20.6 Inflation Volatility

- `CPIVolatilityStructure`, `ConstantCPIVolatility`.
- `YoYInflationOptionletVolatilityStructure`.

### 20.7 ABCD Volatility

- `AbcdFunction` — ABCD parametric vol function a(t) = (a + b·t)·exp(-c·t) + d.
- `AbcdCalibration` — calibration to market data.
- `AbcdInterpolation` — ABCD-based interpolation.

### 20.8 CMS Market & Calibration

- `CmsMarket` — CMS swap market quotes.
- `CmsMarketCalibration` — calibrate CMS model to market.

### 20.9 Stochastic Local Volatility (SLV)

- `HestonSLVFDMModel` — Heston stochastic local vol via FD.
- `HestonSLVMCModel` — Heston SLV via Monte Carlo.
- `HestonSLVProcess` — process with leverage function.
- `PiecewiseTimeDependentHestonModel` — time-dependent Heston params.

### 20.10 Phase 18 Deliverables

- [ ] SVI smile matches QuantLib's calibration to 1e-8.
- [ ] ZABR smile matches QuantLib to 1e-6.
- [ ] NoArb-SABR produces arbitrage-free density.
- [ ] Andreasen-Huge interpolation exact at input points.
- [ ] Swaption vol cube queried at arbitrary strike/expiry/tenor.
- [ ] Optionlet stripping from cap vols matches QuantLib.
- [ ] SLV leverage function calibrated from local/stochastic vol.
- [ ] Benchmark: SABR calibration < 1 ms, SVI fit < 500 μs.

---

## 21. Phase 19: Advanced Finite Difference Framework

**Duration estimate:** 5–6 weeks
**Depends on:** Phase 7 (basic FD), Phase 6 (Heston), Phase 16 (short-rate models)
**Milestone:** *Production-grade multi-dimensional FD framework with all QuantLib meshers, operators, schemes, solvers, and step conditions.*

### 21.1 FDM Meshers

| Mesher | Description |
|---|---|
| `Fdm1DMesher` trait | Base 1D mesher interface |
| `Uniform1DMesher` | Uniform grid |
| `Concentrating1DMesher` | Grid with concentration around a point (strikes) |
| `FdmBlackScholesMesher` | Log-spot mesher for BS |
| `FdmBlackScholesMultiStrikeMesher` | Multi-strike concentrating mesher |
| `FdmHestonVarianceMesher` | Variance mesher for Heston |
| `FdmSimpleProcess1DMesher` | Generic 1D process mesher |
| `Predefined1DMesher` | User-supplied grid points |
| `ExponentialJump1DMesher` | Mesher for jump-diffusion |
| `FdmCEV1DMesher` | CEV model mesher |
| `FdmMesher` trait | Multi-dimensional mesher |
| `FdmMesherComposite` | Composite N-dimensional mesher |
| `UniformGridMesher` | N-dimensional uniform grid |

### 21.2 FDM Operators

| Operator | Description |
|---|---|
| `FdmLinearOp` trait | Base linear operator |
| `TripleBandLinearOp` | Tridiagonal operator |
| `NinePointLinearOp` | 9-point 2D stencil |
| `FirstDerivativeOp` | First derivative operator |
| `SecondDerivativeOp` | Second derivative operator |
| `SecondOrderMixedDerivativeOp` | Cross-derivative for 2D |
| `NthOrderDerivativeOp` | Arbitrary-order derivative |
| `ModTripleBandLinearOp` | Modified tridiagonal |
| `FdmLinearOpComposite` trait | Composite operator (ADI splitting) |
| `FdmBlackScholesOp` | 1D BS operator |
| `Fdm2dBlackScholesOp` | 2D correlated BS operator |
| `FdmHestonOp` | 2D Heston (spot × variance) |
| `FdmHestonFwdOp` | Forward Heston PDE operator |
| `FdmHestonHullWhiteOp` | 3D Heston-HW hybrid operator |
| `FdmBatesOp` | Heston + jump integral |
| `FdmHullWhiteOp` | Hull-White short-rate operator |
| `FdmG2Op` | G2 two-factor rate operator |
| `FdmCIROp` | CIR process operator |
| `FdmCEVOp` | CEV model operator |
| `FdmSABROp` | SABR model operator |
| `FdmOrnsteinUhlenbeckOp` | OU process operator |
| `FdmBlackScholesFwdOp` | Forward BS PDE |
| `FdmLocalVolFwdOp` | Forward local vol PDE |
| `FdmSquareRootFwdOp` | Forward square-root (CIR) PDE |
| `FdmWienerOp` | Wiener process operator |
| `FdmLinearOpLayout` | Grid layout for N-dimensional problems |
| `FdmLinearOpIterator` | Iterator over grid points |

### 21.3 FDM Schemes (Time-Stepping)

| Scheme | Description |
|---|---|
| `CrankNicolsonScheme` | θ = 0.5 implicit-explicit |
| `DouglasScheme` | Douglas ADI splitting |
| `HundsdorferScheme` | Hundsdorfer ADI |
| `CraigSneydScheme` | Craig-Sneyd ADI |
| `ModifiedCraigSneydScheme` | Modified Craig-Sneyd |
| `ExplicitEulerScheme` | Forward Euler |
| `ImplicitEulerScheme` | Backward Euler |
| `MethodOfLinesScheme` | Method of lines |
| `TRBDF2Scheme` | TR-BDF2 (high-order) |
| `MixedScheme` | Weighted combination |
| `BoundaryConditionSchemeHelper` | BC handling for all schemes |

### 21.4 FDM Solvers

| Solver | Description |
|---|---|
| `Fdm1DimSolver` | 1D PDE solver |
| `Fdm2DimSolver` | 2D PDE solver |
| `Fdm3DimSolver` | 3D PDE solver |
| `FdmNDimSolver` | General N-dimensional solver |
| `FdmBackwardSolver` | Backward-in-time solver |
| `FdmBlackScholesSolver` | Specialized BS solver |
| `FdmHestonSolver` | Specialized Heston solver |
| `FdmHestonHullWhiteSolver` | 3D hybrid solver |
| `FdmBatesSolver` | Bates model solver |
| `FdmHullWhiteSolver` | HW rate solver |
| `FdmG2Solver` | G2 two-factor solver |
| `FdmCIRSolver` | CIR model solver |
| `Fdm2dBlackScholesSolver` | 2D correlated BS |
| `FdmSimple2DBSSolver` | Simplified 2D BS solver |
| `FdmSolverDesc` | Solver configuration descriptor |

### 21.5 Step Conditions

| Condition | Description |
|---|---|
| `FdmAmericanStepCondition` | American exercise boundary |
| `FdmBermudanStepCondition` | Bermudan exercise dates |
| `FdmArithmeticAverageCondition` | Asian averaging |
| `FdmSimpleStorageCondition` | Storage/swing option |
| `FdmSimpleSwingCondition` | Swing option exercise |
| `FdmSnapshotCondition` | Snapshot for Greeks |
| `FdmStepConditionComposite` | Composite conditions |

### 21.6 FDM Utilities

- `FdmBoundaryConditionSet` — Dirichlet, discount, time-dependent boundaries.
- `FdmDirichletBoundary`, `FdmDiscountDirichletBoundary`, `FdmTimeDependentDirichletBoundary`.
- `FdmDividendHandler` — discrete dividend handling in FD.
- `FdmInnerValueCalculator` trait + implementations.
- `FdmQuantoHelper` — quanto adjustment in FD.
- `FdmAffineModelTermStructure`, `FdmAffineModelSwapInnerValue`.
- Risk-neutral density calculators: `BSMRndCalculator`, `HestonRndCalculator`, `LocalVolRndCalculator`, `CEVRndCalculator`, `GBSMRndCalculator`, `SquareRootProcessRndCalculator`.
- `FdmMesherIntegral` — numerical integration over mesher.
- `FdmIndicesOnBoundary` — boundary index management.
- `FdmHestonGreensFct` — Green's function for Heston.

### 21.7 Additional FD Engines

| Engine | Model | Notes |
|---|---|---|
| `FdHestonVanillaEngine` | Heston | 2D PDE American/European |
| `FdHestonHullWhiteVanillaEngine` | Heston-HW hybrid | 3D PDE |
| `FdCEVVanillaEngine` | CEV | 1D PDE |
| `FdCIRVanillaEngine` | CIR | 1D PDE |
| `FdSABRVanillaEngine` | SABR | 2D PDE |
| `FdSimpleBSSwingEngine` | BS | Swing option |
| `FdMultiPeriodEngine` | Generic | Multi-exercise periods |

### 21.8 Phase 19 Deliverables

- [ ] Concentrating mesher produces denser grid around strike.
- [ ] Heston 2D FD matches analytic Heston to 1e-4 on 100×100 grid.
- [ ] Douglas ADI matches Crank-Nicolson to 1e-3 for 2D problems.
- [ ] 3D Heston-HW FD runs successfully on 50×50×50 grid.
- [ ] American exercise boundary is correct (early exercise premium > 0).
- [ ] Bermudan swaption FD matches tree to 1e-3.
- [ ] All schemes converge at expected order (CN: 2nd order, explicit: 1st order).
- [ ] Benchmark: 1D FD (200 grid) < 10 ms, 2D FD (100×100) < 1 s.

---

## 22. Phase 20: LIBOR Market Model Framework

**Duration estimate:** 6–8 weeks
**Depends on:** Phase 7 (MC), Phase 8 (vol surfaces), Phase 16 (short-rate models)
**Milestone:** *Complete LIBOR Market Model (BGM) framework: curve states, evolvers, multi-step products, calibration, pathwise Greeks.*

### 22.1 Market Model Infrastructure

- `MarketModel` trait — base trait for market models.
- `EvolutionDescription` — time grid, rate tenors, alive rates.
- `CurveState` trait — state of the yield curve at each time step.
  - `LMMCurveState` — forward LIBOR measure.
  - `CoterminalSwapCurveState` — coterminal swap measure.
  - `CMSwapCurveState` — constant maturity swap measure.
- `Discounter` — path-based discounting.

### 22.2 Brownian Generators

- `BrownianGenerator` trait — interface for Brownian increments.
- `MTBrownianGenerator` — Mersenne Twister based.
- `SobolBrownianGenerator` — Sobol quasi-random.
- Bridge construction variants for variance reduction.

### 22.3 Drift Computation

- `LMMDriftCalculator` — log-normal LMM drift.
- `LMMNormalDriftCalculator` — normal LMM drift.
- `SMMDriftCalculator` — swap market model drift.
- `CmSMMDriftCalculator` — CMS market model drift.

### 22.4 Evolvers

| Evolver | Measure | Scheme |
|---|---|---|
| `LogNormalFwdRatePC` | Forward | Predictor-corrector |
| `LogNormalFwdRateEuler` | Forward | Euler |
| `LogNormalFwdRateBalland` | Forward | Balland-Tran |
| `LogNormalFwdRateEulerConstrained` | Forward | Constrained Euler |
| `LogNormalFwdRateIPC` | Forward | Iterative PC |
| `LogNormalFwdRateIBalland` | Forward | Iterative Balland |
| `LogNormalCotSwapRatePC` | Coterminal swap | Predictor-corrector |
| `LogNormalCMSwapRatePC` | CMS | Predictor-corrector |
| `NormalFwdRatePC` | Forward (normal) | Predictor-corrector |
| `SvdDFwdRatePC` | Forward (SVD) | SVD-based |

### 22.5 Market Model Volatility

- `AbcdVol` — ABCD parameterization for forward rate vols.
- `FlatVol` — constant vol for all rates.
- `PiecewiseConstantVariance` — piecewise constant per rate.
- `PiecewiseConstantAbcdVariance` — ABCD with piecewise parameters.
- `VolatilityInterpolationSpecifier`, `VolatilityInterpolationSpecifierAbcd`.
- `MarketModelVolProcess` — stochastic vol on rates (Andersen's QE).
- `SquareRootAndersen` — square-root vol-of-vol process.

### 22.6 Correlations

- `PiecewiseConstantCorrelation` — base correlation structure.
- `TimeHomogeneousForwardCorrelation` — time-homogeneous forward rate correlation.
- `ExponentialCorrelations` — exponentially decaying correlation.
- `CotSwapFromFwdCorrelation` — derive coterminal from forward.

### 22.7 Multi-Step Products

| Product | Description |
|---|---|
| `MultiStepSwap` | Generic multi-step swap |
| `MultiStepSwaption` | Multi-step Bermudan swaption |
| `MultiStepOptionlets` | Sequence of caplets |
| `MultiStepForwards` | Forward rate agreements |
| `MultiStepCoterminalSwaps` | Coterminal swap portfolio |
| `MultiStepCoterminalSwaptions` | Coterminal swaption |
| `MultiStepCoinitialSwaps` | Coinitial swaps |
| `MultiStepInverseFloater` | Inverse floater |
| `MultiStepRatchet` | Ratchet product |
| `MultiStepTarn` | Target redemption note |
| `MultiStepNothing` | Null product (benchmarking) |
| `MultiStepPeriodCapletSwaptions` | Period caplet/swaption |
| `CallSpecifiedMultiProduct` | Callable multi-step |
| `ExerciseAdapter` | Early exercise adapter |
| `CashRebate` | Cash rebate on exercise |

### 22.8 One-Step Products

- `OneStepForwards`, `OneStepOptionlets`.
- `OneStepCoterminalSwaps`, `OneStepCoinitialSwaps`.

### 22.9 Accounting & Greeks

- `AccountingEngine` — pathwise P&L accumulation.
- `PathwiseAccountingEngine` — pathwise with AAD.
- `PathwiseDiscounter` — pathwise discount factors.
- `ProxyGreekEngine` — proxy Greeks via bump-and-revalue.
- Pathwise Greeks: `BumpInstrumentJacobian`, `RatePseudoRootJacobian`, `SwaptionPseudoJacobian`, `VegaBumpCluster`.

### 22.10 Calibration

- `CapletCoterminalAlphaCalibration` — alpha calibration.
- `CapletCoterminalMaxHomogeneity` — max homogeneity calibration.
- `CapletCoterminalPeriodic` — periodic calibration.
- `CapletCoterminalSwaptionCalibration` — joint caplet/swaption calibration.
- `CTSMMCapletCalibration` — CTSMM caplet calibration.
- `CotSwapToFwdAdapter`, `FwdToCotSwapAdapter`, `FwdPeriodAdapter` — model adapters.
- `PseudoRootFacade` — pseudo-root access.
- `AlphaFinder`, `AlphaForm`, `AlphaFormConcrete` — alpha shape functions.
- `ForwardForwardMappings`, `SwapForwardMappings` — mapping utilities.
- `HistoricalForwardRatesAnalysis`, `HistoricalRatesAnalysis` — historical analysis.

### 22.11 Callability (Bermudan)

- `ExerciseValue` trait, `BermudanSwaptionExerciseValue`.
- `LSStrategy` — Longstaff-Schwartz strategy.
- `UpperBoundEngine` — Andersen upper bound.
- `MarketModelBasisSystem`, `MarketModelParametricExercise`.
- `SwapBasisSystem`, `SwapForwardBasisSystem`.
- `SwapRateTrigger`, `TriggeredSwapExercise`.
- `CollectNodeData`, `NodeDataProvider`.
- `ParametricExerciseAdapter`, `NothingExerciseValue`.

### 22.12 Phase 20 Deliverables

- [ ] LMM forward rate evolution preserves drift condition.
- [ ] Coterminal swaption prices match Black swaption prices at flat vol.
- [ ] Predictor-corrector evolver converges to exact at fine time steps.
- [ ] Caplet calibration reproduces market caplet vols to 0.5 bp.
- [ ] Bermudan swaption with LS regression produces sensible exercise boundary.
- [ ] Pathwise vega matches bump-and-revalue to 5%.
- [ ] Multi-step swap NPV matches discounting engine to 1e-4.
- [ ] Example: `calibrate_lmm.rs` — full LMM calibration + Bermudan swaption.
- [ ] Benchmark: LMM 10k paths, 40 rates, 10 steps < 5 s.

---

## 23. Phase 21: Advanced Credit Models

**Duration estimate:** 4–5 weeks
**Depends on:** Phase 9 (basic credit), Phase 7 (MC), Phase 22 (math: copulas)
**Milestone:** *CDO tranches, nth-to-default baskets, CDS options, portfolio credit models with copulas.*

### 23.1 Portfolio Credit Instruments

| Instrument | Description |
|---|---|
| `SyntheticCDO` | Synthetic CDO tranche |
| `NthToDefault` | n-th to default basket |
| `CDSOption` | Option on a CDS |
| `Basket` | Credit basket (pool of names) |

### 23.2 Credit Engines

| Engine | Description |
|---|---|
| `IntegralCDOEngine` | Numerical integration CDO pricing |
| `MidpointCDOEngine` | Midpoint integration CDO |
| `IntegralNtdEngine` | Nth-to-default engine |
| `IsdaCdsEngine` | ISDA standard CDS pricing |
| `IntegralCdsEngine` | Numerical integration CDS |
| `BlackCdsOptionEngine` | Black model CDS option |

### 23.3 Default Loss Models

| Model | Description |
|---|---|
| `DefaultLossModel` trait | Base trait for portfolio loss |
| `GaussianLHPLossModel` | Large homogeneous portfolio (Vasicek) |
| `BinomialLossModel` | Binomial loss distribution |
| `RecursiveLossModel` | Recursive (Andersen-Sidenius) |
| `SaddlepointLossModel` | Saddlepoint approximation |
| `RandomDefaultLossModel` | MC default simulation |

### 23.4 Copula Models

| Model | Description |
|---|---|
| `OneFactorGaussianCopula` | Standard Gaussian copula |
| `OneFactorStudentCopula` | Student-t copula |
| `OneFactorAffineSurvival` | Affine survival |
| `DefaultProbabilityLatentModel` | Latent factor model |
| `ConstantLossLatentModel` | Constant LGD latent model |
| `SpotLossLatentModel` | Spot loss latent model |
| `RandomLossLatentModel` | Random loss latent model |
| `RandomDefaultLatentModel` | Random default latent model |

### 23.5 Base Correlation

- `BaseCorrelationStructure` — base correlation term structure.
- `BaseCorrelationLossModel` — tranche pricing via base correlation.
- `CorrelationStructure` — general correlation interface.

### 23.6 Credit Utilities

- `Pool` — pool of issuers.
- `Issuer` — individual issuer with default curves.
- `DefaultEvent`, `DefaultType`, `DefaultProbabilityKey`.
- `RecoveryRateModel`, `RecoveryRateQuote`.
- `Loss`, `LossDistribution`.
- `HomogeneousPoolDef`, `InhomogeneousPoolDef` — pool definitions.
- `Claim` — recovery claim type.
- `FactorSpreadedhazardRateCurve`, `SpreadedhazardRateCurve`.
- `InterpolatedAffineHazardRateCurve`.
- `RiskyAssetSwap`, `RiskyAssetSwapOption` — credit-risky instruments.

### 23.7 Phase 21 Deliverables

- [ ] Gaussian copula CDO tranche matches QuantLib to 1e-4.
- [ ] Nth-to-default basket matches MC to 1e-3.
- [ ] ISDA CDS engine matches standard ISDA calculator.
- [ ] CDS option Black model matches QuantLib to 1e-6.
- [ ] Base correlation term structure interpolates correctly.
- [ ] MC default simulation converges to semi-analytic at 100k paths.
- [ ] Benchmark: Gaussian copula CDO (125 names, 7 tranches) < 2 s.

---

## 24. Phase 22: Math Library Extensions

**Duration estimate:** 4–5 weeks
**Depends on:** Phase 2 (math base)
**Milestone:** *Complete math library parity: all interpolations, distributions, copulas, optimizers, statistics, RNG, and matrix utilities.*

### 24.1 Additional Interpolations

| Interpolation | Description |
|---|---|
| `BackwardFlatInterpolation` | Backward-flat step function |
| `ForwardFlatInterpolation` | Forward-flat step function |
| `SABRInterpolation` | SABR-based smile interpolation |
| `XABRInterpolation` | Generalized XABR framework |
| `ABCDInterpolation` | ABCD parametric vol interpolation |
| `LagrangeInterpolation` | Lagrange polynomial |
| `ChebyshevInterpolation` | Chebyshev polynomial |
| `ConvexMonotoneInterpolation` | Monotone-convex preserving |
| `MixedInterpolation` | Hybrid interpolation |
| `KernelInterpolation` | Kernel (RBF) interpolation |
| `BSpline` | B-spline basis functions |
| `BilinearInterpolation` | 2D bilinear |
| `BicubicSplineInterpolation` | 2D bicubic spline |
| `KernelInterpolation2D` | 2D kernel interpolation |
| `BackwardFlatLinearInterpolation` | 2D backward-flat + linear |
| `FlatExtrapolation2D` | 2D flat extrapolation |
| `MultiCubicSpline` | N-dimensional cubic spline |
| `Interpolation2D` trait | Base 2D interpolation |

### 24.2 Additional Distributions

| Distribution | Description |
|---|---|
| `BinomialDistribution` | Binomial CDF/PDF |
| `PoissonDistribution` | Poisson CDF/PDF |
| `GammaDistribution` | Gamma CDF/PDF |
| `ChiSquareDistribution` | Chi-square CDF/PDF |
| `StudentTDistribution` | Student-t CDF/PDF |
| `BivariateNormalDistribution` | 2D cumulative normal |
| `BivariateStudentTDistribution` | 2D cumulative Student-t |

### 24.3 Copulas

| Copula | Description |
|---|---|
| `GaussianCopula` | Standard Gaussian |
| `ClaytonCopula` | Clayton (lower tail dependence) |
| `FrankCopula` | Frank (symmetric) |
| `GumbelCopula` | Gumbel (upper tail dependence) |
| `PlackettCopula` | Plackett |
| `AliMikhailHaqCopula` | Ali-Mikhail-Haq |
| `FarlieGumbelMorgensternCopula` | FGM |
| `GalambosCopula` | Galambos |
| `HuslerReissCopula` | Husler-Reiss |
| `MarshallOlkinCopula` | Marshall-Olkin |
| `IndependentCopula` | Product copula |
| `MaxCopula`, `MinCopula` | Fréchet bounds |

### 24.4 Additional Optimizers

| Optimizer | Description |
|---|---|
| `BFGS` | Broyden-Fletcher-Goldfarb-Shanno |
| `ConjugateGradient` | Nonlinear conjugate gradient |
| `SteepestDescent` | Gradient descent |
| `DifferentialEvolution` | Evolutionary global optimizer |
| `SimulatedAnnealing` | Stochastic global optimizer |
| `ArmijoLineSearch` | Armijo step-size rule |
| `GoldsteinLineSearch` | Goldstein conditions |
| `SphereCylinder` | Constrained optimization on sphere/cylinder |
| `ProjectedCostFunction` | Projected optimization (subset of parameters) |
| `ProjectedConstraint` | Projected constraints |

### 24.5 Additional Solvers

| Solver | Description |
|---|---|
| `Secant` | Secant method |
| `Ridder` | Ridder's method |
| `FalsePosition` | False position (regula falsi) |
| `Halley` | Halley's method (cubic convergence) |
| `NewtonSafe` | Newton with bisection safeguard |
| `FiniteDifferenceNewtonSafe` | Newton-safe with FD Jacobian |

### 24.6 Statistics

| Type | Description |
|---|---|
| `GeneralStatistics` | Mean, variance, skewness, kurtosis, percentiles |
| `IncrementalStatistics` | Online (streaming) statistics |
| `GaussianStatistics` | Extended with Gaussian-based metrics |
| `RiskStatistics` | VaR, CVaR, shortfall, average shortfall |
| `SequenceStatistics` | Multi-variate statistics (correlation matrix) |
| `ConvergenceStatistics` | Track convergence (for MC) |
| `DiscrepancyStatistics` | Low-discrepancy sequence quality |
| `Histogram` | Histogram accumulator |

### 24.7 Additional Random Number Generators

| Generator | Description |
|---|---|
| `Xoshiro256StarStarUniformRng` | Xoshiro256** (fast, modern) |
| `KnuthUniformRng` | Knuth's subtractive RNG |
| `LecuyerUniformRng` | L'Ecuyer combined RNG |
| `RanluxUniformRng` | RANLUX luxury random |
| `HaltonRsg` | Halton quasi-random sequence |
| `FaureRsg` | Faure quasi-random sequence |
| `LatticeRsg` | Lattice rules |
| `RandomizedLDS` | Randomized low-discrepancy |
| `SobolBrownianBridgeRsg` | Sobol + Brownian bridge |
| `Burley2020SobolRsg` | Burley (2020) scrambled Sobol |
| `BoxMullerGaussianRng` | Box-Muller normal transform |
| `ZigguratGaussianRng` | Ziggurat normal (very fast) |
| `CentralLimitGaussianRng` | CLT-based normal |
| `StochasticCollocationInvCDF` | Stochastic collocation CDF |

### 24.8 Matrix Utilities

| Utility | Description |
|---|---|
| `CholeskyDecomposition` | Lower-triangular Cholesky |
| `SVD` | Singular value decomposition |
| `QRDecomposition` | QR decomposition |
| `SymmetricSchurDecomposition` | Symmetric eigendecomposition |
| `TQREigenDecomposition` | Tridiagonal QR eigen |
| `PseudoSqrt` | Pseudo-square root (Spectral, Salvaging, Hypersphere) |
| `MatrixExponential` (expm) | Matrix exponential |
| `Householder` | Householder transformation |
| `GetCovariance` | Covariance from vol + correlation |
| `BasisIncompleteOrdered` | Incomplete basis |
| `FactorReduction` | Factor reduction |
| `BiCGstab` | Bi-conjugate gradient stabilized |
| `GMRES` | Generalized minimal residual |
| `SparseMatrix` | Sparse matrix type |
| `SparseILUPreconditioner` | ILU(0) preconditioner |
| `TAPCorrelations` | TAP correlation structure |

### 24.9 Additional Math Functions

- `FastFourierTransform` — FFT for option pricing (Carr-Madan).
- `RichardsonExtrapolation` — convergence acceleration.
- `GeneralLinearLeastSquares`, `LinearLeastSquaresRegression` — regression.
- `BernsteinPolynomial` — Bernstein basis polynomials.
- `ModifiedBessel` — modified Bessel functions (I, K).
- `Beta` function, `IncompletGamma` function.
- `Factorial`, `PascalTriangle`, `PrimeNumbers`.
- `Rounding` — decimal rounding modes.
- `Autocovariance` — autocovariance function.
- `KernelFunctions` — Gaussian, Epanechnikov kernels.
- `AdaptiveRungeKutta` — ODE solver.
- `Quadratic` — quadratic equation solver.
- `TransformedGrid` — coordinate transforms.

### 24.10 Additional Integrals

| Integral | Description |
|---|---|
| `SimpsonIntegral` | Simpson's rule |
| `TrapezoidIntegral` | Trapezoidal rule |
| `SegmentIntegral` | Segment-based |
| `KronrodIntegral` | Gauss-Kronrod adaptive |
| `GaussianQuadratures` | Gauss-Legendre, Laguerre, Hermite, Jacobi, etc. |
| `GaussianOrthogonalPolynomial` | Orthogonal polynomial basis |
| `TwoDimensionalIntegral` | 2D integration |
| `DiscreteIntegrals` | Discrete (trapezoid, Simpson) |
| `FilonIntegral` | Highly oscillatory integrals |
| `TanhSinhIntegral` | Double-exponential (tanh-sinh) |
| `ExpSinhIntegral` | Exp-sinh integration |
| `ExponentialIntegrals` | Ei(x) exponential integral |
| `GaussLaguerreCosinePolynomial` | Laguerre-cosine for Heston |
| `MomentBasedGaussianPolynomial` | Moment-based quadrature |

### 24.11 Phase 22 Deliverables

- [ ] All 14 copulas match QuantLib CDF/PDF values to 1e-10.
- [ ] Bivariate normal CDF matches QuantLib to 1e-12.
- [ ] SABR interpolation matches QuantLib SABR to 1e-10.
- [ ] 2D bicubic spline smooth and matches QuantLib.
- [ ] BFGS optimizer converges on Rosenbrock function.
- [ ] Differential evolution finds global minimum of Rastrigin function.
- [ ] Cholesky decomposition matches `nalgebra` to 1e-14.
- [ ] FFT matches direct DFT to 1e-10.
- [ ] All statistics accumulators match QuantLib reference values.
- [ ] Halton, Faure, lattice sequences pass discrepancy tests.
- [ ] Benchmark: FFT (8192 points) < 1 ms, Cholesky (100×100) < 1 ms.

---

## 25. Phase 23: Advanced Cash Flows & Coupons

**Duration estimate:** 3–4 weeks
**Depends on:** Phase 4 (basic cash flows), Phase 3 (indexes)
**Milestone:** *Complete coupon/cash flow coverage: CMS coupons, digital coupons, range accruals, sub-period coupons, inflation coupons.*

### 25.1 CMS Coupons

- `CmsCoupon` — Constant Maturity Swap coupon.
- `CmsCouponPricer` — CMS convexity adjustment.
- `ConundrumPricer` — Hagan, Andersen & Piterbarg replication-based pricing.
- `LinearTSRPricer` — linear terminal swap rate model pricer.
- `DigitalCmsCoupon` — digital CMS coupon.

### 25.2 Digital Coupons

- `DigitalCoupon` — generic digital coupon (call/put, cash/asset).
- `DigitalIborCoupon` — digital IBOR coupon.
- `CapFlooredCoupon` — capped/floored coupon.
- `CapFlooredInflationCoupon` — capped/floored inflation coupon.

### 25.3 Inflation Coupons

- `CPICoupon` + `CPICouponPricer` — CPI-linked coupon.
- `YoYInflationCoupon` — year-on-year inflation coupon.
- `InflationCoupon` (base), `InflationCouponPricer`.
- `ZeroInflationCashflow` — zero-coupon inflation cash flow.
- `IndexedCashflow` — generic index-linked cash flow.

### 25.4 Other Coupon Types

- `AverageBMACoupon` — BMA (Bond Market Association) averaged coupon.
- `SubPeriodCoupon` — sub-period compounding/averaging coupon.
- `MultipleResetsCoupon` — coupon with multiple resets per period.
- `RangeAccrualCoupon` — range accrual (corridor) coupon.
- `OvernightIndexedCouponPricer` — extended pricer with lookback/lockout.
- `EquityCashflow` — equity dividend/total return cash flow.
- `Dividend` — discrete dividend payment.

### 25.5 Cash Flow Utilities

- `CashFlows` analytics: `npv`, `bps`, `atmRate`, `duration`, `convexity`, `basisPointValue`, `zSpread`.
- `TimeBasket` — time-bucketed cash flow aggregation.
- `CashFlowVectors` — utility for building coupon vectors.
- `RateAveraging` — compounding vs. averaging methods (ISDA 2021).
- `Replication` — replication model for CMS convexity.
- `Duration` enum — `Simple`, `Macaulay`, `Modified`.

### 25.6 Phase 23 Deliverables

- [ ] CMS coupon rate matches QuantLib's Hagan replication to 0.5 bp.
- [ ] Digital coupon cash flow matches QuantLib to 1e-8.
- [ ] Range accrual matches QuantLib under flat vol to 1e-6.
- [ ] CPI coupon indexation matches QuantLib.
- [ ] Sub-period compounding matches manual calculation to 1e-12.
- [ ] Cash flow `zSpread` matches QuantLib to 1e-8.

---

## 26. Phase 24: Advanced Yield Curve & Fitting

**Duration estimate:** 2–3 weeks
**Depends on:** Phase 3 (basic term structures)
**Milestone:** *Fitted bond discount curves (Nelson-Siegel, Svensson), OIS helpers, multi-curve framework.*

### 26.1 Fitted Bond Discount Curve

- `FittedBondDiscountCurve` — fit discount function to bond prices.
- `NonlinearFittingMethods`:
  - `NelsonSiegelFitting` — 4-parameter Nelson-Siegel.
  - `SvenssonFitting` — 6-parameter Svensson.
  - `ExponentialSplinesFitting` — exponential spline fitting.
  - `CubicBSplinesFitting` — cubic B-spline fitting.
  - `SimplePolynomialFitting` — polynomial fitting.
  - `SpreadFittingMethod` — spread over reference curve.
- `BondHelper` — rate helper for bond-based bootstrapping.

### 26.2 Advanced Yield Curves

- `CompositeZeroYieldStructure` — sum/product of two zero-rate curves.
- `ImpliedTermStructure` — derived from a base curve.
- `InterpolatedSimpleZeroCurve` — zero curve with simple compounding.
- `ForwardCurve` — interpolated instantaneous forward curve.
- `PiecewiseSpreadYieldCurve` — bootstrap spread over reference.
- `PiecewiseForwardSpreadedTermStructure` — piecewise forward spread.
- `PiecewiseZeroSpreadedTermStructure` — piecewise zero spread.
- `SpreadDiscountCurve` — discount curve with additive spread.
- `UltimateForwardTermStructure` — Smith-Wilson extrapolation.
- `QuantoTermStructure` — quanto-adjusted yield curve.

### 26.3 Additional Rate Helpers

- `OISRateHelper` (extended) — overnight index swap helper.
- `OvernightIndexFutureRateHelper` — SOFR futures helper.
- `BondHelper` — bootstrap from bond prices.
- `FuturesRateHelper` — Eurodollar/SOFR futures.

### 26.4 Phase 24 Deliverables

- [ ] Nelson-Siegel fit matches QuantLib to 1e-6 on 10 bonds.
- [ ] Svensson fit produces smooth forward curve.
- [ ] OIS curve bootstrap matches QuantLib to 1e-8.
- [ ] Smith-Wilson extrapolation converges to UFR.
- [ ] Composite curve C1 + C2 matches manual computation.
- [ ] Benchmark: Nelson-Siegel fit (20 bonds) < 100 ms.

---

## 27. Cross-Cutting Concerns

These apply across ALL phases and should be set up during workspace bootstrap:

### 27.1 Error Handling

- All public APIs return `QLResult<T>`.
- No `unwrap()` in library code (only in examples/tests).
- Use `thiserror` for library errors, `anyhow` only in binaries.

### 27.2 Logging & Tracing

- Use `tracing` crate with structured spans.
- Key instrumentation points:
  - Calibration iterations: `tracing::info!(iteration, cost, params)`.
  - Bootstrap steps: `tracing::debug!(pillar_date, quote, implied)`.
  - MC simulation: `tracing::info!(paths, mean, std_error, elapsed)`.

### 27.3 Serialization

- All domain types derive `Serialize` + `Deserialize`.
- Enables: JSON config files, persistence (Phase 11), REST API (future).

### 27.4 Documentation

- Every public type/function has `///` doc comments with examples.
- Each crate has a `//! # ql-time` module-level doc.
- `cargo doc --workspace --no-deps` generates a browsable documentation site.

### 27.5 Feature Flags

| Flag | Crate | Effect |
|---|---|---|
| `parallel` | `ql-methods` | Enable `rayon` parallelism in MC/FD (default: on) |
| `redb` | `ql-persistence` | Enable embedded redb backend |
| `postgres` | `ql-persistence` | Enable PostgreSQL backend |
| `analytics` | `ql-persistence` | Enable DuckDB + Parquet export |
| `serde` | All | Enable serialization (default: on) |

---

## 28. Dependency Graph Between Phases

```
Phase 0: Workspace Bootstrap
    │
    ▼
Phase 1: ql-core + ql-time ◄──────────────────────────────┐
    │                                                      │
    ├──► Phase 2: ql-math ──────────────────────┐          │
    │       │                                   │          │
    │       │                          Phase 22: Math Ext  │
    │       ▼                                              │
    ├──► Phase 3: ql-currencies + ql-indexes               │
    │       + ql-termstructures ────► Phase 24: Adv Curves  │
    │       │                                              │
    │       ▼                                              │
    ├──► Phase 4: ql-cashflows ────► Phase 23: Adv Coupons │
    │       │                                              │
    │       ▼                                              │
    ├──► Phase 5: ql-instruments + ql-pricingengines       │
    │       │                                              │
    │       ├──► Phase 6: ql-processes + ql-models         │
    │       │       │                                      │
    │       │       ├──► Phase 14: Bates & Jump-Diffusion  │
    │       │       │                                      │
    │       │       ▼                                      │
    │       ├──► Phase 7: ql-methods (MC, FD, lattice)     │
    │       │       │                                      │
    │       │       ├──► Phase 8: Vol surfaces              │
    │       │       │       └──► Phase 18: Adv Vol Surfaces │
    │       │       │                                      │
    │       │       ├──► Phase 9: Credit & inflation        │
    │       │       │       └──► Phase 21: Advanced Credit  │
    │       │       │                                      │
    │       │       ├──► Phase 10: Exotics (expanded)       │
    │       │       │                                      │
    │       │       ├──► Phase 13: Advanced American        │
    │       │       │                                      │
    │       │       ├──► Phase 15: Multi-Asset & Basket     │
    │       │       │                                      │
    │       │       ├──► Phase 19: Advanced FD Framework    │
    │       │       │                                      │
    │       │       ├──► Phase 16: Short-Rate Models (full) │
    │       │       │       │                              │
    │       │       │       └──► Phase 17: Adv Swaption    │
    │       │       │                                      │
    │       │       └──► Phase 20: LMM Framework            │
    │       │                                              │
    │       └──► Phase 12: Integration / CLI                │
    │                                                      │
    └──► Phase 11: ql-persistence ─────────────────────────┘
         (can start in parallel after Phase 1)
```

**Key insight:** Phase 11 (persistence) depends only on `ql-core` types. It can be developed **in parallel** with Phases 3–24. Phases 13–24 can largely proceed in parallel once their dependencies are met.

---

## 29. Testing & Validation Strategy

### 29.1 Unit Tests (Every Phase)

- Every module has inline `#[cfg(test)] mod tests`.
- Use `approx::assert_relative_eq!` for all floating-point comparisons.
- Target: > 90% line coverage on core computation code.

### 29.2 Cross-Validation Against QuantLib C++

For each phase, write "golden file" tests that compare results against QuantLib C++ output:

1. Write a small C++ program that uses QuantLib to compute reference values.
2. Save the results as JSON/CSV in `tests/data/`.
3. Rust tests load and compare against these golden values.

| Phase | Golden Test |
|---|---|
| 1 | Calendar holidays, schedule dates, day count fractions |
| 2 | Interpolation values, solver results, distribution CDF/PDF |
| 3 | Bootstrapped discount factors, zero rates, forward rates |
| 4 | Fixed/float leg NPVs, accrued interest |
| 5 | BS option prices + Greeks, swap NPVs, bond prices |
| 6 | Heston prices, HW swaption prices |
| 7 | MC/FD/binomial prices |
| 8 | SABR vols, swaption prices, cap/floor prices |
| 9 | CDS NPV, default curve survival probabilities |
| 10 | Callable bond, barrier option, Asian option, lookback, cliquet |
| 13 | BAW, Bjerksund-Stensland, QD+, Longstaff-Schwartz prices |
| 14 | Bates prices, Merton jump-diffusion, GJR-GARCH |
| 15 | Basket option MC, spread option Kirk, Margrabe formula |
| 16 | Vasicek/CIR bond prices, BK/GSR/G2 swaption prices |
| 17 | Gaussian 1D swaption, Jamshidian, FD HW swaption |
| 18 | SVI/ZABR/NoArb-SABR vols, Andreasen-Huge, vol cube |
| 19 | 2D FD Heston, 3D Heston-HW, FD barrier, FD Bates |
| 20 | LMM caplet prices, Bermudan swaption, pathwise Greeks |
| 21 | CDO tranche, nth-to-default, ISDA CDS, CDS option |
| 22 | Copula values, bivariate normal, FFT, statistics |
| 23 | CMS coupon, digital coupon, range accrual, CPI coupon |
| 24 | Nelson-Siegel fit, Svensson, OIS bootstrap, Smith-Wilson |

### 29.3 Integration Tests

Located in `tests/` directory:

- `test_yield_curve_pipeline.rs` — market data → bootstrap → discount → forward rate.
- `test_swap_pricing_pipeline.rs` — curve → legs → swap → NPV.
- `test_option_pricing_pipeline.rs` — vol surface → process → engine → Greeks.
- `test_persistence_round_trip.rs` — create trade → persist → load → verify.

### 29.4 Property-Based Testing

Use `proptest` or `quickcheck` for invariants:

- `discount(t) ∈ (0, 1]` for all `t > 0`.
- Put-call parity: `C - P = S·exp(-qT) - K·exp(-rT)`.
- Monotonicity: `discount(t1) >= discount(t2)` for `t1 < t2`.
- American option price ≥ European option price.

### 29.5 Fuzzing (Optional, Phase 10+)

Use `cargo-fuzz` on date parsing, schedule generation, and deserialization to find panics.

---

## 30. Performance Benchmarking Plan

### 30.1 Criterion Benchmarks

Located in `benches/`:

| Benchmark | Target |
|---|---|
| `bench_date_arithmetic` | Date add/sub: < 1 ns |
| `bench_day_count` | `year_fraction()`: < 5 ns |
| `bench_calendar_advance` | `advance(30 business days)`: < 100 ns |
| `bench_interpolation` | Linear/cubic lookup: < 10 ns |
| `bench_bs_analytic` | BS price + 5 Greeks: < 200 ns |
| `bench_bootstrap` | 15-node yield curve: < 1 ms |
| `bench_mc_european_100k` | 100k paths, 252 steps: < 500 ms |
| `bench_mc_european_1M` | 1M paths, 252 steps: < 5 s |
| `bench_fd_american` | 200×200 grid: < 50 ms |
| `bench_heston_analytic` | Heston price: < 100 μs |
| `bench_heston_calibration` | 20-point calibration: < 2 s |
| `bench_baw_american` | BAW approximation: < 1 μs |
| `bench_bjerksund_stensland` | BJS approximation: < 1 μs |
| `bench_ls_mc_10k` | Longstaff-Schwartz 10k paths: < 100 ms |
| `bench_bates_analytic` | Bates characteristic fn price: < 500 μs |
| `bench_fd_bates_2d` | 2D FD Bates (100×100): < 5 s |
| `bench_basket_mc_5asset` | 5-asset basket MC (100k): < 2 s |
| `bench_kirk_spread` | Kirk spread approximation: < 1 μs |
| `bench_vasicek_bond` | Vasicek bond price: < 100 ns |
| `bench_g2_swaption` | G2 analytic swaption: < 10 ms |
| `bench_sabr_calibration` | SABR calibration per-expiry: < 1 ms |
| `bench_svi_fit` | SVI smile fit: < 500 μs |
| `bench_fd_heston_2d` | 2D FD Heston (100×100): < 1 s |
| `bench_fd_heston_hw_3d` | 3D FD Heston-HW (50³): < 30 s |
| `bench_lmm_10k_paths` | LMM 10k paths, 40 rates: < 5 s |
| `bench_gaussian_copula_cdo` | CDO (125 names, 7 tranches): < 2 s |
| `bench_fft_8192` | FFT 8192 points: < 1 ms |
| `bench_cholesky_100` | Cholesky 100×100: < 1 ms |
| `bench_nelson_siegel_fit` | Nelson-Siegel (20 bonds): < 100 ms |
| `bench_cms_coupon_pricing` | CMS coupon replication: < 10 ms |

### 30.2 Comparison Framework

Write a benchmark harness that runs the same computation in QuantLib C++ (via `cc` crate or subprocess) and reports the Rust / C++ speed ratio.

---

## 31. Risk Register & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| QuantLib C++ semantics are subtle (e.g., settlement lag, fixing conventions) | Incorrect prices | High | Cross-validate every phase against QuantLib test suite |
| Observer pattern + interior mutability → complex borrow-checker fights | Development slowdown | Medium | Start with single-threaded `Rc<RefCell>`, upgrade to `Arc<RwLock>` only where needed |
| `argmin` crate doesn't cover all QuantLib optimizers (e.g., LM constraints) | Can't calibrate some models | Medium | Implement custom optimizer wrapper; contribute to `argmin` upstream |
| Calendar/holiday data maintenance burden | Stale calendars | Medium | Auto-generate from QuantLib's calendar source; keep a `calendars/` data directory |
| Scope creep (QuantLib has 2,300+ source files) | Never-ending project | High | Strict phase gates; "done" = milestone test passes, not feature parity |
| Persistence layer adds coupling to domain crates | Architecture rot | Medium | `ObjectStore` trait in `ql-core`; all implementations in `ql-persistence` |
| `nalgebra` API churn or breaking changes | Build breakage | Low | Pin versions in workspace dependencies |
| LMM framework is enormous (~100 QuantLib files) | Phase 20 takes too long | High | Start with forward measure only; add coterminal/CMS measures later |
| Multi-dimensional FD (3D+) memory and runtime | Impractical on commodity hardware | Medium | Use ADI splitting; limit 3D grids to ~50³; provide rayon parallelism |
| Copula/credit models require extensive test data | Can't validate without market data | Medium | Use synthetic portfolios; cross-validate against QuantLib test suite |
| Advanced vol models (SLV, ZABR) have numerical edge cases | Divergence or NaN | Medium | Extensive property-based testing; fallback to simpler models |
| Extended phases may break existing API contracts | Regression | Medium | Golden tests freeze existing behavior; CI runs full test suite |

---

## 32. Estimated Timeline

Assuming one full-time developer:

### 32.1 Core Phases (Phases 0–12)

| Phase | Duration | Cumulative | Key Milestone |
|---|---|---|---|
| 0: Bootstrap | 2–3 days | Week 1 | Workspace compiles |
| 1: Foundations | 2–3 weeks | Week 3–4 | Coupon schedules ✓ |
| 2: Math | 2–3 weeks | Week 6–7 | Root-finding + interpolation ✓ |
| 3: Rates | 3–4 weeks | Week 10–11 | Yield curve bootstrap ✓ |
| 4: Cash Flows | 2 weeks | Week 12–13 | Swap legs ✓ |
| 5: Instruments | 3–4 weeks | Week 16–17 | BS pricing + swaps + bonds ✓ |
| 6: Processes & Models | 3–4 weeks | Week 20–21 | Heston calibration ✓ |
| 7: Advanced Methods | 4–5 weeks | Week 25–26 | MC + FD + lattice ✓ |
| 8: Vol Surfaces | 3 weeks | Week 28–29 | SABR + swaptions ✓ |
| 9: Credit & Inflation | 3 weeks | Week 31–32 | CDS + inflation ✓ |
| 10: Exotics | 4–6 weeks | Week 37–38 | Full exotics suite ✓ |
| 11: Persistence | 3–4 weeks | (parallel) | redb + Postgres ✓ |
| 12: Integration | 2–3 weeks | Week 40 | End-to-end ✓ |

**Core Total: ~10 months** for a single developer.

### 32.2 Extended Phases (Phases 13–24)

| Phase | Duration | Depends On | Key Milestone |
|---|---|---|---|
| 13: Advanced American | 3–4 weeks | 5, 7 | BAW + LS-MC ✓ |
| 14: Bates & Jump-Diffusion | 2–3 weeks | 6, 7 | Bates calibration ✓ |
| 15: Multi-Asset & Basket | 3–4 weeks | 6, 7 | Basket MC + spread options ✓ |
| 16: Short-Rate Models (full) | 4–5 weeks | 6, 7 | CIR + Vasicek + BK + GSR + G2 ✓ |
| 17: Adv Swaption & Cap/Floor | 3–4 weeks | 16, 8 | Gaussian 1D + FD HW swaption ✓ |
| 18: Adv Vol Surfaces | 4–5 weeks | 8, 6 | SVI + ZABR + NoArb-SABR + SLV ✓ |
| 19: Advanced FD Framework | 5–6 weeks | 7, 6, 16 | Full FDM: meshers + operators + schemes ✓ |
| 20: LMM Framework | 6–8 weeks | 7, 8, 16 | Full LMM with Bermudan swaptions ✓ |
| 21: Advanced Credit | 4–5 weeks | 9, 7, 22 | CDO + copula models ✓ |
| 22: Math Extensions | 4–5 weeks | 2 | All interpolations + copulas + stats ✓ |
| 23: Adv Cash Flows | 3–4 weeks | 4, 3 | CMS + digital + range accrual ✓ |
| 24: Adv Yield Curve | 2–3 weeks | 3 | Nelson-Siegel + OIS + multi-curve ✓ |

**Extended Total: ~44–56 additional weeks** (sequentially).

### 32.3 Parallelization Opportunities

Many extended phases can run in parallel:

- **Track A (Equity/Vol):** Phases 13, 14, 15, 18 (in sequence after Phase 7)
- **Track B (Rates):** Phases 16, 17, 23, 24 (in sequence after Phase 7)
- **Track C (Numerics):** Phases 19, 20 (after Phase 7)
- **Track D (Credit/Math):** Phases 22, 21 (Phase 22 first, then 21)

With 3–4 parallel tracks, extended phases reduce to ~6–8 months. **Grand total with parallelism: ~16–18 months** for full QuantLib parity.

---

## 33. Definition of Done — Per-Phase Checklist

For each phase, **all** of the following must be satisfied:

- [ ] All `#[test]` functions pass: `cargo test -p <crate>`.
- [ ] Clippy clean: `cargo clippy -p <crate> -- -D warnings`.
- [ ] Documentation builds: `cargo doc -p <crate> --no-deps`.
- [ ] All public APIs have `///` doc comments.
- [ ] No `unwrap()` or `expect()` in library code (only in tests/examples).
- [ ] Cross-validation against QuantLib C++ for at least 3 test cases.
- [ ] Benchmark exists for performance-critical functions.
- [ ] Phase milestone demonstrated (e.g., "can bootstrap a yield curve").
- [ ] CHANGELOG updated.
- [ ] README updated with new capabilities.

---

## Appendix A: File Checklist per Phase

### Phase 1 Files

```
crates/ql-core/src/
├── lib.rs
├── types.rs
├── errors.rs
├── observable.rs
├── lazy.rs
├── handle.rs
├── settings.rs
└── quote.rs

crates/ql-time/src/
├── lib.rs
├── date.rs
├── period.rs
├── calendar.rs
├── calendars/
│   ├── mod.rs
│   ├── null_calendar.rs
│   ├── target.rs
│   ├── united_states.rs
│   ├── weekends_only.rs
│   └── united_kingdom.rs
├── day_counter.rs
├── day_counters/
│   ├── mod.rs
│   ├── actual360.rs
│   ├── actual365_fixed.rs
│   ├── thirty360.rs
│   └── actual_actual.rs
├── schedule.rs
├── business_day_convention.rs
└── imm.rs
```

### Phase 2 Files

```
crates/ql-math/src/
├── lib.rs
├── interpolation.rs
├── interpolations/
│   ├── mod.rs
│   ├── linear.rs
│   ├── log_linear.rs
│   └── cubic.rs
├── solvers/
│   ├── mod.rs
│   ├── brent.rs
│   ├── newton.rs
│   └── bisection.rs
├── distributions.rs
├── optimization.rs
├── integration.rs
├── rng.rs
├── sobol.rs
└── matrix.rs
```

### Phase 3 Files

```
crates/ql-currencies/src/
├── lib.rs
├── currency.rs
├── money.rs
├── exchange_rate.rs
└── currencies/
    ├── mod.rs
    ├── america.rs
    ├── europe.rs
    └── asia.rs

crates/ql-indexes/src/
├── lib.rs
├── index.rs
├── index_manager.rs
├── interest_rate_index.rs
├── ibor/
│   ├── mod.rs
│   ├── euribor.rs
│   └── sofr.rs
├── overnight.rs
├── swap_index.rs
└── inflation.rs

crates/ql-termstructures/src/
├── lib.rs
├── term_structure.rs
├── yield_ts.rs
├── yield_curves/
│   ├── mod.rs
│   ├── flat_forward.rs
│   ├── discount_curve.rs
│   ├── zero_curve.rs
│   ├── piecewise_yield_curve.rs
│   ├── zero_spreaded.rs
│   └── forward_spreaded.rs
├── bootstrap.rs
├── rate_helpers/
│   ├── mod.rs
│   ├── deposit_rate_helper.rs
│   ├── fra_rate_helper.rs
│   ├── swap_rate_helper.rs
│   └── ois_rate_helper.rs
├── interest_rate.rs
├── vol_ts.rs
├── vol_surfaces/
│   ├── mod.rs
│   └── black_constant_vol.rs
├── default_ts.rs
└── inflation_ts.rs
```

### Phase 4 Files

```
crates/ql-cashflows/src/
├── lib.rs
├── cashflow.rs
├── coupon.rs
├── fixed_rate_coupon.rs
├── floating_rate_coupon.rs
├── ibor_coupon.rs
├── overnight_coupon.rs
├── simple_cashflow.rs
├── coupon_pricer.rs
├── leg.rs
└── cashflow_analytics.rs
```

### Phase 5 Files

```
crates/ql-instruments/src/
├── lib.rs
├── instrument.rs
├── payoff.rs
├── exercise.rs
├── vanilla_option.rs
├── swap.rs
├── bond.rs
└── forward.rs

crates/ql-pricingengines/src/
├── lib.rs
├── engine.rs
├── analytic/
│   ├── mod.rs
│   └── analytic_european_engine.rs
├── swap/
│   ├── mod.rs
│   └── discounting_swap_engine.rs
└── bond/
    ├── mod.rs
    └── discounting_bond_engine.rs
```

### Phase 11 Files

```
crates/ql-persistence/src/
├── lib.rs
├── object_store.rs          # ObjectStore trait
├── persistable.rs           # Persistable trait
├── domain/
│   ├── mod.rs
│   ├── trade.rs
│   ├── lifecycle_event.rs
│   ├── market_snapshot.rs
│   └── trade_filter.rs
├── embedded/                # feature = "redb"
│   ├── mod.rs
│   └── redb_store.rs
├── postgres/                # feature = "postgres"
│   ├── mod.rs
│   ├── pg_store.rs
│   └── migrations/
│       └── 001_initial.sql
└── analytics/               # feature = "analytics"
    ├── mod.rs
    └── parquet_export.rs
```

---

## Appendix B: Quick-Start Commands

```bash
# Clone and build
git clone https://github.com/<user>/ql-rust.git
cd ql-rust
cargo build --workspace

# Run all tests
cargo test --workspace

# Run specific phase tests
cargo test -p ql-core
cargo test -p ql-time

# Run examples
cargo run --example price_european_option
cargo run --example bootstrap_yield_curve

# Run benchmarks
cargo bench --bench bench_bs_analytic

# Generate documentation
cargo doc --workspace --no-deps --open

# Lint
cargo clippy --workspace -- -D warnings

# Format
cargo fmt --all
```
