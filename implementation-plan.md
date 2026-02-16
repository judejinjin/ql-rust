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
15. [Cross-Cutting Concerns](#15-cross-cutting-concerns)
16. [Dependency Graph Between Phases](#16-dependency-graph-between-phases)
17. [Testing & Validation Strategy](#17-testing--validation-strategy)
18. [Performance Benchmarking Plan](#18-performance-benchmarking-plan)
19. [Risk Register & Mitigations](#19-risk-register--mitigations)
20. [Estimated Timeline](#20-estimated-timeline)
21. [Definition of Done — Per-Phase Checklist](#21-definition-of-done--per-phase-checklist)

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

**Duration estimate:** 4+ weeks (ongoing)
**Depends on:** Phases 6–9
**Milestone:** *Feature parity with QuantLib's most-used experimental modules.*

| Feature | Priority |
|---|---|
| Callable bonds (Hull-White tree) | High |
| Barrier options (analytic + MC + FD) | High |
| Asian options (MC) | Medium |
| Variance swaps | Medium |
| Stochastic local volatility (SLV) | Low |
| Hybrid Heston-Hull-White | Low |
| Commodity models | Low |

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

## 15. Cross-Cutting Concerns

These apply across ALL phases and should be set up during workspace bootstrap:

### 15.1 Error Handling

- All public APIs return `QLResult<T>`.
- No `unwrap()` in library code (only in examples/tests).
- Use `thiserror` for library errors, `anyhow` only in binaries.

### 15.2 Logging & Tracing

- Use `tracing` crate with structured spans.
- Key instrumentation points:
  - Calibration iterations: `tracing::info!(iteration, cost, params)`.
  - Bootstrap steps: `tracing::debug!(pillar_date, quote, implied)`.
  - MC simulation: `tracing::info!(paths, mean, std_error, elapsed)`.

### 15.3 Serialization

- All domain types derive `Serialize` + `Deserialize`.
- Enables: JSON config files, persistence (Phase 11), REST API (future).

### 15.4 Documentation

- Every public type/function has `///` doc comments with examples.
- Each crate has a `//! # ql-time` module-level doc.
- `cargo doc --workspace --no-deps` generates a browsable documentation site.

### 15.5 Feature Flags

| Flag | Crate | Effect |
|---|---|---|
| `parallel` | `ql-methods` | Enable `rayon` parallelism in MC/FD (default: on) |
| `redb` | `ql-persistence` | Enable embedded redb backend |
| `postgres` | `ql-persistence` | Enable PostgreSQL backend |
| `analytics` | `ql-persistence` | Enable DuckDB + Parquet export |
| `serde` | All | Enable serialization (default: on) |

---

## 16. Dependency Graph Between Phases

```
Phase 0: Workspace Bootstrap
    │
    ▼
Phase 1: ql-core + ql-time ◄──────────────────────┐
    │                                               │
    ├──► Phase 2: ql-math                           │
    │       │                                       │
    │       ▼                                       │
    ├──► Phase 3: ql-currencies + ql-indexes        │
    │       + ql-termstructures                     │
    │       │                                       │
    │       ▼                                       │
    ├──► Phase 4: ql-cashflows                      │
    │       │                                       │
    │       ▼                                       │
    ├──► Phase 5: ql-instruments + ql-pricingengines│
    │       │                                       │
    │       ├──► Phase 6: ql-processes + ql-models  │
    │       │       │                               │
    │       │       ▼                               │
    │       ├──► Phase 7: ql-methods (MC, FD, lattice)
    │       │       │                               │
    │       │       ├──► Phase 8: Vol surfaces      │
    │       │       ├──► Phase 9: Credit & inflation│
    │       │       └──► Phase 10: Exotics          │
    │       │                                       │
    │       └──► Phase 12: Integration / CLI        │
    │                                               │
    └──► Phase 11: ql-persistence ──────────────────┘
         (can start in parallel after Phase 1)
```

**Key insight:** Phase 11 (persistence) depends only on `ql-core` types. It can be developed **in parallel** with Phases 3–10.

---

## 17. Testing & Validation Strategy

### 17.1 Unit Tests (Every Phase)

- Every module has inline `#[cfg(test)] mod tests`.
- Use `approx::assert_relative_eq!` for all floating-point comparisons.
- Target: > 90% line coverage on core computation code.

### 17.2 Cross-Validation Against QuantLib C++

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

### 17.3 Integration Tests

Located in `tests/` directory:

- `test_yield_curve_pipeline.rs` — market data → bootstrap → discount → forward rate.
- `test_swap_pricing_pipeline.rs` — curve → legs → swap → NPV.
- `test_option_pricing_pipeline.rs` — vol surface → process → engine → Greeks.
- `test_persistence_round_trip.rs` — create trade → persist → load → verify.

### 17.4 Property-Based Testing

Use `proptest` or `quickcheck` for invariants:

- `discount(t) ∈ (0, 1]` for all `t > 0`.
- Put-call parity: `C - P = S·exp(-qT) - K·exp(-rT)`.
- Monotonicity: `discount(t1) >= discount(t2)` for `t1 < t2`.
- American option price ≥ European option price.

### 17.5 Fuzzing (Optional, Phase 10+)

Use `cargo-fuzz` on date parsing, schedule generation, and deserialization to find panics.

---

## 18. Performance Benchmarking Plan

### 18.1 Criterion Benchmarks

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

### 18.2 Comparison Framework

Write a benchmark harness that runs the same computation in QuantLib C++ (via `cc` crate or subprocess) and reports the Rust / C++ speed ratio.

---

## 19. Risk Register & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| QuantLib C++ semantics are subtle (e.g., settlement lag, fixing conventions) | Incorrect prices | High | Cross-validate every phase against QuantLib test suite |
| Observer pattern + interior mutability → complex borrow-checker fights | Development slowdown | Medium | Start with single-threaded `Rc<RefCell>`, upgrade to `Arc<RwLock>` only where needed |
| `argmin` crate doesn't cover all QuantLib optimizers (e.g., LM constraints) | Can't calibrate some models | Medium | Implement custom optimizer wrapper; contribute to `argmin` upstream |
| Calendar/holiday data maintenance burden | Stale calendars | Medium | Auto-generate from QuantLib's calendar source; keep a `calendars/` data directory |
| Scope creep (QuantLib has 500+ source files) | Never-ending project | High | Strict phase gates; "done" = milestone test passes, not feature parity |
| Persistence layer adds coupling to domain crates | Architecture rot | Medium | `ObjectStore` trait in `ql-core`; all implementations in `ql-persistence` |
| `nalgebra` API churn or breaking changes | Build breakage | Low | Pin versions in workspace dependencies |

---

## 20. Estimated Timeline

Assuming one full-time developer:

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
| 10: Exotics | 4+ weeks | Week 35–36 | Barriers + Asians ✓ |
| 11: Persistence | 3–4 weeks | (parallel) | redb + Postgres ✓ |
| 12: Integration | 2–3 weeks | Week 38 | End-to-end ✓ |

**Total: ~9 months** for a single developer to reach comprehensive coverage. Phases 11 runs in parallel with 3–10.

With a team of 2–3 developers, the critical path (Phases 0→1→2→3→4→5) is ~16–17 weeks, and parallel work on persistence, math, and advanced methods can bring total time to ~5–6 months.

---

## 21. Definition of Done — Per-Phase Checklist

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
