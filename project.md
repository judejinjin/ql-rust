I want to re-implement open-source quantlib with modern rust language with the state-of-the-art open source crates.

---

# QuantLib Object Model Summary

> Source: `/mnt/c/finance/quantlib/` — the C++ open-source QuantLib library  
> Purpose: Reference architecture for a Rust re-implementation

---

## 1. Foundational Design Patterns (`ql/patterns/`)

QuantLib's entire object model rests on a small set of design patterns:

### 1.1 Observer / Observable (`observable.hpp`)

The backbone of QuantLib. Nearly every object inherits from `Observable`, `Observer`, or both.

- **`Observable`** — maintains a `set<Observer*>`. Calls `notifyObservers()` when state changes.
- **`Observer`** — registers with one or more `Observable` instances via `registerWith()`. Must implement `virtual void update() = 0`.
- **`ObservableSettings`** — singleton that can globally disable/defer notifications (useful for batch updates).
- Thread-safe variant is available behind `#ifdef QL_ENABLE_THREAD_SAFE_OBSERVER_PATTERN`, using `std::mutex` and a `Proxy` indirection layer.

**Rust mapping:** This maps well to a reactive/signal-based system. Consider `Arc<RwLock<>>` + callback closures, or a crate like `futures` channels.

### 1.2 LazyObject (`lazyobject.hpp`)

Implements **calculation-on-demand with result caching**. Inherits both `Observable` and `Observer`.

- `calculate()` — triggers `performCalculations()` only if `!calculated_ && !frozen_`.
- `update()` — called when a dependency changes; sets `calculated_ = false` and forwards notification.
- `freeze()` / `unfreeze()` — pins/unpins cached results.
- `recalculate()` — forces immediate recalculation.
- `forwardFirstNotificationOnly()` — performance optimization: suppress redundant notifications.

Key state: `mutable bool calculated_, frozen_, alwaysForward_`.

**Rust mapping:** A struct with `Cell<bool>` or `RefCell`-based caching. `performCalculations` becomes a trait method.

### 1.3 Singleton (`singleton.hpp`)

Used for global settings and managers (e.g., `Settings`, `IndexManager`, `ObservableSettings`).

### 1.4 Acyclic Visitor (`visitor.hpp`)

- `AcyclicVisitor` — empty base class.
- `Visitor<T>` — templated; defines `virtual void visit(T&) = 0`.
- Used extensively by cash flows and pricing engines to dispatch on concrete types.

### 1.5 Curiously Recurring Template Pattern (`curiouslyrecurring.hpp`)

Used for static polymorphism in a few places.

---

## 2. Core Value Types

### 2.1 Date & Time (`ql/time/`)

| Class | Description |
|---|---|
| `Date` | Serial-number based date (integer days from epoch). Supports arithmetic, comparison. |
| `Period` | A length of time (e.g., `3*Months`). Combines a number with a `TimeUnit`. |
| `Calendar` | Defines business days for a market/country. Has `isBusinessDay()`, `advance()`, `businessDaysBetween()`. |
| `DayCounter` | Computes year fractions between dates (Act/360, Act/365, 30/360, etc.). |
| `Schedule` | A sequence of dates generated from rules (tenor, calendar, convention, stub rules). |
| `Frequency` | Enum: `Annual`, `Semiannual`, `Quarterly`, `Monthly`, etc. |
| `BusinessDayConvention` | Enum: `Following`, `ModifiedFollowing`, `Preceding`, etc. |
| `DateGeneration::Rule` | Enum: `Forward`, `Backward`, `ThirdWednesday`, `CDS`, etc. |

### 2.2 InterestRate (`interestrate.hpp`)

Encapsulates the interest rate compounding algebra:
- Stores: `Rate r_`, `DayCounter dc_`, `Compounding comp_`, `Frequency freq_`.
- Provides: `discountFactor(t)`, `compoundFactor(t)`, `equivalentRate()`, `impliedRate()`.
- `Compounding` enum: `Simple`, `Compounded`, `Continuous`, `SimpleThenCompounded`.

### 2.3 Currency (`currency.hpp`)

Defines ISO currencies (name, code, numeric code, rounding, format). Concrete currencies in `ql/currencies/` by region (America, Europe, Asia, Africa, Oceania, Crypto).

### 2.4 Money (`money.hpp`)

A `Real` value associated with a `Currency`.

### 2.5 Quote (`quote.hpp`)

Purely virtual base for market observables:
- `virtual Real value() const = 0`
- `virtual bool isValid() const = 0`
- Inherits `Observable` so instruments can register to be notified of quote changes.

---

## 3. Handle & RelinkableHandle (`handle.hpp`)

A critical piece of QuantLib's object wiring:

- **`Handle<T>`** — a shared, reference-counted smart pointer to an `Observable`. All copies share the same internal `Link`.
- **`RelinkableHandle<T>`** — extends `Handle<T>` with `linkTo()`, which swaps the underlying pointer. When relinked, **all holders are notified**.
- The internal `Link` is itself an `Observable` + `Observer`, forwarding notifications from the pointee.

**Usage pattern:**
```cpp
RelinkableHandle<YieldTermStructure> discountCurve;
instrument.setPricingEngine(engine_that_uses(discountCurve));
// Later, swap the curve — instrument automatically recalculates:
discountCurve.linkTo(newCurve);
```

**Rust mapping:** `Arc<RwLock<Option<Arc<dyn T>>>>` with a notification channel, or a custom `Handle` type.

---

## 4. Global Settings (`settings.hpp`)

`Settings` is a `Singleton` providing:
- **`evaluationDate()`** — the pricing date. Returns today's date if not explicitly set. Is itself an `ObservableValue`, so term structures/instruments react when it changes.
- `includeReferenceDateEvents()`, `includeTodaysCashFlows()` — control cash-flow inclusion semantics.
- `enforcesTodaysHistoricFixings()` — enforce availability of today's index fixings.
- `SavedSettings` RAII guard for temporarily changing settings.

---

## 5. Instrument Hierarchy (`instrument.hpp`, `ql/instruments/`)

### 5.1 Base: `Instrument` (extends `LazyObject`)

The central abstraction. Key interface:
- `Real NPV() const` — net present value (triggers lazy calculation).
- `Real errorEstimate() const` — Monte Carlo error estimate when available.
- `Date valuationDate() const`
- `T result<T>(tag)` — retrieve named additional results.
- `bool isExpired() const = 0` — pure virtual.
- `void setPricingEngine(shared_ptr<PricingEngine>)` — attach an engine.
- `setupArguments(PricingEngine::arguments*)` — fill engine inputs.
- `fetchResults(PricingEngine::results*)` — read engine outputs.

Calculation flow:
```
NPV() → calculate() → if expired: setupExpired()
                       else: performCalculations()
                              → engine_->reset()
                              → setupArguments(engine_->getArguments())
                              → engine_->getArguments()->validate()
                              → engine_->calculate()
                              → fetchResults(engine_->getResults())
```

### 5.2 Concrete Instruments

| Category | Classes |
|---|---|
| **Options** | `VanillaOption`, `EuropeanOption`, `BarrierOption`, `AsianOption`, `BasketOption`, `CliqueOption`, `CompoundOption`, `DoubleBarrierOption`, `QuantoVanillaOption`, `Swaption`, `CapFloor` |
| **Swaps** | `Swap`, `VanillaSwap`, `FixedVsFloatingSwap`, `FloatFloatSwap`, `BMASwap`, `CPISwap`, `CreditDefaultSwap`, `EquityTotalReturnSwap`, `VarianceSwap`, `ZeroCouponSwap`, `YoYInflationSwap` |
| **Bonds** | `Bond`, `FixedRateBond`, `FloatingRateBond`, `ZeroCouponBond`, `AmortizingFixedRateBond`, `AmortizingFloatingRateBond`, `ConvertibleBond`, `CPIBond`, `CMSRateBond`, `BTP` |
| **Forward** | `Forward`, `ForwardRateAgreement`, `BondForward` |
| **Other** | `Stock`, `CompositeInstrument` |

Inheritance example:
```
LazyObject → Instrument → OneAssetOption → VanillaOption
LazyObject → Instrument → Swap → VanillaSwap
LazyObject → Instrument → Bond → FixedRateBond
```

---

## 6. Pricing Engine Framework (`pricingengine.hpp`, `ql/pricingengines/`)

### 6.1 Base: `PricingEngine`

Abstract interface (inherits `Observable`):
- `arguments* getArguments() const = 0`
- `const results* getResults() const = 0`
- `void reset() = 0`
- `void calculate() const = 0`

### 6.2 `GenericEngine<ArgumentsType, ResultsType>`

Template that stores `mutable ArgumentsType arguments_` and `mutable ResultsType results_`. Derived engines only implement `calculate()`.

### 6.3 Pricing Engine Categories

| Category | Examples |
|---|---|
| **Analytic** | `AnalyticEuropeanEngine`, `AnalyticHestonEngine`, `AnalyticBSMHullWhiteEngine`, `BlackCalculator`, `BachelierCalculator` |
| **Finite Difference** | `FdBlackScholesVanillaEngine`, `FdHestonVanillaEngine`, `FdBatesVanillaEngine`, `FdCIRVanillaEngine` |
| **Monte Carlo** | `MCEuropeanEngine`, `MCAmericanEngine`, `MCHestonHullWhiteEngine` |
| **Lattice / Tree** | `BinomialEngine`, `LatticeShortRateModelEngine` |
| **Bond** | engines under `ql/pricingengines/bond/` |
| **Swap / Swaption** | engines under `ql/pricingengines/swap/`, `swaption/` |

**Design insight:** Instruments are *engine-agnostic*. The same `VanillaOption` can be priced by an analytic Black-Scholes engine, a Heston engine, a finite-difference engine, or a Monte Carlo engine — just call `setPricingEngine()`.

---

## 7. Term Structures (`termstructure.hpp`, `ql/termstructures/`)

### 7.1 Base: `TermStructure`

Inherits `Observer`, `Observable`, and `Extrapolator`. Tracks:
- Reference date (fixed, or floating relative to evaluation date).
- Calendar, day counter, settlement days.
- `maxDate()` — the latest date the curve covers.
- `timeFromReference(Date)` — convert date to year fraction.

### 7.2 Yield Term Structures (`yieldtermstructure.hpp`)

Provides:
- `discount(Date/Time)` — discount factor.
- `zeroRate(Date/Time)` — zero-coupon rate.
- `forwardRate(Date1, Date2)` — forward rate between two dates.

Concrete implementations:
| Type | Classes |
|---|---|
| **Bootstrapped** | `PiecewiseYieldCurve<Traits, Interpolator>` — the workhorse. Bootstraps from market helpers (deposit rates, FRA, futures, swaps). |
| **Parametric** | `FittedBondDiscountCurve` (Nelson-Siegel, Svensson, etc.) |
| **Simple** | `FlatForward`, `DiscountCurve`, `ZeroCurve`, `ForwardCurve` |
| **Spreaded** | `ZeroSpreadedTermStructure`, `ForwardSpreadedTermStructure`, `PiecewiseZeroSpreadedTermStructure` |
| **Implied** | `ImpliedTermStructure`, `QuantoTermStructure` |

### 7.3 Volatility Term Structures (`ql/termstructures/volatility/`)

Organized by product:
- **Equity/FX** — `BlackVolTermStructure`, `LocalVolTermStructure`, `BlackVolSurface`
- **Cap/Floor** — `CapFloorTermVolStructure`, `OptionletVolatilityStructure`
- **Swaption** — `SwaptionVolatilityStructure`, `SwaptionVolCube`
- **Smile sections** — `SmileSection`, `SabrSmileSection`, `KahaleSmileSection`
- **SABR** — `sabr.hpp` (SABR formula implementation)

### 7.4 Other Term Structures

- **Default / Credit** — `DefaultProbabilityTermStructure` (`ql/termstructures/credit/`)
- **Inflation** — `InflationTermStructure`, `ZeroInflationTermStructure`, `YoYInflationTermStructure`

### 7.5 Bootstrapping Framework

- `BootstrapHelper<TermStructure>` — base for rate helpers.
- `IterativeBootstrap` — the default solver; iteratively solves for each node.
- Concrete helpers: `DepositRateHelper`, `FraRateHelper`, `FuturesRateHelper`, `SwapRateHelper`, `OISRateHelper`, `BondHelper`.

---

## 8. Stochastic Processes (`stochasticprocess.hpp`, `ql/processes/`)

### 8.1 Base: `StochasticProcess` (multi-dimensional)

Models $d\mathbf{x}_t = \mu(t, \mathbf{x}_t) dt + \sigma(t, \mathbf{x}_t) \cdot d\mathbf{W}_t$

Interface:
- `size()` — number of state variables.
- `factors()` — number of independent Brownian motions.
- `initialValues()`, `drift(t, x)`, `diffusion(t, x)`
- `expectation()`, `stdDeviation()`, `covariance()`, `evolve()` — discretization methods.
- Has a pluggable `discretization` strategy object.

### 8.2 `StochasticProcess1D`

Scalar specialization with `Real` instead of `Array`/`Matrix`.

### 8.3 Concrete Processes

| Process | Description |
|---|---|
| `GeneralizedBlackScholesProcess` | GBM with term structure of rates and vols |
| `HestonProcess` | Stochastic volatility (Heston 1993) |
| `BatesProcess` | Heston + jumps |
| `HullWhiteProcess` | Short-rate (Hull-White) |
| `G2Process` | Two-factor Gaussian short-rate |
| `Merton76Process` | Jump-diffusion |
| `CoxIngersollRossProcess` | CIR mean-reverting |
| `OrnsteinUhlenbeckProcess` | Mean-reverting OU |
| `GeometricBrownianMotionProcess` | Simple GBM |
| `GJRGARCHProcess` | GJR-GARCH volatility |
| `HybridHestonHullWhiteProcess` | Combined equity + rates |

---

## 9. Calibrated Models (`ql/models/`)

### 9.1 Base: `CalibratedModel`

Inherits `Observer` + `Observable`. Provides:
- `calibrate(helpers, method, endCriteria)` — optimize model parameters to match market prices.
- `params()` / `setParams()` — get/set parameter array.
- `arguments_` — vector of `Parameter` objects (each with constraints, transformation).
- `generateArguments()` — hook called after parameter updates.

### 9.2 `ShortRateModel` (extends `CalibratedModel`)

Adds: `virtual shared_ptr<Lattice> tree(const TimeGrid&) const = 0`

Concrete one-factor models:
- **`Vasicek`** — $dr = a(b - r)dt + \sigma dW$
- **`HullWhite`** — time-dependent Vasicek, fits initial term structure
- **`BlackKarasinski`** — lognormal short rate
- **`CoxIngersollRoss`** — $dr = a(b-r)dt + \sigma\sqrt{r}dW$
- **`GSR`** — Gaussian Short Rate (Gaussian 1-factor)
- **`MarkovFunctional`** — Markov-functional model

Two-factor: `G2` (two correlated Gaussian factors).

### 9.3 Equity Models

- **`HestonModel`** — parameterized Heston stochastic volatility.
- **`BatesModel`** — Heston + Merton jumps.
- **`GJRGARCHModel`** — GJR-GARCH(1,1) model.
- **`HestonSLVFDMModel`** / **`HestonSLVMCModel`** — stochastic-local volatility.

### 9.4 Model Hierarchy

- `AffineModel` — interface for analytically tractable models (`discount`, `discountBondOption`).
- `TermStructureConsistentModel` — holds a `Handle<YieldTermStructure>`, ensuring consistency.

---

## 10. Cash Flows (`cashflow.hpp`, `ql/cashflows/`)

### 10.1 Base: `CashFlow` (extends `Event` + `LazyObject`)

- `virtual Real amount() const = 0`
- `virtual Date date() const = 0`
- `Leg` typedef = `vector<shared_ptr<CashFlow>>`

### 10.2 Concrete Cash Flows

| Type | Description |
|---|---|
| `SimpleCashFlow` | Fixed amount on a date |
| `Coupon` | Base for interest coupons (adds `rate()`, `accrualPeriod()`, `nominal()`) |
| `FixedRateCoupon` | Fixed-rate coupon |
| `FloatingRateCoupon` | Base for IBOR/OIS-linked coupons |
| `IborCoupon` | IBOR-indexed floating coupon |
| `OvernightIndexedCoupon` | Compounded overnight rate coupon |
| `CMSCoupon` | CMS-rate linked coupon |
| `InflationCoupon` | CPI/YoY inflation-linked coupon |
| `DigitalCoupon` | Coupon with digital (binary) feature |
| `CappedFlooredCoupon` | Coupon with cap/floor |
| `Dividend` | Discrete dividend cash flow |

### 10.3 Coupon Pricers (`couponpricer.hpp`)

Floating coupons delegate convexity adjustment and option pricing to `FloatingRateCouponPricer` objects — another example of the strategy pattern.

---

## 11. Indexes (`index.hpp`, `ql/indexes/`)

### 11.1 Base: `Index` (extends `Observable` + `Observer`)

- `name()`, `fixingCalendar()`, `isValidFixingDate()`
- `fixing(date)` — returns historical fixing or forecast.
- `addFixing()` / `addFixings()` — store historical data.
- Uses `IndexManager` singleton for global fixing storage.

### 11.2 Concrete Indexes

| Category | Examples |
|---|---|
| **Interest Rate** | `InterestRateIndex` → `IborIndex` (Euribor, USD LIBOR, SOFR, ESTR, etc.) |
| **Overnight** | `OvernightIndex` (SOFR, ESTR, SONIA, CORRA, etc.) |
| **Swap** | `SwapIndex` — represents a swap rate for a given tenor |
| **Inflation** | `InflationIndex`, `ZeroInflationIndex`, `YoYInflationIndex` |
| **Equity** | `EquityIndex` |

---

## 12. Exercise Types (`exercise.hpp`)

- `Exercise::Type` enum: `American`, `Bermudan`, `European`
- `AmericanExercise`, `BermudanExercise`, `EuropeanExercise`

---

## 13. Payoffs (`payoff.hpp`, `ql/instruments/payoffs.hpp`)

- `Payoff` → `TypePayoff` → `StrikedTypePayoff` → `PlainVanillaPayoff`, `CashOrNothingPayoff`, `AssetOrNothingPayoff`, `GapPayoff`, `SuperSharePayoff`, etc.
- `Option::Type` enum: `Call`, `Put`

---

## 14. Math & Numerical Methods (`ql/math/`, `ql/methods/`)

### 14.1 Math Library

- **Arrays/Matrices** — `Array`, `Matrix` (basic linear algebra).
- **Interpolation** — `LinearInterpolation`, `CubicInterpolation`, `SABRInterpolation`, `LogLinearInterpolation`, etc.
- **Optimization** — `LevenbergMarquardt`, `Simplex`, `ConjugateGradient`, `EndCriteria`, `Constraint`.
- **Distributions** — normal, chi-squared, Poisson, etc.
- **Integration** — `GaussLobattoIntegral`, `SimpsonIntegral`, `TrapezoidIntegral`.
- **Root finding** — `Brent`, `Newton`, `Bisection`, `Ridder`.
- **Random numbers** — Mersenne Twister, Sobol sequences, inverse cumulative.
- **FFT** — `FastFourierTransform`.

### 14.2 Numerical Methods

- **Finite Differences** (`ql/methods/finitedifferences/`) — operators, meshers, solvers for PDE-based pricing.
- **Lattices** (`ql/methods/lattices/`) — binomial/trinomial trees, `Lattice`, `TreeLattice`.
- **Monte Carlo** (`ql/methods/montecarlo/`) — path generators, path pricers, `MonteCarloModel`.

---

## 15. Experimental (`ql/experimental/`)

Contains newer, less-stable features:
- Callable bonds, CAT bonds, variance gamma, Asian options extensions, short rate model extensions, exotic options, commodities, FX barriers, credit models, basis models.

---

## 16. Architecture Summary — Object Dependency Graph

```
Settings (singleton, holds evaluationDate)
    │
    ▼
Quote (market observable, e.g., spot price, rate)
    │
    ▼
Handle<T> / RelinkableHandle<T> (relinkable smart pointer)
    │
    ├──► TermStructure (yield, vol, default, inflation)
    │        │
    │        ├── YieldTermStructure (discount, zero, forward rates)
    │        ├── BlackVolTermStructure (BS implied vol surface)
    │        ├── DefaultProbabilityTermStructure (credit)
    │        └── InflationTermStructure
    │
    ├──► Index (rate fixings, forecasting via term structures)
    │
    └──► StochasticProcess (drift, diffusion — uses term structures)
              │
              ▼
         CalibratedModel (parameters, calibration to helpers)
              │
              ▼
         PricingEngine (calculates using process/model + arguments)
              │
              ▼
         Instrument (holds engine, delegates calculation)
              │
              ├── results: NPV, error estimate, greeks, etc.
              └── contains: Legs (vectors of CashFlows), Payoffs, Exercises
```

**Notification flow:** Quote changes → Handle notifies → TermStructure marks dirty → Instrument marks dirty → next call to `NPV()` triggers full recalculation through the engine.

---

## 17. Rust Re-implementation — Detailed Design Guide

### 17.1 Quick-Reference Pattern Mapping

| QuantLib C++ Pattern | Rust Equivalent |
|---|---|
| `Observer/Observable` | Trait-based callbacks, channels (`tokio::watch`), or a custom signal system |
| `LazyObject` (mutable cache) | `Cell<bool>` + `RefCell<T>` or `OnceCell` / `OnceLock` patterns |
| `Handle<T>` (relinkable ptr) | `Arc<RwLock<Option<Arc<dyn T>>>>` with notification |
| `shared_ptr<T>` | `Arc<T>` or `Rc<T>` |
| `GenericEngine<Args, Results>` | Trait with associated types: `type Arguments; type Results;` |
| Virtual dispatch (inheritance) | Trait objects (`dyn Trait`) or enums |
| Template metaprogramming | Generics + trait bounds |
| Acyclic Visitor | Enum dispatch or `downcast` crate |
| Singleton | `once_cell::sync::Lazy` or `std::sync::OnceLock` |
| `mutable` members | Interior mutability (`Cell`, `RefCell`, `Mutex`) |
| Day counters / calendars | Trait + enum dispatch (no heap allocation needed) |
| Piecewise curve bootstrapping | Generic over interpolation trait + bootstrap trait |

---

### 17.2 Recommended Crate Ecosystem

| Domain | Crate | Purpose |
|---|---|---|
| Linear algebra | `nalgebra` | Dense vectors/matrices, replaces QuantLib `Array`/`Matrix` |
| Linear algebra (alt) | `ndarray` | N-dimensional arrays, good for grids/paths |
| Dates | `chrono` or `time` | Date arithmetic (wrap with serial-number `Date` for performance) |
| Serialization | `serde` + `serde_json` | Serialize curves, fixings, instrument configs |
| Error handling | `thiserror` + `anyhow` | Domain-specific errors + ad-hoc error propagation |
| Logging | `tracing` | Structured logging with span-based diagnostics |
| Parallelism | `rayon` | Data-parallel Monte Carlo paths, parallel bootstrapping |
| Async (optional) | `tokio` | If real-time pricing/streaming market data is needed |
| Random numbers | `rand` + `rand_distr` | Mersenne Twister, normal/uniform distributions |
| Quasi-random | `sobol_burley` or custom | Sobol sequences for quasi-Monte Carlo |
| Optimization | `argmin` | Levenberg-Marquardt, Nelder-Mead, BFGS — replaces QL optimization |
| Root finding | `roots` or custom | Brent, Newton, bisection solvers |
| Interpolation | `interp` or custom | Linear, cubic, log-linear interpolation |
| FFT | `rustfft` | Fast Fourier Transform for characteristic-function pricing |
| Statistical distributions | `statrs` | Normal CDF/PDF, chi-squared, Poisson, etc. |
| Numerical integration | `quadrature` or `gauss_quad` | Gauss-Legendre, Gauss-Lobatto |
| Testing | `approx` | Floating-point approximate equality assertions |
| Benchmarking | `criterion` | Micro-benchmarks for pricing engines |

---

### 17.3 Proposed Crate / Module Layout

```
ql-rust/
├── Cargo.toml                    # workspace root
├── crates/
│   ├── ql-core/                  # foundational types + patterns
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── types.rs          # Real, Rate, Spread, DiscountFactor, Size, etc.
│   │       ├── errors.rs         # QLError enum (thiserror)
│   │       ├── settings.rs       # evaluation date, global settings
│   │       ├── observable.rs     # Observer/Observable trait + registry
│   │       ├── lazy.rs           # LazyObject trait + CachedValue<T>
│   │       └── handle.rs         # Handle<T>, RelinkableHandle<T>
│   │
│   ├── ql-time/                  # date, calendar, day counter, schedule
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── date.rs           # serial-number Date
│   │       ├── period.rs         # Period, TimeUnit, Frequency
│   │       ├── calendar.rs       # Calendar trait + enum dispatch
│   │       ├── calendars/        # concrete calendars by country
│   │       │   ├── united_states.rs
│   │       │   ├── united_kingdom.rs
│   │       │   ├── target.rs
│   │       │   └── ...
│   │       ├── day_counter.rs    # DayCounter trait + enum dispatch
│   │       ├── day_counters/     # Act360, Act365Fixed, Thirty360, etc.
│   │       ├── schedule.rs       # Schedule builder
│   │       ├── business_day_convention.rs
│   │       └── imm.rs            # IMM dates
│   │
│   ├── ql-math/                  # numerical library
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── interpolation.rs  # Interpolation trait
│   │       ├── interpolations/   # Linear, LogLinear, Cubic, SABR, etc.
│   │       ├── optimization.rs   # optimizer traits + EndCriteria
│   │       ├── solvers/          # Brent, Newton, Bisection
│   │       ├── distributions.rs  # normal, chi-sq, Poisson
│   │       ├── integration.rs    # numerical quadrature
│   │       ├── rng.rs            # random number generators
│   │       ├── sobol.rs          # Sobol quasi-random sequences
│   │       └── matrix.rs         # thin wrappers or re-exports from nalgebra
│   │
│   ├── ql-currencies/            # Currency, Money, ExchangeRate
│   │
│   ├── ql-indexes/               # Index trait, IborIndex, OvernightIndex, etc.
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── index.rs          # Index trait + IndexManager
│   │       ├── interest_rate_index.rs
│   │       ├── ibor/             # Euribor, UsdLibor, Sofr, etc.
│   │       ├── overnight.rs
│   │       ├── swap_index.rs
│   │       └── inflation.rs
│   │
│   ├── ql-termstructures/        # term structure traits + implementations
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── term_structure.rs          # base TermStructure trait
│   │       ├── yield_ts.rs                # YieldTermStructure trait
│   │       ├── yield_curves/              # FlatForward, PiecewiseYieldCurve, etc.
│   │       ├── vol_ts.rs                  # BlackVolTermStructure trait
│   │       ├── vol_surfaces/              # BlackConstantVol, BlackVarianceSurface, etc.
│   │       ├── local_vol.rs
│   │       ├── default_ts.rs              # DefaultProbabilityTermStructure
│   │       ├── inflation_ts.rs
│   │       ├── bootstrap.rs               # IterativeBootstrap generic
│   │       └── rate_helpers.rs            # BootstrapHelper, DepositRateHelper, etc.
│   │
│   ├── ql-cashflows/             # CashFlow, Coupon, Leg
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── cashflow.rs       # CashFlow trait
│   │       ├── coupon.rs         # Coupon trait
│   │       ├── fixed_rate_coupon.rs
│   │       ├── floating_rate_coupon.rs
│   │       ├── ibor_coupon.rs
│   │       ├── overnight_coupon.rs
│   │       ├── coupon_pricer.rs
│   │       └── leg.rs            # type Leg = Vec<Box<dyn CashFlow>>
│   │
│   ├── ql-instruments/           # Instrument trait + concrete instruments
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── instrument.rs     # Instrument trait
│   │       ├── payoff.rs         # Payoff enum
│   │       ├── exercise.rs       # Exercise enum
│   │       ├── option.rs         # VanillaOption, BarrierOption, etc.
│   │       ├── swap.rs           # Swap, VanillaSwap
│   │       ├── bond.rs           # Bond, FixedRateBond, FloatingRateBond
│   │       ├── forward.rs
│   │       └── credit.rs         # CDS
│   │
│   ├── ql-processes/             # stochastic processes
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── process.rs        # StochasticProcess trait
│   │       ├── black_scholes.rs  # GeneralizedBlackScholesProcess
│   │       ├── heston.rs
│   │       ├── hull_white.rs
│   │       └── ...
│   │
│   ├── ql-models/                # calibrated models
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── model.rs          # CalibratedModel trait
│   │       ├── parameter.rs      # Parameter with constraints
│   │       ├── short_rate/       # Vasicek, HullWhite, BlackKarasinski, etc.
│   │       └── equity/           # HestonModel, BatesModel
│   │
│   ├── ql-pricingengines/        # pricing engines
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── engine.rs         # PricingEngine trait + GenericEngine
│   │       ├── analytic/         # AnalyticEuropeanEngine, AnalyticHestonEngine
│   │       ├── fd/               # finite difference engines
│   │       ├── mc/               # Monte Carlo engines
│   │       ├── lattice/          # binomial/trinomial tree engines
│   │       ├── bond/
│   │       └── swap/
│   │
│   └── ql-methods/               # numerical pricing methods
│       └── src/
│           ├── lib.rs
│           ├── fd/               # FD operators, meshers, step conditions
│           ├── lattice/          # tree lattice framework
│           └── mc/               # path generators, path pricers
│
├── examples/                     # runnable examples
│   ├── price_european_option.rs
│   ├── bootstrap_yield_curve.rs
│   └── calibrate_heston.rs
│
└── tests/                        # integration tests
    ├── test_day_counters.rs
    ├── test_black_scholes.rs
    └── test_swap_pricing.rs
```

---

### 17.4 Core Pattern Implementations in Rust

#### 17.4.1 Observer / Observable

QuantLib's observer pattern uses raw pointer sets with manual registration. In Rust, we can use a callback-based approach with `Arc` and weak references to avoid cycles:

```rust
use std::sync::{Arc, Weak, RwLock};

/// Unique identifier for an observer registration.
type ObserverId = u64;

/// An observable that notifies registered observers when state changes.
pub struct ObservableState {
    next_id: ObserverId,
    observers: Vec<(ObserverId, Weak<dyn Observer>)>,
}

pub trait Observable {
    fn state(&self) -> &RwLock<ObservableState>;

    fn register_observer(&self, observer: &Arc<dyn Observer>) -> ObserverId {
        let mut state = self.state().write().unwrap();
        let id = state.next_id;
        state.next_id += 1;
        state.observers.push((id, Arc::downgrade(observer)));
        id
    }

    fn unregister_observer(&self, id: ObserverId) {
        let mut state = self.state().write().unwrap();
        state.observers.retain(|(oid, _)| *oid != id);
    }

    fn notify_observers(&self) {
        let state = self.state().read().unwrap();
        for (_, weak) in &state.observers {
            if let Some(observer) = weak.upgrade() {
                observer.update();
            }
        }
    }
}

pub trait Observer: Send + Sync {
    fn update(&self);
}
```

**Alternative (simpler, single-threaded):** Use `Rc<RefCell<>>` and skip `Send + Sync` if thread safety is not needed.

**Alternative (channel-based):** Use `tokio::sync::watch` channels for reactive propagation without manual registration. Each `watch::Receiver` clones the latest value when polled.

#### 17.4.2 LazyObject / CachedValue

QuantLib's `LazyObject` uses `mutable` members for on-demand caching. In Rust, interior mutability achieves the same:

```rust
use std::cell::Cell;

/// Mixin for lazy calculation with caching.
pub struct LazyCache {
    calculated: Cell<bool>,
    frozen: Cell<bool>,
}

impl LazyCache {
    pub fn new() -> Self {
        Self { calculated: Cell::new(false), frozen: Cell::new(false) }
    }

    pub fn is_calculated(&self) -> bool { self.calculated.get() }

    /// Mark as dirty (called when a dependency changes).
    pub fn invalidate(&self) {
        if !self.frozen.get() {
            self.calculated.set(false);
        }
    }

    /// Execute `f` only if not already calculated (and not frozen).
    pub fn ensure_calculated<F: FnOnce()>(&self, f: F) {
        if !self.calculated.get() && !self.frozen.get() {
            self.calculated.set(true); // set first to prevent recursion
            f();
        }
    }

    pub fn freeze(&self) { self.frozen.set(true); }
    pub fn unfreeze(&self) { self.frozen.set(false); }
}
```

For cached results that hold a value:

```rust
use std::cell::RefCell;

/// A lazily-computed, cached value.
pub struct Cached<T> {
    value: RefCell<Option<T>>,
    valid: Cell<bool>,
}

impl<T> Cached<T> {
    pub fn new() -> Self {
        Self { value: RefCell::new(None), valid: Cell::new(false) }
    }

    pub fn get_or_compute<F: FnOnce() -> T>(&self, f: F) -> std::cell::Ref<T> {
        if !self.valid.get() {
            *self.value.borrow_mut() = Some(f());
            self.valid.set(true);
        }
        std::cell::Ref::map(self.value.borrow(), |v| v.as_ref().unwrap())
    }

    pub fn invalidate(&self) { self.valid.set(false); }
}
```

#### 17.4.3 Handle / RelinkableHandle

The key insight: all clones of a `Handle` share the same inner `Link`, and relinking propagates to all holders.

```rust
use std::sync::{Arc, RwLock};

/// Shared, relinkable reference to a term structure or observable.
/// All clones see the same underlying object.
pub struct Handle<T: ?Sized> {
    link: Arc<RwLock<Link<T>>>,
}

struct Link<T: ?Sized> {
    inner: Option<Arc<T>>,
    // Could add observer notification hooks here
}

impl<T: ?Sized> Handle<T> {
    pub fn new(obj: Arc<T>) -> Self {
        Self { link: Arc::new(RwLock::new(Link { inner: Some(obj) })) }
    }

    pub fn empty() -> Self {
        Self { link: Arc::new(RwLock::new(Link { inner: None })) }
    }

    /// Dereference — panics if empty.
    pub fn get(&self) -> Arc<T> {
        self.link.read().unwrap().inner.as_ref()
            .expect("empty Handle dereferenced").clone()
    }

    pub fn is_empty(&self) -> bool {
        self.link.read().unwrap().inner.is_none()
    }
}

impl<T: ?Sized> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Self { link: Arc::clone(&self.link) }
    }
}

/// Only the owner of a RelinkableHandle can call `link_to`.
pub struct RelinkableHandle<T: ?Sized> {
    handle: Handle<T>,
}

impl<T: ?Sized> RelinkableHandle<T> {
    pub fn new(obj: Arc<T>) -> Self {
        Self { handle: Handle::new(obj) }
    }

    pub fn handle(&self) -> Handle<T> {
        self.handle.clone()
    }

    /// Relink — all holders of cloned Handles now see the new object.
    pub fn link_to(&self, obj: Arc<T>) {
        self.handle.link.write().unwrap().inner = Some(obj);
        // TODO: notify observers
    }
}
```

#### 17.4.4 Settings (Global State)

```rust
use std::sync::OnceLock;
use std::sync::RwLock;

pub struct Settings {
    evaluation_date: RwLock<Option<Date>>,
    include_reference_date_events: RwLock<bool>,
    include_todays_cashflows: RwLock<Option<bool>>,
}

static SETTINGS: OnceLock<Settings> = OnceLock::new();

impl Settings {
    pub fn instance() -> &'static Settings {
        SETTINGS.get_or_init(|| Settings {
            evaluation_date: RwLock::new(None),
            include_reference_date_events: RwLock::new(false),
            include_todays_cashflows: RwLock::new(None),
        })
    }

    pub fn evaluation_date(&self) -> Date {
        self.evaluation_date.read().unwrap()
            .unwrap_or_else(Date::today)
    }

    pub fn set_evaluation_date(&self, date: Date) {
        *self.evaluation_date.write().unwrap() = Some(date);
        // notify all registered observers
    }
}
```

---

### 17.5 Core Trait Hierarchies

#### 17.5.1 Date, Calendar, DayCounter (Zero-Cost Abstractions)

These are QuantLib's most-called types. In Rust, use enums + match dispatch to avoid heap allocation:

```rust
/// Serial-number date (days from a fixed epoch).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Date(i32); // serial number

impl Date {
    pub fn from_ymd(year: i32, month: u32, day: u32) -> Self { /* ... */ }
    pub fn today() -> Self { /* chrono::Local::now() → serial */ }
    pub fn serial(&self) -> i32 { self.0 }
    pub fn weekday(&self) -> Weekday { /* ... */ }
    pub fn year(&self) -> i32 { /* ... */ }
    pub fn month(&self) -> Month { /* ... */ }
    pub fn day_of_month(&self) -> u32 { /* ... */ }
}

impl std::ops::Add<i32> for Date {
    type Output = Date;
    fn add(self, days: i32) -> Date { Date(self.0 + days) }
}

impl std::ops::Sub<Date> for Date {
    type Output = i32;
    fn sub(self, other: Date) -> i32 { self.0 - other.0 }
}

/// Enum-based calendar — no vtable, no heap.
#[derive(Debug, Clone, Copy)]
pub enum Calendar {
    Target,
    UnitedStates(USMarket),
    UnitedKingdom(UKMarket),
    Japan,
    NullCalendar,
    // ...
}

impl Calendar {
    pub fn is_business_day(&self, date: Date) -> bool {
        match self {
            Calendar::Target => target::is_business_day(date),
            Calendar::UnitedStates(m) => us::is_business_day(date, *m),
            // ...
        }
    }

    pub fn advance(&self, date: Date, period: Period, convention: BusinessDayConvention) -> Date {
        // ...
    }
}

/// Enum-based day counter — no vtable, no heap.
#[derive(Debug, Clone, Copy)]
pub enum DayCounter {
    Actual360,
    Actual365Fixed,
    Thirty360(Thirty360Convention),
    ActualActual(ActualActualConvention),
    Business252(Calendar),
}

impl DayCounter {
    pub fn year_fraction(&self, d1: Date, d2: Date) -> f64 {
        match self {
            DayCounter::Actual360 => (d2 - d1) as f64 / 360.0,
            DayCounter::Actual365Fixed => (d2 - d1) as f64 / 365.0,
            // ...
        }
    }

    pub fn day_count(&self, d1: Date, d2: Date) -> i32 {
        match self { /* ... */ }
    }
}
```

**Why enums over trait objects?** Calendars and day counters are called millions of times during curve bootstrapping and MC simulation. Enum dispatch is inlined and branch-predicted; trait-object dispatch has vtable overhead and prevents inlining.

#### 17.5.2 InterestRate

```rust
#[derive(Debug, Clone, Copy)]
pub enum Compounding { Simple, Compounded, Continuous, SimpleThenCompounded }

#[derive(Debug, Clone, Copy)]
pub enum Frequency { Once, Annual, Semiannual, Quarterly, Monthly, Weekly, Daily }

#[derive(Debug, Clone, Copy)]
pub struct InterestRate {
    pub rate: f64,
    pub day_counter: DayCounter,
    pub compounding: Compounding,
    pub frequency: Frequency,
}

impl InterestRate {
    pub fn discount_factor(&self, t: f64) -> f64 {
        1.0 / self.compound_factor(t)
    }

    pub fn compound_factor(&self, t: f64) -> f64 {
        match self.compounding {
            Compounding::Simple => 1.0 + self.rate * t,
            Compounding::Compounded => {
                let f = self.frequency as i32 as f64;
                (1.0 + self.rate / f).powf(f * t)
            }
            Compounding::Continuous => (self.rate * t).exp(),
            Compounding::SimpleThenCompounded => {
                if t <= 1.0 / (self.frequency as i32 as f64) {
                    1.0 + self.rate * t
                } else {
                    let f = self.frequency as i32 as f64;
                    (1.0 + self.rate / f).powf(f * t)
                }
            }
        }
    }

    pub fn implied_rate(compound: f64, dc: DayCounter, comp: Compounding,
                        freq: Frequency, t: f64) -> Self { /* ... */ }

    pub fn equivalent_rate(&self, comp: Compounding, freq: Frequency, t: f64) -> Self { /* ... */ }
}
```

#### 17.5.3 Term Structures

```rust
/// Base term structure trait.
pub trait TermStructure: Send + Sync {
    fn reference_date(&self) -> Date;
    fn day_counter(&self) -> DayCounter;
    fn calendar(&self) -> Calendar;
    fn max_date(&self) -> Date;
    fn settlement_days(&self) -> u32 { 0 }

    fn time_from_reference(&self, date: Date) -> f64 {
        self.day_counter().year_fraction(self.reference_date(), date)
    }
}

/// Yield (discount) term structure.
pub trait YieldTermStructure: TermStructure {
    /// Override ONE of these three — the others are derived automatically.
    fn discount_impl(&self, t: f64) -> f64;

    fn discount(&self, date: Date) -> f64 {
        self.discount_impl(self.time_from_reference(date))
    }

    fn zero_rate(&self, date: Date, dc: DayCounter, comp: Compounding,
                 freq: Frequency) -> InterestRate {
        let t = self.time_from_reference(date);
        let compound = 1.0 / self.discount_impl(t);
        InterestRate::implied_rate(compound, dc, comp, freq, t)
    }

    fn forward_rate(&self, d1: Date, d2: Date, dc: DayCounter,
                    comp: Compounding, freq: Frequency) -> InterestRate {
        let t1 = self.time_from_reference(d1);
        let t2 = self.time_from_reference(d2);
        let compound = self.discount_impl(t1) / self.discount_impl(t2);
        InterestRate::implied_rate(compound, dc, comp, freq, t2 - t1)
    }
}
```

**Concrete implementations:**

```rust
/// Flat forward rate curve.
pub struct FlatForward {
    reference_date: Date,
    rate: InterestRate,
    day_counter: DayCounter,
}

impl YieldTermStructure for FlatForward {
    fn discount_impl(&self, t: f64) -> f64 {
        self.rate.discount_factor(t)
    }
}

/// Piecewise bootstrapped yield curve — generic over interpolation.
pub struct PiecewiseYieldCurve<I: Interpolator> {
    reference_date: Date,
    day_counter: DayCounter,
    calendar: Calendar,
    nodes: Vec<(f64, f64)>,       // (time, value) pairs
    interpolator: I,
    // bootstrap state
    helpers: Vec<Box<dyn RateHelper>>,
    cache: LazyCache,
}
```

#### 17.5.4 Instrument + PricingEngine

This is where Rust's type system shines. Use associated types to create a type-safe instrument/engine pairing:

```rust
/// Results common to all instruments.
#[derive(Debug, Default)]
pub struct InstrumentResults {
    pub npv: Option<f64>,
    pub error_estimate: Option<f64>,
    pub valuation_date: Option<Date>,
    pub additional_results: HashMap<String, f64>,
}

/// A pricing engine for a specific instrument type.
pub trait PricingEngine<I: Instrument> {
    fn calculate(&self, args: &I::Arguments) -> I::Results;
}

/// Base instrument trait.
pub trait Instrument: Sized {
    type Arguments;
    type Results: Into<InstrumentResults>;

    fn is_expired(&self) -> bool;
    fn build_arguments(&self) -> Self::Arguments;
}

/// Wrapper that adds lazy caching and engine dispatch.
pub struct PricedInstrument<I: Instrument> {
    instrument: I,
    engine: Option<Box<dyn PricingEngine<I, Results = I::Results>>>,
    cache: LazyCache,
    cached_results: RefCell<Option<I::Results>>,
}

impl<I: Instrument> PricedInstrument<I> {
    pub fn npv(&self) -> f64 {
        self.ensure_calculated();
        self.cached_results.borrow().as_ref()
            .and_then(|r| Into::<InstrumentResults>::into(r).npv)
            .expect("NPV not available")
    }

    pub fn set_pricing_engine(&mut self, engine: Box<dyn PricingEngine<I>>) {
        self.engine = Some(engine);
        self.cache.invalidate();
    }

    fn ensure_calculated(&self) {
        self.cache.ensure_calculated(|| {
            let engine = self.engine.as_ref().expect("no pricing engine set");
            let args = self.instrument.build_arguments();
            let results = engine.calculate(&args);
            *self.cached_results.borrow_mut() = Some(results);
        });
    }
}
```

**Concrete example — Vanilla Option:**

```rust
#[derive(Debug, Clone, Copy)]
pub enum OptionType { Call, Put }

#[derive(Debug, Clone, Copy)]
pub enum ExerciseType {
    European { expiry: Date },
    American { start: Date, expiry: Date },
    Bermudan { dates: Vec<Date> },
}

#[derive(Debug, Clone)]
pub struct VanillaOptionArgs {
    pub option_type: OptionType,
    pub strike: f64,
    pub exercise: ExerciseType,
}

#[derive(Debug, Default)]
pub struct VanillaOptionResults {
    pub base: InstrumentResults,
    pub delta: Option<f64>,
    pub gamma: Option<f64>,
    pub theta: Option<f64>,
    pub vega: Option<f64>,
    pub rho: Option<f64>,
}

pub struct VanillaOption {
    pub option_type: OptionType,
    pub strike: f64,
    pub exercise: ExerciseType,
}

impl Instrument for VanillaOption {
    type Arguments = VanillaOptionArgs;
    type Results = VanillaOptionResults;

    fn is_expired(&self) -> bool {
        match &self.exercise {
            ExerciseType::European { expiry } => *expiry < Settings::instance().evaluation_date(),
            ExerciseType::American { expiry, .. } => *expiry < Settings::instance().evaluation_date(),
            ExerciseType::Bermudan { dates } => dates.last().map_or(true, |d| *d < Settings::instance().evaluation_date()),
        }
    }

    fn build_arguments(&self) -> VanillaOptionArgs {
        VanillaOptionArgs {
            option_type: self.option_type,
            strike: self.strike,
            exercise: self.exercise.clone(),
        }
    }
}
```

**Concrete engine — Analytic Black-Scholes:**

```rust
pub struct AnalyticEuropeanEngine {
    process: Arc<BlackScholesProcess>,
}

impl PricingEngine<VanillaOption> for AnalyticEuropeanEngine {
    fn calculate(&self, args: &VanillaOptionArgs) -> VanillaOptionResults {
        let expiry = match args.exercise {
            ExerciseType::European { expiry } => expiry,
            _ => panic!("AnalyticEuropeanEngine requires European exercise"),
        };
        let t = self.process.time(expiry);
        let spot = self.process.spot();
        let df = self.process.risk_free_rate().discount(expiry);
        let div_df = self.process.dividend_yield().discount(expiry);
        let vol = self.process.volatility(t, spot);

        let d1 = ((spot * div_df / (args.strike * df)).ln() + 0.5 * vol * vol * t)
                  / (vol * t.sqrt());
        let d2 = d1 - vol * t.sqrt();

        let norm = Normal::new(0.0, 1.0).unwrap();
        let (npv, delta) = match args.option_type {
            OptionType::Call => {
                let npv = spot * div_df * norm.cdf(d1) - args.strike * df * norm.cdf(d2);
                let delta = div_df * norm.cdf(d1);
                (npv, delta)
            }
            OptionType::Put => {
                let npv = args.strike * df * norm.cdf(-d2) - spot * div_df * norm.cdf(-d1);
                let delta = -div_df * norm.cdf(-d1);
                (npv, delta)
            }
        };

        VanillaOptionResults {
            base: InstrumentResults { npv: Some(npv), ..Default::default() },
            delta: Some(delta),
            gamma: Some(div_df * norm.pdf(d1) / (spot * vol * t.sqrt())),
            vega: Some(spot * div_df * norm.pdf(d1) * t.sqrt()),
            theta: None, // compute if needed
            rho: None,
        }
    }
}
```

#### 17.5.5 Stochastic Processes

```rust
/// Multi-dimensional stochastic process.
pub trait StochasticProcess: Send + Sync {
    fn size(&self) -> usize;
    fn factors(&self) -> usize { self.size() }
    fn initial_values(&self) -> DVector<f64>;
    fn drift(&self, t: f64, x: &DVector<f64>) -> DVector<f64>;
    fn diffusion(&self, t: f64, x: &DVector<f64>) -> DMatrix<f64>;

    fn evolve(&self, t0: f64, x0: &DVector<f64>, dt: f64, dw: &DVector<f64>) -> DVector<f64> {
        let mu = self.drift(t0, x0);
        let sigma = self.diffusion(t0, x0);
        x0 + mu * dt + sigma * dw * dt.sqrt()
    }

    fn expectation(&self, t0: f64, x0: &DVector<f64>, dt: f64) -> DVector<f64> {
        x0 + self.drift(t0, x0) * dt
    }
}

/// 1-D specialization for performance.
pub trait StochasticProcess1D: Send + Sync {
    fn x0(&self) -> f64;
    fn drift(&self, t: f64, x: f64) -> f64;
    fn diffusion(&self, t: f64, x: f64) -> f64;

    fn evolve(&self, t0: f64, x0: f64, dt: f64, dw: f64) -> f64 {
        x0 + self.drift(t0, x0) * dt + self.diffusion(t0, x0) * dw * dt.sqrt()
    }

    fn expectation(&self, t0: f64, x0: f64, dt: f64) -> f64 {
        x0 + self.drift(t0, x0) * dt
    }

    fn variance(&self, t0: f64, x0: f64, dt: f64) -> f64 {
        let s = self.diffusion(t0, x0);
        s * s * dt
    }
}

/// Generalized Black-Scholes process.
pub struct BlackScholesProcess {
    spot: Handle<dyn Quote>,
    risk_free_rate: Handle<dyn YieldTermStructure>,
    dividend_yield: Handle<dyn YieldTermStructure>,
    volatility: Handle<dyn BlackVolTermStructure>,
}
```

#### 17.5.6 Calibrated Models

```rust
/// A parameter with constraints and transformations.
pub struct Parameter {
    values: Vec<f64>,
    constraint: Box<dyn Constraint>,
}

pub trait Constraint: Send + Sync {
    fn test(&self, params: &[f64]) -> bool;
    fn upper_bound(&self, params: &[f64]) -> Vec<f64>;
    fn lower_bound(&self, params: &[f64]) -> Vec<f64>;
}

/// Base trait for calibrated models.
pub trait CalibratedModel: Send + Sync {
    fn parameters(&self) -> &[Parameter];
    fn set_params(&mut self, params: &[f64]);

    fn calibrate(
        &mut self,
        helpers: &[Box<dyn CalibrationHelper>],
        optimizer: &mut dyn Optimizer,   // from argmin
        end_criteria: &EndCriteria,
    ) -> Result<(), QLError> {
        // Build cost function from helpers
        // Run optimizer
        // Set optimized parameters
        todo!()
    }
}

/// Short-rate model adds a tree builder.
pub trait ShortRateModel: CalibratedModel {
    fn tree(&self, grid: &TimeGrid) -> Box<dyn Lattice>;
}
```

#### 17.5.7 Cash Flows & Legs

```rust
/// Base cash flow trait.
pub trait CashFlow: Send + Sync {
    fn date(&self) -> Date;
    fn amount(&self) -> f64;
    fn has_occurred(&self, ref_date: Date) -> bool {
        self.date() < ref_date
    }
}

/// Coupon extends CashFlow with accrual info.
pub trait Coupon: CashFlow {
    fn nominal(&self) -> f64;
    fn rate(&self) -> f64;
    fn accrual_start(&self) -> Date;
    fn accrual_end(&self) -> Date;
    fn accrual_period(&self) -> f64;
    fn day_counter(&self) -> DayCounter;
}

/// A leg is a sequence of cash flows.
pub type Leg = Vec<Box<dyn CashFlow>>;

/// Fixed rate coupon — no heap allocation for the coupon itself.
pub struct FixedRateCoupon {
    pub payment_date: Date,
    pub nominal: f64,
    pub rate: InterestRate,
    pub accrual_start: Date,
    pub accrual_end: Date,
}

impl CashFlow for FixedRateCoupon {
    fn date(&self) -> Date { self.payment_date }
    fn amount(&self) -> f64 {
        self.nominal * self.rate.compound_factor(
            self.rate.day_counter().year_fraction(self.accrual_start, self.accrual_end)
        ) - self.nominal
    }
}
```

#### 17.5.8 Bootstrapping Framework

```rust
/// Trait for interpolation strategies.
pub trait Interpolator: Clone {
    type Interpolation: Interpolation;
    fn interpolate(&self, x: &[f64], y: &[f64]) -> Self::Interpolation;
}

pub trait Interpolation {
    fn value(&self, x: f64) -> f64;
    fn primitive(&self, x: f64) -> f64;   // integral
    fn derivative(&self, x: f64) -> f64;
}

/// Bootstrap trait — what the curve stores (discounts, zero rates, or forward rates).
pub trait BootstrapTraits {
    fn initial_value() -> f64;
    fn initial_guess() -> f64;
    fn max_iterations() -> usize { 100 }

    /// Convert from the interpolated quantity to a discount factor.
    fn discount(ts: &dyn YieldTermStructure, t: f64) -> f64;
}

/// Rate helper — a market instrument used to bootstrap a curve.
pub trait RateHelper {
    fn pillar_date(&self) -> Date;
    fn quote_value(&self) -> f64;
    fn implied_quote(&self, curve: &dyn YieldTermStructure) -> f64;
    fn quote_error(&self, curve: &dyn YieldTermStructure) -> f64 {
        self.implied_quote(curve) - self.quote_value()
    }
}

/// Generic piecewise curve.
pub struct PiecewiseYieldCurve<Traits: BootstrapTraits, I: Interpolator> {
    reference_date: Date,
    day_counter: DayCounter,
    helpers: Vec<Box<dyn RateHelper>>,
    times: Vec<f64>,
    data: Vec<f64>,
    interpolation: Option<I::Interpolation>,
    cache: LazyCache,
    _traits: PhantomData<Traits>,
}

impl<Traits: BootstrapTraits, I: Interpolator> PiecewiseYieldCurve<Traits, I> {
    /// Iterative bootstrap: solve for each node sequentially.
    fn bootstrap(&mut self) -> Result<(), QLError> {
        // Sort helpers by pillar date
        // For each helper, use root-finding (Brent) to solve for the
        // interpolation node value that makes implied_quote == market_quote
        todo!()
    }
}
```

---

### 17.6 Monte Carlo Framework in Rust

Rust's `rayon` crate makes parallelizing path generation trivial:

```rust
use rayon::prelude::*;
use rand::prelude::*;
use rand_distr::StandardNormal;

pub struct MonteCarloEngine<P: StochasticProcess1D, V: PathPricer> {
    process: P,
    pricer: V,
    num_paths: usize,
    time_steps: usize,
}

pub trait PathPricer: Send + Sync {
    fn value(&self, path: &[f64], times: &[f64]) -> f64;
}

impl<P: StochasticProcess1D + Send + Sync, V: PathPricer> MonteCarloEngine<P, V> {
    pub fn calculate(&self) -> (f64, f64) {
        let dt = self.maturity() / self.time_steps as f64;
        let times: Vec<f64> = (0..=self.time_steps).map(|i| i as f64 * dt).collect();

        // Parallel path generation with rayon
        let results: Vec<f64> = (0..self.num_paths)
            .into_par_iter()
            .map(|_| {
                let mut rng = thread_rng();
                let mut path = Vec::with_capacity(self.time_steps + 1);
                path.push(self.process.x0());

                for i in 0..self.time_steps {
                    let dw: f64 = rng.sample(StandardNormal);
                    let x = path.last().copied().unwrap();
                    path.push(self.process.evolve(times[i], x, dt, dw));
                }

                self.pricer.value(&path, &times)
            })
            .collect();

        let n = results.len() as f64;
        let mean = results.iter().sum::<f64>() / n;
        let variance = results.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std_error = (variance / n).sqrt();

        (mean, std_error)
    }
}
```

---

### 17.7 Error Handling Strategy

Replace QuantLib's `QL_REQUIRE` / `QL_ENSURE` / `QL_FAIL` macros with Rust's `Result` type:

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum QLError {
    #[error("null pricing engine")]
    NullEngine,

    #[error("empty handle cannot be dereferenced")]
    EmptyHandle,

    #[error("{field} not provided")]
    MissingResult { field: &'static str },

    #[error("date {0} is outside curve range")]
    DateOutOfRange(Date),

    #[error("negative {quantity}: {value}")]
    NegativeValue { quantity: &'static str, value: f64 },

    #[error("calibration failed: {0}")]
    CalibrationFailure(String),

    #[error("root not found after {0} iterations")]
    RootNotFound(usize),

    #[error("{0}")]
    Other(String),
}

pub type QLResult<T> = Result<T, QLError>;

// Usage — replaces QL_REQUIRE:
fn npv(&self) -> QLResult<f64> {
    self.ensure_calculated()?;
    self.cached_npv.ok_or(QLError::MissingResult { field: "NPV" })
}
```

---

### 17.8 Advantages of Rust over C++ for QuantLib

| Aspect | C++ QuantLib | Rust Re-implementation |
|---|---|---|
| **Memory safety** | Manual; shared_ptr cycles possible | Ownership system prevents leaks and dangling refs at compile time |
| **Thread safety** | Opt-in `#ifdef`; data races possible | `Send`/`Sync` enforced by compiler; fearless concurrency |
| **Parallelism** | Requires OpenMP or manual threading | `rayon` for trivial data-parallel MC; zero-cost |
| **Error handling** | Exceptions (hidden control flow) | `Result<T, E>` — errors are explicit values, zero-cost when Ok |
| **Build system** | CMake + autotools + vcproj chaos | `cargo` — unified, reproducible, cross-platform |
| **Dependencies** | Boost (huge), optional BLAS | Targeted crates: `nalgebra`, `rand`, `statrs`, etc. |
| **Performance** | Excellent (but UB risk) | Equivalent; no GC, zero-cost abstractions, SIMD via nalgebra |
| **Enum dispatch** | `virtual` + vtable overhead | `enum` + `match` = inlined, branch-predicted, no heap |
| **Testing** | Separate test suite, Boost.Test | Built-in `#[test]`, `#[bench]`, `cargo test` |
| **Documentation** | Doxygen (external) | `cargo doc` generates from `///` comments, with examples |
| **Package distribution** | Compile from source | `cargo publish` to crates.io |

---

### 17.9 Suggested Implementation Order

Build from the bottom up, testing each layer before moving to the next:

| Phase | Modules | Milestone |
|---|---|---|
| **Phase 1: Foundations** | `ql-core` (types, errors, settings), `ql-time` (Date, Calendar, DayCounter, Schedule) | Can generate coupon schedules |
| **Phase 2: Math** | `ql-math` (interpolation, root-finding, distributions, optimization) | Can bootstrap a curve |
| **Phase 3: Rates** | `ql-currencies`, `ql-indexes`, `ql-termstructures` (yield curves, flat forward, piecewise bootstrap) | Can build yield curves from market data |
| **Phase 4: Cash Flows** | `ql-cashflows` (fixed/floating coupons, legs, coupon pricers) | Can generate and value swap legs |
| **Phase 5: Instruments** | `ql-instruments` (instrument trait, VanillaOption, Swap, Bond), `ql-pricingengines` (AnalyticEuropeanEngine, DiscountingSwapEngine, DiscountingBondEngine) | Can price vanilla options, swaps, bonds |
| **Phase 6: Processes & Models** | `ql-processes` (Black-Scholes, Heston), `ql-models` (CalibratedModel, HullWhite, HestonModel) | Can calibrate Heston to market vols |
| **Phase 7: Advanced Engines** | `ql-methods` (MC, FD, lattice), MC/FD engines | Can price exotics and American options |
| **Phase 8: Vol Surfaces** | Vol term structures, SABR, smile interpolation, local vol | Full vol surface construction |
| **Phase 9: Credit & Inflation** | Default curves, CDS pricing, inflation curves/swaps | Full fixed-income coverage |
| **Phase 10: Experimental** | Callable bonds, exotic options, hybrid models | Feature parity |

---

### 17.10 Testing Strategy

Port QuantLib's test suite (`/mnt/c/finance/quantlib/test-suite/`) as integration tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_european_option_bs_price() {
        // Known-good values from QuantLib test suite
        let spot = 100.0;
        let strike = 100.0;
        let r = 0.05;
        let q = 0.02;
        let vol = 0.20;
        let t = 1.0;

        let engine = AnalyticEuropeanEngine::new(/* ... */);
        let option = VanillaOption::european(OptionType::Call, strike, expiry);
        let results = engine.calculate(&option.build_arguments());

        assert_relative_eq!(results.base.npv.unwrap(), 10.4506, epsilon = 1e-4);
        assert_relative_eq!(results.delta.unwrap(), 0.6368, epsilon = 1e-4);
    }

    #[test]
    fn test_yield_curve_bootstrap() {
        // Bootstrap from deposits + swaps, verify discount factors
    }

    #[test]
    fn test_heston_calibration() {
        // Calibrate to vol surface, verify parameters
    }
}
```

Use `criterion` for performance regression testing:

```rust
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_mc_european(c: &mut Criterion) {
    c.bench_function("MC European 100k paths", |b| {
        b.iter(|| {
            let engine = MCEuropeanEngine::new(process, 100_000, 252);
            engine.calculate(&args)
        })
    });
}

criterion_group!(benches, bench_mc_european);
criterion_main!(benches);
```

---

## 18. Persistence Layer — Technology Evaluation for SecDB/Beacon-Style Storage

A SecDB/Beacon-style system needs a persistence layer that supports:

1. **Versioned, typed objects** (instruments, trades, curves) — not flat relational rows.
2. **Event sourcing** — immutable lifecycle event log as the source of truth.
3. **Bitemporal queries** — "what did we know, and when did we know it?"
4. **Graph-aware storage** — objects reference each other (trade → instrument → curve → quotes).
5. **High-throughput reads** — pricing and risk loops read market data + trades millions of times.
6. **Snapshot + replay** — materialize current state for speed, replay events for audit.
7. **Schema evolution** — instrument types and fields change over time.

Below is an evaluation of candidate technologies, from simplest to most ambitious.

---

### 18.1 Option A: Embedded Rust-Native Storage (Recommended for Phase 1)

**Technology:** [`redb`](https://crates.io/crates/redb) or [`sled`](https://crates.io/crates/sled) + `serde`

| Aspect | Details |
|---|---|
| **What** | Embedded key-value store, no external server, compiled into the binary |
| **Serialization** | `serde` + `bincode` (fast binary) or `serde_json` (human-readable) |
| **Schema** | Defined by Rust types + `#[derive(Serialize, Deserialize)]` |
| **Versioning** | Key = `(ObjectId, Version)` tuple; append-only writes |
| **Transactions** | `redb` supports full ACID transactions |
| **Bitemporal** | Manual: store `(as_of_date, knowledge_time)` in the key |
| **Performance** | Excellent — single-digit μs reads, memory-mapped I/O |
| **Complexity** | Very low — `cargo add redb serde bincode` |

**Best for:** Local single-user pricing workstation, backtesting, research. No external infrastructure needed.

```rust
use redb::{Database, TableDefinition, ReadableTable};
use serde::{Serialize, Deserialize};

const TRADES: TableDefinition<&str, &[u8]> = TableDefinition::new("trades");
const EVENTS: TableDefinition<(u64, u64), &[u8]> = TableDefinition::new("events");
// key = (trade_id_hash, event_sequence_number)

pub struct EmbeddedStore {
    db: Database,
}

impl EmbeddedStore {
    pub fn open(path: &str) -> Result<Self> {
        let db = Database::create(path)?;
        Ok(Self { db })
    }

    pub fn save_trade<T: Serialize>(&self, id: &str, trade: &T) -> Result<()> {
        let bytes = bincode::serialize(trade)?;
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(TRADES)?;
            table.insert(id, bytes.as_slice())?;
        }
        txn.commit()?;
        Ok(())
    }

    pub fn load_trade<T: for<'de> Deserialize<'de>>(&self, id: &str) -> Result<T> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(TRADES)?;
        let bytes = table.get(id)?.ok_or(QLError::NotFound)?;
        Ok(bincode::deserialize(bytes.value())?)
    }

    pub fn append_event<E: Serialize>(&self, trade_id: u64, seq: u64, event: &E) -> Result<()> {
        let bytes = bincode::serialize(event)?;
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(EVENTS)?;
            table.insert((trade_id, seq), bytes.as_slice())?;
        }
        txn.commit()?;
        Ok(())
    }

    pub fn replay_events<E: for<'de> Deserialize<'de>>(&self, trade_id: u64) -> Result<Vec<E>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(EVENTS)?;
        let range = table.range((trade_id, 0)..=(trade_id, u64::MAX))?;
        range
            .map(|entry| {
                let (_, bytes) = entry?;
                Ok(bincode::deserialize(bytes.value())?)
            })
            .collect()
    }
}
```

**Crates:**

| Crate | Purpose |
|---|---|
| `redb` | Embedded ACID key-value store (pure Rust, no C deps) |
| `sled` | Alternative embedded store (higher-level, but less mature ACID) |
| `bincode` | Fast binary serialization (10-100x faster than JSON) |
| `serde` + `serde_json` | Serialization framework + JSON for debugging/export |
| `rkyv` | Zero-copy deserialization — read objects directly from mmap without parsing |

---

### 18.2 Option B: PostgreSQL with JSONB (Recommended for Production Multi-User)

**Technology:** PostgreSQL 15+ with JSONB columns + [`sqlx`](https://crates.io/crates/sqlx) or [`diesel`](https://crates.io/crates/diesel)

| Aspect | Details |
|---|---|
| **What** | Relational DB used as a versioned document store |
| **Schema** | Thin relational wrapper around JSONB documents |
| **Versioning** | Built-in: `version` column + `valid_from`/`valid_to` for bitemporal |
| **Bitemporal** | Native — PostgreSQL temporal tables or manual range columns |
| **Event sourcing** | Dedicated `lifecycle_events` table with append-only inserts |
| **Graph queries** | Recursive CTEs or `ltree` extension for object graph traversal |
| **Performance** | Good (10k-100k trades), GIN indexes on JSONB for fast lookups |
| **Concurrency** | MVCC — excellent multi-user concurrent access |
| **Ecosystem** | Mature: backups, replication, monitoring, cloud-managed (RDS/CloudSQL) |

**Schema design:**

```sql
-- Object type registry
CREATE TABLE object_types (
    type_name       TEXT PRIMARY KEY,
    parent_type     TEXT REFERENCES object_types(type_name),
    schema_version  INT NOT NULL DEFAULT 1
);

-- Core object table — versioned documents
CREATE TABLE objects (
    object_id       UUID NOT NULL,
    object_type     TEXT NOT NULL REFERENCES object_types(type_name),
    version         INT NOT NULL,
    valid_from      TIMESTAMPTZ NOT NULL DEFAULT now(),
    valid_to        TIMESTAMPTZ NOT NULL DEFAULT 'infinity',
    created_by      TEXT NOT NULL,
    data            JSONB NOT NULL,
    PRIMARY KEY (object_id, version)
);

-- GIN index for fast JSONB queries
CREATE INDEX idx_objects_data ON objects USING GIN (data);
-- Partial index for current versions only
CREATE INDEX idx_objects_current ON objects (object_id)
    WHERE valid_to = 'infinity';

-- Lifecycle events — append-only, immutable
CREATE TABLE lifecycle_events (
    event_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trade_id        UUID NOT NULL,
    event_type      TEXT NOT NULL,
    event_date      DATE NOT NULL,
    effective_date  DATE,
    entered_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    entered_by      TEXT NOT NULL,
    payload         JSONB NOT NULL,
    -- resulting new version of the trade
    resulting_version INT REFERENCES objects(version)
);

CREATE INDEX idx_events_trade ON lifecycle_events (trade_id, entered_at);

-- Market data snapshots
CREATE TABLE market_snapshots (
    snapshot_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_date   DATE NOT NULL,
    snapshot_time   TIMESTAMPTZ NOT NULL,
    snapshot_type   TEXT NOT NULL,  -- 'EOD', 'INTRADAY', 'CUSTOM'
    data            JSONB NOT NULL
);

-- Example queries:

-- Get current version of a trade
SELECT data FROM objects
WHERE object_id = $1 AND valid_to = 'infinity';

-- Get trade as it was known at a specific time (bitemporal)
SELECT data FROM objects
WHERE object_id = $1
  AND valid_from <= $2 AND valid_to > $2
ORDER BY version DESC LIMIT 1;

-- Get all lifecycle events for a trade
SELECT * FROM lifecycle_events
WHERE trade_id = $1
ORDER BY entered_at;

-- Find all active trades for a counterparty
SELECT data FROM objects
WHERE object_type = 'Trade'
  AND valid_to = 'infinity'
  AND data->>'status' = 'Active'
  AND data->>'counterparty' = $1;

-- Find all trades referencing a specific instrument
SELECT data FROM objects
WHERE object_type = 'Trade'
  AND valid_to = 'infinity'
  AND data->>'instrument_id' = $1;
```

**Rust integration with `sqlx`:**

```rust
use sqlx::{PgPool, FromRow};
use uuid::Uuid;
use chrono::{DateTime, Utc};

#[derive(Debug, FromRow)]
struct ObjectRow {
    object_id: Uuid,
    object_type: String,
    version: i32,
    valid_from: DateTime<Utc>,
    data: serde_json::Value,
}

pub struct PgObjectStore {
    pool: PgPool,
}

impl PgObjectStore {
    pub async fn get_current<T: for<'de> Deserialize<'de>>(
        &self, id: Uuid,
    ) -> Result<T> {
        let row = sqlx::query_as::<_, ObjectRow>(
            "SELECT * FROM objects WHERE object_id = $1 AND valid_to = 'infinity'"
        )
        .bind(id)
        .fetch_one(&self.pool)
        .await?;

        Ok(serde_json::from_value(row.data)?)
    }

    pub async fn get_as_of<T: for<'de> Deserialize<'de>>(
        &self, id: Uuid, as_of: DateTime<Utc>,
    ) -> Result<T> {
        let row = sqlx::query_as::<_, ObjectRow>(
            "SELECT * FROM objects
             WHERE object_id = $1 AND valid_from <= $2 AND valid_to > $2
             ORDER BY version DESC LIMIT 1"
        )
        .bind(id).bind(as_of)
        .fetch_one(&self.pool)
        .await?;

        Ok(serde_json::from_value(row.data)?)
    }

    pub async fn save_new_version<T: Serialize>(
        &self, id: Uuid, obj_type: &str, data: &T, user: &str,
    ) -> Result<i32> {
        let json = serde_json::to_value(data)?;
        let mut tx = self.pool.begin().await?;

        // Close current version
        sqlx::query(
            "UPDATE objects SET valid_to = now()
             WHERE object_id = $1 AND valid_to = 'infinity'"
        ).bind(id).execute(&mut *tx).await?;

        // Get next version number
        let next_version: i32 = sqlx::query_scalar(
            "SELECT COALESCE(MAX(version), 0) + 1 FROM objects WHERE object_id = $1"
        ).bind(id).fetch_one(&mut *tx).await?;

        // Insert new version
        sqlx::query(
            "INSERT INTO objects (object_id, object_type, version, created_by, data)
             VALUES ($1, $2, $3, $4, $5)"
        )
        .bind(id).bind(obj_type).bind(next_version).bind(user).bind(&json)
        .execute(&mut *tx).await?;

        tx.commit().await?;
        Ok(next_version)
    }

    pub async fn append_event(
        &self, trade_id: Uuid, event: &LifecycleEvent, user: &str,
    ) -> Result<Uuid> {
        let event_id = Uuid::new_v4();
        let payload = serde_json::to_value(event)?;

        sqlx::query(
            "INSERT INTO lifecycle_events
             (event_id, trade_id, event_type, event_date, entered_by, payload)
             VALUES ($1, $2, $3, $4, $5, $6)"
        )
        .bind(event_id).bind(trade_id)
        .bind(event.event_type_name())
        .bind(event.event_date())
        .bind(user).bind(&payload)
        .execute(&self.pool).await?;

        Ok(event_id)
    }
}
```

**Crates:**

| Crate | Purpose |
|---|---|
| `sqlx` | Async, compile-time-checked SQL queries for PostgreSQL |
| `diesel` | Alternative: type-safe ORM with schema migrations |
| `uuid` | UUID generation for object/event IDs |
| `chrono` | Timestamps for bitemporal columns |

---

### 18.3 Option C: EventStoreDB for Pure Event Sourcing

**Technology:** [EventStoreDB](https://www.eventstore.com/) + Rust gRPC client

| Aspect | Details |
|---|---|
| **What** | Purpose-built event-sourcing database |
| **Model** | Streams of immutable events per aggregate (e.g., per trade) |
| **Projections** | Built-in server-side projections to materialize read models |
| **Subscriptions** | Real-time push when new events arrive — ideal for live risk |
| **Bitemporal** | Natural: events are timestamped and never deleted |
| **Performance** | Optimized for append + sequential read (100k+ events/sec) |
| **Complexity** | Moderate — requires separate read-model store for queries |

**Pattern:**

```
Stream: "trade-TRD-2025-0012345"
  Event 0: TradeExecuted { instrument: "IRS_5Y", notional: 50M, ... }
  Event 1: TradeAmended  { field: "notional", new_value: 40M, ... }
  Event 2: CashSettled   { amount: -437500, date: 2025-12-17 }
  Event 3: CashSettled   { amount: -425000, date: 2026-06-17 }
  ...

Projection → Materialized "current_trades" read model (in Redis / Postgres)
```

**Best for:** Systems where the event log IS the product — regulatory reporting, P&L attribution, audit-heavy environments. Requires a separate read store (Postgres, Redis) for complex queries.

---

### 18.4 Option D: SurrealDB — Document + Graph + Temporal in One

**Technology:** [SurrealDB](https://surrealdb.com/) — a multi-model database written in Rust

| Aspect | Details |
|---|---|
| **What** | Document store + graph database + temporal support in a single engine |
| **Written in** | Rust — excellent FFI, can embed directly |
| **Schema** | Flexible: schemaless or strict, per-table |
| **Graph** | First-class graph edges: `RELATE trade:123 -> booked_in -> book:rates_nyc` |
| **Temporal** | Built-in versioning with `VERSION` clause |
| **Query language** | SurrealQL — SQL-like with graph traversal and record links |
| **Embedding** | Can run in-process (like SQLite) or as a server |
| **Maturity** | Younger than Postgres — fast-moving, API still evolving |

**Why it fits SecDB's model well:**

```sql
-- Define instrument with nested structure
CREATE instrument:irs_5y SET
    type = 'InterestRateSwap',
    fixed_rate = 0.035,
    float_index = index:usd_sofr,
    maturity = '5Y',
    legs = [
        { type: 'FixedLeg', rate: 0.035, frequency: 'Semiannual' },
        { type: 'FloatLeg', index: index:usd_sofr, spread: 0.001 }
    ];

-- Create trade referencing instrument via graph edge
CREATE trade:trd_001 SET
    notional = 50000000,
    direction = 'Payer',
    status = 'Active',
    trade_date = '2025-06-15';

-- Graph relationship
RELATE trade:trd_001 -> trades_instrument -> instrument:irs_5y;
RELATE trade:trd_001 -> counterparty -> entity:jpm;
RELATE trade:trd_001 -> booked_in -> book:rates_nyc;

-- Lifecycle event
CREATE event:evt_001 SET
    type = 'Amendment',
    payload = { field: 'notional', old: 50000000, new: 40000000 };
RELATE event:evt_001 -> applies_to -> trade:trd_001;

-- Query: all active trades for a counterparty, traversing graph
SELECT * FROM trade
    WHERE status = 'Active'
    AND ->counterparty->entity.name = 'JPMorgan';

-- Query: full lifecycle of a trade
SELECT * FROM event WHERE ->applies_to->trade = trade:trd_001
    ORDER BY time.created ASC;

-- Query: all trades referencing a specific curve (graph traversal)
SELECT <-trades_instrument<-trade.* FROM instrument
    WHERE float_index = index:usd_sofr;

-- Temporal: get trade as it was at a previous time
SELECT * FROM trade:trd_001 VERSION '2025-09-01T00:00:00Z';
```

**Rust integration (embedded mode):**

```rust
use surrealdb::Surreal;
use surrealdb::engine::local::RocksDb;  // embedded, no server needed

pub struct SurrealStore {
    db: Surreal<surrealdb::engine::local::Db>,
}

impl SurrealStore {
    pub async fn open(path: &str) -> Result<Self> {
        let db = Surreal::new::<RocksDb>(path).await?;
        db.use_ns("quantlib").use_db("trading").await?;
        Ok(Self { db })
    }

    pub async fn save_trade(&self, trade: &Trade) -> Result<()> {
        let _: Option<Trade> = self.db
            .create(("trade", &trade.trade_id))
            .content(trade)
            .await?;
        Ok(())
    }

    pub async fn get_trade(&self, id: &str) -> Result<Trade> {
        let trade: Option<Trade> = self.db
            .select(("trade", id))
            .await?;
        trade.ok_or(QLError::NotFound)
    }

    pub async fn get_trade_as_of(&self, id: &str, as_of: &str) -> Result<Trade> {
        let mut result = self.db
            .query(format!("SELECT * FROM trade:{} VERSION '{}'", id, as_of))
            .await?;
        let trade: Option<Trade> = result.take(0)?;
        trade.ok_or(QLError::NotFound)
    }

    pub async fn add_lifecycle_event(
        &self, trade_id: &str, event: &LifecycleEvent,
    ) -> Result<()> {
        let event_id = Uuid::new_v4().to_string();
        let _: Option<LifecycleEvent> = self.db
            .create(("event", &event_id))
            .content(event)
            .await?;
        // Create graph edge
        self.db.query(
            "RELATE event:$event_id -> applies_to -> trade:$trade_id"
        )
        .bind(("event_id", &event_id))
        .bind(("trade_id", trade_id))
        .await?;
        Ok(())
    }

    pub async fn active_trades_for_counterparty(&self, name: &str) -> Result<Vec<Trade>> {
        let mut result = self.db.query(
            "SELECT * FROM trade
             WHERE status = 'Active'
             AND ->counterparty->entity.name = $name"
        ).bind(("name", name)).await?;
        Ok(result.take(0)?)
    }
}
```

**Crates:**

| Crate | Purpose |
|---|---|
| `surrealdb` | SurrealDB Rust client (embedded or remote) |

---

### 18.5 Option E: Apache Kafka + Materialized Views (Large-Scale Streaming)

**Technology:** Kafka for the event log + a materialized read store (Postgres / Redis / ClickHouse)

| Aspect | Details |
|---|---|
| **What** | Distributed, durable event log with consumer-driven projections |
| **Throughput** | Millions of events/sec, horizontally scalable |
| **Event sourcing** | Natural: Kafka topics are append-only logs with retention |
| **Real-time** | Consumers get events in real time — ideal for live risk, P&L |
| **Read model** | Separate: Kafka Connect → Postgres/Redis/ClickHouse for queries |
| **Complexity** | High — requires Kafka cluster, schema registry, consumer orchestration |
| **Best for** | Large trading desks, multi-region, 100k+ trades, real-time streaming |

**Pattern:**

```
Kafka Topics:
  trades.lifecycle-events   ← all lifecycle events (partitioned by trade_id)
  market-data.quotes        ← real-time quote updates
  risk.results              ← computed risk metrics

Consumers:
  trade-state-projector     → reads events → writes current trade state to Postgres
  risk-engine               → reads events + quotes → computes NPV/Greeks → writes to risk topic
  pnl-calculator            → reads risk results → computes P&L → writes to pnl topic
  audit-archiver            → reads all events → writes to cold storage (S3/Parquet)
```

**Rust crates:**

| Crate | Purpose |
|---|---|
| `rdkafka` | High-performance Kafka client (wraps librdkafka) |
| `apache-avro` | Avro serialization for schema-registry compatibility |
| `protobuf` / `prost` | Protobuf serialization alternative |

---

### 18.6 Option F: DuckDB for Analytics + Parquet for Archive

**Technology:** [DuckDB](https://duckdb.org/) (embedded OLAP) + Parquet files

| Aspect | Details |
|---|---|
| **What** | Columnar analytical database, embedded (like SQLite but for analytics) |
| **Strength** | Blazing fast aggregation queries — portfolio-level risk, P&L attribution |
| **Storage** | Reads/writes Parquet files directly; also has its own storage |
| **Integration** | Can query Parquet on S3 directly |
| **Use case** | Not for OLTP (trade booking) — use for historical analytics, EOD snapshots |

**Pattern:** Use Postgres/SurrealDB for live trades → nightly export to Parquet → query with DuckDB.

```rust
use duckdb::Connection;

let conn = Connection::open("analytics.duckdb")?;

// Query historical P&L across all trades
conn.execute_batch("
    CREATE VIEW trade_pnl AS
    SELECT * FROM read_parquet('s3://data/pnl/2025/*.parquet');
")?;

let mut stmt = conn.prepare(
    "SELECT book, SUM(daily_pnl) as total_pnl, COUNT(*) as trade_count
     FROM trade_pnl
     WHERE trade_date BETWEEN '2025-01-01' AND '2025-12-31'
     GROUP BY book
     ORDER BY total_pnl DESC"
)?;
```

**Crates:**

| Crate | Purpose |
|---|---|
| `duckdb` | DuckDB Rust bindings |
| `parquet` (arrow-rs) | Read/write Parquet files |
| `arrow` | Apache Arrow in-memory columnar format |

---

### 18.7 Recommended Architecture — Layered Approach

Don't pick just one. Use the right tool for each concern:

```
┌────────────────────────────────────────────────────────────────────┐
│                         Application Layer                          │
│   Rust: ql-core, ql-instruments, ql-pricingengines, ql-models     │
│         (pure computation — no persistence dependency)             │
└───────────────────────────────┬────────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │  ObjectStore Trait     │  ← Abstract persistence API
                    │  (defined in ql-core)  │
                    └───────────┬───────────┘
                                │
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
┌────────▼────────┐  ┌──────────▼──────────┐  ┌───────▼────────┐
│  EmbeddedStore  │  │  PostgresStore      │  │  SurrealStore  │
│  (redb/sled)    │  │  (sqlx + JSONB)     │  │  (surrealdb)   │
│                 │  │                     │  │                │
│  Dev / Research │  │  Production OLTP    │  │  Graph-first   │
│  Single-user    │  │  Multi-user         │  │  Experimental  │
└─────────────────┘  └──────────┬──────────┘  └────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Event Log            │
                    │  (Postgres table or   │
                    │   Kafka for scale)    │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Analytics / Archive  │
                    │  DuckDB + Parquet     │
                    │  (historical queries) │
                    └───────────────────────┘
```

**The abstract trait that unifies them all:**

```rust
use async_trait::async_trait;

/// Unique object identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ObjectId(pub String);

/// Abstract object store — swap implementations without changing business logic.
#[async_trait]
pub trait ObjectStore: Send + Sync {
    /// Retrieve the current version of an object.
    async fn get<T: Persistable>(&self, id: &ObjectId) -> QLResult<T>;

    /// Retrieve an object as it was at a specific point in time.
    async fn get_as_of<T: Persistable>(
        &self, id: &ObjectId, as_of: DateTime<Utc>,
    ) -> QLResult<T>;

    /// Save a new version of an object (returns the new version number).
    async fn put<T: Persistable>(
        &self, id: &ObjectId, obj: &T, user: &str,
    ) -> QLResult<u64>;

    /// Append a lifecycle event (returns event id).
    async fn append_event(
        &self, trade_id: &ObjectId, event: &LifecycleEvent, user: &str,
    ) -> QLResult<ObjectId>;

    /// Replay all lifecycle events for a trade.
    async fn replay_events(
        &self, trade_id: &ObjectId,
    ) -> QLResult<Vec<LifecycleEvent>>;

    /// Query trades by predicate.
    async fn query_trades(
        &self, filter: &TradeFilter,
    ) -> QLResult<Vec<Trade>>;

    /// Save a market data snapshot.
    async fn save_snapshot(
        &self, snapshot: &MarketSnapshot,
    ) -> QLResult<ObjectId>;

    /// Load a market data snapshot for a date.
    async fn load_snapshot(
        &self, date: Date, snapshot_type: SnapshotType,
    ) -> QLResult<MarketSnapshot>;
}

/// Marker trait for types that can be persisted.
pub trait Persistable: Serialize + for<'de> Deserialize<'de> + Send + Sync {
    fn object_type() -> &'static str;
}
```

---

### 18.8 Decision Matrix

| Criteria | redb (embedded) | PostgreSQL + JSONB | SurrealDB | EventStoreDB | Kafka + Postgres |
|---|---|---|---|---|---|
| **Setup complexity** | ★☆☆☆☆ | ★★☆☆☆ | ★★☆☆☆ | ★★★☆☆ | ★★★★★ |
| **Single-user dev** | ✅ Excellent | ⚠️ Overkill | ✅ Embedded mode | ⚠️ Overkill | ❌ Way overkill |
| **Multi-user prod** | ❌ | ✅ Excellent | ✅ Good | ✅ Good | ✅ Excellent |
| **Bitemporal** | Manual | ✅ Native | ✅ Native | ✅ Natural | Manual |
| **Graph queries** | ❌ | ⚠️ CTEs | ✅ Native | ❌ | ❌ |
| **Event sourcing** | Manual | ✅ Table-based | ✅ | ✅ Native | ✅ Native |
| **Scalability** | Single node | Vertical | Horizontal | Horizontal | Horizontal |
| **Rust ecosystem** | ✅ Pure Rust | ✅ Mature (sqlx) | ✅ Rust-native | ⚠️ gRPC client | ✅ rdkafka |
| **Maturity** | Good | ✅ Battle-tested | ⚠️ Young | Good | ✅ Battle-tested |
| **Best for** | Phase 1, research | Phase 2, production | If graph queries are key | If event log is core | Enterprise scale |

**Recommended path:**

1. **Start with `redb`** — zero-infrastructure embedded store for development and backtesting.
2. **Graduate to PostgreSQL** — when you need multi-user access, production reliability, and SQL queries.
3. **Add DuckDB + Parquet** — when you need portfolio-level analytics and historical reporting.
4. **Consider Kafka** — only at enterprise scale with real-time streaming requirements.
