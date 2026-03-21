# ql-rust

A modern Rust reimplementation of the [QuantLib](https://www.quantlib.org/) quantitative finance library.

[![Build](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Tests](https://img.shields.io/badge/tests-3028_passing-brightgreen)]()
[![Rust](https://img.shields.io/badge/rust-2021_edition-orange)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

## Overview

**ql-rust** brings the battle-tested financial models of QuantLib to Rust, offering:

- **Zero-cost abstractions** — Rust's type system and ownership model prevent common errors at compile time
- **High performance** — BS pricing + Greeks in ~200 ns, curve bootstrap in ~1 ms
- **Thread safety** — All core types are `Send + Sync`; Monte Carlo engine uses Rayon for parallel path generation
- **Modular architecture** — 17 focused crates; depend on only what you need
- **Automatic differentiation** — Forward-mode (`Dual`, `DualVec<N>`) and reverse-mode (`AReal`) AD via the `Number` trait; 100+ generic engines support exact Greeks without finite-difference bumping
- **Embedded persistence** — Trade booking, lifecycle events, and versioning via redb (no external DB required)

## Quick Start

```toml
# Cargo.toml
[dependencies]
ql-rust = { path = "crates/ql-rust" }
```

### Price a European Call Option

```rust
use ql_rust::*;

let today = Date::from_ymd(2025, Month::January, 15);
let call = VanillaOption::european_call(105.0, today + 365);

let result = price_european(&call, 100.0, 0.05, 0.02, 0.20, 1.0);

println!("NPV:   {:.4}", result.npv);
println!("Delta: {:.4}", result.delta);
println!("Gamma: {:.4}", result.gamma);
println!("Vega:  {:.4}", result.vega);
println!("Theta: {:.4}", result.theta);
println!("Rho:   {:.4}", result.rho);
```

### Bootstrap a Yield Curve

```rust
use ql_rust::*;

let today = Date::from_ymd(2025, Month::January, 15);
let dc = DayCounter::Actual365Fixed;

let mut helpers: Vec<Box<dyn RateHelper>> = vec![
    Box::new(DepositRateHelper::new(0.045, today, today + 91, dc)),
    Box::new(DepositRateHelper::new(0.046, today, today + 182, dc)),
    Box::new(SwapRateHelper::new(0.050, today,
        (1..=5).map(|y| today + y * 365).collect(), dc)),
];

let curve = PiecewiseYieldCurve::new(today, &mut helpers, dc, 1e-12)
    .expect("Bootstrap failed");

println!("2Y discount factor: {:.6}", curve.discount_t(2.0));
```

### Price a Vanilla Swap

```rust
use ql_rust::*;
use ql_cashflows::{fixed_leg, ibor_leg};

let today = Date::from_ymd(2025, Month::January, 15);
let dc = DayCounter::Actual365Fixed;
let curve = FlatForward::new(today, 0.05, dc);
let schedule = Schedule::from_dates(vec![
    Date::from_ymd(2025, Month::January, 15),
    Date::from_ymd(2025, Month::July, 15),
    Date::from_ymd(2026, Month::January, 15),
]);

let index = ql_indexes::IborIndex::euribor_6m();
let notionals = [1_000_000.0; 2];
let fixed = fixed_leg(&schedule, &notionals, &[0.05; 2], dc);
let floating = ibor_leg(&schedule, &notionals, &index, &[0.0; 2], dc);
let swap = VanillaSwap::new(SwapType::Payer, 1_000_000.0, fixed, floating, 0.05, 0.0);

let result = price_swap(&swap, &curve, today);
println!("Swap NPV: {:.2}", result.npv);
println!("Fair rate: {:.4}", result.fair_rate);
```

### Monte Carlo & Finite Differences

```rust
use ql_rust::*;

// Monte Carlo European call (500K paths, antithetic variates)
let mc = mc_european(100.0, 105.0, 0.05, 0.0, 0.20, 1.0,
    OptionType::Call, 500_000, true, 42);
println!("MC price: {:.4} ± {:.4}", mc.npv, mc.std_error);

// Finite differences American put (200×200 grid)
let fd = fd_black_scholes(100.0, 110.0, 0.05, 0.0, 0.30, 1.0,
    false, true, 200, 200);
println!("FD American put: {:.4}", fd.npv);

// Binomial CRR (500 steps)
let crr = binomial_crr(100.0, 105.0, 0.05, 0.0, 0.20, 1.0,
    true, false, 500);
println!("CRR European call: {:.4}", crr.npv);
```

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        ql-rust (facade)                      │
│               Re-exports all public types & functions        │
├──────────────────────────────────────────────────────────────┤
│  ql-cli          │  Command-line interface (price, curve,    │
│                  │  trade, list, risk)                       │
├──────────────────┼───────────────────────────────────────────┤
│  ql-python       │  Python bindings via PyO3 (maturin)      │
├──────────────────┼───────────────────────────────────────────┤
│  ql-persistence  │  Trade store, lifecycle events, redb      │
├──────────────────┼───────────────────────────────────────────┤
│  ql-methods      │  Monte Carlo, FD (1D + 2D Heston),       │
│                  │  lattice, FDM meshers & operators         │
├──────────────────┼───────────────────────────────────────────┤
│  ql-pricingengines│ Analytic BS, swap/bond/swaption pricing  │
│                  │  + 100 generic engines (T: Number)        │
├──────────────────┼───────────────────────────────────────────┤
│  ql-aad          │  Automatic differentiation: Dual, DualVec,│
│                  │  AReal (tape), Number trait               │
├──────────────────┼───────────────────────────────────────────┤
│  ql-models       │  Heston, Hull-White, Vasicek, CIR, G2,   │
│                  │  Bates, Black-Karasinski, LMM             │
├──────────────────┼───────────────────────────────────────────┤
│  ql-processes    │  GBM, Heston, Hull-White, Bates, CIR      │
├──────────────────┼───────────────────────────────────────────┤
│  ql-instruments  │  Options, swaps, bonds, swaptions,        │
│                  │  caps/floors, CDS, exotics                │
├──────────────────┼───────────────────────────────────────────┤
│  ql-cashflows    │  Fixed/floating coupons, CMS, digital,    │
│                  │  range-accrual, sub-period, analytics     │
├──────────────────┼───────────────────────────────────────────┤
│  ql-termstructures│ Yield curves, vol surfaces, inflation,   │
│                  │  credit, local vol, SABR, SVI, ZABR,     │
│                  │  Nelson-Siegel, Smith-Wilson              │
├──────────────────┼───────────────────────────────────────────┤
│  ql-indexes      │  IBOR indices, interest rate compounding  │
├──────────────────┼───────────────────────────────────────────┤
│  ql-currencies   │  30+ ISO 4217 currencies                  │
├──────────────────┼───────────────────────────────────────────┤
│  ql-math         │  Interpolation, root-finding, integration,│
│                  │  optimization, copulas, FFT, quasi-random │
├──────────────────┼───────────────────────────────────────────┤
│  ql-time         │  Dates, day counters, calendars,          │
│                  │  schedules, business day conventions       │
├──────────────────┼───────────────────────────────────────────┤
│  ql-core         │  Error types, quote abstraction            │
└──────────────────┴───────────────────────────────────────────┘
```

## Crate Map

| Crate | Description | Key Types |
|-------|-------------|-----------|
| **ql-core** | Error handling, market quotes | `QLError`, `QLResult`, `SimpleQuote` |
| **ql-time** | Date arithmetic, calendars, schedules | `Date`, `Calendar`, `DayCounter`, `Schedule` |
| **ql-math** | Numerical methods | `LinearInterpolation`, `CubicSpline`, `Brent` |
| **ql-currencies** | ISO 4217 currency definitions | `Currency`, `USD`, `EUR`, `GBP` |
| **ql-indexes** | Interest rate indices | `IborIndex`, `InterestRate`, `Compounding` |
| **ql-termstructures** | Term structure models | `FlatForward`, `PiecewiseYieldCurve`, `NelsonSiegelFitting`, `SviSmileSection` |
| **ql-cashflows** | Cash flow generation & analytics | `CashFlow`, `Leg`, `CmsCoupon`, `DigitalCoupon`, `convexity`, `dv01` |
| **ql-instruments** | Financial instrument types | `VanillaOption`, `VanillaSwap`, `FixedRateBond` |
| **ql-processes** | Stochastic processes | `GeneralizedBlackScholesProcess`, `HestonProcess`, `HullWhiteProcess` |
| **ql-models** | Calibrated models | `HestonModel`, `HullWhiteModel`, `VasicekModel`, `CIRModel`, `G2Model` |
| **ql-pricingengines** | Analytic pricing engines | `price_european`, `price_swap`, `barone_adesi_whaley`, `mc_basket` |
| **ql-aad** | Automatic differentiation | `Dual`, `DualVec<N>`, `AReal`, `Number` trait |
| **ql-methods** | Numerical pricing methods | `mc_european`, `fd_black_scholes`, `fd_heston_solve` |
| **ql-persistence** | Trade storage & lifecycle | `Trade`, `EmbeddedStore`, `ObjectStore` |
| **ql-cli** | Command-line interface | Binary: `ql-cli` |
| **ql-python** | Python bindings (PyO3) | `Date`, `FlatForward`, `price_european_bs`, `mc_european_py` |
| **ql-rust** | Façade re-exporting all crates | — |

## Supported Instruments

| Category | Instruments |
|----------|-------------|
| **Equity** | European/American options, barrier options, lookback options, Asian options, compound options, variance swaps, basket options, spread options, exchange options |
| **Rates** | Vanilla swaps, swaptions (European & Bermudan), caps/floors, fixed-rate bonds, callable bonds |
| **Credit** | Credit default swaps, CDS options, CDO tranches (LHP), N-th to default baskets |
| **Hybrid** | Convertible bonds |
| **Multi-Asset** | Stulz max/min, Kirk spread, Margrabe exchange, MC basket (N-asset) |

## Pricing Engines

| Engine | Method | Instruments |
|--------|--------|-------------|
| Analytic Black-Scholes | Closed-form | European options |
| BAW / BJS / QD+ | Analytic approximation | American options |
| Longstaff-Schwartz | Least-squares MC | American options |
| Heston semi-analytic | Fourier integration | European options (stochastic vol) |
| Bates / Merton JD | Jump-diffusion | European options |
| Monte Carlo | Simulation (parallel) | European, barrier, Asian, Heston, Bates, basket |
| Finite Differences 1D | Crank-Nicolson | European & American options |
| Finite Differences 2D | Douglas ADI | Heston PDE |
| Binomial CRR | Lattice | European & American options |
| Trinomial tree | Short-rate tree | Bonds, swaptions, caps/floors |
| Hull-White analytic | Closed-form | Bond options, caplets, swaptions |
| Analytic swap/bond | Discounted cash flows | Swaps, bonds |
| Black / Bachelier | Closed-form | Swaptions, caps/floors |
| Gaussian copula LHP | Semi-analytic | CDO tranche pricing |
| Black CDS option | Closed-form | CDS options |

## CLI Usage

```bash
# Build the CLI
cargo build -p ql-cli --release

# Price a European call
ql-cli price --instrument call --spot 100 --strike 105 --vol 0.20 \
  --rate 0.05 --div 0.02 --expiry 1.0

# Bootstrap a yield curve
ql-cli curve --deposits 0.045:91,0.046:182 --swaps 0.050:5

# Book a trade
ql-cli trade --type option --counterparty "ACME" --book equity \
  --notional 1000000 --direction buy

# List trades
ql-cli list --book equity
```

## Serialization

All ~190 domain types across 13 crates implement serde `Serialize` + `Deserialize`,
enabling JSON (and any serde backend) round-trips for instruments, term structures,
pricing results, schedules, processes, and models.

```rust
use ql_rust::*;

// Serialize a vanilla option to JSON
let option = VanillaOption::european_call(100.0, Date::from_ymd(2026, Month::June, 15));
let json = serde_json::to_string_pretty(&option)?;

// Deserialize back
let restored: VanillaOption = serde_json::from_str(&json)?;
assert_eq!(restored.strike(), option.strike());

// Works for pricing results too
let greeks = price_european(&option, 100.0, 0.05, 0.0, 0.20, 1.0);
let json = serde_json::to_string(&greeks)?;  // {"npv":10.45,"delta":0.637,...}

// And term structures
let curve = FlatForward::new(Date::from_ymd(2025, Month::January, 15), 0.05, DayCounter::Actual365Fixed);
let json = serde_json::to_string(&curve)?;
```

See [`examples/serde_round_trip.rs`](crates/ql-rust/examples/serde_round_trip.rs) for a
complete runnable demo covering VanillaOption, BarrierOption, FlatForward,
NelsonSiegelFitting, Schedule, AnalyticEuropeanResults, CreditDefaultSwap, and Date.

## Testing

```bash
# Run all 3028 tests
cargo test --workspace

# Run integration tests only
cargo test -p ql-rust --tests

# Run AD integration tests (26 first-order + 7 higher-order Greeks)
cargo test -p ql-rust --test test_ad_generic_engines

# Run property-based tests (proptest)
cargo test -p ql-rust --test test_property_based

# Run benchmarks (including AD performance comparisons)
cargo bench -p ql-rust
```

### Test Categories

| Category | Count | Description |
|----------|-------|-------------|
| Unit tests | ~2700 | Per-crate functionality |
| Integration tests | 80+ | Cross-crate pipelines (options, swaps, yield curve, American, multi-asset, short-rate, cashflows, E2E workflows) |
| AD integration tests | 33 | AD types (Dual, DualVec, AReal) through generic engines + higher-order Greeks (gamma, vanna, volga, charm) |
| Property-based tests | 11 | Mathematical invariants via proptest (put-call parity, bounds, monotonicity) |
| Doc-tests | 50+ | Verified examples on public APIs |
| Calendar validation | 124 | Holiday verification (TARGET, NYSE, UK) against known dates |
| Golden cross-validation | 36 | BS, American, Nelson-Siegel, short-rate, FD, credit, LMM, CMS, advanced curves |

### Benchmarks

| Benchmark | Description |
|-----------|-------------|
| `bs_european_call_price_and_greeks` | Analytic BS pricing + all Greeks |
| `implied_volatility_newton` | Newton's method implied vol solver |
| `yield_curve_bootstrap_6_helpers` | Piecewise yield curve (6 instruments) |
| `mc_european/10k_paths` | Monte Carlo with 10K paths |
| `mc_european/100k_paths` | Monte Carlo with 100K paths |
| `fd_american_put_200x200` | Crank-Nicolson FD (200×200 grid) |
| `binomial_crr/{100,500,1000}_steps` | CRR lattice at various step counts |
| `fixed_rate_bond_pricing` | Bond NPV + clean/dirty prices |
| `vanilla_swap_pricing` | Swap NPV + fair rate |
| `heston_analytic_price` | Heston semi-analytic pricing (Fourier) |
| `heston_calibration_5_helpers` | 5-point Heston model calibration |
| `calendar_advance_30bd` | Calendar.advance 30 business days |
| `interpolation_linear_lookup` | Linear interpolation point lookup |
| `interpolation_cubic_spline_lookup` | Cubic spline interpolation point lookup |
| `date_add_days` | Date + integer days arithmetic |
| `day_counter_year_fraction` | Year fraction calculation |
| `american_baw_put` | Barone-Adesi-Whaley American put |
| `american_bjerksund_stensland_put` | Bjerksund-Stensland American put |
| `american_qd_plus_put` | QD+ high-precision American put |
| `nelson_siegel_fit_11_points` | Nelson-Siegel 4-param curve fitting |
| `vasicek_bond_5y` | Vasicek analytic bond price (5Y) |
| `g2_swaption_10y` | G2 two-factor swaption pricing |
| `fft_8192` | In-place FFT on 8192-point complex array |
| `cholesky_50x50` | Cholesky decomposition (50×50 matrix) |
| `cms_caplet_pricing` | CMS caplet via linear TSR model |
| `lmm_cap_10k_paths` | LMM cap pricing (10K MC paths) |
| `gaussian_copula_cdo_tranche` | CDO equity tranche expected loss (LHP) |
| `cds_option_black` | CDS option via Black's formula |

### AD Performance Benchmarks

These benchmarks compare f64 (baseline) against forward-mode `Dual` (single Greek),
forward-mode `DualVec<5>` (5 Greeks in one pass), and reverse-mode `AReal` (all partials)
for the same generic engine call.

| Benchmark | f64 | Dual | DualVec&lt;5&gt; | AReal |
|-----------|----:|-----:|----------:|------:|
| BS European | ~70 ns | ~120 ns (1.7×) | ~260 ns (3.7×) | ~2.5 µs (35×) |
| BAW American | ~1.4 µs | ~2.1 µs (1.5×) | ~3.4 µs (2.4×) | ~37 µs (26×) |
| Merton JD | ~1.0 µs | ~1.8 µs (1.8×) | ~3.4 µs (3.3×) | ~33 µs (32×) |
| Chooser | ~70 ns | ~118 ns (1.7×) | — | ~3.5 µs (50×) |

Run: `cargo bench -p ql-rust -- ad_`

## Automatic Differentiation (AAD)

The `ql-aad` crate provides exact, tape-free algorithmic differentiation through the
`Number` trait. All 100+ generic engines in `ql-pricingengines::generic` and
`ql-methods::generic` accept any `T: Number`, enabling:

- **`Dual`** — forward-mode, single derivative seed (one Greek per pass)
- **`DualVec<N>`** — forward-mode, N derivative seeds (N Greeks in one pass)
- **`AReal`** — reverse-mode, tape-based (all partials in one backward sweep)
- **`f64`** — zero-overhead when no derivatives are needed

```rust
use ql_aad::{Dual, DualVec};
use ql_pricingengines::generic::bs_european_generic;

// Forward-mode: exact delta in a single pass
let spot = Dual::new(100.0, 1.0);  // seed = 1.0 → ∂/∂spot
let res = bs_european_generic(
    spot,
    Dual::constant(100.0),  // strike
    Dual::constant(0.05),   // r
    Dual::constant(0.0),    // q
    Dual::constant(0.20),   // vol
    Dual::constant(1.0),    // T
    true,                   // is_call
);
let price = res.npv.val;   // 10.45...
let delta = res.npv.dot;   // 0.637...  (exact, no bumping)

// Multi-seed: all 5 Greeks in one pass
type D5 = DualVec<5>;
let res = bs_european_generic(
    D5::variable(100.0, 0),  // ∂/∂spot
    D5::constant(100.0),
    D5::variable(0.05, 1),   // ∂/∂r
    D5::variable(0.0, 2),    // ∂/∂q
    D5::variable(0.20, 3),   // ∂/∂vol
    D5::variable(1.0, 4),    // ∂/∂T
    true,
);
// res.npv.dot == [delta, rho, ∂V/∂q, vega, ∂V/∂T]
```

Higher-order Greeks (gamma, vanna, volga) are computed via FD-on-AD: a central finite
difference on the exact first-order AD derivative. See `tests/test_ad_generic_engines.rs`
for examples.

## Building

```bash
# Debug build
cargo build --workspace

# Release build
cargo build --workspace --release

# Generate documentation
cargo doc --workspace --no-deps --open

# Lint
cargo clippy --workspace -- -D warnings
```

## Project Structure

```
ql-rust/
├── Cargo.toml              # Workspace manifest
├── README.md
├── CHANGELOG.md
├── project.md              # Implementation plan
└── crates/
    ├── ql-core/            # Error types, quotes
    ├── ql-time/            # Dates, calendars, schedules
    ├── ql-math/            # Interpolation, solvers, optimization
    ├── ql-currencies/      # ISO 4217 currencies
    ├── ql-indexes/         # IBOR indices, interest rates
    ├── ql-termstructures/  # Yield curves, vol surfaces
    ├── ql-cashflows/       # Cash flow generation
    ├── ql-instruments/     # Financial instruments
    ├── ql-processes/       # Stochastic processes
    ├── ql-models/          # Calibrated models
    ├── ql-pricingengines/  # Analytic pricing (+ 100 generic engines)
    ├── ql-aad/             # Automatic differentiation (Dual, DualVec, AReal)
    ├── ql-methods/         # MC, FD, lattice
    ├── ql-persistence/     # Trade store (redb)
    ├── ql-rust/            # Facade crate + integration tests + benchmarks
    ├── ql-cli/             # CLI binary
    └── ql-python/          # Python bindings (PyO3 + maturin)
```

## License

MIT OR Apache-2.0
