# ql-rust

A modern Rust reimplementation of the [QuantLib](https://www.quantlib.org/) quantitative finance library.

[![Build](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Tests](https://img.shields.io/badge/tests-963_passing-brightgreen)]()
[![Rust](https://img.shields.io/badge/rust-2021_edition-orange)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

## Overview

**ql-rust** brings the battle-tested financial models of QuantLib to Rust, offering:

- **Zero-cost abstractions** — Rust's type system and ownership model prevent common errors at compile time
- **High performance** — BS pricing + Greeks in ~200 ns, curve bootstrap in ~1 ms
- **Thread safety** — All core types are `Send + Sync`; Monte Carlo engine uses Rayon for parallel path generation
- **Modular architecture** — 15 focused crates; depend on only what you need
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
│  ql-persistence  │  Trade store, lifecycle events, redb      │
├──────────────────┼───────────────────────────────────────────┤
│  ql-methods      │  Monte Carlo, FD (1D + 2D Heston),       │
│                  │  lattice, FDM meshers & operators         │
├──────────────────┼───────────────────────────────────────────┤
│  ql-pricingengines│ Analytic BS, swap/bond/swaption pricing  │
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
| **ql-methods** | Numerical pricing methods | `mc_european`, `fd_black_scholes`, `fd_heston_solve` |
| **ql-persistence** | Trade storage & lifecycle | `Trade`, `EmbeddedStore`, `ObjectStore` |
| **ql-cli** | Command-line interface | Binary: `ql-cli` |
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

## Testing

```bash
# Run all 963 tests
cargo test --workspace

# Run integration tests only
cargo test -p ql-rust --tests

# Run property-based tests (proptest)
cargo test -p ql-rust --test test_property_based

# Run benchmarks
cargo bench -p ql-rust
```

### Test Categories

| Category | Count | Description |
|----------|-------|-------------|
| Unit tests | 845 | Per-crate functionality |
| Integration tests | 42 | Cross-crate pipelines (options, swaps, yield curve, American, multi-asset, short-rate, cashflows) |
| Property-based tests | 11 | Mathematical invariants via proptest (put-call parity, bounds, monotonicity) |
| Golden cross-validation | 65 | Calendar, yield curve, BS, American, spread, Nelson-Siegel, short-rate, FD, credit, LMM, CMS, advanced curves |

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
    ├── ql-pricingengines/  # Analytic pricing
    ├── ql-methods/         # MC, FD, lattice
    ├── ql-persistence/     # Trade store (redb)
    ├── ql-rust/            # Facade crate + integration tests + benchmarks
    └── ql-cli/             # CLI binary
```

## License

MIT
