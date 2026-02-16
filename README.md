# ql-rust

A comprehensive quantitative finance library in Rust — a modern re-implementation of QuantLib.

## Crate Map

| Crate | Description |
|---|---|
| `ql-core` | Foundational types, errors, observer pattern, handles, settings |
| `ql-time` | Date, Calendar, DayCounter, Schedule, Period |
| `ql-math` | Interpolation, root-finding, optimization, distributions, RNG |
| `ql-currencies` | Currency, Money, ExchangeRate |
| `ql-indexes` | Index trait, IborIndex, OvernightIndex, SwapIndex |
| `ql-termstructures` | Term structure traits, yield curves, bootstrapping |
| `ql-cashflows` | CashFlow, Coupon, Leg, coupon pricers |
| `ql-instruments` | Instrument trait, options, swaps, bonds |
| `ql-processes` | Stochastic processes (Black-Scholes, Heston, Hull-White) |
| `ql-models` | Calibrated models (Heston, Hull-White, Vasicek) |
| `ql-pricingengines` | Pricing engines (analytic, MC, FD, lattice) |
| `ql-methods` | Numerical methods (Monte Carlo, finite differences, lattices) |
| `ql-persistence` | SecDB/Beacon-style persistence (redb, PostgreSQL, DuckDB) |

## Build

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo doc --workspace --no-deps --open
```

## License

Dual-licensed under Apache 2.0 and MIT.
