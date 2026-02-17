//! Performance benchmarks for key ql-rust operations.
//!
//! Run with: `cargo bench -p ql-rust`

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

use ql_cashflows::fixed_leg;
use ql_indexes::IborIndex;
use ql_instruments::{FixedRateBond, OptionType, VanillaOption, SwapType, VanillaSwap};
use ql_methods::{binomial_crr, fd_black_scholes, mc_european};
use ql_pricingengines::{implied_volatility, price_bond, price_european, price_swap};
use ql_termstructures::{
    DepositRateHelper, FlatForward, PiecewiseYieldCurve, RateHelper, SwapRateHelper,
    YieldTermStructure,
};
use ql_time::{Date, DayCounter, Month, Schedule};

// ── Black-Scholes analytic pricing + Greeks ──────────────────────────────────

fn bench_bs_pricing(c: &mut Criterion) {
    let today = Date::from_ymd(2025, Month::January, 15);
    let call = VanillaOption::european_call(105.0, today + 365);

    c.bench_function("bs_european_call_price_and_greeks", |b| {
        b.iter(|| {
            price_european(
                black_box(&call),
                black_box(100.0),
                black_box(0.05),
                black_box(0.02),
                black_box(0.20),
                black_box(1.0),
            )
        })
    });
}

// ── Implied volatility (Newton solver) ───────────────────────────────────────

fn bench_implied_vol(c: &mut Criterion) {
    let today = Date::from_ymd(2025, Month::January, 15);
    let call = VanillaOption::european_call(105.0, today + 365);
    let target_price = price_european(&call, 100.0, 0.05, 0.02, 0.20, 1.0).npv;

    c.bench_function("implied_volatility_newton", |b| {
        b.iter(|| {
            implied_volatility(
                black_box(&call),
                black_box(target_price),
                black_box(100.0),
                black_box(0.05),
                black_box(0.02),
                black_box(1.0),
            )
        })
    });
}

// ── Yield curve bootstrapping ────────────────────────────────────────────────

fn bench_curve_bootstrap(c: &mut Criterion) {
    let today = Date::from_ymd(2025, Month::January, 15);
    let dc = DayCounter::Actual365Fixed;

    c.bench_function("yield_curve_bootstrap_6_helpers", |b| {
        b.iter(|| {
            let mut helpers: Vec<Box<dyn RateHelper>> = vec![
                Box::new(DepositRateHelper::new(0.045, today, today + 91, dc)),
                Box::new(DepositRateHelper::new(0.046, today, today + 182, dc)),
                Box::new(SwapRateHelper::new(0.048, today, vec![today + 365, today + 730], dc)),
                Box::new(SwapRateHelper::new(
                    0.050, today,
                    vec![today + 365, today + 730, today + 1095, today + 1461, today + 1826],
                    dc,
                )),
                Box::new(SwapRateHelper::new(
                    0.052, today,
                    (1..=7).map(|y| today + y * 365).collect(),
                    dc,
                )),
                Box::new(SwapRateHelper::new(
                    0.053, today,
                    (1..=10).map(|y| today + y * 365).collect(),
                    dc,
                )),
            ];
            PiecewiseYieldCurve::new(
                black_box(today),
                black_box(&mut helpers),
                black_box(dc),
                black_box(1e-12),
            )
        })
    });
}

// ── Discount factor lookup ───────────────────────────────────────────────────

fn bench_discount_factor(c: &mut Criterion) {
    let today = Date::from_ymd(2025, Month::January, 15);
    let dc = DayCounter::Actual365Fixed;
    let curve = FlatForward::new(today, 0.05, dc);

    c.bench_function("flat_forward_discount_t", |b| {
        b.iter(|| curve.discount_t(black_box(2.5)))
    });
}

// ── Monte Carlo European option ──────────────────────────────────────────────

fn bench_mc_european(c: &mut Criterion) {
    let mut group = c.benchmark_group("mc_european");
    for &paths in &[10_000u64, 100_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}k_paths", paths / 1000)),
            &paths,
            |b, &paths| {
                b.iter(|| {
                    mc_european(
                        black_box(100.0),
                        black_box(105.0),
                        black_box(0.05),
                        black_box(0.0),
                        black_box(0.20),
                        black_box(1.0),
                        black_box(OptionType::Call),
                        black_box(paths as usize),
                        black_box(true),
                        black_box(42),
                    )
                })
            },
        );
    }
    group.finish();
}

// ── Finite Differences American option ───────────────────────────────────────

fn bench_fd_american(c: &mut Criterion) {
    c.bench_function("fd_american_put_200x200", |b| {
        b.iter(|| {
            fd_black_scholes(
                black_box(100.0),
                black_box(110.0),
                black_box(0.05),
                black_box(0.0),
                black_box(0.30),
                black_box(1.0),
                black_box(false),
                black_box(true),
                black_box(200),
                black_box(200),
            )
        })
    });
}

// ── Binomial CRR ─────────────────────────────────────────────────────────────

fn bench_binomial_crr(c: &mut Criterion) {
    let mut group = c.benchmark_group("binomial_crr");
    for &steps in &[100u64, 500, 1000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_steps", steps)),
            &steps,
            |b, &steps| {
                b.iter(|| {
                    binomial_crr(
                        black_box(100.0),
                        black_box(105.0),
                        black_box(0.05),
                        black_box(0.0),
                        black_box(0.20),
                        black_box(1.0),
                        black_box(true),
                        black_box(false),
                        black_box(steps as usize),
                    )
                })
            },
        );
    }
    group.finish();
}

// ── Bond pricing ─────────────────────────────────────────────────────────────

fn bench_bond_pricing(c: &mut Criterion) {
    let today = Date::from_ymd(2025, Month::January, 15);
    let dc = DayCounter::Actual365Fixed;
    let schedule = Schedule::from_dates(vec![
        Date::from_ymd(2025, Month::January, 15),
        Date::from_ymd(2025, Month::July, 15),
        Date::from_ymd(2026, Month::January, 15),
        Date::from_ymd(2026, Month::July, 15),
        Date::from_ymd(2027, Month::January, 15),
    ]);
    let bond = FixedRateBond::new(100.0, 2, &schedule, 0.05, dc);
    let curve = FlatForward::new(today, 0.05, dc);

    c.bench_function("fixed_rate_bond_pricing", |b| {
        b.iter(|| price_bond(black_box(&bond), black_box(&curve), black_box(today)))
    });
}

// ── Swap pricing ─────────────────────────────────────────────────────────────

fn bench_swap_pricing(c: &mut Criterion) {
    let today = Date::from_ymd(2025, Month::January, 15);
    let dc = DayCounter::Actual365Fixed;
    let curve = FlatForward::new(today, 0.05, dc);

    let schedule = Schedule::from_dates(vec![
        Date::from_ymd(2025, Month::January, 15),
        Date::from_ymd(2025, Month::July, 15),
        Date::from_ymd(2026, Month::January, 15),
        Date::from_ymd(2026, Month::July, 15),
        Date::from_ymd(2027, Month::January, 15),
    ]);

    let notionals = [1_000_000.0; 4];
    let index = IborIndex::euribor_6m();
    let fixed = fixed_leg(&schedule, &notionals, &[0.05; 4], dc);
    let floating = ql_cashflows::ibor_leg(&schedule, &notionals, &index, &[0.0; 4], dc);
    let swap = VanillaSwap::new(SwapType::Payer, 1_000_000.0, fixed, floating, 0.05, 0.0);

    c.bench_function("vanilla_swap_pricing", |b| {
        b.iter(|| price_swap(black_box(&swap), black_box(&curve), black_box(today)))
    });
}

// ── Date arithmetic ──────────────────────────────────────────────────────────

fn bench_date_arithmetic(c: &mut Criterion) {
    let d = Date::from_ymd(2025, Month::January, 15);
    let dc = DayCounter::Actual365Fixed;

    c.bench_function("date_add_days", |b| {
        b.iter(|| black_box(d) + black_box(365))
    });

    c.bench_function("day_counter_year_fraction", |b| {
        let d2 = d + 730;
        b.iter(|| dc.year_fraction(black_box(d), black_box(d2)))
    });
}

criterion_group!(
    benches,
    bench_bs_pricing,
    bench_implied_vol,
    bench_curve_bootstrap,
    bench_discount_factor,
    bench_mc_european,
    bench_fd_american,
    bench_binomial_crr,
    bench_bond_pricing,
    bench_swap_pricing,
    bench_date_arithmetic,
);
criterion_main!(benches);
