//! Performance benchmarks for key ql-rust operations.
//!
//! Run with: `cargo bench -p ql-rust`

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

use ql_cashflows::fixed_leg;
use ql_indexes::IborIndex;
use ql_instruments::{FixedRateBond, OptionType, VanillaOption, SwapType, VanillaSwap};
use ql_math::interpolation::{CubicSplineInterpolation, Interpolation, LinearInterpolation};
use ql_math::optimization::EndCriteria;
use ql_methods::{binomial_crr, fd_black_scholes, fd_heston_solve, mc_asian, mc_barrier, mc_bates, mc_european, mc_heston};
use ql_models::{CalibrationHelper, HestonModel, calibrate};
use ql_pricingengines::{
    heston_price, implied_volatility, price_bond, price_european, price_swap,
    barone_adesi_whaley, bjerksund_stensland, qd_plus_american,
    mc_american_longstaff_schwartz, LSMBasis,
    kirk_spread_call, mc_basket, BasketType,
    hw_jamshidian_swaption,
    merton_jump_diffusion, bates_price_flat,
};
use ql_termstructures::{
    DepositRateHelper, FlatForward, NelsonSiegelFitting, PiecewiseYieldCurve,
    RateHelper, SwapRateHelper, YieldTermStructure,
    sabr_volatility, svi_calibrate,
};
use ql_time::{
    BusinessDayConvention, Calendar, Date, DayCounter, Month, Period, Schedule,
};

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

// ── Heston analytic pricing ──────────────────────────────────────────────────

fn bench_heston_analytic(c: &mut Criterion) {
    let model = HestonModel::new(100.0, 0.05, 0.02, 0.04, 2.0, 0.04, 0.3, -0.5);

    c.bench_function("heston_analytic_price", |b| {
        b.iter(|| {
            heston_price(
                black_box(&model),
                black_box(105.0),
                black_box(1.0),
                black_box(true),
            )
        })
    });
}

// ── Heston calibration ──────────────────────────────────────────────────────

/// CalibrationHelper for benchmarking Heston calibration.
struct BenchHestonHelper {
    spot: f64,
    rate: f64,
    dividend: f64,
    strike: f64,
    time_to_expiry: f64,
    market_price: f64,
}

impl CalibrationHelper for BenchHestonHelper {
    fn market_value(&self) -> f64 {
        self.market_price
    }
    fn model_value_with_params(&self, params: &[f64]) -> f64 {
        let model = HestonModel::new(
            self.spot, self.rate, self.dividend,
            params[0], params[1], params[2], params[3], params[4],
        );
        heston_price(&model, self.strike, self.time_to_expiry, true).npv
    }
}

fn bench_heston_calibration(c: &mut Criterion) {
    let true_model = HestonModel::new(100.0, 0.05, 0.02, 0.04, 2.0, 0.04, 0.3, -0.5);
    let strikes = [90.0, 95.0, 100.0, 105.0, 110.0];

    let helpers: Vec<Box<dyn CalibrationHelper>> = strikes
        .iter()
        .map(|&k| {
            let mkt_price = heston_price(&true_model, k, 1.0, true).npv;
            Box::new(BenchHestonHelper {
                spot: 100.0,
                rate: 0.05,
                dividend: 0.02,
                strike: k,
                time_to_expiry: 1.0,
                market_price: mkt_price,
            }) as Box<dyn CalibrationHelper>
        })
        .collect();

    let criteria = EndCriteria {
        max_iterations: 500,
        max_stationary_iterations: 50,
        ..EndCriteria::default()
    };

    c.bench_function("heston_calibration_5_helpers", |b| {
        b.iter(|| {
            let mut model = HestonModel::new(
                100.0, 0.05, 0.02, 0.06, 1.0, 0.06, 0.5, -0.3,
            );
            calibrate(black_box(&mut model), black_box(&helpers), black_box(&criteria))
        })
    });
}

// ── Calendar advance ─────────────────────────────────────────────────────────

fn bench_calendar_advance(c: &mut Criterion) {
    let cal = Calendar::UnitedStates(ql_time::calendar::USMarket::NYSE);
    let d = Date::from_ymd(2025, Month::January, 15);
    let period = Period::months(1);

    c.bench_function("calendar_advance_30bd", |b| {
        b.iter(|| {
            cal.advance(
                black_box(d),
                black_box(period),
                black_box(BusinessDayConvention::ModifiedFollowing),
                black_box(false),
            )
        })
    });
}

// ── Interpolation ────────────────────────────────────────────────────────────

fn bench_interpolation(c: &mut Criterion) {
    let xs: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| (x * 0.5).sin()).collect();

    let linear = LinearInterpolation::new(xs.clone(), ys.clone()).unwrap();
    let cubic = CubicSplineInterpolation::new(xs, ys).unwrap();

    c.bench_function("interpolation_linear_lookup", |b| {
        b.iter(|| linear.value(black_box(4.73)))
    });

    c.bench_function("interpolation_cubic_spline_lookup", |b| {
        b.iter(|| cubic.value(black_box(4.73)))
    });
}

// ── American approximation methods ───────────────────────────────────────────

fn bench_american_approx(c: &mut Criterion) {
    c.bench_function("american_baw_put", |b| {
        b.iter(|| {
            barone_adesi_whaley(
                black_box(100.0), black_box(110.0),
                black_box(0.05), black_box(0.02),
                black_box(0.30), black_box(1.0),
                black_box(false),
            )
        })
    });

    c.bench_function("american_bjerksund_stensland_put", |b| {
        b.iter(|| {
            bjerksund_stensland(
                black_box(100.0), black_box(110.0),
                black_box(0.05), black_box(0.02),
                black_box(0.30), black_box(1.0),
                black_box(false),
            )
        })
    });

    c.bench_function("american_qd_plus_put", |b| {
        b.iter(|| {
            qd_plus_american(
                black_box(100.0), black_box(110.0),
                black_box(0.05), black_box(0.02),
                black_box(0.30), black_box(1.0),
                black_box(false),
            )
        })
    });
}

// ── Nelson-Siegel curve fitting ──────────────────────────────────────────────

fn bench_nelson_siegel_fit(c: &mut Criterion) {
    let maturities = vec![0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0];
    let yields = vec![
        0.030, 0.032, 0.035, 0.040, 0.043, 0.048, 0.050, 0.052, 0.053, 0.054, 0.055,
    ];

    c.bench_function("nelson_siegel_fit_11_points", |b| {
        b.iter(|| {
            NelsonSiegelFitting::fit(
                black_box(&maturities),
                black_box(&yields),
            )
        })
    });
}

// ── MC Barrier (down-and-out call) ───────────────────────────────────────────

fn bench_mc_barrier(c: &mut Criterion) {
    c.bench_function("mc_barrier_down_and_out_50k", |b| {
        b.iter(|| {
            mc_barrier(
                black_box(100.0), black_box(100.0), black_box(80.0), black_box(0.0),
                black_box(0.05), black_box(0.0), black_box(0.2), black_box(1.0),
                black_box(OptionType::Call), black_box(false), black_box(false),
                black_box(50_000), black_box(252), black_box(42),
            )
        })
    });
}

// ── MC Asian (arithmetic average call) ───────────────────────────────────────

fn bench_mc_asian(c: &mut Criterion) {
    c.bench_function("mc_asian_arithmetic_50k", |b| {
        b.iter(|| {
            mc_asian(
                black_box(100.0), black_box(100.0),
                black_box(0.05), black_box(0.0), black_box(0.2), black_box(1.0),
                black_box(OptionType::Call), black_box(true),
                black_box(50_000), black_box(252), black_box(42),
            )
        })
    });
}

// ── MC Heston European ──────────────────────────────────────────────────────

fn bench_mc_heston(c: &mut Criterion) {
    c.bench_function("mc_heston_european_50k", |b| {
        b.iter(|| {
            mc_heston(
                black_box(100.0), black_box(100.0),
                black_box(0.05), black_box(0.0),
                black_box(0.04), black_box(1.5), black_box(0.04),
                black_box(0.3), black_box(-0.7), black_box(1.0),
                black_box(OptionType::Call),
                black_box(50_000), black_box(252), black_box(42),
            )
        })
    });
}

// ── MC Bates (Heston + jumps) ───────────────────────────────────────────────

fn bench_mc_bates(c: &mut Criterion) {
    c.bench_function("mc_bates_european_50k", |b| {
        b.iter(|| {
            mc_bates(
                black_box(100.0), black_box(100.0),
                black_box(0.05), black_box(0.0),
                black_box(0.04), black_box(1.5), black_box(0.04),
                black_box(0.3), black_box(-0.7),
                black_box(0.5), black_box(-0.1), black_box(0.15),
                black_box(1.0), black_box(OptionType::Call),
                black_box(50_000), black_box(252), black_box(42),
            )
        })
    });
}

// ── FD Heston 2D Douglas ADI ────────────────────────────────────────────────

fn bench_fd_heston(c: &mut Criterion) {
    c.bench_function("fd_heston_european_50x30x50", |b| {
        b.iter(|| {
            fd_heston_solve(
                black_box(100.0), black_box(100.0),
                black_box(0.05), black_box(0.0),
                black_box(0.04), black_box(1.5), black_box(0.04),
                black_box(0.3), black_box(-0.7), black_box(1.0),
                black_box(true), black_box(false),
                black_box(50), black_box(30), black_box(50),
            )
        })
    });
}

// ── Longstaff-Schwartz MC American ──────────────────────────────────────────

fn bench_lsm_american(c: &mut Criterion) {
    c.bench_function("lsm_american_put_50k", |b| {
        b.iter(|| {
            mc_american_longstaff_schwartz(
                black_box(100.0), black_box(110.0),
                black_box(0.05), black_box(0.0),
                black_box(0.30), black_box(1.0),
                black_box(OptionType::Put),
                black_box(50_000), black_box(50),
                black_box(3), black_box(LSMBasis::Laguerre),
                black_box(42),
            )
        })
    });
}

// ── Kirk spread option ──────────────────────────────────────────────────────

fn bench_kirk_spread(c: &mut Criterion) {
    c.bench_function("kirk_spread_call", |b| {
        b.iter(|| {
            kirk_spread_call(
                black_box(100.0), black_box(96.0), black_box(3.0),
                black_box(0.05), black_box(0.02), black_box(0.01),
                black_box(0.20), black_box(0.25), black_box(0.5),
                black_box(1.0),
            )
        })
    });
}

// ── MC Basket (3-asset correlated) ──────────────────────────────────────────

fn bench_mc_basket(c: &mut Criterion) {
    let spots = [100.0, 100.0, 100.0];
    let weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
    let divs = [0.02, 0.01, 0.03];
    let vols = [0.20, 0.25, 0.30];
    // 3×3 correlation matrix (row-major)
    #[rustfmt::skip]
    let corr = [
        1.0, 0.5, 0.3,
        0.5, 1.0, 0.4,
        0.3, 0.4, 1.0,
    ];

    c.bench_function("mc_basket_3_asset_50k", |b| {
        b.iter(|| {
            mc_basket(
                black_box(&spots), black_box(&weights), black_box(100.0),
                black_box(0.05), black_box(&divs), black_box(&vols),
                black_box(&corr), black_box(1.0), black_box(true),
                black_box(BasketType::WeightedAverage),
                black_box(50_000), black_box(42),
            )
        })
    });
}

// ── SABR volatility ─────────────────────────────────────────────────────────

fn bench_sabr(c: &mut Criterion) {
    c.bench_function("sabr_implied_vol", |b| {
        b.iter(|| {
            sabr_volatility(
                black_box(0.03), black_box(0.025),
                black_box(5.0), black_box(0.035),
                black_box(0.5), black_box(-0.3), black_box(0.4),
            )
        })
    });
}

// ── SVI calibration ─────────────────────────────────────────────────────────

fn bench_svi_calibrate(c: &mut Criterion) {
    let strikes = vec![80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0];
    let vols = vec![0.28, 0.25, 0.22, 0.20, 0.19, 0.20, 0.22, 0.25, 0.29];

    c.bench_function("svi_calibrate_9_strikes", |b| {
        b.iter(|| {
            svi_calibrate(
                black_box(&strikes), black_box(&vols),
                black_box(100.0), black_box(1.0),
            )
        })
    });
}

// ── HW Jamshidian swaption ──────────────────────────────────────────────────

fn bench_hw_swaption(c: &mut Criterion) {
    let swap_tenors: Vec<f64> = (1..=10).map(|y| y as f64).collect();
    let dfs: Vec<f64> = swap_tenors.iter().map(|&t| (-0.04 * t).exp()).collect();

    c.bench_function("hw_jamshidian_swaption_10y", |b| {
        b.iter(|| {
            hw_jamshidian_swaption(
                black_box(0.03), black_box(0.01),
                black_box(1.0), black_box(&swap_tenors),
                black_box(0.04), black_box(&dfs),
                black_box((-0.04_f64).exp()), black_box(1_000_000.0),
                black_box(true),
            )
        })
    });
}

// ── Merton jump-diffusion ───────────────────────────────────────────────────

fn bench_merton_jd(c: &mut Criterion) {
    c.bench_function("merton_jump_diffusion_call", |b| {
        b.iter(|| {
            merton_jump_diffusion(
                black_box(100.0), black_box(100.0),
                black_box(0.05), black_box(0.0),
                black_box(0.15), black_box(1.0),
                black_box(0.5), black_box(-0.1), black_box(0.15),
                black_box(true),
            )
        })
    });
}

// ── Bates analytic (flat) ───────────────────────────────────────────────────

fn bench_bates_analytic(c: &mut Criterion) {
    c.bench_function("bates_analytic_flat_call", |b| {
        b.iter(|| {
            bates_price_flat(
                black_box(100.0), black_box(100.0),
                black_box(0.05), black_box(0.0), black_box(1.0),
                black_box(0.04), black_box(1.5), black_box(0.04),
                black_box(0.3), black_box(-0.7),
                black_box(0.5), black_box(-0.1), black_box(0.15),
                black_box(true),
            )
        })
    });
}

// ── Schedule generation ─────────────────────────────────────────────────────

fn bench_schedule_generation(c: &mut Criterion) {
    use ql_time::schedule::DateGenerationRule;
    use ql_time::Frequency;

    let effective = Date::from_ymd(2025, Month::January, 15);
    let termination = Date::from_ymd(2055, Month::January, 15);
    let cal = Calendar::UnitedStates(ql_time::calendar::USMarket::NYSE);

    c.bench_function("schedule_30y_semiannual", |b| {
        b.iter(|| {
            Schedule::builder()
                .effective_date(black_box(effective))
                .termination_date(black_box(termination))
                .frequency(black_box(Frequency::Semiannual))
                .calendar(black_box(cal.clone()))
                .convention(black_box(BusinessDayConvention::ModifiedFollowing))
                .rule(black_box(DateGenerationRule::Forward))
                .build()
        })
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
    bench_heston_analytic,
    bench_heston_calibration,
    bench_calendar_advance,
    bench_interpolation,
    bench_american_approx,
    bench_nelson_siegel_fit,
    // -- New benchmarks --
    bench_mc_barrier,
    bench_mc_asian,
    bench_mc_heston,
    bench_mc_bates,
    bench_fd_heston,
    bench_lsm_american,
    bench_kirk_spread,
    bench_mc_basket,
    bench_sabr,
    bench_svi_calibrate,
    bench_hw_swaption,
    bench_merton_jd,
    bench_bates_analytic,
    bench_schedule_generation,
);
criterion_main!(benches);
