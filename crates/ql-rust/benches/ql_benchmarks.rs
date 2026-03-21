//! Performance benchmarks for key ql-rust operations.
//!
//! Run with: `cargo bench -p ql-rust`

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

use ql_aad::dual::Dual;
use ql_aad::dual_vec::DualVec;
use ql_aad::tape::{adjoint_tl, with_tape, AReal};
use ql_aad::Number;
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
    GaussianCopulaLHP, cds_option_black,
    double_barrier_knockout, chooser_price, cliquet_price,
};
use ql_pricingengines::generic::{
    bs_european_generic, barone_adesi_whaley_generic, merton_jd_generic,
    chooser_generic,
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
                .unwrap()
        })
    });
}

// ── Vasicek bond pricing ─────────────────────────────────────────────────────

fn bench_vasicek_bond(c: &mut Criterion) {
    let model = ql_models::VasicekModel::new(0.3, 0.05, 0.02, 0.05);

    c.bench_function("vasicek_bond_5y", |b| {
        b.iter(|| model.bond_price(black_box(5.0)))
    });
}

// ── G2 swaption pricing ─────────────────────────────────────────────────────

fn bench_g2_swaption(c: &mut Criterion) {
    let model = ql_models::G2Model::new(0.1, 0.01, 0.15, 0.008, -0.5, 0.04);
    let swap_tenors: Vec<f64> = (1..=10).map(|y| y as f64).collect();

    c.bench_function("g2_swaption_10y", |b| {
        b.iter(|| {
            model.swaption_price(
                black_box(1.0),
                black_box(&swap_tenors),
                black_box(0.04),
                black_box(1_000_000.0),
                black_box(true),
            )
        })
    });
}

// ── FFT benchmark ────────────────────────────────────────────────────────────

fn bench_fft(c: &mut Criterion) {
    use ql_math::fft::{fft, Complex};

    let n = 8192;
    let data: Vec<Complex> = (0..n)
        .map(|i| Complex {
            re: (i as f64 / n as f64 * 6.28).sin(),
            im: 0.0,
        })
        .collect();

    c.bench_function("fft_8192", |b| {
        b.iter(|| {
            let mut d = data.clone();
            fft(black_box(&mut d), false);
            d
        })
    });
}

// ── Cholesky decomposition ──────────────────────────────────────────────────

fn bench_cholesky(c: &mut Criterion) {
    use ql_math::matrix::Matrix;

    // Build a 50×50 positive definite matrix
    let n = 50;
    let mut m = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            m[(i, j)] = if i == j {
                n as f64 + 1.0
            } else {
                0.5_f64.powi((i as i32 - j as i32).unsigned_abs() as i32)
            };
        }
    }

    c.bench_function("cholesky_50x50", |b| {
        b.iter(|| ql_math::matrix::cholesky(black_box(&m)))
    });
}

// ── CMS coupon pricing ──────────────────────────────────────────────────────

fn bench_cms_caplet(c: &mut Criterion) {
    c.bench_function("cms_caplet_pricing", |b| {
        b.iter(|| {
            ql_cashflows::cms_caplet_price(
                black_box(0.04), black_box(0.05),
                black_box(0.20), black_box(1.0),
                black_box(1.0), black_box(0.95),
                black_box(1_000_000.0), black_box(0.5),
            )
        })
    });
}

// ── LMM cap pricing ─────────────────────────────────────────────────────────

fn bench_lmm_cap(c: &mut Criterion) {
    let config = ql_models::LmmConfig::flat(10, 0.04, 0.5, 0.15, 0.5);

    c.bench_function("lmm_cap_10k_paths", |b| {
        b.iter(|| {
            ql_models::lmm_cap_price(
                black_box(&config),
                black_box(0.05),
                black_box(10_000),
                black_box(42),
            )
        })
    });
}

// ── Gaussian copula CDO tranche ─────────────────────────────────────────────

fn bench_gaussian_copula_cdo(c: &mut Criterion) {
    let default_prob = 1.0 - (-0.01_f64 * 5.0).exp();
    let copula = GaussianCopulaLHP::new(125, 0.3, 0.4, default_prob);

    c.bench_function("gaussian_copula_cdo_tranche", |b| {
        b.iter(|| {
            copula.tranche_expected_loss(black_box(0.0), black_box(0.03))
        })
    });
}

// ── CDS option (Black) ──────────────────────────────────────────────────────

fn bench_cds_option(c: &mut Criterion) {
    c.bench_function("cds_option_black", |b| {
        b.iter(|| {
            cds_option_black(
                black_box(0.01), black_box(0.012),
                black_box(0.40), black_box(1.0),
                black_box(4.5), black_box(true),
            )
        })
    });
}

// ── Double-barrier knock-out ─────────────────────────────────────────────────

fn bench_double_barrier_ko(c: &mut Criterion) {
    c.bench_function("double_barrier_ko_call", |b| {
        b.iter(|| {
            double_barrier_knockout(
                black_box(100.0), black_box(100.0),
                black_box(0.05), black_box(0.0),
                black_box(0.20), black_box(1.0),
                black_box(80.0), black_box(120.0),
                OptionType::Call, 20,
            )
        })
    });
}

// ── Chooser option ───────────────────────────────────────────────────────────

fn bench_chooser(c: &mut Criterion) {
    c.bench_function("chooser_rubinstein", |b| {
        b.iter(|| {
            chooser_price(
                black_box(50.0), black_box(50.0),
                black_box(0.08), black_box(0.0),
                black_box(0.25), black_box(0.25),
                black_box(0.50),
            )
        })
    });
}

// ── Cliquet option ───────────────────────────────────────────────────────────

fn bench_cliquet(c: &mut Criterion) {
    let resets = vec![0.25, 0.50, 0.75, 1.00];
    c.bench_function("cliquet_4_period_call", |b| {
        b.iter(|| {
            cliquet_price(
                black_box(100.0), black_box(0.05), black_box(0.02),
                black_box(0.20), black_box(&resets),
                black_box(-0.05), black_box(0.10),
                black_box(0.0), black_box(1.0),
                black_box(1_000_000.0), OptionType::Call,
            )
        })
    });
}

// ── CDS pricing via midpoint engine ──────────────────────────────────────────

fn bench_cds_pricing(c: &mut Criterion) {
    use std::sync::Arc;
    use ql_instruments::credit_default_swap::{CreditDefaultSwap, CdsPremiumPeriod, CdsProtectionSide};
    use ql_termstructures::default_term_structure::{DefaultProbabilityTermStructure, FlatHazardRate};

    let ref_date = Date::from_ymd(2025, Month::March, 20);
    let dc = DayCounter::Actual365Fixed;
    let spread = 0.01;

    let schedule: Vec<CdsPremiumPeriod> = (1..=20)
        .map(|i| {
            let q = (i - 1) % 4;
            let m = [Month::March, Month::June, Month::September, Month::December][q];
            let y0 = 2025 + (i - 1) / 4;
            let eq = i % 4;
            let em = [Month::March, Month::June, Month::September, Month::December][eq];
            let y1 = 2025 + i / 4;
            CdsPremiumPeriod {
                accrual_start: Date::from_ymd(y0 as i32, m, 20),
                accrual_end: Date::from_ymd(y1 as i32, em, 20),
                payment_date: Date::from_ymd(y1 as i32, em, 20),
                accrual_fraction: 0.25,
            }
        })
        .collect();

    let cds = CreditDefaultSwap::new(CdsProtectionSide::Buyer, 10_000_000.0, spread,
        Date::from_ymd(2030, Month::March, 20), 0.4, schedule);
    let default_curve = Arc::new(FlatHazardRate::from_spread(ref_date, spread, 0.4, dc));
    let yield_curve = Arc::new(FlatForward::new(ref_date, 0.03, dc));

    c.bench_function("cds_midpoint_5y", |b| {
        b.iter(|| {
            ql_pricingengines::midpoint_cds_engine(
                black_box(&cds),
                &(default_curve.clone() as Arc<dyn DefaultProbabilityTermStructure>),
                &(yield_curve.clone() as Arc<dyn YieldTermStructure>),
                0.0,
            )
        })
    });
}

// ── Risk analytics: key-rate durations & scenario analysis ───────────────────

fn bench_risk_analytics(c: &mut Criterion) {
    use ql_pricingengines::{
        key_rate_durations, scenario_analysis, YieldCurveScenario, equity_risk_ladder,
        EquityMarketParams,
    };

    let ref_date = Date::from_ymd(2025, Month::January, 15);
    let dc = DayCounter::Actual360;
    let curve = FlatForward::new(ref_date, 0.04, dc);

    // Build a 30-semi leg for KRD / scenario benchmarks
    let sched = Schedule::from_dates(
        (0..=60).map(|i| {
            let y = 2025 + i / 2;
            let m = if i % 2 == 0 { Month::January } else { Month::July };
            Date::from_ymd(y, m, 15)
        }).collect(),
    );
    let leg = fixed_leg(&sched, &[1_000_000.0], &[0.05], dc);

    let tenors: Vec<f64> = vec![0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0];
    c.bench_function("key_rate_durations_10_tenors", |b| {
        b.iter(|| key_rate_durations(black_box(&leg), black_box(&curve), ref_date, &tenors, 1.0))
    });

    let scenarios = vec![
        ("Up 100bp".into(), YieldCurveScenario::ParallelShift(0.01)),
        ("Down 100bp".into(), YieldCurveScenario::ParallelShift(-0.01)),
        ("Steepen".into(), YieldCurveScenario::SteepenerFlattener { short_shift: -0.005, long_shift: 0.01 }),
        ("Flatten".into(), YieldCurveScenario::SteepenerFlattener { short_shift: 0.005, long_shift: -0.005 }),
        ("Custom".into(), YieldCurveScenario::Custom {
            tenors: vec![1.0, 5.0, 10.0, 30.0],
            shifts: vec![0.002, 0.005, 0.008, 0.012],
        }),
    ];
    c.bench_function("scenario_analysis_5", |b| {
        b.iter(|| scenario_analysis(black_box(&leg), black_box(&curve), ref_date, &scenarios))
    });

    // Equity risk ladder (bump-and-reprice Greeks)
    let call = VanillaOption::european_call(105.0, Date::from_ymd(2026, Month::January, 15));
    let params = EquityMarketParams {
        spot: 100.0,
        risk_free_rate: 0.05,
        dividend_yield: 0.02,
        volatility: 0.20,
        time_to_expiry: 1.0,
    };
    c.bench_function("equity_risk_ladder", |b| {
        b.iter(|| equity_risk_ladder(black_box(&call), black_box(&params)))
    });
}

// ── Svensson curve fitting ───────────────────────────────────────────────────

fn bench_svensson_fit(c: &mut Criterion) {
    use ql_termstructures::SvenssonFitting;

    let maturities = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0];
    let rates = [0.040, 0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.047, 0.048, 0.048, 0.049];

    c.bench_function("svensson_fit_11_points", |b| {
        b.iter(|| SvenssonFitting::fit(black_box(&maturities), black_box(&rates)).unwrap())
    });
}

// ── Lookback option pricing ──────────────────────────────────────────────────

fn bench_lookback(c: &mut Criterion) {
    use ql_instruments::lookback_option::{LookbackOption, LookbackType};

    let opt = LookbackOption {
        lookback_type: LookbackType::FloatingStrike,
        option_type: OptionType::Call,
        min_so_far: 90.0,
        max_so_far: 110.0,
        strike: 0.0,
        time_to_expiry: 1.0,
    };

    c.bench_function("lookback_floating_call", |b| {
        b.iter(|| {
            ql_pricingengines::analytic_lookback(
                black_box(&opt), black_box(100.0), black_box(0.05),
                black_box(0.0), black_box(0.25),
            )
        })
    });
}

// ── Variance swap pricing ────────────────────────────────────────────────────

fn bench_variance_swap(c: &mut Criterion) {
    use ql_instruments::variance_swap::VarianceSwap;

    let vs = VarianceSwap::from_vol_strike(1_000_000.0, 0.20, 1.0);
    c.bench_function("variance_swap_1y", |b| {
        b.iter(|| {
            ql_pricingengines::price_variance_swap(
                black_box(&vs), black_box(0.22), black_box(0.05),
            )
        })
    });
}

// ── Serde round-trip benchmark ───────────────────────────────────────────────

fn bench_serde_round_trip(c: &mut Criterion) {
    let today = Date::from_ymd(2025, Month::January, 15);
    let call = VanillaOption::european_call(105.0, today + 365);

    let json = serde_json::to_string(&call).unwrap();

    c.bench_function("serde_vanilla_option_serialize", |b| {
        b.iter(|| serde_json::to_string(black_box(&call)).unwrap())
    });

    c.bench_function("serde_vanilla_option_deserialize", |b| {
        b.iter(|| serde_json::from_str::<VanillaOption>(black_box(&json)).unwrap())
    });
}

// ── Reactive pricing infrastructure ─────────────────────────────────────────

// Shared helper types for the reactive benchmarks.
#[derive(Clone)]
struct BenchParams {
    strike: f64,
}
#[derive(Clone)]
struct BenchResult {
    npv: f64,
}
impl ql_core::portfolio::HasNpv for BenchResult {
    fn npv_value(&self) -> f64 {
        self.npv
    }
}

fn make_bench_lazy(
    spot: std::sync::Arc<ql_core::quote::SimpleQuote>,
    strike: f64,
) -> std::sync::Arc<ql_core::engine::LazyInstrument<BenchParams, BenchResult>> {
    use ql_core::errors::QLResult;
    use ql_core::quote::Quote;
    let s = std::sync::Arc::clone(&spot);
    let engine = ql_core::engine::ClosureEngine::new(move |p: &BenchParams| {
        let sv: QLResult<f64> = s.value();
        Ok(BenchResult { npv: (sv? - p.strike).max(0.0) })
    });
    std::sync::Arc::new(ql_core::engine::LazyInstrument::new(
        BenchParams { strike },
        Box::new(engine),
    ))
}

/// Hot path: NPV is already cached — measures lock + atomic-flag overhead only.
fn bench_reactive_lazy_cache_hit(c: &mut Criterion) {
    use std::sync::Arc;
    use ql_core::observable::{Observable, Observer};
    use ql_core::portfolio::NpvProvider;
    use ql_core::quote::SimpleQuote;

    let spot  = Arc::new(SimpleQuote::new(110.0));
    let instr = make_bench_lazy(spot.clone(), 100.0);
    spot.register_observer(&(instr.clone() as Arc<dyn Observer>));
    let _ = NpvProvider::npv(instr.as_ref()); // warm-up cache

    c.bench_function("reactive_lazy_cache_hit", |b| {
        b.iter(|| NpvProvider::npv(black_box(instr.as_ref())))
    });
}

/// Cold path: quote changes each iteration → cache miss → full engine invocation.
fn bench_reactive_lazy_cache_miss(c: &mut Criterion) {
    use std::sync::Arc;
    use ql_core::observable::{Observable, Observer};
    use ql_core::portfolio::NpvProvider;
    use ql_core::quote::SimpleQuote;

    let spot  = Arc::new(SimpleQuote::new(110.0));
    let instr = make_bench_lazy(spot.clone(), 100.0);
    spot.register_observer(&(instr.clone() as Arc<dyn Observer>));

    let mut sv = 110.0_f64;
    c.bench_function("reactive_lazy_cache_miss", |b| {
        b.iter(|| {
            sv += 0.001;
            spot.set_value(black_box(sv));
            NpvProvider::npv(black_box(instr.as_ref()))
        })
    });
}

/// Cached total NPV across 5 instruments — hot path, nothing dirty.
fn bench_reactive_portfolio_cached(c: &mut Criterion) {
    use std::sync::Arc;
    use ql_core::observable::{Observable, Observer};
    use ql_core::portfolio::{wire_entry, NpvProvider, ReactivePortfolio};
    use ql_core::quote::SimpleQuote;

    let portfolio = Arc::new(ReactivePortfolio::new("bench-book"));
    for i in 0..5_u32 {
        let spot  = Arc::new(SimpleQuote::new(100.0 + i as f64));
        let instr = make_bench_lazy(spot.clone(), 90.0);
        spot.register_observer(&(instr.clone() as Arc<dyn Observer>));
        wire_entry(&portfolio, instr as Arc<dyn NpvProvider>);
    }
    let _ = portfolio.total_npv(); // warm-up

    c.bench_function("reactive_portfolio_5_instruments_cached", |b| {
        b.iter(|| portfolio.total_npv())
    });
}

/// One quote changes per iteration → portfolio dirty → full 5-instrument reprice.
fn bench_reactive_portfolio_invalidate_reprice(c: &mut Criterion) {
    use std::sync::Arc;
    use ql_core::observable::{Observable, Observer};
    use ql_core::portfolio::{wire_entry, NpvProvider, ReactivePortfolio};
    use ql_core::quote::SimpleQuote;

    let portfolio = Arc::new(ReactivePortfolio::new("bench-book-inv"));
    let spot0 = Arc::new(SimpleQuote::new(100.0));
    let instr0 = make_bench_lazy(spot0.clone(), 90.0);
    spot0.register_observer(&(instr0.clone() as Arc<dyn Observer>));
    wire_entry(&portfolio, instr0 as Arc<dyn NpvProvider>);

    for i in 1..5_u32 {
        let spot  = Arc::new(SimpleQuote::new(100.0 + i as f64));
        let instr = make_bench_lazy(spot.clone(), 90.0);
        spot.register_observer(&(instr.clone() as Arc<dyn Observer>));
        wire_entry(&portfolio, instr as Arc<dyn NpvProvider>);
    }

    let mut sv = 100.0_f64;
    c.bench_function("reactive_portfolio_invalidate_reprice", |b| {
        b.iter(|| {
            sv += 0.001;
            spot0.set_value(black_box(sv));
            portfolio.total_npv()
        })
    });
}

/// Feed publish throughput with a single subscriber.
fn bench_feed_publish_1_subscriber(c: &mut Criterion) {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering as O};
    use ql_core::market_data::{FeedEvent, InMemoryFeed, MarketDataFeed};

    let feed = Arc::new(InMemoryFeed::new("bench-feed-1"));
    let counter = Arc::new(AtomicU64::new(0));
    let cc = counter.clone();
    feed.subscribe("AAPL", Arc::new(move |_: FeedEvent| { cc.fetch_add(1, O::Relaxed); }));

    let mut price = 100.0_f64;
    c.bench_function("feed_publish_1_subscriber", |b| {
        b.iter(|| {
            price += 0.001;
            feed.publish(FeedEvent::new("AAPL", black_box(price)))
        })
    });
}

/// Feed publish throughput with ten subscribers on the same ticker.
fn bench_feed_publish_10_subscribers(c: &mut Criterion) {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering as O};
    use ql_core::market_data::{FeedEvent, InMemoryFeed, MarketDataFeed};

    let feed = Arc::new(InMemoryFeed::new("bench-feed-10"));
    for _ in 0..10 {
        let counter = Arc::new(AtomicU64::new(0));
        feed.subscribe(
            "AAPL",
            Arc::new(move |_: FeedEvent| { counter.fetch_add(1, O::Relaxed); }),
        );
    }

    let mut price = 100.0_f64;
    c.bench_function("feed_publish_10_subscribers", |b| {
        b.iter(|| {
            price += 0.001;
            feed.publish(FeedEvent::new("AAPL", black_box(price)))
        })
    });
}

// ══════════════════════════════════════════════════════════════════════════════
// AD (Automatic Differentiation) benchmark group
// ══════════════════════════════════════════════════════════════════════════════
//
// Compares f64 (baseline) vs forward-mode Dual (single Greek) vs
// forward-mode DualVec<5> (all Greeks in one pass) vs reverse-mode AReal
// for key generic engines.

fn bench_ad_bs_european(c: &mut Criterion) {
    let mut group = c.benchmark_group("ad_bs_european");

    group.bench_function("f64", |b| {
        b.iter(|| {
            bs_european_generic::<f64>(
                black_box(100.0), black_box(100.0),
                black_box(0.05), black_box(0.0),
                black_box(0.20), black_box(1.0), true,
            )
        })
    });

    group.bench_function("Dual_delta", |b| {
        b.iter(|| {
            bs_european_generic(
                Dual::new(black_box(100.0), 1.0),
                Dual::constant(black_box(100.0)),
                Dual::constant(black_box(0.05)),
                Dual::constant(black_box(0.0)),
                Dual::constant(black_box(0.20)),
                Dual::constant(black_box(1.0)),
                true,
            )
        })
    });

    group.bench_function("DualVec5_all_greeks", |b| {
        b.iter(|| {
            type D5 = DualVec<5>;
            bs_european_generic(
                D5::variable(black_box(100.0), 0),
                D5::constant(black_box(100.0)),
                D5::variable(black_box(0.05), 1),
                D5::variable(black_box(0.0), 2),
                D5::variable(black_box(0.20), 3),
                D5::variable(black_box(1.0), 4),
                true,
            )
        })
    });

    group.bench_function("AReal_reverse", |b| {
        b.iter(|| {
            let npv = with_tape(|tape| {
                let s = tape.input(black_box(100.0));
                let k = tape.input(black_box(100.0));
                let r = tape.input(black_box(0.05));
                let q = tape.input(black_box(0.0));
                let v = tape.input(black_box(0.20));
                let t = AReal::from_f64(black_box(1.0));
                bs_european_generic(s, k, r, q, v, t, true).npv
            });
            adjoint_tl(npv)
        })
    });

    group.finish();
}

fn bench_ad_baw_american(c: &mut Criterion) {
    let mut group = c.benchmark_group("ad_baw_american");

    group.bench_function("f64", |b| {
        b.iter(|| {
            barone_adesi_whaley_generic::<f64>(
                black_box(100.0), black_box(100.0),
                black_box(0.05), black_box(0.02),
                black_box(0.20), black_box(1.0), false,
            )
        })
    });

    group.bench_function("Dual_delta", |b| {
        b.iter(|| {
            barone_adesi_whaley_generic(
                Dual::new(black_box(100.0), 1.0),
                Dual::constant(black_box(100.0)),
                Dual::constant(black_box(0.05)),
                Dual::constant(black_box(0.02)),
                Dual::constant(black_box(0.20)),
                Dual::constant(black_box(1.0)),
                false,
            )
        })
    });

    group.bench_function("DualVec5_all_greeks", |b| {
        b.iter(|| {
            type D5 = DualVec<5>;
            barone_adesi_whaley_generic(
                D5::variable(black_box(100.0), 0),
                D5::constant(black_box(100.0)),
                D5::variable(black_box(0.05), 1),
                D5::variable(black_box(0.02), 2),
                D5::variable(black_box(0.20), 3),
                D5::variable(black_box(1.0), 4),
                false,
            )
        })
    });

    group.bench_function("AReal_reverse", |b| {
        b.iter(|| {
            let npv = with_tape(|tape| {
                let s = tape.input(black_box(100.0));
                let k = tape.input(black_box(100.0));
                let r = tape.input(black_box(0.05));
                let q = tape.input(black_box(0.02));
                let v = tape.input(black_box(0.20));
                let t = AReal::from_f64(black_box(1.0));
                barone_adesi_whaley_generic(s, k, r, q, v, t, false).npv
            });
            adjoint_tl(npv)
        })
    });

    group.finish();
}

fn bench_ad_merton_jd(c: &mut Criterion) {
    let mut group = c.benchmark_group("ad_merton_jd");

    group.bench_function("f64", |b| {
        b.iter(|| {
            merton_jd_generic::<f64>(
                black_box(100.0), black_box(100.0),
                black_box(0.05), black_box(0.0),
                black_box(0.15), black_box(1.0),
                black_box(0.5), black_box(-0.1), black_box(0.15),
                true,
            )
        })
    });

    group.bench_function("Dual_delta", |b| {
        b.iter(|| {
            merton_jd_generic(
                Dual::new(black_box(100.0), 1.0),
                Dual::constant(black_box(100.0)),
                Dual::constant(black_box(0.05)),
                Dual::constant(black_box(0.0)),
                Dual::constant(black_box(0.15)),
                Dual::constant(black_box(1.0)),
                Dual::constant(black_box(0.5)),
                Dual::constant(black_box(-0.1)),
                Dual::constant(black_box(0.15)),
                true,
            )
        })
    });

    group.bench_function("DualVec5_all_greeks", |b| {
        b.iter(|| {
            type D5 = DualVec<5>;
            merton_jd_generic(
                D5::variable(black_box(100.0), 0),
                D5::constant(black_box(100.0)),
                D5::variable(black_box(0.05), 1),
                D5::variable(black_box(0.0), 2),
                D5::variable(black_box(0.15), 3),
                D5::constant(black_box(1.0)),
                D5::constant(black_box(0.5)),
                D5::constant(black_box(-0.1)),
                D5::variable(black_box(0.15), 4),
                true,
            )
        })
    });

    group.bench_function("AReal_reverse", |b| {
        b.iter(|| {
            let npv = with_tape(|tape| {
                let s = tape.input(black_box(100.0));
                let k = tape.input(black_box(100.0));
                let r = tape.input(black_box(0.05));
                let q = tape.input(black_box(0.0));
                let vol = tape.input(black_box(0.15));
                let t = AReal::from_f64(black_box(1.0));
                let lam = tape.input(black_box(0.5));
                let nu = tape.input(black_box(-0.1));
                let delta = tape.input(black_box(0.15));
                merton_jd_generic(s, k, r, q, vol, t, lam, nu, delta, true).npv
            });
            adjoint_tl(npv)
        })
    });

    group.finish();
}

fn bench_ad_chooser(c: &mut Criterion) {
    let mut group = c.benchmark_group("ad_chooser");

    group.bench_function("f64", |b| {
        b.iter(|| {
            chooser_generic::<f64>(
                black_box(50.0), black_box(50.0),
                black_box(0.08), black_box(0.0),
                black_box(0.25), black_box(0.25), black_box(0.50),
            )
        })
    });

    group.bench_function("Dual_delta", |b| {
        b.iter(|| {
            chooser_generic(
                Dual::new(black_box(50.0), 1.0),
                Dual::constant(black_box(50.0)),
                Dual::constant(black_box(0.08)),
                Dual::constant(black_box(0.0)),
                Dual::constant(black_box(0.25)),
                Dual::constant(black_box(0.25)),
                Dual::constant(black_box(0.50)),
            )
        })
    });

    group.bench_function("AReal_reverse", |b| {
        b.iter(|| {
            let npv = with_tape(|tape| {
                let s = tape.input(black_box(50.0));
                let k = tape.input(black_box(50.0));
                let r = tape.input(black_box(0.08));
                let q = tape.input(black_box(0.0));
                let vol = tape.input(black_box(0.25));
                let tc = AReal::from_f64(black_box(0.25));
                let te = AReal::from_f64(black_box(0.50));
                chooser_generic(s, k, r, q, vol, tc, te)
            });
            adjoint_tl(npv)
        })
    });

    group.finish();
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
    // -- Phase 16+ benchmarks --
    bench_vasicek_bond,
    bench_g2_swaption,
    bench_fft,
    bench_cholesky,
    bench_cms_caplet,
    bench_lmm_cap,
    bench_gaussian_copula_cdo,
    bench_cds_option,
    // -- Phase 26: exotic engines --
    bench_double_barrier_ko,
    bench_chooser,
    bench_cliquet,
    // -- Phase 31: new hot-path benchmarks --
    bench_cds_pricing,
    bench_risk_analytics,
    bench_svensson_fit,
    bench_lookback,
    bench_variance_swap,
    bench_serde_round_trip,
    // -- Phase 25: reactive pricing infrastructure --
    bench_reactive_lazy_cache_hit,
    bench_reactive_lazy_cache_miss,
    bench_reactive_portfolio_cached,
    bench_reactive_portfolio_invalidate_reprice,
    bench_feed_publish_1_subscriber,
    bench_feed_publish_10_subscribers,
    // -- AD (automatic differentiation) benchmarks --
    bench_ad_bs_european,
    bench_ad_baw_american,
    bench_ad_merton_jd,
    bench_ad_chooser,
);
criterion_main!(benches);
