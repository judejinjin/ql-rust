//! Criterion benchmarks for ql-aad — comparative f64 vs Dual vs AReal vs SIMD vs JIT.
//!
//! Run with:
//!   cargo bench -p ql-aad                       # without JIT
//!   cargo bench -p ql-aad --features jit        # with JIT benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use ql_aad::bs::{bs_greeks_forward_ad, bs_price_generic, OptionKind};
use ql_aad::dual::Dual;
use ql_aad::heston::{heston_greeks_ad, heston_price_generic};
use ql_aad::mc::{mc_european_aad, mc_european_forward, mc_heston_aad};
use ql_aad::simd::{mc_european_simd4, mc_heston_simd4};
use ql_aad::tape::{adjoint_tl, with_tape, AReal};

// ── Parameters ───────────────────────────────────────────────────────────────

const SPOT: f64 = 100.0;
const STRIKE: f64 = 100.0;
const R: f64 = 0.05;
const Q: f64 = 0.02;
const VOL: f64 = 0.20;
const T: f64 = 1.0;
const SEED: u64 = 42;

// Heston parameters
const V0: f64 = 0.04;
const KAPPA: f64 = 1.5;
const THETA: f64 = 0.04;
const SIGMA: f64 = 0.3;
const RHO: f64 = -0.7;

// MC path counts
const MC_PATHS_SMALL: usize = 10_000;
const MC_PATHS_LARGE: usize = 50_000;

// ── Black-Scholes pricing ────────────────────────────────────────────────────

fn bench_bs_f64(c: &mut Criterion) {
    c.bench_function("aad_bs_price_f64", |b| {
        b.iter(|| {
            bs_price_generic::<f64>(
                black_box(SPOT),
                black_box(STRIKE),
                black_box(R),
                black_box(Q),
                black_box(VOL),
                black_box(T),
                OptionKind::Call,
            )
        })
    });
}

fn bench_bs_dual(c: &mut Criterion) {
    c.bench_function("aad_bs_price_dual", |b| {
        b.iter(|| {
            let spot = Dual { val: SPOT, dot: 1.0 };
            let strike = Dual { val: STRIKE, dot: 0.0 };
            let r = Dual { val: R, dot: 0.0 };
            let q = Dual { val: Q, dot: 0.0 };
            let vol = Dual { val: VOL, dot: 0.0 };
            let t = Dual { val: T, dot: 0.0 };
            bs_price_generic::<Dual>(
                black_box(spot),
                black_box(strike),
                black_box(r),
                black_box(q),
                black_box(vol),
                black_box(t),
                OptionKind::Call,
            )
        })
    });
}

fn bench_bs_dual_vec5(c: &mut Criterion) {
    c.bench_function("aad_bs_greeks_dual_vec5", |b| {
        b.iter(|| {
            bs_greeks_forward_ad(
                black_box(SPOT),
                black_box(STRIKE),
                black_box(R),
                black_box(Q),
                black_box(VOL),
                black_box(T),
                OptionKind::Call,
            )
        })
    });
}

fn bench_bs_areal(c: &mut Criterion) {
    c.bench_function("aad_bs_price_areal", |b| {
        b.iter(|| {
            with_tape(|tape| {
                let spot = tape.input(SPOT);
                let strike = tape.input(STRIKE);
                let r = tape.input(R);
                let q = tape.input(Q);
                let vol = tape.input(VOL);
                let t = tape.input(T);
                let price = bs_price_generic::<AReal>(spot, strike, r, q, vol, t, OptionKind::Call);
                let _adj = adjoint_tl(price);
            })
        })
    });
}

// ── Heston pricing ──────────────────────────────────────────────────────────

fn bench_heston_f64(c: &mut Criterion) {
    c.bench_function("aad_heston_price_f64", |b| {
        b.iter(|| {
            heston_price_generic::<f64>(
                black_box(SPOT),
                black_box(STRIKE),
                black_box(R),
                black_box(Q),
                black_box(T),
                black_box(V0),
                black_box(KAPPA),
                black_box(THETA),
                black_box(SIGMA),
                black_box(RHO),
                true,
            )
        })
    });
}

fn bench_heston_greeks_ad(c: &mut Criterion) {
    c.bench_function("aad_heston_greeks_adjoint", |b| {
        b.iter(|| {
            heston_greeks_ad(
                black_box(SPOT),
                black_box(STRIKE),
                black_box(R),
                black_box(Q),
                black_box(T),
                black_box(V0),
                black_box(KAPPA),
                black_box(THETA),
                black_box(SIGMA),
                black_box(RHO),
                true,
            )
        })
    });
}

// ── MC European pricing ─────────────────────────────────────────────────────

fn bench_mc_european_aad_10k(c: &mut Criterion) {
    c.bench_function("aad_mc_european_areal_10k", |b| {
        b.iter(|| {
            mc_european_aad(
                black_box(SPOT),
                black_box(STRIKE),
                black_box(R),
                black_box(Q),
                black_box(VOL),
                black_box(T),
                OptionKind::Call,
                MC_PATHS_SMALL,
                SEED,
            )
        })
    });
}

fn bench_mc_european_forward_10k(c: &mut Criterion) {
    c.bench_function("aad_mc_european_forward_10k", |b| {
        b.iter(|| {
            mc_european_forward(
                black_box(SPOT),
                black_box(STRIKE),
                black_box(R),
                black_box(Q),
                black_box(VOL),
                black_box(T),
                OptionKind::Call,
                MC_PATHS_SMALL,
                SEED,
            )
        })
    });
}

fn bench_mc_european_simd4_10k(c: &mut Criterion) {
    c.bench_function("aad_mc_european_simd4_10k", |b| {
        b.iter(|| {
            mc_european_simd4(
                black_box(SPOT),
                black_box(STRIKE),
                black_box(R),
                black_box(Q),
                black_box(VOL),
                black_box(T),
                OptionKind::Call,
                MC_PATHS_SMALL,
                SEED,
            )
        })
    });
}

fn bench_mc_european_aad_50k(c: &mut Criterion) {
    c.bench_function("aad_mc_european_areal_50k", |b| {
        b.iter(|| {
            mc_european_aad(
                black_box(SPOT),
                black_box(STRIKE),
                black_box(R),
                black_box(Q),
                black_box(VOL),
                black_box(T),
                OptionKind::Call,
                MC_PATHS_LARGE,
                SEED,
            )
        })
    });
}

fn bench_mc_european_simd4_50k(c: &mut Criterion) {
    c.bench_function("aad_mc_european_simd4_50k", |b| {
        b.iter(|| {
            mc_european_simd4(
                black_box(SPOT),
                black_box(STRIKE),
                black_box(R),
                black_box(Q),
                black_box(VOL),
                black_box(T),
                OptionKind::Call,
                MC_PATHS_LARGE,
                SEED,
            )
        })
    });
}

// ── MC Heston pricing ───────────────────────────────────────────────────────

fn bench_mc_heston_aad_10k(c: &mut Criterion) {
    c.bench_function("aad_mc_heston_areal_10k", |b| {
        b.iter(|| {
            mc_heston_aad(
                black_box(SPOT),
                black_box(STRIKE),
                black_box(R),
                black_box(Q),
                black_box(V0),
                black_box(KAPPA),
                black_box(THETA),
                black_box(SIGMA),
                black_box(RHO),
                black_box(T),
                OptionKind::Call,
                MC_PATHS_SMALL,
                50,
                SEED,
            )
        })
    });
}

fn bench_mc_heston_simd4_10k(c: &mut Criterion) {
    c.bench_function("aad_mc_heston_simd4_10k", |b| {
        b.iter(|| {
            mc_heston_simd4(
                black_box(SPOT),
                black_box(STRIKE),
                black_box(R),
                black_box(Q),
                black_box(V0),
                black_box(KAPPA),
                black_box(THETA),
                black_box(SIGMA),
                black_box(RHO),
                black_box(T),
                OptionKind::Call,
                MC_PATHS_SMALL,
                50,
                SEED,
            )
        })
    });
}

// ── JIT benchmarks (feature-gated) ──────────────────────────────────────────

#[cfg(feature = "jit")]
mod jit_benches {
    use super::*;
    use ql_aad::jit::{mc_european_jit, mc_heston_jit};

    pub fn bench_mc_european_jit_10k(c: &mut Criterion) {
        c.bench_function("aad_mc_european_jit_10k", |b| {
            b.iter(|| {
                mc_european_jit(
                    black_box(SPOT),
                    black_box(STRIKE),
                    black_box(R),
                    black_box(Q),
                    black_box(VOL),
                    black_box(T),
                    OptionKind::Call,
                    MC_PATHS_SMALL,
                    SEED,
                )
            })
        });
    }

    pub fn bench_mc_european_jit_50k(c: &mut Criterion) {
        c.bench_function("aad_mc_european_jit_50k", |b| {
            b.iter(|| {
                mc_european_jit(
                    black_box(SPOT),
                    black_box(STRIKE),
                    black_box(R),
                    black_box(Q),
                    black_box(VOL),
                    black_box(T),
                    OptionKind::Call,
                    MC_PATHS_LARGE,
                    SEED,
                )
            })
        });
    }

    pub fn bench_mc_heston_jit_10k(c: &mut Criterion) {
        c.bench_function("aad_mc_heston_jit_10k", |b| {
            b.iter(|| {
                mc_heston_jit(
                    black_box(SPOT),
                    black_box(STRIKE),
                    black_box(R),
                    black_box(Q),
                    black_box(V0),
                    black_box(KAPPA),
                    black_box(THETA),
                    black_box(SIGMA),
                    black_box(RHO),
                    black_box(T),
                    OptionKind::Call,
                    MC_PATHS_SMALL,
                    50,
                    SEED,
                )
            })
        });
    }
}

// ── Criterion groups ────────────────────────────────────────────────────────

criterion_group!(
    bs_benches,
    bench_bs_f64,
    bench_bs_dual,
    bench_bs_dual_vec5,
    bench_bs_areal,
);

criterion_group!(
    heston_benches,
    bench_heston_f64,
    bench_heston_greeks_ad,
);

criterion_group!(
    mc_european_benches,
    bench_mc_european_aad_10k,
    bench_mc_european_forward_10k,
    bench_mc_european_simd4_10k,
    bench_mc_european_aad_50k,
    bench_mc_european_simd4_50k,
);

criterion_group!(
    mc_heston_benches,
    bench_mc_heston_aad_10k,
    bench_mc_heston_simd4_10k,
);

#[cfg(feature = "jit")]
criterion_group!(
    jit_mc_benches,
    jit_benches::bench_mc_european_jit_10k,
    jit_benches::bench_mc_european_jit_50k,
    jit_benches::bench_mc_heston_jit_10k,
);

#[cfg(feature = "jit")]
criterion_main!(bs_benches, heston_benches, mc_european_benches, mc_heston_benches, jit_mc_benches);

#[cfg(not(feature = "jit"))]
criterion_main!(bs_benches, heston_benches, mc_european_benches, mc_heston_benches);
