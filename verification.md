# ql-rust Verification Plan

Cross-validate ql-rust pricing and AD outputs against **QuantLib-Risks-Py** (v1.33.3),
a C++ QuantLib fork with XAD reverse-mode algorithmic differentiation.

**Source scripts:** `/mnt/c/finance/ad_tutorial/benchmarks/*.py`

---

## Scope

21 benchmark scripts covering 4 asset classes, 3 sensitivity methods
(finite-difference, forward-mode AD, reverse-mode AD), and 5 second-order
Hessian computations. Each script must be re-implemented as a Rust integration
test in ql-rust that reproduces the reference NPV and Greeks to within stated
tolerances.

### Sensitivity Methods Compared

| Method | QuantLib-Risks-Py | ql-rust Equivalent |
|--------|-------------------|--------------------|
| Finite differences (bump & reprice) | `SimpleQuote` bump loop | `SimpleQuote` bump loop or manual bump |
| AAD reverse-mode | XAD tape record → backward sweep | `AReal` tape record → backward sweep |
| AAD forward-mode | *(not used in benchmarks)* | `Dual` / `DualVec<N>` via generic engines |

### Tolerances

| Comparison | Tolerance | Notes |
|------------|-----------|-------|
| NPV vs reference | 1e-4 relative | Exact for analytic engines; MC may need wider |
| First-order Greeks (AAD vs FD) | 1e-3 relative | Matches QuantLib-Risks results |
| Second-order Hessian (FD-over-AAD) | 1e-2 relative | Inherits FD noise from bump step |
| Jacobian round-trip (K×J ≈ I) | 1e-4 max entry | Per QuantLib-Risks verification |

---

## Benchmark Matrix

### Group A — Equity Derivatives (First-Order)

#### A1. European Option — BSM Closed-Form
- **Script:** `european_option_benchmarks.py`
- **Engine:** `AnalyticEuropeanEngine` → ql-rust `bs_european_generic`
- **Parameters:** S=7.0, K=8.0, σ=0.10, r=0.05, q=0.05, T=1Y (European Call)
- **Inputs (4):** spot, div-yield, vol, rate

| Output | Reference Value |
|--------|----------------:|
| NPV | 0.0303344207 |
| ∂V/∂S (delta) | 0.09509987 |
| ∂V/∂q (div-rho) | −0.66934673 |
| ∂V/∂σ (vega) | 1.17147727 |
| ∂V/∂r (rho) | 0.63884610 |

#### A2. American Option — 4 Engines
- **Script:** `american_option_benchmarks.py`
- **Parameters:** S=36.0, K=40.0, σ=0.20, r=0.06, q=0.00, T=1Y (American Put)
- **Inputs (4):** spot, rate, div-yield, vol

| Engine | NPV | ∂V/∂S | ∂V/∂r | ∂V/∂q | ∂V/∂σ |
|--------|------:|------:|------:|------:|------:|
| BAW | 4.4622 | −0.6907 | −10.3683 | 9.3026 | 10.9987 |
| Bjerksund-Stensland | 4.4557 | −0.7029 | −9.7203 | 8.7084 | 10.5930 |
| FD-BS (PDE) | 4.4887 | −0.6960 | −10.3835 | 9.0911 | 10.9782 |
| QD+ | 4.4997 | −0.6981 | −10.2895 | 8.9972 | 10.9630 |

#### A3. Basket Option — MC 2-Asset
- **Script:** `basket_option_benchmarks.py`
- **Engine:** `MCEuropeanBasketEngine` → ql-rust `mc_basket`
- **Parameters:** S1=S2=7.0, K=8.0, σ1=σ2=0.10, r=0.05, q1=q2=0.05, ρ=0.5, T=1Y
- **Inputs (5):** spot1, spot2, vol1, vol2, rate
- **Note:** 32768 low-discrepancy samples, seed=42. MC tolerance wider (~1e-2).

| Output | Reference Value |
|--------|----------------:|
| NPV | 0.0534 |
| ∂V/∂S1 | 0.08055 |
| ∂V/∂S2 | 0.08068 |
| ∂V/∂σ1 | 1.00328 |
| ∂V/∂σ2 | 1.00429 |
| ∂V/∂r | 1.07524 |

#### A4. Swing Option — FD PDE
- **Script:** `swing_option_benchmarks.py`
- **Engine:** `FdSimpleBSSwingEngine` → ql-rust FD solver (if available)
- **Parameters:** S=30.0, K=30.0, σ=0.20, r=0.05, q=0.00, 31 exercise dates
- **Inputs (3):** spot, vol, rate

| Output | Reference Value |
|--------|----------------:|
| NPV | 47.1723 |
| ∂V/∂S | 18.3779 |
| ∂V/∂σ | 197.1571 |
| ∂V/∂r | 147.0830 |

---

### Group B — Credit Derivatives (First-Order)

#### B1. CDS — MidPoint Engine (100 Scenarios)
- **Script:** `cds_benchmarks.py`
- **Engine:** `MidPointCdsEngine` → ql-rust `midpoint_cds`
- **Parameters:** 2Y CDS, 150bp coupon, nominal=1M, recovery=0.50, RF=1%
- **Inputs (6):** 4 CDS spreads (3M, 6M, 1Y, 2Y) + recovery + risk-free rate
- **Batch:** 100 MC scenarios with bumped inputs

#### B2. ISDA CDS Engine (100 Scenarios)
- **Script:** `isda_cds_benchmarks.py`
- **Engine:** `IsdaCdsEngine` (ISDA standard model)
- **Parameters:** 10Y CDS, 10bp spread, recovery=40%, notional=10M
- **Inputs (20):** 6 deposit rates (1M–12M) + 14 swap rates (2Y–30Y)
- **Note:** Requires full ISDA-standard bootstrap. JIT not applicable (data-dependent branching).

#### B3. Risky Bond — Survival-Weighted Discounting
- **Script:** `risky_bond_benchmarks.py`
- **Engine:** `RiskyBondEngine` → ql-rust risky bond pricing
- **Parameters:** 5Y fixed-rate bond, 5% semiannual, notional=100
- **Inputs (14):** 9 OIS rates (1M–30Y) + 4 CDS spreads (1Y–5Y) + 1 recovery

| Output | Reference Value |
|--------|----------------:|
| NPV | 100.6120 |
| ∂V/∂(OIS 1M) | −1.1174 |
| ∂V/∂(OIS 5Y) | −417.8971 |
| ∂V/∂(CDS 1Y) | −0.4425 |
| ∂V/∂(CDS 5Y) | −439.3196 |
| ∂V/∂(Recovery) | −0.2004 |

---

### Group C — Rates Derivatives (First-Order)

#### C1. Vanilla IRS — Bootstrapped Curve (100 Scenarios)
- **Script:** `monte_carlo_irs_benchmarks.py`
- **Engine:** `DiscountingSwapEngine` + `PiecewiseFlatForward`
- **Parameters:** 5Y Payer IRS, Euribor3M curve
- **Inputs (17):** 1 deposit + 3 FRA + 8 futures + 5 swap rates
- **FD÷AAD ratio:** 212×

#### C2. OIS-Bootstrapped SOFR Swap (100 Scenarios)
- **Script:** `ois_bootstrapped_IRS_benchmarks.py`
- **Engine:** `DiscountingSwapEngine` + `PiecewiseLogLinearDiscount`
- **Parameters:** 5Y SOFR OIS, pay fixed, receive SOFR, $10M notional
- **Inputs (9):** OIS par rates at 1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 10Y, 30Y
- **FD÷AAD ratio:** 136×

#### C3. Callable Bond — HW Tree (100 Scenarios)
- **Script:** `monte_carlo_bond_benchmarks.py`
- **Engine:** `TreeCallableFixedRateBondEngine` (HW, 40 steps)
- **Parameters:** Callable fixed-rate bond, flat rate=4.65%, HW a=0.06, σ=0.20
- **Inputs (3):** rate, mean-reversion, volatility

#### C4. IR Cap — Black Engine
- **Script:** (from `run_more_benchmarks.py`, instrument E)
- **Engine:** `BlackCapFloorEngine` + bootstrapped `PiecewiseFlatForward`
- **Parameters:** 10Y Cap on Euribor3M, strike=5%, notional=1M
- **Inputs (18):** 17 rate curve inputs + 1 Black flat vol (20%)

| Output | Reference Value |
|--------|----------------:|
| NPV | 54534.5174 |
| ∂V/∂(Swap 10Y) | 4955428.4163 |
| ∂V/∂(Black Vol) | 287818.8501 |

#### C5. European Swaption — Jamshidian/HW
- **Script:** (from `run_more_benchmarks.py`, instrument F)
- **Engine:** `JamshidianSwaptionEngine` + `HullWhite`
- **Parameters:** European payer swaption (10Y into 5Y)
- **Inputs (3):** rate, HW mean-reversion, HW volatility

---

### Group D — Curve Jacobians

#### D1. CDS Spread Jacobian
- **Script:** `cds_spread_jacobian_benchmarks.py`
- **Instrument:** 5Y CDS, Protection Buyer, 100bp coupon, $10M notional
- **Inputs (4):** CDS spreads at 1Y, 2Y, 3Y, 5Y (50, 75, 100, 125 bp)
- **Outputs:**
  - ∂NPV/∂(CDS spread) — 4 sensitivities
  - Bootstrap Jacobian J = ∂h/∂s (4×4 lower-triangular)
  - Reverse Jacobian K = ∂s/∂h (4×4)
  - Round-trip verification: max |K×J − I| < 1.3e-10

| Tenor | ∂NPV/∂s (AAD) |
|-------|---------------:|
| 1Y | −37805.21 |
| 2Y | −70354.80 |
| 3Y | −160755.56 |
| 5Y | 44392831.43 |

#### D2. Hazard Rate Jacobian
- **Script:** `hazard_rate_jacobian_benchmarks.py`
- **Instrument:** Same 5Y CDS as D1
- **Outputs:**
  - ∂NPV/∂(hazard rate) — 4 sensitivities
  - Round-trip Jᵀ × ∂NPV/∂h = ∂NPV/∂s validation

| Pillar | ∂NPV/∂h (AAD) |
|--------|---------------:|
| 2025-12-22 | 6301013.91 |
| 2026-12-21 | 5396482.92 |
| 2027-12-20 | 5118347.71 |
| 2029-12-20 | 9502664.34 |

#### D3. Zero Rate Jacobian
- **Script:** `zero_rate_jacobian_benchmarks.py`
- **Instrument:** 5Y SOFR OIS, $10M notional
- **Inputs (9):** Zero rates at 9 tenors (derived from US Treasury par rates)
- **Outputs:**
  - ∂NPV/∂(zero rate) — 9 sensitivities
  - Bootstrap Jacobian J = ∂z/∂r (9×9 lower-triangular)
  - FD÷AAD ratio: 35,548×

| Pillar | ∂NPV/∂z (AAD) |
|--------|---------------:|
| 2026-04-02 | −109530.01 |
| 2027-03-02 | 353125.57 |
| 2028-03-02 | 682341.40 |
| 2029-03-02 | 1616187.48 |
| 2031-03-03 | 44105719.81 |

#### D4. Par (MM) Rate Jacobian
- **Script:** `mm_rate_jacobian_benchmarks.py`
- **Instrument:** Same 5Y SOFR OIS
- **Inputs (9):** OIS par rates at 9 tenors
- **Outputs:**
  - ∂NPV/∂(par rate) — 9 sensitivities
  - Reverse Jacobian K = ∂r/∂z (9×9)
  - Round-trip: max |K×J − I| < 4.66e-5

---

### Group E — Second-Order Sensitivities (Hessians)

All Hessians computed via FD-over-AAD: bump each input by h=1e-5,
re-run AAD backward sweep, compute ∂²V/∂xᵢ∂xⱼ ≈ (∂V/∂xⱼ(xᵢ+h) − ∂V/∂xⱼ(xᵢ−h)) / 2h.

#### E1. European Option Hessian (4×4)
- **Script:** `second_order_european.py`
- **Same instrument as A1**

| | S | q | σ | r |
|---|------:|------:|------:|------:|
| **S** | 0.2378 | −1.7692 | 2.3062 | 1.6736 |
| **q** | −1.7690 | 12.4511 | −16.2313 | −11.7781 |
| **σ** | 2.3062 | −16.2317 | 20.7436 | 15.0537 |
| **r** | 1.6736 | −11.7797 | 15.0543 | 11.1373 |

Analytic BSM validation: Gamma=0.2374, Vanna=2.3015, Volga=20.7070

#### E2. IRS Hessian (17×17)
- **Script:** `second_order_irs.py`
- **Same instrument as C1** (5Y IRS, fixed=4%)
- **NPV:** 11109.11
- **Dominant entry:** Swap 5Y × Swap 5Y = −12,684,044

#### E3. CDS Hessian (6×6)
- **Script:** `second_order_cds.py`
- **Parameters:** 2Y CDS, Protection Seller, nominal=1M, coupon=150bp
- **Inputs (6):** 4 CDS spreads + recovery + risk-free rate
- **NPV:** 41.09
- **Dominant cross-gamma:** CDS 2Y × Recovery = 124,815

#### E4. IR Cap Hessian (18×18)
- **Script:** `second_order_ir_cap.py`
- **Same instrument as C4** (10Y Cap, 18 inputs)
- **NPV:** 54534.52
- **Dominant entry:** Swap 7Y × Swap 7Y = 529,107,728
- **Vega-gamma (Vol²):** 36,121

#### E5. Risky Bond Hessian (14×14)
- **Script:** `second_order_risky_bond.py`
- **Same instrument as B3** (5Y risky bond, 14 inputs)
- **NPV:** 102.33
- **Block maxima:**
  - IR×IR: OIS 5Y × OIS 5Y = 777
  - CR×CR: CDS 5Y × CDS 5Y = 1,242
  - IR×CR: OIS 5Y × CDS 5Y = 2,884

---

## Implementation Plan

### Phase 1 — Analytic Engines (A1, A2, A4)

These have exact closed-form solutions already in ql-rust. Implement as
integration tests comparing against the reference values above.

| Test | ql-rust Function | Status |
|------|-----------------|--------|
| A1 European | `bs_european_generic` | Existing engine, need test |
| A2 BAW | `barone_adesi_whaley_generic` | Existing engine, need test |
| A2 Bjerksund | `bjerksund_stensland_generic` | Existing engine, need test |
| A2 FD-BS | `fd_black_scholes` | Existing engine, need test |
| A2 QD+ | `qd_plus_american` | Existing engine, need test |
| A4 Swing | FD swing engine | May need new engine |

### Phase 2 — Credit Engines (B1, B2, B3)

| Test | ql-rust Function | Status |
|------|-----------------|--------|
| B1 CDS MidPoint | `midpoint_cds` | Existing, need curve bootstrap |
| B2 ISDA CDS | Need ISDA engine | May need new engine |
| B3 Risky Bond | `RiskyBondEngine` | May need new engine |

### Phase 3 — Rates Engines (C1–C5)

| Test | ql-rust Function | Status |
|------|-----------------|--------|
| C1 IRS (bootstrapped) | `price_swap` + `PiecewiseYieldCurve` | Existing |
| C2 OIS SOFR | OIS swap + log-linear discount | May need OIS helpers |
| C3 Callable Bond HW | `tree_callable_bond` | Existing |
| C4 IR Cap Black | `BlackCapFloorEngine` | Existing |
| C5 Swaption Jamshidian | `hw_jamshidian_swaption` | Existing |

### Phase 4 — Curve Jacobians (D1–D4)

These test the AD system at the curve-building level — sensitivities of NPV
w.r.t. bootstrapping instrument quotes flowing through the solver.

| Test | Challenge |
|------|-----------|
| D1 CDS Spread Jacobian | AD through `PiecewiseFlatHazardRate` bootstrap |
| D2 Hazard Rate Jacobian | Direct `HazardRateCurve<T>` with AD |
| D3 Zero Rate Jacobian | Direct `ZeroCurve<T>` with AD |
| D4 Par Rate Jacobian | AD through `PiecewiseLinearZero` bootstrap |

### Phase 5 — Second-Order Hessians (E1–E5)

FD-over-AAD: for each input xᵢ, bump by ±h, run `AReal` backward sweep,
collect ∂V/∂xⱼ at both bump points, finite-difference to get ∂²V/∂xᵢ∂xⱼ.

| Test | Dimension | ql-rust Approach |
|------|----------:|------------------|
| E1 European | 4×4 | FD-over-`Dual` or FD-over-`AReal` |
| E2 IRS | 17×17 | FD-over-`AReal` through bootstrap |
| E3 CDS | 6×6 | FD-over-`AReal` through hazard curve |
| E4 IR Cap | 18×18 | FD-over-`AReal` through curve + Black vol |
| E5 Risky Bond | 14×14 | FD-over-`AReal` through IR + credit curves |

### Phase 6 — MC & Batch Scenarios (A3, B1 batch, B2 batch, C1–C3 batch)

| Test | Notes |
|------|-------|
| A3 Basket MC | 32768 quasi-random paths, seed=42. Tolerance ~1e-2 for MC noise. |
| B1 CDS 100 scen | Batch loop: bump inputs per scenario, re-price |
| B2 ISDA 100 scen | Full ISDA bootstrap per scenario |
| C1 IRS 100 scen | Curve rebuild + swap reprice per scenario |
| C2 OIS 100 scen | OIS bootstrap per scenario |
| C3 Bond 100 scen | HW tree rebuild per scenario |

---

## Test File Structure

```
crates/ql-rust/tests/
    test_verification_equity.rs       # A1–A4
    test_verification_credit.rs       # B1–B3
    test_verification_rates.rs        # C1–C5
    test_verification_jacobians.rs    # D1–D4
    test_verification_hessians.rs     # E1–E5
    test_verification_mc_batch.rs     # A3, batch scenarios
```

Each test function will:
1. Construct the instrument with identical parameters to the Python script
2. Price with the appropriate engine
3. Compare NPV against the reference value
4. Compute first-order Greeks via `Dual` / `DualVec<N>` / `AReal`
5. Compare Greeks against the reference values
6. For Hessians: run FD-over-AAD and compare the full matrix

---

## Gap Analysis vs Current ql-rust

| Capability | Current ql-rust | Needed for Verification |
|------------|----------------|------------------------|
| BSM European + Greeks | ✅ `bs_european_generic` | ✅ Ready |
| BAW American + Greeks | ✅ `barone_adesi_whaley_generic` | ✅ Ready |
| Bjerksund-Stensland | ✅ `bjerksund_stensland_generic` | ✅ Ready |
| QD+ American | ✅ `qd_plus_american` | ✅ Ready |
| FD-BS American | ✅ `fd_black_scholes` | ✅ Ready |
| MC Basket (correlated) | ✅ `mc_basket` | ✅ Ready (seed control needed) |
| CDS MidPoint | ✅ `midpoint_cds` | ✅ Ready |
| CDS hazard curve bootstrap | ✅ `PiecewiseFlatHazardRate` | ⚠️ Need AD-through-bootstrap |
| ISDA CDS Engine | ❌ Not implemented | 🔴 New engine needed |
| Risky Bond Engine | ❌ Not implemented | 🔴 New engine needed |
| Swing Option FD | ❌ Not implemented | 🔴 New engine needed |
| PiecewiseFlatForward bootstrap | ✅ `PiecewiseYieldCurve` | ⚠️ Need FRA/futures helpers |
| PiecewiseLogLinearDiscount | ❌ Only linear zero | 🔴 New interpolation mode |
| OIS swap pricing | ✅ `OvernightIndexedSwap` | ⚠️ Need OIS rate helper integration |
| IR Cap (Black) | ✅ `BlackCapFloorEngine` | ✅ Ready |
| Swaption Jamshidian | ✅ `hw_jamshidian_swaption` | ✅ Ready |
| Callable Bond HW Tree | ✅ `tree_callable_bond` | ✅ Ready |
| AD through curve bootstrap | ⚠️ Generic `Number` bootstrap partial | 🔴 Full AD tape through solver |
| FD-over-AAD Hessians | ⚠️ Demonstrated in tests | ✅ Pattern exists |
| Jacobian extraction (multi-sweep) | ❌ Not implemented | 🟡 New utility needed |

**Summary:** 11 of 21 benchmarks can be verified with existing engines immediately.
5 need minor extensions (helpers, interpolation modes). 5 need new engines or
significant work (ISDA CDS, Risky Bond, Swing, AD-through-bootstrap, log-linear discount).

---

## Priority Order

1. **Phase 1** (A1, A2) — immediate, validates core AD against known analytics
2. **Phase 2** (B1, B3) — validates credit; ISDA CDS (B2) deferred if complex
3. **Phase 3** (C1, C4, C5) — validates rates; C2/C3 after helpers built
4. **Phase 4** (D1–D4) — highest AD complexity; validates curve-level AD
5. **Phase 5** (E1–E5) — second-order; builds on Phase 1–4 infrastructure
6. **Phase 6** (batch) — performance validation; last priority
