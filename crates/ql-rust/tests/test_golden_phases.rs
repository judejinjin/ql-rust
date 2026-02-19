//! Golden cross-validation tests — Phases 16–24 coverage.
//!
//! Tests short-rate models, swaptions, FD Heston, credit models, LMM,
//! and advanced cashflows against known reference values.

use approx::assert_abs_diff_eq;
use serde_json::Value;

fn load_golden(name: &str) -> Value {
    let path = format!("tests/data/{name}");
    let data = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
    serde_json::from_str(&data).expect("parse golden json")
}

// ===========================================================================
// Short-Rate Models (Phase 16)
// ===========================================================================

mod short_rate {
    use super::*;
    use ql_models::{CIRModel, VasicekModel};
    use ql_pricingengines::hw_jamshidian_swaption;

    fn g() -> Value {
        load_golden("golden_shortrate.json")
    }

    #[test]
    fn vasicek_bond_1y() {
        let tc = &g()["test_cases"]["vasicek_bond_1y"];
        let m = &tc["model"];
        let model = VasicekModel::new(
            m["kappa"].as_f64().unwrap(),
            m["theta"].as_f64().unwrap(),
            m["sigma"].as_f64().unwrap(),
            m["r0"].as_f64().unwrap(),
        );
        let price = model.bond_price(tc["maturity"].as_f64().unwrap());
        let lo = tc["expected_price"]["min"].as_f64().unwrap();
        let hi = tc["expected_price"]["max"].as_f64().unwrap();
        assert!(
            price >= lo && price <= hi,
            "Vasicek bond 1Y price {price:.6} not in [{lo}, {hi}]"
        );
    }

    #[test]
    fn vasicek_bond_5y() {
        let tc = &g()["test_cases"]["vasicek_bond_5y"];
        let m = &tc["model"];
        let model = VasicekModel::new(
            m["kappa"].as_f64().unwrap(),
            m["theta"].as_f64().unwrap(),
            m["sigma"].as_f64().unwrap(),
            m["r0"].as_f64().unwrap(),
        );
        let price = model.bond_price(tc["maturity"].as_f64().unwrap());
        let lo = tc["expected_price"]["min"].as_f64().unwrap();
        let hi = tc["expected_price"]["max"].as_f64().unwrap();
        assert!(
            price >= lo && price <= hi,
            "Vasicek bond 5Y price {price:.6} not in [{lo}, {hi}]"
        );
    }

    #[test]
    fn cir_bond_1y() {
        let tc = &g()["test_cases"]["cir_bond_1y"];
        let m = &tc["model"];
        let model = CIRModel::new(
            m["kappa"].as_f64().unwrap(),
            m["theta"].as_f64().unwrap(),
            m["sigma"].as_f64().unwrap(),
            m["r0"].as_f64().unwrap(),
        );
        let price = model.bond_price(tc["maturity"].as_f64().unwrap());
        let lo = tc["expected_price"]["min"].as_f64().unwrap();
        let hi = tc["expected_price"]["max"].as_f64().unwrap();
        assert!(
            price >= lo && price <= hi,
            "CIR bond 1Y price {price:.6} not in [{lo}, {hi}]"
        );
    }

    #[test]
    fn vasicek_yield_positive() {
        let model = VasicekModel::new(0.3, 0.05, 0.02, 0.05);
        for t in [0.5, 1.0, 2.0, 5.0, 10.0] {
            let y = model.yield_rate(t);
            assert!(y > 0.0, "Vasicek yield at t={t} should be positive: {y}");
        }
    }

    #[test]
    fn cir_feller_condition() {
        // 2*kappa*theta > sigma^2 for Feller condition
        let model = CIRModel::new(0.5, 0.05, 0.1, 0.05);
        assert!(
            model.feller_satisfied(),
            "2*0.5*0.05=0.05 > 0.01=0.1^2 → Feller satisfied"
        );
    }

    #[test]
    fn hw_jamshidian_swaption_positive() {
        let swap_tenors: Vec<f64> = (1..=10).map(|y| y as f64).collect();
        let dfs: Vec<f64> = swap_tenors.iter().map(|&t| (-0.04 * t).exp()).collect();

        let result = hw_jamshidian_swaption(
            0.03,
            0.01,
            1.0,
            &swap_tenors,
            0.04,
            &dfs,
            (-0.04_f64).exp(),
            1_000_000.0,
            true,
        );
        assert!(
            result.npv > 0.0,
            "HW Jamshidian swaption NPV should be positive: {}",
            result.npv
        );
    }

    #[test]
    fn vasicek_bond_option_positive() {
        let model = VasicekModel::new(0.3, 0.05, 0.02, 0.05);
        let call_price = model.bond_option(1.0, 5.0, 0.80, true);
        assert!(
            call_price > 0.0,
            "Vasicek bond option should be positive: {call_price}"
        );
    }
}

// ===========================================================================
// FD & Advanced Numerical Methods (Phase 19)
// ===========================================================================

mod fd_advanced {
    use super::*;
    use ql_methods::fd_heston_solve;
    use ql_models::{BatesModel, HestonModel};
    use ql_pricingengines::{bates_price, heston_price};

    fn g() -> Value {
        load_golden("golden_fd_advanced.json")
    }

    #[test]
    fn fd_heston_european_call() {
        let tc = &g()["test_cases"]["fd_heston_european_call"];
        let result = fd_heston_solve(
            tc["spot"].as_f64().unwrap(),
            tc["strike"].as_f64().unwrap(),
            tc["r"].as_f64().unwrap(),
            tc["q"].as_f64().unwrap(),
            tc["v0"].as_f64().unwrap(),
            tc["kappa"].as_f64().unwrap(),
            tc["theta"].as_f64().unwrap(),
            tc["sigma"].as_f64().unwrap(),
            tc["rho"].as_f64().unwrap(),
            tc["time_to_expiry"].as_f64().unwrap(),
            true,
            false,
            tc["ns"].as_u64().unwrap() as usize,
            tc["nv"].as_u64().unwrap() as usize,
            tc["nt"].as_u64().unwrap() as usize,
        );
        let lo = tc["expected_npv"]["min"].as_f64().unwrap();
        let hi = tc["expected_npv"]["max"].as_f64().unwrap();
        assert!(
            result.price >= lo && result.price <= hi,
            "FD Heston call {:.4} not in [{lo}, {hi}]",
            result.price
        );
    }

    #[test]
    fn fd_heston_vs_analytic() {
        // FD Heston should agree with analytic Heston within a few percent
        let model = HestonModel::new(100.0, 0.05, 0.0, 0.04, 1.5, 0.04, 0.3, -0.7);
        let analytic = heston_price(&model, 100.0, 1.0, true);
        let fd = fd_heston_solve(100.0, 100.0, 0.05, 0.0, 0.04, 1.5, 0.04, 0.3, -0.7, 1.0, true, false, 80, 40, 80);
        assert_abs_diff_eq!(fd.price, analytic.npv, epsilon = 1.0);
    }

    #[test]
    fn fd_bs_american_put() {
        let tc = &g()["test_cases"]["fd_bs_american_put"];
        let result = ql_methods::fd_black_scholes(
            tc["spot"].as_f64().unwrap(),
            tc["strike"].as_f64().unwrap(),
            tc["r"].as_f64().unwrap(),
            tc["q"].as_f64().unwrap(),
            tc["vol"].as_f64().unwrap(),
            tc["time_to_expiry"].as_f64().unwrap(),
            false,
            true,
            tc["ns"].as_u64().unwrap() as usize,
            tc["nt"].as_u64().unwrap() as usize,
        );
        let lo = tc["expected_npv"]["min"].as_f64().unwrap();
        let hi = tc["expected_npv"]["max"].as_f64().unwrap();
        assert!(
            result.npv >= lo && result.npv <= hi,
            "FD BS American put {:.4} not in [{lo}, {hi}]",
            result.npv
        );
    }

    #[test]
    fn merton_jd_call() {
        let tc = &g()["test_cases"]["merton_jd_call"];
        let result = ql_pricingengines::merton_jump_diffusion(
            tc["spot"].as_f64().unwrap(),
            tc["strike"].as_f64().unwrap(),
            tc["r"].as_f64().unwrap(),
            tc["q"].as_f64().unwrap(),
            tc["vol"].as_f64().unwrap(),
            tc["time_to_expiry"].as_f64().unwrap(),
            tc["lambda"].as_f64().unwrap(),
            tc["nu"].as_f64().unwrap(),
            tc["delta"].as_f64().unwrap(),
            true,
        );
        let price = result.npv;
        let lo = tc["expected_npv"]["min"].as_f64().unwrap();
        let hi = tc["expected_npv"]["max"].as_f64().unwrap();
        assert!(
            price >= lo && price <= hi,
            "Merton JD call {price:.4} not in [{lo}, {hi}]"
        );
    }

    #[test]
    fn bates_reduces_to_heston() {
        let tc = &g()["test_cases"]["bates_vs_heston_no_jumps"];
        let heston_model = HestonModel::new(100.0, 0.05, 0.0, 0.04, 1.5, 0.04, 0.3, -0.7);
        let bates_model = BatesModel::new(100.0, 0.05, 0.0, 0.04, 1.5, 0.04, 0.3, -0.7, 0.0, 0.0, 0.01);

        let h = heston_price(&heston_model, 100.0, 1.0, true);
        let b = bates_price(&bates_model, 100.0, 1.0, true);
        let tol = tc["tolerance"].as_f64().unwrap();
        assert_abs_diff_eq!(h.npv, b.npv, epsilon = tol);
    }
}

// ===========================================================================
// Credit Models (Phase 21)
// ===========================================================================

mod credit_models {
    use super::*;
    use ql_pricingengines::{cds_option_black, GaussianCopulaLHP};

    fn g() -> Value {
        load_golden("golden_credit.json")
    }

    #[test]
    fn gaussian_copula_cdo_equity() {
        let tc = &g()["test_cases"]["gaussian_copula_cdo_equity"];
        let default_prob = 1.0 - (-tc["hazard_rate"].as_f64().unwrap() * tc["maturity"].as_f64().unwrap()).exp();
        let copula = GaussianCopulaLHP::new(
            tc["num_names"].as_u64().unwrap() as usize,
            tc["correlation"].as_f64().unwrap(),
            tc["recovery"].as_f64().unwrap(),
            default_prob,
        );
        let loss = copula.tranche_expected_loss(
            tc["attachment"].as_f64().unwrap(),
            tc["detachment"].as_f64().unwrap(),
        );
        assert!(
            loss >= 0.0 && loss <= 1.0,
            "Equity tranche normalized loss {loss:.6} should be in [0, 1]"
        );
    }

    #[test]
    fn gaussian_copula_cdo_mezzanine() {
        let tc = &g()["test_cases"]["gaussian_copula_cdo_mezzanine"];
        let default_prob = 1.0 - (-tc["hazard_rate"].as_f64().unwrap() * tc["maturity"].as_f64().unwrap()).exp();
        let copula = GaussianCopulaLHP::new(
            tc["num_names"].as_u64().unwrap() as usize,
            tc["correlation"].as_f64().unwrap(),
            tc["recovery"].as_f64().unwrap(),
            default_prob,
        );
        let mezz_loss = copula.tranche_expected_loss(
            tc["attachment"].as_f64().unwrap(),
            tc["detachment"].as_f64().unwrap(),
        );
        // Mezzanine tranche normalized loss should be in [0, 1]
        assert!(
            mezz_loss >= 0.0 && mezz_loss <= 1.0,
            "Mezzanine normalized loss should be in [0, 1]: {mezz_loss:.6}"
        );
        // Mezzanine should have less expected loss than equity (subordination)
        let equity_loss = copula.tranche_expected_loss(0.0, tc["attachment"].as_f64().unwrap());
        assert!(
            mezz_loss <= equity_loss + 0.01,
            "Mezzanine loss {mezz_loss:.4} should be ≤ equity loss {equity_loss:.4} (subordination)"
        );
    }

    #[test]
    fn cds_option_black_positive() {
        let tc = &g()["test_cases"]["cds_option_black"];
        let premium = cds_option_black(
            tc["fwd_spread"].as_f64().unwrap(),
            tc["strike_spread"].as_f64().unwrap(),
            tc["vol"].as_f64().unwrap(),
            tc["time_to_expiry"].as_f64().unwrap(),
            tc["annuity"].as_f64().unwrap(),
            true,
        );
        assert!(
            premium > 0.0,
            "CDS option premium should be positive: {premium:.4}"
        );
    }

    #[test]
    fn cds_option_parity() {
        // Payer - Receiver = (fwd - strike) * annuity
        let fwd = 0.01;
        let strike = 0.012;
        let vol = 0.40;
        let t = 1.0;
        let annuity = 4.5;

        let payer = cds_option_black(fwd, strike, vol, t, annuity, true);
        let receiver = cds_option_black(fwd, strike, vol, t, annuity, false);
        let parity = payer - receiver;
        let expected = (fwd - strike) * annuity;
        assert_abs_diff_eq!(parity, expected, epsilon = 1e-6);
    }
}

// ===========================================================================
// LMM (Phase 20)
// ===========================================================================

mod lmm_tests {
    use ql_models::{lmm_cap_price, lmm_swaption_price, LmmConfig};

    #[test]
    fn lmm_cap_positive() {
        let config = LmmConfig::flat(10, 0.04, 0.5, 0.15, 0.5);
        let result = lmm_cap_price(&config, 0.05, 10_000, 42);
        assert!(
            result.price > 0.0,
            "LMM cap price should be positive: {:.4}",
            result.price
        );
    }

    #[test]
    fn lmm_swaption_positive() {
        let config = LmmConfig::flat(10, 0.04, 0.5, 0.15, 0.5);
        let result = lmm_swaption_price(&config, 2, 10, 0.04, 10_000, true, 42);
        assert!(
            result.price >= 0.0,
            "LMM swaption price should be non-negative: {:.4}",
            result.price
        );
    }

    #[test]
    fn lmm_cap_monotone_in_strike() {
        let config = LmmConfig::flat(10, 0.04, 0.5, 0.15, 0.5);
        let cap_low = lmm_cap_price(&config, 0.03, 10_000, 42);
        let cap_high = lmm_cap_price(&config, 0.06, 10_000, 42);
        assert!(
            cap_low.price >= cap_high.price - 3.0 * cap_high.std_error,
            "Lower strike cap {:.4} should cost more than higher strike {:.4}",
            cap_low.price,
            cap_high.price
        );
    }
}

// ===========================================================================
// Cash Flow Analytics (Phase 23)
// ===========================================================================

mod cashflow_analytics {
    use approx::assert_abs_diff_eq;
    use ql_cashflows::{cms_caplet_price, cms_convexity_adjustment, fixed_leg};
    use ql_time::{Date, DayCounter, Month, Schedule};

    #[test]
    fn cms_convexity_adjustment_nonzero() {
        let adj = cms_convexity_adjustment(0.04, 0.20, 1.0, 5.0, 0.0);
        // CMS convexity adjustment is nonzero when vol > 0
        assert!(
            adj.abs() > 1e-6,
            "CMS convexity adjustment should be nonzero: {adj:.6}"
        );
    }

    #[test]
    fn cms_convexity_adjustment_magnitude_increases_with_vol() {
        let adj_low = cms_convexity_adjustment(0.04, 0.10, 1.0, 5.0, 0.0);
        let adj_high = cms_convexity_adjustment(0.04, 0.30, 1.0, 5.0, 0.0);
        assert!(
            adj_high.abs() > adj_low.abs(),
            "Higher vol should give larger magnitude CMS adjustment: low={adj_low:.6} high={adj_high:.6}"
        );
    }

    #[test]
    fn cms_caplet_positive() {
        let price = cms_caplet_price(0.04, 0.05, 0.20, 1.0, 1.0, 0.95, 1_000_000.0, 0.5);
        assert!(
            price > 0.0,
            "CMS caplet price should be positive: {price:.2}"
        );
    }

    #[test]
    fn fixed_leg_npv_consistency() {
        let today = Date::from_ymd(2025, Month::January, 15);
        let schedule = Schedule::from_dates(vec![
            today,
            today + 182,
            today + 365,
            today + 547,
            today + 730,
        ]);
        let dc = DayCounter::Actual365Fixed;
        let notionals = [1_000_000.0; 4];
        let rates = [0.05; 4];
        let leg = fixed_leg(&schedule, &notionals, &rates, dc);

        // Total coupons should roughly equal notional * rate * T
        let total: f64 = leg.iter().map(|cf| cf.amount()).sum();
        assert_abs_diff_eq!(total, 100_000.0, epsilon = 5000.0);
    }
}

// ===========================================================================
// Advanced Yield Curves (Phase 24)
// ===========================================================================

mod advanced_curves {
    use approx::assert_abs_diff_eq;
    use ql_termstructures::{NelsonSiegelFitting, SvenssonFitting};

    #[test]
    fn svensson_6param_fit() {
        let maturities = vec![0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0];
        let yields = vec![
            0.030, 0.032, 0.035, 0.040, 0.043, 0.048, 0.050, 0.052, 0.053, 0.054, 0.055,
        ];
        let fit = SvenssonFitting::fit(&maturities, &yields).expect("Svensson fit");
        // Check that fitted yields are close to input at 5Y
        let y5 = fit.zero_rate(5.0);
        assert_abs_diff_eq!(y5, 0.048, epsilon = 0.005);
    }

    #[test]
    fn nelson_siegel_vs_svensson() {
        let maturities = vec![0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0];
        let yields = vec![
            0.030, 0.032, 0.035, 0.040, 0.043, 0.048, 0.050, 0.052, 0.053, 0.054, 0.055,
        ];
        let ns = NelsonSiegelFitting::fit(&maturities, &yields).expect("NS fit");
        let sv = SvenssonFitting::fit(&maturities, &yields).expect("Svensson fit");

        // Both should give similar yields at 5Y
        let ns5 = ns.zero_rate(5.0);
        let sv5 = sv.zero_rate(5.0);
        assert_abs_diff_eq!(ns5, sv5, epsilon = 0.005);
    }

    #[test]
    fn nelson_siegel_monotone_long_end() {
        let maturities = vec![0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0];
        let yields = vec![0.030, 0.032, 0.035, 0.040, 0.048, 0.052, 0.055];
        let ns = NelsonSiegelFitting::fit(&maturities, &yields).expect("NS fit");

        // NS asymptotes to beta0, so 30Y should be close to beta0
        let y30 = ns.zero_rate(30.0);
        assert!(
            y30 > 0.04 && y30 < 0.07,
            "NS 30Y yield should be reasonable: {y30:.4}"
        );
    }
}
