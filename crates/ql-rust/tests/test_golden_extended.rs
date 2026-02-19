//! Extended golden cross-validation tests driven by JSON reference data.
//!
//! Each test loads a golden JSON file and asserts that ql-rust produces
//! results within the specified tolerance of QuantLib C++ reference values.

use serde_json::Value;

// ===========================================================================
// Helpers
// ===========================================================================

fn load_golden(name: &str) -> Value {
    let path = format!("tests/data/{name}");
    let data = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
    serde_json::from_str(&data).expect("parse golden json")
}

// ===========================================================================
// Heston golden tests
// ===========================================================================

mod golden_heston {
    use super::*;
    use ql_models::HestonModel;
    use ql_pricingengines::heston_price;

    fn case(name: &str) -> Value {
        let golden = load_golden("golden_heston.json");
        golden["test_cases"][name].clone()
    }

    #[test]
    fn heston_call_standard() {
        let c = case("heston_call_standard");
        let model = HestonModel::new(
            c["spot"].as_f64().unwrap(),
            c["r"].as_f64().unwrap(),
            c["q"].as_f64().unwrap(),
            c["v0"].as_f64().unwrap(),
            c["kappa"].as_f64().unwrap(),
            c["theta"].as_f64().unwrap(),
            c["sigma"].as_f64().unwrap(),
            c["rho"].as_f64().unwrap(),
        );
        let result = heston_price(
            &model,
            c["strike"].as_f64().unwrap(),
            c["time_to_expiry"].as_f64().unwrap(),
            c["is_call"].as_bool().unwrap(),
        );
        let expected = c["expected_npv"].as_f64().unwrap();
        let tol = c["tolerance"].as_f64().unwrap();
        assert!(
            (result.npv - expected).abs() < tol,
            "Heston call: {:.4} vs expected {:.4} (tol={tol})",
            result.npv,
            expected,
        );
    }

    #[test]
    fn heston_put_otm() {
        let c = case("heston_put_otm");
        let model = HestonModel::new(
            c["spot"].as_f64().unwrap(),
            c["r"].as_f64().unwrap(),
            c["q"].as_f64().unwrap(),
            c["v0"].as_f64().unwrap(),
            c["kappa"].as_f64().unwrap(),
            c["theta"].as_f64().unwrap(),
            c["sigma"].as_f64().unwrap(),
            c["rho"].as_f64().unwrap(),
        );
        let result = heston_price(
            &model,
            c["strike"].as_f64().unwrap(),
            c["time_to_expiry"].as_f64().unwrap(),
            c["is_call"].as_bool().unwrap(),
        );
        let expected = c["expected_npv"].as_f64().unwrap();
        let tol = c["tolerance"].as_f64().unwrap();
        assert!(
            (result.npv - expected).abs() < tol,
            "Heston OTM put: {:.4} vs expected {:.4} (tol={tol})",
            result.npv,
            expected,
        );
    }

    #[test]
    fn heston_high_vol_of_vol() {
        let c = case("heston_high_vol_of_vol");
        let model = HestonModel::new(
            c["spot"].as_f64().unwrap(),
            c["r"].as_f64().unwrap(),
            c["q"].as_f64().unwrap(),
            c["v0"].as_f64().unwrap(),
            c["kappa"].as_f64().unwrap(),
            c["theta"].as_f64().unwrap(),
            c["sigma"].as_f64().unwrap(),
            c["rho"].as_f64().unwrap(),
        );
        let result = heston_price(
            &model,
            c["strike"].as_f64().unwrap(),
            c["time_to_expiry"].as_f64().unwrap(),
            c["is_call"].as_bool().unwrap(),
        );
        let expected = c["expected_npv"].as_f64().unwrap();
        let tol = c["tolerance"].as_f64().unwrap();
        assert!(
            (result.npv - expected).abs() < tol,
            "Heston high σ: {:.4} vs expected {:.4} (tol={tol})",
            result.npv,
            expected,
        );
    }

    #[test]
    fn heston_short_dated() {
        let c = case("heston_short_dated");
        let model = HestonModel::new(
            c["spot"].as_f64().unwrap(),
            c["r"].as_f64().unwrap(),
            c["q"].as_f64().unwrap(),
            c["v0"].as_f64().unwrap(),
            c["kappa"].as_f64().unwrap(),
            c["theta"].as_f64().unwrap(),
            c["sigma"].as_f64().unwrap(),
            c["rho"].as_f64().unwrap(),
        );
        let result = heston_price(
            &model,
            c["strike"].as_f64().unwrap(),
            c["time_to_expiry"].as_f64().unwrap(),
            c["is_call"].as_bool().unwrap(),
        );
        let expected = c["expected_npv"].as_f64().unwrap();
        let tol = c["tolerance"].as_f64().unwrap();
        assert!(
            (result.npv - expected).abs() < tol,
            "Heston 1M: {:.4} vs expected {:.4} (tol={tol})",
            result.npv,
            expected,
        );
    }
}

// ===========================================================================
// American options golden tests
// ===========================================================================

mod golden_american {
    use super::*;
    use ql_instruments::VanillaOption;
    use ql_pricingengines::{
        barone_adesi_whaley, bjerksund_stensland, price_european, qd_plus_american,
    };
    use ql_time::{Date, Month};

    fn case(name: &str) -> Value {
        let golden = load_golden("golden_american.json");
        golden["test_cases"][name].clone()
    }

    #[test]
    fn baw_atm_put_short() {
        let c = case("baw_atm_put_short");
        let result = barone_adesi_whaley(
            c["spot"].as_f64().unwrap(),
            c["strike"].as_f64().unwrap(),
            c["r"].as_f64().unwrap(),
            c["q"].as_f64().unwrap(),
            c["vol"].as_f64().unwrap(),
            c["time_to_expiry"].as_f64().unwrap(),
            c["is_call"].as_bool().unwrap(),
        );
        let expected = c["expected_npv"].as_f64().unwrap();
        let tol = c["tolerance"].as_f64().unwrap();
        assert!(
            (result.npv - expected).abs() < tol,
            "BAW ATM put: {:.4} vs {:.4} (tol={tol})",
            result.npv,
            expected,
        );
    }

    #[test]
    fn baw_otm_call() {
        let c = case("baw_otm_call");
        let result = barone_adesi_whaley(
            c["spot"].as_f64().unwrap(),
            c["strike"].as_f64().unwrap(),
            c["r"].as_f64().unwrap(),
            c["q"].as_f64().unwrap(),
            c["vol"].as_f64().unwrap(),
            c["time_to_expiry"].as_f64().unwrap(),
            c["is_call"].as_bool().unwrap(),
        );
        let expected = c["expected_npv"].as_f64().unwrap();
        let tol = c["tolerance"].as_f64().unwrap();
        assert!(
            (result.npv - expected).abs() < tol,
            "BAW OTM call: {:.4} vs {:.4} (tol={tol})",
            result.npv,
            expected,
        );
    }

    #[test]
    fn qd_itm_put() {
        let c = case("qd_itm_put");
        let result = qd_plus_american(
            c["spot"].as_f64().unwrap(),
            c["strike"].as_f64().unwrap(),
            c["r"].as_f64().unwrap(),
            c["q"].as_f64().unwrap(),
            c["vol"].as_f64().unwrap(),
            c["time_to_expiry"].as_f64().unwrap(),
            c["is_call"].as_bool().unwrap(),
        );
        let expected = c["expected_npv"].as_f64().unwrap();
        let tol = c["tolerance"].as_f64().unwrap();
        assert!(
            (result.npv - expected).abs() < tol,
            "QD+ ITM put: {:.4} vs {:.4} (tol={tol})",
            result.npv,
            expected,
        );
    }

    #[test]
    fn bjs_vs_baw_atm_call() {
        let c = case("bjs_vs_baw_atm_call");
        let spot = c["spot"].as_f64().unwrap();
        let strike = c["strike"].as_f64().unwrap();
        let r = c["r"].as_f64().unwrap();
        let q = c["q"].as_f64().unwrap();
        let vol = c["vol"].as_f64().unwrap();
        let t = c["time_to_expiry"].as_f64().unwrap();
        let is_call = c["is_call"].as_bool().unwrap();
        let rel_tol = c["relative_tolerance"].as_f64().unwrap();

        let baw = barone_adesi_whaley(spot, strike, r, q, vol, t, is_call);
        let bjs = bjerksund_stensland(spot, strike, r, q, vol, t, is_call);
        let rel_err = ((baw.npv - bjs.npv) / baw.npv).abs();
        assert!(
            rel_err < rel_tol,
            "BAW {:.4} vs BJS {:.4} differ by {:.2}% (max {:.0}%)",
            baw.npv,
            bjs.npv,
            rel_err * 100.0,
            rel_tol * 100.0,
        );
    }

    #[test]
    fn american_european_bound() {
        let c = case("american_european_bound");
        let spot = c["spot"].as_f64().unwrap();
        let strike = c["strike"].as_f64().unwrap();
        let r = c["r"].as_f64().unwrap();
        let q = c["q"].as_f64().unwrap();
        let vol = c["vol"].as_f64().unwrap();
        let t = c["time_to_expiry"].as_f64().unwrap();
        let tol = c["tolerance"].as_f64().unwrap();

        let today = Date::from_ymd(2025, Month::January, 15);
        let option = VanillaOption::european_call(strike, today + (t * 365.0) as i32);
        let european = price_european(&option, spot, r, q, vol, t);

        // American call with no dividends should equal European
        let american = barone_adesi_whaley(spot, strike, r, q, vol, t, true);
        assert!(
            (american.npv - european.npv).abs() < tol,
            "American call (no divs) {:.4} should ≈ European {:.4}",
            american.npv,
            european.npv,
        );
    }
}

// ===========================================================================
// SABR / SVI golden tests
// ===========================================================================

mod golden_sabr_svi {
    use super::*;
    use ql_termstructures::{sabr_volatility, svi_calibrate, svi_volatility};

    fn case(name: &str) -> Value {
        let golden = load_golden("golden_sabr_svi.json");
        golden["test_cases"][name].clone()
    }

    #[test]
    fn sabr_atm_swaption() {
        let c = case("sabr_atm_swaption");
        let vol = sabr_volatility(
            c["strike"].as_f64().unwrap(),
            c["forward"].as_f64().unwrap(),
            c["expiry"].as_f64().unwrap(),
            c["alpha"].as_f64().unwrap(),
            c["beta"].as_f64().unwrap(),
            c["rho"].as_f64().unwrap(),
            c["nu"].as_f64().unwrap(),
        );
        let expected = c["expected_vol"].as_f64().unwrap();
        let tol = c["tolerance"].as_f64().unwrap();
        assert!(
            (vol - expected).abs() < tol,
            "SABR ATM vol: {:.6} vs {:.6} (tol={tol})",
            vol,
            expected,
        );
    }

    #[test]
    fn sabr_otm_call_in_range() {
        let c = case("sabr_otm_call");
        let vol = sabr_volatility(
            c["strike"].as_f64().unwrap(),
            c["forward"].as_f64().unwrap(),
            c["expiry"].as_f64().unwrap(),
            c["alpha"].as_f64().unwrap(),
            c["beta"].as_f64().unwrap(),
            c["rho"].as_f64().unwrap(),
            c["nu"].as_f64().unwrap(),
        );
        let lo = c["expected_vol_range_low"].as_f64().unwrap();
        let hi = c["expected_vol_range_high"].as_f64().unwrap();
        assert!(
            vol >= lo && vol <= hi,
            "SABR OTM vol: {:.6} not in [{lo}, {hi}]",
            vol,
        );
    }

    #[test]
    fn sabr_smile_symmetry() {
        let c = case("sabr_smile_symmetry");
        let forward = c["forward"].as_f64().unwrap();
        let expiry = c["expiry"].as_f64().unwrap();
        let alpha = c["alpha"].as_f64().unwrap();
        let beta = c["beta"].as_f64().unwrap();
        let rho = c["rho"].as_f64().unwrap();
        let nu = c["nu"].as_f64().unwrap();
        let offsets: Vec<f64> = c["strikes_offsets_bp"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let tol = c["tolerance"].as_f64().unwrap();

        // With rho=0, vol(F+Δ) ≈ vol(F-Δ)
        for &offset in &offsets {
            let k_plus = forward + offset * 0.0001;
            let k_minus = forward - offset * 0.0001;
            if k_plus > 0.0 && k_minus > 0.0 && offset.abs() > 1e-10 {
                let v_plus = sabr_volatility(k_plus, forward, expiry, alpha, beta, rho, nu);
                let v_minus = sabr_volatility(k_minus, forward, expiry, alpha, beta, rho, nu);
                assert!(
                    (v_plus - v_minus).abs() < tol,
                    "SABR symmetry at ±{offset}bp: {v_plus:.6} vs {v_minus:.6}",
                );
            }
        }
    }

    #[test]
    fn svi_calibration_roundtrip() {
        let c = case("svi_calibration_roundtrip");
        let strikes: Vec<f64> = c["strikes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let market_vols: Vec<f64> = c["market_vols"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let forward = c["forward"].as_f64().unwrap();
        let expiry = c["expiry"].as_f64().unwrap();
        let tol = c["tolerance"].as_f64().unwrap();

        let (a, b, rho, m, sigma) = svi_calibrate(&strikes, &market_vols, forward, expiry);

        for (&k, &mkt_vol) in strikes.iter().zip(market_vols.iter()) {
            let fitted_vol = svi_volatility(k, forward, expiry, a, b, rho, m, sigma);
            assert!(
                (fitted_vol - mkt_vol).abs() < tol,
                "SVI roundtrip at K={k}: fitted {fitted_vol:.4} vs market {mkt_vol:.4} (tol={tol}), params: a={a:.4} b={b:.4} rho={rho:.4} m={m:.4} sigma={sigma:.4}",
            );
        }
    }
}

// ===========================================================================
// Multi-asset golden tests
// ===========================================================================

mod golden_multi_asset {
    use super::*;
    use ql_pricingengines::{
        kirk_spread_call, margrabe_exchange, mc_basket, merton_jump_diffusion,
        BasketType,
    };

    fn case(name: &str) -> Value {
        let golden = load_golden("golden_multi_asset.json");
        golden["test_cases"][name].clone()
    }

    #[test]
    fn kirk_spread_atm() {
        let c = case("kirk_spread_atm");
        let result = kirk_spread_call(
            c["s1"].as_f64().unwrap(),
            c["s2"].as_f64().unwrap(),
            c["strike"].as_f64().unwrap(),
            c["r"].as_f64().unwrap(),
            c["q1"].as_f64().unwrap(),
            c["q2"].as_f64().unwrap(),
            c["vol1"].as_f64().unwrap(),
            c["vol2"].as_f64().unwrap(),
            c["rho"].as_f64().unwrap(),
            c["time_to_expiry"].as_f64().unwrap(),
        );
        let lo = c["expected_price_range_low"].as_f64().unwrap();
        let hi = c["expected_price_range_high"].as_f64().unwrap();
        assert!(
            result >= lo && result <= hi,
            "Kirk spread call: {result:.4} not in [{lo}, {hi}]",
        );
    }

    #[test]
    fn margrabe_exchange_positive() {
        let c = case("margrabe_exchange_symmetry");
        let result = margrabe_exchange(
            c["s1"].as_f64().unwrap(),
            c["s2"].as_f64().unwrap(),
            c["q1"].as_f64().unwrap(),
            c["q2"].as_f64().unwrap(),
            c["vol1"].as_f64().unwrap(),
            c["vol2"].as_f64().unwrap(),
            c["rho"].as_f64().unwrap(),
            c["time_to_expiry"].as_f64().unwrap(),
        );
        // With equal spots the exchange option should be positive
        assert!(result > 0.0, "Margrabe should be positive, got {result:.4}");
        assert!(result < 30.0, "Margrabe unreasonably large: {result:.4}");
    }

    #[test]
    fn basket_weighted_avg_call() {
        let c = case("basket_weighted_avg_call");
        let spots: Vec<f64> = c["spots"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let weights: Vec<f64> = c["weights"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let divs: Vec<f64> = c["dividends"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let vols: Vec<f64> = c["vols"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let corr: Vec<f64> = c["correlation_matrix"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let result = mc_basket(
            &spots,
            &weights,
            c["strike"].as_f64().unwrap(),
            c["r"].as_f64().unwrap(),
            &divs,
            &vols,
            &corr,
            c["time_to_expiry"].as_f64().unwrap(),
            true,
            BasketType::WeightedAverage,
            c["num_paths"].as_u64().unwrap() as usize,
            42,
        );
        let lo = c["expected_price_range_low"].as_f64().unwrap();
        let hi = c["expected_price_range_high"].as_f64().unwrap();
        assert!(
            result.npv >= lo && result.npv <= hi,
            "Basket call: {:.4} not in [{lo}, {hi}]",
            result.npv,
        );
    }

    #[test]
    fn merton_jd_no_jumps() {
        let c = case("merton_jd_no_jumps");
        let result = merton_jump_diffusion(
            c["spot"].as_f64().unwrap(),
            c["strike"].as_f64().unwrap(),
            c["r"].as_f64().unwrap(),
            c["q"].as_f64().unwrap(),
            c["vol"].as_f64().unwrap(),
            c["time_to_expiry"].as_f64().unwrap(),
            c["lambda"].as_f64().unwrap(),
            c["nu"].as_f64().unwrap(),
            c["delta"].as_f64().unwrap(),
            true,
        );
        let expected = c["expected_call_price"].as_f64().unwrap();
        let tol = c["tolerance"].as_f64().unwrap();
        assert!(
            (result.npv - expected).abs() < tol,
            "Merton JD (λ=0): {:.4} vs BS {:.4} (tol={tol})",
            result.npv,
            expected,
        );
    }

    #[test]
    fn merton_jd_with_jumps() {
        let c = case("merton_jd_with_jumps");
        let result = merton_jump_diffusion(
            c["spot"].as_f64().unwrap(),
            c["strike"].as_f64().unwrap(),
            c["r"].as_f64().unwrap(),
            c["q"].as_f64().unwrap(),
            c["vol"].as_f64().unwrap(),
            c["time_to_expiry"].as_f64().unwrap(),
            c["lambda"].as_f64().unwrap(),
            c["nu"].as_f64().unwrap(),
            c["delta"].as_f64().unwrap(),
            true,
        );
        let lo = c["expected_call_price_range_low"].as_f64().unwrap();
        let hi = c["expected_call_price_range_high"].as_f64().unwrap();
        assert!(
            result.npv >= lo && result.npv <= hi,
            "Merton JD (with jumps): {:.4} not in [{lo}, {hi}]",
            result.npv,
        );
    }
}
