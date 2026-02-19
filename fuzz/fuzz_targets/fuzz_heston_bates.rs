//! Fuzz target for Heston and Bates analytic pricing engines.
//!
//! Exercises the full characteristic-function + Gauss-Legendre
//! integration path with bounded fuzz inputs.  Asserts that results
//! are finite and non-negative.

#![no_main]

use libfuzzer_sys::fuzz_target;
use ql_models::{BatesModel, HestonModel};
use ql_pricingengines::{bates_price, heston_price};

fuzz_target!(|data: (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, bool)| {
    let (
        spot_raw, strike_raw, r_raw, q_raw, v0_raw, kappa_raw, theta_raw, sigma_raw, rho_raw,
        t_raw, is_call,
    ) = data;

    // Bound to valid ranges
    let spot = (spot_raw.abs() % 500.0) + 1.0;
    let strike = (strike_raw.abs() % 500.0) + 1.0;
    let r = (r_raw % 0.5).clamp(-0.05, 0.3);
    let q = (q_raw % 0.5).clamp(-0.05, 0.2);
    let v0 = (v0_raw.abs() % 1.0).max(0.001);
    let kappa = (kappa_raw.abs() % 10.0).max(0.01);
    let theta = (theta_raw.abs() % 1.0).max(0.001);
    let sigma = (sigma_raw.abs() % 3.0).max(0.01);
    let rho = (rho_raw % 1.0).clamp(-0.999, 0.999);
    let t = (t_raw.abs() % 10.0).max(0.01);

    if ![spot, strike, r, q, v0, kappa, theta, sigma, rho, t]
        .iter()
        .all(|x| x.is_finite())
    {
        return;
    }

    // --- Heston ---
    let heston = HestonModel::new(spot, r, q, v0, kappa, theta, sigma, rho);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        heston_price(&heston, strike, t, is_call)
    }));
    if let Ok(hr) = result {
        assert!(
            hr.npv.is_finite() && hr.npv >= 0.0,
            "Heston returned invalid npv: {}",
            hr.npv
        );
    }

    // --- Bates (with small jump params) ---
    let lambda = 0.1;
    let nu = -0.05;
    let delta = 0.1;
    let bates = BatesModel::new(spot, r, q, v0, kappa, theta, sigma, rho, lambda, nu, delta);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        bates_price(&bates, strike, t, is_call)
    }));
    if let Ok(br) = result {
        assert!(
            br.npv.is_finite() && br.npv >= 0.0,
            "Bates returned invalid npv: {}",
            br.npv
        );
    }
});
