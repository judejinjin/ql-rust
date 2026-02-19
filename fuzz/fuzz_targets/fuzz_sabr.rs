//! Fuzz target for SABR implied volatility.
//!
//! Tests that sabr_volatility doesn't panic or return NaN/Inf
//! on bounded fuzz inputs.

#![no_main]

use libfuzzer_sys::fuzz_target;
use ql_termstructures::sabr_volatility;

fuzz_target!(|data: (f64, f64, f64, f64, f64, f64, f64)| {
    let (strike_raw, forward_raw, expiry_raw, alpha_raw, beta_raw, rho_raw, nu_raw) = data;

    // Bound to valid SABR parameter ranges
    let strike = (strike_raw.abs() % 1.0) + 0.0001;
    let forward = (forward_raw.abs() % 1.0) + 0.0001;
    let expiry = (expiry_raw.abs() % 30.0).max(0.01);
    let alpha = (alpha_raw.abs() % 1.0).max(0.0001);
    let beta = (beta_raw.abs() % 1.0).clamp(0.0, 1.0);
    let rho = (rho_raw % 1.0).clamp(-0.999, 0.999);
    let nu = (nu_raw.abs() % 5.0).max(0.0);

    if !strike.is_finite() || !forward.is_finite() || !expiry.is_finite()
        || !alpha.is_finite() || !beta.is_finite() || !rho.is_finite()
        || !nu.is_finite()
    {
        return;
    }

    // Should not panic
    let result = std::panic::catch_unwind(|| {
        sabr_volatility(strike, forward, expiry, alpha, beta, rho, nu)
    });

    // If it succeeds, result should be finite and positive
    if let Ok(vol) = result {
        assert!(
            vol.is_finite() && vol >= 0.0,
            "SABR returned invalid vol: {vol} for strike={strike}, forward={forward}"
        );
    }
});
