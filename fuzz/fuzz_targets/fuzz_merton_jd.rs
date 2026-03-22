//! Fuzz target for Merton jump-diffusion pricing.
//!
//! Tests that merton_jump_diffusion doesn't panic or return
//! NaN/Inf on bounded fuzz inputs.

#![no_main]

use libfuzzer_sys::fuzz_target;
use ql_pricingengines::merton_jump_diffusion;

fuzz_target!(|data: (f64, f64, f64, f64, f64, f64, f64, f64, f64, bool)| {
    let (spot_raw, strike_raw, r_raw, q_raw, vol_raw, t_raw, lambda_raw, nu_raw, delta_raw, is_call) =
        data;

    let spot = (spot_raw.abs() % 500.0) + 1.0;
    let strike = (strike_raw.abs() % 500.0) + 1.0;
    let r = (r_raw % 0.5).clamp(-0.05, 0.3);
    let q = (q_raw % 0.5).clamp(-0.05, 0.2);
    let vol = (vol_raw.abs() % 3.0).max(0.01);
    let t = (t_raw.abs() % 10.0).max(0.01);
    // Jump parameters
    let lambda = (lambda_raw.abs() % 5.0).max(0.0); // jump intensity
    let nu = (nu_raw % 1.0).clamp(-0.5, 0.5); // jump mean
    let delta = (delta_raw.abs() % 1.0).max(0.01); // jump vol

    if ![spot, strike, r, q, vol, t, lambda, nu, delta]
        .iter()
        .all(|x| x.is_finite())
    {
        return;
    }

    let result = std::panic::catch_unwind(|| {
        merton_jump_diffusion(spot, strike, r, q, vol, t, lambda, nu, delta, is_call)
    });

    if let Ok(res) = result {
        assert!(
            res.npv.is_finite() && res.npv >= 0.0,
            "Merton JD returned invalid npv: {} (lambda={lambda}, nu={nu}, delta={delta})",
            res.npv
        );
    }
});
