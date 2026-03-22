//! Fuzz target for analytic Asian option engines.
//!
//! Tests geometric continuous/strike, Turnbull-Wakeman, and Levy
//! engines with bounded inputs.  All results must be finite and
//! non-negative.

#![no_main]

use libfuzzer_sys::fuzz_target;
use ql_instruments::OptionType;
use ql_pricingengines::{
    asian_geometric_continuous_avg_price, asian_geometric_continuous_avg_strike, asian_levy,
    asian_turnbull_wakeman,
};

fuzz_target!(|data: (f64, f64, f64, f64, f64, f64, f64, f64, bool)| {
    let (spot_raw, strike_raw, r_raw, q_raw, vol_raw, t_raw, t0_raw, a_raw, is_call) = data;

    let spot = (spot_raw.abs() % 500.0) + 1.0;
    let strike = (strike_raw.abs() % 500.0) + 1.0;
    let r = (r_raw % 0.5).clamp(-0.05, 0.3);
    let q = (q_raw % 0.5).clamp(-0.05, 0.2);
    let vol = (vol_raw.abs() % 3.0).max(0.01);
    let t = (t_raw.abs() % 10.0).max(0.01);
    let t0 = (t0_raw.abs() % t).max(0.0); // elapsed <= total
    let a = (a_raw.abs() % 1000.0).max(0.01); // running average

    if ![spot, strike, r, q, vol, t, t0, a].iter().all(|x| x.is_finite()) {
        return;
    }

    let opt = if is_call { OptionType::Call } else { OptionType::Put };

    // --- Geometric continuous avg price ---
    let r1 = std::panic::catch_unwind(|| {
        asian_geometric_continuous_avg_price(spot, strike, r, q, vol, t, opt)
    });
    if let Ok(res) = r1 {
        assert!(
            res.npv.is_finite() && res.npv >= 0.0,
            "Geo cont avg-price invalid: {}",
            res.npv
        );
    }

    // --- Geometric continuous avg strike ---
    let r2 = std::panic::catch_unwind(|| {
        asian_geometric_continuous_avg_strike(spot, r, q, vol, t, opt)
    });
    if let Ok(res) = r2 {
        assert!(
            res.npv.is_finite() && res.npv >= 0.0,
            "Geo cont avg-strike invalid: {}",
            res.npv
        );
    }

    // --- Turnbull-Wakeman ---
    let r3 = std::panic::catch_unwind(|| {
        asian_turnbull_wakeman(spot, strike, r, q, vol, t, t0, a, opt)
    });
    if let Ok(res) = r3 {
        assert!(
            res.npv.is_finite() && res.npv >= 0.0,
            "Turnbull-Wakeman invalid: {}",
            res.npv
        );
    }

    // --- Levy ---
    let r4 = std::panic::catch_unwind(|| {
        asian_levy(spot, strike, r, q, vol, t, opt)
    });
    if let Ok(res) = r4 {
        assert!(
            res.npv.is_finite() && res.npv >= 0.0,
            "Levy invalid: {}",
            res.npv
        );
    }
});
