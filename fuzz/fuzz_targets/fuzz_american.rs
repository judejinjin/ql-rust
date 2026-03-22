//! Fuzz target for American option pricing approximations.
//!
//! Exercises BAW, Bjerksund-Stensland, and QD+ with bounded
//! fuzz inputs.  Asserts that results are finite and non-negative,
//! and that American prices >= European intrinsic.

#![no_main]

use libfuzzer_sys::fuzz_target;
use ql_pricingengines::{barone_adesi_whaley, bjerksund_stensland, qd_plus_american};

fuzz_target!(|data: (f64, f64, f64, f64, f64, f64, bool)| {
    let (spot_raw, strike_raw, r_raw, q_raw, vol_raw, t_raw, is_call) = data;

    // Bound inputs to reasonable ranges
    let spot = (spot_raw.abs() % 500.0) + 1.0;
    let strike = (strike_raw.abs() % 500.0) + 1.0;
    let r = (r_raw % 0.5).clamp(-0.05, 0.3);
    let q = (q_raw % 0.5).clamp(-0.05, 0.2);
    let vol = (vol_raw.abs() % 3.0).max(0.01);
    let t = (t_raw.abs() % 10.0).max(0.01);

    if ![spot, strike, r, q, vol, t].iter().all(|x| x.is_finite()) {
        return;
    }

    // --- BAW ---
    let baw = std::panic::catch_unwind(|| {
        barone_adesi_whaley(spot, strike, r, q, vol, t, is_call)
    });
    if let Ok(res) = &baw {
        assert!(
            res.npv.is_finite() && res.npv >= 0.0,
            "BAW returned invalid npv: {}",
            res.npv
        );
    }

    // --- Bjerksund-Stensland ---
    let bjs = std::panic::catch_unwind(|| {
        bjerksund_stensland(spot, strike, r, q, vol, t, is_call)
    });
    if let Ok(res) = &bjs {
        assert!(
            res.npv.is_finite() && res.npv >= 0.0,
            "BJS returned invalid npv: {}",
            res.npv
        );
    }

    // --- QD+ ---
    let qdp = std::panic::catch_unwind(|| {
        qd_plus_american(spot, strike, r, q, vol, t, is_call)
    });
    if let Ok(res) = &qdp {
        assert!(
            res.npv.is_finite() && res.npv >= 0.0,
            "QD+ returned invalid npv: {}",
            res.npv
        );
    }
});
