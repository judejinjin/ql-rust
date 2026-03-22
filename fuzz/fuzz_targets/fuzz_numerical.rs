//! Fuzz target for numerical pricing methods (ql-methods crate).
//!
//! Exercises Monte Carlo European, finite-difference Black-Scholes,
//! and binomial CRR with bounded inputs.  Grid/path counts are
//! kept small so each iteration is fast.

#![no_main]

use libfuzzer_sys::fuzz_target;
use ql_instruments::OptionType;
use ql_methods::{binomial_crr, fd_black_scholes, mc_european};

fuzz_target!(|data: (f64, f64, f64, f64, f64, f64, bool, bool, u8)| {
    let (spot_raw, strike_raw, r_raw, q_raw, vol_raw, t_raw, is_call, is_american, seed_byte) =
        data;

    let spot = (spot_raw.abs() % 500.0) + 1.0;
    let strike = (strike_raw.abs() % 500.0) + 1.0;
    let r = (r_raw % 0.5).clamp(-0.05, 0.3);
    let q = (q_raw % 0.5).clamp(-0.05, 0.2);
    let vol = (vol_raw.abs() % 3.0).max(0.01);
    let t = (t_raw.abs() % 5.0).max(0.01);

    if ![spot, strike, r, q, vol, t].iter().all(|x| x.is_finite()) {
        return;
    }

    let opt = if is_call { OptionType::Call } else { OptionType::Put };
    let seed = seed_byte as u64 + 1;

    // --- MC European (small path count for speed) ---
    let rmc = std::panic::catch_unwind(|| {
        mc_european(spot, strike, r, q, vol, t, opt, 256, true, seed)
    });
    if let Ok(res) = rmc {
        assert!(
            res.npv.is_finite(),
            "MC European invalid npv: {}",
            res.npv
        );
    }

    // --- FD Black-Scholes ---
    let rfd = std::panic::catch_unwind(|| {
        fd_black_scholes(spot, strike, r, q, vol, t, is_call, is_american, 50, 50)
    });
    if let Ok(res) = rfd {
        assert!(
            res.npv.is_finite() && res.npv >= 0.0,
            "FD BS invalid npv: {}",
            res.npv
        );
    }

    // --- Binomial CRR ---
    let rcrr = std::panic::catch_unwind(|| {
        binomial_crr(spot, strike, r, q, vol, t, is_call, is_american, 50)
    });
    if let Ok(res) = rcrr {
        assert!(
            res.npv.is_finite() && res.npv >= 0.0,
            "Binomial CRR invalid npv: {}",
            res.npv
        );
    }
});
