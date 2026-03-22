//! Fuzz target for multi-asset option pricing engines.
//!
//! Exercises Margrabe exchange, Stulz max/min call, and quanto
//! European engines with bounded correlated inputs.

#![no_main]

use libfuzzer_sys::fuzz_target;
use ql_pricingengines::{margrabe_exchange, quanto_european, stulz_max_call, stulz_min_call};

fuzz_target!(
    |data: (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, bool)| {
        let (
            s1_raw, s2_raw, strike_raw, r_raw, q1_raw, q2_raw, vol1_raw, vol2_raw, rho_raw,
            t_raw, fx_raw, is_call,
        ) = data;

        let s1 = (s1_raw.abs() % 500.0) + 1.0;
        let s2 = (s2_raw.abs() % 500.0) + 1.0;
        let strike = (strike_raw.abs() % 500.0) + 1.0;
        let r = (r_raw % 0.5).clamp(-0.05, 0.3);
        let q1 = (q1_raw % 0.5).clamp(-0.05, 0.2);
        let q2 = (q2_raw % 0.5).clamp(-0.05, 0.2);
        let vol1 = (vol1_raw.abs() % 3.0).max(0.01);
        let vol2 = (vol2_raw.abs() % 3.0).max(0.01);
        let rho = (rho_raw % 1.0).clamp(-0.999, 0.999);
        let t = (t_raw.abs() % 10.0).max(0.01);
        let fx = (fx_raw.abs() % 10.0).max(0.01);

        if ![s1, s2, strike, r, q1, q2, vol1, vol2, rho, t, fx]
            .iter()
            .all(|x| x.is_finite())
        {
            return;
        }

        // --- Margrabe exchange ---
        let rm = std::panic::catch_unwind(|| {
            margrabe_exchange(s1, s2, q1, q2, vol1, vol2, rho, t)
        });
        if let Ok(price) = rm {
            assert!(
                price.is_finite() && price >= 0.0,
                "Margrabe invalid: {price}"
            );
        }

        // --- Stulz max-of-two call ---
        let rmax = std::panic::catch_unwind(|| {
            stulz_max_call(s1, s2, strike, r, q1, q2, vol1, vol2, rho, t)
        });
        if let Ok(price) = rmax {
            assert!(
                price.is_finite() && price >= 0.0,
                "Stulz max call invalid: {price}"
            );
        }

        // --- Stulz min-of-two call ---
        let rmin = std::panic::catch_unwind(|| {
            stulz_min_call(s1, s2, strike, r, q1, q2, vol1, vol2, rho, t)
        });
        if let Ok(price) = rmin {
            assert!(
                price.is_finite() && price >= 0.0,
                "Stulz min call invalid: {price}"
            );
        }

        // --- Quanto European ---
        let rq = std::panic::catch_unwind(|| {
            quanto_european(s1, strike, r, q1, vol1, vol2, rho, t, fx, is_call)
        });
        if let Ok(res) = rq {
            assert!(
                res.price.is_finite() && res.price >= 0.0,
                "Quanto invalid: {}",
                res.price
            );
        }
    }
);
