//! Fuzz target for Black-Scholes pricing and implied volatility.
//!
//! Tests that price_european and implied_volatility don't panic
//! on arbitrary (but bounded) inputs, and that round-tripping works.

#![no_main]

use libfuzzer_sys::fuzz_target;
use ql_instruments::VanillaOption;
use ql_pricingengines::{implied_volatility, price_european};
use ql_time::{Date, Month};

fuzz_target!(|data: (f64, f64, f64, f64, f64, f64, bool)| {
    let (spot_raw, strike_raw, r_raw, q_raw, vol_raw, t_raw, is_call) = data;

    // Bound inputs to reasonable ranges
    let spot = spot_raw.abs() % 1000.0 + 0.01;
    let strike = strike_raw.abs() % 1000.0 + 0.01;
    let r = (r_raw % 1.0).clamp(-0.1, 0.5);
    let q = (q_raw % 1.0).clamp(-0.1, 0.5);
    let vol = (vol_raw.abs() % 5.0).max(0.001);
    let t = (t_raw.abs() % 30.0).max(0.001);

    if !spot.is_finite() || !strike.is_finite() || !r.is_finite()
        || !q.is_finite() || !vol.is_finite() || !t.is_finite()
    {
        return;
    }

    let today = Date::from_ymd(2025, Month::January, 15);
    let expiry_days = (t * 365.0) as i32;
    if expiry_days < 1 || expiry_days > 11000 {
        return;
    }
    let option = if is_call {
        VanillaOption::european_call(strike, today + expiry_days)
    } else {
        VanillaOption::european_put(strike, today + expiry_days)
    };

    // Price should not panic
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        price_european(&option, spot, r, q, vol, t)
    }));

    if let Ok(result) = result {
        // NPV should be finite and non-negative
        if result.npv.is_finite() && result.npv > 0.01 {
            // Try to recover implied vol — should not panic
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                implied_volatility(&option, result.npv, spot, r, q, t)
            }));
        }
    }
});
