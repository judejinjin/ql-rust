//! Fuzz target for exotic equity engines.
//!
//! Exercises chooser, forward-start, power, digital-barrier,
//! and double-barrier-knockout engines with bounded inputs.

#![no_main]

use libfuzzer_sys::fuzz_target;
use ql_instruments::OptionType;
use ql_pricingengines::{
    chooser_price, digital_barrier, double_barrier_knockout, forward_start_option, power_option,
    DigitalBarrierType,
};

fuzz_target!(
    |data: (f64, f64, f64, f64, f64, f64, f64, f64, f64, bool, u8)| {
        let (
            spot_raw, strike_raw, r_raw, q_raw, vol_raw, t1_raw, t2_raw, _barrier_lo_raw,
            _barrier_hi_raw, is_call, variant,
        ) = data;

        let spot = (spot_raw.abs() % 500.0) + 10.0;
        let strike = (strike_raw.abs() % 500.0) + 1.0;
        let r = (r_raw % 0.5).clamp(-0.05, 0.3);
        let q = (q_raw % 0.5).clamp(-0.05, 0.2);
        let vol = (vol_raw.abs() % 3.0).max(0.01);
        let t1 = (t1_raw.abs() % 5.0).max(0.01); // choose / reset time
        let t2_offset = (t2_raw.abs() % 5.0).max(0.01);
        let t2 = t1 + t2_offset; // expiry > choose/reset

        if ![spot, strike, r, q, vol, t1, t2].iter().all(|x| x.is_finite()) {
            return;
        }

        // --- Chooser ---
        let rc = std::panic::catch_unwind(|| chooser_price(spot, strike, r, q, vol, t1, t2));
        if let Ok(res) = rc {
            assert!(
                res.npv.is_finite() && res.npv >= 0.0,
                "Chooser invalid: {}",
                res.npv
            );
        }

        // --- Forward-start ---
        let alpha = (strike / spot).clamp(0.5, 2.0); // strike proportion
        let rf = std::panic::catch_unwind(|| {
            forward_start_option(spot, r, q, vol, t1, t2, alpha, is_call)
        });
        if let Ok(res) = rf {
            assert!(
                res.price.is_finite() && res.price >= 0.0,
                "Forward-start invalid: {}",
                res.price
            );
        }

        // --- Power option (exponent 1–3) ---
        let exponent = (variant % 3) as f64 + 1.0;
        let rp = std::panic::catch_unwind(|| {
            power_option(spot, strike, r, q, vol, t2, exponent, is_call)
        });
        if let Ok(res) = rp {
            assert!(res.price.is_finite(), "Power invalid: {}", res.price);
        }

        // --- Digital barrier ---
        let barrier = if is_call {
            spot * 1.2 // upper barrier for call
        } else {
            spot * 0.8 // lower barrier for put
        };
        if barrier.is_finite() && barrier > 0.0 {
            let dt = if variant % 2 == 0 {
                DigitalBarrierType::OneTouch
            } else {
                DigitalBarrierType::NoTouch
            };
            let rebate = 1.0;
            let rd = std::panic::catch_unwind(|| {
                digital_barrier(spot, barrier, rebate, r, q, vol, t2, dt, is_call)
            });
            if let Ok(res) = rd {
                assert!(res.price.is_finite(), "Digital barrier invalid: {}", res.price);
            }
        }

        // --- Double barrier knockout ---
        let lower = spot * 0.7;
        let upper = spot * 1.3;
        if lower.is_finite() && upper.is_finite() && lower > 0.0 && upper > lower && spot > lower && spot < upper {
            let opt = if is_call { OptionType::Call } else { OptionType::Put };
            let rdb = std::panic::catch_unwind(|| {
                double_barrier_knockout(spot, strike, r, q, vol, t2, lower, upper, opt, 20)
            });
            if let Ok(res) = rdb {
                assert!(
                    res.npv.is_finite() && res.npv >= 0.0,
                    "Double barrier KO invalid: {}",
                    res.npv
                );
            }
        }
    }
);
