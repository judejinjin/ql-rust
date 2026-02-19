//! Fuzz target for serde deserialization of instrument types.
//!
//! Feeds arbitrary bytes as JSON to `serde_json::from_slice` for
//! VanillaOption, BarrierOption, and AsianOption.  Ensures no panics
//! on malformed input and round-trip consistency on valid input.

#![no_main]

use libfuzzer_sys::fuzz_target;
use ql_instruments::{AsianOption, BarrierOption, VanillaOption};

fuzz_target!(|data: &[u8]| {
    // --- VanillaOption ---
    if let Ok(opt) = serde_json::from_slice::<VanillaOption>(data) {
        // Round-trip: serialize then deserialize again
        let json = serde_json::to_vec(&opt).expect("serialize should not fail");
        let opt2: VanillaOption =
            serde_json::from_slice(&json).expect("round-trip deserialize should not fail");
        // Strike must survive
        assert_eq!(
            format!("{:?}", opt.payoff),
            format!("{:?}", opt2.payoff),
            "VanillaOption round-trip mismatch"
        );
    }

    // --- BarrierOption ---
    if let Ok(opt) = serde_json::from_slice::<BarrierOption>(data) {
        let json = serde_json::to_vec(&opt).expect("serialize should not fail");
        let opt2: BarrierOption =
            serde_json::from_slice(&json).expect("round-trip deserialize should not fail");
        assert!(
            (opt.barrier - opt2.barrier).abs() < 1e-15,
            "BarrierOption barrier round-trip mismatch"
        );
    }

    // --- AsianOption ---
    if let Ok(opt) = serde_json::from_slice::<AsianOption>(data) {
        let json = serde_json::to_vec(&opt).expect("serialize should not fail");
        let opt2: AsianOption =
            serde_json::from_slice(&json).expect("round-trip deserialize should not fail");
        assert_eq!(
            format!("{:?}", opt.averaging_type),
            format!("{:?}", opt2.averaging_type),
            "AsianOption round-trip mismatch"
        );
    }
});
