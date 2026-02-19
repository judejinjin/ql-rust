//! Fuzz target for serde deserialization of term-structure types.
//!
//! Feeds arbitrary bytes as JSON to `serde_json::from_slice` for
//! FlatForward, DiscountCurve, ZeroCurve, and SabrSmileSection.
//! Ensures no panics on malformed input and consistency on valid input.

#![no_main]

use libfuzzer_sys::fuzz_target;
use ql_termstructures::{DiscountCurve, FlatForward, SabrSmileSection, ZeroCurve};

fuzz_target!(|data: &[u8]| {
    // --- FlatForward ---
    if let Ok(curve) = serde_json::from_slice::<FlatForward>(data) {
        let json = serde_json::to_vec(&curve).expect("serialize should not fail");
        let _curve2: FlatForward =
            serde_json::from_slice(&json).expect("round-trip deserialize should not fail");
    }

    // --- DiscountCurve ---
    if let Ok(curve) = serde_json::from_slice::<DiscountCurve>(data) {
        let json = serde_json::to_vec(&curve).expect("serialize should not fail");
        let _curve2: DiscountCurve =
            serde_json::from_slice(&json).expect("round-trip deserialize should not fail");
    }

    // --- ZeroCurve ---
    if let Ok(curve) = serde_json::from_slice::<ZeroCurve>(data) {
        let json = serde_json::to_vec(&curve).expect("serialize should not fail");
        let _curve2: ZeroCurve =
            serde_json::from_slice(&json).expect("round-trip deserialize should not fail");
    }

    // --- SabrSmileSection ---
    if let Ok(smile) = serde_json::from_slice::<SabrSmileSection>(data) {
        let json = serde_json::to_vec(&smile).expect("serialize should not fail");
        let _smile2: SabrSmileSection =
            serde_json::from_slice(&json).expect("round-trip deserialize should not fail");
    }
});
