//! Fuzz target for interpolation construction and lookup.
//!
//! Tests LinearInterpolation and CubicSplineInterpolation with
//! fuzzed node points and query values.

#![no_main]

use libfuzzer_sys::fuzz_target;
use ql_math::interpolation::{CubicSplineInterpolation, Interpolation, LinearInterpolation};

fuzz_target!(|data: (Vec<f64>, f64)| {
    let (mut raw_points, query) = data;

    // Need at least 2 points for interpolation
    if raw_points.len() < 4 {
        return;
    }

    // Truncate to reasonable size
    if raw_points.len() > 200 {
        raw_points.truncate(200);
    }

    // Filter out NaN/Inf
    if raw_points.iter().any(|x| !x.is_finite()) || !query.is_finite() {
        return;
    }

    // Split into xs and ys
    let n = raw_points.len() / 2;
    if n < 2 {
        return;
    }
    let mut xs: Vec<f64> = raw_points[..n].to_vec();
    let ys: Vec<f64> = raw_points[n..2 * n].to_vec();

    // Sort xs and ensure strictly increasing
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs.dedup();
    if xs.len() < 2 || xs.len() != ys.len().min(xs.len()) {
        return;
    }
    let ys = ys[..xs.len()].to_vec();

    // Try linear interpolation — should not panic
    if let Ok(linear) = LinearInterpolation::new(xs.clone(), ys.clone()) {
        let _ = linear.value(query);
    }

    // Try cubic spline interpolation — should not panic
    if xs.len() >= 3 {
        if let Ok(cubic) = CubicSplineInterpolation::new(xs, ys) {
            let _ = cubic.value(query);
        }
    }
});
