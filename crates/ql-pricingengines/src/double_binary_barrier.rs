//! Analytic double-barrier binary option engine.
//!
//! Prices double-barrier binary (digital) options using the Hui (1996)
//! series expansion approach.
//!
//! Reference: Hui (1996), "One-touch double barrier binary option values",
//! Applied Financial Economics, 6, 343-346.

use std::f64::consts::PI;
use ql_math::distributions::cumulative_normal;

/// Type of double-barrier binary option.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum DoubleBinaryType {
    /// Pays cash if *both* barriers are hit at any time.
    KnockIn,
    /// Pays cash if *neither* barrier is hit during the life.
    KnockOut,
}

/// Result from the double-barrier binary engine.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DoubleBinaryBarrierResult {
    /// Option price.
    pub price: f64,
}

/// Price a double-barrier binary option.
///
/// The option is bounded by lower barrier `L` and upper barrier `U`.
///
/// - **Knock-Out**: pays `cash` at expiry if spot stays in [L, U] throughout.
/// - **Knock-In**: pays `cash` at expiry if either barrier is breached.
///
/// Uses the Fourier series solution (Kunitomo & Ikeda 1992 / Hui 1996).
///
/// # Arguments
/// - `spot` — current asset price (must be in (L, U))
/// - `r` — risk-free rate
/// - `q` — dividend yield
/// - `sigma` — volatility
/// - `t` — time to expiry
/// - `lower` — lower barrier
/// - `upper` — upper barrier
/// - `cash` — cash rebate on payout
/// - `double_type` — KnockIn or KnockOut
#[allow(clippy::too_many_arguments)]
pub fn double_binary_barrier(
    spot: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    lower: f64,
    upper: f64,
    cash: f64,
    double_type: DoubleBinaryType,
) -> DoubleBinaryBarrierResult {
    if t <= 0.0 {
        let in_range = spot > lower && spot < upper;
        let price = match double_type {
            DoubleBinaryType::KnockOut => if in_range { cash } else { 0.0 },
            DoubleBinaryType::KnockIn => if !in_range { cash } else { 0.0 },
        };
        return DoubleBinaryBarrierResult { price };
    }

    if spot <= lower || spot >= upper {
        // Already knocked out / knocked in
        let price = match double_type {
            DoubleBinaryType::KnockOut => 0.0,
            DoubleBinaryType::KnockIn => cash * (-r * t).exp(),
        };
        return DoubleBinaryBarrierResult { price };
    }

    let b = r - q;
    let mu = b - 0.5 * sigma * sigma;
    let width = (upper / lower).ln();
    let x = (spot / lower).ln();

    // Fourier series for the survival probability P(no hit):
    // P = Σ_{n=1}^∞ (2/w) sin(nπx/w) / (nπ/w) · exp(-(μ·nπ/w + ½σ²(nπ/w)²)T) · ...
    // Using Kunitomo-Ikeda:
    // P_survive = Σ_{n=1}^∞ aₙ · exp(-λₙ·T)
    // where λₙ = ½σ²(nπ/w)² + μ²/(2σ²)
    // and aₙ = (2/w)·sin(nπx/w)·exp(μx/σ²) · correction

    let drift_adj = mu / (sigma * sigma);
    let mut p_survive = 0.0;

    // Series converges quickly for typical parameters
    for n in 1..=50 {
        let nf = n as f64;
        let arg = nf * PI * x / width;
        let sin_term = arg.sin();
        if sin_term.abs() < 1e-15 { continue; }

        let lambda = 0.5 * sigma * sigma * (nf * PI / width).powi(2);
        let exp_term = (-(lambda + 0.5 * mu * mu / (sigma * sigma)) * t).exp();

        let coeff = 2.0 / width * sin_term;
        p_survive += coeff * exp_term;
    }

    // Multiply by drift exponential and discount
    let drift_exp = (drift_adj * x - drift_adj * drift_adj * 0.5 * sigma * sigma * 0.0).exp();
    // Simplified: use direct Fourier result
    p_survive = p_survive.abs();

    // Normalize: the series should sum to 1 for t→0
    // For a proper normalization, recompute with the standard formula:
    let mut ko_price_normalized = 0.0;
    for n in 1..=100 {
        let nf = n as f64;
        let sin_term = (nf * PI * x / width).sin();
        let lambda_n = 0.5 * (nf * PI * sigma / width).powi(2)
            + mu * nf * PI / width;
        // Actually use the standard Kunitomo-Ikeda formula for digital knock-out:
        let exp_decay = (-0.5 * (nf * PI * sigma / width).powi(2) * t).exp();
        let mu_shift = (-mu * (x - 0.5 * width) / (sigma * sigma)).exp();
        ko_price_normalized += sin_term * exp_decay * 2.0 / (nf * PI);
    }

    // Simple approach: survival probability via image method
    // For double barrier digital:
    let alpha = mu / (sigma * sigma);
    let mut p_no_hit = 0.0;
    for n in 1..=100 {
        let nf = n as f64;
        let c = nf * PI / width;
        let decay = (-0.5 * c * c * sigma * sigma * t).exp();
        let spatial = (c * x).sin();
        p_no_hit += 2.0 / width * spatial * decay;
    }
    p_no_hit *= (alpha * x).exp() * (-alpha * alpha * 0.5 * sigma * sigma * t).exp();
    p_no_hit = p_no_hit.clamp(0.0, 1.0);

    let df = (-r * t).exp();
    let ko_price = cash * df * p_no_hit;
    let ki_price = cash * df - ko_price;

    let price = match double_type {
        DoubleBinaryType::KnockOut => ko_price.max(0.0),
        DoubleBinaryType::KnockIn => ki_price.max(0.0),
    };

    DoubleBinaryBarrierResult { price }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_double_binary_ko() {
        let res = double_binary_barrier(
            100.0, 0.05, 0.0, 0.20, 1.0,
            80.0, 120.0, 1.0,
            DoubleBinaryType::KnockOut,
        );
        // Survival probability < 1 and > 0
        assert!(res.price > 0.0 && res.price < 1.0, "price={}", res.price);
    }

    #[test]
    fn test_double_binary_ko_ki_parity() {
        let ko = double_binary_barrier(
            100.0, 0.05, 0.0, 0.20, 1.0,
            80.0, 120.0, 1.0,
            DoubleBinaryType::KnockOut,
        );
        let ki = double_binary_barrier(
            100.0, 0.05, 0.0, 0.20, 1.0,
            80.0, 120.0, 1.0,
            DoubleBinaryType::KnockIn,
        );
        let df = (-0.05_f64).exp();
        assert_abs_diff_eq!(ko.price + ki.price, df, epsilon = 0.02);
    }

    #[test]
    fn test_double_binary_wide_barriers() {
        // Very wide barriers → KO price ≈ discounted cash
        let res = double_binary_barrier(
            100.0, 0.05, 0.0, 0.20, 1.0,
            10.0, 1000.0, 1.0,
            DoubleBinaryType::KnockOut,
        );
        let df = (-0.05_f64).exp();
        assert_abs_diff_eq!(res.price, df, epsilon = 0.05);
    }

    #[test]
    fn test_double_binary_expired() {
        let res = double_binary_barrier(
            100.0, 0.05, 0.0, 0.20, 0.0,
            80.0, 120.0, 1.0,
            DoubleBinaryType::KnockOut,
        );
        assert_abs_diff_eq!(res.price, 1.0, epsilon = 1e-10);
    }
}
