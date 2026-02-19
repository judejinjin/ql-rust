//! Variance swap pricing engine.
//!
//! Prices variance swaps using the replication approach.
//! Under the Black-Scholes model, the fair variance equals σ².
//! For partially-elapsed swaps, the value is a combination of
//! realized and implied variance.

use ql_instruments::variance_swap::VarianceSwap;

/// Result from the variance swap engine.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VarianceSwapResult {
    /// Net present value.
    pub npv: f64,
    /// Fair variance (the variance strike that makes NPV = 0).
    pub fair_variance: f64,
    /// Fair volatility (sqrt of fair variance).
    pub fair_volatility: f64,
}

/// Price a variance swap under the Black-Scholes model.
///
/// Under BS, the fair delivery variance equals σ². The NPV is:
///   PV = Notional × discount × (expected_variance − K_var)
///
/// For partially-elapsed swaps:
///   expected_variance = w × realized_variance + (1-w) × implied_variance²
///
/// where w = elapsed_fraction and implied_variance² comes from the
/// current implied vol.
///
/// # Arguments
/// * `vs` — the variance swap
/// * `implied_vol` — current implied volatility (annualized)
/// * `r` — risk-free rate
pub fn price_variance_swap(
    vs: &VarianceSwap,
    implied_vol: f64,
    r: f64,
) -> VarianceSwapResult {
    let implied_var = implied_vol * implied_vol;
    let t_remaining = vs.time_to_expiry;

    // Fair variance: weighted average of realized and implied
    let w = vs.elapsed_fraction;
    let fair_variance = w * vs.realized_variance + (1.0 - w) * implied_var;
    let fair_volatility = fair_variance.sqrt();

    // Discount factor
    let df = (-r * t_remaining).exp();

    // NPV = Notional × DF × (fair_variance − variance_strike)
    let npv = vs.variance_notional * df * (fair_variance - vs.variance_strike);

    VarianceSwapResult {
        npv,
        fair_variance,
        fair_volatility,
    }
}

/// Compute the vega notional from variance notional.
///
/// Vega notional = 2 × variance_notional × vol_strike
pub fn vega_notional(vs: &VarianceSwap) -> f64 {
    2.0 * vs.variance_notional * vs.vol_strike()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn fair_variance_equals_implied_squared() {
        let vs = VarianceSwap::from_vol_strike(100.0, 0.20, 1.0);
        let result = price_variance_swap(&vs, 0.25, 0.05);

        // Fair variance should be implied_vol² = 0.0625
        assert_abs_diff_eq!(result.fair_variance, 0.0625, epsilon = 1e-10);
        assert_abs_diff_eq!(result.fair_volatility, 0.25, epsilon = 1e-10);
    }

    #[test]
    fn npv_zero_at_fair_strike() {
        // If implied vol = vol strike, NPV should be zero
        let vs = VarianceSwap::from_vol_strike(100.0, 0.20, 1.0);
        let result = price_variance_swap(&vs, 0.20, 0.05);

        assert_abs_diff_eq!(result.npv, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn buyer_positive_when_implied_exceeds_strike() {
        // implied vol (25%) > vol strike (20%) → buyer benefits
        let vs = VarianceSwap::from_vol_strike(100.0, 0.20, 1.0);
        let result = price_variance_swap(&vs, 0.25, 0.05);

        assert!(
            result.npv > 0.0,
            "Variance swap buyer should profit when implied > strike: {:.4}",
            result.npv
        );
    }

    #[test]
    fn seller_positive_when_strike_exceeds_implied() {
        // implied vol (15%) < vol strike (20%) → buyer loses
        let vs = VarianceSwap::from_vol_strike(100.0, 0.20, 1.0);
        let result = price_variance_swap(&vs, 0.15, 0.05);

        assert!(
            result.npv < 0.0,
            "Variance swap should have negative NPV when implied < strike: {:.4}",
            result.npv
        );
    }

    #[test]
    fn partially_elapsed_swap() {
        // Half elapsed with realized variance 0.05, implied vol = 0.20
        let vs = VarianceSwap::with_realized(100.0, 0.04, 0.5, 0.05, 0.5);
        let result = price_variance_swap(&vs, 0.20, 0.05);

        // Fair variance = 0.5 × 0.05 + 0.5 × 0.04 = 0.045
        assert_abs_diff_eq!(result.fair_variance, 0.045, epsilon = 1e-10);

        // NPV = 100 × exp(-0.05×0.5) × (0.045 - 0.04) > 0
        assert!(result.npv > 0.0);
    }

    #[test]
    fn vega_notional_calculation() {
        let vs = VarianceSwap::from_vol_strike(100.0, 0.20, 1.0);
        let vn = vega_notional(&vs);
        // vega_notional = 2 × 100 × 0.20 = 40
        assert_abs_diff_eq!(vn, 40.0, epsilon = 1e-10);
    }

    #[test]
    fn npv_scales_with_notional() {
        let vs1 = VarianceSwap::from_vol_strike(100.0, 0.20, 1.0);
        let vs2 = VarianceSwap::from_vol_strike(200.0, 0.20, 1.0);

        let r1 = price_variance_swap(&vs1, 0.25, 0.05);
        let r2 = price_variance_swap(&vs2, 0.25, 0.05);

        assert_abs_diff_eq!(r2.npv, 2.0 * r1.npv, epsilon = 1e-10);
    }
}
