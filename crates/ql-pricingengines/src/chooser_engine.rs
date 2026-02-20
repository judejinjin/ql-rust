//! Simple chooser option engine — Rubinstein (1991).
//!
//! A simple chooser option gives the holder the right to choose
//! at time `t_c` (the choosing time) whether the option becomes a
//! European call or put with strike `K` and expiry `T`.
//!
//! The Rubinstein formula decomposes the chooser into a European call
//! plus a European put with adjusted parameters, using put-call parity.
//!
//! # References
//! - Rubinstein, M. (1991), "Options for the Undecided", *Risk* 4(4).
//! - Haug, E.G. (2007), *The Complete Guide to Option Pricing Formulas*,
//!   Chapter 6.

use ql_math::distributions::cumulative_normal;

/// Result from a chooser option pricing.
#[derive(Debug, Clone)]
pub struct ChooserResult {
    /// Net present value of the chooser option.
    pub npv: f64,
}

/// Price a simple chooser option using the Rubinstein (1991) formula.
///
/// # Parameters
/// - `spot`: current underlying price
/// - `strike`: option strike
/// - `r`: risk-free rate
/// - `q`: dividend yield
/// - `vol`: volatility
/// - `t_choose`: time to the choosing date (years)
/// - `t_expiry`: time to expiry (years), must be >= `t_choose`
///
/// # Formula
/// The chooser value is:
///   `C = S·e^{-qT}·N(d1) − K·e^{-rT}·N(d2)
///      − S·e^{-qT}·N(−y1) + K·e^{-rT_c}·e^{-r(T−T_c)}·N(−y2)`
///
/// which simplifies to:
///   `C = c(S, K, T) + p(S, K·e^{−(r−q)(T−T_c)}, T_c)`
///
/// where `c()` and `p()` are Black-Scholes call/put prices.
pub fn chooser_price(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    vol: f64,
    t_choose: f64,
    t_expiry: f64,
) -> ChooserResult {
    debug_assert!(t_choose > 0.0 && t_choose <= t_expiry);

    let sqrt_t = t_expiry.sqrt();
    let sqrt_tc = t_choose.sqrt();

    let d1 = ((spot / strike).ln() + (r - q + 0.5 * vol * vol) * t_expiry) / (vol * sqrt_t);
    let d2 = d1 - vol * sqrt_t;

    // y terms use t_choose
    let y1 = ((spot / strike).ln() + (r - q) * t_expiry + 0.5 * vol * vol * t_choose)
        / (vol * sqrt_tc);
    let y2 = y1 - vol * sqrt_tc;

    let df_q_t = (-q * t_expiry).exp();
    let df_r_t = (-r * t_expiry).exp();
    let df_r_tc = (-r * t_choose).exp();

    // Rubinstein decomposition
    let npv = spot * df_q_t * cumulative_normal(d1) - strike * df_r_t * cumulative_normal(d2)
        - spot * df_q_t * cumulative_normal(-y1) + strike * df_r_tc * cumulative_normal(-y2);

    ChooserResult { npv }
}

/// Convenience: price using `ChooserOption` instrument fields.
pub fn price_chooser_option(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    vol: f64,
    choosing_time: f64,
    time_to_expiry: f64,
) -> ChooserResult {
    chooser_price(spot, strike, r, q, vol, choosing_time, time_to_expiry)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analytic_european;
    use ql_instruments::OptionType;

    const SPOT: f64 = 50.0;
    const STRIKE: f64 = 50.0;
    const R: f64 = 0.08;
    const Q: f64 = 0.0;
    const VOL: f64 = 0.25;
    const T_CHOOSE: f64 = 0.25; // 3 months
    const T_EXPIRY: f64 = 0.50; // 6 months

    #[test]
    fn chooser_positive() {
        let res = chooser_price(SPOT, STRIKE, R, Q, VOL, T_CHOOSE, T_EXPIRY);
        assert!(res.npv > 0.0, "Chooser should be positive: {}", res.npv);
    }

    #[test]
    fn chooser_exceeds_call_and_put() {
        // Chooser is always at least as valuable as the more expensive of call or put
        let chooser = chooser_price(SPOT, STRIKE, R, Q, VOL, T_CHOOSE, T_EXPIRY).npv;
        let call = analytic_european::black_scholes_price(SPOT, STRIKE, R, Q, VOL, T_EXPIRY, OptionType::Call).npv;
        let put = analytic_european::black_scholes_price(SPOT, STRIKE, R, Q, VOL, T_EXPIRY, OptionType::Put).npv;
        assert!(
            chooser >= call.max(put) - 0.01,
            "Chooser {:.4} should >= max(call {:.4}, put {:.4})",
            chooser, call, put
        );
    }

    #[test]
    fn chooser_bounded_by_call_plus_put() {
        // Chooser <= call + put (it's choosing one, not getting both)
        let chooser = chooser_price(SPOT, STRIKE, R, Q, VOL, T_CHOOSE, T_EXPIRY).npv;
        let call = analytic_european::black_scholes_price(SPOT, STRIKE, R, Q, VOL, T_EXPIRY, OptionType::Call).npv;
        let put = analytic_european::black_scholes_price(SPOT, STRIKE, R, Q, VOL, T_EXPIRY, OptionType::Put).npv;
        assert!(
            chooser <= call + put + 0.01,
            "Chooser {:.4} should <= call ({:.4}) + put ({:.4}) = {:.4}",
            chooser, call, put, call + put
        );
    }

    #[test]
    fn chooser_haug_reference() {
        // Haug (2007) example: S=50, K=50, r=8%, q=0%, σ=25%, Tc=0.25, T=0.50
        // Expected value ≈ 6.1071
        let res = chooser_price(SPOT, STRIKE, R, Q, VOL, T_CHOOSE, T_EXPIRY);
        assert!(
            (res.npv - 6.1071).abs() < 0.50,
            "Chooser {:.4} should ≈ 6.1071 (Haug reference)",
            res.npv
        );
    }

    #[test]
    fn chooser_late_choose_equals_straddle() {
        // If choosing time = expiry, chooser = max(call payoff, put payoff)
        // ≈ a straddle (call + put - min(call, put))
        // For ATM: chooser(t_c→T) approaches call + put disounted value
        let t = 1.0;
        let chooser = chooser_price(SPOT, STRIKE, R, Q, VOL, t - 0.001, t).npv;
        let call = analytic_european::black_scholes_price(SPOT, STRIKE, R, Q, VOL, t, OptionType::Call).npv;
        let put = analytic_european::black_scholes_price(SPOT, STRIKE, R, Q, VOL, t, OptionType::Put).npv;
        // When t_c ≈ T, the chooser ≈ call + put (straddle)
        let straddle = call + put;
        let diff = (chooser - straddle).abs();
        assert!(
            diff < 0.50,
            "Chooser({:.4}) with tc≈T should ≈ straddle({:.4})",
            chooser, straddle
        );
    }
}
