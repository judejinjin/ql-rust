//! Cliquet (ratchet) option engine.
//!
//! A cliquet option is a chain of forward-starting options.  At each
//! reset date the strike resets to the then-current spot.  The payoff
//! is the sum of locally capped/floored per-period returns, subject
//! to a global cap and floor on the accumulated return.
//!
//! This engine uses the closed-form sum of Black-Scholes forward-starting
//! option values.  Each period contributes a BS value with the forward
//! moneyness equal to 1 (ATM forward-start), and the per-period return
//! is capped/floored using call-spread replication.
//!
//! # References
//! - Wilmott, P. (2006), *Paul Wilmott on Quantitative Finance*,
//!   Chapter 26: Cliquets.
//! - Haug, E.G. (2007), *The Complete Guide to Option Pricing Formulas*,
//!   Chapter 7.

use ql_instruments::OptionType;
use crate::analytic_european;

/// Result from a cliquet option pricing.
#[derive(Debug, Clone)]
#[must_use]
pub struct CliquetResult {
    /// Net present value.
    pub npv: f64,
    /// Per-period forward-start option values (before capping/flooring).
    pub period_values: Vec<f64>,
}

/// Price a cliquet option using a sum of forward-starting BS options.
///
/// Each period `[t_{i-1}, t_i]` contributes a forward-starting ATM option
/// whose value is computed at `t_{i-1}` and discounted to today.
///
/// The per-period return `R_i = S(t_i)/S(t_{i-1}) - 1` is floored at
/// `local_floor` and capped at `local_cap`.  The accumulated return
/// `Σ R_i` is then floored/capped globally.
///
/// # Parameters
/// - `spot`: current underlying price
/// - `r`: risk-free rate
/// - `q`: dividend yield
/// - `vol`: volatility (assumed flat for all periods)
/// - `reset_times`: sorted year-fractions for reset dates (last = expiry)
/// - `local_floor`, `local_cap`: per-period return bounds
/// - `global_floor`, `global_cap`: overall accumulated return bounds
/// - `notional`: notional amount
/// - `option_type`: Call or Put (determines sign of per-period returns)
#[allow(clippy::too_many_arguments)]
pub fn cliquet_price(
    _spot: f64,
    r: f64,
    q: f64,
    vol: f64,
    reset_times: &[f64],
    local_floor: f64,
    local_cap: f64,
    global_floor: f64,
    global_cap: f64,
    notional: f64,
    option_type: OptionType,
) -> CliquetResult {
    if reset_times.is_empty() {
        return CliquetResult {
            npv: 0.0,
            period_values: vec![],
        };
    }

    let n_periods = reset_times.len();
    let mut period_values = Vec::with_capacity(n_periods);
    let mut total_fwd_value = 0.0;

    for i in 0..n_periods {
        let t_start = if i == 0 { 0.0 } else { reset_times[i - 1] };
        let t_end = reset_times[i];
        let dt = t_end - t_start;
        if dt <= 0.0 {
            period_values.push(0.0);
            continue;
        }

        // For a call cliquet, the per-period capped return is:
        //   capped_R = min(max(R_i, floor), cap)
        //            = floor + max(R_i - floor, 0) - max(R_i - cap, 0)
        //
        // where R_i = S(t_i)/S(t_{i-1}) - 1 for calls,
        //       R_i = 1 - S(t_i)/S(t_{i-1}) for puts.
        //
        // For calls: max(R_i - f, 0) = max(S_T/S_0 - (1+f), 0) → BS call(1, 1+f)
        // For puts:  max(R_i - f, 0) = max((1-f) - S_T/S_0, 0) → BS put(1, 1-f)

        let (k_floor, k_cap, bs_type) = match option_type {
            OptionType::Call => (
                (1.0 + local_floor).max(1e-10),
                (1.0 + local_cap).max(1e-10),
                OptionType::Call,
            ),
            OptionType::Put => (
                (1.0 - local_floor).max(1e-10),
                (1.0 - local_cap).max(1e-10),
                OptionType::Put,
            ),
        };

        // BS(1, K, r, q, vol, dt) = e^{-r·dt} · E[max(±(S_T/S_0 - K), 0)]
        let v_floor = analytic_european::black_scholes_price(
            1.0, k_floor, r, q, vol, dt, bs_type,
        )
        .npv;
        let v_cap = analytic_european::black_scholes_price(
            1.0, k_cap, r, q, vol, dt, bs_type,
        )
        .npv;

        // PV at time 0 of the capped return:
        //   e^{-r·t_end}·floor + e^{-r·t_start}·(v_floor - v_cap)
        //
        // The e^{-r·t_start} factor discounts: BS already discounts over dt,
        // so we need to discount from t_start to 0 for the BS prices;
        // the floor term is received at t_end, hence e^{-r·t_end}.
        let df_end = (-r * t_end).exp();
        let df_start = (-r * t_start).exp();
        let pv_period = df_end * local_floor + df_start * (v_floor - v_cap);

        period_values.push(pv_period);
        total_fwd_value += pv_period;
    }

    // Apply global floor/cap (approximation: apply to expected PV sum)
    let t_final = *reset_times.last().unwrap();
    let df_final = (-r * t_final).exp();
    let capped = total_fwd_value
        .max(global_floor * df_final)
        .min(global_cap * df_final);

    CliquetResult {
        npv: capped * notional,
        period_values,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const SPOT: f64 = 100.0;
    const R: f64 = 0.05;
    const Q: f64 = 0.02;
    const VOL: f64 = 0.20;

    #[test]
    fn cliquet_call_positive() {
        let resets = vec![0.25, 0.50, 0.75, 1.00];
        let res = cliquet_price(
            SPOT, R, Q, VOL, &resets, -1.0, 1.0, -1.0, 1.0, 1.0, OptionType::Call,
        );
        assert!(res.npv > 0.0, "Cliquet call should be positive: {}", res.npv);
        assert_eq!(res.period_values.len(), 4);
    }

    #[test]
    fn cliquet_put_has_value() {
        // A put cliquet with wide floor/cap and r > q has negative expected returns
        // when q < r (carry cost). With a global floor of 0, the value should be ≥ 0.
        let resets = vec![0.25, 0.50, 0.75, 1.00];
        let res = cliquet_price(
            SPOT, R, Q, VOL, &resets, -1.0, 1.0, 0.0, 1.0, 1.0, OptionType::Put,
        );
        assert!(res.npv >= 0.0, "Cliquet put with global floor 0 should be ≥ 0: {}", res.npv);
    }

    #[test]
    fn cliquet_cap_reduces_value() {
        let resets = vec![0.25, 0.50, 0.75, 1.00];
        let uncapped = cliquet_price(
            SPOT, R, Q, VOL, &resets, -1.0, 1.0, -1.0, 1.0, 1.0, OptionType::Call,
        );
        let capped = cliquet_price(
            SPOT, R, Q, VOL, &resets, -1.0, 0.05, -1.0, 1.0, 1.0, OptionType::Call,
        );
        assert!(
            capped.npv <= uncapped.npv + 0.01,
            "Capped cliquet ({:.4}) should be <= uncapped ({:.4})",
            capped.npv,
            uncapped.npv
        );
    }

    #[test]
    fn cliquet_global_floor_provides_minimum() {
        let resets = vec![0.25, 0.50, 0.75, 1.00];
        let res = cliquet_price(
            SPOT, R, Q, VOL, &resets, -1.0, 1.0, 0.02, 1.0, 1_000_000.0, OptionType::Call,
        );
        // Global floor of 2% on 1M notional → PV ≥ 0.02 * df * 1M
        let df = (-R * 1.0_f64).exp();
        assert!(
            res.npv >= 0.02 * df * 1_000_000.0 - 1.0,
            "Global floor should guarantee minimum: npv = {:.2}",
            res.npv
        );
    }

    #[test]
    fn cliquet_notional_scales_linearly() {
        let resets = vec![0.50, 1.00];
        let res1 = cliquet_price(
            SPOT, R, Q, VOL, &resets, -0.05, 0.10, 0.0, 1.0, 1.0, OptionType::Call,
        );
        let res2 = cliquet_price(
            SPOT, R, Q, VOL, &resets, -0.05, 0.10, 0.0, 1.0, 100.0, OptionType::Call,
        );
        let ratio = res2.npv / res1.npv;
        assert!(
            (ratio - 100.0).abs() < 0.01,
            "Notional scaling: ratio = {:.6}, expected 100",
            ratio
        );
    }

    #[test]
    fn cliquet_empty_resets() {
        let res = cliquet_price(SPOT, R, Q, VOL, &[], -0.05, 0.10, 0.0, 1.0, 1.0, OptionType::Call);
        assert!((res.npv).abs() < 1e-12);
        assert!(res.period_values.is_empty());
    }
}
