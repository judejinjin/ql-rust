//! Analytic European option pricing engine (Black-Scholes).
//!
//! Computes NPV and Greeks for European vanilla options using the
//! Black-Scholes closed-form solution.

use ql_instruments::{OptionType, VanillaOption};
use ql_math::distributions::NormalDistribution;
use tracing::info_span;

/// Results from the analytic European engine.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AnalyticEuropeanResults {
    /// Net present value.
    pub npv: f64,
    /// Delta: ∂V/∂S.
    pub delta: f64,
    /// Gamma: ∂²V/∂S².
    pub gamma: f64,
    /// Vega: ∂V/∂σ (per 1% move, not per 100%).
    pub vega: f64,
    /// Theta: ∂V/∂t (daily decay).
    pub theta: f64,
    /// Rho: ∂V/∂r (per 1% move).
    pub rho: f64,
}

/// Black-Scholes analytic pricing engine for European options.
///
/// # Parameters
/// - `spot`: current underlying price
/// - `risk_free_rate`: continuously compounded risk-free rate
/// - `dividend_yield`: continuously compounded dividend yield
/// - `volatility`: annualized volatility (σ)
/// - `time_to_expiry`: time to expiry in years
pub fn price_european(
    option: &VanillaOption,
    spot: f64,
    risk_free_rate: f64,
    dividend_yield: f64,
    volatility: f64,
    time_to_expiry: f64,
) -> AnalyticEuropeanResults {
    let _span = info_span!("price_european", spot, volatility, time_to_expiry).entered();
    // Validate that this is a European option with PlainVanilla payoff
    let strike = option.strike();
    let omega = option.option_type().sign(); // +1 call, -1 put

    if time_to_expiry <= 0.0 {
        // Expired option: intrinsic value only
        let intrinsic = (omega * (spot - strike)).max(0.0);
        return AnalyticEuropeanResults {
            npv: intrinsic,
            delta: if omega * (spot - strike) > 0.0 { omega } else { 0.0 },
            gamma: 0.0,
            vega: 0.0,
            theta: 0.0,
            rho: 0.0,
        };
    }

    let t = time_to_expiry;
    let sigma = volatility;
    let r = risk_free_rate;
    let q = dividend_yield;
    let sqrt_t = t.sqrt();

    // d1, d2
    let d1 = ((spot / strike).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;

    let n = NormalDistribution::standard();
    let nd1 = n.cdf(omega * d1);
    let nd2 = n.cdf(omega * d2);
    let npdf_d1 = n.pdf(d1);

    let df_q = (-q * t).exp(); // discount factor for dividends
    let df_r = (-r * t).exp(); // discount factor for risk-free rate

    // NPV
    let npv = omega * (spot * df_q * nd1 - strike * df_r * nd2);

    // Delta: ∂V/∂S
    let delta = omega * df_q * nd1;

    // Gamma: ∂²V/∂S²
    let gamma = df_q * npdf_d1 / (spot * sigma * sqrt_t);

    // Vega: ∂V/∂σ  (scaled to 1% = 0.01)
    let vega = spot * df_q * npdf_d1 * sqrt_t * 0.01;

    // Theta: ∂V/∂t (per calendar day = 1/365)
    let theta_continuous = -spot * df_q * npdf_d1 * sigma / (2.0 * sqrt_t)
        - omega * r * strike * df_r * nd2
        + omega * q * spot * df_q * nd1;
    let theta = theta_continuous / 365.0;

    // Rho: ∂V/∂r (scaled to 1% = 0.01)
    let rho = omega * strike * t * df_r * nd2 * 0.01;

    AnalyticEuropeanResults {
        npv,
        delta,
        gamma,
        vega,
        theta,
        rho,
    }
}

/// Lightweight Black-Scholes price (NPV only, no Greeks) for use by
/// exotic engines that don't need a full `VanillaOption` instrument.
pub fn black_scholes_price(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    option_type: OptionType,
) -> AnalyticEuropeanResults {
    use ql_math::distributions::NormalDistribution;

    let omega = option_type.sign();
    if t <= 0.0 {
        let intrinsic = (omega * (spot - strike)).max(0.0);
        return AnalyticEuropeanResults {
            npv: intrinsic,
            delta: 0.0,
            gamma: 0.0,
            vega: 0.0,
            theta: 0.0,
            rho: 0.0,
        };
    }
    let sqrt_t = t.sqrt();
    let d1 = ((spot / strike).ln() + (r - q + 0.5 * vol * vol) * t) / (vol * sqrt_t);
    let d2 = d1 - vol * sqrt_t;
    let n = NormalDistribution::standard();
    let nd1 = n.cdf(omega * d1);
    let nd2 = n.cdf(omega * d2);
    let df_q = (-q * t).exp();
    let df_r = (-r * t).exp();
    let npv = omega * (spot * df_q * nd1 - strike * df_r * nd2);
    AnalyticEuropeanResults {
        npv,
        delta: 0.0,
        gamma: 0.0,
        vega: 0.0,
        theta: 0.0,
        rho: 0.0,
    }
}

/// Compute implied volatility for a European option using Brent solver.
///
/// Finds the volatility σ such that BS_price(σ) = target_price.
pub fn implied_volatility(
    option: &VanillaOption,
    target_price: f64,
    spot: f64,
    risk_free_rate: f64,
    dividend_yield: f64,
    time_to_expiry: f64,
) -> Result<f64, ql_core::errors::QLError> {
    use ql_math::solvers1d::{Brent, Solver1D};

    let solver = Brent;

    let objective = |vol: f64| -> f64 {
        let result = price_european(option, spot, risk_free_rate, dividend_yield, vol, time_to_expiry);
        result.npv - target_price
    };

    // Bracket: vol between 0.001 (0.1%) and 5.0 (500%)
    solver.solve(objective, 0.0, 0.2, 0.001, 5.0, 1e-12, 100)
}

/// Price a European call or put with **discrete dividends** using the
/// escrowed dividend model.
///
/// The spot is adjusted by subtracting the PV of cash dividends and
/// multiplying by the cumulative proportional factor. Black-Scholes
/// is then applied to the adjusted spot with `q = 0`.
///
/// This approach is exact when all dividends are known and the volatility
/// applies to the forward (ex-dividend) stock price process.
///
/// # Returns
///
/// The same `AnalyticEuropeanResults` struct with NPV and Greeks.
/// Delta is the *total* delta w.r.t. the unadjusted spot.
#[allow(clippy::too_many_arguments)]
pub fn price_european_discrete_dividends(
    spot: f64,
    strike: f64,
    r: f64,
    vol: f64,
    t: f64,
    option_type: OptionType,
    dividends: &ql_cashflows::DividendSchedule,
) -> AnalyticEuropeanResults {
    let s_adj = dividends.escrowed_spot(spot, r, t);
    if s_adj <= 0.0 {
        // Dividends exceed spot: call worthless, put deep ITM
        let omega = option_type.sign();
        let df_r = (-r * t).exp();
        return AnalyticEuropeanResults {
            npv: if omega < 0.0 {
                (strike * df_r - s_adj.max(0.0)).max(0.0)
            } else {
                0.0
            },
            delta: 0.0,
            gamma: 0.0,
            vega: 0.0,
            theta: 0.0,
            rho: 0.0,
        };
    }
    // Use BS with adjusted spot and q=0
    let result = black_scholes_price(s_adj, strike, r, 0.0, vol, t, option_type);

    // Adjust delta: d(NPV)/dS = d(NPV)/dS* × dS*/dS
    // where dS*/dS = proportional_factor(t)
    let prop_factor = dividends.proportional_factor(t);
    AnalyticEuropeanResults {
        npv: result.npv,
        delta: result.delta * prop_factor,
        gamma: result.gamma * prop_factor * prop_factor,
        vega: result.vega,
        theta: result.theta,
        rho: result.rho,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_instruments::VanillaOption;
    use ql_time::{Date, Month};

    // Standard test case: S=100, K=100, r=5%, q=0%, σ=20%, T=1Y
    fn standard_option_call() -> (VanillaOption, f64, f64, f64, f64, f64) {
        let opt = VanillaOption::european_call(100.0, Date::from_ymd(2026, Month::January, 2));
        (opt, 100.0, 0.05, 0.0, 0.20, 1.0)
    }

    fn standard_option_put() -> (VanillaOption, f64, f64, f64, f64, f64) {
        let opt = VanillaOption::european_put(100.0, Date::from_ymd(2026, Month::January, 2));
        (opt, 100.0, 0.05, 0.0, 0.20, 1.0)
    }

    #[test]
    fn bs_call_price() {
        let (opt, spot, r, q, vol, t) = standard_option_call();
        let result = price_european(&opt, spot, r, q, vol, t);
        // QuantLib reference: BS call with S=K=100, r=5%, σ=20%, T=1 ≈ 10.4506
        assert_abs_diff_eq!(result.npv, 10.4506, epsilon = 0.001);
    }

    #[test]
    fn bs_put_price() {
        let (opt, spot, r, q, vol, t) = standard_option_put();
        let result = price_european(&opt, spot, r, q, vol, t);
        // Put-call parity: P = C - S + K*exp(-rT)
        let (_, _, _, _, _, _) = standard_option_call();
        let call_result = price_european(
            &VanillaOption::european_call(100.0, Date::from_ymd(2026, Month::January, 2)),
            spot, r, q, vol, t,
        );
        let parity = call_result.npv - spot + 100.0 * (-r * t).exp();
        assert_abs_diff_eq!(result.npv, parity, epsilon = 1e-10);
    }

    #[test]
    fn bs_delta_call() {
        let (opt, spot, r, q, vol, t) = standard_option_call();
        let result = price_european(&opt, spot, r, q, vol, t);
        // ATM call delta should be ~0.6368
        assert_abs_diff_eq!(result.delta, 0.6368, epsilon = 0.002);
    }

    #[test]
    fn bs_delta_put() {
        let (opt, spot, r, q, vol, t) = standard_option_put();
        let result = price_european(&opt, spot, r, q, vol, t);
        // Put delta should be call_delta - 1
        let call_result = price_european(
            &VanillaOption::european_call(100.0, Date::from_ymd(2026, Month::January, 2)),
            spot, r, q, vol, t,
        );
        assert_abs_diff_eq!(result.delta, call_result.delta - 1.0, epsilon = 1e-10);
    }

    #[test]
    fn bs_gamma() {
        let (opt, spot, r, q, vol, t) = standard_option_call();
        let result = price_european(&opt, spot, r, q, vol, t);
        // Gamma should be same for call and put
        let put_result = price_european(
            &VanillaOption::european_put(100.0, Date::from_ymd(2026, Month::January, 2)),
            spot, r, q, vol, t,
        );
        assert_abs_diff_eq!(result.gamma, put_result.gamma, epsilon = 1e-10);
        // ATM gamma ~0.0188
        assert_abs_diff_eq!(result.gamma, 0.0188, epsilon = 0.001);
    }

    #[test]
    fn bs_vega() {
        let (opt, spot, r, q, vol, t) = standard_option_call();
        let result = price_european(&opt, spot, r, q, vol, t);
        // Vega should be positive
        assert!(result.vega > 0.0);
        // ATM vega per 1% vol ~ 0.375
        assert_abs_diff_eq!(result.vega, 0.375, epsilon = 0.02);
    }

    #[test]
    fn bs_put_call_parity() {
        let (call_opt, spot, r, q, vol, t) = standard_option_call();
        let (put_opt, _, _, _, _, _) = standard_option_put();
        let call = price_european(&call_opt, spot, r, q, vol, t);
        let put = price_european(&put_opt, spot, r, q, vol, t);
        // C - P = S*exp(-qT) - K*exp(-rT)
        let lhs = call.npv - put.npv;
        let rhs = spot * (-q * t).exp() - 100.0 * (-r * t).exp();
        assert_abs_diff_eq!(lhs, rhs, epsilon = 1e-10);
    }

    #[test]
    fn implied_vol_atm_call() {
        let (opt, spot, r, q, vol, t) = standard_option_call();
        let result = price_european(&opt, spot, r, q, vol, t);
        let iv = implied_volatility(&opt, result.npv, spot, r, q, t).unwrap();
        assert_abs_diff_eq!(iv, vol, epsilon = 1e-8);
    }

    #[test]
    fn implied_vol_itm_put() {
        let opt = VanillaOption::european_put(110.0, Date::from_ymd(2026, Month::January, 2));
        let result = price_european(&opt, 100.0, 0.05, 0.0, 0.25, 1.0);
        let iv = implied_volatility(&opt, result.npv, 100.0, 0.05, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(iv, 0.25, epsilon = 1e-8);
    }

    #[test]
    fn implied_vol_otm_call() {
        let opt = VanillaOption::european_call(120.0, Date::from_ymd(2026, Month::January, 2));
        let result = price_european(&opt, 100.0, 0.05, 0.0, 0.30, 1.0);
        let iv = implied_volatility(&opt, result.npv, 100.0, 0.05, 0.0, 1.0).unwrap();
        assert_abs_diff_eq!(iv, 0.30, epsilon = 1e-8);
    }

    #[test]
    fn bs_with_dividend_yield() {
        let opt = VanillaOption::european_call(100.0, Date::from_ymd(2026, Month::January, 2));
        let no_div = price_european(&opt, 100.0, 0.05, 0.0, 0.20, 1.0);
        let with_div = price_european(&opt, 100.0, 0.05, 0.02, 0.20, 1.0);
        // Dividend yield should reduce call price
        assert!(with_div.npv < no_div.npv, "Dividend yield should reduce call price");
    }

    #[test]
    fn expired_option() {
        let opt = VanillaOption::european_call(100.0, Date::from_ymd(2024, Month::January, 2));
        let result = price_european(&opt, 110.0, 0.05, 0.0, 0.20, 0.0);
        assert_abs_diff_eq!(result.npv, 10.0, epsilon = 1e-10); // intrinsic value
    }

    #[test]
    fn discrete_div_reduces_call() {
        use ql_cashflows::{Dividend, DividendSchedule};
        let divs = DividendSchedule::new(vec![
            Dividend::cash(0.25, 2.0),
            Dividend::cash(0.75, 2.0),
        ]);
        let no_div = black_scholes_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Call);
        let with_div = price_european_discrete_dividends(
            100.0, 100.0, 0.05, 0.20, 1.0, OptionType::Call, &divs,
        );
        assert!(
            with_div.npv < no_div.npv,
            "Discrete dividends should reduce call price: {} vs {}",
            with_div.npv, no_div.npv
        );
    }

    #[test]
    fn discrete_div_empty_matches_no_div() {
        use ql_cashflows::DividendSchedule;
        let divs = DividendSchedule::empty();
        let no_div = black_scholes_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Call);
        let with_div = price_european_discrete_dividends(
            100.0, 100.0, 0.05, 0.20, 1.0, OptionType::Call, &divs,
        );
        assert_abs_diff_eq!(with_div.npv, no_div.npv, epsilon = 1e-12);
    }

    #[test]
    fn discrete_div_put_call_parity() {
        use ql_cashflows::{Dividend, DividendSchedule};
        let divs = DividendSchedule::new(vec![Dividend::cash(0.5, 3.0)]);
        let call = price_european_discrete_dividends(
            100.0, 100.0, 0.05, 0.20, 1.0, OptionType::Call, &divs,
        );
        let put = price_european_discrete_dividends(
            100.0, 100.0, 0.05, 0.20, 1.0, OptionType::Put, &divs,
        );
        let s_adj = divs.escrowed_spot(100.0, 0.05, 1.0);
        let parity = s_adj - 100.0 * (-0.05_f64).exp();
        assert_abs_diff_eq!(call.npv - put.npv, parity, epsilon = 0.01);
    }

    #[test]
    fn discrete_proportional_div() {
        use ql_cashflows::{Dividend, DividendSchedule};
        let divs = DividendSchedule::new(vec![Dividend::proportional(0.5, 0.03)]);
        let result = price_european_discrete_dividends(
            100.0, 100.0, 0.05, 0.20, 1.0, OptionType::Call, &divs,
        );
        // With 3% proportional dividend, effective spot is 97 → lower call price
        let no_div = black_scholes_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Call);
        assert!(result.npv < no_div.npv);
        assert!(result.npv > 0.0);
    }
}
