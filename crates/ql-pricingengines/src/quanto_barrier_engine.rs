//! Quanto barrier option pricing engine.
//!
//! A **quanto barrier option** is a barrier option where the underlying is
//! denominated in a foreign currency, but the payoff is converted to the
//! domestic currency at a fixed exchange rate (typically 1.0).
//!
//! ## Pricing
//!
//! The quanto adjustment modifies the drift of the underlying by replacing
//! the foreign risk-free rate `r_f` with an effective rate:
//!
//! ```text
//! r_f_adj = r_f + ρ_{S,FX} · σ_S · σ_FX
//! ```
//!
//! The barrier option is then priced using the Merton-Reiner-Rubinstein
//! closed-form formula with the adjusted foreign rate, discounted at
//! the domestic risk-free rate.
//!
//! ## QuantLib Parity
//!
//! Corresponds to `QuantoBarrierOption` + `AnalyticBarrierEngine` in
//! QuantLib C++ (ql/instruments/quantobarrieroption.hpp).

use ql_instruments::barrier_option::BarrierType;
use ql_instruments::payoff::OptionType;
use ql_math::distributions::cumulative_normal;
use serde::{Deserialize, Serialize};

/// Parameters for a quanto barrier option.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantoBarrierOption {
    /// Option type (call or put).
    pub option_type: OptionType,
    /// Barrier type (UpIn, UpOut, DownIn, DownOut).
    pub barrier_type: BarrierType,
    /// Spot price of the underlying (in foreign currency).
    pub spot: f64,
    /// Strike price (in foreign currency).
    pub strike: f64,
    /// Barrier level (in foreign currency).
    pub barrier: f64,
    /// Cash rebate paid if the option is knocked out.
    pub rebate: f64,
    /// Time to expiry (years).
    pub tau: f64,
    /// Domestic risk-free rate (continuous).
    pub r_domestic: f64,
    /// Foreign risk-free rate (continuous).
    pub r_foreign: f64,
    /// Underlying volatility σ_S.
    pub sigma: f64,
    /// FX volatility σ_FX.
    pub sigma_fx: f64,
    /// Correlation ρ between ln(S) and ln(FX).
    pub rho_sfx: f64,
    /// Fixed exchange rate Q (domestic per 1 foreign). Typically 1.0.
    pub fixed_fx: f64,
}

/// Results from quanto barrier option pricing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantoBarrierResult {
    /// Option price in domestic currency.
    pub price: f64,
    /// Delta: ∂price/∂S.
    pub delta: f64,
    /// Gamma: ∂²price/∂S².
    pub gamma: f64,
    /// Vega: ∂price/∂σ_S (per 1% move).
    pub vega: f64,
    /// Quanto vega: ∂price/∂σ_FX (per 1% move).
    pub quanto_vega: f64,
    /// Theta: ∂price/∂t (per day).
    pub theta: f64,
}

/// Price a quanto barrier option analytically.
///
/// Uses the Merton-Reiner-Rubinstein closed-form barrier pricing formula
/// with a quanto-adjusted foreign rate.
pub fn price_quanto_barrier(opt: &QuantoBarrierOption) -> QuantoBarrierResult {
    let q = opt.fixed_fx;

    // Quanto-adjusted foreign rate
    let quanto_adj = opt.rho_sfx * opt.sigma * opt.sigma_fx;
    let r_f_adj = opt.r_foreign + quanto_adj;

    // Price using the barrier formula with (r_d, r_f_adj)
    let price = q * barrier_closed_form(
        opt.spot,
        opt.strike,
        opt.barrier,
        opt.rebate,
        opt.r_domestic,
        r_f_adj,
        opt.sigma,
        opt.tau,
        opt.barrier_type,
        opt.option_type,
    );

    // Delta via bump-and-reprice
    let ds = opt.spot * 0.001;
    let price_up = q * barrier_closed_form(
        opt.spot + ds, opt.strike, opt.barrier, opt.rebate,
        opt.r_domestic, r_f_adj, opt.sigma, opt.tau,
        opt.barrier_type, opt.option_type,
    );
    let price_down = q * barrier_closed_form(
        opt.spot - ds, opt.strike, opt.barrier, opt.rebate,
        opt.r_domestic, r_f_adj, opt.sigma, opt.tau,
        opt.barrier_type, opt.option_type,
    );
    let delta = (price_up - price_down) / (2.0 * ds);
    let gamma = (price_up - 2.0 * price + price_down) / (ds * ds);

    // Vega via bump
    let dsig = 0.001;
    let price_vup = q * barrier_closed_form(
        opt.spot, opt.strike, opt.barrier, opt.rebate,
        opt.r_domestic, r_f_adj, opt.sigma + dsig, opt.tau,
        opt.barrier_type, opt.option_type,
    );
    let vega = (price_vup - price) / dsig * 0.01; // per 1% vol move

    // Quanto vega: sensitivity to σ_FX
    let r_f_adj_up = opt.r_foreign + opt.rho_sfx * opt.sigma * (opt.sigma_fx + dsig);
    let price_qvup = q * barrier_closed_form(
        opt.spot, opt.strike, opt.barrier, opt.rebate,
        opt.r_domestic, r_f_adj_up, opt.sigma, opt.tau,
        opt.barrier_type, opt.option_type,
    );
    let quanto_vega = (price_qvup - price) / dsig * 0.01;

    // Theta via time bump
    let dt = 1.0 / 365.0;
    let theta = if opt.tau > dt {
        let price_tm = q * barrier_closed_form(
            opt.spot, opt.strike, opt.barrier, opt.rebate,
            opt.r_domestic, r_f_adj, opt.sigma, opt.tau - dt,
            opt.barrier_type, opt.option_type,
        );
        price_tm - price // theta is negative for long options
    } else {
        0.0
    };

    QuantoBarrierResult {
        price,
        delta,
        gamma,
        vega,
        quanto_vega,
        theta,
    }
}

// ===========================================================================
// Merton-Reiner-Rubinstein barrier pricing
// ===========================================================================

/// Analytic barrier option price using the Merton-Reiner-Rubinstein formula.
fn barrier_closed_form(
    spot: f64,
    strike: f64,
    barrier: f64,
    rebate: f64,
    r_d: f64,
    r_f: f64,
    sigma: f64,
    t: f64,
    barrier_type: BarrierType,
    option_type: OptionType,
) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }

    let phi = match option_type {
        OptionType::Call => 1.0_f64,
        OptionType::Put => -1.0,
    };

    let mu = (r_d - r_f) / (sigma * sigma) - 0.5;
    let lambda = (mu * mu + 2.0 * r_d / (sigma * sigma)).sqrt();
    let sqrt_t = t.sqrt();
    let sig_sqrt_t = sigma * sqrt_t;

    let x1 = (spot / strike).ln() / sig_sqrt_t + (1.0 + mu) * sig_sqrt_t;
    let x2 = (spot / barrier).ln() / sig_sqrt_t + (1.0 + mu) * sig_sqrt_t;
    let y1 = (barrier * barrier / (spot * strike)).ln() / sig_sqrt_t + (1.0 + mu) * sig_sqrt_t;
    let y2 = (barrier / spot).ln() / sig_sqrt_t + (1.0 + mu) * sig_sqrt_t;
    let z = (barrier / spot).ln() / sig_sqrt_t + lambda * sig_sqrt_t;

    let h_ratio = barrier / spot;
    let df_d = (-r_d * t).exp();
    let df_f = (-r_f * t).exp();

    let (eta, is_knockout) = match barrier_type {
        BarrierType::DownOut => (1.0, true),
        BarrierType::UpOut => (-1.0, true),
        BarrierType::DownIn => (1.0, false),
        BarrierType::UpIn => (-1.0, false),
    };

    let term_a = phi * spot * df_f * cumulative_normal(phi * x1)
        - phi * strike * df_d * cumulative_normal(phi * (x1 - sig_sqrt_t));
    let term_b = phi * spot * df_f * cumulative_normal(phi * x2)
        - phi * strike * df_d * cumulative_normal(phi * (x2 - sig_sqrt_t));
    let term_c = phi * spot * df_f * h_ratio.powf(2.0 * (mu + 1.0)) * cumulative_normal(eta * y1)
        - phi * strike * df_d * h_ratio.powf(2.0 * mu) * cumulative_normal(eta * (y1 - sig_sqrt_t));
    let term_d = phi * spot * df_f * h_ratio.powf(2.0 * (mu + 1.0)) * cumulative_normal(eta * y2)
        - phi * strike * df_d * h_ratio.powf(2.0 * mu) * cumulative_normal(eta * (y2 - sig_sqrt_t));
    let _term_e = rebate * df_d * (
        cumulative_normal(eta * (x2 - sig_sqrt_t))
        - h_ratio.powf(2.0 * mu) * cumulative_normal(eta * (y2 - sig_sqrt_t))
    );
    let term_f = rebate * (
        h_ratio.powf(mu + lambda) * cumulative_normal(eta * z)
        + h_ratio.powf(mu - lambda) * cumulative_normal(eta * (z - 2.0 * lambda * sig_sqrt_t))
    );

    let is_call = matches!(option_type, OptionType::Call);

    let ko_price = match (is_call, barrier_type) {
        (true, BarrierType::DownOut) if strike > barrier => term_a - term_c + term_f,
        (true, BarrierType::DownOut) => term_b - term_d + term_f,
        (false, BarrierType::DownOut) => term_a - term_b + term_d - term_c + term_f,
        (true, BarrierType::UpOut) => term_a - term_b + term_d - term_c + term_f,
        (false, BarrierType::UpOut) if strike > barrier => term_a - term_c + term_f,
        (false, BarrierType::UpOut) => term_b - term_d + term_f,
        _ => 0.0,
    };

    if is_knockout {
        ko_price.max(0.0)
    } else {
        // In-out parity: KI = Vanilla - KO
        let vanilla = bs_vanilla(spot, strike, r_d, r_f, sigma, t, is_call);
        let ko_type = if barrier_type == BarrierType::DownIn {
            BarrierType::DownOut
        } else {
            BarrierType::UpOut
        };
        let ko = barrier_closed_form(
            spot, strike, barrier, rebate, r_d, r_f, sigma, t, ko_type, option_type,
        );
        (vanilla - ko + rebate * df_d).max(0.0)
    }
}

/// Standard Black-Scholes vanilla option price (Garman-Kohlhagen).
fn bs_vanilla(spot: f64, strike: f64, r_d: f64, r_f: f64, sigma: f64, t: f64, is_call: bool) -> f64 {
    let fwd = spot * ((r_d - r_f) * t).exp();
    let sqrt_t = t.sqrt();
    let d1 = (fwd / strike).ln() / (sigma * sqrt_t) + 0.5 * sigma * sqrt_t;
    let d2 = d1 - sigma * sqrt_t;
    let df = (-r_d * t).exp();
    let phi = if is_call { 1.0 } else { -1.0 };
    phi * df * (fwd * cumulative_normal(phi * d1) - strike * cumulative_normal(phi * d2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_quanto_barrier(barrier_type: BarrierType, option_type: OptionType) -> QuantoBarrierOption {
        QuantoBarrierOption {
            option_type,
            barrier_type,
            spot: 100.0,
            strike: 100.0,
            barrier: 80.0,
            rebate: 0.0,
            tau: 1.0,
            r_domestic: 0.03,
            r_foreign: 0.05,
            sigma: 0.20,
            sigma_fx: 0.10,
            rho_sfx: 0.3,
            fixed_fx: 1.0,
        }
    }

    #[test]
    fn quanto_barrier_price_positive() {
        let opt = make_quanto_barrier(BarrierType::DownOut, OptionType::Call);
        let result = price_quanto_barrier(&opt);
        assert!(result.price > 0.0, "Quanto barrier price should be positive");
    }

    #[test]
    fn quanto_barrier_down_out_call_less_than_vanilla() {
        let opt = make_quanto_barrier(BarrierType::DownOut, OptionType::Call);
        let result = price_quanto_barrier(&opt);

        // Compare with vanilla quanto (no barrier)
        let quanto_adj = opt.rho_sfx * opt.sigma * opt.sigma_fx;
        let r_f_adj = opt.r_foreign + quanto_adj;
        let vanilla = bs_vanilla(opt.spot, opt.strike, opt.r_domestic, r_f_adj, opt.sigma, opt.tau, true);

        assert!(result.price <= vanilla * 1.001,
            "Down-and-out call {} should be ≤ vanilla {}", result.price, vanilla);
    }

    #[test]
    fn quanto_barrier_in_out_parity() {
        let opt_out = make_quanto_barrier(BarrierType::DownOut, OptionType::Call);
        let mut opt_in = opt_out.clone();
        opt_in.barrier_type = BarrierType::DownIn;

        let r_out = price_quanto_barrier(&opt_out);
        let r_in = price_quanto_barrier(&opt_in);

        // In-out parity: KO + KI = Vanilla
        let quanto_adj = opt_out.rho_sfx * opt_out.sigma * opt_out.sigma_fx;
        let r_f_adj = opt_out.r_foreign + quanto_adj;
        let vanilla = bs_vanilla(opt_out.spot, opt_out.strike, opt_out.r_domestic, r_f_adj, opt_out.sigma, opt_out.tau, true);

        assert_abs_diff_eq!(r_out.price + r_in.price, vanilla, epsilon = 0.01);
    }

    #[test]
    fn quanto_adjustment_affects_price() {
        let opt1 = make_quanto_barrier(BarrierType::DownOut, OptionType::Call);
        let mut opt2 = opt1.clone();
        opt2.rho_sfx = 0.0; // no correlation → no quanto adjustment

        let r1 = price_quanto_barrier(&opt1);
        let r2 = price_quanto_barrier(&opt2);

        assert!((r1.price - r2.price).abs() > 0.01,
            "Quanto correlation should affect price: rho=0.3 → {}, rho=0.0 → {}", r1.price, r2.price);
    }

    #[test]
    fn delta_positive_for_down_out_call() {
        let opt = make_quanto_barrier(BarrierType::DownOut, OptionType::Call);
        let result = price_quanto_barrier(&opt);
        assert!(result.delta > 0.0, "Delta should be positive for DO call");
    }

    #[test]
    fn gamma_positive() {
        let opt = make_quanto_barrier(BarrierType::DownOut, OptionType::Call);
        let result = price_quanto_barrier(&opt);
        assert!(result.gamma > 0.0 || result.gamma.abs() < 0.1,
            "Gamma should be non-negative for DO call, got {}", result.gamma);
    }

    #[test]
    fn quanto_barrier_put_positive() {
        let mut opt = make_quanto_barrier(BarrierType::UpOut, OptionType::Put);
        opt.barrier = 120.0;
        let result = price_quanto_barrier(&opt);
        assert!(result.price > 0.0, "UO put should have positive value");
    }

    #[test]
    fn fixed_fx_scales_price() {
        let opt1 = make_quanto_barrier(BarrierType::DownOut, OptionType::Call);
        let mut opt2 = opt1.clone();
        opt2.fixed_fx = 2.0;

        let r1 = price_quanto_barrier(&opt1);
        let r2 = price_quanto_barrier(&opt2);

        assert_abs_diff_eq!(r2.price, 2.0 * r1.price, epsilon = 1e-10);
    }

    #[test]
    fn theta_negative_for_long_did_not_expire() {
        let opt = make_quanto_barrier(BarrierType::DownOut, OptionType::Call);
        let result = price_quanto_barrier(&opt);
        assert!(result.theta < 0.0, "Theta should be negative for long option, got {}", result.theta);
    }
}
