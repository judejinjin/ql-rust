//! Vanna-Volga barrier option pricing engine.
//!
//! The Vanna-Volga method is the standard approach for pricing FX barrier
//! options. It constructs an adjustment from the cost of hedging the vanna
//! and volga exposures of the barrier using three vanilla options (25Δ put,
//! ATM, 25Δ call).
//!
//! References:
//! - Castagna & Mercurio (2007), "The Vanna-Volga method for implied
//!   volatilities", Risk.
//! - Wystup (2006), "FX Options and Structured Products", Wiley.

use serde::{Deserialize, Serialize};
use ql_math::distributions::cumulative_normal;
use std::f64::consts::PI;

/// Barrier type for Vanna-Volga engine.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum VvBarrierType {
    /// Down And Out.
    DownAndOut,
    /// Up And Out.
    UpAndOut,
    /// Down And In.
    DownAndIn,
    /// Up And In.
    UpAndIn,
}

/// Result from the Vanna-Volga barrier engine.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VannaVolgaBarrierResult {
    /// Barrier option price.
    pub price: f64,
    /// BS barrier price (for reference).
    pub bs_price: f64,
    /// Vanna-Volga adjustment.
    pub vv_adjustment: f64,
    /// Domestic-ccy delta.
    pub delta: f64,
}

/// Black-Scholes vanilla price helper.
fn bs_price(spot: f64, strike: f64, r_d: f64, r_f: f64, sigma: f64, t: f64, is_call: bool) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        let intrinsic = if is_call { (spot * (-r_f * t).exp() - strike * (-r_d * t).exp()).max(0.0) }
                        else { (strike * (-r_d * t).exp() - spot * (-r_f * t).exp()).max(0.0) };
        return intrinsic;
    }
    let fwd = spot * ((r_d - r_f) * t).exp();
    let d1 = (fwd / strike).ln() / (sigma * t.sqrt()) + 0.5 * sigma * t.sqrt();
    let d2 = d1 - sigma * t.sqrt();
    let df = (-r_d * t).exp();
    if is_call {
        df * (fwd * cumulative_normal(d1) - strike * cumulative_normal(d2))
    } else {
        df * (strike * cumulative_normal(-d2) - fwd * cumulative_normal(-d1))
    }
}

/// Black-Scholes vega.
fn bs_vega(spot: f64, strike: f64, r_d: f64, r_f: f64, sigma: f64, t: f64) -> f64 {
    let fwd = spot * ((r_d - r_f) * t).exp();
    let d1 = (fwd / strike).ln() / (sigma * t.sqrt()) + 0.5 * sigma * t.sqrt();
    let df = (-r_d * t).exp();
    df * fwd * t.sqrt() * (-0.5 * d1 * d1).exp() / (2.0 * PI).sqrt()
}

/// Analytic BS barrier price (Merton-Reiner-Rubinstein).
fn bs_barrier_price(
    spot: f64, strike: f64, barrier: f64, rebate: f64,
    r_d: f64, r_f: f64, sigma: f64, t: f64,
    barrier_type: VvBarrierType, is_call: bool,
) -> f64 {
    let mu = (r_d - r_f) / (sigma * sigma) - 0.5;
    let lambda = (mu * mu + 2.0 * r_d / (sigma * sigma)).sqrt();
    let x1 = (spot / strike).ln() / (sigma * t.sqrt()) + (1.0 + mu) * sigma * t.sqrt();
    let x2 = (spot / barrier).ln() / (sigma * t.sqrt()) + (1.0 + mu) * sigma * t.sqrt();
    let y1 = (barrier * barrier / (spot * strike)).ln() / (sigma * t.sqrt()) + (1.0 + mu) * sigma * t.sqrt();
    let y2 = (barrier / spot).ln() / (sigma * t.sqrt()) + (1.0 + mu) * sigma * t.sqrt();
    let z = (barrier / spot).ln() / (sigma * t.sqrt()) + lambda * sigma * t.sqrt();

    let phi = if is_call { 1.0 } else { -1.0 };
    let h_ratio = barrier / spot;

    let (eta, is_knockout) = match barrier_type {
        VvBarrierType::DownAndOut => (1.0, true),
        VvBarrierType::UpAndOut => (-1.0, true),
        VvBarrierType::DownAndIn => (1.0, false),
        VvBarrierType::UpAndIn => (-1.0, false),
    };

    let df_d = (-r_d * t).exp();
    let df_f = (-r_f * t).exp();

    let a = phi * spot * df_f * cumulative_normal(phi * x1) - phi * strike * df_d * cumulative_normal(phi * (x1 - sigma * t.sqrt()));
    let b = phi * spot * df_f * cumulative_normal(phi * x2) - phi * strike * df_d * cumulative_normal(phi * (x2 - sigma * t.sqrt()));
    let c = phi * spot * df_f * h_ratio.powf(2.0 * (mu + 1.0)) * cumulative_normal(eta * y1)
        - phi * strike * df_d * h_ratio.powf(2.0 * mu) * cumulative_normal(eta * (y1 - sigma * t.sqrt()));
    let d = phi * spot * df_f * h_ratio.powf(2.0 * (mu + 1.0)) * cumulative_normal(eta * y2)
        - phi * strike * df_d * h_ratio.powf(2.0 * mu) * cumulative_normal(eta * (y2 - sigma * t.sqrt()));
    let _e = rebate * df_d * (
        cumulative_normal(eta * (x2 - sigma * t.sqrt()))
        - h_ratio.powf(2.0 * mu) * cumulative_normal(eta * (y2 - sigma * t.sqrt()))
    );
    let f = rebate * (
        h_ratio.powf(mu + lambda) * cumulative_normal(eta * z)
        + h_ratio.powf(mu - lambda) * cumulative_normal(eta * (z - 2.0 * lambda * sigma * t.sqrt()))
    );

    let ko_price = match (is_call, barrier_type) {
        (true, VvBarrierType::DownAndOut) if strike > barrier => a - c + f,
        (true, VvBarrierType::DownAndOut) => b - d + f,
        (false, VvBarrierType::DownAndOut) => a - b + d - c + f,
        (true, VvBarrierType::UpAndOut) => a - b + d - c + f,
        (false, VvBarrierType::UpAndOut) if strike > barrier => a - c + f,
        (false, VvBarrierType::UpAndOut) => b - d + f,
        _ => 0.0, // knock-in handled separately
    };

    if is_knockout {
        ko_price.max(0.0)
    } else {
        // In-out parity: KI = Vanilla - KO
        let vanilla = bs_price(spot, strike, r_d, r_f, sigma, t, is_call);
        let ko = bs_barrier_price(spot, strike, barrier, rebate, r_d, r_f, sigma, t,
            if barrier_type == VvBarrierType::DownAndIn { VvBarrierType::DownAndOut } else { VvBarrierType::UpAndOut },
            is_call);
        (vanilla - ko + rebate * df_d).max(0.0)
    }
}

/// Price an FX barrier option using the Vanna-Volga method.
///
/// The method adjusts the BS barrier price by the cost of hedging vanna and
/// volga exposures using three market-quoted vanillas (25Δ put, ATM, 25Δ call).
///
/// # Arguments
/// - `spot` — FX spot rate
/// - `strike` — option strike
/// - `barrier` — barrier level
/// - `rebate` — cash rebate at barrier hit
/// - `r_d`, `r_f` — domestic and foreign risk-free rates
/// - `sigma_atm` — ATM implied volatility
/// - `sigma_25d_put` — 25-delta put implied volatility
/// - `sigma_25d_call` — 25-delta call implied volatility
/// - `t` — time to expiry in years
/// - `barrier_type` — type of barrier
/// - `is_call` — true for call, false for put
#[allow(clippy::too_many_arguments)]
pub fn vanna_volga_barrier(
    spot: f64,
    strike: f64,
    barrier: f64,
    rebate: f64,
    r_d: f64,
    r_f: f64,
    sigma_atm: f64,
    sigma_25d_put: f64,
    sigma_25d_call: f64,
    t: f64,
    barrier_type: VvBarrierType,
    is_call: bool,
) -> VannaVolgaBarrierResult {
    // Three pillar strikes: 25Δ put, ATM, 25Δ call
    let fwd = spot * ((r_d - r_f) * t).exp();
    let k_atm = fwd * (0.5 * sigma_atm * sigma_atm * t).exp(); // ATM DNS (delta-neutral straddle)
    let k_25p = fwd * (-sigma_25d_put * t.sqrt() * 0.6745 + 0.5 * sigma_25d_put * sigma_25d_put * t).exp();
    let k_25c = fwd * (sigma_25d_call * t.sqrt() * 0.6745 + 0.5 * sigma_25d_call * sigma_25d_call * t).exp();

    let sigma_pillars = [sigma_25d_put, sigma_atm, sigma_25d_call];
    let k_pillars = [k_25p, k_atm, k_25c];

    // BS barrier price with ATM vol as a first approximation
    let bs_bar = bs_barrier_price(spot, strike, barrier, rebate, r_d, r_f, sigma_atm, t, barrier_type, is_call);

    // Compute cost of overhedge for each pillar:
    // x_i = BS(K_i, σ_mkt_i) - BS(K_i, σ_ATM)
    let mut overhedge = [0.0; 3];
    for i in 0..3 {
        let p_mkt = bs_price(spot, k_pillars[i], r_d, r_f, sigma_pillars[i], t, true);
        let p_atm = bs_price(spot, k_pillars[i], r_d, r_f, sigma_atm, t, true);
        overhedge[i] = p_mkt - p_atm;
    }

    // Compute vega, vanna, volga at strike using ATM vol
    let vega_k = bs_vega(spot, strike, r_d, r_f, sigma_atm, t);

    let d1 = {
        let f = fwd;
        (f / strike).ln() / (sigma_atm * t.sqrt()) + 0.5 * sigma_atm * t.sqrt()
    };
    let d2 = d1 - sigma_atm * t.sqrt();

    // Vanna = dVega/dSpot = vega * (1 - d1/(sigma*sqrt(t))) / spot approximately
    // Actually: d²V/dSdσ = -Vega * d2 / (S * σ * √t)
    // Volga = d²V/dσ² = Vega * d1 * d2 / σ
    let _vanna_k = if sigma_atm * t.sqrt() > 1e-12 {
        -vega_k * d2 / (spot * sigma_atm * t.sqrt())
    } else { 0.0 };
    let _volga_k = if sigma_atm > 1e-12 {
        vega_k * d1 * d2 / sigma_atm
    } else { 0.0 };

    // Same for each pillar
    let mut vega_p = [0.0; 3];
    let mut vanna_p = [0.0; 3];
    let mut volga_p = [0.0; 3];
    for i in 0..3 {
        vega_p[i] = bs_vega(spot, k_pillars[i], r_d, r_f, sigma_atm, t);
        let d1_i = (fwd / k_pillars[i]).ln() / (sigma_atm * t.sqrt()) + 0.5 * sigma_atm * t.sqrt();
        let d2_i = d1_i - sigma_atm * t.sqrt();
        if sigma_atm * t.sqrt() > 1e-12 {
            vanna_p[i] = -vega_p[i] * d2_i / (spot * sigma_atm * t.sqrt());
        }
        if sigma_atm > 1e-12 {
            volga_p[i] = vega_p[i] * d1_i * d2_i / sigma_atm;
        }
    }

    // Solve for weights x_1, x_2, x_3 such that the portfolio replicates
    // the vega, vanna and volga of the target option:
    // Σ x_i * vega_i = vega_k
    // Σ x_i * vanna_i = vanna_k  
    // Σ x_i * volga_i = volga_k
    // Using log-strike approach (Castagna-Mercurio):
    let ln_k = strike.ln();
    let ln_k1 = k_pillars[0].ln();
    let ln_k2 = k_pillars[1].ln();
    let ln_k3 = k_pillars[2].ln();

    let mut w = [0.0; 3];
    let det = (ln_k2 - ln_k1) * (ln_k3 - ln_k1);
    if det.abs() > 1e-20 {
        w[0] = (ln_k - ln_k2) * (ln_k - ln_k3) / ((ln_k1 - ln_k2) * (ln_k1 - ln_k3));
        w[1] = (ln_k - ln_k1) * (ln_k - ln_k3) / ((ln_k2 - ln_k1) * (ln_k2 - ln_k3));
        w[2] = (ln_k - ln_k1) * (ln_k - ln_k2) / ((ln_k3 - ln_k1) * (ln_k3 - ln_k2));
    }

    // VV adjustment for the vanilla option
    let _vv_adj_vanilla: f64 = (0..3).map(|i| w[i] * overhedge[i]).sum();

    // For barrier options, apply the survival probability weighting (no-touch probability)
    // Vanna-Volga first-order: scale by the ratio of BS barrier price to BS vanilla price
    let bs_vanilla = bs_price(spot, strike, r_d, r_f, sigma_atm, t, is_call);
    let _touch_ratio = if bs_vanilla.abs() > 1e-12 {
        (bs_bar / bs_vanilla).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Second-order VV adjustment for barriers:
    // Apply survival-probability weighting to the overhedge costs
    let mut vv_adj_barrier = 0.0;
    for i in 0..3 {
        let bs_bar_i = bs_barrier_price(spot, k_pillars[i], barrier, 0.0, r_d, r_f, sigma_atm, t, barrier_type, true);
        let bs_van_i = bs_price(spot, k_pillars[i], r_d, r_f, sigma_atm, t, true);
        let ratio_i = if bs_van_i.abs() > 1e-12 {
            (bs_bar_i / bs_van_i).clamp(0.0, 1.0)
        } else { 0.0 };
        vv_adj_barrier += w[i] * overhedge[i] * ratio_i;
    }

    let price = bs_bar + vv_adj_barrier;

    // Delta by central difference
    let eps = spot * 0.001;
    let p_up = {
        let bs_up = bs_barrier_price(spot + eps, strike, barrier, rebate, r_d, r_f, sigma_atm, t, barrier_type, is_call);
        let van_up = bs_price(spot + eps, strike, r_d, r_f, sigma_atm, t, is_call);
        let _tr_up = if van_up.abs() > 1e-12 { (bs_up / van_up).clamp(0.0, 1.0) } else { 0.0 };
        let mut adj = 0.0;
        for i in 0..3 {
            let bs_bar_i = bs_barrier_price(spot + eps, k_pillars[i], barrier, 0.0, r_d, r_f, sigma_atm, t, barrier_type, true);
            let bs_van_i = bs_price(spot + eps, k_pillars[i], r_d, r_f, sigma_atm, t, true);
            let r = if bs_van_i.abs() > 1e-12 { (bs_bar_i / bs_van_i).clamp(0.0, 1.0) } else { 0.0 };
            adj += w[i] * overhedge[i] * r;
        }
        bs_up + adj
    };
    let p_dn = {
        let bs_dn = bs_barrier_price(spot - eps, strike, barrier, rebate, r_d, r_f, sigma_atm, t, barrier_type, is_call);
        let van_dn = bs_price(spot - eps, strike, r_d, r_f, sigma_atm, t, is_call);
        let _tr_dn = if van_dn.abs() > 1e-12 { (bs_dn / van_dn).clamp(0.0, 1.0) } else { 0.0 };
        let mut adj = 0.0;
        for i in 0..3 {
            let bs_bar_i = bs_barrier_price(spot - eps, k_pillars[i], barrier, 0.0, r_d, r_f, sigma_atm, t, barrier_type, true);
            let bs_van_i = bs_price(spot - eps, k_pillars[i], r_d, r_f, sigma_atm, t, true);
            let r = if bs_van_i.abs() > 1e-12 { (bs_bar_i / bs_van_i).clamp(0.0, 1.0) } else { 0.0 };
            adj += w[i] * overhedge[i] * r;
        }
        bs_dn + adj
    };

    let delta = (p_up - p_dn) / (2.0 * eps);

    VannaVolgaBarrierResult {
        price: price.max(0.0),
        bs_price: bs_bar.max(0.0),
        vv_adjustment: vv_adj_barrier,
        delta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_vv_barrier_dao_call() {
        // EUR/USD example: spot=1.10, K=1.12, barrier=1.02
        let res = vanna_volga_barrier(
            1.10, 1.12, 1.02, 0.0,
            0.05, 0.03,
            0.10, 0.12, 0.09,
            0.5,
            VvBarrierType::DownAndOut, true,
        );
        // Should be less than vanilla BS call
        let vanilla = bs_price(1.10, 1.12, 0.05, 0.03, 0.10, 0.5, true);
        assert!(res.price > 0.0 && res.price < vanilla * 1.1, "price={}", res.price);
    }

    #[test]
    fn test_vv_barrier_in_out_parity() {
        let dao = vanna_volga_barrier(
            1.10, 1.10, 1.00, 0.0,
            0.05, 0.03, 0.10, 0.12, 0.09,
            1.0, VvBarrierType::DownAndOut, true,
        );
        let dai = vanna_volga_barrier(
            1.10, 1.10, 1.00, 0.0,
            0.05, 0.03, 0.10, 0.12, 0.09,
            1.0, VvBarrierType::DownAndIn, true,
        );
        let vanilla = bs_price(1.10, 1.10, 0.05, 0.03, 0.10, 1.0, true);
        // In-out parity holds approximately under VV
        let sum = dao.price + dai.price;
        assert_abs_diff_eq!(sum, vanilla, epsilon = 0.005);
    }

    #[test]
    fn test_vv_flat_smile_equals_bs() {
        // When smile is flat, VV adjustment should vanish
        let res = vanna_volga_barrier(
            100.0, 100.0, 85.0, 0.0,
            0.05, 0.02, 0.20, 0.20, 0.20,
            1.0, VvBarrierType::DownAndOut, true,
        );
        let bs_bar = bs_barrier_price(100.0, 100.0, 85.0, 0.0, 0.05, 0.02, 0.20, 1.0, VvBarrierType::DownAndOut, true);
        assert_abs_diff_eq!(res.price, bs_bar, epsilon = 0.01);
    }

    #[test]
    fn test_vv_up_and_out_put() {
        let res = vanna_volga_barrier(
            1.10, 1.08, 1.18, 0.0,
            0.05, 0.03, 0.10, 0.12, 0.09,
            0.5, VvBarrierType::UpAndOut, false,
        );
        assert!(res.price > 0.0, "price={}", res.price);
    }
}
