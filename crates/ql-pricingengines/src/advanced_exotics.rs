//! Advanced exotic option pricing engines.
//!
//! ## Quanto Options
//! FX-adjusted options paying in a domestic currency on a foreign asset.
//! Uses the Garman-Kohlhagen style quanto adjustment.
//!
//! ## Power Options
//! Payoff = max(S^α − K, 0). Closed-form BS modification.
//!
//! ## Forward-Start Options
//! Strike set at α × S(t₁) at future date t₁, priced at time 0.
//!
//! ## Digital Barrier Options (One-Touch / No-Touch)
//! Binary payoffs triggered (or not) by a barrier breach.

use std::f64::consts::{FRAC_1_SQRT_2, PI};

// ===========================================================================
//  Normal distribution helpers
// ===========================================================================

fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x * FRAC_1_SQRT_2))
}

fn erf(x: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26 approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let t = 1.0 / (1.0 + p * x.abs());
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

// ===========================================================================
//  Quanto Options
// ===========================================================================

/// Result from quanto option pricing.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QuantoResult {
    pub price: f64,
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
}

/// Price a quanto European option (foreign asset, domestic settlement).
///
/// The quanto-adjusted drift replaces the foreign risk-free rate with
/// `r_f − ρ σ_S σ_fx`, keeping the option's currency dimension collapsed.
///
/// # Parameters
/// - `spot`: foreign asset spot price (in foreign currency)
/// - `strike`: strike price (in foreign currency)
/// - `r_d`: domestic risk-free rate  
/// - `r_f`: foreign risk-free rate
/// - `vol_s`: foreign asset volatility
/// - `vol_fx`: FX rate volatility (domestic per foreign)
/// - `rho`: correlation between asset and FX rate
/// - `t`: time to expiry (years)
/// - `fx_rate`: current FX rate (domestic per 1 foreign) — for PV conversion
/// - `is_call`: true for call, false for put
pub fn quanto_european(
    spot: f64,
    strike: f64,
    r_d: f64,
    r_f: f64,
    vol_s: f64,
    vol_fx: f64,
    rho: f64,
    t: f64,
    fx_rate: f64,
    is_call: bool,
) -> QuantoResult {
    // Quanto-adjusted cost of carry
    let b_q = r_f - rho * vol_s * vol_fx;
    let df = (-r_d * t).exp();

    let d1 = ((spot / strike).ln() + (b_q + 0.5 * vol_s * vol_s) * t) / (vol_s * t.sqrt());
    let d2 = d1 - vol_s * t.sqrt();

    let (price, delta) = if is_call {
        let p = spot * ((b_q - r_d) * t).exp() * norm_cdf(d1)
            - strike * df * norm_cdf(d2);
        let d = ((b_q - r_d) * t).exp() * norm_cdf(d1);
        (p, d)
    } else {
        let p = strike * df * norm_cdf(-d2)
            - spot * ((b_q - r_d) * t).exp() * norm_cdf(-d1);
        let d = -((b_q - r_d) * t).exp() * norm_cdf(-d1);
        (p, d)
    };

    let gamma_val = ((b_q - r_d) * t).exp() * norm_pdf(d1) / (spot * vol_s * t.sqrt());
    let vega_val = spot * ((b_q - r_d) * t).exp() * norm_pdf(d1) * t.sqrt();

    QuantoResult {
        price: price * fx_rate,
        delta: delta * fx_rate,
        gamma: gamma_val * fx_rate,
        vega: vega_val * fx_rate,
    }
}

// ===========================================================================
//  Power Options
// ===========================================================================

/// Result from power option pricing.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PowerResult {
    pub price: f64,
}

/// Price a power option: payoff = max(S^α − K, 0) for call.
///
/// Uses the substitution F = S^α with adjusted vol σ_p = α × σ.
pub fn power_option(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    alpha: f64,
    is_call: bool,
) -> PowerResult {
    let vol_p = (alpha * vol).abs();
    let s_alpha = spot.powf(alpha);
    // Risk-neutral drift for S^α: r_α = α(r-q) + ½α(α-1)σ²
    let drift_alpha = alpha * (r - q) + 0.5 * alpha * (alpha - 1.0) * vol * vol;
    let forward = s_alpha * (drift_alpha * t).exp();
    let df = (-r * t).exp();

    if vol_p * t.sqrt() < 1e-15 {
        let intrinsic = if is_call {
            (forward - strike).max(0.0)
        } else {
            (strike - forward).max(0.0)
        };
        return PowerResult { price: df * intrinsic };
    }

    let d1 = ((forward / strike).ln() + 0.5 * vol_p * vol_p * t) / (vol_p * t.sqrt());
    let d2 = d1 - vol_p * t.sqrt();

    let price = if is_call {
        df * (forward * norm_cdf(d1) - strike * norm_cdf(d2))
    } else {
        df * (strike * norm_cdf(-d2) - forward * norm_cdf(-d1))
    };

    PowerResult { price }
}

// ===========================================================================
//  Forward-Start Options
// ===========================================================================

/// Result from forward-start option pricing.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ForwardStartResult {
    pub price: f64,
}

/// Price a forward-start European option.
///
/// Strike is set at `t1` as `K = alpha × S(t1)`. The option expires at `t2`.
/// Using Rubinstein (1990): the forward-start call reduces to a scaled
/// standard BS call.
///
/// # Parameters
/// - `spot`: current spot price
/// - `r`: risk-free rate
/// - `q`: dividend yield
/// - `vol`: volatility
/// - `t1`: time to strike-setting date
/// - `t2`: time to expiry (t2 > t1)
/// - `alpha`: strike ratio (K = alpha × S(t1))
/// - `is_call`: true for call
pub fn forward_start_option(
    spot: f64,
    r: f64,
    q: f64,
    vol: f64,
    t1: f64,
    t2: f64,
    alpha: f64,
    is_call: bool,
) -> ForwardStartResult {
    let tau = t2 - t1; // remaining time after strike set
    if tau <= 0.0 {
        return ForwardStartResult { price: 0.0 };
    }

    // Standard BS on a unit-spot call with strike = alpha
    let d1 = ((1.0 / alpha).ln() + (r - q + 0.5 * vol * vol) * tau) / (vol * tau.sqrt());
    let d2 = d1 - vol * tau.sqrt();

    // Forward-start = e^{-q*t1} × S × BS(1, alpha, r, q, vol, tau)
    let scale = spot * (-q * t1).exp();

    let price = if is_call {
        scale * ((-q * tau).exp() * norm_cdf(d1) - alpha * (-r * tau).exp() * norm_cdf(d2))
    } else {
        scale * (alpha * (-r * tau).exp() * norm_cdf(-d2) - (-q * tau).exp() * norm_cdf(-d1))
    };

    ForwardStartResult { price }
}

// ===========================================================================
//  Digital Barrier Options (One-Touch / No-Touch)
// ===========================================================================

/// Result from digital barrier option pricing.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DigitalBarrierResult {
    pub price: f64,
}

/// Type of digital barrier option.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DigitalBarrierType {
    /// One-Touch: pays rebate if barrier is hit at any time before expiry.
    OneTouch,
    /// No-Touch: pays rebate only if barrier is never hit before expiry.
    NoTouch,
}

/// Price a digital barrier option (one-touch / no-touch).
///
/// Uses closed-form formulas for continuous-monitoring barrier digitals.
///
/// # Parameters
/// - `spot`: current spot
/// - `barrier`: barrier level
/// - `rebate`: amount paid (either at hit or at expiry)
/// - `r`: risk-free rate
/// - `q`: dividend yield
/// - `vol`: volatility
/// - `t`: time to expiry
/// - `barrier_type`: `OneTouch` or `NoTouch`
/// - `is_upper`: true if barrier is above spot (up), false if below (down)
pub fn digital_barrier(
    spot: f64,
    barrier: f64,
    rebate: f64,
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    barrier_type: DigitalBarrierType,
    is_upper: bool,
) -> DigitalBarrierResult {
    let sig2 = vol * vol;
    // mu = (r - q - σ²/2) / σ²
    let mu = (r - q - 0.5 * sig2) / sig2;
    let lambda = ((mu * sig2).powi(2) + 2.0 * r * sig2).sqrt() / sig2;

    let _h = (barrier / spot).ln() / vol;
    let sqrt_t = t.sqrt();

    // One-touch formula (pays $1 at hit, discounted to PV)
    // P(hit) = (H/S)^{μ+λ} N(η z₁) + (H/S)^{μ-λ} N(η z₂)
    // where z₁ = [ln(H/S) + λσ²t] / (σ√t), z₂ = [ln(H/S) - λσ²t] / (σ√t)
    let eta = if is_upper { -1.0 } else { 1.0 }; // barrier direction
    let ratio = barrier / spot;

    let z1 = (ratio.ln() + lambda * sig2 * t) / (vol * sqrt_t);
    let z2 = (ratio.ln() - lambda * sig2 * t) / (vol * sqrt_t);

    let term1 = ratio.powf(mu + lambda) * norm_cdf(eta * z1);
    let term2 = ratio.powf(mu - lambda) * norm_cdf(eta * z2);

    let one_touch_pv = rebate * (term1 + term2);

    let price = match barrier_type {
        DigitalBarrierType::OneTouch => one_touch_pv,
        DigitalBarrierType::NoTouch => rebate * (-r * t).exp() - one_touch_pv,
    };

    DigitalBarrierResult {
        price: price.max(0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn quanto_call_positive() {
        let r = quanto_european(
            100.0, 100.0, 0.02, 0.05, 0.20, 0.10, 0.3, 1.0, 1.5, true,
        );
        assert!(r.price > 0.0, "Quanto call should be positive");
        assert!(r.delta > 0.0, "Quanto call delta should be positive");
    }

    #[test]
    fn quanto_put_call_parity() {
        let c = quanto_european(
            100.0, 100.0, 0.02, 0.05, 0.20, 0.10, 0.3, 1.0, 1.5, true,
        );
        let p = quanto_european(
            100.0, 100.0, 0.02, 0.05, 0.20, 0.10, 0.3, 1.0, 1.5, false,
        );
        // Quanto put-call parity: C - P = fx × [S × e^{(b-r)T} - K × e^{-rT}]
        let b_q = 0.05 - 0.3 * 0.20 * 0.10;
        let parity_rhs = 1.5 * (100.0_f64 * ((b_q - 0.02) * 1.0_f64).exp() - 100.0_f64 * (-0.02_f64 * 1.0_f64).exp());
        assert_abs_diff_eq!(c.price - p.price, parity_rhs, epsilon = 0.01);
    }

    #[test]
    fn power_option_alpha_1_matches_bs() {
        // With alpha=1, power option = standard BS
        let p = power_option(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 1.0, true);
        // Standard BS ATM call: S=100,K=100,r=5%,q=0%,vol=20%,T=1
        // forward = 100*exp(0.05) ≈ 105.13, d1≈0.35, d2≈0.15
        // price ≈ 10.45
        assert!(p.price > 8.0 && p.price < 13.0,
            "Power(alpha=1) should match BS, got {}", p.price);
    }

    #[test]
    fn power_option_alpha_2() {
        let p = power_option(100.0, 10000.0, 0.05, 0.0, 0.20, 1.0, 2.0, true);
        assert!(p.price > 0.0, "Power call should be positive");
    }

    #[test]
    fn forward_start_positive() {
        let r = forward_start_option(100.0, 0.05, 0.02, 0.20, 0.5, 1.0, 1.0, true);
        assert!(r.price > 0.0, "Forward-start ATM call should be positive");
    }

    #[test]
    fn forward_start_versus_spot() {
        // Forward-start with t1=0 should be very close to standard BS
        let fs = forward_start_option(100.0, 0.05, 0.0, 0.20, 0.001, 1.0, 1.0, true);
        let bs = power_option(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 1.0, true);
        assert_abs_diff_eq!(fs.price, bs.price, epsilon = 0.5);
    }

    #[test]
    fn one_touch_down_positive() {
        let r = digital_barrier(
            100.0, 90.0, 1.0, 0.05, 0.0, 0.20, 1.0,
            DigitalBarrierType::OneTouch, false,
        );
        assert!(r.price > 0.0, "One-touch should have positive value");
        assert!(r.price <= 1.0, "One-touch should not exceed rebate");
    }

    #[test]
    fn no_touch_plus_one_touch_equals_pv_rebate() {
        let ot = digital_barrier(
            100.0, 90.0, 1.0, 0.05, 0.0, 0.20, 1.0,
            DigitalBarrierType::OneTouch, false,
        );
        let nt = digital_barrier(
            100.0, 90.0, 1.0, 0.05, 0.0, 0.20, 1.0,
            DigitalBarrierType::NoTouch, false,
        );
        let pv_rebate = (-0.05 * 1.0_f64).exp();
        assert_abs_diff_eq!(ot.price + nt.price, pv_rebate, epsilon = 0.01);
    }

    #[test]
    fn one_touch_up_positive() {
        let r = digital_barrier(
            100.0, 110.0, 1.0, 0.05, 0.0, 0.20, 1.0,
            DigitalBarrierType::OneTouch, true,
        );
        assert!(r.price > 0.0);
    }
}
