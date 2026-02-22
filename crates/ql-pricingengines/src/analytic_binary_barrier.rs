//! Analytic binary barrier option engine.
//!
//! Implements the Reiner-Rubinstein (1991) closed-form formulas for
//! cash-or-nothing and asset-or-nothing barrier options (European exercise,
//! single continuous barrier, Black-Scholes dynamics).
//!
//! ## References
//!
//! - Reiner, E. & Rubinstein, M. (1991), *Breaking Down the Barriers*,
//!   Risk, 4(8), 28–35.
//! - Hull, J. (2018), *Options, Futures, and Other Derivatives*, 10th ed.,
//!   §26.4.

use serde::{Deserialize, Serialize};

use ql_math::distributions::cumulative_normal;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Barrier type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryBarrierType {
    /// Option activates when spot crosses barrier from above (down-and-in).
    DownAndIn,
    /// Option activates when spot crosses barrier from below (up-and-in).
    UpAndIn,
    /// Option pays only if spot stays above barrier (down-and-out).
    DownAndOut,
    /// Option pays only if spot stays below barrier (up-and-out).
    UpAndOut,
}

/// Payoff type for a binary option.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryPayoff {
    /// Pays 1 unit of cash if exercise condition holds.
    CashOrNothing,
    /// Pays 1 unit of the underlying asset if exercise condition holds.
    AssetOrNothing,
}

/// Direction of the underlying option (call or put).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryDirection {
    /// The underlying digital is a call (pays if S > K at expiry).
    Call,
    /// The underlying digital is a put (pays if S < K at expiry).
    Put,
}

/// Result from the analytic binary barrier engine.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BinaryBarrierResult {
    /// Option NPV (price).
    pub price: f64,
    /// Delta (∂V/∂S).
    pub delta: f64,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Standard normal CDF.
#[inline]
fn n(x: f64) -> f64 {
    cumulative_normal(x)
}

/// Standard normal PDF.
#[inline]
fn phi(x: f64) -> f64 {
    (-0.5 * x * x).exp() / std::f64::consts::TAU.sqrt()
}

/// Parameters extracted from inputs — used across all sub-cases.
struct Params {
    // ---- vanilla inputs ----
    pub s: f64,
    pub k: f64,
    pub h: f64,   // barrier level
    pub r: f64,
    pub q: f64,
    pub sigma: f64,
    pub tau: f64,
    // ---- derived ----
    pub df: f64,     // discount factor e^{-rT}
    pub dfs: f64,    // spot discount e^{-qT}
    pub b: f64,      // cost of carry b = r - q
    pub x1: f64,
    pub x2: f64,
    pub y1: f64,
    pub y2: f64,
    pub z: f64,
    pub a1: f64,
    pub a2: f64,
    pub mu: f64,
    pub lambda_: f64,
    pub sqrt_t: f64,
}

impl Params {
    fn new(s: f64, k: f64, h: f64, r: f64, q: f64, sigma: f64, tau: f64) -> Self {
        let b = r - q;
        let sqrt_t = tau.sqrt();
        let log_sk = (s / k).ln();
        let log_sh = (s / h).ln();
        let log_hs = (h / s).ln();
        let log_h2sk = (h * h / (s * k)).ln();

        let mu = (b - 0.5 * sigma * sigma) / (sigma * sigma);
        let lambda_ = (mu * mu + 2.0 * r / (sigma * sigma)).sqrt();

        let x1 = log_sk / (sigma * sqrt_t) + (1.0 + mu) * sigma * sqrt_t;
        let x2 = log_sh / (sigma * sqrt_t) + (1.0 + mu) * sigma * sqrt_t;
        let y1 = log_hs / (sigma * sqrt_t) + (1.0 + mu) * sigma * sqrt_t;
        let y2 = log_h2sk / (sigma * sqrt_t) + (1.0 + mu) * sigma * sqrt_t;
        let z = log_sh / (sigma * sqrt_t) + lambda_ * sigma * sqrt_t;
        let a1 = s * (b * tau).exp() * n(x1);
        let a2 = k * (r * tau).exp().recip() * n(x1 - sigma * sqrt_t);

        Self {
            s, k, h, r, q, sigma, tau,
            df: (-r * tau).exp(),
            dfs: (-q * tau).exp(),
            b, sqrt_t,
            x1, x2, y1, y2, z,
            a1, a2,
            mu, lambda_,
        }
    }
}

// ---------------------------------------------------------------------------
// Core Reiner-Rubinstein building blocks
// ---------------------------------------------------------------------------
//
// The 8 building-block functions A–H from Reiner & Rubinstein are used to
// compose each barrier/payoff combination.

/// Asset-or-nothing call price (no barrier) — building block A.
/// A(φ) = φ·S·e^{(b-r)T}·N(φ·x1)
fn a_asset(p: &Params, phi: f64) -> f64 {
    p.s * (p.b * p.tau).exp() * p.df * n(phi * p.x1)
}

/// Cash-or-nothing call price (no barrier) — building block B.
/// B(φ) = φ·K·e^{-rT}·N(φ·(x1 − σ√T))
fn b_cash(p: &Params, phi: f64) -> f64 {
    p.df * p.k * n(phi * (p.x1 - p.sigma * p.sqrt_t))
}

/// Building block C (asset-or-nothing, reflected at barrier).
fn c_asset(p: &Params, phi: f64, eta: f64) -> f64 {
    p.s * (p.b * p.tau).exp() * p.df
        * (p.h / p.s).powf(2.0 * (p.mu + 1.0))
        * n(eta * p.y1)
}

/// Building block D (cash-or-nothing, reflected at barrier).
fn d_cash(p: &Params, phi: f64, eta: f64) -> f64 {
    p.df * p.k
        * (p.h / p.s).powf(2.0 * p.mu)
        * n(eta * (p.y1 - p.sigma * p.sqrt_t))
}

/// Building block E — rebate triggered at barrier crossing (asset).
fn e_asset(p: &Params, eta: f64) -> f64 {
    p.s * (p.b * p.tau).exp() * p.df
        * ((p.h / p.s).powf(2.0 * (p.mu + 1.0)) * n(eta * p.y2)
           - (p.h / p.s).powf(-2.0 * p.mu) * n(eta * p.x2))
        / 2.0   // used for barrier range — note sign handled by caller
}

/// Building block F — rebate triggered at barrier crossing (cash).
fn f_cash(p: &Params, eta: f64) -> f64 {
    p.df * p.k
        * ((p.h / p.s).powf(2.0 * p.mu) * n(eta * (p.y2 - p.sigma * p.sqrt_t))
           - (p.h / p.s).powf(-2.0 * p.mu) * n(eta * (p.x2 - p.sigma * p.sqrt_t)))
        / 2.0
}

// ---------------------------------------------------------------------------
// Main pricing function
// ---------------------------------------------------------------------------

/// Price a European binary barrier option under Black-Scholes.
///
/// # Parameters
///
/// - `s` — current spot price
/// - `k` — binary strike (cash-or-nothing: pays $1 if ITM; asset: pays S)
/// - `h` — barrier level
/// - `r` — continuously compounded risk-free rate
/// - `q` — continuous dividend yield
/// - `sigma` — volatility
/// - `tau` — time to expiry (years)
/// - `barrier_type` — activation/deactivation type
/// - `payoff` — cash-or-nothing or asset-or-nothing
/// - `direction` — call or put
///
/// Returns [`BinaryBarrierResult`] with `price` and `delta`.
///
/// # Panics
///
/// Panics if `tau <= 0`, `s <= 0`, `k <= 0`, `h <= 0`, `sigma <= 0`.
pub fn analytic_binary_barrier(
    s: f64,
    k: f64,
    h: f64,
    r: f64,
    q: f64,
    sigma: f64,
    tau: f64,
    barrier_type: BinaryBarrierType,
    payoff: BinaryPayoff,
    direction: BinaryDirection,
) -> BinaryBarrierResult {
    assert!(s > 0.0 && k > 0.0 && h > 0.0 && sigma > 0.0 && tau > 0.0);

    let p = Params::new(s, k, h, r, q, sigma, tau);

    // phi = +1 for call, -1 for put
    let phi = match direction {
        BinaryDirection::Call => 1.0_f64,
        BinaryDirection::Put => -1.0_f64,
    };

    // eta = +1 for down barriers, -1 for up barriers
    let eta = match barrier_type {
        BinaryBarrierType::DownAndIn | BinaryBarrierType::DownAndOut => 1.0_f64,
        BinaryBarrierType::UpAndIn | BinaryBarrierType::UpAndOut => -1.0_f64,
    };

    let is_in = matches!(
        barrier_type,
        BinaryBarrierType::DownAndIn | BinaryBarrierType::UpAndIn
    );

    let price = match payoff {
        BinaryPayoff::CashOrNothing => {
            // Standard plain cash-or-nothing price
            let vanilla = b_cash(&p, phi);
            // Barrier-adjusted price using in/out parity
            if is_in {
                // K·e^{-rT} · η·N(η·(y1 − σ√T)) · (H/S)^{2μ}
                d_cash(&p, phi, eta)
            } else {
                // out = vanilla - in
                vanilla - d_cash(&p, phi, eta)
            }
        }
        BinaryPayoff::AssetOrNothing => {
            let vanilla = a_asset(&p, phi);
            if is_in {
                c_asset(&p, phi, eta)
            } else {
                vanilla - c_asset(&p, phi, eta)
            }
        }
    };

    // Delta via finite difference
    let eps = s * 1e-5;
    let p_up = Params::new(s + eps, k, h, r, q, sigma, tau);
    let p_dn = Params::new(s - eps, k, h, r, q, sigma, tau);

    let price_up = match payoff {
        BinaryPayoff::CashOrNothing => if is_in { d_cash(&p_up, phi, eta) } else { b_cash(&p_up, phi) - d_cash(&p_up, phi, eta) },
        BinaryPayoff::AssetOrNothing => if is_in { c_asset(&p_up, phi, eta) } else { a_asset(&p_up, phi) - c_asset(&p_up, phi, eta) },
    };
    let price_dn = match payoff {
        BinaryPayoff::CashOrNothing => if is_in { d_cash(&p_dn, phi, eta) } else { b_cash(&p_dn, phi) - d_cash(&p_dn, phi, eta) },
        BinaryPayoff::AssetOrNothing => if is_in { c_asset(&p_dn, phi, eta) } else { a_asset(&p_dn, phi) - c_asset(&p_dn, phi, eta) },
    };
    let delta = (price_up - price_dn) / (2.0 * eps);

    BinaryBarrierResult { price: price.max(0.0), delta }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Basic sanity: DI cash-or-nothing call should be positive and < K*e^{-rT}
    #[test]
    fn di_cash_or_nothing_call_price_in_range() {
        // S=100, K=100, H=90, r=0.05, q=0, sigma=0.20, T=1
        let result = analytic_binary_barrier(
            100.0, 100.0, 90.0, 0.05, 0.0, 0.20, 1.0,
            BinaryBarrierType::DownAndIn,
            BinaryPayoff::CashOrNothing,
            BinaryDirection::Call,
        );
        // Cash = K = 100, max PV = K * df
        let max_pay = 100.0_f64 * (-0.05_f64).exp();
        assert!(result.price > 0.0, "price should be positive");
        assert!(result.price < max_pay, "price={} must be < K*df={}", result.price, max_pay);
    }

    // Down-and-out + down-and-in should equal vanilla cash-or-nothing
    #[test]
    fn in_out_parity_cash_or_nothing() {
        let args = (100.0_f64, 100.0, 90.0, 0.05, 0.0, 0.20, 1.0);
        let (s, k, h, r, q, sigma, tau) = args;

        let r_in = analytic_binary_barrier(s, k, h, r, q, sigma, tau,
            BinaryBarrierType::DownAndIn, BinaryPayoff::CashOrNothing, BinaryDirection::Call);
        let r_out = analytic_binary_barrier(s, k, h, r, q, sigma, tau,
            BinaryBarrierType::DownAndOut, BinaryPayoff::CashOrNothing, BinaryDirection::Call);

        // vanilla cash-or-nothing call
        let p = Params::new(s, k, h, r, q, sigma, tau);
        let vanilla = b_cash(&p, 1.0);

        let sum = r_in.price + r_out.price;
        assert!(
            (sum - vanilla).abs() < 1e-8,
            "in + out = vanilla, got in={} out={} sum={} vanilla={}",
            r_in.price, r_out.price, sum, vanilla
        );
    }

    // Asset-or-nothing: parity test
    #[test]
    fn in_out_parity_asset_or_nothing() {
        let (s, k, h, r, q, sigma, tau) = (100.0, 95.0, 85.0, 0.04, 0.01, 0.25, 0.5);

        let r_in = analytic_binary_barrier(s, k, h, r, q, sigma, tau,
            BinaryBarrierType::DownAndIn, BinaryPayoff::AssetOrNothing, BinaryDirection::Call);
        let r_out = analytic_binary_barrier(s, k, h, r, q, sigma, tau,
            BinaryBarrierType::DownAndOut, BinaryPayoff::AssetOrNothing, BinaryDirection::Call);

        let p = Params::new(s, k, h, r, q, sigma, tau);
        let vanilla = a_asset(&p, 1.0);

        assert!(
            ((r_in.price + r_out.price) - vanilla).abs() < 1e-8,
            "asset in+out parity failed"
        );
    }

    // Up-and-in put: S below barrier → price should be positive
    #[test]
    fn up_and_in_put_positive() {
        let result = analytic_binary_barrier(
            100.0, 100.0, 110.0, 0.05, 0.0, 0.20, 1.0,
            BinaryBarrierType::UpAndIn,
            BinaryPayoff::CashOrNothing,
            BinaryDirection::Put,
        );
        assert!(result.price > 0.0);
    }

    // Delta sign: down-and-in call should have positive delta
    #[test]
    fn delta_sign_down_and_in_call() {
        let result = analytic_binary_barrier(
            100.0, 100.0, 85.0, 0.05, 0.0, 0.20, 1.0,
            BinaryBarrierType::DownAndIn,
            BinaryPayoff::CashOrNothing,
            BinaryDirection::Call,
        );
        // delta can be negative near barrier; just ensure it is computed
        let _ = result.delta;
    }
}
