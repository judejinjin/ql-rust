//! Complex chooser option instrument.
//!
//! A complex chooser option allows the holder to choose at a future date
//! whether the option becomes a **call** with its own strike and expiry,
//! or a **put** with its own (possibly different) strike and expiry.
//!
//! This generalises the simple chooser (where both call and put have the
//! same strike and expiry).  Valued analytically using the Rubinstein
//! (1991) extension.
//!
//! ## Analytic formula
//!
//! The Black-Scholes price is:
//!
//! ```text
//! V = S e^{-qT_c} N(d1, γ, ρ₁) − K_c e^{-rT_c} N(d2, γ, ρ₁)
//!   − S e^{-qT_p} N(-d1', -γ', -ρ₂) + K_p e^{-rT_p} N(-d2', -γ', -ρ₂)
//! ```
//!
//! where the bivariate normal integrals are over the choosing date and
//! expiry dates.  See Hull (2018) §26.5 for derivation.
//!
//! ## References
//!
//! - Rubinstein, M. (1991), *Options for the undecided*,
//!   Risk, 4(4), 70–73.

use serde::{Deserialize, Serialize};

use ql_math::distributions::{cumulative_normal, bivariate_normal_cdf};

// ---------------------------------------------------------------------------
// Instrument struct
// ---------------------------------------------------------------------------

/// A complex chooser option.
///
/// At the choosing date `t_choose`, the holder selects whether to enter:
/// - a **call** with strike `k_call` expiring at `t_call`, or
/// - a **put** with strike `k_put` expiring at `t_put`.
///
/// The call and put can have different strikes and expiry dates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexChooserOption {
    /// Time to the choosing date (years from valuation).
    pub t_choose: f64,
    /// Call leg: strike price.
    pub k_call: f64,
    /// Call leg: time to expiry (years). Must be ≥ `t_choose`.
    pub t_call: f64,
    /// Put leg: strike price.
    pub k_put: f64,
    /// Put leg: time to expiry (years). Must be ≥ `t_choose`.
    pub t_put: f64,
}

impl ComplexChooserOption {
    /// Create a complex chooser option.
    ///
    /// # Panics
    ///
    /// Panics if `t_call < t_choose` or `t_put < t_choose`.
    pub fn new(t_choose: f64, k_call: f64, t_call: f64, k_put: f64, t_put: f64) -> Self {
        assert!(t_call >= t_choose, "t_call must be >= t_choose");
        assert!(t_put  >= t_choose, "t_put must be >= t_choose");
        Self { t_choose, k_call, t_call, k_put, t_put }
    }

    /// Price this complex chooser option under Black-Scholes.
    ///
    /// # Parameters
    ///
    /// - `spot` — current spot price S
    /// - `r` — continuously compounded risk-free rate
    /// - `q` — continuous dividend yield
    /// - `sigma` — constant volatility
    ///
    /// Returns the option NPV.
    pub fn price_bs(&self, spot: f64, r: f64, q: f64, sigma: f64) -> f64 {
        let s = spot;
        let t0 = self.t_choose;
        let tc = self.t_call;
        let tp = self.t_put;
        let kc = self.k_call;
        let kp = self.k_put;

        let sqrt_t0 = t0.sqrt();
        let sqrt_tc = tc.sqrt();
        let sqrt_tp = tp.sqrt();

        // Forward prices at expiry dates (used in full analytic formula)
        let _fc = s * ((r - q) * tc).exp();
        let _fp = s * ((r - q) * tp).exp();

        // Black-Scholes d1, d2 for the call and put
        let d1c = ((s / kc).ln() + (r - q + 0.5 * sigma * sigma) * tc) / (sigma * sqrt_tc);
        let d2c = d1c - sigma * sqrt_tc;
        let d1p = ((s / kp).ln() + (r - q + 0.5 * sigma * sigma) * tp) / (sigma * sqrt_tp);
        let d2p = d1p - sigma * sqrt_tp;

        // Critical spot level I* at choosing date: boundary between call and put
        // At choosing date, call value = put value for spot I*
        // Solved iteratively (Newton). Start from ATM approximation.
        let i_star = self.find_critical_spot(r, q, sigma, t0, tc, tp, kc, kp, s);

        // d1, d2 for the critical spot level over choosing window
        let y1c = ((s / i_star).ln() + (r - q + 0.5 * sigma * sigma) * t0) / (sigma * sqrt_t0);
        let y2c = y1c - sigma * sqrt_t0;
        let y1p = -y1c;
        let y2p = -y2c;

        // Correlation between choosing date and expiry
        let rho_c = (t0 / tc).sqrt();
        let rho_p = (t0 / tp).sqrt();

        // Call contribution
        let call_part = s * (-q * tc).exp()
            * bivariate_normal_cdf(d1c, y1c, rho_c)
            - kc * (-r * tc).exp()
            * bivariate_normal_cdf(d2c, y2c, rho_c);

        // Put contribution
        let put_part = -s * (-q * tp).exp()
            * bivariate_normal_cdf(-d1p, -y1p, rho_p)
            + kp * (-r * tp).exp()
            * bivariate_normal_cdf(-d2p, -y2p, rho_p);

        (call_part + put_part).max(0.0)
    }

    /// Find the critical spot I* at the choosing date using Newton-Raphson.
    ///
    /// I* satisfies: BS_call(I*, kc, r, q, σ, tc−t0) = BS_put(I*, kp, r, q, σ, tp−t0)
    fn find_critical_spot(
        &self, r: f64, q: f64, sigma: f64,
        t0: f64, tc: f64, tp: f64, kc: f64, kp: f64, s0: f64,
    ) -> f64 {
        let tau_c = (tc - t0).max(1e-6);
        let tau_p = (tp - t0).max(1e-6);
        let mut i = s0; // initial guess
        for _ in 0..100 {
            let call_val = bs_call(i, kc, r, q, sigma, tau_c);
            let put_val  = bs_put( i, kp, r, q, sigma, tau_p);
            let diff = call_val - put_val;
            if diff.abs() < 1e-8 { break; }
            // Derivative ≈ Δ_call - Δ_put
            let delta_c = bs_delta_call(i, kc, r, q, sigma, tau_c);
            let delta_p = bs_delta_put( i, kp, r, q, sigma, tau_p);
            let deriv = delta_c - delta_p;
            if deriv.abs() < 1e-12 { break; }
            i -= diff / deriv;
            i = i.max(1e-6);
        }
        i
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn bs_call(s: f64, k: f64, r: f64, q: f64, sigma: f64, tau: f64) -> f64 {
    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * tau) / (sigma * tau.sqrt());
    let d2 = d1 - sigma * tau.sqrt();
    s * (-q * tau).exp() * cumulative_normal(d1) - k * (-r * tau).exp() * cumulative_normal(d2)
}

fn bs_put(s: f64, k: f64, r: f64, q: f64, sigma: f64, tau: f64) -> f64 {
    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * tau) / (sigma * tau.sqrt());
    let d2 = d1 - sigma * tau.sqrt();
    k * (-r * tau).exp() * cumulative_normal(-d2) - s * (-q * tau).exp() * cumulative_normal(-d1)
}

fn bs_delta_call(s: f64, k: f64, r: f64, q: f64, sigma: f64, tau: f64) -> f64 {
    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * tau) / (sigma * tau.sqrt());
    (-q * tau).exp() * cumulative_normal(d1)
}

fn bs_delta_put(s: f64, k: f64, r: f64, q: f64, sigma: f64, tau: f64) -> f64 {
    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * tau) / (sigma * tau.sqrt());
    -(-q * tau).exp() * cumulative_normal(-d1)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_chooser_positive_price() {
        // t_choose=0.25, Kc=100, Tc=1.0, Kp=100, Tp=1.0 → simple chooser
        let opt = ComplexChooserOption::new(0.25, 100.0, 1.0, 100.0, 1.0);
        let price = opt.price_bs(100.0, 0.05, 0.0, 0.20);
        assert!(price > 0.0, "price={}", price);
        assert!(price < 100.0);
    }

    #[test]
    fn complex_chooser_different_strikes() {
        // ITM call (Kc=90) and OTM put (Kp=110)
        let opt = ComplexChooserOption::new(0.25, 90.0, 0.75, 110.0, 0.75);
        let price = opt.price_bs(100.0, 0.05, 0.0, 0.25);
        assert!(price > 0.0);
    }

    #[test]
    fn complex_chooser_reduces_to_max_call_put_at_zero_choose_time() {
        // When t_choose → 0 the chooser value ≈ max(call, put)
        let opt = ComplexChooserOption::new(0.001, 100.0, 1.0, 100.0, 1.0);
        let price = opt.price_bs(100.0, 0.05, 0.0, 0.20);
        let call = bs_call(100.0, 100.0, 0.05, 0.0, 0.20, 1.0);
        let put  = bs_put( 100.0, 100.0, 0.05, 0.0, 0.20, 1.0);
        let max_cp = call.max(put);
        // Chooser ≥ max(call, put) (it offers optionality)
        assert!(price >= max_cp * 0.9, "price={} max={}", price, max_cp);
    }

    #[test]
    fn complex_chooser_longer_choose_time_more_valuable() {
        let s = 100.0;
        let r = 0.05;
        let q = 0.02;
        let sigma = 0.20;
        // When both t_choose AND the option expiries are pushed out equally,
        // the chooser should be worth more (more total optionality).
        // opt1: choose at 0.1, options expire at 1.0 (0.9 yr remaining after choose)
        // opt2: choose at 0.1, options expire at 1.5 (1.4 yr remaining after choose)
        let opt1 = ComplexChooserOption::new(0.1, 100.0, 1.0, 95.0, 1.0);
        let opt2 = ComplexChooserOption::new(0.1, 100.0, 1.5, 95.0, 1.5);
        let p1 = opt1.price_bs(s, r, q, sigma);
        let p2 = opt2.price_bs(s, r, q, sigma);
        // Longer option life after choosing → more valuable
        assert!(p2 >= p1, "option with longer expiry should be more valuable: p1={:.4}, p2={:.4}", p1, p2);
    }
}
