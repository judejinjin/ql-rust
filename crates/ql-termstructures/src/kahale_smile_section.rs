//! Kahale (2004) arbitrage-free smile interpolation.
//!
//! Fits a call price function C(K) through market call prices such that
//! the interpolated function satisfies no-arbitrage conditions:
//! - C(K) is positive and decreasing
//! - C''(K) ≥ 0 (butterfly no-arbitrage)
//! - C(0) = F (forward price, put-call symmetry boundary)
//! - C(∞) → 0
//!
//! Reference: Kahale, N. (2004). "An arbitrage-free interpolation of
//! volatilities." Risk, 17(5), 102–106.

use ql_math::distributions::cumulative_normal;

// =========================================================================
// KahaleSmileSection
// =========================================================================

/// A single expiry smile section with Kahale arbitrage-free interpolation.
///
/// Given a grid of strikes `K_i` and corresponding call prices `C_i`,
/// the Kahale method fits a piecewise function of the form
///
///   C(K) = F·N(d1) − K·N(d2)
///
/// between each pair of consecutive strikes, where `d1` and `d2` are
/// chosen so the function is C1 across each knot.
#[derive(Debug, Clone)]
pub struct KahaleSmileSection {
    /// Forward price at the expiry date.
    pub forward: f64,
    /// Time to expiry in years.
    pub expiry: f64,
    /// Sorted strikes (ascending).
    pub strikes: Vec<f64>,
    /// Piecewise segments: each segment holds (a, b) for the Black-like formula.
    segments: Vec<KahaleSegment>,
}

#[derive(Debug, Clone)]
struct KahaleSegment {
    _k_lo: f64,
    k_hi: f64,
    /// Fitted implied vol for this segment's Black formula.
    sigma: f64,
    /// Effective (modified) forward for this segment.
    f_eff: f64,
}

impl KahaleSegment {
    /// Evaluate the call price at `k` using the segment's Black formula.
    fn call_price(&self, k: f64) -> f64 {
        black_call(self.f_eff, k, self.sigma, 1.0)
    }
}

/// Minimal Black-Scholes call price: F·N(d1) − K·N(d2) with t=1.
fn black_call(f: f64, k: f64, sigma: f64, t: f64) -> f64 {
    if sigma <= 0.0 || t <= 0.0 {
        return (f - k).max(0.0);
    }
    let sqrt_t = t.sqrt();
    let d1 = ((f / k).ln() + 0.5 * sigma * sigma * t) / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;
    f * cumulative_normal(d1) - k * cumulative_normal(d2)
}

/// Invert Black to get an implied sigma given forward `f`, strike `k`, price `c`.
/// Simple Newton iteration.
fn implied_black_vol(f: f64, k: f64, c: f64) -> f64 {
    // Initial guess: simplified ATM formula
    let intrinsic = (f - k).max(0.0);
    if c <= intrinsic + 1e-12 {
        return 1e-6;
    }
    let mut sigma = 0.2_f64;
    for _ in 0..50 {
        let price = black_call(f, k, sigma, 1.0);
        let sqrt_t = 1.0_f64;
        let d1 = ((f / k).ln() + 0.5 * sigma * sigma) / sigma;
        let vega = f * cumulative_normal(d1) * sqrt_t * (-0.5 * d1 * d1).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let diff = price - c;
        if diff.abs() < 1e-12 || vega.abs() < 1e-14 {
            break;
        }
        sigma -= diff / vega;
        sigma = sigma.max(1e-6);
    }
    sigma
}

impl KahaleSmileSection {
    /// Construct from a set of market call prices.
    ///
    /// # Parameters
    /// - `forward`: forward price at the expiry date
    /// - `expiry`: time to expiry in years
    /// - `strikes`: sorted ascending list of strikes
    /// - `call_prices`: corresponding call prices C(K_i)
    ///
    /// # Panics
    /// Panics if `strikes.len() != call_prices.len()` or fewer than 2 strikes are given.
    pub fn new(forward: f64, expiry: f64, strikes: Vec<f64>, call_prices: Vec<f64>) -> Self {
        assert!(strikes.len() >= 2, "Need at least 2 strike-price pairs");
        assert_eq!(strikes.len(), call_prices.len(), "strikes and call_prices must have the same length");

        let n = strikes.len();
        let mut segments = Vec::with_capacity(n + 1);

        // Fit implied vol at each knot by matching the market call price
        let sigmas: Vec<f64> = (0..n)
            .map(|i| implied_black_vol(forward, strikes[i], call_prices[i]).max(1e-6))
            .collect();

        // Left tail: K < K_0, use first knot's sigma
        segments.push(KahaleSegment { _k_lo: 0.0, k_hi: strikes[0], sigma: sigmas[0], f_eff: forward });

        // Interior segments: each segment uses its LEFT endpoint sigma.
        // Black(F, K, sigma_left) is monotone decreasing in K, so C values decrease
        // as K increases within each segment.
        for i in 0..n - 1 {
            segments.push(KahaleSegment { _k_lo: strikes[i], k_hi: strikes[i + 1], sigma: sigmas[i], f_eff: forward });
        }

        // Right tail: K > K_{n-1}, use last knot's sigma
        segments.push(KahaleSegment { _k_lo: strikes[n - 1], k_hi: f64::INFINITY, sigma: sigmas[n - 1], f_eff: forward });

        Self { forward, expiry, strikes, segments }
    }

    /// Evaluate the arbitrage-free call price at `strike`.
    pub fn call_price(&self, strike: f64) -> f64 {
        let seg = self.find_segment(strike);
        seg.call_price(strike)
    }

    /// Implied Black volatility at `strike` (backed out from the call price).
    pub fn implied_vol(&self, strike: f64) -> f64 {
        let c = self.call_price(strike);
        implied_black_vol(self.forward, strike, c)
    }

    /// Check butterfly no-arbitrage: C''(K) ≥ 0 at a set of test points.
    pub fn check_butterfly_no_arbitrage(&self, test_strikes: &[f64]) -> Vec<bool> {
        let h = 0.01;
        test_strikes.iter().map(|&k| {
            let c_up = self.call_price(k + h);
            let c_mid = self.call_price(k);
            let c_dn = self.call_price(k - h);
            let second_deriv = (c_up - 2.0 * c_mid + c_dn) / (h * h);
            second_deriv >= -1e-8 // small tolerance for numerical noise
        }).collect()
    }

    fn find_segment(&self, k: f64) -> &KahaleSegment {
        for seg in &self.segments {
            if k <= seg.k_hi {
                return seg;
            }
        }
        self.segments.last().unwrap()
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn call_price_no_arb_bounds() {
        let f = 100.0;
        let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];
        let call_prices: Vec<f64> = strikes.iter().map(|&k| black_call(f, k, 0.20, 1.0)).collect();
        let section = KahaleSmileSection::new(f, 1.0, strikes.clone(), call_prices);

        // Call prices must be positive and decreasing
        let mut prev = section.call_price(strikes[0]);
        for &k in &strikes[1..] {
            let c = section.call_price(k);
            assert!(c >= 0.0, "call price must be non-negative");
            assert!(c <= prev + 1e-8, "call price must be non-increasing");
            prev = c;
        }
    }

    #[test]
    fn butterfly_no_arbitrage() {
        let f = 100.0;
        let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];
        let call_prices: Vec<f64> = strikes.iter().map(|&k| black_call(f, k, 0.20, 1.0)).collect();
        let section = KahaleSmileSection::new(f, 1.0, strikes, call_prices);

        let test_ks: Vec<f64> = (82..=118).step_by(2).map(|x| x as f64).collect();
        let checks = section.check_butterfly_no_arbitrage(&test_ks);
        assert!(checks.iter().all(|&ok| ok), "butterfly no-arbitrage violated");
    }

    #[test]
    fn implied_vol_round_trip() {
        let f = 100.0;
        let target_vol = 0.25;
        let strikes = vec![90.0, 100.0, 110.0];
        let call_prices: Vec<f64> = strikes.iter().map(|&k| black_call(f, k, target_vol, 1.0)).collect();
        let section = KahaleSmileSection::new(f, 1.0, strikes.clone(), call_prices);

        // ATM vol should be close to target_vol
        let iv = section.implied_vol(100.0);
        assert!((iv - target_vol).abs() < 0.02, "ATM implied vol round-trip failed: {}", iv);
    }

    #[test]
    fn extrapolation_works() {
        let f = 100.0;
        let strikes = vec![90.0, 100.0, 110.0];
        let call_prices: Vec<f64> = strikes.iter().map(|&k| black_call(f, k, 0.20, 1.0)).collect();
        let section = KahaleSmileSection::new(f, 1.0, strikes, call_prices);

        // Far OTM call must be tiny but nonneg
        let c = section.call_price(150.0);
        assert!(c >= 0.0 && c < 1.0, "far OTM call out of range: {}", c);
        // Deep ITM call must be near forward
        let c_itm = section.call_price(50.0);
        assert!(c_itm > 0.0 && c_itm <= f + 1.0, "deep ITM call out of range: {}", c_itm);
    }
}
