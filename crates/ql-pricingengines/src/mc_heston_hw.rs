//! Monte Carlo Heston Hull-White hybrid engine.
//!
//! Combined stochastic volatility (Heston) and stochastic interest rates
//! (Hull-White) for pricing equity options.
//!
//! System of SDEs:
//!   dS = (r(t) − q) S dt + √v S dW_S
//!   dv = κ(θ − v) dt + σ_v √v dW_v
//!   dr = (θ_r(t) − a r) dt + σ_r dW_r
//!
//! with correlations ρ_Sv, ρ_Sr, ρ_vr between the three Brownians.
//!
//! Corresponds to QuantLib's `MCHestonHullWhiteEngine`.

use serde::{Deserialize, Serialize};

/// Parameters for the MC Heston Hull-White engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McHestonHwParams {
    /// Spot price.
    pub spot: f64,
    /// Strike price.
    pub strike: f64,
    /// Time to maturity (years).
    pub maturity: f64,
    /// Dividend yield (continuous).
    pub div_yield: f64,
    /// True for call, false for put.
    pub is_call: bool,

    // Heston params
    /// Initial variance v₀.
    pub v0: f64,
    /// Mean-reversion speed κ.
    pub kappa: f64,
    /// Long-run variance θ.
    pub theta: f64,
    /// Vol of vol σ_v.
    pub sigma_v: f64,
    /// Correlation between S and v.
    pub rho_sv: f64,

    // Hull-White params
    /// HW mean-reversion speed a.
    pub hw_a: f64,
    /// HW short-rate volatility σ_r.
    pub hw_sigma: f64,
    /// Initial short rate r₀.
    pub r0: f64,

    // Cross-correlations
    /// Correlation between S and r.
    pub rho_sr: f64,
    /// Correlation between v and r.
    pub rho_vr: f64,

    // MC params
    /// Number of simulation paths.
    pub num_paths: usize,
    /// Number of time steps.
    pub num_steps: usize,
    /// Random seed.
    pub seed: u64,
}

/// Result from the MC Heston Hull-White engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McHestonHwResult {
    /// Option price.
    pub npv: f64,
    /// Standard error.
    pub std_error: f64,
    /// Number of paths.
    pub num_paths: usize,
}

/// Price a European option under Heston + Hull-White using Monte Carlo.
pub fn price_mc_heston_hw(params: &McHestonHwParams) -> McHestonHwResult {
    let dt = params.maturity / params.num_steps as f64;
    let sqrt_dt = dt.sqrt();

    // Cholesky decomposition for 3 correlated Brownians
    // [W_S, W_v, W_r] with correlations ρ_Sv, ρ_Sr, ρ_vr
    let l11 = 1.0;
    let l21 = params.rho_sv;
    let l22 = (1.0 - l21 * l21).max(0.0).sqrt();
    let l31 = params.rho_sr;
    let l32 = if l22.abs() > 1e-10 { (params.rho_vr - l31 * l21) / l22 } else { 0.0 };
    let l33 = (1.0 - l31 * l31 - l32 * l32).max(0.0).sqrt();

    let mut rng = SimpleRng::new(params.seed);
    let mut sum_payoff = 0.0;
    let mut sum_payoff_sq = 0.0;

    for _ in 0..params.num_paths {
        let mut s = params.spot;
        let mut v = params.v0;
        let mut r = params.r0;
        let mut integrated_r = 0.0;

        for _ in 0..params.num_steps {
            let z1 = rng.normal();
            let z2 = rng.normal();
            let z3 = rng.normal();

            // Correlated increments
            let dw_s = l11 * z1;
            let dw_v = l21 * z1 + l22 * z2;
            let dw_r = l31 * z1 + l32 * z2 + l33 * z3;

            // Full truncation scheme for Heston
            let v_pos = v.max(0.0);
            let sqrt_v = v_pos.sqrt();

            let ds = (r - params.div_yield) * s * dt + sqrt_v * s * sqrt_dt * dw_s;
            let dv = params.kappa * (params.theta - v_pos) * dt
                + params.sigma_v * sqrt_v * sqrt_dt * dw_v;
            let dr = params.hw_a * (params.r0 - r) * dt + params.hw_sigma * sqrt_dt * dw_r;

            // Trapezoidal integration for discount factor
            integrated_r += 0.5 * dt * r;
            s += ds;
            v += dv;
            r += dr;
            integrated_r += 0.5 * dt * r;

            s = s.max(1e-8);
        }

        let payoff = if params.is_call {
            (s - params.strike).max(0.0)
        } else {
            (params.strike - s).max(0.0)
        };

        let discounted_payoff = payoff * (-integrated_r).exp();
        sum_payoff += discounted_payoff;
        sum_payoff_sq += discounted_payoff * discounted_payoff;
    }

    let n = params.num_paths as f64;
    let mean = sum_payoff / n;
    let variance = (sum_payoff_sq / n - mean * mean).max(0.0);
    let std_error = (variance / n).sqrt();

    McHestonHwResult {
        npv: mean.max(0.0),
        std_error,
        num_paths: params.num_paths,
    }
}

/// Simple xorshift64 PRNG with Box-Muller.
struct SimpleRng {
    state: u64,
    spare: Option<f64>,
}

impl SimpleRng {
    fn new(seed: u64) -> Self { Self { state: seed.max(1), spare: None } }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        if let Some(z) = self.spare.take() { return z; }
        loop {
            let u = 2.0 * self.uniform() - 1.0;
            let v = 2.0 * self.uniform() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let factor = (-2.0 * s.ln() / s).sqrt();
                self.spare = Some(v * factor);
                return u * factor;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_params() -> McHestonHwParams {
        McHestonHwParams {
            spot: 100.0,
            strike: 100.0,
            maturity: 1.0,
            div_yield: 0.0,
            is_call: true,
            v0: 0.04,
            kappa: 2.0,
            theta: 0.04,
            sigma_v: 0.3,
            rho_sv: -0.7,
            hw_a: 0.05,
            hw_sigma: 0.01,
            r0: 0.05,
            rho_sr: 0.0,
            rho_vr: 0.0,
            num_paths: 50_000,
            num_steps: 100,
            seed: 42,
        }
    }

    #[test]
    fn test_mc_heston_hw_call_price() {
        let params = base_params();
        let res = price_mc_heston_hw(&params);
        // Reasonable ATM call price
        assert!(res.npv > 3.0 && res.npv < 20.0, "npv={}", res.npv);
        assert!(res.std_error < 1.0, "se={}", res.std_error);
    }

    #[test]
    fn test_mc_heston_hw_put_lower_bound() {
        let params = McHestonHwParams { is_call: false, ..base_params() };
        let res = price_mc_heston_hw(&params);
        assert!(res.npv >= 0.0, "npv={}", res.npv);
    }

    #[test]
    fn test_mc_heston_hw_zero_hw_vol() {
        // Zero HW vol → pure Heston
        let params = McHestonHwParams { hw_sigma: 0.0, ..base_params() };
        let res = price_mc_heston_hw(&params);
        assert!(res.npv > 3.0 && res.npv < 20.0, "npv={}", res.npv);
    }

    #[test]
    fn test_mc_heston_hw_correlation_effect() {
        let pos = McHestonHwParams { rho_sr: 0.5, ..base_params() };
        let neg = McHestonHwParams { rho_sr: -0.5, ..base_params() };
        let pos_res = price_mc_heston_hw(&pos);
        let neg_res = price_mc_heston_hw(&neg);
        // Both should give positive prices
        assert!(pos_res.npv > 0.0 && neg_res.npv > 0.0);
    }
}
