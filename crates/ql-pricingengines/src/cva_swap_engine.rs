//! CVA (Credit Valuation Adjustment) swap engine using Hull-White short-rate
//! simulation.
//!
//! Computes unilateral and bilateral CVA for interest-rate swaps using
//! Monte Carlo simulation of the Hull-White one-factor model to generate
//! future exposure profiles.
//!
//! CVA = ∫₀ᵀ E[max(V(t), 0)] · (1 − R_C) · dPD_C(t)
//! DVA = ∫₀ᵀ E[max(-V(t), 0)] · (1 − R_B) · dPD_B(t)
//!
//! where PD_C is the counterparty default probability and PD_B is the
//! own-institution default probability.
//!
//! Reference:
//! - QuantLib: CounterpartyAdjSwapEngine (cva_swap.hpp)
//! - Gregory, J. (2012), "Counterparty Credit Risk and CVA"

use serde::{Deserialize, Serialize};
use rand::prelude::*;
use rand_distr::StandardNormal;

/// CVA swap engine result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CvaSwapResult {
    /// Risk-free NPV of the swap.
    pub risk_free_npv: f64,
    /// Unilateral CVA.
    pub cva: f64,
    /// DVA (own-default benefit).
    pub dva: f64,
    /// Bilateral CVA = CVA − DVA.
    pub bilateral_cva: f64,
    /// CVA-adjusted NPV = risk_free_npv − CVA + DVA.
    pub adjusted_npv: f64,
    /// Expected positive exposure profile at each time step.
    pub epe_profile: Vec<f64>,
    /// Expected negative exposure profile.
    pub ene_profile: Vec<f64>,
    /// Potential future exposure (97.5th percentile).
    pub pfe_profile: Vec<f64>,
}

/// Compute CVA for an interest-rate swap via Hull-White Monte Carlo.
///
/// # Arguments
/// - `notional` — swap notional
/// - `fixed_rate` — fixed leg rate
/// - `swap_tenor` — total swap tenor in years
/// - `payment_freq` — payments per year (e.g. 2 = semi-annual)
/// - `risk_free_rate` — initial short rate (r₀)
/// - `hw_kappa` — Hull-White mean reversion speed
/// - `hw_sigma` — Hull-White volatility
/// - `hw_theta` — Hull-White long-run rate (simplified: constant θ)
/// - `counterparty_hazard_rate` — counterparty hazard rate (constant)
/// - `counterparty_recovery` — counterparty recovery rate
/// - `own_hazard_rate` — own-institution hazard rate
/// - `own_recovery` — own recovery rate
/// - `n_paths` — number of Monte Carlo paths
/// - `n_steps` — number of time steps per path
/// - `seed` — RNG seed
#[allow(clippy::too_many_arguments)]
pub fn cva_swap_engine(
    notional: f64,
    fixed_rate: f64,
    swap_tenor: f64,
    payment_freq: u32,
    risk_free_rate: f64,
    hw_kappa: f64,
    hw_sigma: f64,
    hw_theta: f64,
    counterparty_hazard_rate: f64,
    counterparty_recovery: f64,
    own_hazard_rate: f64,
    own_recovery: f64,
    n_paths: usize,
    n_steps: usize,
    seed: u64,
) -> CvaSwapResult {
    let dt = swap_tenor / n_steps as f64;
    let n_payments = (swap_tenor * payment_freq as f64).round() as usize;
    let payment_times: Vec<f64> = (1..=n_payments)
        .map(|i| i as f64 / payment_freq as f64)
        .collect();
    let accrual = 1.0 / payment_freq as f64;

    let mut rng = StdRng::seed_from_u64(seed);

    // Storage for exposure profiles
    let mut epe_sum = vec![0.0_f64; n_steps + 1];
    let mut ene_sum = vec![0.0_f64; n_steps + 1];
    let mut exposures: Vec<Vec<f64>> = vec![Vec::with_capacity(n_paths); n_steps + 1];
    let mut npv_sum = 0.0_f64;

    for _path in 0..n_paths {
        // Simulate Hull-White short rate path
        let mut r = risk_free_rate;
        let mut rates = Vec::with_capacity(n_steps + 1);
        rates.push(r);

        for _step in 0..n_steps {
            let dw: f64 = rng.sample::<f64, _>(StandardNormal);
            r += hw_kappa * (hw_theta - r) * dt + hw_sigma * dt.sqrt() * dw;
            rates.push(r);
        }

        // At each time step, compute the swap MTM (remaining coupons)
        for step in 0..=n_steps {
            let t = step as f64 * dt;
            let short_rate = rates[step];

            // Remaining swap value: Σ (fixed - float) × τ × DF(t, t_i) × N
            // Simplified: float leg refixes at current short rate
            let mut swap_mtm = 0.0;
            for &tp in &payment_times {
                if tp > t + 1e-8 {
                    let dt_pay = tp - t;
                    // Discount using the Hull-White simulated rate (simplified)
                    let df = (-short_rate * dt_pay).exp();
                    swap_mtm += (fixed_rate - short_rate) * accrual * df * notional;
                }
            }

            epe_sum[step] += swap_mtm.max(0.0);
            ene_sum[step] += (-swap_mtm).max(0.0);
            exposures[step].push(swap_mtm);
        }

        // Risk-free NPV: use final swap MTM at t=0
        npv_sum += exposures[0].last().copied().unwrap_or(0.0);
    }

    let risk_free_npv = npv_sum / n_paths as f64;

    // EPE and ENE profiles
    let epe_profile: Vec<f64> = epe_sum.iter().map(|s| s / n_paths as f64).collect();
    let ene_profile: Vec<f64> = ene_sum.iter().map(|s| s / n_paths as f64).collect();

    // PFE (97.5th percentile)
    let pfe_profile: Vec<f64> = exposures.iter().map(|exps| {
        let mut sorted = exps.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((0.975 * sorted.len() as f64) as usize).min(sorted.len().saturating_sub(1));
        sorted[idx]
    }).collect();

    // CVA = Σ EPE(t_i) × DP_C(t_{i-1}, t_i) × (1 − R_C) × Δt
    let lgd_c = 1.0 - counterparty_recovery;
    let lgd_b = 1.0 - own_recovery;
    let mut cva = 0.0;
    let mut dva = 0.0;

    for step in 1..=n_steps {
        let t_prev = (step - 1) as f64 * dt;
        let t = step as f64 * dt;
        let sp_prev = (-counterparty_hazard_rate * t_prev).exp();
        let sp = (-counterparty_hazard_rate * t).exp();
        let dp = sp_prev - sp;

        let sp_own_prev = (-own_hazard_rate * t_prev).exp();
        let sp_own = (-own_hazard_rate * t).exp();
        let dp_own = sp_own_prev - sp_own;

        let df = (-risk_free_rate * t).exp();

        cva += df * epe_profile[step] * lgd_c * dp;
        dva += df * ene_profile[step] * lgd_b * dp_own;
    }

    let bilateral = cva - dva;
    let adjusted = risk_free_npv - cva + dva;

    CvaSwapResult {
        risk_free_npv,
        cva,
        dva,
        bilateral_cva: bilateral,
        adjusted_npv: adjusted,
        epe_profile,
        ene_profile,
        pfe_profile,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cva_basic() {
        let res = cva_swap_engine(
            10_000_000.0,  // notional
            0.03,          // fixed rate
            5.0,           // 5Y swap
            2,             // semi-annual
            0.03,          // r0
            0.1,           // kappa
            0.01,          // sigma
            0.03,          // theta
            0.02,          // counterparty hazard
            0.40,          // counterparty recovery
            0.01,          // own hazard
            0.40,          // own recovery
            1000,          // paths
            50,            // steps
            42,            // seed
        );

        // CVA should be positive
        assert!(res.cva >= 0.0, "cva={}", res.cva);
        // DVA should be positive
        assert!(res.dva >= 0.0, "dva={}", res.dva);
        // EPE profile should be non-negative
        for &e in &res.epe_profile {
            assert!(e >= 0.0);
        }
        // PFE profile should have correct length
        assert_eq!(res.pfe_profile.len(), 51);
    }

    #[test]
    fn test_cva_zero_hazard() {
        let res = cva_swap_engine(
            10_000_000.0, 0.03, 5.0, 2,
            0.03, 0.1, 0.01, 0.03,
            0.0,   // no counterparty default risk
            0.40,
            0.0,   // no own default risk
            0.40,
            500, 20, 42,
        );
        // With zero hazard rates, CVA and DVA should be zero
        assert!(res.cva.abs() < 10.0, "cva={}", res.cva);
        assert!(res.dva.abs() < 10.0, "dva={}", res.dva);
    }

    #[test]
    fn test_cva_atm_swap() {
        // At-the-money swap: fixed rate = initial floating rate
        let res = cva_swap_engine(
            10_000_000.0, 0.04, 5.0, 2,
            0.04, 0.1, 0.01, 0.04,
            0.03, 0.40, 0.01, 0.40,
            500, 20, 123,
        );
        // Just verify reasonable output
        assert!(res.cva.abs() < 1_000_000.0, "cva={}", res.cva);
        assert!(res.adjusted_npv.is_finite());
    }
}
