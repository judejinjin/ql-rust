//! Portfolio-level AAD — aggregate sensitivities across instruments.
//!
//! For a portfolio of M instruments sharing N market inputs, a single
//! forward pass computes total PnL and a single backward pass yields
//! ∂PnL/∂x_i for all i — cost is O(M), not O(M·N).
//!
//! # Example
//!
//! ```
//! use ql_aad::portfolio::{Portfolio, PortfolioGreeks};
//! use ql_aad::cashflows::{Cashflow, fixed_rate_bond_cashflows};
//! use ql_aad::curves::DiscountCurveAD;
//! use ql_aad::tape::AReal;
//!
//! let times = vec![1.0, 2.0, 5.0, 10.0];
//! let rates = vec![0.03, 0.032, 0.035, 0.04];
//!
//! let mut port = Portfolio::new(&times, &rates);
//! port.add_bond(0.04, 100.0, 5, 1.0);     // 5y 4% bond, weight 1
//! port.add_bond(0.05, 100.0, 10, -0.5);   // 10y 5% bond, weight -0.5 (short)
//!
//! let greeks = port.compute();
//! assert!(greeks.total_npv.is_finite());
//! assert!(greeks.key_rate_durations.len() == 4);
//! ```

use crate::cashflows::{fixed_rate_bond_cashflows, npv, Cashflow};
use crate::curves::DiscountCurveAD;
use crate::number::Number;
use crate::tape::{adjoint_tl, with_tape, AReal};

// ===========================================================================
// Types
// ===========================================================================

/// A single instrument in the portfolio.
#[derive(Debug, Clone)]
struct PortfolioInstrument {
    /// Cashflows for this instrument.
    cashflows: Vec<Cashflow>,
    /// Portfolio weight (notional multiplier; negative = short).
    weight: f64,
    /// Description label.
    #[allow(dead_code)]
    label: String,
}

/// Result of portfolio-level AAD.
#[derive(Debug, Clone)]
pub struct PortfolioGreeks {
    /// Total portfolio NPV (weighted sum of instrument NPVs).
    pub total_npv: f64,
    /// Per-instrument NPVs (before weighting).
    pub instrument_npvs: Vec<f64>,
    /// Key-rate durations: ∂NPV/∂r_i × 0.0001 for each pillar i.
    pub key_rate_durations: Vec<f64>,
    /// Raw sensitivities: ∂NPV/∂r_i for each pillar i.
    pub rate_sensitivities: Vec<f64>,
    /// Pillar times (for labeling).
    pub pillar_times: Vec<f64>,
    /// Number of instruments.
    pub num_instruments: usize,
}

/// Portfolio builder for AAD-enabled risk computation.
pub struct Portfolio {
    instruments: Vec<PortfolioInstrument>,
    pillar_times: Vec<f64>,
    pillar_rates: Vec<f64>,
}

impl Portfolio {
    /// Create a new portfolio with shared curve pillars.
    pub fn new(pillar_times: &[f64], pillar_rates: &[f64]) -> Self {
        assert_eq!(pillar_times.len(), pillar_rates.len());
        Self {
            instruments: Vec::new(),
            pillar_times: pillar_times.to_vec(),
            pillar_rates: pillar_rates.to_vec(),
        }
    }

    /// Add a fixed-rate bond to the portfolio.
    ///
    /// * `coupon_rate` — annual coupon rate (e.g. 0.05 = 5%)
    /// * `face` — face value
    /// * `maturity_years` — years to maturity (integer)
    /// * `weight` — portfolio weight (1.0 = long, -1.0 = short)
    pub fn add_bond(&mut self, coupon_rate: f64, face: f64, maturity_years: usize, weight: f64) {
        let cfs = fixed_rate_bond_cashflows(coupon_rate, face, maturity_years);
        self.instruments.push(PortfolioInstrument {
            cashflows: cfs,
            weight,
            label: format!("Bond {}y {}%", maturity_years, coupon_rate * 100.0),
        });
    }

    /// Add a custom cashflow instrument.
    pub fn add_cashflows(&mut self, cashflows: Vec<Cashflow>, weight: f64, label: &str) {
        self.instruments.push(PortfolioInstrument {
            cashflows,
            weight,
            label: label.to_string(),
        });
    }

    /// Number of instruments in the portfolio.
    pub fn num_instruments(&self) -> usize {
        self.instruments.len()
    }

    /// Compute portfolio NPV + all key-rate durations via a single AAD pass.
    ///
    /// Records the entire portfolio valuation on one tape, then a single
    /// adjoint pass yields ∂(total NPV)/∂r_i for every pillar i.
    pub fn compute(&self) -> PortfolioGreeks {
        let n_instruments = self.instruments.len();

        let (total_npv, instrument_npvs, sensitivities) = with_tape(|tape| {
            // Register all pillar rates as tape inputs
            let rates: Vec<AReal> = self
                .pillar_rates
                .iter()
                .map(|&r| tape.input(r))
                .collect();

            let curve = DiscountCurveAD::from_zero_rates(&self.pillar_times, &rates);

            // Price all instruments on the same tape
            let mut total = AReal::zero();
            let mut inst_npvs = Vec::with_capacity(n_instruments);

            for inst in &self.instruments {
                let inst_npv: AReal = npv(&inst.cashflows, &curve);
                inst_npvs.push(inst_npv.val);
                total = total + inst_npv * AReal::from_f64(inst.weight);
            }

            // Single adjoint pass for the entire portfolio
            let adj = adjoint_tl(total);
            let sens: Vec<f64> = rates.iter().map(|a| adj[a.idx]).collect();
            (total.val, inst_npvs, sens)
        });

        let key_rate_durations: Vec<f64> = sensitivities.iter().map(|&s| s * 0.0001).collect();

        PortfolioGreeks {
            total_npv,
            instrument_npvs,
            key_rate_durations,
            rate_sensitivities: sensitivities,
            pillar_times: self.pillar_times.clone(),
            num_instruments: n_instruments,
        }
    }
}

// ===========================================================================
// Standalone helper: per-instrument decomposition
// ===========================================================================

/// Compute per-instrument KRDs (one AAD pass per instrument).
///
/// This is useful when you need instrument-level risk attribution, not just
/// portfolio totals. Cost: O(M) where M = number of instruments (one tape
/// per instrument).
pub fn per_instrument_krds(
    instruments: &[Vec<Cashflow>],
    pillar_times: &[f64],
    pillar_rates: &[f64],
) -> Vec<Vec<f64>> {
    instruments
        .iter()
        .map(|cfs| {
            with_tape(|tape| {
                let rates: Vec<AReal> =
                    pillar_rates.iter().map(|&r| tape.input(r)).collect();
                let curve = DiscountCurveAD::from_zero_rates(pillar_times, &rates);
                let total: AReal = npv(cfs, &curve);
                let adj = adjoint_tl(total);
                rates.iter().map(|a| adj[a.idx] * 0.0001).collect()
            })
        })
        .collect()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cashflows::Cashflow;
    use approx::assert_abs_diff_eq;

    fn sample_pillars() -> (Vec<f64>, Vec<f64>) {
        (
            vec![1.0, 2.0, 3.0, 5.0, 10.0],
            vec![0.03, 0.032, 0.034, 0.038, 0.042],
        )
    }

    #[test]
    fn single_bond_portfolio() {
        let (times, rates) = sample_pillars();
        let mut port = Portfolio::new(&times, &rates);
        port.add_bond(0.05, 100.0, 5, 1.0);

        let g = port.compute();
        assert!(g.total_npv > 90.0 && g.total_npv < 120.0,
                "bond NPV = {}", g.total_npv);
        assert_eq!(g.instrument_npvs.len(), 1);
        assert_eq!(g.key_rate_durations.len(), 5);

        // All KRDs should be negative for a long bond
        for (i, &krd) in g.key_rate_durations.iter().enumerate() {
            assert!(krd <= 0.0 + 1e-10, "KRD[{}] = {} should be <= 0", i, krd);
        }
    }

    #[test]
    fn hedged_portfolio_near_zero_duration() {
        let (times, rates) = sample_pillars();
        let mut port = Portfolio::new(&times, &rates);

        // Long 5y bond
        port.add_bond(0.05, 100.0, 5, 1.0);
        // Short 5y bond with same coupon → net duration ≈ 0
        port.add_bond(0.05, 100.0, 5, -1.0);

        let g = port.compute();
        assert_abs_diff_eq!(g.total_npv, 0.0, epsilon = 1e-10);

        for &krd in g.key_rate_durations.iter() {
            assert_abs_diff_eq!(krd, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn two_bond_portfolio() {
        let (times, rates) = sample_pillars();
        let mut port = Portfolio::new(&times, &rates);
        port.add_bond(0.04, 100.0, 3, 1.0);
        port.add_bond(0.05, 100.0, 5, 0.5);

        let g = port.compute();
        assert_eq!(g.num_instruments, 2);
        assert_eq!(g.instrument_npvs.len(), 2);
        assert!(g.total_npv > 0.0);
    }

    #[test]
    fn portfolio_vs_individual_instruments() {
        let (times, rates) = sample_pillars();

        // MKT sensitivities via portfolio
        let mut port = Portfolio::new(&times, &rates);
        port.add_bond(0.04, 100.0, 3, 1.0);
        port.add_bond(0.05, 100.0, 5, 1.0);
        let g = port.compute();

        // MKT sensitivities via per-instrument decomposition
        let cfs_1 = fixed_rate_bond_cashflows(0.04, 100.0, 3);
        let cfs_2 = fixed_rate_bond_cashflows(0.05, 100.0, 5);
        let instruments = vec![cfs_1, cfs_2];
        let per_inst = per_instrument_krds(&instruments, &times, &rates);

        // Portfolio KRD should equal sum of per-instrument KRDs
        for j in 0..times.len() {
            let sum_krd: f64 = per_inst.iter().map(|krds| krds[j]).sum();
            assert_abs_diff_eq!(g.key_rate_durations[j], sum_krd, epsilon = 1e-10);
        }
    }

    #[test]
    fn portfolio_npv_equals_sum_of_weighted() {
        let (times, rates) = sample_pillars();
        let mut port = Portfolio::new(&times, &rates);
        port.add_bond(0.04, 100.0, 3, 2.0);
        port.add_bond(0.05, 100.0, 5, -0.5);

        let g = port.compute();

        let expected_npv = g.instrument_npvs[0] * 2.0 + g.instrument_npvs[1] * (-0.5);
        assert_abs_diff_eq!(g.total_npv, expected_npv, epsilon = 1e-10);
    }

    #[test]
    fn custom_cashflows() {
        let (times, rates) = sample_pillars();
        let mut port = Portfolio::new(&times, &rates);

        // A floating-rate note proxy: single large cashflow at maturity
        let cfs = vec![Cashflow { time: 2.0, amount: 100.0 }];
        port.add_cashflows(cfs, 1.0, "ZCB 2y");

        let g = port.compute();
        assert_eq!(g.num_instruments, 1);
        assert!(g.total_npv > 0.0);
    }

    #[test]
    fn per_instrument_krds_match_portfolio() {
        let (times, rates) = sample_pillars();
        let cfs = vec![fixed_rate_bond_cashflows(0.05, 100.0, 5)];
        let krds = per_instrument_krds(&cfs, &times, &rates);

        assert_eq!(krds.len(), 1);
        assert_eq!(krds[0].len(), 5);

        // All should be negative
        for &k in &krds[0] {
            assert!(k <= 0.0 + 1e-10);
        }
    }

    #[test]
    fn sensitivity_vs_bump() {
        let (times, rates) = sample_pillars();
        let mut port = Portfolio::new(&times, &rates);
        port.add_bond(0.05, 100.0, 5, 1.0);

        let g = port.compute();

        // Verify against bump-and-reprice
        let bump = 1e-4; // 1 bp
        for i in 0..times.len() {
            let mut rates_up = rates.clone();
            rates_up[i] += bump;
            let curve_up = DiscountCurveAD::from_zero_rates(&times, &rates_up);
            let cfs = fixed_rate_bond_cashflows(0.05, 100.0, 5);
            let npv_up: f64 = npv(&cfs, &curve_up);

            let mut rates_dn = rates.clone();
            rates_dn[i] -= bump;
            let curve_dn = DiscountCurveAD::from_zero_rates(&times, &rates_dn);
            let npv_dn: f64 = npv(&cfs, &curve_dn);

            let bump_krd = (npv_up - npv_dn) / 2.0; // already per 1bp (bump was 1bp)
            assert_abs_diff_eq!(g.key_rate_durations[i], bump_krd, epsilon = 1e-5);
        }
    }
}
