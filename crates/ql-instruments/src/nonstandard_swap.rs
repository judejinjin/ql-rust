//! Nonstandard (irregular) interest rate swap.
//!
//! Supports per-period notionals, per-period fixed rates, per-period
//! spreads, amortizing schedules, and step-up/step-down structures
//! that cannot be represented by a `VanillaSwap`.
//!
//! ## Examples
//!
//! ```rust,no_run
//! use ql_instruments::nonstandard_swap::{NonstandardSwap, AmortizationType};
//! use ql_instruments::vanilla_swap::SwapType;
//!
//! // Amortizing swap: 10M notional, linear amortization over 5 annual periods
//! let swap = NonstandardSwap::amortizing(
//!     SwapType::Payer,
//!     10_000_000.0,
//!     5,
//!     AmortizationType::Linear,
//!     &[0.03],       // fixed rate (constant)
//!     &[0.0],        // spread (constant)
//! );
//! assert_eq!(swap.notionals.len(), 5);
//! ```

use serde::{Deserialize, Serialize};

use ql_cashflows::Leg;
use crate::vanilla_swap::SwapType;

/// Amortization type for an amortizing swap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AmortizationType {
    /// Equal reduction of notional each period (notional / n).
    Linear,
    /// Mortgage-style: equal total payment (principal + interest).
    Annuity,
    /// No amortization — bullet (constant notional).
    Bullet,
}

/// A nonstandard/irregular interest rate swap.
///
/// Unlike `VanillaSwap`, supports:
/// - Per-period notionals (amortizing, step-up, custom schedule)
/// - Per-period fixed rates (step-up coupon)
/// - Per-period floating leg spreads
/// - Differently-sized fixed and floating legs
///
/// The fixed and floating legs can be pre-built externally and attached,
/// or constructed from the per-period vectors.
#[derive(Debug)]
pub struct NonstandardSwap {
    /// Payer or receiver (from fixed-leg perspective).
    pub swap_type: SwapType,
    /// Per-period notionals (one per period).
    pub notionals: Vec<f64>,
    /// Per-period fixed rates (one per period; for step-up coupons).
    pub fixed_rates: Vec<f64>,
    /// Per-period floating leg spreads.
    pub spreads: Vec<f64>,
    /// Fixed leg (pre-built).
    pub fixed_leg: Leg,
    /// Floating leg (pre-built).
    pub floating_leg: Leg,
}

impl NonstandardSwap {
    /// Create from pre-built legs and per-period parameter vectors.
    pub fn new(
        swap_type: SwapType,
        notionals: Vec<f64>,
        fixed_rates: Vec<f64>,
        spreads: Vec<f64>,
        fixed_leg: Leg,
        floating_leg: Leg,
    ) -> Self {
        Self {
            swap_type,
            notionals,
            fixed_rates,
            spreads,
            fixed_leg,
            floating_leg,
        }
    }

    /// Create an amortizing swap with automatic notional schedule.
    ///
    /// * `initial_notional` — starting notional
    /// * `n_periods` — number of payment periods
    /// * `amort_type` — Linear, Annuity, or Bullet
    /// * `rates` — fixed rate(s) (last value extended if fewer than n_periods)
    /// * `spreads` — floating spread(s) (last extended)
    ///
    /// Returns a swap without pre-built legs (legs will be empty).
    /// Use `build_legs` or pass to a pricing function that constructs legs.
    pub fn amortizing(
        swap_type: SwapType,
        initial_notional: f64,
        n_periods: usize,
        amort_type: AmortizationType,
        rates: &[f64],
        spreads: &[f64],
    ) -> Self {
        let notionals = match amort_type {
            AmortizationType::Bullet => vec![initial_notional; n_periods],
            AmortizationType::Linear => {
                let step = initial_notional / n_periods as f64;
                (0..n_periods)
                    .map(|i| initial_notional - i as f64 * step)
                    .collect()
            }
            AmortizationType::Annuity => {
                // Mortgage-style: assume average fixed rate for annuity calculation
                let r = if rates.is_empty() { 0.05 } else { rates[0] };
                if r.abs() < 1e-12 {
                    // Zero rate → linear
                    let step = initial_notional / n_periods as f64;
                    (0..n_periods)
                        .map(|i| initial_notional - i as f64 * step)
                        .collect()
                } else {
                    let annuity = initial_notional * r / (1.0 - (1.0 + r).powi(-(n_periods as i32)));
                    let mut nots = Vec::with_capacity(n_periods);
                    let mut outstanding = initial_notional;
                    for _ in 0..n_periods {
                        nots.push(outstanding);
                        let interest = outstanding * r;
                        let principal = annuity - interest;
                        outstanding = (outstanding - principal).max(0.0);
                    }
                    nots
                }
            }
        };

        let fixed_rates = extend_vec(rates, n_periods);
        let spread_vec = extend_vec(spreads, n_periods);

        Self {
            swap_type,
            notionals,
            fixed_rates,
            spreads: spread_vec,
            fixed_leg: Vec::new(),
            floating_leg: Vec::new(),
        }
    }

    /// Create a step-up swap where the fixed rate increases each period.
    pub fn step_up(
        swap_type: SwapType,
        notional: f64,
        fixed_rates: Vec<f64>,
        spread: f64,
    ) -> Self {
        let n = fixed_rates.len();
        Self {
            swap_type,
            notionals: vec![notional; n],
            fixed_rates,
            spreads: vec![spread; n],
            fixed_leg: Vec::new(),
            floating_leg: Vec::new(),
        }
    }

    /// Number of periods.
    pub fn n_periods(&self) -> usize {
        self.notionals.len()
    }

    /// Total initial notional.
    pub fn initial_notional(&self) -> f64 {
        self.notionals.first().copied().unwrap_or(0.0)
    }

    /// Weighted average fixed rate.
    pub fn weighted_average_rate(&self) -> f64 {
        if self.notionals.is_empty() {
            return 0.0;
        }
        let total_not: f64 = self.notionals.iter().sum();
        if total_not.abs() < 1e-15 {
            return 0.0;
        }
        self.notionals
            .iter()
            .zip(self.fixed_rates.iter())
            .map(|(&n, &r)| n * r)
            .sum::<f64>()
            / total_not
    }
}

/// Extend a slice to length `n` by repeating the last element.
fn extend_vec(values: &[f64], n: usize) -> Vec<f64> {
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let v = if i < values.len() {
            values[i]
        } else {
            values.last().copied().unwrap_or(0.0)
        };
        result.push(v);
    }
    result
}

/// Price a nonstandard swap using per-period NPV calculation.
///
/// Rather than using pre-built legs, this computes each period's contribution
/// directly from the per-period notionals, rates, spreads, and discount factors.
///
/// # Arguments
/// * `swap` — the nonstandard swap
/// * `floating_rates` — per-period forward floating rates (e.g., from a yield curve)
/// * `discount_factors` — per-period discount factors at payment dates
///
/// # Returns
/// The swap NPV and per-leg NPVs.
pub fn price_nonstandard_swap(
    swap: &NonstandardSwap,
    floating_rates: &[f64],
    discount_factors: &[f64],
    year_fractions: &[f64],
) -> NonstandardSwapResults {
    let n = swap.n_periods();
    assert_eq!(floating_rates.len(), n, "floating_rates length mismatch");
    assert_eq!(discount_factors.len(), n, "discount_factors length mismatch");
    assert_eq!(year_fractions.len(), n, "year_fractions length mismatch");

    let sign = match swap.swap_type {
        SwapType::Payer => 1.0,
        SwapType::Receiver => -1.0,
    };

    let mut fixed_npv = 0.0;
    let mut floating_npv = 0.0;

    for i in 0..n {
        let df = discount_factors[i];
        let tau = year_fractions[i];
        let not = swap.notionals[i];
        let fixed_rate = swap.fixed_rates[i];
        let float_rate = floating_rates[i] + swap.spreads[i];

        fixed_npv += df * tau * not * fixed_rate;
        floating_npv += df * tau * not * float_rate;
    }

    let npv = sign * (floating_npv - fixed_npv);

    // Fair rate: the rate that makes NPV = 0
    let annuity: f64 = (0..n)
        .map(|i| discount_factors[i] * year_fractions[i] * swap.notionals[i])
        .sum();
    let fair_rate = if annuity.abs() > 1e-15 {
        floating_npv / annuity
    } else {
        0.0
    };

    NonstandardSwapResults {
        npv,
        fixed_leg_npv: fixed_npv,
        floating_leg_npv: floating_npv,
        fair_rate,
    }
}

/// Results from pricing a nonstandard swap.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NonstandardSwapResults {
    /// Net present value.
    pub npv: f64,
    /// Fixed leg NPV.
    pub fixed_leg_npv: f64,
    /// Floating leg NPV.
    pub floating_leg_npv: f64,
    /// Fair rate (makes NPV = 0).
    pub fair_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn bullet_swap_constant_notional() {
        let swap = NonstandardSwap::amortizing(
            SwapType::Payer,
            1_000_000.0,
            5,
            AmortizationType::Bullet,
            &[0.03],
            &[0.0],
        );
        assert_eq!(swap.n_periods(), 5);
        for &n in &swap.notionals {
            assert_abs_diff_eq!(n, 1_000_000.0);
        }
    }

    #[test]
    fn linear_amortizing_notionals() {
        let swap = NonstandardSwap::amortizing(
            SwapType::Payer,
            1_000_000.0,
            5,
            AmortizationType::Linear,
            &[0.03],
            &[0.0],
        );
        assert_eq!(swap.notionals.len(), 5);
        // Notionals should decrease: 1M, 800k, 600k, 400k, 200k
        assert_abs_diff_eq!(swap.notionals[0], 1_000_000.0);
        assert_abs_diff_eq!(swap.notionals[1], 800_000.0);
        assert_abs_diff_eq!(swap.notionals[4], 200_000.0);
    }

    #[test]
    fn annuity_amortizing_decreasing() {
        let swap = NonstandardSwap::amortizing(
            SwapType::Payer,
            1_000_000.0,
            10,
            AmortizationType::Annuity,
            &[0.05],
            &[0.0],
        );
        // Notionals should be decreasing
        for i in 1..swap.notionals.len() {
            assert!(
                swap.notionals[i] < swap.notionals[i - 1],
                "Annuity notionals should decrease: {} vs {}",
                swap.notionals[i],
                swap.notionals[i - 1]
            );
        }
    }

    #[test]
    fn step_up_rates() {
        let rates = vec![0.02, 0.025, 0.03, 0.035, 0.04];
        let swap = NonstandardSwap::step_up(SwapType::Payer, 1_000_000.0, rates.clone(), 0.001);
        assert_eq!(swap.fixed_rates, rates);
        assert_eq!(swap.notionals, vec![1_000_000.0; 5]);
    }

    #[test]
    fn weighted_average_rate() {
        // 60% at 3%, 40% at 4%
        let swap = NonstandardSwap {
            swap_type: SwapType::Payer,
            notionals: vec![600_000.0, 400_000.0],
            fixed_rates: vec![0.03, 0.04],
            spreads: vec![0.0, 0.0],
            fixed_leg: Vec::new(),
            floating_leg: Vec::new(),
        };
        let expected = (600_000.0 * 0.03 + 400_000.0 * 0.04) / 1_000_000.0;
        assert_abs_diff_eq!(swap.weighted_average_rate(), expected, epsilon = 1e-10);
    }

    #[test]
    fn price_flat_curve() {
        let swap = NonstandardSwap::amortizing(
            SwapType::Payer,
            1_000_000.0,
            5,
            AmortizationType::Bullet,
            &[0.03],
            &[0.0],
        );
        let r = 0.04;
        let dfs: Vec<f64> = (1..=5).map(|i| (-r * i as f64).exp()).collect();
        let fwd_rates = vec![0.04; 5]; // flat forward = 4%
        let yfs = vec![1.0; 5]; // annual

        let result = price_nonstandard_swap(&swap, &fwd_rates, &dfs, &yfs);
        // Payer receives floating (4%) and pays fixed (3%) → positive NPV
        assert!(result.npv > 0.0, "Payer swap NPV should be positive: {}", result.npv);
    }

    #[test]
    fn fair_rate_makes_npv_zero() {
        let swap = NonstandardSwap::amortizing(
            SwapType::Payer,
            1_000_000.0,
            5,
            AmortizationType::Linear,
            &[0.03], // placeholder
            &[0.0],
        );
        let r = 0.04;
        let dfs: Vec<f64> = (1..=5).map(|i| (-r * i as f64).exp()).collect();
        let fwd_rates = vec![0.04; 5];
        let yfs = vec![1.0; 5];

        let result = price_nonstandard_swap(&swap, &fwd_rates, &dfs, &yfs);

        // Price at fair rate should give ~0 NPV
        let mut swap2 = NonstandardSwap::amortizing(
            SwapType::Payer,
            1_000_000.0,
            5,
            AmortizationType::Linear,
            &[result.fair_rate],
            &[0.0],
        );
        swap2.fixed_rates = vec![result.fair_rate; 5];
        let result2 = price_nonstandard_swap(&swap2, &fwd_rates, &dfs, &yfs);
        assert_abs_diff_eq!(result2.npv, 0.0, epsilon = 1.0);
    }

    #[test]
    fn payer_receiver_symmetry() {
        let dfs: Vec<f64> = (1..=3).map(|i| (-0.04 * i as f64).exp()).collect();
        let fwd = vec![0.035; 3];
        let yfs = vec![1.0; 3];

        let payer = NonstandardSwap::amortizing(
            SwapType::Payer, 1_000_000.0, 3, AmortizationType::Bullet, &[0.03], &[0.0],
        );
        let receiver = NonstandardSwap::amortizing(
            SwapType::Receiver, 1_000_000.0, 3, AmortizationType::Bullet, &[0.03], &[0.0],
        );
        let p = price_nonstandard_swap(&payer, &fwd, &dfs, &yfs);
        let rec = price_nonstandard_swap(&receiver, &fwd, &dfs, &yfs);
        assert_abs_diff_eq!(p.npv, -rec.npv, epsilon = 0.01);
    }
}
