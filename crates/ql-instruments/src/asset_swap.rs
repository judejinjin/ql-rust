//! Asset swap and equity total return swap instruments.
//!
//! ## Asset Swap
//!
//! An asset swap strips the credit risk from a bond by exchanging the bond's
//! fixed coupons (plus any premium/discount) for LIBOR/SOFR + spread.
//! The **asset swap spread** is the spread that makes the package NPV = 0.
//!
//! ## Equity Total Return Swap (TRS)
//!
//! An equity TRS exchanges the total return on an equity (price appreciation
//! + dividends) for a floating rate + spread. Used for synthetic equity exposure.

use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════
// Asset Swap
// ═══════════════════════════════════════════════════════════════

/// Asset swap convention (par or market value).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssetSwapConvention {
    /// Par asset swap: investor pays par for the bond and receives
    /// the bond's coupons plus any premium/discount upfront.
    Par,
    /// Market-value asset swap: investor pays the bond's market price
    /// and there is no upfront adjustment.
    MarketValue,
}

/// An asset swap: bond + IRS that converts bond coupons to floating rate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetSwap {
    /// Par or market-value convention.
    pub convention: AssetSwapConvention,
    /// Bond's clean price (as a percentage of par, e.g., 98.5).
    pub bond_clean_price: f64,
    /// Bond's coupon rate (fixed).
    pub bond_coupon_rate: f64,
    /// Par amount / notional.
    pub notional: f64,
    /// Remaining coupon times (years from valuation).
    pub coupon_times: Vec<f64>,
    /// Year fractions for each coupon period.
    pub year_fractions: Vec<f64>,
    /// Discount factors at each coupon payment date.
    pub discount_factors: Vec<f64>,
}

/// Result of asset swap pricing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetSwapResult {
    /// Asset swap spread (over floating rate) that makes NPV = 0.
    pub asset_swap_spread: f64,
    /// NPV of the bond leg (fixed coupons + par redemption).
    pub bond_leg_npv: f64,
    /// Annuity (PV01) of the floating leg.
    pub floating_annuity: f64,
}

/// Compute the asset swap spread for a par or market-value asset swap.
///
/// For a **par** asset swap:
/// $$\text{ASW} = \frac{(100 - P) + \sum c_i \cdot \tau_i \cdot D_i - 100 \cdot D_n + 100}{\sum \tau_i \cdot D_i}$$
///
/// simplified to:
/// $$\text{ASW} = \frac{100 - P + \text{bond\_leg\_pv} - 100 \cdot D_n}{\text{annuity}}$$
///
/// For a **market-value** asset swap:
/// $$\text{ASW} = \frac{\text{bond\_leg\_pv} - P}{P \cdot \text{annuity} / 100}$$
pub fn price_asset_swap(swap: &AssetSwap) -> AssetSwapResult {
    let n = swap.coupon_times.len();
    assert!(!swap.coupon_times.is_empty(), "Asset swap needs at least one period");

    // Bond leg PV: PV of remaining coupons + par redemption
    let coupon_pv: f64 = (0..n)
        .map(|i| {
            swap.notional * swap.bond_coupon_rate * swap.year_fractions[i] * swap.discount_factors[i]
        })
        .sum();
    let par_pv = swap.notional * swap.discount_factors[n - 1]; // par at maturity
    let bond_leg_npv = coupon_pv + par_pv;

    // Floating leg annuity: sum(τ_i × D_i × N)
    let annuity: f64 = (0..n)
        .map(|i| swap.notional * swap.year_fractions[i] * swap.discount_factors[i])
        .sum();

    let asset_swap_spread = match swap.convention {
        AssetSwapConvention::Par => {
            // Investor pays par (100%) for the bond worth P.
            // Makes up the difference through the ASW spread.
            // ASW = (coupon_pv + par_pv - notional × P/100) / annuity
            let p = swap.bond_clean_price / 100.0; // convert to fraction
            (bond_leg_npv - swap.notional * p) / annuity
        }
        AssetSwapConvention::MarketValue => {
            // Investor pays market price. No upfront adjustment.
            // ASW = (bond_coupon_rate × annuity + par_pv - notional × P/100) / annuity
            // = bond_coupon_rate + (par_pv - notional × P/100) / annuity
            let p = swap.bond_clean_price / 100.0;
            swap.bond_coupon_rate + (par_pv - swap.notional * p) / annuity
        }
    };

    AssetSwapResult {
        asset_swap_spread,
        bond_leg_npv,
        floating_annuity: annuity,
    }
}

// ═══════════════════════════════════════════════════════════════
// Equity Total Return Swap
// ═══════════════════════════════════════════════════════════════

/// An equity total return swap.
///
/// The equity receiver gets total return (price + dividends) on the
/// reference equity and pays floating rate + spread on the notional.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquityTRS {
    /// Notional amount.
    pub notional: f64,
    /// Initial equity price (at trade inception).
    pub initial_price: f64,
    /// Current equity price.
    pub current_price: f64,
    /// Accrued dividends (already paid or accrued during the period).
    pub accrued_dividends: f64,
    /// Floating rate for the funding leg (annualized).
    pub floating_rate: f64,
    /// Spread over the floating rate.
    pub spread: f64,
    /// Accrual period (year fraction since last reset).
    pub accrual_period: f64,
    /// Discount factor to payment date.
    pub discount_factor: f64,
}

/// Result of equity TRS pricing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquityTRSResult {
    /// NPV of the equity leg (total return to receiver).
    pub equity_leg_npv: f64,
    /// NPV of the funding leg (floating + spread to payer).
    pub funding_leg_npv: f64,
    /// Net NPV (equity leg - funding leg) from equity receiver's perspective.
    pub npv: f64,
    /// Return on equity since last reset.
    pub equity_return: f64,
}

/// Price an equity total return swap.
///
/// Equity leg NPV = D × N × [(S_t/S_0 - 1) + div/S_0]
/// Funding leg NPV = D × N × (r + s) × τ
///
/// NPV from equity receiver's perspective = equity_leg - funding_leg.
pub fn price_equity_trs(trs: &EquityTRS) -> EquityTRSResult {
    let equity_return = (trs.current_price - trs.initial_price) / trs.initial_price
        + trs.accrued_dividends / trs.initial_price;

    let equity_leg_npv = trs.discount_factor * trs.notional * equity_return;

    let funding_leg_npv = trs.discount_factor
        * trs.notional
        * (trs.floating_rate + trs.spread)
        * trs.accrual_period;

    let npv = equity_leg_npv - funding_leg_npv;

    EquityTRSResult {
        equity_leg_npv,
        funding_leg_npv,
        npv,
        equity_return,
    }
}

/// Compute the fair spread of an equity TRS (spread that makes NPV = 0).
///
/// fair_spread = equity_return / τ - floating_rate
pub fn equity_trs_fair_spread(trs: &EquityTRS) -> f64 {
    if trs.accrual_period.abs() < 1e-15 {
        return 0.0;
    }
    let equity_return = (trs.current_price - trs.initial_price) / trs.initial_price
        + trs.accrued_dividends / trs.initial_price;
    equity_return / trs.accrual_period - trs.floating_rate
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ── Asset Swap tests ─────────────────────────────────────

    #[test]
    fn par_asset_swap_at_par() {
        // Bond trading at par: ASW spread ≈ coupon rate - swap rate
        let swap = AssetSwap {
            convention: AssetSwapConvention::Par,
            bond_clean_price: 100.0,
            bond_coupon_rate: 0.05,
            notional: 1_000_000.0,
            coupon_times: vec![0.5, 1.0, 1.5, 2.0],
            year_fractions: vec![0.5; 4],
            discount_factors: vec![0.98, 0.96, 0.94, 0.92],
        };
        let result = price_asset_swap(&swap);
        // At par, ASW = coupon_rate + (par_pv - notional) / annuity
        assert!(result.asset_swap_spread > 0.0);
    }

    #[test]
    fn par_asset_swap_discount_bond() {
        // Bond at 95: should have higher ASW spread than at par
        let at_par = AssetSwap {
            convention: AssetSwapConvention::Par,
            bond_clean_price: 100.0,
            bond_coupon_rate: 0.04,
            notional: 1_000_000.0,
            coupon_times: vec![1.0, 2.0, 3.0],
            year_fractions: vec![1.0; 3],
            discount_factors: vec![0.96, 0.92, 0.88],
        };
        let at_95 = AssetSwap {
            bond_clean_price: 95.0,
            ..at_par.clone()
        };
        let r_par = price_asset_swap(&at_par);
        let r_95 = price_asset_swap(&at_95);
        assert!(
            r_95.asset_swap_spread > r_par.asset_swap_spread,
            "Discount bond ASW ({}) > par ASW ({})",
            r_95.asset_swap_spread,
            r_par.asset_swap_spread
        );
    }

    #[test]
    fn market_value_asset_swap() {
        let swap = AssetSwap {
            convention: AssetSwapConvention::MarketValue,
            bond_clean_price: 102.0,
            bond_coupon_rate: 0.05,
            notional: 1_000_000.0,
            coupon_times: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            year_fractions: vec![1.0; 5],
            discount_factors: vec![0.96, 0.92, 0.88, 0.85, 0.82],
        };
        let result = price_asset_swap(&swap);
        // Premium bond with 5% coupon should have positive ASW
        assert!(result.asset_swap_spread > 0.0);
        assert!(result.floating_annuity > 0.0);
    }

    // ── Equity TRS tests ─────────────────────────────────────

    #[test]
    fn equity_trs_positive_return() {
        let trs = EquityTRS {
            notional: 10_000_000.0,
            initial_price: 100.0,
            current_price: 110.0,
            accrued_dividends: 2.0,
            floating_rate: 0.05,
            spread: 0.002,
            accrual_period: 0.25,
            discount_factor: 0.99,
        };
        let result = price_equity_trs(&trs);
        assert!(result.equity_return > 0.0);
        assert!(result.equity_leg_npv > 0.0);
        assert!(result.funding_leg_npv > 0.0);
    }

    #[test]
    fn equity_trs_negative_return() {
        let trs = EquityTRS {
            notional: 10_000_000.0,
            initial_price: 100.0,
            current_price: 90.0,
            accrued_dividends: 1.0,
            floating_rate: 0.05,
            spread: 0.002,
            accrual_period: 0.25,
            discount_factor: 0.99,
        };
        let result = price_equity_trs(&trs);
        // Price down 10%, dividends +1% → return = -9%
        assert_abs_diff_eq!(result.equity_return, -0.09, epsilon = 1e-10);
    }

    #[test]
    fn equity_trs_fair_spread_test() {
        let trs = EquityTRS {
            notional: 10_000_000.0,
            initial_price: 100.0,
            current_price: 105.0,
            accrued_dividends: 1.0,
            floating_rate: 0.05,
            spread: 0.0, // placeholder
            accrual_period: 0.25,
            discount_factor: 0.99,
        };
        let fs = equity_trs_fair_spread(&trs);
        // At fair spread, NPV should be ~0
        let trs_fair = EquityTRS { spread: fs, ..trs };
        let result = price_equity_trs(&trs_fair);
        assert_abs_diff_eq!(result.npv, 0.0, epsilon = 1.0);
    }

    #[test]
    fn equity_trs_no_change() {
        let trs = EquityTRS {
            notional: 10_000_000.0,
            initial_price: 100.0,
            current_price: 100.0,
            accrued_dividends: 0.0,
            floating_rate: 0.05,
            spread: 0.01,
            accrual_period: 0.25,
            discount_factor: 1.0,
        };
        let result = price_equity_trs(&trs);
        assert_abs_diff_eq!(result.equity_return, 0.0);
        assert_abs_diff_eq!(result.equity_leg_npv, 0.0);
        // Funding leg should still be positive
        assert!(result.funding_leg_npv > 0.0);
        // NPV negative for equity receiver when equity flat
        assert!(result.npv < 0.0);
    }
}
