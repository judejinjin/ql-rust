//! Integration tests: Short-rate model pricing pipeline.

use ql_models::{CIRModel, VasicekModel};
use ql_pricingengines::{
    hw_caplet, hw_floorlet, hw_jamshidian_swaption,
    tree_bond_price, tree_swaption,
};
use ql_termstructures::{FlatForward, YieldTermStructure};
use ql_time::{Date, DayCounter, Month};

/// HW caplet + floorlet: both should be non-negative.
#[test]
fn hw_caplet_floorlet_bounds() {
    let a = 0.1;
    let sigma = 0.01;
    // Discount factors: P(0, 1.0) and P(0, 1.5) from flat 5%
    let today = Date::from_ymd(2025, Month::January, 15);
    let dc = DayCounter::Actual365Fixed;
    let curve = FlatForward::new(today, 0.05, dc);
    let p_fix = curve.discount_t(1.0);
    let p_pay = curve.discount_t(1.5);

    let caplet = hw_caplet(a, sigma, p_fix, p_pay, 1.0, 1.5, 0.05, 1_000_000.0);
    let floorlet = hw_floorlet(a, sigma, p_fix, p_pay, 1.0, 1.5, 0.05, 1_000_000.0);

    assert!(caplet.npv >= 0.0, "Caplet should be non-negative");
    assert!(floorlet.npv >= 0.0, "Floorlet should be non-negative");
    assert!(caplet.npv > 0.0 || floorlet.npv > 0.0, "At least one should be positive");
}

/// Trinomial tree bond price should converge to Vasicek analytic.
#[test]
fn tree_bond_converges_to_analytic() {
    let a = 0.1;
    let sigma = 0.01;
    let r0 = 0.05;
    let t = 5.0;

    let vasicek = VasicekModel::new(a, 0.05, sigma, r0);
    let analytic = vasicek.bond_price(t);
    let tree = tree_bond_price(a, sigma, r0, t, 500);

    let rel_err = ((tree.npv - analytic) / analytic).abs();
    assert!(
        rel_err < 0.005,
        "Tree vs analytic rel error {:.4}% (tree={:.6}, analytic={:.6})",
        rel_err * 100.0, tree.npv, analytic,
    );
}

/// Vasicek and CIR bond prices should be in (0, 1) for positive rates.
#[test]
fn short_rate_bond_prices_bounds() {
    let vasicek = VasicekModel::new(0.15, 0.04, 0.01, 0.04);
    let cir = CIRModel::new(0.15, 0.04, 0.05, 0.04);

    for t in [0.5, 1.0, 2.0, 5.0, 10.0, 30.0] {
        let pv = vasicek.bond_price(t);
        assert!(pv > 0.0 && pv < 1.0, "Vasicek P(0,{}) = {} not in (0,1)", t, pv);
        let pc = cir.bond_price(t);
        assert!(pc > 0.0 && pc < 1.0, "CIR P(0,{}) = {} not in (0,1)", t, pc);
    }
}

/// HW Jamshidian swaption should produce a positive price.
#[test]
fn hw_swaption_positive() {
    let a = 0.1;
    let sigma = 0.01;
    let today = Date::from_ymd(2025, Month::January, 15);
    let dc = DayCounter::Actual365Fixed;
    let curve = FlatForward::new(today, 0.05, dc);

    let swap_tenors = [1.5, 2.0, 2.5, 3.0];
    let dfs: Vec<f64> = swap_tenors.iter().map(|&t| curve.discount_t(t)).collect();
    let p_option = curve.discount_t(1.0);

    let result = hw_jamshidian_swaption(
        a, sigma, 1.0, &swap_tenors, 0.05, &dfs, p_option, 1_000_000.0, true,
    );
    assert!(result.npv > 0.0, "Payer swaption should be positive");
}

/// Tree swaption and Jamshidian should both produce positive prices
/// with the same order of magnitude. (They use different internal calibrations
/// so exact agreement isn't guaranteed for low step counts.)
#[test]
fn tree_swaption_and_jamshidian_both_positive() {
    let a = 0.1;
    let sigma = 0.01;
    let r0 = 0.05;
    let today = Date::from_ymd(2025, Month::January, 15);
    let dc = DayCounter::Actual365Fixed;
    let curve = FlatForward::new(today, r0, dc);

    let swap_tenors = [1.5, 2.0, 2.5, 3.0];
    let dfs: Vec<f64> = swap_tenors.iter().map(|&t| curve.discount_t(t)).collect();
    let p_option = curve.discount_t(1.0);

    let analytic = hw_jamshidian_swaption(
        a, sigma, 1.0, &swap_tenors, 0.05, &dfs, p_option, 1_000_000.0, true,
    );
    let tree = tree_swaption(
        a, sigma, r0, 1.0, &swap_tenors, 0.05, 1_000_000.0, true, 200,
    );

    assert!(analytic.npv > 0.0, "Jamshidian swaption should be positive");
    assert!(tree.npv > 0.0, "Tree swaption should be positive");
    // Same order of magnitude
    let ratio = tree.npv / analytic.npv;
    assert!(
        ratio > 0.1 && ratio < 10.0,
        "Tree/Jamshidian ratio {:.2} outside [0.1, 10]",
        ratio,
    );
}
