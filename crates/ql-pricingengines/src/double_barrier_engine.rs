//! Analytic double-barrier option engine (eigenfunction series).
//!
//! Prices European double-barrier knock-out options using the
//! eigenfunction expansion of the absorbed log-normal diffusion.
//! The series converges exponentially fast.  Knock-in prices are
//! obtained via in-out parity: KI = Vanilla − KO.
//!
//! # References
//! - Ikeda, M. and Kunitomo, N. (1992), "Pricing Options with Curved
//!   Boundaries", *Mathematical Finance* 2(4).
//! - Haug, E.G. (2007), *The Complete Guide to Option Pricing Formulas*,
//!   Chapter 4.

use std::f64::consts::PI;

use ql_instruments::OptionType;

use crate::analytic_european;

/// Result of a double-barrier pricing.
#[derive(Debug, Clone)]
pub struct DoubleBarrierResult {
    /// Net present value.
    pub npv: f64,
}

/// Price a European double-barrier **knock-out** option analytically.
///
/// Uses the eigenfunction expansion for the absorbed GBM process
/// between logarithmic barriers.  The series converges exponentially
/// and 20 terms suffice for all practical cases.
///
/// The spot must lie strictly between the two barriers:
/// `lower < spot < upper`.
#[allow(clippy::too_many_arguments)]
pub fn double_barrier_knockout(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    lower: f64,
    upper: f64,
    option_type: OptionType,
    n_terms: usize,
) -> DoubleBarrierResult {
    debug_assert!(lower < spot && spot < upper, "spot must lie between barriers");
    debug_assert!(lower < upper);

    let sigma2 = vol * vol;
    // Log-space variables
    let a = lower.ln(); // lower barrier in log
    let b = upper.ln(); // upper barrier in log
    let l = b - a; // barrier gap = ln(U/L)
    let x0 = spot.ln() - a; // ln(S/L), position within [0, l]

    // Drift parameter
    let alpha = (r - q - 0.5 * sigma2) / sigma2; // nu / sigma^2
    let beta = alpha * alpha * sigma2 / 2.0 + r; // eigenvalue base

    let ln_k = strike.ln();
    // Where the payoff starts being positive in log-space
    let c = (ln_k - a).max(0.0).min(l); // clamp to [0, l]

    // Prefactor
    let prefactor = (2.0 / l) * (-beta * t - alpha * (spot.ln())).exp();

    let mut sum = 0.0;
    for n in 1..=n_terms {
        let nf = n as f64;
        let npi_l = nf * PI / l;

        // Eigenvalue decay
        let eigen_decay = (-(npi_l * npi_l) * sigma2 * t / 2.0).exp();
        if eigen_decay < 1e-30 {
            break; // remaining terms are negligible
        }

        let sin_x0 = (npi_l * x0).sin();

        // Payoff integral J_n = ∫ payoff(e^{u+a}) * e^{alpha*(u+a)} * sin(n*pi*u/l) du
        // For a call:  J_n = ∫_c^l (e^{u+a} - K) * e^{alpha*(u+a)} * sin(nπu/l) du
        //            = L^{alpha} * [L * I(alpha+1, c, l) - K * I(alpha, c, l)]
        // where I(p, c_lo, c_hi) = ∫_{c_lo}^{c_hi} e^{p*u} * sin(nπu/l) du
        //
        // For a put:  J_n = ∫_0^c (K - e^{u+a}) * e^{alpha*(u+a)} * sin(nπu/l) du
        //           = L^{alpha} * [K * I(alpha, 0, c) - L * I(alpha+1, 0, c)]

        let la = lower.powf(alpha);
        let j_n = match option_type {
            OptionType::Call => {
                let i1 = integral_exp_sin(alpha + 1.0, npi_l, c, l);
                let i2 = integral_exp_sin(alpha, npi_l, c, l);
                la * (lower * i1 - strike * i2)
            }
            OptionType::Put => {
                let i1 = integral_exp_sin(alpha, npi_l, 0.0, c);
                let i2 = integral_exp_sin(alpha + 1.0, npi_l, 0.0, c);
                la * (strike * i1 - lower * i2)
            }
        };

        sum += sin_x0 * eigen_decay * j_n;
    }

    let npv = (prefactor * sum).max(0.0);
    DoubleBarrierResult { npv }
}

/// Compute ∫_{c_lo}^{c_hi} e^{p*u} * sin(b*u) du   (closed-form).
fn integral_exp_sin(p: f64, b: f64, c_lo: f64, c_hi: f64) -> f64 {
    let denom = p * p + b * b;
    if denom < 1e-30 {
        return 0.0;
    }
    let f = |u: f64| {
        let eu = (p * u).exp();
        eu * (p * (b * u).sin() - b * (b * u).cos()) / denom
    };
    f(c_hi) - f(c_lo)
}

/// Price a European double-barrier **knock-in** option via in-out parity.
///
/// knock_in = vanilla − knock_out
#[allow(clippy::too_many_arguments)]
pub fn double_barrier_knockin(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    vol: f64,
    t: f64,
    lower: f64,
    upper: f64,
    option_type: OptionType,
    n_terms: usize,
) -> DoubleBarrierResult {
    let vanilla =
        analytic_european::black_scholes_price(spot, strike, r, q, vol, t, option_type);
    let ko =
        double_barrier_knockout(spot, strike, r, q, vol, t, lower, upper, option_type, n_terms);
    DoubleBarrierResult {
        npv: (vanilla.npv - ko.npv).max(0.0),
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const SPOT: f64 = 100.0;
    const STRIKE: f64 = 100.0;
    const R: f64 = 0.05;
    const Q: f64 = 0.0;
    const VOL: f64 = 0.20;
    const T: f64 = 1.0;
    const LOWER: f64 = 80.0;
    const UPPER: f64 = 120.0;

    #[test]
    fn ko_call_positive_and_less_than_vanilla() {
        let ko = double_barrier_knockout(SPOT, STRIKE, R, Q, VOL, T, LOWER, UPPER, OptionType::Call, 20);
        let van = analytic_european::black_scholes_price(SPOT, STRIKE, R, Q, VOL, T, OptionType::Call);
        assert!(ko.npv > 0.0, "KO call should be positive: {}", ko.npv);
        assert!(
            ko.npv < van.npv + 0.01,
            "KO call {} should be <= vanilla call {}",
            ko.npv,
            van.npv
        );
    }

    #[test]
    fn ko_put_positive_and_less_than_vanilla() {
        let ko = double_barrier_knockout(SPOT, STRIKE, R, Q, VOL, T, LOWER, UPPER, OptionType::Put, 20);
        let van = analytic_european::black_scholes_price(SPOT, STRIKE, R, Q, VOL, T, OptionType::Put);
        assert!(ko.npv > 0.0, "KO put should be positive: {}", ko.npv);
        assert!(
            ko.npv < van.npv + 0.01,
            "KO put {} should be <= vanilla put {}",
            ko.npv,
            van.npv
        );
    }

    #[test]
    fn ki_plus_ko_equals_vanilla() {
        let ko = double_barrier_knockout(SPOT, STRIKE, R, Q, VOL, T, LOWER, UPPER, OptionType::Call, 20);
        let ki = double_barrier_knockin(SPOT, STRIKE, R, Q, VOL, T, LOWER, UPPER, OptionType::Call, 20);
        let van = analytic_european::black_scholes_price(SPOT, STRIKE, R, Q, VOL, T, OptionType::Call);
        let diff = (ki.npv + ko.npv - van.npv).abs();
        assert!(
            diff < 0.10,
            "KI ({:.4}) + KO ({:.4}) = {:.4} should ≈ vanilla {:.4}",
            ki.npv, ko.npv, ki.npv + ko.npv, van.npv
        );
    }

    #[test]
    fn wide_barriers_approach_vanilla() {
        // Moderately wide barriers → KO should still be a substantial fraction of vanilla
        let ko = double_barrier_knockout(SPOT, STRIKE, R, Q, VOL, T, 50.0, 200.0, OptionType::Call, 30);
        let van = analytic_european::black_scholes_price(SPOT, STRIKE, R, Q, VOL, T, OptionType::Call);
        // With barriers at 50/200 (50% away), most paths don't hit → KO should be close to vanilla
        assert!(
            ko.npv > van.npv * 0.5,
            "Wide barriers KO {:.4} should be >50% of vanilla {:.4}",
            ko.npv, van.npv
        );
    }

    #[test]
    fn ko_decreases_with_tighter_barriers() {
        let ko_wide = double_barrier_knockout(SPOT, STRIKE, R, Q, VOL, T, 70.0, 130.0, OptionType::Call, 20);
        let ko_tight = double_barrier_knockout(SPOT, STRIKE, R, Q, VOL, T, 90.0, 110.0, OptionType::Call, 20);
        assert!(
            ko_tight.npv < ko_wide.npv,
            "Tighter barriers ({:.4}) should give lower KO than wider ({:.4})",
            ko_tight.npv, ko_wide.npv
        );
    }
}
