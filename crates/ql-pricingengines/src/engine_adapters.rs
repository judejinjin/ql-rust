//! Concrete `PricingEngine<I>` implementations that bridge the trait-based
//! dispatch pattern to the existing free-function pricing API.
//!
//! Each engine struct holds the market data parameters required by the
//! underlying pricing function and implements [`PricingEngine::calculate`]
//! by delegating to it.
//!
//! ## Available Engines
//!
//! | Engine | Instrument | Pricing Function |
//! |--------|-----------|-----------------|
//! | [`AnalyticEuropeanEngine`] | `VanillaOption` | `price_european` |
//! | [`DiscountingSwapEngine`] | `VanillaSwap` | `price_swap` |
//! | [`DiscountingBondEngine`] | `FixedRateBond` | `price_bond` |
//! | [`MCEuropeanEngine`] | `VanillaOption` | `mc_european` |
//! | [`BinomialCRREngine`] | `VanillaOption` | `binomial_crr` |
//!
//! ## Example
//!
//! ```
//! use ql_core::engine::PricingEngine;
//! use ql_instruments::VanillaOption;
//! use ql_pricingengines::engine_adapters::AnalyticEuropeanEngine;
//! use ql_time::{Date, Month};
//!
//! let option = VanillaOption::european_call(100.0, Date::from_ymd(2026, Month::January, 15));
//! let engine = AnalyticEuropeanEngine {
//!     spot: 100.0,
//!     risk_free_rate: 0.05,
//!     dividend_yield: 0.0,
//!     volatility: 0.20,
//!     time_to_expiry: 1.0,
//! };
//! let result = engine.calculate(&option).unwrap();
//! assert!((result.npv - 10.45).abs() < 0.1);
//! ```

use ql_core::engine::PricingEngine;
use ql_core::errors::QLResult;
use ql_instruments::bond::FixedRateBond;
use ql_instruments::vanilla_swap::VanillaSwap;
use ql_instruments::VanillaOption;
use ql_time::Date;

use crate::analytic_european::{price_european, AnalyticEuropeanResults};
use crate::discounting::{price_bond, price_swap, BondResults, SwapResults};

// ═══════════════════════════════════════════════════════════════
// AnalyticEuropeanEngine
// ═══════════════════════════════════════════════════════════════

/// Black-Scholes analytic engine for European vanilla options.
///
/// Holds market data as simple scalars. For reactive (observable-based)
/// pricing, wrap in a [`LazyInstrument`](ql_core::engine::LazyInstrument).
#[derive(Debug, Clone)]
pub struct AnalyticEuropeanEngine {
    /// Current underlying spot price.
    pub spot: f64,
    /// Continuously compounded risk-free rate.
    pub risk_free_rate: f64,
    /// Continuously compounded dividend yield.
    pub dividend_yield: f64,
    /// Annualized Black-Scholes volatility (σ).
    pub volatility: f64,
    /// Time to expiry in years.
    pub time_to_expiry: f64,
}

impl PricingEngine<VanillaOption> for AnalyticEuropeanEngine {
    type Result = AnalyticEuropeanResults;

    fn calculate(&self, option: &VanillaOption) -> QLResult<AnalyticEuropeanResults> {
        Ok(price_european(
            option,
            self.spot,
            self.risk_free_rate,
            self.dividend_yield,
            self.volatility,
            self.time_to_expiry,
        ))
    }
}

// ═══════════════════════════════════════════════════════════════
// DiscountingSwapEngine
// ═══════════════════════════════════════════════════════════════

/// Discounting engine for vanilla interest rate swaps.
///
/// Prices a swap by discounting both legs using the provided yield curve.
pub struct DiscountingSwapEngine<'a> {
    /// Yield curve used for discounting (and forecasting, in single-curve mode).
    pub curve: &'a dyn ql_termstructures::YieldTermStructure,
    /// Settlement date.
    pub settle: Date,
}

impl<'a> PricingEngine<VanillaSwap> for DiscountingSwapEngine<'a> {
    type Result = SwapResults;

    fn calculate(&self, swap: &VanillaSwap) -> QLResult<SwapResults> {
        Ok(price_swap(swap, self.curve, self.settle))
    }
}

// ═══════════════════════════════════════════════════════════════
// DiscountingBondEngine
// ═══════════════════════════════════════════════════════════════

/// Discounting engine for fixed-rate bonds.
///
/// Prices a bond by discounting its cashflows using the provided yield curve.
pub struct DiscountingBondEngine<'a> {
    /// Yield curve used for discounting.
    pub curve: &'a dyn ql_termstructures::YieldTermStructure,
    /// Settlement date.
    pub settle: Date,
}

impl<'a> PricingEngine<FixedRateBond> for DiscountingBondEngine<'a> {
    type Result = BondResults;

    fn calculate(&self, bond: &FixedRateBond) -> QLResult<BondResults> {
        Ok(price_bond(bond, self.curve, self.settle))
    }
}

// ═══════════════════════════════════════════════════════════════
// MCEuropeanEngine
// ═══════════════════════════════════════════════════════════════

/// Monte Carlo engine for European options under GBM.
///
/// Uses exact log-normal sampling (single time step) with optional
/// antithetic variance reduction.
#[derive(Debug, Clone)]
pub struct MCEuropeanEngine {
    /// Current underlying spot price.
    pub spot: f64,
    /// Continuously compounded risk-free rate.
    pub risk_free_rate: f64,
    /// Continuously compounded dividend yield.
    pub dividend_yield: f64,
    /// Annualized volatility.
    pub volatility: f64,
    /// Time to expiry in years.
    pub time_to_expiry: f64,
    /// Number of Monte Carlo paths.
    pub num_paths: usize,
    /// Whether to use antithetic variance reduction.
    pub antithetic: bool,
    /// Random seed.
    pub seed: u64,
}

impl PricingEngine<VanillaOption> for MCEuropeanEngine {
    type Result = ql_methods::mc_engines::MCResult;

    fn calculate(&self, option: &VanillaOption) -> QLResult<ql_methods::mc_engines::MCResult> {
        Ok(ql_methods::mc_engines::mc_european(
            self.spot,
            option.strike(),
            self.risk_free_rate,
            self.dividend_yield,
            self.volatility,
            self.time_to_expiry,
            option.option_type(),
            self.num_paths,
            self.antithetic,
            self.seed,
        ))
    }
}

// ═══════════════════════════════════════════════════════════════
// BinomialCRREngine
// ═══════════════════════════════════════════════════════════════

/// Binomial CRR (Cox-Ross-Rubinstein) lattice engine.
///
/// Supports both European and American exercise via backward induction.
#[derive(Debug, Clone)]
pub struct BinomialCRREngine {
    /// Current underlying spot price.
    pub spot: f64,
    /// Continuously compounded risk-free rate.
    pub risk_free_rate: f64,
    /// Continuously compounded dividend yield.
    pub dividend_yield: f64,
    /// Annualized volatility.
    pub volatility: f64,
    /// Time to expiry in years.
    pub time_to_expiry: f64,
    /// Number of binomial tree steps.
    pub num_steps: usize,
}

impl PricingEngine<VanillaOption> for BinomialCRREngine {
    type Result = ql_methods::lattice::LatticeResult;

    fn calculate(&self, option: &VanillaOption) -> QLResult<ql_methods::lattice::LatticeResult> {
        use ql_instruments::payoff::Exercise;

        let is_call = option.option_type() == ql_instruments::OptionType::Call;
        let is_american = matches!(option.exercise, Exercise::American { .. });

        Ok(ql_methods::lattice::binomial_crr(
            self.spot,
            option.strike(),
            self.risk_free_rate,
            self.dividend_yield,
            self.volatility,
            self.time_to_expiry,
            is_call,
            is_american,
            self.num_steps,
        ))
    }
}

// ═══════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use ql_core::engine::PricingEngine;
    use ql_instruments::VanillaOption;
    use ql_time::{Date, Month};

    #[test]
    fn analytic_european_engine_call() {
        let option = VanillaOption::european_call(100.0, Date::from_ymd(2026, Month::January, 15));
        let engine = AnalyticEuropeanEngine {
            spot: 100.0,
            risk_free_rate: 0.05,
            dividend_yield: 0.0,
            volatility: 0.20,
            time_to_expiry: 1.0,
        };
        let result = engine.calculate(&option).unwrap();
        assert!((result.npv - 10.45).abs() < 0.1, "BS call ≈ 10.45, got {}", result.npv);
        assert!(result.delta > 0.5, "ATM call delta > 0.5");
    }

    #[test]
    fn analytic_european_engine_put() {
        let option = VanillaOption::european_put(100.0, Date::from_ymd(2026, Month::January, 15));
        let engine = AnalyticEuropeanEngine {
            spot: 100.0,
            risk_free_rate: 0.05,
            dividend_yield: 0.0,
            volatility: 0.20,
            time_to_expiry: 1.0,
        };
        let result = engine.calculate(&option).unwrap();
        // Put-call parity: P = C - S + K·e^(-rT) ≈ 10.45 - 100 + 100·0.9512 ≈ 5.57
        assert!((result.npv - 5.57).abs() < 0.1, "BS put ≈ 5.57, got {}", result.npv);
        assert!(result.delta < 0.0, "put delta negative");
    }

    #[test]
    fn mc_european_engine_matches_bs() {
        let option = VanillaOption::european_call(100.0, Date::from_ymd(2026, Month::January, 15));
        let engine = MCEuropeanEngine {
            spot: 100.0,
            risk_free_rate: 0.05,
            dividend_yield: 0.0,
            volatility: 0.20,
            time_to_expiry: 1.0,
            num_paths: 200_000,
            antithetic: true,
            seed: 42,
        };
        let result = engine.calculate(&option).unwrap();
        assert!((result.npv - 10.45).abs() < 0.3, "MC ≈ BS, got {}", result.npv);
    }

    #[test]
    fn binomial_crr_engine_european() {
        let option = VanillaOption::european_call(100.0, Date::from_ymd(2026, Month::January, 15));
        let engine = BinomialCRREngine {
            spot: 100.0,
            risk_free_rate: 0.05,
            dividend_yield: 0.0,
            volatility: 0.20,
            time_to_expiry: 1.0,
            num_steps: 500,
        };
        let result = engine.calculate(&option).unwrap();
        assert!((result.npv - 10.45).abs() < 0.2, "CRR → BS, got {}", result.npv);
    }

    #[test]
    fn binomial_crr_engine_american_put() {
        use ql_instruments::payoff::{Exercise, Payoff};
        let expiry = Date::from_ymd(2026, Month::January, 15);
        let option = VanillaOption::new(
            Payoff::PlainVanilla {
                option_type: ql_instruments::OptionType::Put,
                strike: 100.0,
            },
            Exercise::American {
                earliest: Date::from_ymd(2025, Month::January, 15),
                expiry,
            },
        );
        let engine = BinomialCRREngine {
            spot: 100.0,
            risk_free_rate: 0.05,
            dividend_yield: 0.0,
            volatility: 0.20,
            time_to_expiry: 1.0,
            num_steps: 500,
        };
        let result = engine.calculate(&option).unwrap();
        // American put ≥ European put ≈ 5.57
        assert!(result.npv > 5.57, "American put > European put");
        assert!(result.npv < 7.0, "but not by too much: {}", result.npv);
    }

    #[test]
    fn engine_polymorphism() {
        // Demonstrate that different engines can be stored as trait objects
        let option = VanillaOption::european_call(100.0, Date::from_ymd(2026, Month::January, 15));

        let bs_engine = AnalyticEuropeanEngine {
            spot: 100.0,
            risk_free_rate: 0.05,
            dividend_yield: 0.0,
            volatility: 0.20,
            time_to_expiry: 1.0,
        };

        let mc_engine = MCEuropeanEngine {
            spot: 100.0,
            risk_free_rate: 0.05,
            dividend_yield: 0.0,
            volatility: 0.20,
            time_to_expiry: 1.0,
            num_paths: 100_000,
            antithetic: true,
            seed: 42,
        };

        // Both compute NPV close to each other
        let bs_npv = bs_engine.calculate(&option).unwrap().npv;
        let mc_npv = mc_engine.calculate(&option).unwrap().npv;
        assert!((bs_npv - mc_npv).abs() < 0.5, "BS={bs_npv}, MC={mc_npv}");
    }

    #[test]
    fn discounting_swap_engine() {
        use ql_termstructures::FlatForward;
        use ql_time::DayCounter;

        let today = Date::from_ymd(2025, Month::January, 15);
        let curve = FlatForward::new(today, 0.03, DayCounter::Actual365Fixed);

        // Build a simple swap: fixed 3% vs floating
        let swap = VanillaSwap::new(
            ql_instruments::SwapType::Payer,
            1_000_000.0,
            vec![], // fixed leg (empty for this test)
            vec![], // floating leg (empty for this test)
            0.03,
            0.0,
        );

        let engine = DiscountingSwapEngine {
            curve: &curve,
            settle: today,
        };
        let result = engine.calculate(&swap).unwrap();
        // With empty legs, NPV should be 0
        assert!(result.npv.abs() < 1e-10);
    }
}
