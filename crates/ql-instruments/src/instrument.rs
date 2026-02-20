//! Unified `Instrument` trait for financial instruments.
//!
//! Provides a common interface that every financial instrument can implement,
//! enabling heterogeneous portfolio management via `dyn Instrument`.
//!
//! # Design
//! - Methods return `Option` when not all instruments can answer a query
//!   (e.g. time-parameterised exotics don't track dates).
//! - No global state, no interior mutability — all methods take `&self`.
//! - Pricing remains in free functions; this trait is for identification
//!   and portfolio-level queries.

use ql_time::Date;

/// Minimal interface for financial instruments.
///
/// All concrete instruments (bonds, swaps, options, exotics) implement this
/// trait so that portfolios can hold `Vec<Box<dyn Instrument>>`.
pub trait Instrument {
    /// Human-readable instrument type name (e.g. `"FixedRateBond"`, `"VanillaOption"`).
    fn instrument_type(&self) -> &str;

    /// Whether the instrument has expired as of `ref_date`.
    ///
    /// Returns `None` if the instrument doesn't track dates natively
    /// (e.g. time-parameterised exotics that only store year-fractions).
    fn is_expired(&self, ref_date: Date) -> Option<bool>;

    /// The maturity / last relevant date, if available.
    fn maturity_date(&self) -> Option<Date>;

    /// Notional or face amount, if applicable.
    fn notional(&self) -> Option<f64> {
        None
    }
}

// ──────────────────── Implementations ────────────────────

use crate::bond::FixedRateBond;
use crate::floating_rate_bond::FloatingRateBond;
use crate::zero_coupon_bond::ZeroCouponBond;
use crate::amortizing_bond::AmortizingBond;
use crate::callable_bond::CallableBond;
use crate::convertible_bond::ConvertibleBond;
use crate::inflation_linked_bond::InflationLinkedBond;
use crate::vanilla_option::VanillaOption;
use crate::barrier_option::BarrierOption;
use crate::double_barrier_option::DoubleBarrierOption;
use crate::asian_option::AsianOption;
use crate::lookback_option::LookbackOption;
use crate::compound_option::CompoundOption;
use crate::chooser_option::ChooserOption;
use crate::cliquet_option::CliquetOption;
use crate::variance_swap::VarianceSwap;
use crate::vanilla_swap::VanillaSwap;
use crate::ois_swap::OISSwap;
use crate::fra::ForwardRateAgreement;
use crate::basis_swap::BasisSwap;
use crate::cross_currency_swap::{CrossCurrencySwap, FloatFloatSwap};
use crate::swaption::Swaption;
use crate::cap_floor::CapFloor;
use crate::credit_default_swap::CreditDefaultSwap;
use crate::stock::Stock;
use crate::bond_forward::BondForward;
use crate::composite_instrument::CompositeInstrument;

// ── Bonds ────────────────────────────────────────────────

impl Instrument for FixedRateBond {
    fn instrument_type(&self) -> &str { "FixedRateBond" }
    fn is_expired(&self, ref_date: Date) -> Option<bool> {
        Some(self.maturity_date < ref_date)
    }
    fn maturity_date(&self) -> Option<Date> { Some(self.maturity_date) }
    fn notional(&self) -> Option<f64> { Some(self.face_amount) }
}

impl Instrument for FloatingRateBond {
    fn instrument_type(&self) -> &str { "FloatingRateBond" }
    fn is_expired(&self, ref_date: Date) -> Option<bool> {
        Some(self.maturity_date < ref_date)
    }
    fn maturity_date(&self) -> Option<Date> { Some(self.maturity_date) }
    fn notional(&self) -> Option<f64> { Some(self.face_amount) }
}

impl Instrument for ZeroCouponBond {
    fn instrument_type(&self) -> &str { "ZeroCouponBond" }
    fn is_expired(&self, ref_date: Date) -> Option<bool> {
        Some(self.maturity_date < ref_date)
    }
    fn maturity_date(&self) -> Option<Date> { Some(self.maturity_date) }
    fn notional(&self) -> Option<f64> { Some(self.face_amount) }
}

impl Instrument for AmortizingBond {
    fn instrument_type(&self) -> &str { "AmortizingBond" }
    fn is_expired(&self, ref_date: Date) -> Option<bool> {
        Some(self.maturity_date < ref_date)
    }
    fn maturity_date(&self) -> Option<Date> { Some(self.maturity_date) }
    fn notional(&self) -> Option<f64> { Some(self.face_amount) }
}

impl Instrument for CallableBond {
    fn instrument_type(&self) -> &str { "CallableBond" }
    fn is_expired(&self, ref_date: Date) -> Option<bool> {
        Some(self.maturity_date < ref_date)
    }
    fn maturity_date(&self) -> Option<Date> { Some(self.maturity_date) }
    fn notional(&self) -> Option<f64> { Some(self.face_amount) }
}

impl Instrument for ConvertibleBond {
    fn instrument_type(&self) -> &str { "ConvertibleBond" }
    fn is_expired(&self, ref_date: Date) -> Option<bool> {
        Some(self.maturity_date < ref_date)
    }
    fn maturity_date(&self) -> Option<Date> { Some(self.maturity_date) }
    fn notional(&self) -> Option<f64> { Some(self.face_amount) }
}

impl Instrument for InflationLinkedBond {
    fn instrument_type(&self) -> &str { "InflationLinkedBond" }
    fn is_expired(&self, ref_date: Date) -> Option<bool> {
        Some(self.maturity < ref_date)
    }
    fn maturity_date(&self) -> Option<Date> { Some(self.maturity) }
    fn notional(&self) -> Option<f64> { Some(self.notional) }
}

// ── Options (date-based) ─────────────────────────────────

impl Instrument for VanillaOption {
    fn instrument_type(&self) -> &str { "VanillaOption" }
    fn is_expired(&self, ref_date: Date) -> Option<bool> {
        Some(self.exercise.last_date() < ref_date)
    }
    fn maturity_date(&self) -> Option<Date> {
        Some(self.exercise.last_date())
    }
}

impl Instrument for BarrierOption {
    fn instrument_type(&self) -> &str { "BarrierOption" }
    fn is_expired(&self, ref_date: Date) -> Option<bool> {
        Some(self.exercise.last_date() < ref_date)
    }
    fn maturity_date(&self) -> Option<Date> {
        Some(self.exercise.last_date())
    }
}

impl Instrument for DoubleBarrierOption {
    fn instrument_type(&self) -> &str { "DoubleBarrierOption" }
    fn is_expired(&self, ref_date: Date) -> Option<bool> {
        Some(self.exercise.last_date() < ref_date)
    }
    fn maturity_date(&self) -> Option<Date> {
        Some(self.exercise.last_date())
    }
}

impl Instrument for AsianOption {
    fn instrument_type(&self) -> &str { "AsianOption" }
    fn is_expired(&self, ref_date: Date) -> Option<bool> {
        Some(self.exercise.last_date() < ref_date)
    }
    fn maturity_date(&self) -> Option<Date> {
        Some(self.exercise.last_date())
    }
}

// ── Options (time-parameterised: no dates) ───────────────

impl Instrument for LookbackOption {
    fn instrument_type(&self) -> &str { "LookbackOption" }
    fn is_expired(&self, _ref_date: Date) -> Option<bool> { None }
    fn maturity_date(&self) -> Option<Date> { None }
}

impl Instrument for CompoundOption {
    fn instrument_type(&self) -> &str { "CompoundOption" }
    fn is_expired(&self, _ref_date: Date) -> Option<bool> { None }
    fn maturity_date(&self) -> Option<Date> { None }
}

impl Instrument for ChooserOption {
    fn instrument_type(&self) -> &str { "ChooserOption" }
    fn is_expired(&self, ref_date: Date) -> Option<bool> {
        // ChooserOption has an Exercise field
        Some(self.exercise.last_date() < ref_date)
    }
    fn maturity_date(&self) -> Option<Date> {
        Some(self.exercise.last_date())
    }
}

impl Instrument for CliquetOption {
    fn instrument_type(&self) -> &str { "CliquetOption" }
    fn is_expired(&self, _ref_date: Date) -> Option<bool> { None }
    fn maturity_date(&self) -> Option<Date> { None }
    fn notional(&self) -> Option<f64> { Some(self.notional) }
}

impl Instrument for VarianceSwap {
    fn instrument_type(&self) -> &str { "VarianceSwap" }
    fn is_expired(&self, _ref_date: Date) -> Option<bool> { None }
    fn maturity_date(&self) -> Option<Date> { None }
    fn notional(&self) -> Option<f64> { Some(self.variance_notional) }
}

// ── Swaps ────────────────────────────────────────────────

impl Instrument for VanillaSwap {
    fn instrument_type(&self) -> &str { "VanillaSwap" }
    fn is_expired(&self, ref_date: Date) -> Option<bool> {
        Some(VanillaSwap::is_expired(self, ref_date))
    }
    fn maturity_date(&self) -> Option<Date> {
        VanillaSwap::maturity_date(self)
    }
    fn notional(&self) -> Option<f64> { Some(self.nominal) }
}

impl Instrument for OISSwap {
    fn instrument_type(&self) -> &str { "OISSwap" }
    fn is_expired(&self, ref_date: Date) -> Option<bool> {
        Some(OISSwap::is_expired(self, ref_date))
    }
    fn maturity_date(&self) -> Option<Date> {
        OISSwap::maturity_date(self)
    }
    fn notional(&self) -> Option<f64> { Some(self.nominal) }
}

impl Instrument for ForwardRateAgreement {
    fn instrument_type(&self) -> &str { "ForwardRateAgreement" }
    fn is_expired(&self, ref_date: Date) -> Option<bool> {
        Some(self.maturity_date < ref_date)
    }
    fn maturity_date(&self) -> Option<Date> { Some(self.maturity_date) }
    fn notional(&self) -> Option<f64> { Some(self.notional) }
}

impl Instrument for BasisSwap {
    fn instrument_type(&self) -> &str { "BasisSwap" }
    fn is_expired(&self, _ref_date: Date) -> Option<bool> {
        // BasisSwap doesn't store maturity directly
        None
    }
    fn maturity_date(&self) -> Option<Date> { None }
    fn notional(&self) -> Option<f64> { Some(self.nominal) }
}

impl Instrument for CrossCurrencySwap {
    fn instrument_type(&self) -> &str { "CrossCurrencySwap" }
    fn is_expired(&self, _ref_date: Date) -> Option<bool> { None }
    fn maturity_date(&self) -> Option<Date> { None }
    fn notional(&self) -> Option<f64> { Some(self.leg1.notional) }
}

impl Instrument for FloatFloatSwap {
    fn instrument_type(&self) -> &str { "FloatFloatSwap" }
    fn is_expired(&self, _ref_date: Date) -> Option<bool> { None }
    fn maturity_date(&self) -> Option<Date> { None }
    fn notional(&self) -> Option<f64> { Some(self.nominal) }
}

// ── Rate derivatives ─────────────────────────────────────

impl Instrument for Swaption {
    fn instrument_type(&self) -> &str { "Swaption" }
    fn is_expired(&self, ref_date: Date) -> Option<bool> {
        Some(self.expiry < ref_date)
    }
    fn maturity_date(&self) -> Option<Date> { Some(self.swap_maturity) }
}

impl Instrument for CapFloor {
    fn instrument_type(&self) -> &str { "CapFloor" }
    fn is_expired(&self, ref_date: Date) -> Option<bool> {
        self.caplets.last().map(|c| c.payment_date < ref_date)
    }
    fn maturity_date(&self) -> Option<Date> {
        self.caplets.last().map(|c| c.accrual_end)
    }
    fn notional(&self) -> Option<f64> {
        self.caplets.first().map(|c| c.notional)
    }
}

// ── Credit ───────────────────────────────────────────────

impl Instrument for CreditDefaultSwap {
    fn instrument_type(&self) -> &str { "CreditDefaultSwap" }
    fn is_expired(&self, ref_date: Date) -> Option<bool> {
        Some(self.maturity < ref_date)
    }
    fn maturity_date(&self) -> Option<Date> { Some(self.maturity) }
    fn notional(&self) -> Option<f64> { Some(self.notional) }
}

// ── Equity / Forwards / Composites ──────────────────────

impl Instrument for Stock {
    fn instrument_type(&self) -> &str { "Stock" }
    fn is_expired(&self, _ref_date: Date) -> Option<bool> {
        // Stocks don't expire.
        Some(false)
    }
    fn maturity_date(&self) -> Option<Date> { None }
}

impl Instrument for BondForward {
    fn instrument_type(&self) -> &str { "BondForward" }
    fn is_expired(&self, ref_date: Date) -> Option<bool> {
        Some(self.settlement_date < ref_date)
    }
    fn maturity_date(&self) -> Option<Date> { Some(self.settlement_date) }
    fn notional(&self) -> Option<f64> { Some(self.face_amount) }
}

impl Instrument for CompositeInstrument {
    fn instrument_type(&self) -> &str { "CompositeInstrument" }
    fn is_expired(&self, _ref_date: Date) -> Option<bool> {
        // Composite doesn't have a single expiry.
        None
    }
    fn maturity_date(&self) -> Option<Date> { None }
}

// ── Utility functions ────────────────────────────────────

/// Count instruments in a portfolio that have expired.
pub fn expired_count(instruments: &[&dyn Instrument], ref_date: Date) -> usize {
    instruments
        .iter()
        .filter(|i| i.is_expired(ref_date) == Some(true))
        .count()
}

/// Collect all instruments matching a given type name.
pub fn filter_by_type<'a>(
    instruments: &[&'a dyn Instrument],
    type_name: &str,
) -> Vec<&'a dyn Instrument> {
    instruments
        .iter()
        .filter(|i| i.instrument_type() == type_name)
        .copied()
        .collect()
}

/// Total notional across all instruments that report one.
pub fn total_notional(instruments: &[&dyn Instrument]) -> f64 {
    instruments
        .iter()
        .filter_map(|i| i.notional())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::payoff::{Exercise, OptionType, Payoff};
    use ql_time::{Date, DayCounter, Month, Schedule};

    fn sample_bond() -> FixedRateBond {
        let schedule = Schedule::from_dates(vec![
            Date::from_ymd(2024, Month::January, 15),
            Date::from_ymd(2024, Month::July, 15),
            Date::from_ymd(2025, Month::January, 15),
        ]);
        FixedRateBond::new(
            100_000.0,
            2,
            &schedule,
            0.05,
            DayCounter::Actual365Fixed,
        )
    }

    fn sample_option() -> VanillaOption {
        VanillaOption {
            payoff: Payoff::PlainVanilla {
                option_type: OptionType::Call,
                strike: 100.0,
            },
            exercise: Exercise::European {
                expiry: Date::from_ymd(2025, Month::June, 15),
            },
        }
    }

    #[test]
    fn trait_instrument_type() {
        let bond = sample_bond();
        let opt = sample_option();
        assert_eq!(Instrument::instrument_type(&bond), "FixedRateBond");
        assert_eq!(Instrument::instrument_type(&opt), "VanillaOption");
    }

    #[test]
    fn trait_is_expired() {
        let bond = sample_bond();
        let before = Date::from_ymd(2024, Month::June, 1);
        let after = Date::from_ymd(2026, Month::January, 1);
        assert_eq!(Instrument::is_expired(&bond, before), Some(false));
        assert_eq!(Instrument::is_expired(&bond, after), Some(true));
    }

    #[test]
    fn trait_maturity_date() {
        let bond = sample_bond();
        assert_eq!(
            Instrument::maturity_date(&bond),
            Some(Date::from_ymd(2025, Month::January, 15))
        );
    }

    #[test]
    fn trait_notional() {
        let bond = sample_bond();
        assert_eq!(Instrument::notional(&bond), Some(100_000.0));
        let opt = sample_option();
        assert_eq!(Instrument::notional(&opt), None);
    }

    #[test]
    fn time_parameterised_returns_none() {
        let lb = LookbackOption {
            option_type: OptionType::Call,
            lookback_type: crate::lookback_option::LookbackType::FloatingStrike,
            strike: 0.0,
            min_so_far: 90.0,
            max_so_far: 110.0,
            time_to_expiry: 1.0,
        };
        assert_eq!(Instrument::is_expired(&lb, Date::from_ymd(2025, Month::January, 1)), None);
        assert_eq!(Instrument::maturity_date(&lb), None);
    }

    #[test]
    fn heterogeneous_portfolio() {
        let bond = sample_bond();
        let opt = sample_option();
        let portfolio: Vec<&dyn Instrument> = vec![&bond, &opt];

        let ref_date = Date::from_ymd(2026, Month::January, 1);
        assert_eq!(expired_count(&portfolio, ref_date), 2);

        let bonds = filter_by_type(&portfolio, "FixedRateBond");
        assert_eq!(bonds.len(), 1);
    }

    #[test]
    fn total_notional_portfolio() {
        let bond = sample_bond();
        let opt = sample_option();
        let portfolio: Vec<&dyn Instrument> = vec![&bond, &opt];
        // bond = 100k, option = None (skipped)
        assert!((total_notional(&portfolio) - 100_000.0).abs() < 1e-10);
    }
}
