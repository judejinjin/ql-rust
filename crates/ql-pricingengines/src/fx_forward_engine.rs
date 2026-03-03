//! FX forward pricing engine.
//!
//! Prices an FX forward by discounting the base and quote notionals
//! with their respective currency yield curves and converting to a
//! single-currency NPV using the spot FX rate.
//!
//! ## Covered Interest Rate Parity
//!
//! The theoretical forward rate satisfies:
//!   F = S × D_base(T) / D_quote(T)
//!
//! where S is the spot rate, D_base(T) is the base-currency discount factor
//! to maturity T, and D_quote(T) is the quote-currency discount factor.
//!
//! ## QuantLib Parity
//!
//! Corresponds to `DiscountingFxForwardEngine` in QuantLib C++.

use ql_instruments::fx_forward::{FxForward, FxForwardType};
use ql_termstructures::YieldTermStructure;
use ql_time::Date;

/// Results from the FX forward pricing engine.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FxForwardResult {
    /// NPV in base (domestic) currency.
    pub npv: f64,
    /// Fair forward rate (quote per base) implied by the curves.
    pub fair_forward_rate: f64,
    /// Forward points (fair_forward_rate - spot_fx).
    pub forward_points: f64,
    /// Base leg PV (in base currency).
    pub base_leg_pv: f64,
    /// Quote leg PV (in base currency, converted at spot).
    pub quote_leg_pv: f64,
    /// Delta (sensitivity to spot FX rate change).
    pub delta: f64,
}

/// Price an FX forward.
///
/// # Arguments
/// * `fwd` — the FX forward instrument
/// * `base_curve` — yield curve for the base currency
/// * `quote_curve` — yield curve for the quote currency
/// * `spot_fx` — current spot FX rate: units of quote ccy per 1 unit of base ccy
/// * `settle` — settlement / valuation date
pub fn price_fx_forward(
    fwd: &FxForward,
    base_curve: &dyn YieldTermStructure,
    quote_curve: &dyn YieldTermStructure,
    spot_fx: f64,
    settle: Date,
) -> FxForwardResult {
    let dc_base = base_curve.day_counter();
    let dc_quote = quote_curve.day_counter();
    let t_base = dc_base.year_fraction(settle, fwd.maturity_date);
    let t_quote = dc_quote.year_fraction(settle, fwd.maturity_date);

    if t_base <= 0.0 {
        // Expired forward: intrinsic value
        let intrinsic = match fwd.forward_type {
            FxForwardType::Buy => fwd.base_notional * spot_fx - fwd.quote_notional,
            FxForwardType::Sell => fwd.quote_notional - fwd.base_notional * spot_fx,
        };
        return FxForwardResult {
            npv: intrinsic,
            fair_forward_rate: spot_fx,
            forward_points: 0.0,
            base_leg_pv: fwd.base_notional,
            quote_leg_pv: fwd.quote_notional / spot_fx,
            delta: 0.0,
        };
    }

    let df_base = base_curve.discount_t(t_base);
    let df_quote = quote_curve.discount_t(t_quote);

    // Fair forward rate via covered interest rate parity
    // F = S × D_base(T) / D_quote(T)
    let fair_fwd = spot_fx * df_base / df_quote;

    // NPV of base leg: base_notional × df_base (in base currency)
    let base_leg_pv = fwd.base_notional * df_base;
    // NPV of quote leg: quote_notional × df_quote (in quote currency) → converted at spot
    let quote_leg_pv_in_base = fwd.quote_notional * df_quote / spot_fx;

    let npv = match fwd.forward_type {
        // Buy base: receive base_notional at T, pay quote_notional at T
        // NPV = base_notional × df_base - quote_notional × df_quote / spot
        FxForwardType::Buy => base_leg_pv - quote_leg_pv_in_base,
        // Sell base: pay base_notional at T, receive quote_notional at T
        FxForwardType::Sell => quote_leg_pv_in_base - base_leg_pv,
    };

    // Delta: dNPV/dSpot
    // For a buyer: d/dSpot [base_notional × df_base - quote_notional × df_quote / spot]
    //            = quote_notional × df_quote / spot^2
    let delta = match fwd.forward_type {
        FxForwardType::Buy => fwd.quote_notional * df_quote / (spot_fx * spot_fx),
        FxForwardType::Sell => -fwd.quote_notional * df_quote / (spot_fx * spot_fx),
    };

    FxForwardResult {
        npv,
        fair_forward_rate: fair_fwd,
        forward_points: fair_fwd - spot_fx,
        base_leg_pv,
        quote_leg_pv: quote_leg_pv_in_base,
        delta,
    }
}

/// Compute the theoretical forward FX rate from two yield curves.
///
/// F = S × D_base(T) / D_quote(T)
pub fn fx_forward_rate(
    spot_fx: f64,
    base_curve: &dyn YieldTermStructure,
    quote_curve: &dyn YieldTermStructure,
    settle: Date,
    maturity: Date,
) -> f64 {
    let t_base = base_curve.day_counter().year_fraction(settle, maturity);
    let t_quote = quote_curve.day_counter().year_fraction(settle, maturity);
    let df_base = base_curve.discount_t(t_base.max(0.0));
    let df_quote = quote_curve.discount_t(t_quote.max(0.0));
    spot_fx * df_base / df_quote
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_currencies::currency::Currency;
    use ql_instruments::fx_forward::FxForward;
    use ql_termstructures::FlatForward;
    use ql_time::{DayCounter, Month};

    #[test]
    fn fx_forward_pricing_at_fair_rate() {
        let settle = Date::from_ymd(2025, Month::January, 15);
        let maturity = Date::from_ymd(2026, Month::January, 15);
        let spot = 1.10; // 1.10 USD per EUR (quote per base)

        // base=EUR (3%), quote=USD (5%)
        let eur_curve = FlatForward::new(settle, 0.03, DayCounter::Actual360);
        let usd_curve = FlatForward::new(settle, 0.05, DayCounter::Actual360);

        // Compute fair forward rate: F = S × D_EUR / D_USD
        // With USD rate > EUR rate, D_USD < D_EUR → F > S (EUR at premium)
        let fair = fx_forward_rate(spot, &eur_curve, &usd_curve, settle, maturity);
        assert!(fair > spot, "Forward should exceed spot when quote rate > base rate");

        // Create forward AT the fair rate → NPV should be ~0
        let fwd = FxForward::from_rate(
            FxForwardType::Buy,
            Currency::eur(),
            Currency::usd(),
            1_000_000.0,
            fair,
            maturity,
        );
        let result = price_fx_forward(&fwd, &eur_curve, &usd_curve, spot, settle);
        assert_abs_diff_eq!(result.npv, 0.0, epsilon = 1.0);
    }

    #[test]
    fn fx_forward_buyer_benefits_from_spot_increase() {
        let settle = Date::from_ymd(2025, Month::January, 15);
        let maturity = Date::from_ymd(2025, Month::July, 15);

        // base=EUR (3%), quote=USD (4%)
        let eur_curve = FlatForward::new(settle, 0.03, DayCounter::Actual360);
        let usd_curve = FlatForward::new(settle, 0.04, DayCounter::Actual360);

        let fwd = FxForward::from_rate(
            FxForwardType::Buy,
            Currency::eur(),
            Currency::usd(),
            1_000_000.0,
            1.10,
            maturity,
        );

        let _r1 = price_fx_forward(&fwd, &eur_curve, &usd_curve, 1.05, settle);
        let _r2 = price_fx_forward(&fwd, &eur_curve, &usd_curve, 1.15, settle);

        // Buyer benefits when the quoted rate goes up (base currency appreciates)
        assert!(_r1.delta > 0.0, "Delta should be positive for buyer");
    }

    #[test]
    fn fx_forward_seller_symmetry() {
        let settle = Date::from_ymd(2025, Month::January, 15);
        let maturity = Date::from_ymd(2025, Month::July, 15);

        let base_curve = FlatForward::new(settle, 0.04, DayCounter::Actual360);
        let quote_curve = FlatForward::new(settle, 0.03, DayCounter::Actual360);

        let buy = FxForward::from_rate(
            FxForwardType::Buy,
            Currency::usd(),
            Currency::eur(),
            1_000_000.0,
            0.90,
            maturity,
        );
        let sell = FxForward::from_rate(
            FxForwardType::Sell,
            Currency::usd(),
            Currency::eur(),
            1_000_000.0,
            0.90,
            maturity,
        );

        let r_buy = price_fx_forward(&buy, &base_curve, &quote_curve, 0.90, settle);
        let r_sell = price_fx_forward(&sell, &base_curve, &quote_curve, 0.90, settle);

        // Buy + Sell should net to zero
        assert_abs_diff_eq!(r_buy.npv + r_sell.npv, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn fx_forward_expired() {
        let settle = Date::from_ymd(2025, Month::July, 15);
        let maturity = Date::from_ymd(2025, Month::January, 15); // already passed

        let base_curve = FlatForward::new(settle, 0.04, DayCounter::Actual360);
        let quote_curve = FlatForward::new(settle, 0.03, DayCounter::Actual360);

        let fwd = FxForward::from_rate(
            FxForwardType::Buy,
            Currency::usd(),
            Currency::eur(),
            1_000_000.0,
            0.90,
            maturity,
        );

        let result = price_fx_forward(&fwd, &base_curve, &quote_curve, 0.95, settle);
        assert_eq!(result.delta, 0.0, "Expired forward has zero delta");
    }

    #[test]
    fn fx_forward_covered_interest_parity() {
        let settle = Date::from_ymd(2025, Month::January, 15);
        let maturity = Date::from_ymd(2026, Month::January, 15);
        let spot = 1.10;

        // base rate 2%, quote rate 5% → F = S × D_base/D_quote > S
        let base_curve = FlatForward::new(settle, 0.02, DayCounter::Actual365Fixed);
        let quote_curve = FlatForward::new(settle, 0.05, DayCounter::Actual365Fixed);

        let fair = fx_forward_rate(spot, &base_curve, &quote_curve, settle, maturity);

        // Forward points should reflect interest rate differential
        let fwd_points = fair - spot;
        // With quote rate > base rate, forward > spot (base currency at premium)
        assert!(fwd_points > 0.0, "Forward points should be positive when quote rate > base rate");
    }
}
