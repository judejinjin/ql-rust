//! Forward Rate Agreement (FRA).
//!
//! A FRA is a cash-settled OTC contract where one party pays a fixed rate
//! and receives a floating (IBOR) rate on a notional for a single future
//! period. Settlement is at the start of the rate period.
//!
//! ## Payoff
//!
//! ```text
//! payer:    N * (L - K) * τ / (1 + L * τ)
//! receiver: N * (K - L) * τ / (1 + L * τ)
//! ```
//!
//! where `L` is the IBOR fixing, `K` is the FRA rate, `τ` is the accrual
//! period, and discounting is done at the fixing rate (standard FRA convention).

use ql_indexes::IborIndex;
use ql_time::{Date, DayCounter};

use crate::vanilla_swap::SwapType;

/// A Forward Rate Agreement.
#[derive(Debug, Clone)]
pub struct ForwardRateAgreement {
    /// Payer (pay fixed, receive floating) or Receiver.
    pub fra_type: SwapType,
    /// Notional principal.
    pub notional: f64,
    /// The agreed fixed rate.
    pub strike_rate: f64,
    /// Start of the rate period (= settlement date for most FRAs).
    pub value_date: Date,
    /// End of the rate period.
    pub maturity_date: Date,
    /// The IBOR index the FRA references.
    pub index: IborIndex,
    /// Day counter for accrual fraction.
    pub day_counter: DayCounter,
}

impl ForwardRateAgreement {
    /// Create a new FRA.
    pub fn new(
        fra_type: SwapType,
        notional: f64,
        strike_rate: f64,
        value_date: Date,
        maturity_date: Date,
        index: IborIndex,
        day_counter: DayCounter,
    ) -> Self {
        Self {
            fra_type,
            notional,
            strike_rate,
            value_date,
            maturity_date,
            index,
            day_counter,
        }
    }

    /// Year fraction for the FRA period.
    pub fn accrual_period(&self) -> f64 {
        self.day_counter
            .year_fraction(self.value_date, self.maturity_date)
    }

    /// Compute the FRA's implied forward rate from a yield curve.
    pub fn implied_rate(
        &self,
        curve: &dyn ql_termstructures::YieldTermStructure,
    ) -> f64 {
        let df_start = curve.discount(self.value_date);
        let df_end = curve.discount(self.maturity_date);
        self.index
            .forecast_fixing(self.value_date, self.maturity_date, df_start, df_end)
    }

    /// NPV of the FRA given a yield curve.
    ///
    /// FRA NPV at `t=0` for a **payer** (pay fixed K, receive floating L):
    /// ```text
    /// NPV = N * (L - K) * τ / (1 + L * τ) * df(value_date)
    /// ```
    /// where `df(value_date)` discounts from the settlement date back to today.
    pub fn npv(&self, curve: &dyn ql_termstructures::YieldTermStructure) -> f64 {
        let forward = self.implied_rate(curve);
        let tau = self.accrual_period();
        // Cash-settled FRA payoff at value_date
        let payoff_at_settle = self.notional * (forward - self.strike_rate) * tau
            / (1.0 + forward * tau);
        let sign = match self.fra_type {
            SwapType::Payer => 1.0,
            SwapType::Receiver => -1.0,
        };
        sign * payoff_at_settle * curve.discount(self.value_date)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_currencies::currency::Currency;
    use ql_termstructures::FlatForward;
    use ql_time::{BusinessDayConvention, Calendar, Month, Period, TimeUnit};

    fn make_fra(rate: f64) -> ForwardRateAgreement {
        let index = IborIndex::new(
            "USD-LIBOR-3M",
            Period::new(3, TimeUnit::Months),
            2,
            Currency::usd(),
            Calendar::Target,
            BusinessDayConvention::ModifiedFollowing,
            false,
            DayCounter::Actual360,
        );
        ForwardRateAgreement::new(
            SwapType::Payer,
            1_000_000.0,
            rate,
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2025, Month::October, 15),
            index,
            DayCounter::Actual360,
        )
    }

    #[test]
    fn fra_at_forward_rate_has_near_zero_npv() {
        let ref_date = Date::from_ymd(2025, Month::January, 2);
        let curve = FlatForward::new(ref_date, 0.04, DayCounter::Actual360);
        // For a flat curve, implied forward is roughly the curve rate
        let fra = make_fra(0.04);
        let implied = fra.implied_rate(&curve);
        // Strike at implied forward → NPV ≈ 0
        let fra_at_fwd = ForwardRateAgreement::new(
            SwapType::Payer,
            1_000_000.0,
            implied,
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2025, Month::October, 15),
            fra.index.clone(),
            DayCounter::Actual360,
        );
        let npv = fra_at_fwd.npv(&curve);
        assert_abs_diff_eq!(npv, 0.0, epsilon = 0.01);
    }

    #[test]
    fn fra_payer_positive_when_forward_above_strike() {
        let ref_date = Date::from_ymd(2025, Month::January, 2);
        let curve = FlatForward::new(ref_date, 0.05, DayCounter::Actual360);
        let fra = make_fra(0.03); // Strike below forward
        assert!(fra.npv(&curve) > 0.0, "Payer FRA should have positive NPV");
    }

    #[test]
    fn fra_receiver_positive_when_forward_below_strike() {
        let ref_date = Date::from_ymd(2025, Month::January, 2);
        let curve = FlatForward::new(ref_date, 0.03, DayCounter::Actual360);
        let fra = ForwardRateAgreement::new(
            SwapType::Receiver,
            1_000_000.0,
            0.05,
            Date::from_ymd(2025, Month::July, 15),
            Date::from_ymd(2025, Month::October, 15),
            IborIndex::new(
                "USD-LIBOR-3M",
                Period::new(3, TimeUnit::Months),
                2,
                Currency::usd(),
                Calendar::Target,
                BusinessDayConvention::ModifiedFollowing,
                false,
                DayCounter::Actual360,
            ),
            DayCounter::Actual360,
        );
        assert!(fra.npv(&curve) > 0.0);
    }
}
