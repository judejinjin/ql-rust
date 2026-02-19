//! Interest rate representation (rate + day counter + compounding).

use ql_core::errors::{QLError, QLResult};
use ql_time::DayCounter;
use serde::{Deserialize, Serialize};

/// Compounding convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Compounding {
    /// No compounding: `1 + r·t`.
    Simple,
    /// Continuous: `e^{r·t}`.
    Continuous,
    /// Compounded: `(1 + r/n)^{n·t}`.
    Compounded,
    /// Simple then compounded (used in some money-market conventions).
    SimpleThenCompounded,
}

/// An interest rate with its associated conventions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterestRate {
    /// The rate value (e.g. 0.05 for 5%).
    pub rate: f64,
    /// Day counter for computing year fractions.
    pub day_counter: DayCounter,
    /// Compounding convention.
    pub compounding: Compounding,
    /// Compounding frequency (events per year). Only used for `Compounded`.
    pub frequency: u32,
}

impl InterestRate {
    /// Create a new interest rate.
    pub fn new(
        rate: f64,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: u32,
    ) -> Self {
        Self {
            rate,
            day_counter,
            compounding,
            frequency,
        }
    }

    /// Compute the discount factor for a given time `t` (in years).
    pub fn discount_factor(&self, t: f64) -> QLResult<f64> {
        self.compound_factor(t).map(|cf| 1.0 / cf)
    }

    /// Compute the compound factor for a given time `t` (in years).
    pub fn compound_factor(&self, t: f64) -> QLResult<f64> {
        if t < 0.0 {
            return Err(QLError::InvalidArgument(
                "negative time not allowed".into(),
            ));
        }
        if t.abs() < 1e-15 {
            return Ok(1.0);
        }
        match self.compounding {
            Compounding::Simple => Ok(1.0 + self.rate * t),
            Compounding::Continuous => Ok((self.rate * t).exp()),
            Compounding::Compounded => {
                let n = self.frequency as f64;
                if n < 1.0 {
                    return Err(QLError::InvalidArgument(
                        "compounding frequency must be >= 1".into(),
                    ));
                }
                Ok((1.0 + self.rate / n).powf(n * t))
            }
            Compounding::SimpleThenCompounded => {
                if t <= 1.0 / self.frequency as f64 {
                    Ok(1.0 + self.rate * t)
                } else {
                    let n = self.frequency as f64;
                    Ok((1.0 + self.rate / n).powf(n * t))
                }
            }
        }
    }

    /// Implied rate from a compound factor over time `t`.
    pub fn implied_rate(
        compound: f64,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: u32,
        t: f64,
    ) -> QLResult<Self> {
        if compound <= 0.0 {
            return Err(QLError::InvalidArgument(
                "compound factor must be positive".into(),
            ));
        }
        if t.abs() < 1e-15 {
            return Ok(Self::new(0.0, day_counter, compounding, frequency));
        }
        let rate = match compounding {
            Compounding::Simple => (compound - 1.0) / t,
            Compounding::Continuous => compound.ln() / t,
            Compounding::Compounded => {
                let n = frequency as f64;
                n * (compound.powf(1.0 / (n * t)) - 1.0)
            }
            Compounding::SimpleThenCompounded => {
                if t <= 1.0 / frequency as f64 {
                    (compound - 1.0) / t
                } else {
                    let n = frequency as f64;
                    n * (compound.powf(1.0 / (n * t)) - 1.0)
                }
            }
        };
        Ok(Self::new(rate, day_counter, compounding, frequency))
    }

    /// Convert this rate to an equivalent rate under different conventions.
    pub fn equivalent_rate(
        &self,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: u32,
        t: f64,
    ) -> QLResult<InterestRate> {
        let compound = self.compound_factor(t)?;
        InterestRate::implied_rate(compound, day_counter, compounding, frequency, t)
    }
}

impl std::fmt::Display for InterestRate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.4} {:?} {:?}", self.rate, self.compounding, self.day_counter)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn simple_compounding() {
        let r = InterestRate::new(0.05, DayCounter::Actual365Fixed, Compounding::Simple, 1);
        let cf = r.compound_factor(1.0).unwrap();
        assert_abs_diff_eq!(cf, 1.05, epsilon = 1e-15);
    }

    #[test]
    fn continuous_compounding() {
        let r = InterestRate::new(0.05, DayCounter::Actual365Fixed, Compounding::Continuous, 1);
        let cf = r.compound_factor(1.0).unwrap();
        assert_abs_diff_eq!(cf, (0.05_f64).exp(), epsilon = 1e-14);
    }

    #[test]
    fn annual_compounding() {
        let r = InterestRate::new(0.05, DayCounter::Actual365Fixed, Compounding::Compounded, 1);
        let cf = r.compound_factor(2.0).unwrap();
        assert_abs_diff_eq!(cf, 1.05_f64.powi(2), epsilon = 1e-14);
    }

    #[test]
    fn semiannual_compounding() {
        let r = InterestRate::new(0.05, DayCounter::Actual365Fixed, Compounding::Compounded, 2);
        let cf = r.compound_factor(1.0).unwrap();
        assert_abs_diff_eq!(cf, (1.025_f64).powi(2), epsilon = 1e-14);
    }

    #[test]
    fn discount_factor() {
        let r = InterestRate::new(0.05, DayCounter::Actual365Fixed, Compounding::Continuous, 1);
        let df = r.discount_factor(1.0).unwrap();
        assert_abs_diff_eq!(df, (-0.05_f64).exp(), epsilon = 1e-14);
    }

    #[test]
    fn zero_time_gives_one() {
        let r = InterestRate::new(0.10, DayCounter::Actual365Fixed, Compounding::Continuous, 1);
        assert_abs_diff_eq!(r.compound_factor(0.0).unwrap(), 1.0, epsilon = 1e-15);
    }

    #[test]
    fn implied_rate_roundtrip() {
        let r = InterestRate::new(0.05, DayCounter::Actual365Fixed, Compounding::Continuous, 1);
        let cf = r.compound_factor(2.0).unwrap();
        let implied = InterestRate::implied_rate(
            cf,
            DayCounter::Actual365Fixed,
            Compounding::Continuous,
            1,
            2.0,
        )
        .unwrap();
        assert_abs_diff_eq!(implied.rate, 0.05, epsilon = 1e-14);
    }

    #[test]
    fn equivalent_rate_continuous_to_annual() {
        let r = InterestRate::new(0.05, DayCounter::Actual365Fixed, Compounding::Continuous, 1);
        let eq = r
            .equivalent_rate(DayCounter::Actual365Fixed, Compounding::Compounded, 1, 1.0)
            .unwrap();
        // e^0.05 = (1 + r_annual), so r_annual = e^0.05 - 1
        assert_abs_diff_eq!(eq.rate, (0.05_f64).exp() - 1.0, epsilon = 1e-14);
    }

    #[test]
    fn negative_time_rejected() {
        let r = InterestRate::new(0.05, DayCounter::Actual365Fixed, Compounding::Simple, 1);
        assert!(r.compound_factor(-1.0).is_err());
    }
}
