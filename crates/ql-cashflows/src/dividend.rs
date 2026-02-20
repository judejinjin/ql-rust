//! Discrete dividend representations and schedule utilities.
//!
//! Provides [`Dividend`] (cash or proportional) and [`DividendSchedule`]
//! for use in equity option pricing models that handle discrete dividends.

use serde::{Deserialize, Serialize};

/// A single discrete dividend payment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Dividend {
    /// Fixed cash dividend at time `t` (years from valuation).
    Cash {
        /// Time of ex-dividend date (years from valuation date).
        t: f64,
        /// Absolute dividend amount.
        amount: f64,
    },
    /// Proportional (percentage) dividend at time `t`.
    Proportional {
        /// Time of ex-dividend date (years from valuation date).
        t: f64,
        /// Dividend yield as a fraction (e.g., 0.02 for 2%).
        rate: f64,
    },
}

impl Dividend {
    /// Create a new fixed cash dividend.
    pub fn cash(t: f64, amount: f64) -> Self {
        Dividend::Cash { t, amount }
    }

    /// Create a new proportional dividend.
    pub fn proportional(t: f64, rate: f64) -> Self {
        Dividend::Proportional { t, rate }
    }

    /// Time of the dividend payment (years).
    pub fn time(&self) -> f64 {
        match self {
            Dividend::Cash { t, .. } | Dividend::Proportional { t, .. } => *t,
        }
    }

    /// True if this is a cash dividend.
    pub fn is_cash(&self) -> bool {
        matches!(self, Dividend::Cash { .. })
    }
}

/// A schedule of discrete dividends, sorted by time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DividendSchedule {
    dividends: Vec<Dividend>,
}

impl DividendSchedule {
    /// Create from a vector of dividends (will be sorted by time).
    pub fn new(mut dividends: Vec<Dividend>) -> Self {
        dividends.sort_by(|a, b| a.time().partial_cmp(&b.time()).unwrap());
        DividendSchedule { dividends }
    }

    /// Empty schedule (no discrete dividends).
    pub fn empty() -> Self {
        DividendSchedule {
            dividends: Vec::new(),
        }
    }

    /// Iterate over dividends.
    pub fn iter(&self) -> std::slice::Iter<'_, Dividend> {
        self.dividends.iter()
    }

    /// Number of dividends.
    pub fn len(&self) -> usize {
        self.dividends.len()
    }

    /// True if no dividends.
    pub fn is_empty(&self) -> bool {
        self.dividends.is_empty()
    }

    /// Present value of all cash dividends before time `t`,
    /// discounted at continuous rate `r`.
    pub fn pv_cash_dividends(&self, r: f64, t: f64) -> f64 {
        self.dividends
            .iter()
            .filter_map(|d| match d {
                Dividend::Cash { t: td, amount } if *td <= t && *td > 0.0 => {
                    Some(amount * (-r * td).exp())
                }
                _ => None,
            })
            .sum()
    }

    /// Cumulative proportional factor for dividends before time `t`:
    /// ∏(1 - rate_i) for all proportional dividends at t_i ≤ t.
    pub fn proportional_factor(&self, t: f64) -> f64 {
        self.dividends
            .iter()
            .filter_map(|d| match d {
                Dividend::Proportional { t: td, rate } if *td <= t && *td > 0.0 => {
                    Some(1.0 - rate)
                }
                _ => None,
            })
            .product()
    }

    /// Adjusted spot for the escrowed dividend model:
    /// S* = S · proportional_factor(T) - PV(cash dividends before T)
    pub fn escrowed_spot(&self, spot: f64, r: f64, t: f64) -> f64 {
        spot * self.proportional_factor(t) - self.pv_cash_dividends(r, t)
    }

    /// Dividends between time `t0` (exclusive) and `t1` (inclusive).
    pub fn dividends_between(&self, t0: f64, t1: f64) -> Vec<&Dividend> {
        self.dividends
            .iter()
            .filter(|d| d.time() > t0 && d.time() <= t1)
            .collect()
    }
}

impl IntoIterator for DividendSchedule {
    type Item = Dividend;
    type IntoIter = std::vec::IntoIter<Dividend>;
    fn into_iter(self) -> Self::IntoIter {
        self.dividends.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn cash_dividend_creation() {
        let d = Dividend::cash(0.25, 2.0);
        assert_abs_diff_eq!(d.time(), 0.25);
        assert!(d.is_cash());
    }

    #[test]
    fn proportional_dividend_creation() {
        let d = Dividend::proportional(0.5, 0.02);
        assert_abs_diff_eq!(d.time(), 0.5);
        assert!(!d.is_cash());
    }

    #[test]
    fn schedule_sorts_by_time() {
        let sched = DividendSchedule::new(vec![
            Dividend::cash(0.75, 1.5),
            Dividend::cash(0.25, 2.0),
            Dividend::cash(0.50, 1.0),
        ]);
        let times: Vec<f64> = sched.iter().map(|d| d.time()).collect();
        assert_eq!(times, vec![0.25, 0.50, 0.75]);
    }

    #[test]
    fn pv_cash_dividends() {
        let r = 0.05;
        let sched = DividendSchedule::new(vec![
            Dividend::cash(0.25, 2.0),
            Dividend::cash(0.50, 3.0),
        ]);
        let pv = sched.pv_cash_dividends(r, 1.0);
        let expected = 2.0 * (-r * 0.25_f64).exp() + 3.0 * (-r * 0.5_f64).exp();
        assert_abs_diff_eq!(pv, expected, epsilon = 1e-10);
    }

    #[test]
    fn proportional_factor() {
        let sched = DividendSchedule::new(vec![
            Dividend::proportional(0.25, 0.02),
            Dividend::proportional(0.75, 0.03),
        ]);
        let factor = sched.proportional_factor(1.0);
        assert_abs_diff_eq!(factor, 0.98 * 0.97, epsilon = 1e-10);
    }

    #[test]
    fn escrowed_spot() {
        let spot = 100.0;
        let r = 0.05;
        let sched = DividendSchedule::new(vec![
            Dividend::cash(0.25, 2.0),
            Dividend::cash(0.75, 2.0),
        ]);
        let s_star = sched.escrowed_spot(spot, r, 1.0);
        let expected = spot - 2.0 * (-r * 0.25_f64).exp() - 2.0 * (-r * 0.75_f64).exp();
        assert_abs_diff_eq!(s_star, expected, epsilon = 1e-10);
    }

    #[test]
    fn dividends_between() {
        let sched = DividendSchedule::new(vec![
            Dividend::cash(0.1, 1.0),
            Dividend::cash(0.3, 2.0),
            Dividend::cash(0.5, 3.0),
            Dividend::cash(0.7, 4.0),
        ]);
        let between = sched.dividends_between(0.2, 0.6);
        assert_eq!(between.len(), 2);
    }

    #[test]
    fn empty_schedule() {
        let sched = DividendSchedule::empty();
        assert!(sched.is_empty());
        assert_abs_diff_eq!(sched.pv_cash_dividends(0.05, 1.0), 0.0);
        assert_abs_diff_eq!(sched.proportional_factor(1.0), 1.0);
        assert_abs_diff_eq!(sched.escrowed_spot(100.0, 0.05, 1.0), 100.0);
    }
}
