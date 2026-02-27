//! Perpetual futures instrument.
//!
//! Perpetual futures (common in crypto exchanges) have no expiry date.
//! They use a funding rate mechanism to anchor the futures price to the
//! spot price.

use serde::{Deserialize, Serialize};

/// A perpetual futures contract.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerpetualFuture {
    /// Current mark price.
    pub mark_price: f64,
    /// Index price (spot reference).
    pub index_price: f64,
    /// Notional (contract size * quantity).
    pub notional: f64,
    /// Current funding rate (per period, e.g., per 8 hours).
    pub funding_rate: f64,
    /// Annualized interest rate ("premium" component).
    pub interest_rate: f64,
    /// Funding interval in hours (typically 8).
    pub funding_interval_hours: f64,
    /// True if long, false if short.
    pub is_long: bool,
    /// Underlying asset name/symbol.
    pub symbol: String,
}

/// Result of perpetual futures pricing/analysis.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerpetualFutureResult {
    /// Fair funding rate.
    pub fair_funding_rate: f64,
    /// Annualized funding cost/income.
    pub annualized_funding: f64,
    /// Basis (mark - index).
    pub basis: f64,
    /// Basis as percentage.
    pub basis_pct: f64,
    /// PnL from a given price move.
    pub unrealized_pnl: f64,
}

impl PerpetualFuture {
    /// Create a new perpetual future.
    pub fn new(
        mark_price: f64,
        index_price: f64,
        notional: f64,
        funding_rate: f64,
        interest_rate: f64,
        is_long: bool,
        symbol: impl Into<String>,
    ) -> Self {
        Self {
            mark_price,
            index_price,
            notional,
            funding_rate,
            interest_rate,
            funding_interval_hours: 8.0,
            is_long,
            symbol: symbol.into(),
        }
    }

    /// Fair value premium/discount.
    pub fn basis(&self) -> f64 {
        self.mark_price - self.index_price
    }

    /// Basis as a percentage of index price.
    pub fn basis_pct(&self) -> f64 {
        self.basis() / self.index_price * 100.0
    }

    /// Compute the fair funding rate.
    ///
    /// Fair funding = premium / index + interest rate adjustment.
    /// funding_rate = max(-0.05%, min(0.05%, premium/index)) + interest_rate
    pub fn fair_funding_rate(&self) -> f64 {
        let premium_rate = self.basis() / self.index_price;
        let clamp_rate = premium_rate.clamp(-0.0005, 0.0005);
        clamp_rate + self.interest_rate / (365.0 * 24.0 / self.funding_interval_hours)
    }

    /// Funding payment for one period.
    ///
    /// Positive means the position pays funding; negative means it receives.
    pub fn funding_payment(&self) -> f64 {
        let direction = if self.is_long { 1.0 } else { -1.0 };
        direction * self.funding_rate * self.notional
    }

    /// Annualized funding cost.
    pub fn annualized_funding_cost(&self) -> f64 {
        let periods_per_year = 365.0 * 24.0 / self.funding_interval_hours;
        self.funding_payment() * periods_per_year
    }

    /// PnL from a price change.
    pub fn pnl(&self, new_mark_price: f64) -> f64 {
        let direction = if self.is_long { 1.0 } else { -1.0 };
        let quantity = self.notional / self.mark_price;
        direction * quantity * (new_mark_price - self.mark_price)
    }
}

/// Analyze a perpetual future position.
pub fn analyze_perpetual(perp: &PerpetualFuture) -> PerpetualFutureResult {
    PerpetualFutureResult {
        fair_funding_rate: perp.fair_funding_rate(),
        annualized_funding: perp.annualized_funding_cost(),
        basis: perp.basis(),
        basis_pct: perp.basis_pct(),
        unrealized_pnl: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_perpetual_basis() {
        let perp = PerpetualFuture::new(
            50_100.0, 50_000.0, 10_000.0, 0.0001, 0.0, true, "BTC",
        );
        assert_abs_diff_eq!(perp.basis(), 100.0, epsilon = 1e-10);
        assert_abs_diff_eq!(perp.basis_pct(), 0.2, epsilon = 1e-10);
    }

    #[test]
    fn test_perpetual_funding() {
        let perp = PerpetualFuture::new(
            50_000.0, 50_000.0, 10_000.0, 0.0001, 0.0, true, "BTC",
        );
        // Long pays 0.01% * 10000 = 1.0 per period
        assert_abs_diff_eq!(perp.funding_payment(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_perpetual_pnl() {
        let perp = PerpetualFuture::new(
            50_000.0, 50_000.0, 50_000.0, 0.0001, 0.0, true, "BTC",
        );
        // Long 1 BTC, price goes up by 1000
        let pnl = perp.pnl(51_000.0);
        assert_abs_diff_eq!(pnl, 1_000.0, epsilon = 1.0);
    }
}
