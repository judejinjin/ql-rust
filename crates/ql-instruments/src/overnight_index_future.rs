//! Overnight index futures instrument.
//!
//! Overnight index futures (e.g., Fed Funds futures, SOFR futures) are
//! cash-settled futures contracts referencing the compounded overnight rate
//! over a specific accrual period.

use serde::{Deserialize, Serialize};

/// An overnight index future (e.g., Fed Funds futures, SOFR futures).
///
/// The contract settles at 100 − R, where R is the arithmetic or compounded
/// average of the overnight rate during the reference period.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OvernightIndexFuture {
    /// Notional amount.
    pub notional: f64,
    /// Futures price (quoted as 100 - implied rate in %).
    pub futures_price: f64,
    /// Start date of the reference period (as f64 year fraction from today).
    pub start_time: f64,
    /// End date of the reference period.
    pub end_time: f64,
    /// Name of the overnight index (e.g., "SOFR", "FedFunds").
    pub index_name: String,
    /// Whether compounding is used (true) or simple average (false).
    pub compounding: bool,
}

impl OvernightIndexFuture {
    /// Create a new overnight index future.
    pub fn new(
        notional: f64,
        futures_price: f64,
        start_time: f64,
        end_time: f64,
        index_name: impl Into<String>,
        compounding: bool,
    ) -> Self {
        Self {
            notional,
            futures_price,
            start_time,
            end_time,
            index_name: index_name.into(),
            compounding,
        }
    }

    /// Implied average overnight rate from the futures price.
    pub fn implied_rate(&self) -> f64 {
        (100.0 - self.futures_price) / 100.0
    }

    /// Day count fraction for the reference period.
    pub fn day_count_fraction(&self) -> f64 {
        self.end_time - self.start_time
    }

    /// Mark-to-market PV given a current futures price.
    pub fn mtm(&self, current_price: f64) -> f64 {
        // Each basis point is worth notional * dcf / 100
        let dcf = self.day_count_fraction();
        self.notional * dcf * (current_price - self.futures_price) / 100.0
    }

    /// Convexity adjustment (simple approximation).
    ///
    /// Futures rate ≈ forward rate + convexity adjustment.
    /// Using σ² * T * dcf approximation.
    pub fn convexity_adjustment(&self, sigma: f64) -> f64 {
        let t = self.start_time;
        let dcf = self.day_count_fraction();
        0.5 * sigma * sigma * t * dcf
    }

    /// Adjusted implied rate (forward rate − convexity adjustment).
    pub fn adjusted_forward_rate(&self, sigma: f64) -> f64 {
        self.implied_rate() - self.convexity_adjustment(sigma)
    }
}

/// Result of pricing an overnight index future.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OvernightIndexFutureResult {
    pub fair_price: f64,
    pub implied_rate: f64,
    pub convexity_adjustment: f64,
    pub adjusted_rate: f64,
    pub pv: f64,
}

/// Price an overnight index future given an overnight rate curve.
///
/// # Arguments
/// - `future` — the future contract
/// - `overnight_rates` — vector of (time, rate) pairs for daily overnight rates
/// - `sigma` — rate volatility (for convexity adjustment)
pub fn price_overnight_index_future(
    future: &OvernightIndexFuture,
    overnight_rates: &[(f64, f64)],
    sigma: f64,
) -> OvernightIndexFutureResult {
    // Compute average/compounded rate over the reference period
    let relevant_rates: Vec<_> = overnight_rates
        .iter()
        .filter(|(t, _)| *t >= future.start_time && *t < future.end_time)
        .collect();

    let dcf = future.day_count_fraction();

    let implied_rate = if future.compounding && !relevant_rates.is_empty() {
        // Compounded: ∏(1 + r_i * δ_i) − 1 / dcf
        let n = relevant_rates.len();
        let dt_daily = dcf / n as f64;
        let mut product = 1.0;
        for &(_, r) in &relevant_rates {
            product *= 1.0 + r * dt_daily;
        }
        (product - 1.0) / dcf
    } else if !relevant_rates.is_empty() {
        // Simple average
        let sum: f64 = relevant_rates.iter().map(|(_, r)| r).sum();
        sum / relevant_rates.len() as f64
    } else {
        future.implied_rate()
    };

    let conv_adj = future.convexity_adjustment(sigma);
    let adjusted_rate = implied_rate - conv_adj;
    let fair_price = 100.0 - implied_rate * 100.0;
    let pv = future.mtm(fair_price);

    OvernightIndexFutureResult {
        fair_price,
        implied_rate,
        convexity_adjustment: conv_adj,
        adjusted_rate,
        pv,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_overnight_future_implied_rate() {
        let fut = OvernightIndexFuture::new(
            1_000_000.0, 95.0, 0.25, 0.50, "SOFR", true,
        );
        assert_abs_diff_eq!(fut.implied_rate(), 0.05, epsilon = 1e-10);
        assert_abs_diff_eq!(fut.day_count_fraction(), 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_overnight_future_mtm() {
        let fut = OvernightIndexFuture::new(
            1_000_000.0, 95.0, 0.25, 0.50, "FedFunds", false,
        );
        // If price goes to 95.10, gain = 1M * 0.25 * 0.10/100 = 250
        let mtm = fut.mtm(95.10);
        assert_abs_diff_eq!(mtm, 250.0, epsilon = 1.0);
    }

    #[test]
    fn test_convexity_adjustment() {
        let fut = OvernightIndexFuture::new(
            1_000_000.0, 95.0, 1.0, 1.25, "SOFR", true,
        );
        let ca = fut.convexity_adjustment(0.01);
        // 0.5 * 0.01^2 * 1.0 * 0.25 = 0.0000125
        assert_abs_diff_eq!(ca, 0.0000125, epsilon = 1e-8);
    }
}
