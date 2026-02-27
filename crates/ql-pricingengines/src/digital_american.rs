//! Analytic Digital American option engine.
//!
//! Prices digital (binary) American options using the Rubinstein & Reiner (1991)
//! one-touch / no-touch barrier replication approach with an intensity adjustment.
//!
//! Reference: Haug (2007), "The Complete Guide to Option Pricing Formulas",
//! Chapter 9.

use ql_math::distributions::cumulative_normal;

/// Result from the digital American engine.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DigitalAmericanResult {
    /// Digital option price (as fraction of cash rebate).
    pub price: f64,
}

/// Type of digital American option.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum DigitalAmericanType {
    /// Cash-or-nothing call: pays `cash` if S > K at any time before T.
    CashOrNothingCall,
    /// Cash-or-nothing put: pays `cash` if S < K at any time before T.
    CashOrNothingPut,
    /// Asset-or-nothing call: pays S if S > K at any time before T.
    AssetOrNothingCall,
    /// Asset-or-nothing put: pays S if S < K at any time before T.
    AssetOrNothingPut,
}

/// Price a digital (binary) American option.
///
/// A digital American option pays a fixed cash amount (or asset value) the first
/// time the underlying hits the barrier/strike. This is equivalent to a one-touch
/// barrier digital.
///
/// # Arguments
/// - `spot` — current asset price
/// - `strike` — digital strike / barrier
/// - `r` — risk-free rate
/// - `q` — continuous dividend yield
/// - `sigma` — volatility
/// - `t` — time to expiry (years)
/// - `cash` — cash rebate amount (for cash-or-nothing types)
/// - `digital_type` — type of digital option
#[allow(clippy::too_many_arguments)]
pub fn digital_american(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    cash: f64,
    digital_type: DigitalAmericanType,
) -> DigitalAmericanResult {
    if t <= 0.0 || sigma <= 0.0 {
        let payoff = match digital_type {
            DigitalAmericanType::CashOrNothingCall => if spot > strike { cash } else { 0.0 },
            DigitalAmericanType::CashOrNothingPut => if spot < strike { cash } else { 0.0 },
            DigitalAmericanType::AssetOrNothingCall => if spot > strike { spot } else { 0.0 },
            DigitalAmericanType::AssetOrNothingPut => if spot < strike { spot } else { 0.0 },
        };
        return DigitalAmericanResult { price: payoff };
    }

    let b = r - q;
    let sqrt_t = t.sqrt();

    // For American digitals, we use the one-touch formula.
    // If spot is already past the barrier, immediate exercise:
    match digital_type {
        DigitalAmericanType::CashOrNothingCall | DigitalAmericanType::AssetOrNothingCall => {
            if spot >= strike {
                let p = match digital_type {
                    DigitalAmericanType::CashOrNothingCall => cash,
                    DigitalAmericanType::AssetOrNothingCall => spot,
                    _ => unreachable!(),
                };
                return DigitalAmericanResult { price: p };
            }
        }
        DigitalAmericanType::CashOrNothingPut | DigitalAmericanType::AssetOrNothingPut => {
            if spot <= strike {
                let p = match digital_type {
                    DigitalAmericanType::CashOrNothingPut => cash,
                    DigitalAmericanType::AssetOrNothingPut => spot,
                    _ => unreachable!(),
                };
                return DigitalAmericanResult { price: p };
            }
        }
    }

    // One-touch barrier formula (Reiner & Rubinstein 1991)
    // For an up-and-in cash digital (spot < strike):
    //   P = cash · [ (H/S)^(a+b) N(η·z) + (H/S)^(a-b) N(η·z - 2bσ√T) ]
    // where H = strike, η = +1 for up, -1 for down
    //   a = (b - σ²/2) / σ²,  b_coeff = sqrt(a² + 2r/σ²)
    //   z = ln(H/S)/(σ√T) + b·σ√T
    //
    // Simplified: use intensity approach λ = (√(μ² + 2rσ²) - μ) / σ²
    //   where μ = b - σ²/2

    let mu = b - 0.5 * sigma * sigma;
    let lambda = ((mu * mu + 2.0 * r * sigma * sigma).sqrt()) ;
    let a = mu / (sigma * sigma);
    let b_coeff = lambda / (sigma * sigma);

    let (is_up, eta) = match digital_type {
        DigitalAmericanType::CashOrNothingCall | DigitalAmericanType::AssetOrNothingCall => (true, -1.0_f64),
        DigitalAmericanType::CashOrNothingPut | DigitalAmericanType::AssetOrNothingPut => (false, 1.0_f64),
    };

    let h = strike;
    let x = (h / spot).ln() / (sigma * sqrt_t);

    let z1 = x + b_coeff * sigma * sqrt_t;
    let z2 = x - b_coeff * sigma * sqrt_t;

    let pow_plus = (h / spot).powf(a + b_coeff);
    let pow_minus = (h / spot).powf(a - b_coeff);

    let cash_price = cash * (pow_plus * cumulative_normal(eta * z1)
        + pow_minus * cumulative_normal(eta * z2));

    let price = match digital_type {
        DigitalAmericanType::CashOrNothingCall | DigitalAmericanType::CashOrNothingPut => {
            cash_price
        }
        DigitalAmericanType::AssetOrNothingCall => {
            // Asset-or-nothing: need to adjust. The payout is S_τ when S first hits H.
            // Since at the hitting time S_τ = H, the value is H · P(one-touch) / cash · rebate=1
            let one_touch = cash_price / cash;
            strike * one_touch // pays H (= strike) at hitting time
        }
        DigitalAmericanType::AssetOrNothingPut => {
            let one_touch = cash_price / cash;
            strike * one_touch
        }
    };

    DigitalAmericanResult {
        price: price.max(0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_cash_or_nothing_call() {
        let res = digital_american(
            100.0, 110.0, 0.05, 0.0, 0.20, 1.0, 1.0,
            DigitalAmericanType::CashOrNothingCall,
        );
        // One-touch probability should be between 0 and 1
        assert!(res.price > 0.0 && res.price < 1.0, "price={}", res.price);
    }

    #[test]
    fn test_cash_or_nothing_put() {
        let res = digital_american(
            100.0, 90.0, 0.05, 0.0, 0.20, 1.0, 1.0,
            DigitalAmericanType::CashOrNothingPut,
        );
        assert!(res.price > 0.0 && res.price < 1.0, "price={}", res.price);
    }

    #[test]
    fn test_already_hit_call() {
        let res = digital_american(
            110.0, 100.0, 0.05, 0.0, 0.20, 1.0, 15.0,
            DigitalAmericanType::CashOrNothingCall,
        );
        assert_abs_diff_eq!(res.price, 15.0, epsilon = 1e-10);
    }

    #[test]
    fn test_already_hit_put() {
        let res = digital_american(
            90.0, 100.0, 0.05, 0.0, 0.20, 1.0, 15.0,
            DigitalAmericanType::CashOrNothingPut,
        );
        assert_abs_diff_eq!(res.price, 15.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expired() {
        let res = digital_american(
            100.0, 110.0, 0.05, 0.0, 0.20, 0.0, 1.0,
            DigitalAmericanType::CashOrNothingCall,
        );
        assert_abs_diff_eq!(res.price, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_asset_or_nothing_call() {
        let res = digital_american(
            100.0, 110.0, 0.05, 0.0, 0.20, 1.0, 1.0,
            DigitalAmericanType::AssetOrNothingCall,
        );
        // Asset-or-nothing pays strike=110 at hitting time, discounted
        assert!(res.price > 0.0 && res.price < 110.0, "price={}", res.price);
    }
}
