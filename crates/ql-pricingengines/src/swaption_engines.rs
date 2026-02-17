//! Swaption pricing engines.
//!
//! Implements Black (log-normal) and Bachelier (normal) swaption pricing.

use ql_instruments::swaption::{Swaption, SwaptionType};
use ql_math::distributions::NormalDistribution;

/// Result from a swaption pricing engine.
#[derive(Debug, Clone)]
pub struct SwaptionResult {
    /// Net present value.
    pub npv: f64,
    /// Vega (∂V/∂σ).
    pub vega: f64,
}

/// Price a European swaption using Black's model (log-normal rates).
///
/// # Parameters
/// - `swaption`: the swaption instrument (contains forward rate, annuity, strike, etc.)
/// - `volatility`: Black (log-normal) volatility of the forward swap rate
/// - `time_to_expiry`: time to swaption expiry in years
///
/// # Formula
/// Payer: A · [F·N(d₁) − K·N(d₂)]
/// Receiver: A · [K·N(−d₂) − F·N(−d₁)]
/// where d₁ = [ln(F/K) + ½σ²T] / (σ√T), d₂ = d₁ − σ√T
pub fn black_swaption(
    swaption: &Swaption,
    volatility: f64,
    time_to_expiry: f64,
) -> SwaptionResult {
    let f = swaption.forward_rate;
    let k = swaption.strike;
    let a = swaption.annuity;
    let sigma = volatility;
    let t = time_to_expiry;

    if t <= 0.0 {
        let intrinsic = a * (swaption.swaption_type.sign() * (f - k)).max(0.0);
        return SwaptionResult {
            npv: intrinsic,
            vega: 0.0,
        };
    }

    let n = NormalDistribution::standard();
    let sqrt_t = t.sqrt();
    let d1 = ((f / k).ln() + 0.5 * sigma * sigma * t) / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;

    let npv = match swaption.swaption_type {
        SwaptionType::Payer => a * (f * n.cdf(d1) - k * n.cdf(d2)),
        SwaptionType::Receiver => a * (k * n.cdf(-d2) - f * n.cdf(-d1)),
    };

    let vega = a * f * n.pdf(d1) * sqrt_t;

    SwaptionResult { npv, vega }
}

/// Price a European swaption using Bachelier's model (normal rates).
///
/// # Parameters
/// - `swaption`: the swaption instrument
/// - `volatility`: normal (Bachelier) volatility of the forward swap rate (in absolute terms)
/// - `time_to_expiry`: time to swaption expiry in years
///
/// # Formula
/// Payer: A · [(F−K)·N(d) + σ√T·n(d)]
/// where d = (F−K) / (σ√T)
pub fn bachelier_swaption(
    swaption: &Swaption,
    volatility: f64,
    time_to_expiry: f64,
) -> SwaptionResult {
    let f = swaption.forward_rate;
    let k = swaption.strike;
    let a = swaption.annuity;
    let sigma = volatility;
    let t = time_to_expiry;

    if t <= 0.0 {
        let intrinsic = a * (swaption.swaption_type.sign() * (f - k)).max(0.0);
        return SwaptionResult {
            npv: intrinsic,
            vega: 0.0,
        };
    }

    let n = NormalDistribution::standard();
    let sqrt_t = t.sqrt();
    let omega = swaption.swaption_type.sign();
    let d = omega * (f - k) / (sigma * sqrt_t);

    let npv = a * (omega * (f - k) * n.cdf(d) + sigma * sqrt_t * n.pdf(d));
    let vega = a * sqrt_t * n.pdf(d);

    SwaptionResult { npv, vega }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_instruments::swaption::{SettlementType, SwaptionType};
    use ql_time::{Date, Month};

    fn sample_swaption(swaption_type: SwaptionType) -> Swaption {
        Swaption::new(
            swaption_type,
            0.03,  // strike = 3%
            Date::from_ymd(2026, Month::January, 15),
            Date::from_ymd(2026, Month::January, 17),
            Date::from_ymd(2031, Month::January, 17),
            5.0,   // 5y swap tenor
            4.5,   // annuity
            0.035, // forward rate = 3.5%
            SettlementType::Physical,
        )
    }

    #[test]
    fn black_payer_swaption_positive() {
        let s = sample_swaption(SwaptionType::Payer);
        let result = black_swaption(&s, 0.20, 1.0);
        assert!(result.npv > 0.0);
        // Forward > strike, so payer is ITM
        // NPV should be at least intrinsic: A * (F - K) = 4.5 * 0.005 = 0.0225
        assert!(result.npv > 0.022);
    }

    #[test]
    fn black_receiver_swaption_positive() {
        let s = sample_swaption(SwaptionType::Receiver);
        let result = black_swaption(&s, 0.20, 1.0);
        assert!(result.npv > 0.0);
        // Forward > strike, so receiver is OTM but still has time value
    }

    #[test]
    fn black_put_call_parity() {
        // Payer - Receiver = A * (F - K)
        let payer = sample_swaption(SwaptionType::Payer);
        let receiver = sample_swaption(SwaptionType::Receiver);

        let p = black_swaption(&payer, 0.20, 1.0);
        let r = black_swaption(&receiver, 0.20, 1.0);

        let expected = payer.annuity * (payer.forward_rate - payer.strike);
        assert_abs_diff_eq!(p.npv - r.npv, expected, epsilon = 1e-10);
    }

    #[test]
    fn bachelier_payer_swaption_positive() {
        let s = sample_swaption(SwaptionType::Payer);
        let result = bachelier_swaption(&s, 0.005, 1.0); // 50bp normal vol
        assert!(result.npv > 0.0);
    }

    #[test]
    fn bachelier_put_call_parity() {
        let payer = sample_swaption(SwaptionType::Payer);
        let receiver = sample_swaption(SwaptionType::Receiver);

        let p = bachelier_swaption(&payer, 0.005, 1.0);
        let r = bachelier_swaption(&receiver, 0.005, 1.0);

        let expected = payer.annuity * (payer.forward_rate - payer.strike);
        assert_abs_diff_eq!(p.npv - r.npv, expected, epsilon = 1e-10);
    }

    #[test]
    fn black_vega_positive() {
        let s = sample_swaption(SwaptionType::Payer);
        let result = black_swaption(&s, 0.20, 1.0);
        assert!(result.vega > 0.0);
    }

    #[test]
    fn bachelier_vega_positive() {
        let s = sample_swaption(SwaptionType::Payer);
        let result = bachelier_swaption(&s, 0.005, 1.0);
        assert!(result.vega > 0.0);
    }

    #[test]
    fn expired_swaption_intrinsic() {
        let s = sample_swaption(SwaptionType::Payer);
        let result = black_swaption(&s, 0.20, 0.0);
        let expected = s.annuity * (s.forward_rate - s.strike);
        assert_abs_diff_eq!(result.npv, expected, epsilon = 1e-10);
    }
}
