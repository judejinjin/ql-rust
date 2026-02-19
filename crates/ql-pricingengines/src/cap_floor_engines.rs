//! Cap/Floor pricing engines.
//!
//! Implements Black (log-normal) and Bachelier (normal) caplet/floorlet pricing,
//! summed across the strip.

use ql_instruments::cap_floor::CapFloor;
use ql_math::distributions::NormalDistribution;

/// Result from a cap/floor pricing engine.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CapFloorResult {
    /// Net present value.
    pub npv: f64,
    /// Total vega (sum of caplet vegas).
    pub vega: f64,
}

/// Price a cap or floor using Black's model (log-normal forward rates).
///
/// Each caplet/floorlet is priced as:
///   df · τ · N · [ω·F·N(ω·d₁) − ω·K·N(ω·d₂)]
/// where ω = +1 for cap, -1 for floor.
pub fn black_cap_floor(
    cap_floor: &CapFloor,
    volatility: f64,
    time_to_first_fixing: f64,
) -> CapFloorResult {
    let n = NormalDistribution::standard();
    let omega = cap_floor.cap_floor_type.sign();
    let k = cap_floor.strike;

    let mut total_npv = 0.0;
    let mut total_vega = 0.0;

    for (i, caplet) in cap_floor.caplets.iter().enumerate() {
        // Time to fixing for this caplet
        let t = time_to_first_fixing + i as f64 * caplet.accrual_fraction;
        if t <= 0.0 {
            // Already fixed: intrinsic value
            let intrinsic = caplet.notional
                * caplet.accrual_fraction
                * caplet.discount
                * (omega * (caplet.forward_rate - k)).max(0.0);
            total_npv += intrinsic;
            continue;
        }

        let f = caplet.forward_rate;
        let sigma = volatility;
        let sqrt_t = t.sqrt();

        if f <= 0.0 || k <= 0.0 {
            continue;
        }

        let d1 = ((f / k).ln() + 0.5 * sigma * sigma * t) / (sigma * sqrt_t);
        let d2 = d1 - sigma * sqrt_t;

        let caplet_npv = caplet.discount
            * caplet.accrual_fraction
            * caplet.notional
            * (omega * f * n.cdf(omega * d1) - omega * k * n.cdf(omega * d2));

        let caplet_vega = caplet.discount
            * caplet.accrual_fraction
            * caplet.notional
            * f
            * n.pdf(d1)
            * sqrt_t;

        total_npv += caplet_npv;
        total_vega += caplet_vega;
    }

    CapFloorResult {
        npv: total_npv,
        vega: total_vega,
    }
}

/// Price a cap or floor using Bachelier's model (normal forward rates).
///
/// Each caplet/floorlet is priced as:
///   df · τ · N · [ω·(F−K)·N(ω·d) + σ√T·n(d)]
/// where d = ω·(F−K) / (σ√T).
pub fn bachelier_cap_floor(
    cap_floor: &CapFloor,
    volatility: f64,
    time_to_first_fixing: f64,
) -> CapFloorResult {
    let n = NormalDistribution::standard();
    let omega = cap_floor.cap_floor_type.sign();
    let k = cap_floor.strike;

    let mut total_npv = 0.0;
    let mut total_vega = 0.0;

    for (i, caplet) in cap_floor.caplets.iter().enumerate() {
        let t = time_to_first_fixing + i as f64 * caplet.accrual_fraction;
        if t <= 0.0 {
            let intrinsic = caplet.notional
                * caplet.accrual_fraction
                * caplet.discount
                * (omega * (caplet.forward_rate - k)).max(0.0);
            total_npv += intrinsic;
            continue;
        }

        let f = caplet.forward_rate;
        let sigma = volatility;
        let sqrt_t = t.sqrt();
        let d = omega * (f - k) / (sigma * sqrt_t);

        let caplet_npv = caplet.discount
            * caplet.accrual_fraction
            * caplet.notional
            * (omega * (f - k) * n.cdf(d) + sigma * sqrt_t * n.pdf(d));

        let caplet_vega = caplet.discount
            * caplet.accrual_fraction
            * caplet.notional
            * sqrt_t
            * n.pdf(d);

        total_npv += caplet_npv;
        total_vega += caplet_vega;
    }

    CapFloorResult {
        npv: total_npv,
        vega: total_vega,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_instruments::cap_floor::{Caplet, CapFloorType};
    use ql_time::{Date, Month};

    fn sample_caplets(forward_rate: f64) -> Vec<Caplet> {
        (0..4)
            .map(|i| Caplet {
                accrual_start: Date::from_ymd(2026 + i, Month::January, 15),
                accrual_end: Date::from_ymd(2026 + i, Month::July, 15),
                payment_date: Date::from_ymd(2026 + i, Month::July, 17),
                accrual_fraction: 0.5,
                notional: 1_000_000.0,
                forward_rate,
                discount: 0.97_f64.powi(i as i32 + 1),
            })
            .collect()
    }

    #[test]
    fn black_cap_positive_when_forward_exceeds_strike() {
        let cap = CapFloor::new(CapFloorType::Cap, 0.03, sample_caplets(0.04));
        let result = black_cap_floor(&cap, 0.20, 1.0);
        assert!(result.npv > 0.0);
    }

    #[test]
    fn black_floor_positive_when_strike_exceeds_forward() {
        let floor = CapFloor::new(CapFloorType::Floor, 0.05, sample_caplets(0.04));
        let result = black_cap_floor(&floor, 0.20, 1.0);
        assert!(result.npv > 0.0);
    }

    #[test]
    fn black_cap_floor_parity() {
        // Cap - Floor = sum of (forward - strike) * df * tau * notional
        let caplets = sample_caplets(0.04);
        let cap = CapFloor::new(CapFloorType::Cap, 0.03, caplets.clone());
        let floor = CapFloor::new(CapFloorType::Floor, 0.03, caplets.clone());

        let cap_result = black_cap_floor(&cap, 0.20, 1.0);
        let floor_result = black_cap_floor(&floor, 0.20, 1.0);

        let forward_value: f64 = caplets
            .iter()
            .map(|c| c.discount * c.accrual_fraction * c.notional * (c.forward_rate - 0.03))
            .sum();

        assert_abs_diff_eq!(
            cap_result.npv - floor_result.npv,
            forward_value,
            epsilon = 1e-6
        );
    }

    #[test]
    fn bachelier_cap_positive() {
        let cap = CapFloor::new(CapFloorType::Cap, 0.03, sample_caplets(0.04));
        let result = bachelier_cap_floor(&cap, 0.005, 1.0);
        assert!(result.npv > 0.0);
    }

    #[test]
    fn bachelier_cap_floor_parity() {
        let caplets = sample_caplets(0.035);
        let cap = CapFloor::new(CapFloorType::Cap, 0.03, caplets.clone());
        let floor = CapFloor::new(CapFloorType::Floor, 0.03, caplets.clone());

        let cap_result = bachelier_cap_floor(&cap, 0.005, 1.0);
        let floor_result = bachelier_cap_floor(&floor, 0.005, 1.0);

        let forward_value: f64 = caplets
            .iter()
            .map(|c| c.discount * c.accrual_fraction * c.notional * (c.forward_rate - 0.03))
            .sum();

        assert_abs_diff_eq!(
            cap_result.npv - floor_result.npv,
            forward_value,
            epsilon = 1e-6
        );
    }

    #[test]
    fn black_cap_vega_positive() {
        let cap = CapFloor::new(CapFloorType::Cap, 0.03, sample_caplets(0.04));
        let result = black_cap_floor(&cap, 0.20, 1.0);
        assert!(result.vega > 0.0);
    }

    #[test]
    fn cap_more_expensive_with_higher_vol() {
        let cap = CapFloor::new(CapFloorType::Cap, 0.03, sample_caplets(0.04));
        let low_vol = black_cap_floor(&cap, 0.10, 1.0);
        let high_vol = black_cap_floor(&cap, 0.30, 1.0);
        assert!(high_vol.npv > low_vol.npv);
    }
}
