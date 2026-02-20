//! Pricing engines for inflation caps and floors.
//!
//! Implements Black (log-normal) and Bachelier (normal) pricing for
//! YoY inflation caplets/floorlets and zero-coupon inflation caps/floors.

use ql_instruments::inflation_cap_floor::{
    YoYInflationCapFloor, ZeroCouponInflationCapFloor,
};
use ql_math::distributions::NormalDistribution;

/// Result of an inflation cap/floor pricing.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InflationCapFloorResult {
    /// Net present value.
    pub npv: f64,
    /// Vega (sensitivity to 1bp vol change).
    pub vega: f64,
}

/// Price a YoY inflation cap/floor using the Black (log-normal) model.
///
/// Each caplet is priced as:
/// $$\text{NPV}_i = D(t_i) \, \tau_i \, N \, [\omega F_i \Phi(\omega d_1) - \omega K \Phi(\omega d_2)]$$
///
/// where $F_i$ is the forward YoY rate, $K$ is the strike, $\sigma$ is
/// the (log-normal) volatility of the YoY rate, and $d_{1,2}$ are the
/// usual Black-Scholes variables.
///
/// # Arguments
/// * `cap_floor` — the YoY inflation cap/floor instrument
/// * `volatility` — flat log-normal volatility of YoY inflation rate
pub fn black_yoy_inflation_cap_floor(
    cap_floor: &YoYInflationCapFloor,
    volatility: f64,
) -> InflationCapFloorResult {
    let omega = cap_floor.cap_floor_type.sign();
    let strike = cap_floor.strike;
    let n = NormalDistribution::standard();
    let mut total_npv = 0.0;
    let mut total_vega = 0.0;

    for caplet in &cap_floor.caplets {
        let t = caplet.fixing_time;
        if t <= 0.0 {
            // Already fixed
            let payoff = (omega * (caplet.forward_rate - strike)).max(0.0);
            total_npv += caplet.discount * caplet.accrual_fraction * caplet.notional * payoff;
            continue;
        }

        let f = caplet.forward_rate;
        let vol = volatility;

        if f <= 0.0 || strike <= 0.0 {
            // Log-normal model not valid for non-positive rates
            // Use intrinsic value
            let payoff = (omega * (f - strike)).max(0.0);
            total_npv += caplet.discount * caplet.accrual_fraction * caplet.notional * payoff;
            continue;
        }

        let sqrt_t = t.sqrt();
        let d1 = ((f / strike).ln() + 0.5 * vol * vol * t) / (vol * sqrt_t);
        let d2 = d1 - vol * sqrt_t;

        let caplet_npv = caplet.discount
            * caplet.accrual_fraction
            * caplet.notional
            * (omega * f * n.cdf(omega * d1) - omega * strike * n.cdf(omega * d2));

        let caplet_vega = caplet.discount
            * caplet.accrual_fraction
            * caplet.notional
            * f
            * n.pdf(d1)
            * sqrt_t
            * 0.0001; // per 1bp

        total_npv += caplet_npv;
        total_vega += caplet_vega;
    }

    InflationCapFloorResult {
        npv: total_npv,
        vega: total_vega,
    }
}

/// Price a YoY inflation cap/floor using the Bachelier (normal) model.
///
/// Each caplet is priced as:
/// $$\text{NPV}_i = D(t_i) \, \tau_i \, N \, [\omega(F_i - K)\Phi(\omega d) + \sigma\sqrt{T}\phi(d)]$$
///
/// where $d = (F_i - K) / (\sigma\sqrt{T})$.
///
/// This model handles negative forward rates naturally.
pub fn bachelier_yoy_inflation_cap_floor(
    cap_floor: &YoYInflationCapFloor,
    volatility: f64,
) -> InflationCapFloorResult {
    let omega = cap_floor.cap_floor_type.sign();
    let strike = cap_floor.strike;
    let n = NormalDistribution::standard();
    let mut total_npv = 0.0;
    let mut total_vega = 0.0;

    for caplet in &cap_floor.caplets {
        let t = caplet.fixing_time;
        if t <= 0.0 {
            let payoff = (omega * (caplet.forward_rate - strike)).max(0.0);
            total_npv += caplet.discount * caplet.accrual_fraction * caplet.notional * payoff;
            continue;
        }

        let f = caplet.forward_rate;
        let vol = volatility;
        let sqrt_t = t.sqrt();
        let vol_sqrt_t = vol * sqrt_t;

        if vol_sqrt_t < 1e-15 {
            let payoff = (omega * (f - strike)).max(0.0);
            total_npv += caplet.discount * caplet.accrual_fraction * caplet.notional * payoff;
            continue;
        }

        let d = (f - strike) / vol_sqrt_t;

        let caplet_npv = caplet.discount
            * caplet.accrual_fraction
            * caplet.notional
            * (omega * (f - strike) * n.cdf(omega * d) + vol_sqrt_t * n.pdf(d));

        let caplet_vega = caplet.discount
            * caplet.accrual_fraction
            * caplet.notional
            * sqrt_t
            * n.pdf(d)
            * 0.0001; // per 1bp

        total_npv += caplet_npv;
        total_vega += caplet_vega;
    }

    InflationCapFloorResult {
        npv: total_npv,
        vega: total_vega,
    }
}

/// Price a zero-coupon inflation cap/floor using the Black model.
///
/// Models the annualized zero-coupon inflation rate as log-normal:
///
/// $$\text{NPV} = D(T) \, N \, T \, [\omega F \Phi(\omega d_1) - \omega K \Phi(\omega d_2)]$$
///
/// where $F$ is the forward annualized ZC inflation rate.
pub fn black_zc_inflation_cap_floor(
    instrument: &ZeroCouponInflationCapFloor,
    volatility: f64,
) -> InflationCapFloorResult {
    let omega = instrument.cap_floor_type.sign();
    let f = instrument.forward_rate;
    let k = instrument.strike;
    let t = instrument.maturity;
    let n_dist = NormalDistribution::standard();

    if t <= 0.0 || f <= 0.0 || k <= 0.0 {
        let payoff = (omega * (f - k)).max(0.0);
        return InflationCapFloorResult {
            npv: instrument.discount * instrument.notional * t * payoff,
            vega: 0.0,
        };
    }

    let sqrt_t = t.sqrt();
    let d1 = ((f / k).ln() + 0.5 * volatility * volatility * t) / (volatility * sqrt_t);
    let d2 = d1 - volatility * sqrt_t;

    let npv = instrument.discount
        * instrument.notional
        * t
        * (omega * f * n_dist.cdf(omega * d1) - omega * k * n_dist.cdf(omega * d2));

    let vega = instrument.discount
        * instrument.notional
        * t
        * f
        * n_dist.pdf(d1)
        * sqrt_t
        * 0.0001;

    InflationCapFloorResult { npv, vega }
}

/// Price a zero-coupon inflation cap/floor using the Bachelier model.
pub fn bachelier_zc_inflation_cap_floor(
    instrument: &ZeroCouponInflationCapFloor,
    volatility: f64,
) -> InflationCapFloorResult {
    let omega = instrument.cap_floor_type.sign();
    let f = instrument.forward_rate;
    let k = instrument.strike;
    let t = instrument.maturity;
    let n_dist = NormalDistribution::standard();

    let sqrt_t = t.sqrt();
    let vol_sqrt_t = volatility * sqrt_t;

    if vol_sqrt_t < 1e-15 || t <= 0.0 {
        let payoff = (omega * (f - k)).max(0.0);
        return InflationCapFloorResult {
            npv: instrument.discount * instrument.notional * t * payoff,
            vega: 0.0,
        };
    }

    let d = (f - k) / vol_sqrt_t;

    let npv = instrument.discount
        * instrument.notional
        * t
        * (omega * (f - k) * n_dist.cdf(omega * d) + vol_sqrt_t * n_dist.pdf(d));

    let vega = instrument.discount
        * instrument.notional
        * t
        * sqrt_t
        * n_dist.pdf(d)
        * 0.0001;

    InflationCapFloorResult { npv, vega }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_instruments::inflation_cap_floor::{
        InflationCapFloorType, build_yoy_cap_floor, ZeroCouponInflationCapFloor,
    };

    #[test]
    fn yoy_cap_positive() {
        let cap = build_yoy_cap_floor(
            InflationCapFloorType::Cap,
            0.02,           // 2% strike
            1_000_000.0,
            5,
            &[0.025, 0.026, 0.027, 0.028, 0.029],
            &[0.95, 0.90, 0.86, 0.82, 0.78],
        );
        let result = black_yoy_inflation_cap_floor(&cap, 0.005); // 0.5% vol
        assert!(result.npv > 0.0, "YoY cap NPV should be positive: {}", result.npv);
    }

    #[test]
    fn yoy_floor_positive() {
        let floor = build_yoy_cap_floor(
            InflationCapFloorType::Floor,
            0.03,           // 3% strike (above forwards)
            1_000_000.0,
            5,
            &[0.025, 0.026, 0.027, 0.028, 0.029],
            &[0.95, 0.90, 0.86, 0.82, 0.78],
        );
        let result = black_yoy_inflation_cap_floor(&floor, 0.005);
        assert!(result.npv > 0.0, "YoY floor NPV should be positive: {}", result.npv);
    }

    #[test]
    fn yoy_cap_floor_parity() {
        // Cap - Floor = sum of forward minus strike annuities
        let fwds = [0.025, 0.026, 0.027, 0.028, 0.029];
        let dfs = [0.95, 0.90, 0.86, 0.82, 0.78];
        let strike = 0.025;

        let cap = build_yoy_cap_floor(
            InflationCapFloorType::Cap, strike, 1_000_000.0, 5, &fwds, &dfs,
        );
        let floor = build_yoy_cap_floor(
            InflationCapFloorType::Floor, strike, 1_000_000.0, 5, &fwds, &dfs,
        );

        let cap_price = black_yoy_inflation_cap_floor(&cap, 0.005).npv;
        let floor_price = black_yoy_inflation_cap_floor(&floor, 0.005).npv;

        // Parity: Cap - Floor = sum(DF * τ * N * (F - K))
        let swap_value: f64 = fwds
            .iter()
            .zip(dfs.iter())
            .map(|(&f, &df)| df * 1.0 * 1_000_000.0 * (f - strike))
            .sum();

        assert_abs_diff_eq!(cap_price - floor_price, swap_value, epsilon = 1.0);
    }

    #[test]
    fn bachelier_yoy_cap_positive() {
        let cap = build_yoy_cap_floor(
            InflationCapFloorType::Cap,
            0.02,
            1_000_000.0,
            5,
            &[0.025, 0.026, 0.027, 0.028, 0.029],
            &[0.95, 0.90, 0.86, 0.82, 0.78],
        );
        let result = bachelier_yoy_inflation_cap_floor(&cap, 0.005);
        assert!(result.npv > 0.0, "Bachelier YoY cap should be positive: {}", result.npv);
    }

    #[test]
    fn zc_inflation_cap_positive() {
        let cap = ZeroCouponInflationCapFloor {
            cap_floor_type: InflationCapFloorType::Cap,
            strike: 0.025,
            maturity: 5.0,
            notional: 1_000_000.0,
            forward_rate: 0.03,
            discount: 0.78,
        };
        let result = black_zc_inflation_cap_floor(&cap, 0.005);
        assert!(result.npv > 0.0, "ZC inflation cap should be positive: {}", result.npv);
    }

    #[test]
    fn zc_inflation_cap_floor_parity() {
        let strike = 0.028;
        let cap = ZeroCouponInflationCapFloor {
            cap_floor_type: InflationCapFloorType::Cap,
            strike,
            maturity: 5.0,
            notional: 1_000_000.0,
            forward_rate: 0.03,
            discount: 0.78,
        };
        let floor = ZeroCouponInflationCapFloor {
            cap_floor_type: InflationCapFloorType::Floor,
            strike,
            maturity: 5.0,
            notional: 1_000_000.0,
            forward_rate: 0.03,
            discount: 0.78,
        };

        let c = black_zc_inflation_cap_floor(&cap, 0.005).npv;
        let f = black_zc_inflation_cap_floor(&floor, 0.005).npv;
        let swap = 0.78 * 1_000_000.0 * 5.0 * (0.03 - strike);
        assert_abs_diff_eq!(c - f, swap, epsilon = 1.0);
    }

    #[test]
    fn bachelier_zc_cap_positive() {
        let cap = ZeroCouponInflationCapFloor {
            cap_floor_type: InflationCapFloorType::Cap,
            strike: 0.025,
            maturity: 5.0,
            notional: 1_000_000.0,
            forward_rate: 0.03,
            discount: 0.78,
        };
        let result = bachelier_zc_inflation_cap_floor(&cap, 0.005);
        assert!(result.npv > 0.0);
    }

    #[test]
    fn zero_vol_gives_intrinsic() {
        let cap = ZeroCouponInflationCapFloor {
            cap_floor_type: InflationCapFloorType::Cap,
            strike: 0.02,
            maturity: 5.0,
            notional: 1_000_000.0,
            forward_rate: 0.03,
            discount: 0.80,
        };
        let result = bachelier_zc_inflation_cap_floor(&cap, 0.0);
        let intrinsic = 0.80 * 1_000_000.0 * 5.0 * (0.03 - 0.02);
        assert_abs_diff_eq!(result.npv, intrinsic, epsilon = 0.01);
    }
}
