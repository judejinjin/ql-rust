//! Choi (2018) moment-matching Asian engine.
//!
//! Prices arithmetic average Asian options using the moment-matching approach
//! of Choi (2018), "Sum of all Black-Scholes-Merton models: An efficient
//! pricing method for spread, basket, and Asian options".
//!
//! This engine matches the first two moments of the arithmetic average
//! to a log-normal distribution, then prices using Black-Scholes.

use ql_math::distributions::cumulative_normal;

/// Result from the Choi Asian engine.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ChoiAsianResult {
    /// Option price.
    pub price: f64,
    /// Effective volatility of the arithmetic average.
    pub effective_vol: f64,
}

/// Price an arithmetic average price Asian option using the Choi (2018)
/// moment-matching approach.
///
/// This is more accurate than Turnbull-Wakeman for OTM options and
/// handles the correlation structure of discrete fixings correctly.
///
/// # Arguments
/// - `spot`, `strike`, `r`, `q`, `sigma` — standard BS inputs
/// - `t` — time to expiry
/// - `n_fixings` — number of equally-spaced fixing dates
/// - `is_call` — true for call, false for put
#[allow(clippy::too_many_arguments)]
pub fn choi_asian(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    n_fixings: usize,
    is_call: bool,
) -> ChoiAsianResult {
    let omega = if is_call { 1.0 } else { -1.0 };
    let n = n_fixings as f64;
    let dt = t / n;
    let b = r - q;
    let df = (-r * t).exp();

    // First moment: E[A] = (1/n) Σ S · e^{b·tᵢ}
    // where tᵢ = i·dt for i = 1..n
    let mut m1 = 0.0;
    for i in 1..=n_fixings {
        let ti = i as f64 * dt;
        m1 += (b * ti).exp();
    }
    m1 *= spot / n;

    // Second moment: E[A²] = (1/n²) Σᵢ Σⱼ S² · e^{(b + σ²)·min(tᵢ,tⱼ)} · e^{b·|tᵢ−tⱼ|}
    // More precisely: E[SᵢSⱼ] = S² · e^{b(tᵢ+tⱼ)} · e^{σ²·min(tᵢ,tⱼ)}
    let mut m2 = 0.0;
    for i in 1..=n_fixings {
        let ti = i as f64 * dt;
        for j in 1..=n_fixings {
            let tj = j as f64 * dt;
            let t_min = ti.min(tj);
            m2 += ((b * (ti + tj)) + sigma * sigma * t_min).exp();
        }
    }
    m2 *= spot * spot / (n * n);

    // Variance of ln(A): match to log-normal
    // V = ln(M₂/M₁²)
    let v = (m2 / (m1 * m1)).ln();
    if v <= 0.0 || v.is_nan() {
        // Degenerate case: deterministic average
        let price = df * (omega * (m1 - strike)).max(0.0);
        return ChoiAsianResult {
            price,
            effective_vol: 0.0,
        };
    }

    let sigma_a = v.sqrt();
    let effective_vol = sigma_a / t.sqrt();

    // Adjusted forward: F_A = M₁
    let d1 = ((m1 / strike).ln() + 0.5 * v) / sigma_a;
    let d2 = d1 - sigma_a;

    let price = df * omega * (m1 * cumulative_normal(omega * d1) - strike * cumulative_normal(omega * d2));

    ChoiAsianResult {
        price: price.max(0.0),
        effective_vol,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_choi_asian_call() {
        let res = choi_asian(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 12, true);
        // ATM Asian call, 20% vol, 1y, monthly fixings
        assert!(res.price > 3.0 && res.price < 8.0, "price={}", res.price);
        assert!(res.effective_vol > 0.0 && res.effective_vol < 0.25);
    }

    #[test]
    fn test_choi_asian_put() {
        let res = choi_asian(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 12, false);
        assert!(res.price > 0.5 && res.price < 6.0, "price={}", res.price);
    }

    #[test]
    fn test_choi_vs_high_fixings() {
        // More fixings → closer to continuous → cheaper
        let r12 = choi_asian(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 12, true);
        let r252 = choi_asian(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, 252, true);
        assert!(r252.price <= r12.price + 0.5, "252={} vs 12={}", r252.price, r12.price);
    }

    #[test]
    fn test_choi_asian_otm_put() {
        let res = choi_asian(100.0, 80.0, 0.05, 0.0, 0.20, 1.0, 12, false);
        // Deep OTM put should be very cheap
        assert!(res.price < 1.0, "price={}", res.price);
    }
}
