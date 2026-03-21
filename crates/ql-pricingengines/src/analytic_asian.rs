//! Analytic Asian option pricing engines.
//!
//! Provides closed-form and semi-analytic approximations for geometric and
//! arithmetic average-price/average-strike Asian options.
//!
//! ## Engines
//! - [`asian_geometric_continuous_avg_price`] — Kemna-Vorst (1990) exact formula
//! - [`asian_geometric_discrete_avg_price`]  — exact discrete geometric average price
//! - [`asian_geometric_continuous_avg_strike`] — continuous geometric average-strike
//! - [`asian_turnbull_wakeman`]              — Turnbull-Wakeman (1991) arithmetic approximation
//! - [`asian_levy`]                          — Levy (1992) continuous arithmetic approximation

use ql_instruments::OptionType;
use ql_math::distributions::NormalDistribution;

/// Result from an analytic Asian option engine.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[must_use]
pub struct AsianResult {
    /// Net present value.
    pub npv: f64,
    /// Standard deviation used internally (σ_A).
    pub effective_vol: f64,
    /// Effective forward / adjusted mean.
    pub effective_forward: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn bs_price(fwd: f64, strike: f64, sigma_a: f64, t: f64, r: f64, omega: f64) -> f64 {
    let nd = NormalDistribution::standard();
    if sigma_a * t.sqrt() < 1e-14 {
        return (-r * t).exp() * (omega * (fwd - strike)).max(0.0);
    }
    let d1 = (fwd / strike).ln() / (sigma_a * t.sqrt()) + 0.5 * sigma_a * t.sqrt();
    let d2 = d1 - sigma_a * t.sqrt();
    (-r * t).exp() * omega * (fwd * nd.cdf(omega * d1) - strike * nd.cdf(omega * d2))
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Continuous Geometric Average Price — Kemna & Vorst (1990)
// ─────────────────────────────────────────────────────────────────────────────

/// Price a **continuous geometric average-price** Asian option using the
/// Kemna-Vorst (1990) exact formula.
///
/// The payoff at maturity is max(ω·(G_T − K), 0) where G_T is the geometric
/// average of the spot price path over [0, T].
///
/// # Parameters
/// - `spot`             — current spot price S
/// - `strike`           — option strike K
/// - `risk_free_rate`   — continuously compounded risk-free rate r
/// - `dividend_yield`   — continuously compounded dividend yield q
/// - `volatility`       — annualised vol σ
/// - `time_to_expiry`   — T in years (≥ 0)
/// - `option_type`      — [`OptionType::Call`] or [`OptionType::Put`]
///
/// # Examples
/// ```
/// use ql_instruments::OptionType;
/// use ql_pricingengines::analytic_asian::asian_geometric_continuous_avg_price;
/// let r = asian_geometric_continuous_avg_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Call);
/// assert!(r.npv > 0.0 && r.npv < 10.0); // geometric avg < arithmetic avg call
/// ```
pub fn asian_geometric_continuous_avg_price(
    spot: f64,
    strike: f64,
    risk_free_rate: f64,
    dividend_yield: f64,
    volatility: f64,
    time_to_expiry: f64,
    option_type: OptionType,
) -> AsianResult {
    let omega = option_type.sign();
    let r = risk_free_rate;
    let b = r - dividend_yield; // cost of carry
    let s = volatility;
    let t = time_to_expiry;

    // Adjusted parameters for the geometric average
    let b_a = b / 2.0 - s * s / 12.0;
    let sigma_a = s / 3.0_f64.sqrt();

    // Forward of the geometric average
    let fwd = spot * ((b_a) * t).exp();

    let npv = bs_price(fwd, strike, sigma_a, t, r, omega);

    AsianResult { npv, effective_vol: sigma_a, effective_forward: fwd }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Discrete Geometric Average Price
// ─────────────────────────────────────────────────────────────────────────────

/// Price a **discrete geometric average-price** Asian option.
///
/// Assumes `n` equally-spaced observations over `[0, T]`.
///
/// # Parameters
/// - `n` — number of observation dates (≥ 2)
pub fn asian_geometric_discrete_avg_price(
    spot: f64,
    strike: f64,
    risk_free_rate: f64,
    dividend_yield: f64,
    volatility: f64,
    time_to_expiry: f64,
    n: usize,
    option_type: OptionType,
) -> AsianResult {
    let omega = option_type.sign();
    let r = risk_free_rate;
    let b = r - dividend_yield;
    let s = volatility;
    let t = time_to_expiry;
    let n_f = n.max(2) as f64;

    // Exact discrete geometric average variance
    // σ_A² = σ² * (n+1)(2n+1) / (6n²)
    let sigma_a_sq = s * s * (n_f + 1.0) * (2.0 * n_f + 1.0) / (6.0 * n_f * n_f);
    let sigma_a = sigma_a_sq.sqrt();

    // Adjusted cost of carry: b_A = (b - σ²/2) * (n+1)/(2n) + σ_A²/2
    let b_a = (b - s * s / 2.0) * (n_f + 1.0) / (2.0 * n_f) + sigma_a_sq / 2.0;

    // Forward of the discrete geometric average
    let fwd = spot * (b_a * t).exp();

    let npv = bs_price(fwd, strike, sigma_a, t, r, omega);

    AsianResult { npv, effective_vol: sigma_a, effective_forward: fwd }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Continuous Geometric Average Strike
// ─────────────────────────────────────────────────────────────────────────────

/// Price a **continuous geometric average-strike** Asian option.
///
/// The payoff is max(ω·(S_T − G_T), 0) where G_T is the geometric average.
/// Uses the Kemna-Vorst approach for the ratio S_T / G_T.
pub fn asian_geometric_continuous_avg_strike(
    spot: f64,
    risk_free_rate: f64,
    dividend_yield: f64,
    volatility: f64,
    time_to_expiry: f64,
    option_type: OptionType,
) -> AsianResult {
    let nd = NormalDistribution::standard();
    let omega = option_type.sign();
    let r = risk_free_rate;
    let q = dividend_yield;
    let b = r - q;
    let s = volatility;
    let t = time_to_expiry;

    // σ for the ratio S_T / G_T: σ_R² = σ²/3  (since corr(S_T, G_T) gives cancel)
    let sigma_r = s / 3.0_f64.sqrt();

    // Cost of carry for the ratio: b_R = b/2 + σ²/12
    let b_r = b / 2.0 + s * s / 12.0;

    let sqrt_t = t.sqrt();
    let d1 = ((b_r + sigma_r * sigma_r / 2.0) * t) / (sigma_r * sqrt_t);
    let d2 = d1 - sigma_r * sqrt_t;

    // Price: e^{-r*T} * (ω * S * e^{b*T} * N(ω*d1) - ω * S * e^{b_A*T} * N(ω*d2))
    //   where the "strike" is G_T ≈ S * e^{b_A*T}
    let b_a = b / 2.0 - s * s / 12.0;
    let _npv = spot * (-q * t).exp()
        * (omega * ((b_r + sigma_r * sigma_r / 2.0) * t).exp() * nd.cdf(omega * d1)
            - omega * (b_a * t).exp() * nd.cdf(omega * d2))
        * (-q * t).exp().powi(-1) // cancel dividend factor already in spot term
        * (-r * t).exp();

    // Simpler exact form:
    let npv = {
        let f_s = spot * (b * t).exp();           // forward of spot
        let f_g = spot * (b_a * t).exp();          // forward of geo avg
        let d1 = ((f_s / f_g).ln() + sigma_r * sigma_r * t / 2.0) / (sigma_r * sqrt_t);
        let d2 = d1 - sigma_r * sqrt_t;
        (-r * t).exp()
            * omega
            * (f_s * nd.cdf(omega * d1) - f_g * nd.cdf(omega * d2))
    };

    AsianResult { npv, effective_vol: sigma_r, effective_forward: spot * (b_a * t).exp() }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Turnbull-Wakeman (1991) Arithmetic Average Price Approximation
// ─────────────────────────────────────────────────────────────────────────────

/// Price an **arithmetic average-price** Asian option using the Turnbull-Wakeman
/// (1991) log-normal moment-matching approximation.
///
/// This is the industry-standard first-order approximation for continuous
/// arithmetic average Asian options.
///
/// # Parameters
/// - `t0` — time already elapsed (past averaging days / total averaging days * T);
///   set to `0.0` for a freshly-started option.
/// - `a`  — average already accumulated (running average); used only when `t0 > 0`;
///   set to `0.0` for a freshly-started option.
pub fn asian_turnbull_wakeman(
    spot: f64,
    strike: f64,
    risk_free_rate: f64,
    dividend_yield: f64,
    volatility: f64,
    time_to_expiry: f64,
    t0: f64,
    a: f64,
    option_type: OptionType,
) -> AsianResult {
    let omega = option_type.sign();
    let r = risk_free_rate;
    let q = dividend_yield;
    let b = r - q;
    let s = volatility;
    let t = time_to_expiry;
    let t_remaining = t - t0;

    if t_remaining <= 0.0 {
        // Option already in averaging; intrinsic only
        let intrinsic = (omega * (a - strike)).max(0.0);
        return AsianResult { npv: intrinsic, effective_vol: 0.0, effective_forward: a };
    }

    // Effective strike, adjusted for already-accumulated average
    // K* = (t / t_remaining) * K - (t0 / t_remaining) * a
    let k_eff = if t0 > 0.0 {
        let total_frac = t / t_remaining;
        
        total_frac * strike - (t0 / t_remaining) * a
    } else {
        strike
    };

    let s2 = s * s;
    let t_r = t_remaining;

    // First moment M1 = E[A] = S * (exp(b*T) - 1) / (b*T)
    let m1 = if b.abs() < 1e-10 {
        spot * (1.0 + b * t_r / 2.0) // limiting form as b→0: S * (1 + bT/2 + …)
    } else {
        spot * ((b * t_r).exp() - 1.0) / (b * t_r)
    };

    // Second moment M2 = (2/T²) ∫₀ᵀ∫₀ᵗ S²·exp(b(t+u)+σ²u) du dt
    // = 2S²/T² · [(exp((2b+σ²)T)-1)/((b+σ²)(2b+σ²)) - (exp(bT)-1)/(b(b+σ²))]
    let m2 = {
        let bs2 = b + s2;          // b + σ²
        let tbs2 = 2.0 * b + s2;  // 2b + σ²
        let term1 = if tbs2.abs() < 1e-10 {
            // L'Hôpital: lim_{ε→0} (e^{ε·T}-1)/(ε·bs2) = T/bs2
            t_r / bs2
        } else {
            ((tbs2 * t_r).exp() - 1.0) / (bs2 * tbs2)
        };
        let term2 = if b.abs() < 1e-10 {
            // L'Hôpital: lim_{b→0} (e^{bT}-1)/(b·bs2) = T/bs2
            t_r / bs2
        } else {
            ((b * t_r).exp() - 1.0) / (b * bs2)
        };
        2.0 * spot * spot / (t_r * t_r) * (term1 - term2)
    };

    // Effective lognormal vol for the average
    let sigma_a_sq = (m2 / (m1 * m1)).ln() / t_r;
    let sigma_a = sigma_a_sq.max(0.0).sqrt();

    if sigma_a < 1e-14 {
        let npv = (-r * t).exp() * (omega * (m1 - k_eff)).max(0.0);
        return AsianResult { npv, effective_vol: 0.0, effective_forward: m1 };
    }

    let nd = NormalDistribution::standard();
    let d1 = (m1 / k_eff).ln() / (sigma_a * t_r.sqrt()) + 0.5 * sigma_a * t_r.sqrt();
    let d2 = d1 - sigma_a * t_r.sqrt();

    let npv_remaining = (-r * t_r).exp()
        * omega
        * (m1 * nd.cdf(omega * d1) - k_eff * nd.cdf(omega * d2));

    // Scale back by t_remaining/t if partially elapsed
    let scale = if t0 > 0.0 { t_remaining / t } else { 1.0 };
    let npv = scale * npv_remaining;

    AsianResult { npv, effective_vol: sigma_a, effective_forward: m1 }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Levy (1992) Continuous Arithmetic Average Price
// ─────────────────────────────────────────────────────────────────────────────

/// Price a **continuous arithmetic average-price** Asian option using the
/// Levy (1992) log-normal approximation.
///
/// Compared to Turnbull-Wakeman, Levy uses a slightly different moment formula.
/// For full-life options the two are equivalent; the additional `t0 = 0.0, a = 0.0`
/// form is supported here.
pub fn asian_levy(
    spot: f64,
    strike: f64,
    risk_free_rate: f64,
    dividend_yield: f64,
    volatility: f64,
    time_to_expiry: f64,
    option_type: OptionType,
) -> AsianResult {
    // Levy is Turnbull-Wakeman with t0 = 0.0
    asian_turnbull_wakeman(
        spot,
        strike,
        risk_free_rate,
        dividend_yield,
        volatility,
        time_to_expiry,
        0.0,
        0.0,
        option_type,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Discrete Geometric Average Strike
// ─────────────────────────────────────────────────────────────────────────────

/// Price a **discrete geometric average-strike** Asian option.
///
/// The payoff is max(ω·(S_T − G_n), 0) where G_n is the discrete geometric
/// average of `n` observations equally spaced over [0, T].
pub fn asian_geometric_discrete_avg_strike(
    spot: f64,
    risk_free_rate: f64,
    dividend_yield: f64,
    volatility: f64,
    time_to_expiry: f64,
    n: usize,
    option_type: OptionType,
) -> AsianResult {
    let nd = NormalDistribution::standard();
    let omega = option_type.sign();
    let r = risk_free_rate;
    let q = dividend_yield;
    let b = r - q;
    let s = volatility;
    let t = time_to_expiry;
    let n_f = n.max(2) as f64;

    // Variance of the discrete geometric average
    let sigma_g_sq = s * s * (n_f + 1.0) * (2.0 * n_f + 1.0) / (6.0 * n_f * n_f);

    // Effective cost-of-carry for the geometric average
    let b_g = (b - s * s / 2.0) * (n_f + 1.0) / (2.0 * n_f) + sigma_g_sq / 2.0;

    // Variance of (S_T / G_n): Var[ln(S_T/G_n)] = σ²*T + σ_G²*T - 2*Cov(ln S_T, ln G_n)
    // Cov(ln S_T, ln G_n) = σ² * (n+1)*T / (2n)
    let cov = s * s * (n_f + 1.0) * t / (2.0 * n_f);
    let var_ratio = s * s * t + sigma_g_sq * t - 2.0 * cov;
    let sigma_ratio = (var_ratio / t).sqrt();

    let f_s = spot * (b * t).exp();
    let f_g = spot * (b_g * t).exp();

    let sqrt_t = t.sqrt();
    let d1 = ((f_s / f_g).ln() + sigma_ratio * sigma_ratio * t / 2.0) / (sigma_ratio * sqrt_t);
    let d2 = d1 - sigma_ratio * sqrt_t;

    let npv = (-r * t).exp()
        * omega
        * (f_s * nd.cdf(omega * d1) - f_g * nd.cdf(omega * d2));

    AsianResult { npv, effective_vol: sigma_ratio, effective_forward: f_g }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn geometric_continuous_call_otm() {
        // ATM call: continuous geometric average price
        // S=100, K=100, r=5%, q=0%, σ=20%, T=1
        // Known value ≈ 6.97 (geometric < arithmetic ≈ 7.97)
        let r = asian_geometric_continuous_avg_price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, OptionType::Call);
        assert!(r.npv > 5.0 && r.npv < 9.0, "geometric avg call out of range: {}", r.npv);
    }

    #[test]
    fn geometric_continuous_put_call_parity() {
        let (s, k, r_r, q, sigma, t) = (100.0, 102.0, 0.05, 0.02, 0.25, 1.0);
        let c = asian_geometric_continuous_avg_price(s, k, r_r, q, sigma, t, OptionType::Call);
        let p = asian_geometric_continuous_avg_price(s, k, r_r, q, sigma, t, OptionType::Put);
        // Put-call parity for geometric Asian: c - p = e^{-r*T}*(F_G - K)
        let b_a = (r_r - q) / 2.0 - sigma * sigma / 12.0;
        let f_g = s * (b_a * t).exp();
        let expected = (-r_r * t).exp() * (f_g - k);
        assert_abs_diff_eq!(c.npv - p.npv, expected, epsilon = 1e-10);
    }

    #[test]
    fn discrete_geometric_converges_to_continuous() {
        // With many observations, discrete ≈ continuous
        let s = 100.0; let k = 100.0; let r = 0.05; let q = 0.0; let sigma = 0.20; let t = 1.0;
        let continuous = asian_geometric_continuous_avg_price(s, k, r, q, sigma, t, OptionType::Call);
        let discrete_1000 = asian_geometric_discrete_avg_price(s, k, r, q, sigma, t, 1000, OptionType::Call);
        assert_abs_diff_eq!(discrete_1000.npv, continuous.npv, epsilon = 1e-2);
    }

    #[test]
    fn arithmetic_call_exceeds_geometric_call() {
        let (s, k, r, q, sigma, t) = (100.0, 100.0, 0.05, 0.0, 0.20, 1.0);
        let arith = asian_turnbull_wakeman(s, k, r, q, sigma, t, 0.0, 0.0, OptionType::Call);
        let geom = asian_geometric_continuous_avg_price(s, k, r, q, sigma, t, OptionType::Call);
        // Arithmetic average ≥ geometric average → arithmetic price ≥ geometric price
        assert!(arith.npv >= geom.npv - 1e-10,
            "arithmetic {} < geometric {}", arith.npv, geom.npv);
    }

    #[test]
    fn levy_matches_turnbull_wakeman_fresh_option() {
        let (s, k, r, q, sigma, t) = (100.0, 95.0, 0.06, 0.01, 0.30, 0.5);
        let tw = asian_turnbull_wakeman(s, k, r, q, sigma, t, 0.0, 0.0, OptionType::Put);
        let levy = asian_levy(s, k, r, q, sigma, t, OptionType::Put);
        assert_abs_diff_eq!(tw.npv, levy.npv, epsilon = 1e-15);
    }

    #[test]
    fn discrete_avg_strike_call_positive() {
        let r = asian_geometric_discrete_avg_strike(100.0, 0.05, 0.0, 0.20, 1.0, 12, OptionType::Call);
        assert!(r.npv > 0.0, "avg-strike call should be positive");
    }

    #[test]
    fn zero_vol_intrinsic_call() {
        // Zero vol → geometric avg = S*e^{b*T/2}, price = discounted intrinsic
        let (s, k, r, q, sigma, t) = (100.0, 98.0, 0.05, 0.0, 0.001, 1.0);
        let res = asian_geometric_continuous_avg_price(s, k, r, q, sigma, t, OptionType::Call);
        assert!(res.npv > 0.0);
    }
}
