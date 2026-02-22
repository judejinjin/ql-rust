//! Analytic Variance Gamma option pricing engine.
//!
//! Prices European vanilla options under the Variance Gamma model using the
//! Fourier-cosine (COS) method (Fang & Oosterlee 2008) with the VG
//! characteristic function.
//!
//! ## Algorithm
//!
//! 1. Compute the VG log-return characteristic function φ_VG(u, T).
//! 2. Apply the COS expansion over the truncation interval [a, b] chosen from
//!    the first two VG cumulants.
//! 3. The cosine payoff coefficients U_k are the same as for Heston/BS.
//!
//! ## References
//! - Madan, D.B., Carr, P., Chang, E.C. (1998). "The variance gamma process."
//! - Fang, F. & Oosterlee, C.W. (2008). *SIAM J. Sci. Comput.*

use std::f64::consts::PI;

use ql_core::errors::{QLError, QLResult};
use ql_instruments::OptionType;
use ql_models::variance_gamma_model::VarianceGammaModel;

/// Result from the VG pricing engine.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VgResult {
    /// Option price.
    pub price: f64,
    /// Approximate delta (finite-difference).
    pub delta: f64,
    /// Approximate vega w.r.t. σ.
    pub vega: f64,
}

/// Price a European vanilla option under the Variance Gamma model.
///
/// Uses the COS method with N cosine terms. For most applications
/// N=128 at L=10 gives machine-precision accuracy.
///
/// # Arguments
/// - `model` — calibrated VG model (σ, ν, θ)
/// - `spot` — current spot S₀
/// - `strike` — option strike K
/// - `tau` — time to maturity T (years)
/// - `risk_free_rate` — continuous risk-free rate r
/// - `dividend_yield` — continuous dividend yield q
/// - `opt_type` — call or put
/// - `n_terms` — COS terms (default 128 when 0)
/// - `l` — truncation half-width in cumulant std-devs (default 12 when ≤ 0)
pub fn vg_cos_price(
    model: &VarianceGammaModel,
    spot: f64,
    strike: f64,
    tau: f64,
    risk_free_rate: f64,
    dividend_yield: f64,
    opt_type: OptionType,
    n_terms: usize,
    l: f64,
) -> QLResult<VgResult> {
    if spot <= 0.0 || strike <= 0.0 || tau <= 0.0 {
        return Err(QLError::InvalidArgument("spot, strike, tau must be positive".into()));
    }
    let omega = model.omega().ok_or_else(|| {
        QLError::InvalidArgument("VG parameters violate martingale condition".into())
    })?;
    let n = if n_terms == 0 { 128 } else { n_terms };
    let l = if l <= 0.0 { 12.0 } else { l };

    let r = risk_free_rate;
    let q = dividend_yield;
    let log_fk = (spot / strike).ln() + (r - q) * tau;

    // Cumulants for interval selection
    let c1 = (r - q + omega + model.theta) * tau;
    let c2 = model.c2(tau);
    let c2s = c2.abs().sqrt().max(1e-4);
    let a = c1 - l * c2s;
    let b = c1 + l * c2s;
    let range = b - a;

    let df = (-r * tau).exp();

    // Payoff coefficients (same structure as COS-Heston)
    let u_k: Vec<f64> = match opt_type {
        OptionType::Call => {
            (0..n).map(|k| {
                let kf = k as f64;
                let kpi = kf * PI;
                let chi = chi_k(kf, 0.0, b, a, range);
                let psi = psi_k(kpi, 0.0, b, a, range);
                2.0 / range * strike * (chi - psi)
            }).collect()
        }
        OptionType::Put => {
            (0..n).map(|k| {
                let kf = k as f64;
                let kpi = kf * PI;
                let chi = chi_k(kf, a, 0.0, a, range);
                let psi = psi_k(kpi, a, 0.0, a, range);
                2.0 / range * strike * (-chi + psi)
            }).collect()
        }
    };

    // Sum COS series
    let mut price = 0.0_f64;
    for k in 0..n {
        let freq = k as f64 * PI / range;
        let (cf_re, cf_im) = vg_cf_with_forward(model, freq, tau, omega, r - q, log_fk);
        // Phase: e^{-ikπa/(b-a)}
        let phase_arg = -(k as f64) * PI * a / range;
        let (cos_ph, sin_ph) = (phase_arg.cos(), phase_arg.sin());
        // Re[ φ(freq) × e^{i·phase} ] = re(cf)·cos - im(cf)·sin
        let re_cf_phase = cf_re * cos_ph - cf_im * sin_ph;
        let weight = if k == 0 { 0.5 } else { 1.0 };
        price += weight * re_cf_phase * u_k[k];
    }
    price *= df;
    let price = price.max(0.0);

    // Greeks via finite difference
    let eps = spot * 1e-4;
    let p_up = vg_cos_scalar(model, spot + eps, strike, tau, r, q, opt_type, n, l, omega, a, b);
    let p_dn = vg_cos_scalar(model, spot - eps, strike, tau, r, q, opt_type, n, l, omega, a, b);
    let delta = (p_up - p_dn) / (2.0 * eps);

    let dsig = 1e-4;
    let m_up = VarianceGammaModel::new(model.sigma + dsig, model.nu, model.theta);
    let m_dn = VarianceGammaModel::new((model.sigma - dsig).max(1e-6), model.nu, model.theta);
    let v_up = vg_price_only(&m_up, spot, strike, tau, r, q, opt_type, n, l);
    let v_dn = vg_price_only(&m_dn, spot, strike, tau, r, q, opt_type, n, l);
    let vega = (v_up - v_dn) / (2.0 * dsig);

    Ok(VgResult { price, delta, vega })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// VG characteristic function of X(T) = ln(S_T/K) including forward/drift.
///
/// Returns `(re, im)` of φ(u) × exp(i·u·log_fk).
fn vg_cf_with_forward(
    model: &VarianceGammaModel,
    u: f64,
    tau: f64,
    omega: f64,
    b: f64,  // r - q
    log_fk: f64,
) -> (f64, f64) {
    // Full log-forward: ln(S₀/K) + (r-q+ω)·T
    // log_fk already = ln(S₀/K) + (r-q)·T; add ω·T to get E[ln(S_T/K)] from drift
    let (vg_re, vg_im) = model.log_cf(u, tau);
    // Multiply by exp(i·u·(log_fk + ω·T))
    let phase = u * (log_fk + omega * tau);
    let (cos_ph, sin_ph) = (phase.cos(), phase.sin());
    let re = vg_re * cos_ph - vg_im * sin_ph;
    let im = vg_re * sin_ph + vg_im * cos_ph;
    let _ = b; // b (r-q) already incorporated via log_fk
    (re, im)
}

/// Scalar price calculation (no greeks) for finite-difference use.
fn vg_cos_scalar(
    model: &VarianceGammaModel,
    spot: f64,
    strike: f64,
    tau: f64,
    r: f64,
    q: f64,
    opt_type: OptionType,
    n: usize,
    l: f64,
    omega: f64,
    a: f64,
    b: f64,
) -> f64 {
    let log_fk = (spot / strike).ln() + (r - q) * tau;
    let range = b - a;
    let df = (-r * tau).exp();
    let u_k: Vec<f64> = match opt_type {
        OptionType::Call => (0..n).map(|k| {
            let kf = k as f64;
            let kpi = kf * PI;
            2.0 / range * strike * (chi_k(kf, 0.0, b, a, range) - psi_k(kpi, 0.0, b, a, range))
        }).collect(),
        OptionType::Put => (0..n).map(|k| {
            let kf = k as f64;
            let kpi = kf * PI;
            2.0 / range * strike * (-chi_k(kf, a, 0.0, a, range) + psi_k(kpi, a, 0.0, a, range))
        }).collect(),
    };
    let mut price = 0.0;
    for k in 0..n {
        let freq = k as f64 * PI / range;
        let (cf_re, cf_im) = vg_cf_with_forward(model, freq, tau, omega, r - q, log_fk);
        let phase_arg = -(k as f64) * PI * a / range;
        let re_cf_phase = cf_re * phase_arg.cos() - cf_im * phase_arg.sin();
        let weight = if k == 0 { 0.5 } else { 1.0 };
        price += weight * re_cf_phase * u_k[k];
    }
    (price * df).max(0.0)
}

/// Price-only (no greeks) helper used for vega finite differences to avoid
/// infinite recursion that would occur if `vg_cos_price` called itself.
fn vg_price_only(
    model: &VarianceGammaModel,
    spot: f64,
    strike: f64,
    tau: f64,
    r: f64,
    q: f64,
    opt_type: OptionType,
    n: usize,
    l: f64,
) -> f64 {
    let omega = match model.omega() {
        Some(v) => v,
        None => return 0.0,
    };
    let c1 = (r - q + omega + model.theta) * tau;
    let c2 = model.c2(tau);
    let c2s = c2.abs().sqrt().max(1e-4);
    let a = c1 - l * c2s;
    let b = c1 + l * c2s;
    vg_cos_scalar(model, spot, strike, tau, r, q, opt_type, n, l, omega, a, b)
}

fn chi_k(k: f64, c: f64, d: f64, a: f64, range: f64) -> f64 {
    let kpi_over_range = k * PI / range;
    let denom = 1.0 + kpi_over_range * kpi_over_range;
    let term_d = (d - a) * kpi_over_range;
    let term_c = (c - a) * kpi_over_range;
    (1.0 / denom)
        * (d.exp() * (term_d.cos() + kpi_over_range * term_d.sin())
           - c.exp() * (term_c.cos() + kpi_over_range * term_c.sin()))
}

fn psi_k(kpi: f64, c: f64, d: f64, a: f64, range: f64) -> f64 {
    if kpi.abs() < 1e-12 { return d - c; }
    let kpi_over_range = kpi / range;
    (((d - a) * kpi_over_range).sin() - ((c - a) * kpi_over_range).sin()) / kpi_over_range
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn mcc_model() -> VarianceGammaModel {
        // Madan-Carr-Chang (1998) calibrated parameters for S&P500
        // σ=0.12, ν=0.017, θ=−0.14
        VarianceGammaModel::new(0.12, 0.017, -0.14)
    }

    #[test]
    fn vg_call_positive() {
        let m = mcc_model();
        let res = vg_cos_price(&m, 100.0, 100.0, 1.0, 0.05, 0.0, OptionType::Call, 128, 12.0)
            .unwrap();
        assert!(res.price > 0.0 && res.price < 50.0, "price={}", res.price);
    }

    #[test]
    fn vg_put_call_parity() {
        let m = mcc_model();
        let s = 100.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let q = 0.0;
        let call = vg_cos_price(&m, s, k, t, r, q, OptionType::Call, 256, 14.0).unwrap();
        let put = vg_cos_price(&m, s, k, t, r, q, OptionType::Put, 256, 14.0).unwrap();
        let fwd = s * (-q * t).exp() - k * (-r * t).exp();
        let err = (call.price - put.price - fwd).abs();
        assert!(err < 0.5, "pcp err={} call={} put={}", err, call.price, put.price);
    }

    #[test]
    fn vg_call_decreasing_in_strike() {
        let m = VarianceGammaModel::new(0.20, 0.10, -0.10);
        let c90 = vg_cos_price(&m, 100.0, 90.0, 1.0, 0.05, 0.0, OptionType::Call, 128, 12.0)
            .unwrap().price;
        let c100 = vg_cos_price(&m, 100.0, 100.0, 1.0, 0.05, 0.0, OptionType::Call, 128, 12.0)
            .unwrap().price;
        let c110 = vg_cos_price(&m, 100.0, 110.0, 1.0, 0.05, 0.0, OptionType::Call, 128, 12.0)
            .unwrap().price;
        assert!(c90 > c100 && c100 > c110, "c90={} c100={} c110={}", c90, c100, c110);
    }

    #[test]
    fn vg_call_positive_delta() {
        let m = mcc_model();
        let res = vg_cos_price(&m, 100.0, 100.0, 1.0, 0.05, 0.0, OptionType::Call, 128, 12.0)
            .unwrap();
        assert!(res.delta > 0.0 && res.delta < 1.0, "delta={}", res.delta);
    }

    #[test]
    fn vg_invalid_params() {
        let m = VarianceGammaModel::new(0.10, 100.0, 100.0); // θν > 1: omega undefined
        let res = vg_cos_price(&m, 100.0, 100.0, 1.0, 0.05, 0.0, OptionType::Call, 64, 10.0);
        assert!(res.is_err(), "should fail for invalid params");
    }
}
