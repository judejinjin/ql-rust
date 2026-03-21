//! COS (Fourier-cosine series) Heston pricing engine.
//!
//! Implements the Fang & Oosterlee (2008) COS method for European options
//! under the Heston stochastic-volatility model.  The pricing algorithm
//! uses an N-term cosine expansion of the risk-neutral density and is
//! typically 5–10× faster than the standard Gauss-Legendre Fourier
//! integration at equal accuracy.
//!
//! ## Algorithm summary
//!
//! Given X = ln(S_T / K), the price is:
//!
//! ```text
//! V ≈ e^{-rT} ∑_{k=0}^{N-1} ' Re[ φ_X(kπ/(b-a)) · e^{-ikπa/(b-a)} ] · V_k
//! ```
//!
//! where `' ` means the k=0 term is halved, φ_X is the log-return
//! characteristic function of the Heston model, and V_k are the cosine
//! payoff coefficients for call or put.
//!
//! ## References
//!
//! - Fang, F. & Oosterlee, C.W. (2008). *A novel pricing method for
//!   European options based on Fourier-cosine series expansions.*
//!   SIAM Journal on Scientific Computing.

use std::f64::consts::PI;

use ql_core::errors::{QLError, QLResult};
use ql_instruments::OptionType;
use ql_models::HestonModel;

// ---------------------------------------------------------------------------
// Minimal inline complex arithmetic
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct C {
    re: f64,
    im: f64,
}

#[allow(dead_code)]
impl C {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    fn from_real(re: f64) -> Self { Self { re, im: 0.0 } }
    fn exp(self) -> Self {
        let r = self.re.exp();
        Self { re: r * self.im.cos(), im: r * self.im.sin() }
    }
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
    fn add(self, rhs: Self) -> Self {
        Self { re: self.re + rhs.re, im: self.im + rhs.im }
    }
    fn scale(self, s: f64) -> Self {
        Self { re: self.re * s, im: self.im * s }
    }
    fn conj_re(self) -> f64 { self.re }
}

// ---------------------------------------------------------------------------
// Heston characteristic function of log(S_T/K) = log(S_T) - log(K)
//
// We use the "rotation-count free" Albrecher et al. (2007) formulation.
// ---------------------------------------------------------------------------

fn heston_log_cf(
    u: f64,            // frequency
    tau: f64,          // time to maturity
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    _rd: f64,           // risk-free rate
    _rd_minus_q: f64,   // r − q (forward measure adjustment)
    x0_over_k: f64,    // ln(F/K) where F = S·e^{(r-q)T}
) -> C {
    // u is the purely real frequency argument; characteristic function φ(u)
    // where X = ln(S_T/K) under the risk-neutral measure with forward.
    let _xi = C::new(kappa - rho * sigma * u, -rho * sigma * 0.0); // kappa - rho*sigma*(iu)
    // For logCF we evaluate at iu (purely imaginary)
    let _a = kappa;
    let _b = C::new(
        kappa - rho * sigma * /* nothing */ 0.0,
        -rho * sigma * u,
    );
    // d = sqrt((b - i*sigma*u)^2 + sigma^2 * u^2)
    //   with the Albrecher choice  d = sqrt((rho*sigma*iu - kappa)^2 + sigma^2(iu+u^2))
    // Standard formula: we compute φ(u) = exp(A + B*v0) where
    //   d^2 = (kappa - i*rho*sigma*u)^2 + sigma^2*(u^2 + i*u)
    let _iu = C::new(0.0, u);
    let _neg_iu = C::new(0.0, -u);
    // alpha_hat = -0.5*(u^2 + i*u)
    // beta = kappa - rho*sigma*i*u
    let alpha = C::new(-0.5 * (u * u), -0.5 * u); // -½(u² + iu)
    let beta = C::new(kappa, -rho * sigma * u);
    // gamma = 0.5*sigma²
    let _half_sig2 = 0.5 * sigma * sigma;
    // d = sqrt(beta² - 2*gamma*alpha*2) ... use standard form
    // d² = beta² - sigma²*alpha  (since 2*gamma = sigma²)
    let beta_sq = C::new(
        beta.re * beta.re - beta.im * beta.im,
        2.0 * beta.re * beta.im,
    );
    let sig2_alpha = C::new(sigma * sigma * alpha.re, sigma * sigma * alpha.im);
    let d_sq = C::new(beta_sq.re - sig2_alpha.re, beta_sq.im - sig2_alpha.im);
    // d = sqrt(d_sq)
    let d_abs = (d_sq.re * d_sq.re + d_sq.im * d_sq.im).sqrt().sqrt();
    let d_arg = d_sq.im.atan2(d_sq.re) * 0.5;
    let d = C::new(d_abs * d_arg.cos(), d_abs * d_arg.sin());

    // g = (beta - d) / (beta + d)
    let num = C::new(beta.re - d.re, beta.im - d.im);
    let den = C::new(beta.re + d.re, beta.im + d.im);
    let den_norm = den.re * den.re + den.im * den.im;
    let g = if den_norm < 1e-30 {
        C::new(0.0, 0.0)
    } else {
        C::new(
            (num.re * den.re + num.im * den.im) / den_norm,
            (num.im * den.re - num.re * den.im) / den_norm,
        )
    };

    // e^{d*tau}
    let d_tau = C::new(d.re * tau, d.im * tau);
    let exp_dtau = d_tau.exp();

    // (1 − g·e^{d*tau}) / (1 − g)
    let ge_dtau = g.mul(exp_dtau);
    let _one = C::new(1.0, 0.0);
    let num2 = C::new(1.0 - ge_dtau.re, -ge_dtau.im);
    let den2 = C::new(1.0 - g.re, -g.im);
    let den2_norm = den2.re * den2.re + den2.im * den2.im;
    let _ratio = if den2_norm < 1e-30 {
        C::new(1.0, 0.0)
    } else {
        C::new(
            (num2.re * den2.re + num2.im * den2.im) / den2_norm,
            (num2.im * den2.re - num2.re * den2.im) / den2_norm,
        )
    };

    // B = (beta - d) / sigma² * (1 - e^{-d*tau}) / (1 - g*e^{-d*tau})
    // Use the numerically stable form in terms of d and g above.
    // Following Fang & Oosterlee, we express:
    //   B(u) = (beta - d)/sigma² · (1 - exp(-d*tau))/(1 - g*exp(-d*tau))
    // Here we compute via:
    //   B = 2*alpha/(beta + d) · (1 - exp(-d*tau))/(1 - g*exp(-d*tau))
    // but we'll use the simpler  B = (beta-d)/sigma² * ...

    // 1 - exp(-d*tau):
    let neg_d_tau = C::new(-d.re * tau, -d.im * tau);
    let exp_neg_dtau = neg_d_tau.exp();
    let one_minus_exp = C::new(1.0 - exp_neg_dtau.re, -exp_neg_dtau.im);
    // 1 - g*exp(-d*tau):
    let g_exp_neg = g.mul(exp_neg_dtau);
    let denom_b = C::new(1.0 - g_exp_neg.re, -g_exp_neg.im);
    let denom_b_norm = denom_b.re * denom_b.re + denom_b.im * denom_b.im;
    let frac_b = if denom_b_norm < 1e-30 {
        C::new(0.0, 0.0)
    } else {
        C::new(
            (one_minus_exp.re * denom_b.re + one_minus_exp.im * denom_b.im) / denom_b_norm,
            (one_minus_exp.im * denom_b.re - one_minus_exp.re * denom_b.im) / denom_b_norm,
        )
    };
    let sigma_sq = sigma * sigma;
    let bm_d = C::new(beta.re - d.re, beta.im - d.im);
    let big_b = bm_d.mul(frac_b).scale(1.0 / sigma_sq);

    // A = kappa*theta/sigma² * [(beta-d)*tau - 2*ln((1 - g*e^{-d*tau})/(1-g))]
    let inner_log_num = C::new(1.0 - g_exp_neg.re, -g_exp_neg.im);
    let inner_log_den = C::new(1.0 - g.re, -g.im);
    let iln_den_norm = inner_log_den.re * inner_log_den.re + inner_log_den.im * inner_log_den.im;
    let log_ratio_re = if iln_den_norm < 1e-30 {
        0.0
    } else {
        let real_q = (inner_log_num.re * inner_log_den.re + inner_log_num.im * inner_log_den.im) / iln_den_norm;
        let imag_q = (inner_log_num.im * inner_log_den.re - inner_log_num.re * inner_log_den.im) / iln_den_norm;
        let abs_q = (real_q * real_q + imag_q * imag_q).sqrt();
        
        // Result re of ln(ratio)
        abs_q.ln()
    };
    let log_ratio_im = if iln_den_norm < 1e-30 {
        0.0
    } else {
        let real_q = (inner_log_num.re * inner_log_den.re + inner_log_num.im * inner_log_den.im) / iln_den_norm;
        let imag_q = (inner_log_num.im * inner_log_den.re - inner_log_num.re * inner_log_den.im) / iln_den_norm;
        imag_q.atan2(real_q)
    };

    let bt_tau = C::new(bm_d.re * tau, bm_d.im * tau);
    let big_a_bracket = C::new(
        bt_tau.re - 2.0 * log_ratio_re,
        bt_tau.im - 2.0 * log_ratio_im,
    );
    let kappa_theta_over_sig2 = kappa * theta / sigma_sq;
    let big_a = big_a_bracket.scale(kappa_theta_over_sig2);

    // φ(u) = e^{A + B*v0 + i*u*x0_over_k}
    let b_v0 = big_b.scale(v0);
    let iu_x = C::new(0.0, u * x0_over_k);
    let exponent = C::new(
        big_a.re + b_v0.re + iu_x.re,
        big_a.im + b_v0.im + iu_x.im,
    );
    exponent.exp()
}

// ---------------------------------------------------------------------------
// Payoff cosine coefficients  U_k  for call and put
// ---------------------------------------------------------------------------

/// Cosine payoff coefficients for a European call/put on [a, b].
///
/// For a call (exercise at K): payoff(x) = K·(e^x − 1)^+  where x = ln(S_T/K)
/// For a put:  payoff(x) = K·(1 − e^x)^+
fn payoff_cos_coeffs(n: usize, a: f64, b: f64, strike: f64, opt_type: OptionType) -> Vec<f64> {
    let range = b - a;
    match opt_type {
        OptionType::Call => {
            // Integration from 0 to b
            (0..n)
                .map(|k| {
                    let kf = k as f64;
                    let kpi = kf * PI;
                    let chi = chi_k(kf, 0.0, b, a, range);
                    let psi = psi_k(kpi, 0.0, b, a, range);
                    2.0 / range * strike * (chi - psi)
                })
                .collect()
        }
        OptionType::Put => {
            // Integration from a to 0
            (0..n)
                .map(|k| {
                    let kf = k as f64;
                    let kpi = kf * PI;
                    let chi = chi_k(kf, a, 0.0, a, range);
                    let psi = psi_k(kpi, a, 0.0, a, range);
                    2.0 / range * strike * (-chi + psi)
                })
                .collect()
        }
    }
}

/// χ_k(c, d) = ∫_c^d e^x cos(kπ(x-a)/(b-a)) dx
fn chi_k(k: f64, c: f64, d: f64, a: f64, range: f64) -> f64 {
    let kpi_over_range = k * PI / range;
    let denom = 1.0 + kpi_over_range * kpi_over_range;
    let term_d = (d - a) * kpi_over_range;
    let term_c = (c - a) * kpi_over_range;
    (1.0 / denom)
        * (d.exp() * (term_d.cos() + kpi_over_range * term_d.sin())
            - c.exp() * (term_c.cos() + kpi_over_range * term_c.sin()))
}

/// ψ_k(c, d) = ∫_c^d cos(kπ(x-a)/(b-a)) dx
fn psi_k(kpi: f64, c: f64, d: f64, a: f64, range: f64) -> f64 {
    if kpi.abs() < 1e-12 {
        return d - c;
    }
    let kpi_over_range = kpi / range;
    (((d - a) * kpi_over_range).sin() - ((c - a) * kpi_over_range).sin())
        / kpi_over_range
}

// ---------------------------------------------------------------------------
// COS pricing result
// ---------------------------------------------------------------------------

/// Result from the COS Heston pricing engine.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CosHestonResult {
    /// Option price.
    pub price: f64,
    /// Number of COS terms used.
    pub n_terms: usize,
    /// Truncation interval [a, b] used.
    pub ab: (f64, f64),
}

/// Price a European option under the Heston model using the COS method.
///
/// # Arguments
/// - `model` — calibrated Heston model
/// - `spot` — current spot price S₀
/// - `strike` — option strike K
/// - `tau` — time to maturity T (in years)
/// - `risk_free_rate` — continuously compounded risk-free rate r
/// - `dividend_yield` — continuous dividend yield q
/// - `opt_type` — call or put
/// - `n_terms` — number of cosine series terms (default: 128; use 64–256)
/// - `L` — truncation parameter (default: 12.0; controls [a,b] width)
#[allow(clippy::needless_range_loop)]
pub fn cos_heston_price(
    model: &HestonModel,
    spot: f64,
    strike: f64,
    tau: f64,
    risk_free_rate: f64,
    dividend_yield: f64,
    opt_type: OptionType,
    n_terms: usize,
    l: f64,
) -> QLResult<CosHestonResult> {
    if strike <= 0.0 || spot <= 0.0 || tau <= 0.0 {
        return Err(QLError::InvalidArgument(
            "spot, strike, and tau must be positive".into(),
        ));
    }
    let n = if n_terms == 0 { 128 } else { n_terms };
    let l = if l <= 0.0 { 12.0 } else { l };

    let v0 = model.v0();
    let kappa = model.kappa();
    let theta = model.theta();
    let sigma = model.sigma();
    let rho = model.rho();

    // Log-forward adjusted x₀ = ln(S₀/K) + (r-q)*τ = ln(F/K)
    let log_fk = (spot / strike).ln() + (risk_free_rate - dividend_yield) * tau;

    // Cumulants of log(S_T/K) for interval width estimation
    // c1 = log(F/K) + (mean of log-return)
    // We use the leading Heston cumulants (Fang & Oosterlee 2008, Appendix)
    let e_kappa_tau = (-kappa * tau).exp();
    let c1 = log_fk
        + (1.0 - e_kappa_tau) * (theta - v0) / (2.0 * kappa)
        - theta * tau / 2.0;
    let c2 = theta * tau / (2.0 * kappa)
        * (sigma.powi(2) - 2.0 * kappa * theta)
        + v0 * (1.0 - e_kappa_tau) * (1.0 - e_kappa_tau) / (4.0 * kappa)
            * (sigma.powi(2) - 2.0 * kappa * theta)
        + (1.0 - e_kappa_tau) * theta / (2.0) * kappa;

    let c2 = c2.abs().max(0.01);
    let a = c1 - l * c2.sqrt();
    let b = c1 + l * c2.sqrt();

    // Discount factor
    let df = (-risk_free_rate * tau).exp();

    // Payoff cosine coefficients
    let u_k = payoff_cos_coeffs(n, a, b, strike, opt_type);

    // Sum the COS series
    let mut price = 0.0_f64;
    for k in 0..n {
        let freq = k as f64 * PI / (b - a);
        let cf = heston_log_cf(freq, tau, v0, kappa, theta, sigma, rho, risk_free_rate, risk_free_rate - dividend_yield, log_fk);
        // Multiply CF by e^{-i*k*pi*a/(b-a)}
        let phase_arg = -(k as f64) * PI * a / (b - a);
        let phase = C::new(0.0, phase_arg).exp();
        let re_cf_phase = cf.mul(phase).conj_re();
        let weight = if k == 0 { 0.5 } else { 1.0 };
        price += weight * re_cf_phase * u_k[k];
    }
    price *= df;

    Ok(CosHestonResult {
        price: price.max(0.0),
        n_terms: n,
        ab: (a, b),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_model() -> HestonModel {
        // s0=100, r=0.05, q=0, v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.7
        HestonModel::new(100.0, 0.05, 0.0, 0.04, 1.5, 0.04, 0.3, -0.7)
    }

    #[test]
    fn cos_call_price_positive() {
        let model = make_model();
        let res = cos_heston_price(&model, 100.0, 100.0, 1.0, 0.05, 0.0, OptionType::Call, 128, 12.0)
            .unwrap();
        assert!(res.price > 0.0 && res.price < 100.0, "price={}", res.price);
    }

    #[test]
    fn cos_put_call_parity() {
        let model = make_model();
        let s = 100.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let q = 0.0;
        let call = cos_heston_price(&model, s, k, t, r, q, OptionType::Call, 128, 12.0).unwrap();
        let put = cos_heston_price(&model, s, k, t, r, q, OptionType::Put, 128, 12.0).unwrap();
        // Put-call parity: C - P = S*e^{-qT} - K*e^{-rT}
        let fwd = s * (-q * t).exp() - k * (-r * t).exp();
        assert!((call.price - put.price - fwd).abs() < 0.1, "pcp err={}", (call.price - put.price - fwd).abs());
    }

    #[test]
    fn cos_otm_call_less_than_forward() {
        let model = make_model();
        let res = cos_heston_price(&model, 100.0, 120.0, 1.0, 0.05, 0.0, OptionType::Call, 128, 12.0)
            .unwrap();
        assert!(res.price < 10.0);
    }

    #[test]
    fn cos_n_terms_default() {
        let model = make_model();
        let res = cos_heston_price(&model, 100.0, 100.0, 1.0, 0.05, 0.0, OptionType::Call, 0, 0.0)
            .unwrap();
        assert_eq!(res.n_terms, 128);
    }
}
