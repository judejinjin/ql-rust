//! Analytic Bates pricing engine.
//!
//! Prices European options under the Bates model (Heston + jumps) using
//! the characteristic function approach.
//!
//! The Bates characteristic function extends Heston by adding a jump term:
//!
//!   ln f_Bates(φ) = ln f_Heston(φ) + λτ [e^{iφν − ½δ²φ²} − 1 − iφk̄]
//!
//! where k̄ = e^{ν + δ²/2} − 1 is the jump compensator.
//!
//! # References
//! - Bates, D.S. (1996), "Jumps and stochastic volatility: exchange rate
//!   processes implicit in Deutsche Mark options", *Review of Financial Studies* 9.

use std::f64::consts::PI;

use ql_math::integration::{GaussLegendreIntegral, Integrator};
use ql_models::BatesModel;

// ---------------------------------------------------------------------------
// Complex number helper (same as in analytic_heston)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct C {
    re: f64,
    im: f64,
}

impl C {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    fn from_real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }

    fn norm_sq(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    fn abs(self) -> f64 {
        self.norm_sq().sqrt()
    }

    fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    fn exp(self) -> Self {
        let r = self.re.exp();
        Self {
            re: r * self.im.cos(),
            im: r * self.im.sin(),
        }
    }

    fn ln(self) -> Self {
        Self {
            re: self.abs().ln(),
            im: self.arg(),
        }
    }

    fn sqrt(self) -> Self {
        let r = self.abs().sqrt();
        let theta = self.arg() / 2.0;
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }
}

impl std::ops::Add for C {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl std::ops::Sub for C {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl std::ops::Mul for C {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl std::ops::Mul<f64> for C {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

impl std::ops::Div for C {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let d = rhs.norm_sq();
        Self {
            re: (self.re * rhs.re + self.im * rhs.im) / d,
            im: (self.im * rhs.re - self.re * rhs.im) / d,
        }
    }
}

impl std::ops::Neg for C {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

// ---------------------------------------------------------------------------
// Heston characteristic function (reproduced here with Bates jump extension)
// ---------------------------------------------------------------------------

/// Heston log-characteristic function component (Albrecher formulation).
/// Returns the complex value of f_j(phi) for j=1 or j=2.
#[allow(clippy::too_many_arguments)]
fn heston_cf(
    phi: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    tau: f64,
    j: u8,
) -> C {
    let b_j: f64 = if j == 1 { kappa - rho * sigma } else { kappa };
    let u_j: f64 = if j == 1 { 0.5 } else { -0.5 };
    let s2 = sigma * sigma;

    let bj_minus = C::new(b_j, -rho * sigma * phi);
    let d_sq = bj_minus * bj_minus + C::from_real(s2) * C::new(phi * phi, -2.0 * u_j * phi);
    let d = d_sq.sqrt();

    let r_minus = bj_minus - d;
    let g = r_minus / (bj_minus + d);

    let one = C::from_real(1.0);
    let exp_neg_dt = (-d * tau).exp();

    let one_minus_gexp = one - g * exp_neg_dt;
    let big_d = (r_minus * (1.0 / s2)) * ((one - exp_neg_dt) / one_minus_gexp);

    let log_ratio = (one_minus_gexp / (one - g)).ln();
    let big_c = (r_minus * tau - log_ratio * 2.0) * (kappa * theta / s2);

    (big_c + big_d * v0).exp()
}

/// Bates characteristic function for Pⱼ.
///
/// f_Bates(φ) = f_Heston(φ) · exp(jump_term)
///
/// jump_term = λτ · [exp(iφν − ½δ²φ²) − 1 − iφk̄]
#[allow(clippy::too_many_arguments)]
fn bates_cf(
    phi: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    tau: f64,
    lambda: f64,
    nu: f64,
    delta: f64,
    j: u8,
) -> C {
    let f_heston = heston_cf(phi, v0, kappa, theta, sigma, rho, tau, j);

    // Jump compensator: k_bar = exp(nu + delta^2/2) - 1
    let k_bar = (nu + 0.5 * delta * delta).exp() - 1.0;

    // Jump CF term: exp(i*phi*nu - 0.5*delta^2*phi^2)
    let jump_cf_re = -0.5 * delta * delta * phi * phi;
    let jump_cf_im = phi * nu;
    let jump_cf = C::new(jump_cf_re, jump_cf_im).exp();

    // Full jump term: lambda * tau * (jump_cf - 1 - i*phi*k_bar)
    let one = C::from_real(1.0);
    let i_phi_kbar = C::new(0.0, phi * k_bar);
    let jump_term = (jump_cf - one - i_phi_kbar) * (lambda * tau);

    // f_Bates = f_Heston * exp(jump_term)
    f_heston * jump_term.exp()
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Results from the Bates analytic engine.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BatesResult {
    /// Net present value of the option.
    pub npv: f64,
    /// P1 probability (delta-related).
    pub p1: f64,
    /// P2 probability (exercise probability).
    pub p2: f64,
}

/// Price a European option under the Bates model.
///
/// Uses the characteristic function representation identical to Heston,
/// but with the Bates CF that includes jump contributions.
pub fn bates_price(
    model: &BatesModel,
    strike: f64,
    time_to_expiry: f64,
    is_call: bool,
) -> BatesResult {
    let s = model.spot();
    let k = strike;
    let tau = time_to_expiry;
    let r = model.risk_free_rate();
    let q = model.dividend_yield();
    let v0 = model.v0();
    let kappa = model.kappa();
    let theta = model.theta();
    let sigma = model.sigma();
    let rho = model.rho();
    let lambda = model.lambda();
    let nu = model.nu();
    let delta = model.delta();

    let df = (-r * tau).exp();
    let fwd = s * ((r - q) * tau).exp();
    let x = (fwd / k).ln();

    let make_integrand = |j: u8| {
        move |phi: f64| -> f64 {
            if phi < 1e-8 {
                return 0.0;
            }
            let f = bates_cf(phi, v0, kappa, theta, sigma, rho, tau, lambda, nu, delta, j);
            let cos_px = (phi * x).cos();
            let sin_px = (phi * x).sin();
            (f.re * sin_px + f.im * cos_px) / phi
        }
    };

    let integrator = GaussLegendreIntegral::new(128).expect("GL128");
    let upper = 200.0;

    let i1 = integrator.integrate(make_integrand(1), 1e-8, upper).unwrap_or(0.0);
    let i2 = integrator.integrate(make_integrand(2), 1e-8, upper).unwrap_or(0.0);

    let p1 = 0.5 + i1 / PI;
    let p2 = 0.5 + i2 / PI;

    let call = fwd * df * p1 - k * df * p2;

    let npv = if is_call {
        call
    } else {
        call - s * (-q * tau).exp() + k * df
    };

    BatesResult {
        npv: npv.max(0.0),
        p1,
        p2,
    }
}

/// Price a European option under the Bates model using flat parameters
/// (no model object needed).
#[allow(clippy::too_many_arguments)]
pub fn bates_price_flat(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    time_to_expiry: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    lambda: f64,
    nu: f64,
    delta: f64,
    is_call: bool,
) -> f64 {
    let model = BatesModel::new(spot, r, q, v0, kappa, theta, sigma, rho, lambda, nu, delta);
    bates_price(&model, strike, time_to_expiry, is_call).npv
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_models::HestonModel;

    fn make_bates() -> BatesModel {
        BatesModel::new(
            100.0, 0.05, 0.0, // s0, r, q
            0.04, 1.5, 0.04, 0.3, -0.7, // v0, κ, θ, σ, ρ
            0.5, -0.1, 0.15, // λ, ν, δ
        )
    }

    fn make_heston_equivalent() -> HestonModel {
        HestonModel::new(100.0, 0.05, 0.0, 0.04, 1.5, 0.04, 0.3, -0.7)
    }

    #[test]
    fn bates_reduces_to_heston_when_no_jumps() {
        // With λ=0, Bates should equal Heston
        let bates_model = BatesModel::new(
            100.0, 0.05, 0.0,
            0.04, 1.5, 0.04, 0.3, -0.7,
            0.0, 0.0, 0.01, // λ=0 means no jumps (delta nonzero but irrelevant)
        );
        let heston_model = make_heston_equivalent();

        let bates_call = bates_price(&bates_model, 100.0, 1.0, true);
        let heston_call = crate::heston_price(&heston_model, 100.0, 1.0, true);

        assert_abs_diff_eq!(bates_call.npv, heston_call.npv, epsilon = 1e-4);
    }

    #[test]
    fn bates_call_positive() {
        let m = make_bates();
        let res = bates_price(&m, 100.0, 1.0, true);
        assert!(res.npv > 0.0, "Bates call should be positive: {}", res.npv);
    }

    #[test]
    fn bates_put_call_parity() {
        let m = make_bates();
        let s = m.spot();
        let r = m.risk_free_rate();
        let q = m.dividend_yield();
        let k = 100.0;
        let t = 1.0;

        let call = bates_price(&m, k, t, true).npv;
        let put = bates_price(&m, k, t, false).npv;

        let parity = call - put - s * (-q * t).exp() + k * (-r * t).exp();
        assert_abs_diff_eq!(parity, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn bates_jumps_change_price_vs_heston() {
        // Jumps change option prices relative to Heston
        let bates_model = make_bates();
        let heston_model = make_heston_equivalent();

        let bates_call = bates_price(&bates_model, 100.0, 1.0, true).npv;
        let heston_call = crate::heston_price(&heston_model, 100.0, 1.0, true).npv;

        // They should differ when jumps are present
        assert!(
            (bates_call - heston_call).abs() > 0.1,
            "Bates call {} should differ from Heston call {} with jumps",
            bates_call,
            heston_call
        );
    }

    #[test]
    fn bates_probabilities_valid() {
        let m = make_bates();
        let res = bates_price(&m, 100.0, 1.0, true);
        assert!(res.p1 > 0.0 && res.p1 < 1.0, "P1={}", res.p1);
        assert!(res.p2 > 0.0 && res.p2 < 1.0, "P2={}", res.p2);
    }

    #[test]
    fn bates_flat_matches_model() {
        let m = make_bates();
        let model_price = bates_price(&m, 100.0, 1.0, true).npv;
        let flat_price = bates_price_flat(
            100.0, 100.0, 0.05, 0.0, 1.0,
            0.04, 1.5, 0.04, 0.3, -0.7,
            0.5, -0.1, 0.15, true,
        );
        assert_abs_diff_eq!(model_price, flat_price, epsilon = 1e-10);
    }

    #[test]
    fn bates_otm_put_small() {
        // Deep OTM put should be small
        let m = make_bates();
        let res = bates_price(&m, 60.0, 1.0, false);
        assert!(res.npv < 1.0, "Deep OTM Bates put should be small: {}", res.npv);
    }

    #[test]
    fn bates_itm_call_near_intrinsic() {
        // Deep ITM call should be near intrinsic + time value
        let m = make_bates();
        let res = bates_price(&m, 60.0, 1.0, true);
        let intrinsic = 100.0 - 60.0;
        assert!(
            res.npv > intrinsic * 0.9,
            "Deep ITM Bates call {} should be near intrinsic {}",
            res.npv,
            intrinsic
        );
    }
}
