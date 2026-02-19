//! Analytic Heston pricing engine.
//!
//! Prices European options under the Heston stochastic volatility model
//! using the characteristic function approach with Gauss-Legendre integration.
//!
//! Uses the Gatheral (2006) log-strike formulation with the Albrecher et al.
//! (2007) rotation-count correction for numerical stability.

use std::f64::consts::PI;

use ql_math::integration::{GaussLegendreIntegral, Integrator};
use ql_models::HestonModel;
use tracing::info_span;

// ---------------------------------------------------------------------------
// Minimal complex number helper (avoids pulling in num-complex)
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
// Heston characteristic function
// ---------------------------------------------------------------------------

/// Compute the Heston log-characteristic function for Pⱼ (j=1,2).
///
/// Uses the Albrecher et al. (2007) formulation (avoids the "little Heston trap"):
///
///   d  = sqrt( (bⱼ − ρσiφ)² + σ²(2uⱼiφ + φ²) )
///   rⱼ⁻ = bⱼ − ρσiφ − d
///   g  = rⱼ⁻ / (bⱼ − ρσiφ + d)
///   D  = (rⱼ⁻ / σ²) · (1 − e^{−dτ}) / (1 − g·e^{−dτ})
///   C  = (κθ/σ²) · [rⱼ⁻·τ − 2·ln((1 − g·e^{−dτ})/(1−g))]
///
/// with b₁ = κ − ρσ, u₁ = ½;  b₂ = κ, u₂ = −½.
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

    // bⱼ − ρσiφ
    let bj_minus = C::new(b_j, -rho * sigma * phi);

    // d² = (bⱼ − ρσiφ)² + σ²(φ² − 2uⱼiφ)
    //
    // This follows QuantLib's convention (Heston 1993, Gatheral formulation):
    //   for j=1 (u=0.5):  σ²(φ² − iφ)
    //   for j=2 (u=-0.5): σ²(φ² + iφ)
    // Note the sign: it's MINUS 2uⱼiφ, not plus.
    let d_sq = bj_minus * bj_minus + C::from_real(s2) * C::new(phi * phi, -2.0 * u_j * phi);
    let d = d_sq.sqrt();

    // rⱼ⁻ = bⱼ − ρσiφ − d
    let r_minus = bj_minus - d;

    // g = rⱼ⁻ / rⱼ⁺ = (bⱼ − ρσiφ − d) / (bⱼ − ρσiφ + d)
    let g = r_minus / (bj_minus + d);

    // e^{−dτ}
    let one = C::from_real(1.0);
    let exp_neg_dt = (-d * tau).exp();

    // D = (rⱼ⁻ / σ²) · (1 − e^{−dτ}) / (1 − g·e^{−dτ})
    let one_minus_gexp = one - g * exp_neg_dt;
    let big_d = (r_minus * (1.0 / s2)) * ((one - exp_neg_dt) / one_minus_gexp);

    // C = (κθ/σ²) · [rⱼ⁻·τ − 2·ln((1 − g·e^{−dτ})/(1−g))]
    let log_ratio = (one_minus_gexp / (one - g)).ln();
    let big_c = (r_minus * tau - log_ratio * 2.0) * (kappa * theta / s2);

    // f_j = exp(C + D·v₀)
    (big_c + big_d * v0).exp()
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Results from the Heston analytic engine.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HestonResult {
    /// Net present value of the option.
    pub npv: f64,
    /// P1 probability (delta-related).
    pub p1: f64,
    /// P2 probability (exercise probability).
    pub p2: f64,
}

/// Price a European option under the Heston model.
///
/// Uses the characteristic function representation:
///   C = S e^{−qT} P₁ − K e^{−rT} P₂
///   P = K e^{−rT}(1−P₂) − S e^{−qT}(1−P₁)
///
/// where P₁ and P₂ are computed via Fourier inversion.
pub fn heston_price(
    model: &HestonModel,
    strike: f64,
    time_to_expiry: f64,
    is_call: bool,
) -> HestonResult {
    let _span = info_span!("heston_price", strike, time_to_expiry, is_call).entered();
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

    let df = (-r * tau).exp();
    let fwd = s * ((r - q) * tau).exp();
    let x = (fwd / k).ln(); // log-moneyness

    // P_j = ½ + (1/π) ∫₀^∞ Re[ e^{iφx} f_j(φ) / (iφ) ] dφ
    //
    // Let f_j = a + bi, e^{iφx} = cos(φx) + i·sin(φx)
    // f_j · e^{iφx} = (a·cos − b·sin) + i·(a·sin + b·cos)
    // Divide by iφ:  multiply numerator by −i/φ:
    //   Re = (a·sin(φx) + b·cos(φx)) / φ
    let make_integrand = |j: u8| {
        move |phi: f64| -> f64 {
            if phi < 1e-8 {
                return 0.0;
            }
            let f = heston_cf(phi, v0, kappa, theta, sigma, rho, tau, j);
            let cos_px = (phi * x).cos();
            let sin_px = (phi * x).sin();
            (f.re * sin_px + f.im * cos_px) / phi
        }
    };

    let integrator = GaussLegendreIntegral::new(128).expect("GL128");
    let upper = 200.0;

    let i1 = integrator
        .integrate(make_integrand(1), 1e-8, upper)
        .unwrap_or(0.0);
    let i2 = integrator
        .integrate(make_integrand(2), 1e-8, upper)
        .unwrap_or(0.0);

    let p1 = 0.5 + i1 / PI;
    let p2 = 0.5 + i2 / PI;

    let call = fwd * df * p1 - k * df * p2;

    let npv = if is_call {
        call
    } else {
        // Put-call parity
        call - s * (-q * tau).exp() + k * df
    };

    HestonResult {
        npv: npv.max(0.0),
        p1,
        p2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_model() -> HestonModel {
        // Standard test parameters
        HestonModel::new(
            100.0, // s0
            0.05,  // r
            0.0,   // q (no dividends for simpler validation)
            0.04,  // v0 (σ_impl ≈ 20%)
            1.5,   // kappa
            0.04,  // theta
            0.3,   // sigma (vol of vol)
            -0.7,  // rho
        )
    }

    #[test]
    fn heston_call_positive() {
        let model = make_model();
        let result = heston_price(&model, 100.0, 1.0, true);
        assert!(
            result.npv > 0.0,
            "Call price should be positive, got {}",
            result.npv
        );
        // ATM call under ~20% vol, 1yr should be roughly 8-14
        assert!(
            result.npv > 5.0 && result.npv < 20.0,
            "ATM call price {} not in expected range",
            result.npv
        );
    }

    #[test]
    fn heston_put_positive() {
        let model = make_model();
        let result = heston_price(&model, 100.0, 1.0, false);
        assert!(
            result.npv > 0.0,
            "Put price should be positive, got {}",
            result.npv
        );
    }

    #[test]
    fn heston_put_call_parity() {
        let model = make_model();
        let tau = 1.0;
        let k = 100.0;
        let call = heston_price(&model, k, tau, true);
        let put = heston_price(&model, k, tau, false);

        let s = model.spot();
        let r = model.risk_free_rate();
        let q = model.dividend_yield();

        // C - P = S e^{-qT} - K e^{-rT}
        let lhs = call.npv - put.npv;
        let rhs = s * (-q * tau).exp() - k * (-r * tau).exp();
        assert_abs_diff_eq!(lhs, rhs, epsilon = 0.05);
    }

    #[test]
    fn heston_otm_call_cheaper() {
        let model = make_model();
        let atm = heston_price(&model, 100.0, 1.0, true);
        let otm = heston_price(&model, 120.0, 1.0, true);
        assert!(
            otm.npv < atm.npv,
            "OTM call {} should be cheaper than ATM {}",
            otm.npv,
            atm.npv
        );
    }

    #[test]
    fn heston_longer_maturity_more_expensive() {
        let model = make_model();
        let short = heston_price(&model, 100.0, 0.25, true);
        let long = heston_price(&model, 100.0, 1.0, true);
        assert!(
            long.npv > short.npv,
            "Longer maturity ATM call {} should > short {}",
            long.npv,
            short.npv
        );
    }

    #[test]
    fn heston_p1_p2_in_range() {
        let model = make_model();
        let result = heston_price(&model, 100.0, 1.0, true);
        assert!(
            result.p1 > 0.0 && result.p1 < 1.0,
            "P1 out of range: {}",
            result.p1
        );
        assert!(
            result.p2 > 0.0 && result.p2 < 1.0,
            "P2 out of range: {}",
            result.p2
        );
    }

    #[test]
    fn heston_zero_vol_of_vol_approaches_bs() {
        // With σ → 0, Heston → Black-Scholes
        let model = HestonModel::new(
            100.0, 0.05, 0.0,
            0.04, // v0 → σ=20%
            1.5, 0.04,
            0.001, // very small vol-of-vol
            0.0,   // zero correlation for cleaner comparison
        );
        let heston = heston_price(&model, 100.0, 1.0, true);
        // BS price with σ=20%, S=100, K=100, r=5%, T=1: ~10.45
        assert_abs_diff_eq!(heston.npv, 10.45, epsilon = 0.5);
    }

    #[test]
    fn heston_deep_itm_call_near_intrinsic() {
        let model = make_model();
        let result = heston_price(&model, 60.0, 1.0, true);
        let intrinsic = 100.0 - 60.0 * (-0.05_f64).exp();
        assert!(
            result.npv > intrinsic * 0.95,
            "Deep ITM call {} should be near intrinsic {}",
            result.npv,
            intrinsic
        );
    }
}
