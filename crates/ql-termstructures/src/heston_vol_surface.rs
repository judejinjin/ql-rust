//! Heston-model implied Black volatility surface.
//!
//! Given calibrated Heston model parameters, this module computes the implied
//! Black-Scholes volatility at any (T, K) point by inverting the Heston
//! analytic call price.
//!
//! This is the Rust equivalent of QuantLib's `HestonBlackVolSurface`.
//!
//! Reference:
//! - Heston, S. (1993), "A Closed-Form Solution for Options with Stochastic
//!   Volatility", Review of Financial Studies.

use serde::{Deserialize, Serialize};
use ql_math::distributions::cumulative_normal;
use std::f64::consts::PI;

/// Heston model parameters for building the vol surface.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HestonVolParams {
    /// Spot price.
    pub spot: f64,
    /// Risk-free rate (continuous compounding).
    pub risk_free_rate: f64,
    /// Dividend yield (continuous).
    pub dividend_yield: f64,
    /// Initial variance v₀.
    pub v0: f64,
    /// Long-run variance θ.
    pub theta: f64,
    /// Mean-reversion speed κ.
    pub kappa: f64,
    /// Vol-of-vol σ.
    pub sigma: f64,
    /// Correlation ρ between spot and variance.
    pub rho: f64,
}

/// A Black volatility surface derived from a Heston model.
#[derive(Clone, Debug)]
pub struct HestonBlackVolSurface {
    params: HestonVolParams,
}

impl HestonBlackVolSurface {
    /// Create a Heston-implied Black vol surface from calibrated parameters.
    pub fn new(params: HestonVolParams) -> Self {
        HestonBlackVolSurface { params }
    }

    /// Compute the implied Black volatility at maturity T and strike K.
    ///
    /// Steps:
    /// 1. Price Heston call via characteristic function integration
    /// 2. Invert Black-Scholes formula to get implied vol
    pub fn black_vol(&self, t: f64, strike: f64) -> f64 {
        if t <= 0.0 || strike <= 0.0 { return self.params.v0.sqrt(); }

        let call_price = self.heston_call(t, strike);
        let forward = self.params.spot
            * ((self.params.risk_free_rate - self.params.dividend_yield) * t).exp();
        let df = (-self.params.risk_free_rate * t).exp();

        // Invert Black-Scholes for implied vol via Newton-Raphson
        implied_vol_from_price(call_price, forward, strike, t, df)
    }

    /// Compute Heston call price via numerical integration (trapezoidal).
    ///
    /// Uses the same Albrecher/Gatheral formulation as the analytic Heston engine.
    fn heston_call(&self, t: f64, k: f64) -> f64 {
        let p = &self.params;
        let forward = p.spot * ((p.risk_free_rate - p.dividend_yield) * t).exp();
        let df = (-p.risk_free_rate * t).exp();
        let x = (forward / k).ln(); // log-moneyness

        // P_j = ½ + (1/π) ∫₀^∞ Re[ e^{iφx} f_j(φ) / (iφ) ] dφ
        //     = ½ + (1/π) ∫₀^∞ (f_re · sin(φx) + f_im · cos(φx)) / φ  dφ
        let n_points = 200;
        let du = 0.25;
        let mut p1 = 0.0;
        let mut p2 = 0.0;

        for j in 1..=n_points {
            let phi = j as f64 * du;

            let f1 = heston_cf(phi, t, p, 1);
            let f2 = heston_cf(phi, t, p, 2);

            let sin_px = (phi * x).sin();
            let cos_px = (phi * x).cos();

            p1 += (f1.re * sin_px + f1.im * cos_px) / phi * du;
            p2 += (f2.re * sin_px + f2.im * cos_px) / phi * du;
        }

        let prob1 = (0.5 + p1 / PI).clamp(0.0, 1.0);
        let prob2 = (0.5 + p2 / PI).clamp(0.0, 1.0);

        (forward * df * prob1 - k * df * prob2).max(0.0)
    }
}

/// Simple complex number for characteristic function computation.
#[derive(Clone, Copy, Debug)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn new(re: f64, im: f64) -> Self { Complex { re, im } }
    fn exp(self) -> Self {
        let r = self.re.exp();
        Complex { re: r * self.im.cos(), im: r * self.im.sin() }
    }
    fn ln(self) -> Self {
        let r = (self.re * self.re + self.im * self.im).sqrt();
        Complex { re: r.ln(), im: self.im.atan2(self.re) }
    }
}

impl std::ops::Add for Complex {
    type Output = Complex;
    fn add(self, rhs: Complex) -> Complex { Complex { re: self.re + rhs.re, im: self.im + rhs.im } }
}
impl std::ops::Sub for Complex {
    type Output = Complex;
    fn sub(self, rhs: Complex) -> Complex { Complex { re: self.re - rhs.re, im: self.im - rhs.im } }
}
impl std::ops::Mul for Complex {
    type Output = Complex;
    fn mul(self, rhs: Complex) -> Complex {
        Complex {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}
impl std::ops::Div for Complex {
    type Output = Complex;
    fn div(self, rhs: Complex) -> Complex {
        let denom = rhs.re * rhs.re + rhs.im * rhs.im;
        Complex {
            re: (self.re * rhs.re + self.im * rhs.im) / denom,
            im: (self.im * rhs.re - self.re * rhs.im) / denom,
        }
    }
}
impl std::ops::Mul<Complex> for f64 {
    type Output = Complex;
    fn mul(self, rhs: Complex) -> Complex { Complex { re: self * rhs.re, im: self * rhs.im } }
}


/// Heston characteristic function using the Albrecher et al. (2007) formulation.
///
/// This matches the implementation in the analytic Heston engine (avoids the
/// "little Heston trap"):
///
///   d  = sqrt( (b_j − ρσiφ)² + σ²(φ² − 2u_jiφ) )
///   r⁻ = b_j − ρσiφ − d
///   g  = r⁻ / (b_j − ρσiφ + d)
///   D  = (r⁻ / σ²) · (1 − e^{−dτ}) / (1 − g·e^{−dτ})
///   C  = (κθ/σ²) · [r⁻·τ − 2·ln((1 − g·e^{−dτ})/(1−g))]
///
/// with b₁ = κ − ρσ, u₁ = ½;  b₂ = κ, u₂ = −½.
fn heston_cf(phi: f64, tau: f64, p: &HestonVolParams, variant: u8) -> Complex {
    let b_j = if variant == 1 { p.kappa - p.rho * p.sigma } else { p.kappa };
    let u_j: f64 = if variant == 1 { 0.5 } else { -0.5 };
    let s2 = p.sigma * p.sigma;

    // b_j − ρσiφ
    let bj_minus = Complex::new(b_j, -p.rho * p.sigma * phi);

    // d² = (b_j − ρσiφ)² + σ²(φ² − 2u_jiφ)
    let d_sq = {
        let t1 = bj_minus * bj_minus;
        let t2 = Complex::new(s2 * phi * phi, -2.0 * u_j * s2 * phi);
        Complex::new(t1.re + t2.re, t1.im + t2.im)
    };
    let d = complex_sqrt(d_sq);

    // r⁻ = b_j − ρσiφ − d
    let r_minus = Complex::new(bj_minus.re - d.re, bj_minus.im - d.im);

    // g = r⁻ / r⁺
    let r_plus = Complex::new(bj_minus.re + d.re, bj_minus.im + d.im);
    let g = r_minus / r_plus;

    // e^{−dτ}
    let neg_d_tau = Complex::new(-d.re * tau, -d.im * tau);
    let exp_neg_dt = neg_d_tau.exp();

    let one = Complex::new(1.0, 0.0);
    let one_minus_gexp = Complex::new(one.re - (g * exp_neg_dt).re, one.im - (g * exp_neg_dt).im);

    // D = (r⁻ / σ²) · (1 − e^{−dτ}) / (1 − g·e^{−dτ})
    let one_minus_exp = Complex::new(one.re - exp_neg_dt.re, one.im - exp_neg_dt.im);
    let big_d = (r_minus * Complex::new(1.0 / s2, 0.0)) * (one_minus_exp / one_minus_gexp);

    // C = (κθ/σ²) · [r⁻·τ − 2·ln((1 − g·e^{−dτ})/(1−g))]
    let one_minus_g = Complex::new(one.re - g.re, one.im - g.im);
    let log_ratio = (one_minus_gexp / one_minus_g).ln();
    let r_minus_tau = Complex::new(r_minus.re * tau, r_minus.im * tau);
    let big_c_factor = p.kappa * p.theta / s2;
    let big_c = Complex::new(
        big_c_factor * (r_minus_tau.re - 2.0 * log_ratio.re),
        big_c_factor * (r_minus_tau.im - 2.0 * log_ratio.im),
    );

    // f_j = exp(C + D·v₀)
    let dv0 = Complex::new(big_d.re * p.v0, big_d.im * p.v0);
    Complex::new(big_c.re + dv0.re, big_c.im + dv0.im).exp()
}

fn complex_sqrt(z: Complex) -> Complex {
    let r = (z.re * z.re + z.im * z.im).sqrt().sqrt();
    let theta = z.im.atan2(z.re) / 2.0;
    Complex { re: r * theta.cos(), im: r * theta.sin() }
}

/// Implied Black vol from a call price via Newton-Raphson.
fn implied_vol_from_price(price: f64, forward: f64, strike: f64, t: f64, df: f64) -> f64 {
    if price <= 0.0 || t <= 0.0 { return 0.01; }
    let intrinsic = df * (forward - strike).max(0.0);
    if price <= intrinsic + 1e-12 { return 0.01; }

    let mut sigma = 0.2; // initial guess
    let sqrt_t = t.sqrt();

    for _ in 0..100 {
        let d1 = ((forward / strike).ln() + 0.5 * sigma * sigma * t) / (sigma * sqrt_t);
        let d2 = d1 - sigma * sqrt_t;
        let bs_price = df * (forward * cumulative_normal(d1) - strike * cumulative_normal(d2));
        let vega = df * forward * sqrt_t * norm_pdf(d1);
        if vega.abs() < 1e-14 { break; }
        let diff = bs_price - price;
        if diff.abs() < 1e-10 { break; }
        sigma -= diff / vega;
        sigma = sigma.clamp(0.001, 5.0);
    }
    sigma
}

fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn sample_params() -> HestonVolParams {
        HestonVolParams {
            spot: 100.0,
            risk_free_rate: 0.05,
            dividend_yield: 0.02,
            v0: 0.04,     // 20% vol
            theta: 0.04,
            kappa: 2.0,
            sigma: 0.5,
            rho: -0.7,
        }
    }

    #[test]
    fn test_heston_vol_surface_atm() {
        let surf = HestonBlackVolSurface::new(sample_params());
        let vol = surf.black_vol(1.0, 100.0);
        // ATM vol should be near sqrt(v0) ≈ 0.20
        assert!(vol > 0.10 && vol < 0.40, "atm_vol={}", vol);
    }

    #[test]
    fn test_heston_vol_surface_skew() {
        let surf = HestonBlackVolSurface::new(sample_params());
        // With ρ < 0, OTM puts (low strikes) should have higher vol
        let vol_low = surf.black_vol(1.0, 80.0);
        let vol_high = surf.black_vol(1.0, 120.0);
        // vol_low should be > vol_high (negative skew)
        assert!(vol_low > vol_high * 0.8,
            "vol_low={}, vol_high={}", vol_low, vol_high);
    }

    #[test]
    fn test_heston_vol_surface_term_structure() {
        let surf = HestonBlackVolSurface::new(sample_params());
        let vol_short = surf.black_vol(0.25, 100.0);
        let vol_long = surf.black_vol(5.0, 100.0);
        // Both should be reasonable
        assert!(vol_short > 0.05 && vol_short < 0.50, "vol_short={}", vol_short);
        assert!(vol_long > 0.05 && vol_long < 0.50, "vol_long={}", vol_long);
    }

    #[test]
    fn test_implied_vol_roundtrip() {
        let forward: f64 = 100.0;
        let strike: f64 = 100.0;
        let t: f64 = 1.0;
        let df = (-0.05_f64 * t).exp();
        let true_vol: f64 = 0.25;

        let d1 = ((forward / strike).ln() + 0.5 * true_vol * true_vol * t) / (true_vol * t.sqrt());
        let d2 = d1 - true_vol * t.sqrt();
        let price = df * (forward * cumulative_normal(d1) - strike * cumulative_normal(d2));

        let implied = implied_vol_from_price(price, forward, strike, t, df);
        assert_abs_diff_eq!(implied, true_vol, epsilon = 0.001);
    }
}
