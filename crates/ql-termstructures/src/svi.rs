#![allow(clippy::too_many_arguments)]
//! SVI (Stochastic Volatility Inspired) smile parameterization.
//!
//! The SVI raw parameterization of total implied variance:
//!   w(k) = a + b [ρ(k − m) + √((k − m)² + σ²)]
//! where k = ln(K/F) is log-moneyness.
//!
//! This produces a symmetric (when ρ=0) or skewed smile that is
//! arbitrage-free under certain parameter constraints.
//!
//! Reference: Gatheral (2004), "A parsimonious arbitrage-free implied
//! volatility parameterization with application to the valuation of
//! volatility derivatives."

/// Compute total implied variance w(k) under the SVI raw parameterization.
///
/// w(k) = a + b [ρ(k − m) + √((k − m)² + σ²)]
///
/// # Parameters
/// - `k` — log-moneyness ln(K/F)
/// - `a` — overall variance level
/// - `b` — slope (b > 0)
/// - `rho` — rotation parameter ∈ (−1, 1)
/// - `m` — translation (center of smile)
/// - `sigma` — ATM curvature (σ > 0)
pub fn svi_total_variance(k: f64, a: f64, b: f64, rho: f64, m: f64, sigma: f64) -> f64 {
    let dk = k - m;
    a + b * (rho * dk + (dk * dk + sigma * sigma).sqrt())
}

/// Compute Black implied volatility under SVI.
///
/// σ_BS(k, T) = √(w(k) / T)
///
/// # Parameters
/// - `strike` — option strike
/// - `forward` — forward price
/// - `expiry` — time to expiry
/// - `a`, `b`, `rho`, `m`, `sigma` — SVI raw parameters
pub fn svi_volatility(
    strike: f64,
    forward: f64,
    expiry: f64,
    a: f64,
    b: f64,
    rho: f64,
    m: f64,
    sigma: f64,
) -> f64 {
    let k = (strike / forward).ln();
    let w = svi_total_variance(k, a, b, rho, m, sigma);
    if w <= 0.0 || expiry <= 0.0 {
        return 0.0;
    }
    (w / expiry).sqrt()
}

/// SVI smile section for a single expiry.
///
/// Stores the SVI raw parameters and provides volatility/variance
/// queries at arbitrary strikes.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SviSmileSection {
    pub forward: f64,
    pub expiry: f64,
    pub a: f64,
    pub b: f64,
    pub rho: f64,
    pub m: f64,
    pub sigma: f64,
}

impl SviSmileSection {
    /// Create a new SVI smile section.
    pub fn new(
        forward: f64,
        expiry: f64,
        a: f64,
        b: f64,
        rho: f64,
        m: f64,
        sigma: f64,
    ) -> Self {
        Self {
            forward,
            expiry,
            a,
            b,
            rho,
            m,
            sigma,
        }
    }

    /// Black implied volatility at the given strike.
    pub fn volatility(&self, strike: f64) -> f64 {
        svi_volatility(
            strike,
            self.forward,
            self.expiry,
            self.a,
            self.b,
            self.rho,
            self.m,
            self.sigma,
        )
    }

    /// Total implied variance at the given strike.
    pub fn variance(&self, strike: f64) -> f64 {
        let k = (strike / self.forward).ln();
        svi_total_variance(k, self.a, self.b, self.rho, self.m, self.sigma)
    }

    /// ATM implied volatility.
    pub fn atm_vol(&self) -> f64 {
        self.volatility(self.forward)
    }

    /// Check the no-static-arbitrage condition (Gatheral & Jacquier 2014).
    ///
    /// The SVI parameterization is free of static arbitrage if:
    /// 1. a + b σ √(1 − ρ²) ≥ 0 (ensures w(k) ≥ 0 for all k)
    /// 2. b ≥ 0
    /// 3. |ρ| < 1
    /// 4. σ > 0
    /// 5. b(1 + |ρ|) ≤ 4 / T  (Roger Lee's moment formula bound)
    pub fn is_arbitrage_free(&self) -> bool {
        self.b >= 0.0
            && self.sigma > 0.0
            && self.rho.abs() < 1.0
            && self.a + self.b * self.sigma * (1.0 - self.rho * self.rho).sqrt() >= 0.0
            && self.b * (1.0 + self.rho.abs()) <= 4.0 / self.expiry
    }
}

/// Calibrate SVI parameters to a set of market implied volatilities.
///
/// Uses a simplified grid search + Nelder-Mead approach.
///
/// # Parameters
/// - `strikes` — option strikes
/// - `vols` — market implied volatilities
/// - `forward` — forward price
/// - `expiry` — time to expiry
///
/// # Returns
/// The calibrated SVI raw parameters (a, b, ρ, m, σ).
pub fn svi_calibrate(
    strikes: &[f64],
    vols: &[f64],
    forward: f64,
    expiry: f64,
) -> (f64, f64, f64, f64, f64) {
    assert_eq!(
        strikes.len(),
        vols.len(),
        "strikes and vols must have same length"
    );

    // Target: total variance
    let market_w: Vec<f64> = vols.iter().map(|&v| v * v * expiry).collect();
    let k_vals: Vec<f64> = strikes.iter().map(|&s| (s / forward).ln()).collect();

    // Initial guess
    let atm_var = vols
        .iter()
        .zip(strikes.iter())
        .min_by(|(_, s1), (_, s2)| {
            ((*s1 - forward).abs())
                .partial_cmp(&((*s2 - forward).abs()))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(v, _)| v * v * expiry)
        .unwrap_or(0.04);

    let mut best_params = (atm_var, 0.1, 0.0, 0.0, 0.1);
    let mut best_err = f64::MAX;

    // Simple grid search for initial guess
    for &b in &[0.05, 0.1, 0.2, 0.5] {
        for &rho in &[-0.5, -0.25, 0.0, 0.25, 0.5] {
            for &sigma in &[0.05, 0.1, 0.2, 0.5] {
                let a = atm_var - b * sigma * (1.0_f64 - rho * rho).sqrt();
                let m = 0.0;

                let err: f64 = k_vals
                    .iter()
                    .zip(market_w.iter())
                    .map(|(&k, &mw)| {
                        let w = svi_total_variance(k, a, b, rho, m, sigma);
                        (w - mw) * (w - mw)
                    })
                    .sum();

                if err < best_err {
                    best_err = err;
                    best_params = (a, b, rho, m, sigma);
                }
            }
        }
    }

    // Refine with simplex-like perturbation
    let (mut a, mut b, mut rho, mut m_val, mut sigma) = best_params;
    let perturbations = [0.01, 0.005, 0.002, 0.001, 0.0005];

    for &pert in &perturbations {
        let candidates = [
            (a + pert, b, rho, m_val, sigma),
            (a - pert, b, rho, m_val, sigma),
            (a, b + pert, rho, m_val, sigma),
            (a, (b - pert).max(0.001), rho, m_val, sigma),
            (a, b, (rho + pert).min(0.99), m_val, sigma),
            (a, b, (rho - pert).max(-0.99), m_val, sigma),
            (a, b, rho, m_val + pert, sigma),
            (a, b, rho, m_val - pert, sigma),
            (a, b, rho, m_val, (sigma + pert).max(0.001)),
            (a, b, rho, m_val, (sigma - pert).max(0.001)),
        ];

        for &(ca, cb, cr, cm, cs) in &candidates {
            let err: f64 = k_vals
                .iter()
                .zip(market_w.iter())
                .map(|(&k, &mw)| {
                    let w = svi_total_variance(k, ca, cb, cr, cm, cs);
                    (w - mw) * (w - mw)
                })
                .sum();

            if err < best_err {
                best_err = err;
                a = ca;
                b = cb;
                rho = cr;
                m_val = cm;
                sigma = cs;
            }
        }
    }

    (a, b, rho, m_val, sigma)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_section() -> SviSmileSection {
        // a=0.04, b=0.2, rho=-0.3, m=0.0, sigma=0.1, F=100, T=1
        SviSmileSection::new(100.0, 1.0, 0.04, 0.2, -0.3, 0.0, 0.1)
    }

    #[test]
    fn svi_atm_variance() {
        let s = make_section();
        // At k=0: w(0) = a + b*(ρ*0 + √(0 + σ²)) = a + b*σ
        let expected = 0.04 + 0.2 * 0.1;
        assert_abs_diff_eq!(s.variance(100.0), expected, epsilon = 1e-12);
    }

    #[test]
    fn svi_atm_vol() {
        let s = make_section();
        // σ_ATM = √(w(0)/T) = √(0.06)
        let expected = 0.06_f64.sqrt();
        assert_abs_diff_eq!(s.atm_vol(), expected, epsilon = 1e-10);
    }

    #[test]
    fn svi_symmetric_when_rho_zero() {
        let s = SviSmileSection::new(100.0, 1.0, 0.04, 0.2, 0.0, 0.0, 0.1);
        let v_otm_call = s.volatility(110.0);
        let v_otm_put = s.volatility(100.0 * 100.0 / 110.0); // reciprocal moneyness
        assert_abs_diff_eq!(v_otm_call, v_otm_put, epsilon = 1e-10);
    }

    #[test]
    fn svi_vol_positive() {
        let s = make_section();
        for k in [80.0, 90.0, 100.0, 110.0, 120.0] {
            let v = s.volatility(k);
            assert!(v > 0.0, "Vol should be positive at K={k}: {v}");
        }
    }

    #[test]
    fn svi_smile_curvature() {
        // With b > 0, OTM options should have higher variance than ATM
        let s = SviSmileSection::new(100.0, 1.0, 0.04, 0.3, 0.0, 0.0, 0.1);
        let v_atm = s.variance(100.0);
        let v_otm = s.variance(130.0);
        assert!(
            v_otm > v_atm,
            "OTM variance should exceed ATM: otm={v_otm}, atm={v_atm}"
        );
    }

    #[test]
    fn svi_arbitrage_free_check() {
        let good = SviSmileSection::new(100.0, 1.0, 0.02, 0.1, -0.3, 0.0, 0.2);
        assert!(good.is_arbitrage_free());

        // b*(1+|rho|) > 4/T should fail
        let bad = SviSmileSection::new(100.0, 1.0, 0.02, 5.0, 0.5, 0.0, 0.2);
        assert!(!bad.is_arbitrage_free());
    }

    #[test]
    fn svi_calibrate_recovers_params() {
        // Generate market data from known SVI parameters
        let true_params = (0.04, 0.2, -0.2, 0.0, 0.15);
        let forward = 100.0;
        let expiry = 1.0;

        let strikes = vec![80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0];
        let vols: Vec<f64> = strikes
            .iter()
            .map(|&s| {
                svi_volatility(
                    s,
                    forward,
                    expiry,
                    true_params.0,
                    true_params.1,
                    true_params.2,
                    true_params.3,
                    true_params.4,
                )
            })
            .collect();

        let (a, b, _rho, _m, _sigma) = svi_calibrate(&strikes, &vols, forward, expiry);

        // Calibrated should reproduce market vols reasonably well
        let max_err: f64 = strikes
            .iter()
            .zip(vols.iter())
            .map(|(&s, &mv)| {
                let cv = svi_volatility(s, forward, expiry, a, b, _rho, _m, _sigma);
                (cv - mv).abs()
            })
            .fold(0.0, f64::max);

        assert!(
            max_err < 0.01,
            "Calibration max error should be small: {max_err}"
        );
    }

    #[test]
    fn svi_total_variance_function() {
        // Direct function test
        let w = svi_total_variance(0.0, 0.04, 0.2, 0.0, 0.0, 0.1);
        assert_abs_diff_eq!(w, 0.04 + 0.2 * 0.1, epsilon = 1e-12);
    }
}
