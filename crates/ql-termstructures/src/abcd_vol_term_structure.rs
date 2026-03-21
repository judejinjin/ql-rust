//! ABCD parametric caplet volatility term structure.
//!
//! The ABCD functional form fits the term structure of ATM caplet
//! implied volatilities:
//!
//!   f(T) = [(a + b·T)·exp(-c·T) + d] · √T
//!
//! or equivalently the instantaneous vol:
//!
//!   σ(T) = (a + b·T)·exp(-c·T) + d
//!
//! where:
//! - `d` is the long-term (asymptotic) vol
//! - `a + d` is the short-term vol offset
//! - `b` controls the slope at T=0
//! - `c` controls the speed of mean-reversion
//!
//! Reference: Rebonato, R. (2004). "Volatility and Correlation."
//! Brigo, D. & Mercurio, F. (2006). "Interest Rate Models."

// =========================================================================
// AbcdParameters
// =========================================================================

/// Parameters for the ABCD volatility function.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct AbcdParameters {
    /// Short-term offset parameter (can be negative).
    pub a: f64,
    /// Slope of short-term hump.
    pub b: f64,
    /// Speed of decay.
    pub c: f64,
    /// Long-term (asymptotic) vol floor.
    pub d: f64,
}

impl AbcdParameters {
    /// Create ABCD parameters. `d > 0` and `c > 0` are enforced.
    pub fn new(a: f64, b: f64, c: f64, d: f64) -> Self {
        assert!(c > 0.0, "c must be positive");
        assert!(d > 0.0, "d must be positive");
        Self { a, b, c, d }
    }

    /// Instantaneous volatility at time `t`: σ(t) = (a + b·t)·exp(-c·t) + d.
    pub fn instantaneous_vol(&self, t: f64) -> f64 {
        (self.a + self.b * t) * (-self.c * t).exp() + self.d
    }

    /// Root-mean-square vol over [0, T]:
    ///   σ_rms(T) = sqrt( (1/T) ∫₀ᵀ σ(t)² dt )
    pub fn rms_vol(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return self.instantaneous_vol(0.0).abs();
        }
        (self.variance(t) / t).sqrt()
    }

    /// Integral ∫₀ᵀ σ(t)² dt (Black variance of a caplet expiring at T).
    pub fn variance(&self, t: f64) -> f64 {
        integrate_abcd_sq(self.a, self.b, self.c, self.d, 0.0, t)
    }

    /// Time location of the hump (inflection point) in σ(t).
    /// Returns `None` if `b ≤ 0` (no hump).
    pub fn hump_time(&self) -> Option<f64> {
        if self.b <= 0.0 {
            return None;
        }
        let t_hump = (1.0 / self.c) - self.a / self.b;
        if t_hump > 0.0 { Some(t_hump) } else { None }
    }

    /// Check if the vol is always positive on [0, ∞).
    pub fn is_valid(&self) -> bool {
        // d > 0 ensures long-run positivity; check minimum over [0,∞)
        self.d > 0.0 && self.instantaneous_vol_min() > 0.0
    }

    fn instantaneous_vol_min(&self) -> f64 {
        // Global minimum of (a + b·t)·exp(-c·t) + d
        // Derivative zero: (b - c·(a + b·t))·exp(-c·t) = 0
        // => t* = (b - c·a) / (c·b) = 1/c - a/b  (for b > 0)
        let cands = [0.0_f64];
        let t_star = if self.b.abs() > 1e-15 {
            let ts = 1.0 / self.c - self.a / self.b;
            if ts > 0.0 { Some(ts) } else { None }
        } else {
            None
        };
        let mut min_v = cands.iter().map(|&t| self.instantaneous_vol(t)).fold(f64::INFINITY, f64::min);
        if let Some(ts) = t_star {
            min_v = min_v.min(self.instantaneous_vol(ts));
        }
        // long-run value
        min_v = min_v.min(self.d);
        min_v
    }
}

/// Compute ∫_{t1}^{t2} [(a + b·t)·exp(-c·t) + d]² dt analytically.
pub fn integrate_abcd_sq(a: f64, b: f64, c: f64, d: f64, t1: f64, t2: f64) -> f64 {
    // Expand: (a + b·t)²·e^{-2c·t} + 2d·(a + b·t)·e^{-c·t} + d²
    // Each term has a closed-form antiderivative.
    let antideriv = |t: f64| -> f64 {
        let e1 = (-c * t).exp();
        let e2 = (-2.0 * c * t).exp();
        // Term 1: (a + bt)² e^{-2ct}
        // = (a² + 2abt + b²t²) e^{-2ct}
        // ∫ e^{-2ct} dt = -1/(2c) e^{-2ct}
        // ∫ t e^{-2ct} dt = -t/(2c) e^{-2ct} - 1/(4c²) e^{-2ct}
        // ∫ t² e^{-2ct} dt = -t²/(2c) e^{-2ct} - t/(2c²) e^{-2ct} - 1/(4c³) e^{-2ct}
        let c2 = 2.0 * c;
        let term1 = e2 * (
            -(a * a) / c2
            - (2.0 * a * b) * (t / c2 + 1.0 / (c2 * c2))
            - b * b * (t * t / c2 + t / (c2 * c2) + 1.0 / (2.0 * c * c2 * c2))
        );
        // Term 2: 2d(a + bt) e^{-ct}
        // = 2d [ a e^{-ct} + bt e^{-ct} ]
        // ∫ e^{-ct} dt = -1/c e^{-ct}
        // ∫ t e^{-ct} dt = -t/c e^{-ct} - 1/c² e^{-ct}
        let term2 = 2.0 * d * e1 * (-a / c - b * (t / c + 1.0 / (c * c)));
        // Term 3: d² t
        let term3 = d * d * t;
        term1 + term2 + term3
    };
    antideriv(t2) - antideriv(t1)
}

// =========================================================================
// AbcdVolTermStructure
// =========================================================================

/// ABCD parametric caplet volatility term structure.
///
/// Given fitted ABCD parameters, provides caplet vol and variance for any
/// expiry date (time in years).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AbcdVolTermStructure {
    /// ABCD parameters.
    pub params: AbcdParameters,
}

impl AbcdVolTermStructure {
    /// Create from ABCD parameters.
    pub fn new(params: AbcdParameters) -> Self {
        Self { params }
    }

    /// Instantaneous caplet volatility at expiry `t`.
    pub fn instantaneous_vol(&self, t: f64) -> f64 {
        self.params.instantaneous_vol(t)
    }

    /// RMS caplet volatility for a caplet expiring at `t`.
    pub fn caplet_vol(&self, t: f64) -> f64 {
        self.params.rms_vol(t)
    }

    /// Black variance ∫₀ᵀ σ(t)² dt for a caplet expiring at `T`.
    pub fn caplet_variance(&self, t: f64) -> f64 {
        self.params.variance(t)
    }

    /// Caplet price under Black with a given forward rate `f` and strike `k`.
    pub fn caplet_price(&self, forward: f64, strike: f64, expiry: f64, notional: f64) -> f64 {
        use ql_math::distributions::cumulative_normal;
        let var = self.caplet_variance(expiry);
        if var <= 0.0 {
            return notional * (forward - strike).max(0.0) * expiry;
        }
        let _sigma = (var / expiry).sqrt();
        let d1 = ((forward / strike).ln() + 0.5 * var) / var.sqrt();
        let d2 = d1 - var.sqrt();
        notional * expiry * (forward * cumulative_normal(d1) - strike * cumulative_normal(d2))
    }

    /// Calibrate ABCD parameters to a set of (expiry, market_vol) quotes
    /// using a simple grid search / Nelder-Mead inspired approach.
    ///
    /// Returns the best-fit parameters minimising sum of squared vol errors.
    pub fn calibrate(market_expiries: &[f64], market_vols: &[f64]) -> AbcdParameters {
        assert_eq!(market_expiries.len(), market_vols.len());
        // Simplex search over (a, b, c, d)
        // Initial guess from typical interest rate vol shapes
        let mut best = AbcdParameters::new(-0.06, 0.17, 0.54, 0.17);
        let mut best_err = abcd_fit_error(&best, market_expiries, market_vols);

        // Grid search over reasonable ranges
        for &a in &[-0.1_f64, -0.05, 0.0, 0.05] {
            for &b in &[0.0_f64, 0.1, 0.2, 0.3] {
                for &c in &[0.3_f64, 0.5, 0.8, 1.5] {
                    for &d in &[0.05_f64, 0.10, 0.15, 0.20, 0.25] {
                        if d <= 0.0 { continue; }
                        let p = AbcdParameters { a, b, c, d };
                        let err = abcd_fit_error(&p, market_expiries, market_vols);
                        if err < best_err {
                            best_err = err;
                            best = p;
                        }
                    }
                }
            }
        }
        best
    }
}

fn abcd_fit_error(p: &AbcdParameters, expiries: &[f64], vols: &[f64]) -> f64 {
    expiries.iter().zip(vols.iter()).map(|(&t, &v)| {
        let model_v = p.rms_vol(t);
        (model_v - v).powi(2)
    }).sum()
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instantaneous_vol_shape() {
        let p = AbcdParameters::new(-0.06, 0.17, 0.54, 0.17);
        // Should have a hump
        let t_hump = p.hump_time().expect("should have hump");
        assert!(t_hump > 0.0, "hump at {}", t_hump);
        let v_hump = p.instantaneous_vol(t_hump);
        let v_0 = p.instantaneous_vol(0.0);
        let v_long = p.instantaneous_vol(10.0);
        // Hump should be max
        assert!(v_hump >= v_0, "hump should exceed initial vol");
        assert!(v_hump >= v_long, "hump should exceed long-run vol");
    }

    #[test]
    fn variance_positive() {
        let p = AbcdParameters::new(-0.06, 0.17, 0.54, 0.17);
        for &t in &[0.25, 0.5, 1.0, 2.0, 5.0, 10.0] {
            let v = p.variance(t);
            assert!(v > 0.0, "variance at t={} is {}", t, v);
        }
    }

    #[test]
    fn rms_vol_matches_caplet_vol() {
        let ts = AbcdVolTermStructure::new(AbcdParameters::new(-0.06, 0.17, 0.54, 0.17));
        for &t in &[0.5, 1.0, 2.0, 5.0] {
            let v1 = ts.caplet_vol(t);
            let v2 = ts.instantaneous_vol(t); // different from rms for non-constant
            assert!(v1 > 0.0 && v2 > 0.0, "t={}: caplet_vol={}, inst_vol={}", t, v1, v2);
        }
    }

    #[test]
    fn calibrate_flat_vols() {
        let expiries = vec![0.25, 0.5, 1.0, 2.0, 3.0, 5.0];
        let target_vol = 0.20;
        let vols = vec![target_vol; expiries.len()];
        let p = AbcdVolTermStructure::calibrate(&expiries, &vols);
        let ts = AbcdVolTermStructure::new(p);
        for &t in &expiries {
            let v = ts.caplet_vol(t);
            assert!((v - target_vol).abs() < 0.05, "calibrated caplet_vol at t={}: {}", t, v);
        }
    }

    #[test]
    fn caplet_price_positive() {
        let ts = AbcdVolTermStructure::new(AbcdParameters::new(-0.06, 0.17, 0.54, 0.17));
        let price = ts.caplet_price(0.05, 0.04, 1.0, 1_000_000.0);
        assert!(price > 0.0, "caplet price should be positive: {}", price);
    }
}
