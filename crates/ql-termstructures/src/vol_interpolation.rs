//! Volatility interpolation utilities.
//!
//! Provides:
//! - `BlackVarianceCurve`: ATM variance term structure (strike-independent)
//! - `VolatilitySmileSurface`: full strike×expiry surface using SmileSection objects
//! - Variance interpolation helpers

/// ATM Black variance curve: σ²(T) interpolated across expiry dates.
///
/// This is a strike-independent term structure suitable for
/// ATM swaption or cap vol quotes.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BlackVarianceCurve {
    /// Expiry dates (years).
    pub times: Vec<f64>,
    /// Total variances σ²T at each expiry.
    pub total_variances: Vec<f64>,
}

impl BlackVarianceCurve {
    /// Create from expiries and ATM Black vols.
    pub fn from_vols(times: &[f64], vols: &[f64]) -> Self {
        assert_eq!(times.len(), vols.len());
        let total_variances = times
            .iter()
            .zip(vols.iter())
            .map(|(&t, &v)| v * v * t)
            .collect();
        Self {
            times: times.to_vec(),
            total_variances,
        }
    }

    /// Create from expiries and total variances directly.
    pub fn from_total_variances(times: &[f64], total_variances: &[f64]) -> Self {
        assert_eq!(times.len(), total_variances.len());
        Self {
            times: times.to_vec(),
            total_variances: total_variances.to_vec(),
        }
    }

    /// Linearly interpolate total variance, then return implied vol.
    pub fn black_vol(&self, t: f64) -> f64 {
        let w = self.total_variance(t);
        if t <= 0.0 || w <= 0.0 {
            return 0.0;
        }
        (w / t).sqrt()
    }

    /// Linearly interpolate total variance w(t) = σ²t.
    ///
    /// Linear interpolation in total variance is the standard market convention
    /// because it preserves calendar-spread arbitrage freedom (w must be non-decreasing).
    pub fn total_variance(&self, t: f64) -> f64 {
        if self.times.is_empty() {
            return 0.0;
        }
        if t <= self.times[0] {
            // Flat extrapolation in vol → linear in variance
            return self.total_variances[0] * t / self.times[0];
        }
        if t >= *self.times.last().unwrap() {
            let n = self.times.len();
            return self.total_variances[n - 1] * t / self.times[n - 1];
        }

        // Find bracket
        let mut i = 0;
        while i < self.times.len() - 1 && self.times[i + 1] < t {
            i += 1;
        }

        let frac = (t - self.times[i]) / (self.times[i + 1] - self.times[i]);
        self.total_variances[i] * (1.0 - frac) + self.total_variances[i + 1] * frac
    }

    /// Forward variance between t1 and t2: w(t2) − w(t1).
    pub fn forward_variance(&self, t1: f64, t2: f64) -> f64 {
        (self.total_variance(t2) - self.total_variance(t1)).max(0.0)
    }

    /// Forward volatility between t1 and t2.
    pub fn forward_vol(&self, t1: f64, t2: f64) -> f64 {
        let dt = t2 - t1;
        if dt <= 0.0 {
            return self.black_vol(t1);
        }
        let fwd_var = self.forward_variance(t1, t2);
        (fwd_var / dt).sqrt()
    }
}

/// A volatility smile surface that holds smile sections at discrete expiries.
///
/// Each smile section provides σ(K) at a given expiry.
/// Between expiries, variance is linearly interpolated.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SmileSectionSurface {
    /// Expiry times.
    pub expiries: Vec<f64>,
    /// For each expiry, a set of (strike, vol) pairs defining the smile.
    pub smiles: Vec<Vec<(f64, f64)>>,
}

impl SmileSectionSurface {
    /// Create a surface from expiry-indexed smiles.
    pub fn new(expiries: Vec<f64>, smiles: Vec<Vec<(f64, f64)>>) -> Self {
        assert_eq!(expiries.len(), smiles.len());
        Self { expiries, smiles }
    }

    /// Interpolate vol at (strike, expiry) via bilinear interpolation.
    pub fn black_vol(&self, strike: f64, expiry: f64) -> f64 {
        if self.expiries.is_empty() {
            return 0.0;
        }

        // Find bracketing expiries
        if expiry <= self.expiries[0] {
            return interpolate_smile(&self.smiles[0], strike);
        }
        if expiry >= *self.expiries.last().unwrap() {
            return interpolate_smile(self.smiles.last().unwrap(), strike);
        }

        let mut i = 0;
        while i < self.expiries.len() - 1 && self.expiries[i + 1] < expiry {
            i += 1;
        }

        let t1 = self.expiries[i];
        let t2 = self.expiries[i + 1];
        let v1 = interpolate_smile(&self.smiles[i], strike);
        let v2 = interpolate_smile(&self.smiles[i + 1], strike);

        // Linear interpolation in total variance
        let w1 = v1 * v1 * t1;
        let w2 = v2 * v2 * t2;
        let frac = (expiry - t1) / (t2 - t1);
        let w = w1 * (1.0 - frac) + w2 * frac;

        if w <= 0.0 || expiry <= 0.0 {
            return 0.0;
        }
        (w / expiry).sqrt()
    }
}

/// Linearly interpolate a smile (sorted by strike) at a given strike.
fn interpolate_smile(smile: &[(f64, f64)], strike: f64) -> f64 {
    if smile.is_empty() {
        return 0.0;
    }
    if smile.len() == 1 || strike <= smile[0].0 {
        return smile[0].1;
    }
    if strike >= smile.last().unwrap().0 {
        return smile.last().unwrap().1;
    }

    let mut i = 0;
    while i < smile.len() - 1 && smile[i + 1].0 < strike {
        i += 1;
    }

    let (k1, v1) = smile[i];
    let (k2, v2) = smile[i + 1];
    let frac = (strike - k1) / (k2 - k1);
    v1 * (1.0 - frac) + v2 * frac
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn variance_curve_from_vols() {
        let curve = BlackVarianceCurve::from_vols(&[1.0, 2.0, 3.0], &[0.20, 0.22, 0.25]);
        assert_abs_diff_eq!(curve.total_variances[0], 0.04, epsilon = 1e-10);
        assert_abs_diff_eq!(curve.total_variances[1], 0.0968, epsilon = 1e-10);
    }

    #[test]
    fn variance_curve_interpolation() {
        let curve = BlackVarianceCurve::from_vols(&[1.0, 2.0], &[0.20, 0.20]);
        // With constant 20% vol, total var at t=1.5 should be 0.04 * 1.5 = 0.06
        let w = curve.total_variance(1.5);
        assert_abs_diff_eq!(w, 0.06, epsilon = 1e-10);
        let v = curve.black_vol(1.5);
        assert_abs_diff_eq!(v, 0.20, epsilon = 1e-10);
    }

    #[test]
    fn variance_curve_extrapolation() {
        let curve = BlackVarianceCurve::from_vols(&[1.0, 2.0], &[0.20, 0.20]);
        // Flat vol extrapolation
        let v = curve.black_vol(5.0);
        assert_abs_diff_eq!(v, 0.20, epsilon = 1e-10);
        let v0 = curve.black_vol(0.5);
        assert_abs_diff_eq!(v0, 0.20, epsilon = 1e-10);
    }

    #[test]
    fn forward_vol_constant_vol() {
        let curve = BlackVarianceCurve::from_vols(&[1.0, 2.0, 3.0], &[0.20, 0.20, 0.20]);
        let fv = curve.forward_vol(1.0, 2.0);
        assert_abs_diff_eq!(fv, 0.20, epsilon = 1e-10);
    }

    #[test]
    fn forward_variance_non_negative() {
        let curve = BlackVarianceCurve::from_vols(&[1.0, 2.0, 3.0], &[0.20, 0.22, 0.25]);
        let fw = curve.forward_variance(1.0, 2.0);
        assert!(fw >= 0.0);
    }

    #[test]
    fn smile_surface_single_expiry() {
        let smiles = vec![
            vec![(80.0, 0.30), (100.0, 0.20), (120.0, 0.25)],
        ];
        let surface = SmileSectionSurface::new(vec![1.0], smiles);
        // ATM vol
        assert_abs_diff_eq!(surface.black_vol(100.0, 1.0), 0.20, epsilon = 1e-10);
        // ITM strike
        let v = surface.black_vol(80.0, 1.0);
        assert_abs_diff_eq!(v, 0.30, epsilon = 1e-10);
    }

    #[test]
    fn smile_surface_interpolation_across_expiry() {
        let smiles = vec![
            vec![(100.0, 0.20)],
            vec![(100.0, 0.30)],
        ];
        let surface = SmileSectionSurface::new(vec![1.0, 2.0], smiles);
        // Interpolate at t=1.5
        let v = surface.black_vol(100.0, 1.5);
        // w(1) = 0.04, w(2) = 0.18, w(1.5) = 0.11, v = sqrt(0.11/1.5) ≈ 0.2708
        let expected = (0.11_f64 / 1.5).sqrt();
        assert_abs_diff_eq!(v, expected, epsilon = 1e-4);
    }

    #[test]
    fn smile_interpolation_linear() {
        let smile = [(80.0, 0.30), (100.0, 0.20), (120.0, 0.25)];
        let v = interpolate_smile(&smile, 90.0);
        // Linear interp between (80, 0.30) and (100, 0.20)
        assert_abs_diff_eq!(v, 0.25, epsilon = 1e-10);
    }
}
