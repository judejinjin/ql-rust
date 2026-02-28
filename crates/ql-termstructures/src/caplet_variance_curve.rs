//! Caplet variance curve.
//!
//! A term structure of caplet variance as a function of option expiry.
//! This is useful for cap/floor pricing and interpolation of implied
//! volatilities.
//!
//! Corresponds to QuantLib's `CapletVarianceCurve`.

use serde::{Deserialize, Serialize};

/// A term structure of caplet variances (σ²·T) indexed by expiry time.
///
/// Supports flat, linear, and log-linear interpolation between pillars.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapletVarianceCurve {
    /// Pillar expiry times (sorted, in years).
    pub times: Vec<f64>,
    /// Caplet total variances at each pillar (σ² × T).
    pub total_variances: Vec<f64>,
    /// Interpolation method.
    pub interp: CapletVarInterp,
}

/// Interpolation method for caplet variance curve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapletVarInterp {
    /// Flat (piecewise-constant) volatility between pillars.
    Flat,
    /// Linear interpolation in total variance.
    Linear,
    /// Linear interpolation in variance rate (σ²), then multiply by T.
    LinearInVarianceRate,
}

impl CapletVarianceCurve {
    /// Create a new caplet variance curve from expiry times and flat vols.
    ///
    /// `vols` are annualized flat caplet volatilities at each pillar.
    pub fn from_vols(times: Vec<f64>, vols: Vec<f64>, interp: CapletVarInterp) -> Self {
        assert_eq!(times.len(), vols.len());
        let total_variances = times.iter().zip(vols.iter())
            .map(|(&t, &v)| v * v * t)
            .collect();
        Self { times, total_variances, interp }
    }

    /// Create a new caplet variance curve from expiry times and total variances.
    pub fn from_total_variances(times: Vec<f64>, total_variances: Vec<f64>, interp: CapletVarInterp) -> Self {
        assert_eq!(times.len(), total_variances.len());
        Self { times, total_variances, interp }
    }

    /// Total variance (σ² × T) at time t.
    pub fn total_variance(&self, t: f64) -> f64 {
        if self.times.is_empty() { return 0.0; }
        if t <= self.times[0] { return self.total_variances[0] * (t / self.times[0]).max(0.0); }
        if t >= *self.times.last().unwrap() { return *self.total_variances.last().unwrap(); }

        // Find bracketing interval
        let idx = self.times.partition_point(|&x| x < t).saturating_sub(1);
        let idx = idx.min(self.times.len() - 2);
        let t0 = self.times[idx];
        let t1 = self.times[idx + 1];
        let v0 = self.total_variances[idx];
        let v1 = self.total_variances[idx + 1];
        let w = (t - t0) / (t1 - t0);

        match self.interp {
            CapletVarInterp::Flat => {
                // Piecewise-constant vol: σ² = v0/t0, total_var = σ² * t
                let sigma_sq = if t0 > 0.0 { v0 / t0 } else { v1 / t1 };
                sigma_sq * t
            }
            CapletVarInterp::Linear => {
                // Linear in total variance
                v0 + w * (v1 - v0)
            }
            CapletVarInterp::LinearInVarianceRate => {
                // Linear in σ²
                let rate0 = if t0 > 0.0 { v0 / t0 } else { 0.0 };
                let rate1 = v1 / t1;
                let rate = rate0 + w * (rate1 - rate0);
                rate * t
            }
        }
    }

    /// Implied (Black) flat volatility at time t.
    pub fn volatility(&self, t: f64) -> f64 {
        if t <= 0.0 { return 0.0; }
        let tv = self.total_variance(t);
        (tv / t).max(0.0).sqrt()
    }

    /// Instantaneous forward variance at time t.
    /// d(σ²T)/dT evaluated at t.
    pub fn forward_variance(&self, t: f64) -> f64 {
        let eps = 1e-4;
        let tv_plus = self.total_variance(t + eps);
        let tv_minus = self.total_variance((t - eps).max(0.0));
        let dt = if t > eps { 2.0 * eps } else { t + eps };
        (tv_plus - tv_minus) / dt
    }

    /// Number of pillar points.
    pub fn size(&self) -> usize {
        self.times.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_caplet_var_curve_from_vols() {
        let times = vec![0.5, 1.0, 2.0, 3.0, 5.0];
        let vols = vec![0.20, 0.22, 0.21, 0.19, 0.18];
        let curve = CapletVarianceCurve::from_vols(times.clone(), vols.clone(), CapletVarInterp::Linear);

        // At pillar points, volatility should match
        for (&t, &v) in times.iter().zip(vols.iter()) {
            let vol = curve.volatility(t);
            assert!((vol - v).abs() < 1e-10, "t={}, expected={}, got={}", t, v, vol);
        }
    }

    #[test]
    fn test_caplet_var_curve_interpolation() {
        let times = vec![1.0, 2.0, 5.0];
        let vols = vec![0.20, 0.25, 0.22];
        let curve = CapletVarianceCurve::from_vols(times, vols, CapletVarInterp::Linear);

        let vol_1_5 = curve.volatility(1.5);
        // Between 20% and 25%
        assert!(vol_1_5 > 0.20 && vol_1_5 < 0.26, "vol={}", vol_1_5);
    }

    #[test]
    fn test_caplet_var_curve_flat_interp() {
        let times = vec![1.0, 2.0, 5.0];
        let vols = vec![0.20, 0.20, 0.20];
        let curve = CapletVarianceCurve::from_vols(times, vols, CapletVarInterp::Flat);

        // Flat vol everywhere
        for t in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0] {
            let vol = curve.volatility(t);
            assert!((vol - 0.20).abs() < 0.01, "t={}, vol={}", t, vol);
        }
    }

    #[test]
    fn test_caplet_forward_variance() {
        let times = vec![1.0, 2.0, 3.0];
        let vols = vec![0.20, 0.20, 0.20];
        let curve = CapletVarianceCurve::from_vols(times, vols, CapletVarInterp::Linear);

        // For flat vol, forward variance ≈ σ² = 0.04
        let fvar = curve.forward_variance(1.5);
        assert!((fvar - 0.04).abs() < 0.005, "fvar={}", fvar);
    }

    #[test]
    fn test_caplet_var_curve_total_variance_monotone() {
        let times = vec![1.0, 2.0, 5.0];
        let vols = vec![0.20, 0.22, 0.21];
        let curve = CapletVarianceCurve::from_vols(times, vols, CapletVarInterp::Linear);

        let tv1 = curve.total_variance(1.0);
        let tv2 = curve.total_variance(2.0);
        let tv5 = curve.total_variance(5.0);
        assert!(tv2 > tv1, "tv2={} <= tv1={}", tv2, tv1);
        assert!(tv5 > tv2, "tv5={} <= tv2={}", tv5, tv2);
    }
}
