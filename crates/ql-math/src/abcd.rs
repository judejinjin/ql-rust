//! ABCD parametric volatility function.
//!
//! The ABCD function is used for time-dependent volatility parametrisation,
//! especially in LMM calibration:
//!
//!   σ(t) = (a + b·t) · e^{−c·t} + d
//!
//! where a, b, c, d are parameters with constraints:
//!   - a + d > 0  (volatility at t=0)
//!   - d ≥ 0      (long-end floor)
//!   - c ≥ 0      (decay rate)
//!
//! Also provides calibration via Levenberg-Marquardt.

use crate::optimization::{Simplex, CostFunction, EndCriteria};

/// ABCD parametric volatility function: σ(t) = (a + b·t) · e^{−c·t} + d.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct AbcdFunction {
    /// A.
    pub a: f64,
    /// B.
    pub b: f64,
    /// C.
    pub c: f64,
    /// D.
    pub d: f64,
}

impl AbcdFunction {
    /// Create a new ABCD function with given parameters.
    pub fn new(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self { a, b, c, d }
    }

    /// Evaluate σ(t).
    pub fn value(&self, t: f64) -> f64 {
        (self.a + self.b * t) * (-self.c * t).exp() + self.d
    }

    /// Time of maximum volatility: t_max = (1/c − a/b) if b ≠ 0 and c > 0.
    pub fn max_time(&self) -> Option<f64> {
        if self.b.abs() < 1e-15 || self.c <= 0.0 {
            return None;
        }
        let t = 1.0 / self.c - self.a / self.b;
        if t > 0.0 { Some(t) } else { None }
    }

    /// Maximum volatility value.
    pub fn max_value(&self) -> f64 {
        match self.max_time() {
            Some(t) => self.value(t),
            None => self.value(0.0), // max at t=0
        }
    }

    /// Long-end asymptotic volatility: σ(∞) = d.
    pub fn long_end(&self) -> f64 {
        self.d
    }

    /// Short-end volatility: σ(0) = a + d.
    pub fn short_end(&self) -> f64 {
        self.a + self.d
    }

    /// Integral of σ²(t) from 0 to T (analytic closed form).
    ///
    /// ∫₀ᵀ σ²(t) dt where σ(t) = (a+bt)e^{-ct} + d.
    pub fn variance_integral(&self, big_t: f64) -> f64 {
        if big_t <= 0.0 {
            return 0.0;
        }
        let a = self.a;
        let b = self.b;
        let c = self.c;
        let d = self.d;

        if c.abs() < 1e-15 {
            // No decay: σ(t) = a + bt + d
            let ad = a + d;
            return ad * ad * big_t + ad * b * big_t * big_t + b * b * big_t.powi(3) / 3.0;
        }

        let e2ct = (-2.0 * c * big_t).exp();
        let ect = (-c * big_t).exp();

        // ∫(a+bt)²e^{-2ct} dt from 0 to T
        let i1 = {
            let aa = a * a / (2.0 * c) * (1.0 - e2ct);
            let ab = 2.0 * a * b / (4.0 * c * c) * (1.0 - e2ct * (1.0 + 2.0 * c * big_t));
            let bb = b * b / (4.0 * c * c * c)
                * (1.0 - e2ct * (1.0 + 2.0 * c * big_t + 2.0 * c * c * big_t * big_t));
            aa + ab + bb
        };

        // 2d·∫(a+bt)e^{-ct} dt from 0 to T
        let i2 = {
            let ia = a / c * (1.0 - ect);
            let ib = b / (c * c) * (1.0 - ect * (1.0 + c * big_t));
            2.0 * d * (ia + ib)
        };

        // d²T
        let i3 = d * d * big_t;

        i1 + i2 + i3
    }
}

/// Result of ABCD calibration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AbcdCalibrationResult {
    /// Calibrated ABCD function.
    pub abcd: AbcdFunction,
    /// RMS error in volatility units.
    pub rms_error: f64,
    /// Maximum absolute error.
    pub max_error: f64,
}

struct AbcdObjective<'a> {
    times: &'a [f64],
    vols: &'a [f64],
}

impl CostFunction for AbcdObjective<'_> {
    fn value(&self, params: &[f64]) -> f64 {
        let abcd = AbcdFunction::new(params[0], params[1], params[2].abs(), params[3].abs());
        self.times
            .iter()
            .zip(self.vols.iter())
            .map(|(&t, &v)| {
                let diff = abcd.value(t) - v;
                diff * diff
            })
            .sum()
    }

    fn dimension(&self) -> usize {
        4
    }
}

/// Calibrate ABCD parameters to observed volatility data.
///
/// # Parameters
/// - `times` — observation times (years)
/// - `vols` — observed volatilities at those times
///
/// # Returns
/// Calibrated ABCD function with error statistics.
pub fn abcd_calibrate(times: &[f64], vols: &[f64]) -> AbcdCalibrationResult {
    assert_eq!(times.len(), vols.len());
    assert!(!times.is_empty());

    // Initial guess from data
    let short_vol = vols[0];
    let long_vol = *vols.last().unwrap();
    let mid_vol = vols[vols.len() / 2];

    let d0 = long_vol.max(0.01);
    let a0 = (short_vol - d0).max(-0.5);
    let c0 = if times.len() > 1 {
        1.0 / times[times.len() / 2].max(0.1)
    } else {
        0.5
    };
    let b0 = if mid_vol > short_vol + 0.001 { 0.1 } else { -0.01 };

    let initial = vec![a0, b0, c0, d0];
    let obj = AbcdObjective { times, vols };
    let criteria = EndCriteria {
        max_iterations: 2000,
        max_stationary_iterations: 200,
        root_epsilon: 1e-12,
        function_epsilon: 1e-12,
        gradient_epsilon: 1e-12,
    };
    let simplex = Simplex::new(0.05);
    let result = simplex.minimize(&obj, &initial, &criteria).unwrap_or_else(|_| {
        crate::optimization::OptimizationResult {
            parameters: initial.clone(),
            value: obj.value(&initial),
            iterations: 0,
        }
    });

    let abcd = AbcdFunction::new(
        result.parameters[0],
        result.parameters[1],
        result.parameters[2].abs(),
        result.parameters[3].abs(),
    );

    let mut max_err = 0.0_f64;
    let mut sum_sq = 0.0;
    for (&t, &v) in times.iter().zip(vols.iter()) {
        let err = (abcd.value(t) - v).abs();
        max_err = max_err.max(err);
        sum_sq += err * err;
    }
    let rms = (sum_sq / times.len() as f64).sqrt();

    AbcdCalibrationResult {
        abcd,
        rms_error: rms,
        max_error: max_err,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abcd_value_at_zero() {
        let f = AbcdFunction::new(0.10, 0.05, 0.50, 0.15);
        assert!((f.value(0.0) - 0.25).abs() < 1e-12); // a + d = 0.10 + 0.15 = 0.25
    }

    #[test]
    fn abcd_long_end() {
        let f = AbcdFunction::new(0.10, 0.05, 0.50, 0.15);
        assert!((f.value(100.0) - 0.15).abs() < 0.01); // approaches d
    }

    #[test]
    fn abcd_max_time() {
        // b>0, c>0 → hump-shaped
        let f = AbcdFunction::new(-0.05, 0.10, 0.50, 0.20);
        let t = f.max_time().expect("Should have a max time");
        assert!(t > 0.0, "Max time should be positive: {t}");
        // Value at max should exceed endpoints
        let v_max = f.value(t);
        let v_0 = f.value(0.0);
        assert!(v_max >= v_0 - 1e-10, "Max {v_max} should >= start {v_0}");
    }

    #[test]
    fn variance_integral_flat_vol() {
        // a=0, b=0, d=0.2 → σ(t) = 0.2, ∫σ² dt = 0.04T
        let f = AbcdFunction::new(0.0, 0.0, 0.0, 0.2);
        let integral = f.variance_integral(5.0);
        assert!((integral - 0.04 * 5.0).abs() < 1e-10);
    }

    #[test]
    fn variance_integral_positive() {
        let f = AbcdFunction::new(0.10, 0.05, 0.50, 0.15);
        let integral = f.variance_integral(2.0);
        assert!(integral > 0.0, "Variance integral should be positive: {integral}");
    }

    #[test]
    fn calibrate_to_flat_vol() {
        let times: Vec<f64> = (1..=10).map(|i| i as f64 * 0.5).collect();
        let vols: Vec<f64> = vec![0.20; 10];
        let result = abcd_calibrate(&times, &vols);
        assert!(
            result.rms_error < 0.01,
            "RMS error should be small for flat vol: {}",
            result.rms_error
        );
    }

    #[test]
    fn calibrate_to_humped_vol() {
        // Generate data from known ABCD
        let true_f = AbcdFunction::new(0.05, 0.08, 0.40, 0.15);
        let times: Vec<f64> = (1..=20).map(|i| i as f64 * 0.25).collect();
        let vols: Vec<f64> = times.iter().map(|&t| true_f.value(t)).collect();

        let result = abcd_calibrate(&times, &vols);
        assert!(
            result.rms_error < 0.005,
            "RMS error should be small for humped data: {}",
            result.rms_error
        );
        // Calibrated function should reproduce values
        for (&t, &v) in times.iter().zip(vols.iter()) {
            let err = (result.abcd.value(t) - v).abs();
            assert!(err < 0.01, "At t={t}: error {err}");
        }
    }
}
