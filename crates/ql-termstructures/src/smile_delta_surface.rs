//! FX delta volatility surface (SmileDeltaTermStructure).
//!
//! An FX volatility surface parameterized in the **delta space** across
//! multiple tenors. At each tenor, the smile is defined by:
//!
//! - ATM volatility
//! - 25Δ risk reversal (25Δ call vol − 25Δ put vol)
//! - 25Δ butterfly (½(25Δ call vol + 25Δ put vol) − ATM vol)
//!
//! Optionally, 10Δ quotes can be added for a 5-point smile.
//!
//! ## Interpolation
//!
//! - **In time**: linear in total variance (σ²τ)
//! - **In delta**: cubic or linear interpolation of vols across delta values
//!
//! ## QuantLib Parity
//!
//! Corresponds to `DeltaVolQuote` + `FxBlackVolatilitySurface` in QuantLib C++.

use ql_math::black_delta::{AtmType, BlackDeltaCalculator, DeltaType};
use serde::{Deserialize, Serialize};

/// A single smile section at one expiry, defined by delta-space quotes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmileDeltaSection {
    /// Time to expiry in years.
    pub tau: f64,
    /// ATM volatility.
    pub atm_vol: f64,
    /// 25Δ risk reversal: σ(25Δ call) − σ(25Δ put).
    pub rr_25d: f64,
    /// 25Δ butterfly: ½(σ(25Δ call) + σ(25Δ put)) − σ(ATM).
    pub bf_25d: f64,
    /// Optional 10Δ risk reversal.
    pub rr_10d: Option<f64>,
    /// Optional 10Δ butterfly.
    pub bf_10d: Option<f64>,
}

impl SmileDeltaSection {
    /// Create a 3-point smile (ATM, 25Δ RR, 25Δ BF).
    pub fn new_3pt(tau: f64, atm_vol: f64, rr_25d: f64, bf_25d: f64) -> Self {
        Self {
            tau,
            atm_vol,
            rr_25d,
            bf_25d,
            rr_10d: None,
            bf_10d: None,
        }
    }

    /// Create a 5-point smile (ATM, 25Δ RR/BF, 10Δ RR/BF).
    pub fn new_5pt(
        tau: f64,
        atm_vol: f64,
        rr_25d: f64,
        bf_25d: f64,
        rr_10d: f64,
        bf_10d: f64,
    ) -> Self {
        Self {
            tau,
            atm_vol,
            rr_25d,
            bf_25d,
            rr_10d: Some(rr_10d),
            bf_10d: Some(bf_10d),
        }
    }

    /// Compute individual volatilities from the market quotes.
    ///
    /// Returns (vol_25d_put, vol_atm, vol_25d_call) for 3-point,
    /// or (vol_10d_put, vol_25d_put, vol_atm, vol_25d_call, vol_10d_call) for 5-point.
    pub fn vols(&self) -> Vec<f64> {
        let vol_25d_call = self.atm_vol + self.bf_25d + 0.5 * self.rr_25d;
        let vol_25d_put = self.atm_vol + self.bf_25d - 0.5 * self.rr_25d;

        if let (Some(rr_10d), Some(bf_10d)) = (self.rr_10d, self.bf_10d) {
            let vol_10d_call = self.atm_vol + bf_10d + 0.5 * rr_10d;
            let vol_10d_put = self.atm_vol + bf_10d - 0.5 * rr_10d;
            vec![vol_10d_put, vol_25d_put, self.atm_vol, vol_25d_call, vol_10d_call]
        } else {
            vec![vol_25d_put, self.atm_vol, vol_25d_call]
        }
    }

    /// Compute (delta, vol) pairs for this smile section.
    ///
    /// Deltas are absolute values: 0.10, 0.25, 0.50, 0.75, 0.90 etc.
    pub fn delta_vol_pairs(&self) -> Vec<(f64, f64)> {
        let vols = self.vols();
        if vols.len() == 5 {
            vec![
                (0.10, vols[0]),
                (0.25, vols[1]),
                (0.50, vols[2]),
                (0.75, vols[3]),
                (0.90, vols[4]),
            ]
        } else {
            vec![
                (0.25, vols[0]),
                (0.50, vols[1]),
                (0.75, vols[2]),
            ]
        }
    }

    /// Interpolate volatility at a given absolute delta (e.g. 0.30).
    pub fn vol_at_delta(&self, delta: f64) -> f64 {
        let pairs = self.delta_vol_pairs();
        if pairs.is_empty() {
            return self.atm_vol;
        }
        if delta <= pairs[0].0 {
            return pairs[0].1;
        }
        let n = pairs.len();
        if delta >= pairs[n - 1].0 {
            return pairs[n - 1].1;
        }
        // Linear interpolation
        let idx = pairs.partition_point(|(d, _)| *d < delta).min(n - 1).max(1);
        let (d0, v0) = pairs[idx - 1];
        let (d1, v1) = pairs[idx];
        v0 + (v1 - v0) * (delta - d0) / (d1 - d0)
    }

    /// Get the strike corresponding to a given delta using BlackDeltaCalculator.
    ///
    /// Returns (strike, vol) for the given delta.
    pub fn strike_at_delta(
        &self,
        delta: f64,
        spot: f64,
        rd: f64,
        rf: f64,
        delta_type: DeltaType,
    ) -> Option<(f64, f64)> {
        let vol = self.vol_at_delta(delta);
        let calc = BlackDeltaCalculator::new(true, spot, rd, rf, vol, self.tau).ok()?;
        let strike = calc.strike_from_delta(delta, delta_type).ok()?;
        Some((strike, vol))
    }
}

// ===========================================================================
// SmileDeltaTermStructure
// ===========================================================================

/// FX delta volatility term structure.
///
/// A surface of implied volatilities parameterized by delta and time,
/// built from smile sections at discrete tenors.
///
/// Interpolation:
/// - **Time**: linear in total variance (σ²τ) between pillars
/// - **Delta**: linear interpolation within each smile section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmileDeltaTermStructure {
    /// Smile sections sorted by expiry time.
    pub sections: Vec<SmileDeltaSection>,
    /// Spot FX rate, for strike conversions.
    pub spot: f64,
    /// Domestic risk-free rate.
    pub rd: f64,
    /// Foreign risk-free rate.
    pub rf: f64,
    /// Delta convention.
    pub delta_type: DeltaType,
    /// ATM convention.
    pub atm_type: AtmType,
}

impl SmileDeltaTermStructure {
    /// Create a new delta vol surface from smile sections.
    ///
    /// Sections are sorted by time to expiry internally.
    pub fn new(
        mut sections: Vec<SmileDeltaSection>,
        spot: f64,
        rd: f64,
        rf: f64,
        delta_type: DeltaType,
        atm_type: AtmType,
    ) -> Self {
        sections.sort_by(|a, b| a.tau.partial_cmp(&b.tau).unwrap());
        Self {
            sections,
            spot,
            rd,
            rf,
            delta_type,
            atm_type,
        }
    }

    /// Get volatility at a specific (time, delta) point.
    ///
    /// Interpolates in total-variance space between tenor pillars,
    /// then in delta space within each smile section.
    pub fn vol(&self, tau: f64, delta: f64) -> f64 {
        if self.sections.is_empty() {
            return 0.0;
        }
        if self.sections.len() == 1 {
            return self.sections[0].vol_at_delta(delta);
        }

        // Find the two surrounding sections
        let idx = self
            .sections
            .partition_point(|s| s.tau < tau)
            .min(self.sections.len() - 1)
            .max(1);

        let s0 = &self.sections[idx - 1];
        let s1 = &self.sections[idx];

        if tau <= s0.tau {
            return s0.vol_at_delta(delta);
        }
        if tau >= s1.tau {
            return s1.vol_at_delta(delta);
        }

        // Linear interpolation in total variance: var = σ²τ
        let v0 = s0.vol_at_delta(delta);
        let v1 = s1.vol_at_delta(delta);
        let var0 = v0 * v0 * s0.tau;
        let var1 = v1 * v1 * s1.tau;

        let w = (tau - s0.tau) / (s1.tau - s0.tau);
        let var_interp = var0 + w * (var1 - var0);

        if var_interp > 0.0 && tau > 0.0 {
            (var_interp / tau).sqrt()
        } else {
            v0 + w * (v1 - v0) // fallback to linear in vol
        }
    }

    /// Get the ATM volatility at a given time.
    pub fn atm_vol(&self, tau: f64) -> f64 {
        self.vol(tau, 0.50)
    }

    /// Get the 25Δ risk reversal at a given time.
    pub fn rr_25d(&self, tau: f64) -> f64 {
        self.vol(tau, 0.75) - self.vol(tau, 0.25)
    }

    /// Get the 25Δ butterfly at a given time.
    pub fn bf_25d(&self, tau: f64) -> f64 {
        0.5 * (self.vol(tau, 0.75) + self.vol(tau, 0.25)) - self.vol(tau, 0.50)
    }

    /// Get the smile at a given time as a vector of (delta, vol) pairs.
    pub fn smile_at(&self, tau: f64) -> Vec<(f64, f64)> {
        let deltas = vec![0.10, 0.25, 0.50, 0.75, 0.90];
        deltas
            .iter()
            .map(|&d| (d, self.vol(tau, d)))
            .collect()
    }

    /// Convert a (tau, delta) point to a (tau, strike, vol) triple.
    pub fn to_strike_vol(
        &self,
        tau: f64,
        delta: f64,
    ) -> (f64, f64) {
        let vol = self.vol(tau, delta);
        let calc = BlackDeltaCalculator::new(true, self.spot, self.rd, self.rf, vol, tau);
        match calc {
            Ok(c) => match c.strike_from_delta(delta, self.delta_type) {
                Ok(strike) => (strike, vol),
                Err(_) => (self.spot, vol),
            },
            Err(_) => (self.spot, vol), // fallback
        }
    }

    /// Number of tenor pillars.
    pub fn num_tenors(&self) -> usize {
        self.sections.len()
    }

    /// Tenor times.
    pub fn tenors(&self) -> Vec<f64> {
        self.sections.iter().map(|s| s.tau).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_surface() -> SmileDeltaTermStructure {
        let sections = vec![
            SmileDeltaSection::new_3pt(0.25, 0.10, 0.015, 0.005),  // 3M
            SmileDeltaSection::new_3pt(0.50, 0.105, 0.018, 0.006), // 6M
            SmileDeltaSection::new_3pt(1.00, 0.11, 0.020, 0.007),  // 1Y
        ];
        SmileDeltaTermStructure::new(
            sections,
            1.10,  // EUR/USD spot
            0.05,  // USD rate
            0.03,  // EUR rate
            DeltaType::Spot,
            AtmType::AtmDeltaNeutral,
        )
    }

    #[test]
    fn smile_section_3pt_vols() {
        let s = SmileDeltaSection::new_3pt(0.25, 0.10, 0.015, 0.005);
        let vols = s.vols();
        assert_eq!(vols.len(), 3);
        // 25Δ put: ATM + BF - RR/2 = 0.10 + 0.005 - 0.0075 = 0.0975
        assert_abs_diff_eq!(vols[0], 0.0975, epsilon = 1e-10);
        // ATM
        assert_abs_diff_eq!(vols[1], 0.10, epsilon = 1e-10);
        // 25Δ call: ATM + BF + RR/2 = 0.10 + 0.005 + 0.0075 = 0.1125
        assert_abs_diff_eq!(vols[2], 0.1125, epsilon = 1e-10);
    }

    #[test]
    fn smile_section_5pt_vols() {
        let s = SmileDeltaSection::new_5pt(0.25, 0.10, 0.015, 0.005, 0.025, 0.010);
        let vols = s.vols();
        assert_eq!(vols.len(), 5);
        // 10Δ put: ATM + BF10 - RR10/2 = 0.10 + 0.010 - 0.0125 = 0.0975
        assert_abs_diff_eq!(vols[0], 0.0975, epsilon = 1e-10);
    }

    #[test]
    fn surface_atm_at_pillar() {
        let surf = make_surface();
        assert_abs_diff_eq!(surf.atm_vol(0.25), 0.10, epsilon = 1e-10);
        assert_abs_diff_eq!(surf.atm_vol(1.00), 0.11, epsilon = 1e-10);
    }

    #[test]
    fn surface_vol_interpolation_in_time() {
        let surf = make_surface();
        // At 0.75Y (between 6M=0.50 and 1Y=1.00), ATM should be interpolated
        let vol_075 = surf.atm_vol(0.75);
        assert!(vol_075 > 0.105 && vol_075 < 0.11,
            "Interpolated 9M ATM vol {} should be between 6M (0.105) and 1Y (0.11)", vol_075);
    }

    #[test]
    fn surface_rr_positive() {
        let surf = make_surface();
        let rr = surf.rr_25d(0.50);
        assert!(rr > 0.0, "25Δ RR should be positive when calls have higher vol");
    }

    #[test]
    fn surface_bf_positive() {
        let surf = make_surface();
        let bf = surf.bf_25d(0.50);
        assert!(bf > 0.0, "25Δ BF should be positive (smile effect)");
    }

    #[test]
    fn surface_smile_at_returns_5_points() {
        let surf = make_surface();
        let smile = surf.smile_at(0.50);
        assert_eq!(smile.len(), 5);
    }

    #[test]
    fn surface_extrapolation_flat() {
        let surf = make_surface();
        // Before first pillar → flat
        let vol_short = surf.atm_vol(0.10);
        assert_abs_diff_eq!(vol_short, 0.10, epsilon = 1e-10);
        // After last pillar → flat
        let vol_long = surf.atm_vol(2.0);
        assert_abs_diff_eq!(vol_long, 0.11, epsilon = 1e-10);
    }

    #[test]
    fn to_strike_vol_returns_valid_strike() {
        let surf = make_surface();
        let (strike, vol) = surf.to_strike_vol(0.50, 0.25);
        assert!(strike > 0.0, "Strike should be positive");
        assert!(vol > 0.0, "Vol should be positive");
    }

    #[test]
    fn tenors_sorted() {
        let surf = make_surface();
        let tenors = surf.tenors();
        assert_eq!(tenors, vec![0.25, 0.50, 1.00]);
    }

    #[test]
    fn total_variance_interpolation_is_arbitrage_free() {
        let surf = make_surface();
        // Total variance should be non-decreasing in time
        let var_3m = surf.atm_vol(0.25).powi(2) * 0.25;
        let var_6m = surf.atm_vol(0.50).powi(2) * 0.50;
        let var_9m = surf.atm_vol(0.75).powi(2) * 0.75;
        let var_1y = surf.atm_vol(1.00).powi(2) * 1.00;
        assert!(var_3m <= var_6m, "Variance should increase: 3M {} > 6M {}", var_3m, var_6m);
        assert!(var_6m <= var_9m, "Variance should increase: 6M {} > 9M {}", var_6m, var_9m);
        assert!(var_9m <= var_1y, "Variance should increase: 9M {} > 1Y {}", var_9m, var_1y);
    }
}
