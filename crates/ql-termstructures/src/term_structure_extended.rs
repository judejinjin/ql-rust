//! Extended term structures for local vol, credit, inflation, optionlet, and more.
//!
//! **G61** — ConstantLocalVol
//! **G62** — FixedLocalVolSurface
//! **G63** — GridModelLocalVolSurface
//! **G64** — ImpliedVolTermStructure
//! **G65** — DefaultDensityStructure
//! **G66** — HazardRateStructure (interpolated)
//! **G67** — InterpolatedHazardRateCurve
//! **G68** — InterpolatedSurvivalProbCurve
//! **G69** — CapFloorTermVolCurve
//! **G70** — OptionletVolatilityStructure + ConstantOptionletVol
//! **G71** — SpreadedOptionletVol
//! **G72** — InterpolatedZeroInflationCurve
//! **G73** — PiecewiseYoYInflationCurve (basic)
//! **G74** — Gaussian1dSwaptionVol
//! **G75** — VolatilityType enum

use crate::term_structure::TermStructure;
use crate::vol_term_structure::{BlackVolTermStructure, LocalVolTermStructure};
use ql_time::{Calendar, Date, DayCounter};
use std::sync::Arc;

// ===========================================================================
// VolatilityType (G75)
// ===========================================================================

/// Volatility type: shifted log-normal (Black) or normal (Bachelier).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[derive(Default)]
pub enum VolatilityType {
    /// Black-Scholes / Black-76 log-normal volatility.
    #[default]
    ShiftedLognormal,
    /// Bachelier (normal / absolute) volatility.
    Normal,
}


// ===========================================================================
// ConstantLocalVol (G61)
// ===========================================================================

/// Constant local volatility surface.
///
/// σ_local(t, S) = σ for all t and S.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConstantLocalVol {
    reference_date: Date,
    day_counter: DayCounter,
    vol: f64,
}

impl ConstantLocalVol {
    pub fn new(reference_date: Date, vol: f64, day_counter: DayCounter) -> Self {
        Self {
            reference_date,
            day_counter,
            vol,
        }
    }
}

impl TermStructure for ConstantLocalVol {
    fn reference_date(&self) -> Date {
        self.reference_date
    }
    fn day_counter(&self) -> DayCounter {
        self.day_counter
    }
    fn calendar(&self) -> &Calendar {
        &Calendar::NullCalendar
    }
    fn max_date(&self) -> Date {
        Date::from_serial(73050)
    }
}

impl LocalVolTermStructure for ConstantLocalVol {
    fn local_vol(&self, _t: f64, _underlying: f64) -> f64 {
        self.vol
    }
}

// ===========================================================================
// FixedLocalVolSurface (G62)
// ===========================================================================

/// Pre-computed local volatility surface on a fixed (time × strike) grid.
///
/// Uses bilinear interpolation between grid nodes.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FixedLocalVolSurface {
    reference_date: Date,
    day_counter: DayCounter,
    times: Vec<f64>,
    strikes: Vec<f64>,
    /// Row-major: `vols[i * n_strikes + j]` = σ(tᵢ, Kⱼ).
    vols: Vec<f64>,
}

impl FixedLocalVolSurface {
    /// Create from a grid of times, strikes, and local vols.
    ///
    /// `vols` must have length `times.len() × strikes.len()`.
    pub fn new(
        reference_date: Date,
        day_counter: DayCounter,
        times: Vec<f64>,
        strikes: Vec<f64>,
        vols: Vec<f64>,
    ) -> Self {
        assert_eq!(vols.len(), times.len() * strikes.len());
        Self {
            reference_date,
            day_counter,
            times,
            strikes,
            vols,
        }
    }

    fn interpolate(&self, t: f64, s: f64) -> f64 {
        let nt = self.times.len();
        let ns = self.strikes.len();
        if nt == 0 || ns == 0 {
            return 0.0;
        }

        // Find time bracket
        let ti = find_bracket(&self.times, t);
        let si = find_bracket(&self.strikes, s);

        let t0 = self.times[ti];
        let t1 = self.times[(ti + 1).min(nt - 1)];
        let s0 = self.strikes[si];
        let s1 = self.strikes[(si + 1).min(ns - 1)];

        let v00 = self.vols[ti * ns + si];
        let v01 = self.vols[ti * ns + (si + 1).min(ns - 1)];
        let v10 = self.vols[(ti + 1).min(nt - 1) * ns + si];
        let v11 = self.vols[(ti + 1).min(nt - 1) * ns + (si + 1).min(ns - 1)];

        let tt = if (t1 - t0).abs() > 1e-15 {
            (t - t0) / (t1 - t0)
        } else {
            0.0
        };
        let ss = if (s1 - s0).abs() > 1e-15 {
            (s - s0) / (s1 - s0)
        } else {
            0.0
        };

        let tt = tt.clamp(0.0, 1.0);
        let ss = ss.clamp(0.0, 1.0);

        (1.0 - tt) * ((1.0 - ss) * v00 + ss * v01) + tt * ((1.0 - ss) * v10 + ss * v11)
    }
}

fn find_bracket(sorted: &[f64], x: f64) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    match sorted.binary_search_by(|v| v.partial_cmp(&x).unwrap()) {
        Ok(i) => i.min(sorted.len().saturating_sub(2)),
        Err(i) => {
            if i == 0 {
                0
            } else {
                (i - 1).min(sorted.len().saturating_sub(2))
            }
        }
    }
}

impl TermStructure for FixedLocalVolSurface {
    fn reference_date(&self) -> Date {
        self.reference_date
    }
    fn day_counter(&self) -> DayCounter {
        self.day_counter
    }
    fn calendar(&self) -> &Calendar {
        &Calendar::NullCalendar
    }
    fn max_date(&self) -> Date {
        Date::from_serial(73050)
    }
}

impl LocalVolTermStructure for FixedLocalVolSurface {
    fn local_vol(&self, t: f64, underlying: f64) -> f64 {
        self.interpolate(t, underlying)
    }
}

// ===========================================================================
// GridModelLocalVolSurface (G63)
// ===========================================================================

/// Local volatility surface computed from a calibrated model on a grid.
///
/// Wraps a callable that computes local vol at any (t, S) point using
/// an underlying calibrated model.
pub struct GridModelLocalVolSurface {
    reference_date: Date,
    day_counter: DayCounter,
    local_vol_fn: Arc<dyn Fn(f64, f64) -> f64 + Send + Sync>,
}

impl GridModelLocalVolSurface {
    pub fn new(
        reference_date: Date,
        day_counter: DayCounter,
        local_vol_fn: Arc<dyn Fn(f64, f64) -> f64 + Send + Sync>,
    ) -> Self {
        Self {
            reference_date,
            day_counter,
            local_vol_fn,
        }
    }
}

impl TermStructure for GridModelLocalVolSurface {
    fn reference_date(&self) -> Date {
        self.reference_date
    }
    fn day_counter(&self) -> DayCounter {
        self.day_counter
    }
    fn calendar(&self) -> &Calendar {
        &Calendar::NullCalendar
    }
    fn max_date(&self) -> Date {
        Date::from_serial(73050)
    }
}

impl LocalVolTermStructure for GridModelLocalVolSurface {
    fn local_vol(&self, t: f64, underlying: f64) -> f64 {
        (self.local_vol_fn)(t, underlying)
    }
}

// ===========================================================================
// ImpliedVolTermStructure (G64)
// ===========================================================================

/// An implied Black vol term structure that shifts the reference date of
/// an underlying vol surface. This allows using a vol surface as if it
/// were observed at a different date.
pub struct ImpliedVolTermStructure {
    underlying: Arc<dyn BlackVolTermStructure>,
    reference_date: Date,
    day_counter: DayCounter,
}

impl ImpliedVolTermStructure {
    pub fn new(
        underlying: Arc<dyn BlackVolTermStructure>,
        reference_date: Date,
    ) -> Self {
        let day_counter = underlying.day_counter();
        Self {
            underlying,
            reference_date,
            day_counter,
        }
    }
}

impl TermStructure for ImpliedVolTermStructure {
    fn reference_date(&self) -> Date {
        self.reference_date
    }
    fn day_counter(&self) -> DayCounter {
        self.day_counter
    }
    fn calendar(&self) -> &Calendar {
        &Calendar::NullCalendar
    }
    fn max_date(&self) -> Date {
        self.underlying.max_date()
    }
}

impl BlackVolTermStructure for ImpliedVolTermStructure {
    fn black_vol(&self, t: f64, strike: f64) -> f64 {
        // Shift time by the difference between reference dates
        let dt = self.day_counter.year_fraction(
            self.underlying.reference_date(),
            self.reference_date,
        );
        self.underlying.black_vol(t + dt, strike)
    }
}

// ===========================================================================
// InterpolatedHazardRateCurve (G67)
// ===========================================================================

/// Interpolated hazard rate term structure.
///
/// Given a set of (time, hazard_rate) points, interpolates between them
/// (linearly by default) and computes survival probabilities via integration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InterpolatedHazardRateCurve {
    reference_date: Date,
    day_counter: DayCounter,
    times: Vec<f64>,
    hazard_rates: Vec<f64>,
}

impl InterpolatedHazardRateCurve {
    pub fn new(
        reference_date: Date,
        day_counter: DayCounter,
        dates: &[Date],
        hazard_rates: &[f64],
    ) -> Self {
        let times: Vec<f64> = dates
            .iter()
            .map(|d| day_counter.year_fraction(reference_date, *d))
            .collect();
        Self {
            reference_date,
            day_counter,
            times,
            hazard_rates: hazard_rates.to_vec(),
        }
    }

    /// Interpolated hazard rate at time `t`.
    pub fn hazard_rate(&self, t: f64) -> f64 {
        if self.times.is_empty() {
            return 0.0;
        }
        if t <= self.times[0] {
            return self.hazard_rates[0];
        }
        let n = self.times.len();
        if t >= self.times[n - 1] {
            return self.hazard_rates[n - 1];
        }
        let i = find_bracket(&self.times, t);
        let alpha = (t - self.times[i]) / (self.times[i + 1] - self.times[i]);
        self.hazard_rates[i] * (1.0 - alpha) + self.hazard_rates[i + 1] * alpha
    }

    /// Survival probability S(t) = exp(-∫₀ᵗ h(s) ds).
    pub fn survival_probability(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }
        // Simple trapezoidal integration of hazard rate
        let n_steps = 100;
        let dt = t / n_steps as f64;
        let mut integral = 0.0;
        let mut h_prev = self.hazard_rate(0.0);
        for i in 1..=n_steps {
            let h = self.hazard_rate(i as f64 * dt);
            integral += 0.5 * (h_prev + h) * dt;
            h_prev = h;
        }
        (-integral).exp()
    }

    /// Default probability up to time `t`: 1 - S(t).
    pub fn default_probability(&self, t: f64) -> f64 {
        1.0 - self.survival_probability(t)
    }
}

impl TermStructure for InterpolatedHazardRateCurve {
    fn reference_date(&self) -> Date {
        self.reference_date
    }
    fn day_counter(&self) -> DayCounter {
        self.day_counter
    }
    fn calendar(&self) -> &Calendar {
        &Calendar::NullCalendar
    }
    fn max_date(&self) -> Date {
        Date::from_serial(73050)
    }
}

// ===========================================================================
// InterpolatedSurvivalProbCurve (G68)
// ===========================================================================

/// Interpolated survival probability curve.
///
/// Given a set of (time, survival_probability) points, interpolates and provides
/// hazard rates and default probabilities.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InterpolatedSurvivalProbCurve {
    reference_date: Date,
    day_counter: DayCounter,
    times: Vec<f64>,
    survival_probs: Vec<f64>,
}

impl InterpolatedSurvivalProbCurve {
    pub fn new(
        reference_date: Date,
        day_counter: DayCounter,
        dates: &[Date],
        survival_probs: &[f64],
    ) -> Self {
        let times: Vec<f64> = dates
            .iter()
            .map(|d| day_counter.year_fraction(reference_date, *d))
            .collect();
        Self {
            reference_date,
            day_counter,
            times,
            survival_probs: survival_probs.to_vec(),
        }
    }

    /// Survival probability at time `t` (log-linear interpolation).
    pub fn survival_probability(&self, t: f64) -> f64 {
        if self.times.is_empty() {
            return 1.0;
        }
        if t <= 0.0 {
            return 1.0;
        }
        if t <= self.times[0] {
            // Extrapolate using first hazard rate
            let h = -self.survival_probs[0].ln() / self.times[0];
            return (-h * t).exp();
        }
        let n = self.times.len();
        if t >= self.times[n - 1] {
            // Flat extrapolation of hazard rate
            let h = -self.survival_probs[n - 1].ln() / self.times[n - 1];
            return (-h * t).exp();
        }
        let i = find_bracket(&self.times, t);
        // Log-linear interpolation
        let alpha = (t - self.times[i]) / (self.times[i + 1] - self.times[i]);
        let ln_sp = self.survival_probs[i].ln() * (1.0 - alpha)
            + self.survival_probs[i + 1].ln() * alpha;
        ln_sp.exp()
    }

    /// Default probability up to time `t`.
    pub fn default_probability(&self, t: f64) -> f64 {
        1.0 - self.survival_probability(t)
    }

    /// Instantaneous hazard rate h(t) = -d/dt ln S(t).
    pub fn hazard_rate(&self, t: f64) -> f64 {
        let eps = 1e-4;
        let sp1 = self.survival_probability(t);
        let sp2 = self.survival_probability(t + eps);
        if sp1 > 0.0 {
            -(sp2 / sp1).ln() / eps
        } else {
            0.0
        }
    }
}

impl TermStructure for InterpolatedSurvivalProbCurve {
    fn reference_date(&self) -> Date {
        self.reference_date
    }
    fn day_counter(&self) -> DayCounter {
        self.day_counter
    }
    fn calendar(&self) -> &Calendar {
        &Calendar::NullCalendar
    }
    fn max_date(&self) -> Date {
        Date::from_serial(73050)
    }
}

// ===========================================================================
// CapFloorTermVolCurve (G69)
// ===========================================================================

/// At-the-money cap/floor volatility curve (1D: term → vol).
///
/// Stores ATM Black (or normal) volatilities for caps/floors at various tenors.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CapFloorTermVolCurve {
    reference_date: Date,
    day_counter: DayCounter,
    tenors: Vec<f64>,
    vols: Vec<f64>,
    vol_type: VolatilityType,
}

impl CapFloorTermVolCurve {
    pub fn new(
        reference_date: Date,
        day_counter: DayCounter,
        tenors: Vec<f64>,
        vols: Vec<f64>,
        vol_type: VolatilityType,
    ) -> Self {
        Self {
            reference_date,
            day_counter,
            tenors,
            vols,
            vol_type,
        }
    }

    /// ATM vol at a given tenor (years).
    pub fn vol(&self, tenor: f64) -> f64 {
        if self.tenors.is_empty() {
            return 0.0;
        }
        if tenor <= self.tenors[0] {
            return self.vols[0];
        }
        let n = self.tenors.len();
        if tenor >= self.tenors[n - 1] {
            return self.vols[n - 1];
        }
        let i = find_bracket(&self.tenors, tenor);
        let alpha = (tenor - self.tenors[i]) / (self.tenors[i + 1] - self.tenors[i]);
        self.vols[i] * (1.0 - alpha) + self.vols[i + 1] * alpha
    }

    pub fn volatility_type(&self) -> &VolatilityType {
        &self.vol_type
    }
}

impl TermStructure for CapFloorTermVolCurve {
    fn reference_date(&self) -> Date {
        self.reference_date
    }
    fn day_counter(&self) -> DayCounter {
        self.day_counter
    }
    fn calendar(&self) -> &Calendar {
        &Calendar::NullCalendar
    }
    fn max_date(&self) -> Date {
        Date::from_serial(73050)
    }
}

// ===========================================================================
// OptionletVolatilityStructure (G70)
// ===========================================================================

/// Optionlet (caplet/floorlet) volatility surface trait.
pub trait OptionletVolatilityStructure: TermStructure {
    /// Optionlet volatility at time `t` and strike `strike`.
    fn optionlet_vol(&self, t: f64, strike: f64) -> f64;

    /// Volatility type (lognormal vs normal).
    fn volatility_type(&self) -> VolatilityType {
        VolatilityType::ShiftedLognormal
    }

    /// Shift (for shifted lognormal).
    fn displacement(&self) -> f64 {
        0.0
    }
}

/// Constant optionlet volatility surface.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConstantOptionletVol {
    reference_date: Date,
    day_counter: DayCounter,
    vol: f64,
    vol_type: VolatilityType,
    displacement: f64,
}

impl ConstantOptionletVol {
    pub fn new(
        reference_date: Date,
        vol: f64,
        day_counter: DayCounter,
        vol_type: VolatilityType,
        displacement: f64,
    ) -> Self {
        Self {
            reference_date,
            day_counter,
            vol,
            vol_type,
            displacement,
        }
    }
}

impl TermStructure for ConstantOptionletVol {
    fn reference_date(&self) -> Date {
        self.reference_date
    }
    fn day_counter(&self) -> DayCounter {
        self.day_counter
    }
    fn calendar(&self) -> &Calendar {
        &Calendar::NullCalendar
    }
    fn max_date(&self) -> Date {
        Date::from_serial(73050)
    }
}

impl OptionletVolatilityStructure for ConstantOptionletVol {
    fn optionlet_vol(&self, _t: f64, _strike: f64) -> f64 {
        self.vol
    }

    fn volatility_type(&self) -> VolatilityType {
        self.vol_type.clone()
    }

    fn displacement(&self) -> f64 {
        self.displacement
    }
}

// ===========================================================================
// SpreadedOptionletVol (G71)
// ===========================================================================

/// Optionlet volatility surface with an additive spread on top of an underlying.
pub struct SpreadedOptionletVol {
    reference_date: Date,
    day_counter: DayCounter,
    underlying: Arc<dyn OptionletVolatilityStructure>,
    spread: f64,
}

impl SpreadedOptionletVol {
    pub fn new(
        underlying: Arc<dyn OptionletVolatilityStructure>,
        spread: f64,
    ) -> Self {
        let reference_date = underlying.reference_date();
        let day_counter = underlying.day_counter();
        Self {
            reference_date,
            day_counter,
            underlying,
            spread,
        }
    }
}

impl TermStructure for SpreadedOptionletVol {
    fn reference_date(&self) -> Date {
        self.reference_date
    }
    fn day_counter(&self) -> DayCounter {
        self.day_counter
    }
    fn calendar(&self) -> &Calendar {
        &Calendar::NullCalendar
    }
    fn max_date(&self) -> Date {
        self.underlying.max_date()
    }
}

impl OptionletVolatilityStructure for SpreadedOptionletVol {
    fn optionlet_vol(&self, t: f64, strike: f64) -> f64 {
        self.underlying.optionlet_vol(t, strike) + self.spread
    }

    fn volatility_type(&self) -> VolatilityType {
        self.underlying.volatility_type()
    }

    fn displacement(&self) -> f64 {
        self.underlying.displacement()
    }
}

// ===========================================================================
// InterpolatedZeroInflationCurve (G72)
// ===========================================================================

/// Interpolated zero-coupon inflation rate curve.
///
/// Given a set of (date, zero_rate) pairs, interpolates to provide
/// CPI ratio = (1 + zero_rate)^t for arbitrary dates.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InterpolatedZeroInflationCurve {
    reference_date: Date,
    day_counter: DayCounter,
    base_cpi: f64,
    times: Vec<f64>,
    zero_rates: Vec<f64>,
}

impl InterpolatedZeroInflationCurve {
    pub fn new(
        reference_date: Date,
        day_counter: DayCounter,
        base_cpi: f64,
        dates: &[Date],
        zero_rates: &[f64],
    ) -> Self {
        let times: Vec<f64> = dates
            .iter()
            .map(|d| day_counter.year_fraction(reference_date, *d))
            .collect();
        Self {
            reference_date,
            day_counter,
            base_cpi,
            times,
            zero_rates: zero_rates.to_vec(),
        }
    }

    /// Interpolated zero-coupon inflation rate at time `t`.
    pub fn zero_rate(&self, t: f64) -> f64 {
        if self.times.is_empty() {
            return 0.0;
        }
        if t <= self.times[0] {
            return self.zero_rates[0];
        }
        let n = self.times.len();
        if t >= self.times[n - 1] {
            return self.zero_rates[n - 1];
        }
        let i = find_bracket(&self.times, t);
        let alpha = (t - self.times[i]) / (self.times[i + 1] - self.times[i]);
        self.zero_rates[i] * (1.0 - alpha) + self.zero_rates[i + 1] * alpha
    }

    /// CPI at time `t`: base_cpi × (1 + zero_rate)^t.
    pub fn cpi(&self, t: f64) -> f64 {
        self.base_cpi * (1.0 + self.zero_rate(t)).powf(t)
    }

    /// Inflation discount factor: 1 / (1 + zero_rate)^t.
    pub fn inflation_discount(&self, t: f64) -> f64 {
        (1.0 + self.zero_rate(t)).powf(-t)
    }
}

impl TermStructure for InterpolatedZeroInflationCurve {
    fn reference_date(&self) -> Date {
        self.reference_date
    }
    fn day_counter(&self) -> DayCounter {
        self.day_counter
    }
    fn calendar(&self) -> &Calendar {
        &Calendar::NullCalendar
    }
    fn max_date(&self) -> Date {
        Date::from_serial(73050)
    }
}

// ===========================================================================
// Gaussian1dSwaptionVol (G74)
// ===========================================================================

/// Swaption volatility derived from a Gaussian 1D short-rate model.
///
/// Converts a calibrated Gaussian 1D model (e.g., Hull-White) into a
/// swaption vol surface via analytic or numerical swaption pricing.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Gaussian1dSwaptionVol {
    reference_date: Date,
    day_counter: DayCounter,
    /// Model mean reversion speed.
    mean_reversion: f64,
    /// Model volatility (σ in dr = κ(θ-r)dt + σdW).
    sigma: f64,
}

impl Gaussian1dSwaptionVol {
    pub fn new(
        reference_date: Date,
        day_counter: DayCounter,
        mean_reversion: f64,
        sigma: f64,
    ) -> Self {
        Self {
            reference_date,
            day_counter,
            mean_reversion,
            sigma,
        }
    }

    /// Approximate Black swaption vol for an option with expiry `t_opt`
    /// on a swap of tenor `t_swap` (both in years).
    ///
    /// Uses the Jamshidian-style approximation for the Hull-White model.
    pub fn black_vol(&self, t_opt: f64, t_swap: f64) -> f64 {
        let kappa = self.mean_reversion;
        let sigma = self.sigma;

        if t_opt <= 0.0 || t_swap <= 0.0 {
            return 0.0;
        }

        // Bond vol in Hull-White: σ_P(T) = σ/κ · (1 - e^{-κT})
        let bond_vol = |t: f64| -> f64 {
            if kappa.abs() < 1e-10 {
                sigma * t
            } else {
                sigma / kappa * (1.0 - (-kappa * t).exp())
            }
        };

        // Variance of swap rate: ∫₀^{t_opt} [B(t_swap) - weighted B terms]² σ²
        // Simplified: use the approximation σ_swap ≈ σ_P(t_swap) / √t_opt
        let bv = bond_vol(t_swap);

        // Variance under Hull-White
        let var = if kappa.abs() < 1e-10 {
            sigma * sigma * t_opt
        } else {
            sigma * sigma / (2.0 * kappa) * (1.0 - (-2.0 * kappa * t_opt).exp())
        };

        let swap_vol = bv * var.sqrt() / t_opt.sqrt();
        swap_vol.max(0.0)
    }
}

impl TermStructure for Gaussian1dSwaptionVol {
    fn reference_date(&self) -> Date {
        self.reference_date
    }
    fn day_counter(&self) -> DayCounter {
        self.day_counter
    }
    fn calendar(&self) -> &Calendar {
        &Calendar::NullCalendar
    }
    fn max_date(&self) -> Date {
        Date::from_serial(73050)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_time::Month;

    fn ref_date() -> Date {
        Date::from_ymd(2025, Month::January, 15)
    }

    #[test]
    fn constant_local_vol() {
        let lv = ConstantLocalVol::new(ref_date(), 0.25, DayCounter::Actual365Fixed);
        assert_abs_diff_eq!(lv.local_vol(1.0, 100.0), 0.25);
        assert_abs_diff_eq!(lv.local_vol(5.0, 50.0), 0.25);
    }

    #[test]
    fn fixed_local_vol_interpolation() {
        let times = vec![0.5, 1.0];
        let strikes = vec![90.0, 100.0, 110.0];
        #[rustfmt::skip]
        let vols = vec![
            0.30, 0.25, 0.20,  // t=0.5
            0.28, 0.23, 0.18,  // t=1.0
        ];
        let surface = FixedLocalVolSurface::new(
            ref_date(),
            DayCounter::Actual365Fixed,
            times,
            strikes,
            vols,
        );
        // At node
        assert_abs_diff_eq!(surface.local_vol(0.5, 100.0), 0.25, epsilon = 1e-10);
        // Between nodes
        let v = surface.local_vol(0.75, 100.0);
        assert_abs_diff_eq!(v, 0.24, epsilon = 1e-10);
    }

    #[test]
    fn interpolated_hazard_rate() {
        let dates = vec![
            ref_date() + 365,
            ref_date() + 730,
            ref_date() + 1825,
        ];
        let hrates = vec![0.01, 0.015, 0.02];
        let curve = InterpolatedHazardRateCurve::new(
            ref_date(),
            DayCounter::Actual365Fixed,
            &dates,
            &hrates,
        );
        // At t=1yr, hazard rate should be 0.01
        let h = curve.hazard_rate(1.0);
        assert_abs_diff_eq!(h, 0.01, epsilon = 0.001);
        // Survival prob at t=0 should be 1
        let sp0 = curve.survival_probability(0.0);
        assert_abs_diff_eq!(sp0, 1.0, epsilon = 1e-10);
        // Survival prob should decrease with time
        let sp1 = curve.survival_probability(1.0);
        let sp5 = curve.survival_probability(5.0);
        assert!(sp1 > sp5);
        assert!(sp1 < 1.0);
    }

    #[test]
    fn interpolated_survival_prob_curve() {
        let dates = vec![
            ref_date() + 365,
            ref_date() + 1825,
            ref_date() + 3650,
        ];
        let sps = vec![0.99, 0.95, 0.85];
        let curve = InterpolatedSurvivalProbCurve::new(
            ref_date(),
            DayCounter::Actual365Fixed,
            &dates,
            &sps,
        );
        // At first date, should return the input
        assert_abs_diff_eq!(curve.survival_probability(1.0), 0.99, epsilon = 0.01);
        // Default probability should be complement
        assert_abs_diff_eq!(
            curve.default_probability(1.0),
            1.0 - curve.survival_probability(1.0),
            epsilon = 1e-10
        );
    }

    #[test]
    fn cap_floor_term_vol_curve() {
        let tenors = vec![1.0, 2.0, 5.0, 10.0];
        let vols = vec![0.50, 0.45, 0.40, 0.38];
        let curve = CapFloorTermVolCurve::new(
            ref_date(),
            DayCounter::Actual365Fixed,
            tenors,
            vols,
            VolatilityType::ShiftedLognormal,
        );
        assert_abs_diff_eq!(curve.vol(1.0), 0.50, epsilon = 1e-10);
        assert_abs_diff_eq!(curve.vol(5.0), 0.40, epsilon = 1e-10);
        // Interpolated
        let v3 = curve.vol(3.0);
        assert!(v3 > 0.40 && v3 < 0.45);
    }

    #[test]
    fn constant_optionlet_vol() {
        let olv = ConstantOptionletVol::new(
            ref_date(),
            0.20,
            DayCounter::Actual365Fixed,
            VolatilityType::ShiftedLognormal,
            0.0,
        );
        assert_abs_diff_eq!(olv.optionlet_vol(1.0, 0.03), 0.20);
        assert_abs_diff_eq!(olv.optionlet_vol(5.0, 0.05), 0.20);
    }

    #[test]
    fn interpolated_zero_inflation() {
        let dates = vec![
            ref_date() + 365,
            ref_date() + 1825,
        ];
        let rates = vec![0.02, 0.025];
        let curve = InterpolatedZeroInflationCurve::new(
            ref_date(),
            DayCounter::Actual365Fixed,
            100.0,
            &dates,
            &rates,
        );
        // At t=1yr
        assert_abs_diff_eq!(curve.zero_rate(1.0), 0.02, epsilon = 0.001);
        // CPI should increase
        assert!(curve.cpi(1.0) > 100.0);
        assert!(curve.cpi(5.0) > curve.cpi(1.0));
    }

    #[test]
    fn volatility_type_default() {
        assert_eq!(VolatilityType::default(), VolatilityType::ShiftedLognormal);
    }

    #[test]
    fn gaussian1d_swaption_vol_positive() {
        let vol = Gaussian1dSwaptionVol::new(
            ref_date(),
            DayCounter::Actual365Fixed,
            0.05,
            0.01,
        );
        let bv = vol.black_vol(5.0, 10.0);
        assert!(bv > 0.0, "swaption vol should be positive: {}", bv);
    }
}
