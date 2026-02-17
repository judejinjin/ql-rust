//! Volatility term structure traits and implementations.
//!
//! Provides Black volatility and local volatility term structures.

use crate::term_structure::TermStructure;

// =========================================================================
// Black vol term structure trait
// =========================================================================

/// Black volatility term structure: maps (time, strike) → Black vol / variance.
pub trait BlackVolTermStructure: TermStructure {
    /// Black volatility at time `t` (years) and `strike`.
    fn black_vol(&self, t: f64, strike: f64) -> f64;

    /// Black variance = σ²·t at time `t` and `strike`.
    fn black_variance(&self, t: f64, strike: f64) -> f64 {
        let v = self.black_vol(t, strike);
        v * v * t
    }
}

// =========================================================================
// Local vol term structure trait
// =========================================================================

/// Local volatility term structure (Dupire): maps (t, underlying) → local vol.
pub trait LocalVolTermStructure: TermStructure {
    /// Local volatility at time `t` and underlying level `underlying`.
    fn local_vol(&self, t: f64, underlying: f64) -> f64;
}

// =========================================================================
// BlackConstantVol
// =========================================================================

use ql_time::{Calendar, Date, DayCounter};

/// Flat (constant) Black volatility surface.
#[derive(Debug, Clone)]
pub struct BlackConstantVol {
    reference_date: Date,
    day_counter: DayCounter,
    vol: f64,
}

impl BlackConstantVol {
    /// Create a constant Black vol surface.
    pub fn new(reference_date: Date, vol: f64, day_counter: DayCounter) -> Self {
        Self {
            reference_date,
            day_counter,
            vol,
        }
    }

    /// The constant volatility value.
    pub fn volatility(&self) -> f64 {
        self.vol
    }
}

impl TermStructure for BlackConstantVol {
    fn reference_date(&self) -> Date {
        self.reference_date
    }
    fn day_counter(&self) -> DayCounter {
        self.day_counter
    }
    fn calendar(&self) -> Calendar {
        Calendar::NullCalendar
    }
    fn max_date(&self) -> Date {
        Date::from_serial(73050) // ~2099-12-31
    }
}

impl BlackVolTermStructure for BlackConstantVol {
    fn black_vol(&self, _t: f64, _strike: f64) -> f64 {
        self.vol
    }
}

// =========================================================================
// BlackVarianceSurface
// =========================================================================

use ql_math::interpolation::{Interpolation, LinearInterpolation};

/// Interpolated Black variance surface on a (strike × expiry) grid.
///
/// Stores variances σ²·t at grid points and uses bilinear interpolation.
/// Input `vols` is indexed as `vols[expiry_idx][strike_idx]`.
#[derive(Debug, Clone)]
pub struct BlackVarianceSurface {
    reference_date: Date,
    day_counter: DayCounter,
    strikes: Vec<f64>,
    expiries: Vec<f64>, // in year fractions
    /// Variance grid: variances[i][j] = vol[i][j]² × expiries[i]
    variances: Vec<Vec<f64>>,
}

impl BlackVarianceSurface {
    /// Construct from a grid of Black vols.
    ///
    /// - `strikes`: sorted strike values
    /// - `expiries_yf`: sorted year fractions
    /// - `vols`: vols\[expiry\]\[strike\], same dimensions as (expiries × strikes)
    pub fn new(
        reference_date: Date,
        day_counter: DayCounter,
        strikes: Vec<f64>,
        expiries_yf: Vec<f64>,
        vols: &[Vec<f64>],
    ) -> Self {
        assert_eq!(vols.len(), expiries_yf.len());
        let variances: Vec<Vec<f64>> = vols
            .iter()
            .enumerate()
            .map(|(i, row)| {
                assert_eq!(row.len(), strikes.len());
                row.iter().map(|&v| v * v * expiries_yf[i]).collect()
            })
            .collect();
        Self {
            reference_date,
            day_counter,
            strikes,
            expiries: expiries_yf,
            variances,
        }
    }

    /// Bilinear interpolation of variance, then convert to vol.
    fn interpolate_variance(&self, t: f64, strike: f64) -> f64 {
        // Find expiry bracket
        let ne = self.expiries.len();
        let ns = self.strikes.len();

        // Clamp to grid boundaries
        let t_clamped = t.clamp(self.expiries[0], self.expiries[ne - 1]);
        let k_clamped = strike.clamp(self.strikes[0], self.strikes[ns - 1]);

        // Interpolate along strike for each expiry, then interpolate along time
        let mut var_at_t_per_expiry = Vec::with_capacity(ne);
        for i in 0..ne {
            let interp = LinearInterpolation::new(
                self.strikes.clone(),
                self.variances[i].clone(),
            )
            .unwrap();
            var_at_t_per_expiry.push(interp.value(k_clamped).unwrap());
        }

        // Interpolate along time
        let time_interp = LinearInterpolation::new(
            self.expiries.clone(),
            var_at_t_per_expiry,
        )
        .unwrap();
        time_interp.value(t_clamped).unwrap()
    }
}

impl TermStructure for BlackVarianceSurface {
    fn reference_date(&self) -> Date {
        self.reference_date
    }
    fn day_counter(&self) -> DayCounter {
        self.day_counter
    }
    fn calendar(&self) -> Calendar {
        Calendar::NullCalendar
    }
    fn max_date(&self) -> Date {
        Date::from_serial(73050)
    }
}

impl BlackVolTermStructure for BlackVarianceSurface {
    fn black_vol(&self, t: f64, strike: f64) -> f64 {
        if t <= 0.0 {
            // At t=0, return the shortest-expiry vol
            let interp = LinearInterpolation::new(
                self.strikes.clone(),
                self.variances[0]
                    .iter()
                    .map(|v| (v / self.expiries[0]).sqrt())
                    .collect(),
            )
            .unwrap();
            let k = strike.clamp(self.strikes[0], *self.strikes.last().unwrap());
            return interp.value(k).unwrap();
        }
        let var = self.interpolate_variance(t, strike);
        (var / t).max(0.0).sqrt()
    }

    fn black_variance(&self, t: f64, strike: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        self.interpolate_variance(t, strike).max(0.0)
    }
}

// =========================================================================
// LocalVolSurface (Dupire)
// =========================================================================

use std::sync::Arc;
use crate::yield_term_structure::YieldTermStructure;

/// Local volatility surface derived from a Black vol surface via Dupire's formula.
///
/// σ_local²(t,S) = [∂w/∂t + (r-q)·K·∂w/∂K + ½·∂²w/∂K² · ???]
///
/// Simplified Dupire formula (no dividends, continuous):
///   σ_local²(K,T) = [∂w/∂T] / [1 - y/w·∂w/∂y + ¼(-¼ - 1/w + y²/w²)(∂w/∂y)² + ½·∂²w/∂y²]
///
/// where w = σ²T (total variance), y = ln(K/F).
///
/// We use a simpler finite-difference approximation directly on Black vol.
pub struct LocalVolSurface {
    reference_date: Date,
    day_counter: DayCounter,
    black_vol: Arc<dyn BlackVolTermStructure>,
    risk_free: Arc<dyn YieldTermStructure>,
    dividend_yield: Arc<dyn YieldTermStructure>,
    spot: f64,
}

impl LocalVolSurface {
    pub fn new(
        black_vol: Arc<dyn BlackVolTermStructure>,
        risk_free: Arc<dyn YieldTermStructure>,
        dividend_yield: Arc<dyn YieldTermStructure>,
        spot: f64,
    ) -> Self {
        let reference_date = black_vol.reference_date();
        let day_counter = black_vol.day_counter();
        Self {
            reference_date,
            day_counter,
            black_vol,
            risk_free,
            dividend_yield,
            spot,
        }
    }
}

impl TermStructure for LocalVolSurface {
    fn reference_date(&self) -> Date {
        self.reference_date
    }
    fn day_counter(&self) -> DayCounter {
        self.day_counter
    }
    fn calendar(&self) -> Calendar {
        Calendar::NullCalendar
    }
    fn max_date(&self) -> Date {
        Date::from_serial(73050)
    }
}

impl LocalVolTermStructure for LocalVolSurface {
    /// Compute local vol via Dupire's formula using finite differences on the
    /// Black variance surface.
    fn local_vol(&self, t: f64, underlying: f64) -> f64 {
        let t = t.max(1e-6);
        let strike = underlying; // evaluate at K = S_local

        // Forward price
        let df_r = self.risk_free.discount_t(t);
        let df_q = self.dividend_yield.discount_t(t);
        let forward = self.spot * df_q / df_r;

        // Total variance w(T,K) = σ(T,K)² · T
        let eps_t = 1e-4_f64;
        let eps_k = strike * 1e-4_f64;

        let w = |tt: f64, kk: f64| -> f64 {
            let v = self.black_vol.black_vol(tt.max(1e-8), kk);
            v * v * tt.max(1e-8)
        };

        // ∂w/∂T
        let dw_dt = (w(t + eps_t, strike) - w(t - eps_t.min(t - 1e-8), strike))
            / (2.0 * eps_t.min(t - 1e-8 + eps_t));

        // ∂w/∂K, ∂²w/∂K²
        let w0 = w(t, strike);
        let wp = w(t, strike + eps_k);
        let wm = w(t, strike - eps_k);
        let dw_dk = (wp - wm) / (2.0 * eps_k);
        let d2w_dk2 = (wp - 2.0 * w0 + wm) / (eps_k * eps_k);

        // Dupire formula:
        // σ_local² = dw/dT / [1 - (y/w)·dw/dy + ¼(-¼ - 1/w + y²/w²)(dw/dy)² + ½·d²w/dy²]
        // where y = ln(K/F), dy = dK/K
        let y = (strike / forward).ln();
        let dw_dy = dw_dk * strike;
        let d2w_dy2 = d2w_dk2 * strike * strike + dw_dk * strike;

        let denominator =
            1.0 - y / w0 * dw_dy
                + 0.25 * (-0.25 - 1.0 / w0 + y * y / (w0 * w0)) * dw_dy * dw_dy
                + 0.5 * d2w_dy2;

        let local_var = if denominator > 0.0 {
            dw_dt / denominator
        } else {
            // Fallback to Black vol
            let v = self.black_vol.black_vol(t, strike);
            v * v
        };

        local_var.max(0.0).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ql_time::Month;

    #[test]
    fn constant_vol_returns_same_for_any_strike_and_time() {
        let ref_date = Date::from_ymd(2025, Month::January, 15);
        let cv = BlackConstantVol::new(ref_date, 0.2, DayCounter::Actual365Fixed);
        assert_abs_diff_eq!(cv.black_vol(0.5, 100.0), 0.2);
        assert_abs_diff_eq!(cv.black_vol(2.0, 50.0), 0.2);
        assert_abs_diff_eq!(cv.black_vol(0.01, 200.0), 0.2);
    }

    #[test]
    fn constant_vol_variance_scales_with_time() {
        let ref_date = Date::from_ymd(2025, Month::January, 15);
        let cv = BlackConstantVol::new(ref_date, 0.3, DayCounter::Actual365Fixed);
        assert_abs_diff_eq!(cv.black_variance(1.0, 100.0), 0.09, epsilon = 1e-12);
        assert_abs_diff_eq!(cv.black_variance(2.0, 100.0), 0.18, epsilon = 1e-12);
    }

    #[test]
    fn variance_surface_interpolates_correctly() {
        let ref_date = Date::from_ymd(2025, Month::January, 15);
        let strikes = vec![90.0, 100.0, 110.0];
        let expiries = vec![0.25, 0.5, 1.0];
        // Flat 20% vol across the surface
        let vols = vec![
            vec![0.20, 0.20, 0.20],
            vec![0.20, 0.20, 0.20],
            vec![0.20, 0.20, 0.20],
        ];
        let surface = BlackVarianceSurface::new(
            ref_date,
            DayCounter::Actual365Fixed,
            strikes,
            expiries,
            &vols,
        );
        // Flat vol should be recovered at any point
        assert_abs_diff_eq!(surface.black_vol(0.5, 100.0), 0.20, epsilon = 1e-10);
        assert_abs_diff_eq!(surface.black_vol(0.75, 95.0), 0.20, epsilon = 1e-6);
    }

    #[test]
    fn variance_surface_smile() {
        let ref_date = Date::from_ymd(2025, Month::January, 15);
        let strikes = vec![80.0, 100.0, 120.0];
        let expiries = vec![0.5, 1.0];
        // Smile: higher vol at wings
        let vols = vec![
            vec![0.25, 0.20, 0.24],
            vec![0.24, 0.19, 0.23],
        ];
        let surface = BlackVarianceSurface::new(
            ref_date,
            DayCounter::Actual365Fixed,
            strikes,
            expiries,
            &vols,
        );
        // ATM vol should be lower than wing
        let atm_vol = surface.black_vol(0.5, 100.0);
        let wing_vol = surface.black_vol(0.5, 80.0);
        assert!(wing_vol > atm_vol);
    }

    #[test]
    fn local_vol_from_constant_black_vol() {
        use crate::yield_curves::FlatForward;

        let ref_date = Date::from_ymd(2025, Month::January, 15);
        let vol = 0.25;
        let black_vol = Arc::new(BlackConstantVol::new(
            ref_date,
            vol,
            DayCounter::Actual365Fixed,
        ));
        let r_curve = Arc::new(FlatForward::new(ref_date, 0.05, DayCounter::Actual365Fixed));
        let q_curve = Arc::new(FlatForward::new(ref_date, 0.02, DayCounter::Actual365Fixed));
        let spot = 100.0;

        let local_vol_surface = LocalVolSurface::new(
            black_vol,
            r_curve as Arc<dyn YieldTermStructure>,
            q_curve as Arc<dyn YieldTermStructure>,
            spot,
        );

        // For constant Black vol, local vol should equal Black vol
        let lv = local_vol_surface.local_vol(0.5, 100.0);
        assert_abs_diff_eq!(lv, vol, epsilon = 0.01);
    }

    #[test]
    fn local_vol_positive() {
        use crate::yield_curves::FlatForward;

        let ref_date = Date::from_ymd(2025, Month::January, 15);
        let black_vol = Arc::new(BlackConstantVol::new(
            ref_date,
            0.30,
            DayCounter::Actual365Fixed,
        ));
        let r_curve = Arc::new(FlatForward::new(ref_date, 0.03, DayCounter::Actual365Fixed));
        let q_curve = Arc::new(FlatForward::new(ref_date, 0.0, DayCounter::Actual365Fixed));

        let local_vol_surface = LocalVolSurface::new(
            black_vol,
            r_curve as Arc<dyn YieldTermStructure>,
            q_curve as Arc<dyn YieldTermStructure>,
            100.0,
        );

        // Local vol should be positive for various spots and times
        for &t in &[0.1, 0.5, 1.0, 2.0] {
            for &s in &[80.0, 100.0, 120.0] {
                let lv = local_vol_surface.local_vol(t, s);
                assert!(lv > 0.0, "Local vol should be positive at t={t}, S={s}");
            }
        }
    }
}
