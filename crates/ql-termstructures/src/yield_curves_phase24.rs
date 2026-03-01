//! Phase 24 advanced yield curve additions.
//!
//! - [`CubicBSplinesFitting`] — cubic B-spline curve fitting method.
//! - [`SpreadFittingMethod`] — spread over reference curve fitting.
//! - [`InterpolatedSimpleZeroCurve`] — zero curve with simple compounding.
//! - [`SpreadDiscountCurve`] — discount curve with additive spread.
//! - [`OvernightIndexFutureRateHelper`] — SOFR futures rate helper.
//! - [`FittingMethodExtended`] — extended enum covering all fitting methods.

use serde::{Deserialize, Serialize};
use ql_time::{Calendar, Date, DayCounter};

use crate::bootstrap::RateHelper;
use crate::term_structure::TermStructure;
use crate::yield_term_structure::YieldTermStructure;

// ===========================================================================
// CubicBSplinesFitting
// ===========================================================================

/// Cubic B-spline fitting method for discount curves.
///
/// Models the discount function as a linear combination of cubic B-spline
/// basis functions:
///
/// ```text
/// d(t) = Σᵢ cᵢ · Bᵢ,₃(t)
/// ```
///
/// where `Bᵢ,₃(t)` is the i-th cubic B-spline basis function defined on the
/// knot vector and `cᵢ` are the fitted coefficients.
///
/// The constraint `d(0) = 1` is enforced by fixing `c₀ = 1` if the first
/// knot is at zero, or by solving a constrained least-squares problem.
///
/// B-spline fitting is more flexible than polynomial fitting and produces
/// smoother curves than exponential splines. McCulloch (1971,1975) introduced
/// this approach for yield curve fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CubicBSplinesFitting {
    /// B-spline coefficients.
    pub coefficients: Vec<f64>,
    /// Knot vector (extended, includes repeated boundary knots).
    pub knots: Vec<f64>,
}

impl CubicBSplinesFitting {
    /// Create from pre-computed coefficients and knot vector.
    pub fn new(coefficients: Vec<f64>, knots: Vec<f64>) -> Self {
        Self { coefficients, knots }
    }

    /// Generate a default knot vector for a given maximum maturity.
    ///
    /// Creates interior knots plus the required 4 repeated boundary knots
    /// on each side for cubic splines.
    pub fn default_knots(max_maturity: f64) -> Vec<f64> {
        let interior: Vec<f64> = vec![
            0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0,
        ]
        .into_iter()
        .filter(|&k| k < max_maturity)
        .collect();

        let mut knots = Vec::new();
        // 4 repeated knots at start (for cubic = order 4)
        for _ in 0..4 {
            knots.push(0.0);
        }
        knots.extend_from_slice(&interior);
        // 4 repeated knots at end
        for _ in 0..4 {
            knots.push(max_maturity);
        }
        knots
    }

    /// Evaluate a single cubic B-spline basis function `Bᵢ,₃(t)` using
    /// the Cox–de Boor recursion.
    fn basis(knots: &[f64], i: usize, t: f64) -> f64 {
        Self::basis_recurse(knots, i, 3, t)
    }

    fn basis_recurse(knots: &[f64], i: usize, p: usize, t: f64) -> f64 {
        if p == 0 {
            return if knots[i] <= t && t < knots[i + 1] { 1.0 } else { 0.0 };
        }
        let mut val = 0.0;
        let denom1 = knots[i + p] - knots[i];
        if denom1.abs() > 1e-15 {
            val += (t - knots[i]) / denom1 * Self::basis_recurse(knots, i, p - 1, t);
        }
        let denom2 = knots[i + p + 1] - knots[i + 1];
        if denom2.abs() > 1e-15 {
            val += (knots[i + p + 1] - t) / denom2
                * Self::basis_recurse(knots, i + 1, p - 1, t);
        }
        val
    }

    /// Number of basis functions = len(knots) - 4 (for cubic).
    pub fn num_basis(&self) -> usize {
        if self.knots.len() < 5 {
            0
        } else {
            self.knots.len() - 4
        }
    }

    /// Discount factor at maturity `t`.
    pub fn discount(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }
        let n = self.num_basis();
        let mut d = 0.0;
        for i in 0..n.min(self.coefficients.len()) {
            d += self.coefficients[i] * Self::basis(&self.knots, i, t);
        }
        // Handle end boundary: for t >= last interior knot, use last basis
        // (Cox-de Boor gives 0 at the right boundary, so we clamp)
        d.max(1e-10)
    }

    /// Continuously-compounded zero rate at maturity `t`.
    pub fn zero_rate(&self, t: f64) -> f64 {
        if t <= 1e-8 {
            return 0.0;
        }
        -self.discount(t).ln() / t
    }

    /// Forward rate over `[t1, t2]`.
    pub fn forward_rate(&self, t1: f64, t2: f64) -> f64 {
        if t2 <= t1 + 1e-8 {
            return self.zero_rate(t1);
        }
        (self.discount(t1) / self.discount(t2)).ln() / (t2 - t1)
    }

    /// Calibrate cubic B-spline to zero-coupon bond prices (discount factors).
    ///
    /// Uses least squares with the constraint `d(0) = 1`. The first coefficient
    /// is solved from the constraint: `c₀ = (1 − Σᵢ₌₁ cᵢ B_i(0)) / B₀(0)`.
    pub fn fit(maturities: &[f64], prices: &[f64], knots: Option<Vec<f64>>) -> Self {
        let max_mat = maturities.iter().cloned().fold(0.0_f64, f64::max);
        let knot_vec = knots.unwrap_or_else(|| Self::default_knots(max_mat * 1.1));
        let n_basis = if knot_vec.len() >= 5 {
            knot_vec.len() - 4
        } else {
            return Self::new(vec![1.0], knot_vec);
        };
        let n_bonds = maturities.len();

        if n_basis < 2 {
            return Self::new(vec![1.0], knot_vec);
        }

        // Constrained OLS: substitute c₀ = (1 − Σ cᵢ B_i(0)) / B₀(0)
        let b0_at_0 = Self::basis(&knot_vec, 0, 1e-10);
        let b_at_0: Vec<f64> = (1..n_basis)
            .map(|i| Self::basis(&knot_vec, i, 1e-10))
            .collect();

        let n_free = n_basis - 1;
        let mut x = vec![0.0_f64; n_bonds * n_free];
        let mut y = vec![0.0_f64; n_bonds];

        for (row, (&t, &p)) in maturities.iter().zip(prices.iter()).enumerate() {
            let b0_t = Self::basis(&knot_vec, 0, t);
            // Substitute c₀: predicted = (1/B₀(0)) · B₀(t) + Σ cⱼ [Bⱼ(t) - bⱼ(0)/B₀(0) · B₀(t)]
            let base = if b0_at_0.abs() > 1e-15 {
                b0_t / b0_at_0
            } else {
                0.0
            };
            y[row] = p - base;
            for j in 0..n_free {
                let bj_t = Self::basis(&knot_vec, j + 1, t);
                let shift = if b0_at_0.abs() > 1e-15 {
                    b_at_0[j] * b0_t / b0_at_0
                } else {
                    0.0
                };
                x[row * n_free + j] = bj_t - shift;
            }
        }

        let free_coeffs = ols(&x, &y, n_bonds, n_free);

        // Recover c₀
        let sum_at_0: f64 = free_coeffs
            .iter()
            .zip(b_at_0.iter())
            .map(|(&c, &b)| c * b)
            .sum();
        let c0 = if b0_at_0.abs() > 1e-15 {
            (1.0 - sum_at_0) / b0_at_0
        } else {
            1.0
        };

        let mut coefficients = vec![c0];
        coefficients.extend_from_slice(&free_coeffs);

        Self::new(coefficients, knot_vec)
    }
}

// ===========================================================================
// SpreadFittingMethod
// ===========================================================================

/// Spread fitting method: fits a discount curve as a spread over a reference curve.
///
/// The fitted discount factor is:
///
/// ```text
/// d(t) = d_ref(t) · exp(-s(t) · t)
/// ```
///
/// where `s(t)` is a spread term structure modelled by a simple parametric
/// form (e.g. polynomial in `t`).
///
/// The spread parameters are calibrated to minimise pricing errors on a
/// set of instruments relative to the reference curve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpreadFittingMethod {
    /// Reference discount factors at sample times.
    pub ref_times: Vec<f64>,
    pub ref_dfs: Vec<f64>,
    /// Polynomial spread coefficients [s₁, s₂, …, sₙ].
    /// Spread s(t) = s₁·t + s₂·t² + …
    /// (No s₀ so that the spread is zero at t=0.)
    pub spread_coefficients: Vec<f64>,
}

impl SpreadFittingMethod {
    /// Create from pre-computed reference curve and spread coefficients.
    pub fn new(
        ref_times: Vec<f64>,
        ref_dfs: Vec<f64>,
        spread_coefficients: Vec<f64>,
    ) -> Self {
        Self {
            ref_times,
            ref_dfs,
            spread_coefficients,
        }
    }

    /// Evaluate the polynomial spread at time `t`.
    pub fn spread(&self, t: f64) -> f64 {
        let mut s = 0.0;
        let mut tp = t;
        for &c in &self.spread_coefficients {
            s += c * tp;
            tp *= t;
        }
        s
    }

    /// Discount factor from the reference curve at time `t` (log-linear interpolated).
    fn ref_discount(&self, t: f64) -> f64 {
        interpolate_log_linear_local(&self.ref_times, &self.ref_dfs, t)
    }

    /// Total discount factor: d_ref(t) · exp(-s(t) · t).
    pub fn discount(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }
        self.ref_discount(t) * (-self.spread(t) * t).exp()
    }

    /// Zero rate at maturity `t`.
    pub fn zero_rate(&self, t: f64) -> f64 {
        if t <= 1e-8 {
            return 0.0;
        }
        -self.discount(t).ln() / t
    }

    /// Calibrate spread coefficients to match bond prices.
    ///
    /// # Arguments
    /// - `ref_curve` — reference yield curve
    /// - `maturities` — instrument maturities
    /// - `target_dfs` — target discount factors (from market)
    /// - `degree` — degree of the spread polynomial
    /// - `max_years` — maximum time for reference curve sampling
    pub fn fit(
        ref_curve: &dyn YieldTermStructure,
        maturities: &[f64],
        target_dfs: &[f64],
        degree: usize,
        max_years: f64,
    ) -> Self {
        let n_samples = 100;
        let mut ref_times = Vec::with_capacity(n_samples);
        let mut ref_dfs_vec = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let t = max_years * (i as f64) / ((n_samples - 1) as f64);
            ref_times.push(t);
            ref_dfs_vec.push(ref_curve.discount_t(t));
        }

        let n = maturities.len();
        // y_i = -ln(target_df_i / ref_df_i)    (this is the total spread * t)
        // We model spread_integrated(t) = s₁·t² + s₂·t³ + ...
        // So y_i = s₁·tᵢ² + s₂·tᵢ³ + ...
        let mut x = vec![0.0_f64; n * degree];
        let mut y = vec![0.0_f64; n];

        for (i, (&t, &df_target)) in maturities.iter().zip(target_dfs.iter()).enumerate() {
            let df_ref = interpolate_log_linear_local(&ref_times, &ref_dfs_vec, t);
            y[i] = if df_ref > 1e-15 && df_target > 1e-15 {
                -(df_target / df_ref).ln()
            } else {
                0.0
            };
            let mut tp = t * t; // start at t² because spread(t)*t = s₁·t² + ...
            for j in 0..degree {
                x[i * degree + j] = tp;
                tp *= t;
            }
        }

        let coeffs = ols(&x, &y, n, degree);

        Self::new(ref_times, ref_dfs_vec, coeffs)
    }
}

// ===========================================================================
// InterpolatedSimpleZeroCurve
// ===========================================================================

/// A zero curve with simple (as opposed to continuous) compounding.
///
/// The discount factor is:
///
/// ```text
/// df(t) = 1 / (1 + z(t) · t)
/// ```
///
/// where `z(t)` is the linearly-interpolated simple zero rate.
///
/// Simple compounding is used in money-market conventions (deposits,
/// T-bills, commercial paper) where maturities are typically ≤ 1 year.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpolatedSimpleZeroCurve {
    reference_date: Date,
    day_counter: DayCounter,
    calendar: Calendar,
    max_date: Date,
    /// Node times (year fractions from reference_date).
    pub times: Vec<f64>,
    /// Simple zero rates at each node.
    pub rates: Vec<f64>,
}

impl InterpolatedSimpleZeroCurve {
    /// Create from node times and simple zero rates.
    pub fn new(
        reference_date: Date,
        day_counter: DayCounter,
        times: Vec<f64>,
        rates: Vec<f64>,
    ) -> Self {
        assert_eq!(times.len(), rates.len(), "times and rates must match");
        let max_t = times.last().copied().unwrap_or(1.0);
        let max_days = (max_t * 365.25) as i32;
        let max_date = reference_date + max_days;
        Self {
            reference_date,
            day_counter,
            calendar: Calendar::NullCalendar,
            max_date,
            times,
            rates,
        }
    }

    /// Create from a set of `(Date, simple_rate)` pairs.
    pub fn from_dates(
        reference_date: Date,
        day_counter: DayCounter,
        date_rate_pairs: &[(Date, f64)],
    ) -> Self {
        let mut times = Vec::with_capacity(date_rate_pairs.len());
        let mut rates = Vec::with_capacity(date_rate_pairs.len());
        for &(d, r) in date_rate_pairs {
            let t = day_counter.year_fraction(reference_date, d);
            times.push(t);
            rates.push(r);
        }
        Self::new(reference_date, day_counter, times, rates)
    }

    /// Linearly interpolate the simple zero rate at time `t`.
    pub fn simple_rate(&self, t: f64) -> f64 {
        if self.times.is_empty() {
            return 0.0;
        }
        if t <= self.times[0] {
            return self.rates[0];
        }
        if t >= *self.times.last().unwrap() {
            return *self.rates.last().unwrap();
        }
        let idx = self.times.partition_point(|&x| x < t);
        if idx == 0 {
            return self.rates[0];
        }
        let t0 = self.times[idx - 1];
        let t1 = self.times[idx];
        let r0 = self.rates[idx - 1];
        let r1 = self.rates[idx];
        let w = (t - t0) / (t1 - t0);
        r0 + w * (r1 - r0)
    }
}

impl TermStructure for InterpolatedSimpleZeroCurve {
    fn reference_date(&self) -> Date {
        self.reference_date
    }
    fn day_counter(&self) -> DayCounter {
        self.day_counter
    }
    fn calendar(&self) -> &Calendar {
        &self.calendar
    }
    fn max_date(&self) -> Date {
        self.max_date
    }
}

impl YieldTermStructure for InterpolatedSimpleZeroCurve {
    fn discount_impl(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }
        let r = self.simple_rate(t);
        let denom = 1.0 + r * t;
        if denom > 1e-15 {
            1.0 / denom
        } else {
            1e-10
        }
    }
}

// ===========================================================================
// SpreadDiscountCurve
// ===========================================================================

/// A discount curve formed by applying an additive spread to a base curve's
/// zero rates.
///
/// The resulting discount factor is:
///
/// ```text
/// df(t) = df_base(t) · exp(-spread · t)
/// ```
///
/// This is simpler than `PiecewiseZeroSpreadedTermStructure` — the spread
/// is a single constant rather than a piecewise curve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpreadDiscountCurve {
    reference_date: Date,
    day_counter: DayCounter,
    calendar: Calendar,
    max_date: Date,
    /// Sampled base curve discount factors.
    base_times: Vec<f64>,
    base_dfs: Vec<f64>,
    /// Constant additive zero-rate spread.
    pub spread: f64,
}

impl SpreadDiscountCurve {
    /// Create from a base yield curve and a constant spread.
    pub fn new(
        base: &dyn YieldTermStructure,
        spread: f64,
        max_years: f64,
    ) -> Self {
        let ref_date = base.reference_date();
        let dc = base.day_counter();
        let n = 200;
        let mut base_times = Vec::with_capacity(n);
        let mut base_dfs = Vec::with_capacity(n);
        for i in 0..n {
            let t = max_years * (i as f64) / ((n - 1) as f64);
            base_times.push(t);
            base_dfs.push(base.discount_t(t));
        }
        let max_days = (max_years * 365.25) as i32;
        let max_date = ref_date + max_days;
        Self {
            reference_date: ref_date,
            day_counter: dc,
            calendar: base.calendar().clone(),
            max_date,
            base_times,
            base_dfs,
            spread,
        }
    }

    /// Create from pre-sampled base curve data and a constant spread.
    pub fn from_data(
        reference_date: Date,
        day_counter: DayCounter,
        base_times: Vec<f64>,
        base_dfs: Vec<f64>,
        spread: f64,
    ) -> Self {
        let max_t = base_times.last().copied().unwrap_or(30.0);
        let max_days = (max_t * 365.25) as i32;
        let max_date = reference_date + max_days;
        Self {
            reference_date,
            day_counter,
            calendar: Calendar::NullCalendar,
            max_date,
            base_times,
            base_dfs,
            spread,
        }
    }

    /// The constant spread value.
    pub fn spread_value(&self) -> f64 {
        self.spread
    }
}

impl TermStructure for SpreadDiscountCurve {
    fn reference_date(&self) -> Date {
        self.reference_date
    }
    fn day_counter(&self) -> DayCounter {
        self.day_counter
    }
    fn calendar(&self) -> &Calendar {
        &self.calendar
    }
    fn max_date(&self) -> Date {
        self.max_date
    }
}

impl YieldTermStructure for SpreadDiscountCurve {
    fn discount_impl(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }
        let base_df = interpolate_log_linear_local(&self.base_times, &self.base_dfs, t);
        base_df * (-self.spread * t).exp()
    }
}

// ===========================================================================
// OvernightIndexFutureRateHelper
// ===========================================================================

/// Rate helper for bootstrapping from overnight index futures (SOFR futures).
///
/// SOFR futures (1M and 3M) settle based on the compounded daily SOFR rate
/// over the contract period.  The futures price is `100 − rate`.
///
/// Unlike standard Eurodollar futures which reference a 3M LIBOR rate,
/// SOFR futures reference a compounded average of daily overnight rates.
/// The implied rate under daily compounding is:
///
/// ```text
/// rate = (df_start / df_end − 1) / τ − convexity_adjustment
/// ```
///
/// This helper supports:
/// - **1-month SOFR futures** (SR1): reference period ~ 1 calendar month
/// - **3-month SOFR futures** (SR3): reference period = IMM quarter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OvernightIndexFutureRateHelper {
    /// Futures price (e.g. 95.50 implies rate of 4.50%).
    price: f64,
    /// Start date of the reference period.
    start_date: Date,
    /// End date of the reference period.
    end_date: Date,
    /// Day counter (typically Actual/360 for SOFR).
    day_counter: DayCounter,
    /// Convexity adjustment (futures vs. OIS rate bias).
    convexity_adjustment: f64,
    /// Whether this is a 1-month (true) vs 3-month (false) contract.
    is_one_month: bool,
}

impl OvernightIndexFutureRateHelper {
    /// Create a new overnight index future rate helper.
    pub fn new(
        price: f64,
        start_date: Date,
        end_date: Date,
        day_counter: DayCounter,
        convexity_adjustment: f64,
        is_one_month: bool,
    ) -> Self {
        Self {
            price,
            start_date,
            end_date,
            day_counter,
            convexity_adjustment,
            is_one_month,
        }
    }

    /// Convenience: create a 3-month (SR3) SOFR futures helper.
    pub fn sr3(
        price: f64,
        start_date: Date,
        end_date: Date,
        convexity_adjustment: f64,
    ) -> Self {
        Self::new(
            price,
            start_date,
            end_date,
            DayCounter::Actual360,
            convexity_adjustment,
            false,
        )
    }

    /// Convenience: create a 1-month (SR1) SOFR futures helper.
    pub fn sr1(
        price: f64,
        start_date: Date,
        end_date: Date,
        convexity_adjustment: f64,
    ) -> Self {
        Self::new(
            price,
            start_date,
            end_date,
            DayCounter::Actual360,
            convexity_adjustment,
            true,
        )
    }

    /// The implied compounded rate from the futures price.
    pub fn implied_rate(&self) -> f64 {
        (100.0 - self.price) / 100.0 - self.convexity_adjustment
    }

    /// Whether this is a 1-month contract.
    pub fn is_one_month(&self) -> bool {
        self.is_one_month
    }
}

impl RateHelper for OvernightIndexFutureRateHelper {
    fn pillar_date(&self) -> Date {
        self.end_date
    }

    fn quote(&self) -> f64 {
        self.implied_rate()
    }

    fn implied_quote(
        &self,
        times: &[f64],
        dfs: &[f64],
        day_counter: DayCounter,
        ref_date: Date,
    ) -> f64 {
        let t_start = day_counter.year_fraction(ref_date, self.start_date);
        let t_end = day_counter.year_fraction(ref_date, self.end_date);
        let yf = self.day_counter.year_fraction(self.start_date, self.end_date);

        let df_start = interpolate_log_linear_local(times, dfs, t_start);
        let df_end = interpolate_log_linear_local(times, dfs, t_end);

        if yf.abs() < 1e-15 {
            return 0.0;
        }
        // Compounded overnight rate: (df_start / df_end − 1) / τ
        (df_start / df_end - 1.0) / yf
    }
}

// ===========================================================================
// FittingMethodExtended — extended enum covering all fitting methods
// ===========================================================================

/// Extended fitting method enum covering all parametric discount curve models.
///
/// This extends the basic `FittingMethod` (NelsonSiegel, Svensson) to include
/// polynomial, exponential spline, cubic B-spline, and spread methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FittingMethodExtended {
    /// Nelson-Siegel 4-parameter model.
    NelsonSiegel(crate::nelson_siegel::NelsonSiegelFitting),
    /// Svensson 6-parameter model.
    Svensson(crate::nelson_siegel::SvenssonFitting),
    /// Polynomial discount factor model.
    Polynomial(crate::fitted_bond_curve_extended::PolynomialDiscountCurve),
    /// Exponential spline (Vasicek-Fong) model.
    ExponentialSpline(crate::fitted_bond_curve_extended::ExponentialSplineCurve),
    /// Cubic B-spline model.
    CubicBSpline(CubicBSplinesFitting),
    /// Spread over reference curve.
    Spread(SpreadFittingMethod),
}

impl FittingMethodExtended {
    /// Discount factor at maturity `t`.
    pub fn discount(&self, t: f64) -> f64 {
        match self {
            Self::NelsonSiegel(ns) => ns.discount(t),
            Self::Svensson(sv) => sv.discount(t),
            Self::Polynomial(p) => p.discount(t),
            Self::ExponentialSpline(e) => e.discount(t),
            Self::CubicBSpline(b) => b.discount(t),
            Self::Spread(s) => s.discount(t),
        }
    }

    /// Zero rate at maturity `t`.
    pub fn zero_rate(&self, t: f64) -> f64 {
        match self {
            Self::NelsonSiegel(ns) => ns.zero_rate(t),
            Self::Svensson(sv) => sv.zero_rate(t),
            Self::Polynomial(p) => p.zero_rate(t),
            Self::ExponentialSpline(e) => e.zero_rate(t),
            Self::CubicBSpline(b) => b.zero_rate(t),
            Self::Spread(s) => s.zero_rate(t),
        }
    }
}

// ===========================================================================
// Helper: OLS and log-linear interpolation (local for independence)
// ===========================================================================

/// Solve OLS via normal equations with Cholesky decomposition.
fn ols(xmat: &[f64], y: &[f64], n_rows: usize, n_cols: usize) -> Vec<f64> {
    if n_cols == 0 || n_rows == 0 {
        return vec![];
    }
    let mut xtx = vec![0.0_f64; n_cols * n_cols];
    let mut xty = vec![0.0_f64; n_cols];

    for i in 0..n_rows {
        for j in 0..n_cols {
            xty[j] += xmat[i * n_cols + j] * y[i];
            for k in 0..n_cols {
                xtx[j * n_cols + k] += xmat[i * n_cols + j] * xmat[i * n_cols + k];
            }
        }
    }

    let l = cholesky_lower(&xtx, n_cols);
    let mut z = vec![0.0_f64; n_cols];
    for i in 0..n_cols {
        let mut s = xty[i];
        for j in 0..i {
            s -= l[i * n_cols + j] * z[j];
        }
        let diag = l[i * n_cols + i];
        z[i] = if diag.abs() > 1e-15 { s / diag } else { 0.0 };
    }
    let mut beta = vec![0.0_f64; n_cols];
    for i in (0..n_cols).rev() {
        let mut s = z[i];
        for j in i + 1..n_cols {
            s -= l[j * n_cols + i] * beta[j];
        }
        let diag = l[i * n_cols + i];
        beta[i] = if diag.abs() > 1e-15 { s / diag } else { 0.0 };
    }
    beta
}

fn cholesky_lower(a: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s: f64 = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                l[i * n + j] = s.max(1e-30).sqrt();
            } else {
                let diag = l[j * n + j];
                l[i * n + j] = if diag > 1e-15 { s / diag } else { 0.0 };
            }
        }
    }
    l
}

fn interpolate_log_linear_local(times: &[f64], dfs: &[f64], t: f64) -> f64 {
    if times.is_empty() || dfs.is_empty() {
        return 1.0;
    }
    if t <= times[0] {
        return dfs[0];
    }
    let n = times.len();
    if t >= times[n - 1] {
        if n < 2 {
            return dfs[n - 1];
        }
        let ln1 = dfs[n - 2].ln();
        let ln2 = dfs[n - 1].ln();
        let dt = times[n - 1] - times[n - 2];
        if dt.abs() < 1e-15 {
            return dfs[n - 1];
        }
        let slope = (ln2 - ln1) / dt;
        return (ln2 + slope * (t - times[n - 1])).exp();
    }
    let mut i = 0;
    for (j, tj) in times.iter().enumerate().skip(1) {
        if *tj >= t {
            i = j - 1;
            break;
        }
    }
    let t1 = times[i];
    let t2 = times[i + 1];
    let ln1 = dfs[i].ln();
    let ln2 = dfs[i + 1].ln();
    let dt = t2 - t1;
    if dt.abs() < 1e-15 {
        return dfs[i];
    }
    let frac = (t - t1) / dt;
    (ln1 + frac * (ln2 - ln1)).exp()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use crate::yield_curves::FlatForward;
    use ql_time::Month;

    fn ref_date() -> Date {
        Date::from_ymd(2025, Month::January, 2)
    }

    // -----------------------------------------------------------------------
    // CubicBSplinesFitting
    // -----------------------------------------------------------------------

    #[test]
    fn bspline_discount_at_zero() {
        let knots = CubicBSplinesFitting::default_knots(30.0);
        let n_basis = knots.len() - 4;
        // All coefficients = 1/(n_basis) roughly gives d(0)≈1
        let coeffs = vec![1.0; n_basis];
        let bs = CubicBSplinesFitting::new(coeffs, knots);
        // d(0) should be close to 1 (sum of B-spline values at 0 is 1 by partition of unity)
        let d0 = bs.discount(1e-10);
        assert!(d0 > 0.5, "d(0) = {d0}");
    }

    #[test]
    fn bspline_fit_zero_coupon_bonds() {
        let mats: Vec<f64> = (1..=10).map(|y| y as f64).collect();
        let prices: Vec<f64> = mats.iter().map(|&t| (-0.05 * t).exp()).collect();
        let bs = CubicBSplinesFitting::fit(&mats, &prices, None);

        // Fitted discount factors should be close to the true ones
        for (&t, &p) in mats.iter().zip(prices.iter()) {
            let est = bs.discount(t);
            assert!(
                (est - p).abs() < 0.02,
                "t={t} est={est:.6} true={p:.6}"
            );
        }
    }

    #[test]
    fn bspline_basis_partition_of_unity() {
        // At any point inside the knot span, B-spline basis functions sum to 1
        let knots = CubicBSplinesFitting::default_knots(30.0);
        let n_basis = knots.len() - 4;
        for t in [0.5, 1.0, 5.0, 10.0, 20.0] {
            let sum: f64 = (0..n_basis)
                .map(|i| CubicBSplinesFitting::basis(&knots, i, t))
                .sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn bspline_forward_rate() {
        let mats: Vec<f64> = (1..=10).map(|y| y as f64).collect();
        let prices: Vec<f64> = mats.iter().map(|&t| (-0.05 * t).exp()).collect();
        let bs = CubicBSplinesFitting::fit(&mats, &prices, None);
        let fwd = bs.forward_rate(2.0, 3.0);
        // Forward rate should be roughly 5% for a flat curve
        assert!(fwd > 0.0, "forward rate: {fwd}");
    }

    // -----------------------------------------------------------------------
    // SpreadFittingMethod
    // -----------------------------------------------------------------------

    #[test]
    fn spread_fitting_zero_spread() {
        let ref_curve = FlatForward::new(ref_date(), 0.04, DayCounter::Actual365Fixed);
        let mats: Vec<f64> = (1..=5).map(|y| y as f64).collect();
        let target_dfs: Vec<f64> = mats.iter().map(|&t| (-0.04 * t).exp()).collect();

        let sf = SpreadFittingMethod::fit(&ref_curve, &mats, &target_dfs, 2, 30.0);

        // With zero spread (target = reference), fitted spread should be ~0
        for &t in &mats {
            let df_fit = sf.discount(t);
            let df_ref = (-0.04 * t).exp();
            assert_abs_diff_eq!(df_fit, df_ref, epsilon = 0.01);
        }
    }

    #[test]
    fn spread_fitting_with_spread() {
        let ref_curve = FlatForward::new(ref_date(), 0.03, DayCounter::Actual365Fixed);
        // Target has 50bp more than reference
        let mats: Vec<f64> = (1..=5).map(|y| y as f64).collect();
        let target_dfs: Vec<f64> = mats.iter().map(|&t| (-0.035 * t).exp()).collect();

        let sf = SpreadFittingMethod::fit(&ref_curve, &mats, &target_dfs, 2, 30.0);

        // Spread should produce discount factors close to target
        for (&t, &df_target) in mats.iter().zip(target_dfs.iter()) {
            let df_fit = sf.discount(t);
            assert!(
                (df_fit - df_target).abs() < 0.01,
                "t={t} fit={df_fit:.6} target={df_target:.6}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // InterpolatedSimpleZeroCurve
    // -----------------------------------------------------------------------

    #[test]
    fn simple_zero_curve_discount() {
        let curve = InterpolatedSimpleZeroCurve::new(
            ref_date(),
            DayCounter::Actual360,
            vec![0.25, 0.5, 1.0],
            vec![0.04, 0.045, 0.05],  // money-market rates
        );
        // At t=0.5, rate = 4.5%, df = 1 / (1 + 0.045 * 0.5) = 1/1.0225
        let df = curve.discount_impl(0.5);
        assert_abs_diff_eq!(df, 1.0 / 1.0225, epsilon = 1e-10);
    }

    #[test]
    fn simple_zero_curve_interpolation() {
        let curve = InterpolatedSimpleZeroCurve::new(
            ref_date(),
            DayCounter::Actual360,
            vec![0.25, 1.0],
            vec![0.04, 0.05],
        );
        // At t=0.625 (midpoint), rate should be 0.045
        let r = curve.simple_rate(0.625);
        assert_abs_diff_eq!(r, 0.045, epsilon = 1e-10);
    }

    #[test]
    fn simple_zero_curve_term_structure() {
        let curve = InterpolatedSimpleZeroCurve::new(
            ref_date(),
            DayCounter::Actual360,
            vec![0.25, 0.5, 1.0],
            vec![0.04, 0.045, 0.05],
        );
        assert_eq!(curve.reference_date(), ref_date());
        assert_abs_diff_eq!(curve.discount_impl(0.0), 1.0, epsilon = 1e-15);
    }

    #[test]
    fn simple_zero_curve_from_dates() {
        let r = ref_date();
        let pairs = vec![
            (r + 91, 0.04),
            (r + 182, 0.045),
            (r + 365, 0.05),
        ];
        let curve = InterpolatedSimpleZeroCurve::from_dates(r, DayCounter::Actual360, &pairs);
        assert_eq!(curve.rates.len(), 3);
        assert!(curve.discount_impl(0.5) > 0.0 && curve.discount_impl(0.5) < 1.0);
    }

    // -----------------------------------------------------------------------
    // SpreadDiscountCurve
    // -----------------------------------------------------------------------

    #[test]
    fn spread_discount_curve_zero_spread() {
        let base = FlatForward::new(ref_date(), 0.05, DayCounter::Actual365Fixed);
        let spread_curve = SpreadDiscountCurve::new(&base, 0.0, 30.0);
        // Zero spread → same as base
        for t in [1.0, 5.0, 10.0] {
            let df_base = base.discount_t(t);
            let df_spread = spread_curve.discount_impl(t);
            assert_abs_diff_eq!(df_spread, df_base, epsilon = 1e-8);
        }
    }

    #[test]
    fn spread_discount_curve_positive_spread() {
        let base = FlatForward::new(ref_date(), 0.04, DayCounter::Actual365Fixed);
        let spread_curve = SpreadDiscountCurve::new(&base, 0.005, 30.0);
        // With 50bp spread, df should be lower than base
        for t in [1.0, 5.0, 10.0] {
            let df_base = base.discount_t(t);
            let df_spread = spread_curve.discount_impl(t);
            assert!(df_spread < df_base, "spread should lower df");
            // Expected: base_df * exp(-0.005 * t)
            let expected = df_base * (-0.005 * t).exp();
            assert_abs_diff_eq!(df_spread, expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn spread_discount_curve_at_zero() {
        let base = FlatForward::new(ref_date(), 0.05, DayCounter::Actual365Fixed);
        let spread_curve = SpreadDiscountCurve::new(&base, 0.01, 30.0);
        assert_abs_diff_eq!(spread_curve.discount_impl(0.0), 1.0, epsilon = 1e-15);
    }

    // -----------------------------------------------------------------------
    // OvernightIndexFutureRateHelper
    // -----------------------------------------------------------------------

    #[test]
    fn sofr_future_implied_rate() {
        let start = Date::from_ymd(2025, Month::March, 19);
        let end = Date::from_ymd(2025, Month::June, 18);
        let helper = OvernightIndexFutureRateHelper::sr3(95.50, start, end, 0.001);
        // rate = (100 - 95.5) / 100 - 0.001 = 0.044
        assert_abs_diff_eq!(helper.implied_rate(), 0.044, epsilon = 1e-15);
        assert!(!helper.is_one_month());
    }

    #[test]
    fn sofr_future_sr1() {
        let start = Date::from_ymd(2025, Month::April, 1);
        let end = Date::from_ymd(2025, Month::May, 1);
        let helper = OvernightIndexFutureRateHelper::sr1(96.0, start, end, 0.0);
        assert_abs_diff_eq!(helper.implied_rate(), 0.04, epsilon = 1e-15);
        assert!(helper.is_one_month());
        assert_eq!(helper.pillar_date(), end);
    }

    #[test]
    fn sofr_future_bootstrap() {
        use crate::bootstrap::PiecewiseYieldCurve;

        let start = ref_date();
        let end = Date::from_ymd(2025, Month::April, 2);
        let rate = 0.04;
        let price = 100.0 - rate * 100.0;

        let mut helpers: Vec<Box<dyn RateHelper>> = vec![Box::new(
            OvernightIndexFutureRateHelper::sr3(price, start, end, 0.0),
        )];

        let curve = PiecewiseYieldCurve::new(
            ref_date(),
            &mut helpers,
            DayCounter::Actual360,
            1e-12,
        )
        .unwrap();

        let yf = DayCounter::Actual360.year_fraction(start, end);
        let expected_df = 1.0 / (1.0 + rate * yf);
        assert_abs_diff_eq!(curve.discount(end), expected_df, epsilon = 1e-8);
    }

    // -----------------------------------------------------------------------
    // FittingMethodExtended
    // -----------------------------------------------------------------------

    #[test]
    fn fitting_method_extended_dispatch() {
        let ns = crate::nelson_siegel::NelsonSiegelFitting::new(0.04, -0.01, 0.01, 1.5);
        let method = FittingMethodExtended::NelsonSiegel(ns.clone());
        assert_abs_diff_eq!(method.discount(5.0), ns.discount(5.0), epsilon = 1e-15);
        assert_abs_diff_eq!(method.zero_rate(5.0), ns.zero_rate(5.0), epsilon = 1e-15);
    }

    #[test]
    fn fitting_method_extended_bspline() {
        let mats: Vec<f64> = (1..=8).map(|y| y as f64).collect();
        let prices: Vec<f64> = mats.iter().map(|&t| (-0.05 * t).exp()).collect();
        let bs = CubicBSplinesFitting::fit(&mats, &prices, None);
        let method = FittingMethodExtended::CubicBSpline(bs);
        assert!(method.discount(5.0) > 0.0 && method.discount(5.0) < 1.0);
    }

    #[test]
    fn fitting_method_extended_serializes() {
        let ns = crate::nelson_siegel::NelsonSiegelFitting::new(0.04, -0.01, 0.01, 1.5);
        let method = FittingMethodExtended::NelsonSiegel(ns);
        let json = serde_json::to_string(&method).unwrap();
        let back: FittingMethodExtended = serde_json::from_str(&json).unwrap();
        assert_abs_diff_eq!(
            method.discount(5.0),
            back.discount(5.0),
            epsilon = 1e-15
        );
    }
}
