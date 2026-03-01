//! Extended volatility surfaces and smile sections (Phase 18 — Vol Surfaces & Smile).
//!
//! **G106** — `LocalVolCurve` — Local vol as a function of time only (1D).
//! **G107** — `LocalConstantVol` — Constant local volatility term structure.
//! **G108** — `NoExceptLocalVolSurface` — Local vol surface with fallback on invalid inputs.
//! **G109** — `InterpolatedSmileSection` — Smile section from discrete strike-vol points.
//! **G110** — `AtmAdjustedSmileSection` — Wraps a smile section, adjusting ATM vol.
//! **G111** — `AtmSmileSection` / `FlatSmileSection` — Constant-vol smile sections.
//! **G112** — `CPIVolatilityStructure` / `ConstantCPIVolatility` — CPI inflation vol.
//! **G113** — `YoYInflationOptionletVolatilityStructure` — YoY inflation optionlet vol.
//! **G114** — `CmsMarket` / `CmsMarketCalibration` — CMS market data and calibration.
//! **G115** — `Gaussian1dSwaptionVolatility` — Swaption vol from Gaussian 1D model.
//! **G116** — `SwaptionVolDiscrete` — Discretely-sampled swaption vol matrix.

use serde::{Deserialize, Serialize};

// ===========================================================================
// Inline linear interpolation helper
// ===========================================================================

/// Linearly interpolate a sorted vector of (x, y) pairs at the given `x`.
/// Clamps to boundary values outside the range.
fn linear_interp(data: &[(f64, f64)], x: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    if data.len() == 1 {
        return data[0].1;
    }
    if x <= data[0].0 {
        return data[0].1;
    }
    let last = data.len() - 1;
    if x >= data[last].0 {
        return data[last].1;
    }
    // Binary search for bracketing interval
    let mut lo = 0;
    let mut hi = last;
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if data[mid].0 <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let x0 = data[lo].0;
    let x1 = data[hi].0;
    let y0 = data[lo].1;
    let y1 = data[hi].1;
    let dx = x1 - x0;
    if dx.abs() < 1e-15 {
        return y0;
    }
    y0 + (y1 - y0) * (x - x0) / dx
}

/// Find the bracketing index and weight for bilinear interpolation in a sorted slice.
fn bracket(xs: &[f64], x: f64) -> (usize, f64) {
    if xs.is_empty() {
        return (0, 0.0);
    }
    let n = xs.len();
    if x <= xs[0] {
        return (0, 0.0);
    }
    if x >= xs[n - 1] {
        return (n - 1, 0.0);
    }
    let mut lo = 0;
    let mut hi = n - 1;
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if xs[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let w = (x - xs[lo]) / (xs[hi] - xs[lo]);
    (lo, w)
}

// ===========================================================================
// G106: LocalVolCurve
// ===========================================================================

/// Local volatility as a function of time only (1D curve).
///
/// Takes a vector of `(time, vol)` pairs and linearly interpolates between them.
/// The strike dimension is ignored — this represents a time-only local vol.
///
/// # Example
/// ```
/// use ql_termstructures::vol_surfaces_extended::LocalVolCurve;
/// let curve = LocalVolCurve::new(vec![(0.0, 0.20), (1.0, 0.25), (2.0, 0.22)]);
/// let v = curve.vol(0.5);
/// assert!((v - 0.225).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalVolCurve {
    /// Sorted `(time, vol)` nodes.
    pub nodes: Vec<(f64, f64)>,
}

impl LocalVolCurve {
    /// Create a new local vol curve from `(time, vol)` pairs.
    ///
    /// The pairs are sorted by time on construction.
    pub fn new(mut nodes: Vec<(f64, f64)>) -> Self {
        nodes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        Self { nodes }
    }

    /// Local vol at time `t` (linearly interpolated).
    pub fn vol(&self, t: f64) -> f64 {
        linear_interp(&self.nodes, t)
    }

    /// Local vol at `(t, strike)` — strike is ignored.
    pub fn local_vol(&self, t: f64, _strike: f64) -> f64 {
        self.vol(t)
    }
}

// ===========================================================================
// G107: LocalConstantVol
// ===========================================================================

/// Constant local volatility term structure.
///
/// Returns the same volatility regardless of time or strike.
///
/// # Example
/// ```
/// use ql_termstructures::vol_surfaces_extended::LocalConstantVol;
/// let cv = LocalConstantVol::new(0.30);
/// assert_eq!(cv.vol(1.0, 100.0), 0.30);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalConstantVol {
    /// The constant local volatility.
    pub vol_value: f64,
}

impl LocalConstantVol {
    /// Create a constant local vol with the given value.
    pub fn new(vol: f64) -> Self {
        Self { vol_value: vol }
    }

    /// Local vol at `(t, strike)` — always returns the constant.
    pub fn vol(&self, _t: f64, _strike: f64) -> f64 {
        self.vol_value
    }

    /// Local vol at time `t` only — always returns the constant.
    pub fn vol_t(&self, _t: f64) -> f64 {
        self.vol_value
    }
}

// ===========================================================================
// G108: NoExceptLocalVolSurface
// ===========================================================================

/// Local vol surface that returns a fallback value instead of panicking on
/// invalid inputs (e.g., negative time, NaN, etc.).
///
/// Wraps a grid-based local vol surface and catches out-of-range queries
/// by returning a user-specified fallback volatility.
///
/// # Example
/// ```
/// use ql_termstructures::vol_surfaces_extended::NoExceptLocalVolSurface;
/// let surface = NoExceptLocalVolSurface::new(
///     vec![0.5, 1.0],
///     vec![90.0, 100.0, 110.0],
///     vec![
///         0.30, 0.25, 0.20,
///         0.28, 0.23, 0.18,
///     ],
///     0.25,
/// );
/// // Negative time → fallback
/// assert_eq!(surface.vol(-1.0, 100.0), 0.25);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoExceptLocalVolSurface {
    /// Time axis (sorted ascending).
    pub times: Vec<f64>,
    /// Strike axis (sorted ascending).
    pub strikes: Vec<f64>,
    /// Row-major vol grid: `vols[i * strikes.len() + j]`.
    pub vols: Vec<f64>,
    /// Fallback vol returned for out-of-range or invalid inputs.
    pub fallback_vol: f64,
}

impl NoExceptLocalVolSurface {
    /// Create a new no-except local vol surface.
    ///
    /// `vols` must have length `times.len() * strikes.len()`.
    pub fn new(
        times: Vec<f64>,
        strikes: Vec<f64>,
        vols: Vec<f64>,
        fallback_vol: f64,
    ) -> Self {
        assert_eq!(vols.len(), times.len() * strikes.len());
        Self {
            times,
            strikes,
            vols,
            fallback_vol,
        }
    }

    /// Local vol at `(t, strike)` with fallback for invalid inputs.
    pub fn vol(&self, t: f64, strike: f64) -> f64 {
        // Guard: negative time, NaN, empty grid
        if t.is_nan() || strike.is_nan() || t < 0.0 || strike <= 0.0 {
            return self.fallback_vol;
        }
        if self.times.is_empty() || self.strikes.is_empty() {
            return self.fallback_vol;
        }

        let nt = self.times.len();
        let ns = self.strikes.len();

        let (ti, tw) = bracket(&self.times, t);
        let (si, sw) = bracket(&self.strikes, strike);

        let ti2 = (ti + 1).min(nt - 1);
        let si2 = (si + 1).min(ns - 1);

        let v00 = self.vols[ti * ns + si];
        let v01 = self.vols[ti * ns + si2];
        let v10 = self.vols[ti2 * ns + si];
        let v11 = self.vols[ti2 * ns + si2];

        let v0 = v00 + tw * (v10 - v00);
        let v1 = v01 + tw * (v11 - v01);
        v0 + sw * (v1 - v0)
    }
}

// ===========================================================================
// G109: InterpolatedSmileSection
// ===========================================================================

/// A smile section (vol as a function of strike) built from discrete
/// strike-vol points with linear interpolation.
///
/// # Example
/// ```
/// use ql_termstructures::vol_surfaces_extended::InterpolatedSmileSection;
/// let section = InterpolatedSmileSection::new(
///     1.0,
///     vec![(90.0, 0.30), (100.0, 0.25), (110.0, 0.22)],
/// );
/// assert!((section.vol(100.0) - 0.25).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpolatedSmileSection {
    /// The option expiry time (in years).
    pub expiry: f64,
    /// Sorted `(strike, vol)` nodes.
    pub nodes: Vec<(f64, f64)>,
}

impl InterpolatedSmileSection {
    /// Create a smile section from `(strike, vol)` pairs.
    ///
    /// Pairs are sorted by strike on construction.
    pub fn new(expiry: f64, mut nodes: Vec<(f64, f64)>) -> Self {
        nodes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        Self { expiry, nodes }
    }

    /// Volatility at a given `strike` (linearly interpolated).
    pub fn vol(&self, strike: f64) -> f64 {
        linear_interp(&self.nodes, strike)
    }

    /// Total variance σ²·T at a given `strike`.
    pub fn variance(&self, strike: f64) -> f64 {
        let v = self.vol(strike);
        v * v * self.expiry
    }

    /// ATM vol (at the middle strike).
    pub fn atm_vol(&self) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        let mid_idx = self.nodes.len() / 2;
        self.nodes[mid_idx].1
    }
}

// ===========================================================================
// G110: AtmAdjustedSmileSection
// ===========================================================================

/// Wraps another smile section and adjusts all vols so that the ATM vol
/// matches a given target value.
///
/// The adjustment is additive: `adjusted_vol(K) = base_vol(K) + (target_atm - base_atm)`.
///
/// # Example
/// ```
/// use ql_termstructures::vol_surfaces_extended::{
///     InterpolatedSmileSection, AtmAdjustedSmileSection,
/// };
/// let base = InterpolatedSmileSection::new(
///     1.0,
///     vec![(90.0, 0.30), (100.0, 0.25), (110.0, 0.22)],
/// );
/// let adjusted = AtmAdjustedSmileSection::new(base, 100.0, 0.28);
/// // ATM vol should now be 0.28
/// assert!((adjusted.vol(100.0) - 0.28).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtmAdjustedSmileSection {
    /// The underlying smile section.
    pub base: InterpolatedSmileSection,
    /// ATM strike used to determine the base ATM vol.
    pub atm_strike: f64,
    /// Target ATM vol.
    pub target_atm_vol: f64,
    /// Additive adjustment: `target_atm_vol - base.vol(atm_strike)`.
    adjustment: f64,
}

impl AtmAdjustedSmileSection {
    /// Create an ATM-adjusted smile section.
    ///
    /// - `base`: the underlying smile section
    /// - `atm_strike`: the ATM strike to calibrate to
    /// - `target_atm_vol`: desired ATM vol at `atm_strike`
    pub fn new(
        base: InterpolatedSmileSection,
        atm_strike: f64,
        target_atm_vol: f64,
    ) -> Self {
        let base_atm = base.vol(atm_strike);
        let adjustment = target_atm_vol - base_atm;
        Self {
            base,
            atm_strike,
            target_atm_vol,
            adjustment,
        }
    }

    /// Adjusted volatility at a given `strike`.
    pub fn vol(&self, strike: f64) -> f64 {
        (self.base.vol(strike) + self.adjustment).max(0.0)
    }

    /// The expiry time of the underlying section.
    pub fn expiry(&self) -> f64 {
        self.base.expiry
    }

    /// Total variance σ²·T at a given `strike`.
    pub fn variance(&self, strike: f64) -> f64 {
        let v = self.vol(strike);
        v * v * self.base.expiry
    }
}

// ===========================================================================
// G111: AtmSmileSection / FlatSmileSection
// ===========================================================================

/// A smile section that returns a constant ATM volatility for all strikes.
///
/// Useful as a simple benchmark or placeholder smile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtmSmileSection {
    /// Option expiry time in years.
    pub expiry: f64,
    /// ATM volatility (returned for every strike).
    pub atm_vol: f64,
}

impl AtmSmileSection {
    /// Create an ATM smile section.
    pub fn new(expiry: f64, atm_vol: f64) -> Self {
        Self { expiry, atm_vol }
    }

    /// Volatility at `strike` — always returns `atm_vol`.
    pub fn vol(&self, _strike: f64) -> f64 {
        self.atm_vol
    }

    /// Total variance σ²·T.
    pub fn variance(&self, _strike: f64) -> f64 {
        self.atm_vol * self.atm_vol * self.expiry
    }
}

/// A flat smile section: returns a constant vol for all strikes.
///
/// Identical to `AtmSmileSection` but named for clarity when the vol
/// is not necessarily ATM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatSmileSection {
    /// Option expiry time in years.
    pub expiry: f64,
    /// Flat volatility (returned for every strike).
    pub flat_vol: f64,
}

impl FlatSmileSection {
    /// Create a flat smile section.
    pub fn new(expiry: f64, flat_vol: f64) -> Self {
        Self { expiry, flat_vol }
    }

    /// Volatility at `strike` — always returns `flat_vol`.
    pub fn vol(&self, _strike: f64) -> f64 {
        self.flat_vol
    }

    /// Total variance σ²·T.
    pub fn variance(&self, _strike: f64) -> f64 {
        self.flat_vol * self.flat_vol * self.expiry
    }
}

// ===========================================================================
// G112: CPIVolatilityStructure / ConstantCPIVolatility
// ===========================================================================

/// CPI inflation option volatility surface.
///
/// Maps `(maturity_time, strike)` → Black/normal vol for zero-coupon CPI
/// caps and floors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CPIVolatilityStructure {
    /// Maturity times (in years), sorted ascending.
    pub maturities: Vec<f64>,
    /// Strike grid, sorted ascending.
    pub strikes: Vec<f64>,
    /// Volatility matrix: `vols[i][j]` = vol at maturity `i`, strike `j`.
    pub vols: Vec<Vec<f64>>,
    /// Base CPI level.
    pub base_cpi: f64,
    /// Observation lag in months.
    pub observation_lag_months: u32,
    /// Whether volatilities are normal (true) or lognormal (false).
    pub is_normal: bool,
}

impl CPIVolatilityStructure {
    /// Create a new CPI volatility structure.
    pub fn new(
        maturities: Vec<f64>,
        strikes: Vec<f64>,
        vols: Vec<Vec<f64>>,
        base_cpi: f64,
        observation_lag_months: u32,
        is_normal: bool,
    ) -> Self {
        Self {
            maturities,
            strikes,
            vols,
            base_cpi,
            observation_lag_months,
            is_normal,
        }
    }

    /// CPI vol at given `maturity_time` and `strike` (bilinear interpolation).
    pub fn vol(&self, t: f64, strike: f64) -> f64 {
        if self.maturities.is_empty() || self.strikes.is_empty() {
            return 0.0;
        }
        let (mi, mw) = bracket(&self.maturities, t);
        let (si, sw) = bracket(&self.strikes, strike);

        let mi2 = (mi + 1).min(self.maturities.len() - 1);
        let si2 = (si + 1).min(self.strikes.len() - 1);

        let v00 = self.vols[mi][si];
        let v01 = self.vols[mi][si2];
        let v10 = self.vols[mi2][si];
        let v11 = self.vols[mi2][si2];

        let v0 = v00 + mw * (v10 - v00);
        let v1 = v01 + mw * (v11 - v01);
        v0 + sw * (v1 - v0)
    }

    /// Total variance σ²·T.
    pub fn total_variance(&self, t: f64, strike: f64) -> f64 {
        let v = self.vol(t, strike);
        v * v * t
    }
}

/// Constant (flat) CPI volatility — returns the same vol for every maturity and strike.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantCPIVolatility {
    /// The constant CPI vol.
    pub vol_value: f64,
    /// Whether the vol is normal (true) or lognormal (false).
    pub is_normal: bool,
}

impl ConstantCPIVolatility {
    /// Create a constant CPI volatility.
    pub fn new(vol: f64, is_normal: bool) -> Self {
        Self {
            vol_value: vol,
            is_normal,
        }
    }

    /// CPI vol at any `(t, strike)` — always returns `vol_value`.
    pub fn vol(&self, _t: f64, _strike: f64) -> f64 {
        self.vol_value
    }
}

// ===========================================================================
// G113: YoYInflationOptionletVolatilityStructure
// ===========================================================================

/// Year-on-year inflation optionlet volatility surface.
///
/// Provides volatilities for YoY inflation caps/floors as a function of
/// maturity and strike. Similar to `CPIVolatilityStructure` but specific
/// to year-on-year options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YoYInflationOptionletVolatilityStructure {
    /// Maturity times (in years), sorted ascending.
    pub maturities: Vec<f64>,
    /// Strike grid, sorted ascending.
    pub strikes: Vec<f64>,
    /// Optionlet volatilities: `vols[i][j]` = vol at maturity `i`, strike `j`.
    pub vols: Vec<Vec<f64>>,
    /// Whether volatilities are normal (true) or lognormal (false).
    pub is_normal: bool,
    /// Observation lag in months.
    pub observation_lag_months: u32,
}

impl YoYInflationOptionletVolatilityStructure {
    /// Create a new YoY inflation optionlet vol surface.
    pub fn new(
        maturities: Vec<f64>,
        strikes: Vec<f64>,
        vols: Vec<Vec<f64>>,
        is_normal: bool,
        observation_lag_months: u32,
    ) -> Self {
        Self {
            maturities,
            strikes,
            vols,
            is_normal,
            observation_lag_months,
        }
    }

    /// YoY optionlet vol at given `maturity_time` and `strike`.
    pub fn vol(&self, t: f64, strike: f64) -> f64 {
        if self.maturities.is_empty() || self.strikes.is_empty() {
            return 0.0;
        }
        let (mi, mw) = bracket(&self.maturities, t);
        let (si, sw) = bracket(&self.strikes, strike);

        let mi2 = (mi + 1).min(self.maturities.len() - 1);
        let si2 = (si + 1).min(self.strikes.len() - 1);

        let v00 = self.vols[mi][si];
        let v01 = self.vols[mi][si2];
        let v10 = self.vols[mi2][si];
        let v11 = self.vols[mi2][si2];

        let v0 = v00 + mw * (v10 - v00);
        let v1 = v01 + mw * (v11 - v01);
        v0 + sw * (v1 - v0)
    }

    /// Total variance σ²·T.
    pub fn total_variance(&self, t: f64, strike: f64) -> f64 {
        let v = self.vol(t, strike);
        v * v * t
    }

    /// ATM vol at a given maturity (at the middle strike).
    pub fn atm_vol(&self, t: f64) -> f64 {
        if self.strikes.is_empty() {
            return 0.0;
        }
        let mid = self.strikes[self.strikes.len() / 2];
        self.vol(t, mid)
    }
}

/// Constant YoY inflation optionlet vol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantYoYInflationOptionletVol {
    /// The constant optionlet vol.
    pub vol_value: f64,
    /// Whether the vol is normal (true) or lognormal (false).
    pub is_normal: bool,
}

impl ConstantYoYInflationOptionletVol {
    /// Create a constant YoY inflation optionlet vol.
    pub fn new(vol: f64, is_normal: bool) -> Self {
        Self {
            vol_value: vol,
            is_normal,
        }
    }

    /// YoY optionlet vol at any `(t, strike)`.
    pub fn vol(&self, _t: f64, _strike: f64) -> f64 {
        self.vol_value
    }
}

// ===========================================================================
// G114: CmsMarket / CmsMarketCalibration
// ===========================================================================

/// A single CMS market data point: `(tenor, rate, spread)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CmsMarketQuote {
    /// CMS swap tenor in years.
    pub tenor: f64,
    /// CMS swap rate (e.g. 0.035 = 3.5%).
    pub rate: f64,
    /// CMS spread (e.g. over Libor/Euribor).
    pub spread: f64,
}

/// CMS market data container.
///
/// Stores market-observed CMS swap rates and spreads for various tenors,
/// along with the underlying swaption vol surface used in CMS pricing.
///
/// # Example
/// ```
/// use ql_termstructures::vol_surfaces_extended::{CmsMarket, CmsMarketQuote};
/// let market = CmsMarket::new(
///     vec![
///         CmsMarketQuote { tenor: 2.0, rate: 0.030, spread: 0.0015 },
///         CmsMarketQuote { tenor: 5.0, rate: 0.035, spread: 0.0020 },
///         CmsMarketQuote { tenor: 10.0, rate: 0.040, spread: 0.0025 },
///     ],
///     0.05, // discount rate
/// );
/// assert_eq!(market.quotes.len(), 3);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CmsMarket {
    /// CMS market quotes: `(tenor, rate, spread)`.
    pub quotes: Vec<CmsMarketQuote>,
    /// Discount rate used for CMS replication pricing.
    pub discount_rate: f64,
}

impl CmsMarket {
    /// Create a new CMS market container.
    pub fn new(quotes: Vec<CmsMarketQuote>, discount_rate: f64) -> Self {
        Self {
            quotes,
            discount_rate,
        }
    }

    /// Retrieve the CMS rate at a given tenor (linearly interpolated).
    pub fn cms_rate(&self, tenor: f64) -> f64 {
        let data: Vec<(f64, f64)> = self.quotes.iter().map(|q| (q.tenor, q.rate)).collect();
        linear_interp(&data, tenor)
    }

    /// Retrieve the CMS spread at a given tenor (linearly interpolated).
    pub fn cms_spread(&self, tenor: f64) -> f64 {
        let data: Vec<(f64, f64)> = self.quotes.iter().map(|q| (q.tenor, q.spread)).collect();
        linear_interp(&data, tenor)
    }
}

/// CMS market calibration result.
///
/// The calibration adjusts a mean-reversion parameter so that model CMS rates
/// match observed market CMS rates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CmsMarketCalibration {
    /// Calibrated mean reversion speed.
    pub mean_reversion: f64,
    /// Model-implied CMS rates after calibration (one per quote).
    pub model_cms_rates: Vec<f64>,
    /// Calibration errors: `market_rate - model_rate` for each quote.
    pub errors: Vec<f64>,
    /// Root-mean-square calibration error.
    pub rmse: f64,
}

/// Calibrate CMS mean reversion to market quotes.
///
/// Uses a simple bisection to find the mean-reversion `κ` such that the
/// static-replication CMS convexity correction matches market CMS rates.
///
/// The CMS convexity adjustment under a simple model is approximated as:
///
/// $$\text{CMS rate} \approx \text{swap rate} + \frac{\sigma^2 \cdot T \cdot S}{1 + S \cdot \tau}$$
///
/// where the vol `σ` is adjusted via the annuity mapping function that depends
/// on mean reversion `κ`.
pub fn calibrate_cms_market(
    market: &CmsMarket,
    swaption_vol: f64,
) -> CmsMarketCalibration {
    // Simplified CMS convexity correction:
    // cms_rate ≈ swap_rate + σ² · T · S² / (1 + S·T) · (1 - κ·T/2)
    // We calibrate κ to minimise the sum of squared differences.

    let best_kappa = {
        let mut best = 0.0_f64;
        let mut best_err = f64::MAX;
        // Search over κ ∈ [0, 0.5]
        let n_steps = 1000;
        for i in 0..=n_steps {
            let kappa = i as f64 * 0.5 / n_steps as f64;
            let mut total_sq_err = 0.0;
            for q in &market.quotes {
                let model_rate = cms_model_rate(q.rate, swaption_vol, q.tenor, kappa);
                let target = q.rate + q.spread;
                let err = target - model_rate;
                total_sq_err += err * err;
            }
            if total_sq_err < best_err {
                best_err = total_sq_err;
                best = kappa;
            }
        }
        best
    };

    let mut model_rates = Vec::with_capacity(market.quotes.len());
    let mut errors = Vec::with_capacity(market.quotes.len());
    let mut sum_sq = 0.0;

    for q in &market.quotes {
        let mr = cms_model_rate(q.rate, swaption_vol, q.tenor, best_kappa);
        let target = q.rate + q.spread;
        let err = target - mr;
        model_rates.push(mr);
        errors.push(err);
        sum_sq += err * err;
    }

    let n = market.quotes.len().max(1) as f64;
    let rmse = (sum_sq / n).sqrt();

    CmsMarketCalibration {
        mean_reversion: best_kappa,
        model_cms_rates: model_rates,
        errors,
        rmse,
    }
}

/// Simple CMS model rate given swap rate, vol, tenor, and mean reversion.
fn cms_model_rate(swap_rate: f64, vol: f64, tenor: f64, kappa: f64) -> f64 {
    // CMS convexity correction:
    // Δ = σ² · T · S² / (1 + S·T) · g(κ,T)
    // where g(κ,T) ≈ (1 - exp(-κ·T)) / (κ·T) for κ > 0, else 1
    let g = if kappa.abs() < 1e-10 {
        1.0
    } else {
        (1.0 - (-kappa * tenor).exp()) / (kappa * tenor)
    };
    let convexity = vol * vol * tenor * swap_rate * swap_rate / (1.0 + swap_rate * tenor) * g;
    swap_rate + convexity
}

// ===========================================================================
// G115: Gaussian1dSwaptionVolatility
// ===========================================================================

/// Swaption volatility surface implied by a Gaussian 1D short-rate model
/// (Hull-White).
///
/// Given model parameters (mean reversion `κ` and short-rate vol `σ`), this
/// computes implied Black swaption volatilities using Jamshidian's formula.
///
/// The bond volatility in Hull-White is:
/// $$\sigma_P(T) = \frac{\sigma}{\kappa}\bigl(1 - e^{-\kappa T}\bigr)$$
///
/// The integrated variance over `[0, t]` is:
/// $$V(t) = \frac{\sigma^2}{2\kappa}\bigl(1 - e^{-2\kappa t}\bigr)$$
///
/// The approximate swaption vol is then:
/// $$\sigma_{\text{swaption}} \approx \frac{B(T_{\text{swap}}) \cdot \sqrt{V(T_{\text{opt}})}}{A \cdot \sqrt{T_{\text{opt}}}}$$
///
/// where $A$ is the swap annuity approximation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gaussian1dSwaptionVolatility {
    /// Mean reversion speed κ.
    pub mean_reversion: f64,
    /// Short-rate volatility σ.
    pub sigma: f64,
    /// Number of fixed payments per year in the underlying swap (e.g. 2 for semiannual).
    pub fixed_frequency: u32,
}

impl Gaussian1dSwaptionVolatility {
    /// Create a new Gaussian 1D swaption volatility calculator.
    pub fn new(mean_reversion: f64, sigma: f64, fixed_frequency: u32) -> Self {
        Self {
            mean_reversion,
            sigma,
            fixed_frequency,
        }
    }

    /// Hull-White bond vol function: B(T) = (1 - exp(-κT)) / κ.
    fn bond_vol_factor(&self, t: f64) -> f64 {
        let k = self.mean_reversion;
        if k.abs() < 1e-10 {
            t
        } else {
            (1.0 - (-k * t).exp()) / k
        }
    }

    /// Hull-White integrated variance over [0, t]: σ²/(2κ) · (1 - exp(-2κt)).
    fn integrated_variance(&self, t: f64) -> f64 {
        let k = self.mean_reversion;
        let s = self.sigma;
        if k.abs() < 1e-10 {
            s * s * t
        } else {
            s * s / (2.0 * k) * (1.0 - (-2.0 * k * t).exp())
        }
    }

    /// Compute the implied Black swaption vol for an option with expiry
    /// `t_opt` on a swap of tenor `t_swap` (both in years).
    ///
    /// Uses Jamshidian's formula for the Hull-White model.
    pub fn vol(&self, t_opt: f64, t_swap: f64) -> f64 {
        if t_opt <= 0.0 || t_swap <= 0.0 {
            return 0.0;
        }

        let freq = self.fixed_frequency.max(1) as f64;
        let n_payments = (t_swap * freq).round() as usize;
        if n_payments == 0 {
            return 0.0;
        }

        // Bond vol difference: B(T_opt + T_swap) - B(T_opt)
        // Jamshidian: σ_swaption ≈ |Σ cᵢ Bᵢ| · √V / (A · √T)
        // where cᵢ are coupon cashflow weights and Bᵢ = B(tᵢ) - B(T_opt)

        let v = self.integrated_variance(t_opt);
        if v <= 0.0 {
            return 0.0;
        }

        // Compute annuity-weighted bond vol
        let dt = 1.0 / freq;
        let mut weighted_bv = 0.0;
        let mut annuity = 0.0;

        for i in 1..=n_payments {
            let ti = t_opt + i as f64 * dt;
            let bi = self.bond_vol_factor(ti) - self.bond_vol_factor(t_opt);
            // Weight: discount factor approximation ≈ 1 (flat rate)
            let w = dt;
            annuity += w;
            weighted_bv += w * bi;
        }

        if annuity.abs() < 1e-15 {
            return 0.0;
        }

        let avg_bv = weighted_bv / annuity;
        let swap_vol = avg_bv * v.sqrt() / t_opt.sqrt();
        swap_vol.abs()
    }

    /// Implied swaption vol at `(t_opt, t_swap, strike)` — strike is ignored
    /// because the Gaussian model produces a strike-independent approximate vol.
    pub fn vol_with_strike(&self, t_opt: f64, t_swap: f64, _strike: f64) -> f64 {
        self.vol(t_opt, t_swap)
    }
}

// ===========================================================================
// G116: SwaptionVolDiscrete
// ===========================================================================

/// Base representation for discretely-sampled swaption volatilities.
///
/// Stores a matrix of `option_tenors × swap_tenors → vol`, with bilinear
/// interpolation between nodes.
///
/// Unlike `SwaptionVolMatrix` (which requires the `ql-math` interpolation
/// module), this struct is self-contained with inline interpolation.
///
/// # Example
/// ```
/// use ql_termstructures::vol_surfaces_extended::SwaptionVolDiscrete;
/// let svd = SwaptionVolDiscrete::new(
///     vec![0.25, 0.5, 1.0],   // option tenors
///     vec![2.0, 5.0, 10.0],   // swap tenors
///     vec![
///         vec![0.45, 0.40, 0.35],
///         vec![0.44, 0.39, 0.34],
///         vec![0.40, 0.36, 0.31],
///     ],
/// );
/// let v = svd.vol(0.5, 5.0);
/// assert!((v - 0.39).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwaptionVolDiscrete {
    /// Option tenors (in years), sorted ascending.
    pub option_tenors: Vec<f64>,
    /// Swap tenors (in years), sorted ascending.
    pub swap_tenors: Vec<f64>,
    /// Volatilities: `vols[i][j]` = vol for option tenor `i`, swap tenor `j`.
    pub vols: Vec<Vec<f64>>,
}

impl SwaptionVolDiscrete {
    /// Create a new discrete swaption vol matrix.
    pub fn new(
        option_tenors: Vec<f64>,
        swap_tenors: Vec<f64>,
        vols: Vec<Vec<f64>>,
    ) -> Self {
        assert_eq!(vols.len(), option_tenors.len());
        for row in &vols {
            assert_eq!(row.len(), swap_tenors.len());
        }
        Self {
            option_tenors,
            swap_tenors,
            vols,
        }
    }

    /// Swaption vol at `(option_tenor, swap_tenor)` via bilinear interpolation.
    pub fn vol(&self, option_tenor: f64, swap_tenor: f64) -> f64 {
        if self.option_tenors.is_empty() || self.swap_tenors.is_empty() {
            return 0.0;
        }

        let (oi, ow) = bracket(&self.option_tenors, option_tenor);
        let (si, sw) = bracket(&self.swap_tenors, swap_tenor);

        let no = self.option_tenors.len();
        let ns = self.swap_tenors.len();

        let oi2 = (oi + 1).min(no - 1);
        let si2 = (si + 1).min(ns - 1);

        let v00 = self.vols[oi][si];
        let v01 = self.vols[oi][si2];
        let v10 = self.vols[oi2][si];
        let v11 = self.vols[oi2][si2];

        let v0 = v00 + ow * (v10 - v00);
        let v1 = v01 + ow * (v11 - v01);
        v0 + sw * (v1 - v0)
    }

    /// Swaption vol at `(option_tenor, swap_tenor, strike)` — strike ignored
    /// (ATM only in the discrete matrix representation).
    pub fn vol_with_strike(&self, option_tenor: f64, swap_tenor: f64, _strike: f64) -> f64 {
        self.vol(option_tenor, swap_tenor)
    }

    /// Number of option tenors.
    pub fn n_option_tenors(&self) -> usize {
        self.option_tenors.len()
    }

    /// Number of swap tenors.
    pub fn n_swap_tenors(&self) -> usize {
        self.swap_tenors.len()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // -----------------------------------------------------------------------
    // G106: LocalVolCurve
    // -----------------------------------------------------------------------

    #[test]
    fn local_vol_curve_interpolation() {
        let curve = LocalVolCurve::new(vec![(0.0, 0.20), (1.0, 0.25), (2.0, 0.22)]);
        assert_abs_diff_eq!(curve.vol(0.0), 0.20, epsilon = 1e-12);
        assert_abs_diff_eq!(curve.vol(0.5), 0.225, epsilon = 1e-12);
        assert_abs_diff_eq!(curve.vol(1.0), 0.25, epsilon = 1e-12);
        assert_abs_diff_eq!(curve.vol(1.5), 0.235, epsilon = 1e-12);
        assert_abs_diff_eq!(curve.vol(2.0), 0.22, epsilon = 1e-12);
    }

    #[test]
    fn local_vol_curve_boundary_extrapolation() {
        let curve = LocalVolCurve::new(vec![(1.0, 0.30), (3.0, 0.20)]);
        // Below min → flat at first node
        assert_abs_diff_eq!(curve.vol(0.0), 0.30, epsilon = 1e-12);
        // Above max → flat at last node
        assert_abs_diff_eq!(curve.vol(5.0), 0.20, epsilon = 1e-12);
        // Strike ignored
        assert_abs_diff_eq!(curve.local_vol(2.0, 999.0), 0.25, epsilon = 1e-12);
    }

    // -----------------------------------------------------------------------
    // G107: LocalConstantVol
    // -----------------------------------------------------------------------

    #[test]
    fn local_constant_vol_returns_constant() {
        let cv = LocalConstantVol::new(0.30);
        assert_abs_diff_eq!(cv.vol(0.5, 100.0), 0.30, epsilon = 1e-15);
        assert_abs_diff_eq!(cv.vol(10.0, 50.0), 0.30, epsilon = 1e-15);
    }

    #[test]
    fn local_constant_vol_time_only() {
        let cv = LocalConstantVol::new(0.18);
        assert_abs_diff_eq!(cv.vol_t(0.0), 0.18, epsilon = 1e-15);
        assert_abs_diff_eq!(cv.vol_t(100.0), 0.18, epsilon = 1e-15);
    }

    // -----------------------------------------------------------------------
    // G108: NoExceptLocalVolSurface
    // -----------------------------------------------------------------------

    #[test]
    fn no_except_returns_fallback_on_invalid() {
        let surface = NoExceptLocalVolSurface::new(
            vec![0.5, 1.0],
            vec![90.0, 100.0, 110.0],
            vec![
                0.30, 0.25, 0.20,
                0.28, 0.23, 0.18,
            ],
            0.99,
        );
        // Negative time
        assert_abs_diff_eq!(surface.vol(-1.0, 100.0), 0.99, epsilon = 1e-15);
        // NaN strike
        assert_abs_diff_eq!(surface.vol(0.5, f64::NAN), 0.99, epsilon = 1e-15);
        // Zero strike
        assert_abs_diff_eq!(surface.vol(0.5, 0.0), 0.99, epsilon = 1e-15);
    }

    #[test]
    fn no_except_bilinear_interpolation() {
        let surface = NoExceptLocalVolSurface::new(
            vec![0.5, 1.0],
            vec![90.0, 110.0],
            vec![
                0.30, 0.20,  // t=0.5
                0.28, 0.18,  // t=1.0
            ],
            0.99,
        );
        // At node (t=0.5, K=90)
        assert_abs_diff_eq!(surface.vol(0.5, 90.0), 0.30, epsilon = 1e-12);
        // At node (t=1.0, K=110)
        assert_abs_diff_eq!(surface.vol(1.0, 110.0), 0.18, epsilon = 1e-12);
        // Midpoint
        let v = surface.vol(0.75, 100.0);
        // Expected: bilinear at center of grid
        // t weight = (0.75 - 0.5) / (1.0 - 0.5) = 0.5
        // s weight = (100 - 90) / (110 - 90) = 0.5
        // v00=0.30, v01=0.20, v10=0.28, v11=0.18
        // v0 = 0.30 + 0.5*(0.28 - 0.30) = 0.29
        // v1 = 0.20 + 0.5*(0.18 - 0.20) = 0.19
        // v  = 0.29 + 0.5*(0.19 - 0.29) = 0.24
        assert_abs_diff_eq!(v, 0.24, epsilon = 1e-12);
    }

    // -----------------------------------------------------------------------
    // G109: InterpolatedSmileSection
    // -----------------------------------------------------------------------

    #[test]
    fn interpolated_smile_at_nodes() {
        let section = InterpolatedSmileSection::new(
            1.0,
            vec![(90.0, 0.30), (100.0, 0.25), (110.0, 0.22)],
        );
        assert_abs_diff_eq!(section.vol(90.0), 0.30, epsilon = 1e-12);
        assert_abs_diff_eq!(section.vol(100.0), 0.25, epsilon = 1e-12);
        assert_abs_diff_eq!(section.vol(110.0), 0.22, epsilon = 1e-12);
    }

    #[test]
    fn interpolated_smile_between_nodes() {
        let section = InterpolatedSmileSection::new(
            2.0,
            vec![(80.0, 0.35), (100.0, 0.25), (120.0, 0.30)],
        );
        // At K=90: linear between (80,0.35) and (100,0.25) → 0.30
        assert_abs_diff_eq!(section.vol(90.0), 0.30, epsilon = 1e-12);
        // Variance at K=100: 0.25² × 2 = 0.125
        assert_abs_diff_eq!(section.variance(100.0), 0.125, epsilon = 1e-12);
    }

    // -----------------------------------------------------------------------
    // G110: AtmAdjustedSmileSection
    // -----------------------------------------------------------------------

    #[test]
    fn atm_adjusted_matches_target() {
        let base = InterpolatedSmileSection::new(
            1.0,
            vec![(90.0, 0.30), (100.0, 0.25), (110.0, 0.22)],
        );
        let adjusted = AtmAdjustedSmileSection::new(base, 100.0, 0.28);
        assert_abs_diff_eq!(adjusted.vol(100.0), 0.28, epsilon = 1e-12);
    }

    #[test]
    fn atm_adjusted_preserves_smile_shape() {
        let base = InterpolatedSmileSection::new(
            1.0,
            vec![(90.0, 0.30), (100.0, 0.25), (110.0, 0.22)],
        );
        let adjusted = AtmAdjustedSmileSection::new(base, 100.0, 0.28);
        // Adjustment = 0.28 - 0.25 = 0.03
        assert_abs_diff_eq!(adjusted.vol(90.0), 0.33, epsilon = 1e-12);
        assert_abs_diff_eq!(adjusted.vol(110.0), 0.25, epsilon = 1e-12);
        // Expiry preserved
        assert_abs_diff_eq!(adjusted.expiry(), 1.0, epsilon = 1e-15);
    }

    // -----------------------------------------------------------------------
    // G111: AtmSmileSection / FlatSmileSection
    // -----------------------------------------------------------------------

    #[test]
    fn atm_smile_section_constant() {
        let section = AtmSmileSection::new(1.0, 0.25);
        assert_abs_diff_eq!(section.vol(80.0), 0.25, epsilon = 1e-15);
        assert_abs_diff_eq!(section.vol(120.0), 0.25, epsilon = 1e-15);
        assert_abs_diff_eq!(section.variance(100.0), 0.0625, epsilon = 1e-12);
    }

    #[test]
    fn flat_smile_section_constant() {
        let section = FlatSmileSection::new(2.0, 0.18);
        assert_abs_diff_eq!(section.vol(50.0), 0.18, epsilon = 1e-15);
        assert_abs_diff_eq!(section.vol(150.0), 0.18, epsilon = 1e-15);
        // Variance = 0.18² × 2 = 0.0648
        assert_abs_diff_eq!(section.variance(100.0), 0.0648, epsilon = 1e-12);
    }

    // -----------------------------------------------------------------------
    // G112: CPIVolatilityStructure / ConstantCPIVolatility
    // -----------------------------------------------------------------------

    #[test]
    fn cpi_vol_structure_at_nodes() {
        let cpi = CPIVolatilityStructure::new(
            vec![1.0, 5.0],
            vec![0.01, 0.03, 0.05],
            vec![
                vec![0.06, 0.05, 0.04],  // t=1
                vec![0.07, 0.06, 0.05],  // t=5
            ],
            100.0,
            3,
            false,
        );
        assert_abs_diff_eq!(cpi.vol(1.0, 0.01), 0.06, epsilon = 1e-12);
        assert_abs_diff_eq!(cpi.vol(5.0, 0.05), 0.05, epsilon = 1e-12);
    }

    #[test]
    fn constant_cpi_vol() {
        let cv = ConstantCPIVolatility::new(0.04, false);
        assert_abs_diff_eq!(cv.vol(1.0, 0.02), 0.04, epsilon = 1e-15);
        assert_abs_diff_eq!(cv.vol(10.0, 0.05), 0.04, epsilon = 1e-15);
    }

    #[test]
    fn cpi_vol_total_variance() {
        let cpi = CPIVolatilityStructure::new(
            vec![1.0],
            vec![0.03],
            vec![vec![0.05]],
            100.0,
            3,
            false,
        );
        // Total variance = 0.05² × 1.0 = 0.0025
        assert_abs_diff_eq!(cpi.total_variance(1.0, 0.03), 0.0025, epsilon = 1e-12);
    }

    // -----------------------------------------------------------------------
    // G113: YoYInflationOptionletVolatilityStructure
    // -----------------------------------------------------------------------

    #[test]
    fn yoy_optionlet_vol_at_nodes() {
        let yoy = YoYInflationOptionletVolatilityStructure::new(
            vec![1.0, 3.0, 5.0],
            vec![0.01, 0.03],
            vec![
                vec![0.05, 0.04],
                vec![0.06, 0.05],
                vec![0.07, 0.06],
            ],
            false,
            3,
        );
        assert_abs_diff_eq!(yoy.vol(1.0, 0.01), 0.05, epsilon = 1e-12);
        assert_abs_diff_eq!(yoy.vol(5.0, 0.03), 0.06, epsilon = 1e-12);
    }

    #[test]
    fn yoy_optionlet_vol_interpolation() {
        let yoy = YoYInflationOptionletVolatilityStructure::new(
            vec![1.0, 3.0],
            vec![0.01, 0.03],
            vec![
                vec![0.05, 0.04],
                vec![0.07, 0.06],
            ],
            false,
            3,
        );
        // At t=2 (mid), K=0.02 (mid)
        // mw = (2-1)/(3-1) = 0.5, sw = (0.02-0.01)/(0.03-0.01) = 0.5
        // v00=0.05, v01=0.04, v10=0.07, v11=0.06
        // v0 = 0.05 + 0.5*(0.07-0.05) = 0.06
        // v1 = 0.04 + 0.5*(0.06-0.04) = 0.05
        // v  = 0.06 + 0.5*(0.05-0.06) = 0.055
        assert_abs_diff_eq!(yoy.vol(2.0, 0.02), 0.055, epsilon = 1e-12);
    }

    #[test]
    fn constant_yoy_vol() {
        let cv = ConstantYoYInflationOptionletVol::new(0.03, true);
        assert_abs_diff_eq!(cv.vol(1.0, 0.02), 0.03, epsilon = 1e-15);
        assert_abs_diff_eq!(cv.vol(10.0, 0.05), 0.03, epsilon = 1e-15);
    }

    // -----------------------------------------------------------------------
    // G114: CmsMarket / CmsMarketCalibration
    // -----------------------------------------------------------------------

    #[test]
    fn cms_market_rate_interpolation() {
        let market = CmsMarket::new(
            vec![
                CmsMarketQuote { tenor: 2.0, rate: 0.030, spread: 0.0015 },
                CmsMarketQuote { tenor: 5.0, rate: 0.035, spread: 0.0020 },
                CmsMarketQuote { tenor: 10.0, rate: 0.040, spread: 0.0025 },
            ],
            0.05,
        );
        assert_abs_diff_eq!(market.cms_rate(2.0), 0.030, epsilon = 1e-12);
        assert_abs_diff_eq!(market.cms_rate(10.0), 0.040, epsilon = 1e-12);
        // Midpoint between 2Y and 5Y: (0.030 + 0.035) / 2 = 0.0325
        assert_abs_diff_eq!(market.cms_rate(3.5), 0.0325, epsilon = 1e-12);
    }

    #[test]
    fn cms_market_calibration_runs() {
        let market = CmsMarket::new(
            vec![
                CmsMarketQuote { tenor: 5.0, rate: 0.035, spread: 0.0010 },
                CmsMarketQuote { tenor: 10.0, rate: 0.040, spread: 0.0015 },
            ],
            0.05,
        );
        let result = calibrate_cms_market(&market, 0.20);
        // Should produce a meaningful calibration
        assert!(result.mean_reversion >= 0.0);
        assert_eq!(result.model_cms_rates.len(), 2);
        assert_eq!(result.errors.len(), 2);
        // RMSE should be small (model is approximate but should get close)
        assert!(result.rmse < 0.01);
    }

    // -----------------------------------------------------------------------
    // G115: Gaussian1dSwaptionVolatility
    // -----------------------------------------------------------------------

    #[test]
    fn gaussian1d_swaption_vol_basic() {
        let g1d = Gaussian1dSwaptionVolatility::new(0.03, 0.01, 2);
        let v = g1d.vol(1.0, 5.0);
        // Should be positive
        assert!(v > 0.0);
        // With zero mean reversion, the vol simplifies
        let g1d_zero = Gaussian1dSwaptionVolatility::new(0.0, 0.01, 2);
        let v0 = g1d_zero.vol(1.0, 5.0);
        assert!(v0 > 0.0);
    }

    #[test]
    fn gaussian1d_swaption_vol_zero_inputs() {
        let g1d = Gaussian1dSwaptionVolatility::new(0.03, 0.01, 2);
        assert_abs_diff_eq!(g1d.vol(0.0, 5.0), 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(g1d.vol(1.0, 0.0), 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(g1d.vol(-1.0, 5.0), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn gaussian1d_swaption_vol_strike_independence() {
        let g1d = Gaussian1dSwaptionVolatility::new(0.05, 0.008, 4);
        let v1 = g1d.vol_with_strike(2.0, 10.0, 0.03);
        let v2 = g1d.vol_with_strike(2.0, 10.0, 0.05);
        assert_abs_diff_eq!(v1, v2, epsilon = 1e-15);
    }

    // -----------------------------------------------------------------------
    // G116: SwaptionVolDiscrete
    // -----------------------------------------------------------------------

    #[test]
    fn swaption_vol_discrete_at_nodes() {
        let svd = SwaptionVolDiscrete::new(
            vec![0.25, 0.5, 1.0],
            vec![2.0, 5.0, 10.0],
            vec![
                vec![0.45, 0.40, 0.35],
                vec![0.44, 0.39, 0.34],
                vec![0.40, 0.36, 0.31],
            ],
        );
        assert_abs_diff_eq!(svd.vol(0.25, 2.0), 0.45, epsilon = 1e-12);
        assert_abs_diff_eq!(svd.vol(0.5, 5.0), 0.39, epsilon = 1e-12);
        assert_abs_diff_eq!(svd.vol(1.0, 10.0), 0.31, epsilon = 1e-12);
    }

    #[test]
    fn swaption_vol_discrete_interpolation() {
        let svd = SwaptionVolDiscrete::new(
            vec![0.5, 1.0],
            vec![5.0, 10.0],
            vec![
                vec![0.40, 0.35],  // opt=0.5
                vec![0.36, 0.31],  // opt=1.0
            ],
        );
        // Midpoint: opt=0.75, swap=7.5
        // ow = (0.75 - 0.5) / (1.0 - 0.5) = 0.5
        // sw = (7.5 - 5.0) / (10.0 - 5.0) = 0.5
        // v00=0.40, v01=0.35, v10=0.36, v11=0.31
        // v0 = 0.40 + 0.5*(0.36-0.40) = 0.38
        // v1 = 0.35 + 0.5*(0.31-0.35) = 0.33
        // v  = 0.38 + 0.5*(0.33-0.38) = 0.355
        assert_abs_diff_eq!(svd.vol(0.75, 7.5), 0.355, epsilon = 1e-12);
    }

    #[test]
    fn swaption_vol_discrete_dimensions() {
        let svd = SwaptionVolDiscrete::new(
            vec![0.25, 0.5, 1.0, 2.0],
            vec![2.0, 5.0, 10.0],
            vec![
                vec![0.45, 0.40, 0.35],
                vec![0.44, 0.39, 0.34],
                vec![0.40, 0.36, 0.31],
                vec![0.38, 0.34, 0.29],
            ],
        );
        assert_eq!(svd.n_option_tenors(), 4);
        assert_eq!(svd.n_swap_tenors(), 3);
    }

    // -----------------------------------------------------------------------
    // Serialization round-trip tests
    // -----------------------------------------------------------------------

    #[test]
    fn serde_roundtrip_local_vol_curve() {
        let curve = LocalVolCurve::new(vec![(0.5, 0.22), (1.0, 0.25)]);
        let json = serde_json::to_string(&curve).unwrap();
        let deser: LocalVolCurve = serde_json::from_str(&json).unwrap();
        assert_abs_diff_eq!(deser.vol(0.75), curve.vol(0.75), epsilon = 1e-15);
    }

    #[test]
    fn serde_roundtrip_swaption_vol_discrete() {
        let svd = SwaptionVolDiscrete::new(
            vec![0.5, 1.0],
            vec![5.0, 10.0],
            vec![
                vec![0.40, 0.35],
                vec![0.36, 0.31],
            ],
        );
        let json = serde_json::to_string(&svd).unwrap();
        let deser: SwaptionVolDiscrete = serde_json::from_str(&json).unwrap();
        assert_abs_diff_eq!(deser.vol(0.75, 7.5), svd.vol(0.75, 7.5), epsilon = 1e-15);
    }
}
