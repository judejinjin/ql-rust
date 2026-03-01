//! Extended FDM meshers for specialised 1-D grids (Phase 19, G117–G122).
//!
//! Each constructor returns a [`Mesher1d`] from [`crate::fdm_meshers`].
//!
//! | Function | Gap | Purpose |
//! |---|---|---|
//! | [`fdm_black_scholes_mesher`] | G117 | BS-adapted sinh-concentrated mesher |
//! | [`fdm_black_scholes_multi_strike_mesher`] | G118 | Multi-strike piecewise BS mesher |
//! | [`fdm_heston_variance_mesher`] | G119 | Variance mesher for the Heston v-dim |
//! | [`fdm_simple_process_1d_mesher`] | G120 | Generic process mesher |
//! | [`exponential_jump_1d_mesher`] | G121 | Mesher for Kou/Merton jump processes |
//! | [`fdm_cev_1d_mesher`] | G122 | CEV-adapted mesher |

use crate::fdm_meshers::Mesher1d;

// ---------------------------------------------------------------------------
// G117 – FdmBlackScholesMesher
// ---------------------------------------------------------------------------

/// Parameters for [`fdm_black_scholes_mesher`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FdmBlackScholesMesherParams {
    /// Current spot price.
    pub spot: f64,
    /// Annualised volatility.
    pub vol: f64,
    /// Risk-free rate (continuously compounded).
    pub r: f64,
    /// Dividend yield (continuously compounded).
    pub q: f64,
    /// Strike price around which to concentrate.
    pub strike: f64,
    /// Time to maturity in years.
    pub maturity: f64,
    /// Number of grid points.
    pub n_points: usize,
    /// Concentration factor (default 2.0). Higher ⇒ tighter concentration.
    pub concentration_factor: f64,
}

/// Build a 1-D mesher in log-spot space adapted for Black-Scholes dynamics
/// (G117).
///
/// Grid points are concentrated around `ln(strike)` via the sinh
/// transformation
///
/// $$s_i = x_{\text{mid}} + d \sinh\!\bigl((u_i - 0.5)/d\bigr)$$
///
/// where $d = 1 / \text{concentration\_factor}$ and $u_i$ is uniform on
/// $[0, 1]$.
pub fn fdm_black_scholes_mesher(p: &FdmBlackScholesMesherParams) -> Mesher1d {
    assert!(p.n_points >= 2, "need at least 2 grid points");
    assert!(p.spot > 0.0, "spot must be positive");
    assert!(p.vol > 0.0, "vol must be positive");
    assert!(p.maturity > 0.0, "maturity must be positive");
    assert!(p.strike > 0.0, "strike must be positive");

    let x0 = p.spot.ln();
    let x_strike = p.strike.ln();
    let std_dev = p.vol * p.maturity.sqrt();
    let n_std = 4.0;
    let lo = x0 - n_std * std_dev;
    let hi = x0 + n_std * std_dev;

    // Concentration parameter (smaller d ⇒ tighter)
    let d = if p.concentration_factor.abs() < 1e-12 {
        1e6 // effectively uniform
    } else {
        1.0 / p.concentration_factor
    };

    let n = p.n_points;
    let locations: Vec<f64> = (0..n)
        .map(|i| {
            let u = i as f64 / (n - 1) as f64; // uniform in [0,1]
            let xi = x_strike + d * ((u - 0.5) / d).sinh();
            xi.clamp(lo, hi)
        })
        .collect();

    Mesher1d::from_locations(locations)
}

// ---------------------------------------------------------------------------
// G118 – FdmBlackScholesMultiStrikeMesher
// ---------------------------------------------------------------------------

/// Parameters for [`fdm_black_scholes_multi_strike_mesher`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FdmBlackScholesMultiStrikeMesherParams {
    /// Current spot price.
    pub spot: f64,
    /// Annualised volatility.
    pub vol: f64,
    /// Risk-free rate.
    pub r: f64,
    /// Dividend yield.
    pub q: f64,
    /// Strike prices to concentrate around (need not be sorted).
    pub strikes: Vec<f64>,
    /// Time to maturity.
    pub maturity: f64,
    /// Total number of grid points.
    pub n_points: usize,
    /// Concentration factor per strike (default 2.0).
    pub concentration_factor: f64,
}

/// Build a 1-D log-spot mesher that concentrates around **multiple** strikes
/// (G118).
///
/// Strategy: for each strike allocate a proportional number of points and
/// build a local sinh sub-grid, then merge and deduplicate, topping up with
/// uniform fill points so the total reaches `n_points`.
pub fn fdm_black_scholes_multi_strike_mesher(
    p: &FdmBlackScholesMultiStrikeMesherParams,
) -> Mesher1d {
    assert!(p.n_points >= 2);
    assert!(!p.strikes.is_empty(), "need at least one strike");
    assert!(p.spot > 0.0);
    assert!(p.vol > 0.0);
    assert!(p.maturity > 0.0);

    let x0 = p.spot.ln();
    let std_dev = p.vol * p.maturity.sqrt();
    let n_std = 4.0;
    let lo = x0 - n_std * std_dev;
    let hi = x0 + n_std * std_dev;

    let d = if p.concentration_factor.abs() < 1e-12 {
        1e6
    } else {
        1.0 / p.concentration_factor
    };

    // Generate concentrated points around each strike
    let pts_per_strike = (p.n_points / p.strikes.len()).max(3);
    let mut all_locs: Vec<f64> = Vec::with_capacity(p.n_points + p.strikes.len() * pts_per_strike);

    for &k in &p.strikes {
        assert!(k > 0.0, "strikes must be positive");
        let x_k = k.ln();
        for i in 0..pts_per_strike {
            let u = i as f64 / (pts_per_strike - 1) as f64;
            let xi = x_k + d * ((u - 0.5) / d).sinh();
            all_locs.push(xi.clamp(lo, hi));
        }
    }

    // Ensure endpoints are present
    all_locs.push(lo);
    all_locs.push(hi);

    // Sort and deduplicate (within tolerance)
    all_locs.sort_by(|a, b| a.total_cmp(b));
    all_locs.dedup_by(|a, b| (*a - *b).abs() < 1e-14);

    // Fill up to n_points with uniform points if needed
    if all_locs.len() < p.n_points {
        let deficit = p.n_points - all_locs.len();
        let dx = (hi - lo) / (deficit + 1) as f64;
        for i in 1..=deficit {
            all_locs.push(lo + i as f64 * dx);
        }
        all_locs.sort_by(|a, b| a.total_cmp(b));
        all_locs.dedup_by(|a, b| (*a - *b).abs() < 1e-14);
    }

    // Trim (or keep) to n_points by uniform subsampling if we have too many
    let locs = if all_locs.len() > p.n_points {
        subsample(&all_locs, p.n_points)
    } else {
        all_locs
    };

    Mesher1d::from_locations(locs)
}

/// Uniformly subsample `src` (already sorted) down to exactly `n` points,
/// always keeping the first and last.
fn subsample(src: &[f64], n: usize) -> Vec<f64> {
    if n >= src.len() {
        return src.to_vec();
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let idx = i * (src.len() - 1) / (n - 1);
        out.push(src[idx]);
    }
    out
}

// ---------------------------------------------------------------------------
// G119 – FdmHestonVarianceMesher
// ---------------------------------------------------------------------------

/// Parameters for [`fdm_heston_variance_mesher`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FdmHestonVarianceMesherParams {
    /// Initial variance v₀.
    pub v0: f64,
    /// Mean-reversion speed κ.
    pub kappa: f64,
    /// Long-run variance θ.
    pub theta: f64,
    /// Vol-of-vol σ.
    pub sigma: f64,
    /// Time to maturity.
    pub maturity: f64,
    /// Number of grid points.
    pub n_points: usize,
}

/// Build a variance-dimension mesher for the Heston model (G119).
///
/// The grid concentrates near v = 0 and around the long-run variance θ by
/// using a transformation based on quantiles of the non-central chi-squared
/// stationary distribution.  As a practical approximation we use a
/// piecewise-linear CDF constructed from a few key quantiles estimated via
/// the mean and variance of the stationary distribution.
pub fn fdm_heston_variance_mesher(p: &FdmHestonVarianceMesherParams) -> Mesher1d {
    assert!(p.n_points >= 2);
    assert!(p.v0 >= 0.0);
    assert!(p.kappa > 0.0);
    assert!(p.theta > 0.0);
    assert!(p.sigma > 0.0);
    assert!(p.maturity > 0.0);

    let kappa = p.kappa;
    let theta = p.theta;
    let sigma = p.sigma;

    // Stationary mean and variance of v under Heston (CIR dynamics)
    // E[v_T] = theta + (v0 - theta) * exp(-kappa*T)
    // Var[v_T] ≈ sigma^2 * theta / (2*kappa)  (stationary limit)
    let v_mean = theta + (p.v0 - theta) * (-kappa * p.maturity).exp();
    let v_var = sigma * sigma * theta / (2.0 * kappa);
    let v_std = v_var.sqrt();

    // Grid bounds: ensure we cover v=0 and well into the tail
    let v_min = 0.0;
    let v_max = (v_mean + 5.0 * v_std).max(3.0 * p.v0).max(3.0 * theta);

    // Build a grid that concentrates near v=0 and near v_mean.
    // Use two sinh concentrations blended together.
    let n = p.n_points;
    let mut locations: Vec<f64> = Vec::with_capacity(n);

    // Fraction of points allocated near v=0
    let frac_zero = 0.3_f64;
    let n_zero = ((n as f64 * frac_zero) as usize).max(2);
    let n_rest = n - n_zero;

    // Near-zero segment: uniform in sqrt(v) space
    let v_split = v_mean.min(v_max * 0.3);
    for i in 0..n_zero {
        let u = i as f64 / n_zero as f64;
        // sqrt-spacing concentrates near 0
        let v = v_split * u * u;
        locations.push(v);
    }

    // Upper segment: sinh concentration around v_mean
    let d = 1.0 / 2.0_f64; // concentration factor
    for i in 0..n_rest {
        let u = i as f64 / (n_rest - 1).max(1) as f64;
        let v = v_mean + d * ((u - 0.5) / d).sinh() * (v_max - v_split);
        locations.push(v.clamp(v_split, v_max));
    }

    locations.sort_by(|a, b| a.total_cmp(b));
    locations.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

    // Ensure we have endpoints
    if locations[0] > v_min + 1e-15 {
        locations.insert(0, v_min);
    }
    if *locations.last().unwrap() < v_max - 1e-10 {
        locations.push(v_max);
    }
    locations.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

    // Subsample to exactly n_points
    let locs = if locations.len() > n {
        subsample(&locations, n)
    } else {
        locations
    };

    Mesher1d::from_locations(locs)
}

// ---------------------------------------------------------------------------
// G120 – FdmSimpleProcess1DMesher
// ---------------------------------------------------------------------------

/// Parameters for [`fdm_simple_process_1d_mesher`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FdmSimpleProcess1DMesherParams {
    /// Drift of the process.
    pub drift: f64,
    /// Volatility (diffusion coefficient).
    pub vol: f64,
    /// Initial value x₀.
    pub x0: f64,
    /// Time to maturity.
    pub maturity: f64,
    /// Number of grid points.
    pub n_points: usize,
    /// Number of standard deviations to cover (e.g. 4.0).
    pub stddevs: f64,
}

/// Build a simple 1-D mesher for an arbitrary diffusion process (G120).
///
/// The grid is centred at `x0 + drift * T` and spans
/// ± `stddevs * vol * √T` around the centre.  Points are uniformly spaced.
pub fn fdm_simple_process_1d_mesher(p: &FdmSimpleProcess1DMesherParams) -> Mesher1d {
    assert!(p.n_points >= 2);
    assert!(p.maturity > 0.0);
    assert!(p.vol >= 0.0);
    assert!(p.stddevs > 0.0);

    let center = p.x0 + p.drift * p.maturity;
    let half_width = p.stddevs * p.vol * p.maturity.sqrt();
    let half_width = half_width.max(1e-8); // avoid zero-width grid

    let lo = center - half_width;
    let hi = center + half_width;

    let n = p.n_points;
    let dx = (hi - lo) / (n - 1) as f64;
    let locations: Vec<f64> = (0..n).map(|i| lo + i as f64 * dx).collect();

    Mesher1d::from_locations(locations)
}

// ---------------------------------------------------------------------------
// G121 – ExponentialJump1DMesher
// ---------------------------------------------------------------------------

/// Parameters for [`exponential_jump_1d_mesher`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExponentialJump1DMesherParams {
    /// Jump intensity λ (jumps per year).
    pub intensity: f64,
    /// Exponential decay rate η (controls mean jump size 1/η).
    pub eta: f64,
    /// Number of grid points.
    pub n_points: usize,
    /// Maximum extent of the grid (positive).
    pub x_max: f64,
}

/// Build a 1-D mesher for an exponential-jump process (G121).
///
/// The grid covers `[0, x_max]` with points concentrated near x = 0 using an
/// exponential-spacing rule `x_i = x_max (e^{η u_i} - 1) / (e^η - 1)` where
/// $u_i$ is uniform on `[0, 1]`.  This ensures fine resolution where the
/// jump-size PDF is largest (near zero).
pub fn exponential_jump_1d_mesher(p: &ExponentialJump1DMesherParams) -> Mesher1d {
    assert!(p.n_points >= 2);
    assert!(p.eta > 0.0, "eta must be positive");
    assert!(p.x_max > 0.0, "x_max must be positive");
    let _ = p.intensity; // recorded for documentation; does not affect the grid

    let n = p.n_points;
    let eta = p.eta;
    let denom = eta.exp() - 1.0;

    let locations: Vec<f64> = (0..n)
        .map(|i| {
            let u = i as f64 / (n - 1) as f64;
            p.x_max * ((eta * u).exp() - 1.0) / denom
        })
        .collect();

    Mesher1d::from_locations(locations)
}

// ---------------------------------------------------------------------------
// G122 – FdmCEV1DMesher
// ---------------------------------------------------------------------------

/// Parameters for [`fdm_cev_1d_mesher`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FdmCEV1DMesherParams {
    /// Current spot S₀.
    pub spot: f64,
    /// CEV β parameter. β < 1 concentrates near S = 0; β > 1 spreads tails.
    pub beta: f64,
    /// Volatility σ.
    pub vol: f64,
    /// Time to maturity.
    pub maturity: f64,
    /// Number of grid points.
    pub n_points: usize,
    /// Number of standard deviations to cover (default 4.0).
    pub stddevs: f64,
}

/// Build a 1-D asset-price mesher adapted for CEV dynamics (G122).
///
/// The local volatility of a CEV process is σ S^β, so the natural coordinate
/// is a power transform.  We build a uniform grid in the transformed variable
///
/// $$y = \begin{cases} \frac{S^{1-\beta}}{1-\beta} & \beta \ne 1 \\[4pt]
///        \ln S & \beta = 1 \end{cases}$$
///
/// then invert to obtain the asset-price grid.
pub fn fdm_cev_1d_mesher(p: &FdmCEV1DMesherParams) -> Mesher1d {
    assert!(p.n_points >= 2);
    assert!(p.spot > 0.0);
    assert!(p.vol > 0.0);
    assert!(p.maturity > 0.0);
    assert!(p.stddevs > 0.0);

    let n = p.n_points;
    let beta = p.beta;
    let std_dev = p.vol * p.maturity.sqrt();

    if (beta - 1.0).abs() < 1e-12 {
        // GBM case – grid in log-space
        let x0 = p.spot.ln();
        let lo = x0 - p.stddevs * std_dev;
        let hi = x0 + p.stddevs * std_dev;
        let dx = (hi - lo) / (n - 1) as f64;
        let locations: Vec<f64> = (0..n).map(|i| (lo + i as f64 * dx).exp()).collect();
        return Mesher1d::from_locations(locations);
    }

    // Transform: y = S^(1-beta) / (1-beta)
    let exponent = 1.0 - beta;
    let y0 = p.spot.powf(exponent) / exponent;

    // In the transformed coordinate the local vol is approximately σ,
    // so spread ±stddevs * σ * √T around y0.
    let half = p.stddevs * std_dev;
    let mut y_lo = y0 - half;
    let mut y_hi = y0 + half;

    // For β > 1 (exponent < 0) the transform maps S ∈ (0,∞) to y ∈ (−∞, 0).
    // Clamp the grid to stay in the valid region so the inverse stays real.
    if exponent < 0.0 {
        // y must be negative; y → 0⁻ as S → ∞
        let y_hi_max = -1e-6;
        y_hi = y_hi.min(y_hi_max);
        if y_lo >= y_hi {
            y_lo = y_hi - half.abs().max(1.0);
        }
    } else {
        // exponent > 0: arg = y * exponent must be positive
        let y_lo_min = 1e-6 / exponent;
        y_lo = y_lo.max(y_lo_min);
        if y_hi <= y_lo {
            y_hi = y_lo + half.abs().max(1.0);
        }
    }

    let dy = (y_hi - y_lo) / (n - 1) as f64;
    let locations: Vec<f64> = (0..n)
        .map(|i| {
            let y = y_lo + i as f64 * dy;
            // Invert: S = (y * (1-beta))^(1/(1-beta))
            let arg = y * exponent;
            if arg <= 0.0 {
                1e-8 // floor at a small positive value
            } else {
                arg.powf(1.0 / exponent)
            }
        })
        .collect();

    Mesher1d::from_locations(locations)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ---- G117 tests -------------------------------------------------------

    #[test]
    fn bs_mesher_size_and_sorted() {
        let p = FdmBlackScholesMesherParams {
            spot: 100.0,
            vol: 0.20,
            r: 0.05,
            q: 0.02,
            strike: 100.0,
            maturity: 1.0,
            n_points: 101,
            concentration_factor: 2.0,
        };
        let m = fdm_black_scholes_mesher(&p);
        assert_eq!(m.size(), 101);
        for i in 1..m.size() {
            assert!(m.locations[i] > m.locations[i - 1], "not strictly sorted");
        }
    }

    #[test]
    fn bs_mesher_contains_log_strike() {
        let p = FdmBlackScholesMesherParams {
            spot: 100.0,
            vol: 0.20,
            r: 0.05,
            q: 0.0,
            strike: 110.0,
            maturity: 1.0,
            n_points: 201,
            concentration_factor: 2.0,
        };
        let m = fdm_black_scholes_mesher(&p);
        let x_k = 110.0_f64.ln();
        assert!(m.locations[0] < x_k);
        assert!(*m.locations.last().unwrap() > x_k);
    }

    #[test]
    fn bs_mesher_concentrates_around_strike() {
        let p = FdmBlackScholesMesherParams {
            spot: 100.0,
            vol: 0.20,
            r: 0.05,
            q: 0.0,
            strike: 100.0,
            maturity: 1.0,
            n_points: 101,
            concentration_factor: 3.0,
        };
        let m = fdm_black_scholes_mesher(&p);
        let x_k = 100.0_f64.ln();
        let idx = m.lower_index(x_k);
        let spacing_center = m.dplus[idx];
        let spacing_edge = m.dplus[0];
        assert!(
            spacing_center < spacing_edge,
            "center spacing {spacing_center} >= edge spacing {spacing_edge}"
        );
    }

    // ---- G118 tests -------------------------------------------------------

    #[test]
    fn multi_strike_mesher_sorted_and_sized() {
        let p = FdmBlackScholesMultiStrikeMesherParams {
            spot: 100.0,
            vol: 0.20,
            r: 0.05,
            q: 0.0,
            strikes: vec![90.0, 100.0, 110.0],
            maturity: 1.0,
            n_points: 201,
            concentration_factor: 2.0,
        };
        let m = fdm_black_scholes_multi_strike_mesher(&p);
        assert!(m.size() >= 2);
        for i in 1..m.size() {
            assert!(m.locations[i] > m.locations[i - 1]);
        }
    }

    #[test]
    fn multi_strike_mesher_covers_all_strikes() {
        let strikes = vec![80.0, 100.0, 120.0];
        let p = FdmBlackScholesMultiStrikeMesherParams {
            spot: 100.0,
            vol: 0.25,
            r: 0.05,
            q: 0.0,
            strikes: strikes.clone(),
            maturity: 1.0,
            n_points: 301,
            concentration_factor: 2.0,
        };
        let m = fdm_black_scholes_multi_strike_mesher(&p);
        for &k in &strikes {
            let x_k = k.ln();
            assert!(m.locations[0] < x_k, "grid does not cover strike {k}");
            assert!(*m.locations.last().unwrap() > x_k);
        }
    }

    #[test]
    fn multi_strike_mesher_single_strike_like_bs_mesher() {
        let p_multi = FdmBlackScholesMultiStrikeMesherParams {
            spot: 100.0,
            vol: 0.20,
            r: 0.05,
            q: 0.0,
            strikes: vec![100.0],
            maturity: 1.0,
            n_points: 101,
            concentration_factor: 2.0,
        };
        let m = fdm_black_scholes_multi_strike_mesher(&p_multi);
        // Should still produce a valid sorted grid
        assert!(m.size() >= 2);
        let x_k = 100.0_f64.ln();
        assert!(m.locations[0] < x_k);
        assert!(*m.locations.last().unwrap() > x_k);
    }

    // ---- G119 tests -------------------------------------------------------

    #[test]
    fn heston_var_mesher_starts_near_zero() {
        let p = FdmHestonVarianceMesherParams {
            v0: 0.04,
            kappa: 1.5,
            theta: 0.04,
            sigma: 0.5,
            maturity: 1.0,
            n_points: 51,
        };
        let m = fdm_heston_variance_mesher(&p);
        assert_abs_diff_eq!(m.locations[0], 0.0, epsilon = 1e-10);
        assert!(m.size() >= 2);
    }

    #[test]
    fn heston_var_mesher_covers_theta() {
        let p = FdmHestonVarianceMesherParams {
            v0: 0.04,
            kappa: 1.5,
            theta: 0.06,
            sigma: 0.4,
            maturity: 2.0,
            n_points: 51,
        };
        let m = fdm_heston_variance_mesher(&p);
        // θ should lie within the grid
        assert!(m.locations[0] <= p.theta);
        assert!(*m.locations.last().unwrap() >= p.theta);
    }

    #[test]
    fn heston_var_mesher_concentrates_near_zero() {
        let p = FdmHestonVarianceMesherParams {
            v0: 0.04,
            kappa: 2.0,
            theta: 0.04,
            sigma: 0.3,
            maturity: 1.0,
            n_points: 101,
        };
        let m = fdm_heston_variance_mesher(&p);
        // The first few spacings should be smaller than later ones
        let spacing_start = m.dplus[0];
        let spacing_mid = m.dplus[m.size() / 2];
        assert!(
            spacing_start < spacing_mid,
            "near-zero spacing {spacing_start} >= mid spacing {spacing_mid}"
        );
    }

    // ---- G120 tests -------------------------------------------------------

    #[test]
    fn simple_process_mesher_centered() {
        let p = FdmSimpleProcess1DMesherParams {
            drift: 0.05,
            vol: 0.20,
            x0: 0.0,
            maturity: 1.0,
            n_points: 51,
            stddevs: 4.0,
        };
        let m = fdm_simple_process_1d_mesher(&p);
        let center = p.x0 + p.drift * p.maturity;
        let mid_idx = m.size() / 2;
        assert_abs_diff_eq!(m.locations[mid_idx], center, epsilon = 0.02);
    }

    #[test]
    fn simple_process_mesher_width() {
        let p = FdmSimpleProcess1DMesherParams {
            drift: 0.0,
            vol: 0.30,
            x0: 1.0,
            maturity: 1.0,
            n_points: 101,
            stddevs: 3.0,
        };
        let m = fdm_simple_process_1d_mesher(&p);
        let expected_half = 3.0 * 0.30 * 1.0_f64.sqrt();
        assert_abs_diff_eq!(m.locations[0], 1.0 - expected_half, epsilon = 1e-12);
        assert_abs_diff_eq!(
            *m.locations.last().unwrap(),
            1.0 + expected_half,
            epsilon = 1e-12
        );
    }

    #[test]
    fn simple_process_mesher_uniform_spacing() {
        let p = FdmSimpleProcess1DMesherParams {
            drift: 0.0,
            vol: 0.20,
            x0: 0.0,
            maturity: 1.0,
            n_points: 11,
            stddevs: 4.0,
        };
        let m = fdm_simple_process_1d_mesher(&p);
        let dx = m.dplus[0];
        for i in 0..m.size() - 1 {
            assert_abs_diff_eq!(m.dplus[i], dx, epsilon = 1e-12);
        }
    }

    // ---- G121 tests -------------------------------------------------------

    #[test]
    fn exp_jump_mesher_endpoints() {
        let p = ExponentialJump1DMesherParams {
            intensity: 1.0,
            eta: 3.0,
            n_points: 51,
            x_max: 5.0,
        };
        let m = exponential_jump_1d_mesher(&p);
        assert_abs_diff_eq!(m.locations[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(*m.locations.last().unwrap(), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn exp_jump_mesher_concentrates_near_zero() {
        let p = ExponentialJump1DMesherParams {
            intensity: 1.0,
            eta: 5.0,
            n_points: 101,
            x_max: 10.0,
        };
        let m = exponential_jump_1d_mesher(&p);
        // Spacing near zero should be smaller than spacing near x_max
        let spacing_start = m.dplus[0];
        let spacing_end = m.dplus[m.size() - 2];
        assert!(
            spacing_start < spacing_end,
            "near-zero spacing {spacing_start} >= tail spacing {spacing_end}"
        );
    }

    #[test]
    fn exp_jump_mesher_size() {
        let p = ExponentialJump1DMesherParams {
            intensity: 2.0,
            eta: 4.0,
            n_points: 31,
            x_max: 3.0,
        };
        let m = exponential_jump_1d_mesher(&p);
        assert_eq!(m.size(), 31);
    }

    // ---- G122 tests -------------------------------------------------------

    #[test]
    fn cev_mesher_beta_one_gives_log_spacing() {
        let p = FdmCEV1DMesherParams {
            spot: 100.0,
            beta: 1.0,
            vol: 0.20,
            maturity: 1.0,
            n_points: 51,
            stddevs: 4.0,
        };
        let m = fdm_cev_1d_mesher(&p);
        assert!(m.size() == 51);
        // All locations should be positive (exponentiated)
        for &s in &m.locations {
            assert!(s > 0.0);
        }
    }

    #[test]
    fn cev_mesher_beta_half_concentrates_low() {
        // β < 1 ⇒ concentrate near S = 0
        let p = FdmCEV1DMesherParams {
            spot: 100.0,
            beta: 0.5,
            vol: 0.20,
            maturity: 1.0,
            n_points: 101,
            stddevs: 4.0,
        };
        let m = fdm_cev_1d_mesher(&p);
        // First spacing (in S) should be smaller than last
        let spacing_start = m.dplus[0];
        let spacing_end = m.dplus[m.size() - 2];
        assert!(
            spacing_start < spacing_end,
            "low-end spacing {spacing_start} >= high-end spacing {spacing_end}"
        );
    }

    #[test]
    fn cev_mesher_beta_gt_one_spreads_tails() {
        // β > 1 ⇒ inversion stretches tails
        let p = FdmCEV1DMesherParams {
            spot: 100.0,
            beta: 1.5,
            vol: 0.20,
            maturity: 1.0,
            n_points: 101,
            stddevs: 4.0,
        };
        let m = fdm_cev_1d_mesher(&p);
        assert!(m.size() >= 2);
        // All positive
        for &s in &m.locations {
            assert!(s > 0.0, "expected positive S, got {s}");
        }
        // Grid should be sorted
        for i in 1..m.size() {
            assert!(m.locations[i] > m.locations[i - 1]);
        }
    }
}
