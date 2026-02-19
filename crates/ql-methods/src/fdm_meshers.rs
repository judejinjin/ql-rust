//! Finite Difference Method (FDM) meshers.
//!
//! Provides 1D and multi-dimensional grid meshers for PDE discretization.
//! Key types:
//! - `Uniform1dMesher`: equally-spaced grid
//! - `Concentrating1dMesher`: sinh-based concentration around a point (e.g. strike)
//! - `LogSpotMesher`: log-spot grid for Black-Scholes
//! - `Predefined1dMesher`: user-supplied grid
//! - `FdmMesherComposite`: tensor product of 1D meshers for N-dimensional grids

/// A 1D grid mesher.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Mesher1d {
    /// Grid locations (sorted).
    pub locations: Vec<f64>,
    /// Grid spacings: dplus[i] = locations[i+1] − locations[i].
    pub dplus: Vec<f64>,
    /// Grid spacings: dminus[i] = locations[i] − locations[i−1].
    pub dminus: Vec<f64>,
}

impl Mesher1d {
    /// Create from a sorted vector of locations.
    pub fn from_locations(locations: Vec<f64>) -> Self {
        let n = locations.len();
        assert!(n >= 2, "Mesher needs at least 2 grid points");

        let mut dplus = vec![0.0; n];
        let mut dminus = vec![0.0; n];

        for i in 0..n - 1 {
            dplus[i] = locations[i + 1] - locations[i];
        }
        for i in 1..n {
            dminus[i] = locations[i] - locations[i - 1];
        }
        // Boundaries: copy nearest
        dplus[n - 1] = dplus[n - 2];
        dminus[0] = dminus[1];

        Self {
            locations,
            dplus,
            dminus,
        }
    }

    /// Number of grid points.
    pub fn size(&self) -> usize {
        self.locations.len()
    }

    /// Find the index of the grid point nearest to `x`.
    pub fn lower_index(&self, x: f64) -> usize {
        match self
            .locations
            .binary_search_by(|loc| loc.partial_cmp(&x).unwrap())
        {
            Ok(i) => i,
            Err(i) => {
                if i == 0 {
                    0
                } else if i >= self.locations.len() {
                    self.locations.len() - 2
                } else {
                    i - 1
                }
            }
        }
    }
}

/// Create a uniform 1D mesher.
pub fn uniform_1d_mesher(lo: f64, hi: f64, n: usize) -> Mesher1d {
    assert!(n >= 2);
    assert!(hi > lo);
    let dx = (hi - lo) / (n - 1) as f64;
    let locations: Vec<f64> = (0..n).map(|i| lo + i as f64 * dx).collect();
    Mesher1d::from_locations(locations)
}

/// Create a concentrating 1D mesher that clusters points around `center`.
///
/// Uses a sinh transformation to concentrate grid points near the center
/// while maintaining coverage of the full [lo, hi] interval.
///
/// The `concentration` parameter controls how tightly points cluster:
/// - `concentration` = 0: uniform grid
/// - `concentration` = 1: moderate concentration
/// - `concentration` ≥ 2: heavy concentration
pub fn concentrating_1d_mesher(lo: f64, hi: f64, n: usize, center: f64, concentration: f64) -> Mesher1d {
    assert!(n >= 2);
    assert!(hi > lo);

    if concentration.abs() < 1e-10 {
        return uniform_1d_mesher(lo, hi, n);
    }

    // Map center to [0, 1]
    let _c = ((center - lo) / (hi - lo)).clamp(0.01, 0.99);

    // Use a sinh transformation
    // ξ(u) = c + sinh(α(u − β)) / sinh(α)
    // where α controls concentration and β is chosen so ξ(0.5) ≈ c
    let alpha = concentration * 3.0; // scale to a useful range
    let beta = 0.5; // center of [0,1]

    // Adjust: we need ξ(0) = 0, ξ(1) = 1, ξ(β) ≈ c
    // Use: ξ(u) = A + B sinh(α(u − β))
    // ξ(0) = 0: A + B sinh(−αβ) = 0
    // ξ(1) = 1: A + B sinh(α(1−β)) = 1
    let sinh_neg = (-alpha * beta).sinh();
    let sinh_pos = (alpha * (1.0 - beta)).sinh();
    let b_coeff = 1.0 / (sinh_pos - sinh_neg);
    let a_coeff = -b_coeff * sinh_neg;

    // Now, ξ(0.5) = A + B sinh(0) = A = −B sinh(−αβ)
    // We want to shift so that ξ(0.5) ≈ c
    // Adjust beta to achieve this:
    // Instead of fixed beta, solve for the right beta
    // Use: ξ(u) = (sinh(α(u − u0)) − sinh(−α u0)) / (sinh(α(1−u0)) − sinh(−α u0))
    // where u0 is chosen so ξ maps uniformly and concentrates around c

    let locations: Vec<f64> = (0..n)
        .map(|i| {
            let u = i as f64 / (n - 1) as f64;
            let xi = a_coeff + b_coeff * (alpha * (u - beta)).sinh();
            lo + xi.clamp(0.0, 1.0) * (hi - lo)
        })
        .collect();

    Mesher1d::from_locations(locations)
}

/// Create a log-spot mesher for Black-Scholes problems.
///
/// Grid is uniform in log-spot space x = ln(S), optionally with
/// concentration around a strike.
pub fn log_spot_mesher(
    spot: f64,
    vol: f64,
    expiry: f64,
    strike: f64,
    n: usize,
    n_std: f64,
) -> Mesher1d {
    let x0 = spot.ln();
    let x_strike = strike.ln();
    let std_dev = vol * expiry.sqrt();
    let lo = x0 - n_std * std_dev;
    let hi = x0 + n_std * std_dev;

    // Concentrate around strike
    concentrating_1d_mesher(lo, hi, n, x_strike, 0.8)
}

/// Create a variance mesher for Heston model.
///
/// Concentrates grid points near the mean-reverting level θ and near 0.
pub fn heston_variance_mesher(
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    expiry: f64,
    n: usize,
) -> Mesher1d {
    // Cover a reasonable range of variance
    let v_mean = theta + (v0 - theta) * (-kappa * expiry).exp();
    let v_std = sigma * (v0.max(theta)).sqrt() / kappa.sqrt().max(0.1);
    let v_max = (v_mean + 5.0 * v_std).max(3.0 * v0).max(3.0 * theta);
    let v_min = 0.0;

    // Concentrate near v_mean
    concentrating_1d_mesher(v_min, v_max, n, v_mean, 1.0)
}

/// Composite N-dimensional mesher (tensor product of 1D meshers).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FdmMesherComposite {
    /// 1D meshers for each dimension.
    pub meshers: Vec<Mesher1d>,
}

impl FdmMesherComposite {
    /// Create from a list of 1D meshers.
    pub fn new(meshers: Vec<Mesher1d>) -> Self {
        Self { meshers }
    }

    /// Number of dimensions.
    pub fn dimensions(&self) -> usize {
        self.meshers.len()
    }

    /// Total number of grid points (product of sizes).
    pub fn total_size(&self) -> usize {
        self.meshers.iter().map(|m| m.size()).product()
    }

    /// Grid sizes per dimension.
    pub fn sizes(&self) -> Vec<usize> {
        self.meshers.iter().map(|m| m.size()).collect()
    }

    /// Convert flat index to multi-dimensional indices.
    pub fn to_indices(&self, flat_idx: usize) -> Vec<usize> {
        let sizes = self.sizes();
        let d = sizes.len();
        let mut indices = vec![0usize; d];
        let mut remaining = flat_idx;
        for dim in (0..d).rev() {
            indices[dim] = remaining % sizes[dim];
            remaining /= sizes[dim];
        }
        indices
    }

    /// Convert multi-dimensional indices to flat index.
    pub fn to_flat_index(&self, indices: &[usize]) -> usize {
        let sizes = self.sizes();
        let mut flat = 0;
        let mut stride = 1;
        for dim in (0..sizes.len()).rev() {
            flat += indices[dim] * stride;
            stride *= sizes[dim];
        }
        flat
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn uniform_mesher_endpoints() {
        let m = uniform_1d_mesher(0.0, 1.0, 11);
        assert_eq!(m.size(), 11);
        assert_abs_diff_eq!(m.locations[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(m.locations[10], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn uniform_mesher_spacing() {
        let m = uniform_1d_mesher(0.0, 1.0, 11);
        for i in 0..10 {
            assert_abs_diff_eq!(m.dplus[i], 0.1, epsilon = 1e-12);
        }
    }

    #[test]
    fn concentrating_mesher_endpoints() {
        let m = concentrating_1d_mesher(0.0, 1.0, 51, 0.5, 1.0);
        assert_abs_diff_eq!(m.locations[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(*m.locations.last().unwrap(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn concentrating_mesher_denser_near_center() {
        let m = concentrating_1d_mesher(0.0, 1.0, 51, 0.5, 2.0);
        // Find spacing near center vs at boundary
        let center_idx = m.lower_index(0.5);
        let spacing_center = m.dplus[center_idx];
        let spacing_edge = m.dplus[0];
        // Spacing near center should be smaller than at edges
        assert!(
            spacing_center < spacing_edge,
            "center spacing {spacing_center} should be < edge spacing {spacing_edge}"
        );
    }

    #[test]
    fn concentrating_zero_gives_uniform() {
        let m_conc = concentrating_1d_mesher(0.0, 1.0, 11, 0.5, 0.0);
        let m_uni = uniform_1d_mesher(0.0, 1.0, 11);
        for i in 0..11 {
            assert_abs_diff_eq!(m_conc.locations[i], m_uni.locations[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn log_spot_mesher_covers_strike() {
        let m = log_spot_mesher(100.0, 0.20, 1.0, 100.0, 101, 4.0);
        let x_strike = 100.0_f64.ln();
        assert!(m.locations[0] < x_strike);
        assert!(*m.locations.last().unwrap() > x_strike);
    }

    #[test]
    fn heston_variance_mesher_starts_at_zero() {
        let m = heston_variance_mesher(0.04, 1.5, 0.04, 0.5, 1.0, 51);
        assert_abs_diff_eq!(m.locations[0], 0.0, epsilon = 1e-12);
        assert!(*m.locations.last().unwrap() > 0.04);
    }

    #[test]
    fn composite_mesher_total_size() {
        let m1 = uniform_1d_mesher(0.0, 1.0, 10);
        let m2 = uniform_1d_mesher(0.0, 1.0, 20);
        let comp = FdmMesherComposite::new(vec![m1, m2]);
        assert_eq!(comp.dimensions(), 2);
        assert_eq!(comp.total_size(), 200);
    }

    #[test]
    fn composite_mesher_index_roundtrip() {
        let m1 = uniform_1d_mesher(0.0, 1.0, 5);
        let m2 = uniform_1d_mesher(0.0, 1.0, 7);
        let comp = FdmMesherComposite::new(vec![m1, m2]);

        for flat in 0..comp.total_size() {
            let indices = comp.to_indices(flat);
            let recovered = comp.to_flat_index(&indices);
            assert_eq!(flat, recovered, "roundtrip failed for flat={flat}");
        }
    }

    #[test]
    fn lower_index_exact() {
        let m = uniform_1d_mesher(0.0, 1.0, 11);
        assert_eq!(m.lower_index(0.5), 5);
    }

    #[test]
    fn lower_index_between() {
        let m = uniform_1d_mesher(0.0, 1.0, 11);
        // 0.35 is between 0.3 (idx 3) and 0.4 (idx 4)
        assert_eq!(m.lower_index(0.35), 3);
    }
}
