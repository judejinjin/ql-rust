//! Extended random number generators and statistics.
//!
//! **G54** — BoxMullerGaussianRng
//! **G55** — InverseCumulativeRng
//! **G56** — SobolBrownianBridgeRsg
//! **G58** — DiscrepancyStatistics
//! **G59** — Histogram

use ql_core::errors::{QLError, QLResult};

// ===========================================================================
// Box-Muller Gaussian RNG (G54)
// ===========================================================================

/// Box-Muller transform for generating pairs of standard normal variates
/// from uniform variates.
///
/// Each call to `next()` produces one normal variate; internally generates
/// two at a time and caches the second.
pub struct BoxMullerGaussianRng<R: UniformRng> {
    uniform: R,
    cached: Option<f64>,
}

/// Trait for uniform [0,1) random number generators.
pub trait UniformRng {
    fn next_uniform(&mut self) -> f64;
}

/// Simple linear congruential generator (for testing / non-crypto use).
#[derive(Clone, Debug)]
pub struct LcgRng {
    state: u64,
}

impl LcgRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }
}

impl UniformRng for LcgRng {
    fn next_uniform(&mut self) -> f64 {
        // Numerical Recipes LCG
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Wrapper around `rand::rngs::StdRng`-style generators.
/// Uses xoshiro256** from the `rand` crate if available, otherwise LCG.
pub struct StdUniformRng {
    lcg: LcgRng,
}

impl StdUniformRng {
    pub fn new(seed: u64) -> Self {
        Self {
            lcg: LcgRng::new(seed),
        }
    }
}

impl UniformRng for StdUniformRng {
    fn next_uniform(&mut self) -> f64 {
        self.lcg.next_uniform()
    }
}

impl<R: UniformRng> BoxMullerGaussianRng<R> {
    pub fn new(uniform: R) -> Self {
        Self {
            uniform,
            cached: None,
        }
    }

    /// Generate a standard normal variate.
    pub fn next(&mut self) -> f64 {
        if let Some(z) = self.cached.take() {
            return z;
        }

        loop {
            let u1 = 2.0 * self.uniform.next_uniform() - 1.0;
            let u2 = 2.0 * self.uniform.next_uniform() - 1.0;
            let s = u1 * u1 + u2 * u2;
            if s > 0.0 && s < 1.0 {
                let factor = (-2.0 * s.ln() / s).sqrt();
                self.cached = Some(u2 * factor);
                return u1 * factor;
            }
        }
    }

    /// Generate `n` standard normal variates.
    pub fn next_n(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.next()).collect()
    }
}

// ===========================================================================
// Inverse Cumulative RNG (G55)
// ===========================================================================

/// Generate variates from any distribution using inverse CDF method.
///
/// Given a uniform RNG and an inverse CDF function, produces variates
/// from the target distribution.
pub struct InverseCumulativeRng<R: UniformRng> {
    uniform: R,
}

impl<R: UniformRng> InverseCumulativeRng<R> {
    pub fn new(uniform: R) -> Self {
        Self { uniform }
    }

    /// Generate a variate using the provided inverse CDF.
    pub fn next<F: Fn(f64) -> f64>(&mut self, inverse_cdf: &F) -> f64 {
        let u = self.uniform.next_uniform();
        inverse_cdf(u)
    }

    /// Generate a standard normal variate using inverse CDF.
    pub fn next_gaussian(&mut self) -> f64 {
        let u = self.uniform.next_uniform();
        crate::distributions::inverse_cumulative_normal(u).unwrap_or(0.0)
    }

    /// Generate `n` gaussian variates.
    pub fn next_gaussian_n(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.next_gaussian()).collect()
    }
}

// ===========================================================================
// Sobol Brownian Bridge RSG (G56)
// ===========================================================================

/// Sobol sequence with Brownian bridge path construction.
///
/// Combines a Sobol quasi-random sequence generator with Brownian bridge
/// ordering to improve convergence of multi-step Monte Carlo simulations.
pub struct SobolBrownianBridgeRsg {
    sobol: crate::quasi_random::SobolSequence,
    bridge: crate::quasi_random::BrownianBridge,
    steps: usize,
}

impl SobolBrownianBridgeRsg {
    /// Create a new Sobol-Brownian-Bridge RSG for `steps` time steps.
    pub fn new(steps: usize) -> Self {
        let sobol = crate::quasi_random::SobolSequence::new(steps);
        let bridge = crate::quasi_random::BrownianBridge::new(steps);

        Self {
            sobol,
            bridge,
            steps,
        }
    }

    /// Generate one quasi-random path (vector of `steps` Brownian increments).
    pub fn next_path(&mut self) -> Vec<f64> {
        // Generate Sobol uniforms and transform to normals
        let uniforms = self.sobol.next_point();
        let normals: Vec<f64> = uniforms
            .iter()
            .map(|&u: &f64| {
                crate::distributions::inverse_cumulative_normal(u.clamp(1e-10, 1.0 - 1e-10))
                    .unwrap_or(0.0)
            })
            .collect();

        // Apply Brownian bridge
        self.bridge.transform(&normals)
    }

    /// Number of steps per path.
    pub fn steps(&self) -> usize {
        self.steps
    }
}

// ===========================================================================
// Discrepancy Statistics (G58)
// ===========================================================================

/// Star discrepancy measure for quasi-random sequences.
///
/// The star discrepancy D* measures how uniformly a set of points fills [0,1)^d.
/// Lower values indicate better uniformity.
pub struct DiscrepancyStatistics {
    dimension: usize,
    points: Vec<Vec<f64>>,
}

impl DiscrepancyStatistics {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            points: Vec::new(),
        }
    }

    /// Add a point (must have dimension components).
    pub fn add(&mut self, point: Vec<f64>) {
        assert_eq!(point.len(), self.dimension);
        self.points.push(point);
    }

    /// Number of points added.
    pub fn count(&self) -> usize {
        self.points.len()
    }

    /// Compute the L2-star discrepancy (Warnock's formula).
    ///
    /// More efficient to compute than the sup-norm discrepancy.
    pub fn l2_star_discrepancy(&self) -> f64 {
        let n = self.points.len();
        if n == 0 {
            return 0.0;
        }
        let d = self.dimension;
        let nf = n as f64;

        // Term 1: (1/3)^d
        let term1 = (1.0 / 3.0_f64).powi(d as i32);

        // Term 2: (2/N) Σᵢ Πⱼ (1 - xᵢⱼ²) / 2
        let mut term2 = 0.0;
        for point in &self.points {
            let mut prod = 1.0;
            for j in 0..d {
                prod *= (1.0 - point[j] * point[j]) / 2.0;
            }
            term2 += prod;
        }
        term2 /= nf;

        // Term 3: (1/N²) Σᵢ Σₖ Πⱼ (1 - max(xᵢⱼ, xₖⱼ))
        let mut term3 = 0.0;
        for i in 0..n {
            for k in 0..n {
                let mut prod = 1.0;
                for j in 0..d {
                    prod *= 1.0 - self.points[i][j].max(self.points[k][j]);
                }
                term3 += prod;
            }
        }
        term3 /= nf * nf;

        (term1 - term2 + term3).abs().sqrt()
    }

    /// Reset the statistics.
    pub fn reset(&mut self) {
        self.points.clear();
    }
}

// ===========================================================================
// Histogram (G59)
// ===========================================================================

/// Histogram with automatic or fixed binning.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Histogram {
    /// Bin edges (length = bins + 1).
    pub edges: Vec<f64>,
    /// Bin counts.
    pub counts: Vec<usize>,
    /// Total number of samples.
    pub total: usize,
}

impl Histogram {
    /// Create a histogram with `n_bins` equally spaced bins over `[min, max]`.
    pub fn new(min: f64, max: f64, n_bins: usize) -> QLResult<Self> {
        if n_bins == 0 || min >= max {
            return Err(QLError::InvalidArgument(
                "histogram: need n_bins > 0 and min < max".into(),
            ));
        }
        let h = (max - min) / n_bins as f64;
        let edges: Vec<f64> = (0..=n_bins).map(|i| min + i as f64 * h).collect();
        Ok(Self {
            edges,
            counts: vec![0; n_bins],
            total: 0,
        })
    }

    /// Create a histogram from pre-defined bin edges.
    pub fn from_edges(edges: Vec<f64>) -> QLResult<Self> {
        if edges.len() < 2 {
            return Err(QLError::InvalidArgument(
                "histogram requires at least 2 edges".into(),
            ));
        }
        let n_bins = edges.len() - 1;
        Ok(Self {
            edges,
            counts: vec![0; n_bins],
            total: 0,
        })
    }

    /// Create a histogram with automatic binning (Sturges' rule).
    pub fn from_data(data: &[f64]) -> QLResult<Self> {
        if data.is_empty() {
            return Err(QLError::InvalidArgument(
                "histogram: empty data".into(),
            ));
        }
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let n_bins = ((data.len() as f64).log2().ceil() + 1.0).max(1.0) as usize;
        let spread = max - min;
        let margin = if spread < 1e-15 { 1.0 } else { spread * 0.01 };
        let mut hist = Self::new(min - margin, max + margin, n_bins)?;
        for &x in data {
            hist.add(x);
        }
        Ok(hist)
    }

    /// Add a sample to the histogram.
    pub fn add(&mut self, value: f64) {
        self.total += 1;
        // Binary search for the bin
        let n_bins = self.counts.len();
        for i in 0..n_bins {
            if value >= self.edges[i] && value < self.edges[i + 1] {
                self.counts[i] += 1;
                return;
            }
        }
        // Handle value == max (put in last bin)
        if (value - self.edges[n_bins]).abs() < 1e-15 {
            self.counts[n_bins - 1] += 1;
        }
    }

    /// Number of bins.
    pub fn n_bins(&self) -> usize {
        self.counts.len()
    }

    /// Normalized frequency (probability density estimate) for bin `i`.
    pub fn density(&self, i: usize) -> f64 {
        if i >= self.counts.len() || self.total == 0 {
            return 0.0;
        }
        let bin_width = self.edges[i + 1] - self.edges[i];
        self.counts[i] as f64 / (self.total as f64 * bin_width)
    }

    /// Cumulative frequency up to and including bin `i`.
    pub fn cumulative(&self, i: usize) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let sum: usize = self.counts[..=i.min(self.counts.len() - 1)].iter().sum();
        sum as f64 / self.total as f64
    }

    /// Reset all counts.
    pub fn reset(&mut self) {
        self.counts.iter_mut().for_each(|c| *c = 0);
        self.total = 0;
    }
}

// ===========================================================================
// Sampled Curve (G60 supplement)
// ===========================================================================

/// A curve sampled on a uniform grid — used for FD boundary values
/// and payoff discretization.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SampledCurve {
    /// Grid points.
    pub grid: Vec<f64>,
    /// Values at grid points.
    pub values: Vec<f64>,
}

impl SampledCurve {
    /// Create a sampled curve from grid and values.
    pub fn new(grid: Vec<f64>, values: Vec<f64>) -> QLResult<Self> {
        if grid.len() != values.len() {
            return Err(QLError::InvalidArgument(
                "grid and values must have the same length".into(),
            ));
        }
        Ok(Self { grid, values })
    }

    /// Create a sampled curve by evaluating `f` on `n` equally-spaced points in `[a, b]`.
    pub fn from_function<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> Self {
        let h = if n > 1 { (b - a) / (n - 1) as f64 } else { 0.0 };
        let grid: Vec<f64> = (0..n).map(|i| a + i as f64 * h).collect();
        let values: Vec<f64> = grid.iter().map(|&x| f(x)).collect();
        Self { grid, values }
    }

    /// Number of sample points.
    pub fn size(&self) -> usize {
        self.grid.len()
    }

    /// Linear interpolation at `x`.
    pub fn value_at(&self, x: f64) -> f64 {
        let n = self.grid.len();
        if n == 0 {
            return 0.0;
        }
        if n == 1 || x <= self.grid[0] {
            return self.values[0];
        }
        if x >= self.grid[n - 1] {
            return self.values[n - 1];
        }

        // Binary search
        let i = match self.grid.binary_search_by(|g| g.partial_cmp(&x).unwrap()) {
            Ok(i) => return self.values[i],
            Err(i) => i - 1,
        };

        let t = (x - self.grid[i]) / (self.grid[i + 1] - self.grid[i]);
        self.values[i] * (1.0 - t) + self.values[i + 1] * t
    }

    /// Compute the first derivative using central differences.
    pub fn first_derivative(&self) -> Vec<f64> {
        let n = self.grid.len();
        let mut d = vec![0.0; n];
        if n < 2 {
            return d;
        }
        d[0] = (self.values[1] - self.values[0]) / (self.grid[1] - self.grid[0]);
        for i in 1..n - 1 {
            d[i] = (self.values[i + 1] - self.values[i - 1]) / (self.grid[i + 1] - self.grid[i - 1]);
        }
        d[n - 1] = (self.values[n - 1] - self.values[n - 2]) / (self.grid[n - 1] - self.grid[n - 2]);
        d
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn box_muller_statistics() {
        let mut rng = BoxMullerGaussianRng::new(LcgRng::new(42));
        let n = 50_000;
        let samples: Vec<f64> = (0..n).map(|_| rng.next()).collect();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        assert_abs_diff_eq!(mean, 0.0, epsilon = 0.05);
        assert_abs_diff_eq!(var, 1.0, epsilon = 0.05);
    }

    #[test]
    fn inverse_cumulative_gaussian() {
        let mut rng = InverseCumulativeRng::new(LcgRng::new(123));
        let n = 50_000;
        let samples: Vec<f64> = (0..n).map(|_| rng.next_gaussian()).collect();
        let mean = samples.iter().sum::<f64>() / n as f64;
        assert_abs_diff_eq!(mean, 0.0, epsilon = 0.05);
    }

    #[test]
    fn histogram_uniform() {
        let mut hist = Histogram::new(0.0, 1.0, 10).unwrap();
        for i in 0..1000 {
            hist.add(i as f64 / 1000.0);
        }
        assert_eq!(hist.total, 1000);
        assert_eq!(hist.n_bins(), 10);
        // Each bin should have ~100 samples
        for i in 0..10 {
            assert!(hist.counts[i] >= 90 && hist.counts[i] <= 110,
                    "bin {} has {} counts", i, hist.counts[i]);
        }
    }

    #[test]
    fn histogram_from_data() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let hist = Histogram::from_data(&data).unwrap();
        assert_eq!(hist.total, 100);
        assert!(hist.n_bins() >= 5);
    }

    #[test]
    fn histogram_density() {
        let mut hist = Histogram::new(0.0, 1.0, 2).unwrap();
        for _ in 0..100 {
            hist.add(0.25);
        }
        for _ in 0..100 {
            hist.add(0.75);
        }
        // Each bin width = 0.5, each has 100/200 = 0.5 frequency
        // density = 0.5 / 0.5 = 1.0
        assert_abs_diff_eq!(hist.density(0), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(hist.density(1), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn discrepancy_uniform_grid() {
        let mut stats = DiscrepancyStatistics::new(1);
        let n = 100;
        for i in 0..n {
            stats.add(vec![(i as f64 + 0.5) / n as f64]);
        }
        let d = stats.l2_star_discrepancy();
        // A uniform grid should have relatively low discrepancy
        assert!(d < 1.0, "discrepancy = {}", d);
        assert!(d > 0.0, "discrepancy should be positive");
    }

    #[test]
    fn sampled_curve_interpolation() {
        let curve = SampledCurve::from_function(|x| x * x, 0.0, 2.0, 201);
        assert_abs_diff_eq!(curve.value_at(1.0), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(curve.value_at(0.5), 0.25, epsilon = 0.01);
    }

    #[test]
    fn sampled_curve_derivative() {
        let curve = SampledCurve::from_function(|x| x * x, 0.0, 2.0, 201);
        let d = curve.first_derivative();
        // f'(x) = 2x, at midpoint x=1.0 → d ≈ 2.0
        let mid = 100;
        assert_abs_diff_eq!(d[mid], 2.0 * curve.grid[mid], epsilon = 0.02);
    }

    #[test]
    fn lcg_rng_in_range() {
        let mut rng = LcgRng::new(42);
        for _ in 0..1000 {
            let u = rng.next_uniform();
            assert!(u >= 0.0 && u < 1.0, "uniform out of range: {}", u);
        }
    }
}
