//! Statistics accumulators: general, incremental (online), risk metrics.

/// General statistics accumulator — collects all samples then computes.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GeneralStatistics {
    samples: Vec<f64>,
    sorted: bool,
}

impl GeneralStatistics {
    /// New.
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            sorted: false,
        }
    }

    /// Add.
    pub fn add(&mut self, value: f64) {
        self.samples.push(value);
        self.sorted = false;
    }

    /// Add weighted.
    pub fn add_weighted(&mut self, value: f64, _weight: f64) {
        // For simplicity, treat as unweighted
        self.samples.push(value);
        self.sorted = false;
    }

    /// Reset.
    pub fn reset(&mut self) {
        self.samples.clear();
        self.sorted = false;
    }

    /// Count.
    pub fn count(&self) -> usize {
        self.samples.len()
    }

    /// Mean.
    pub fn mean(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        self.samples.iter().sum::<f64>() / self.samples.len() as f64
    }

    /// Variance.
    pub fn variance(&self) -> f64 {
        let n = self.samples.len();
        if n < 2 {
            return 0.0;
        }
        let m = self.mean();
        let sum_sq: f64 = self.samples.iter().map(|&x| (x - m) * (x - m)).sum();
        sum_sq / (n - 1) as f64
    }

    /// Standard deviation.
    pub fn standard_deviation(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Skewness.
    pub fn skewness(&self) -> f64 {
        let n = self.samples.len();
        if n < 3 {
            return 0.0;
        }
        let m = self.mean();
        let s = self.standard_deviation();
        if s < 1e-30 {
            return 0.0;
        }
        let sum: f64 = self.samples.iter().map(|&x| ((x - m) / s).powi(3)).sum();
        let nf = n as f64;
        sum * nf / ((nf - 1.0) * (nf - 2.0))
    }

    /// Kurtosis.
    pub fn kurtosis(&self) -> f64 {
        let n = self.samples.len();
        if n < 4 {
            return 0.0;
        }
        let m = self.mean();
        let s = self.standard_deviation();
        if s < 1e-30 {
            return 0.0;
        }
        let sum: f64 = self.samples.iter().map(|&x| ((x - m) / s).powi(4)).sum();
        let nf = n as f64;
        // Excess kurtosis
        (nf * (nf + 1.0) * sum) / ((nf - 1.0) * (nf - 2.0) * (nf - 3.0))
            - 3.0 * (nf - 1.0) * (nf - 1.0) / ((nf - 2.0) * (nf - 3.0))
    }

    /// Min.
    pub fn min(&self) -> f64 {
        self.samples.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// Max.
    pub fn max(&self) -> f64 {
        self.samples
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    fn ensure_sorted(&mut self) {
        if !self.sorted {
            self.samples.sort_by(|a, b| a.total_cmp(b));
            self.sorted = true;
        }
    }

    /// Percentile (0..100).
    pub fn percentile(&mut self, p: f64) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        self.ensure_sorted();
        let n = self.samples.len();
        let idx = (p / 100.0) * (n - 1) as f64;
        let lo = idx.floor() as usize;
        let hi = idx.ceil() as usize;
        let frac = idx - lo as f64;
        if lo >= n {
            return self.samples[n - 1];
        }
        if hi >= n {
            return self.samples[n - 1];
        }
        self.samples[lo] * (1.0 - frac) + self.samples[hi] * frac
    }

    /// Median (50th percentile).
    pub fn median(&mut self) -> f64 {
        self.percentile(50.0)
    }
}

impl Default for GeneralStatistics {
    fn default() -> Self {
        Self::new()
    }
}

/// Incremental (online/streaming) statistics — Welford's algorithm.
///
/// Computes mean, variance, skewness, kurtosis in a single pass.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IncrementalStatistics {
    n: usize,
    mean: f64,
    m2: f64,
    m3: f64,
    m4: f64,
    min_val: f64,
    max_val: f64,
}

impl IncrementalStatistics {
    /// New.
    pub fn new() -> Self {
        Self {
            n: 0,
            mean: 0.0,
            m2: 0.0,
            m3: 0.0,
            m4: 0.0,
            min_val: f64::INFINITY,
            max_val: f64::NEG_INFINITY,
        }
    }

    /// Add.
    pub fn add(&mut self, x: f64) {
        let n1 = self.n as f64;
        self.n += 1;
        let n = self.n as f64;

        let delta = x - self.mean;
        let delta_n = delta / n;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * n1;

        self.mean += delta_n;
        self.m4 += term1 * delta_n2 * (n * n - 3.0 * n + 3.0)
            + 6.0 * delta_n2 * self.m2
            - 4.0 * delta_n * self.m3;
        self.m3 += term1 * delta_n * (n - 2.0) - 3.0 * delta_n * self.m2;
        self.m2 += term1;

        if x < self.min_val {
            self.min_val = x;
        }
        if x > self.max_val {
            self.max_val = x;
        }
    }

    /// Reset.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Count.
    pub fn count(&self) -> usize {
        self.n
    }

    /// Mean.
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Variance.
    pub fn variance(&self) -> f64 {
        if self.n < 2 {
            return 0.0;
        }
        self.m2 / (self.n - 1) as f64
    }

    /// Standard deviation.
    pub fn standard_deviation(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Skewness.
    pub fn skewness(&self) -> f64 {
        if self.n < 3 {
            return 0.0;
        }
        let n = self.n as f64;
        (n.sqrt() * self.m3) / self.m2.powf(1.5)
    }

    /// Kurtosis.
    pub fn kurtosis(&self) -> f64 {
        if self.n < 4 {
            return 0.0;
        }
        let n = self.n as f64;
        (n * self.m4) / (self.m2 * self.m2) - 3.0
    }

    /// Min.
    pub fn min(&self) -> f64 {
        self.min_val
    }

    /// Max.
    pub fn max(&self) -> f64 {
        self.max_val
    }
}

impl Default for IncrementalStatistics {
    fn default() -> Self {
        Self::new()
    }
}

/// Risk statistics — VaR, CVaR (Expected Shortfall), shortfall probability.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RiskStatistics {
    inner: GeneralStatistics,
}

impl RiskStatistics {
    /// New.
    pub fn new() -> Self {
        Self {
            inner: GeneralStatistics::new(),
        }
    }

    /// Add.
    pub fn add(&mut self, value: f64) {
        self.inner.add(value);
    }

    /// Reset.
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// Count.
    pub fn count(&self) -> usize {
        self.inner.count()
    }

    /// Mean.
    pub fn mean(&self) -> f64 {
        self.inner.mean()
    }

    /// Variance.
    pub fn variance(&self) -> f64 {
        self.inner.variance()
    }

    /// Standard deviation.
    pub fn standard_deviation(&self) -> f64 {
        self.inner.standard_deviation()
    }

    /// Value at Risk at given confidence level (e.g. 0.95 for 95%).
    /// Returns the quantile: losses exceeding this with probability (1 - level).
    pub fn value_at_risk(&mut self, level: f64) -> f64 {
        // VaR is negative of the (1-level) percentile for loss distribution
        let pctile = (1.0 - level) * 100.0;
        self.inner.percentile(pctile)
    }

    /// Conditional VaR (Expected Shortfall) at given confidence level.
    /// Average of losses exceeding VaR.
    pub fn expected_shortfall(&mut self, level: f64) -> f64 {
        let var = self.value_at_risk(level);
        let tail: Vec<f64> = self
            .inner
            .samples
            .iter()
            .cloned()
            .filter(|&x| x <= var)
            .collect();
        if tail.is_empty() {
            return var;
        }
        tail.iter().sum::<f64>() / tail.len() as f64
    }

    /// Shortfall probability: P(X < target).
    pub fn shortfall_probability(&self, target: f64) -> f64 {
        if self.inner.samples.is_empty() {
            return 0.0;
        }
        let count = self.inner.samples.iter().filter(|&&x| x < target).count();
        count as f64 / self.inner.samples.len() as f64
    }

    /// Average shortfall: E[target - X | X < target].
    pub fn average_shortfall(&self, target: f64) -> f64 {
        let shortfalls: Vec<f64> = self
            .inner
            .samples
            .iter()
            .filter(|&&x| x < target)
            .map(|&x| target - x)
            .collect();
        if shortfalls.is_empty() {
            return 0.0;
        }
        shortfalls.iter().sum::<f64>() / shortfalls.len() as f64
    }
}

impl Default for RiskStatistics {
    fn default() -> Self {
        Self::new()
    }
}

/// Convergence tracker — records running mean/stderr for MC convergence checks.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConvergenceStatistics {
    /// Sample sizes.
    pub sample_sizes: Vec<usize>,
    /// Means.
    pub means: Vec<f64>,
    /// Std errors.
    pub std_errors: Vec<f64>,
    inner: IncrementalStatistics,
    next_checkpoint: usize,
}

impl ConvergenceStatistics {
    /// New.
    pub fn new() -> Self {
        Self {
            sample_sizes: Vec::new(),
            means: Vec::new(),
            std_errors: Vec::new(),
            inner: IncrementalStatistics::new(),
            next_checkpoint: 1,
        }
    }

    /// Add.
    pub fn add(&mut self, value: f64) {
        self.inner.add(value);
        if self.inner.count() >= self.next_checkpoint {
            self.sample_sizes.push(self.inner.count());
            self.means.push(self.inner.mean());
            if self.inner.count() > 1 {
                self.std_errors
                    .push(self.inner.standard_deviation() / (self.inner.count() as f64).sqrt());
            } else {
                self.std_errors.push(f64::INFINITY);
            }
            self.next_checkpoint *= 2;
        }
    }

    /// Count.
    pub fn count(&self) -> usize {
        self.inner.count()
    }

    /// Mean.
    pub fn mean(&self) -> f64 {
        self.inner.mean()
    }

    /// Standard error.
    pub fn standard_error(&self) -> f64 {
        if self.inner.count() > 1 {
            self.inner.standard_deviation() / (self.inner.count() as f64).sqrt()
        } else {
            f64::INFINITY
        }
    }
}

impl Default for ConvergenceStatistics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn general_stats_basic() {
        let mut s = GeneralStatistics::new();
        for i in 1..=10 {
            s.add(i as f64);
        }
        assert_eq!(s.count(), 10);
        assert_abs_diff_eq!(s.mean(), 5.5, epsilon = 1e-10);
        assert_abs_diff_eq!(s.variance(), 9.166_666_666_666, epsilon = 1e-6);
        assert_abs_diff_eq!(s.min(), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(s.max(), 10.0, epsilon = 1e-10);
    }

    #[test]
    fn general_stats_skewness_symmetric() {
        let mut s = GeneralStatistics::new();
        // Symmetric distribution: skewness ≈ 0
        for &x in &[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0] {
            s.add(x);
        }
        assert_abs_diff_eq!(s.skewness(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn general_stats_percentile() {
        let mut s = GeneralStatistics::new();
        for i in 1..=100 {
            s.add(i as f64);
        }
        assert_abs_diff_eq!(s.percentile(50.0), 50.5, epsilon = 0.6);
        assert_abs_diff_eq!(s.percentile(0.0), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(s.percentile(100.0), 100.0, epsilon = 1e-10);
    }

    #[test]
    fn incremental_stats_matches_general() {
        let data: Vec<f64> = (1..=20).map(|i| i as f64 * 0.7 - 3.0).collect();

        let mut gs = GeneralStatistics::new();
        let mut is = IncrementalStatistics::new();
        for &x in &data {
            gs.add(x);
            is.add(x);
        }

        assert_abs_diff_eq!(gs.mean(), is.mean(), epsilon = 1e-10);
        assert_abs_diff_eq!(gs.variance(), is.variance(), epsilon = 1e-8);
        assert_abs_diff_eq!(gs.standard_deviation(), is.standard_deviation(), epsilon = 1e-8);
    }

    #[test]
    fn risk_stats_var() {
        let mut rs = RiskStatistics::new();
        // Simulate 1000 returns
        for i in 0..1000 {
            let x = (i as f64 / 1000.0 - 0.5) * 2.0; // uniform[-1, 1)
            rs.add(x);
        }
        let var95 = rs.value_at_risk(0.95);
        // 5th percentile of uniform[-1,1) ≈ -0.9
        assert!(var95 < -0.5, "VaR95 should be significantly negative: {var95}");
    }

    #[test]
    fn risk_stats_expected_shortfall() {
        let mut rs = RiskStatistics::new();
        for i in 0..1000 {
            let x = (i as f64 / 1000.0 - 0.5) * 2.0;
            rs.add(x);
        }
        let es95 = rs.expected_shortfall(0.95);
        let var95 = rs.value_at_risk(0.95);
        // ES should be more extreme than VaR
        assert!(es95 <= var95, "ES {es95} should be <= VaR {var95}");
    }

    #[test]
    fn risk_stats_shortfall_probability() {
        let mut rs = RiskStatistics::new();
        for i in 0..1000 {
            rs.add(i as f64);
        }
        let prob = rs.shortfall_probability(500.0);
        assert_abs_diff_eq!(prob, 0.5, epsilon = 0.01);
    }

    #[test]
    fn convergence_tracker() {
        let mut cs = ConvergenceStatistics::new();
        for i in 0..1000 {
            cs.add(i as f64);
        }
        assert!(cs.sample_sizes.len() >= 9, "Should have log2(1000)≈10 checkpoints");
        // Final mean (including all 1000 samples) should be 499.5
        assert_abs_diff_eq!(cs.mean(), 499.5, epsilon = 1e-10);
        // Last recorded checkpoint mean should also be close
        let last_mean = *cs.means.last().unwrap();
        let last_n = *cs.sample_sizes.last().unwrap();
        // Last checkpoint is at 512, mean of 0..511 = 255.5
        assert_abs_diff_eq!(last_mean, (last_n - 1) as f64 / 2.0, epsilon = 1e-10);
    }

    #[test]
    fn incremental_stats_kurtosis() {
        // Normal-like data — excess kurtosis near 0
        let mut is = IncrementalStatistics::new();
        // Use known kurtosis: uniform distribution has excess kurtosis = -6/5 = -1.2
        for i in 0..10000 {
            let x = i as f64 / 10000.0;
            is.add(x);
        }
        let k = is.kurtosis();
        assert_abs_diff_eq!(k, -1.2, epsilon = 0.1);
    }
}

// ===========================================================================
// Sequence Statistics (multi-dimensional path statistics)
// ===========================================================================

/// Multi-dimensional running statistics for Monte-Carlo path analysis.
///
/// Equivalent to QuantLib's `SequenceStatistics<IncrementalStatistics>`.
/// Maintains one [`IncrementalStatistics`] per dimension, plus cross-moment
/// (covariance/correlation) estimates via an incremental algorithm.
#[derive(Clone, Debug)]
pub struct SequenceStatistics {
    dim: usize,
    /// Per-dimension accumulators.
    stats: Vec<IncrementalStatistics>,
    /// Upper-triangular cross-sum: cov_accum[i][j] (i ≤ j) stores
    /// sum of (x_i − μ_i)(x_j − μ_j) so far.
    cross: Vec<Vec<f64>>,
    n: usize,
}

impl SequenceStatistics {
    /// Create a new `SequenceStatistics` for `dim`-dimensional samples.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            stats: vec![IncrementalStatistics::new(); dim],
            cross: vec![vec![0.0; dim]; dim],
            n: 0,
        }
    }

    /// Add a single `dim`-dimensional sample.
    ///
    /// # Panics
    /// Panics if `sample.len() != self.dim`.
    pub fn add(&mut self, sample: &[f64]) {
        assert_eq!(sample.len(), self.dim, "sample dimension mismatch");
        self.n += 1;
        let n = self.n as f64;
        // Compute delta from old mean (before this sample is incorporated)
        let old_means: Vec<f64> = self.stats.iter().map(|s| s.mean()).collect();
        // Update per-dimension stats
        #[allow(clippy::needless_range_loop)]
        for i in 0..self.dim {
            self.stats[i].add(sample[i]);
        }
        let new_means: Vec<f64> = self.stats.iter().map(|s| s.mean()).collect();
        // Welford online cross-covariance update:
        //   C[i][j] += (x_i - old_mean_i) * (x_j - new_mean_j)
        // This gives the unscaled sum used for covariance: cov = C / (n-1)
        if self.n >= 2 {
            for i in 0..self.dim {
                for j in i..self.dim {
                    let delta_i = sample[i] - old_means[i];
                    let delta_j = sample[j] - new_means[j];
                    self.cross[i][j] += delta_i * delta_j;
                }
            }
        }
        let _ = n;
    }

    /// Number of samples added.
    pub fn count(&self) -> usize {
        self.n
    }

    /// Mean vector.
    pub fn mean(&self) -> Vec<f64> {
        self.stats.iter().map(|s| s.mean()).collect()
    }

    /// Variance vector (sample variance, n−1 denominator).
    pub fn variance(&self) -> Vec<f64> {
        self.stats.iter().map(|s| s.variance()).collect()
    }

    /// Standard deviation vector.
    pub fn std_dev(&self) -> Vec<f64> {
        self.stats.iter().map(|s| s.standard_deviation()).collect()
    }

    /// Sample covariance between dimensions `i` and `j`.
    ///
    /// Returns `None` if fewer than 2 samples.
    pub fn covariance(&self, i: usize, j: usize) -> Option<f64> {
        if self.n < 2 {
            return None;
        }
        let (ii, jj) = if i <= j { (i, j) } else { (j, i) };
        Some(self.cross[ii][jj] / (self.n - 1) as f64)
    }

    /// Pearson correlation between dimensions `i` and `j`.
    ///
    /// Returns `None` if fewer than 2 samples or either dimension has
    /// zero variance.
    pub fn correlation(&self, i: usize, j: usize) -> Option<f64> {
        let cov = self.covariance(i, j)?;
        let si = self.stats[i].standard_deviation();
        let sj = self.stats[j].standard_deviation();
        if si < 1e-14 || sj < 1e-14 {
            return None;
        }
        Some(cov / (si * sj))
    }

    /// Full covariance matrix (dim × dim).
    ///
    /// Returns `None` if fewer than 2 samples.
    pub fn covariance_matrix(&self) -> Option<Vec<Vec<f64>>> {
        if self.n < 2 {
            return None;
        }
        let mut mat = vec![vec![0.0f64; self.dim]; self.dim];
        #[allow(clippy::needless_range_loop)]
        for i in 0..self.dim {
            for j in 0..self.dim {
                mat[i][j] = self.covariance(i, j).unwrap_or(0.0);
            }
        }
        Some(mat)
    }

    /// Full correlation matrix (dim × dim).
    ///
    /// Returns `None` if fewer than 2 samples.
    pub fn correlation_matrix(&self) -> Option<Vec<Vec<f64>>> {
        if self.n < 2 {
            return None;
        }
        let mut mat = vec![vec![0.0f64; self.dim]; self.dim];
        #[allow(clippy::needless_range_loop)]
        for i in 0..self.dim {
            for j in 0..self.dim {
                mat[i][j] = self.correlation(i, j).unwrap_or(if i == j { 1.0 } else { 0.0 });
            }
        }
        Some(mat)
    }

    /// Minimum value per dimension.
    pub fn min(&self) -> Vec<f64> {
        self.stats.iter().map(|s| s.min()).collect()
    }

    /// Maximum value per dimension.
    pub fn max(&self) -> Vec<f64> {
        self.stats.iter().map(|s| s.max()).collect()
    }
}

#[cfg(test)]
mod seqstats_tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn seq_stats_mean_and_variance() {
        let mut ss = SequenceStatistics::new(2);
        let data = vec![
            vec![1.0, 4.0],
            vec![2.0, 5.0],
            vec![3.0, 6.0],
        ];
        for d in &data {
            ss.add(d);
        }
        let mean = ss.mean();
        assert_abs_diff_eq!(mean[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(mean[1], 5.0, epsilon = 1e-10);
        let var = ss.variance();
        assert_abs_diff_eq!(var[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(var[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn seq_stats_perfect_correlation() {
        let mut ss = SequenceStatistics::new(2);
        for i in 0..100 {
            let x = i as f64;
            ss.add(&[x, 2.0 * x]);
        }
        let r = ss.correlation(0, 1).unwrap();
        assert_abs_diff_eq!(r, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn seq_stats_no_samples() {
        let ss = SequenceStatistics::new(3);
        assert_eq!(ss.count(), 0);
        assert!(ss.covariance(0, 1).is_none());
    }

    #[test]
    #[should_panic]
    fn seq_stats_dimension_mismatch() {
        let mut ss = SequenceStatistics::new(2);
        ss.add(&[1.0, 2.0, 3.0]); // wrong dim
    }
}
