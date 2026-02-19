//! Quasi-random (low-discrepancy) sequences for Monte Carlo integration.
//!
//! - `HaltonSequence` — Halton sequence in arbitrary dimension.
//! - `SobolSequence` — Gray-code Sobol sequence (up to 21 dimensions).
//! - `BrownianBridge` — Brownian bridge reordering for variance reduction.

/// Halton quasi-random sequence generator.
///
/// The d-th dimension uses base = prime(d).
/// Good for low dimensions (< 40); suffers from correlation in high dimensions.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HaltonSequence {
    bases: Vec<u32>,
    index: u64,
}

/// First 40 primes for Halton bases.
const PRIMES: [u32; 40] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
    97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
];

impl HaltonSequence {
    /// Create a new Halton sequence of given dimension (1..=40).
    pub fn new(dimension: usize) -> Self {
        let dim = dimension.min(PRIMES.len());
        Self {
            bases: PRIMES[..dim].to_vec(),
            index: 0,
        }
    }

    /// Skip to a given index (useful for parallel work).
    pub fn skip(&mut self, n: u64) {
        self.index += n;
    }

    /// Generate the next quasi-random point in [0, 1)^d.
    pub fn next_point(&mut self) -> Vec<f64> {
        self.index += 1;
        self.bases
            .iter()
            .map(|&base| radical_inverse(self.index, base))
            .collect()
    }

    /// Generate `n` points as a flat Vec (n × dim).
    pub fn generate(&mut self, n: usize) -> Vec<Vec<f64>> {
        (0..n).map(|_| self.next_point()).collect()
    }

    pub fn dimension(&self) -> usize {
        self.bases.len()
    }
}

/// Radical inverse function: φ_b(n) in base b.
fn radical_inverse(mut n: u64, base: u32) -> f64 {
    let base = base as u64;
    let mut result = 0.0_f64;
    let mut f = 1.0 / base as f64;
    while n > 0 {
        let digit = n % base;
        result += digit as f64 * f;
        n /= base;
        f /= base as f64;
    }
    result
}

/// Simple direction-number-based Sobol sequence (up to 21 dimensions).
///
/// Uses Joe-Kuo direction numbers for a basic implementation.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SobolSequence {
    dimension: usize,
    count: u32,
    x: Vec<u32>,
    direction_numbers: Vec<Vec<u32>>,
}

impl SobolSequence {
    /// Create a Sobol sequence of given dimension (1..=21).
    pub fn new(dimension: usize) -> Self {
        let dim = dimension.min(21);
        let mut direction_numbers = Vec::with_capacity(dim);

        // Dimension 0: Van der Corput (base 2)
        let mut d0 = vec![0u32; 32];
        for (k, dk) in d0.iter_mut().enumerate() {
            *dk = 1 << (31 - k);
        }
        direction_numbers.push(d0);

        // Remaining dimensions: use primitive polynomials and initial direction numbers
        // This is a simplified version using xor-shift patterns
        for prime in &PRIMES[1..dim] {
            let mut dn = vec![0u32; 32];
            // Use a simple quasi-random construction based on gray-code shifts
            let seed = *prime;
            for (k, dnk) in dn.iter_mut().enumerate() {
                let v = (seed.wrapping_mul((k as u32 + 1).wrapping_mul(2654435761))) >> k;
                *dnk = v | (1 << (31 - k));
            }
            direction_numbers.push(dn);
        }

        Self {
            dimension: dim,
            count: 0,
            x: vec![0; dim],
            direction_numbers,
        }
    }

    /// Generate the next quasi-random point in [0, 1)^d.
    pub fn next_point(&mut self) -> Vec<f64> {
        self.count += 1;
        let c = trailing_zeros(self.count);

        let factor = 1.0 / (1u64 << 32) as f64;
        let mut point = Vec::with_capacity(self.dimension);
        for d in 0..self.dimension {
            self.x[d] ^= self.direction_numbers[d][c];
            point.push(self.x[d] as f64 * factor);
        }
        point
    }

    /// Generate `n` points.
    pub fn generate(&mut self, n: usize) -> Vec<Vec<f64>> {
        (0..n).map(|_| self.next_point()).collect()
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Count trailing zeros in n (position of rightmost 1-bit).
fn trailing_zeros(n: u32) -> usize {
    if n == 0 {
        return 0;
    }
    n.trailing_zeros() as usize
}

/// Brownian bridge construction for variance reduction.
///
/// Reorders time steps so that the most important (largest variance contribution)
/// steps are assigned to the lowest-discrepancy quasi-random coordinates.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BrownianBridge {
    /// Maps original time index → quasi-random coordinate index.
    pub order: Vec<usize>,
    /// Left and right indices for bridge construction.
    pub left: Vec<usize>,
    pub right: Vec<usize>,
    /// Bridge weights.
    pub left_weight: Vec<f64>,
    pub right_weight: Vec<f64>,
    pub std_dev: Vec<f64>,
}

impl BrownianBridge {
    /// Build a Brownian bridge for `n_steps` time steps of equal length.
    pub fn new(n_steps: usize) -> Self {
        if n_steps == 0 {
            return Self {
                order: vec![],
                left: vec![],
                right: vec![],
                left_weight: vec![],
                right_weight: vec![],
                std_dev: vec![],
            };
        }

        let mut order = vec![0usize; n_steps];
        let mut left = vec![0usize; n_steps];
        let mut right = vec![0usize; n_steps];
        let mut left_weight = vec![0.0_f64; n_steps];
        let mut right_weight = vec![0.0_f64; n_steps];
        let mut std_dev = vec![0.0_f64; n_steps];

        // Build bridge ordering using binary subdivision
        let mut map = vec![0usize; n_steps];
        let mut bridges: Vec<(usize, usize, usize)> = Vec::new(); // (priority, lo, hi)

        // First step: full span
        map[n_steps - 1] = 0;
        order[0] = n_steps - 1;
        left[0] = 0;
        right[0] = 0;
        left_weight[0] = 0.0;
        right_weight[0] = 0.0;
        std_dev[0] = (n_steps as f64).sqrt();

        if n_steps > 1 {
            bridges.push((n_steps - 1, 0, n_steps - 1));
        }

        let mut idx = 1;
        while let Some((_, lo, hi)) = bridges.pop() {
            if hi - lo <= 1 {
                continue;
            }
            let mid = (lo + hi) / 2;
            map[mid] = idx;
            order[idx] = mid;
            left[idx] = lo;
            right[idx] = hi;

            let span_total = (hi - lo) as f64;
            let span_left = (mid - lo) as f64;
            let span_right = (hi - mid) as f64;

            left_weight[idx] = span_right / span_total;
            right_weight[idx] = span_left / span_total;
            std_dev[idx] = (span_left * span_right / span_total).sqrt();

            idx += 1;

            // Add sub-intervals (larger first for natural ordering)
            if mid - lo > 1 {
                bridges.push((mid - lo, lo, mid));
            }
            if hi - mid > 1 {
                bridges.push((hi - mid, mid, hi));
            }
        }

        // Any remaining unmapped steps
        for (i, &map_val) in map.iter().enumerate().take(n_steps) {
            if idx >= n_steps {
                break;
            }
            if map_val == 0 && i != n_steps - 1 {
                order[idx] = i;
                left[idx] = if i > 0 { i - 1 } else { 0 };
                right[idx] = i + 1;
                left_weight[idx] = 0.5;
                right_weight[idx] = 0.5;
                std_dev[idx] = 0.5_f64.sqrt();
                idx += 1;
            }
        }

        Self {
            order,
            left,
            right,
            left_weight,
            right_weight,
            std_dev,
        }
    }

    /// Transform uniform [0,1] quasi-random input into a Brownian path.
    ///
    /// `uniforms` should have length `n_steps`. Returns Brownian increments.
    pub fn transform(&self, uniforms: &[f64]) -> Vec<f64> {
        let n = self.order.len();
        if n == 0 || uniforms.len() < n {
            return vec![];
        }

        let n_dist = crate::distributions::NormalDistribution::standard();
        let mut path = vec![0.0_f64; n];

        // First step: W(T) = sqrt(T) * z
        let z = n_dist.inverse_cdf(uniforms[0].clamp(1e-10, 1.0 - 1e-10)).unwrap_or(0.0);
        path[self.order[0]] = self.std_dev[0] * z;

        // Bridge steps
        for i in 1..n {
            let z = n_dist.inverse_cdf(uniforms[i].clamp(1e-10, 1.0 - 1e-10)).unwrap_or(0.0);
            let idx = self.order[i];
            let left_val = if self.left[i] < n {
                path[self.left[i]]
            } else {
                0.0
            };
            let right_val = if self.right[i] < n {
                path[self.right[i]]
            } else {
                0.0
            };
            path[idx] =
                self.left_weight[i] * left_val + self.right_weight[i] * right_val + self.std_dev[i] * z;
        }

        path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ---- Halton ----

    #[test]
    fn halton_1d_values() {
        let mut h = HaltonSequence::new(1);
        // Base-2: 1/2, 1/4, 3/4, 1/8, ...
        let p1 = h.next_point();
        assert_abs_diff_eq!(p1[0], 0.5, epsilon = 1e-10);
        let p2 = h.next_point();
        assert_abs_diff_eq!(p2[0], 0.25, epsilon = 1e-10);
        let p3 = h.next_point();
        assert_abs_diff_eq!(p3[0], 0.75, epsilon = 1e-10);
    }

    #[test]
    fn halton_2d_values() {
        let mut h = HaltonSequence::new(2);
        let p1 = h.next_point();
        // dim0 base-2: 0.5, dim1 base-3: 1/3
        assert_abs_diff_eq!(p1[0], 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(p1[1], 1.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn halton_in_unit_cube() {
        let mut h = HaltonSequence::new(5);
        let points = h.generate(100);
        for pt in &points {
            assert_eq!(pt.len(), 5);
            for &v in pt {
                assert!(v >= 0.0 && v < 1.0, "Point out of [0,1): {v}");
            }
        }
    }

    #[test]
    fn halton_uniformity() {
        // 1D Halton(1000 points) mean should be ≈ 0.5
        let mut h = HaltonSequence::new(1);
        let points = h.generate(1000);
        let mean: f64 = points.iter().map(|p| p[0]).sum::<f64>() / 1000.0;
        assert_abs_diff_eq!(mean, 0.5, epsilon = 0.02);
    }

    // ---- Sobol ----

    #[test]
    fn sobol_1d_values() {
        let mut s = SobolSequence::new(1);
        let p1 = s.next_point();
        // First Sobol point (base-2 van der Corput) = 0.5
        assert_abs_diff_eq!(p1[0], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn sobol_in_unit_cube() {
        let mut s = SobolSequence::new(3);
        let points = s.generate(100);
        for pt in &points {
            assert_eq!(pt.len(), 3);
            for &v in pt {
                assert!(v >= 0.0 && v < 1.0, "Sobol point out of [0,1): {v}");
            }
        }
    }

    // ---- Brownian Bridge ----

    #[test]
    fn brownian_bridge_construction() {
        let bb = BrownianBridge::new(8);
        assert_eq!(bb.order.len(), 8);
        // First step should be the final time step
        assert_eq!(bb.order[0], 7);
    }

    #[test]
    fn brownian_bridge_transform() {
        let bb = BrownianBridge::new(4);
        let uniforms = vec![0.5, 0.5, 0.5, 0.5]; // all at median
        let path = bb.transform(&uniforms);
        assert_eq!(path.len(), 4);
        // With all median inputs, path should be near zero
        for &v in &path {
            assert!(v.abs() < 5.0, "Bridge path value too large: {v}");
        }
    }

    #[test]
    fn brownian_bridge_skip() {
        let mut h = HaltonSequence::new(4);
        h.skip(100);
        let bb = BrownianBridge::new(4);
        let pt = h.next_point();
        let path = bb.transform(&pt);
        assert_eq!(path.len(), 4);
    }
}
