//! Halton quasi-random sequence generator.
//!
//! Generates low-discrepancy sequences for quasi-Monte Carlo methods.
//! The Halton sequence uses different prime bases for each dimension,
//! providing good uniform coverage of the unit hypercube.
//!
//! Corresponds to QuantLib's `HaltonRsg`.

use serde::{Deserialize, Serialize};

/// First 50 primes for Halton sequence bases.
const PRIMES: [u32; 50] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
];

/// Halton quasi-random sequence generator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HaltonSequence {
    /// Dimensionality.
    pub dimension: usize,
    /// Current sequence index.
    pub index: usize,
}

impl HaltonSequence {
    /// Create a new Halton sequence generator with given dimensionality.
    ///
    /// Maximum dimension is 50 (limited by available primes).
    pub fn new(dimension: usize) -> Self {
        assert!(dimension >= 1 && dimension <= PRIMES.len(),
            "dimension must be 1..={}", PRIMES.len());
        Self { dimension, index: 0 }
    }

    /// Create with a starting index (skip first `start` points).
    pub fn with_offset(dimension: usize, start: usize) -> Self {
        assert!(dimension >= 1 && dimension <= PRIMES.len());
        Self { dimension, index: start }
    }

    /// Generate the next point in the sequence.
    pub fn next_point(&mut self) -> Vec<f64> {
        self.index += 1;
        let point: Vec<f64> = (0..self.dimension)
            .map(|d| halton_element(self.index, PRIMES[d]))
            .collect();
        point
    }

    /// Generate a batch of n points.
    pub fn sample(&mut self, n: usize) -> Vec<Vec<f64>> {
        (0..n).map(|_| self.next_point()).collect()
    }

    /// Get the i-th element of dimension d (0-based index, 1-based count).
    pub fn element(i: usize, d: usize) -> f64 {
        assert!(d < PRIMES.len());
        halton_element(i + 1, PRIMES[d])
    }

    /// Reset the sequence to start from the beginning.
    pub fn reset(&mut self) {
        self.index = 0;
    }

    /// Current sequence index.
    pub fn current_index(&self) -> usize {
        self.index
    }
}

/// Compute the i-th element of the van der Corput sequence in base b.
///
/// ψ_b(i) = Σ_k d_k b^{-(k+1)} where i = Σ_k d_k b^k is the base-b
/// representation.
fn halton_element(mut i: usize, base: u32) -> f64 {
    let mut result = 0.0;
    let mut f = 1.0 / base as f64;
    while i > 0 {
        result += (i % base as usize) as f64 * f;
        i /= base as usize;
        f /= base as f64;
    }
    result
}

/// Faure sequence generator — a permuted variant of Halton for better uniformity
/// in higher dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaureSequence {
    /// Internal Halton generator.
    halton: HaltonSequence,
    /// Permutation tables for each dimension.
    permutations: Vec<Vec<usize>>,
}

impl FaureSequence {
    /// Create a Faure sequence generator.
    pub fn new(dimension: usize) -> Self {
        assert!(dimension >= 1 && dimension <= PRIMES.len());
        let permutations = (0..dimension)
            .map(|d| {
                let base = PRIMES[d] as usize;
                // Braaten-Weller permutation
                let mut perm: Vec<usize> = (0..base).collect();
                if base > 2 {
                    // Simple reversal-based permutation for improvement
                    for i in 1..base / 2 {
                        perm.swap(i, base - i);
                    }
                }
                perm
            })
            .collect();
        Self {
            halton: HaltonSequence::new(dimension),
            permutations,
        }
    }

    /// Generate the next point.
    pub fn next_point(&mut self) -> Vec<f64> {
        self.halton.index += 1;
        let i = self.halton.index;
        (0..self.halton.dimension)
            .map(|d| permuted_halton_element(i, PRIMES[d], &self.permutations[d]))
            .collect()
    }

    /// Generate a batch of n points.
    pub fn sample(&mut self, n: usize) -> Vec<Vec<f64>> {
        (0..n).map(|_| self.next_point()).collect()
    }

    /// Reset to beginning.
    pub fn reset(&mut self) {
        self.halton.reset();
    }
}

/// Permuted van der Corput sequence element.
fn permuted_halton_element(mut i: usize, base: u32, perm: &[usize]) -> f64 {
    let mut result = 0.0;
    let mut f = 1.0 / base as f64;
    while i > 0 {
        let digit = i % base as usize;
        let permuted = perm[digit];
        result += permuted as f64 * f;
        i /= base as usize;
        f /= base as f64;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_halton_base2() {
        // Van der Corput in base 2: 1/2, 1/4, 3/4, 1/8, 5/8, 3/8, 7/8, ...
        let expected = [0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875];
        for (i, &e) in expected.iter().enumerate() {
            let val = halton_element(i + 1, 2);
            assert!((val - e).abs() < 1e-14, "i={}, val={}, expected={}", i, val, e);
        }
    }

    #[test]
    fn test_halton_2d() {
        let mut seq = HaltonSequence::new(2);
        let points = seq.sample(100);
        assert_eq!(points.len(), 100);
        // All points in [0, 1)²
        for p in &points {
            assert!(p[0] >= 0.0 && p[0] < 1.0, "x={}", p[0]);
            assert!(p[1] >= 0.0 && p[1] < 1.0, "y={}", p[1]);
        }
    }

    #[test]
    fn test_halton_uniformity() {
        // Check that Halton points are roughly uniform
        let mut seq = HaltonSequence::new(1);
        let points = seq.sample(1000);
        let in_lower_half = points.iter().filter(|p| p[0] < 0.5).count();
        // Should be roughly 500
        assert!(in_lower_half > 400 && in_lower_half < 600,
            "in_lower_half={}", in_lower_half);
    }

    #[test]
    fn test_halton_element_static() {
        let val = HaltonSequence::element(0, 0); // i=1 base 2
        assert!((val - 0.5).abs() < 1e-14);
    }

    #[test]
    fn test_faure_sequence() {
        let mut seq = FaureSequence::new(3);
        let points = seq.sample(50);
        assert_eq!(points.len(), 50);
        for p in &points {
            assert_eq!(p.len(), 3);
            for &x in p {
                assert!(x >= 0.0 && x < 1.0, "x={}", x);
            }
        }
    }

    #[test]
    fn test_halton_discrepancy() {
        // Star discrepancy test: Halton should have lower discrepancy than random
        let mut seq = HaltonSequence::new(1);
        let n = 256;
        let points = seq.sample(n);

        // Sort and check max deviation from uniform CDF
        let mut sorted: Vec<f64> = points.iter().map(|p| p[0]).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut max_disc = 0.0_f64;
        for (i, &x) in sorted.iter().enumerate() {
            let expected = (i as f64 + 0.5) / n as f64;
            max_disc = max_disc.max((x - expected).abs());
        }
        // Halton discrepancy for 256 points should be much better than random (~0.03)
        assert!(max_disc < 0.05, "discrepancy={}", max_disc);
    }
}
