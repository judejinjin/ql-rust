//! Brownian bridge path construction.
//!
//! Constructs a discretized Brownian motion path using the bridge
//! technique, which fills in intermediate points given the endpoints
//! and uses conditional distributions.
//!
//! This is particularly useful for variance reduction in Monte Carlo
//! simulation and for generating paths with quasi-random numbers.
//!
//! Corresponds to QuantLib's `BrownianBridge`.

use serde::{Deserialize, Serialize};

/// Brownian bridge path constructor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrownianBridge {
    /// Number of time steps.
    pub steps: usize,
    /// Time increments (Δt_i).
    pub dt: Vec<f64>,
    /// Total time.
    pub total_time: f64,
    /// Bridge ordering: indices in the order they are filled.
    bridge_order: Vec<usize>,
    /// Left index for each bridge step.
    left_index: Vec<usize>,
    /// Right index for each bridge step.
    right_index: Vec<usize>,
    /// Left weight for each bridge step.
    left_weight: Vec<f64>,
    /// Right weight for each bridge step.
    right_weight: Vec<f64>,
    /// Standard deviation for each bridge step.
    stddev: Vec<f64>,
}

impl BrownianBridge {
    /// Create a Brownian bridge for uniform time steps.
    pub fn new(steps: usize, total_time: f64) -> Self {
        let dt_val = total_time / steps as f64;
        let dt = vec![dt_val; steps];
        Self::from_times(dt, total_time)
    }

    /// Create a Brownian bridge for non-uniform time steps.
    pub fn from_times(dt: Vec<f64>, total_time: f64) -> Self {
        let steps = dt.len();
        assert!(steps >= 1, "need at least 1 step");

        // Cumulative times
        let mut times = Vec::with_capacity(steps + 1);
        times.push(0.0);
        let mut t = 0.0;
        for &d in &dt {
            t += d;
            times.push(t);
        }

        // Build bridge ordering (divide and conquer)
        let mut bridge_order = Vec::with_capacity(steps);
        let mut left_index = Vec::with_capacity(steps);
        let mut right_index = Vec::with_capacity(steps);
        let mut left_weight = Vec::with_capacity(steps);
        let mut right_weight = Vec::with_capacity(steps);
        let mut stddev = Vec::with_capacity(steps);

        // First step: set the endpoint (index = steps)
        bridge_order.push(steps); // fill the last point first
        left_index.push(0);
        right_index.push(0); // boundary
        left_weight.push(0.0);
        right_weight.push(0.0);
        stddev.push(total_time.sqrt());

        // Build the bridge tree using a queue
        let mut queue: Vec<(usize, usize)> = vec![(0, steps)];

        while let Some((left, right)) = queue.pop() {
            if right - left <= 1 { continue; }
            let mid = (left + right) / 2;
            bridge_order.push(mid);

            let t_left = times[left];
            let t_right = times[right];
            let t_mid = times[mid];

            let span = t_right - t_left;
            if span > 1e-14 {
                let lw = (t_right - t_mid) / span;
                let rw = (t_mid - t_left) / span;
                let sd = ((t_mid - t_left) * (t_right - t_mid) / span).sqrt();
                left_weight.push(lw);
                right_weight.push(rw);
                stddev.push(sd);
            } else {
                left_weight.push(0.5);
                right_weight.push(0.5);
                stddev.push(0.0);
            }
            left_index.push(left);
            right_index.push(right);

            // Enqueue sub-intervals
            queue.push((left, mid));
            queue.push((mid, right));
        }

        Self {
            steps,
            dt,
            total_time,
            bridge_order,
            left_index,
            right_index,
            left_weight,
            right_weight,
            stddev,
        }
    }

    /// Transform a sequence of independent standard normal variates into
    /// a Brownian bridge path.
    ///
    /// `normals` must have length `steps`. Returns the path increments.
    pub fn transform(&self, normals: &[f64]) -> Vec<f64> {
        assert!(normals.len() >= self.steps, "need {} normals, got {}", self.steps, normals.len());

        // Path values at each time point (0 to steps)
        let mut path = vec![0.0_f64; self.steps + 1];

        for (k, &normal) in normals.iter().enumerate().take(self.bridge_order.len()) {
            let idx = self.bridge_order[k];
            if k == 0 {
                // Set endpoint
                path[idx] = self.stddev[k] * normal;
            } else {
                let li = self.left_index[k];
                let ri = self.right_index[k];
                path[idx] = self.left_weight[k] * path[li]
                    + self.right_weight[k] * path[ri]
                    + self.stddev[k] * normal;
            }
        }

        // Convert to increments
        let mut increments = Vec::with_capacity(self.steps);
        for i in 1..=self.steps {
            increments.push(path[i] - path[i - 1]);
        }
        increments
    }

    /// Generate a Brownian motion path (cumulative values) from normal variates.
    pub fn path(&self, normals: &[f64]) -> Vec<f64> {
        let increments = self.transform(normals);
        let mut path = Vec::with_capacity(self.steps + 1);
        path.push(0.0);
        let mut w = 0.0;
        for inc in &increments {
            w += inc;
            path.push(w);
        }
        path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brownian_bridge_endpoint() {
        let bb = BrownianBridge::new(10, 1.0);
        let normals = vec![1.0; 10];
        let path = bb.path(&normals);

        assert_eq!(path.len(), 11);
        assert!((path[0]).abs() < 1e-14, "path[0]={}", path[0]);
    }

    #[test]
    fn test_brownian_bridge_increments_sum() {
        let bb = BrownianBridge::new(8, 2.0);
        let normals = vec![0.5, -0.3, 1.2, -0.8, 0.1, 0.7, -0.4, 0.9];
        let increments = bb.transform(&normals);

        assert_eq!(increments.len(), 8);
        let sum: f64 = increments.iter().sum();
        let path = bb.path(&normals);
        let endpoint = path[8];
        assert!((sum - endpoint).abs() < 1e-10, "sum={}, endpoint={}", sum, endpoint);
    }

    #[test]
    fn test_brownian_bridge_zero_input() {
        let bb = BrownianBridge::new(5, 1.0);
        let normals = vec![0.0; 5];
        let path = bb.path(&normals);

        for &v in &path {
            assert!(v.abs() < 1e-14, "nonzero path value: {}", v);
        }
    }

    #[test]
    fn test_brownian_bridge_non_uniform() {
        let dt = vec![0.1, 0.3, 0.2, 0.4];
        let bb = BrownianBridge::from_times(dt, 1.0);
        let normals = vec![1.0, 0.5, -0.5, 0.0];
        let path = bb.path(&normals);

        assert_eq!(path.len(), 5);
        assert!((path[0]).abs() < 1e-14);
    }

    #[test]
    fn test_brownian_bridge_variance() {
        // Endpoint variance should be approximately T
        let bb = BrownianBridge::new(100, 1.0);
        let mut sum_sq = 0.0;
        let n_samples = 1000;
        let mut rng_state = 42u64;

        for _ in 0..n_samples {
            let normals: Vec<f64> = (0..100)
                .map(|_| {
                    rng_state ^= rng_state << 13;
                    rng_state ^= rng_state >> 7;
                    rng_state ^= rng_state << 17;
                    // Simple approx normal via central limit
                    let u: f64 = (rng_state >> 11) as f64 / (1u64 << 53) as f64;
                    // Inverse-ish normal, rough
                    (u - 0.5) * 3.46
                })
                .collect();
            let path = bb.path(&normals);
            sum_sq += path[100] * path[100];
        }
        let var = sum_sq / n_samples as f64;
        // Variance should be roughly T=1.0 (very rough due to bad RNG)
        assert!(var > 0.1 && var < 10.0, "var={}", var);
    }
}
