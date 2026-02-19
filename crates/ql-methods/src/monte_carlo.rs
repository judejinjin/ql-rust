//! Monte Carlo path generation.
//!
//! Generates sample paths from a `StochasticProcess1D` for use in
//! Monte Carlo pricing.

use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use ql_processes::StochasticProcess1D;

/// A single simulated path (time grid + spot values).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Path {
    /// Time points: `[0, dt, 2*dt, ..., T]`.
    pub times: Vec<f64>,
    /// Spot values at each time point: `values[0] = S₀`.
    pub values: Vec<f64>,
}

impl Path {
    /// Number of time steps (excluding the initial point).
    pub fn steps(&self) -> usize {
        self.times.len() - 1
    }
}

/// Generates 1-D sample paths using a `StochasticProcess1D`.
pub struct PathGenerator<P: StochasticProcess1D> {
    process: P,
    time_to_maturity: f64,
    num_steps: usize,
    rng: SmallRng,
}

impl<P: StochasticProcess1D> PathGenerator<P> {
    /// Create a new path generator.
    pub fn new(process: P, time_to_maturity: f64, num_steps: usize, seed: u64) -> Self {
        Self {
            process,
            time_to_maturity,
            num_steps,
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    /// Generate a single path.
    pub fn next_path(&mut self) -> Path {
        let dt = self.time_to_maturity / self.num_steps as f64;
        let mut times = Vec::with_capacity(self.num_steps + 1);
        let mut values = Vec::with_capacity(self.num_steps + 1);

        times.push(0.0);
        values.push(self.process.x0());

        let mut x = self.process.x0();
        for i in 0..self.num_steps {
            let t = i as f64 * dt;
            let dw: f64 = StandardNormal.sample(&mut self.rng);
            x = self.process.evolve_1d(t, x, dt, dw);
            times.push((i + 1) as f64 * dt);
            values.push(x);
        }

        Path { times, values }
    }

    /// Generate a path and its antithetic counterpart (negated shocks).
    pub fn next_antithetic_pair(&mut self) -> (Path, Path) {
        let dt = self.time_to_maturity / self.num_steps as f64;
        let mut times = Vec::with_capacity(self.num_steps + 1);
        let mut values1 = Vec::with_capacity(self.num_steps + 1);
        let mut values2 = Vec::with_capacity(self.num_steps + 1);

        times.push(0.0);
        values1.push(self.process.x0());
        values2.push(self.process.x0());

        let mut x1 = self.process.x0();
        let mut x2 = self.process.x0();

        for i in 0..self.num_steps {
            let t = i as f64 * dt;
            let dw: f64 = StandardNormal.sample(&mut self.rng);
            x1 = self.process.evolve_1d(t, x1, dt, dw);
            x2 = self.process.evolve_1d(t, x2, dt, -dw);
            times.push((i + 1) as f64 * dt);
            values1.push(x1);
            values2.push(x2);
        }

        let path1 = Path {
            times: times.clone(),
            values: values1,
        };
        let path2 = Path {
            times,
            values: values2,
        };
        (path1, path2)
    }
}

/// Multi-dimensional path for correlated processes.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MultiPath {
    /// Time points.
    pub times: Vec<f64>,
    /// Values for each factor at each time point.
    /// `values[factor][time_index]`.
    pub values: Vec<Vec<f64>>,
}

impl MultiPath {
    /// Number of time steps.
    pub fn steps(&self) -> usize {
        self.times.len() - 1
    }

    /// Number of factors (dimensions).
    pub fn factors(&self) -> usize {
        self.values.len()
    }
}

/// Type alias for multi-dimensional process coefficient functions.
type ProcessFn = Box<dyn Fn(f64, &[f64]) -> f64 + Send + Sync>;

/// Generates multi-dimensional correlated paths using Cholesky decomposition.
pub struct MultiPathGenerator {
    /// Initial values for each factor.
    initial_values: Vec<f64>,
    /// Drift function for each factor: drift(t, x, factor_index) -> f64.
    drifts: Vec<ProcessFn>,
    /// Diffusion function for each factor: diffusion(t, x, factor_index) -> f64.
    diffusions: Vec<ProcessFn>,
    /// Lower-triangular Cholesky factor of the correlation matrix.
    cholesky: Vec<Vec<f64>>,
    time_to_maturity: f64,
    num_steps: usize,
    rng: SmallRng,
}

impl MultiPathGenerator {
    /// Create a new multi-path generator.
    ///
    /// `correlation` is the NxN correlation matrix (must be positive definite).
    pub fn new(
        initial_values: Vec<f64>,
        drifts: Vec<ProcessFn>,
        diffusions: Vec<ProcessFn>,
        correlation: &[Vec<f64>],
        time_to_maturity: f64,
        num_steps: usize,
        seed: u64,
    ) -> Self {
        let n = initial_values.len();
        assert_eq!(drifts.len(), n);
        assert_eq!(diffusions.len(), n);
        assert_eq!(correlation.len(), n);

        // Cholesky decomposition
        let cholesky = cholesky_decompose(correlation);

        Self {
            initial_values,
            drifts,
            diffusions,
            cholesky,
            time_to_maturity,
            num_steps,
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    /// Generate a single multi-dimensional path.
    pub fn next_path(&mut self) -> MultiPath {
        let n = self.initial_values.len();
        let dt = self.time_to_maturity / self.num_steps as f64;
        let sqrt_dt = dt.sqrt();

        let mut times = Vec::with_capacity(self.num_steps + 1);
        let mut values = vec![Vec::with_capacity(self.num_steps + 1); n];

        times.push(0.0);
        let mut x = self.initial_values.clone();
        for (j, v) in values.iter_mut().enumerate() {
            v.push(x[j]);
        }

        for i in 0..self.num_steps {
            let t = i as f64 * dt;

            // Generate independent normals
            let z: Vec<f64> = (0..n)
                .map(|_| StandardNormal.sample(&mut self.rng))
                .collect();

            // Correlate: w = L·z
            let w: Vec<f64> = (0..n)
                .map(|j| {
                    (0..=j)
                        .map(|k| self.cholesky[j][k] * z[k])
                        .sum::<f64>()
                })
                .collect();

            // Euler step for each factor
            let x_prev = x.clone();
            for j in 0..n {
                let drift = (self.drifts[j])(t, &x_prev);
                let diff = (self.diffusions[j])(t, &x_prev);
                x[j] = x_prev[j] + drift * dt + diff * sqrt_dt * w[j];
            }

            times.push((i + 1) as f64 * dt);
            for (j, v) in values.iter_mut().enumerate() {
                v.push(x[j]);
            }
        }

        MultiPath { times, values }
    }
}

/// Cholesky decomposition of a symmetric positive-definite matrix.
///
/// Returns the lower-triangular factor L such that A = L·Lᵀ.
fn cholesky_decompose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for (li_k, lj_k) in l[i].iter().zip(l[j].iter()).take(j) {
                sum += li_k * lj_k;
            }
            if i == j {
                l[i][j] = (a[i][i] - sum).sqrt();
            } else {
                l[i][j] = (a[i][j] - sum) / l[j][j];
            }
        }
    }

    l
}

#[cfg(test)]
mod tests {
    use super::*;
    use ql_processes::black_scholes_process::GeneralizedBlackScholesProcess;
    use ql_termstructures::yield_curves::FlatForward;
    use ql_time::{Date, Month, DayCounter};
    use std::sync::Arc;

    fn make_gbm() -> GeneralizedBlackScholesProcess {
        let ref_date = Date::from_ymd(2025, Month::January, 1);
        let r = Arc::new(FlatForward::new(ref_date, 0.05, DayCounter::Actual365Fixed));
        let q = Arc::new(FlatForward::new(ref_date, 0.0, DayCounter::Actual365Fixed));
        GeneralizedBlackScholesProcess::new(100.0, r, q, 0.2)
    }

    #[test]
    fn path_generator_produces_correct_length() {
        let gbm = make_gbm();
        let mut gen = PathGenerator::new(gbm, 1.0, 252, 42);
        let path = gen.next_path();
        assert_eq!(path.values.len(), 253); // 252 steps + initial
        assert_eq!(path.steps(), 252);
    }

    #[test]
    fn path_starts_at_spot() {
        let gbm = make_gbm();
        let mut gen = PathGenerator::new(gbm, 1.0, 100, 42);
        let path = gen.next_path();
        assert!((path.values[0] - 100.0).abs() < 1e-12);
    }

    #[test]
    fn antithetic_paths_differ() {
        let gbm = make_gbm();
        let mut gen = PathGenerator::new(gbm, 1.0, 100, 42);
        let (p1, p2) = gen.next_antithetic_pair();
        // The two paths should generally differ
        let differ = p1
            .values
            .iter()
            .zip(p2.values.iter())
            .skip(1)
            .any(|(a, b)| (a - b).abs() > 1e-12);
        assert!(differ);
    }

    #[test]
    fn cholesky_identity() {
        let id = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let l = cholesky_decompose(&id);
        assert!((l[0][0] - 1.0).abs() < 1e-12);
        assert!((l[1][1] - 1.0).abs() < 1e-12);
        assert!((l[1][0]).abs() < 1e-12);
    }

    #[test]
    fn cholesky_correlated() {
        let rho = 0.5;
        let corr = vec![vec![1.0, rho], vec![rho, 1.0]];
        let l = cholesky_decompose(&corr);
        // L[0][0] = 1, L[1][0] = rho, L[1][1] = sqrt(1 - rho^2)
        assert!((l[0][0] - 1.0).abs() < 1e-12);
        assert!((l[1][0] - rho).abs() < 1e-12);
        assert!((l[1][1] - (1.0 - rho * rho).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn multi_path_generator_shape() {
        let n = 2;
        let initial = vec![100.0, 0.04];
        let drifts: Vec<Box<dyn Fn(f64, &[f64]) -> f64 + Send + Sync>> =
            vec![Box::new(|_, _| 0.05), Box::new(|_, _| 0.0)];
        let diffs: Vec<Box<dyn Fn(f64, &[f64]) -> f64 + Send + Sync>> =
            vec![Box::new(|_, x: &[f64]| 0.2 * x[0]), Box::new(|_, _| 0.3)];
        let corr = vec![vec![1.0, -0.5], vec![-0.5, 1.0]];
        let mut gen = MultiPathGenerator::new(initial, drifts, diffs, &corr, 1.0, 100, 42);
        let mpath = gen.next_path();
        assert_eq!(mpath.factors(), n);
        assert_eq!(mpath.steps(), 100);
        assert_eq!(mpath.values[0].len(), 101);
    }
}
