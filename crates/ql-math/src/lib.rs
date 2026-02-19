//! # ql-math
//!
//! Numerical library: interpolation, root-finding, optimization, distributions,
//! numerical integration, linear algebra utilities, copulas, statistics, FFT,
//! and quasi-random sequences.
//!
//! ## Overview
//!
//! | Module | Purpose |
//! |---|---|
//! | [`interpolation`] | Linear, log-linear, and cubic spline interpolation |
//! | [`interpolation_extended`] | Monotone convex, Hermite, Steffen, bilinear interpolation |
//! | [`solvers1d`] | Root-finding: Brent, Newton, bisection, secant, Ridder |
//! | [`optimization`] | Levenberg-Marquardt, simplex, BFGS, differential evolution |
//! | [`distributions`] | Normal, chi-squared, Poisson, Student-t CDFs and inverse CDFs |
//! | [`integration`] | Gauss-Legendre, adaptive Simpson, Gauss-Kronrod quadrature |
//! | [`matrix`] | Cholesky decomposition, eigenvalue, pseudo-sqrt utilities |
//! | [`copulas`] | Gaussian, Student-t, Clayton, Frank, Gumbel copulas |
//! | [`statistics`] | Running statistics, exponentially weighted, risk measures |
//! | [`fft`] | Fast Fourier transform (radix-2 Cooley-Tukey) |
//! | [`quasi_random`] | Sobol quasi-random sequences for Monte Carlo |
//!
//! ## Quick Start
//!
//! ```rust
//! use ql_math::interpolation::{LinearInterpolation, Interpolation};
//! use ql_math::solvers1d::{Brent, Solver1D};
//!
//! // Linear interpolation
//! let xs = vec![0.0, 1.0, 2.0];
//! let ys = vec![0.0, 1.0, 4.0];
//! let interp = LinearInterpolation::new(xs, ys).unwrap();
//! assert!((interp.value(0.5).unwrap() - 0.5).abs() < 1e-10);
//!
//! // Root finding: solve x^2 - 2 = 0
//! let root = Brent.solve(|x| x * x, 2.0, 1.5, 1.0, 2.0, 1e-12, 100).unwrap();
//! assert!((root - std::f64::consts::SQRT_2).abs() < 1e-10);
//! ```

pub mod copulas;
pub mod distributions;
pub mod fft;
pub mod integration;
pub mod interpolation;
pub mod interpolation_extended;
pub mod matrix;
pub mod optimization;
pub mod quasi_random;
pub mod solvers1d;
pub mod statistics;
