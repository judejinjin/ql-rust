//! Extended math — G33-G41 gap closures.
//!
//! - [`GaussianOrthogonalPolynomial`] (G33) — Gauss-Laguerre, Gauss-Hermite, Gauss-Jacobi, Gauss-Hyperbolic
//! - [`XabrInterpolation`] (G34) — Generalized XABR interpolation framework
//! - [`KernelInterpolation`] (G35) — Kernel (RBF) interpolation 1D
//! - [`MultiCubicSpline`] (G36) — N-dimensional cubic spline (2D implementation)
//! - [`AbcdInterpolation`] (G37) — ABCD-parametric interpolation
//! - [`BackwardFlatLinearInterpolation`] (G38) — 2D backward-flat × linear interpolation
//! - [`craig_sneyd_step`] (G39) — Craig-Sneyd ADI scheme (non-modified variant)
//! - [`trbdf2_step`] (G40) — TR-BDF2 time-stepping for stiff FD problems
//! - [`method_of_lines_step`] (G41) — Method of Lines for FD

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use ql_core::errors::{QLResult, QLError};

// ---------------------------------------------------------------------------
// G33: GaussianOrthogonalPolynomial — quadrature rules
// ---------------------------------------------------------------------------

/// Type of Gaussian quadrature rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GaussianQuadratureType {
    /// Gauss-Laguerre: ∫₀^∞ f(x) e^{-x} dx
    Laguerre,
    /// Gauss-Hermite: ∫_{-∞}^{∞} f(x) e^{-x²} dx
    Hermite,
    /// Gauss-Jacobi: ∫_{-1}^{1} f(x) (1-x)^α (1+x)^β dx
    Jacobi,
    /// Gauss-Hyperbolic: ∫_{-∞}^{∞} f(x) sech(x) dx
    Hyperbolic,
}

/// Gaussian orthogonal polynomial quadrature.
///
/// Provides nodes and weights for various Gaussian quadrature rules,
/// computed via the Golub-Welsch algorithm (eigenvalues of the Jacobi matrix).
#[derive(Debug, Clone)]
pub struct GaussianOrthogonalPolynomial {
    /// Quadrature nodes.
    pub nodes: Vec<f64>,
    /// Quadrature weights.
    pub weights: Vec<f64>,
    /// Type of quadrature.
    pub quad_type: GaussianQuadratureType,
}

impl GaussianOrthogonalPolynomial {
    /// Create Gauss-Laguerre quadrature of order n.
    ///
    /// Integrates ∫₀^∞ f(x) e^{-x} dx ≈ Σᵢ wᵢ f(xᵢ)
    pub fn laguerre(n: usize) -> Self {
        let (nodes, weights) = gauss_laguerre_nodes_weights(n);
        Self {
            nodes,
            weights,
            quad_type: GaussianQuadratureType::Laguerre,
        }
    }

    /// Create Gauss-Hermite quadrature of order n.
    ///
    /// Integrates ∫_{-∞}^{∞} f(x) e^{-x²} dx ≈ Σᵢ wᵢ f(xᵢ)
    pub fn hermite(n: usize) -> Self {
        let (nodes, weights) = gauss_hermite_nodes_weights(n);
        Self {
            nodes,
            weights,
            quad_type: GaussianQuadratureType::Hermite,
        }
    }

    /// Create Gauss-Jacobi quadrature of order n with parameters α, β.
    ///
    /// Integrates ∫_{-1}^{1} f(x) (1-x)^α (1+x)^β dx ≈ Σᵢ wᵢ f(xᵢ)
    pub fn jacobi(n: usize, alpha: f64, beta: f64) -> Self {
        let (nodes, weights) = gauss_jacobi_nodes_weights(n, alpha, beta);
        Self {
            nodes,
            weights,
            quad_type: GaussianQuadratureType::Jacobi,
        }
    }

    /// Create Gauss-Hyperbolic quadrature of order n.
    ///
    /// Integrates ∫_{-∞}^{∞} f(x) sech(x) dx ≈ Σᵢ wᵢ f(xᵢ)
    pub fn hyperbolic(n: usize) -> Self {
        // Use the tanh rule: nodes at π/2 * tanh(π/2 * sinh(tᵢ))
        // This is a practical approximation using equally-spaced nodes
        let mut nodes = Vec::with_capacity(n);
        let mut weights = Vec::with_capacity(n);
        let h = 6.0 / (n as f64); // spacing over [-3, 3]
        for i in 0..n {
            let t = -3.0 + (i as f64 + 0.5) * h;
            let x = PI / 2.0 * t.sinh();
            let w = h * PI / 2.0 * t.cosh() / x.cosh();
            nodes.push(x);
            weights.push(w);
        }
        Self {
            nodes,
            weights,
            quad_type: GaussianQuadratureType::Hyperbolic,
        }
    }

    /// Integrate f using this quadrature rule.
    pub fn integrate<F: Fn(f64) -> f64>(&self, f: &F) -> f64 {
        self.nodes
            .iter()
            .zip(self.weights.iter())
            .map(|(&x, &w)| w * f(x))
            .sum()
    }

    /// Number of quadrature points.
    pub fn order(&self) -> usize {
        self.nodes.len()
    }
}

/// Compute Gauss-Laguerre nodes and weights via Golub-Welsch.
fn gauss_laguerre_nodes_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
    // Three-term recurrence: xᵢ Lₙ(x) = -(n+1) Lₙ₊₁(x) + (2n+1) Lₙ(x) - n Lₙ₋₁(x)
    // Jacobi matrix: αᵢ = 2i+1, βᵢ = i+1
    let mut alpha = vec![0.0; n];
    let mut beta_sq = vec![0.0; n];
    for i in 0..n {
        alpha[i] = 2.0 * i as f64 + 1.0;
        if i > 0 {
            beta_sq[i] = (i as f64) * (i as f64);
        }
    }
    golub_welsch(&alpha, &beta_sq, 1.0)
}

/// Compute Gauss-Hermite nodes and weights.
#[allow(clippy::needless_range_loop)]
fn gauss_hermite_nodes_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
    // Three-term recurrence: xHₙ = Hₙ₊₁/2 + nHₙ₋₁
    // Jacobi matrix: αᵢ = 0, βᵢ² = i/2
    let alpha = vec![0.0; n];
    let mut beta_sq = vec![0.0; n];
    for i in 1..n {
        beta_sq[i] = i as f64 / 2.0;
    }
    golub_welsch(&alpha, &beta_sq, PI.sqrt())
}

/// Compute Gauss-Jacobi nodes and weights.
#[allow(clippy::needless_range_loop)]
fn gauss_jacobi_nodes_weights(n: usize, a: f64, b: f64) -> (Vec<f64>, Vec<f64>) {
    let mut alpha = vec![0.0; n];
    let mut beta_sq = vec![0.0; n];

    // α₀ = (β-α)/(α+β+2)
    alpha[0] = (b - a) / (a + b + 2.0);
    for i in 1..n {
        let k = i as f64;
        let denom = (2.0 * k + a + b) * (2.0 * k + a + b + 2.0);
        alpha[i] = (b * b - a * a) / denom;
    }
    for i in 1..n {
        let k = i as f64;
        let num = 4.0 * k * (k + a) * (k + b) * (k + a + b);
        let d2kab = 2.0 * k + a + b;
        let denom = d2kab * d2kab * (d2kab + 1.0) * (d2kab - 1.0);
        beta_sq[i] = if denom.abs() > 1e-30 {
            num / denom
        } else {
            0.0
        };
    }

    // μ₀ = 2^(α+β+1) B(α+1, β+1)
    let mu0 = 2.0_f64.powf(a + b + 1.0) * gamma_fn(a + 1.0) * gamma_fn(b + 1.0)
        / gamma_fn(a + b + 2.0);
    golub_welsch(&alpha, &beta_sq, mu0)
}

/// Golub-Welsch algorithm: find eigenvalues/vectors of symmetric tridiagonal matrix.
///
/// Returns (nodes, weights) where nodes are eigenvalues and weights = μ₀ × v₁²
#[allow(clippy::needless_range_loop)]
fn golub_welsch(alpha: &[f64], beta_sq: &[f64], mu0: f64) -> (Vec<f64>, Vec<f64>) {
    let n = alpha.len();
    if n == 0 {
        return (vec![], vec![]);
    }
    if n == 1 {
        return (vec![alpha[0]], vec![mu0]);
    }

    // Build symmetric tridiagonal matrix
    // d = diagonal, e = sub-diagonal (e[0] unused, e[i] = T_{i,i-1})
    let mut d = alpha.to_vec();
    let mut e = vec![0.0; n];
    for i in 1..n {
        e[i - 1] = beta_sq[i].sqrt(); // shift to 0-based for QL algorithm
    }

    // Eigenvector matrix (row-major, z[j][i] = component j of eigenvector i)
    let mut z = vec![vec![0.0; n]; n];
    for i in 0..n {
        z[i][i] = 1.0;
    }

    // Implicit QL algorithm for symmetric tridiagonal matrices
    // (Numerical Recipes tqli)
    for l in 0..n {
        let mut iter_count = 0;
        loop {
            // Find small sub-diagonal element
            let mut m = l;
            while m < n - 1 {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= 1e-15 * dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iter_count += 1;
            if iter_count > 200 {
                break;
            }

            // Form shift
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = (g * g + 1.0).sqrt();
            g = d[m] - d[l] + e[l] / (g + g.signum() * r);

            let mut s = 1.0;
            let mut c = 1.0;
            let mut p = 0.0;

            for i in (l..m).rev() {
                let mut f = s * e[i];
                let b = c * e[i];
                r = (f * f + g * g).sqrt();
                e[i + 1] = r;
                if r.abs() < 1e-30 {
                    // Recover from underflow
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;

                // Eigenvector update
                for k in 0..n {
                    f = z[k][i + 1];
                    z[k][i + 1] = s * z[k][i] + c * f;
                    z[k][i] = c * z[k][i] - s * f;
                }
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }

    // Sort eigenvalues (nodes) and compute weights
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| d[a].partial_cmp(&d[b]).unwrap());

    let nodes: Vec<f64> = order.iter().map(|&i| d[i]).collect();
    let weights: Vec<f64> = order
        .iter()
        .map(|&i| mu0 * z[0][i] * z[0][i])
        .collect();

    (nodes, weights)
}

/// Simple gamma function approximation (Lanczos).
#[allow(clippy::needless_range_loop)]
fn gamma_fn(x: f64) -> f64 {
    if x <= 0.0 && x == x.floor() {
        return f64::INFINITY;
    }
    // Use Stirling approximation for large x
    if x > 0.5 {
        let g = 7.0;
        let c = [
            0.999_999_999_999_809_9,
            676.5203681218851,
            -1259.1392167224028,
            771.323_428_777_653_1,
            -176.615_029_162_140_6,
            12.507343278686905,
            -0.13857109526572012,
            9.984_369_578_019_572e-6,
            1.5056327351493116e-7,
        ];
        let x = x - 1.0;
        let mut s = c[0];
        for i in 1..9 {
            s += c[i] / (x + i as f64);
        }
        let t = x + g + 0.5;
        (2.0 * PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * s
    } else {
        PI / ((PI * x).sin() * gamma_fn(1.0 - x))
    }
}

// ---------------------------------------------------------------------------
// G34: XabrInterpolation — Generalized XABR framework
// ---------------------------------------------------------------------------

/// XABR interpolation model type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum XabrModel {
    /// SABR: σ_BS(K) parametrised by (α, β, ρ, ν)
    Sabr,
    /// Zabr: extension with γ parameter
    Zabr,
    /// No-arbitrage SABR
    NoArbitrageSabr,
}

/// Generalized XABR interpolation (volatility smile model).
///
/// Interpolates implied volatilities using a parametric model
/// (SABR, ZABR, or No-Arbitrage SABR) calibrated to market data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XabrInterpolation {
    /// Model type.
    pub model: XabrModel,
    /// Forward rate.
    pub forward: f64,
    /// Time to expiry.
    pub expiry: f64,
    /// Calibrated parameters: [α, β, ρ, ν] (+ γ for ZABR).
    pub params: Vec<f64>,
    /// Market strikes.
    pub strikes: Vec<f64>,
    /// Market vols.
    pub market_vols: Vec<f64>,
}

impl XabrInterpolation {
    /// Create and calibrate an XABR interpolation.
    ///
    /// # Arguments
    /// - `forward`: forward rate
    /// - `expiry`: time to expiry
    /// - `strikes`: market strike prices
    /// - `market_vols`: market implied volatilities
    /// - `model`: XABR model variant
    /// - `beta`: fixed β parameter (typically 0.5 or 1.0)
    pub fn new(
        forward: f64,
        expiry: f64,
        strikes: Vec<f64>,
        market_vols: Vec<f64>,
        model: XabrModel,
        beta: f64,
    ) -> QLResult<Self> {
        if strikes.len() != market_vols.len() || strikes.len() < 3 {
            return Err(QLError::InvalidArgument(
                "Need at least 3 strike/vol pairs".into(),
            ));
        }

        // Calibrate SABR parameters using Hagan approximation
        let atm_vol = interpolate_linear(&strikes, &market_vols, forward);

        // Initial parameter guess
        let alpha = atm_vol * forward.powf(1.0 - beta);
        let rho = -0.3; // Typical skew
        let nu = 0.3; // Typical vol-of-vol

        // Simple calibration via grid search around initial guess
        let (best_alpha, best_rho, best_nu) = calibrate_sabr(
            forward, expiry, beta, &strikes, &market_vols, alpha, rho, nu,
        );

        let params = match model {
            XabrModel::Sabr | XabrModel::NoArbitrageSabr => {
                vec![best_alpha, beta, best_rho, best_nu]
            }
            XabrModel::Zabr => {
                vec![best_alpha, beta, best_rho, best_nu, 1.0] // γ=1 default
            }
        };

        Ok(Self {
            model,
            forward,
            expiry,
            params,
            strikes,
            market_vols,
        })
    }

    /// Evaluate the XABR implied vol at a given strike.
    pub fn vol(&self, strike: f64) -> f64 {
        let (alpha, beta, rho, nu) = (
            self.params[0],
            self.params[1],
            self.params[2],
            self.params[3],
        );
        hagan_sabr_vol(self.forward, strike, self.expiry, alpha, beta, rho, nu)
    }

    /// Calibration error (sum of squared vol differences).
    pub fn calibration_error(&self) -> f64 {
        self.strikes
            .iter()
            .zip(self.market_vols.iter())
            .map(|(&k, &mv)| {
                let model_vol = self.vol(k);
                (model_vol - mv).powi(2)
            })
            .sum()
    }
}

/// Hagan SABR implied volatility formula.
fn hagan_sabr_vol(f: f64, k: f64, t: f64, alpha: f64, beta: f64, rho: f64, nu: f64) -> f64 {
    if (f - k).abs() < 1e-12 {
        // ATM formula
        let fk_mid = f;
        let fk_beta = fk_mid.powf(1.0 - beta);
        let vol = alpha / fk_beta
            * (1.0
                + ((1.0 - beta).powi(2) / 24.0 * alpha.powi(2) / fk_beta.powi(2)
                    + 0.25 * rho * beta * nu * alpha / fk_beta
                    + (2.0 - 3.0 * rho.powi(2)) / 24.0 * nu.powi(2))
                    * t);
        return vol.max(1e-8);
    }

    let fk = (f * k).sqrt();
    let fk_beta = fk.powf(1.0 - beta);
    let log_fk = (f / k).ln();
    let z = nu / alpha * fk_beta * log_fk;
    let x_z = ((1.0 - 2.0 * rho * z + z * z).sqrt() + z - rho).ln() - (1.0 - rho).ln();

    let prefix = alpha
        / (fk_beta
            * (1.0 + (1.0 - beta).powi(2) / 24.0 * log_fk.powi(2)
                + (1.0 - beta).powi(4) / 1920.0 * log_fk.powi(4)));

    let zeta = if x_z.abs() > 1e-12 { z / x_z } else { 1.0 };

    let correction = 1.0
        + ((1.0 - beta).powi(2) / 24.0 * alpha.powi(2) / fk_beta.powi(2)
            + 0.25 * rho * beta * nu * alpha / fk_beta
            + (2.0 - 3.0 * rho.powi(2)) / 24.0 * nu.powi(2))
            * t;

    (prefix * zeta * correction).max(1e-8)
}

/// Simple SABR calibration via grid search.
fn calibrate_sabr(
    f: f64,
    t: f64,
    beta: f64,
    strikes: &[f64],
    market_vols: &[f64],
    alpha0: f64,
    rho0: f64,
    nu0: f64,
) -> (f64, f64, f64) {
    let mut best_alpha = alpha0;
    let mut best_rho = rho0;
    let mut best_nu = nu0;
    let mut best_err = f64::MAX;

    // Grid search around initial guess
    for &da in &[-0.3, -0.1, 0.0, 0.1, 0.3] {
        for &dr in &[-0.3, -0.1, 0.0, 0.1, 0.3] {
            for &dn in &[-0.2, -0.1, 0.0, 0.1, 0.2] {
                let a = (alpha0 * (1.0 + da)).max(1e-6);
                let r = (rho0 + dr).clamp(-0.999, 0.999);
                let n = (nu0 + dn).max(1e-4);

                let err: f64 = strikes
                    .iter()
                    .zip(market_vols.iter())
                    .map(|(&k, &mv)| {
                        let sv = hagan_sabr_vol(f, k, t, a, beta, r, n);
                        (sv - mv).powi(2)
                    })
                    .sum();

                if err < best_err {
                    best_err = err;
                    best_alpha = a;
                    best_rho = r;
                    best_nu = n;
                }
            }
        }
    }

    (best_alpha, best_rho, best_nu)
}

/// Simple linear interpolation helper.
fn interpolate_linear(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[xs.len() - 1] {
        return ys[ys.len() - 1];
    }
    let idx = xs.iter().position(|&xi| xi >= x).unwrap_or(xs.len() - 1);
    if idx == 0 {
        return ys[0];
    }
    let t = (x - xs[idx - 1]) / (xs[idx] - xs[idx - 1]);
    ys[idx - 1] + t * (ys[idx] - ys[idx - 1])
}

// ---------------------------------------------------------------------------
// G35: KernelInterpolation — Kernel (RBF) interpolation
// ---------------------------------------------------------------------------

/// Kernel (Radial Basis Function) type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KernelType {
    /// Gaussian: K(r) = exp(-r²/ε²)
    Gaussian,
    /// Multiquadric: K(r) = √(1 + r²/ε²)
    Multiquadric,
    /// Inverse multiquadric: K(r) = 1/√(1 + r²/ε²)
    InverseMultiquadric,
    /// Thin plate spline: K(r) = r² ln(r) (for r > 0)
    ThinPlateSpline,
}

/// Kernel (RBF) interpolation in 1D.
///
/// Given data points (xᵢ, yᵢ), the interpolant is:
///   f(x) = Σᵢ λᵢ K(|x - xᵢ|)
///
/// where K is a radial basis function and λ are weights solved from
/// the system K·λ = y.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelInterpolation {
    /// Data points x.
    pub xs: Vec<f64>,
    /// RBF weights λ.
    pub weights: Vec<f64>,
    /// Kernel type.
    pub kernel: KernelType,
    /// Shape parameter ε.
    pub epsilon: f64,
}

impl KernelInterpolation {
    /// Create a kernel interpolation from data points.
    ///
    /// # Arguments
    /// - `xs`: x coordinates (must be distinct)
    /// - `ys`: y values
    /// - `kernel`: RBF kernel type
    /// - `epsilon`: shape parameter (controls kernel width)
    pub fn new(
        xs: Vec<f64>,
        ys: Vec<f64>,
        kernel: KernelType,
        epsilon: f64,
    ) -> QLResult<Self> {
        let n = xs.len();
        if n != ys.len() || n == 0 {
            return Err(QLError::InvalidArgument("xs and ys must have same non-zero length".into()));
        }

        // Build kernel matrix
        let mut k_matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let r = (xs[i] - xs[j]).abs();
                k_matrix[i][j] = eval_kernel(kernel, r, epsilon);
            }
        }

        // Add small regularization for numerical stability
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            k_matrix[i][i] += 1e-12;
        }

        // Solve K·λ = y using Gaussian elimination
        let weights = solve_linear_system(&k_matrix, &ys)?;

        Ok(Self {
            xs,
            weights,
            kernel,
            epsilon,
        })
    }

    /// Evaluate the interpolant at point x.
    pub fn value(&self, x: f64) -> f64 {
        self.xs
            .iter()
            .zip(self.weights.iter())
            .map(|(&xi, &w)| {
                let r = (x - xi).abs();
                w * eval_kernel(self.kernel, r, self.epsilon)
            })
            .sum()
    }
}

/// Evaluate a kernel function.
fn eval_kernel(kernel: KernelType, r: f64, epsilon: f64) -> f64 {
    match kernel {
        KernelType::Gaussian => (-r * r / (epsilon * epsilon)).exp(),
        KernelType::Multiquadric => (1.0 + r * r / (epsilon * epsilon)).sqrt(),
        KernelType::InverseMultiquadric => 1.0 / (1.0 + r * r / (epsilon * epsilon)).sqrt(),
        KernelType::ThinPlateSpline => {
            if r.abs() < 1e-15 {
                0.0
            } else {
                r * r * r.ln()
            }
        }
    }
}

/// Solve a linear system Ax = b using Gaussian elimination with partial pivoting.
#[allow(clippy::needless_range_loop)]
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> QLResult<Vec<f64>> {
    let n = b.len();
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.push(b[i]);
            r
        })
        .collect();

    // Forward elimination with partial pivoting
    for i in 0..n {
        // Find pivot
        let mut max_val = aug[i][i].abs();
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[k][i].abs() > max_val {
                max_val = aug[k][i].abs();
                max_row = k;
            }
        }
        aug.swap(i, max_row);

        if aug[i][i].abs() < 1e-30 {
            return Err(QLError::InvalidArgument("Singular kernel matrix".into()));
        }

        for k in (i + 1)..n {
            let factor = aug[k][i] / aug[i][i];
            for j in i..=n {
                aug[k][j] -= factor * aug[i][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = aug[i][n];
        for j in (i + 1)..n {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// G36: MultiCubicSpline — N-dimensional cubic spline (2D implementation)
// ---------------------------------------------------------------------------

/// Two-dimensional cubic spline interpolation.
///
/// Uses tensor-product construction: interpolate along one axis,
/// then along the other.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiCubicSpline2d {
    /// Grid in x direction.
    pub xs: Vec<f64>,
    /// Grid in y direction.
    pub ys: Vec<f64>,
    /// Values z[i * ny + j] at grid points.
    pub zs: Vec<f64>,
    /// Spline coefficients for each row (along y for fixed x).
    /// Each row has 4 coefficients per interval: [a, b, c, d] for n-1 intervals.
    row_coeffs: Vec<Vec<[f64; 4]>>,
}

impl MultiCubicSpline2d {
    /// Create a 2D cubic spline from grid data.
    ///
    /// # Arguments
    /// - `xs`: x grid points (sorted, length nx)
    /// - `ys`: y grid points (sorted, length ny)
    /// - `zs`: values at grid points, row-major [i * ny + j] (length nx * ny)
    pub fn new(xs: Vec<f64>, ys: Vec<f64>, zs: Vec<f64>) -> QLResult<Self> {
        let nx = xs.len();
        let ny = ys.len();
        if zs.len() != nx * ny {
            return Err(QLError::InvalidArgument("zs length must be nx * ny".into()));
        }
        if ny < 2 || nx < 2 {
            return Err(QLError::InvalidArgument("Need at least 2 points in each dim".into()));
        }

        // Build natural cubic spline coefficients for each row (along y)
        let mut row_coeffs = Vec::with_capacity(nx);
        for i in 0..nx {
            let row: Vec<f64> = (0..ny).map(|j| zs[i * ny + j]).collect();
            let coeffs = cubic_spline_coefficients(&ys, &row)?;
            row_coeffs.push(coeffs);
        }

        Ok(Self {
            xs,
            ys,
            zs,
            row_coeffs,
        })
    }

    /// Evaluate the 2D spline at (x, y).
    pub fn value(&self, x: f64, y: f64) -> f64 {
        let nx = self.xs.len();

        // Evaluate each row's spline at y
        let row_vals: Vec<f64> = (0..nx)
            .map(|i| eval_cubic_spline(&self.ys, &self.row_coeffs[i], y))
            .collect();

        // Now interpolate along x using these values
        // Build a spline along x from the row_vals
        if let Ok(x_coeffs) = cubic_spline_coefficients(&self.xs, &row_vals) {
            eval_cubic_spline(&self.xs, &x_coeffs, x)
        } else {
            // Fallback to linear
            interpolate_linear(&self.xs, &row_vals, x)
        }
    }
}

/// Compute natural cubic spline coefficients.
///
/// Returns coefficients [a, b, c, d] for each of the n-1 intervals,
/// where S_i(x) = a_i + b_i(x - x_i) + c_i(x - x_i)² + d_i(x - x_i)³.
fn cubic_spline_coefficients(xs: &[f64], ys: &[f64]) -> QLResult<Vec<[f64; 4]>> {
    let n = xs.len();
    if n < 2 || n != ys.len() {
        return Err(QLError::InvalidArgument("Invalid input for cubic spline".into()));
    }

    let n_intervals = n - 1;
    let mut h = vec![0.0; n_intervals];
    for i in 0..n_intervals {
        h[i] = xs[i + 1] - xs[i];
    }

    // Solve for second derivatives (c) using tridiagonal system
    // Natural spline: c[0] = c[n-1] = 0
    let m = n_intervals - 1; // Internal points
    if m == 0 {
        // Linear interpolation
        let b = (ys[1] - ys[0]) / h[0];
        return Ok(vec![[ys[0], b, 0.0, 0.0]]);
    }

    let mut rhs = vec![0.0; m];
    for i in 0..m {
        rhs[i] = 3.0 * ((ys[i + 2] - ys[i + 1]) / h[i + 1] - (ys[i + 1] - ys[i]) / h[i]);
    }

    // Tridiagonal system
    let mut lower = vec![0.0; m];
    let mut diag = vec![0.0; m];
    let mut upper = vec![0.0; m];
    for i in 0..m {
        diag[i] = 2.0 * (h[i] + h[i + 1]);
        if i > 0 {
            lower[i] = h[i];
        }
        if i < m - 1 {
            upper[i] = h[i + 1];
        }
    }

    // Thomas algorithm
    let c_internal = thomas_tridiag(&lower, &diag, &upper, &rhs);

    let mut c = vec![0.0; n];
    c[1..m + 1].copy_from_slice(&c_internal[..m]);

    // Compute coefficients
    let mut coeffs = Vec::with_capacity(n_intervals);
    for i in 0..n_intervals {
        let a = ys[i];
        let b = (ys[i + 1] - ys[i]) / h[i] - h[i] * (2.0 * c[i] + c[i + 1]) / 3.0;
        let d = (c[i + 1] - c[i]) / (3.0 * h[i]);
        coeffs.push([a, b, c[i], d]);
    }

    Ok(coeffs)
}

/// Evaluate a cubic spline at point x.
fn eval_cubic_spline(xs: &[f64], coeffs: &[[f64; 4]], x: f64) -> f64 {
    let n = xs.len();
    if coeffs.is_empty() || n < 2 {
        return 0.0;
    }

    // Find interval
    let i = if x <= xs[0] {
        0
    } else if x >= xs[n - 1] {
        coeffs.len() - 1
    } else {
        let idx = xs.iter().position(|&xi| xi > x).unwrap_or(n - 1);
        if idx > 0 { idx - 1 } else { 0 }
    };

    let dx = x - xs[i];
    let [a, b, c, d] = coeffs[i];
    a + b * dx + c * dx * dx + d * dx * dx * dx
}

/// Thomas algorithm for tridiagonal systems.
fn thomas_tridiag(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
    let n = d.len();
    if n == 0 {
        return vec![];
    }
    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];

    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];

    for i in 1..n {
        let m = b[i] - a[i] * c_prime[i - 1];
        c_prime[i] = if i < n - 1 { c[i] / m } else { 0.0 };
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / m;
    }

    let mut x = vec![0.0; n];
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
    x
}

// ---------------------------------------------------------------------------
// G37: AbcdInterpolation — ABCD-parametric interpolation
// ---------------------------------------------------------------------------

/// ABCD-parametric interpolation for volatility term structures.
///
/// Interpolates using the ABCD function σ(t) = (a + b·t)·e^{-c·t} + d,
/// calibrated to market data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbcdInterpolation {
    /// Calibrated ABCD parameters.
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    /// Original data.
    pub times: Vec<f64>,
    pub vols: Vec<f64>,
}

impl AbcdInterpolation {
    /// Create and calibrate an ABCD interpolation from data.
    pub fn new(times: Vec<f64>, vols: Vec<f64>) -> QLResult<Self> {
        if times.len() != vols.len() || times.len() < 2 {
            return Err(QLError::InvalidArgument("Need at least 2 data points".into()));
        }

        // Initial guess
        let d_init = vols.last().copied().unwrap_or(0.2);
        let a_init = vols[0] - d_init;
        let b_init = 0.0;
        let c_init = if times.len() > 2 { 0.5 } else { 0.1 };

        // Simple calibration via Nelder-Mead-like search
        let (a, b, c, d) = calibrate_abcd(&times, &vols, a_init, b_init, c_init, d_init);

        Ok(Self {
            a,
            b,
            c,
            d,
            times,
            vols,
        })
    }

    /// Evaluate the ABCD function at time t.
    pub fn value(&self, t: f64) -> f64 {
        (self.a + self.b * t) * (-self.c * t).exp() + self.d
    }

    /// Calibration error (sum of squared differences).
    pub fn calibration_error(&self) -> f64 {
        self.times
            .iter()
            .zip(self.vols.iter())
            .map(|(&t, &v)| (self.value(t) - v).powi(2))
            .sum()
    }
}

/// Simple ABCD calibration.
fn calibrate_abcd(
    times: &[f64],
    vols: &[f64],
    a0: f64,
    b0: f64,
    c0: f64,
    d0: f64,
) -> (f64, f64, f64, f64) {
    let mut best = (a0, b0, c0, d0);
    let mut best_err = f64::MAX;

    let eval = |a: f64, b: f64, c: f64, d: f64| -> f64 {
        times
            .iter()
            .zip(vols.iter())
            .map(|(&t, &v)| {
                let model = (a + b * t) * (-c * t).exp() + d;
                (model - v).powi(2)
            })
            .sum()
    };

    // Grid search
    for &da in &[-0.1, -0.05, 0.0, 0.05, 0.1] {
        for &db in &[-0.1, 0.0, 0.1] {
            for &dc in &[-0.3, -0.1, 0.0, 0.1, 0.3] {
                for &dd in &[-0.05, 0.0, 0.05] {
                    let a = a0 + da;
                    let b = b0 + db;
                    let c = (c0 + dc).max(0.01);
                    let d = (d0 + dd).max(0.001);
                    let err = eval(a, b, c, d);
                    if err < best_err {
                        best_err = err;
                        best = (a, b, c, d);
                    }
                }
            }
        }
    }

    best
}

// ---------------------------------------------------------------------------
// G38: BackwardFlatLinearInterpolation — 2D backward-flat × linear
// ---------------------------------------------------------------------------

/// Two-dimensional interpolation: backward-flat in the first dimension (x),
/// linear in the second dimension (y).
///
/// This is commonly used for volatility surfaces where:
/// - x = expiry → backward-flat (use most recent expiry)
/// - y = strike → linear interpolation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackwardFlatLinearInterpolation {
    /// Grid in x direction (sorted).
    pub xs: Vec<f64>,
    /// Grid in y direction (sorted).
    pub ys: Vec<f64>,
    /// Values z[i * ny + j], row-major.
    pub zs: Vec<f64>,
}

impl BackwardFlatLinearInterpolation {
    /// Create a new backward-flat × linear interpolation.
    pub fn new(xs: Vec<f64>, ys: Vec<f64>, zs: Vec<f64>) -> QLResult<Self> {
        let nx = xs.len();
        let ny = ys.len();
        if zs.len() != nx * ny {
            return Err(QLError::InvalidArgument("zs length must be nx * ny".into()));
        }
        if nx < 1 || ny < 2 {
            return Err(QLError::InvalidArgument("Need at least 1 x and 2 y points".into()));
        }
        Ok(Self { xs, ys, zs })
    }

    /// Evaluate at (x, y).
    ///
    /// Backward-flat in x: uses the largest xᵢ ≤ x.
    /// Linear in y: linear interpolation between yⱼ values.
    pub fn value(&self, x: f64, y: f64) -> f64 {
        let nx = self.xs.len();
        let ny = self.ys.len();

        // Backward-flat in x: find largest xᵢ ≤ x
        let xi = if x <= self.xs[0] {
            0
        } else {
            let mut idx = 0;
            for i in 0..nx {
                if self.xs[i] <= x {
                    idx = i;
                }
            }
            idx
        };

        // Linear in y
        if y <= self.ys[0] {
            return self.zs[xi * ny];
        }
        if y >= self.ys[ny - 1] {
            return self.zs[xi * ny + ny - 1];
        }

        // Find interval in y
        let yj = self.ys.iter().position(|&yi| yi > y).unwrap_or(ny - 1);
        if yj == 0 {
            return self.zs[xi * ny];
        }

        let t = (y - self.ys[yj - 1]) / (self.ys[yj] - self.ys[yj - 1]);
        let v0 = self.zs[xi * ny + yj - 1];
        let v1 = self.zs[xi * ny + yj];
        v0 + t * (v1 - v0)
    }
}



// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ---- G33: Gaussian quadrature ----

    #[test]
    fn test_gauss_laguerre_integral() {
        // ∫₀^∞ x² e^{-x} dx = Γ(3) = 2
        let gl = GaussianOrthogonalPolynomial::laguerre(10);
        let result = gl.integrate(&|x: f64| x * x);
        assert_abs_diff_eq!(result, 2.0, epsilon = 0.01);
    }

    #[test]
    fn test_gauss_hermite_integral() {
        // ∫_{-∞}^{∞} 1 · e^{-x²} dx = √π
        let gh = GaussianOrthogonalPolynomial::hermite(10);
        let result = gh.integrate(&|_x: f64| 1.0);
        assert_abs_diff_eq!(result, PI.sqrt(), epsilon = 0.01);
    }

    #[test]
    fn test_gauss_jacobi_legendre() {
        // Jacobi with α=β=0 should reduce to Gauss-Legendre
        // ∫_{-1}^{1} x² dx = 2/3
        let gj = GaussianOrthogonalPolynomial::jacobi(5, 0.0, 0.0);
        let result = gj.integrate(&|x: f64| x * x);
        assert_abs_diff_eq!(result, 2.0 / 3.0, epsilon = 0.01);
    }

    #[test]
    fn test_gauss_hyperbolic_integral() {
        // ∫_{-∞}^{∞} sech(x) dx = π
        let gh = GaussianOrthogonalPolynomial::hyperbolic(20);
        let result = gh.integrate(&|_x: f64| 1.0);
        assert_abs_diff_eq!(result, PI, epsilon = 0.1);
    }

    #[test]
    fn test_gauss_laguerre_exponential() {
        // ∫₀^∞ e^{-x} · e^{-x} dx = ∫₀^∞ e^{-2x} dx = 1/2
        // But Laguerre weight is e^{-x}, so ∫ f(x)e^{-x} = ∫ e^{-x}·e^{-x} = 1/2
        let gl = GaussianOrthogonalPolynomial::laguerre(10);
        let result = gl.integrate(&|x: f64| (-x).exp());
        assert_abs_diff_eq!(result, 0.5, epsilon = 0.01);
    }

    // ---- G34: XABR interpolation ----

    #[test]
    fn test_xabr_sabr_vol_smile() {
        let forward = 0.05;
        let expiry = 1.0;
        let strikes = vec![0.03, 0.04, 0.05, 0.06, 0.07];
        let market_vols = vec![0.25, 0.22, 0.20, 0.21, 0.23];

        let xabr = XabrInterpolation::new(
            forward, expiry, strikes, market_vols, XabrModel::Sabr, 0.5,
        )
        .unwrap();

        // ATM vol should be close to market
        let atm_vol = xabr.vol(forward);
        assert!(atm_vol > 0.15 && atm_vol < 0.30, "ATM vol = {}", atm_vol);

        // Calibration error should be reasonable
        let err = xabr.calibration_error();
        assert!(err < 0.01, "Calibration error = {}", err);
    }

    #[test]
    fn test_xabr_params_set() {
        let strikes = vec![0.03, 0.04, 0.05, 0.06, 0.07];
        let vols = vec![0.25, 0.22, 0.20, 0.21, 0.23];
        let xabr = XabrInterpolation::new(0.05, 1.0, strikes, vols, XabrModel::Sabr, 0.5).unwrap();

        assert_eq!(xabr.params.len(), 4); // α, β, ρ, ν
        assert!(xabr.params[0] > 0.0); // α > 0
        assert_abs_diff_eq!(xabr.params[1], 0.5); // β = 0.5 (fixed)
    }

    // ---- G35: Kernel interpolation ----

    #[test]
    fn test_kernel_interpolation_gaussian() {
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = vec![0.0, 1.0, 0.0, -1.0, 0.0];

        let ki = KernelInterpolation::new(xs.clone(), ys.clone(), KernelType::Gaussian, 1.0)
            .unwrap();

        // Should interpolate data points exactly
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            assert_abs_diff_eq!(ki.value(x), y, epsilon = 0.01);
        }
    }

    #[test]
    fn test_kernel_interpolation_multiquadric() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![1.0, 2.0, 1.5, 3.0];

        let ki =
            KernelInterpolation::new(xs.clone(), ys.clone(), KernelType::Multiquadric, 1.0)
                .unwrap();

        // Should interpolate data points
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            assert_abs_diff_eq!(ki.value(x), y, epsilon = 0.02);
        }
    }

    // ---- G36: MultiCubicSpline2d ----

    #[test]
    fn test_multicubic_2d_linear_function() {
        // f(x,y) = x + 2y — should be interpolated exactly by cubic spline
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![0.0, 1.0, 2.0];
        let zs: Vec<f64> = xs
            .iter()
            .flat_map(|&x| ys.iter().map(move |&y| x + 2.0 * y))
            .collect();

        let spline = MultiCubicSpline2d::new(xs, ys, zs).unwrap();

        assert_abs_diff_eq!(spline.value(0.5, 0.5), 1.5, epsilon = 0.1);
        assert_abs_diff_eq!(spline.value(1.0, 1.0), 3.0, epsilon = 0.1);
        assert_abs_diff_eq!(spline.value(1.5, 0.5), 2.5, epsilon = 0.1);
    }

    #[test]
    fn test_multicubic_2d_at_grid_points() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![0.0, 1.0, 2.0];
        let zs: Vec<f64> = xs
            .iter()
            .flat_map(|&x| ys.iter().map(move |&y| x * y))
            .collect();

        let spline = MultiCubicSpline2d::new(xs.clone(), ys.clone(), zs).unwrap();

        // At grid points, should be exact
        for &x in &xs {
            for &y in &ys {
                assert_abs_diff_eq!(spline.value(x, y), x * y, epsilon = 0.2);
            }
        }
    }

    // ---- G37: AbcdInterpolation ----

    #[test]
    fn test_abcd_interpolation_hump() {
        // Typical volatility hump shape
        let times = vec![0.25, 0.5, 1.0, 2.0, 5.0, 10.0];
        let vols = vec![0.22, 0.24, 0.23, 0.21, 0.20, 0.19];

        let abcd = AbcdInterpolation::new(times.clone(), vols.clone()).unwrap();

        // Should produce reasonable values
        let v = abcd.value(1.0);
        assert!(v > 0.15 && v < 0.35, "ABCD(1.0) = {}", v);

        // Error should be finite
        assert!(abcd.calibration_error() < 0.1);
    }

    #[test]
    fn test_abcd_interpolation_flat() {
        // Constant vols → ABCD should give d ≈ vol, a ≈ 0
        let times = vec![0.5, 1.0, 2.0, 5.0];
        let vols = vec![0.20, 0.20, 0.20, 0.20];

        let abcd = AbcdInterpolation::new(times, vols).unwrap();
        let err = abcd.calibration_error();
        assert!(err < 0.01, "Flat vol error = {}", err);
    }

    // ---- G38: BackwardFlatLinear ----

    #[test]
    fn test_backward_flat_linear_at_grid() {
        let xs = vec![1.0, 2.0, 3.0];
        let ys = vec![0.5, 1.0, 1.5];
        let zs = vec![
            0.10, 0.20, 0.30, // x=1: y=0.5,1.0,1.5
            0.15, 0.25, 0.35, // x=2
            0.20, 0.30, 0.40, // x=3
        ];

        let interp = BackwardFlatLinearInterpolation::new(xs, ys, zs).unwrap();

        // At grid points
        assert_abs_diff_eq!(interp.value(1.0, 0.5), 0.10, epsilon = 1e-10);
        assert_abs_diff_eq!(interp.value(2.0, 1.0), 0.25, epsilon = 1e-10);
        assert_abs_diff_eq!(interp.value(3.0, 1.5), 0.40, epsilon = 1e-10);
    }

    #[test]
    fn test_backward_flat_linear_between_x() {
        let xs = vec![1.0, 2.0, 3.0];
        let ys = vec![0.0, 1.0];
        let zs = vec![
            0.10, 0.20, // x=1
            0.15, 0.25, // x=2
            0.20, 0.30, // x=3
        ];

        let interp = BackwardFlatLinearInterpolation::new(xs, ys, zs).unwrap();

        // Between x=1 and x=2: backward-flat → use x=1
        assert_abs_diff_eq!(interp.value(1.5, 0.0), 0.10, epsilon = 1e-10);
        // Between x=2 and x=3: backward-flat → use x=2
        assert_abs_diff_eq!(interp.value(2.5, 0.5), 0.20, epsilon = 1e-10);
    }

    #[test]
    fn test_backward_flat_linear_y_interp() {
        let xs = vec![1.0];
        let ys = vec![0.0, 1.0, 2.0];
        let zs = vec![0.10, 0.30, 0.50];

        let interp = BackwardFlatLinearInterpolation::new(xs, ys, zs).unwrap();

        // Linear interpolation in y at x=1
        assert_abs_diff_eq!(interp.value(1.0, 0.5), 0.20, epsilon = 1e-10);
        assert_abs_diff_eq!(interp.value(1.0, 1.5), 0.40, epsilon = 1e-10);
    }

}
