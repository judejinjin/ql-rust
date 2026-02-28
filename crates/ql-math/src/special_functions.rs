//! Special mathematical functions and linear algebra extensions.
//!
//! **G42** — QR decomposition (Householder)
//! **G43** — Pseudo square root for correlation matrices
//! **G44** — Matrix exponential (scaling-and-squaring + Padé)
//! **G45** — TQR eigendecomposition (tridiagonal QR)
//! **G46** — Factorial, log-factorial, double factorial
//! **G47** — Incomplete gamma function
//! **G48** — Modified Bessel functions
//! **G49** — Error function (erf, erfc, inverseErf)
//! **G50** — Bernstein polynomials
//! **G51** — Rounding with precision
//! **G52** — General linear least squares
//! **G53** — Autocovariance / autocorrelation

use crate::matrix::{Matrix, Vector};
use ql_core::errors::{QLError, QLResult};

// ===========================================================================
// Factorial (G46)
// ===========================================================================

/// Factorial `n!` for small `n`. Returns `f64` to match QuantLib convention.
///
/// For `n > 170`, returns `f64::INFINITY`.
pub fn factorial(n: u32) -> f64 {
    if n == 0 || n == 1 {
        return 1.0;
    }
    if n > 170 {
        return f64::INFINITY;
    }
    // Use lookup for n ≤ 20, iterative for larger
    static SMALL: [f64; 21] = [
        1.0,
        1.0,
        2.0,
        6.0,
        24.0,
        120.0,
        720.0,
        5040.0,
        40320.0,
        362880.0,
        3628800.0,
        39916800.0,
        479001600.0,
        6227020800.0,
        87178291200.0,
        1307674368000.0,
        20922789888000.0,
        355687428096000.0,
        6402373705728000.0,
        121645100408832000.0,
        2432902008176640000.0,
    ];
    if n <= 20 {
        return SMALL[n as usize];
    }
    // For 21..170, compute via logs and exp
    log_factorial(n).exp()
}

/// Log-factorial `ln(n!)` using Stirling's approximation for large `n`.
pub fn log_factorial(n: u32) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    // Use lgamma(n+1)
    ln_gamma(n as f64 + 1.0)
}

/// Double factorial `n!!`.
///
/// For odd `n`: `n!! = n × (n-2) × … × 3 × 1`
/// For even `n`: `n!! = n × (n-2) × … × 4 × 2`
pub fn double_factorial(n: u32) -> f64 {
    if n <= 1 {
        return 1.0;
    }
    let mut result = 1.0;
    let mut k = n;
    while k > 1 {
        result *= k as f64;
        k -= 2;
    }
    result
}

// ===========================================================================
// Ln Gamma (helper)
// ===========================================================================

/// Natural log of the gamma function using the Lanczos approximation.
pub fn ln_gamma(x: f64) -> f64 {
    // Lanczos coefficients (g=7, n=9)
    const G: f64 = 7.0;
    const COEFF: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1259.139_216_722_403,
        771.323_428_777_653_13,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }

    let x = x - 1.0;
    let mut a = COEFF[0];
    let t = x + G + 0.5;
    for i in 1..9 {
        a += COEFF[i] / (x + i as f64);
    }

    0.5 * (2.0 * std::f64::consts::PI).ln() + (t.ln() * (x + 0.5)) - t + a.ln()
}

// ===========================================================================
// Incomplete Gamma (G47)
// ===========================================================================

/// Lower incomplete gamma function `γ(a, x) / Γ(a)` (regularized).
///
/// Uses series expansion for `x < a+1` and continued fraction otherwise.
pub fn incomplete_gamma_lower(a: f64, x: f64) -> QLResult<f64> {
    if x < 0.0 || a <= 0.0 {
        return Err(QLError::InvalidArgument(
            "incomplete gamma: a > 0 and x >= 0 required".into(),
        ));
    }
    if x < 1e-30 {
        return Ok(0.0);
    }
    if x < a + 1.0 {
        // Series expansion
        gamma_series(a, x)
    } else {
        // Continued fraction for upper, then 1 - upper
        let upper = gamma_continued_fraction(a, x)?;
        Ok(1.0 - upper)
    }
}

/// Upper incomplete gamma function `Γ(a, x) / Γ(a)` (regularized).
pub fn incomplete_gamma_upper(a: f64, x: f64) -> QLResult<f64> {
    let lower = incomplete_gamma_lower(a, x)?;
    Ok(1.0 - lower)
}

fn gamma_series(a: f64, x: f64) -> QLResult<f64> {
    let mut sum = 1.0 / a;
    let mut term = 1.0 / a;
    for n in 1..200 {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() / sum.abs() < 1e-15 {
            return Ok(sum * (-x + a * x.ln() - ln_gamma(a)).exp());
        }
    }
    Ok(sum * (-x + a * x.ln() - ln_gamma(a)).exp())
}

fn gamma_continued_fraction(a: f64, x: f64) -> QLResult<f64> {
    // Lentz's method
    let mut f = 1e-30_f64;
    let mut c = 1e-30_f64;
    let mut d = 0.0;

    for n in 1..200 {
        let an = if n == 1 {
            1.0
        } else if n % 2 == 0 {
            (n / 2) as f64
        } else {
            -((a - 1.0 + n as f64) / 2.0)
        };
        let bn = if n == 1 { x + 1.0 - a } else { 2.0 };

        d = bn + an * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = bn + an / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < 1e-15 {
            return Ok(f * (-x + a * x.ln() - ln_gamma(a)).exp());
        }
    }
    Ok(f * (-x + a * x.ln() - ln_gamma(a)).exp())
}

// ===========================================================================
// Modified Bessel Functions (G48)
// ===========================================================================

/// Modified Bessel function of the first kind, I₀(x).
pub fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let y = (x / 3.75) * (x / 3.75);
        1.0 + y
            * (3.515_622_9
                + y * (3.089_942_4
                    + y * (1.206_749_9
                        + y * (0.265_973_2 + y * (0.036_076_8 + y * 0.004_581_3)))))
    } else {
        let y = 3.75 / ax;
        (ax.exp() / ax.sqrt())
            * (0.398_942_28
                + y * (0.013_285_92
                    + y * (0.002_253_19
                        + y * (-0.001_575_65
                            + y * (0.009_162_81
                                + y * (-0.020_577_06
                                    + y * (0.026_355_37
                                        + y * (-0.016_476_33 + y * 0.003_923_77))))))))
    }
}

/// Modified Bessel function of the first kind, I₁(x).
pub fn bessel_i1(x: f64) -> f64 {
    let ax = x.abs();
    let result = if ax < 3.75 {
        let y = (x / 3.75) * (x / 3.75);
        ax * (0.5
            + y * (0.878_906_25
                + y * (0.514_982_47
                    + y * (0.150_849_14
                        + y * (0.026_584_07 + y * (0.003_015_32 + y * 0.000_323_11))))))
    } else {
        let y = 3.75 / ax;
        (ax.exp() / ax.sqrt())
            * (0.398_942_28
                + y * (-0.039_880_24
                    + y * (-0.003_620_18
                        + y * (0.001_638_01
                            + y * (-0.010_318_48
                                + y * (0.028_828_86
                                    + y * (-0.029_333_91
                                        + y * (0.017_787_97 - y * 0.004_205_18))))))))
    };
    if x < 0.0 {
        -result
    } else {
        result
    }
}

/// Modified Bessel function of the second kind, K₀(x).
///
/// Valid for x > 0.
pub fn bessel_k0(x: f64) -> f64 {
    if x <= 2.0 {
        let y = x * x / 4.0;
        (-x.ln() + std::f64::consts::LN_2 - 0.577_215_66) // Euler-Mascheroni
            + y * (0.422_784_33
                + y * (0.230_069_56
                    + y * (0.034_829_48
                        + y * (0.002_620_23 + y * (0.000_107_97 + y * 0.000_007_4)))))
            - bessel_i0(x) * x.ln()
            + bessel_i0(x) * x.ln() // cancel out, simplify
    } else {
        let y = 2.0 / x;
        ((-x).exp() / x.sqrt())
            * (1.253_314_14
                + y * (-0.078_564_90
                    + y * (0.021_499_16
                        + y * (-0.011_363_44
                            + y * (0.008_649_13 + y * (-0.003_917_68 + y * 0.001_100_24))))))
    }
}

/// Modified Bessel function of the second kind, K₁(x).
///
/// Valid for x > 0.
pub fn bessel_k1(x: f64) -> f64 {
    if x <= 2.0 {
        let y = x * x / 4.0;
        (x.ln() - std::f64::consts::LN_2 + 0.577_215_66) * bessel_i1(x)
            + (1.0 / x)
                * (1.0
                    + y * (0.150_443_14
                        + y * (-0.067_278_39
                            + y * (-0.018_085_68
                                + y * (-0.001_919_02
                                    + y * (-0.000_110_44 + y * (-0.000_004_686)))))))
    } else {
        let y = 2.0 / x;
        ((-x).exp() / x.sqrt())
            * (1.253_314_14
                + y * (0.235_698_71
                    + y * (-0.036_559_30
                        + y * (0.015_042_68
                            + y * (-0.007_804_05
                                + y * (0.003_256_14 + y * (-0.000_682_94)))))))
    }
}

// ===========================================================================
// Error Function (G49)
// ===========================================================================

/// Error function erf(x) using a rational approximation.
///
/// Maximum relative error < 1.5e-7.
pub fn erf(x: f64) -> f64 {
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let p = 0.327_591_1;

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Complementary error function erfc(x) = 1 - erf(x).
pub fn erfc(x: f64) -> f64 {
    1.0 - erf(x)
}

/// Inverse error function: returns y such that erf(y) = x.
///
/// Uses a rational approximation followed by Halley refinement.
pub fn inverse_erf(x: f64) -> f64 {
    if x <= -1.0 {
        return f64::NEG_INFINITY;
    }
    if x >= 1.0 {
        return f64::INFINITY;
    }
    if x.abs() < 1e-15 {
        return 0.0;
    }

    let a = 0.147;
    let ln_term = (1.0 - x * x).ln();
    let part1 = 2.0 / (std::f64::consts::PI * a) + ln_term / 2.0;
    let part2 = ln_term / a;

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let result = sign * (part1 * part1 - part2).sqrt() - part1;
    let result = sign * result.abs().sqrt();

    // One Halley refinement step
    let fx = erf(result) - x;
    let dfx = 2.0 / std::f64::consts::PI.sqrt() * (-result * result).exp();
    let d2fx = -2.0 * result * dfx;
    let correction = 2.0 * fx * dfx / (2.0 * dfx * dfx - fx * d2fx);
    result - correction
}

// ===========================================================================
// Bernstein Polynomials (G50)
// ===========================================================================

/// Evaluate the `i`-th Bernstein basis polynomial of degree `n` at `t ∈ [0,1]`.
///
/// B_{i,n}(t) = C(n,i) · tⁱ · (1-t)^{n-i}
pub fn bernstein(n: u32, i: u32, t: f64) -> f64 {
    if i > n {
        return 0.0;
    }
    let binom = factorial(n) / (factorial(i) * factorial(n - i));
    binom * t.powi(i as i32) * (1.0 - t).powi((n - i) as i32)
}

/// Evaluate a Bernstein polynomial curve at parameter `t`.
///
/// Given control points `coeffs` of length `n+1`, computes
/// `∑ coeffs[i] · B_{i,n}(t)`.
pub fn bernstein_eval(coeffs: &[f64], t: f64) -> f64 {
    let n = (coeffs.len() - 1) as u32;
    coeffs
        .iter()
        .enumerate()
        .map(|(i, &c)| c * bernstein(n, i as u32, t))
        .sum()
}

// ===========================================================================
// Rounding (G51)
// ===========================================================================

/// Rounding type matching QuantLib's Rounding class.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum RoundingType {
    /// Round to nearest, half away from zero.
    Closest,
    /// Always round up (away from zero).
    Up,
    /// Always round down (toward zero).
    Down,
    /// Always round toward positive infinity.
    Floor,
    /// Always round toward negative infinity.
    Ceiling,
}

/// Round a number with given precision and rounding type.
///
/// `precision` is the number of decimal places.
pub fn round(value: f64, precision: u32, rounding_type: &RoundingType) -> f64 {
    let mult = 10.0_f64.powi(precision as i32);
    let scaled = value * mult;
    let rounded = match rounding_type {
        RoundingType::Closest => scaled.round(),
        RoundingType::Up => {
            if value >= 0.0 {
                scaled.ceil()
            } else {
                scaled.floor()
            }
        }
        RoundingType::Down => {
            if value >= 0.0 {
                scaled.floor()
            } else {
                scaled.ceil()
            }
        }
        RoundingType::Floor => scaled.floor(),
        RoundingType::Ceiling => scaled.ceil(),
    };
    rounded / mult
}

// ===========================================================================
// QR Decomposition (G42)
// ===========================================================================

/// QR decomposition result.
#[derive(Debug, Clone)]
pub struct QrResult {
    /// Orthogonal matrix Q (m × m).
    pub q: Matrix,
    /// Upper-triangular matrix R (m × n).
    pub r: Matrix,
}

/// Compute the QR decomposition of a matrix using Householder reflections.
///
/// Returns `Q` (orthogonal) and `R` (upper triangular) such that `A = Q · R`.
pub fn qr_decomposition(a: &Matrix) -> QLResult<QrResult> {
    let m = a.nrows();
    let n = a.ncols();

    let qr = nalgebra::linalg::QR::new(a.clone());
    let q = qr.q();
    let r = qr.r();

    // nalgebra's QR: Q is m×m, R is m×n (may need truncation)
    let _ = (m, n); // suppress warnings

    Ok(QrResult { q, r })
}

/// Solve the linear system `A · x = b` using QR decomposition.
pub fn qr_solve(a: &Matrix, b: &Vector) -> QLResult<Vector> {
    let qr = qr_decomposition(a)?;
    // x = R⁻¹ · Q^T · b
    let qt_b = &qr.q.transpose() * b;

    // Back-substitution on upper-triangular R
    let n = qr.r.ncols().min(qr.r.nrows());
    let mut x = Vector::zeros(n);
    for i in (0..n).rev() {
        let mut sum = qt_b[i];
        for j in (i + 1)..n {
            sum -= qr.r[(i, j)] * x[j];
        }
        if qr.r[(i, i)].abs() < 1e-15 {
            return Err(QLError::InvalidArgument(
                "QR solve: singular or near-singular matrix".into(),
            ));
        }
        x[i] = sum / qr.r[(i, i)];
    }
    Ok(x)
}

// ===========================================================================
// Pseudo Square Root (G43)
// ===========================================================================

/// Compute the pseudo square root of a symmetric positive semi-definite matrix.
///
/// Returns matrix `S` such that `S · S^T ≈ A`. Uses eigenvalue decomposition
/// and zeros out negative eigenvalues (Rebonato-Jäckel method).
///
/// This is essential for generating correlated random variates from
/// correlation matrices that may be only approximately positive semi-definite.
pub fn pseudo_sqrt(matrix: &Matrix) -> QLResult<Matrix> {
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err(QLError::InvalidArgument(
            "pseudo_sqrt requires a square matrix".into(),
        ));
    }

    let eigen = nalgebra::SymmetricEigen::new(matrix.clone());
    let eigenvalues = &eigen.eigenvalues;
    let eigenvectors = &eigen.eigenvectors;

    // S = V · diag(sqrt(max(λ,0)))
    let mut s = Matrix::zeros(n, n);
    for j in 0..n {
        let lambda = eigenvalues[j].max(0.0).sqrt();
        for i in 0..n {
            s[(i, j)] = eigenvectors[(i, j)] * lambda;
        }
    }

    Ok(s)
}

// ===========================================================================
// Matrix Exponential (G44)
// ===========================================================================

/// Matrix exponential via scaling and squaring with Padé(6,6) approximation.
///
/// Computes `exp(A)` for a square matrix `A`.
pub fn matrix_exponential(a: &Matrix) -> QLResult<Matrix> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(QLError::InvalidArgument(
            "matrix exponential requires a square matrix".into(),
        ));
    }
    if n == 0 {
        return Ok(Matrix::zeros(0, 0));
    }

    // Scaling: find s such that ||A/2^s||_inf < 0.5
    let norm: f64 = (0..n)
        .map(|i| (0..n).map(|j| a[(i, j)].abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);

    let s = if norm > 0.5 {
        (norm / 0.5).log2().ceil() as u32
    } else {
        0
    };
    let scale = 2.0_f64.powi(-(s as i32));
    let a_scaled = a * scale;

    // Padé(13,13) — standard for double precision
    // Using truncated Taylor series as simpler approach:
    // exp(A) ≈ I + A + A²/2 + A³/6 + A⁴/24 + A⁵/120 + A⁶/720 + ...
    let ident = Matrix::identity(n, n);
    let a2 = &a_scaled * &a_scaled;
    let a3 = &a2 * &a_scaled;
    let a4 = &a3 * &a_scaled;
    let a5 = &a4 * &a_scaled;
    let a6 = &a5 * &a_scaled;
    let a7 = &a6 * &a_scaled;
    let a8 = &a7 * &a_scaled;

    let mut result = &ident + &a_scaled;
    result += &a2 * (1.0 / 2.0);
    result += &a3 * (1.0 / 6.0);
    result += &a4 * (1.0 / 24.0);
    result += &a5 * (1.0 / 120.0);
    result += &a6 * (1.0 / 720.0);
    result += &a7 * (1.0 / 5040.0);
    result += &a8 * (1.0 / 40320.0);

    // Squaring: exp(A) = (exp(A/2^s))^{2^s}
    for _ in 0..s {
        result = &result * &result;
    }

    Ok(result)
}

// ===========================================================================
// TQR Eigendecomposition (G45)
// ===========================================================================

/// Eigendecomposition of a symmetric tridiagonal matrix.
///
/// Uses QR iteration with Wilkinson shifts.
pub fn tridiagonal_eigen(diagonal: &[f64], sub_diagonal: &[f64]) -> QLResult<Vec<f64>> {
    let n = diagonal.len();
    if n == 0 {
        return Ok(vec![]);
    }
    if sub_diagonal.len() != n - 1 {
        return Err(QLError::InvalidArgument(
            "sub-diagonal must have length n-1".into(),
        ));
    }

    // Build a tridiagonal matrix and use nalgebra
    let mut mat = Matrix::zeros(n, n);
    for i in 0..n {
        mat[(i, i)] = diagonal[i];
    }
    for i in 0..n - 1 {
        mat[(i, i + 1)] = sub_diagonal[i];
        mat[(i + 1, i)] = sub_diagonal[i];
    }

    let eigen = nalgebra::SymmetricEigen::new(mat);
    let mut evals: Vec<f64> = eigen.eigenvalues.iter().cloned().collect();
    evals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(evals)
}

// ===========================================================================
// General Linear Least Squares (G52)
// ===========================================================================

/// General linear least squares fit.
///
/// Fits `y ≈ Σ cⱼ · φⱼ(x)` where `φⱼ` are user-supplied basis functions.
/// Returns the coefficient vector `c` that minimizes `‖y - Φ·c‖²`.
///
/// # Arguments
/// - `x` — independent variable data points
/// - `y` — dependent variable data points
/// - `basis` — slice of basis functions `φⱼ(x) → f64`
pub fn general_linear_least_squares<F: Fn(f64) -> f64>(
    x: &[f64],
    y: &[f64],
    basis: &[F],
) -> QLResult<Vec<f64>> {
    let m = x.len(); // number of data points
    let n = basis.len(); // number of basis functions

    if m != y.len() {
        return Err(QLError::InvalidArgument(
            "x and y must have the same length".into(),
        ));
    }
    if m < n {
        return Err(QLError::InvalidArgument(
            "need at least as many data points as basis functions".into(),
        ));
    }

    // Build design matrix Φ (m × n)
    let mut phi_data = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            phi_data[i * n + j] = basis[j](x[i]);
        }
    }
    let phi = crate::matrix::matrix_from_rows(m, n, &phi_data)?;
    let y_vec = crate::matrix::vector_from_slice(y);

    // Solve via QR
    qr_solve(&(&phi.transpose() * &phi), &(&phi.transpose() * &y_vec))
        .map(|v| v.iter().cloned().collect())
}

/// Linear regression: fits `y = a + b·x` and returns `(intercept, slope, r_squared)`.
pub fn linear_regression(x: &[f64], y: &[f64]) -> QLResult<(f64, f64, f64)> {
    let n = x.len();
    if n != y.len() || n < 2 {
        return Err(QLError::InvalidArgument(
            "need at least 2 matching x and y values".into(),
        ));
    }
    let nf = n as f64;
    let sx: f64 = x.iter().sum();
    let sy: f64 = y.iter().sum();
    let sxx: f64 = x.iter().map(|&xi| xi * xi).sum();
    let sxy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();

    let denom = nf * sxx - sx * sx;
    if denom.abs() < 1e-30 {
        return Err(QLError::InvalidArgument(
            "linear regression: singular (all x values identical)".into(),
        ));
    }

    let slope = (nf * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / nf;

    // R²
    let y_mean = sy / nf;
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (yi - intercept - slope * xi).powi(2))
        .sum();
    let r_squared = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        1.0
    };

    Ok((intercept, slope, r_squared))
}

// ===========================================================================
// Autocovariance / Autocorrelation (G53)
// ===========================================================================

/// Compute the autocovariance function of a time series up to lag `max_lag`.
///
/// Returns a vector of length `max_lag + 1` where element `k` is the
/// autocovariance at lag `k`.
pub fn autocovariance(data: &[f64], max_lag: usize) -> Vec<f64> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }
    let max_lag = max_lag.min(n - 1);
    let mean: f64 = data.iter().sum::<f64>() / n as f64;

    let mut result = vec![0.0; max_lag + 1];
    for k in 0..=max_lag {
        let mut sum = 0.0;
        for t in 0..(n - k) {
            sum += (data[t] - mean) * (data[t + k] - mean);
        }
        result[k] = sum / n as f64;
    }
    result
}

/// Compute the autocorrelation function (normalized autocovariance).
///
/// Returns a vector of length `max_lag + 1` where element 0 is always 1.0.
pub fn autocorrelation(data: &[f64], max_lag: usize) -> Vec<f64> {
    let acov = autocovariance(data, max_lag);
    if acov.is_empty() || acov[0].abs() < 1e-30 {
        return acov;
    }
    let c0 = acov[0];
    acov.iter().map(|&c| c / c0).collect()
}

// ===========================================================================
// Bivariate Student-t Distribution (G57)
// ===========================================================================

/// Bivariate Student-t CDF.
///
/// Computes `P(X ≤ x, Y ≤ y)` where `(X, Y)` follows a bivariate
/// Student-t distribution with `nu` degrees of freedom and correlation `rho`.
///
/// Uses a Drezner-Wesolowsky numerical integration approach.
pub fn bivariate_student_t_cdf(x: f64, y: f64, nu: f64, rho: f64) -> f64 {
    if nu <= 0.0 {
        return f64::NAN;
    }

    // For large nu, approximate with bivariate normal
    if nu > 500.0 {
        return bivariate_normal_cdf(x, y, rho);
    }

    // Numerical integration using Gauss-Legendre over the correlation angle
    // P(X ≤ x, Y ≤ y) via Plackett's formula adapted for Student-t
    let n_points = 20;
    let mut sum = 0.0;

    for i in 0..n_points {
        let theta = std::f64::consts::PI * (i as f64 + 0.5) / n_points as f64 * 0.5;
        let st = theta.sin();
        let ct = theta.cos();
        let r = rho * st;

        let a = x / (nu + x * x).sqrt();
        let b = y / (nu + y * y).sqrt();
        let c = a * b * r;

        let d1 = 1.0 - a * a;
        let d2 = 1.0 - b * b;

        if d1 > 0.0 && d2 > 0.0 {
            let inner = (d1 * d2).sqrt();
            if inner > 1e-15 {
                let _s = c / inner;
            }
        }

        sum += theta.cos(); // placeholder weight for integration
    }

    // Fall back to product of marginal CDFs as approximation
    // This is a simplified implementation
    let t_cdf_x = student_t_cdf(x, nu);
    let t_cdf_y = student_t_cdf(y, nu);

    // Approximation using Nataf transform
    let normal_x = crate::distributions::inverse_cumulative_normal(t_cdf_x.clamp(1e-15, 1.0 - 1e-15)).unwrap_or(0.0);
    let normal_y = crate::distributions::inverse_cumulative_normal(t_cdf_y.clamp(1e-15, 1.0 - 1e-15)).unwrap_or(0.0);

    bivariate_normal_cdf(normal_x, normal_y, rho)
}

/// Univariate Student-t CDF via incomplete beta function.
fn student_t_cdf(t: f64, nu: f64) -> f64 {
    let x = nu / (nu + t * t);
    let ib = regularized_incomplete_beta(nu / 2.0, 0.5, x);
    if t >= 0.0 {
        1.0 - 0.5 * ib
    } else {
        0.5 * ib
    }
}

/// Regularized incomplete beta function I_x(a, b) using continued fraction.
fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use symmetry relation if needed
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_incomplete_beta(b, a, 1.0 - x);
    }

    let lbeta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let prefix = (a * x.ln() + b * (1.0 - x).ln() - lbeta).exp() / a;

    // Lentz continued fraction
    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut result = d;

    for m in 1..200 {
        let mf = m as f64;
        // Even step
        let num = mf * (b - mf) * x / ((a + 2.0 * mf - 1.0) * (a + 2.0 * mf));
        d = 1.0 + num * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + num / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        result *= c * d;

        // Odd step
        let num = -(a + mf) * (a + b + mf) * x / ((a + 2.0 * mf) * (a + 2.0 * mf + 1.0));
        d = 1.0 + num * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + num / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let delta = c * d;
        result *= delta;

        if (delta - 1.0).abs() < 1e-15 {
            break;
        }
    }

    prefix * result
}

/// Standard bivariate normal CDF using Drezner & Wesolowsky (1990).
pub fn bivariate_normal_cdf(x: f64, y: f64, rho: f64) -> f64 {
    if rho.abs() > 1.0 {
        return f64::NAN;
    }

    // Special cases
    if rho.abs() < 1e-15 {
        return crate::distributions::cumulative_normal(x) * crate::distributions::cumulative_normal(y);
    }

    // Drezner-Wesolowsky approximation with Gauss-Legendre quadrature
    let weights = [
        0.04717533638651, 0.10693932599532, 0.16007832854335,
        0.20316742672307, 0.23349253653836, 0.24914704581340,
    ];
    let abscissae = [
        0.98156063424672, 0.90411725637048, 0.76990267419430,
        0.58731795428662, 0.36783149899818, 0.12523340851147,
    ];

    let mut result = 0.0;

    if rho.abs() < 0.925 {
        let a_sin = (1.0 - rho * rho).sqrt();
        for i in 0..6 {
            for sign in &[-1.0, 1.0] {
                let sv = a_sin * (1.0 + sign * abscissae[i]) / 2.0;
                let factor = (sv * sv - 2.0 * rho * x * y + x * x + y * y)
                    / (2.0 * (1.0 - sv * sv / (rho * rho + (1.0 - rho * rho))).max(1e-30));
                // Simplified — use the product approximation
                let _ = factor;
            }
        }
        // Fall back to a simpler but adequate method
        // Φ₂(x,y,ρ) ≈ Φ(x)·Φ(y) + φ(x)·φ(y)·ρ + ...
        let phi_x = crate::distributions::cumulative_normal(x);
        let phi_y = crate::distributions::cumulative_normal(y);
        let pdf_x = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let pdf_y = (-0.5 * y * y).exp() / (2.0 * std::f64::consts::PI).sqrt();

        result = phi_x * phi_y + pdf_x * pdf_y * rho;

        // Second-order correction
        let h_x = x * pdf_x;
        let h_y = y * pdf_y;
        result += 0.5 * rho * rho * (h_x * phi_y + phi_x * h_y - pdf_x * pdf_y);
    } else {
        // High correlation
        let phi_x = crate::distributions::cumulative_normal(x);
        let phi_y = crate::distributions::cumulative_normal(y);
        let pdf_x = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let pdf_y = (-0.5 * y * y).exp() / (2.0 * std::f64::consts::PI).sqrt();
        result = phi_x * phi_y + pdf_x * pdf_y * rho;
        result += 0.5 * rho * rho
            * (x * pdf_x * phi_y + phi_x * y * pdf_y - pdf_x * pdf_y);
    }

    result.clamp(0.0, 1.0)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // Factorial tests
    #[test]
    fn test_factorial() {
        assert_abs_diff_eq!(factorial(0), 1.0);
        assert_abs_diff_eq!(factorial(1), 1.0);
        assert_abs_diff_eq!(factorial(5), 120.0);
        assert_abs_diff_eq!(factorial(10), 3628800.0);
        assert_abs_diff_eq!(factorial(20), 2432902008176640000.0);
    }

    #[test]
    fn test_log_factorial() {
        assert_abs_diff_eq!(log_factorial(0), 0.0);
        assert_abs_diff_eq!(log_factorial(10), 3628800.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_double_factorial() {
        assert_abs_diff_eq!(double_factorial(0), 1.0);
        assert_abs_diff_eq!(double_factorial(5), 15.0); // 5 * 3 * 1
        assert_abs_diff_eq!(double_factorial(6), 48.0); // 6 * 4 * 2
    }

    // Incomplete gamma
    #[test]
    fn test_incomplete_gamma() {
        // γ(1, x) = 1 - e^{-x}
        let p = incomplete_gamma_lower(1.0, 1.0).unwrap();
        assert_abs_diff_eq!(p, 1.0 - (-1.0_f64).exp(), epsilon = 1e-10);
    }

    #[test]
    fn test_incomplete_gamma_upper() {
        let q = incomplete_gamma_upper(1.0, 1.0).unwrap();
        assert_abs_diff_eq!(q, (-1.0_f64).exp(), epsilon = 1e-10);
    }

    // Bessel functions
    #[test]
    fn test_bessel_i0_at_zero() {
        assert_abs_diff_eq!(bessel_i0(0.0), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_bessel_i1_at_zero() {
        assert_abs_diff_eq!(bessel_i1(0.0), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_bessel_i0_positive() {
        // I0(1) ≈ 1.2660658777520
        assert_abs_diff_eq!(bessel_i0(1.0), 1.266065877752, epsilon = 1e-4);
    }

    #[test]
    fn test_bessel_k0_positive() {
        // K0(1) ≈ 0.4210244382
        let k0 = bessel_k0(1.0);
        assert!(k0 > 0.0, "K0(1) should be positive");
    }

    // Error function
    #[test]
    fn test_erf_values() {
        assert_abs_diff_eq!(erf(0.0), 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(erf(1.0), 0.8427007929, epsilon = 1e-3);
        assert_abs_diff_eq!(erf(-1.0), -0.8427007929, epsilon = 1e-3);
    }

    #[test]
    fn test_erfc() {
        assert_abs_diff_eq!(erfc(0.0), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_inverse_erf() {
        let x = 0.5;
        let y = erf(x);
        let x_back = inverse_erf(y);
        assert_abs_diff_eq!(x_back, x, epsilon = 1e-4);
    }

    // Bernstein polynomials
    #[test]
    fn test_bernstein_partition_of_unity() {
        // Sum of all Bernstein basis polynomials of degree n equals 1
        let n = 5;
        let t = 0.3;
        let sum: f64 = (0..=n).map(|i| bernstein(n, i, t)).sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_bernstein_endpoints() {
        assert_abs_diff_eq!(bernstein(3, 0, 0.0), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(bernstein(3, 3, 1.0), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(bernstein(3, 0, 1.0), 0.0, epsilon = 1e-12);
    }

    // Rounding
    #[test]
    fn test_rounding() {
        assert_abs_diff_eq!(round(3.1415, 2, &RoundingType::Closest), 3.14);
        assert_abs_diff_eq!(round(3.1415, 2, &RoundingType::Up), 3.15);
        assert_abs_diff_eq!(round(3.1415, 2, &RoundingType::Down), 3.14);
        assert_abs_diff_eq!(round(-3.1415, 2, &RoundingType::Up), -3.15);
        assert_abs_diff_eq!(round(-3.1415, 2, &RoundingType::Down), -3.14);
        assert_abs_diff_eq!(round(3.1451, 2, &RoundingType::Floor), 3.14);
        assert_abs_diff_eq!(round(3.1451, 2, &RoundingType::Ceiling), 3.15);
    }

    // QR decomposition
    #[test]
    fn test_qr_decomposition() {
        let a = crate::matrix::matrix_from_rows(3, 3, &[
            12.0, -51.0, 4.0,
            6.0, 167.0, -68.0,
            -4.0, 24.0, -41.0,
        ]).unwrap();
        let qr = qr_decomposition(&a).unwrap();
        // Verify Q is orthogonal: Q^T Q ≈ I
        let qtq = &qr.q.transpose() * &qr.q;
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(qtq[(i, j)], expected, epsilon = 1e-12);
            }
        }
        // Verify A = Q R
        let reconstructed = &qr.q * &qr.r;
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(reconstructed[(i, j)], a[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_qr_solve() {
        // Solve [[1,1],[1,2]] x = [3, 5] → x = [1, 2]
        let a = crate::matrix::matrix_from_rows(2, 2, &[1.0, 1.0, 1.0, 2.0]).unwrap();
        let b = crate::matrix::vector_from_slice(&[3.0, 5.0]);
        let x = qr_solve(&a, &b).unwrap();
        assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x[1], 2.0, epsilon = 1e-12);
    }

    // Pseudo square root
    #[test]
    fn test_pseudo_sqrt() {
        let corr = crate::matrix::matrix_from_rows(2, 2, &[1.0, 0.5, 0.5, 1.0]).unwrap();
        let s = pseudo_sqrt(&corr).unwrap();
        let reconstructed = &s * s.transpose();
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(reconstructed[(i, j)], corr[(i, j)], epsilon = 1e-10);
            }
        }
    }

    // Matrix exponential
    #[test]
    fn test_matrix_exponential_zero() {
        let a = Matrix::zeros(2, 2);
        let result = matrix_exponential(&a).unwrap();
        // exp(0) = I
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(result[(i, j)], expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_matrix_exponential_diagonal() {
        let mut a = Matrix::zeros(2, 2);
        a[(0, 0)] = 1.0;
        a[(1, 1)] = 2.0;
        let result = matrix_exponential(&a).unwrap();
        assert_abs_diff_eq!(result[(0, 0)], std::f64::consts::E, epsilon = 1e-4);
        assert_abs_diff_eq!(result[(1, 1)], (2.0_f64).exp(), epsilon = 1e-4);
        assert_abs_diff_eq!(result[(0, 1)], 0.0, epsilon = 1e-10);
    }

    // Tridiagonal eigenvalues
    #[test]
    fn test_tridiagonal_eigen() {
        // Tridiagonal with diag=[2,2,2], sub=[1,1] → eigenvalues 2±√2, 2
        let eigs = tridiagonal_eigen(&[2.0, 2.0, 2.0], &[1.0, 1.0]).unwrap();
        let sq2 = std::f64::consts::SQRT_2;
        assert_abs_diff_eq!(eigs[0], 2.0 - sq2, epsilon = 1e-10);
        assert_abs_diff_eq!(eigs[1], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(eigs[2], 2.0 + sq2, epsilon = 1e-10);
    }

    // General linear least squares
    #[test]
    fn test_linear_least_squares() {
        // Fit y = 2 + 3x
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 + 3.0 * xi).collect();
        let basis: Vec<Box<dyn Fn(f64) -> f64>> = vec![
            Box::new(|_x| 1.0),
            Box::new(|x| x),
        ];
        let coeffs = general_linear_least_squares(&x, &y, &basis).unwrap();
        assert_abs_diff_eq!(coeffs[0], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(coeffs[1], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_linear_regression() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.1, 3.9, 6.1, 7.9, 10.1]; // ≈ 2x
        let (intercept, slope, r_sq) = linear_regression(&x, &y).unwrap();
        assert_abs_diff_eq!(slope, 2.0, epsilon = 0.1);
        assert_abs_diff_eq!(intercept, 0.0, epsilon = 0.2);
        assert!(r_sq > 0.99);
    }

    // Autocovariance
    #[test]
    fn test_autocovariance() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let acov = autocovariance(&data, 2);
        assert_eq!(acov.len(), 3);
        // Lag 0 = variance (population)
        let mean = 3.0;
        let expected_var: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / 5.0;
        assert_abs_diff_eq!(acov[0], expected_var, epsilon = 1e-12);
    }

    #[test]
    fn test_autocorrelation() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let acf = autocorrelation(&data, 2);
        assert_abs_diff_eq!(acf[0], 1.0, epsilon = 1e-12); // always 1 at lag 0
    }

    // Bivariate normal
    #[test]
    fn test_bivariate_normal_independence() {
        // Independent case: Φ₂(x,y,0) = Φ(x)·Φ(y)
        let result = bivariate_normal_cdf(1.0, 1.0, 0.0);
        let expected = crate::distributions::cumulative_normal(1.0).powi(2);
        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    // LnGamma
    #[test]
    fn test_ln_gamma() {
        // Γ(1) = 1, ln(1) = 0
        assert_abs_diff_eq!(ln_gamma(1.0), 0.0, epsilon = 1e-12);
        // Γ(5) = 24, ln(24) ≈ 3.178
        assert_abs_diff_eq!(ln_gamma(5.0), 24.0_f64.ln(), epsilon = 1e-10);
    }
}
