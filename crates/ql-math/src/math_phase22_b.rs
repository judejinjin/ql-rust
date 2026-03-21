//! Phase 22 Math Extensions — Part B (G231–G237)
//!
//! - [`LatticeRsg`] / [`RandomizedLDS`] (G231) — Lattice and randomised low-discrepancy sequences
//! - [`SymmetricSchurDecomposition`] (G232) — Symmetric Schur eigendecomposition
//! - [`get_covariance`] / [`CovarianceDecomposition`] / [`BasisIncompleteOrdered`] / [`factor_reduction`] / [`TAPCorrelations`] (G233)
//! - [`PascalTriangle`] / [`PrimeNumbers`] / [`TransformedGrid`] (G234)
//! - [`beta_function`] / [`exponential_integral_ei`] / [`sine_integral`] / [`cosine_integral`] (G235)
//! - [`DiscreteTrapezoidIntegral`] / [`DiscreteSimpsonIntegral`] (G236)
//! - [`GaussLaguerreCosinePolynomial`] / [`GaussLaguerreSinePolynomial`] / [`MomentBasedGaussianPolynomial`] (G237)

use ql_core::errors::{QLError, QLResult};

// ===========================================================================
// G231: LatticeRsg / RandomizedLDS
// ===========================================================================

/// Rank-1 lattice rule quasi-random sequence generator.
///
/// Generates the sequence `x_i = frac(i * z / N)` where `z` is the
/// generating vector and `N` is the number of points.
#[derive(Clone, Debug)]
pub struct LatticeRsg {
    dimensionality: usize,
    n: usize,
    z: Vec<f64>,
    counter: usize,
}

impl LatticeRsg {
    /// Create a new lattice RSG.
    ///
    /// * `dimensionality` — number of dimensions
    /// * `z` — generating vector (length = dimensionality)  
    /// * `n` — number of lattice points
    pub fn new(dimensionality: usize, z: Vec<f64>, n: usize) -> QLResult<Self> {
        if z.len() != dimensionality {
            return Err(QLError::InvalidArgument(format!(
                "generating vector length {} != dimensionality {}",
                z.len(),
                dimensionality
            )));
        }
        Ok(Self {
            dimensionality,
            n,
            z,
            counter: 0,
        })
    }

    /// Skip ahead by `n` points.
    pub fn skip_to(&mut self, n: usize) {
        self.counter = n;
    }

    /// Generate the next lattice point.
    pub fn next_sequence(&mut self) -> Vec<f64> {
        let i = self.counter as f64;
        let n = self.n as f64;
        let result: Vec<f64> = self
            .z
            .iter()
            .map(|&zj| {
                let v = i * zj / n;
                v - v.floor()
            })
            .collect();
        self.counter += 1;
        result
    }

    pub fn dimension(&self) -> usize {
        self.dimensionality
    }
}

/// Randomised low-discrepancy sequence.
///
/// Applies a random shift to each point of a low-discrepancy sequence:
/// `x_i = (lds_i + shift) mod 1`.
#[derive(Clone, Debug)]
pub struct RandomizedLDS {
    dimensionality: usize,
    /// The underlying LDS points (or generator)
    lds_points: Vec<Vec<f64>>,
    /// Random shift vector
    shift: Vec<f64>,
    counter: usize,
}

impl RandomizedLDS {
    /// Create from a pre-generated set of LDS points.
    pub fn new(lds_points: Vec<Vec<f64>>, shift: Vec<f64>) -> QLResult<Self> {
        let dim = shift.len();
        for (i, pt) in lds_points.iter().enumerate() {
            if pt.len() != dim {
                return Err(QLError::InvalidArgument(format!(
                    "LDS point {} has dimension {} != shift dimension {}",
                    i,
                    pt.len(),
                    dim
                )));
            }
        }
        Ok(Self {
            dimensionality: dim,
            lds_points,
            shift,
            counter: 0,
        })
    }

    /// Create from a lattice RSG with a random shift.
    pub fn from_lattice(mut lattice: LatticeRsg, n_points: usize, seed: u64) -> Self {
        let dim = lattice.dimension();

        // Generate shift from seed
        let mut rng = crate::rng_extended::LcgRng::new(seed);
        let shift: Vec<f64> = (0..dim)
            .map(|_| crate::rng_extended::UniformRng::next_uniform(&mut rng))
            .collect();

        // Generate all LDS points
        let lds_points: Vec<Vec<f64>> = (0..n_points).map(|_| lattice.next_sequence()).collect();

        Self {
            dimensionality: dim,
            lds_points,
            shift,
            counter: 0,
        }
    }

    /// Generate the next randomised point.
    pub fn next_sequence(&mut self) -> Vec<f64> {
        let idx = self.counter % self.lds_points.len();
        let result: Vec<f64> = self.lds_points[idx]
            .iter()
            .zip(&self.shift)
            .map(|(&l, &s)| {
                let v = l + s;
                v - v.floor()
            })
            .collect();
        self.counter += 1;
        result
    }

    /// Reset the counter and regenerate the random shift.
    pub fn next_randomizer(&mut self, seed: u64) {
        let mut rng = crate::rng_extended::LcgRng::new(seed);
        for s in &mut self.shift {
            *s = crate::rng_extended::UniformRng::next_uniform(&mut rng);
        }
        self.counter = 0;
    }

    pub fn dimension(&self) -> usize {
        self.dimensionality
    }
}

// ===========================================================================
// G232: SymmetricSchurDecomposition
// ===========================================================================

/// Symmetric Schur eigendecomposition using Jacobi rotations.
///
/// For a symmetric matrix S, computes eigenvalues and eigenvectors
/// sorted in descending order.
#[derive(Clone, Debug)]
pub struct SymmetricSchurDecomposition {
    /// Eigenvalues (sorted descending).
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors as columns (n × n, sorted with eigenvalues).
    pub eigenvectors: Vec<Vec<f64>>,
}

impl SymmetricSchurDecomposition {
    /// Decompose a symmetric matrix.
    ///
    /// Input: flat row-major n × n symmetric matrix.
    pub fn new(matrix: &[f64], n: usize) -> QLResult<Self> {
        if matrix.len() != n * n {
            return Err(QLError::InvalidArgument(format!(
                "matrix size {} != n² = {}",
                matrix.len(),
                n * n
            )));
        }

        // Use nalgebra for the heavy lifting
        let mat = nalgebra::DMatrix::from_row_slice(n, n, matrix);
        let sym = nalgebra::SymmetricEigen::new(mat);

        let mut pairs: Vec<(f64, Vec<f64>)> = Vec::with_capacity(n);
        for j in 0..n {
            let eigval = sym.eigenvalues[j];
            let eigvec: Vec<f64> = (0..n).map(|i| sym.eigenvectors[(i, j)]).collect();
            pairs.push((eigval, eigvec));
        }

        // Sort descending by eigenvalue
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Sign convention: flip if first component < 0
        for pair in &mut pairs {
            if !pair.1.is_empty() && pair.1[0] < 0.0 {
                for v in &mut pair.1 {
                    *v = -*v;
                }
            }
        }

        // Zero out tiny eigenvalues
        let max_eigval = pairs.first().map(|p| p.0.abs()).unwrap_or(1.0);
        for pair in &mut pairs {
            if pair.0.abs() < 1e-16 * max_eigval {
                pair.0 = 0.0;
            }
        }

        let eigenvalues: Vec<f64> = pairs.iter().map(|p| p.0).collect();
        let eigenvectors: Vec<Vec<f64>> = pairs.iter().map(|p| p.1.clone()).collect();

        Ok(Self {
            eigenvalues,
            eigenvectors,
        })
    }

    /// Get the i-th eigenvalue (descending order).
    pub fn eigenvalue(&self, i: usize) -> f64 {
        self.eigenvalues[i]
    }

    /// Get the i-th eigenvector (descending order).
    pub fn eigenvector(&self, i: usize) -> &[f64] {
        &self.eigenvectors[i]
    }
}

// ===========================================================================
// G233: GetCovariance / CovarianceDecomposition / BasisIncompleteOrdered /
//       FactorReduction / TAPCorrelations
// ===========================================================================

/// Construct a covariance matrix from standard deviations and a correlation matrix.
///
/// `cov[i][j] = std_devs[i] * std_devs[j] * 0.5 * (corr[i][j] + corr[j][i])`
pub fn get_covariance(std_devs: &[f64], corr: &[f64], n: usize) -> QLResult<Vec<f64>> {
    if std_devs.len() != n || corr.len() != n * n {
        return Err(QLError::InvalidArgument(
            "get_covariance: dimension mismatch".into(),
        ));
    }
    let mut cov = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            cov[i * n + j] =
                std_devs[i] * std_devs[j] * 0.5 * (corr[i * n + j] + corr[j * n + i]);
        }
    }
    Ok(cov)
}

/// Decompose a covariance matrix into standard deviations and a correlation matrix.
#[derive(Clone, Debug)]
pub struct CovarianceDecomposition {
    pub variances: Vec<f64>,
    pub std_devs: Vec<f64>,
    pub correlation_matrix: Vec<f64>,
    #[allow(dead_code)]
    n: usize,
}

impl CovarianceDecomposition {
    pub fn new(covariance: &[f64], n: usize) -> QLResult<Self> {
        if covariance.len() != n * n {
            return Err(QLError::InvalidArgument(
                "CovarianceDecomposition: dimension mismatch".into(),
            ));
        }

        let variances: Vec<f64> = (0..n).map(|i| covariance[i * n + i]).collect();
        let std_devs: Vec<f64> = variances.iter().map(|&v| v.max(0.0).sqrt()).collect();

        let mut corr = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                if std_devs[i] > 1e-30 && std_devs[j] > 1e-30 {
                    corr[i * n + j] = covariance[i * n + j] / (std_devs[i] * std_devs[j]);
                } else if i == j {
                    corr[i * n + j] = 1.0;
                }
            }
        }

        Ok(Self {
            variances,
            std_devs,
            correlation_matrix: corr,
            n,
        })
    }
}

/// Incrementally build an orthonormal basis via Gram-Schmidt.
#[derive(Clone, Debug)]
pub struct BasisIncompleteOrdered {
    euclidean_dim: usize,
    basis: Vec<Vec<f64>>,
}

impl BasisIncompleteOrdered {
    pub fn new(euclidean_dim: usize) -> Self {
        Self {
            euclidean_dim,
            basis: Vec::new(),
        }
    }

    /// Add a vector to the basis. Returns `true` if it was linearly independent.
    pub fn add_vector(&mut self, v: &[f64]) -> bool {
        if v.len() != self.euclidean_dim {
            return false;
        }
        if self.basis.len() >= self.euclidean_dim {
            return false; // basis already full
        }

        // Gram-Schmidt: subtract projections onto existing basis vectors
        let mut w: Vec<f64> = v.to_vec();
        for basis_vec in &self.basis {
            let dot: f64 = w.iter().zip(basis_vec).map(|(a, b)| a * b).sum();
            for (wi, &bi) in w.iter_mut().zip(basis_vec) {
                *wi -= dot * bi;
            }
        }

        // Check norm
        let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-12 {
            return false; // linearly dependent
        }

        // Normalise
        for wi in &mut w {
            *wi /= norm;
        }
        self.basis.push(w);
        true
    }

    pub fn basis_size(&self) -> usize {
        self.basis.len()
    }

    pub fn euclidean_dimension(&self) -> usize {
        self.euclidean_dim
    }

    /// Get basis as row-major n_basis × euclidean_dim matrix.
    pub fn get_basis_as_rows(&self) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.basis.len() * self.euclidean_dim);
        for v in &self.basis {
            result.extend_from_slice(v);
        }
        result
    }
}

/// Reduce a correlation matrix to a single-factor dependence vector.
///
/// Iteratively extracts the dominant eigenvector of the correlation matrix.
pub fn factor_reduction(corr: &[f64], n: usize, max_iters: usize) -> QLResult<Vec<f64>> {
    if corr.len() != n * n {
        return Err(QLError::InvalidArgument(
            "factor_reduction: dimension mismatch".into(),
        ));
    }

    // Initial guess from column norms
    let mut factors: Vec<f64> = (0..n)
        .map(|j| {
            let col_norm: f64 = (0..n).map(|i| corr[i * n + j].powi(2)).sum::<f64>().sqrt();
            (col_norm / n as f64).sqrt()
        })
        .collect();

    for _ in 0..max_iters {
        let old_factors = factors.clone();

        // Build target = corr - diag adjustment
        let mut target = corr.to_vec();
        for i in 0..n {
            target[i * n + i] = factors[i] * factors[i];
        }

        // Eigendecompose
        let decomp = SymmetricSchurDecomposition::new(&target, n)?;

        // Extract dominant eigenvector, scaled
        let lambda = decomp.eigenvalue(0).max(0.0).sqrt();
        factors = decomp.eigenvector(0).iter().map(|&v| v * lambda).collect();

        // Clamp to [-1, 1]
        for f in &mut factors {
            *f = f.clamp(-1.0, 1.0);
        }

        // Check convergence
        let diff: f64 = factors
            .iter()
            .zip(&old_factors)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        if diff < 1e-6 {
            break;
        }
    }

    Ok(factors)
}

/// Triangular angles parametrisation (TAP) for correlation matrices.
///
/// Constructs a pseudo-root matrix from angles using the cosine/sine chain.
pub struct TAPCorrelations;

impl TAPCorrelations {
    /// Build a pseudo-root from angles.
    ///
    /// `angles` contains the angles row by row (upper-left triangular entries).
    /// The pseudo-root B has dimensions `n_rows × rank`, and `B B^T` is the
    /// resulting correlation matrix.
    pub fn from_angles(angles: &[f64], n_rows: usize, rank: usize) -> Vec<f64> {
        let mut b = vec![0.0; n_rows * rank];
        let mut angle_idx = 0;

        for i in 0..n_rows {
            let mut remaining = 1.0;
            let cols = rank.min(i + 1);
            for j in 0..cols {
                if j < cols - 1 && angle_idx < angles.len() {
                    let c = angles[angle_idx].cos();
                    let s = angles[angle_idx].sin();
                    b[i * rank + j] = remaining * c;
                    remaining *= s;
                    angle_idx += 1;
                } else {
                    b[i * rank + j] = remaining;
                }
            }
        }

        b
    }

    /// Build pseudo-root from unconstrained parameters (arctangent mapping).
    pub fn from_unconstrained(params: &[f64], n_rows: usize, rank: usize) -> Vec<f64> {
        let angles: Vec<f64> = params
            .iter()
            .map(|&x| std::f64::consts::FRAC_PI_2 - x.atan())
            .collect();
        Self::from_angles(&angles, n_rows, rank)
    }

    /// Compute Frobenius distance between `B B^T` and `target`.
    pub fn frobenius_cost(target: &[f64], pseudo_root: &[f64], n: usize, rank: usize) -> f64 {
        let mut cost = 0.0;
        for i in 0..n {
            for j in 0..n {
                let mut bb_ij = 0.0;
                for k in 0..rank {
                    bb_ij += pseudo_root[i * rank + k] * pseudo_root[j * rank + k];
                }
                let diff = bb_ij - target[i * n + j];
                cost += diff * diff;
            }
        }
        cost
    }
}

// ===========================================================================
// G234: PascalTriangle / PrimeNumbers / TransformedGrid
// ===========================================================================

/// Pascal's triangle: compute binomial coefficients.
pub struct PascalTriangle;

impl PascalTriangle {
    /// Get row `n` of Pascal's triangle: `[C(n,0), C(n,1), ..., C(n,n)]`.
    pub fn get(order: usize) -> Vec<u64> {
        let mut row = vec![1u64; order + 1];
        for i in 1..order {
            // Build from right to left to avoid overwriting
            for j in (1..=i).rev() {
                row[j] = row[j].saturating_add(row[j - 1]);
            }
        }
        row
    }

    /// Binomial coefficient `C(n, k)`.
    pub fn binomial(n: usize, k: usize) -> u64 {
        if k > n {
            return 0;
        }
        let k = k.min(n - k);
        let mut result = 1u64;
        for i in 0..k {
            result = result.saturating_mul((n - i) as u64) / (i + 1) as u64;
        }
        result
    }
}

/// Prime number sieve and lookup.
pub struct PrimeNumbers {
    primes: Vec<u64>,
}

impl Default for PrimeNumbers {
    fn default() -> Self {
        Self::new()
    }
}

impl PrimeNumbers {
    pub fn new() -> Self {
        Self {
            primes: vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47],
        }
    }

    /// Get the i-th prime number (0-indexed: get(0) = 2).
    pub fn get(&mut self, index: usize) -> u64 {
        while self.primes.len() <= index {
            self.next_prime();
        }
        self.primes[index]
    }

    fn next_prime(&mut self) {
        let mut candidate = self.primes.last().unwrap() + 2;
        loop {
            let sqrt_c = (candidate as f64).sqrt() as u64;
            let is_prime = self.primes.iter().take_while(|&&p| p <= sqrt_c).all(|&p| !candidate.is_multiple_of(p));
            if is_prime {
                self.primes.push(candidate);
                return;
            }
            candidate += 2;
        }
    }

    /// Get the first `n` primes.
    pub fn first_n(&mut self, n: usize) -> Vec<u64> {
        while self.primes.len() < n {
            self.next_prime();
        }
        self.primes[..n].to_vec()
    }
}

/// Transformed grid for finite-difference methods.
///
/// Stores a grid along with its forward, backward, and central differences
/// after an optional transformation.
#[derive(Clone, Debug)]
pub struct TransformedGrid {
    /// Original grid points.
    pub grid: Vec<f64>,
    /// Transformed grid points.
    pub transformed_grid: Vec<f64>,
    /// Backward differences: `dxm[i] = tg[i] - tg[i-1]` (dxm[0] = 0).
    pub dxm: Vec<f64>,
    /// Forward differences: `dxp[i] = tg[i+1] - tg[i]` (dxp[n-1] = 0).
    pub dxp: Vec<f64>,
    /// Central differences: `dx[i] = dxm[i] + dxp[i]`.
    pub dx: Vec<f64>,
}

impl TransformedGrid {
    /// Create with identity transform.
    pub fn new(grid: Vec<f64>) -> Self {
        Self::with_transform(grid, |x| x)
    }

    /// Create with a custom transformation.
    pub fn with_transform<F: Fn(f64) -> f64>(grid: Vec<f64>, f: F) -> Self {
        let n = grid.len();
        let transformed_grid: Vec<f64> = grid.iter().map(|&x| f(x)).collect();

        let mut dxm = vec![0.0; n];
        let mut dxp = vec![0.0; n];
        let mut dx = vec![0.0; n];

        for i in 1..n {
            dxm[i] = transformed_grid[i] - transformed_grid[i - 1];
        }
        for i in 0..n.saturating_sub(1) {
            dxp[i] = transformed_grid[i + 1] - transformed_grid[i];
        }
        for i in 0..n {
            dx[i] = dxm[i] + dxp[i];
        }

        Self {
            grid,
            transformed_grid,
            dxm,
            dxp,
            dx,
        }
    }

    /// Create a log-transformed grid.
    pub fn log_grid(grid: Vec<f64>) -> Self {
        Self::with_transform(grid, f64::ln)
    }

    pub fn size(&self) -> usize {
        self.grid.len()
    }
}

// ===========================================================================
// G235: Beta function / ExponentialIntegral Ei(x)
// ===========================================================================

/// Beta function: B(z, w) = Γ(z)Γ(w) / Γ(z+w).
pub fn beta_function(z: f64, w: f64) -> f64 {
    (crate::special_functions::ln_gamma(z) + crate::special_functions::ln_gamma(w)
        - crate::special_functions::ln_gamma(z + w))
    .exp()
}

/// Incomplete beta function I_x(a, b) (regularised).
pub fn incomplete_beta(a: f64, b: f64, x: f64) -> QLResult<f64> {
    if !(0.0..=1.0).contains(&x) {
        return Err(QLError::InvalidArgument("x must be in [0, 1]".into()));
    }
    if x == 0.0 || x == 1.0 {
        return Ok(x);
    }

    // Use symmetry: I_x(a,b) = 1 - I_{1-x}(b,a)
    if x > (a + 1.0) / (a + b + 2.0) {
        return Ok(1.0 - incomplete_beta(b, a, 1.0 - x)?);
    }

    let log_front =
        a * x.ln() + b * (1.0 - x).ln() - beta_function(a, b).ln();
    let front = log_front.exp() / a;

    // Lentz continued fraction
    let cf = beta_continued_fraction(a, b, x, 1e-16, 200);
    Ok(front * cf)
}

/// Continued fraction for the incomplete beta function.
fn beta_continued_fraction(a: f64, b: f64, x: f64, accuracy: f64, max_iter: usize) -> f64 {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=max_iter {
        let m_f = m as f64;

        // Even step
        let aa = m_f * (b - m_f) * x / ((qam + 2.0 * m_f) * (a + 2.0 * m_f));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd step
        let aa = -(a + m_f) * (qab + m_f) * x / ((a + 2.0 * m_f) * (qap + 2.0 * m_f));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < accuracy {
            break;
        }
    }

    h
}

/// Exponential integral Ei(x) for real x.
///
/// `Ei(x) = -PV ∫_{-x}^∞ e^{-t}/t dt = PV ∫_{-∞}^x e^t/t dt`
pub fn exponential_integral_ei(x: f64) -> f64 {
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }

    let euler_gamma = 0.5772156649015329;

    if x.abs() < 40.0 {
        // Power series: Ei(x) = γ + ln|x| + Σ x^n / (n · n!)
        let mut sum = 0.0;
        let mut term = 1.0;
        for n in 1..200 {
            term *= x / n as f64;
            sum += term / n as f64;
            if (term / n as f64).abs() < 1e-16 * sum.abs() {
                break;
            }
        }
        euler_gamma + x.abs().ln() + sum
    } else {
        // Asymptotic expansion: Ei(x) ≈ e^x / x · Σ n! / x^n
        let mut sum = 1.0;
        let mut term = 1.0;
        for n in 1..50 {
            let old_term = term;
            term *= n as f64 / x;
            // Diverging series — stop when terms grow
            if term.abs() > old_term.abs() {
                break;
            }
            sum += term;
        }
        x.exp() / x * sum
    }
}

/// Sine integral Si(x) = ∫_0^x sin(t)/t dt.
pub fn sine_integral(x: f64) -> f64 {
    if x.abs() < 1e-15 {
        return x;
    }

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    if x <= 4.0 {
        // Power series
        let mut sum = 0.0;
        let mut term = x;
        for n in 0..100 {
            let k = 2 * n + 1;
            sum += term / k as f64;
            term *= -x * x / ((2 * n + 2) as f64 * (2 * n + 3) as f64);
            if term.abs() < 1e-16 * sum.abs() {
                break;
            }
        }
        sign * sum
    } else {
        // Auxiliary functions f, g
        let (f, g) = si_ci_auxiliary(x);
        sign * (std::f64::consts::FRAC_PI_2 - f * x.cos() - g * x.sin())
    }
}

/// Cosine integral Ci(x) = γ + ln(x) + ∫_0^x (cos(t)-1)/t dt.
///
/// Defined for x > 0.
pub fn cosine_integral(x: f64) -> QLResult<f64> {
    if x <= 0.0 {
        return Err(QLError::InvalidArgument(
            "Cosine integral undefined for x <= 0".into(),
        ));
    }

    let euler_gamma = 0.5772156649015329;

    if x <= 4.0 {
        // Power series
        let mut sum = 0.0;
        let mut term = -x * x / 2.0;
        sum += term / 2.0;
        for n in 2..100 {
            let k = 2 * n;
            term *= -x * x / ((k - 1) as f64 * k as f64);
            sum += term / k as f64;
            if term.abs() < 1e-16 * sum.abs().max(1.0) {
                break;
            }
        }
        Ok(euler_gamma + x.ln() + sum)
    } else {
        let (f, g) = si_ci_auxiliary(x);
        Ok(f * x.sin() - g * x.cos())
    }
}

/// Auxiliary functions f and g for Si/Ci at large x.
fn si_ci_auxiliary(x: f64) -> (f64, f64) {
    // Rational Padé approximation for f(x) ~ 1/x and g(x) ~ 1/x²
    let x2 = x * x;

    // Simple asymptotic approximation
    let mut f = 1.0 / x;
    let mut g = 1.0 / x2;

    // Higher-order terms
    let mut f_sum = 1.0;
    let mut g_sum = 1.0;
    let mut term_f = 1.0;
    let mut term_g = 1.0;

    for n in 1..20 {
        let k = 2 * n;
        term_f *= -(k as f64 * (k - 1) as f64) / x2;
        term_g *= -((k + 1) as f64 * k as f64) / x2;
        if term_f.abs() > 1.0 || term_g.abs() > 1.0 {
            break;
        }
        f_sum += term_f;
        g_sum += term_g;
    }

    f *= f_sum;
    g *= g_sum;

    (f, g)
}

// ===========================================================================
// G236: DiscreteIntegrals
// ===========================================================================

/// Discrete trapezoid integration over tabulated data.
///
/// Computes `∫ f dx ≈ Σ 0.5 (x_{i+1} - x_i)(f_i + f_{i+1})`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DiscreteTrapezoidIntegral;

impl DiscreteTrapezoidIntegral {
    /// Integrate tabulated data.
    pub fn integrate(x: &[f64], f: &[f64]) -> QLResult<f64> {
        if x.len() != f.len() || x.len() < 2 {
            return Err(QLError::InvalidArgument(
                "DiscreteTrapezoidIntegral: need at least 2 points with matching lengths".into(),
            ));
        }
        let mut sum = 0.0;
        for i in 0..x.len() - 1 {
            sum += 0.5 * (x[i + 1] - x[i]) * (f[i] + f[i + 1]);
        }
        Ok(sum)
    }
}

/// Discrete Simpson integration over tabulated data.
///
/// Uses Simpson's rule on pairs of intervals with non-uniform spacing.
/// Falls back to trapezoid for the last interval if the number of points is even.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DiscreteSimpsonIntegral;

impl DiscreteSimpsonIntegral {
    /// Integrate tabulated data.
    pub fn integrate(x: &[f64], f: &[f64]) -> QLResult<f64> {
        if x.len() != f.len() || x.len() < 2 {
            return Err(QLError::InvalidArgument(
                "DiscreteSimpsonIntegral: need at least 2 points with matching lengths".into(),
            ));
        }

        let n = x.len();
        if n == 2 {
            // Only one interval: use trapezoid
            return Ok(0.5 * (x[1] - x[0]) * (f[0] + f[1]));
        }

        let mut sum = 0.0;
        let mut j = 0;

        // Process pairs of intervals
        while j + 2 < n {
            let dx0 = x[j + 1] - x[j];
            let dx1 = x[j + 2] - x[j + 1];
            let dx = dx0 + dx1;

            // Non-uniform Simpson weights
            let alpha = (2.0 * dx1 - dx0) * dx / (6.0 * dx0 + 1e-30);
            let beta = dx * dx * dx / (6.0 * dx0 * dx1 + 1e-30);
            let gamma = (2.0 * dx0 - dx1) * dx / (6.0 * dx1 + 1e-30);

            sum += alpha * f[j] + beta * f[j + 1] + gamma * f[j + 2];
            j += 2;
        }

        // If n is even, handle the last interval with trapezoid
        if j + 1 < n {
            sum += 0.5 * (x[j + 1] - x[j]) * (f[j] + f[j + 1]);
        }

        Ok(sum)
    }
}

/// Discrete trapezoid integrator (wraps a function, not tabulated data).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DiscreteTrapezoidIntegrator {
    pub evaluations: usize,
}

impl DiscreteTrapezoidIntegrator {
    pub fn new(evaluations: usize) -> Self {
        Self { evaluations }
    }

    /// Integrate a function over [a, b].
    pub fn integrate<F: Fn(f64) -> f64>(&self, f: &F, a: f64, b: f64) -> f64 {
        let n = self.evaluations.max(2) - 1;
        let h = (b - a) / n as f64;
        let mut sum = 0.5 * (f(a) + f(b));
        for i in 1..n {
            sum += f(a + i as f64 * h);
        }
        sum * h
    }
}

/// Discrete Simpson integrator (wraps a function).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DiscreteSimpsonIntegrator {
    pub evaluations: usize,
}

impl DiscreteSimpsonIntegrator {
    pub fn new(evaluations: usize) -> Self {
        Self { evaluations }
    }

    /// Integrate a function over [a, b].
    pub fn integrate<F: Fn(f64) -> f64>(&self, f: &F, a: f64, b: f64) -> f64 {
        let n = (self.evaluations.max(3) - 1) | 1; // ensure even number of intervals, but n is odd count
        let n = if n.is_multiple_of(2) { n + 1 } else { n }; // Actually need even number of intervals
        // Uniform Simpson 1/3: n must be odd (even number of intervals)
        let n_intervals = n.max(2);
        let _n_points = n_intervals + 1;
        let h = (b - a) / n_intervals as f64;

        let mut _sum = f(a) + f(b);

        // Process pairs
        let mut i = 1;
        while i < n_intervals {
            if i + 1 < n_intervals {
                _sum += 4.0 * f(a + i as f64 * h) + 2.0 * f(a + (i + 1) as f64 * h);
                i += 2;
            } else {
                // Odd remaining — trapezoid fallback
                _sum += f(a + i as f64 * h);
                i += 1;
            }
        }
        // Correct the last 2.0 to not double-count b
        // Actually, standard Simpson:
        // For n_intervals even: S = h/3 * (f0 + 4f1 + 2f2 + 4f3 + ... + 4f_{n-1} + f_n)
        // Let's just do it properly:
        let n_intervals = (self.evaluations.max(3) / 2) * 2; // even number of intervals
        let h = (b - a) / n_intervals as f64;
        let mut sum = f(a) + f(b);
        for i in 1..n_intervals {
            let coeff = if i % 2 == 1 { 4.0 } else { 2.0 };
            sum += coeff * f(a + i as f64 * h);
        }
        sum * h / 3.0
    }
}

// ===========================================================================
// G237: GaussLaguerreCosinePolynomial / MomentBasedGaussianPolynomial
// ===========================================================================

/// Trait for orthogonal polynomials defined by their moments.
///
/// Implements the Golub-Welsch algorithm to derive three-term recurrence
/// coefficients from moments.
pub trait MomentBasedGaussianPolynomial {
    /// The k-th moment: μ_k = ∫ x^k w(x) dx.
    fn moment(&self, k: usize) -> f64;

    /// μ_0 (zeroth moment).
    fn mu_0(&self) -> f64 {
        self.moment(0)
    }

    /// α coefficient for the three-term recurrence.
    fn alpha(&self, k: usize) -> f64 {
        let order = 2 * k + 2;
        let z = self.z_table(order);

        if z[k][k].abs() < 1e-30 {
            return 0.0;
        }

        let result = z[k][k + 1] / z[k][k];
        if k > 0 && z[k - 1][k - 1].abs() > 1e-30 {
            result - z[k - 1][k] / z[k - 1][k - 1]
        } else {
            result
        }
    }

    /// β coefficient for the three-term recurrence.
    fn beta(&self, k: usize) -> f64 {
        if k == 0 {
            return self.mu_0();
        }
        let order = 2 * k + 1;
        let z = self.z_table(order);

        if z[k - 1][k - 1].abs() < 1e-30 {
            return 0.0;
        }
        z[k][k] / z[k - 1][k - 1]
    }

    /// Build the Z table for the Golub-Welsch recurrence.
    fn z_table(&self, max_order: usize) -> Vec<Vec<f64>> {
        let n = max_order + 1;

        // z[0][i] = μ_i  — we need moments up to 2*n for the recurrence
        // z arrays have length n+1 to accommodate z[k-1][i+1] access
        let m = n + 1;
        let mut z: Vec<Vec<f64>> = vec![vec![0.0; m]; n];
        #[allow(clippy::needless_range_loop)]
        for i in 0..m {
            z[0][i] = self.moment(i);
        }

        // z[k][i] recurrence
        for k in 1..n {
            for i in k..n {
                let a = if z[k - 1][k - 1].abs() > 1e-30 {
                    let alpha_km1 = z[k - 1][k] / z[k - 1][k - 1];
                    z[k - 1][i + 1] - alpha_km1 * z[k - 1][i]
                } else {
                    z[k - 1][i + 1]
                };

                let b = if k >= 2 && z[k - 2][k - 2].abs() > 1e-30 {
                    let beta_km1 = z[k - 1][k - 1] / z[k - 2][k - 2];
                    beta_km1 * z[k - 2][i]
                } else {
                    0.0
                };

                z[k][i] = a - b;
            }
        }

        z
    }
}

/// Gauss-Laguerre trigonometric base for `w(x) = e^{-x} * trig(u*x)`.
struct GaussLaguerreTrigBase {
    u: f64,
    /// First trigonometric moment m0.
    m0: f64,
    /// Second trigonometric moment m1.
    m1: f64,
    /// Cached factorials.
    #[allow(dead_code)]
    factorials: Vec<f64>,
    /// Cached moments.
    #[allow(dead_code)]
    cached_moments: Vec<Option<f64>>,
}

impl GaussLaguerreTrigBase {
    fn trig_moment(&self, n: usize) -> f64 {
        if n == 0 {
            return self.m0;
        }
        if n == 1 {
            return self.m1;
        }

        // Recurrence: m_n = (2n m_{n-1} - n(n-1) m_{n-2}) / (1 + u²)
        let u2 = 1.0 + self.u * self.u;
        let mut m_prev2 = self.m0;
        let mut m_prev1 = self.m1;
        for k in 2..=n {
            let m = (2.0 * k as f64 * m_prev1 - k as f64 * (k - 1) as f64 * m_prev2) / u2;
            m_prev2 = m_prev1;
            m_prev1 = m;
        }
        m_prev1
    }

    #[allow(dead_code)]
    fn factorial(&mut self, n: usize) -> f64 {
        while self.factorials.len() <= n {
            let k = self.factorials.len();
            let prev = *self.factorials.last().unwrap_or(&1.0);
            self.factorials.push(if k == 0 { 1.0 } else { prev * k as f64 });
        }
        self.factorials[n]
    }
}

/// Gauss-Laguerre cosine polynomial.
///
/// Weight function: `w(x; u) = e^{-x} (1 + cos(u·x)) / m0'`
/// where `m0' = 1 + 1/(1+u²)`.
pub struct GaussLaguerreCosinePolynomial {
    base: GaussLaguerreTrigBase,
    m0_prime: f64,
}

impl GaussLaguerreCosinePolynomial {
    pub fn new(u: f64) -> Self {
        let u2 = u * u;
        let denom = 1.0 + u2;
        let m0 = 1.0 / denom;
        let m1 = (1.0 - u2) / (denom * denom);
        let m0_prime = 1.0 + m0;

        Self {
            base: GaussLaguerreTrigBase {
                u,
                m0,
                m1,
                factorials: vec![1.0],
                cached_moments: Vec::new(),
            },
            m0_prime,
        }
    }
}

impl MomentBasedGaussianPolynomial for GaussLaguerreCosinePolynomial {
    fn moment(&self, k: usize) -> f64 {
        let trig_m = self.base.trig_moment(k);
        let fact = {
            let mut f = 1.0;
            for i in 1..=k {
                f *= i as f64;
            }
            f
        };
        (trig_m + fact) / self.m0_prime
    }
}

/// Gauss-Laguerre sine polynomial.
///
/// Weight function: `w(x; u) = e^{-x} (1 + sin(u·x)) / m0'`
/// where `m0' = 1 + u/(1+u²)`.
pub struct GaussLaguerreSinePolynomial {
    base: GaussLaguerreTrigBase,
    m0_prime: f64,
}

impl GaussLaguerreSinePolynomial {
    pub fn new(u: f64) -> Self {
        let u2 = u * u;
        let denom = 1.0 + u2;
        let m0 = u / denom;
        let m1 = 2.0 * u / (denom * denom);
        let m0_prime = 1.0 + m0;

        Self {
            base: GaussLaguerreTrigBase {
                u,
                m0,
                m1,
                factorials: vec![1.0],
                cached_moments: Vec::new(),
            },
            m0_prime,
        }
    }
}

impl MomentBasedGaussianPolynomial for GaussLaguerreSinePolynomial {
    fn moment(&self, k: usize) -> f64 {
        let trig_m = self.base.trig_moment(k);
        let fact = {
            let mut f = 1.0;
            for i in 1..=k {
                f *= i as f64;
            }
            f
        };
        (trig_m + fact) / self.m0_prime
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ----- G231: LatticeRsg -----
    #[test]
    fn lattice_rsg_basic() {
        let mut rsg = LatticeRsg::new(2, vec![1.0, 3.0], 7).unwrap();
        let p0 = rsg.next_sequence();
        assert_eq!(p0.len(), 2);
        // i=0: frac(0*1/7) = 0, frac(0*3/7) = 0
        assert_abs_diff_eq!(p0[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(p0[1], 0.0, epsilon = 1e-10);

        let p1 = rsg.next_sequence();
        // i=1: frac(1/7) = 1/7, frac(3/7) = 3/7
        assert_abs_diff_eq!(p1[0], 1.0 / 7.0, epsilon = 1e-10);
        assert_abs_diff_eq!(p1[1], 3.0 / 7.0, epsilon = 1e-10);
    }

    #[test]
    fn randomized_lds_in_unit_cube() {
        let lattice = LatticeRsg::new(2, vec![1.0, 3.0], 7).unwrap();
        let mut rds = RandomizedLDS::from_lattice(lattice, 7, 42);
        for _ in 0..7 {
            let pt = rds.next_sequence();
            for &v in &pt {
                assert!(v >= 0.0 && v < 1.0, "value out of [0,1): {}", v);
            }
        }
    }

    // ----- G232: SymmetricSchurDecomposition -----
    #[test]
    fn schur_identity() {
        let mat = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let decomp = SymmetricSchurDecomposition::new(&mat, 3).unwrap();
        for &e in &decomp.eigenvalues {
            assert_abs_diff_eq!(e, 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn schur_diagonal() {
        let mat = vec![3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0];
        let decomp = SymmetricSchurDecomposition::new(&mat, 3).unwrap();
        // Descending order
        assert_abs_diff_eq!(decomp.eigenvalues[0], 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(decomp.eigenvalues[1], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(decomp.eigenvalues[2], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn schur_symmetric() {
        // [2 1; 1 2] -> eigenvalues 3, 1
        let mat = vec![2.0, 1.0, 1.0, 2.0];
        let decomp = SymmetricSchurDecomposition::new(&mat, 2).unwrap();
        assert_abs_diff_eq!(decomp.eigenvalues[0], 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(decomp.eigenvalues[1], 1.0, epsilon = 1e-12);
    }

    // ----- G233: Covariance / BasisIncompleteOrdered / factor_reduction -----
    #[test]
    fn get_covariance_basic() {
        let std_devs = vec![1.0, 2.0];
        let corr = vec![1.0, 0.5, 0.5, 1.0];
        let cov = get_covariance(&std_devs, &corr, 2).unwrap();
        assert_abs_diff_eq!(cov[0], 1.0, epsilon = 1e-12); // σ1² = 1
        assert_abs_diff_eq!(cov[3], 4.0, epsilon = 1e-12); // σ2² = 4
        assert_abs_diff_eq!(cov[1], 1.0, epsilon = 1e-12); // σ1·σ2·ρ = 1
        assert_abs_diff_eq!(cov[2], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn covariance_decomposition_roundtrip() {
        let cov = vec![4.0, 2.0, 2.0, 9.0];
        let decomp = CovarianceDecomposition::new(&cov, 2).unwrap();
        assert_abs_diff_eq!(decomp.std_devs[0], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(decomp.std_devs[1], 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(
            decomp.correlation_matrix[1],
            2.0 / 6.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn basis_incomplete_ordered() {
        let mut basis = BasisIncompleteOrdered::new(3);
        assert!(basis.add_vector(&[1.0, 0.0, 0.0]));
        assert!(basis.add_vector(&[0.0, 1.0, 0.0]));
        assert!(basis.add_vector(&[0.0, 0.0, 1.0]));
        assert!(!basis.add_vector(&[1.0, 1.0, 0.0])); // full
        assert_eq!(basis.basis_size(), 3);
    }

    #[test]
    fn basis_incomplete_dependent() {
        let mut basis = BasisIncompleteOrdered::new(3);
        assert!(basis.add_vector(&[1.0, 0.0, 0.0]));
        assert!(!basis.add_vector(&[2.0, 0.0, 0.0])); // dependent
        assert_eq!(basis.basis_size(), 1);
    }

    #[test]
    fn factor_reduction_identity_corr() {
        let corr = vec![1.0, 0.9, 0.9, 1.0];
        let factors = factor_reduction(&corr, 2, 25).unwrap();
        assert_eq!(factors.len(), 2);
        // Both factors should be close to sqrt(0.95) ≈ 0.975
        for f in &factors {
            assert!(f.abs() > 0.8);
        }
    }

    // ----- G233: TAPCorrelations -----
    #[test]
    fn tap_pseudo_root_is_correlation() {
        let angles = vec![0.3, 0.5, 0.7];
        let b = TAPCorrelations::from_angles(&angles, 3, 2);
        // B B^T should have diagonal = 1
        for i in 0..3 {
            let mut diag = 0.0;
            for k in 0..2 {
                diag += b[i * 2 + k] * b[i * 2 + k];
            }
            assert_abs_diff_eq!(diag, 1.0, epsilon = 1e-10);
        }
    }

    // ----- G234: PascalTriangle / PrimeNumbers / TransformedGrid -----
    #[test]
    fn pascal_triangle_row4() {
        let row = PascalTriangle::get(4);
        assert_eq!(row, vec![1, 4, 6, 4, 1]);
    }

    #[test]
    fn pascal_binomial() {
        assert_eq!(PascalTriangle::binomial(10, 3), 120);
        assert_eq!(PascalTriangle::binomial(5, 0), 1);
        assert_eq!(PascalTriangle::binomial(5, 5), 1);
    }

    #[test]
    fn prime_numbers_first_10() {
        let mut pn = PrimeNumbers::new();
        let primes: Vec<u64> = (0..10).map(|i| pn.get(i)).collect();
        assert_eq!(primes, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
    }

    #[test]
    fn prime_numbers_50th() {
        let mut pn = PrimeNumbers::new();
        assert_eq!(pn.get(49), 229); // 50th prime
    }

    #[test]
    fn transformed_grid_identity() {
        let grid = TransformedGrid::new(vec![0.0, 1.0, 3.0, 6.0]);
        assert_abs_diff_eq!(grid.dxp[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(grid.dxp[1], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(grid.dxm[2], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(grid.dx[1], 3.0, epsilon = 1e-12); // dxm+dxp = 1+2
    }

    #[test]
    fn transformed_grid_log() {
        let grid = TransformedGrid::log_grid(vec![1.0, 2.0, 4.0]);
        assert_abs_diff_eq!(grid.transformed_grid[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(
            grid.transformed_grid[1],
            2.0_f64.ln(),
            epsilon = 1e-12
        );
    }

    // ----- G235: Beta / ExponentialIntegral -----
    #[test]
    fn beta_function_values() {
        // B(1,1) = 1
        assert_abs_diff_eq!(beta_function(1.0, 1.0), 1.0, epsilon = 1e-10);
        // B(2,2) = 1/6
        assert_abs_diff_eq!(beta_function(2.0, 2.0), 1.0 / 6.0, epsilon = 1e-10);
    }

    #[test]
    fn incomplete_beta_endpoints() {
        assert_abs_diff_eq!(incomplete_beta(2.0, 3.0, 0.0).unwrap(), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(incomplete_beta(2.0, 3.0, 1.0).unwrap(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn incomplete_beta_half() {
        // I_{0.5}(1, 1) = 0.5
        let val = incomplete_beta(1.0, 1.0, 0.5).unwrap();
        assert_abs_diff_eq!(val, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn exponential_integral_known_values() {
        // Ei(1) ≈ 1.8951178
        assert_abs_diff_eq!(exponential_integral_ei(1.0), 1.8951178, epsilon = 1e-5);
        // Ei(2) ≈ 4.9542344
        assert_abs_diff_eq!(exponential_integral_ei(2.0), 4.9542344, epsilon = 1e-4);
    }

    #[test]
    fn sine_integral_known_values() {
        // Si(π) ≈ 1.8519370
        assert_abs_diff_eq!(
            sine_integral(std::f64::consts::PI),
            1.8519370,
            epsilon = 1e-4
        );
    }

    #[test]
    fn cosine_integral_known_values() {
        // Ci(1) ≈ 0.3374039
        let val = cosine_integral(1.0).unwrap();
        assert_abs_diff_eq!(val, 0.3374039, epsilon = 1e-4);
    }

    // ----- G236: DiscreteIntegrals -----
    #[test]
    fn discrete_trapezoid_linear() {
        // ∫_0^1 x dx = 0.5
        let x: Vec<f64> = (0..=100).map(|i| i as f64 / 100.0).collect();
        let f: Vec<f64> = x.iter().map(|&xi| xi).collect();
        let result = DiscreteTrapezoidIntegral::integrate(&x, &f).unwrap();
        assert_abs_diff_eq!(result, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn discrete_simpson_quadratic() {
        // ∫_0^1 x² dx = 1/3
        let x: Vec<f64> = (0..=10).map(|i| i as f64 / 10.0).collect();
        let f: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        let result = DiscreteSimpsonIntegral::integrate(&x, &f).unwrap();
        assert_abs_diff_eq!(result, 1.0 / 3.0, epsilon = 1e-4);
    }

    #[test]
    fn discrete_trapezoid_integrator_sin() {
        // ∫_0^π sin(x) dx = 2
        let integrator = DiscreteTrapezoidIntegrator::new(1000);
        let result = integrator.integrate(&f64::sin, 0.0, std::f64::consts::PI);
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-4);
    }

    #[test]
    fn discrete_simpson_integrator_sin() {
        // ∫_0^π sin(x) dx = 2
        let integrator = DiscreteSimpsonIntegrator::new(100);
        let result = integrator.integrate(&f64::sin, 0.0, std::f64::consts::PI);
        assert_abs_diff_eq!(result, 2.0, epsilon = 1e-4);
    }

    // ----- G237: GaussLaguerreCosinePolynomial -----
    #[test]
    fn gauss_laguerre_cosine_moments() {
        let poly = GaussLaguerreCosinePolynomial::new(1.0);
        let m0 = poly.moment(0);
        // m0 = (1/(1+u²) + 0!) / (1 + 1/(1+u²)) = (0.5 + 1) / 1.5 = 1.0
        assert_abs_diff_eq!(m0, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn gauss_laguerre_cosine_alpha_beta() {
        let poly = GaussLaguerreCosinePolynomial::new(0.5);
        let a0 = poly.alpha(0);
        let b0 = poly.beta(0);
        // beta(0) = mu_0
        assert_abs_diff_eq!(b0, poly.mu_0(), epsilon = 1e-10);
        // alpha should be finite
        assert!(a0.is_finite());
    }

    #[test]
    fn gauss_laguerre_sine_moments() {
        let poly = GaussLaguerreSinePolynomial::new(1.0);
        let m0 = poly.moment(0);
        // m0 = (u/(1+u²) + 0!) / (1 + u/(1+u²)) = (0.5 + 1) / 1.5 = 1.0
        assert_abs_diff_eq!(m0, 1.0, epsilon = 1e-10);
    }
}
