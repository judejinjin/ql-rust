//! Sparse matrix storage (CSR), ILU(0) preconditioner, and iterative solvers.
//!
//! ## Compressed Sparse Row (CSR)
//!
//! The standard sparse storage for large systems arising in FD grids.
//! Supports matrix-vector multiply, transpose, and conversion from triplets.
//!
//! ## Preconditioned iterative solvers
//!
//! - **BiCGStab** — Bi-Conjugate Gradient Stabilised (van der Vorst, 1992)
//! - **GMRES(m)** — Generalised Minimum Residual with restart (Saad & Schultz, 1986)
//!
//! Both accept an optional ILU(0) preconditioner.

/// Compressed Sparse Row (CSR) matrix.
#[derive(Debug, Clone)]
pub struct CsrMatrix {
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// Row pointers: `row_ptr[i]..row_ptr[i+1]` index into `col_idx` / `values`.
    pub row_ptr: Vec<usize>,
    /// Column indices for each non-zero.
    pub col_idx: Vec<usize>,
    /// Non-zero values.
    pub values: Vec<f64>,
}

/// Result from an iterative linear solver.
#[derive(Debug, Clone)]
pub struct IterSolveResult {
    /// Solution vector x.
    pub x: Vec<f64>,
    /// Residual norm ‖b − Ax‖.
    pub residual: f64,
    /// Number of iterations.
    pub iterations: usize,
    /// Whether the solver converged within tolerance.
    pub converged: bool,
}

impl CsrMatrix {
    /// Build a CSR matrix from (row, col, value) triplets.
    ///
    /// Duplicate entries are summed.
    pub fn from_triplets(nrows: usize, ncols: usize, triplets: &[(usize, usize, f64)]) -> Self {
        // Count entries per row
        let mut row_counts = vec![0usize; nrows];
        for &(r, _, _) in triplets {
            row_counts[r] += 1;
        }

        // Build sorted triplets per row
        let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); nrows];
        for &(r, c, v) in triplets {
            rows[r].push((c, v));
        }
        for row in &mut rows {
            row.sort_by_key(|&(c, _)| c);
        }

        // Merge duplicate columns
        let mut row_ptr = vec![0usize; nrows + 1];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();

        for (i, row) in rows.iter().enumerate() {
            let start = col_idx.len();
            for &(c, v) in row {
                if let Some(&last_c) = col_idx.last() {
                    if last_c == c && col_idx.len() > start {
                        *values.last_mut().unwrap() += v;
                        continue;
                    }
                }
                col_idx.push(c);
                values.push(v);
            }
            row_ptr[i + 1] = col_idx.len();
        }

        Self {
            nrows,
            ncols,
            row_ptr,
            col_idx,
            values,
        }
    }

    /// Create an identity matrix in CSR format.
    pub fn identity(n: usize) -> Self {
        let row_ptr: Vec<usize> = (0..=n).collect();
        let col_idx: Vec<usize> = (0..n).collect();
        let values = vec![1.0; n];
        Self {
            nrows: n,
            ncols: n,
            row_ptr,
            col_idx,
            values,
        }
    }

    /// Number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Matrix-vector product: y = A * x.
    #[allow(clippy::needless_range_loop)]
    pub fn matvec(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.ncols);
        let mut y = vec![0.0; self.nrows];
        for i in 0..self.nrows {
            let mut sum = 0.0;
            for idx in self.row_ptr[i]..self.row_ptr[i + 1] {
                sum += self.values[idx] * x[self.col_idx[idx]];
            }
            y[i] = sum;
        }
        y
    }

    /// Get the diagonal element of row `i`. Returns 0 if not stored.
    pub fn diagonal(&self, i: usize) -> f64 {
        for idx in self.row_ptr[i]..self.row_ptr[i + 1] {
            if self.col_idx[idx] == i {
                return self.values[idx];
            }
        }
        0.0
    }
}

// ---------------------------------------------------------------------------
// ILU(0) — Incomplete LU factorisation with zero fill-in
// ---------------------------------------------------------------------------

/// ILU(0) preconditioner stored in CSR-like format.
#[derive(Debug, Clone)]
pub struct Ilu0 {
    nrows: usize,
    /// Stores L (lower, unit diagonal) and U (upper) in the same sparsity pattern as A.
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    values: Vec<f64>,
    /// Position of the diagonal in each row (for fast triangular solves).
    diag_idx: Vec<usize>,
}

impl Ilu0 {
    /// Compute ILU(0) of a CSR matrix.
    ///
    /// Panics if a zero pivot is encountered.
    pub fn new(a: &CsrMatrix) -> Self {
        assert_eq!(a.nrows, a.ncols, "ILU requires a square matrix");
        let n = a.nrows;
        let mut values = a.values.clone();
        let row_ptr = a.row_ptr.clone();
        let col_idx = a.col_idx.clone();

        // Find diagonal positions
        let mut diag_idx = vec![0usize; n];
        for i in 0..n {
            let mut found = false;
            #[allow(clippy::needless_range_loop)]
            for idx in row_ptr[i]..row_ptr[i + 1] {
                if col_idx[idx] == i {
                    diag_idx[i] = idx;
                    found = true;
                    break;
                }
            }
            assert!(found, "ILU(0): missing diagonal in row {i}");
        }

        // IKJ variant of ILU(0)
        for i in 1..n {
            for idx_ik in row_ptr[i]..diag_idx[i] {
                let k = col_idx[idx_ik];
                let pivot = values[diag_idx[k]];
                assert!(pivot.abs() > 1e-30, "ILU(0): zero pivot at row {k}");
                values[idx_ik] /= pivot;

                let l_ik = values[idx_ik];
                // Update U part: for each j > k in row i, subtract L(i,k)*U(k,j)
                for idx_kj in (diag_idx[k] + 1)..row_ptr[k + 1] {
                    let j = col_idx[idx_kj];
                    // Find j in row i
                    for idx_ij in (diag_idx[i])..row_ptr[i + 1] {
                        if col_idx[idx_ij] == j {
                            values[idx_ij] -= l_ik * values[idx_kj];
                            break;
                        }
                    }
                }
            }
        }

        Self {
            nrows: n,
            row_ptr,
            col_idx,
            values,
            diag_idx,
        }
    }

    /// Solve (LU) z = r.
    pub fn solve(&self, r: &[f64]) -> Vec<f64> {
        let n = self.nrows;
        // Forward: L y = r (L has unit diagonal)
        let mut y = r.to_vec();
        for i in 0..n {
            for idx in self.row_ptr[i]..self.diag_idx[i] {
                let k = self.col_idx[idx];
                y[i] -= self.values[idx] * y[k];
            }
        }
        // Backward: U x = y
        let mut x = y;
        for i in (0..n).rev() {
            for idx in (self.diag_idx[i] + 1)..self.row_ptr[i + 1] {
                let k = self.col_idx[idx];
                x[i] -= self.values[idx] * x[k];
            }
            x[i] /= self.values[self.diag_idx[i]];
        }
        x
    }
}

// ---------------------------------------------------------------------------
// BiCGStab
// ---------------------------------------------------------------------------

/// Solve Ax = b using BiCGStab with optional ILU(0) preconditioner.
///
/// # Arguments
/// - `a`          — CSR coefficient matrix
/// - `b`          — right-hand side vector
/// - `x0`         — initial guess (if `None`, uses zero vector)
/// - `precond`    — optional ILU(0) preconditioner
/// - `tol`        — convergence tolerance on ‖r‖/‖b‖
/// - `max_iter`   — maximum iterations
pub fn bicgstab(
    a: &CsrMatrix,
    b: &[f64],
    x0: Option<&[f64]>,
    precond: Option<&Ilu0>,
    tol: f64,
    max_iter: usize,
) -> IterSolveResult {
    let n = b.len();
    let mut x: Vec<f64> = x0.map(|v| v.to_vec()).unwrap_or_else(|| vec![0.0; n]);
    let ax = a.matvec(&x);
    let mut r: Vec<f64> = (0..n).map(|i| b[i] - ax[i]).collect();
    let r_hat: Vec<f64> = r.clone();

    let b_norm = dot(b, b).sqrt();
    if b_norm < 1e-30 {
        return IterSolveResult {
            x,
            residual: 0.0,
            iterations: 0,
            converged: true,
        };
    }

    let mut rho = 1.0_f64;
    let mut alpha = 1.0_f64;
    let mut omega = 1.0_f64;

    let mut v = vec![0.0; n];
    let mut p = vec![0.0; n];
    let mut converged = false;
    let mut iterations = 0;

    for _iter in 0..max_iter {
        let rho_new = dot(&r_hat, &r);
        if rho_new.abs() < 1e-30 {
            break;
        }
        let beta = (rho_new / rho) * (alpha / omega);
        rho = rho_new;

        // p = r + β(p − ω·v)
        for i in 0..n {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }

        // Precondition: p_hat = M^{-1} p
        let p_hat = match precond {
            Some(pc) => pc.solve(&p),
            None => p.clone(),
        };

        v = a.matvec(&p_hat);
        alpha = rho / dot(&r_hat, &v);

        // s = r − α·v
        let s: Vec<f64> = (0..n).map(|i| r[i] - alpha * v[i]).collect();

        let s_norm = dot(&s, &s).sqrt();
        if s_norm / b_norm < tol {
            for i in 0..n {
                x[i] += alpha * p_hat[i];
            }
            r = s;
            converged = true;
            iterations = _iter + 1;
            break;
        }

        // Precondition: s_hat = M^{-1} s
        let s_hat = match precond {
            Some(pc) => pc.solve(&s),
            None => s.clone(),
        };

        let t = a.matvec(&s_hat);
        omega = dot(&t, &s) / dot(&t, &t);

        for i in 0..n {
            x[i] += alpha * p_hat[i] + omega * s_hat[i];
            r[i] = s[i] - omega * t[i];
        }

        iterations = _iter + 1;

        let r_norm = dot(&r, &r).sqrt();
        if r_norm / b_norm < tol {
            converged = true;
            break;
        }
    }

    let residual = dot(&r, &r).sqrt();
    IterSolveResult {
        x,
        residual,
        iterations,
        converged,
    }
}

// ---------------------------------------------------------------------------
// GMRES(m) — restarted GMRES
// ---------------------------------------------------------------------------

/// Solve Ax = b using restarted GMRES(m) with optional ILU(0) preconditioner.
///
/// # Arguments
/// - `a`         — CSR coefficient matrix
/// - `b`         — right-hand side
/// - `x0`        — initial guess (if `None`, uses zero)
/// - `precond`   — optional ILU(0) preconditioner
/// - `restart`   — restart parameter m (Krylov subspace size before restart)
/// - `tol`       — relative convergence tolerance
/// - `max_iter`  — max number of outer (restart) iterations
pub fn gmres(
    a: &CsrMatrix,
    b: &[f64],
    x0: Option<&[f64]>,
    precond: Option<&Ilu0>,
    restart: usize,
    tol: f64,
    max_iter: usize,
) -> IterSolveResult {
    let n = b.len();
    let mut x: Vec<f64> = x0.map(|v| v.to_vec()).unwrap_or_else(|| vec![0.0; n]);

    let b_norm = dot(b, b).sqrt();
    if b_norm < 1e-30 {
        return IterSolveResult {
            x,
            residual: 0.0,
            iterations: 0,
            converged: true,
        };
    }

    let mut total_iter = 0;
    let mut converged = false;

    for _outer in 0..max_iter {
        let ax = a.matvec(&x);
        let r0: Vec<f64> = (0..n).map(|i| b[i] - ax[i]).collect();
        let beta = dot(&r0, &r0).sqrt();
        if beta / b_norm < tol {
            converged = true;
            break;
        }

        // Arnoldi process
        let m = restart.min(n);
        let mut v_basis: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
        v_basis.push(r0.iter().map(|&ri| ri / beta).collect());

        let mut h = vec![vec![0.0; m]; m + 1]; // (m+1) × m upper Hessenberg
        let mut g = vec![0.0; m + 1]; // RHS for least squares
        g[0] = beta;

        // Givens rotations
        let mut cs = vec![0.0; m];
        let mut sn = vec![0.0; m];

        let mut k_final = 0;

        for k in 0..m {
            // w = A · M^{-1} · v_k  (right preconditioning)
            let vk = &v_basis[k];
            let z = match precond {
                Some(pc) => pc.solve(vk),
                None => vk.clone(),
            };
            let mut w = a.matvec(&z);
            total_iter += 1;

            // Modified Gram-Schmidt
            for j in 0..=k {
                h[j][k] = dot(&w, &v_basis[j]);
                for i in 0..n {
                    w[i] -= h[j][k] * v_basis[j][i];
                }
            }
            h[k + 1][k] = dot(&w, &w).sqrt();

            if h[k + 1][k] > 1e-30 {
                let inv = 1.0 / h[k + 1][k];
                let v_new: Vec<f64> = w.iter().map(|&wi| wi * inv).collect();
                v_basis.push(v_new);
            } else {
                v_basis.push(vec![0.0; n]);
            }

            // Apply previous Givens rotations to column k of H
            for j in 0..k {
                let temp = cs[j] * h[j][k] + sn[j] * h[j + 1][k];
                h[j + 1][k] = -sn[j] * h[j][k] + cs[j] * h[j + 1][k];
                h[j][k] = temp;
            }

            // Compute new Givens rotation
            let rr = (h[k][k] * h[k][k] + h[k + 1][k] * h[k + 1][k]).sqrt();
            if rr > 1e-30 {
                cs[k] = h[k][k] / rr;
                sn[k] = h[k + 1][k] / rr;
            } else {
                cs[k] = 1.0;
                sn[k] = 0.0;
            }
            h[k][k] = cs[k] * h[k][k] + sn[k] * h[k + 1][k];
            h[k + 1][k] = 0.0;

            let temp = cs[k] * g[k] + sn[k] * g[k + 1];
            g[k + 1] = -sn[k] * g[k] + cs[k] * g[k + 1];
            g[k] = temp;

            k_final = k + 1;

            if g[k + 1].abs() / b_norm < tol {
                converged = true;
                break;
            }
        }

        // Back-substitution for y
        let mut y = vec![0.0; k_final];
        for i in (0..k_final).rev() {
            y[i] = g[i];
            for j in (i + 1)..k_final {
                y[i] -= h[i][j] * y[j];
            }
            if h[i][i].abs() > 1e-30 {
                y[i] /= h[i][i];
            }
        }

        // Update x = x + M^{-1} * V * y (right preconditioning)
        for k in 0..k_final {
            let zk = match precond {
                Some(pc) => pc.solve(&v_basis[k]),
                None => v_basis[k].clone(),
            };
            for i in 0..n {
                x[i] += y[k] * zk[i];
            }
        }

        if converged {
            break;
        }
    }

    let ax = a.matvec(&x);
    let residual = (0..n)
        .map(|i| (b[i] - ax[i]).powi(2))
        .sum::<f64>()
        .sqrt();

    IterSolveResult {
        x,
        residual,
        iterations: total_iter,
        converged,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_tridiag(n: usize) -> CsrMatrix {
        // −1, 2, −1 tridiagonal (SPD)
        let mut trips = Vec::new();
        for i in 0..n {
            trips.push((i, i, 2.0));
            if i > 0 {
                trips.push((i, i - 1, -1.0));
            }
            if i + 1 < n {
                trips.push((i, i + 1, -1.0));
            }
        }
        CsrMatrix::from_triplets(n, n, &trips)
    }

    #[test]
    fn csr_identity_matvec() {
        let eye = CsrMatrix::identity(4);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = eye.matvec(&x);
        assert_eq!(y, x);
    }

    #[test]
    fn csr_tridiag_matvec() {
        let a = make_tridiag(4);
        let x = vec![1.0, 0.0, 0.0, 0.0];
        let y = a.matvec(&x);
        // [2, -1, 0, 0]
        assert_abs_diff_eq!(y[0], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(y[1], -1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(y[2], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn csr_nnz_and_diagonal() {
        let a = make_tridiag(5);
        assert_eq!(a.nnz(), 13); // 5 diag + 4 upper + 4 lower
        for i in 0..5 {
            assert_abs_diff_eq!(a.diagonal(i), 2.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn ilu0_identity() {
        let eye = CsrMatrix::identity(3);
        let ilu = Ilu0::new(&eye);
        let r = vec![1.0, 2.0, 3.0];
        let z = ilu.solve(&r);
        assert_abs_diff_eq!(z[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(z[1], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(z[2], 3.0, epsilon = 1e-12);
    }

    #[test]
    fn bicgstab_tridiag() {
        let n = 50;
        let a = make_tridiag(n);
        let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let res = bicgstab(&a, &b, None, None, 1e-10, 200);
        assert!(res.converged, "BiCGStab did not converge");
        // Verify A·x ≈ b
        let ax = a.matvec(&res.x);
        for i in 0..n {
            assert_abs_diff_eq!(ax[i], b[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn bicgstab_with_ilu_precond() {
        let n = 100;
        let a = make_tridiag(n);
        let b: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let ilu = Ilu0::new(&a);
        let res = bicgstab(&a, &b, None, Some(&ilu), 1e-12, 200);
        assert!(res.converged, "preconditioned BiCGStab did not converge");
        let ax = a.matvec(&res.x);
        for i in 0..n {
            assert_abs_diff_eq!(ax[i], b[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn gmres_tridiag() {
        let n = 50;
        let a = make_tridiag(n);
        let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let res = gmres(&a, &b, None, None, 30, 1e-10, 20);
        assert!(res.converged, "GMRES did not converge");
        let ax = a.matvec(&res.x);
        for i in 0..n {
            assert_abs_diff_eq!(ax[i], b[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn gmres_with_ilu_precond() {
        let n = 100;
        let a = make_tridiag(n);
        let b: Vec<f64> = (0..n).map(|i| (i as f64).cos()).collect();
        let ilu = Ilu0::new(&a);
        let res = gmres(&a, &b, None, Some(&ilu), 30, 1e-12, 20);
        assert!(res.converged, "preconditioned GMRES did not converge: iter={}, res={}", res.iterations, res.residual);
        let ax = a.matvec(&res.x);
        for i in 0..n {
            assert_abs_diff_eq!(ax[i], b[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn bicgstab_converges_faster_with_precond() {
        let n = 100;
        let a = make_tridiag(n);
        let b: Vec<f64> = vec![1.0; n];
        let res_no = bicgstab(&a, &b, None, None, 1e-10, 500);
        let ilu = Ilu0::new(&a);
        let res_pc = bicgstab(&a, &b, None, Some(&ilu), 1e-10, 500);
        assert!(res_no.converged);
        assert!(res_pc.converged);
        assert!(res_pc.iterations <= res_no.iterations,
                "precond should converge faster: {} vs {}", res_pc.iterations, res_no.iterations);
    }
}
