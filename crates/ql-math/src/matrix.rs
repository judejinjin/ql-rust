//! Matrix and vector utilities.
//!
//! Thin wrappers and type aliases around `nalgebra` for QuantLib-compatible
//! linear algebra operations.

use ql_core::errors::{QLError, QLResult};

/// Dynamic-sized column vector of `f64`.
pub type Vector = nalgebra::DVector<f64>;

/// Dynamic-sized matrix of `f64`.
pub type Matrix = nalgebra::DMatrix<f64>;

// ---------------------------------------------------------------------------
// Construction helpers
// ---------------------------------------------------------------------------

/// Create a `Vector` from a slice.
pub fn vector_from_slice(data: &[f64]) -> Vector {
    Vector::from_column_slice(data)
}

/// Create a `Matrix` from row-major data.
pub fn matrix_from_rows(rows: usize, cols: usize, data: &[f64]) -> QLResult<Matrix> {
    if data.len() != rows * cols {
        return Err(QLError::InvalidArgument(format!(
            "expected {} elements for {}x{} matrix, got {}",
            rows * cols,
            rows,
            cols,
            data.len()
        )));
    }
    // nalgebra stores column-major; we accept row-major and transpose
    let col_major = nalgebra::DMatrix::from_row_slice(rows, cols, data);
    Ok(col_major)
}

// ---------------------------------------------------------------------------
// Cholesky Decomposition
// ---------------------------------------------------------------------------

/// Compute the Cholesky decomposition of a symmetric positive-definite matrix.
///
/// Returns the lower-triangular matrix `L` such that `A = L * L^T`.
/// This is used in correlated path generation for Monte Carlo simulation.
pub fn cholesky(matrix: &Matrix) -> QLResult<Matrix> {
    let chol = nalgebra::linalg::Cholesky::new(matrix.clone()).ok_or_else(|| {
        QLError::InvalidArgument(
            "Cholesky decomposition failed: matrix is not positive definite".into(),
        )
    })?;
    Ok(chol.l())
}

// ---------------------------------------------------------------------------
// Pseudo-inverse (Moore-Penrose)
// ---------------------------------------------------------------------------

/// Compute the pseudo-inverse of a matrix using SVD.
pub fn pseudo_inverse(matrix: &Matrix, tolerance: f64) -> QLResult<Matrix> {
    let svd = nalgebra::linalg::SVD::new(matrix.clone(), true, true);
    let u = svd
        .u
        .as_ref()
        .ok_or_else(|| QLError::InvalidArgument("SVD failed to compute U".into()))?;
    let v_t = svd
        .v_t
        .as_ref()
        .ok_or_else(|| QLError::InvalidArgument("SVD failed to compute V^T".into()))?;

    // nalgebra thin SVD: U is m×k, V_t is k×n, singular_values has k entries
    // where k = min(m, n)
    let k = svd.singular_values.len();

    // Build Σ⁺ as k×k diagonal with reciprocals
    let mut sigma_inv = Matrix::zeros(k, k);
    for i in 0..k {
        let s = svd.singular_values[i];
        if s > tolerance {
            sigma_inv[(i, i)] = 1.0 / s;
        }
    }

    // A⁺ = V * Σ⁺ * Uᵀ  (n×k * k×k * k×m = n×m)
    Ok(v_t.transpose() * sigma_inv * u.transpose())
}

// ---------------------------------------------------------------------------
// Determinant
// ---------------------------------------------------------------------------

/// Compute the determinant of a square matrix.
pub fn determinant(matrix: &Matrix) -> QLResult<f64> {
    if matrix.nrows() != matrix.ncols() {
        return Err(QLError::InvalidArgument(
            "determinant requires a square matrix".into(),
        ));
    }
    Ok(matrix.determinant())
}

// ---------------------------------------------------------------------------
// Eigenvalues (symmetric)
// ---------------------------------------------------------------------------

/// Compute eigenvalues of a symmetric matrix (sorted ascending).
pub fn symmetric_eigenvalues(matrix: &Matrix) -> QLResult<Vector> {
    if matrix.nrows() != matrix.ncols() {
        return Err(QLError::InvalidArgument(
            "eigenvalue decomposition requires a square matrix".into(),
        ));
    }
    let sym = nalgebra::SymmetricEigen::new(matrix.clone());
    Ok(sym.eigenvalues)
}

// ---------------------------------------------------------------------------
// Singular Value Decomposition
// ---------------------------------------------------------------------------

/// Full SVD decomposition result.
#[derive(Debug, Clone)]
pub struct SvdResult {
    /// Left singular vectors (m × min(m,n)), columns are the left singular vectors.
    pub u: Matrix,
    /// Singular values in descending order.
    pub singular_values: Vector,
    /// Right singular vectors transposed (min(m,n) × n).
    pub v_t: Matrix,
}

/// Compute the (thin) Singular Value Decomposition of a matrix.
///
/// Returns `U`, `Σ` (as a vector), and `V^T` such that `A = U · diag(Σ) · V^T`.
/// Singular values are returned in descending order.
pub fn svd(matrix: &Matrix) -> QLResult<SvdResult> {
    let decomp = nalgebra::linalg::SVD::new(matrix.clone(), true, true);
    let u = decomp
        .u
        .ok_or_else(|| QLError::InvalidArgument("SVD failed to compute U".into()))?;
    let v_t = decomp
        .v_t
        .ok_or_else(|| QLError::InvalidArgument("SVD failed to compute V^T".into()))?;

    // nalgebra doesn't guarantee descending order — sort if needed
    let sv = &decomp.singular_values;
    let k = sv.len();
    let mut indices: Vec<usize> = (0..k).collect();
    indices.sort_by(|&a, &b| sv[b].partial_cmp(&sv[a]).unwrap());

    let mut sorted_sv = Vector::zeros(k);
    let mut sorted_u = Matrix::zeros(u.nrows(), k);
    let mut sorted_vt = Matrix::zeros(k, v_t.ncols());
    for (new_i, &old_i) in indices.iter().enumerate() {
        sorted_sv[new_i] = sv[old_i];
        sorted_u.set_column(new_i, &u.column(old_i));
        sorted_vt.set_row(new_i, &v_t.row(old_i));
    }

    Ok(SvdResult {
        u: sorted_u,
        singular_values: sorted_sv,
        v_t: sorted_vt,
    })
}

/// Compute the rank of a matrix (number of singular values above tolerance).
pub fn matrix_rank(matrix: &Matrix, tolerance: f64) -> QLResult<usize> {
    let decomp = svd(matrix)?;
    Ok(decomp.singular_values.iter().filter(|&&s| s > tolerance).count())
}

/// Compute the condition number (ratio of largest to smallest singular value).
///
/// Returns `f64::INFINITY` if the matrix is singular.
pub fn condition_number(matrix: &Matrix) -> QLResult<f64> {
    let decomp = svd(matrix)?;
    let k = decomp.singular_values.len();
    if k == 0 {
        return Ok(f64::INFINITY);
    }
    let s_max = decomp.singular_values[0];
    let s_min = decomp.singular_values[k - 1];
    if s_min < 1e-15 {
        Ok(f64::INFINITY)
    } else {
        Ok(s_max / s_min)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn vector_basics() {
        let v = vector_from_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(v.len(), 3);
        assert_abs_diff_eq!(v[0], 1.0);
        assert_abs_diff_eq!(v.dot(&v), 14.0, epsilon = 1e-14);
    }

    #[test]
    fn matrix_construction() {
        let m = matrix_from_rows(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 3);
        assert_abs_diff_eq!(m[(0, 0)], 1.0);
        assert_abs_diff_eq!(m[(0, 2)], 3.0);
        assert_abs_diff_eq!(m[(1, 0)], 4.0);
    }

    #[test]
    fn matrix_bad_size() {
        assert!(matrix_from_rows(2, 3, &[1.0, 2.0]).is_err());
    }

    #[test]
    fn cholesky_identity() {
        let m = Matrix::identity(3, 3);
        let l = cholesky(&m).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(l[(i, j)], expected, epsilon = 1e-14);
            }
        }
    }

    #[test]
    fn cholesky_correlation() {
        // Correlation matrix: [[1, 0.5], [0.5, 1]]
        let m = matrix_from_rows(2, 2, &[1.0, 0.5, 0.5, 1.0]).unwrap();
        let l = cholesky(&m).unwrap();
        // Verify L * L^T = m
        let reconstructed = &l * l.transpose();
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(reconstructed[(i, j)], m[(i, j)], epsilon = 1e-14);
            }
        }
    }

    #[test]
    fn cholesky_not_positive_definite() {
        let m = matrix_from_rows(2, 2, &[1.0, 2.0, 2.0, 1.0]).unwrap();
        assert!(cholesky(&m).is_err());
    }

    #[test]
    fn determinant_2x2() {
        let m = matrix_from_rows(2, 2, &[3.0, 8.0, 4.0, 6.0]).unwrap();
        // det = 3*6 - 8*4 = -14
        assert_abs_diff_eq!(determinant(&m).unwrap(), -14.0, epsilon = 1e-12);
    }

    #[test]
    fn determinant_identity() {
        let m = Matrix::identity(4, 4);
        assert_abs_diff_eq!(determinant(&m).unwrap(), 1.0, epsilon = 1e-14);
    }

    #[test]
    fn pseudo_inverse_identity() {
        let m = Matrix::identity(3, 3);
        let pi = pseudo_inverse(&m, 1e-10).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(pi[(i, j)], expected, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn pseudo_inverse_rectangular() {
        // [[1, 0], [0, 1], [0, 0]] — pseudo-inverse should be [[1, 0, 0], [0, 1, 0]]
        let m = matrix_from_rows(3, 2, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
        let pi = pseudo_inverse(&m, 1e-10).unwrap();
        assert_eq!(pi.nrows(), 2);
        assert_eq!(pi.ncols(), 3);
        assert_abs_diff_eq!(pi[(0, 0)], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(pi[(1, 1)], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(pi[(0, 2)], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn eigenvalues_diagonal() {
        let m = matrix_from_rows(3, 3, &[3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0]).unwrap();
        let eigs = symmetric_eigenvalues(&m).unwrap();
        let mut sorted: Vec<f64> = eigs.iter().cloned().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_abs_diff_eq!(sorted[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(sorted[1], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(sorted[2], 3.0, epsilon = 1e-12);
    }

    #[test]
    fn matrix_multiply() {
        let a = matrix_from_rows(2, 2, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = matrix_from_rows(2, 2, &[5.0, 6.0, 7.0, 8.0]).unwrap();
        let c = &a * &b;
        // [[19, 22], [43, 50]]
        assert_abs_diff_eq!(c[(0, 0)], 19.0, epsilon = 1e-12);
        assert_abs_diff_eq!(c[(0, 1)], 22.0, epsilon = 1e-12);
        assert_abs_diff_eq!(c[(1, 0)], 43.0, epsilon = 1e-12);
        assert_abs_diff_eq!(c[(1, 1)], 50.0, epsilon = 1e-12);
    }

    #[test]
    fn svd_identity() {
        let m = Matrix::identity(3, 3);
        let res = svd(&m).unwrap();
        // All singular values should be 1
        for i in 0..3 {
            assert_abs_diff_eq!(res.singular_values[i], 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn svd_reconstruction() {
        let m = matrix_from_rows(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let res = svd(&m).unwrap();
        // Reconstruct: A = U * diag(s) * V^T
        let k = res.singular_values.len();
        let mut sigma = Matrix::zeros(k, k);
        for i in 0..k {
            sigma[(i, i)] = res.singular_values[i];
        }
        let reconstructed = &res.u * sigma * &res.v_t;
        for i in 0..2 {
            for j in 0..3 {
                assert_abs_diff_eq!(reconstructed[(i, j)], m[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn svd_singular_values_descending() {
        let m = matrix_from_rows(3, 3, &[1.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0]).unwrap();
        let res = svd(&m).unwrap();
        for i in 1..res.singular_values.len() {
            assert!(res.singular_values[i] <= res.singular_values[i - 1] + 1e-14,
                    "SVs not descending: {} > {}", res.singular_values[i], res.singular_values[i - 1]);
        }
    }

    #[test]
    fn matrix_rank_full() {
        let m = Matrix::identity(4, 4);
        assert_eq!(matrix_rank(&m, 1e-10).unwrap(), 4);
    }

    #[test]
    fn matrix_rank_deficient() {
        // Rank 1: [[1,2],[2,4]]
        let m = matrix_from_rows(2, 2, &[1.0, 2.0, 2.0, 4.0]).unwrap();
        assert_eq!(matrix_rank(&m, 1e-10).unwrap(), 1);
    }

    #[test]
    fn condition_number_identity() {
        let m = Matrix::identity(3, 3);
        assert_abs_diff_eq!(condition_number(&m).unwrap(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn condition_number_singular() {
        let m = matrix_from_rows(2, 2, &[1.0, 2.0, 2.0, 4.0]).unwrap();
        assert!(condition_number(&m).unwrap().is_infinite());
    }
}
