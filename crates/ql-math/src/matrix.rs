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
}
