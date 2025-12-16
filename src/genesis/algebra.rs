use anyhow::{bail, Result};
use nalgebra::{Cholesky, DMatrix, DVector};

/// Tool 1: The Sherman-Morrison Formula (Rank-1 Covariance Updates)
///
/// Efficiently updates the inverse of a matrix A when a rank-1 perturbation uv^T is added.
/// Complexity: O(N^2) instead of O(N^3) for full inversion.
///
/// Formula: (A + uv^T)^-1 = A^-1 - (A^-1 u v^T A^-1) / (1 + v^T A^-1 u)
pub fn update_inverse_rank1(
    a_inv: &DMatrix<f32>,
    u: &DVector<f32>,
    v: &DVector<f32>,
) -> Result<DMatrix<f32>> {
    let dim = a_inv.nrows();
    if a_inv.ncols() != dim || u.len() != dim || v.len() != dim {
        bail!("Dimension mismatch in Sherman-Morrison update");
    }

    let a_inv_u = a_inv * u;
    let v_t_a_inv = v.transpose() * a_inv;

    // Scalar denominator: 1 + v^T A^-1 u
    let denominator = 1.0 + (v.transpose() * &a_inv_u)[(0, 0)];

    if denominator.abs() < 1e-6 {
        bail!("Sherman-Morrison singularity: denominator close to zero");
    }

    let numerator = &a_inv_u * &v_t_a_inv;
    let update = numerator / denominator;

    Ok(a_inv - update)
}

/// Tool 2: The Woodbury Matrix Identity (Rank-k Batch Updates)
///
/// Generalizes Sherman-Morrison to rank-k updates.
/// Useful for "Densification" and subspace projections.
///
/// Formula: (A + UCV)^-1 = A^-1 - A^-1 U (C^-1 + V A^-1 U)^-1 V A^-1
/// Where U is n x k, C is k x k, V is k x n.
pub fn update_inverse_rank_k(
    a_inv: &DMatrix<f32>,
    u: &DMatrix<f32>,
    c: &DMatrix<f32>,
    v: &DMatrix<f32>,
) -> Result<DMatrix<f32>> {
    let n = a_inv.nrows();
    let k = u.ncols();

    // Validate dimensions
    if a_inv.ncols() != n
        || u.nrows() != n
        || c.nrows() != k
        || c.ncols() != k
        || v.nrows() != k
        || v.ncols() != n
    {
        bail!("Dimension mismatch in Woodbury update");
    }

    let a_inv_u = a_inv * u;
    let v_a_inv = v * a_inv;

    // Inner term: (C^-1 + V A^-1 U)
    // For simplicity, assuming C is already invertible or provided.
    // In many Woodbury applications C is Identity, but here we keep it generic.
    // If C is singular, this fails.
    let c_inv = c
        .clone()
        .try_inverse()
        .ok_or_else(|| anyhow::anyhow!("C matrix is singular"))?;

    let inner = c_inv + (v * &a_inv_u);
    let inner_inv = inner
        .try_inverse()
        .ok_or_else(|| anyhow::anyhow!("Woodbury inner matrix singular"))?;

    let update = &a_inv_u * inner_inv * &v_a_inv;

    Ok(a_inv - update)
}

/// Tool 3: Cholesky Decomposition and Rank-1 Update
///
/// Maintains the Cholesky factor L (where A = LL^T) under updates.
/// Ensures positive definiteness is preserved.
///
/// Note: A full efficient O(N^2) Cholesky update is complex to implement from scratch.
/// For this genesis implementation, we provide the wrapper that validates PD-ness
/// and recomputes if necessary, or performs a simplified diagonal update check.
///
/// Real O(N^2) update requires careful rotation logic (Givens rotations).
/// Here we implement a "Safe Update" that falls back to decomposition if needed,
/// ensuring stability as the primary goal described in the report.
pub fn cholesky_update(l: &DMatrix<f32>, x: &DVector<f32>) -> Result<DMatrix<f32>> {
    // Reconstruct A from L: A = L * L^T
    let a = l * l.transpose();

    // Perform rank-1 update: A_new = A + x * x^T
    let a_new = a + x * x.transpose();

    // Re-decompose
    match Cholesky::new(a_new) {
        Some(cholesky) => Ok(cholesky.l()),
        None => bail!("Matrix no longer positive definite after update"),
    }
}

/// Tool 3b: Cholesky Downdate (Pruning)
///
/// A_new = A - x * x^T
pub fn cholesky_downdate(l: &DMatrix<f32>, x: &DVector<f32>) -> Result<DMatrix<f32>> {
    let a = l * l.transpose();
    let a_new = a - x * x.transpose();

    match Cholesky::new(a_new) {
        Some(cholesky) => Ok(cholesky.l()),
        None => bail!("Matrix no longer positive definite after downdate (Pruning failed)"),
    }
}
