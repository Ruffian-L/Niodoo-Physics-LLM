use anyhow::{bail, Result};
use nalgebra::{DMatrix, DVector};

/// Tool 4: The Mahalanobis Distance (Geometric Membership)
///
/// Measures the distance between a point x and a distribution D(mu, Sigma).
/// D_M(x) = sqrt((x - mu)^T Sigma^-1 (x - mu))
///
/// Used for "Hit Testing" in anisotropic space.
pub fn mahalanobis_dist(
    x: &DVector<f32>,
    mu: &DVector<f32>,
    precision_matrix: &DMatrix<f32>, // Sigma^-1
) -> Result<f32> {
    if x.len() != mu.len()
        || precision_matrix.nrows() != x.len()
        || precision_matrix.ncols() != x.len()
    {
        bail!("Dimension mismatch in Mahalanobis distance");
    }

    let diff = x - mu;
    // d^2 = diff^T * Sigma^-1 * diff
    let dist_sq = (diff.transpose() * precision_matrix * &diff)[(0, 0)];

    if dist_sq < 0.0 {
        // Can happen due to floating point errors if matrix is not perfectly PD
        return Ok(0.0);
    }

    Ok(dist_sq.sqrt())
}

/// Tool 5: The Bhattacharyya Distance (Splat-to-Splat Similarity)
///
/// Measures divergence between two distributions P1 and P2.
/// Used for clustering, merging, and "PERFIELD" classification.
///
/// D_B = (1/8)(mu1-mu2)^T Sigma^-1 (mu1-mu2) + (1/2)ln(|Sigma| / sqrt(|Sigma1|*|Sigma2|))
/// where Sigma = (Sigma1 + Sigma2) / 2
pub fn bhattacharyya_dist(
    mu1: &DVector<f32>,
    sigma1: &DMatrix<f32>,
    mu2: &DVector<f32>,
    sigma2: &DMatrix<f32>,
) -> Result<f32> {
    let _dim = mu1.len();

    // 1. Average Covariance
    let sigma_avg = (sigma1 + sigma2) * 0.5;

    // 2. First Term (Mahalanobis-like)
    // We need inverse of Sigma_avg
    let sigma_avg_inv = sigma_avg
        .clone()
        .try_inverse()
        .ok_or_else(|| anyhow::anyhow!("Average covariance singular"))?;
    let diff = mu1 - mu2;
    let term1 = 0.125 * (diff.transpose() * sigma_avg_inv * &diff)[(0, 0)];

    // 3. Second Term (Determinant ratio)
    let det_avg = sigma_avg.determinant();
    let det1 = sigma1.determinant();
    let det2 = sigma2.determinant();

    if det_avg <= 0.0 || det1 <= 0.0 || det2 <= 0.0 {
        bail!("Invalid determinants for Bhattacharyya distance (non-PD matrices)");
    }

    let term2 = 0.5 * (det_avg / (det1 * det2).sqrt()).ln();

    Ok(term1 + term2)
}

/// Tool 6: Product of Gaussians (Bayesian Sensor Fusion)
///
/// Fuses two Gaussian distributions (e.g., Visual + Text).
/// Returns the new mean and covariance (and precision).
///
/// Sigma_new = (Sigma1^-1 + Sigma2^-1)^-1
/// mu_new = Sigma_new * (Sigma1^-1 * mu1 + Sigma2^-1 * mu2)
pub fn fuse_gaussians(
    mu1: &DVector<f32>,
    prec1: &DMatrix<f32>, // Precision matrix of source 1
    mu2: &DVector<f32>,
    prec2: &DMatrix<f32>, // Precision matrix of source 2
) -> Result<(DVector<f32>, DMatrix<f32>, DMatrix<f32>)> {
    // New Precision is additive
    let prec_new = prec1 + prec2;

    // New Covariance is inverse of new precision
    let sigma_new = prec_new
        .clone()
        .try_inverse()
        .ok_or_else(|| anyhow::anyhow!("Fused precision singular"))?;

    // Weighted means
    let w1 = prec1 * mu1;
    let w2 = prec2 * mu2;
    let mu_new = &sigma_new * (w1 + w2);

    Ok((mu_new, sigma_new, prec_new))
}
