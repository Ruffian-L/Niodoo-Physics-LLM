use anyhow::{bail, Result};
use flate2::write::ZlibEncoder;
use flate2::Compression;
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use std::io::Write;

/// Tool 7: Negative Relevance Feedback (The "Negative Radiance" Mechanism)
///
/// Updates a query vector based on positive and negative examples.
/// Uses Rocchio Algorithm: q_new = alpha*q + beta*avg(pos) - gamma*avg(neg)
pub fn rocchio_update(
    query: &DVector<f32>,
    positive_docs: &[DVector<f32>],
    negative_docs: &[DVector<f32>],
    alpha: f32,
    beta: f32,
    gamma: f32,
) -> DVector<f32> {
    let mut q_new = query * alpha;

    if !positive_docs.is_empty() {
        let mut pos_sum = DVector::zeros(query.len());
        for doc in positive_docs {
            pos_sum += doc;
        }
        let pos_avg = pos_sum / (positive_docs.len() as f32);
        q_new += pos_avg * beta;
    }

    if !negative_docs.is_empty() {
        let mut neg_sum = DVector::zeros(query.len());
        for doc in negative_docs {
            neg_sum += doc;
        }
        let neg_avg = neg_sum / (negative_docs.len() as f32);
        q_new -= neg_avg * gamma; // Subtraction (Repulsion)
    }

    q_new
}

/// Tool 8: Zlib Entropy Proxy
///
/// Measures information density via compression ratio.
/// H_zlib(x) = len(compress(x)) / len(x)
///
/// Used for hallucination detection and texture pruning.
pub fn compute_zlib_entropy(data: &[u8]) -> Result<f32> {
    if data.is_empty() {
        return Ok(0.0);
    }

    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(data)?;
    let compressed = encoder.finish()?;

    let raw_ratio = compressed.len() as f32 / data.len() as f32;

    // LENGTH CORRECTION:
    // Penalize short strings (high ratio artifact)
    // Boost long strings (true complexity)
    // log(len) / 5.0 gives a factor ~0.6 for short strings, ~1.2 for long
    let len = data.len() as f32;
    let length_factor = (len.ln() / 5.0).clamp(0.5, 1.5);

    // Adjusted entropy: High means "Dense & Substantial", Low means "Repetitive or Tiny"
    Ok(raw_ratio * length_factor)
}

/// Tool 10: Dimensionality Reduction via PCA (The Compression Layer)
///
/// Projects high-dim embeddings (e.g. 768) to lower manifold (e.g. 64).
/// Uses Singular Value Decomposition (SVD) of the covariance matrix.
pub fn compress_embeddings(embeddings: &[Vec<f32>], target_dim: usize) -> Result<Vec<Vec<f32>>> {
    if embeddings.is_empty() {
        return Ok(vec![]);
    }

    let n_samples = embeddings.len();
    let n_features = embeddings[0].len();

    if target_dim > n_features {
        bail!(
            "Target dimension {} cannot be greater than feature dimension {}",
            target_dim,
            n_features
        );
    }

    // 1. Construct Data Matrix X (n_samples x n_features)
    let mut x = DMatrix::from_element(n_samples, n_features, 0.0);
    for (i, vec) in embeddings.iter().enumerate() {
        if vec.len() != n_features {
            bail!("Inconsistent embedding dimensions");
        }
        for (j, &val) in vec.iter().enumerate() {
            x[(i, j)] = val;
        }
    }

    // 2. Center the data (subtract mean of each feature)
    let mut means = DVector::zeros(n_features);
    for j in 0..n_features {
        let mut sum = 0.0;
        for i in 0..n_samples {
            sum += x[(i, j)];
        }
        means[j] = sum / n_samples as f32;
    }

    for i in 0..n_samples {
        for j in 0..n_features {
            x[(i, j)] -= means[j];
        }
    }

    // 3. Compute Covariance Matrix: C = (X^T * X) / (n - 1)
    // Note: for large n_samples, this is n_features x n_features (e.g. 768x768).
    // This is manageable for SVD.
    let cov = (x.transpose() * &x) / (n_samples as f32 - 1.0);

    // 4. Eigendecomposition of Covariance Matrix
    // SymmetricEigen is generally faster and stable for covariance matrices
    let eigen = SymmetricEigen::new(cov);

    // Eigenvalues are sorted ascending by default in nalgebra SymmetricEigen?
    // Actually nalgebra docs say "eigenvalues are not sorted".
    // We need to sort them descending.

    let mut indices: Vec<usize> = (0..n_features).collect();
    let eigenvalues = eigen.eigenvalues;
    indices.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());

    // 5. Select top k eigenvectors (Principal Components)
    let eigenvectors = eigen.eigenvectors;
    let mut projection_matrix = DMatrix::zeros(n_features, target_dim);

    for (k, &idx) in indices.iter().take(target_dim).enumerate() {
        let col = eigenvectors.column(idx);
        projection_matrix.set_column(k, &col);
    }

    // 6. Project Data: Y = X * W
    // X is (n x d), W is (d x k) -> Y is (n x k)
    let projected = x * projection_matrix;

    // 7. Convert back to Vec<Vec<f32>>
    let mut result = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let mut row_vec = Vec::with_capacity(target_dim);
        for j in 0..target_dim {
            row_vec.push(projected[(i, j)]);
        }
        result.push(row_vec);
    }

    Ok(result)
}
