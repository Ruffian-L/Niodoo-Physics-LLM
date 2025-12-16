use crate::genesis::statistics::bhattacharyya_dist;
use anyhow::Result;
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::physics::gaussian::SemanticGaussian as Rank1Gaussian;

/// The "Cell" of the cognitive tissue.
/// Represents a semantic concept as a Gaussian distribution in the cognitive manifold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticGaussian {
    // Spatial Properties (Position)
    pub mean: DVector<f32>, // mu (Position in high-dim or projected space)

    // Geometric Properties (Shape)
    pub scaling: DVector<f32>,  // Eigenvalues (Scale along axes)
    pub rotation: DMatrix<f32>, // Eigenvectors (Orientation matrix R)

    // Radiance Properties (Appearance)
    // Spherical Harmonics coeffs for view-dependent semantic "color"
    pub sh_coeffs: DVector<f32>,

    // Information Properties
    pub entropy: f32, // Zlib compression ratio (Information density)

    // Metadata
    pub id: u64,
    pub text: String,
}

impl SemanticGaussian {
    pub fn new(
        mean: DVector<f32>,
        scaling: DVector<f32>,
        rotation: DMatrix<f32>,
        sh_coeffs: DVector<f32>,
        entropy: f32,
        id: u64,
        text: String,
    ) -> Self {
        Self {
            mean,
            scaling,
            rotation,
            sh_coeffs,
            entropy,
            id,
            text,
        }
    }

    /// Computes the Covariance Matrix Sigma = R * S * S * R^T
    pub fn covariance(&self) -> DMatrix<f32> {
        let dim = self.scaling.len();
        let mut s_diag = DMatrix::zeros(dim, dim);
        for i in 0..dim {
            s_diag[(i, i)] = self.scaling[i];
        }

        // Sigma = R * S * S * R^T = (R*S) * (R*S)^T
        let rs = &self.rotation * &s_diag;
        &rs * rs.transpose()
    }

    /// Computes Precision Matrix (Inverse Covariance) = R * S^-2 * R^T
    pub fn precision_matrix(&self) -> DMatrix<f32> {
        let dim = self.scaling.len();
        let mut s_inv_sq_diag = DMatrix::zeros(dim, dim);
        for i in 0..dim {
            let s = self.scaling[i];
            // Avoid division by zero
            let val = if s.abs() < 1e-6 { 1e6 } else { 1.0 / (s * s) };
            s_inv_sq_diag[(i, i)] = val;
        }

        // P = R * S^-2 * R^T
        let rs = &self.rotation * &s_inv_sq_diag;
        &rs * self.rotation.transpose()
    }

    /// Returns flattened precision matrix for GPU upload
    pub fn precision_vec(&self) -> Vec<f32> {
        self.precision_matrix().as_slice().to_vec()
    }

    /// Computes Log Determinant of Covariance
    /// ln|Sigma| = Sum(ln(S^2_i)) = 2 * Sum(ln(S_i))
    pub fn log_det_cov(&self) -> f32 {
        self.scaling.iter().map(|s| 2.0 * s.ln()).sum()
    }

    /// Perceive the semantic "color" (meaning) from a specific "angle" (context/query vector).
    /// This uses SH coefficients to modulate the output based on viewing direction.
    ///
    /// view_angle: A normalized vector representing the "direction" of the query.
    pub fn perceive(&self, view_angle: &DVector<f32>) -> DVector<f32> {
        // Simplified SH evaluation for L=1 (Ambient + Linear)
        // Coeffs structure: [Ambient_R, Ambient_G, Ambient_B, Dir_X_R, Dir_Y_R, Dir_Z_R, ...]
        // For generalized N-dim semantics, we treat sh_coeffs as base value + directional modulation.

        // Base perception (Ambient) - first N coeffs?
        // Let's implement a simpler model for the abstract "Semantic" SH:
        // Result = Base + (Direction dot Gradient)

        let dim = self.mean.len();
        if self.sh_coeffs.len() < dim * 2 {
            // Fallback if not enough coeffs: just return mean-like identity or first dim coeffs
            // This assumes sh_coeffs stores [Base_1...Base_N, Grad_1...Grad_N]
            return self.mean.clone(); // Placeholder if malformed
        }

        let mut result = DVector::zeros(dim);

        // Assume first dim coeffs are "Ambient" (Base meaning)
        for i in 0..dim {
            result[i] = self.sh_coeffs[i];
        }

        // Apply directional modulation if view_angle provided and matches dimension
        if view_angle.len() == dim {
            // Gradient is stored in the second half
            for i in 0..dim {
                // Determine gradient vector for this dimension i.
                // This is a simplification. Real SH is more complex.
                // We'll treat the second block as a "gradient strength" vector.
                let grad_strength = self.sh_coeffs[dim + i];

                // Modulate by alignment with view angle
                // This is a "directional derivative" of meaning
                result[i] += grad_strength * view_angle.dot(&self.mean.normalize());
            }
        }

        result
    }

    /// Calculates the overlap (similarity) with another SemanticGaussian.
    /// Uses Bhattacharyya distance converted to a similarity score [0, 1].
    pub fn overlap(&self, other: &Self) -> Result<f32> {
        let sigma1 = self.covariance();
        let sigma2 = other.covariance();

        let dist = bhattacharyya_dist(&self.mean, &sigma1, &other.mean, &sigma2)?;

        // Convert distance to similarity score (0 to 1)
        // Dist 0 -> Score 1
        // Dist Inf -> Score 0
        Ok((-dist).exp())
    }

    /// Splits this Gaussian into two children (Mitosis).
    /// Returns the two children.
    ///
    /// Logic:
    /// 1. Find principal axis of variance (Column 0 of Rotation).
    /// 2. Split along that axis by +/- 0.5 * sigma.
    /// 3. Reduce scaling along that axis for children (volume conservation).
    pub fn split(&self) -> (Self, Self) {
        // Direction of maximum variance
        let principal_axis = self.rotation.column(0);
        let principal_sigma = self.scaling[0].sqrt();

        // Perturb means
        let offset = &principal_axis * (0.5 * principal_sigma);
        let mean1 = &self.mean + &offset;
        let mean2 = &self.mean - &offset;

        // Adjust scaling: Reduce variance along split axis to reduce overlap
        // e.g. new sigma = old sigma * 0.7
        let mut scaling_new = self.scaling.clone();
        scaling_new[0] *= 0.5; // Halve variance -> Sigma / sqrt(2)

        // Rotation and SH coeffs are inherited (or slightly perturbed?)
        // For now, inherit.
        let child1 = SemanticGaussian::new(
            mean1,
            scaling_new.clone(),
            self.rotation.clone(),
            self.sh_coeffs.clone(),
            self.entropy,
            self.id, // ID management? Needs unique IDs. Caller should handle re-ID.
            self.text.clone(),
        );

        let child2 = SemanticGaussian::new(
            mean2,
            scaling_new,
            self.rotation.clone(),
            self.sh_coeffs.clone(),
            self.entropy,
            self.id,
            self.text.clone(),
        );

        (child1, child2)
    }
}

impl From<Rank1Gaussian> for SemanticGaussian {
    fn from(g: Rank1Gaussian) -> Self {
        let dim = g.mean.len();

        // 1. Construct Scaling Vector
        // Axis 0: sigma * anisotropy
        // Others: sigma
        let mut scaling = DVector::from_element(dim, g.sigma_iso);
        scaling[0] *= g.anisotropy;

        // 2. Construct Rotation Matrix
        // Col 0: u_vec
        // Col 1..N: Orthogonal basis via Gram-Schmidt
        let mut rotation = DMatrix::zeros(dim, dim);

        // Set first column
        for i in 0..dim {
            rotation[(i, 0)] = g.u_vec[i];
        }

        let mut basis = vec![g.u_vec.clone()];
        let mut rng = rand::thread_rng();

        for col in 1..dim {
            let mut v = DVector::from_iterator(dim, (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0));

            // Orthogonalize against all previous
            for prev in &basis {
                let proj = v.dot(prev);
                v -= prev * proj;
            }
            if v.norm() < 1e-6 {
                // Degenerate case, pick random unit
                v = DVector::from_iterator(dim, (0..dim).map(|_| rng.gen::<f32>()));
                v.normalize_mut();
            } else {
                v.normalize_mut();
            }
            basis.push(v.clone());

            for i in 0..dim {
                rotation[(i, col)] = v[i];
            }
        }

        // 3. Convert SH Coeffs
        // Rank1: [3, D] (Row 0=Ambient, Row 1=Gradient)
        // Tissue: [2*D] (Ambient... Gradient...)
        let mut sh_vec = Vec::with_capacity(dim * 2);
        if g.sh_coeffs.nrows() >= 2 {
            for i in 0..dim {
                sh_vec.push(g.sh_coeffs[(0, i)]);
            }
            for i in 0..dim {
                sh_vec.push(g.sh_coeffs[(1, i)]);
            }
        } else {
            // Fallback
            for _ in 0..dim * 2 {
                sh_vec.push(0.0);
            }
        }

        Self {
            mean: g.mean,
            scaling,
            rotation,
            sh_coeffs: DVector::from_vec(sh_vec),
            entropy: g.entropy,
            id: g.id,
            text: g.text,
        }
    }
}
