use flate2::write::ZlibEncoder;
use flate2::Compression;
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::io::prelude::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SemanticGaussian {
    pub id: u64,                 // Added ID field
    pub mean: DVector<f32>,      // μ  – embedding dimension D
    pub u_vec: DVector<f32>,     // principal needle direction (unit vector)
    pub sigma_iso: f32,          // isotropic “cloud” scale
    pub anisotropy: f32,         // 0.0 = perfect cloud, >100 = extreme needle
    pub sh_coeffs: DMatrix<f32>, // [3, D] – DC + tech_axis + vibe_axis
    pub grad_accum: f32,
    pub entropy: f32,               // Added entropy field (used in ingest)
    pub valence: f32,               // Added valence field (emotional intensity)
    pub discrete_codes: Vec<usize>, // Added discrete RVQ codes
    pub birth: f64,
    pub text: String, // kept for debugging / re-shaping
}

impl Default for SemanticGaussian {
    fn default() -> Self {
        Self {
            id: 0,
            mean: DVector::zeros(0),
            u_vec: DVector::zeros(0),
            sigma_iso: 1.0,
            anisotropy: 1.0,
            sh_coeffs: DMatrix::zeros(0, 0),
            grad_accum: 0.0,
            entropy: 0.0,
            valence: 0.0,
            discrete_codes: Vec::new(),
            birth: 0.0,
            text: String::new(),
        }
    }
}

impl SemanticGaussian {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: u64,
        mean: DVector<f32>,
        u_vec: DVector<f32>,
        sigma_iso: f32,
        anisotropy: f32,
        sh_coeffs: DMatrix<f32>,
        entropy: f32,
        valence: f32,
        discrete_codes: Vec<usize>,
        text: String,
    ) -> Self {
        Self {
            id,
            mean,
            u_vec,
            sigma_iso,
            anisotropy,
            sh_coeffs,
            grad_accum: 0.0,
            entropy,
            valence,
            discrete_codes,
            birth: 0.0,
            text,
        }
    }

    /// Real O(D) Squared Mahalanobis Distance (Tuned)
    pub fn mahalanobis_rank1(&self, query: &SemanticGaussian) -> f32 {
        // 1. View-Dependent Mean Shift
        let query_dir = &query.u_vec;
        let dim = self.mean.len();
        let mut shifted_mean = self.mean.clone();

        if self.sh_coeffs.nrows() >= 2 {
            let gradient = self.sh_coeffs.row(1).transpose();
            for i in 0..dim {
                shifted_mean[i] += gradient[i] * query_dir[i];
            }
        }

        let diff = &query.mean - &shifted_mean;

        // 2. Physics Tuning (The Fix)
        // Clamp sigma to avoid "Singular Needle" explosion
        // Lowered to 0.0001 to allow for "Super Needle" singularities in Hell test.
        let safe_sigma = self.sigma_iso.max(0.0001);

        let lambda = (safe_sigma * self.anisotropy).powi(2);
        let sigma_sq = safe_sigma.powi(2);

        let diff_sq_norm = diff.dot(&diff);
        let proj = self.u_vec.dot(&diff);

        let term1 = diff_sq_norm / sigma_sq;

        let alpha = lambda - sigma_sq;
        let denom = sigma_sq * lambda; // Removed +1e-9, handled by max() above
        let c = alpha / denom;

        let term2 = c * proj.powi(2);

        let dist_sq = (term1 - term2).max(0.0);

        // 3. Dimensionality Normalization
        // In high dims, distances grow naturally. We normalize by sqrt(dim) or a temperature.
        // T = 2.0 makes the exponential curve gentler.
        let temperature = 2.0;
        dist_sq / temperature
    }
}

pub fn compression_entropy(text: &str) -> f32 {
    let mut e = ZlibEncoder::new(Vec::new(), Compression::best());
    e.write_all(text.as_bytes()).unwrap();
    let compressed = e.finish().unwrap();
    compressed.len() as f32 / text.len() as f32
}

pub fn random_orthogonal(v: &DVector<f32>) -> DVector<f32> {
    let mut rng = rand::thread_rng();
    let dim = v.len();
    let mut ortho = DVector::from_iterator(dim, (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0));

    let v_norm_sq = v.dot(v);
    if v_norm_sq > 1e-9 {
        let proj = ortho.dot(v) / v_norm_sq;
        ortho = ortho - v * proj;
    }

    ortho.normalize()
}

impl From<SemanticGaussian> for crate::types::SplatInput {
    fn from(g: SemanticGaussian) -> Self {
        // Dummy conversion for embedding-only tests
        use crate::types::{SplatInput, SplatMeta};
        SplatInput {
            static_points: vec![[0.0, 0.0, 0.0]],
            covariances: vec![[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]],
            motion_velocities: None,
            meta: SplatMeta {
                timestamp: Some(g.birth),
                labels: vec![],
                emotional_state: None,
                fitness_metadata: None,
            },
            normals: None,
            idiv: None,
            ide: None,
            sss_params: None,
            sh_occlusion: None,
        }
    }
}
