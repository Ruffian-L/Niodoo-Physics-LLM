use nalgebra::{DMatrix, DVector};
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualVectorQuantizer {
    pub codebooks: Vec<DMatrix<f32>>, // Each codebook is [dim x num_codes]
    pub dim: usize,
    pub num_layers: usize,
    pub codebook_size: usize,
}

impl ResidualVectorQuantizer {
    pub fn new(dim: usize, num_layers: usize, codebook_size: usize) -> Self {
        Self {
            codebooks: Vec::new(),
            dim,
            num_layers,
            codebook_size,
        }
    }

    /// Initialize codebooks randomly (for testing/mocking)
    pub fn init_random(&mut self) {
        let mut rng = rand::thread_rng();
        self.codebooks.clear();
        for _ in 0..self.num_layers {
            // Initialize with small random values
            let data: Vec<f32> = (0..self.dim * self.codebook_size)
                .map(|_| rng.gen_range(-0.02..0.02))
                .collect();
            let book = DMatrix::from_vec(self.dim, self.codebook_size, data);
            self.codebooks.push(book);
        }
    }

    /// Quantize a vector using RVQ
    /// Returns (indices, reconstructed_vector)
    pub fn quantize(&self, vector: &DVector<f32>) -> (Vec<usize>, DVector<f32>) {
        let mut residual = vector.clone();
        let mut reconstructed = DVector::zeros(self.dim);
        let mut indices = Vec::with_capacity(self.num_layers);

        for layer in 0..self.num_layers {
            if layer >= self.codebooks.len() {
                break;
            }
            let codebook = &self.codebooks[layer];

            // Find nearest neighbor
            // Dist = ||r - c||^2 = r^2 + c^2 - 2rc
            // We want to maximize 2rc - c^2 (since r^2 is constant for this step)
            // Or just compute distance directly.

            let mut best_idx = 0;
            let mut best_dist = f32::MAX;

            for i in 0..self.codebook_size {
                let code = codebook.column(i);
                let diff = &residual - code;
                let dist = diff.norm_squared();
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = i;
                }
            }

            indices.push(best_idx);
            let best_code = codebook.column(best_idx);
            reconstructed += best_code;
            residual -= best_code;
        }

        (indices, reconstructed)
    }

    /// Decode indices to vector
    pub fn decode(&self, indices: &[usize]) -> DVector<f32> {
        let mut reconstructed = DVector::zeros(self.dim);
        for (layer, &idx) in indices.iter().enumerate() {
            if layer < self.codebooks.len() && idx < self.codebook_size {
                reconstructed += self.codebooks[layer].column(idx);
            }
        }
        reconstructed
    }
}
