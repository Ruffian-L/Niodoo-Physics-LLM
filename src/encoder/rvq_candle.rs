// crates/core/src/encoder/rvq_candle.rs
// Manifesto-compliant RVQ using Candle for GPU acceleration and SafeTensors persistence
use candle_core::{DType, Device, IndexOp, Result, Tensor};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

pub const NUM_QUANTIZERS: usize = 8; // PARB-200: 8 stages for better separation
pub const CODEBOOK_SIZE: usize = 256; // POST-COSINE: 256 codes = 8 bits per stage
pub const RVQ_LATENT_DIM: usize = 1024;

/// Residual Vector Quantizer (EnCodec/SoundStream-style)
/// Discretizes 128D embeddings into 12 coarse-to-fine codes for physics-native memory
#[derive(Clone)]
pub struct ResidualVectorQuantizerCandle {
    pub codebooks: Vec<Tensor>, // [NUM_QUANTIZERS, CODEBOOK_SIZE, RVQ_LATENT_DIM]
    pub device: Device,
}

impl ResidualVectorQuantizerCandle {
    /// Load codebooks from SafeTensors file or train if missing
    pub fn load_or_train<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self> {
        let path_ref = path.as_ref();

        if path_ref.exists() {
            // Load from SafeTensors
            Self::load(path_ref, device)
        } else {
            eprintln!("RVQ codebooks not found at {:?}, initializing random (train with train_rvq binary)", path_ref);
            Self::init_random(device)
        }
    }

    /// Load codebooks from SafeTensors
    fn load<P: AsRef<Path>>(path: P, device: &Device) -> Result<Self> {
        let tensors = candle_core::safetensors::load(path, device)?;
        let mut codebooks = Vec::with_capacity(NUM_QUANTIZERS);

        for i in 0..NUM_QUANTIZERS {
            let key = format!("codebook_{}", i);
            let cb = tensors
                .get(&key)
                .ok_or_else(|| candle_core::Error::Msg(format!("Missing codebook_{}", i)))?
                .clone();

            // Verify shape: [CODEBOOK_SIZE, RVQ_LATENT_DIM]
            let dims = cb.dims();
            if dims.len() != 2 || dims[0] != CODEBOOK_SIZE || dims[1] != RVQ_LATENT_DIM {
                return Err(candle_core::Error::Msg(format!(
                    "Invalid codebook_{} shape: {:?}, expected [{}, {}]",
                    i, dims, CODEBOOK_SIZE, RVQ_LATENT_DIM
                )));
            }

            codebooks.push(cb);
        }

        Ok(Self {
            codebooks,
            device: device.clone(),
        })
    }

    /// Initialize random codebooks (for testing or first-time setup)
    fn init_random(device: &Device) -> Result<Self> {
        let mut codebooks = Vec::with_capacity(NUM_QUANTIZERS);

        for _ in 0..NUM_QUANTIZERS {
            // Xavier initialization: ~N(0, sqrt(2 / (in + out)))
            let std = (2.0 / (CODEBOOK_SIZE + RVQ_LATENT_DIM) as f64).sqrt() as f32;
            let cb = Tensor::randn(0f32, std, (CODEBOOK_SIZE, RVQ_LATENT_DIM), device)?;
            codebooks.push(cb);
        }

        Ok(Self {
            codebooks,
            device: device.clone(),
        })
    }

    /// Quantize batch of vectors to discrete RVQ codes
    ///
    /// # Arguments
    /// * `z` - Input tensor of shape (batch, RVQ_LATENT_DIM)
    ///
    /// # Returns
    /// * `indices` - Vec of [u16; NUM_QUANTIZERS] for each batch item
    /// * `reconstructions` - Vec of nested reconstructions (12 tensors, each (batch, RVQ_LATENT_DIM))
    pub fn quantize(&self, z: &Tensor) -> Result<(Vec<[u16; NUM_QUANTIZERS]>, Vec<Tensor>)> {
        let batch_size = z.dims()[0];
        let mut residual = z.clone();
        let mut z_reconstructed =
            Tensor::zeros((batch_size, RVQ_LATENT_DIM), DType::F32, &self.device)?;

        let mut all_indices = vec![[0u16; NUM_QUANTIZERS]; batch_size];
        let mut nested_reconstructions = Vec::with_capacity(NUM_QUANTIZERS);

        for (layer_idx, codebook) in self.codebooks.iter().enumerate() {
            // Compute distances: ||residual - codebook||^2
            // residual: (batch, dim)
            // codebook: (K, dim)
            // Expand: residual.unsqueeze(1) -> (batch, 1, dim)
            //         codebook.unsqueeze(0) -> (1, K, dim)
            // Broadcast subtract and square

            let residual_expanded = residual.unsqueeze(1)?; // (batch, 1, RVQ_LATENT_DIM)
            let codebook_expanded = codebook.unsqueeze(0)?; // (1, CODEBOOK_SIZE, RVQ_LATENT_DIM)

            let diff = residual_expanded.broadcast_sub(&codebook_expanded)?; // (batch, CODEBOOK_SIZE, RVQ_LATENT_DIM)
            let distances = diff.sqr()?.sum(2)?; // (batch, CODEBOOK_SIZE)

            // Find argmin indices
            let indices = distances.argmin(1)?; // (batch,)
            let indices_vec: Vec<u32> = indices.to_vec1()?;

            // Store indices
            for (b, &idx) in indices_vec.iter().enumerate() {
                all_indices[b][layer_idx] = idx as u16;
            }

            // Gather quantized vectors
            let indices_i64 = indices.to_dtype(DType::I64)?;
            let quantized = codebook.index_select(&indices_i64, 0)?; // (batch, RVQ_LATENT_DIM)

            // Update reconstruction
            z_reconstructed = (&z_reconstructed + &quantized)?;
            nested_reconstructions.push(z_reconstructed.clone());

            // Update residual
            residual = (&residual - &quantized)?;
        }

        Ok((all_indices, nested_reconstructions))
    }

    /// Reconstruct vector from RVQ indices with LOD (Level of Detail)
    ///
    /// # Arguments
    /// * `indices` - Array of 12 code indices
    /// * `layers` - Number of layers to use (1-12). Higher = finer detail
    ///
    /// # Returns
    /// * Reconstructed tensor of shape (1, RVQ_LATENT_DIM)
    pub fn reconstruct_coarse(
        &self,
        indices: &[u16; NUM_QUANTIZERS],
        layers: usize,
    ) -> Result<Tensor> {
        let layers = layers.min(NUM_QUANTIZERS).max(1);
        let mut z_reconstructed = Tensor::zeros((1, RVQ_LATENT_DIM), DType::F32, &self.device)?;

        for i in 0..layers {
            let idx = indices[i] as i64;
            let codebook = &self.codebooks[i];

            // Select the code vector
            let code = codebook.get(idx as usize)?; // (RVQ_LATENT_DIM,)
            let code = code.unsqueeze(0)?; // (1, RVQ_LATENT_DIM)

            z_reconstructed = (&z_reconstructed + &code)?;
        }

        Ok(z_reconstructed)
    }

    /// Save codebooks to SafeTensors file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut tensors_map = std::collections::HashMap::new();

        for (i, cb) in self.codebooks.iter().enumerate() {
            tensors_map.insert(format!("codebook_{}", i), cb.clone());
        }

        candle_core::safetensors::save(&tensors_map, path)?;
        Ok(())
    }
}

// Helper to avoid name collision
mod d {
    pub const MINUS_1: usize = usize::MAX;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rvq_quantize_reconstruct() -> Result<()> {
        let device = Device::Cpu;
        let rvq = ResidualVectorQuantizerCandle::init_random(&device)?;

        // Create random input
        let z = Tensor::randn(0f32, 1.0, (4, RVQ_LATENT_DIM), &device)?;

        // Quantize
        let (indices, reconstructions) = rvq.quantize(&z)?;

        // Check indices
        assert_eq!(indices.len(), 4);
        assert_eq!(indices[0].len(), NUM_QUANTIZERS);

        // Check reconstructions are nested
        assert_eq!(reconstructions.len(), NUM_QUANTIZERS);

        // Verify LOD: coarse should differ from fine
        let coarse = rvq.reconstruct_coarse(&indices[0], 2)?;
        let fine = rvq.reconstruct_coarse(&indices[0], 12)?;

        let coarse_vec: Vec<f32> = coarse.to_vec2()?[0].clone();
        let fine_vec: Vec<f32> = fine.to_vec2()?[0].clone();

        // They should be different (unless by extreme chance they're identical)
        let diff: f32 = coarse_vec
            .iter()
            .zip(&fine_vec)
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.0);

        Ok(())
    }

    #[test]
    fn test_rvq_lod_refinement() -> Result<()> {
        let device = Device::Cpu;
        let rvq = ResidualVectorQuantizerCandle::init_random(&device)?;

        let z = Tensor::randn(0f32, 1.0, (1, RVQ_LATENT_DIM), &device)?;
        let (indices, _) = rvq.quantize(&z)?;

        // Reconstruct at different LODs
        let lod_1 = rvq.reconstruct_coarse(&indices[0], 1)?;
        let lod_6 = rvq.reconstruct_coarse(&indices[0], 6)?;
        let lod_12 = rvq.reconstruct_coarse(&indices[0], 12)?;

        // Convert to vecs for comparison
        let v1: Vec<f32> = lod_1.to_vec2()?[0].clone();
        let v6: Vec<f32> = lod_6.to_vec2()?[0].clone();
        let v12: Vec<f32> = lod_12.to_vec2()?[0].clone();

        // Each higher LOD should add more detail (different values)
        let diff_1_6: f32 = v1
            .iter()
            .zip(&v6)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        let diff_6_12: f32 = v6
            .iter()
            .zip(&v12)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        assert!(diff_1_6 > 0.0, "LOD 1 and 6 should differ");
        assert!(diff_6_12 > 0.0, "LOD 6 and 12 should differ");

        Ok(())
    }
}
