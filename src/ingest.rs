// src/ingest.rs
pub mod shaper;

use crate::config::SplatMemoryConfig;
use crate::constants::RVQ_LATENT_DIM;
use crate::curator::{Curator, CuratorDecision};
use crate::embeddings::EmbeddingModel;
use crate::encoder::rvq_candle::ResidualVectorQuantizerCandle;
use crate::encoder::triplane::TriplaneProjector;
use crate::ingest::shaper::Shaper;
use crate::manifold::{load_projector, ManifoldProjector};
use crate::physics::gaussian::SemanticGaussian;
use crate::structs::{SplatGeometry, SplatSemantics, SplatSemanticsV2};
use candle_core::{Device, IndexOp, Tensor};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct IngestionEngine {
    model: EmbeddingModel,
    projector: ManifoldProjector,
    device: Device,
    pub curator: Curator,
    rvq: Arc<ResidualVectorQuantizerCandle>,
}

impl IngestionEngine {
    pub fn new(config: &SplatMemoryConfig, model: EmbeddingModel) -> anyhow::Result<Self> {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        Ok(Self {
            model,
            projector: load_projector(&config.manifold_model_path, &device)?,
            curator: Curator::new(device.clone()),
            rvq: Arc::new(ResidualVectorQuantizerCandle::load_or_train(
                "rvq_codebooks.safetensors",
                &device,
            )?),
            device,
        })
    }

    pub fn process_memory(
        &self,
        new_vec: &Tensor,
        old_vec: &Tensor,
        valence: f32,
    ) -> anyhow::Result<CuratorDecision> {
        let decision = self.curator.judge(new_vec, old_vec, valence)?;

        match decision {
            CuratorDecision::Merge => {
                // Logic for merging would happen here or be signaled
                // For now, we signal Merge
                Ok(CuratorDecision::Merge)
            }
            CuratorDecision::Reject => {
                println!("Curator: Rejected memory due to conflict (Energy > 50, Valence < 0.8)");
                Ok(CuratorDecision::Reject)
            }
            CuratorDecision::Encapsulate => {
                println!("Curator: Encapsulating paradox (Energy > 50, Valence > 0.8)");
                Ok(CuratorDecision::Encapsulate)
            }
        }
    }

    pub fn ingest_batch(
        &self,
        texts: Vec<String>,
        start_id: u64,
        valence_override: Option<f32>,
    ) -> anyhow::Result<
        Vec<(
            u64,
            String,
            SplatGeometry,
            SplatSemanticsV2,
            Vec<f32>, // Embedding
            Vec<SplatGeometry>,
        )>,
    > {
        let shaper = Shaper::new(&self.model);

        // Use batch shaping for GPU efficiency
        let gaussians = shaper.shape_batch(&texts, start_id)?;

        let results: Vec<_> = gaussians
            .into_iter()
            .enumerate()
            .map(|(i, gaussian)| {
                let id = start_id + i as u64;
                let text = texts[i].clone();

                let embedding: Vec<f32> = gaussian.mean.iter().cloned().collect();
                let (geometry, semantics) = self.gaussian_to_splat_v2(&gaussian, valence_override);
                let phoneme_splats = vec![];

                (id, text, geometry, semantics, embedding, phoneme_splats)
            })
            .collect();

        Ok(results)
    }

    fn gaussian_to_splat_v2(
        &self,
        g: &SemanticGaussian,
        valence_override: Option<f32>,
    ) -> (SplatGeometry, SplatSemanticsV2) {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let mean_vec: Vec<f32> = g.mean.iter().cloned().collect();

        // 1. Matryoshka Truncation (768D -> 128D)
        let z_latent = Tensor::from_vec(mean_vec.clone(), (1, mean_vec.len()), &self.device)
            .unwrap()
            .i((.., ..RVQ_LATENT_DIM))
            .unwrap();

        // 2. RVQ Quantization -> [u16; 12]
        let (indices, reconstructions) = self.rvq.quantize(&z_latent).unwrap();
        let rvq_indices = indices[0]; // Take first item from batch of 1

        // 3. Tri-Plane Projection (Coarse Reconstruction -> 6D)
        // Use layer 2 (index 2, i.e., 3 layers) for coarse position
        let coarse_recon = &reconstructions[2];
        let pos_6d = TriplaneProjector::project(coarse_recon).unwrap()[0];

        // 4. Mass from Code Rarity (Self-Information)
        // Use same rarity scoring as chunk_and_quantize_ultra for consistency
        let coarse_mass = calculate_mass_from_rarity(&rvq_indices);

        let valence_val = if let Some(v) = valence_override {
            v
        } else {
            g.valence
        };
        let valence_byte = (valence_val * 127.0) as i8;

        let mut geometry = SplatGeometry {
            position: [pos_6d[0], pos_6d[1], pos_6d[2]], // Use first 3 of tri-plane
            scale: [1.0, 1.0, 1.0],                      // Default scale, physics will adjust
            rotation: [0.0, 0.0, 0.0, 1.0],
            color_rgba: [128, 128, 128, 255],
            physics_props: [128, 128, valence_byte as u8, 0],
            domain_valence: [0.25, 0.25, 0.25, 0.25],
        };

        crate::physics::safety::sanitize_geometry(&mut geometry);

        let mut manifold_vector = [0.0; 64];
        // Just copy first 64 dims of mean_vec for legacy compatibility
        for (i, v) in mean_vec.iter().enumerate().take(64) {
            manifold_vector[i] = *v;
        }

        let semantics = SplatSemanticsV2 {
            payload_id: g.id,
            birth_time: current_time,
            confidence: g.entropy,
            rvq_indices,
            coarse_mass,
            domain_valence: [0.25, 0.25, 0.25, 0.25], // To be filled by domain classifier
            manifold_vector,
            emotional_state: None,
            fitness_metadata: None,
        };

        (geometry, semantics)
    }
    pub fn reconstruct_embedding(&self, indices: &[u16; 8]) -> anyhow::Result<Vec<f32>> {
        let tensor = self.rvq.reconstruct_coarse(indices, 8)?;
        let vec = tensor.to_vec2()?[0].clone();
        Ok(vec)
    }
}

fn calculate_mass_from_rarity(indices: &[u16; 8]) -> u8 {
    // Simple rarity scoring: higher indices = rarer = higher mass
    // In a full implementation, this would use actual frequency statistics
    let rarity_score: u32 = indices.iter().map(|&idx| idx as u32).sum();

    // Normalize to [0.5, 2.0] range for physics simulation, then convert to u8
    let normalized = rarity_score as f32 / (8.0 * 256.0); // PARB-200: 8 stages, 256 codes each
    let mass_float = 0.5 + normalized * 1.5;

    // Convert to u8 range [1, 255] for storage
    (mass_float * 127.0) as u8
}
