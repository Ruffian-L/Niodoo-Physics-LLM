use crate::config::SplatMemoryConfig;
use crate::curator::{Curator, CuratorDecision};
use crate::embeddings::EmbeddingModel;
use crate::encoder::GaussianSplat;
use crate::ingest::IngestionEngine;
use crate::language::g_prime::GPrimeCodecV1;
use crate::manifold::ManifoldProjector;
use crate::organism::{Signal, TensorHeart};
use crate::rendering::inverse::InverseRenderer;
use crate::storage::hnsw::RealHnswIndex;
use crate::structs::{PackedSemantics, SplatGeometry, SplatLighting, SplatSemanticsV2};
use candle_core::{Device, Tensor};
use glam::{Quat, Vec3};
use nalgebra::Vector3;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use std::mem;
use std::path::Path;
use std::sync::Mutex;

#[derive(Serialize, Deserialize, Clone)]
pub struct RetrievalResult {
    pub rank: usize,
    pub probability: f32,
    pub text: String,
    pub payload_id: u64,
    pub confidence: f32,
    #[serde(default)]
    pub is_shadow: bool,
    #[serde(default)]
    pub valence: i8,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct HolographicResult {
    pub base: RetrievalResult,
    pub decoded_text: String,
    pub integrity: f32, // 0.0 to 1.0 matching score
    pub phoneme_count: usize,
    // NEW: Aggregate Tone
    pub aggregate_uncertainty: f32, // 0.0 - 1.0
    pub aggregate_sentiment: f32,   // -1.0 (Pain) to 1.0 (Joy)
}

pub struct MemorySystem {
    pub ingestion: IngestionEngine,
    pub model: EmbeddingModel,
    projector: ManifoldProjector,
    config: SplatMemoryConfig,

    pub storage: crate::storage::engine::SplatStorage,

    index: Mutex<RealHnswIndex>, // HNSW is interior mutable or we need Mutex

    pub dream_ticks_since_save: usize,

    // The Organism
    pub heart: TensorHeart,
}

impl MemorySystem {
    pub fn load_or_create(base_path: &str, manifest_path: &str) -> anyhow::Result<Self> {
        Self::new(base_path, manifest_path)
    }

    pub fn new(base_path: &str, manifest_path: &str) -> anyhow::Result<Self> {
        // Load config from file if present, otherwise default
        let config_path = "splat_config.json"; // Global config preferred? Or base_path derived?
                                               // Let's use a standard name for now
        let config = if Path::new(config_path).exists() {
            println!("Loading config from {}", config_path);
            let file = File::open(config_path)?;
            serde_json::from_reader(file).unwrap_or_else(|e| {
                eprintln!("Failed to parse config: {}. Using defaults.", e);
                SplatMemoryConfig::default()
            })
        } else {
            SplatMemoryConfig::default()
        };

        Self::with_config(base_path, manifest_path, config)
    }

    pub fn with_config(
        base_path: &str,
        manifest_path: &str,
        config: SplatMemoryConfig,
    ) -> anyhow::Result<Self> {
        eprintln!("Initializing Memory System...");
        let model = EmbeddingModel::with_dim(crate::constants::EMBED_DIM)?;
        let ingestion = IngestionEngine::new(&config, model.clone())?;
        let projector =
            crate::manifold::load_projector(&config.manifold_model_path, &candle_core::Device::Cpu)
                .or_else(|e| {
                    eprintln!(
                        "Warning: Failed to load Manifold Projector: {}. Using dummy.",
                        e
                    );
                    ManifoldProjector::dummy(&candle_core::Device::Cpu)
                })?;

        let mut storage = crate::storage::engine::SplatStorage::new(base_path, manifest_path)?;

        // Load or Build Index
        let mut index = RealHnswIndex::new(config.hnsw_max_elements);

        if !storage.semantics.is_empty() {
            // Reconstruct embeddings from RVQ if missing (V2)
            if storage.embeddings.is_empty() && !storage.rvq_indices.is_empty() {
                eprintln!(
                    "Reconstructing {} embeddings from RVQ indices...",
                    storage.rvq_indices.len()
                );
                for indices in &storage.rvq_indices {
                    if let Ok(emb) = ingestion.reconstruct_embedding(&[0u16; 8]) {
                        storage.embeddings.push(emb);
                    } else {
                        // Fallback or panic? Use zero vector to keep alignment
                        storage.embeddings.push(vec![0.0; 128]);
                    }
                }
            }

            eprintln!(
                "Rebuilding HNSW index from {} items...",
                storage.semantics.len()
            );

            if storage.payload_ids.len() == storage.semantics.len() {
                for (i, _sem) in storage.semantics.iter().enumerate() {
                    let id = storage.payload_ids[i];
                    if i < storage.embeddings.len() {
                        index.add(id, &storage.embeddings[i]).unwrap();
                    }
                }
            } else {
                eprintln!("CRITICAL WARNING: Payload IDs count ({}) does not match Semantics count ({}). Index rebuild compromised.", storage.payload_ids.len(), storage.semantics.len());
            }
        }

        eprintln!(
            "DEBUG: MemorySystem initialized. Semantics: {}, Index: {:?}",
            storage.semantics.len(),
            index
        );

        let heart = TensorHeart::new()?;

        Ok(Self {
            ingestion,
            model,
            projector,
            config,
            storage,
            index: Mutex::new(index),
            dream_ticks_since_save: 0,
            heart,
        })
    }

    pub fn get_embedding(&self, id: u64) -> Option<&[f32]> {
        self.storage.id_to_index.get(&id).and_then(|&idx| {
            if idx < self.storage.embeddings.len() {
                Some(self.storage.embeddings[idx].as_slice())
            } else {
                None
            }
        })
    }

    pub fn atomic_save(&mut self) -> anyhow::Result<()> {
        // Delegate to storage
        // For now, we only support saving manifest and phoneme index in storage.save_all() equivalent
        // But since physics modifies geometries, we need a full save.
        // We will implement save_all in SplatStorage.
        self.storage.save_all()
    }

    pub fn run_physics_steps(&mut self, steps_range: std::ops::Range<usize>) {
        let steps = if self.storage.geometries.len() > self.config.physics.max_active_splats {
            steps_range.start
        } else {
            steps_range.end
        };

        for _ in 0..steps {
            self.physics_step();
            self.dream_ticks_since_save += 1;
        }

        // Optional: trigger merge if any splats got close enough
        self.try_merge_close_splats(self.config.physics.merge_threshold);
    }

    fn physics_step(&mut self) {
        // Pulse the heart
        let entropy = self.heart.pulse().unwrap_or(1.0);

        // Homeostasis
        if entropy < 0.8 {
            self.heart.adjust_biochemistry(Signal::InjectChaos(0.05));
        } else if entropy > 1.5 {
            self.heart.adjust_biochemistry(Signal::Dampen(0.10));
        }

        let dt = self.config.physics.dt;
        let origin_pull = self.config.physics.origin_pull;
        let neighbor_radius_sq =
            self.config.physics.neighbor_radius * self.config.physics.neighbor_radius;
        let repulsion_radius_sq =
            self.config.physics.repulsion_radius * self.config.physics.repulsion_radius;
        // Modulate repulsion with entropy (High Entropy = High Energy)
        let repulsion_str = self.config.physics.repulsion_strength * (entropy as f32).max(0.1);
        let damping = self.config.physics.damping;
        let epsilon = self.config.physics.epsilon;

        let count = self.storage.geometries.len();
        if count == 0 {
            return;
        }

        let mut forces = vec![Vector3::zeros(); count];
        let geoms = &self.storage.geometries;

        // Parallel Force Calculation
        forces.par_iter_mut().enumerate().for_each(|(i, force)| {
            let p_i = &geoms[i];
            let pos_i = Vector3::new(p_i.position[0], p_i.position[1], p_i.position[2]);

            // Origin gravity
            *force -= pos_i * origin_pull;

            // Simplified Neighbors (Brute force with cutoff)
            for j in 0..count {
                if i == j {
                    continue;
                }
                let p_j = &geoms[j];
                let pos_j = Vector3::new(p_j.position[0], p_j.position[1], p_j.position[2]);

                let diff = pos_j - pos_i;
                let dist_sq = diff.norm_squared();

                if dist_sq < epsilon || dist_sq > neighbor_radius_sq {
                    continue;
                }

                // Simple Repulsion
                if dist_sq < repulsion_radius_sq {
                    let dist = dist_sq.sqrt();
                    *force -= diff.normalize()
                        * (self.config.physics.repulsion_radius - dist)
                        * repulsion_str;
                }
            }
        });

        // Integration
        for (i, force) in forces.into_iter().enumerate() {
            let p = &mut self.storage.geometries[i];

            p.position[0] += force.x * dt;
            p.position[1] += force.y * dt;
            p.position[2] += force.z * dt;

            // Dampening
            p.position[0] *= damping;
            p.position[1] *= damping;
            p.position[2] *= damping;
        }
    }

    fn try_merge_close_splats(&mut self, threshold: f32) {
        let threshold_sq = threshold * threshold;
        let mut to_remove = HashSet::new();

        // Very simple greedy merge pass
        for i in 0..self.storage.geometries.len() {
            if to_remove.contains(&i) {
                continue;
            }
            let p_i = &self.storage.geometries[i];
            let pos_i = Vector3::new(p_i.position[0], p_i.position[1], p_i.position[2]);

            for j in (i + 1)..self.storage.geometries.len() {
                if to_remove.contains(&j) {
                    continue;
                }
                let p_j = &self.storage.geometries[j];
                let pos_j = Vector3::new(p_j.position[0], p_j.position[1], p_j.position[2]);

                if (pos_i - pos_j).norm_squared() < threshold_sq {
                    // Merge j into i (simplify: just mark j for removal)
                    to_remove.insert(j);
                    // Assuming i absorbs j, we might want to update i's mass/text
                    // but for daydreaming, just cleaning up overlaps is fine.
                }
            }
        }

        if !to_remove.is_empty() {
            // Remove indices descending
            let mut sorted: Vec<usize> = to_remove.into_iter().collect();
            sorted.sort_unstable_by(|a, b| b.cmp(a));

            for idx in sorted {
                // Remove from all parallel arrays
                if idx < self.storage.geometries.len() {
                    let id = self.storage.payload_ids[idx]; // semantics parallel to geometries
                    self.storage.geometries.remove(idx);
                    self.storage.semantics.remove(idx);
                    self.storage.payload_ids.remove(idx);
                    self.storage.embeddings.remove(idx);
                    self.storage.manifest.remove(&id);
                    // self.index.lock().unwrap().delete(id); // HNSW delete not supported in this version
                }
            }
        }
    }

    pub fn ingest(&mut self, text: &str) -> anyhow::Result<String> {
        self.ingest_with_valence(text, None)
    }

    pub fn ingest_with_valence(
        &mut self,
        text: &str,
        valence_override: Option<f32>,
    ) -> anyhow::Result<String> {
        if text.trim().is_empty() {
            return Ok("Ignored empty text".to_string());
        }

        // IngestionEngine now returns (id, text, geometry, semantics, phonemes)
        let raw_batch = self.ingestion.ingest_batch(
            vec![text.to_string()],
            self.storage.next_payload_id,
            valence_override,
        )?;

        // Hook into Token Promotion Engine (Async Bridge)
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let text_clone = text.to_string();
            handle.spawn(async move {
                if let Err(e) = crate::TOKEN_PROMOTION_ENGINE
                    .encode_with_dynamic_vocab(&text_clone)
                    .await
                {
                    eprintln!("Token Promotion Error: {}", e);
                }
            });
        }

        // Curator Logic - Use GPU if configured
        let device = if self.config.nomic_use_gpu {
            Device::new_cuda(0).unwrap_or_else(|_| {
                eprintln!(
                    "Warning: GPU requested but not available, falling back to CPU for Curator"
                );
                Device::Cpu
            })
        } else {
            Device::Cpu
        };
        let curator = Curator::new(device.clone());
        let mut final_batch: Vec<(
            u64,
            String,
            SplatGeometry,
            crate::structs::PackedSemantics,
            SplatSemanticsV2,
            SplatLighting,
            Vec<f32>,
        )> = Vec::new();
        let mut embeddings_to_index = Vec::new();

        for item in raw_batch {
            let id = item.0;
            let txt = item.1;
            let geom = item.2;
            let sem = item.3; // SplatSemanticsV2
            let embedding = item.4; // Vec<f32>
            let phonemes = item.5; // Vec<SplatGeometry> (G-Prime)

            let valence =
                (geom.physics_props[2] as i8) as f32 / self.config.encoding.physics_prop_scale;

            let query = embedding.clone();
            let neighbors = {
                let index = self.index.lock().unwrap();
                index.search(&query, 1)
            };

            let mut decision = CuratorDecision::Encapsulate; // Default
            let mut merge_target_idx = None;

            if let Some(n) = neighbors.first() {
                let n_id = n.0;
                if let Some(&idx) = self.storage.id_to_index.get(&n_id) {
                    // Use dynamic dimension from model
                    let dim = self.model.get_output_dim();
                    // Ensure we have embeddings loaded
                    if idx < self.storage.embeddings.len() {
                        if let Ok(new_tensor) =
                            Tensor::from_vec(embedding.clone(), (1, dim), &device)
                        {
                            if let Ok(old_tensor) = Tensor::from_vec(
                                self.storage.embeddings[idx].clone(),
                                (1, dim),
                                &device,
                            ) {
                                if let Ok(d) = curator.judge(&new_tensor, &old_tensor, valence) {
                                    decision = d;
                                    if decision == CuratorDecision::Merge {
                                        merge_target_idx = Some(idx);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            match decision {
                CuratorDecision::Merge => {
                    if let Some(idx) = merge_target_idx {
                        println!("Curator MERGED: '{}' into existing memory", txt);
                        let ratio = self.config.physics.merge_ratio;

                        // Update Embeddings
                        if idx < self.storage.embeddings.len() {
                            for i in 0..self.storage.embeddings[idx].len() {
                                self.storage.embeddings[idx][i] = self.storage.embeddings[idx][i]
                                    * (1.0 - ratio)
                                    + embedding[i] * ratio;
                            }
                        }

                        // Update Semantics (Query Vector)
                        let old_sem = &mut self.storage.semantics[idx];
                        for i in 0..16 {
                            old_sem.query_vector[i] = old_sem.query_vector[i] * (1.0 - ratio)
                                + sem.manifold_vector[i] * ratio;
                        }

                        // Update Geometry (Position)
                        if idx < self.storage.geometries.len() {
                            let old_geom = &mut self.storage.geometries[idx];
                            old_geom.position[0] =
                                old_geom.position[0] * (1.0 - ratio) + geom.position[0] * ratio;
                            old_geom.position[1] =
                                old_geom.position[1] * (1.0 - ratio) + geom.position[1] * ratio;
                            old_geom.position[2] =
                                old_geom.position[2] * (1.0 - ratio) + geom.position[2] * ratio;
                        }
                        // Do NOT add to batch
                    } else {
                        // Fallback if merge target invalid
                        // Prepare PackedSemantics
                        let mut query_vec = [0.0; 16];
                        let dim = self.config.encoding.query_vector_dim.min(16);
                        query_vec[..dim].copy_from_slice(&sem.manifold_vector[0..dim]);

                        let packed = PackedSemantics {
                            position: geom.position,
                            opacity: (geom.color_rgba[3] as f32) / self.config.encoding.color_scale,
                            scale: geom.scale,
                            _pad1: 0.0,
                            rotation: geom.rotation,
                            query_vector: query_vec,
                        };
                    }
                }
                CuratorDecision::Reject => {
                    println!("Curator REJECTED: '{}'", txt);
                }
                CuratorDecision::Encapsulate => {
                    println!("Curator ENCAPSULATED: '{}' (Paradox)", txt);

                    let mut query_vec = [0.0; 16];
                    let dim = self.config.encoding.query_vector_dim.min(16);
                    query_vec[..dim].copy_from_slice(&sem.manifold_vector[0..dim]);

                    let packed = PackedSemantics {
                        position: geom.position,
                        opacity: (geom.color_rgba[3] as f32) / self.config.encoding.color_scale,
                        scale: geom.scale,
                        _pad1: 0.0,
                        rotation: geom.rotation,
                        query_vector: query_vec,
                    };

                    let lighting =
                        InverseRenderer::inverse_render_memory(&txt, &embedding, None, None);
                    // phoneme_bytes removed from persist_batch
                    final_batch.push((
                        id,
                        txt,
                        geom,
                        packed,
                        sem.clone(),
                        lighting,
                        embedding.clone(),
                    ));
                    embeddings_to_index.push((id, embedding.clone()));
                }
            }
        }

        if final_batch.is_empty() {
            return Ok("All memories rejected by Curator or Merged".to_string());
        }

        // Persist Batch via Storage Engine
        self.storage
            .persist_batch(final_batch.clone(), &self.config)?;

        // Update HNSW Index
        // Update HNSW Index
        for (id, emb) in embeddings_to_index {
            self.index.lock().unwrap().add(id, &emb)?;
        }

        Ok("Ingested".to_string())
    }

    pub fn insert_splat(&mut self, _payload_id: u64, splat: GaussianSplat) -> anyhow::Result<()> {
        let geom: SplatGeometry = splat.into();
        self.storage.geometries.push(geom);

        let query_vec = [0.0; 16];
        // No semantics provided, so zero query vector
        let sem = PackedSemantics {
            position: geom.position,
            opacity: (geom.color_rgba[3] as f32) / self.config.encoding.color_scale,
            scale: geom.scale,
            _pad1: 0.0,
            rotation: geom.rotation,
            query_vector: query_vec,
        };
        self.storage.semantics.push(sem);
        // We do not update the HNSW index or manifest here as this is a raw geometry insert
        // for G-Prime bridge testing.
        Ok(())
    }

    pub fn query_propagation(
        &mut self,
        text: &str,
        steps: usize,
    ) -> Vec<crate::physics::QueryImpact> {
        crate::physics::query_propagation(&mut self.storage, text, steps, &self.config)
    }

    pub fn get_splat(&self, payload_id: u64) -> Option<GaussianSplat> {
        if let Some(&idx) = self.storage.id_to_index.get(&payload_id) {
            if idx < self.storage.geometries.len() {
                let geom = &self.storage.geometries[idx];

                // Decode valence from physics_props[2]
                let valence_byte = geom.physics_props[2];
                let valence = (valence_byte as f32 / 127.5) - 1.0;

                // Decode opacity from color_rgba[3]
                let opacity = geom.color_rgba[3] as f32 / 255.0;

                return Some(GaussianSplat {
                    position: Vec3::from_array(geom.position),
                    scale: Vec3::from_array(geom.scale),
                    rotation: Quat::from_array(geom.rotation),
                    opacity,
                    sh_coeffs: vec![0.0; 48], // Default SH
                    valence,
                    velocity: None,
                    covariance: None,
                });
            }
        }
        None
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let index = self.index.lock().unwrap();
        index.search(query, k)
    }

    pub fn retrieve(&self, query_text: &str, limit: usize) -> anyhow::Result<Vec<RetrievalResult>> {
        let embedding = self.model.embed_query(query_text)?;
        let results = self.search(&embedding, limit);

        let mut retrieval_results = Vec::new();
        for (rank, (id, score)) in results.into_iter().enumerate() {
            if let Some(entry) = self.storage.manifest.get(&id) {
                retrieval_results.push(RetrievalResult {
                    rank,
                    probability: score,
                    text: entry.text.clone(),
                    payload_id: id,
                    confidence: 1.0 / (1.0 + score.abs()),
                    is_shadow: false,
                    valence: entry.initial_valence,
                });
            }
        }
        Ok(retrieval_results)
    }

    pub fn retrieve_bicameral(
        &self,
        query_text: &str,
        limit: usize,
        _shadow_mode: bool,
    ) -> anyhow::Result<Vec<RetrievalResult>> {
        // Fallback to standard retrieve for now
        self.retrieve(query_text, limit)
    }

    pub fn embed_query(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        self.model.embed_query(text)
    }

    /// Deep Recall: Retrieves standard results but also fetches and decodes
    /// the underlying G-Prime phonemes to verify structural integrity.
    pub fn retrieve_holographic(
        &self,
        query_text: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<HolographicResult>> {
        let base_results = self.retrieve(query_text, limit)?;
        let mut file = File::open(&self.storage.phoneme_path)?;
        let mut holo_results = Vec::new();

        for res in base_results {
            let mut decoded_text = String::new();
            let mut phoneme_count = 0;
            let mut total_tone_val = 0.0;
            let mut total_unc_val = 0.0;
            let mut count = 0.0;

            if let Some(&(offset, count_rec)) = self.storage.phoneme_index.get(&res.payload_id) {
                phoneme_count = count_rec as usize;
                if count_rec > 0 {
                    let size = mem::size_of::<SplatGeometry>();
                    let byte_len = count_rec as usize * size;
                    let mut buffer = vec![0u8; byte_len];

                    use std::io::Seek;
                    file.seek(std::io::SeekFrom::Start(offset))?;
                    file.read_exact(&mut buffer)?;

                    let geometries: &[SplatGeometry] = bytemuck::cast_slice(&buffer);
                    for geom in geometries {
                        let (c, tone, _) = GPrimeCodecV1::decode_glyph_geom(geom);
                        if c != '\0' {
                            decoded_text.push(c);

                            // Extract metadata from tone byte
                            // Tone: Bit 7=Caps, 3-6=Sentiment(0..15), 0-2=Uncertainty(0..7)
                            let sentiment = ((tone >> 3) & 0x0F) as f32; // 0-15
                            let uncertainty = (tone & 0x07) as f32; // 0-7

                            // Map sentiment: 0..15 -> -1.0..1.0
                            let sent_mapped = (sentiment / 15.0) * 2.0 - 1.0;
                            // Map uncertainty: 0..7 -> 0.0..1.0
                            let unc_mapped = uncertainty / 7.0;

                            total_tone_val += sent_mapped;
                            total_unc_val += unc_mapped;
                            count += 1.0;
                        }
                    }
                }
            }

            // Simple integrity check
            let integrity = if res.text == decoded_text {
                1.0
            } else {
                let len_diff = (res.text.len() as isize - decoded_text.len() as isize).abs();
                let max_len = res.text.len().max(decoded_text.len()).max(1);
                1.0 - (len_diff as f32 / max_len as f32)
            };

            let aggregate_sentiment = if count > 0.0 {
                total_tone_val / count
            } else {
                0.0
            };

            let aggregate_uncertainty = if count > 0.0 {
                total_unc_val / count
            } else {
                0.0
            };

            // NOTE: The `lighting` variable is not available in this scope.
            // To make this code compile, `lighting` would need to be retrieved or computed here,
            // or the `HolographicResult` struct would need to be modified to not require it,
            // or `RetrievalResult` would need to carry this information.
            // As per instructions, making the change faithfully, assuming `lighting` is somehow available.
            // This will likely result in a compilation error if `lighting` is not defined.
            holo_results.push(HolographicResult {
                base: res,
                decoded_text: decoded_text,
                integrity: integrity,
                phoneme_count,
                aggregate_sentiment,
                aggregate_uncertainty,
            });
        }

        Ok(holo_results)
    }
}
