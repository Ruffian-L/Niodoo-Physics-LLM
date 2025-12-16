use crate::config::SplatMemoryConfig;
use crate::embeddings::EmbeddingModel;
use crate::genesis::semantics::compute_zlib_entropy;
use crate::indexing::text_index::TantivyIndex;
use crate::physics::gaussian::SemanticGaussian;
use crate::storage::SplatBlobStore;
use crate::storage::TopologicalMemoryStore;
use nalgebra::{DMatrix, DVector};
use std::cmp::Ordering;
use std::collections::HashMap;

#[derive(Debug)]
pub struct ScoredMemory {
    pub id: u64,
    pub score: f32,
    pub source: String, // "Grip" (Keyword) or "Brain" (Vector)
    pub radiance: f32,  // The topological/emotional weight
}

/// HybridRetriever now operates on references to allow shared ownership in AppState
pub struct HybridRetriever<'a, B: SplatBlobStore> {
    grip: &'a TantivyIndex,
    brain: &'a TopologicalMemoryStore<B>,
    embedding_model: &'a EmbeddingModel,
    config: &'a SplatMemoryConfig,
}

impl<'a, B: SplatBlobStore> HybridRetriever<'a, B> {
    pub fn new(
        grip: &'a TantivyIndex,
        brain: &'a TopologicalMemoryStore<B>,
        embedding_model: &'a EmbeddingModel,
        config: &'a SplatMemoryConfig,
    ) -> Self {
        Self {
            grip,
            brain,
            embedding_model,
            config,
        }
    }

    /// The "God Protocol" Search (Genesis Physics)
    pub fn search(&self, query: &str, limit: usize) -> Vec<ScoredMemory> {
        // 1. Grip (Keyword Search) - Fast filter
        let keyword_hits = self.grip.search(query, limit * 2).unwrap_or_default();

        // 2. Brain (Physics Scan)
        // We perform a full O(N) physics scan because we need density_bonus
        // which HNSW cannot provide. For personal memory (<50k), this is instant.

        let query_vec = self.embedding_model.embed(query).unwrap_or_default();

        if query_vec.is_empty() {
            return Vec::new();
        }

        // --- 1. WHITENING (Breaking the Cone of Silence) ---
        // Compute global mean of the memory bank
        // In production, cache this. For now, compute O(N).
        let mut global_mean = DVector::zeros(crate::constants::EMBED_DIM);
        let mut count = 0.0;

        // Fast pass to sum vectors (use only first EMBED_DIM dimensions)
        for (_, mem) in self.brain.entries() {
            // We assume embedding is stored as Vec<f16> in StoredMemory
            for (i, val) in mem
                .embedding
                .iter()
                .take(crate::constants::EMBED_DIM)
                .enumerate()
            {
                global_mean[i] += val.to_f32();
            }
            count += 1.0;
        }

        if count > 0.0 {
            global_mean /= count;
        }

        // Whiten the Query
        let q_raw = DVector::from_vec(query_vec.clone());
        let q_centered = &q_raw - &global_mean;
        // Re-normalize after centering (Crucial!)
        let q_mean = if q_centered.norm() > 1e-6 {
            q_centered.normalize()
        } else {
            DVector::zeros(q_centered.len())
        };
        let q_u = q_mean.clone(); // Principal axis

        // Shape Query Gaussian
        let query_gauss = SemanticGaussian::new(
            0,
            q_mean,
            q_u,
            0.8,
            2.0,
            DMatrix::zeros(2, crate::constants::EMBED_DIM),
            0.5,
            0.0,        // Valence
            Vec::new(), // Discrete Codes
            query.to_string(),
        );

        let mut physics_results: Vec<(u64, f32)> = Vec::with_capacity(self.brain.len());

        for (id, memory) in self.brain.entries() {
            // --- 2. RE-INFLATION WITH WHITENING (use only first EMBED_DIM dimensions) ---
            let mem_f32: Vec<f32> = memory
                .embedding
                .iter()
                .take(crate::constants::EMBED_DIM)
                .map(|x| x.to_f32())
                .collect();
            let mem_raw = DVector::from_vec(mem_f32);
            let mem_centered = &mem_raw - &global_mean;
            let mem_vec = if mem_centered.norm() > 1e-6 {
                mem_centered.normalize() // Whiteness applied
            } else {
                DVector::zeros(mem_centered.len())
            };
            let mem_u = mem_vec.clone();

            // Recalculate entropy with the new Length Correction
            // Use TEXT, not labels!
            let entropy = compute_zlib_entropy(memory.text.as_bytes()).unwrap_or(0.5);

            // Shape Logic (THE ONE TRUE LAW - REFINED)
            // Low Entropy = Needle
            // High Entropy = Cloud
            // Adjusted for Symbol Density (Code vs Prose)

            let symbol_density = memory
                .text
                .chars()
                .filter(|c| !c.is_alphanumeric() && !c.is_whitespace())
                .count() as f32
                / (memory.text.chars().count().max(1) as f32);

            let threshold = if symbol_density > 0.10 { 1.30 } else { 1.05 };
            let is_needle = entropy < threshold;

            let anisotropy = if is_needle {
                // Scaling: Lower entropy = Sharper needle
                (20.0 + (threshold - entropy).max(0.0) * 100.0).min(50.0)
            } else {
                1.0
            };

            let sigma_iso = if is_needle { 0.35 } else { 1.5 };

            let mem_gauss = SemanticGaussian::new(
                *id,
                mem_vec,
                mem_u,
                sigma_iso,
                anisotropy,
                DMatrix::zeros(2, crate::constants::EMBED_DIM),
                entropy,
                0.0,        // Valence
                Vec::new(), // Discrete Codes
                "".into(),
            );

            // Physics Distance
            let dist_sq = mem_gauss.mahalanobis_rank1(&query_gauss);
            let similarity = (-dist_sq).exp();

            // --- 3. SIGMOID RADIANCE (Gravity Limit) ---
            // Current Anisotropy ~1.0 to 50.0
            // Tanh(aniso / 20.0) ranges from 0.05 to 0.98
            // Max Boost = 1.0 + 3.0 * 1.0 = 4.0x
            // No more 200x explosions.

            let radiance_boost = 1.0 + 3.0 * (anisotropy / 20.0).tanh();

            // Density Bonus (still useful, but keep it sane)
            // let density = mem_gauss.density_bonus().clamp(1.0, 2.0);
            let density = 1.0; // Placeholder

            let physics_score = similarity * density * radiance_boost;

            if physics_score > 0.001 {
                physics_results.push((*id, physics_score));
            }
        }

        // 3. The Fusion (Reciprocal Rank Fusion)
        let mut scores: HashMap<u64, f32> = HashMap::new();
        let k = 60.0;

        // Process Keyword Hits
        for (rank, (id, _)) in keyword_hits.iter().enumerate() {
            let rrf = 1.0 / (k + rank as f32 + 1.0);
            *scores.entry(*id).or_insert(0.0) += rrf * self.config.alpha_keyword;
        }

        // Process Physics Hits
        // Sort first by physics score
        physics_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        for (rank, (id, raw_score)) in physics_results.iter().enumerate() {
            let rrf = 1.0 / (k + rank as f32 + 1.0);
            // We multiply by the raw physics score to keep the magnitude relevance
            let weighted_score = rrf * self.config.beta_semantic * raw_score.clamp(0.5, 2.0);
            *scores.entry(*id).or_insert(0.0) += weighted_score;
        }

        let mut final_results: Vec<ScoredMemory> = scores
            .into_iter()
            .map(|(id, score)| {
                let radiance = self.brain.get_radiance(id);
                ScoredMemory {
                    id,
                    score: score * (1.0 + radiance.clamp(-0.5, 2.0)), // Radiance Boost
                    source: "Hybrid-Genesis".to_string(),
                    radiance,
                }
            })
            .collect();

        final_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        final_results.into_iter().take(limit).collect()
    }

    fn fallback_gaussian(&self, embedding: Vec<f32>) -> SemanticGaussian {
        let dim = embedding.len();
        SemanticGaussian {
            id: 0, // Placeholder
            mean: DVector::from_vec(embedding),
            u_vec: DVector::zeros(dim), // Zero vector for direction means no anisotropy/needle
            sigma_iso: 0.5,
            anisotropy: 1.0,
            valence: 0.0,
            sh_coeffs: DMatrix::zeros(3, dim),
            grad_accum: 0.0,
            entropy: 0.5, // Default entropy
            discrete_codes: Vec::new(),
            birth: 0.0,
            text: String::new(),
        }
    }
}
