use crate::config::SplatMemoryConfig;
use crate::embeddings::EmbeddingModel;
use crate::indexing::TantivyIndex;
use crate::manifold::ManifoldProjector;
use crate::physics::RadianceField;
use crate::storage::engine::SplatStorage;
use crate::utils::fidelity::robust_dot;
use anyhow::Result;
use cathedral::primitive::PhysicalGaussian;
use nalgebra::{Matrix3, Quaternion, UnitQuaternion, Vector3};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct RetrievalParams {
    pub weight_cosine: f32,
    pub weight_bm25: f32,
    pub weight_radiance: f32,
    pub weight_physics: f32,
    pub diversity: bool,
    pub top_k: usize,
    pub shadow_mode: bool,
    pub use_physics: bool,
}

impl Default for RetrievalParams {
    fn default() -> Self {
        Self {
            weight_cosine: 10.0,
            weight_bm25: 1.0,
            weight_radiance: 5.0,
            weight_physics: 2.0,
            diversity: false,
            top_k: 10,
            shadow_mode: false,
            use_physics: true,
        }
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct RetrievalResult {
    pub rank: usize,
    pub final_score: f32,
    pub rrf_score: f32,
    pub radiance: f32,
    pub cosine: f32,
    pub bm25_score: f32,
    pub distance: f32,
    pub text: String,
    pub payload_id: u64,
    pub valence: i8,
    pub is_shadow: bool,
}

// Helper to compute inverse covariance from scale and rotation
fn compute_cov_inv(scale: [f32; 3], rotation: [f32; 4]) -> [f32; 9] {
    let s = Vector3::new(
        scale[0].max(0.001),
        scale[1].max(0.001),
        scale[2].max(0.001),
    );
    let q = UnitQuaternion::from_quaternion(Quaternion::new(
        rotation[3],
        rotation[0],
        rotation[1],
        rotation[2],
    ));
    let r = q.to_rotation_matrix();

    // Sigma = R * S * S * R^T
    // Sigma^-1 = R * S^-2 * R^T

    let s_inv_sq = Matrix3::new(
        1.0 / (s.x * s.x),
        0.0,
        0.0,
        0.0,
        1.0 / (s.y * s.y),
        0.0,
        0.0,
        0.0,
        1.0 / (s.z * s.z),
    );

    let cov_inv = r.matrix() * s_inv_sq * r.matrix().transpose();

    [
        cov_inv[(0, 0)],
        cov_inv[(0, 1)],
        cov_inv[(0, 2)],
        cov_inv[(1, 0)],
        cov_inv[(1, 1)],
        cov_inv[(1, 2)],
        cov_inv[(2, 0)],
        cov_inv[(2, 1)],
        cov_inv[(2, 2)],
    ]
}

pub struct AdvancedRetriever<'a> {
    pub config: &'a SplatMemoryConfig,
    pub model: &'a EmbeddingModel,
    pub projector: &'a ManifoldProjector,
    pub index: &'a TantivyIndex,
}

impl<'a> AdvancedRetriever<'a> {
    pub fn new(
        config: &'a SplatMemoryConfig,
        model: &'a EmbeddingModel,
        projector: &'a ManifoldProjector,
        index: &'a TantivyIndex,
    ) -> Self {
        Self {
            config,
            model,
            projector,
            index,
        }
    }

    pub fn search(
        &self,
        query: &str,
        storage: &SplatStorage,
        params: RetrievalParams,
    ) -> Result<Vec<RetrievalResult>> {
        // 1. Embed Query
        let mut query_vec = self.model.embed(query)?;

        // Normalize
        let query_norm: f32 = query_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if query_norm > 1e-6 {
            for x in query_vec.iter_mut() {
                *x /= query_norm;
            }
        }

        // 2. Keyword Search (The Grip)
        // Sanitize query
        let safe_query = query
            .chars()
            .map(|c| {
                if "+-&|!(){}[]^\"~*?:\\".contains(c) {
                    ' '
                } else {
                    c
                }
            })
            .collect::<String>();

        let or_query = safe_query
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" OR ");

        let keyword_hits = if !or_query.trim().is_empty() {
            self.index.search(&or_query, 2000)?
        } else {
            Vec::new()
        };

        // 3. Vector Search (The Brain)
        // We need to access semantics from storage
        let semantics = &storage.semantics;
        // Handle missing embeddings gracefully?
        // If storage.embeddings is empty, we can't do vector search.
        // But we can use semantics.query_vector (16D) if available?
        // Or just skip vector search.

        let mut vector_hits: Vec<(u64, f32, usize)> = if !storage.embeddings.is_empty() {
            storage
                .embeddings
                .par_iter()
                .enumerate()
                .map(|(i, emb)| {
                    let dot = robust_dot(emb, &query_vec);
                    (storage.payload_ids[i], dot, i)
                })
                .collect()
        } else {
            Vec::new()
        };

        vector_hits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_vector_hits = vector_hits.iter().take(2000).collect::<Vec<_>>();

        // 4. RRF Fusion
        let k = 60.0;
        let mut rrf_scores: HashMap<u64, f32> = HashMap::new();
        let mut cosine_map: HashMap<u64, f32> = HashMap::new();
        let mut bm25_map: HashMap<u64, f32> = HashMap::new();

        for (rank, (id, score)) in keyword_hits.iter().enumerate() {
            let rrf = 1.0 / (k + rank as f32 + 1.0);
            *rrf_scores.entry(*id).or_insert(0.0) += rrf * params.weight_bm25;
            bm25_map.insert(*id, *score);
        }

        for (rank, (id, score, _idx)) in top_vector_hits.iter().enumerate() {
            let rrf = 1.0 / (k + rank as f32 + 1.0);
            *rrf_scores.entry(*id).or_insert(0.0) += rrf * params.weight_cosine;
            cosine_map.insert(*id, *score);
        }

        // 5. Radiance & Rescoring
        let query_tensor = candle_core::Tensor::from_vec(
            query_vec.clone(),
            (1, query_vec.len()),
            &candle_core::Device::Cpu,
        )
        .unwrap_or(
            candle_core::Tensor::zeros((1, 64), candle_core::DType::F32, &candle_core::Device::Cpu)
                .unwrap(),
        );
        let query_manifold_vector = match self.projector.project(&query_tensor) {
            Ok(geom) => geom
                .mu
                .flatten_all()
                .unwrap_or(
                    candle_core::Tensor::zeros(
                        (64,),
                        candle_core::DType::F32,
                        &candle_core::Device::Cpu,
                    )
                    .unwrap(),
                )
                .to_vec1::<f32>()
                .unwrap_or(vec![0.0; 64]),
            Err(_) => vec![0.0; 64],
        };

        // Use first 3 dims of manifold vector as query position for physics
        let query_pos = [
            query_manifold_vector.get(0).cloned().unwrap_or(0.0),
            query_manifold_vector.get(1).cloned().unwrap_or(0.0),
            query_manifold_vector.get(2).cloned().unwrap_or(0.0),
        ];

        let mut final_results = Vec::new();
        let mut sorted_rrf: Vec<_> = rrf_scores.iter().collect();
        sorted_rrf.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (id, rrf_score) in sorted_rrf {
            if let Some(&idx) = storage.id_to_index.get(id) {
                if idx < storage.geometries.len() {
                    let g = &storage.geometries[idx];
                    let s = if idx < storage.semantics.len() {
                        &storage.semantics[idx]
                    } else {
                        &storage.semantics[0]
                    }; // Fallback?

                    // Radiance (Legacy)
                    let radiance = RadianceField::compute(
                        g,
                        s,
                        &query_manifold_vector,
                        self.config,
                        params.shadow_mode,
                    );

                    // Physics (Manifold Similarity)
                    let mut physics_score = 0.0;
                    if params.use_physics {
                        // Get document's manifold vector from semantics
                        let doc_manifold = if idx < storage.semantics.len() {
                            let s = &storage.semantics[idx];
                            // Use query_vector which is 16D manifold
                            let mut full_vec = vec![0.0f32; 64];
                            for i in 0..16.min(64) {
                                full_vec[i] = s.query_vector[i];
                            }
                            full_vec
                        } else {
                            vec![0.0f32; 64]
                        };

                        // Compute manifold similarity (cosine in manifold space)
                        let dot: f32 = query_manifold_vector
                            .iter()
                            .zip(doc_manifold.iter())
                            .map(|(a, b)| a * b)
                            .sum();
                        let q_norm: f32 = query_manifold_vector
                            .iter()
                            .map(|x| x * x)
                            .sum::<f32>()
                            .sqrt()
                            .max(1e-8);
                        let d_norm: f32 = doc_manifold
                            .iter()
                            .map(|x| x * x)
                            .sum::<f32>()
                            .sqrt()
                            .max(1e-8);
                        let manifold_sim = dot / (q_norm * d_norm);

                        // Also incorporate radiance for domain alignment
                        let radiance = if idx < storage.lighting.len() {
                            let lgt = &storage.lighting[idx];
                            let rad_mag = (lgt.idiv[0] * lgt.idiv[0]
                                + lgt.idiv[1] * lgt.idiv[1]
                                + lgt.idiv[2] * lgt.idiv[2])
                                .sqrt();
                            rad_mag
                        } else {
                            1.0
                        };

                        // Physics score combines manifold similarity with radiance
                        physics_score = manifold_sim * radiance.min(2.0);
                    }

                    let cosine = *cosine_map.get(id).unwrap_or(&0.0);
                    let bm25_raw = *bm25_map.get(id).unwrap_or(&0.0);

                    let normalized_radiance = radiance / (radiance + 1.0);
                    let cosine_norm = (cosine + 1.0) / 2.0;
                    let bm25_norm = bm25_raw / 30.0;

                    let final_score = (bm25_norm * params.weight_bm25)
                        + (cosine_norm * params.weight_cosine)
                        + (normalized_radiance * params.weight_radiance)
                        + (physics_score * params.weight_physics);

                    if let Some(entry) = storage.manifest.get(id) {
                        final_results.push(RetrievalResult {
                            rank: 0,
                            final_score,
                            rrf_score: *rrf_score,
                            radiance,
                            cosine,
                            bm25_score: bm25_raw,
                            distance: 0.0,
                            text: entry.text.clone(),
                            payload_id: *id,
                            valence: g.physics_props[2] as i8,
                            is_shadow: params.shadow_mode,
                        });
                    }
                }
            }
        }

        final_results.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // 6. Diversity (MMR)
        if params.diversity {
            let k_diversity = params.top_k;
            let top_n_candidates = 50.min(final_results.len());

            if top_n_candidates > 0 {
                let mut selected = Vec::with_capacity(k_diversity);
                let mut candidate_indices: Vec<usize> = (0..top_n_candidates).collect();

                selected.push(final_results[0].clone());
                candidate_indices.remove(0);

                let get_vec = |id: u64| -> Vec<f32> {
                    if let Some(&idx) = storage.id_to_index.get(&id) {
                        if idx < storage.semantics.len() {
                            let v = &storage.semantics[idx].query_vector;
                            return v.to_vec();
                        }
                    }
                    vec![0.0; 16]
                };

                let mut selected_vecs = Vec::new();
                selected_vecs.push(get_vec(selected[0].payload_id));

                while selected.len() < k_diversity && !candidate_indices.is_empty() {
                    let mut best_mmr = -f32::INFINITY;
                    let mut best_cand_idx_in_indices = 0;
                    let lambda = 0.5;

                    for (i, &cand_idx) in candidate_indices.iter().enumerate() {
                        let cand = &final_results[cand_idx];
                        let cand_vec = get_vec(cand.payload_id);

                        let mut max_sim = -1.0;
                        for sel_vec in &selected_vecs {
                            let dot: f32 = cand_vec
                                .iter()
                                .zip(sel_vec.iter())
                                .map(|(a, b)| a * b)
                                .sum();
                            if dot > max_sim {
                                max_sim = dot;
                            }
                        }

                        let mmr = lambda * cand.final_score - (1.0 - lambda) * max_sim;
                        if mmr > best_mmr {
                            best_mmr = mmr;
                            best_cand_idx_in_indices = i;
                        }
                    }

                    let best_real_idx = candidate_indices[best_cand_idx_in_indices];
                    let best_cand = final_results[best_real_idx].clone();
                    selected_vecs.push(get_vec(best_cand.payload_id));
                    selected.push(best_cand);
                    candidate_indices.remove(best_cand_idx_in_indices);
                }
                final_results = selected;
            }
        }

        // Fix ranks
        for (i, res) in final_results.iter_mut().enumerate() {
            res.rank = i + 1;
        }

        Ok(final_results.into_iter().take(params.top_k).collect())
    }
}
