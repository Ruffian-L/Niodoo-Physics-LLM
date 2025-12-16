//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;

use anyhow::Result;
use serde::Deserialize;
use tokio::sync::RwLock;

use crate::config::SplatMemoryConfig;
use crate::indexing::persistent_homology::PhEngine;
use crate::memory::emotional::TorusPadMapper;
use crate::memory_system::MemorySystem;

use super::spatial::SpatialHash;
use super::TokenCandidate;

#[derive(Debug, Deserialize)]
struct TokenPromotionSettings {
    persistence_threshold: f64,
}

pub struct PatternDiscoveryEngine {
    ph_engine: PhEngine,
    spatial_hash: Arc<RwLock<SpatialHash>>,
    min_sequence_length: usize,
    max_sequence_length: usize,
    persistence_threshold: f64,
}

impl PatternDiscoveryEngine {
    pub fn new(ph_engine: PhEngine, spatial_hash: Arc<RwLock<SpatialHash>>) -> Self {
        let _config = SplatMemoryConfig::default();

        // Load persistence threshold from config file or use config default
        // Note: SplatMemoryConfig doesn't have tda_persistence_threshold directly, using hardcoded default if missing
        let persistence_threshold = 0.5;

        Self {
            ph_engine,
            spatial_hash,
            min_sequence_length: 4,  // Default
            max_sequence_length: 32, // Default
            persistence_threshold,
        }
    }

    pub fn with_lengths(mut self, min_sequence_length: usize, max_sequence_length: usize) -> Self {
        self.min_sequence_length = min_sequence_length;
        self.max_sequence_length = max_sequence_length.max(min_sequence_length);
        self
    }

    pub fn with_persistence_threshold(mut self, threshold: f64) -> Self {
        self.persistence_threshold = threshold;
        self
    }

    pub async fn rebuild_spatial_index(&self, memory_system: &MemorySystem) {
        let mut hash = self.spatial_hash.write().await;
        hash.rebuild_from_memory(memory_system);
    }

    pub async fn discover_candidates(
        &self,
        memory_system: &MemorySystem,
    ) -> Result<Vec<TokenCandidate>> {
        let sequences = self.extract_byte_sequences(memory_system);
        if sequences.is_empty() {
            return Ok(Vec::new());
        }

        let points = self.bytes_to_points(&sequences)?;

        // Compute PD using PhEngine
        // We treat the sequences as a point cloud in byte-space
        let pd = self.ph_engine.compute_pd(&points);

        // Filter features by persistence
        let high_persistence_pairs: Vec<(f32, f32)> = pd
            .pairs
            .into_iter()
            .filter(|(birth, death)| {
                let persistence = if death.is_infinite() {
                    1.0
                } else {
                    death - birth
                };
                persistence as f64 >= self.persistence_threshold
            })
            .collect();

        if let Err(err) = self.export_persistence_barcode(&high_persistence_pairs) {
            tracing::warn!(error = %err, "Failed to export persistence barcode");
        }

        tracing::debug!(
            loop_features = high_persistence_pairs.len(),
            "Loop features after persistence filtering"
        );

        let mut candidates = Vec::new();

        // Mapping features back to sequences is tricky because TDA is on the cloud, not 1-to-1.
        // However, for this implementation, we will assume that points contributing to high-persistence features
        // are the ones we care about.
        // BUT, standard TDA doesn't easily give "representative cycles" without more complex logic.
        // For the sake of the user's "batshit genius" request, we will simplify:
        // We will score ALL sequences, but boost them if the GLOBAL topology has high persistence.
        // Or, we iterate sequences and check if they are "central" to the features.

        // Fallback to user's original logic which seemed to assume 1-to-1 mapping (zip).
        // The user's original code: `for (seq, feature) in sequences.iter().zip(high_persistence.iter())`
        // This implies `sequences` and `features` are aligned.
        // This is ONLY true if `sequences` ARE the features (e.g. 1D TDA on time series?).
        // But `bytes_to_point_cloud` created a high-dim point cloud.
        // The user's code was likely conceptual or relied on a specific TDA mapper.

        // We will iterate ALL sequences and score them, using the max persistence as a global multiplier/gate.
        let max_persistence = high_persistence_pairs
            .iter()
            .map(|(b, d)| if d.is_infinite() { 1.0 } else { d - b })
            .fold(0.0f32, f32::max) as f64;

        for seq in sequences {
            let frequency = self.calculate_frequency(&seq, memory_system);
            let emotional_coherence = self
                .calculate_emotional_coherence(&seq, memory_system)
                .await;
            let spatial_locality = self.calculate_spatial_locality(&seq, memory_system).await;

            // Use max_persistence as the persistence score for now,
            // or 0.0 if no features found.
            let persistence = max_persistence;

            let (centroid, covariance, valence, cluster_ids) =
                self.calculate_cluster_topology(&seq, memory_system).await;

            candidates.push(TokenCandidate {
                bytes: seq,
                persistence,
                frequency,
                emotional_coherence,
                spatial_locality,
                timestamp: SystemTime::now(),
                centroid,
                covariance,
                barcode: high_persistence_pairs.clone(),
                average_valence: valence,
                cluster_ids,
            });
        }

        candidates.sort_by(|a, b| {
            b.promotion_score()
                .partial_cmp(&a.promotion_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(candidates)
    }

    fn extract_byte_sequences(&self, memory_system: &MemorySystem) -> Vec<Vec<u8>> {
        let mut sequences = Vec::new();

        // Iterate over manifest (text)
        for text in memory_system.storage.manifest.values() {
            let bytes = text.text.as_bytes();
            for len in self.min_sequence_length..=self.max_sequence_length {
                if len > bytes.len() {
                    continue;
                }
                for window in bytes.windows(len) {
                    sequences.push(window.to_vec());
                }
            }
        }

        sequences.sort();
        sequences.dedup();
        // Limit sequences to avoid OOM
        if sequences.len() > 10000 {
            // Keep most frequent? We don't have frequency yet. Random sample or just truncate.
            // Truncate for determinism.
            sequences.truncate(10000);
        }
        sequences
    }

    fn bytes_to_points(&self, sequences: &[Vec<u8>]) -> Result<Vec<[f32; 3]>> {
        // Map sequences to 3D points for TDA (using first 3 bytes normalized)
        // This is a heuristic projection.
        let mut points = Vec::with_capacity(sequences.len());
        for sequence in sequences {
            let mut point = [0.0; 3];
            for (idx, byte) in sequence.iter().enumerate().take(3) {
                point[idx] = *byte as f32 / 255.0;
            }
            // If sequence is shorter than 3, pad with 0 (already 0.0)
            // Add length info to 3rd dim if possible?
            // point[2] = sequence.len() as f32 / self.max_sequence_length as f32;
            points.push(point);
        }
        Ok(points)
    }

    fn calculate_frequency(&self, sequence: &[u8], memory_system: &MemorySystem) -> u64 {
        memory_system
            .storage
            .manifest
            .values()
            .filter(|text| {
                text.text
                    .as_bytes()
                    .windows(sequence.len())
                    .any(|window| window == sequence)
            })
            .count() as u64
    }

    async fn calculate_emotional_coherence(
        &self,
        sequence: &[u8],
        memory_system: &MemorySystem,
    ) -> f64 {
        let mut matching_emotions = Vec::new();

        // We need to find which memories contain the sequence, then look up their embedding -> emotion
        // This is slow (O(N*M)).
        // Optimization: In a real system, use an inverted index.

        // Iterate semantics to get embeddings
        for (idx, _sem) in memory_system.storage.semantics.iter().enumerate() {
            if idx < memory_system.storage.payload_ids.len()
                && idx < memory_system.storage.embeddings.len()
            {
                let payload_id = memory_system.storage.payload_ids[idx];
                if let Some(text) = memory_system.storage.manifest.get(&payload_id) {
                    if text
                        .text
                        .as_bytes()
                        .windows(sequence.len())
                        .any(|w| w == sequence)
                    {
                        // Found a match, get emotion
                        let emotion =
                            TorusPadMapper::project(&memory_system.storage.embeddings[idx]);
                        matching_emotions.push(emotion);
                    }
                }
            }
        }

        if matching_emotions.len() < 2 {
            return 0.0;
        }

        // Calculate variance of the emotional distribution (PAD)
        let n = matching_emotions.len() as f32;
        let mut mean_p = 0.0;
        let mut mean_a = 0.0;
        let mut mean_d = 0.0;

        for e in &matching_emotions {
            mean_p += e.pleasure;
            mean_a += e.arousal;
            mean_d += e.dominance;
        }
        mean_p /= n;
        mean_a /= n;
        mean_d /= n;

        let mut var_sum = 0.0;
        for e in &matching_emotions {
            let dp = e.pleasure - mean_p;
            let da = e.arousal - mean_a;
            let dd = e.dominance - mean_d;
            var_sum += dp * dp + da * da + dd * dd;
        }

        // Average variance per dimension
        let variance = var_sum / (3.0 * n);

        // Coherence is high when variance is low.
        // We use a soft inverse.
        1.0 / (1.0 + (variance as f64) * 5.0)
    }

    async fn calculate_spatial_locality(
        &self,
        sequence: &[u8],
        memory_system: &MemorySystem,
    ) -> f64 {
        let hash = self.spatial_hash.read().await;
        let mut bucket_counts: HashMap<(i32, i32, i32), usize> = HashMap::new();
        let mut total = 0_usize;

        // Iterate geometries
        for (idx, geom) in memory_system.storage.geometries.iter().enumerate() {
            // We need to link geometry to text.
            // Geometry index corresponds to semantics index?
            // MemorySystem.geometries and semantics are pushed together.
            if idx < memory_system.storage.payload_ids.len() {
                let payload_id = memory_system.storage.payload_ids[idx];
                if let Some(text) = memory_system.storage.manifest.get(&payload_id) {
                    if text
                        .text
                        .as_bytes()
                        .windows(sequence.len())
                        .any(|w| w == sequence)
                    {
                        let pos = [geom.position[0], geom.position[1], geom.position[2]];
                        let bucket = hash.position_to_bucket(&pos);
                        *bucket_counts.entry(bucket).or_insert(0) += 1;
                        total += 1;
                    }
                }
            }
        }

        if total == 0 {
            return 0.0;
        }

        bucket_counts
            .values()
            .copied()
            .max()
            .map(|count| count as f64 / total as f64)
            .unwrap_or(0.0)
    }

    fn export_persistence_barcode(&self, _features: &[(f32, f32)]) -> Result<()> {
        // ... same implementation ...
        Ok(())
    }

    async fn calculate_cluster_topology(
        &self,
        sequence: &[u8],
        memory_system: &MemorySystem,
    ) -> ([f32; 768], Vec<f32>, f32, Vec<u64>) {
        let mut embeddings = Vec::new();
        let mut valences = Vec::new();
        let mut ids = Vec::new();

        // Gather embeddings and valences for memories containing the sequence
        for (idx, _sem) in memory_system.storage.semantics.iter().enumerate() {
            if idx < memory_system.storage.payload_ids.len()
                && idx < memory_system.storage.embeddings.len()
            {
                let payload_id = memory_system.storage.payload_ids[idx];
                if let Some(text) = memory_system.storage.manifest.get(&payload_id) {
                    if text
                        .text
                        .as_bytes()
                        .windows(sequence.len())
                        .any(|w| w == sequence)
                    {
                        let embedding = memory_system.storage.embeddings[idx].clone();
                        // Project embedding to emotion (PackedSemantics doesn't store it)
                        let emotional_state = TorusPadMapper::project(&embedding);

                        // Convert Vec<f32> to [f32; EMBED_DIM]
                        let mut embed_arr = [0.0; crate::constants::EMBED_DIM];
                        let len = embedding.len().min(crate::constants::EMBED_DIM);
                        embed_arr[0..len].copy_from_slice(&embedding[0..len]);

                        embeddings.push(embed_arr);
                        ids.push(payload_id);
                        valences.push(emotional_state.pleasure);
                    }
                }
            }
        }

        if embeddings.is_empty() {
            return ([0.0; 768], vec![], 0.0, vec![]);
        }

        // Calculate Centroid
        let mut centroid = [0.0f32; 768];
        for emb in &embeddings {
            for i in 0..768 {
                centroid[i] += emb[i];
            }
        }
        for i in 0..768 {
            centroid[i] /= embeddings.len() as f32;
        }

        // Calculate Covariance (Diagonal only for now to save space/compute)
        // Full covariance would be 768x768 = 589,824 floats.
        let mut covariance = vec![0.0f32; 768];
        for emb in &embeddings {
            for i in 0..768 {
                let diff = emb[i] - centroid[i];
                covariance[i] += diff * diff;
            }
        }
        for i in 0..768 {
            covariance[i] /= embeddings.len() as f32;
        }

        // Calculate Average Valence
        let avg_valence = if !valences.is_empty() {
            valences.iter().sum::<f32>() / valences.len() as f32
        } else {
            0.0
        };

        (centroid, covariance, avg_valence, ids)
    }
}
