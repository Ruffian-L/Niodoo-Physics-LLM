use std::collections::HashSet;

use anyhow::Result;

use crate::indexing::fingerprint::fingerprint_from_splat;
use crate::retrieval::{conscious_recall, RecallResult};
use crate::storage::SplatBlobStore;
use crate::storage::TopologicalMemoryStore;
use crate::tivm::SplatRagConfig;
use crate::types::SplatInput;

pub struct SequenceReconstructor {
    hidden_size: usize,
    #[allow(dead_code)]
    max_sequence_length: usize,
    // Hebbian transition matrix: from_id -> (to_id -> weight)
    transitions: std::collections::HashMap<u64, std::collections::HashMap<u64, f32>>,
    // Cache of memory vectors (simplified for standalone operation)
    memory_vectors: std::collections::HashMap<u64, Vec<f32>>,
}

impl SequenceReconstructor {
    pub fn new(hidden_size: usize, max_sequence_length: usize) -> Self {
        Self {
            hidden_size,
            max_sequence_length,
            transitions: std::collections::HashMap::new(),
            memory_vectors: std::collections::HashMap::new(),
        }
    }

    /// Learn a sequence transition
    pub fn learn_sequence(&mut self, sequence: &[u64], vectors: &[Vec<f32>]) {
        for (i, &id) in sequence.iter().enumerate() {
            if let Some(vec) = vectors.get(i) {
                self.memory_vectors.insert(id, vec.clone());
            }

            if i + 1 < sequence.len() {
                let next_id = sequence[i + 1];
                let entry = self.transitions.entry(id).or_default();
                *entry.entry(next_id).or_insert(0.0) += 1.0;
            }
        }
    }

    pub fn reconstruct(&self, memory_ids: &[u64]) -> Result<Vec<Vec<f32>>> {
        // Reconstruct vectors from IDs using cached memory
        let mut sequence = Vec::new();
        for id in memory_ids {
            if let Some(vec) = self.memory_vectors.get(id) {
                sequence.push(vec.clone());
            } else {
                // If unknown, return zero vector or simplified embedding
                // Real implementation would fetch from store
                sequence.push(vec![0.0; self.hidden_size]);
            }
        }
        Ok(sequence)
    }

    pub fn generate_next(&self, current_state: &[f32]) -> Result<Vec<f32>> {
        // Find memory with closest state
        let mut best_id = None;
        let mut max_sim = f32::NEG_INFINITY;

        for (id, vec) in &self.memory_vectors {
            let sim = self.cosine_similarity(current_state, vec);
            if sim > max_sim {
                max_sim = sim;
                best_id = Some(*id);
            }
        }

        if let Some(id) = best_id {
            // Predict next based on transitions
            if let Some(next_map) = self.transitions.get(&id) {
                // Weighted average of next states
                let mut next_state = vec![0.0; self.hidden_size];
                let mut total_weight = 0.0;

                for (&next_id, &weight) in next_map {
                    if let Some(next_vec) = self.memory_vectors.get(&next_id) {
                        for i in 0..self.hidden_size {
                            if i < next_vec.len() {
                                next_state[i] += next_vec[i] * weight;
                            }
                        }
                        total_weight += weight;
                    }
                }

                if total_weight > 0.0 {
                    for x in &mut next_state {
                        *x /= total_weight;
                    }
                    return Ok(next_state);
                }
            }
        }

        // Fallback: Identity or decay
        Ok(current_state.to_vec())
    }

    fn cosine_similarity(&self, v1: &[f32], v2: &[f32]) -> f32 {
        let dot: f32 = crate::utils::fidelity::robust_dot(v1, v2);
        let mag1: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag2: f32 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();
        if mag1 == 0.0 || mag2 == 0.0 {
            0.0
        } else {
            dot / (mag1 * mag2)
        }
    }
}

/// Iteratively recalls related memories, feeding each result back into the query generator.
/// Stops when `steps` results are collected, the recall stage yields no new IDs, or the
/// `query_gen` callback returns `None`.
pub fn recall_episode<B, F>(
    initial_cue: &SplatInput,
    steps: usize,
    store: &TopologicalMemoryStore<B>,
    config: &SplatRagConfig,
    mut query_gen: F,
) -> Result<Vec<RecallResult>>
where
    B: SplatBlobStore,
    F: FnMut(&RecallResult) -> Option<SplatInput>,
{
    if steps == 0 {
        return Ok(Vec::new());
    }

    let mut results = Vec::with_capacity(steps);
    let mut visited: HashSet<u64> = HashSet::new();
    let mut current_fp = fingerprint_from_splat(initial_cue, config);

    while results.len() < steps {
        let candidates = conscious_recall(store, &current_fp, steps)?;
        let next = candidates
            .into_iter()
            .find(|candidate| !visited.contains(&candidate.splat_id));

        let Some(selected) = next else {
            break;
        };

        visited.insert(selected.splat_id);
        current_fp = match query_gen(&selected) {
            Some(next_cue) => fingerprint_from_splat(&next_cue, config),
            None => {
                results.push(selected);
                break;
            }
        };

        results.push(selected);
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::hnsw::HnswIndex;
    use crate::tivm::SplatRagBuilder;
    use crate::types::{Mat3, Point3, Vec3};
    use crate::{SplatInput, SplatMeta};

    fn make_splat(label: &str, offset: f32) -> SplatInput {
        let mut splat = SplatInput::default();

        // Create connected component with diameter proportional to offset to vary persistence
        // cue (0): d=0.5 -> Barcode [0, 0.5], [0, inf]
        // step1 (1): d=1.0 -> Barcode [0, 1.0], [0, inf]
        // step2 (2): d=1.5 -> Barcode [0, 1.5], [0, inf]
        // Distance(cue, step1) = 0.5 < Distance(cue, step2) = 1.0
        let d = 0.5 + offset * 0.5;
        splat.static_points.push([0.0, 0.0, 0.0]);
        splat.static_points.push([d, 0.0, 0.0]);

        splat
            .covariances
            .push([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        splat
            .covariances
            .push([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

        splat.motion_velocities = Some(vec![[0.0, 1.0, 0.0]]);
        splat.meta = SplatMeta {
            timestamp: None,
            labels: vec![label.into()],
            emotional_state: None,
            fitness_metadata: None,
        };
        splat
    }

    #[test]
    fn recall_episode_walks_sequence() {
        let config = SplatRagBuilder::new().build();
        let blob_store = crate::storage::InMemoryBlobStore::default();
        let hnsw = HnswIndex::new(1000);
        let mut store = TopologicalMemoryStore::with_indexer(config.clone(), blob_store, hnsw);

        let mut splats = Vec::new();
        for (i, label) in ["cue", "step1", "step2"].iter().enumerate() {
            let s = make_splat(label, i as f32);
            let id = store
                .add_splat(
                    &s,
                    crate::storage::OpaqueSplatRef::External(label.to_string()),
                    label.to_string(),
                    vec![i as f32; 384],
                )
                .unwrap();
            splats.push((id, s));
        }

        let id_to_splat = splats
            .iter()
            .map(|(id, splat)| (*id, splat.clone()))
            .collect::<std::collections::HashMap<_, _>>();

        // Initial query matches "cue" (offset 0.0)
        let initial = make_splat("cue", 0.0);
        let episode = recall_episode(&initial, 2, &store, &config, |result| {
            id_to_splat.get(&result.splat_id).cloned()
        })
        .unwrap();

        assert_eq!(episode.len(), 2);
        assert_eq!(episode[0].meta.labels, vec!["cue"]);
        assert_eq!(episode[1].meta.labels, vec!["step1"]);
    }
}
