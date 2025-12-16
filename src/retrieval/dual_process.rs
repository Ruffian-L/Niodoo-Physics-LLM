use std::cmp::Ordering;

use anyhow::Result;

use crate::indexing::fingerprint::{fingerprint_from_splat, wasserstein_distance};
use crate::indexing::TopologicalFingerprint;
use crate::storage::{OpaqueSplatRef, SplatBlobStore, TopologicalMemoryStore};
use crate::tivm::SplatRagConfig;
use crate::types::{SplatId, SplatInput, SplatMeta};

#[derive(Debug, Clone)]
pub struct PrimedContext {
    pub splat_id: SplatId,
    pub distance: f32,
    pub meta: SplatMeta,
}

#[derive(Debug, Clone)]
pub struct RecallResult {
    pub splat_id: SplatId,
    pub distance: f32,
    pub meta: SplatMeta,
    pub blob_handle: Option<OpaqueSplatRef>,
}

/// Stage-1 ANN lookup used for subconscious priming. Returns early if `k` is zero.
pub fn subconscious_priming<B: SplatBlobStore>(
    store: &TopologicalMemoryStore<B>,
    current_input: &SplatInput,
    config: &SplatRagConfig,
    k: usize,
) -> Result<Vec<PrimedContext>> {
    if k == 0 {
        return Ok(Vec::new());
    }

    let fingerprint = fingerprint_from_splat(current_input, config);
    let embedding = fingerprint.to_vector();
    if embedding.is_empty() {
        return Ok(Vec::new());
    }

    let hits = store.search_embeddings(&embedding, k)?;
    let mut contexts = Vec::with_capacity(hits.len());
    for (splat_id, distance) in hits {
        if let Some(record) = store.get(splat_id) {
            contexts.push(PrimedContext {
                splat_id,
                distance,
                meta: record.meta.clone(),
            });
        }
    }

    Ok(contexts)
}

/// Conscious recall over-fetches the ANN stage, then re-ranks using Wasserstein distance and ERAG fitness.
pub fn conscious_recall<B: SplatBlobStore>(
    store: &TopologicalMemoryStore<B>,
    query_fingerprint: &TopologicalFingerprint,
    k: usize,
) -> Result<Vec<RecallResult>> {
    if k == 0 {
        return Ok(Vec::new());
    }

    use crate::constants::RERANK_MULTIPLIER;

    let embedding = query_fingerprint.to_vector();
    if embedding.is_empty() {
        return Ok(Vec::new());
    }

    let ann_k = k.saturating_mul(RERANK_MULTIPLIER).max(k);
    let hits = store.search_embeddings(&embedding, ann_k)?;

    // Parallel Reranking with Rayon
    use rayon::prelude::*;
    let mut scored: Vec<RecallResult> = hits
        .par_iter()
        .filter_map(|(splat_id, _distance)| {
            if let Some(record) = store.get(*splat_id) {
                let wasserstein_dist = wasserstein_distance(query_fingerprint, &record.fingerprint);

                // ERAG Fitness Calculation
                let _similarity_score = 1.0 / (1.0 + wasserstein_dist);

                let emotional_score = if let Some(state) = &record.meta.emotional_state {
                    state.arousal
                } else {
                    0.5
                };

                let adjusted_distance = wasserstein_dist * (1.0 - (emotional_score - 0.5) * 0.2);

                let blob_handle = store.blob(*splat_id);
                Some(RecallResult {
                    splat_id: *splat_id,
                    distance: adjusted_distance,
                    meta: record.meta.clone(),
                    blob_handle,
                })
            } else {
                None
            }
        })
        .collect();

    scored.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(Ordering::Equal)
    });

    // Diversity Enforcement (Jaccard on Labels)
    let mut selected: Vec<RecallResult> = Vec::with_capacity(k);
    for candidate in scored {
        if selected.len() >= k {
            break;
        }

        let mut is_redundant = false;
        for existing in &selected {
            // Jaccard Similarity on Labels
            let set_a: std::collections::HashSet<_> = candidate.meta.labels.iter().collect();
            let set_b: std::collections::HashSet<_> = existing.meta.labels.iter().collect();

            let intersection = set_a.intersection(&set_b).count();
            let union = set_a.union(&set_b).count();

            if union > 0 {
                let jaccard = intersection as f32 / union as f32;
                if jaccard > 0.7 {
                    // Threshold: 0.7 means very similar topics
                    is_redundant = true;
                    break;
                }
            }
        }

        if !is_redundant {
            selected.push(candidate);
        }
    }

    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::hnsw::HnswIndex;
    use crate::tivm::SplatRagBuilder;
    use crate::types::{Mat3, Point3, Vec3};
    use crate::{SplatInput, SplatMeta};

    fn sample_splat(label: &str, offset: f32) -> SplatInput {
        let mut input = SplatInput::default();
        // Perturb position slightly to create distinct fingerprints
        input.static_points.push([offset, offset, offset]);
        // Add a second point to make it more interesting topologically if offset > 0
        if offset > 0.0 {
            input
                .static_points
                .push([offset + 1.0, offset + 1.0, offset + 1.0]);
        }
        input
            .covariances
            .push([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        if offset > 0.0 {
            input
                .covariances
                .push([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        }
        input.motion_velocities = Some(vec![[1.0, 0.0, 0.0]]);
        input.meta = SplatMeta {
            timestamp: None,
            labels: vec![label.into()],
            emotional_state: None,
            fitness_metadata: None,
        };
        input
    }

    #[test]
    fn subconscious_priming_returns_matches() {
        let config = SplatRagBuilder::new().build();
        let blob_store = crate::storage::InMemoryBlobStore::default();
        let hnsw = HnswIndex::new(1000);
        let mut store = TopologicalMemoryStore::with_indexer(config.clone(), blob_store, hnsw);

        let anchor = sample_splat("anchor", 0.0);
        store
            .add_splat(
                &anchor,
                OpaqueSplatRef::External("blob://anchor".into()),
                "anchor text".to_string(),
                vec![0.0; 384],
            )
            .unwrap();

        let contexts = subconscious_priming(&store, &anchor, &config, 1).unwrap();
        assert_eq!(contexts.len(), 1);
        assert_eq!(contexts[0].meta.labels, vec!["anchor"]);
    }

    #[test]
    fn conscious_recall_reranks_by_pd_distance() {
        let config = SplatRagBuilder::new().build();
        let blob_store = crate::storage::InMemoryBlobStore::default();
        let hnsw = HnswIndex::new(1000);
        let mut store = TopologicalMemoryStore::with_indexer(config.clone(), blob_store, hnsw);

        let target = sample_splat("target", 0.0);
        // Distractor has different topology (2 points vs 1 point)
        let distractor = sample_splat("distractor", 5.0);

        store
            .add_splat(
                &target,
                OpaqueSplatRef::External("blob://target".into()),
                "target text".to_string(),
                vec![0.0; 384],
            )
            .unwrap();
        store
            .add_splat(
                &distractor,
                OpaqueSplatRef::External("blob://distractor".into()),
                "distractor text".to_string(),
                vec![0.0; 384],
            )
            .unwrap();

        // Query with target's fingerprint. Target should be closer (distance 0) than distractor.
        let query_fp = fingerprint_from_splat(&target, &config);
        let results = conscious_recall(&store, &query_fp, 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].meta.labels, vec!["target"]);
        assert!(results[0].blob_handle.is_some());
    }

    #[test]
    fn diversity_enforcement_removes_redundant_results() {
        let config = SplatRagBuilder::new().build();
        let blob_store = crate::storage::InMemoryBlobStore::default();
        let hnsw = HnswIndex::new(1000);
        let mut store = TopologicalMemoryStore::with_indexer(config.clone(), blob_store, hnsw);

        // Create 3 splats. 2 are identical in topic ("cats"), 1 is different ("dogs").
        let cat1 = sample_splat("cats", 0.0);
        let cat2 = sample_splat("cats", 0.1); // Slightly different pos, same label
        let dog = sample_splat("dogs", 0.2);

        store
            .add_splat(
                &cat1,
                OpaqueSplatRef::External("b1".into()),
                "c1".into(),
                vec![0.0; 384],
            )
            .unwrap();
        store
            .add_splat(
                &cat2,
                OpaqueSplatRef::External("b2".into()),
                "c2".into(),
                vec![0.0; 384],
            )
            .unwrap();
        store
            .add_splat(
                &dog,
                OpaqueSplatRef::External("b3".into()),
                "d1".into(),
                vec![0.0; 384],
            )
            .unwrap();

        // Query that matches all (all have 0.0 embedding in this mock setup?)
        // Actually sample_splat uses 0.0 embedding in add_splat calls above?
        // Wait, sample_splat returns SplatInput. add_splat takes embedding.
        // In the test above, I passed vec![0.0; 384] for all.
        // So HNSW will return all of them as matches.

        let query_fp = fingerprint_from_splat(&cat1, &config);

        // Request k=3. Should get cat1, dog. cat2 should be filtered if diversity works.
        // But wait, if k=3, and we have 3 items, and 1 is redundant...
        // The diversity logic fills `selected` up to `k`.
        // If cat1 is picked, cat2 is redundant. dog is picked.
        // So we should get 2 items if we ask for 2? Or if we ask for 3, do we get 2?
        // The loop breaks if `selected.len() >= k`.
        // If we skip cat2, we might pick the next one.
        // So if we ask for k=2, we should get cat1 and dog.

        let results = conscious_recall(&store, &query_fp, 2).unwrap();

        // We expect cat1 and dog. cat2 should be skipped because it overlaps "cats" with cat1.
        assert_eq!(results.len(), 2);
        let labels: Vec<&String> = results.iter().map(|r| &r.meta.labels[0]).collect();
        assert!(labels.contains(&&"cats".to_string()));
        assert!(labels.contains(&&"dogs".to_string()));
        // Ensure we don't have 2 cats
        assert_eq!(labels.iter().filter(|&&l| l == "cats").count(), 1);
    }
}
