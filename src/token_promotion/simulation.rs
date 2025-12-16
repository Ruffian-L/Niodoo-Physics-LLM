//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use std::collections::HashSet;
use std::time::Instant;

use crate::memory::emotional::{EmotionalState, TorusPadMapper};
use crate::memory_system::MemorySystem;

/// Result of a standalone promotion simulation cycle.
#[derive(Debug, Clone)]
pub struct PromotionResult {
    pub promoted_count: usize,
    pub pruned_count: usize,
    pub cycle_latency_ms: f64,
}

const PROMOTION_THRESHOLD: f32 = 0.55;
const PRUNE_THRESHOLD: f32 = 0.25;

/// Run a lightweight promotion simulation directly against the Gaussian memory system.
pub fn run_promotion_cycle(memory_system: &mut MemorySystem) -> PromotionResult {
    let cycle_start = Instant::now();
    let total_spheres = memory_system.storage.geometries.len();

    if total_spheres == 0 {
        return PromotionResult {
            promoted_count: 0,
            pruned_count: 0,
            cycle_latency_ms: cycle_start.elapsed().as_secs_f64() * 1000.0,
        };
    }

    // Aggregate Emotion (PAD)
    let mut p_sum = 0.0;
    let mut a_sum = 0.0;
    let mut d_sum = 0.0;

    for embedding in &memory_system.storage.embeddings {
        let emo = TorusPadMapper::project(embedding);
        p_sum += emo.pleasure;
        a_sum += emo.arousal;
        d_sum += emo.dominance;
    }

    let count = total_spheres as f32;
    let _probe_emotion = EmotionalState {
        pleasure: p_sum / count,
        arousal: a_sum / count,
        dominance: d_sum / count,
    };

    // Probe Query: "promotion_probe"
    // MemorySystem doesn't support query by emotion directly in `retrieve`.
    // But `retrieve_bicameral` uses `RadianceField` which uses `manifold_vector`.
    // We can't easily inject emotion into the query string.
    // We will just use a dummy query string "promotion_probe".

    let recall_results = memory_system
        .retrieve("promotion_probe", 20)
        .unwrap_or_default();

    let mut promoted: HashSet<u64> = HashSet::new();
    let mut pruned: HashSet<u64> = HashSet::new();

    for res in recall_results {
        // Use probability/radiance as score
        if res.probability >= PROMOTION_THRESHOLD {
            promoted.insert(res.payload_id);
        } else if res.probability <= PRUNE_THRESHOLD {
            pruned.insert(res.payload_id);
        }
    }

    // Apply feedback (modify covariance?)
    // MemorySystem geometries are `SplatGeometry`.
    // `SplatGeometry` has `covariance: [f32; 6]` (packed upper triangular).
    // We can modify it.

    // We need to map payload_id to index.
    // MemorySystem has `semantics` which has `payload_id`.
    // We can build a map or iterate.

    for (idx, _sem) in memory_system.storage.semantics.iter().enumerate() {
        if idx < memory_system.storage.payload_ids.len() {
            let payload_id = memory_system.storage.payload_ids[idx];

            if promoted.contains(&payload_id) {
                // Highlight promoted tokens (shrink/grow or color change)
                // For now, just a dummy visual effect if we had rendering
                // But we can modify geometry scale slightly to indicate "active"
                if idx < memory_system.storage.geometries.len() {
                    // Pulse effect
                    memory_system.storage.geometries[idx].scale[0] *= 0.9;
                    memory_system.storage.geometries[idx].scale[1] *= 0.9;
                    memory_system.storage.geometries[idx].scale[2] *= 0.9;
                }
            } else if pruned.contains(&payload_id) {
                if idx < memory_system.storage.geometries.len() {
                    // Expand effect (explode?)
                    memory_system.storage.geometries[idx].scale[0] *= 1.1;
                    memory_system.storage.geometries[idx].scale[1] *= 1.1;
                    memory_system.storage.geometries[idx].scale[2] *= 1.1;
                }
            }
        }
    }

    let promoted_count = promoted.len();
    let pruned_count = pruned.len();

    PromotionResult {
        promoted_count,
        pruned_count,
        cycle_latency_ms: cycle_start.elapsed().as_secs_f64() * 1000.0,
    }
}
