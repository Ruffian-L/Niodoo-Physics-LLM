use crate::memory::emotional::{PadGhostState, TemporalDecayConfig, WeightedMemoryMetadata};
use std::time::{SystemTime, UNIX_EPOCH};

pub struct FitnessWeights {
    pub age_weight: f32,
    pub pad_alignment_weight: f32,
    pub beta1_weight: f32,
    pub retrieval_count_weight: f32,
    pub consonance_weight: f32,
    pub resource_penalty_weight: f32,
}

impl Default for FitnessWeights {
    fn default() -> Self {
        Self {
            age_weight: 0.2,
            pad_alignment_weight: 0.3,
            beta1_weight: 0.2,
            retrieval_count_weight: 0.1,
            consonance_weight: 0.1,
            resource_penalty_weight: 0.1,
        }
    }
}

/// Calculate the "Radiance" (Fitness) score of a memory.
///
/// Radiance = w_age * AgeFactor + w_pad * PADAlignment + w_beta1 * Beta1 + ...
///
/// This score determines how "alive" or important a memory is, independent of pure vector similarity.
pub fn calculate_radiance_score(
    birth_time: f64,
    memory_metadata: &WeightedMemoryMetadata,
    _current_pad_state: &PadGhostState,
    weights: &FitnessWeights,
    temporal_config: &TemporalDecayConfig,
) -> f32 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    let age_seconds = (now - birth_time).max(0.0);
    let age_days = age_seconds / 86400.0;

    // 1. Temporal Factor (Decay)
    // Weight decreases over time unless reinforced.
    let half_life = temporal_config.half_life_days as f64;
    let half_life = half_life.max(0.1);
    let decay_factor = (-age_days / half_life).exp() as f32;
    let temporal_score = decay_factor.max(temporal_config.min_weight);

    // 2. PAD Alignment (Emotional Resonance)
    // How well does the memory's emotional state align with the current state?
    // Note: We don't have the memory's PAD state directly in metadata currently,
    // but we can use consonance_score as a proxy or just assume current state context.
    // Niodoo uses trajectory alignment. For now, we use the consonance score stored in metadata.
    // Ideally we would project the memory embedding to PAD and compare.
    // Let's assume high consonance means high alignment.
    let emotional_score = memory_metadata.consonance_score;

    // 3. Topological Connectivity (Beta-1)
    // High Beta-1 means the memory is part of a robust cycle/concept.
    let topology_score = memory_metadata.beta_1_connectivity;

    // 4. Retrieval Count (Reinforcement)
    // Logarithmic boost for frequently retrieved memories.
    let retrieval_boost = (memory_metadata.retrieval_count as f32 + 1.0).ln();

    // Combine components
    let mut score = 0.0;
    score += weights.age_weight * temporal_score;
    score += weights.pad_alignment_weight * emotional_score;
    score += weights.beta1_weight * topology_score;
    score += weights.retrieval_count_weight * retrieval_boost;
    score += weights.consonance_weight * memory_metadata.consonance_score;

    // Resource penalty could be subtracted here if we had resource usage data.

    score
}

/// Calculate diversity penalty using Jaccard similarity of n-grams or simple token overlap.
///
/// Returns a penalty factor (0.0 to 1.0) where 1.0 means "identical to existing results".
pub fn calculate_diversity_penalty(candidate_text: &str, selected_texts: &[String]) -> f32 {
    if selected_texts.is_empty() {
        return 0.0;
    }

    let mut max_similarity = 0.0;

    for selected in selected_texts {
        let sim = jaccard_similarity(candidate_text, selected);
        if sim > max_similarity {
            max_similarity = sim;
        }
    }

    max_similarity
}

fn jaccard_similarity(s1: &str, s2: &str) -> f32 {
    let s1_tokens: std::collections::HashSet<&str> = s1.split_whitespace().collect();
    let s2_tokens: std::collections::HashSet<&str> = s2.split_whitespace().collect();

    if s1_tokens.is_empty() && s2_tokens.is_empty() {
        return 1.0;
    }

    let intersection = s1_tokens.intersection(&s2_tokens).count();
    let union = s1_tokens.union(&s2_tokens).count();

    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}
