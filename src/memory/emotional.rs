use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};

/// PAD Emotional State: The "Feeling" of a memory.
/// Derived deterministically from the embedding space to allow "Bi-Cameral" resonance.
#[derive(
    Debug, Clone, Serialize, Deserialize, Default, Archive, RkyvSerialize, RkyvDeserialize,
)]
pub struct EmotionalState {
    pub pleasure: f32,  // -1.0 to 1.0
    pub arousal: f32,   // -1.0 to 1.0
    pub dominance: f32, // -1.0 to 1.0
}

impl EmotionalState {
    pub fn neutral() -> Self {
        Self {
            pleasure: 0.0,
            arousal: 0.0,
            dominance: 0.0,
        }
    }

    /// Calculate the "intensity" (magnitude) of the emotion
    pub fn intensity(&self) -> f32 {
        (self.pleasure.powi(2) + self.arousal.powi(2) + self.dominance.powi(2)).sqrt()
    }
}

/// Projects the Nomic embedding (768 dim or similar) onto a Topological Torus
/// using deterministic folding.
pub struct TorusPadMapper;

impl TorusPadMapper {
    /// Maps a dense vector to a PAD state using toroidal projection.
    /// We use specific dimensions of the embedding to drive the PAD values.
    /// In a real system, this would be a trained projection matrix,
    /// but here we use a deterministic hashing/folding of the vector.
    pub fn project(embedding: &[f32]) -> EmotionalState {
        if embedding.is_empty() {
            return EmotionalState::neutral();
        }

        // Fold the vector into 3 components using stride
        let mut p_sum = 0.0;
        let mut a_sum = 0.0;
        let mut d_sum = 0.0;

        for (i, val) in embedding.iter().enumerate() {
            match i % 3 {
                0 => p_sum += val,
                1 => a_sum += val,
                2 => d_sum += val,
                _ => {}
            }
        }

        // Normalize to -1.0 to 1.0 (Tanh is good for squashing)
        // Since we sum many small values, the sum can be large, so we might want to scale before tanh
        // Standard BERT/Nomic embeddings are normalized, so individual values are small.
        // Summing 768/3 ~= 256 values. Random walk sigma ~ sqrt(256) ~ 16.
        // Tanh will saturate quickly. We should scale down by sqrt(dim/3).
        let scale = 1.0 / ((embedding.len() as f32 / 3.0).sqrt().max(1.0));

        EmotionalState {
            pleasure: (p_sum * scale).tanh(),
            arousal: (a_sum * scale).tanh(),
            dominance: (d_sum * scale).tanh(),
        }
    }

    /// Calculates the "Mood Distance" on the Torus surface.
    /// Unlike Euclidean distance, this wraps around (cyclic emotions).
    /// Range of PAD is -1.0 to 1.0, so total span is 2.0.
    pub fn toroidal_distance(a: &EmotionalState, b: &EmotionalState) -> f32 {
        let dp = (a.pleasure - b.pleasure).abs();
        let da = (a.arousal - b.arousal).abs();
        let dd = (a.dominance - b.dominance).abs();

        // Wrap around 2.0 (since range is -1 to 1, total span is 2)
        let wp = if dp > 1.0 { 2.0 - dp } else { dp };
        let wa = if da > 1.0 { 2.0 - da } else { da };
        let wd = if dd > 1.0 { 2.0 - dd } else { dd };

        (wp.powi(2) + wa.powi(2) + wd.powi(2)).sqrt()
    }
}

/// Legacy structs for compatibility, aliased or adapted
pub type EmotionalVector = EmotionalState;

#[derive(
    Debug, Clone, Serialize, Deserialize, Default, Archive, RkyvSerialize, RkyvDeserialize,
)]
pub struct PadGhostState {
    // We keep the 7D arrays for legacy/topology compatibility if needed,
    // but for now we just map the core 3 to PAD and rest to 0.
    pub pad: [f64; 7],
    pub entropy: f64,
}

impl From<EmotionalState> for PadGhostState {
    fn from(e: EmotionalState) -> Self {
        let mut pad = [0.0; 7];
        pad[0] = e.pleasure as f64;
        pad[1] = e.arousal as f64;
        pad[2] = e.dominance as f64;
        // 3-6 are ghost dims, leave as 0.0 or derive?
        Self {
            pad,
            entropy: e.intensity() as f64, // Proxy entropy
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct WeightedMemoryMetadata {
    pub retrieval_count: u64,
    pub last_accessed: u64,    // Unix timestamp
    pub consonance_score: f32, // Replaces resonance_score
    pub beta_1_connectivity: f32,
    #[serde(default)]
    pub merged_count: u32,
}

impl Default for WeightedMemoryMetadata {
    fn default() -> Self {
        Self {
            retrieval_count: 0,
            last_accessed: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            consonance_score: 1.0,
            beta_1_connectivity: 0.5,
            merged_count: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct TemporalDecayConfig {
    pub half_life_days: f32,
    pub min_weight: f32,
}

impl Default for TemporalDecayConfig {
    fn default() -> Self {
        Self {
            half_life_days: 30.0,
            min_weight: 0.1,
        }
    }
}
