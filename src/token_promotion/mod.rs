//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use std::time::SystemTime;

pub mod consensus;
pub mod dynamic_tokenizer;
pub mod engine;
pub mod pattern_discovery;
pub mod simulation;
pub mod spatial;

pub use consensus::{ConsensusEngine, ConsensusVote, NodeId};
pub use dynamic_tokenizer::{DynamicTokenizer, TokenizerStats};
pub use engine::{PromotionConfig, PromotionCycleResult, TokenPromotionEngine};
pub use pattern_discovery::PatternDiscoveryEngine;
pub use simulation::{run_promotion_cycle, PromotionResult};

/// A persistent topological feature representing a stable memory cluster.
/// This token DOES NOT map to a learned embedding vector.
/// Instead, it maps to a direct Gaussian constellation in the memory manifold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologicalToken {
    pub token_id: u64,
    #[serde(with = "BigArray")]
    pub centroid: [f32; 768],
    pub covariance: Vec<f32>,     // Flattened covariance or reduced basis
    pub barcode: Vec<(f32, f32)>, // Persistent homology barcode
    pub average_valence: f32,
    pub birth_cycle: u64,
    pub parent_cluster_ids: Vec<u64>,
}

/// Candidate for token promotion discovered during pattern analysis.
#[derive(Debug, Clone)]
pub struct TokenCandidate {
    pub bytes: Vec<u8>,
    pub persistence: f64,
    pub frequency: u64,
    pub emotional_coherence: f64,
    pub spatial_locality: f64,
    pub timestamp: SystemTime,
    // Topological Data
    pub centroid: [f32; 768],
    pub covariance: Vec<f32>,
    pub barcode: Vec<(f32, f32)>,
    pub average_valence: f32,
    pub cluster_ids: Vec<u64>,
}

impl TokenCandidate {
    /// Calculate promotion score using a weighted combination of signals.
    pub fn promotion_score(&self) -> f64 {
        const ALPHA: f64 = 0.5; // Topological persistence
        const BETA: f64 = 0.3; // Usage frequency
        const GAMMA: f64 = 0.2; // Emotional coherence

        let persistence_term = ALPHA * self.persistence;
        // Guard against ln(0).
        let frequency_term = if self.frequency > 0 {
            BETA * (self.frequency as f64).ln()
        } else {
            0.0
        };
        let coherence_term = GAMMA * self.emotional_coherence;

        persistence_term + frequency_term + coherence_term
    }

    /// Human readable representation for logging.
    pub fn display_string(&self) -> String {
        String::from_utf8_lossy(&self.bytes).to_string()
    }
}

/// Runtime token representation ready for integration with the tokenizer.
#[derive(Debug, Clone)]
pub struct PromotedToken {
    pub token_id: u32,
    pub bytes: Vec<u8>,
    pub embedding: Vec<f32>,
    pub promotion_score: f64,
    pub promoted_at: SystemTime,
}
