//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::embeddings::EmbeddingModel;
use crate::memory_system::MemorySystem;

use super::consensus::ConsensusVote;
use super::dynamic_tokenizer::TokenizerStats;
use super::pattern_discovery::PatternDiscoveryEngine;
use super::{ConsensusEngine, DynamicTokenizer, PromotedToken, TokenCandidate, TopologicalToken};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PromotionConfig {
    pub min_promotion_score: f64,
    pub max_candidates_per_cycle: usize,
    pub consensus_threshold: f64,
    pub pruning_min_usage: u64,
}

impl Default for PromotionConfig {
    fn default() -> Self {
        Self {
            min_promotion_score: 0.7,
            max_candidates_per_cycle: 10,
            consensus_threshold: 0.66,
            pruning_min_usage: 10,
        }
    }
}

pub struct TokenPromotionEngine {
    pattern_discovery: Arc<PatternDiscoveryEngine>,
    consensus: Arc<ConsensusEngine>,
    tokenizer: Arc<RwLock<DynamicTokenizer>>,
    embedding_model: Arc<EmbeddingModel>,
    config: PromotionConfig,
}

impl TokenPromotionEngine {
    pub fn new(
        pattern_discovery: Arc<PatternDiscoveryEngine>,
        consensus: Arc<ConsensusEngine>,
        tokenizer: Arc<RwLock<DynamicTokenizer>>,
        embedding_model: Arc<EmbeddingModel>,
    ) -> Self {
        Self {
            pattern_discovery,
            consensus,
            tokenizer,
            embedding_model,
            config: PromotionConfig::default(),
        }
    }

    pub fn with_config(mut self, config: PromotionConfig) -> Self {
        self.config = config;
        self
    }

    /// Encode text using the current dynamic vocabulary, tracking usage statistics.
    pub async fn encode_with_dynamic_vocab(&self, text: &str) -> Result<Vec<u32>> {
        let mut tokenizer = self.tokenizer.write().await;
        tokenizer.encode_extended(text)
    }

    /// Compute a promotion score for an arbitrary byte sequence against the current memory system.
    pub async fn score_candidate(
        &self,
        byte_seq: &[u8],
        memory_system: &MemorySystem,
    ) -> Result<f64> {
        let candidates = self
            .pattern_discovery
            .discover_candidates(memory_system)
            .await?;

        let score = candidates
            .into_iter()
            .find(|candidate| candidate.bytes.as_slice() == byte_seq)
            .map(|candidate| candidate.promotion_score())
            .unwrap_or(0.0);

        Ok(score)
    }

    pub async fn run_promotion_cycle(
        &self,
        memory_system: &MemorySystem,
    ) -> Result<PromotionCycleResult> {
        let start = Instant::now();
        tracing::info!("starting token promotion cycle");

        self.pattern_discovery
            .rebuild_spatial_index(memory_system)
            .await;

        let mut candidates = self
            .pattern_discovery
            .discover_candidates(memory_system)
            .await?;
        tracing::info!(
            candidate_count = candidates.len(),
            "pattern discovery complete"
        );

        candidates
            .retain(|candidate| candidate.promotion_score() >= self.config.min_promotion_score);
        if candidates.len() > self.config.max_candidates_per_cycle {
            candidates.truncate(self.config.max_candidates_per_cycle);
        }

        let mut promoted_tokens = Vec::new();
        let mut rejected_candidates = Vec::new();

        for candidate in candidates {
            let vote = self.consensus.propose_token(&candidate).await?;
            if vote.approved {
                let token = self.promote_candidate(candidate, vote).await?;
                promoted_tokens.push(token);
            } else {
                rejected_candidates.push(candidate);
            }
        }

        let pruned = self
            .tokenizer
            .write()
            .await
            .prune_unused(self.config.pruning_min_usage);
        let duration = start.elapsed();

        Ok(PromotionCycleResult {
            promoted: promoted_tokens,
            rejected: rejected_candidates,
            pruned,
            duration,
        })
    }

    async fn promote_candidate(
        &self,
        candidate: TokenCandidate,
        vote: ConsensusVote,
    ) -> Result<PromotedToken> {
        // Generate real embedding
        let text = String::from_utf8_lossy(&candidate.bytes).to_string();

        // EmbeddingModel::embed_query is blocking, so we wrap it
        let embedding_model = self.embedding_model.clone();
        let embedding =
            tokio::task::spawn_blocking(move || embedding_model.embed_query(&text)).await??;

        // Calculate score before moving fields
        let promotion_score = candidate.promotion_score();

        // MINT TOPOLOGICAL TOKEN
        // We use the reserved topological range
        let token_id = {
            let tokenizer = self.tokenizer.read().await;
            tokenizer.next_topological_id()
        };

        let topo_token = TopologicalToken {
            token_id: token_id as u64,
            centroid: candidate.centroid,
            covariance: candidate.covariance,
            barcode: candidate.barcode,
            average_valence: candidate.average_valence,
            birth_cycle: 0, // TODO: Track cycle count
            parent_cluster_ids: candidate.cluster_ids,
        };

        let promoted = PromotedToken {
            token_id,
            bytes: candidate.bytes.clone(),
            embedding,
            promotion_score,
            promoted_at: SystemTime::now(),
        };

        {
            let mut tokenizer = self.tokenizer.write().await;
            // Register the topological token instead of just adding a string
            tokenizer.register_topological_token(topo_token, candidate.bytes.clone())?;
        }

        tracing::info!(
            token_id = token_id,
            score = promoted.promotion_score,
            votes_for = vote.votes_for,
            votes_against = vote.votes_against,
            "promoted TOPOLOGICAL token"
        );

        Ok(promoted)
    }

    /// Get tokenizer statistics
    pub async fn tokenizer_stats(&self) -> TokenizerStats {
        let tokenizer = self.tokenizer.read().await;
        tokenizer.stats()
    }
}

#[derive(Debug)]
pub struct PromotionCycleResult {
    pub promoted: Vec<PromotedToken>,
    pub rejected: Vec<TokenCandidate>,
    pub pruned: usize,
    pub duration: Duration,
}
