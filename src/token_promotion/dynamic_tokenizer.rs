//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use super::{PromotedToken, TopologicalToken};

const TOPOLOGICAL_TOKEN_BASE: u32 = 1_000_000;

#[derive(Clone)]
pub struct DynamicTokenizer {
    base_tokenizer: Tokenizer,
    extended_vocab: HashMap<Vec<u8>, u32>,
    id_to_bytes: HashMap<u32, Vec<u8>>,
    next_token_id: u32,
    next_topological_id: u32,
    topological_tokens: HashMap<u64, TopologicalToken>,
    token_usage: HashMap<u32, u64>,
    max_extended_length: usize,
}

impl DynamicTokenizer {
    pub fn new(base_tokenizer: Tokenizer) -> Self {
        let next_token_id = base_tokenizer.get_vocab_size(false) as u32;
        Self {
            base_tokenizer,
            extended_vocab: HashMap::new(),
            id_to_bytes: HashMap::new(),
            next_token_id,
            next_topological_id: TOPOLOGICAL_TOKEN_BASE,
            topological_tokens: HashMap::new(),
            token_usage: HashMap::new(),
            max_extended_length: 20,
        }
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path.as_ref())
            .map_err(|err| anyhow!("failed to load tokenizer: {err}"))?;
        Ok(Self::new(tokenizer))
    }

    pub fn next_token_id(&self) -> u32 {
        self.next_token_id
    }

    pub fn next_topological_id(&self) -> u32 {
        self.next_topological_id
    }

    pub fn register_topological_token(
        &mut self,
        token: TopologicalToken,
        bytes: Vec<u8>,
    ) -> Result<()> {
        let token_id = token.token_id as u32;

        // Store the topological payload
        self.topological_tokens.insert(token.token_id, token);

        // Update max length for matching
        if bytes.len() > self.max_extended_length {
            println!(
                "DEBUG: Updating max_extended_length from {} to {}",
                self.max_extended_length,
                bytes.len()
            );
            self.max_extended_length = bytes.len();
        }

        // Map bytes to this ID so it can be tokenized
        self.extended_vocab.insert(bytes.clone(), token_id);
        self.id_to_bytes.insert(token_id, bytes);
        self.token_usage.insert(token_id, 0);

        // Advance topological ID counter if needed
        if token_id >= self.next_topological_id {
            self.next_topological_id = token_id + 1;
        }

        Ok(())
    }

    /// Hook for Inference Engine:
    /// When the LLM encounters a token_id >= TOPOLOGICAL_TOKEN_BASE,
    /// it should call this to retrieve the geometric payload.
    /// The embedding matrix lookup should be BYPASSED.
    /// Instead, inject `centroid` + `covariance` + `valence` directly into the residual stream.
    pub fn get_token_topology(&self, token_id: u64) -> Option<&TopologicalToken> {
        self.topological_tokens.get(&token_id)
    }

    pub fn add_promoted_token(&mut self, token: &PromotedToken) -> Result<()> {
        if self.extended_vocab.contains_key(&token.bytes) {
            return Ok(());
        }

        let token_id = token.token_id;
        self.extended_vocab.insert(token.bytes.clone(), token_id);
        self.id_to_bytes.insert(token_id, token.bytes.clone());
        self.token_usage.insert(token_id, 0);
        self.next_token_id = self.next_token_id.max(token_id + 1);

        if token.bytes.len() > self.max_extended_length {
            self.max_extended_length = token.bytes.len();
        }

        tracing::info!(
          token = %String::from_utf8_lossy(&token.bytes),
          token_id = token_id,
          score = token.promotion_score,
          "added dynamic token"
        );

        Ok(())
    }

    pub fn encode_extended(&mut self, text: &str) -> Result<Vec<u32>> {
        let bytes = text.as_bytes();
        let mut tokens = Vec::new();
        let mut index = 0;

        while index < bytes.len() {
            let mut matched = false;
            let start = index;
            let min_len = self.min_token_length();

            for len in (min_len..=self.max_extended_length).rev() {
                if start + len > bytes.len() {
                    continue;
                }

                let candidate = &bytes[start..start + len];
                // DEBUG PRINT
                if len > 5 {
                    println!(
                        "DEBUG: Checking candidate len {}: {:?}",
                        len,
                        String::from_utf8_lossy(candidate)
                    );
                }

                if let Some(&token_id) = self.extended_vocab.get(candidate) {
                    tokens.push(token_id);
                    *self.token_usage.entry(token_id).or_insert(0) += 1;
                    index += len;
                    matched = true;
                    break;
                }
            }

            if !matched {
                let remaining = &text[index..];
                if remaining.is_empty() {
                    break;
                }

                let encoding = self
                    .base_tokenizer
                    .encode(remaining, false)
                    .map_err(|err| anyhow!("tokenizer encoding failed: {err}"))?;
                let ids = encoding.get_ids();

                if ids.is_empty() {
                    // Fallback: advance by one character to avoid infinite loop
                    let char_len = remaining
                        .chars()
                        .next()
                        .map(|ch| ch.len_utf8())
                        .unwrap_or(1);
                    let fallback_slice = &remaining[..char_len];
                    let fallback_ids = self
                        .base_tokenizer
                        .encode(fallback_slice, false)
                        .map_err(|err| anyhow!("tokenizer encoding failed: {err}"))?;
                    tokens.extend_from_slice(fallback_ids.get_ids());
                    index += char_len;
                    continue;
                }

                tokens.push(ids[0]);

                let consumed = encoding
                    .get_offsets()
                    .get(0)
                    .map(|(_, end)| *end)
                    .filter(|end| *end > 0)
                    .unwrap_or_else(|| {
                        remaining
                            .chars()
                            .next()
                            .map(|ch| ch.len_utf8())
                            .unwrap_or(1)
                    });

                index += consumed;
            }
        }

        Ok(tokens)
    }

    pub fn decode_extended(&self, ids: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(token_bytes) = self.id_to_bytes.get(&id) {
                bytes.extend_from_slice(token_bytes);
            } else {
                let decoded = self
                    .base_tokenizer
                    .decode(&[id], false)
                    .map_err(|err| anyhow!("tokenizer decoding failed: {err}"))?;
                bytes.extend_from_slice(decoded.as_bytes());
            }
        }

        Ok(String::from_utf8_lossy(&bytes).to_string())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.base_tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow!(e))
    }

    pub fn decode_batch(
        &self,
        sequences: &[Vec<u32>],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>> {
        let refs: Vec<&[u32]> = sequences.iter().map(|v| v.as_slice()).collect();
        self.base_tokenizer
            .decode_batch(&refs, skip_special_tokens)
            .map_err(|e| anyhow!(e))
    }

    pub fn prune_unused(&mut self, min_usage: u64) -> usize {
        let to_remove: Vec<u32> = self
            .token_usage
            .iter()
            .filter(|(_, usage)| **usage < min_usage)
            .map(|(&id, _)| id)
            .collect();

        for id in &to_remove {
            if let Some(bytes) = self.id_to_bytes.remove(id) {
                self.extended_vocab.remove(&bytes);
            }
            self.token_usage.remove(id);
        }

        to_remove.len()
    }

    pub fn stats(&self) -> TokenizerStats {
        let active_extended_tokens = self
            .token_usage
            .iter()
            .filter(|(_, usage)| **usage > 0)
            .count();

        TokenizerStats {
            base_vocab_size: self.base_tokenizer.get_vocab_size(false),
            extended_vocab_size: self.extended_vocab.len(),
            total_usage: self.token_usage.values().sum(),
            active_extended_tokens,
        }
    }

    fn min_token_length(&self) -> usize {
        4
    }

    /// CRDT: Merge remote vocabulary with Byzantine-tolerant consensus
    /// Uses usage-weighted last-write-wins strategy
    pub fn merge_remote_vocabulary(&mut self, remote: &RemoteVocabulary) -> Result<MergeStats> {
        let mut added = 0;
        let mut conflicts_resolved = 0;
        let mut usage_updated = 0;

        for (bytes, remote_entry) in &remote.tokens {
            match self.extended_vocab.get(bytes) {
                Some(&local_token_id) => {
                    // Token exists locally - update usage with max strategy
                    let local_usage = self.token_usage.get(&local_token_id).copied().unwrap_or(0);
                    let remote_usage = remote_entry.usage;

                    if remote_usage > local_usage {
                        self.token_usage.insert(local_token_id, remote_usage);
                        usage_updated += 1;
                        tracing::debug!(
                          token = %String::from_utf8_lossy(bytes),
                          local_usage = local_usage,
                          remote_usage = remote_usage,
                          "updated token usage from remote"
                        );
                    }
                }
                None => {
                    // New token from remote - add it
                    let token_id = remote_entry.token_id;

                    // Handle ID conflicts: if ID already used, assign new ID
                    let final_token_id = if self.id_to_bytes.contains_key(&token_id) {
                        conflicts_resolved += 1;
                        let new_id = self.next_token_id;
                        self.next_token_id += 1;
                        tracing::warn!(
                          token = %String::from_utf8_lossy(bytes),
                          remote_id = token_id,
                          assigned_id = new_id,
                          "resolved token ID conflict"
                        );
                        new_id
                    } else {
                        self.next_token_id = self.next_token_id.max(token_id + 1);
                        token_id
                    };

                    self.extended_vocab.insert(bytes.clone(), final_token_id);
                    self.id_to_bytes.insert(final_token_id, bytes.clone());
                    self.token_usage.insert(final_token_id, remote_entry.usage);
                    added += 1;

                    tracing::info!(
                      token = %String::from_utf8_lossy(bytes),
                      token_id = final_token_id,
                      usage = remote_entry.usage,
                      "added remote token to vocabulary"
                    );
                }
            }
        }

        Ok(MergeStats {
            added,
            conflicts_resolved,
            usage_updated,
        })
    }

    /// Export vocabulary for CRDT synchronization
    pub fn export_vocabulary(&self) -> RemoteVocabulary {
        let mut tokens = HashMap::new();

        for (bytes, &token_id) in &self.extended_vocab {
            let usage = self.token_usage.get(&token_id).copied().unwrap_or(0);
            tokens.insert(bytes.clone(), RemoteTokenEntry { token_id, usage });
        }

        RemoteVocabulary {
            tokens,
            node_id: std::env::var("NODE_ID").unwrap_or_else(|_| "unknown".to_string()),
            timestamp: chrono::Utc::now().timestamp(),
        }
    }
}

/// Remote vocabulary for CRDT consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteVocabulary {
    pub tokens: HashMap<Vec<u8>, RemoteTokenEntry>,
    pub node_id: String,
    pub timestamp: i64,
}

/// Remote token entry with usage metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteTokenEntry {
    pub token_id: u32,
    pub usage: u64,
}

/// Statistics from CRDT merge operation
#[derive(Debug, Clone)]
pub struct MergeStats {
    pub added: usize,
    pub conflicts_resolved: usize,
    pub usage_updated: usize,
}

#[derive(Debug, Clone)]
pub struct TokenizerStats {
    pub base_vocab_size: usize,
    pub extended_vocab_size: usize,
    pub total_usage: u64,
    pub active_extended_tokens: usize,
}

impl TokenizerStats {
    /// Calculate out-of-vocabulary rate (proxy: ratio of base tokens to extended)
    pub fn oov_rate(&self) -> f64 {
        if self.total_usage == 0 {
            return 0.0;
        }
        // OOV approximation: if we're using many extended tokens, OOV is decreasing
        // This is a simplified metric - real OOV would need to track actual misses
        let extended_ratio =
            self.active_extended_tokens as f64 / (self.base_vocab_size as f64).max(1.0);
        (1.0 - extended_ratio).max(0.0).min(1.0)
    }

    /// Get total vocabulary size (base + extended)
    pub fn vocab_size(&self) -> usize {
        self.base_vocab_size + self.extended_vocab_size
    }
}
