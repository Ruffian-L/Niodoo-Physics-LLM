pub mod advanced;
pub mod dual_process;
pub mod fitness;
pub mod hippocampal;
pub mod hybrid;

use crate::storage::{SplatBlobStore, TopologicalMemoryStore};
use anyhow::Result;

pub use advanced::{AdvancedRetriever, RetrievalParams, RetrievalResult};
pub use dual_process::{conscious_recall, subconscious_priming, PrimedContext, RecallResult};
pub use hippocampal::recall_episode;
pub use hybrid::{HybridRetriever, ScoredMemory};

pub struct DualProcessQuery {
    _config: QueryConfig,
}

#[derive(Debug, Clone)]
pub struct QueryConfig {
    pub enable_conscious: bool,
    pub enable_subconscious: bool,
    pub top_k: usize,
}

impl Default for QueryConfig {
    fn default() -> Self {
        Self {
            enable_conscious: true,
            enable_subconscious: true,
            top_k: 10,
        }
    }
}

impl DualProcessQuery {
    pub fn new() -> Self {
        Self {
            _config: QueryConfig::default(),
        }
    }

    pub fn with_config(config: QueryConfig) -> Self {
        Self { _config: config }
    }

    pub async fn query<B: SplatBlobStore>(
        &self,
        store: &TopologicalMemoryStore<B>,
        query_vector: &[f32],
    ) -> Result<Vec<u64>> {
        // Perform Dual Process Query
        // 1. Subconscious: Fast ANN search
        let k = self._config.top_k;

        let hits = store.search_embeddings(query_vector, k)?;

        // If conscious recall is enabled, we might want to rerank using TDA
        // But this method signature only takes a vector, not a fingerprint.
        // If the vector IS the fingerprint vector, we can't reconstruct the fingerprint fully
        // (lossy compression).

        // However, for the purpose of this API, we return the ANN results.
        // Ideally, we should take a TopologicalFingerprint as input.
        // But adhering to the current interface (generic vector query):

        Ok(hits.into_iter().map(|(id, _)| id).collect())
    }
}

impl Default for DualProcessQuery {
    fn default() -> Self {
        Self::new()
    }
}

pub struct HippocampalRNN {
    _hidden_size: usize,
}

impl HippocampalRNN {
    pub fn new(hidden_size: usize) -> Self {
        Self {
            _hidden_size: hidden_size,
        }
    }

    pub fn reconstruct_sequence(&self, _memory_ids: &[u64]) -> Result<Vec<Vec<f32>>> {
        anyhow::bail!("Hippocampal sequence reconstruction not implemented yet")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual_process_creation() {
        let query = DualProcessQuery::new();
        assert!(query._config.enable_conscious);
    }

    #[test]
    fn test_hippocampal_creation() {
        let rnn = HippocampalRNN::new(128);
        assert_eq!(rnn._hidden_size, 128);
    }
}
