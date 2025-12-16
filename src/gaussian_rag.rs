//! GAUSSIAN RAG SYSTEM
//! Retrieval Augmented Generation with uncertainty quantification using topology analysis

use crate::memory_topology::{MemoryTopology, TopologyPattern};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct RetrievalResult {
    pub memory_id: String,
    pub content: String,
    pub similarity: f32,
    pub confidence: f32,
    pub topology_pattern: TopologyPattern,
    pub uncertainty_reasoning: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GaussianRAG {
    topology_engine: MemoryTopology,
    retrieval_threshold: f32,
    max_results: usize,
    uncertainty_filter: f32,
}

impl GaussianRAG {
    pub fn new() -> Self {
        Self {
            topology_engine: MemoryTopology::new(),
            retrieval_threshold: 0.3,
            max_results: 5,
            uncertainty_filter: 0.7, // Filter out high uncertainty results
        }
    }

    /// Add document to RAG system with topological analysis
    pub fn add_document(&mut self, doc_id: String, content: String, embedding: Vec<f32>) {
        self.topology_engine.add_memory(doc_id, content, embedding);
    }

    /// Retrieve with Gaussian uncertainty quantification
    pub fn retrieve(&self, query_embedding: &[f32]) -> Vec<RetrievalResult> {
        let raw_results = self
            .topology_engine
            .retrieve_with_uncertainty(query_embedding, self.max_results);

        let mut filtered_results = Vec::new();

        for (memory_id, similarity, confidence) in raw_results {
            // Apply uncertainty filter
            if confidence < self.uncertainty_filter {
                continue;
            }

            // Apply similarity threshold
            if similarity < self.retrieval_threshold {
                continue;
            }

            if let Some(memory) = self.topology_engine.memories.get(&memory_id) {
                let reasoning = self.generate_uncertainty_reasoning(
                    &memory.topology_pattern,
                    memory.uncertainty_score,
                    similarity,
                );

                let result = RetrievalResult {
                    memory_id: memory_id.clone(),
                    content: memory.content.clone(),
                    similarity,
                    confidence,
                    topology_pattern: memory.topology_pattern.clone(),
                    uncertainty_reasoning: reasoning,
                };

                filtered_results.push(result);
            }
        }

        filtered_results
    }

    /// Generate reasoning for uncertainty scores
    fn generate_uncertainty_reasoning(
        &self,
        pattern: &TopologyPattern,
        uncertainty: f32,
        similarity: f32,
    ) -> String {
        match pattern {
            TopologyPattern::VOID => {
                format!("High uncertainty detected ({}). Sparse data may indicate incomplete information.", uncertainty)
            }
            TopologyPattern::LINE => {
                format!(
                    "Low uncertainty ({}). Strong directed relationship with {} similarity.",
                    uncertainty, similarity
                )
            }
            TopologyPattern::PLANE => {
                format!(
                    "Medium uncertainty ({}). Surface-level connection with {} similarity.",
                    uncertainty, similarity
                )
            }
            TopologyPattern::SPHERE => {
                format!(
                    "Low uncertainty ({}). Complete concept with {} similarity.",
                    uncertainty, similarity
                )
            }
            TopologyPattern::CHAOTIC2 => {
                format!("Medium-high uncertainty ({}). Complex organic relationship with {} similarity.", uncertainty, similarity)
            }
            TopologyPattern::COMPLEX1 => {
                format!(
                    "Medium-low uncertainty ({}). System-level connection with {} similarity.",
                    uncertainty, similarity
                )
            }
        }
    }

    /// Find related documents using emergent connections
    pub fn find_related_documents(&self, doc_id: &str) -> Vec<(String, f32)> {
        self.topology_engine.find_emergent_connections(doc_id, 0.4)
    }

    /// Get system statistics
    pub fn get_system_stats(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();

        let topology_stats = self.topology_engine.get_topology_statistics();
        stats.insert(
            "topology_distribution".to_string(),
            serde_json::to_value(&topology_stats).unwrap(),
        );

        let total_memories = self.topology_engine.memories.len();
        stats.insert(
            "total_documents".to_string(),
            serde_json::Value::Number(total_memories.into()),
        );

        let clusters = self.topology_engine.analyze_memory_clusters();
        stats.insert(
            "clusters".to_string(),
            serde_json::to_value(&clusters).unwrap(),
        );

        stats
    }

    /// Adaptive threshold based on system uncertainty
    pub fn adaptive_retrieval(&self, query_embedding: &[f32]) -> Vec<RetrievalResult> {
        // Calculate average uncertainty in system
        let total_uncertainty: f32 = self
            .topology_engine
            .memories
            .values()
            .map(|m| m.uncertainty_score)
            .sum();

        let avg_uncertainty = total_uncertainty / self.topology_engine.memories.len() as f32;

        // Adjust threshold based on system uncertainty
        let adaptive_threshold = if avg_uncertainty > 0.6 {
            self.retrieval_threshold * 0.8 // Lower threshold for high uncertainty systems
        } else if avg_uncertainty < 0.3 {
            self.retrieval_threshold * 1.2 // Raise threshold for confident systems
        } else {
            self.retrieval_threshold
        };

        // Retrieve with adaptive threshold
        let raw_results = self
            .topology_engine
            .retrieve_with_uncertainty(query_embedding, self.max_results * 2);

        let mut filtered_results = Vec::new();

        for (memory_id, similarity, confidence) in raw_results {
            if similarity >= adaptive_threshold && confidence >= self.uncertainty_filter {
                if let Some(memory) = self.topology_engine.memories.get(&memory_id) {
                    let reasoning = format!(
                        "Adaptive threshold: {:.3} (system uncertainty: {:.3})",
                        adaptive_threshold, avg_uncertainty
                    );

                    let result = RetrievalResult {
                        memory_id: memory_id.clone(),
                        content: memory.content.clone(),
                        similarity,
                        confidence,
                        topology_pattern: memory.topology_pattern.clone(),
                        uncertainty_reasoning: reasoning,
                    };

                    filtered_results.push(result);
                }
            }
        }

        filtered_results.truncate(self.max_results);
        filtered_results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rag_retrieval() {
        let mut rag = GaussianRAG::new();

        // Add test documents
        let doc1_embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let doc2_embedding = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];

        rag.add_document(
            "doc1".to_string(),
            "First document content".to_string(),
            doc1_embedding,
        );
        rag.add_document(
            "doc2".to_string(),
            "Second document content".to_string(),
            doc2_embedding,
        );

        // Test retrieval
        let query_embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let results = rag.retrieve(&query_embedding);

        assert!(!results.is_empty());
    }
}
