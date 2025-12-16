use crate::config::SplatMemoryConfig;
use crate::embeddings::EmbeddingModel;
use crate::ingest::IngestionEngine;
use crate::retrieval::advanced::{AdvancedRetriever, RetrievalParams};
use crate::storage::engine::SplatStorage;
use crate::structs::SplatManifest;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;

#[derive(Debug, Deserialize, Clone)]
pub struct BeirDoc {
    pub _id: String,
    pub title: String,
    pub text: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct BeirQuery {
    pub _id: String,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct BeirQrel {
    pub query_id: String,
    pub doc_id: String,
    pub score: i32,
}

pub struct BeirLoader {
    pub corpus: HashMap<String, BeirDoc>,
    pub queries: HashMap<String, BeirQuery>,
    pub qrels: Vec<BeirQrel>,
}

impl BeirLoader {
    pub fn new() -> Self {
        Self {
            corpus: HashMap::new(),
            queries: HashMap::new(),
            qrels: Vec::new(),
        }
    }

    pub fn load_corpus(&mut self, path: &str) -> Result<()> {
        let file = File::open(path).context("Failed to open corpus file")?;
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let line = line?;
            let doc: BeirDoc = serde_json::from_str(&line)?;
            self.corpus.insert(doc._id.clone(), doc);
        }
        Ok(())
    }

    pub fn load_queries(&mut self, path: &str) -> Result<()> {
        let file = File::open(path).context("Failed to open queries file")?;
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let line = line?;
            let query: BeirQuery = serde_json::from_str(&line)?;
            self.queries.insert(query._id.clone(), query);
        }
        Ok(())
    }

    pub fn load_qrels(&mut self, path: &str) -> Result<()> {
        let mut rdr = csv::ReaderBuilder::new()
            .delimiter(b'\t')
            .has_headers(true)
            .from_path(path)?;

        for result in rdr.records() {
            let record = result?;
            let query_id = record.get(0).unwrap().to_string();
            let doc_id = record.get(1).unwrap().to_string();
            let score: i32 = record.get(2).unwrap().parse()?;

            self.qrels.push(BeirQrel {
                query_id,
                doc_id,
                score,
            });
        }
        Ok(())
    }
}

pub struct BeirEvaluator<'a> {
    retriever: &'a AdvancedRetriever<'a>,
    storage: &'a SplatStorage,
    id_map: HashMap<u64, String>,
}

impl<'a> BeirEvaluator<'a> {
    pub fn new(
        retriever: &'a AdvancedRetriever<'a>,
        storage: &'a SplatStorage,
        manifest_path: &str,
    ) -> Result<Self> {
        let file = File::open(manifest_path).context("Failed to open manifest file")?;
        let reader = BufReader::new(file);
        let manifest: SplatManifest = serde_json::from_reader(reader)?;

        let mut id_map = HashMap::new();
        for entry in manifest.entries {
            // Extract BEIR ID from tags
            let mut beir_id = None;
            for tag in entry.tags {
                if tag.starts_with("beir_id:") {
                    beir_id = Some(tag.trim_start_matches("beir_id:").to_string());
                    break;
                }
            }
            if let Some(bid) = beir_id {
                id_map.insert(entry.id, bid);
            }
        }

        Ok(Self {
            retriever,
            storage,
            id_map,
        })
    }

    pub fn evaluate(
        &self,
        queries: &HashMap<String, BeirQuery>,
        top_k: usize,
    ) -> Result<HashMap<String, HashMap<String, f32>>> {
        let mut results = HashMap::new();

        let mut count = 0;
        let total = queries.len();

        for (qid, query) in queries {
            count += 1;
            if count % 10 == 0 {
                println!("Evaluated {}/{} queries...", count, total);
            }

            let params = RetrievalParams {
                top_k,
                weight_bm25: 15.0,    // Boost keyword matching
                weight_cosine: 10.0,  // Keep vector search strong
                weight_radiance: 2.0, // Increase radiance contribution
                weight_physics: 3.0,  // Enable physics with moderate weight
                diversity: true,      // Keep MMR for diversity
                shadow_mode: false,
                use_physics: true, // ENABLE PHYSICS
            };

            let hits = self.retriever.search(&query.text, self.storage, params)?;
            let mut query_results = HashMap::new();
            for hit in hits {
                if let Some(external_id) = self.id_map.get(&hit.payload_id) {
                    query_results.insert(external_id.clone(), hit.final_score);
                }
            }
            results.insert(qid.clone(), query_results);
        }

        Ok(results)
    }

    pub fn compute_ndcg(
        &self,
        results: &HashMap<String, HashMap<String, f32>>,
        qrels: &[BeirQrel],
        k: usize,
    ) -> f32 {
        let mut qrel_map: HashMap<String, HashMap<String, i32>> = HashMap::new();
        for qrel in qrels {
            qrel_map
                .entry(qrel.query_id.clone())
                .or_default()
                .insert(qrel.doc_id.clone(), qrel.score);
        }

        let mut total_ndcg = 0.0;
        let mut count = 0;

        for (qid, doc_scores) in results {
            if let Some(relevant_docs) = qrel_map.get(qid) {
                // Sort retrieved docs by score descending
                let mut sorted_docs: Vec<(&String, &f32)> = doc_scores.iter().collect();
                sorted_docs
                    .sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

                // Cutoff at k
                let cutoff = sorted_docs.len().min(k);

                // Compute DCG
                let mut dcg = 0.0;
                for i in 0..cutoff {
                    let (doc_id, _) = sorted_docs[i];
                    if let Some(&rel) = relevant_docs.get(doc_id.as_str()) {
                        if rel > 0 {
                            dcg += (rel as f32) / (i as f32 + 2.0).log2();
                        }
                    }
                }

                // Compute IDCG
                let mut ideal_rels: Vec<i32> =
                    relevant_docs.values().cloned().filter(|&r| r > 0).collect();
                ideal_rels.sort_by(|a, b| b.cmp(a)); // Descending

                let mut idcg = 0.0;
                for i in 0..ideal_rels.len().min(k) {
                    idcg += (ideal_rels[i] as f32) / (i as f32 + 2.0).log2();
                }

                if idcg > 0.0 {
                    total_ndcg += dcg / idcg;
                }
                count += 1;
            }
        }

        if count > 0 {
            total_ndcg / count as f32
        } else {
            0.0
        }
    }
}
