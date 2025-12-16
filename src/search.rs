use crate::config::{HyperParameters, SplatMemoryConfig};
use crate::embeddings::EmbeddingModel;
use crate::ingest::shaper::Shaper;
// Fixed Import
use crate::physics::gpu_engine::GpuTissue;
use crate::storage::memory::{InMemoryBlobStore, TopologicalMemoryStore};
use crate::structs::SplatManifest;
use anyhow::Result;
use clap::ValueEnum;
use serde::Serialize;
use std::path::Path;

#[derive(Copy, Clone, ValueEnum)]
pub enum SearchMode {
    Focus,
    Rainbow,
}

#[derive(Serialize)]
pub struct SearchResult {
    pub id: u64,
    pub text: String,
    pub score: f32,
}

pub struct Searcher {
    pub store: TopologicalMemoryStore<InMemoryBlobStore>,
    pub manifest: SplatManifest,
    pub model: EmbeddingModel,
    pub gpu_brain: Option<GpuTissue>,
}

use crate::tivm::SplatRagConfig;

impl Searcher {
    pub fn new(config: SplatMemoryConfig, index_path: &Path) -> Result<Self> {
        // Load Store
        let geom_path = index_path.join("mindstream_current.geom");
        let sem_path = index_path.join("mindstream_current.sem");
        let manifest_path = index_path.join("chaos_manifest.bin");

        println!("Loading store from {:?}", index_path);
        let mut store = TopologicalMemoryStore::load_from_split_files(
            &geom_path,
            &sem_path,
            SplatRagConfig::default(),
            InMemoryBlobStore::default(),
        )?;

        // Load Manifest
        println!("Loading manifest from {:?}", manifest_path);
        let manifest_file = std::fs::File::open(manifest_path)?;
        let manifest: SplatManifest = bincode::deserialize_from(manifest_file)?;

        // Load Model
        println!("Loading model...");
        let model = EmbeddingModel::with_dim(crate::constants::EMBED_DIM)?;

        // Convert to SemanticGaussians and Load GPU Brain
        println!("Constructing SemanticGaussians for GPU...");
        let mut memories = Vec::new();
        let manifest_map = manifest.to_map();
        let shaper = Shaper::new(&model);

        let total = store.len();
        // Use entries_mut as it is the only way to iterate currently exposed
        for (i, (id, _entry)) in store.entries_mut().iter().enumerate() {
            if i % 1000 == 0 {
                println!("Processed {}/{} memories...", i, total);
            }

            if let Some(text) = manifest_map.get(id) {
                // Reconstruct SemanticGaussian using Shaper
                // This re-embeds text. Expensive on startup but correct for V2.
                if let Ok(gaussian) = shaper.shape(text, *id) {
                    memories.push(gaussian);
                }
            }
        }

        let gpu_brain = if !memories.is_empty() {
            println!("Uploading to GPU...");
            Some(GpuTissue::from_store(&memories)?)
        } else {
            None
        };

        if let Some(brain) = &gpu_brain {
            println!(
                "GPU Brain online: {} memories in VRAM",
                brain.means.dims()[0]
            );
        }

        Ok(Self {
            store,
            manifest,
            model,
            gpu_brain,
        })
    }

    pub fn search(
        &self,
        query_text: &str,
        _mode: SearchMode,
        _threshold: Option<f32>,
        params: &HyperParameters,
    ) -> Result<Vec<SearchResult>> {
        // 1. Shape Query
        let shaper = Shaper::new(&self.model);
        // Use dummy ID 0 for query
        let query_gaussian = shaper.shape(query_text, 0)?;

        // 2. GPU Query
        if let Some(brain) = &self.gpu_brain {
            let scores = brain.query(&query_gaussian, params)?;

            // Map back to results
            let manifest_map = self.manifest.to_map();
            let results = scores
                .into_iter()
                .map(|(score, id)| {
                    let text = manifest_map.get(&id).cloned().unwrap_or_default();
                    SearchResult { id, text, score }
                })
                .collect();

            Ok(results)
        } else {
            Ok(vec![])
        }
    }
}
