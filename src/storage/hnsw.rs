use anyhow::{Context, Result};
use hnsw_rs::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct RealHnswIndex {
    // We assume Hnsw is serializable via the 'serde' feature in Cargo.toml
    // If not, we save the raw vectors and rebuild on load (safer fallback).
    id_map: HashMap<usize, u64>,
    // Hnsw struct itself isn't easily serializable in all versions.
    // Strategy: Serialize the data points, rebuild tree on load.
    // This is slower but robust.
    stored_vectors: Vec<(u64, Vec<f32>)>,

    #[serde(skip)]
    inner: Option<Hnsw<'static, f32, DistL2>>,
}

impl std::fmt::Debug for RealHnswIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RealHnswIndex")
            .field("count", &self.id_map.len())
            .finish()
    }
}

impl RealHnswIndex {
    pub fn new(max_elements: usize) -> Self {
        let inner = Hnsw::new(32, max_elements, 16, 200, DistL2 {});
        Self {
            inner: Some(inner),
            id_map: HashMap::new(),
            stored_vectors: Vec::new(),
        }
    }

    pub fn add(&mut self, splat_id: u64, embedding: &[f32]) -> Result<()> {
        let id = splat_id as usize;
        if let Some(hnsw) = &self.inner {
            hnsw.insert((embedding, id));
        }
        self.id_map.insert(id, splat_id);
        self.stored_vectors.push((splat_id, embedding.to_vec()));
        Ok(())
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        if let Some(hnsw) = &self.inner {
            hnsw.search(query, k, 30)
                .iter()
                .map(|n| (n.d_id as u64, n.distance))
                .collect()
        } else {
            vec![]
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let file = File::create(path).context("Failed to create index file")?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self).context("Failed to serialize index")?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path).context("Failed to open index file")?;
        let reader = BufReader::new(file);
        let mut index: Self =
            bincode::deserialize_from(reader).context("Failed to deserialize index")?;

        // Rebuild HNSW from stored vectors
        let max_elements = index.stored_vectors.len() + 1000;
        let hnsw = Hnsw::new(32, max_elements, 16, 200, DistL2 {});

        // Parallel insert if possible, otherwise sequential
        for (splat_id, vec) in &index.stored_vectors {
            hnsw.insert((vec.as_slice(), *splat_id as usize));
        }

        index.inner = Some(hnsw);
        Ok(index)
    }
}

pub type HnswIndex = RealHnswIndex;
