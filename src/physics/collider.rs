use candle_core::{Result, Tensor};
use hnsw_rs::prelude::*;
use std::sync::{Arc, RwLock};

pub struct SemanticCollider {
    // Note: Hnsw in 0.3.3 might not use lifetimes in typical usage, but if compilation fails,
    // we check recent updates.
    // If the error was `expected named lifetime parameter`, it implies `Hnsw<'a, ...>`
    // Let's assume standard usage `Hnsw<f32, DistL2>` works if we don't alias it weirdly.
    // However, if the compiler insists, we'll try `'static` because `DistL2` is unit struct.
    // Wait, the error `expected named lifetime parameter` appeared on `Hnsw<f32, DistL2>`.
    // Let's look at `hnsw_rs` docs or source if we could.
    // Since we can't, let's try assuming it takes a lifetime for the Dist?
    // Actually, maybe `DistL2` is fine.
    // Let's try adding `'static` lifetime to Hnsw.
    pub index: Arc<RwLock<Hnsw<'static, f32, DistL2>>>,
    pub dim: usize,
}

impl SemanticCollider {
    pub fn new(dim: usize) -> Self {
        let max_nb_connection = 16;
        let max_elements = 100_000;
        let max_layer = 16;
        let ef_construction = 200;

        let index = Hnsw::new(
            max_nb_connection,
            max_elements,
            max_layer,
            ef_construction,
            DistL2 {},
        );

        Self {
            index: Arc::new(RwLock::new(index)),
            dim,
        }
    }

    pub fn update_positions(&self, tokens: &[u32], vectors: &Tensor) -> Result<()> {
        let vec_data: Vec<Vec<f32>> = vectors.to_vec2()?;
        let mut index = self.index.write().unwrap();

        // Use parallel_insert
        let data_with_ids: Vec<(&Vec<f32>, usize)> = vec_data
            .iter()
            .zip(tokens.iter().map(|&t| t as usize))
            .collect();
        index.parallel_insert(&data_with_ids);

        Ok(())
    }

    pub fn find_interacting_pairs(&self, vector: &[f32]) -> Vec<usize> {
        let index = self.index.read().unwrap();
        let neighbors = index.search(vector, 10, 20);
        neighbors.iter().map(|n| n.d_id).collect()
    }
}
