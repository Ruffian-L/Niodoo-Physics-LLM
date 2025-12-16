pub mod fingerprint;
pub mod persistent_homology;
pub mod tcs;
pub mod text_index;
pub mod vectorize; // Added text_index module

pub use fingerprint::{fingerprint_from_splat, FingerprintConfig, TopologicalFingerprint}; // Re-export from fingerprint.rs
pub use persistent_homology::{PersistenceDiagram, PhConfig, PhEngine, PhStrategy};
pub use tcs::{TcsEngine, TopologicalCognitiveSignature};
pub use text_index::TantivyIndex;
pub use vectorize::vector_persistence_block; // Re-export TantivyIndex

use anyhow::Result;

// Removed duplicate definition of TopologicalFingerprint
// It is now defined in src/indexing/fingerprint.rs and re-exported

pub struct ZigZagPH {
    _config: ZigZagConfig,
    points: Vec<nalgebra::Point3<f32>>,
}

#[derive(Debug, Clone)]
pub struct ZigZagConfig {
    pub max_dimension: usize,
    pub threshold: f32,
}

impl Default for ZigZagConfig {
    fn default() -> Self {
        Self {
            max_dimension: 2,
            threshold: 1.0,
        }
    }
}

impl ZigZagPH {
    pub fn new() -> Self {
        Self {
            _config: ZigZagConfig::default(),
            points: Vec::new(),
        }
    }

    pub fn with_config(config: ZigZagConfig) -> Self {
        Self {
            _config: config,
            points: Vec::new(),
        }
    }

    pub fn compute_persistent_homology(
        &mut self,
        point_cloud: &[nalgebra::Point3<f32>],
    ) -> Result<TopologicalFingerprint> {
        // Update internal state
        self.points = point_cloud.to_vec();

        // Convert nalgebra points to [f32; 3]
        let points: Vec<[f32; 3]> = self.points.iter().map(|p| [p.x, p.y, p.z]).collect();

        // Use PhEngine for real computation
        let engine = PhEngine::new(PhConfig {
            hom_dims: (0..=self._config.max_dimension).collect(),
            strategy: PhStrategy::ExactBatch,
            max_points: 1000,
            connectivity_threshold: 5.0,
            max_dimension: self._config.max_dimension,
            gpu_enabled: true,
            gpu_heap_capacity: 256 * 1024 * 1024,
        });

        let pd = engine.compute_pd(&points);

        // Convert PersistenceDiagram to TopologicalFingerprint
        let h0 = pd.features_by_dim.get(0).cloned().unwrap_or_default();
        let h1 = pd.features_by_dim.get(1).cloned().unwrap_or_default();

        Ok(TopologicalFingerprint::new(h0, h1))
    }

    pub fn update_with_insertion(
        &mut self,
        fingerprint: &mut TopologicalFingerprint,
        point: nalgebra::Point3<f32>,
    ) -> Result<()> {
        self.points.push(point);

        // Recompute full homology (Correctness over Speed for now)
        // In a real ZigZag implementation, we would update the filtration locally.
        let new_fp = self.compute_persistent_homology(&self.points.clone())?;

        *fingerprint = new_fp;
        Ok(())
    }

    pub fn update_with_deletion(
        &mut self,
        fingerprint: &mut TopologicalFingerprint,
        index: usize,
    ) -> Result<()> {
        if index < self.points.len() {
            self.points.remove(index);
            let new_fp = self.compute_persistent_homology(&self.points.clone())?;
            *fingerprint = new_fp;
            Ok(())
        } else {
            anyhow::bail!("Index out of bounds")
        }
    }
}

impl Default for ZigZagPH {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zigzag_creation() {
        let zz = ZigZagPH::new();
        assert_eq!(zz._config.max_dimension, 2);
    }
}
