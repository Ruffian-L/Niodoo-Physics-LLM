#[cfg(feature = "cuda")]
use crate::gpu::GpuPhEngine;
use crate::indexing::persistent_homology::PersistenceDiagram;
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Topological Cognitive Signature (TCS)
///
/// Represents the topological structure of a cognitive state (memory cluster).
/// Replaces "magic numbers" with rigorous Betti number analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologicalCognitiveSignature {
    /// Betti numbers (b0, b1, b2, ...)
    /// b0: Connected components (Fragmentation)
    /// b1: Loops (Recursion/Cycles)
    /// b2: Voids (Missing Information/Unknowns)
    pub betti_numbers: Vec<usize>,

    /// Knot complexity (based on persistence lifetimes)
    pub knot_complexity: f32,

    /// Persistence entropy (measure of topological noise vs signal)
    pub persistence_entropy: f32,
}

impl TopologicalCognitiveSignature {
    pub fn new(betti_numbers: Vec<usize>, knot_complexity: f32, persistence_entropy: f32) -> Self {
        Self {
            betti_numbers,
            knot_complexity,
            persistence_entropy,
        }
    }

    /// Create TCS from a persistence diagram
    pub fn from_diagram(diagram: &PersistenceDiagram, max_dim: usize) -> Result<Self> {
        let mut betti_numbers = vec![0; max_dim + 1];
        let mut total_lifetime = 0.0;
        let mut entropy_sum = 0.0;

        // Filter noise: features with lifetime < threshold
        // This threshold should be dynamic or configurable
        let noise_threshold = 0.1;

        for (dim, features) in diagram.features_by_dim.iter().enumerate() {
            if dim > max_dim {
                continue;
            }

            let mut count = 0;
            let mut lifetimes = Vec::new();
            for (birth, death) in features {
                let lifetime = if *death == f32::INFINITY {
                    10.0 // Cap infinite lifetime for calculation
                } else {
                    death - birth
                };
                lifetimes.push(lifetime);

                if lifetime > noise_threshold {
                    count += 1;
                    total_lifetime += lifetime;
                }
            }
            // Sort descending
            lifetimes.sort_by(|a, b| b.partial_cmp(a).unwrap());
            // let top_10: Vec<_> = lifetimes.iter().take(10).collect();
            // println!("Dim {}: {} features total, {} > threshold. Top lifetimes: {:?}", dim, features.len(), count, top_10);

            betti_numbers[dim] = count;
        }

        // Calculate Persistence Entropy
        if total_lifetime > 0.0 {
            for features in &diagram.features_by_dim {
                for (birth, death) in features {
                    let lifetime = if *death == f32::INFINITY {
                        10.0
                    } else {
                        death - birth
                    };

                    if lifetime > noise_threshold {
                        let p = lifetime / total_lifetime;
                        entropy_sum -= p * p.ln();
                    }
                }
            }
        }

        // Knot complexity is a heuristic based on b1 and b2 interactions
        // For now, simple sum of lifetimes of higher dim features
        let knot_complexity = total_lifetime; // Simplified placeholder

        Ok(Self::new(betti_numbers, knot_complexity, entropy_sum))
    }

    /// Get b0 (Fragmentation)
    pub fn fragmentation(&self) -> usize {
        *self.betti_numbers.get(0).unwrap_or(&0)
    }

    /// Get b1 (Recursion)
    pub fn recursion(&self) -> usize {
        *self.betti_numbers.get(1).unwrap_or(&0)
    }

    /// Get b2 (Unknowns)
    pub fn unknowns(&self) -> usize {
        *self.betti_numbers.get(2).unwrap_or(&0)
    }
}

/// Engine for computing TCS from point clouds
pub struct TcsEngine {
    #[cfg(feature = "cuda")]
    gpu_engine: Option<GpuPhEngine>,
    max_dim: usize,
}

impl TcsEngine {
    pub fn new(max_dim: usize) -> Result<Self> {
        #[cfg(feature = "cuda")]
        let gpu_engine = if crate::gpu::should_use_gpu() {
            Some(GpuPhEngine::new(0, max_dim)?)
        } else {
            None
        };

        Ok(Self {
            #[cfg(feature = "cuda")]
            gpu_engine,
            max_dim,
        })
    }

    /// Compute TCS from a set of points (memory embeddings)
    pub fn compute_signature(&self, points: &[[f32; 3]]) -> Result<TopologicalCognitiveSignature> {
        #[cfg(feature = "cuda")]
        if let Some(engine) = &self.gpu_engine {
            let gpu_pd = engine.compute_persistence_gpu(points, 10.0)?;
            let diagram = PersistenceDiagram {
                dimension: gpu_pd.dimension,
                pairs: gpu_pd.pairs,
                features_by_dim: gpu_pd.features_by_dim,
            };
            return self.analyze_diagram(&diagram);
        }

        // Avoid unused variable warning
        let _ = points;

        // Fallback or error if GPU is required
        // For now, we'll return a dummy signature if no GPU
        // In production, we should have a CPU fallback or fail
        Ok(TopologicalCognitiveSignature::new(
            vec![0; self.max_dim + 1],
            0.0,
            0.0,
        ))
    }

    pub fn analyze_diagram(
        &self,
        diagram: &PersistenceDiagram,
    ) -> Result<TopologicalCognitiveSignature> {
        TopologicalCognitiveSignature::from_diagram(diagram, self.max_dim)
    }
}
