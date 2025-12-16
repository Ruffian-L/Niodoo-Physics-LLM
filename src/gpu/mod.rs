#[cfg(feature = "cuda")]
pub mod context;
#[cfg(feature = "cuda")]
pub mod memory;

// Exposed regardless of GPU feature, handles CPU fallback internally
pub mod lophat;

#[cfg(feature = "cuda")]
pub mod rips;

#[cfg(test)]
mod test_integration;

use crate::tivm::SplatRagConfig;
use crate::types::SplatInput;
use anyhow::Result;

#[cfg(feature = "cuda")]
use cudarc::driver::CudaContext;
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// Check if CUDA is available on this system
#[cfg(feature = "cuda")]
pub fn cuda_available() -> bool {
    CudaContext::new(0).is_ok()
}

#[cfg(not(feature = "cuda"))]
pub fn cuda_available() -> bool {
    false
}

/// Determine if GPU acceleration is requested and available
pub fn should_use_gpu() -> bool {
    if !cfg!(feature = "cuda") {
        eprintln!("⚠️ GPU feature not compiled in");
        return false;
    }

    match std::env::var("SPLATRAG_USE_GPU") {
        Ok(val) if matches!(val.as_str(), "1" | "true" | "TRUE" | "yes" | "YES") => {
            let available = cuda_available();
            if available {
                eprintln!("🚀 GPU ACCELERATION ENABLED - CUDA device available");
            } else {
                eprintln!("⚠️ GPU requested but CUDA not available");
            }
            available
        }
        _ => {
            eprintln!("ℹ️ GPU not requested (set SPLATRAG_USE_GPU=1 to enable)");
            false
        }
    }
}

#[cfg(feature = "cuda")]
pub fn try_gpu_fingerprint(_splat: &SplatInput, _cfg: &SplatRagConfig) -> Result<()> {
    // Legacy function removed
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub fn try_gpu_fingerprint(_splat: &SplatInput, _cfg: &SplatRagConfig) -> Result<()> {
    anyhow::bail!("GPU acceleration feature not enabled");
}

/// Get the number of available CUDA devices
#[cfg(feature = "cuda")]
pub fn device_count() -> Result<usize> {
    Ok(CudaContext::device_count()? as usize)
}

#[cfg(not(feature = "cuda"))]
pub fn device_count() -> Result<usize> {
    Ok(0)
}

#[cfg(feature = "cuda")]
/// GPU-accelerated persistent homology engine
pub struct GpuPhEngine {
    context: Arc<context::GpuContext>,
    max_dim: usize,
}

#[cfg(feature = "cuda")]
impl GpuPhEngine {
    /// Create a new GPU-accelerated engine
    pub fn new(device_id: usize, max_dim: usize) -> Result<Self> {
        let context = Arc::new(context::GpuContext::new(device_id)?);
        Ok(Self { context, max_dim })
    }

    /// Compute persistent homology on GPU
    pub fn compute_persistence_gpu(
        &self,
        points: &[[f32; 3]],
        _threshold_arg: f32,
    ) -> Result<PersistenceDiagram> {
        // 1. Build Distance Matrix (f32)
        let threshold = 10.0f32;
        let d_dists = rips::compute_distances_gpu(
            &self.context.ctx,
            &self.context.stream,
            points,
            threshold,
        )?;

        // 2. Download to CPU for filtration sort
        let dists = self.context.stream.memcpy_dtov(&d_dists)?;
        let n = points.len();

        let mut edges = Vec::with_capacity(n * n / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let d = dists[i * n + j];
                if d <= threshold {
                    edges.push((d, i, j));
                }
            }
        }

        // CRITICAL: Sort by distance to create valid filtration
        edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // 3. Build Boundary Matrix (Sparse)
        let mut boundary_matrix = Vec::with_capacity(n + edges.len());
        // Points (0-dim) have empty boundary
        for _ in 0..n {
            boundary_matrix.push(vec![]);
        }
        // Edges (1-dim) have [u, v] boundary
        for (_, u, v) in &edges {
            let (min, max) = if u < v { (*u, *v) } else { (*v, *u) };
            boundary_matrix.push(vec![min, max]);
        }

        // 4. Reduce on GPU (Matrix Reduction)
        use crate::gpu::lophat::create_decomposer;
        let mut decomposer = create_decomposer(boundary_matrix, true, 10000);
        decomposer.reduce();

        // 5. Extract Lifetimes
        let mut pd = PersistenceDiagram::new(self.max_dim);

        // Iterate edge columns (indices n to end)
        for (edge_idx, (death_time, _, _)) in edges.iter().enumerate() {
            let col_idx = n + edge_idx;
            if let Some(row_idx) = decomposer.get_pivot(col_idx) {
                // Edge killed a component (H0 death)
                if row_idx < n {
                    pd.add_pair(0.0, *death_time);
                }
            } else {
                // Edge created a loop (H1 birth)
                pd.add_pair_with_dim(*death_time, f32::INFINITY, 1);
            }
        }

        Ok(pd)
    }
}

#[derive(Debug, Clone)]
pub struct PersistenceDiagram {
    pub dimension: usize,
    pub pairs: Vec<(f32, f32)>,                // (birth, death)
    pub features_by_dim: Vec<Vec<(f32, f32)>>, // Index k contains pairs for dimension k
}

impl PersistenceDiagram {
    pub fn new(dim: usize) -> Self {
        Self {
            dimension: dim,
            pairs: Vec::new(),
            features_by_dim: vec![Vec::new(); dim + 1],
        }
    }

    pub fn add_pair(&mut self, birth: f32, death: f32) {
        self.pairs.push((birth, death));
        if !self.features_by_dim.is_empty() {
            self.features_by_dim[0].push((birth, death));
        }
    }

    pub fn add_pair_with_dim(&mut self, birth: f32, death: f32, dim: usize) {
        self.pairs.push((birth, death));
        if dim < self.features_by_dim.len() {
            self.features_by_dim[dim].push((birth, death));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_availability() {
        let available = cuda_available();
        println!("CUDA available: {}", available);
        if available {
            let count = device_count().unwrap();
            println!("Found {} CUDA device(s)", count);
        }
    }
}
