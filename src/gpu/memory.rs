//! GPU memory management for sparse matrices and dynamic allocations

use anyhow::Result;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use std::sync::Arc;

/// Sparse matrix in CSC format on GPU
pub struct GpuSparseMatrix {
    pub col_ptr: CudaSlice<u32>, // Column pointers
    pub row_idx: CudaSlice<u32>, // Row indices
    pub num_cols: usize,
    pub num_nonzeros: usize,
}

impl GpuSparseMatrix {
    /// Upload a sparse matrix from host to device
    pub fn from_host(stream: &Arc<CudaStream>, col_ptr: &[u32], row_idx: &[u32]) -> Result<Self> {
        let d_col_ptr = stream.memcpy_stod(col_ptr)?;
        let d_row_idx = stream.memcpy_stod(row_idx)?;

        Ok(Self {
            col_ptr: d_col_ptr,
            row_idx: d_row_idx,
            num_cols: col_ptr.len() - 1,
            num_nonzeros: row_idx.len(),
        })
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        // Estimate based on number of columns and nonzeros
        (self.num_cols + 1) * 4 + self.num_nonzeros * 4
    }
}

/// Result of persistent homology computation on GPU
pub struct GpuPersistenceResult {
    pub pivots: CudaSlice<i32>,
    pub pairs: Vec<(u32, u32)>, // (birth_idx, death_idx)
}

impl GpuPersistenceResult {
    /// Download results from GPU to host
    pub fn to_host(&self, stream: &Arc<CudaStream>) -> Result<Vec<i32>> {
        Ok(stream.memcpy_dtov(&self.pivots)?)
    }
}

/// Memory pool for dynamic allocations during reduction
pub struct MemoryPool {
    chunks: Vec<CudaSlice<u32>>,
    chunk_size: usize,
    ctx: Arc<CudaContext>,
}

impl MemoryPool {
    pub fn new(ctx: Arc<CudaContext>, chunk_size: usize) -> Self {
        Self {
            chunks: Vec::new(),
            chunk_size,
            ctx,
        }
    }

    /// Allocate a new chunk if needed
    pub fn ensure_capacity(&mut self, stream: &Arc<CudaStream>, required: usize) -> Result<()> {
        let current_capacity = self.chunks.len() * self.chunk_size;
        if current_capacity < required {
            let new_chunk = stream.alloc_zeros::<u32>(self.chunk_size)?;
            self.chunks.push(new_chunk);
        }
        Ok(())
    }
}
