/// Common interface for Matrix Reduction (CPU or GPU)
pub trait MatrixDecomposer {
    /// Adds column `source_idx` to `target_idx` (Mod 2 arithmetic)
    fn add_entries(&mut self, target_idx: usize, source_idx: usize);
    /// Returns the pivot (lowest non-zero row index) for a column, or None if empty
    fn get_pivot(&self, col_idx: usize) -> Option<usize>;
    /// Returns the non-zero indices of the reduced column R[col_idx]
    fn get_r_col(&self, col_idx: usize) -> Vec<usize>;

    /// Runs the full reduction (if the backend requires a batch run)
    fn reduce(&mut self);
}

// ------------------------------------------------------------------
// MODULE SELECTION
// ------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub mod cuda;

pub mod cpu;

// Factory to get the correct backend
pub fn create_decomposer(
    boundary_matrix: Vec<Vec<usize>>,
    gpu_enabled: bool,
    heap_capacity: usize,
) -> Box<dyn MatrixDecomposer> {
    #[cfg(feature = "cuda")]
    {
        if gpu_enabled {
            println!("🚀 SPLATRAG: Initializing CUDA LoPhat Backend");
            match cuda::CudaDecomposer::new(boundary_matrix.clone(), heap_capacity) {
                Ok(decomposer) => return Box::new(decomposer),
                Err(e) => {
                    eprintln!("⚠️ CUDA Init Failed: {}. Falling back to CPU.", e);
                }
            }
        } else {
            println!("🐢 SPLATRAG: GPU Disabled by Config. Using CPU Backend.");
        }
    }

    // Fallback or if CUDA disabled
    #[cfg(not(feature = "cuda"))]
    {
        println!("🐢 SPLATRAG: CUDA Feature Not Enabled. Using CPU Backend.");
    }

    Box::new(cpu::CpuDecomposer::new(boundary_matrix))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_fallback_config() {
        let matrix = vec![vec![]; 5];
        // Test with GPU disabled
        println!("Testing GPU Disabled:");
        let _decomposer = create_decomposer(matrix.clone(), false, 1024);

        // Test with GPU enabled (might fail if no GPU, but should try)
        println!("Testing GPU Enabled:");
        let _decomposer = create_decomposer(matrix, true, 1024);
    }
}
