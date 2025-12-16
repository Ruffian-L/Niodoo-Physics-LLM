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
pub fn create_decomposer(boundary_matrix: Vec<Vec<usize>>) -> Box<dyn MatrixDecomposer> {
    #[cfg(feature = "cuda")]
    {
        println!("🚀 SPLATRAG: Initializing CUDA LoPhat Backend");
        Box::new(cuda::CudaDecomposer::new(boundary_matrix))
    }
    #[cfg(not(feature = "cuda"))]
    {
        // Only print ONCE per process to avoid spam in large loops
        use std::sync::Once;
        static START: Once = Once::new();
        START.call_once(|| {
            println!("🐢 SPLATRAG: Initializing CPU Fallback Backend (Serial)");
        });
        Box::new(cpu::CpuDecomposer::new(boundary_matrix))
    }
}
