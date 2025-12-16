use super::MatrixDecomposer;
use anyhow::{Context, Result};
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

// We use a flattened Compressed Sparse Row (CSR) format for the GPU
// It's much faster than pointer chasing on a 5080.
pub struct CudaDecomposer {
    device: Arc<CudaDevice>,
    // We keep these CPU-side for quick lookups if the GPU is busy
    cpu_fallback_cache: Option<Vec<Vec<usize>>>, 
    num_cols: usize,
    num_rows: usize,
}

impl CudaDecomposer {
    pub fn new(boundary_matrix: Vec<Vec<usize>>) -> Self {
        let dev = CudaDevice::new(0).expect("Failed to initialize CUDA device. Check drivers.");
        
        // Load the PTX (compiled CUDA code)
        // We assume build.rs compiles 'kernels/reduce.cu' to 'reduce.ptx'
        dev.load_ptx(Ptx::from_file("./target/nvptx/reduce.ptx"), "persistence", &["reduce_kernel"])
            .expect("Failed to load CUDA kernel");

        let rows = boundary_matrix.len(); // logic approximation
        let cols = boundary_matrix.len();

        Self {
            device: dev,
            cpu_fallback_cache: Some(boundary_matrix), // Keep copy for now
            num_cols: cols,
            num_rows: rows,
        }
    }

    /// Flattens the matrix and sends it to the GPU
    fn upload_matrix(&self) -> Result<(cudarc::driver::CudaSlice<usize>, cudarc::driver::CudaSlice<usize>)> {
        let matrix = self.cpu_fallback_cache.as_ref()
            .ok_or_else(|| anyhow::anyhow!("CPU fallback cache not initialized"))?;
        
        let mut col_ptr = Vec::with_capacity(self.num_cols + 1);
        let mut row_indices = Vec::new();
        
        let mut current_ptr = 0;
        col_ptr.push(current_ptr);

        for col in matrix {
            for &row_idx in col {
                row_indices.push(row_idx);
                current_ptr += 1;
            }
            col_ptr.push(current_ptr);
        }

        let dev_col_ptr = self.device.htod_copy(col_ptr)?;
        let dev_row_idx = self.device.htod_copy(row_indices)?;

        Ok((dev_col_ptr, dev_row_idx))
    }
}

impl MatrixDecomposer for CudaDecomposer {
    fn add_entries(&mut self, _target: usize, _source: usize) {
        // On GPU, we don't do single adds. We batch reduce.
    }

    fn get_pivot(&self, col_idx: usize) -> Option<usize> {
        // In a real high-perf scenario, we'd read this from a simplified array on GPU
        // For now, read from cache
        self.cpu_fallback_cache.as_ref()?[col_idx].last().copied()
    }

    fn get_r_col(&self, col_idx: usize) -> Vec<usize> {
        // In production: Copy back specific slice from GPU
        self.cpu_fallback_cache.as_ref()
            .and_then(|cache| cache.get(col_idx))
            .cloned()
            .unwrap_or_default()
    }
    
    fn reduce(&mut self) {
        println!("⚡ 5080-Q: Dispatching Reduction Kernel...");
        
        // 1. Upload Data
        let (mut d_col_ptr, mut d_row_idx) = self.upload_matrix().unwrap();
        
        // 2. Allocate Output Buffer (Pivots)
        let mut d_pivots = self.device.alloc_zeros::<isize>(self.num_cols).unwrap();

        // 3. Launch Config
        let cfg = LaunchConfig::for_num_elems(self.num_cols as u32);
        let func = self.device.get_func("persistence", "reduce_kernel").unwrap();

        // 4. FIRE
        // Params: (col_ptr, row_idx, pivots, num_cols)
        unsafe { func.launch(cfg, (&mut d_col_ptr, &mut d_row_idx, &mut d_pivots, self.num_cols)) }
            .map_err(|e| anyhow::anyhow!("CUDA kernel launch failed: {}", e))?;

        // 5. Sync (Wait for the 5080 to chew through the topology)
        self.device.synchronize().unwrap();
        
        println!("⚡ 5080-Q: Reduction Complete.");
        
        // TODO: Pull back d_row_idx into self.cpu_fallback_cache to update the host
    }
}









