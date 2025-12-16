use super::MatrixDecomposer;
use anyhow::{anyhow, Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::Arc;

// Shared CUDA context to avoid context creation overhead and thread safety issues
static CUDA_CONTEXT: Mutex<Option<Arc<CudaContext>>> = Mutex::new(None);
static CUDA_MODULE: Mutex<Option<Arc<CudaModule>>> = Mutex::new(None);

fn get_cuda_context() -> Result<Arc<CudaContext>> {
    let mut guard = CUDA_CONTEXT.lock();
    if let Some(ctx) = guard.as_ref() {
        return Ok(ctx.clone());
    }

    // Initialize context (default to device 0)
    let ctx = CudaContext::new(0).context("Failed to initialize CUDA context for device 0")?;

    *guard = Some(ctx.clone());
    Ok(ctx)
}

fn get_cuda_module(ctx: &Arc<CudaContext>) -> Result<Arc<CudaModule>> {
    let mut guard = CUDA_MODULE.lock();
    if let Some(module) = guard.as_ref() {
        return Ok(module.clone());
    }

    // Load PTX - try multiple paths for robustness
    let ptx_paths = [
        "./target/nvptx/reduce.ptx",
        "../target/nvptx/reduce.ptx",
        "/usr/local/share/splatrag/reduce.ptx",
    ];

    for path_str in &ptx_paths {
        let path = PathBuf::from(path_str);
        if path.exists() {
            let ptx = Ptx::from_file(&path);
            let module = ctx
                .load_module(ptx)
                .context(format!("Failed to load PTX from {:?}", path))?;
            *guard = Some(module.clone());
            return Ok(module);
        }
    }

    Err(anyhow!(
        "Could not find reduce.ptx in any standard location. Please ensure kernels are compiled."
    ))
}

// We use a flattened Compressed Sparse Row (CSR) format for the GPU
pub struct CudaDecomposer {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    boundary_matrix: Vec<Vec<usize>>,
    num_cols: usize,
    num_rows: usize,
    pivots: Vec<Option<usize>>,
    heap_capacity: usize,
}

impl CudaDecomposer {
    pub fn new(boundary_matrix: Vec<Vec<usize>>, heap_capacity: usize) -> Result<Self> {
        let ctx = get_cuda_context()?;
        let stream = ctx.default_stream();
        let module = get_cuda_module(&ctx)?;
        let num_cols = boundary_matrix.len();
        let num_rows = boundary_matrix
            .iter()
            .map(|col| col.iter().max().copied().unwrap_or(0))
            .max()
            .unwrap_or(0)
            + 1;

        Ok(Self {
            ctx,
            stream,
            module,
            boundary_matrix,
            num_cols,
            num_rows,
            pivots: vec![None; num_cols],
            heap_capacity,
        })
    }

    /// Flattens the matrix and sends it to the GPU
    fn upload_matrix(&self) -> Result<(CudaSlice<usize>, CudaSlice<usize>)> {
        let matrix = &self.boundary_matrix;

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

        let dev_col_ptr = self
            .stream
            .memcpy_stod(&col_ptr)
            .context("Failed to copy col_ptr to GPU")?;
        let dev_row_idx = self
            .stream
            .memcpy_stod(&row_indices)
            .context("Failed to copy row_indices to GPU")?;

        Ok((dev_col_ptr, dev_row_idx))
    }
}

impl MatrixDecomposer for CudaDecomposer {
    fn add_entries(&mut self, _target: usize, _source: usize) {
        // On GPU, we don't do single adds. We batch reduce.
    }

    fn get_pivot(&self, col_idx: usize) -> Option<usize> {
        // Return the cached pivot from GPU reduction
        if col_idx < self.pivots.len() {
            self.pivots[col_idx]
        } else {
            None
        }
    }

    fn get_r_col(&self, col_idx: usize) -> Vec<usize> {
        // In production: Copy back specific slice from GPU
        // For now, fallback to cache (Note: This is unreduced! But get_pivot is correct)
        if col_idx < self.boundary_matrix.len() {
            self.boundary_matrix[col_idx].clone()
        } else {
            vec![]
        }
    }

    fn reduce(&mut self) {
        println!("⚡ 5080-Q: Dispatching Reduction Kernel...");

        // 1. Upload Data
        let (d_col_ptr, d_row_idx) = match self.upload_matrix() {
            Ok(res) => res,
            Err(e) => {
                eprintln!("GPU Upload Failed: {}. Aborting reduction.", e);
                return;
            }
        };

        // 2. Allocate Output Buffer (Pivots)
        let mut d_pivots: CudaSlice<i32> = match self.stream.alloc_zeros::<i32>(self.num_cols) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("GPU Alloc Failed (pivots): {}.", e);
                return;
            }
        };

        // Initialize with -1
        let init_pivots = vec![-1i32; self.num_cols];
        if let Err(e) = self.stream.memcpy_htod(&init_pivots, &mut d_pivots) {
            eprintln!("GPU Memcpy Failed (pivots): {}.", e);
            return;
        }

        // Allocate auxiliary buffers
        let d_is_cleared: CudaSlice<u8> = match self.stream.alloc_zeros::<u8>(self.num_cols) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("GPU Alloc Failed (is_cleared): {}", e);
                return;
            }
        };

        // Dynamic Heap Sizing
        let max_heap_elements = 250_000_000usize;
        let calculated_capacity = self.num_cols.saturating_mul(1000);
        let heap_capacity = calculated_capacity.min(max_heap_elements).max(1024 * 1024);

        let mut d_heap: CudaSlice<i32> = match self.stream.alloc_zeros::<i32>(heap_capacity) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("GPU Alloc Failed (heap): {}", e);
                return;
            }
        };
        let mut d_heap_ptr: CudaSlice<i32> = match self.stream.alloc_zeros::<i32>(1) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("GPU Alloc Failed (heap_ptr): {}", e);
                return;
            }
        };

        // 3. Get function from module
        let func: CudaFunction = match self.module.load_function("lock_free_reduction") {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Kernel 'lock_free_reduction' not found: {}", e);
                return;
            }
        };

        // 4. Launch Config
        let cfg = LaunchConfig::for_num_elems(self.num_cols as u32);

        // 5. FIRE using Builder API
        {
            let ncols_i32 = self.num_cols as i32;
            let heap_cap_i32 = heap_capacity as i32;

            let launch_result = unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(&mut d_pivots)
                    .arg(&d_col_ptr)
                    .arg(&d_row_idx)
                    .arg(&d_is_cleared)
                    .arg(&mut d_heap)
                    .arg(&mut d_heap_ptr)
                    .arg(&ncols_i32)
                    .arg(&heap_cap_i32)
                    .launch(cfg)
            };

            if let Err(e) = launch_result {
                eprintln!("Kernel Launch Failed: {}", e);
                return;
            }
        }

        // 6. Sync
        if let Err(e) = self.stream.synchronize() {
            eprintln!("GPU Sync Failed: {}", e);
            return;
        }

        println!("⚡ 5080-Q: Reduction Complete.");

        // 7. Download Pivots
        let raw_pivots: Vec<i32> = match self.stream.memcpy_dtov(&d_pivots) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("GPU Download Failed: {}", e);
                return;
            }
        };

        // Update host cache
        for (i, &p) in raw_pivots.iter().enumerate() {
            if p >= 0 {
                self.pivots[i] = Some(p as usize);
            } else {
                self.pivots[i] = None;
            }
        }
    }
}
