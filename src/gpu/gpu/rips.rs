use anyhow::Result;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::compile_ptx;
#[cfg(feature = "cuda")]
use std::sync::Arc;

// Helper for Rips Complex structure
pub struct RipsComplex {
    pub adjacency: Vec<u8>, // N*N bitmap
    pub num_points: usize,
}

#[cfg(feature = "cuda")]
pub fn compute_distances_gpu(
    device: &Arc<CudaDevice>, 
    points: &[[f32; 3]], 
    threshold: f32
) -> Result<cudarc::driver::CudaSlice<u8>> {
    let n = points.len();
    if n == 0 {
         return device.alloc_zeros::<u8>(0).map_err(Into::into);
    }
    
    // 1. Upload points
    let points_flat: Vec<f32> = points.iter().flat_map(|p| p.as_slice()).cloned().collect();
    let d_points = device.htod_copy(points_flat)?;
    
    // 2. Allocate Edge Bitmap/List on GPU
    let mut d_adj = device.alloc_zeros::<u8>(n * n)?;

    // 3. Launch Distance Kernel
    // Note: We assume kernels/distance_matrix.cu is compiled or available. 
    // Since we wrote it to source, we compile on the fly using nvrtc.
    let ptx = compile_ptx(include_str!("kernels/distance_matrix.cu"))?;
    
    // Load PTX
    device.load_ptx(ptx, "distance_module", &["compute_distances"])?;
    let f = device.get_func("distance_module", "compute_distances").unwrap();

    let cfg = LaunchConfig::for_num_elems((n * n) as u32);
    unsafe { f.launch(cfg, (&d_points, &mut d_adj, n as i32, threshold)) }?;

    Ok(d_adj)
}

#[cfg(feature = "cuda")]
pub fn build_rips_complex_gpu(
    device: &Arc<CudaDevice>, 
    points: &[[f32; 3]], 
    threshold: f32
) -> Result<RipsComplex> {
    let n = points.len();
    let d_adj = compute_distances_gpu(device, points, threshold)?;
    
    // 4. Download Adjacency
    let adj_host = device.dtoh_sync_copy(&d_adj)?;
    
    Ok(RipsComplex {
        adjacency: adj_host,
        num_points: n,
    })
}

#[cfg(not(feature = "cuda"))]
pub fn build_rips_complex_gpu(
    _device: &(), // dummy
    _points: &[[f32; 3]], 
    _threshold: f32
) -> Result<RipsComplex> {
    anyhow::bail!("GPU acceleration not enabled. Compile with --features cuda")
}
