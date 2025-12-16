use anyhow::Result;

#[cfg(feature = "cuda")]
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::compile_ptx;
#[cfg(feature = "cuda")]
use std::sync::Arc;

// Helper for Rips Complex structure
pub struct RipsComplex {
    pub distances: Vec<f32>,
    pub num_points: usize,
}

#[cfg(feature = "cuda")]
pub fn compute_distances_gpu(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    points: &[[f32; 3]],
    threshold: f32,
) -> Result<CudaSlice<f32>> {
    let n = points.len();
    if n == 0 {
        return Ok(stream.alloc_zeros::<f32>(0)?);
    }

    // 1. Upload points
    let points_flat: Vec<f32> = points.iter().flat_map(|p| p.as_slice()).cloned().collect();
    let d_points = stream.memcpy_stod(&points_flat)?;

    // 2. Allocate Distance Matrix on GPU (float)
    let mut d_dists = stream.alloc_zeros::<f32>(n * n)?;

    // 3. Compile and load kernel
    let ptx = compile_ptx(include_str!("kernels/distance_matrix.cu"))?;
    let module: Arc<CudaModule> = ctx.load_module(ptx)?;
    let f: CudaFunction = module.load_function("compute_distances")?;

    // 4. Launch Kernel
    let cfg = LaunchConfig::for_num_elems((n * n) as u32);
    let n_i32 = n as i32;
    unsafe {
        stream
            .launch_builder(&f)
            .arg(&d_points)
            .arg(&mut d_dists)
            .arg(&n_i32)
            .arg(&threshold)
            .launch(cfg)?;
    }

    Ok(d_dists)
}

#[cfg(feature = "cuda")]
pub fn build_rips_complex_gpu(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    points: &[[f32; 3]],
    threshold: f32,
) -> Result<RipsComplex> {
    let n = points.len();
    let d_dists = compute_distances_gpu(ctx, stream, points, threshold)?;

    // Download Distances
    let dists_host = stream.memcpy_dtov(&d_dists)?;

    Ok(RipsComplex {
        distances: dists_host,
        num_points: n,
    })
}

#[cfg(not(feature = "cuda"))]
pub fn build_rips_complex_gpu(
    _ctx: &(),
    _stream: &(),
    _points: &[[f32; 3]],
    _threshold: f32,
) -> Result<RipsComplex> {
    anyhow::bail!("GPU acceleration not enabled. Compile with --features cuda")
}
