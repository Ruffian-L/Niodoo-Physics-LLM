//! CUDA context management and device memory allocation

use anyhow::{Result, Context};
use cudarc::driver::{CudaDevice, CudaSlice};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// GPU context managing device and persistent allocations
pub struct GpuContext {
    pub device: Arc<CudaDevice>,
    
    // Pre-allocated buffers for reuse
    pub heap: GpuHeap,
    
    // Compiled kernels
    pub kernels: KernelCache,
}

impl GpuContext {
    /// Create a new GPU context on the specified device
    pub fn new(device_id: usize) -> Result<Self> {
        // CudaDevice::new already returns Arc<CudaDevice>
        let device = CudaDevice::new(device_id)
            .context("Failed to initialize CUDA device")?;
        
        // Pre-allocate 1GB heap for sparse matrix operations
        let heap = GpuHeap::new(Arc::clone(&device), 1 << 30)?;
        
        // Compile and cache kernels
        let kernels = KernelCache::new(Arc::clone(&device))?;
        
        Ok(Self {
            device,
            heap,
            kernels,
        })
    }
    
    /// Get device properties
    pub fn device_info(&self) -> DeviceInfo {
        // This would query device properties via cudarc
        DeviceInfo {
            name: "NVIDIA GPU".to_string(),
            compute_capability: (8, 6), // Example: Ampere
            memory_gb: 24,
            sm_count: 84,
        }
    }
}

/// GPU memory heap for dynamic allocations
#[allow(dead_code)]
pub struct GpuHeap {
    device: Arc<CudaDevice>,
    
    // Main heap buffer
    pub data: CudaSlice<u8>,
    
    // Allocation pointer (atomic on device)
    pub alloc_ptr: CudaSlice<u32>,
    
    total_size: usize,
}

impl GpuHeap {
    pub fn new(device: Arc<CudaDevice>, size: usize) -> Result<Self> {
        let data = device.alloc_zeros::<u8>(size)?;
        let alloc_ptr = device.alloc_zeros::<u32>(1)?;
        
        Ok(Self {
            device,
            data,
            alloc_ptr,
            total_size: size,
        })
    }
    
    /// Reset heap to empty
    pub fn reset(&mut self) -> Result<()> {
        // Reset allocation pointer to 0
        let zero = vec![0u32; 1];
        self.device.htod_sync_copy_into(&zero, &mut self.alloc_ptr)?;
        Ok(())
    }
}

/// Cache of compiled CUDA kernels
#[allow(dead_code)]
pub struct KernelCache {
    device: Arc<CudaDevice>,
    
    // Compiled PTX modules
    pub apparent_pairs_ptx: Option<Ptx>,
    pub lock_free_ptx: Option<Ptx>,
}

impl KernelCache {
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        // Kernels will be compiled on first use
        Ok(Self {
            device,
            apparent_pairs_ptx: None,
            lock_free_ptx: None,
        })
    }
    
    /// Compile and cache the apparent pairs kernel
    pub fn compile_apparent_pairs(&mut self) -> Result<()> {
        if self.apparent_pairs_ptx.is_some() {
            return Ok(());
        }
        
        let kernel_src = include_str!("kernels/apparent_pairs.cu");
        let ptx = cudarc::nvrtc::compile_ptx(kernel_src)?;
        self.apparent_pairs_ptx = Some(ptx);
        Ok(())
    }
}

#[derive(Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub compute_capability: (u32, u32),
    pub memory_gb: usize,
    pub sm_count: usize,
}
