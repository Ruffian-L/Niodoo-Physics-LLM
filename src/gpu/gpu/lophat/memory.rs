//! Memory management for GPU LoPHAT
//! 
//! Handles the "Hybrid Heap" and other memory structures required for the lock-free algorithm.

use anyhow::Result;
use cudarc::driver::{CudaDevice, CudaSlice};
use std::sync::Arc;

/// A paged heap allocator on the GPU
#[allow(dead_code)]
pub struct GpuHeap {
    device: Arc<CudaDevice>,
    pub data: CudaSlice<i32>, // The heap itself (indices)
    pub head: CudaSlice<i32>, // Atomic counter for allocation
    pub capacity: usize,
}

impl GpuHeap {
    pub fn new(device: Arc<CudaDevice>, size_elems: usize) -> Result<Self> {
        let data = device.alloc_zeros::<i32>(size_elems)?;
        let head = device.alloc_zeros::<i32>(1)?;
        
        Ok(Self {
            device,
            data,
            head,
            capacity: size_elems,
        })
    }
}
