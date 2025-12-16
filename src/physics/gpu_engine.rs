use crate::config::HyperParameters;
use crate::physics::gaussian::SemanticGaussian;
use candle_core::{DType, Device, Result, Tensor};

pub struct GpuTissue {
    pub device: Device,
    pub means: Tensor, // [N, 384]
    pub ids: Vec<u64>,
}

impl GpuTissue {
    pub fn from_store(memories: &[SemanticGaussian]) -> Result<Self> {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        let n = memories.len();

        if n == 0 {
            return Ok(Self {
                device: device.clone(),
                means: Tensor::zeros((0, 384), DType::F32, &device)?,
                ids: vec![],
            });
        }

        let mut mean_data = Vec::with_capacity(n * 384);
        let mut ids = Vec::with_capacity(n);

        for mem in memories {
            // Nalgebra DVector to slice
            mean_data.extend_from_slice(mem.mean.as_slice());
            ids.push(mem.id);
        }

        let means = Tensor::from_vec(mean_data, (n, 384), &device)?;

        Ok(Self { device, means, ids })
    }

    pub fn query(
        &self,
        query: &SemanticGaussian,
        _params: &HyperParameters,
    ) -> anyhow::Result<Vec<(f32, u64)>> {
        if self.ids.is_empty() {
            return Ok(vec![]);
        }

        let q_vec = query.mean.as_slice().to_vec();
        let _q = Tensor::from_vec(q_vec, (384,), &self.device)?.unsqueeze(0)?;

        Ok(vec![])
    }
}
