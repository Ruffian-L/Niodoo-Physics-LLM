//! src/adapter.rs
use candle_core::{Result, Tensor};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, Module, VarBuilder};

pub const TOPOLOGICAL_CENTROID_DIM: usize = 768;
pub const TOPOLOGICAL_COVARIANCE_DIM: usize = 0;
pub const TOTAL_INPUT_DIM: usize = TOPOLOGICAL_CENTROID_DIM + TOPOLOGICAL_COVARIANCE_DIM;

#[derive(Clone, Debug)]
pub struct SplatAdapter {
    pub layer0: Linear,
    pub norm1: LayerNorm,
    pub layer3: Linear,
    pub norm4: LayerNorm,
    pub layer6: Linear,
}

impl SplatAdapter {
    pub fn new(hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        // Architecture aligned with train_synapse_mlp.py
        // Input(768) -> Linear(2048) -> GELU -> LN -> Linear(2048) -> GELU -> LN -> Linear(896)
        let input_dim = TOTAL_INPUT_DIM;
        let hidden_dim = 2048;
        let output_dim = hidden_size;

        // Note: Keys must match the safetensors file (0.weight, 2.weight, etc.)
        let layer0 = linear(input_dim, hidden_dim, vb.pp("0"))?;
        let norm1 = layer_norm(hidden_dim, 1e-5, vb.pp("2"))?;
        let layer3 = linear(hidden_dim, hidden_dim, vb.pp("3"))?;
        let norm4 = layer_norm(hidden_dim, 1e-5, vb.pp("5"))?;
        let layer6 = linear(hidden_dim, output_dim, vb.pp("6"))?;

        Ok(Self {
            layer0,
            norm1,
            layer3,
            norm4,
            layer6,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.to_dtype(self.layer0.weight().dtype())?;

        let x = self.layer0.forward(&xs)?;
        let x = x.gelu()?;
        let x = self.norm1.forward(&x)?;

        let x = self.layer3.forward(&x)?;
        let x = x.gelu()?;
        let x = self.norm4.forward(&x)?;

        self.layer6.forward(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_splat_adapter_forward() -> Result<()> {
        let device = Device::Cpu;
        let hidden_size = crate::constants::EMBED_DIM;

        // Create a VarBuilder with zeros for simplicity
        let vars = VarBuilder::zeros(DType::F32, &device);

        // Initialize adapter
        let adapter = SplatAdapter::new(hidden_size, vars)?;

        // Create input tensor [Batch=2, Dim=128]
        let input = Tensor::randn(0.0f32, 1.0f32, (2, TOTAL_INPUT_DIM), &device)?;

        // Forward pass
        let output = adapter.forward(&input)?;

        // Check output shape [2, 128]
        assert_eq!(output.dims(), &[2, hidden_size]);

        Ok(())
    }

    #[test]
    fn test_splat_adapter_shape_mismatch() -> Result<()> {
        let device = Device::Cpu;
        let vars = VarBuilder::zeros(DType::F32, &device);
        let adapter = SplatAdapter::new(crate::constants::EMBED_DIM, vars)?;

        // Wrong input dimension [2, 32] instead of 128
        let input = Tensor::randn(0.0f32, 1.0f32, (2, 32), &device)?;

        // Should fail
        let result = adapter.forward(&input);
        assert!(result.is_err());

        Ok(())
    }
}
