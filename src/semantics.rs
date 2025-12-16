use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{linear, Linear, Optimizer, VarBuilder};

pub struct LangSplatAutoencoder {
    encoder: Linear,
    decoder: Linear,
    device: Device,
}

impl LangSplatAutoencoder {
    pub fn new(input_dim: usize, latent_dim: usize, device: &Device) -> Result<Self> {
        // Simple linear autoencoder (PCA-like behavior)
        // In a real LangSplat, this would be scene-optimized (trained per scene)

        // We create random weights initially or load them.
        // For this implementation, we initialize random.

        let map = VarBuilder::zeros(DType::F32, device);

        let encoder = linear(input_dim, latent_dim, map.pp("enc"))?;
        let decoder = linear(latent_dim, input_dim, map.pp("dec"))?;

        Ok(Self {
            encoder,
            decoder,
            device: device.clone(),
        })
    }

    pub fn encode(&self, embedding: &[f32]) -> Result<[u8; 3]> {
        let input = Tensor::from_slice(embedding, (1, embedding.len()), &self.device)?;
        let latent = self.encoder.forward(&input)?;

        // Normalize latent to 0..1 (sigmoid-like) or clamp for RGB
        // LangSplat usually stores latent features in SH or Color.
        // We map 3D latent to RGB [0..255].

        let latent_vec = latent.squeeze(0)?.to_vec1::<f32>()?;

        if latent_vec.len() < 3 {
            return Ok([0, 0, 0]);
        }

        // Simple mapping: (x + 1) / 2 * 255
        let r = ((latent_vec[0].tanh() + 1.0) / 2.0 * 255.0) as u8;
        let g = ((latent_vec[1].tanh() + 1.0) / 2.0 * 255.0) as u8;
        let b = ((latent_vec[2].tanh() + 1.0) / 2.0 * 255.0) as u8;

        Ok([r, g, b])
    }

    /// Decode RGB back to approximate embedding (for querying)
    pub fn decode(&self, color: [u8; 3]) -> Result<Vec<f32>> {
        let r = (color[0] as f32 / 255.0) * 2.0 - 1.0; // Inverse mapping
        let g = (color[1] as f32 / 255.0) * 2.0 - 1.0;
        let b = (color[2] as f32 / 255.0) * 2.0 - 1.0;

        // Use inverse tanh (approximate or skip)
        // Let's just pass raw
        let latent = Tensor::from_slice(&[r, g, b], (1, 3), &self.device)?;
        let output = self.decoder.forward(&latent)?;

        let vec = output.squeeze(0)?.to_vec1::<f32>()?;
        Ok(vec)
    }
}
