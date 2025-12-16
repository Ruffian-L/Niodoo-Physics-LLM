use anyhow::Result;
use candle_core::{Device, Tensor};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct SteeringEngine {
    pub device: Device,
    pub active_attractor: Option<Tensor>, // [3]
    pub strength: f32,                    // 0.0 to 1.0
}

impl SteeringEngine {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
            active_attractor: None,
            strength: 1.0,
        }
    }

    /// Forward Pass: The LLM "thinks" a token, setting a goal in Physics Space.
    /// In a full loop, this would be the embedding of the *next* desired concept.
    /// For now, we manually set it or map from a generated token.
    pub fn set_attractor(&mut self, target_pos: Tensor) {
        self.active_attractor = Some(target_pos);
    }

    pub fn clear_attractor(&mut self) {
        self.active_attractor = None;
    }

    /// Backward Pass: Physics tells the LLM what to say next.
    /// Returns a tensor of shape [vocab_size] with logit biases.
    /// This is a simplified "Inverse Square" bias based on distance to attractor.
    ///
    /// Note: This requires mapping *every* token in vocab to a physics position.
    /// Since we only simulate N particles (subset of vocab), we only boost those N tokens.
    pub fn get_logit_bias(
        &self,
        sim_positions: &Tensor, // [N, 3]
        token_ids: &[u32],      // [N] mapping particle index -> token_id
        vocab_size: usize,
    ) -> Result<Tensor> {
        let mut bias = Tensor::zeros((vocab_size,), candle_core::DType::F32, &self.device)?;

        if let Some(target) = &self.active_attractor {
            // Compute distances: ||sim_pos - target||
            // target: [3] -> [1, 3] broadcast to [N, 3]
            let diff = sim_positions.broadcast_sub(&target.unsqueeze(0)?)?;
            let dist_sq = diff.sqr()?.sum_keepdim(1)?; // [N, 1]
            let dist = dist_sq.sqrt()?;

            // Bias = strength / (dist + epsilon)
            // We want closer things to have HIGHER logits.
            // .scale() is not standard in candle? Use multiply.
            let strength_val = (self.strength * 10.0) as f64;
            let signal = (dist + 0.1)?.recip()? * strength_val;
            let signal = signal?;

            // Scatter these biases into the vocab tensor
            // bias.scatter_add(indices, values, dim)

            let indices = Tensor::new(token_ids, &self.device)?;
            // scatter_add expects: source [N, 1] matching bias dim?
            // bias is [vocab]. indices is [N]. signal is [N, 1].
            // squeeze signal to [N].
            let signal_flat = signal.squeeze(1)?;
            bias = bias.scatter_add(&indices, &signal_flat, 0)?;
        }

        Ok(bias)
    }
}
