use anyhow::{Context, Result};
use candle_core::{Device, Tensor};

/// The "Neurotransmitter" signals
#[derive(Debug, Clone)]
pub enum Signal {
    InjectChaos(f64), // Increase temperature/variance
    Dampen(f64),      // Reduce gain/prune weights
    Stabilize,        // System is in optimal radiance
}

/// The "Heart" of the organism: A generative tensor loop
pub struct TensorHeart {
    state: Tensor,
    pub gain: f64,
    device: Device,
}

impl TensorHeart {
    pub fn new() -> Result<Self> {
        // Homunculus: Auto-detect CUDA, fallback to CPU
        let device = Device::cuda_if_available(0)?;

        // Initial State: Gaussian Noise
        let state = Tensor::randn(0f32, 1f32, (64, 64), &device)
            .context("Failed to initialize heart state tensor")?;

        Ok(Self {
            state,
            gain: 1.0,
            device,
        })
    }

    /// The "Beat": Generates the next state based on internal gain and previous state
    pub fn pulse(&mut self) -> Result<f64> {
        // 1. Linear Projection (Self-Interaction)
        let mixing = self
            .state
            .matmul(&self.state)
            .context("Synaptic mixing failed (matmul)")?;

        // 2. Chaos Injection (Noise)
        let noise = Tensor::randn(0f32, 1f32, mixing.shape(), &self.device)
            .context("Failed to generate chaos noise")?;

        // Apply Gain to Noise (Chaos Factor)
        let scaled_noise = (noise * self.gain).context("Failed to scale noise")?;

        // 3. Integration
        let raw_state = (mixing + scaled_noise).context("State integration failed")?;

        // 4. Normalization (Homeostatic Constraint)
        let mean = raw_state.mean_all().context("Failed to compute mean")?;
        let centered = raw_state.broadcast_sub(&mean).context("Centering failed")?;

        let sq_diff = centered.sqr()?;
        let variance = sq_diff
            .mean_all()
            .context("Variance computation failed")?
            .to_scalar::<f32>()? as f64;

        let std_dev = variance.sqrt();

        // Update State (Normalized)
        self.state = (centered / (std_dev + 1e-5)).context("Normalization failed")?;

        Ok(variance)
    }

    pub fn adjust_biochemistry(&mut self, signal: Signal) {
        match signal {
            Signal::InjectChaos(dose) => {
                self.gain += dose;
            }
            Signal::Dampen(dose) => {
                let new_gain = (self.gain - dose).max(0.1);
                self.gain = new_gain;
            }
            Signal::Stabilize => {}
        }
    }

    /// Get current chaos level (the internal gain factor)
    /// Used by EmergenceController for dynamic gain computation
    /// Higher values = more exploration, more noise injection
    pub fn current_chaos_level(&self) -> f64 {
        self.gain
    }

    /// Get the current device this heart is running on
    pub fn device(&self) -> &Device {
        &self.device
    }
}
