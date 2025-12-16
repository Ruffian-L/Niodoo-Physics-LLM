use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

#[derive(Debug, Clone)]
pub struct SplatGeometry {
    pub mu: Tensor,    // (Batch, 64) Centroid
    pub sigma: Tensor, // (Batch, 64) Standard Deviation (Radius)
}

#[derive(Debug)]
pub struct ManifoldProjector {
    layers: Vec<Linear>,
    pub device: Device,
}

impl ManifoldProjector {
    pub fn new(path: &str) -> Result<Self> {
        let device = Device::cuda_if_available(0)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &device)? };
        Self::load(vb, &device)
    }

    pub fn dummy(device: &Device) -> Result<Self> {
        use std::collections::HashMap;
        let mut tensors = HashMap::new();
        // Create random weights for 768 -> 128 projection
        tensors.insert(
            "dummy.weight".to_string(),
            Tensor::randn(0.0, 0.1, (128, 768), device)?,
        );
        tensors.insert(
            "dummy.bias".to_string(),
            Tensor::zeros((128,), DType::F32, device)?,
        );

        let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
        let layer = linear(768, 128, vb.pp("dummy"))?;

        Ok(Self {
            layers: vec![layer],
            device: device.clone(),
        })
    }

    /// Load the trained projector from safetensors
    pub fn load(vb: VarBuilder, device: &Device) -> Result<Self> {
        let mut layers = Vec::new();

        // Debug: Print available tensors
        // Note: VarBuilder doesn't expose keys easily without loading.
        // But we can try to guess or just print what we are looking for.
        println!("🔍 Checking projector keys...");
        if vb.contains_tensor("encoder.0.weight") {
            println!("✅ Found VAE Encoder keys");
            // Layer 1: 768 -> 512
            layers.push(linear(768, 512, vb.pp("encoder").pp("0"))?);
            // Layer 2: 512 -> 256
            layers.push(linear(512, 256, vb.pp("encoder").pp("2"))?);
            // Layer 3: 256 -> 128
            layers.push(linear(256, 128, vb.pp("encoder").pp("4"))?);
        }
        // Legacy: Single layer adapter (adapter.linear)
        else if vb.contains_tensor("adapter.linear.weight") {
            println!("✅ Found Legacy Adapter keys");
            layers.push(linear(128, 896, vb.pp("adapter").pp("linear"))?);
        }
        // Legacy: Two layer adapter (adapter.fc1, adapter.fc2)
        else if vb.contains_tensor("adapter.fc1.weight") {
            println!("✅ Found Legacy 2-Layer Adapter keys");
            layers.push(linear(128, 1024, vb.pp("adapter").pp("fc1"))?);
            layers.push(linear(1024, 896, vb.pp("adapter").pp("fc2"))?);
        }
        // Check for "net" keys (Debugging the error)
        else if vb.contains_tensor("net.0.weight") {
            println!("✅ Found 'net' keys (Unexpected!)");
            layers.push(linear(768, 512, vb.pp("net").pp("0"))?);
            layers.push(linear(512, 256, vb.pp("net").pp("2"))?);
            layers.push(linear(256, 128, vb.pp("net").pp("4"))?);
        } else {
            return Err(candle_core::Error::Msg(
                "Unknown projector architecture in safetensors".into(),
            ));
        }

        Ok(Self {
            layers,
            device: device.clone(),
        })
    }

    /// Forward pass: Text Embedding -> Splat Geometry
    pub fn forward(&self, input: &Tensor) -> Result<SplatGeometry> {
        let x = input.to_device(&self.device)?;

        // Handle input dimension mismatch
        let (_b, dim) = x.dims2()?;

        // If using Legacy projector (starts with 128) but input is 768, we must truncate (or fail)
        // But if using VAE projector (starts with 768), we use full input.
        // We can check the first layer's input dim.
        let first_layer_in_dim = self.layers[0].weight().dims()[1];

        let mut x = if dim != first_layer_in_dim {
            if dim > first_layer_in_dim {
                // Truncate (Legacy behavior)
                x.narrow(1, 0, first_layer_in_dim)?
            } else {
                // Pad with zeros
                let pad_size = first_layer_in_dim - dim;
                let pad = Tensor::zeros((x.dims()[0], pad_size), x.dtype(), x.device())?;
                Tensor::cat(&[&x, &pad], 1)?
            }
        } else {
            x
        };

        // Pass through layers
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;
            // Apply ReLU for all except last layer
            if i < self.layers.len() - 1 {
                x = x.relu()?;
            }
        }

        // Final output should be 128 (64 mu + 64 logvar)
        // Or 896 (Legacy).
        let out_dim = x.dims2()?.1;

        let (mu, logvar) = if out_dim == 128 {
            let chunks = x.chunk(2, 1)?;
            (chunks[0].clone(), chunks[1].clone())
        } else if out_dim == 896 {
            // Legacy: Take first 128
            let x = x.narrow(1, 0, 128)?;
            let chunks = x.chunk(2, 1)?;
            (chunks[0].clone(), chunks[1].clone())
        } else {
            // Fallback or error
            let x = x.narrow(1, 0, 128)?;
            let chunks = x.chunk(2, 1)?;
            (chunks[0].clone(), chunks[1].clone())
        };

        // Convert LogVar to Sigma (Radius)
        let sigma = (logvar * 0.5)?.exp()?;

        Ok(SplatGeometry { mu, sigma })
    }

    pub fn project(&self, input: &Tensor) -> Result<SplatGeometry> {
        self.forward(input)
    }
}

/// Helper to load directly from a file path
pub fn load_projector(path: &str, device: &Device) -> Result<ManifoldProjector> {
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
    ManifoldProjector::load(vb, device)
}
