use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder};

pub struct Synapse {
    layer1: Linear,
    ln1: LayerNorm,
    layer2: Linear,
    ln2: LayerNorm,
    layer3: Linear,
}

impl Synapse {
    pub fn new(_device: &Device) -> Result<Self> {
        anyhow::bail!("Use Synapse::load()")
    }

    pub fn from_varmap(vb: VarBuilder) -> Result<Self> {
        let layer1 = candle_nn::linear(768, 2048, vb.pp("0"))?;
        let ln1 = candle_nn::layer_norm(2048, 1e-5, vb.pp("2"))?;
        let layer2 = candle_nn::linear(2048, 2048, vb.pp("3"))?;
        let ln2 = candle_nn::layer_norm(2048, 1e-5, vb.pp("5"))?;
        let layer3 = candle_nn::linear(2048, 896, vb.pp("6"))?;

        Ok(Self {
            layer1,
            ln1,
            layer2,
            ln2,
            layer3,
        })
    }

    pub fn load(path: &str, device: &Device) -> Result<Self> {
        // Load safetensors
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };

        // MLP Structure: 768 -> 2048 -> LN -> 2048 -> LN -> 896
        // Keys: 0, 2, 3, 5, 6 (from nn.Sequential)
        let layer1 = candle_nn::linear(768, 2048, vb.pp("0"))?;
        let ln1 = candle_nn::layer_norm(2048, 1e-5, vb.pp("2"))?;
        let layer2 = candle_nn::linear(2048, 2048, vb.pp("3"))?;
        let ln2 = candle_nn::layer_norm(2048, 1e-5, vb.pp("5"))?;
        let layer3 = candle_nn::linear(2048, 896, vb.pp("6"))?;

        Ok(Self {
            layer1,
            ln1,
            layer2,
            ln2,
            layer3,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.layer1.forward(x)?;
        let h = h.gelu()?;
        let h = self.ln1.forward(&h)?;
        let h = self.layer2.forward(&h)?;
        let h = h.gelu()?;
        let h = self.ln2.forward(&h)?;
        let out = self.layer3.forward(&h)?;
        Ok(out)
    }

    pub fn translate(&self, memory_vector: &Tensor) -> Result<Tensor> {
        // memory_vector: [1, 768]
        // let vec_data = memory_vector.flatten_all()?.to_vec1::<f32>()?;
        // println!("Input Vec[:5]: {:?}", &vec_data[..5]);

        let h = self.layer1.forward(memory_vector)?;
        let h = h.gelu()?;
        let h = self.ln1.forward(&h)?;
        // println!("L1 Norm: {}", h.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?);

        let h = self.layer2.forward(&h)?;
        let h = h.gelu()?;
        let h = self.ln2.forward(&h)?;
        // println!("L2 Norm: {}", h.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?);

        let out = self.layer3.forward(&h)?;
        // println!("L3 Norm: {}", out.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?);

        // Return raw output - the MLP is now trained to target the correct norm (~0.45)
        Ok(out)
    }

    pub fn device(&self) -> Device {
        self.layer1.weight().device().clone()
    }
}
