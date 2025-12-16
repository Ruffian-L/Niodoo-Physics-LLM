use candle_core::{Device, Tensor, Result};
use candle_nn::ops::softmax;

// Manual L2 normalization since candle doesn't have .normalize()
fn l2_normalize(t: &Tensor, dim: usize) -> Result<Tensor> {
    let sq = t.sqr()?;
    let sum_sq = sq.sum_keepdim(dim)?;
    let norm = sum_sq.sqrt()?;
    t.broadcast_div(&norm)
}

pub struct ContinuousHopfield {
    stored: Tensor, // (N, 512) fp32 unit-norm on GPU
    beta: f32,
    lambda: f32,
}

impl ContinuousHopfield {
    pub fn new(stored_positions: Tensor, device: &Device) -> Result<Self> {
        // Ensure stored positions are on the correct device and normalized
        let on_device = stored_positions.to_device(device)?;
        let stored = l2_normalize(&on_device, 1)?;
        Ok(Self { stored, beta: 80.0, lambda: 1.0 })
    }

    pub fn device(&self) -> &Device {
        self.stored.device()
    }

    pub fn retrieve(&self, query: &Tensor, steps: usize) -> Result<Tensor> {
        let mut x = l2_normalize(query, 1)?;
        let mut beta = self.beta;
        
        for _ in 0..steps {
            // Similarity: (B, 512) x (512, N) -> (B, N)
            let sim = x.matmul(&self.stored.t()?)?;
            
            // Attention: softmax(beta * sim)
            let scaled = (sim * beta as f64)?;
            let att = softmax(&scaled, 1)?; // Softmax over N (dim 1)
            
            // Update: (B, N) x (N, 512) -> (B, 512)
            let update = att.matmul(&self.stored)?;
            
            // Hopfield Dynamics: x = update - lambda * x
            let x_scaled = (x.clone() * self.lambda as f64)?;
            x = (update - x_scaled)?;
            
            // Renormalize to stay on unit sphere
            x = l2_normalize(&x, 1)?;
            
            // Anneal beta
            beta *= 2.5; 
        }
        Ok(x)
    }

    pub fn search_batch(&self, query: &Tensor) -> Result<Vec<(usize, f32)>> {
        let x = l2_normalize(query, 1)?;
        // sim: (B, N)
        let sim = x.matmul(&self.stored.t()?)?;
        
        // Get Max (Top-1) per query
        let vals = sim.max(1)?;
        let ids = sim.argmax(1)?;
        
        // Extract to Vec
        let vals_vec: Vec<f32> = vals.to_vec1()?;
        let ids_vec: Vec<u32> = ids.to_vec1()?;
        
        let mut res = Vec::with_capacity(vals_vec.len());
        for (v, id) in vals_vec.iter().zip(ids_vec.iter()) {
            res.push((*id as usize, *v));
        }
        Ok(res)
    }
}
