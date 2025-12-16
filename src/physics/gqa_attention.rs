//! Grouped Query Attention (GQA) for Llama-3
//!
//! Implements proper attention for late layers in hybrid physics mode.

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};

/// GQA Attention for Llama-3 (8 KV heads, 32 Q heads)
pub struct GqaAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_size: usize,
}

impl GqaAttention {
    pub fn load(
        vb: VarBuilder,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
    ) -> Result<Self> {
        let head_dim = hidden_size / num_heads;

        // Q: [num_heads * head_dim, hidden_size] = [4096, 4096] for Llama-3 8B
        let q_proj = candle_nn::linear_no_bias(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;

        // K, V: [num_kv_heads * head_dim, hidden_size] = [1024, 4096] for Llama-3 8B
        let k_proj =
            candle_nn::linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj =
            candle_nn::linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;

        // O: [hidden_size, num_heads * head_dim] = [4096, 4096]
        let o_proj = candle_nn::linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_size,
        })
    }

    /// Forward pass with causal mask
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _) = hidden_states.dims3()?;
        let original_dtype = hidden_states.dtype();

        // Project Q, K, V
        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // Reshape for multi-head attention
        // Q: [batch, seq, num_heads, head_dim] -> [batch, num_heads, seq, head_dim]
        let q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        // K, V: [batch, seq, num_kv_heads, head_dim] -> [batch, num_kv_heads, seq, head_dim]
        let k = k
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // GQA: Repeat K, V to match num_heads
        let repeats = self.num_heads / self.num_kv_heads;
        let k = k.repeat(&[1, repeats, 1, 1])?;
        let v = v.repeat(&[1, repeats, 1, 1])?;

        // Scaled dot-product attention (in F32 for stability)
        let q_f32 = q.to_dtype(DType::F32)?;
        let k_f32 = k.to_dtype(DType::F32)?;
        let v_f32 = v.to_dtype(DType::F32)?;

        let scale = (self.head_dim as f64).sqrt();
        let q_f32 = q_f32.contiguous()?;
        let k_transposed = k_f32.transpose(2, 3)?.contiguous()?;
        let attn_weights = (q_f32.matmul(&k_transposed)? / scale)?;

        // Causal mask
        let mask = Self::causal_mask(seq_len, hidden_states.device())?;
        let attn_weights = attn_weights.broadcast_add(&mask)?;

        // Softmax
        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;

        // Apply attention to values
        let attn_output = attn_weights.matmul(&v_f32)?;

        // Reshape back: [batch, num_heads, seq, head_dim] -> [batch, seq, hidden_size]
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((batch, seq_len, self.hidden_size))?
            .to_dtype(original_dtype)?;

        // Output projection
        self.o_proj.forward(&attn_output)
    }

    fn causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
        // Create lower triangular mask manually
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j > i {
                    mask_data[i * seq_len + j] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::from_vec(mask_data, (seq_len, seq_len), device)?;
        mask.unsqueeze(0)?.unsqueeze(0) // [1, 1, seq, seq]
    }
}
