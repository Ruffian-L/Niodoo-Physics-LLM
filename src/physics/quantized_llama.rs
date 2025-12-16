use crate::physics::naked_llama::PhysicsEngine;
use candle_core::quantized::{gguf_file, QMatMul, QTensor};
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Embedding, Module as _, RmsNorm, RotaryEmbedding, VarBuilder};
use std::collections::HashMap;
use std::path::Path;
use tokenizers::Tokenizer;

// Trivial helper to load QMatMul
fn load_q(
    model: &gguf_file::Content,
    file: &mut std::fs::File,
    name: &str,
    device: &Device,
) -> Result<QMatMul> {
    let info = model
        .tensor_infos
        .get(name)
        .ok_or_else(|| candle_core::Error::Msg(format!("Missing tensor {}", name)))?;
    let q = info.read(file, device)?;
    QMatMul::from_qtensor(q)
}

fn load_norm(
    model: &gguf_file::Content,
    file: &mut std::fs::File,
    name: &str,
    device: &Device,
) -> Result<RmsNorm> {
    let info = model
        .tensor_infos
        .get(name)
        .ok_or_else(|| candle_core::Error::Msg(format!("Missing tensor {}", name)))?;
    let t = info.read(file, device)?.dequantize(device)?;
    Ok(RmsNorm::new(t, 1e-5))
}

pub struct QLayer {
    pub attn_q: QMatMul,
    pub attn_k: QMatMul,
    pub attn_v: QMatMul,
    pub attn_output: QMatMul,
    pub mlp_gate: QMatMul,
    pub mlp_up: QMatMul,
    pub mlp_down: QMatMul,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
    pub params: LlamaParams,
    pub index: usize,
}

#[derive(Clone, Copy)]
pub struct LlamaParams {
    pub hidden_size: usize,
    pub head_dim: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub rope_theta: f32,
}

pub struct QuantizedNakedLlama {
    pub embed_tokens: Embedding,
    pub layers: Vec<QLayer>,
    pub norm: RmsNorm,
    pub lm_head: QMatMul,
    pub device: Device,
    pub params: LlamaParams,
    pub context_tokens: Vec<u32>,
    pub rotary: RotaryEmbedding,
    pub tokenizer: Tokenizer,
}

impl QuantizedNakedLlama {
    pub fn new(
        model: &gguf_file::Content,
        mut file: &mut std::fs::File,
        device: &Device,
        tokenizer_path: &Path,
    ) -> Result<Self> {
        let n_layers = model
            .metadata
            .get("llama.block_count")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(32) as usize;
        let n_head = model
            .metadata
            .get("llama.attention.head_count")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(32) as usize;
        let n_kv_head = model
            .metadata
            .get("llama.attention.head_count_kv")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(8) as usize;
        let hidden_size = model
            .metadata
            .get("llama.embedding_length")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(4096) as usize;
        let rope_theta = model
            .metadata
            .get("llama.rope.freq_base")
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(500000.0);
        let head_dim = hidden_size / n_head;

        let params = LlamaParams {
            hidden_size,
            head_dim,
            n_head,
            n_kv_head,
            rope_theta,
        };

        let tok = model
            .tensor_infos
            .get("token_embd.weight")
            .ok_or(candle_core::Error::Msg("Missing token_embd".into()))?;
        let tok_w = tok.read(&mut file, device)?.dequantize(device)?;
        let embed_tokens = Embedding::new(tok_w, hidden_size);

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = format!("blk.{}.", i);
            let attn_output = load_q(
                model,
                &mut file,
                &format!("{}attn_output.weight", p),
                device,
            )?;
            layers.push(QLayer {
                attn_q: load_q(model, &mut file, &format!("{}attn_q.weight", p), device)?,
                attn_k: load_q(model, &mut file, &format!("{}attn_k.weight", p), device)?,
                attn_v: load_q(model, &mut file, &format!("{}attn_v.weight", p), device)?,
                attn_output,
                mlp_gate: load_q(model, &mut file, &format!("{}ffn_gate.weight", p), device)?,
                mlp_up: load_q(model, &mut file, &format!("{}ffn_up.weight", p), device)?,
                mlp_down: load_q(model, &mut file, &format!("{}ffn_down.weight", p), device)?,
                input_layernorm: load_norm(
                    model,
                    &mut file,
                    &format!("{}attn_norm.weight", p),
                    device,
                )?,
                post_attention_layernorm: load_norm(
                    model,
                    &mut file,
                    &format!("{}ffn_norm.weight", p),
                    device,
                )?,
                params,
                index: i,
            });
        }

        let norm = load_norm(model, &mut file, "output_norm.weight", device)?;
        let lm_head = load_q(model, &mut file, "output.weight", device)?;

        let rotary = RotaryEmbedding::new(rope_theta, head_dim, n_head, device, false, DType::F32)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            params,
            context_tokens: Vec::new(),
            rotary,
            tokenizer,
        })
    }

    pub fn forward_physics(
        &mut self,
        input_ids: &Tensor,
        physics: &mut impl PhysicsEngine,
        _graviton_proj: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (b, seq_len) = input_ids.dims2()?;
        let mut x = self.embed_tokens.forward(input_ids)?;

        let pos = (0..seq_len).map(|i| i as i64).collect::<Vec<_>>();
        let pos_t = Tensor::new(&pos[..], &self.device)?.unsqueeze(0)?;

        for layer in self.layers.iter() {
            let residual = x.clone();
            let x_norm = layer.input_layernorm.forward(&x)?;

            let q = layer.attn_q.forward(&x_norm)?;
            let k = layer.attn_k.forward(&x_norm)?;
            let v = layer.attn_v.forward(&x_norm)?;

            let q = q
                .reshape((b, seq_len, layer.params.n_head, layer.params.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b, seq_len, layer.params.n_kv_head, layer.params.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b, seq_len, layer.params.n_kv_head, layer.params.head_dim))?
                .transpose(1, 2)?;

            let (q, k) = self.rotary.forward(&q, &pos_t, &k)?;

            let k = repeat_kv(k, layer.params.n_head / layer.params.n_kv_head)?;
            let v = repeat_kv(v, layer.params.n_head / layer.params.n_kv_head)?;

            let att = (q.matmul(&k.t()?)? / (layer.params.head_dim as f64).sqrt())?;

            let mask = causal_mask(seq_len, &self.device)?;
            let att = broadcast_add(&att, &mask)?;
            let att = candle_nn::ops::softmax(&att, D::Minus1)?;

            let y = att.matmul(&v)?;
            let y = y
                .transpose(1, 2)?
                .reshape((b, seq_len, layer.params.hidden_size))?;

            let attn_out = layer.attn_output.forward(&y)?;

            let physics_force = physics.apply_forces(&x_norm, layer.index)?;

            let physics_f32 = physics_force.to_dtype(DType::F32)?;
            let p_rank = physics_f32.rank();
            let physics_last = if p_rank == 3 && physics_f32.dim(1)? > 1 {
                physics_f32.narrow(1, physics_f32.dim(1)? - 1, 1)?
            } else {
                physics_f32
            };

            let attn_f32 = attn_out.to_dtype(DType::F32)?;
            let blended = if seq_len > 1 {
                let history = attn_f32.narrow(1, 0, seq_len - 1)?;
                let last = attn_f32.narrow(1, seq_len - 1, 1)?;
                let blended_last = ((last * 0.95)? + (physics_last * 0.05)?)?;
                Tensor::cat(&[&history, &blended_last], 1)?
            } else {
                ((attn_f32 * 0.95)? + (physics_last * 0.05)?)?
            };

            x = (residual + blended.to_dtype(DType::F32)?)?;

            let residual_mlp = x.clone();
            let x_norm_mlp = layer.post_attention_layernorm.forward(&x)?;
            let gate = layer.mlp_gate.forward(&x_norm_mlp)?;
            let up = layer.mlp_up.forward(&x_norm_mlp)?;
            let down = layer
                .mlp_down
                .forward(&(&candle_nn::ops::silu(&gate)? * &up)?)?;

            x = (residual_mlp + down)?;
        }

        let x_norm = self.norm.forward(&x)?;
        let logits = self.lm_head.forward(&x_norm)?;

        Ok((logits, x_norm))
    }

    pub fn append_token(&mut self, token: u32) {
        self.context_tokens.push(token);
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b, n_kv_head, seq, head_dim) = x.dims4()?;
    let x = x
        .unsqueeze(2)?
        .broadcast_as((b, n_kv_head, n_rep, seq, head_dim))?;
    x.flatten(1, 2)
}

fn causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
        .collect();
    Tensor::from_vec(mask, (seq_len, seq_len), device)?
        .unsqueeze(0)?
        .unsqueeze(0)
}

fn broadcast_add(x: &Tensor, mask: &Tensor) -> Result<Tensor> {
    x.broadcast_add(mask)
}
