use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Activation, Embedding, Linear, VarBuilder};

fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    if vb.contains_tensor("bias") {
        candle_nn::linear(in_dim, out_dim, vb)
    } else {
        candle_nn::linear_no_bias(in_dim, out_dim, vb)
    }
}

// Qwen2 Config
#[derive(serde::Deserialize, Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: Option<usize>,
    pub max_window_layers: Option<usize>,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub use_sliding_window: bool,
    pub tie_word_embeddings: bool,
}

impl Config {
    pub fn qwen2_1_5b() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 1536,
            intermediate_size: 8960,
            num_hidden_layers: 28,
            num_attention_heads: 12,
            num_key_value_heads: 2,
            max_position_embeddings: 32768,
            sliding_window: None,
            max_window_layers: None,
            rope_theta: 1000000.0,
            rms_norm_eps: 1e-6,
            use_sliding_window: false,
            tie_word_embeddings: true,
        }
    }
}

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = DType::F32;
        let x = x.to_dtype(internal_dtype)?;
        let (_b_sz, _seq_len, hidden_size) = x.dims3()?;
        let norm_x = (x.sqr()?.sum_keepdim(2)? / (hidden_size as f64))?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        let x_normed = x_normed.to_dtype(x_dtype)?;
        x_normed.broadcast_mul(&self.weight)
    }
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let last_dim = x.dim(D::Minus1)?;
    let x1 = x.narrow(D::Minus1, 0, last_dim / 2)?;
    let x2 = x.narrow(D::Minus1, last_dim / 2, last_dim / 2)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / (cfg.rope_theta as f32).powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let freqs = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, _seq_len_in, _d) = q.dims4()?;
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;

        // Manual RoPE implementation to handle broadcasting and shape mismatches
        let d_cos = cos.dim(1)?;

        // If cos/sin are half-dimension (standard for RoPE construction), repeat them
        let cos = if d_cos == _d / 2 {
            Tensor::cat(&[&cos, &cos], 1)?
        } else {
            cos
        };
        let sin = if d_cos == _d / 2 {
            Tensor::cat(&[&sin, &sin], 1)?
        } else {
            sin
        };

        // Broadcast to [B, 1, S, D] for multiplication with [B, H, S, D]
        let cos = cos.broadcast_as((_b_sz, 1, seq_len, _d))?;
        let sin = sin.broadcast_as((_b_sz, 1, seq_len, _d))?;

        let q_embed = (q.broadcast_mul(&cos)? + &rotate_half(q)?.broadcast_mul(&sin)?)?;
        let k_embed = (k.broadcast_mul(&cos)? + &rotate_half(k)?.broadcast_mul(&sin)?)?;

        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let intermediate_size = cfg.intermediate_size;
        let gate_proj = linear(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear(intermediate_size, hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: Activation::Silu,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let lhs = self.gate_proj.forward(x)?.apply(&self.act_fn)?;
        let rhs = self.up_proj.forward(x)?;
        self.down_proj.forward(&(lhs * rhs)?)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = hidden_size / num_heads;
        let q_proj = linear(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        rotary_emb: &RotaryEmbedding,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // eprintln!("DEBUG: Attention::forward x: {:?}", x.shape());
        let (b_sz, seq_len, hidden_size) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // eprintln!("DEBUG: q/k before rope: {:?} / {:?}", q.shape(), k.shape());
        let (q, k) = rotary_emb.apply_rotary_emb_qkv(&q, &k, seq_len)?;

        // Repeat k/v heads if necessary (GQA)
        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        let att = (q.matmul(&k.t()?)? * self.scale)?;
        let att = match mask {
            Some(mask) => att.broadcast_add(mask)?,
            None => att,
        };
        let att = candle_nn::ops::softmax(&att, D::Minus1)?;
        let y = att.matmul(&v)?;
        let y = y.transpose(1, 2)?.reshape((b_sz, seq_len, hidden_size))?;
        self.o_proj.forward(&y)
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        let n_rep = self.num_heads / self.num_kv_heads;
        if n_rep == 1 {
            Ok(x)
        } else {
            let (b, n_kv_head, seq_len, head_dim) = x.dims4()?;
            let x = x
                .unsqueeze(2)?
                .expand((b, n_kv_head, n_rep, seq_len, head_dim))?;
            x.reshape((b, n_kv_head * n_rep, seq_len, head_dim))
        }
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(cfg, vb.pp("self_attn"))?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        rotary_emb: &RotaryEmbedding,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // eprintln!("DEBUG: DecoderLayer::forward");
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(&x, rotary_emb, mask)?;
        let x = (x + residual)?;

        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        let x = (x + residual)?;
        Ok(x)
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    pub embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    rotary_emb: RotaryEmbedding,
    pub device: Device,
    pub config: Config,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, vb.pp(&format!("layers.{}", i)))?);
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;

        // Try to load lm_head, otherwise fallback to tied embeddings
        let lm_head = match linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head")) {
            Ok(head) => head,
            Err(_) => {
                // Tied weights: reuse embedding weights
                candle_nn::Linear::new(embed_tokens.embeddings().clone(), None)
            }
        };

        let rotary_emb = RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary_emb,
            device: vb.device().clone(),
            config: cfg.clone(),
        })
    }

    pub fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn forward_from_embeddings(&self, embeddings: &Tensor) -> Result<Tensor> {
        let (_b, seq_len, _h) = embeddings.dims3()?;
        let mut x = embeddings.clone();

        // Create causal mask
        let mask = if seq_len > 1 {
            let mask: Vec<_> = (0..seq_len)
                .flat_map(|i| {
                    (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 })
                })
                .collect();
            let mask = Tensor::from_slice(&mask, (seq_len, seq_len), &self.device)?;
            let mask = mask.to_dtype(embeddings.dtype())?;
            let mask = mask.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, seq_len, seq_len]
            Some(mask)
        } else {
            None
        };

        for layer in &self.layers {
            x = layer.forward(&x, &self.rotary_emb, mask.as_ref())?;
        }
        let x = self.norm.forward(&x)?;
        let logits = self.lm_head.forward(&x)?;
        Ok(logits)
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let embeddings = self.embed(input_ids)?;
        self.forward_from_embeddings(&embeddings)
    }

    // 1. Rename existing forward to 'forward_ids' (Wrapper)
    pub fn forward_ids(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.forward(input_ids)
    }

    // 2. Create the new 'forward_raw_embed' (The Telepathy Port)
    // This is essentially forward_from_embeddings but we make it explicit as requested
    pub fn forward_raw_embed(&self, embeddings: &Tensor) -> Result<Tensor> {
        self.forward_from_embeddings(embeddings)
    }

    pub fn forward_with_injection(
        &self,
        embeddings: &Tensor,
        layer_idx: usize,
        injection: &Tensor,
        alpha: f64,
    ) -> Result<Tensor> {
        let (_b, seq_len, _h) = embeddings.dims3()?;
        let mut x = embeddings.clone();

        // Create causal mask
        let mask = if seq_len > 1 {
            let mask: Vec<_> = (0..seq_len)
                .flat_map(|i| {
                    (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 })
                })
                .collect();
            let mask = Tensor::from_slice(&mask, (seq_len, seq_len), &self.device)?;
            let mask = mask.to_dtype(embeddings.dtype())?;
            let mask = mask.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, seq_len, seq_len]
            Some(mask)
        } else {
            None
        };

        for (i, layer) in self.layers.iter().enumerate() {
            if i == layer_idx {
                // Injection: Add to the last token
                if seq_len > 0 {
                    let last_idx = seq_len - 1;
                    let prev = x.narrow(1, 0, last_idx)?;
                    let last = x.narrow(1, last_idx, 1)?;

                    // Ensure injection has same shape as last [1, 1, hidden]
                    let inj = if injection.rank() == 2 {
                        injection.unsqueeze(1)?
                    } else {
                        injection.clone()
                    };

                    // Adaptive Norm Scaling (Alpha passed in)

                    // Cast to F32 for calculation to avoid precision mismatch
                    let last_f32 = last.to_dtype(DType::F32)?;
                    let inj_f32 = inj.to_dtype(DType::F32)?;

                    // 1. Calculate Statistics (Mean & Std Dev)
                    let stream_mean = last_f32.mean_all()?.to_scalar::<f32>()? as f64;
                    let memory_mean = inj_f32.mean_all()?.to_scalar::<f32>()? as f64;

                    let stream_var = (last_f32.sqr()?.mean_all()?.to_scalar::<f32>()? as f64)
                        - (stream_mean * stream_mean);
                    let memory_var = (inj_f32.sqr()?.mean_all()?.to_scalar::<f32>()? as f64)
                        - (memory_mean * memory_mean);

                    let stream_std = stream_var.sqrt();
                    let memory_std = memory_var.sqrt();

                    // 2. Normalize Memory to match Stream Voltage (Std Dev)
                    let scaling_factor = if memory_std > 0.0 {
                        (stream_std / memory_std) * alpha
                    } else {
                        0.0
                    };

                    println!(
                        "DEBUG: Stream Std: {:.4}, Memory Std: {:.4}, Alpha: {:.4}, Scaling: {:.4}",
                        stream_std, memory_std, alpha, scaling_factor
                    );

                    // 3. Inject
                    let scaled_memory_f32 = inj_f32.affine(scaling_factor, 0.0)?;
                    let scaled_memory = scaled_memory_f32.to_dtype(x.dtype())?;

                    let new_last = (last + scaled_memory)?;

                    if last_idx > 0 {
                        x = Tensor::cat(&[&prev, &new_last], 1)?;
                    } else {
                        x = new_last;
                    }
                }
            }
            x = layer.forward(&x, &self.rotary_emb, mask.as_ref())?;
        }
        let x = self.norm.forward(&x)?;
        let logits = self.lm_head.forward(&x)?;
        Ok(logits)
    }

    pub fn get_embed_tokens(&self) -> &Embedding {
        &self.embed_tokens
    }
}
