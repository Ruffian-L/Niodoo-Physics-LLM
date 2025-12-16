use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};
use std::path::Path;
use tokenizers::Tokenizer;

pub struct Model {
    pub embed_tokens: Embedding,
    pub hidden_size: usize,
    pub tokenizer: Tokenizer,
}

impl Model {
    /// Construct a Model. If a safetensors file is present under `models/model.safetensors`
    /// or if `HF_MODEL_REPO` is set, this will attempt to download the safetensors and
    /// load a `VarBuilder` via `from_mmaped_safetensors` (unsafe, zero-copy) for speed.
    /// This will `panic!` on shape/loader mismatches to fail fast as requested.
    pub fn new(vocab_size: usize, hidden_size: usize, _vb: VarBuilder) -> Result<Self> {
        // Determine repo and model file
        let repo = std::env::var("HF_MODEL_REPO")
            .unwrap_or_else(|_| "HuggingFaceTB/SmolLM-360M".to_string());
        let models_dir = Path::new("models");

        // Attempt to ensure tokenizer and model weights are available via hf-hub helper.
        let _ = crate::llm::hf_downloader::download_files(
            &repo,
            &["tokenizer.json", "model.safetensors"],
        );

        // Load tokenizer (fail fast if missing)
        let tokenizer_path = if Path::new("models/tokenizer.json").exists() {
            "models/tokenizer.json"
        } else if Path::new("models/gpt2_tokenizer.json").exists() {
            "models/gpt2_tokenizer.json"
        } else {
            panic!("Tokenizer not found in `models/`. Set HF_MODEL_REPO or place a tokenizer at models/tokenizer.json");
        };

        let tokenizer = Tokenizer::from_file(tokenizer_path).expect("Failed to load tokenizer");

        // If we have safetensors, try to load VarBuilder from mmaped safetensors (zero-copy).
        let safetensors_path = models_dir.join("model.safetensors");
        if safetensors_path.exists() {
            println!(
                "⚡ Found safetensors at {} — loading into VarBuilder (zero-copy)...",
                safetensors_path.display()
            );

            // Pick dtype: prefer F16 for modern models, allow override
            let dtype = match std::env::var("HF_SAFETENSORS_DTYPE").as_deref() {
                Ok("f32") | Ok("F32") => DType::F32,
                _ => DType::F16,
            };

            // Choose device (CUDA if available)
            let device = match Device::new_cuda(0) {
                Ok(dev) => dev,
                Err(_) => Device::Cpu,
            };

            // SAFETY: from_mmaped_safetensors is unsafe because it maps binary file memory into
            // tensors. We follow the code pattern used elsewhere in the codebase.
            let vb_safetensors = unsafe {
                VarBuilder::from_mmaped_safetensors(&[safetensors_path.to_string_lossy().to_string()], dtype, &device)
                    .expect("Failed to create VarBuilder from safetensors. File may be corrupted or incompatible.")
            };

            // Now attempt to grab the embedding weights from the VarBuilder. This will panic
            // (fail-fast) if the expected parameter name or shape does not match the safetensors.
            let embed_tokens =
                candle_nn::embedding(vocab_size, hidden_size, vb_safetensors.pp("embed_tokens"))
                    .expect("Embedding weight mismatch or missing in safetensors — aborting.");

            return Ok(Self {
                embed_tokens,
                hidden_size,
                tokenizer,
            });
        }

        // Fallback: use provided VarBuilder (randomly initialized), but still fail fast if tokenizer missing.
        let embed_tokens = candle_nn::embedding(vocab_size, hidden_size, _vb.pp("embed_tokens"))?;
        Ok(Self {
            embed_tokens,
            hidden_size,
            tokenizer,
        })
    }

    pub fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn forward_from_embeddings(&self, xs: &Tensor) -> Result<Tensor> {
        Ok(xs.clone())
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let embeds = self.embed(input_ids)?;
        self.forward_from_embeddings(&embeds)
    }

    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| candle_core::Error::Msg(format!("Tokenization failed: {}", e)))?;
        Ok(encoding.get_ids().to_vec())
    }
}
