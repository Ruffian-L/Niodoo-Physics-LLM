use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

use super::qwen::{Config, Model};

pub struct ModelLoader;

impl ModelLoader {
    pub fn load_qwen2_1_5b_instruct() -> Result<(Model, Tokenizer)> {
        let device = Device::cuda_if_available(0)?;

        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            "Qwen/Qwen2.5-1.5B-Instruct".to_string(),
            RepoType::Model,
            "main".to_string(),
        ));

        let tokenizer_filename = repo.get("tokenizer.json")?;
        let config_filename = repo.get("config.json")?;
        let model_filename = repo.get("model.safetensors")?;

        let config: Config = serde_json::from_slice(&std::fs::read(config_filename)?)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(|e| anyhow::anyhow!(e))?;

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[model_filename], DType::F16, &device)? };

        // Pass vb.pp("model") because the safetensors file has "model.embed_tokens", "model.layers", etc.
        let model = Model::new(&config, vb.pp("model"))?;

        Ok((model, tokenizer))
    }

    /// Load a Qwen model specified by the `HF_MODEL_REPO` env var (or default to
    /// `Qwen/Qwen2.5-0.5B-Instruct`). Returns (Model, Tokenizer).
    pub fn load_qwen_from_env() -> Result<(Model, Tokenizer)> {
        let repo_id = std::env::var("HF_MODEL_REPO")
            .unwrap_or_else(|_| "Qwen/Qwen2.5-0.5B-Instruct".to_string());
        let device = Device::cuda_if_available(0)?;

        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            repo_id.clone(),
            RepoType::Model,
            "main".to_string(),
        ));

        let tokenizer_filename = repo.get("tokenizer.json")?;
        let config_filename = repo.get("config.json")?;
        let model_filename = repo.get("model.safetensors")?;

        let config: Config = serde_json::from_slice(&std::fs::read(config_filename)?)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(|e| anyhow::anyhow!(e))?;

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[model_filename], DType::F16, &device)? };

        // Pass vb.pp("model") here too
        let model = Model::new(&config, vb.pp("model"))?;

        Ok((model, tokenizer))
    }
}
