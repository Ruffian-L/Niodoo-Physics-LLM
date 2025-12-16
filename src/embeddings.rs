use crate::constants::FULL_EMBED_DIM;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};

pub enum EmbeddingUsage {
    Query,
    Document,
    Tokens,
}

#[derive(Debug, Deserialize)]
struct DaemonResponseItem {
    pooled: Vec<f32>,
    #[serde(default)]
    token_embeddings: Vec<Vec<f32>>,
    #[serde(default)]
    tokens: Vec<String>,
}

struct DaemonProcess {
    child: Child,
    stdin: ChildStdin,
    reader: BufReader<ChildStdout>,
}

impl Drop for DaemonProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

#[derive(Clone)]
pub struct EmbeddingModel {
    daemon: Arc<Mutex<DaemonProcess>>,
    pub embedding_dim: usize,
}

impl EmbeddingModel {
    pub fn new(repo: impl AsRef<str>, use_gpu: bool) -> Result<Self> {
        let _ = repo;
        Self::spawn_daemon(FULL_EMBED_DIM, use_gpu)
    }

    pub fn with_dim(dim: usize) -> Result<Self> {
        Self::spawn_daemon(dim, true) // Default to GPU
    }

    fn spawn_daemon(dim: usize, use_gpu: bool) -> Result<Self> {
        eprintln!("🔌 Spawning Nomic Python Daemon...");

        // Discover Python executable more robustly
        let python_exe = std::env::var("SPLATRAG_PYTHON").or_else(|_| -> Result<String> {
            // Try common virtual environment locations
            let candidates = [
                ".venv/bin/python",
                "venv/bin/python",
                "/opt/venv/bin/python",
            ];
            for candidate in &candidates {
                if std::path::Path::new(candidate).exists() {
                    return Ok(candidate.to_string());
                }
            }
            // Fallback to system python
            Ok("python3".to_string())
        })?;

        let mut cmd = Command::new(python_exe);

        // Resolve daemon path robustly
        let daemon_path = if std::path::Path::new("crates/core/src/nomic_daemon.py").exists() {
            "crates/core/src/nomic_daemon.py"
        } else if std::path::Path::new("src/nomic_daemon.py").exists() {
            "src/nomic_daemon.py"
        } else {
            return Err(anyhow::anyhow!(
                "nomic_daemon.py not found in expected locations"
            ));
        };
        cmd.arg(daemon_path);

        // Always use GPU if available? Or pass flag?
        // Existing code had use_gpu arg.
        // Let's assume we want GPU.

        // Use absolute path or relative to CWD. CWD is workspace root.
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()) // Let Python logs show in terminal
            .spawn()
            .context("Failed to spawn nomic_daemon.py")?;

        let stdin = child.stdin.take().context("Failed to open stdin")?;
        let stdout = child.stdout.take().context("Failed to open stdout")?;
        let reader = BufReader::new(stdout);

        // Parse dim from daemon if needed, but we enforce target_dim
        println!("🔌 Nomic Daemon ready. Target dimension: {}", dim);

        Ok(Self {
            daemon: Arc::new(Mutex::new(DaemonProcess {
                child,
                stdin,
                reader,
            })),
            embedding_dim: dim,
        })
    }

    pub fn get_output_dim(&self) -> usize {
        self.embedding_dim
    }

    fn call_daemon(
        &self,
        texts: &[String],
        usage: EmbeddingUsage,
    ) -> Result<Vec<DaemonResponseItem>> {
        let mode = match usage {
            EmbeddingUsage::Query => "search_query",
            EmbeddingUsage::Document => "search_document",
            EmbeddingUsage::Tokens => "embed_tokens",
        };

        let payload = serde_json::json!({
            "texts": texts,
            "mode": mode
        })
        .to_string();

        let mut daemon = self
            .daemon
            .lock()
            .map_err(|_| anyhow::anyhow!("Failed to lock daemon"))?;

        writeln!(daemon.stdin, "{}", payload)?;
        daemon.stdin.flush()?;

        let mut response = String::new();
        daemon.reader.read_line(&mut response)?;

        if response.trim().is_empty() {
            anyhow::bail!("Empty response from daemon");
        }

        let items: Vec<DaemonResponseItem> = serde_json::from_str(&response)?;
        Ok(items)
    }

    pub fn estimate_valence(embedding: &[f32]) -> f32 {
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        norm
    }

    fn normalize(v: &mut Vec<f32>) -> f32 {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-9 {
            for x in v {
                *x /= norm;
            }
        }
        norm
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let items = self.call_daemon(&[text.to_string()], EmbeddingUsage::Query)?;
        let mut emb = items[0].pooled.clone();
        emb.truncate(self.embedding_dim); // Matryoshka
        Self::normalize(&mut emb);
        Ok(emb)
    }

    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.embed(text)
    }

    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let items = self.call_daemon(texts, EmbeddingUsage::Query)?;
        Ok(items
            .into_iter()
            .map(|mut i| {
                i.pooled.truncate(self.embedding_dim);
                i.pooled
            })
            .collect())
    }

    pub fn embed_document(&self, text: &str) -> Result<Vec<f32>> {
        let items = self.call_daemon(&[text.to_string()], EmbeddingUsage::Document)?;
        let mut emb = items[0].pooled.clone();
        emb.truncate(self.embedding_dim); // Matryoshka
        Self::normalize(&mut emb);
        Ok(emb)
    }

    pub fn embed_document_with_valence(&self, text: &str) -> Result<(Vec<f32>, f32)> {
        let items = self.call_daemon(&[text.to_string()], EmbeddingUsage::Document)?;
        let mut emb = items[0].pooled.clone();
        emb.truncate(self.embedding_dim); // Matryoshka
        let valence = Self::normalize(&mut emb);
        Ok((emb, valence))
    }

    pub fn embed_batch_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let items = self.call_daemon(texts, EmbeddingUsage::Document)?;
        let results = items
            .into_iter()
            .map(|item| {
                let mut emb = item.pooled;
                emb.truncate(self.embedding_dim); // Matryoshka
                Self::normalize(&mut emb);
                emb
            })
            .collect();
        Ok(results)
    }

    pub fn embed_tokens(&self, text: &str) -> Result<(Vec<Vec<f32>>, Vec<String>)> {
        let items = self.call_daemon(&[text.to_string()], EmbeddingUsage::Tokens)?;
        let item = &items[0];

        let sliced_tokens: Vec<Vec<f32>> = item
            .token_embeddings
            .iter()
            .map(|t| {
                let mut t = t.clone();
                t.truncate(self.embedding_dim); // Matryoshka for tokens too
                                                // Tokens might not need normalization for PCA, but let's keep it raw?
                                                // Existing code didn't normalize tokens, only pooled.
                t
            })
            .collect();

        Ok((sliced_tokens, item.tokens.clone()))
    }

    pub fn embed_batch_tokens(
        &self,
        texts: &[String],
    ) -> Result<Vec<(Vec<f32>, f32, Vec<Vec<f32>>, Vec<String>)>> {
        let items = self.call_daemon(texts, EmbeddingUsage::Tokens)?;

        let results = items
            .into_iter()
            .map(|item| {
                let mut pooled = item.pooled;
                pooled.truncate(self.embedding_dim); // Matryoshka
                let valence = Self::normalize(&mut pooled);

                let tokens = item
                    .token_embeddings
                    .into_iter()
                    .map(|t| {
                        let mut t = t.clone();
                        t.truncate(self.embedding_dim); // Matryoshka 256D
                        t
                    })
                    .collect();

                (pooled, valence, tokens, item.tokens)
            })
            .collect();
        Ok(results)
    }
}
