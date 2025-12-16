use hf_hub::api::sync::Api;
use std::path::{Path, PathBuf};

/// Download a list of files from a Hugging Face model repo into the local `models/` folder.
/// Returns the local paths of the downloaded files (or existing cached ones).
pub fn download_files(repo_id: &str, files: &[&str]) -> anyhow::Result<Vec<PathBuf>> {
    let api = Api::new()?;
    let repo = api.model(repo_id.to_string());

    let mut out = Vec::new();
    let models_dir = Path::new("models");
    if !models_dir.exists() {
        std::fs::create_dir_all(models_dir)?;
    }

    for f in files {
        // `get` will return a local cached path for the file
        match repo.get(f) {
            Ok(path) => {
                // Copy into our `models/` folder so other code can find it easily
                let filename = Path::new(f)
                    .file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| f.replace('/', "_"));
                let dest = models_dir.join(filename);
                // If dest already exists we leave it alone; otherwise copy from cache.
                if !dest.exists() {
                    std::fs::copy(&path, &dest)?;
                }
                out.push(dest);
            }
            Err(e) => {
                // Best-effort: continue and report missing file
                eprintln!("hf-hub: failed to get {} from {}: {}", f, repo_id, e);
            }
        }
    }

    Ok(out)
}
