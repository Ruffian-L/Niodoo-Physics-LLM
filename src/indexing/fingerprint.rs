use crate::encoder::GaussianSplat;
use crate::tivm::SplatRagConfig; // Corrected import
use anyhow::Result;

// --- Config & Constants ---

#[derive(Debug, Clone)]
pub struct FingerprintConfig {
    pub max_points: usize,
    pub connectivity_threshold: f32,
    pub use_gpu: bool,
}

impl Default for FingerprintConfig {
    fn default() -> Self {
        Self {
            max_points: 2000,
            connectivity_threshold: 2.0,
            use_gpu: true,
        }
    }
}

// Ensure this struct matches other definitions if duplicated in indexing/mod.rs
// The compiler error suggests duplicate definitions.
// We should probably remove this if it's defined elsewhere, but indexing/mod.rs usually re-exports.
// Let's check indexing/mod.rs content. It re-exports from here?
// "note: `fingerprint::TopologicalFingerprint` is defined in module `crate::indexing::fingerprint`"
// "note: `indexing::TopologicalFingerprint` is defined in module `crate::indexing`"
// If `indexing/mod.rs` defines its own struct instead of re-exporting, that's the issue.

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TopologicalFingerprint {
    pub h0_barcode: Vec<(f32, f32)>, // Birth, Death
    pub h1_barcode: Vec<(f32, f32)>,
    // Additional features like Betti curves can be derived
}

// --- Core Logic ---

pub fn fingerprint_from_splat(
    splat: &crate::SplatInput,
    _config: &SplatRagConfig,
) -> TopologicalFingerprint {
    use crate::indexing::persistent_homology::{PhConfig, PhEngine, PhStrategy};

    // Extract points from SplatInput
    // Assuming splat.static_points is Vec<[f32; 3]>
    let points = &splat.static_points;

    if points.is_empty() {
        return TopologicalFingerprint::new(vec![], vec![]);
    }

    // Configure PhEngine
    let engine = PhEngine::new(PhConfig {
        max_dimension: 2,
        gpu_enabled: false,
        gpu_heap_capacity: 1024,
        hom_dims: vec![0, 1], // Compute H0 and H1
        strategy: PhStrategy::ExactBatch,
        max_points: 1000,
        connectivity_threshold: 5.0,
    });

    // Compute Persistence Diagram
    let pd = engine.compute_pd(points);

    // Extract features
    let h0 = pd.features_by_dim.get(0).cloned().unwrap_or_default();
    let h1 = pd.features_by_dim.get(1).cloned().unwrap_or_default();

    TopologicalFingerprint::new(h0, h1)
}

impl TopologicalFingerprint {
    pub fn new(h0: Vec<(f32, f32)>, h1: Vec<(f32, f32)>) -> Self {
        Self {
            h0_barcode: h0,
            h1_barcode: h1,
        }
    }

    pub fn to_vector(&self) -> Vec<f32> {
        let mut v = vec![0.0; 64];
        v[0] = self.h0_barcode.len() as f32;
        v[1] = self.h1_barcode.len() as f32;
        v
    }

    pub fn distance(&self, other: &Self) -> f32 {
        let h0_diff = (self.h0_barcode.len() as f32 - other.h0_barcode.len() as f32).abs();
        let h1_diff = (self.h1_barcode.len() as f32 - other.h1_barcode.len() as f32).abs();
        h0_diff + h1_diff
    }
}

// --- TDA Pipeline Steps ---

pub fn compute_4d_qr_fingerprint(_splats: &[GaussianSplat]) -> Result<TopologicalFingerprint> {
    anyhow::bail!("4D QR Fingerprint computation is not yet implemented.")
}

pub fn compute_fingerprint_from_points(splats: &[GaussianSplat]) -> TopologicalFingerprint {
    use crate::indexing::persistent_homology::{PhConfig, PhEngine, PhStrategy};

    let points: Vec<[f32; 3]> = splats
        .iter()
        .map(|s| [s.position.x, s.position.y, s.position.z])
        .collect();

    if points.is_empty() {
        return TopologicalFingerprint::new(vec![], vec![]);
    }

    let engine = PhEngine::new(PhConfig {
        max_dimension: 2,
        gpu_enabled: false,
        gpu_heap_capacity: 1024,
        hom_dims: vec![0, 1],
        strategy: PhStrategy::ExactBatch,
        max_points: 1000,
        connectivity_threshold: 5.0,
    });

    let pd = engine.compute_pd(&points);

    let h0 = pd.features_by_dim.get(0).cloned().unwrap_or_default();
    let h1 = pd.features_by_dim.get(1).cloned().unwrap_or_default();

    TopologicalFingerprint::new(h0, h1)
}

pub fn cosine_similarity(fp1: &TopologicalFingerprint, fp2: &TopologicalFingerprint) -> f32 {
    let v1 = fp1.to_vector();
    let v2 = fp2.to_vector();
    let dot: f32 = crate::utils::fidelity::robust_dot(&v1, &v2);
    let mag1: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag2: f32 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag1 == 0.0 || mag2 == 0.0 {
        0.0
    } else {
        dot / (mag1 * mag2)
    }
}

// Added dummy function to satisfy dual_process.rs import
pub fn wasserstein_distance(fp1: &TopologicalFingerprint, fp2: &TopologicalFingerprint) -> f32 {
    fp1.distance(fp2)
}
