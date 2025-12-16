// src/constants.rs

/// Scale factor for mapping floating point valence (-1.0 to 1.0 range) to integer storage
pub const VALENCE_SCALE_FACTOR: f32 = 10.0;

/// Default number of Spherical Harmonic coefficients (Degree 3 = 16 * 3 = 48)
pub const SH_COEFF_COUNT: usize = 48;

/// Default constant for Spherical Harmonics (Band 0)
pub const SH_C0: f32 = 0.28209479177387814;

pub const GPRIME_SCALE_RATIOS: [f32; 3] = [1.0, 0.618, 0.382]; // Golden ratio approximations

/// Size of the phoneme space for language processing
pub const PHONEME_SPACE: u16 = 32768;

/// Multiplier for re-ranking candidates in retrieval
pub const RERANK_MULTIPLIER: usize = 4;

/// Full Nomic embedding dimension (before Matryoshka truncation)
pub const FULL_EMBED_DIM: usize = 768;

/// Embedding dimension for the vector space (Matryoshka truncation)
pub const EMBED_DIM: usize = 512;

/// Level of Detail dimensions for far-field approximation
pub const LOD_DIMS: usize = 64;

/// RVQ latent dimension (further compression for discrete codes)
pub const RVQ_LATENT_DIM: usize = 1024;

/// Configuration for Topological Data Analysis (TDA) defaults
pub mod tda {
    pub const DEFAULT_MAX_POINTS: usize = 2000;
    pub const DEFAULT_CONNECTIVITY_THRESHOLD: f32 = 2.0;
    pub const CIRCLE_VARIANCE_THRESHOLD: f32 = 0.5;
    pub const CIRCLE_MIN_RADIUS: f32 = 0.1;
}

/// Default filenames for the system
pub mod filenames {
    pub const DEFAULT_SPLAT_FILE: &str = "mindstream_current";
    pub const DEFAULT_MANIFEST_FILE: &str = "mindstream_manifest.json";
    pub const DEFAULT_GEOMETRY_FILE: &str = "mindstream_current.geom";
    pub const DEFAULT_SEMANTICS_FILE: &str = "mindstream_current.sem";
    pub const DEFAULT_STATE_FILE: &str = "shadow_state.json";
}
