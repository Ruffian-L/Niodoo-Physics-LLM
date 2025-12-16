// use crate::memory::emotional::{EmotionalState, WeightedMemoryMetadata};
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct SplatFileHeader {
    pub magic: [u8; 8],
    pub version: u32,
    pub count: u64,
    pub geometry_size: u32,
    pub semantics_size: u32,
    pub motion_size: u32,
    pub lighting_size: u32,
    pub _pad: [u32; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct SplatManifestEntry {
    pub id: u64,
    pub text: String,
    pub birth_time: f64,
    #[serde(default)]
    pub valence_history: Vec<f32>,
    #[serde(default)]
    pub initial_valence: i8,
    #[serde(default)]
    pub tags: Vec<String>,
}

// The "Static Splat" (Context/Setting)
// 48 bytes
#[repr(C, align(16))]
#[derive(
    Debug,
    Clone,
    Copy,
    // Pod, Zeroable removed
    Serialize,
    Deserialize,
    Archive,
    RkyvSerialize,
    RkyvDeserialize,
)]
pub struct SplatGeometry {
    // Position: Can be interpreted as 3D [x, y, z]
    pub position: [f32; 3], // 12 bytes

    pub scale: [f32; 3],        // 12 bytes
    pub rotation: [f32; 4],     // 16 bytes
    pub color_rgba: [u8; 4],    // 4 bytes (Albedo + Opacity packed)
    pub physics_props: [u8; 4], // 4 bytes (Roughness, Metallic, Valence, Pad)

    // Light-EBM: Domain Crystallization (Phase 2)
    // [Code, Math, Language, Logic] - L1 normalized (sum = 1.0)
    pub domain_valence: [f32; 4], // 16 bytes

                                  // Total: 76 bytes (increased from 64 bytes for tri-plane support)
}

pub type StaticSplat = SplatGeometry;

impl Default for SplatGeometry {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            scale: [1.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            color_rgba: [255, 255, 255, 255],
            physics_props: [0, 0, 0, 0],
            domain_valence: [0.25, 0.25, 0.25, 0.25], // Neutral: uniform distribution
        }
    }
}

// The "Dynamic Splat" (Action/Event)
// 20 bytes -> Pad to 24 or 32?
// For alignment, let's use [f32; 3] + f32 + f32 + f32 = 24 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct SplatMotion {
    pub velocity: [f32; 3],  // 12 bytes
    pub covariance_det: f32, // 4 bytes (Uncertainty)
    pub time_birth: f32,     // 4 bytes
    pub time_death: f32,     // 4 bytes
}

// unsafe impl Zeroable for SplatMotion {}
// unsafe impl Pod for SplatMotion {}

// The "Lighting Splat" (Layered Light Transport)
// Encodes baked lighting data from Inverse Rendering
#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct SplatLighting {
    pub normal: [f32; 3],         // Optimized Surface Orientation
    pub idiv: [f32; 3],           // Integrated Directional Illumination Vector
    pub ide: [f32; 3],            // Integrated Directional Encoding
    pub sss_params: [f32; 4],     // Subsurface Scattering Parameters (Diffusion radius etc)
    pub sh_occlusion: [f32; 7],   // Spherical Harmonics Occlusion (Band 2, reduced from 9)
    pub domain_valence: [f32; 4], // Light-EBM: [Code, Math, Language, Logic]
    pub _pad: [f32; 0],           // Removed - total still 24 floats (96 bytes)
}

impl Default for SplatLighting {
    fn default() -> Self {
        Self {
            normal: [0.0, 1.0, 0.0],
            idiv: [0.0; 3],
            ide: [0.0; 3],
            sss_params: [0.0; 4],
            sh_occlusion: [0.0; 7],
            domain_valence: [0.25, 0.25, 0.25, 0.25], // Neutral
            _pad: [],
        }
    }
}

// unsafe impl Zeroable for SplatLighting {}
// unsafe impl Pod for SplatLighting {}

// COLD: Heavy data, accessed only during RAG/semantic query
#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct SplatSemantics {
    pub payload_id: u64,
    pub birth_time: f64,
    pub confidence: f32,

    #[serde(with = "BigArray")]
    pub embedding: [f32; crate::constants::FULL_EMBED_DIM], // 3072 bytes (768 * 4)

    // Manifold Vector (64-dim subspace)
    #[serde(with = "BigArray")]
    pub manifold_vector: [f32; 64], // 256 bytes

    // --- God Protocol Additions ---
    // Emotional state stubbed
    #[serde(default)]
    pub emotional_state: Option<()>,

    // Metadata stubbed
    #[serde(default)]
    pub fitness_metadata: Option<()>,
}

/// Manifesto-compliant discrete RVQ-based semantics (Version 2)
/// Stores 12 discrete codes instead of dense 768D embedding
/// Storage: ~48 bytes vs ~3072 bytes (64-128× compression)
#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct SplatSemanticsV2 {
    pub payload_id: u64,
    pub birth_time: f64,
    pub confidence: f32,

    /// RVQ discrete codes
    pub rvq_indices: [u16; 16], // Hardcoded constant to avoid crate::encoder import

    /// Mass derived from code[0] rarity (self-information)
    /// Rarer coarse codes = higher mass = stronger gravity wells
    pub coarse_mass: u8,

    /// Domain valence (kept from v1 for compatibility)
    pub domain_valence: [f32; 4],

    /// Manifold vector (64-dim subspace) - kept for compatibility
    #[serde(with = "BigArray")]
    pub manifold_vector: [f32; 64],

    // --- God Protocol Additions (kept) ---
    // Emotional state stubbed
    #[serde(default)]
    pub emotional_state: Option<()>,

    // Metadata stubbed
    #[serde(default)]
    pub fitness_metadata: Option<()>,
}

impl Default for SplatSemanticsV2 {
    fn default() -> Self {
        Self {
            payload_id: 0,
            birth_time: 0.0,
            confidence: 1.0,
            rvq_indices: [0u16; 16],
            coarse_mass: 128, // Neutral mass
            domain_valence: [0.25, 0.25, 0.25, 0.25],
            manifold_vector: [0.0; 64],
            emotional_state: None,
            fitness_metadata: None,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct PackedSemantics {
    pub position: [f32; 3],
    pub opacity: f32,
    pub scale: [f32; 3],
    pub _pad1: f32,
    pub rotation: [f32; 4],
    pub query_vector: [f32; 16],
}

impl Default for PackedSemantics {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            opacity: 1.0,
            scale: [1.0; 3],
            _pad1: 0.0,
            rotation: [0.0, 0.0, 0.0, 1.0],
            query_vector: [0.0; 16],
        }
    }
}

// unsafe impl Zeroable for PackedSemantics {}
// unsafe impl Pod for PackedSemantics {}

#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct SplatManifest {
    pub entries: Vec<SplatManifestEntry>,
}

impl SplatManifest {
    pub fn to_map(&self) -> std::collections::HashMap<u64, String> {
        self.entries
            .iter()
            .map(|e| (e.id, e.text.clone()))
            .collect()
    }
}

impl Default for SplatMotion {
    fn default() -> Self {
        Self {
            velocity: [0.0; 3],
            covariance_det: 1.0,
            time_birth: 0.0,
            time_death: 0.0,
        }
    }
}

impl Default for SplatSemantics {
    fn default() -> Self {
        Self {
            payload_id: 0,
            birth_time: 0.0,
            confidence: 1.0,
            embedding: [0.0; crate::constants::FULL_EMBED_DIM],
            manifold_vector: [0.0; 64],
            emotional_state: None,
            fitness_metadata: None,
        }
    }
}
