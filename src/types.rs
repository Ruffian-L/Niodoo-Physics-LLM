// use crate::memory::emotional::{EmotionalState, WeightedMemoryMetadata};
use serde::{Deserialize, Serialize};

pub type Point3 = [f32; 3];
pub type Vec3 = [f32; 3];
pub type Mat3 = [f32; 9];
pub type SplatId = u64;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SplatMeta {
    pub timestamp: Option<f64>,
    pub labels: Vec<String>,
    // Stubbed fields removed for release compatibility
}

impl SplatMeta {
    pub fn birth_time(&self) -> Option<f64> {
        self.timestamp
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SplatInput {
    pub static_points: Vec<Point3>,
    pub covariances: Vec<Mat3>,
    pub motion_velocities: Option<Vec<Vec3>>,
    pub meta: SplatMeta,

    // --- Layered Light Transport Encoding ---
    #[serde(default)]
    pub normals: Option<Vec<Vec3>>, // Optimized Surface Orientation
    #[serde(default)]
    pub idiv: Option<Vec<Vec3>>, // Integrated Directional Illumination Vector
    #[serde(default)]
    pub ide: Option<Vec<Vec3>>, // Integrated Directional Encoding
    #[serde(default)]
    pub sss_params: Option<Vec<[f32; 4]>>, // Subsurface Scattering Parameters
    #[serde(default)]
    pub sh_occlusion: Option<Vec<[f32; 9]>>, // Spherical Harmonics Occlusion
}
