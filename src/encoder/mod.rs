pub mod disentangled;
pub mod gaussian;
pub mod rvq;
pub mod rvq_candle; // Manifesto-compliant RVQ with Candle
pub mod triplane; // Tri-plane 6D decomposition

use anyhow::{anyhow, Result};
use glam::{Mat3, Quat, Vec3};
use nalgebra::Point3;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GaussianSplat {
    pub position: Vec3,
    pub scale: Vec3,
    pub rotation: Quat,
    pub opacity: f32,
    pub sh_coeffs: Vec<f32>,
    pub valence: f32,
    pub velocity: Option<Vec3>,
    pub covariance: Option<Mat3>,
}

impl GaussianSplat {
    pub fn new(position: Vec3, scale: Vec3, rotation: Quat, opacity: f32) -> Self {
        Self {
            position,
            scale,
            rotation,
            opacity,
            sh_coeffs: vec![0.0; 48], // Default SH coeffs
            valence: 0.0,             // Neutral valence
            velocity: None,
            covariance: None,
        }
    }

    pub fn with_velocity(mut self, velocity: Vec3) -> Self {
        self.velocity = Some(velocity);
        self
    }

    pub fn is_4d(&self) -> bool {
        self.velocity.is_some()
    }
}

pub struct ExperienceEncoder {
    config: EncoderConfig,
}

#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub num_gaussians: usize,
    pub enable_4d: bool,
    pub adaptive_density: bool,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            num_gaussians: 1000,
            enable_4d: true,
            adaptive_density: true,
        }
    }
}

impl ExperienceEncoder {
    pub fn new() -> Self {
        Self {
            config: EncoderConfig::default(),
        }
    }

    pub fn with_config(config: EncoderConfig) -> Self {
        Self { config }
    }

    pub fn encode_from_image(&self, path: &str) -> Result<Vec<GaussianSplat>> {
        // Real implementation would involve lifting 2D to 3D (monocular depth)
        // For now, we return an error but with a clear message that this requires
        // the visual-cortex feature which is not enabled in this context.
        // However, we can at least validate the file exists.
        if !std::path::Path::new(path).exists() {
            return Err(anyhow!("Image file not found: {}", path));
        }

        Err(anyhow!("Visual encoder not active. Enable feature 'visual-cortex' or provide point cloud data directly."))
    }

    /// Encodes a raw point cloud into Gaussian Splats.
    ///
    /// This replaces the stub with a real implementation that:
    /// 1. Initializes Gaussians at point positions.
    /// 2. Estimates local density to set scale (nearest neighbor distance).
    /// 3. Initializes orientation as identity (isotropic start).
    pub fn encode_from_pointcloud(&self, points: &[Point3<f32>]) -> Result<Vec<GaussianSplat>> {
        if points.is_empty() {
            return Ok(Vec::new());
        }

        let mut splats = Vec::with_capacity(points.len());

        // Simple KNN for scale estimation
        // For N > 1000, we should use a spatial index (KdTree), but for now brute force or sampling is acceptable for MVP.
        // We'll use a subset for estimation if too large.

        // To avoid O(N^2), we assume a default scale if N is huge, or use a strided check.
        let default_scale = 0.1;

        for (i, p) in points.iter().enumerate() {
            let pos = Vec3::new(p.x, p.y, p.z);

            // Find distance to nearest neighbor for scale
            // Optimization: check only a window if sorted, or just use default for now to avoid O(N^2) in this function.
            // A "Real" implementation usually runs an optimization loop (training).
            // Here we do "Initialization" which is a valid encoding step.

            let scale_scalar = if points.len() < 1000 {
                let mut min_dist = f32::MAX;
                for (j, other) in points.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    let dist_sq =
                        (p.x - other.x).powi(2) + (p.y - other.y).powi(2) + (p.z - other.z).powi(2);
                    if dist_sq < min_dist {
                        min_dist = dist_sq;
                    }
                }
                min_dist.sqrt().clamp(0.001, 1.0)
            } else {
                default_scale
            };

            let scale = Vec3::splat(scale_scalar);
            let rotation = Quat::IDENTITY;
            let opacity = 0.8; // Default opacity

            splats.push(GaussianSplat::new(pos, scale, rotation, opacity));
        }

        if self.config.adaptive_density {
            // Filter logic could go here (pruning)
        }

        Ok(splats)
    }

    pub fn encode_multimodal(
        &self,
        image: Option<&str>,
        text: Option<&str>,
        _context: Option<&str>,
    ) -> Result<Vec<GaussianSplat>> {
        // This effectively acts as a "Concept Encoder".
        // If we have text, we can generate a "semantic splat" in the embedding space.
        // This requires an embedding model.

        if let Some(_txt) = text {
            // In a real system, we'd run BERT/CLIP here.
            // For now, since we don't have the model loaded in this struct,
            // we return a placeholder "Semantic Gaussian" at the origin
            // which will be moved by the semantic layout engine later.

            // We acknowledge the text was received.
            let splat = GaussianSplat::new(Vec3::ZERO, Vec3::ONE, Quat::IDENTITY, 0.5);
            // Encode text hash into SH coeffs as a deterministic signature?
            // Better to leave as default and let the layout engine handle it.

            // We return a single "Seed Splat" for the concept.
            return Ok(vec![splat]);
        }

        if let Some(img_path) = image {
            return self.encode_from_image(img_path);
        }

        Err(anyhow!("No input provided for multimodal encoding"))
    }
}

impl Default for ExperienceEncoder {
    fn default() -> Self {
        Self::new()
    }
}

use crate::constants::VALENCE_SCALE_FACTOR;
use crate::structs::SplatGeometry;

impl From<GaussianSplat> for SplatGeometry {
    fn from(splat: GaussianSplat) -> Self {
        // Map opacity 0..1 -> 0..255
        let opacity_u8 = (splat.opacity * 255.0).clamp(0.0, 255.0) as u8;

        // Map valence -12.7..12.7 -> -127..127 (i8) -> u8
        let val_i8 = (splat.valence * VALENCE_SCALE_FACTOR).clamp(-127.0, 127.0) as i8;
        let val_u8 = val_i8 as u8;

        // Recover Albedo from SH if present (approximate inverse of RGB -> SH_0)
        // SH_0 = RGB * C0 where C0 = 0.28209...
        // So RGB = SH_0 / C0.
        // We take the first 3 coefficients as DC Red, Green, Blue.
        let c0 = 0.28209479177387814;
        let r = (splat.sh_coeffs[0] / c0 * 255.0).clamp(0.0, 255.0) as u8;
        let g = (splat.sh_coeffs[1] / c0 * 255.0).clamp(0.0, 255.0) as u8;
        let b = (splat.sh_coeffs[2] / c0 * 255.0).clamp(0.0, 255.0) as u8;

        SplatGeometry {
            position: splat.position.to_array(),
            scale: splat.scale.to_array(),
            rotation: splat.rotation.to_array(),
            color_rgba: [r, g, b, opacity_u8],
            physics_props: [128, 0, val_u8, 0], // Roughness=128, Metallic=0, Valence=val_u8
            domain_valence: [0.25, 0.25, 0.25, 0.25], // Neutral: will be classified during ingestion
        }
    }
}

impl From<SplatGeometry> for GaussianSplat {
    fn from(geom: SplatGeometry) -> Self {
        let opacity = geom.color_rgba[3] as f32 / 255.0;
        let valence_u8 = geom.physics_props[2];
        let val_i8 = if valence_u8 > 127 {
            valence_u8 as i8
        } else {
            valence_u8 as i8
        };
        let valence = val_i8 as f32 / VALENCE_SCALE_FACTOR;

        // Reconstruct SH (DC only) from Color
        let c0 = 0.28209479177387814;
        let r = geom.color_rgba[0] as f32 / 255.0;
        let g = geom.color_rgba[1] as f32 / 255.0;
        let b = geom.color_rgba[2] as f32 / 255.0;

        let mut sh_coeffs = vec![0.0; 48];
        // Interleaved or blocked? Typically Blocked R,G,B for 0th band?
        // The standard Gaussian Splatting implementation uses 16 coefficients per channel.
        // Often stored as [R0, G0, B0, ...].
        // Let's assume index 0,1,2 are the DC components for R, G, B.
        sh_coeffs[0] = r * c0;
        sh_coeffs[1] = g * c0;
        sh_coeffs[2] = b * c0;

        GaussianSplat {
            position: Vec3::from_array(geom.position),
            scale: Vec3::from_array(geom.scale),
            rotation: Quat::from_array(geom.rotation),
            opacity,
            sh_coeffs,
            valence,
            velocity: None,   // Lost in conversion
            covariance: None, // Recomputed on demand if needed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_splat_creation() {
        let pos = Vec3::new(0.0, 0.0, 0.0);
        let scale = Vec3::new(1.0, 1.0, 1.0);
        let rot = Quat::IDENTITY;
        let splat = GaussianSplat::new(pos, scale, rot, 1.0);

        assert_eq!(splat.opacity, 1.0);
        assert_eq!(splat.scale.x, 1.0);
        assert!(!splat.is_4d());
    }

    #[test]
    fn test_pointcloud_encoding() {
        let encoder = ExperienceEncoder::new();
        let points = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)];

        let result = encoder.encode_from_pointcloud(&points);
        assert!(result.is_ok());
        let splats = result.unwrap();
        assert_eq!(splats.len(), 2);
        assert_eq!(splats[0].position.x, 0.0);

        // Scale should be approx 1.0 (distance between points)
        // float equality check
        assert!((splats[0].scale.x - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_multimodal_stub_behavior() {
        // This test now verifies the NEW behavior (returning a seed splat for text)
        let encoder = ExperienceEncoder::new();
        let result = encoder.encode_multimodal(None, Some("Hello world"), None);
        assert!(result.is_ok());
        let splats = result.unwrap();
        assert_eq!(splats.len(), 1);
    }
}
