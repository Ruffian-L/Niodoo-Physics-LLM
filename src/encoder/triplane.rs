// crates/core/src/encoder/triplane.rs
// Tri-Plane Decomposition for 6D semantic positioning (Manifesto §4.3)
use candle_core::{IndexOp, Result, Tensor};

/// Projects 128D RVQ latent into 6D tri-plane coordinates
/// Represents a text particle on 3 orthogonal 2D planes: (XY, XZ, YZ)
pub struct TriplaneProjector;

impl TriplaneProjector {
    /// Project a 128D latent vector to 32D manifold coordinates
    ///
    /// Uses the first 32 dimensions of the coarse RVQ reconstruction
    /// and normalizes to a bounded space.
    ///
    /// # Arguments
    /// * `latent_128d` - Tensor of shape (1, 128) or (batch, 128)
    ///
    /// # Returns
    /// * 32D coordinates for each batch item
    pub fn project(latent_128d: &Tensor) -> Result<Vec<[f32; 32]>> {
        let dims = latent_128d.dims();
        let batch_size = if dims.len() == 1 { 1 } else { dims[0] };

        // Extract first 32 dimensions
        let tri_plane = if dims.len() == 1 {
            latent_128d.i(0..32)?
        } else {
            latent_128d.i((.., 0..32))?
        };

        // Normalize to [-10, 10] range (physics sim space)
        // Using tanh for smooth bounded mapping
        let tri_plane = tri_plane.tanh()?;
        let tri_plane = (tri_plane * 10.0)?;

        // Convert to Vec<[f32; 32]>
        let tri_plane_vec: Vec<Vec<f32>> = tri_plane.to_vec2()?;
        let mut result = Vec::with_capacity(batch_size);

        for row in tri_plane_vec {
            let mut coords = [0.0f32; 32];
            for (i, val) in row.iter().enumerate().take(32) {
                coords[i] = *val;
            }
            result.push(coords);
        }

        Ok(result)
    }

    /// Project a single latent vector to 32D (convenience method)
    pub fn project_single(latent: &[f32]) -> [f32; 32] {
        let mut coords = [0.0f32; 32];

        for i in 0..32 {
            // Tanh normalization to [-1, 1], then scale to [-10, 10]
            // Handle case where latent is shorter than 32 (unlikely but safe)
            let val = if i < latent.len() { latent[i] } else { 0.0 };
            coords[i] = val.tanh() * 10.0;
        }

        coords
    }

    /// Decompose 6D coordinates back to 3 plane positions
    /// Returns: ((u1, v1), (u2, v2), (u3, v3))
    pub fn decompose_to_planes(coords_6d: &[f32; 6]) -> ((f32, f32), (f32, f32), (f32, f32)) {
        (
            (coords_6d[0], coords_6d[1]), // Plane 1 (Topic-Context)
            (coords_6d[2], coords_6d[3]), // Plane 2 (Syntax-Structure)
            (coords_6d[4], coords_6d[5]), // Plane 3 (Entity-Attribute)
        )
    }

    /// Compute rough 3D position from tri-plane (for distance culling)
    /// Takes average of the U coordinates
    pub fn approx_3d_position(coords_6d: &[f32; 6]) -> [f32; 3] {
        [
            coords_6d[0],                        // X from Plane 1 U
            coords_6d[1],                        // Y from Plane 1 V
            (coords_6d[2] + coords_6d[4]) / 2.0, // Z from avg of Plane 2,3 U
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_triplane_projection() -> Result<()> {
        let device = Device::Cpu;

        // Create a random 128D latent
        let latent = Tensor::randn(0f32, 1.0, (1, 128), &device)?;

        // Project to tri-plane
        let coords_batch = TriplaneProjector::project(&latent)?;

        assert_eq!(coords_batch.len(), 1);
        let coords = coords_batch[0];

        // Check all coords are in [-10, 10] range (due to tanh)
        for &val in &coords {
            assert!(val >= -10.0 && val <= 10.0, "Coord {} out of bounds", val);
        }

        Ok(())
    }

    #[test]
    fn test_triplane_batch_projection() -> Result<()> {
        let device = Device::Cpu;

        // Batch of 4
        let latent = Tensor::randn(0f32, 1.0, (4, 128), &device)?;
        let coords_batch = TriplaneProjector::project(&latent)?;

        assert_eq!(coords_batch.len(), 4);

        Ok(())
    }

    #[test]
    fn test_triplane_decompose() {
        let coords = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let planes = TriplaneProjector::decompose_to_planes(&coords);

        assert_eq!(planes.0, (1.0, 2.0));
        assert_eq!(planes.1, (3.0, 4.0));
        assert_eq!(planes.2, (5.0, 6.0));
    }

    #[test]
    fn test_approx_3d_position() {
        let coords = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let pos_3d = TriplaneProjector::approx_3d_position(&coords);

        assert_eq!(pos_3d[0], 1.0);
        assert_eq!(pos_3d[1], 2.0);
        assert_eq!(pos_3d[2], (3.0 + 5.0) / 2.0); // Average of plane 2,3 U coords
    }

    #[test]
    fn test_single_projection_bounded() {
        let mut latent = [0.0f32; 128];
        latent[0] = 100.0; // Extreme value
        latent[1] = -100.0;

        let coords = TriplaneProjector::project_single(&latent);

        // Should be bounded by tanh
        assert!(coords[0] >= -10.0 && coords[0] <= 10.0);
        assert!(coords[1] >= -10.0 && coords[1] <= 10.0);
        assert!(coords[0] > 9.0, "Large positive should map near +10");
        assert!(coords[1] < -9.0, "Large negative should map near -10");
    }
}
