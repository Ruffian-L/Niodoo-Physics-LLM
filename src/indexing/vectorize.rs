use crate::config::SplatMemoryConfig;

/// Converts a persistence diagram (birth/death pairs) into a vector.
/// Uses Persistence Landscapes (k=0, dominant features).
pub fn compute_vector_persistence_landscape(
    diagram: &[(f32, f32)],
    config: &SplatMemoryConfig,
) -> Vec<f32> {
    let resolution = config.tda.resolution;
    let mut vector = vec![0.0; resolution];

    if diagram.is_empty() {
        return vector;
    }

    // Find bounds to normalize the landscape
    let min_birth = diagram.iter().map(|p| p.0).fold(f32::INFINITY, f32::min);
    let max_death = diagram
        .iter()
        .map(|p| p.1)
        .fold(f32::NEG_INFINITY, f32::max);

    // Avoid division by zero if all points are identical
    let range = if (max_death - min_birth).abs() < f32::EPSILON {
        1.0
    } else {
        max_death - min_birth
    };

    let step = range / resolution as f32;

    for i in 0..resolution {
        let t = min_birth + (i as f32 * step);

        // Find the maximum landscape height at time t
        // Landscape function: f(t) = max(0, min(t-b, d-t))
        let max_val = diagram
            .iter()
            .map(|(b, d)| (t - b).min(d - t).max(0.0))
            .fold(0.0f32, f32::max);

        vector[i] = max_val;
    }

    vector
}

// Stub for Image Persistence (keep this simple for now)
pub fn compute_vector_persistence_image(
    _diagram: &[(f32, f32)],
    config: &SplatMemoryConfig,
) -> Vec<f32> {
    // Fallback to landscape or return empty.
    vec![0.0; config.tda.resolution]
}

pub fn vector_persistence_block(
    diagram: &crate::indexing::PersistenceDiagram,
    _params: &crate::tivm::VpbParams,
) -> Vec<f32> {
    // Backward compatibility wrapper
    // Convert PersistenceDiagram to slice
    // We ignore params.weight_fn for now as we moved to config-based landscapes

    let pairs: Vec<(f32, f32)> = diagram.pairs.clone();
    let config = SplatMemoryConfig::default(); // Use default config for legacy calls

    let landscape = compute_vector_persistence_landscape(&pairs, &config);

    // Pad to 8 features to match old VpbParams expectation if needed by downstream
    // The old VPB was 8 floats. The new one is 100 (resolution).
    // We should probably return the full landscape now.
    // But to satisfy the trait signature if it expects fixed size...
    // Let's stick to the new resolution.

    landscape
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SplatMemoryConfig;

    #[test]
    fn test_persistence_landscape_generation() {
        // 1. Setup Config
        let mut config = SplatMemoryConfig::default();
        config.tda.resolution = 10; // Keep it small for easy debugging

        // 2. Define two distinct topological scenarios

        // Scenario A: One massive feature (e.g., a large loop)
        // Born at 0.0, Dies at 10.0. Midpoint (peak) at 5.0.
        let diagram_a = vec![(0.0, 10.0)];

        // Scenario B: Two smaller, noisy features
        // Feature 1: Born 0.0, Dies 4.0
        // Feature 2: Born 6.0, Dies 10.0
        let diagram_b = vec![(0.0, 4.0), (6.0, 10.0)];

        // 3. Compute Landscapes
        let vec_a = compute_vector_persistence_landscape(&diagram_a, &config);
        let vec_b = compute_vector_persistence_landscape(&diagram_b, &config);

        // 4. Assertions
        // Check Resolution
        assert_eq!(vec_a.len(), 10, "Vector A length should match resolution");
        assert_eq!(vec_b.len(), 10, "Vector B length should match resolution");

        // Check for Zeros (The "Stub" Check)
        let sum_a: f32 = vec_a.iter().sum();
        assert!(sum_a > 0.0, "Vector A should not be all zeros");

        // Check Differentiation (The "Fingerprint" Check)
        // If the math is working, these two vectors must be different.
        assert_ne!(
            vec_a, vec_b,
            "Different topologies must yield different vectors"
        );

        // Optional: Check logic correctness
        // For Diagram A (0,10), the peak is at t=5.
        // With resolution 10 over range [0,10], index 5 represents t=5.
        // f(5) = min(5-0, 10-5) = 5.
        // Let's check if the middle of the vector has a high value.
        let mid_index = 5;
        assert!(
            vec_a[mid_index] > 0.0,
            "Peak should exist near the middle for Diagram A"
        );

        println!("Vector A (Large Loop): {:?}", vec_a);
        println!("Vector B (Noise):      {:?}", vec_b);
    }
}
