#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::gpu::SplatInput;
    use crate::tivm::SplatRagConfig;
    use crate::types::{Point3, SplatMeta};

    #[test]
    fn test_gpu_availability_check() {
        let available = cuda_available();
        println!("CUDA available: {}", available);

        if available {
            let count = device_count().unwrap();
            println!("Found {} CUDA device(s)", count);
            assert!(count > 0);
        }
    }

    #[test]
    fn test_gpu_env_detection() {
        // Test without env var
        std::env::remove_var("SPLATRAG_USE_GPU");
        assert!(!should_use_gpu());

        // Test with env var but might not have CUDA
        std::env::set_var("SPLATRAG_USE_GPU", "1");
        let expected = cuda_available();
        assert_eq!(should_use_gpu(), expected);

        // Clean up
        std::env::remove_var("SPLATRAG_USE_GPU");
    }

    #[test]
    #[ignore] // Only run when CUDA is available
    fn test_gpu_fingerprint_computation() {
        if !cuda_available() {
            println!("Skipping GPU fingerprint test - CUDA not available");
            return;
        }

        std::env::set_var("SPLATRAG_USE_GPU", "1");

        let splat = SplatInput {
            static_points: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            covariances: vec![[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]; 3],
            motion_velocities: None,
            meta: SplatMeta::default(),
            normals: None,
            idiv: None,
            ide: None,
            sss_params: None,
            sh_occlusion: None,
        };

        let cfg = SplatRagConfig::default();

        // This should use GPU path (stubbed)
        let result = try_gpu_fingerprint(&splat, &cfg);

        assert!(result.is_ok());

        std::env::remove_var("SPLATRAG_USE_GPU");
    }
}
