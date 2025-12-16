use glam::{Mat3, Quat, Vec3};

pub fn compute_covariance_from_scale_rotation(scale: &Vec3, rotation: &Quat) -> Mat3 {
    let s = Mat3::from_diagonal(*scale);
    let r = Mat3::from_quat(*rotation);
    let cov = r * s * s.transpose() * r.transpose();

    // Apply robust clamping
    let mut arr = cov.to_cols_array();
    crate::utils::fidelity::clamp_covariance(&mut arr);
    Mat3::from_cols_array(&arr)
}

pub fn gaussian_3d(point: &Vec3, mean: &Vec3, covariance: &Mat3) -> f32 {
    let diff = *point - *mean;
    let cov_inv = covariance.inverse();

    let exponent = -0.5 * diff.dot(cov_inv * diff);

    let det = covariance.determinant();
    let normalizer = 1.0 / ((2.0 * std::f32::consts::PI).powi(3) * det).sqrt();

    normalizer * exponent.exp()
}

pub fn adaptive_density_control(positions: &[Vec3], threshold: f32) -> Vec<Vec3> {
    let mut result = Vec::new();

    for pos in positions {
        let mut keep = true;
        for existing in &result {
            let dist = pos.distance(*existing);
            if dist < threshold {
                keep = false;
                break;
            }
        }
        if keep {
            result.push(*pos);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_covariance_computation() {
        let scale = Vec3::new(1.0, 1.0, 1.0);
        let rotation = Quat::IDENTITY;
        let cov = compute_covariance_from_scale_rotation(&scale, &rotation);

        // Check Frobenius norm manually since Mat3 doesn't have length()
        let diff = cov - Mat3::IDENTITY;
        let norm_sq = diff.x_axis.length_squared()
            + diff.y_axis.length_squared()
            + diff.z_axis.length_squared();
        assert!(norm_sq < 1e-6);
    }

    #[test]
    fn test_gaussian_3d_at_mean() {
        let mean = Vec3::new(0.0, 0.0, 0.0);
        let cov = Mat3::IDENTITY;

        let value = gaussian_3d(&mean, &mean, &cov);

        assert!(value > 0.0);
    }

    #[test]
    fn test_adaptive_density() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];

        let filtered = adaptive_density_control(&positions, 0.5);

        assert_eq!(filtered.len(), 2);
    }
}
