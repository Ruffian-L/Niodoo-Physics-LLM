/// Computes a robust sum of floating point numbers using f64 accumulation.
/// This is generally sufficient for most applications compared to naive f32 summation.
pub fn robust_sum<I>(iter: I) -> f32
where
    I: Iterator<Item = f32>,
{
    let mut sum: f64 = 0.0;
    for val in iter {
        sum += val as f64;
    }
    sum as f32
}

/// Computes a robust dot product of two vectors using f64 accumulation.
pub fn robust_dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sum: f64 = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        sum += (*x as f64) * (*y as f64);
    }
    sum as f32
}

/// Clamps a covariance matrix (represented as [f32; 9]) to ensure numerical stability.
/// This enforces symmetry and a minimal diagonal value.
pub fn clamp_covariance(cov: &mut [f32; 9]) {
    const EPSILON: f32 = 1e-6;

    // 1. Enforce Symmetry: (A + A^T) / 2
    // Indices: 0 1 2
    //          3 4 5
    //          6 7 8
    let c1 = (cov[1] + cov[3]) * 0.5;
    cov[1] = c1;
    cov[3] = c1;

    let c2 = (cov[2] + cov[6]) * 0.5;
    cov[2] = c2;
    cov[6] = c2;

    let c5 = (cov[5] + cov[7]) * 0.5;
    cov[5] = c5;
    cov[7] = c5;

    // 2. Clamp Diagonal (Eigenvalue approximation)
    cov[0] = cov[0].max(EPSILON);
    cov[4] = cov[4].max(EPSILON);
    cov[8] = cov[8].max(EPSILON);
}
