use super::algebra::*;
use super::semantics::*;
use super::statistics::*;
use nalgebra::{DMatrix, DVector};

#[test]
fn test_sherman_morrison() {
    // A = I (identity)
    // u = [1, 0]
    // v = [1, 0]
    // A + uv^T = [[2, 0], [0, 1]]
    // Inverse should be [[0.5, 0], [0, 1]]

    let a_inv = DMatrix::from_diagonal_element(2, 2, 1.0);
    let u = DVector::from_vec(vec![1.0, 0.0]);
    let v = DVector::from_vec(vec![1.0, 0.0]);

    let result = update_inverse_rank1(&a_inv, &u, &v).unwrap();

    assert!((result[(0, 0)] - 0.5).abs() < 1e-5);
    assert!((result[(1, 1)] - 1.0).abs() < 1e-5);
}

#[test]
fn test_mahalanobis() {
    // Identity covariance (precision = identity)
    // x = [2, 0], mu = [0, 0]
    // dist = 2
    let prec = DMatrix::from_diagonal_element(2, 2, 1.0);
    let x = DVector::from_vec(vec![2.0, 0.0]);
    let mu = DVector::from_vec(vec![0.0, 0.0]);

    let dist = mahalanobis_dist(&x, &mu, &prec).unwrap();
    assert!((dist - 2.0).abs() < 1e-5);
}

#[test]
fn test_rocchio() {
    let q = DVector::from_vec(vec![1.0, 1.0]);
    let pos = vec![DVector::from_vec(vec![2.0, 2.0])];
    let neg = vec![DVector::from_vec(vec![0.0, 0.0])];

    // alpha=1, beta=0.5, gamma=0
    // q_new = [1,1] + 0.5*[2,2] = [2,2]
    let res = rocchio_update(&q, &pos, &neg, 1.0, 0.5, 0.0);
    assert!((res[(0, 0)] - 2.0).abs() < 1e-5);

    // alpha=1, beta=0, gamma=0.5
    // q_new = [1,1] - 0.5*[0,0] = [1,1]
    let res2 = rocchio_update(&q, &pos, &neg, 1.0, 0.0, 0.5);
    assert!((res2[(0, 0)] - 1.0).abs() < 1e-5);
}

#[test]
fn test_zlib_entropy() {
    let data = vec![0u8; 1000]; // Low entropy
    let entropy = compute_zlib_entropy(&data).unwrap();
    assert!(entropy < 0.1); // Should compress very well

    let data2: Vec<u8> = (0..255).cycle().take(1000).collect(); // Higher entropy
    let entropy2 = compute_zlib_entropy(&data2).unwrap();
    assert!(entropy2 > entropy);
}

#[test]
fn test_product_of_gaussians() {
    // N(0, 1) * N(2, 1) -> N(1, 0.5)
    // Precisions: 1 + 1 = 2 -> Variance = 0.5
    // Mean: (1*0 + 1*2) / 2 = 1

    let mu1 = DVector::from_vec(vec![0.0]);
    let prec1 = DMatrix::from_vec(1, 1, vec![1.0]);

    let mu2 = DVector::from_vec(vec![2.0]);
    let prec2 = DMatrix::from_vec(1, 1, vec![1.0]);

    let (mu_new, sigma_new, _) = fuse_gaussians(&mu1, &prec1, &mu2, &prec2).unwrap();

    assert!((mu_new[(0, 0)] - 1.0).abs() < 1e-5);
    assert!((sigma_new[(0, 0)] - 0.5).abs() < 1e-5);
}
