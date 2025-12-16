// src/indexing/persistent_homology.rs
use anyhow::Result;
use nalgebra::Point3;

#[derive(Debug, Clone, Copy)]
pub enum PhStrategy {
    ExactBatch,
    StreamingApprox,
}

pub type PersistenceInterval = (f32, f32);

#[derive(Debug, Clone)]
pub struct PhConfig {
    pub hom_dims: Vec<usize>,
    pub strategy: PhStrategy,
    pub max_points: usize,
    pub connectivity_threshold: f32,
    pub max_dimension: usize,
    pub gpu_enabled: bool,        // New
    pub gpu_heap_capacity: usize, // New
}

#[derive(Debug, Clone)]
pub struct PhEngine {
    config: PhConfig,
}

impl PhEngine {
    pub fn new(config: PhConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &PhConfig {
        &self.config
    }

    /// Computes the Persistence Diagram using Vietoris-Rips filtration
    pub fn compute_pd<const D: usize>(&self, points: &[[f32; D]]) -> PersistenceDiagram {
        let dimension = self.config.max_dimension;

        if points.is_empty() {
            return PersistenceDiagram::new(dimension);
        }

        // Farthest Point Sampling (FPS)
        let max_points = self.config.max_points;
        let sampled_points = if points.len() > max_points {
            farthest_point_sampling(points, max_points)
        } else {
            points.to_vec()
        };

        let n = sampled_points.len();
        let mut edges = Vec::with_capacity(n * (n - 1) / 2);

        // User Requirement: "Remove hard edge filtering that breaks persistence guarantees"
        // We calculate all edges.
        for i in 0..n {
            for j in (i + 1)..n {
                let dist_sq = euclidean_distance_sq(&sampled_points[i], &sampled_points[j]);
                let dist = dist_sq.sqrt();
                edges.push((dist, i, j));
            }
        }
        edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut simplices: Vec<(f32, usize, Vec<usize>)> = Vec::new();

        // 0-simplices
        for i in 0..n {
            simplices.push((0.0, 0, vec![i]));
        }

        // 1-simplices (edges)
        for (dist, u, v) in &edges {
            simplices.push((*dist, 1, vec![*u, *v]));
        }

        // 2-simplices (triangles)
        // Optimization: Skip 2-skeleton entirely when n > 500 with clear warning, or if dimension < 2
        if dimension >= 2 {
            if n > 500 {
                eprintln!("WARN: Skipping 2-skeleton construction for TDA because n={} > 500. Results for H_2 will be empty.", n);
            } else {
                // Optimized Triangle Construction (O(m * d_max) instead of O(n^3))
                // Iterate over edges, intersect neighborhoods.
                // Precompute adjacency list for faster lookup
                let mut adj = vec![vec![]; n];
                for (dist, u, v) in &edges {
                    adj[*u].push((*v, *dist));
                    adj[*v].push((*u, *dist));
                }

                // For each edge (u, v), find common neighbors w
                // To avoid duplicates, we require u < v < w
                for (dist_uv, u, v) in &edges {
                    // We need w > v.
                    // Check neighbors of v.
                    for &(w, dist_vw) in &adj[*v] {
                        if w > *v {
                            // Check if (u, w) exists
                            // We can check adjacency of u.
                            // Since we want exact triangles, (u, w) must exist.
                            // In full VR, it always exists if we didn't filter edges.
                            // But we need the distance.
                            // Linear scan of u's neighbors? Or adjacency matrix?
                            // For n=500, adjacency matrix is fine (250k bools).
                            // Let's use the matrix from the original code but optimized loop.
                            // Actually, let's just use the adjacency list intersection.
                            if let Some((_, dist_uw)) =
                                adj[*u].iter().find(|(neighbor, _)| *neighbor == w)
                            {
                                let d = dist_uv.max(dist_vw).max(*dist_uw);
                                simplices.push((d, 2, vec![*u, *v, w]));
                            }
                        }
                    }
                }
            }
        }

        // Sort simplices by filtration value (diameter), then dimension
        simplices.sort_by(|a, b| {
            if (a.0 - b.0).abs() > 1e-6 {
                a.0.partial_cmp(&b.0).unwrap()
            } else {
                a.1.cmp(&b.1)
            }
        });

        // Map simplex indices to columns
        let mut boundary_matrix_indices: Vec<Vec<usize>> = Vec::with_capacity(simplices.len());

        // Map vertices to simplex index for boundary lookup
        // Using BTreeMap or HashMap? HashMap is O(1).
        let mut simplex_to_idx = std::collections::HashMap::new();

        for (idx, (_, dim, vertices)) in simplices.iter().enumerate() {
            let mut v_sorted = vertices.clone();
            v_sorted.sort();
            simplex_to_idx.insert(v_sorted.clone(), idx);

            let mut boundary = Vec::new();
            if *dim > 0 {
                for i in 0..vertices.len() {
                    let mut face = v_sorted.clone();
                    face.remove(i);
                    // face is already sorted
                    if let Some(&face_idx) = simplex_to_idx.get(&face) {
                        boundary.push(face_idx);
                    }
                }
            }
            boundary.sort_by(|a, b| b.cmp(a));
            boundary_matrix_indices.push(boundary);
        }

        // Run reduction
        use crate::gpu::lophat::create_decomposer;
        // PhConfig now needs gpu settings or we default them?
        // Let's add them to PhConfig to be clean.
        let mut decomposer = create_decomposer(
            boundary_matrix_indices,
            self.config.gpu_enabled,
            self.config.gpu_heap_capacity,
        );
        decomposer.reduce();

        // Extract persistence pairs
        let mut pd = PersistenceDiagram::new(dimension);
        let mut killed_rows = std::collections::HashSet::new();

        for col_idx in 0..simplices.len() {
            if let Some(row_idx) = decomposer.get_pivot(col_idx) {
                killed_rows.insert(row_idx);

                let birth = simplices[row_idx].0;
                let death = simplices[col_idx].0;
                let dim = simplices[row_idx].1;

                if (death - birth) > 1e-6 {
                    pd.add_pair_with_dim(birth, death, dim);
                }
            }
        }

        // Add infinite pairs (essential classes)
        // A simplex is an essential creator if:
        // 1. It is a creator (positive simplex): decomposer.get_pivot(i) is None (it didn't kill anything).
        // 2. It was never killed: !killed_rows.contains(&i).
        for i in 0..simplices.len() {
            if !killed_rows.contains(&i) {
                if decomposer.get_pivot(i).is_none() {
                    let birth = simplices[i].0;
                    let dim = simplices[i].1;
                    // Only report essential classes for requested dimensions
                    if dim <= dimension {
                        pd.add_pair_with_dim(birth, f32::INFINITY, dim);
                    }
                }
            }
        }

        pd
    }
}

fn farthest_point_sampling<const D: usize>(points: &[[f32; D]], k: usize) -> Vec<[f32; D]> {
    if points.is_empty() || k == 0 {
        return Vec::new();
    }
    let n = points.len();
    if k >= n {
        return points.to_vec();
    }

    let mut sampled_indices = Vec::with_capacity(k);
    let mut min_dists = vec![f32::INFINITY; n];

    // Start with random point (e.g., 0)
    let first_idx = 0;
    sampled_indices.push(first_idx);

    // Update distances from first point
    for i in 0..n {
        let d = euclidean_distance_sq(&points[i], &points[first_idx]);
        min_dists[i] = d;
    }

    for _ in 1..k {
        // Find point with max min_dist
        let mut best_idx = 0;
        let mut max_dist = -1.0;

        for i in 0..n {
            if min_dists[i] > max_dist {
                max_dist = min_dists[i];
                best_idx = i;
            }
        }

        sampled_indices.push(best_idx);

        // Update distances
        for i in 0..n {
            let d = euclidean_distance_sq(&points[i], &points[best_idx]);
            if d < min_dists[i] {
                min_dists[i] = d;
            }
        }
    }

    sampled_indices.into_iter().map(|i| points[i]).collect()
}

fn euclidean_distance_sq<const D: usize>(a: &[f32; D], b: &[f32; D]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

fn euclidean_distance<const D: usize>(a: &[f32; D], b: &[f32; D]) -> f32 {
    euclidean_distance_sq(a, b).sqrt()
}

#[derive(Debug, Clone)]
pub struct PersistenceDiagram {
    pub dimension: usize,
    pub pairs: Vec<(f32, f32)>,
    pub features_by_dim: Vec<Vec<(f32, f32)>>,
}

impl PersistenceDiagram {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            pairs: Vec::new(),
            features_by_dim: vec![Vec::new(); dimension + 1],
        }
    }

    pub fn add_pair(&mut self, birth: f32, death: f32) {
        self.add_pair_with_dim(birth, death, 0);
    }

    pub fn add_pair_with_dim(&mut self, birth: f32, death: f32, dim: usize) {
        self.pairs.push((birth, death));
        if dim < self.features_by_dim.len() {
            self.features_by_dim[dim].push((birth, death));
        } else {
            self.features_by_dim.resize(dim + 1, Vec::new());
            self.features_by_dim[dim].push((birth, death));
        }
    }

    pub fn persistence_values(&self) -> Vec<f32> {
        self.pairs
            .iter()
            .map(|(b, d)| if d.is_infinite() { 0.0 } else { d - b })
            .collect()
    }

    pub fn total_persistence(&self) -> f32 {
        crate::utils::fidelity::robust_sum(self.persistence_values().iter().copied())
    }

    pub fn filter_by_persistence(&self, threshold: f32) -> Self {
        let filtered_pairs: Vec<(f32, f32)> = self
            .pairs
            .iter()
            .filter(|(b, d)| (*d - *b) > threshold)
            .copied()
            .collect();

        let filtered_features_by_dim: Vec<Vec<(f32, f32)>> = self
            .features_by_dim
            .iter()
            .map(|features| {
                features
                    .iter()
                    .filter(|(b, d)| (*d - *b) > threshold)
                    .copied()
                    .collect()
            })
            .collect();

        Self {
            dimension: self.dimension,
            pairs: filtered_pairs,
            features_by_dim: filtered_features_by_dim,
        }
    }
}

pub fn compute_vietoris_rips(
    points: &[Point3<f32>],
    max_dimension: usize,
    _max_radius: f32,
) -> Result<Vec<PersistenceDiagram>> {
    let engine = PhEngine::new(PhConfig {
        hom_dims: (0..=max_dimension).collect(),
        strategy: PhStrategy::ExactBatch,
        max_points: 1000,
        connectivity_threshold: f32::INFINITY,
        max_dimension,
        gpu_enabled: true, // Default to true for now, or pass in?
        gpu_heap_capacity: 256 * 1024 * 1024, // Default
    });

    let raw_points: Vec<[f32; 3]> = points.iter().map(|p| [p.x, p.y, p.z]).collect();

    let pd = engine.compute_pd(&raw_points);

    Ok(vec![pd])
}

pub fn compute_alpha_complex(
    points: &[Point3<f32>],
    max_dimension: usize,
) -> Result<Vec<PersistenceDiagram>> {
    compute_vietoris_rips(points, max_dimension, f32::INFINITY)
}
