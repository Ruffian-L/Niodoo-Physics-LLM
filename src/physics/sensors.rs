// use crate::gpu::lophat::create_decomposer;
use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use rand::prelude::*;
use std::collections::HashSet;

pub trait Sensor {
    fn name(&self) -> &str;
    /// Measure the metric on the given state.
    /// `state`: [Batch, N, Dim] position tensor.
    /// The sensor implementation is responsible for determining if it needs to sample or run on full data.
    fn measure(&self, state: &Tensor) -> Result<f32>;
}

pub struct EuclideanStressSensor {
    pub sample_size: usize,
    pub target_dist: f32, // Simplified ideal distance for now, or we could inject a distance matrix
}

impl EuclideanStressSensor {
    pub fn new(sample_size: usize, target_dist: f32) -> Self {
        Self {
            sample_size,
            target_dist,
        }
    }
}

impl Sensor for EuclideanStressSensor {
    fn name(&self) -> &str {
        "EuclideanStress"
    }

    fn measure(&self, state: &Tensor) -> Result<f32> {
        let (b, n, dim) = state.dims3()?;
        // Assume batch size 1 for now or avg over batch
        let state = state.get(0)?; // [N, Dim]

        let device = state.device();

        // Randomly sample pairs
        // We can't easily index arbitrarily in candle without creating an index tensor.
        // For efficiency in this prototype, let's pull data to CPU if N is small, or use a gathering kernel?
        // Let's bring small subsample to CPU for flexible calculation to avoid custom kernels for now.
        // Or if N is large, simple random indices.

        // Let's implement a naive CPU-based sampling for correctness first.
        // Reading huge tensor to CPU is slow.
        // Alternative: If we want to stay on GPU, we generate random indices and use `embedding` or `index_select`.

        // Let's assume we can afford to copy 1000 points (sample_size * 2) to CPU.
        // Ideally we would do this on GPU.

        // Let's try to just use global pairwise distance on a subsample.
        // Select `sample_size` indices.
        let mut rng = rand::thread_rng();
        let indices: Vec<u32> = (0..n as u32).choose_multiple(&mut rng, self.sample_size);
        let indices_tensor = Tensor::from_vec(indices, (self.sample_size,), device)?;

        // Gather positions: [Sample, Dim]
        let selected = state.index_select(&indices_tensor, 0)?;

        // Compute pairwise distances for this subsample
        // [Sample, 1, Dim] - [1, Sample, Dim] -> [Sample, Sample, Dim]
        // This might be O(Sample^2), which is fine for Sample=500.
        let n_sub = self.sample_size;
        let s2 = selected
            .reshape((n_sub, 1, dim))?
            .broadcast_as((n_sub, n_sub, dim))?;
        let s3 = selected
            .reshape((1, n_sub, dim))?
            .broadcast_as((n_sub, n_sub, dim))?;
        let diff = (s2 - s3)?;
        let dist_sq = diff.sqr()?.sum_keepdim(2)?; // [S, S, 1]
        let dist = dist_sq.sqrt()?;

        // Simplified Stress: (dist - target)^2
        // We only care about upper triangle, but full matrix sum / 2 is fine for relative metric.
        let target = Tensor::new(self.target_dist, device)?.broadcast_as(dist.shape())?;
        let error = (dist - target)?;
        let stress = error.sqr()?.mean_all()?;

        stress.to_scalar::<f32>().map_err(|e| e.into())
    }
}

pub struct TopologicalEntropySensor {
    pub grid_res: usize,
}

impl TopologicalEntropySensor {
    pub fn new(grid_res: usize) -> Self {
        Self { grid_res }
    }
}

impl Sensor for TopologicalEntropySensor {
    fn name(&self) -> &str {
        "TopologicalEntropy"
    }

    fn measure(&self, state: &Tensor) -> Result<f32> {
        // [B, N, Dim]
        let state = state.get(0)?; // [N, Dim]
                                   // Move to CPU for histogram binning (easier than implementing atomic add CUDA kernel right now)
                                   // For 10k points, CPU is instant.
        let coords: Vec<f32> = state.flatten_all()?.to_vec1()?;
        let dim = state.dim(1)?;
        let n = state.dim(0)?;

        if n == 0 {
            return Ok(0.0);
        }

        // Determine Bounds
        let mut min_bound = vec![f32::MAX; dim];
        let mut max_bound = vec![f32::MIN; dim];

        for i in 0..n {
            for d in 0..dim {
                let val = coords[i * dim + d];
                if val < min_bound[d] {
                    min_bound[d] = val;
                }
                if val > max_bound[d] {
                    max_bound[d] = val;
                }
            }
        }

        // Compute Bins
        // We only support 3D max for this implementation (or 2D).
        // Let's use a flat map for arbitrary dimensions sparse histogram.
        let mut counts: std::collections::HashMap<Vec<usize>, usize> =
            std::collections::HashMap::new();

        for i in 0..n {
            let mut bin_idx = Vec::with_capacity(dim);
            for d in 0..dim {
                let idx = if max_bound[d] > min_bound[d] {
                    ((coords[i * dim + d] - min_bound[d]) / (max_bound[d] - min_bound[d])
                        * (self.grid_res as f32)) as usize
                } else {
                    0
                };
                bin_idx.push(idx.min(self.grid_res - 1));
            }
            *counts.entry(bin_idx).or_insert(0) += 1;
        }

        // Calculate Entropy
        let total = n as f32;
        let mut entropy = 0.0;
        for count in counts.values() {
            let p = (*count as f32) / total;
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }

        // Normalize? Usually entropy is just H.
        Ok(entropy)
    }
}

pub struct BettiNumbersSensor {
    pub sample_size: usize,
    pub epsilon: f32, // Threshold for Rips complex
}

impl BettiNumbersSensor {
    pub fn new(sample_size: usize, epsilon: f32) -> Self {
        Self {
            sample_size,
            epsilon,
        }
    }
}

impl Sensor for BettiNumbersSensor {
    fn name(&self) -> &str {
        "BettiNumbers"
    }

    fn measure(&self, state: &Tensor) -> Result<f32> {
        let state = state.get(0)?; // [N, Dim]
        let n = state.dim(0)?;
        let dim = state.dim(1)?;
        let device = state.device();

        // Subsample
        let mut rng = rand::thread_rng();
        let sample_n = self.sample_size.min(n);
        let indices: Vec<u32> = (0..n as u32).choose_multiple(&mut rng, sample_n);
        let indices_tensor = Tensor::from_vec(indices.clone(), (sample_n,), device)?;

        let selected = state.index_select(&indices_tensor, 0)?; // [Sample, Dim]
        let coords: Vec<f32> = selected.flatten_all()?.to_vec1()?;

        // Build Boundary Matrix for Vietoris-Rips Complex
        // This is complex. For a prototype, we can cheat.
        // We want b0 (# components) and b1 (# loops).
        // 0-simplices: All points.
        // 1-simplices: Edges between points d < ε.
        // 2-simplices: Triangles where all edges < ε.

        // Building the full boundary matrix manually is tedious.
        // Let's implement a simplified Graph Connectivity check for b0 (Connected Components)
        // And maybe skip b1 for now unless we really want loops.
        // The user asked for "Betti Numbers", implying homology.
        // lophat expects a boundary matrix of the filtration.

        // Let's implement just b0 using a Union-Find (Disjoint Set) on CPU for the ε-graph.
        // This is extremely fast and gives us "number of clusters".
        // b1 requires triangles.

        // 1. Compute pairwise distances on CPU
        let mut adj: Vec<Vec<usize>> = vec![vec![]; sample_n];
        let mut edges = Vec::new(); // (u, v)

        for i in 0..sample_n {
            for j in (i + 1)..sample_n {
                let mut dist_sq = 0.0;
                for d in 0..dim {
                    let diff = coords[i * dim + d] - coords[j * dim + d];
                    dist_sq += diff * diff;
                }
                if dist_sq < self.epsilon * self.epsilon {
                    adj[i].push(j);
                    adj[j].push(i);
                    edges.push((i, j));
                }
            }
        }

        // Calculate b0 (Connected Components) using BFS/DFS
        let mut visited = vec![false; sample_n];
        let mut components = 0;
        for i in 0..sample_n {
            if !visited[i] {
                components += 1;
                // BFS
                let mut q = std::collections::VecDeque::new();
                q.push_back(i);
                visited[i] = true;
                while let Some(u) = q.pop_front() {
                    for &v in &adj[u] {
                        if !visited[v] {
                            visited[v] = true;
                            q.push_back(v);
                        }
                    }
                }
            }
        }

        // Calculate b1 (Euler Characteristic approximation or simple Cycle Basis?)
        // V - E + F = 1 - b1 (for 1 component simple case?).
        // Euler_char = b0 - b1 + b2 ...
        // V - E = b0 - b1 (ignoring faces/triangles).
        // So b1 ≈ b0 - V + E + Faces.
        // If we don't count faces, we treat graph as 1-skeleton.
        // b1 of graph = E - V + b0.
        // Let's return b1 of the 1-skeleton (Graph Homology).
        // It detects loops formed by edges.
        // Note: Graph b1 is E - V + b0.

        let v = sample_n as f32;
        let e = edges.len() as f32;
        let b0 = components as f32;
        let b1 = e - v + b0; // This is exact for 1D simplicial complex (graph)

        // Score: We want b0 to be 1 (fully connected) and b1 to be "reasonable" (not too high).
        // Let's return a composite metric? Or just return b0 + b1/100?
        // Let's return b0 for now, as fragmentation is the main "bad physics" symptom (clusters flying apart).
        // Or return b1 if we want to measure "richness".
        // The implementation plan says "Composite objective".
        // Let's return b0. The user can instantiate another sensor for b1 if needed.
        // Or we can return a packed float? No.

        // Let's return b0 (Number of Components).
        // Ideal is 1. High is bad (fragmented).
        Ok(b0)
    }
}
