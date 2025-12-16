//! Takens' Embedding: State-Space Reconstruction from Neural Rhythms
//!
//! "The magic theorem that lets us see the shape of time itself"
//!
//! Takens' Embedding Theorem: The topological structure of a high-dimensional
//! dynamical system's attractor can be faithfully reconstructed from a time-series
//! of a single scalar observable of that system.

use std::collections::VecDeque;

/// Parameters for Takens' embedding reconstruction
///
/// These parameters determine how we "unfold" 1D time series into
/// multi-dimensional state space that preserves topological structure.
#[derive(Debug, Clone)]
pub struct TakensEmbedding {
    /// Embedding dimension d - how many time delays to use
    /// Typically 3-7 for most systems
    pub dimension: usize,

    /// Time lag τ - how many steps to delay between coordinates
    /// Should capture the system's characteristic time scale
    pub time_lag: usize,

    /// Maximum number of delay vectors to keep in sliding window
    pub window_size: usize,

    /// History buffer for time series data
    history: VecDeque<f64>,
}

impl TakensEmbedding {
    /// Create embedding with biologically-inspired defaults
    pub fn new() -> Self {
        Self {
            dimension: 5,      // 5D reconstruction (good for neural dynamics)
            time_lag: 10,      // 100ms lag (10 steps * 10ms delta_t)
            window_size: 1000, // Keep 1000 most recent vectors
            history: VecDeque::new(),
        }
    }

    /// Create embedding with custom parameters
    pub fn with_params(dimension: usize, time_lag: usize, window_size: usize) -> Self {
        Self {
            dimension: dimension.max(2),       // Minimum 2D for meaningful topology
            time_lag: time_lag.max(1),         // Minimum 1 step lag
            window_size: window_size.max(100), // Minimum window
            history: VecDeque::new(),
        }
    }

    /// Add new observation to time series
    pub fn add_observation(&mut self, value: f64) {
        self.history.push_back(value);

        // Maintain history size (need enough points for embedding)
        let max_history = self.dimension * self.time_lag + self.window_size;
        while self.history.len() > max_history {
            self.history.pop_front();
        }
    }

    /// Reconstruct delay vectors from current time series
    ///
    /// Each delay vector v(t) = [s(t), s(t-τ), s(t-2τ), ..., s(t-(d-1)τ)]
    ///
    /// Returns: Vec of delay vectors in ℝ^d
    pub fn embed_time_series(&self) -> Vec<Vec<f64>> {
        let series: Vec<f64> = self.history.iter().copied().collect();

        if series.len() < self.dimension * self.time_lag {
            return Vec::new(); // Not enough data for embedding
        }

        let mut embedded = Vec::new();

        // Create delay vectors
        for i in (self.dimension * self.time_lag - 1)..series.len() {
            let mut vector = Vec::with_capacity(self.dimension);

            for j in 0..self.dimension {
                let index = i - j * self.time_lag;
                vector.push(series[index]);
            }

            embedded.push(vector);
        }

        // Keep only the most recent window_size vectors
        if embedded.len() > self.window_size {
            let _ = embedded.split_off(embedded.len() - self.window_size);
        }

        embedded
    }

    /// Get the current time series (for debugging)
    pub fn get_time_series(&self) -> Vec<f64> {
        self.history.iter().copied().collect()
    }

    /// Clear all history
    pub fn clear(&mut self) {
        self.history.clear();
    }

    /// Check if we have enough data for embedding
    pub fn has_sufficient_data(&self) -> bool {
        self.history.len() >= self.dimension * self.time_lag
    }

    /// Estimate optimal time lag using mutual information
    ///
    /// First minimum of mutual information is often a good choice for τ
    pub fn estimate_optimal_lag(&self, max_lag: usize) -> usize {
        let series: Vec<f64> = self.history.iter().copied().collect();
        if series.len() < 50 {
            return self.time_lag; // Not enough data for estimation
        }

        let mut best_lag = self.time_lag;
        let mut min_mi = f64::INFINITY;
        let mut found_minimum = false;

        for lag in 1..=max_lag.min(series.len() / 4) {
            let mi = self.compute_mutual_information(&series, lag);

            // Look for first local minimum
            if mi < min_mi {
                min_mi = mi;
                best_lag = lag;
                found_minimum = true;
            } else if found_minimum {
                // We found the minimum and now MI is increasing
                break;
            }
        }

        best_lag
    }

    /// Estimate optimal embedding dimension using false nearest neighbors
    ///
    /// When dimension is too low, neighbors in embedded space are actually
    /// far apart in the true attractor (false neighbors)
    pub fn estimate_optimal_dimension(&self, max_dim: usize) -> usize {
        let series: Vec<f64> = self.history.iter().copied().collect();
        if series.len() < 100 {
            return self.dimension; // Not enough data
        }

        let mut best_dim = self.dimension;

        for dim in 2..=max_dim {
            let fnn_fraction = self.compute_false_nearest_neighbors(&series, dim);

            // When false neighbors drop below threshold, dimension is sufficient
            if fnn_fraction < 0.01 {
                best_dim = dim;
                break;
            } else {
                best_dim = dim;
            }
        }

        best_dim
    }

    /// Compute mutual information between time series and its lagged version
    fn compute_mutual_information(&self, series: &[f64], lag: usize) -> f64 {
        if series.len() <= lag {
            return 0.0;
        }

        // Create histograms for joint distribution
        let bins = 10;
        let mut joint_hist = vec![vec![0; bins]; bins];
        let mut x_hist = vec![0; bins];
        let mut y_hist = vec![0; bins];

        // Find data ranges
        let x_vals: Vec<f64> = series[..series.len() - lag].to_vec();
        let y_vals: Vec<f64> = series[lag..].to_vec();

        let x_min = x_vals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let x_max = x_vals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let y_min = y_vals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let y_max = y_vals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let x_range = x_max - x_min;
        let y_range = y_max - y_min;

        if x_range == 0.0 || y_range == 0.0 {
            return 0.0;
        }

        // Fill histograms
        for (&x, &y) in x_vals.iter().zip(y_vals.iter()) {
            let x_bin = ((x - x_min) / x_range * (bins - 1) as f64) as usize;
            let y_bin = ((y - y_min) / y_range * (bins - 1) as f64) as usize;

            joint_hist[x_bin.min(bins - 1)][y_bin.min(bins - 1)] += 1;
            x_hist[x_bin.min(bins - 1)] += 1;
            y_hist[y_bin.min(bins - 1)] += 1;
        }

        // Compute mutual information
        let total_points = x_vals.len() as f64;
        let mut mi = 0.0;

        for i in 0..bins {
            for j in 0..bins {
                if joint_hist[i][j] > 0 && x_hist[i] > 0 && y_hist[j] > 0 {
                    let p_xy = joint_hist[i][j] as f64 / total_points;
                    let p_x = x_hist[i] as f64 / total_points;
                    let p_y = y_hist[j] as f64 / total_points;

                    mi += p_xy * (p_xy / (p_x * p_y)).ln();
                }
            }
        }

        mi
    }

    /// Compute fraction of false nearest neighbors for given dimension
    fn compute_false_nearest_neighbors(&self, series: &[f64], dimension: usize) -> f64 {
        if series.len() < dimension * 2 {
            return 1.0;
        }

        let embedded = self.embed_with_dimension(series, dimension);
        if embedded.len() < 2 {
            return 1.0;
        }

        let mut false_neighbors = 0;
        let mut total_neighbors = 0;

        // For each point, find its nearest neighbor
        for (i, point) in embedded.iter().enumerate() {
            if i == 0 {
                continue;
            }

            // Find nearest neighbor (excluding self)
            let mut nearest_dist = f64::INFINITY;
            let mut nearest_idx = 0;

            for (j, other) in embedded.iter().enumerate() {
                if i == j {
                    continue;
                }

                let dist = self.euclidean_distance(point, other);
                if dist < nearest_dist {
                    nearest_dist = dist;
                    nearest_idx = j;
                }
            }

            if nearest_idx > 0 && nearest_idx < embedded.len() - 1 {
                // Check if this is a false neighbor
                let current_next = if i + 1 < embedded.len() {
                    &embedded[i + 1]
                } else {
                    continue;
                };
                let neighbor_next = if nearest_idx + 1 < embedded.len() {
                    &embedded[nearest_idx + 1]
                } else {
                    continue;
                };

                let next_dist = self.euclidean_distance(current_next, neighbor_next);

                // False neighbor criterion
                if next_dist / nearest_dist > 15.0 {
                    false_neighbors += 1;
                }
                total_neighbors += 1;
            }
        }

        if total_neighbors == 0 {
            1.0
        } else {
            false_neighbors as f64 / total_neighbors as f64
        }
    }

    /// Embed time series with specific dimension
    fn embed_with_dimension(&self, series: &[f64], dimension: usize) -> Vec<Vec<f64>> {
        let mut embedded = Vec::new();

        for i in (dimension * self.time_lag - 1)..series.len() {
            let mut vector = Vec::with_capacity(dimension);

            for j in 0..dimension {
                let index = i - j * self.time_lag;
                vector.push(series[index]);
            }

            embedded.push(vector);
        }

        embedded
    }

    /// Compute Euclidean distance between two vectors
    fn euclidean_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Get embedding statistics
    pub fn get_statistics(&self) -> EmbeddingStats {
        EmbeddingStats {
            dimension: self.dimension,
            time_lag: self.time_lag,
            window_size: self.window_size,
            history_length: self.history.len(),
            sufficient_data: self.has_sufficient_data(),
            embedded_vectors: self.embed_time_series().len(),
        }
    }
}

/// Statistics about current embedding state
#[derive(Debug, Clone)]
pub struct EmbeddingStats {
    pub dimension: usize,
    pub time_lag: usize,
    pub window_size: usize,
    pub history_length: usize,
    pub sufficient_data: bool,
    pub embedded_vectors: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_takens_embedding_creation() {
        let embedding = TakensEmbedding::new();

        assert_eq!(embedding.dimension, 5);
        assert_eq!(embedding.time_lag, 10);
        assert_eq!(embedding.window_size, 1000);
        assert!(!embedding.has_sufficient_data());
    }

    #[test]
    fn test_takens_embedding_with_params() {
        let embedding = TakensEmbedding::with_params(3, 5, 500);

        assert_eq!(embedding.dimension, 3);
        assert_eq!(embedding.time_lag, 5);
        assert_eq!(embedding.window_size, 500);
    }

    #[test]
    fn test_observation_addition() {
        let mut embedding = TakensEmbedding::with_params(3, 2, 100);

        // Add insufficient data
        for i in 0..5 {
            embedding.add_observation(i as f64);
        }

        assert!(!embedding.has_sufficient_data());

        // Add sufficient data
        for i in 5..10 {
            embedding.add_observation(i as f64);
        }

        assert!(embedding.has_sufficient_data());
    }

    #[test]
    fn test_delay_vector_embedding() {
        let mut embedding = TakensEmbedding::with_params(3, 2, 100);

        // Create simple linear series: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        for i in 0..10 {
            embedding.add_observation(i as f64);
        }

        let embedded = embedding.embed_time_series();

        // Should have vectors like [9, 7, 5], [8, 6, 4], etc.
        assert!(!embedded.is_empty());

        // Check last vector (should be [9, 7, 5])
        if let Some(last) = embedded.last() {
            assert_eq!(last.len(), 3);
            assert!((last[0] - 9.0).abs() < 1e-10);
            assert!((last[1] - 7.0).abs() < 1e-10);
            assert!((last[2] - 5.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_periodic_signal_embedding() {
        let mut embedding = TakensEmbedding::with_params(3, 5, 200);

        // Create periodic signal (sin wave)
        for i in 0..200 {
            let value = (i as f64 * 0.1).sin();
            embedding.add_observation(value);
        }

        let embedded = embedding.embed_time_series();

        // Should successfully embed periodic signal
        assert!(!embedded.is_empty());
        assert!(embedded.len() <= embedding.window_size);

        // All vectors should have correct dimension
        for vector in &embedded {
            assert_eq!(vector.len(), 3);
        }
    }

    #[test]
    fn test_embedding_statistics() {
        let mut embedding = TakensEmbedding::new();

        let stats = embedding.get_statistics();
        assert_eq!(stats.history_length, 0);
        assert!(!stats.sufficient_data);

        // Add some data
        for i in 0..100 {
            embedding.add_observation(i as f64);
        }

        let stats = embedding.get_statistics();
        assert_eq!(stats.history_length, 100);
        assert!(stats.sufficient_data);
        assert!(stats.embedded_vectors > 0);
    }

    #[test]
    fn test_clear_functionality() {
        let mut embedding = TakensEmbedding::new();

        // Add data
        for i in 0..100 {
            embedding.add_observation(i as f64);
        }

        assert!(embedding.has_sufficient_data());

        // Clear and verify
        embedding.clear();
        assert!(!embedding.has_sufficient_data());
        assert_eq!(embedding.history.len(), 0);
    }
}
