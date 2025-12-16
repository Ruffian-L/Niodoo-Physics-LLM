//! TopologicalPerceiver: Converting Neural Rhythms to Shape
//!
//! "The system that feels the topology of its own thoughts"
//!
//! This module bridges the OscillatoryNeuron engine with Topological Data Analysis,
//! allowing the system to perceive the "shape" of its own cognitive dynamics.

use crate::generative::OscillatoryNetwork;
use crate::indexing::vectorize::vector_persistence_block;
use crate::indexing::{PersistenceDiagram, PhConfig, PhEngine, PhStrategy};
use crate::perceptual::TakensEmbedding;
use crate::tivm::VpbParams;
use std::collections::VecDeque;

/// A perceiver that converts neural dynamics into topological features
///
/// This is the "shape sensor" that allows the system to measure its own
/// emergent state and feed it back into the control loop.
pub struct TopologicalPerceiver {
    /// Takens' embedding for state-space reconstruction
    pub embedding: TakensEmbedding,

    /// Time series history for embedding
    time_series: VecDeque<f64>,

    /// TDA engine for computing persistence diagrams
    tda_engine: PhEngine,

    /// Parameters for vectorization of persistence diagrams
    vpb_params: VpbParams,

    /// History of topological features (for trend analysis)
    feature_history: VecDeque<TopologicalFeatures>,

    /// Maximum feature history size
    max_feature_history: usize,
}

/// Topological features extracted from neural dynamics
#[derive(Debug, Clone)]
pub struct TopologicalFeatures {
    /// 8-dimensional vector from persistence diagram
    pub feature_vector: Vec<f32>,

    /// Betti numbers (connected components, loops, voids)
    pub betti_numbers: BettiNumbers,

    /// Persistence entropy (measure of topological complexity)
    pub persistence_entropy: f64,

    /// Maximum persistence in each dimension
    pub max_persistence: PersistenceMeasures,

    /// Timestamp when features were computed
    pub timestamp: f64,
}

/// Betti numbers for different homology dimensions
#[derive(Debug, Clone, Default)]
pub struct BettiNumbers {
    /// β₀: Connected components
    pub b0: f32,
    /// β₁: Loops/tunnels  
    pub b1: f32,
    /// β₂: Voids/cavities
    pub b2: f32,
}

/// Maximum persistence measures by dimension
#[derive(Debug, Clone, Default)]
pub struct PersistenceMeasures {
    /// Max persistence for β₀ features
    pub max_p0: f32,
    /// Max persistence for β₁ features
    pub max_p1: f32,
    /// Max persistence for β₂ features
    pub max_p2: f32,
}

impl TopologicalPerceiver {
    /// Create a new topological perceiver with default parameters
    pub fn new() -> Self {
        Self {
            embedding: TakensEmbedding::new(),
            time_series: VecDeque::new(),
            tda_engine: PhEngine::new(PhConfig {
                max_dimension: 3,
                hom_dims: vec![0, 1, 2],
                strategy: PhStrategy::ExactBatch,
                max_points: 1000,
                connectivity_threshold: 5.0,
                gpu_enabled: false,
                gpu_heap_capacity: 0,
            }),
            vpb_params: VpbParams::default(),
            feature_history: VecDeque::new(),
            max_feature_history: 100,
        }
    }

    /// Create perceiver with custom parameters
    pub fn with_params(
        embedding_dim: usize,
        time_lag: usize,
        window_size: usize,
        feature_history_size: usize,
    ) -> Self {
        Self {
            embedding: TakensEmbedding::with_params(embedding_dim, time_lag, window_size),
            time_series: VecDeque::new(),
            tda_engine: PhEngine::new(PhConfig {
                max_dimension: 3,
                hom_dims: vec![0, 1, 2],
                strategy: PhStrategy::ExactBatch,
                max_points: 1000,
                connectivity_threshold: 5.0,
                gpu_enabled: false,
                gpu_heap_capacity: 0,
            }),
            vpb_params: VpbParams::default(),
            feature_history: VecDeque::new(),
            max_feature_history: feature_history_size,
        }
    }

    /// Perceive the current topological state of the neural network
    ///
    /// This is the core perception loop:
    /// 1. Extract scalar observable from network
    /// 2. Perform Takens' embedding to reconstruct attractor
    /// 3. Compute persistence diagram of embedded state space
    /// 4. Extract topological features
    pub fn perceive_state(&mut self, network: &OscillatoryNetwork) -> TopologicalFeatures {
        // 1. Extract scalar observable (average activation)
        let avg_activation = network.get_average_activation();
        self.time_series.push_back(avg_activation);

        // Maintain time series size
        let max_series_size =
            self.embedding.dimension * self.embedding.time_lag + self.embedding.window_size;
        while self.time_series.len() > max_series_size {
            self.time_series.pop_front();
        }

        // Add observation to embedding
        self.embedding.add_observation(avg_activation);

        // 2. Reconstruct state space via Takens' embedding
        let embedded_points = self.embedding.embed_time_series();

        // 3. Compute persistence diagram
        let persistence_diagram = if embedded_points.len() >= 3 {
            self.compute_persistence_diagram(&embedded_points)
        } else {
            PersistenceDiagram::new(2) // Default empty diagram
        };

        // 4. Extract topological features
        let features = self.extract_features(&persistence_diagram, network.current_time);

        // Store in history
        self.feature_history.push_back(features.clone());
        while self.feature_history.len() > self.max_feature_history {
            self.feature_history.pop_front();
        }

        features
    }

    /// Compute persistence diagram from embedded points
    fn compute_persistence_diagram(&self, embedded_points: &[Vec<f64>]) -> PersistenceDiagram {
        if embedded_points.is_empty() {
            return PersistenceDiagram::new(2);
        }

        // Convert embedded points to 3D points for TDA
        // We use the first 3 dimensions, or pad with zeros if fewer
        let points_3d: Vec<[f32; 3]> = embedded_points
            .iter()
            .map(|point| {
                let mut p = [0.0f32; 3];
                for (i, &coord) in point.iter().take(3).enumerate() {
                    p[i] = coord as f32;
                }
                p
            })
            .collect();

        // Use existing TDA engine
        self.tda_engine.compute_pd(&points_3d)
    }

    /// Extract topological features from persistence diagram
    fn extract_features(
        &self,
        diagram: &PersistenceDiagram,
        timestamp: f64,
    ) -> TopologicalFeatures {
        // 1. Vectorize persistence diagram (8-dimensional feature vector)
        let feature_vector = vector_persistence_block(diagram, &self.vpb_params);

        // 2. Compute Betti numbers
        let betti_numbers = self.compute_betti_numbers(diagram);

        // 3. Compute persistence entropy
        let persistence_entropy = self.compute_persistence_entropy(diagram);

        // 4. Find maximum persistence by dimension
        let max_persistence = self.compute_max_persistence(diagram);

        TopologicalFeatures {
            feature_vector,
            betti_numbers,
            persistence_entropy,
            max_persistence,
            timestamp,
        }
    }

    /// Compute Betti numbers from persistence diagram
    fn compute_betti_numbers(&self, diagram: &PersistenceDiagram) -> BettiNumbers {
        let mut b0 = 0.0f32;
        let mut b1 = 0.0f32;
        let mut b2 = 0.0f32;

        // For simplicity, treat all pairs as β₀ features in this implementation
        // In a full implementation, we'd need dimensional information
        for (birth, death) in &diagram.pairs {
            let persistence = death - birth;

            if persistence > 0.01 {
                b0 += 1.0;
            }
        }

        // Add some simple heuristics for higher dimensions
        if diagram.pairs.len() > 3 {
            b1 = (diagram.pairs.len() / 4) as f32; // Estimate loops
        }
        if diagram.pairs.len() > 6 {
            b2 = (diagram.pairs.len() / 8) as f32; // Estimate voids
        }

        BettiNumbers { b0, b1, b2 }
    }

    /// Compute persistence entropy (measure of topological complexity)
    fn compute_persistence_entropy(&self, diagram: &PersistenceDiagram) -> f64 {
        if diagram.pairs.is_empty() {
            return 0.0;
        }

        // Compute persistence values
        let persistences: Vec<f32> = diagram
            .pairs
            .iter()
            .map(|(birth, death)| death - birth)
            .filter(|&p| p > 0.001) // Filter very small persistences
            .collect();

        if persistences.is_empty() {
            return 0.0;
        }

        let total_persistence: f32 = persistences.iter().sum();
        let mut entropy = 0.0f64;

        for &persistence in &persistences {
            if persistence > 0.0 && total_persistence > 0.0 {
                let probability = persistence / total_persistence;
                entropy -= (probability as f64) * (probability as f64).ln();
            }
        }

        entropy
    }

    /// Compute maximum persistence by dimension
    fn compute_max_persistence(&self, diagram: &PersistenceDiagram) -> PersistenceMeasures {
        let mut max_p0 = 0.0f32;
        let mut max_p1 = 0.0f32;
        let mut max_p2 = 0.0f32;

        // For simplicity, treat all as β₀ in this implementation
        for (birth, death) in &diagram.pairs {
            let persistence = death - birth;
            max_p0 = max_p0.max(persistence);
        }

        // Add some heuristics for higher dimensions
        if diagram.pairs.len() > 2 {
            max_p1 = max_p0 * 0.8; // Estimate
        }
        if diagram.pairs.len() > 4 {
            max_p2 = max_p0 * 0.6; // Estimate
        }

        PersistenceMeasures {
            max_p0,
            max_p1,
            max_p2,
        }
    }

    /// Get recent trend in topological complexity
    pub fn get_complexity_trend(&self) -> ComplexityTrend {
        if self.feature_history.len() < 3 {
            return ComplexityTrend::InsufficientData;
        }

        let recent: Vec<f64> = self
            .feature_history
            .iter()
            .rev()
            .take(5)
            .map(|f| f.persistence_entropy)
            .collect();

        // Compute trend slope (simple linear regression)
        let n = recent.len() as f64;
        let sum_x: f64 = (0..recent.len()).map(|i| i as f64).sum();
        let sum_y: f64 = recent.iter().sum();
        let sum_xy: f64 = recent.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let sum_x2: f64 = (0..recent.len()).map(|i| (i as f64).powi(2)).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2));

        if slope > 0.01 {
            ComplexityTrend::Increasing
        } else if slope < -0.01 {
            ComplexityTrend::Decreasing
        } else {
            ComplexityTrend::Stable
        }
    }

    /// Get current topological regime
    pub fn get_regime(&self) -> TopologicalRegime {
        if let Some(latest) = self.feature_history.back() {
            if latest.persistence_entropy < 0.1 {
                TopologicalRegime::Simple
            } else if latest.persistence_entropy < 0.5 {
                TopologicalRegime::Complex
            } else if latest.persistence_entropy < 1.0 {
                TopologicalRegime::Chaotic
            } else {
                TopologicalRegime::HyperChaotic
            }
        } else {
            TopologicalRegime::Unknown
        }
    }

    /// Get feature history for analysis
    pub fn get_feature_history(&self) -> Vec<TopologicalFeatures> {
        self.feature_history.iter().cloned().collect()
    }

    /// Clear all history
    pub fn clear(&mut self) {
        self.time_series.clear();
        self.embedding.clear();
        self.feature_history.clear();
    }

    /// Get perceiver statistics
    pub fn get_statistics(&self) -> PerceiverStats {
        PerceiverStats {
            embedding_dimension: self.embedding.dimension,
            time_lag: self.embedding.time_lag,
            window_size: self.embedding.window_size,
            time_series_length: self.time_series.len(),
            feature_history_length: self.feature_history.len(),
            current_regime: self.get_regime(),
            complexity_trend: self.get_complexity_trend(),
        }
    }

    /// How novel is the current state compared to recent history?
    /// Returns 0.0-1.0 where 1.0 = completely novel
    /// Used by EmergenceController for dynamic gain computation
    pub fn novelty_score(&self) -> f64 {
        if self.feature_history.len() < 3 {
            return 0.5; // Default: assume moderate novelty when insufficient data
        }

        let latest = match self.feature_history.back() {
            Some(f) => f,
            None => return 0.5,
        };

        // Compute average entropy over recent history
        let history_slice: Vec<f64> = self
            .feature_history
            .iter()
            .rev()
            .skip(1) // Skip the latest (we're comparing against it)
            .take(5)
            .map(|f| f.persistence_entropy)
            .collect();

        if history_slice.is_empty() {
            return 0.5;
        }

        let recent_avg: f64 = history_slice.iter().sum::<f64>() / history_slice.len() as f64;

        // Also compute standard deviation for better novelty detection
        let variance: f64 = history_slice
            .iter()
            .map(|&x| (x - recent_avg).powi(2))
            .sum::<f64>()
            / history_slice.len() as f64;
        let std_dev = variance.sqrt();

        // How many standard deviations is current from recent average?
        let deviation = (latest.persistence_entropy - recent_avg).abs();
        let z_score = if std_dev > 0.001 {
            deviation / std_dev
        } else {
            deviation * 10.0 // High sensitivity when variance is low
        };

        // Convert z-score to 0-1 novelty (2 std devs = 1.0 novelty)
        (z_score / 2.0).min(1.0)
    }

    /// How sharp/defined are the persistence bars?
    /// High = clear topological structure, Low = fuzzy/noisy features
    /// Returns 0.0-1.0 where 1.0 = perfectly sharp definition
    /// Used by EmergenceController for dynamic gain computation
    pub fn persistence_sharpness(&self) -> f64 {
        let latest = match self.feature_history.back() {
            Some(f) => f,
            None => return 0.5, // Default when no data
        };

        let max_p = latest.max_persistence.max_p0 as f64;
        let entropy = latest.persistence_entropy;

        if entropy < 0.01 {
            // Very low entropy = either perfect structure or no data
            // If max_p is also low, it's no data; if high, it's sharp
            if max_p > 0.1 {
                1.0 // Sharp: high persistence, low entropy
            } else {
                0.3 // Unclear: both low
            }
        } else if max_p < 0.01 {
            // No significant features
            0.2
        } else {
            // Normal case: sharpness = ratio of max persistence to entropy
            // High max_p with low entropy = sharp
            // Low max_p with high entropy = fuzzy
            let raw_sharpness = max_p / entropy;
            (raw_sharpness / 2.0).min(1.0) // Scale so ratio of 2 = 1.0 sharpness
        }
    }

    /// Get current persistence entropy (for external access)
    pub fn current_entropy(&self) -> f64 {
        self.feature_history
            .back()
            .map(|f| f.persistence_entropy)
            .unwrap_or(0.5)
    }
}

/// Trend in topological complexity over time
#[derive(Debug, Clone, PartialEq)]
pub enum ComplexityTrend {
    Increasing,
    Decreasing,
    Stable,
    InsufficientData,
}

/// Current topological regime of the system
#[derive(Debug, Clone, PartialEq)]
pub enum TopologicalRegime {
    Simple,       // Low entropy, few features
    Complex,      // Moderate entropy, structured features
    Chaotic,      // High entropy, many noisy features
    HyperChaotic, // Very high entropy, overwhelming complexity
    Unknown,      // Cannot determine
}

/// Statistics about the perceiver state
#[derive(Debug, Clone)]
pub struct PerceiverStats {
    pub embedding_dimension: usize,
    pub time_lag: usize,
    pub window_size: usize,
    pub time_series_length: usize,
    pub feature_history_length: usize,
    pub current_regime: TopologicalRegime,
    pub complexity_trend: ComplexityTrend,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generative::{InputPattern, OscillatoryNetwork};

    #[test]
    fn test_topological_perceiver_creation() {
        let perceiver = TopologicalPerceiver::new();

        assert_eq!(perceiver.embedding.dimension, 5);
        assert_eq!(perceiver.embedding.time_lag, 10);
        assert_eq!(perceiver.max_feature_history, 100);
    }

    #[test]
    fn test_perceiver_with_params() {
        let perceiver = TopologicalPerceiver::with_params(3, 5, 200, 50);

        assert_eq!(perceiver.embedding.dimension, 3);
        assert_eq!(perceiver.embedding.time_lag, 5);
        assert_eq!(perceiver.embedding.window_size, 200);
        assert_eq!(perceiver.max_feature_history, 50);
    }

    #[test]
    fn test_basic_perception() {
        let mut perceiver = TopologicalPerceiver::new();
        let mut network = OscillatoryNetwork::with_size(10);

        // Apply simple input and run
        network.apply_input_pattern(InputPattern::Uniform(0.5));
        network.run_steps(50);

        // Perceive state
        let features = perceiver.perceive_state(&network);

        // Should have extracted features
        assert!(!features.feature_vector.is_empty());
        assert!(features.timestamp >= 0.0);
    }

    #[test]
    fn test_feature_history() {
        let mut perceiver = TopologicalPerceiver::new();
        let mut network = OscillatoryNetwork::with_size(5);

        network.apply_input_pattern(InputPattern::Uniform(0.6));

        // Multiple perceptions should build history
        for _ in 0..5 {
            network.run_steps(20);
            perceiver.perceive_state(&network);
        }

        let history = perceiver.get_feature_history();
        assert_eq!(history.len(), 5);

        // Timestamps should be increasing
        for i in 1..history.len() {
            assert!(history[i].timestamp > history[i - 1].timestamp);
        }
    }

    #[test]
    fn test_complexity_trend() {
        let mut perceiver = TopologicalPerceiver::new();

        // Insufficient data
        assert_eq!(
            perceiver.get_complexity_trend(),
            ComplexityTrend::InsufficientData
        );
    }

    #[test]
    fn test_topological_regime() {
        let mut perceiver = TopologicalPerceiver::new();

        // No data yet
        assert_eq!(perceiver.get_regime(), TopologicalRegime::Unknown);
    }

    #[test]
    fn test_perceiver_statistics() {
        let perceiver = TopologicalPerceiver::new();
        let stats = perceiver.get_statistics();

        assert_eq!(stats.embedding_dimension, 5);
        assert_eq!(stats.time_lag, 10);
        assert_eq!(stats.window_size, 1000);
        assert_eq!(stats.time_series_length, 0);
        assert_eq!(stats.feature_history_length, 0);
        assert_eq!(stats.current_regime, TopologicalRegime::Unknown);
    }

    #[test]
    fn test_clear_functionality() {
        let mut perceiver = TopologicalPerceiver::new();
        let mut network = OscillatoryNetwork::with_size(5);

        // Add some data
        network.apply_input_pattern(InputPattern::Uniform(0.5));
        for _ in 0..60 {
            network.run_steps(1);
            perceiver.perceive_state(&network);
        }

        // Should have data
        assert!(!perceiver.time_series.is_empty());
        assert!(perceiver.embedding.has_sufficient_data());

        // Clear and verify
        perceiver.clear();
        assert!(perceiver.time_series.is_empty());
        assert!(!perceiver.embedding.has_sufficient_data());
        assert!(perceiver.feature_history.is_empty());
    }
}
