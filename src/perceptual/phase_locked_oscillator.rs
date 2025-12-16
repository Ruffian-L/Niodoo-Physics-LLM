//! Phase-Locked Oscillator: Topology → Rhythm → Memory
//!
//! "Where Tokyo alleys learn to sing and cat memories become harmonic resonances"
//!
//! This is the revolutionary bridge between topological memory and oscillatory
//! intelligence. Persistence diagrams don't just get stored - they become
//! rhythmic patterns that the network can feel, remember, and resonate with.

use crate::generative::{OscillatoryNetwork, SimParams};
use crate::indexing::vectorize::vector_persistence_block;
use crate::indexing::{PersistenceDiagram, PhConfig, PhEngine, PhStrategy};
use crate::perceptual::TopologicalPerceiver;
use crate::tivm::VpbParams;
use std::collections::HashMap;
use std::f64::consts::PI;

/// The revolutionary system that converts topology into living rhythm
///
/// When a Tokyo alley splat hits this system:
/// - Linear voids create low-frequency inhibition waves
/// - Cat memory loops resonate at harmonic 3  
/// - Phase drift becomes the feeling of "wrongness"
/// - Harmonic convergence becomes déjà vu
pub struct TopologicalOscillator {
    /// The oscillatory neural network that thinks in cycles
    neuron_grid: OscillatoryNetwork,

    /// Topological perceiver for state reconstruction
    perceiver: TopologicalPerceiver,

    /// Phase-locking strength (how strongly topology affects rhythm)
    phase_lock: f64,

    /// Memory of past rhythmic signatures (for resonance detection)
    resonance_memory: HashMap<String, RhythmicSignature>,

    /// Current rhythmic signature of the system
    pub current_signature: RhythmicSignature,

    /// TDA engine for processing incoming splats
    tda_engine: PhEngine,

    /// Harmonic sensitivity (how responsive to specific frequencies)
    harmonic_sensitivity: f64,

    /// Resonance threshold for detecting "familiar" patterns
    resonance_threshold: f64,
}

/// A rhythmic signature that captures the "feel" of a topological pattern
///
/// This is what allows the system to "remember" how Tokyo at 2am feels
/// and recognize when a cat memory from 3 months ago resonates with it.
#[derive(Debug, Clone)]
pub struct RhythmicSignature {
    /// Dominant frequency of the oscillation (Hz)
    pub dominant_frequency: f64,

    /// Frequency spectrum (harmonic content)
    pub harmonics: Vec<f64>,

    /// Phase relationships between frequency components
    pub phase_pattern: Vec<f64>,

    /// Complexity measure (how "rich" the rhythm is)
    pub complexity: f64,

    /// Inhibition pattern (how selection pressure varies)
    pub inhibition_pattern: Vec<f64>,

    /// Timestamp when this signature was created
    pub timestamp: f64,

    /// Semantic label (if any)
    pub label: Option<String>,
}

/// Resonance memory that stores and retrieves rhythmic patterns
#[derive(Debug, Clone)]
pub struct ResonanceMemory {
    /// Storage of rhythmic signatures with semantic associations
    signatures: HashMap<String, RhythmicSignature>,

    /// Resonance cache for fast lookup
    resonance_cache: HashMap<String, f64>,
}

/// The feeling of recognition when patterns resonate
#[derive(Debug, Clone)]
pub struct ResonanceFeeling {
    /// How strong the resonance is (0.0 to 1.0)
    pub strength: f64,

    /// What memory is resonating
    pub memory_label: String,

    /// The harmonic that's causing the resonance
    pub resonant_harmonic: usize,

    /// Phase difference causing the "feeling"
    pub phase_drift: f64,

    /// Semantic interpretation of the resonance
    pub interpretation: String,
}

impl TopologicalOscillator {
    /// Create a new topological oscillator with default parameters
    pub fn new() -> Self {
        let neuron_grid = OscillatoryNetwork::with_size(256); // Larger grid for rich harmonics
        let perceiver = TopologicalPerceiver::with_params(5, 10, 500, 50);

        Self {
            neuron_grid,
            perceiver,
            phase_lock: 0.7, // Strong topology-rhythm coupling
            resonance_memory: HashMap::new(),
            current_signature: RhythmicSignature::default(),
            tda_engine: PhEngine::new(PhConfig {
                max_dimension: 3,
                hom_dims: vec![0, 1, 2],
                strategy: PhStrategy::ExactBatch,
                max_points: 1000,
                connectivity_threshold: 5.0,
                gpu_enabled: true,
                gpu_heap_capacity: 256 * 1024 * 1024,
            }),
            harmonic_sensitivity: 0.8, // Highly sensitive to harmonics
            resonance_threshold: 0.6,  // Threshold for feeling "familiar"
        }
    }

    /// Create oscillator with custom sensitivity parameters
    pub fn with_sensitivity(
        phase_lock: f64,
        harmonic_sensitivity: f64,
        resonance_threshold: f64,
    ) -> Self {
        let mut oscillator = Self::new();
        oscillator.phase_lock = phase_lock.clamp(0.0, 1.0);
        oscillator.harmonic_sensitivity = harmonic_sensitivity.clamp(0.0, 1.0);
        oscillator.resonance_threshold = resonance_threshold.clamp(0.0, 1.0);
        oscillator
    }

    /// Ingest a splat and convert its topology into rhythm
    ///
    /// This is where the magic happens:
    /// - Splat topology → persistence diagram
    /// - Persistence diagram → frequency modulation
    /// - Frequency modulation → rhythmic signature
    /// - Rhythmic signature → feeling of place
    pub fn ingest_splat(&mut self, splat_points: &[[f32; 3]]) -> RhythmicSignature {
        // 1. Compute persistence diagram from splat
        let persistence_diagram = self.tda_engine.compute_pd(splat_points);

        // 2. Convert topology to frequency modulation
        let frequency_modulation = self.topology_to_frequency(&persistence_diagram);

        // 3. Apply modulation to neuron grid
        self.apply_frequency_modulation(&frequency_modulation);

        // 3. Let the network settle into new rhythm
        self.neuron_grid.run_steps(200); // Increased from 50 to 200 steps

        // 5. Extract current rhythmic signature
        let signature = self.extract_rhythmic_signature();
        self.current_signature = signature.clone();

        signature
    }

    /// Convert persistence diagram to frequency modulation pattern
    fn topology_to_frequency(&self, diagram: &PersistenceDiagram) -> FrequencyModulation {
        let vpb = vector_persistence_block(diagram, &VpbParams::default());

        // Map topological features to frequency changes
        let base_frequency = 10.0; // Alpha rhythm baseline
        let mut frequency_shifts = Vec::new();

        for (i, &feature) in vpb.iter().enumerate() {
            // Different features affect different harmonics
            let harmonic_multiplier = (i + 1) as f64;
            let frequency_shift =
                base_frequency * harmonic_multiplier * feature as f64 * self.phase_lock;
            frequency_shifts.push(frequency_shift);
        }

        // Create inhibition pattern from topological complexity
        let inhibition_strength = vpb.iter().map(|&f| f as f64).sum::<f64>() / vpb.len() as f64;
        let inhibition_pattern = vec![inhibition_strength; self.neuron_grid.size()];

        FrequencyModulation {
            frequency_shifts,
            inhibition_pattern,
            base_frequency,
        }
    }

    /// Apply frequency modulation to the oscillatory network
    fn apply_frequency_modulation(&mut self, modulation: &FrequencyModulation) {
        // Update network parameters based on topology
        let new_frequency =
            modulation.base_frequency + modulation.frequency_shifts.first().unwrap_or(&0.0);

        let new_inhibition = modulation.inhibition_pattern.first().unwrap_or(&1.0);

        let new_params = SimParams::new(
            new_frequency.clamp(0.1, 100.0),
            new_inhibition.clamp(0.0, 10.0),
            0.05,
            0.1, // Keep tau constants stable
        );

        self.neuron_grid.update_params(new_params);

        // Apply spatial modulation across neuron grid
        for (i, inhibition) in modulation.inhibition_pattern.iter().enumerate() {
            if i < self.neuron_grid.inputs.len() {
                self.neuron_grid.set_input(i, *inhibition);
            }
        }
    }

    /// Extract the current rhythmic signature from the oscillating network
    fn extract_rhythmic_signature(&self) -> RhythmicSignature {
        // 1. Get dominant frequency from network oscillation
        let dominant_frequency = self.compute_dominant_frequency();

        // 2. Extract harmonic content
        let harmonics = self.extract_harmonics();

        // 3. Analyze phase relationships
        let phase_pattern = self.analyze_phase_pattern();

        // 4. Compute complexity
        let complexity = self.neuron_grid.get_network_complexity();

        // 5. Get inhibition pattern
        let inhibition_pattern = self.neuron_grid.inputs.clone();

        RhythmicSignature {
            dominant_frequency,
            harmonics,
            phase_pattern,
            complexity,
            inhibition_pattern,
            timestamp: self.neuron_grid.current_time,
            label: None,
        }
    }

    /// Compute dominant frequency from network oscillation
    fn compute_dominant_frequency(&self) -> f64 {
        // Use FFT on activation history to find dominant frequency
        let activation_history = self.neuron_grid.get_activation_history();

        if activation_history.len() < 10 {
            return self.neuron_grid.params.frequency; // Not enough data, return current frequency
        }

        // Simple frequency estimation using zero-crossings
        let mut zero_crossings = 0;
        for i in 1..activation_history.len() {
            let prev = activation_history[i - 1];
            let curr = activation_history[i];

            if (prev >= 0.0 && curr < 0.0) || (prev <= 0.0 && curr > 0.0) {
                zero_crossings += 1;
            }
        }

        let duration = activation_history.len() as f64 * self.neuron_grid.params.delta_t;
        if duration > 0.0 && zero_crossings > 0 {
            zero_crossings as f64 / (2.0 * duration)
        } else {
            self.neuron_grid.params.frequency // Fallback to current frequency
        }
    }

    /// Extract harmonic content from network oscillation
    fn extract_harmonics(&self) -> Vec<f64> {
        let activation_history = self.neuron_grid.get_activation_history();

        if activation_history.len() < 20 {
            return vec![self.neuron_grid.params.frequency];
        }

        // Simple harmonic analysis (in production, use proper FFT)
        let mut harmonics = Vec::new();
        let base_freq = self.neuron_grid.params.frequency;

        for harmonic in 1..=5 {
            harmonics.push(base_freq * harmonic as f64);
        }

        harmonics
    }

    /// Analyze phase relationships between network components
    fn analyze_phase_pattern(&self) -> Vec<f64> {
        // Get activation phases across the network
        let activations = self.neuron_grid.get_activation_vector();

        // Simple phase analysis based on activation levels
        activations.iter().map(|&a| (a * 2.0 * PI).sin()).collect()
    }

    /// Store a rhythmic signature in resonance memory
    pub fn store_signature(&mut self, label: String, signature: RhythmicSignature) {
        let mut labeled_signature = signature.clone();
        labeled_signature.label = Some(label.clone());
        self.resonance_memory.insert(label, labeled_signature);
    }

    /// Check if current signature resonates with any stored memories
    pub fn detect_resonance(&self) -> Option<ResonanceFeeling> {
        let mut best_resonance = None;
        let mut best_strength = 0.0;

        for (_label, stored_signature) in &self.resonance_memory {
            if let Some(resonance) =
                self.compute_resonance(&self.current_signature, stored_signature)
            {
                if resonance.strength > best_strength
                    && resonance.strength > self.resonance_threshold
                {
                    best_strength = resonance.strength;
                    best_resonance = Some(resonance);
                }
            }
        }

        best_resonance
    }

    /// Compute resonance between two rhythmic signatures
    fn compute_resonance(
        &self,
        current: &RhythmicSignature,
        stored: &RhythmicSignature,
    ) -> Option<ResonanceFeeling> {
        // 1. Frequency resonance (harmonic alignment)
        let freq_diff = (current.dominant_frequency - stored.dominant_frequency).abs();
        let freq_resonance = (-freq_diff / self.harmonic_sensitivity).exp();

        // 2. Harmonic pattern matching
        let harmonic_resonance = self.compare_harmonics(&current.harmonics, &stored.harmonics);

        // 3. Phase pattern similarity
        let phase_similarity =
            self.compare_phase_patterns(&current.phase_pattern, &stored.phase_pattern);

        // 4. Overall resonance strength
        let overall_strength =
            freq_resonance * 0.4 + harmonic_resonance * 0.3 + phase_similarity * 0.3;

        if overall_strength > self.resonance_threshold {
            // Find resonant harmonic
            let resonant_harmonic =
                self.find_resonant_harmonic(&current.harmonics, &stored.harmonics);

            // Compute phase drift
            let phase_drift =
                self.compute_phase_drift(&current.phase_pattern, &stored.phase_pattern);

            // Generate interpretation
            let interpretation =
                self.generate_resonance_interpretation(overall_strength, phase_drift);

            Some(ResonanceFeeling {
                strength: overall_strength,
                memory_label: stored.label.clone().unwrap_or_default(),
                resonant_harmonic,
                phase_drift,
                interpretation,
            })
        } else {
            None
        }
    }

    /// Compare harmonic patterns between signatures
    fn compare_harmonics(&self, current: &[f64], stored: &[f64]) -> f64 {
        let min_len = current.len().min(stored.len());
        if min_len == 0 {
            return 0.0;
        }

        let mut similarity = 0.0;
        for i in 0..min_len {
            let diff = (current[i] - stored[i]).abs();
            similarity += (-diff / self.harmonic_sensitivity).exp();
        }

        similarity / min_len as f64
    }

    /// Compare phase patterns
    fn compare_phase_patterns(&self, current: &[f64], stored: &[f64]) -> f64 {
        let min_len = current.len().min(stored.len());
        if min_len == 0 {
            return 0.0;
        }

        let mut similarity = 0.0;
        for i in 0..min_len {
            let phase_diff = (current[i] - stored[i]).abs();
            similarity += (-phase_diff).exp();
        }

        similarity / min_len as f64
    }

    /// Find which harmonic is causing the strongest resonance
    fn find_resonant_harmonic(&self, current: &[f64], stored: &[f64]) -> usize {
        let min_len = current.len().min(stored.len());
        let mut best_harmonic = 0;
        let mut best_alignment = 0.0;

        for i in 0..min_len {
            let alignment = (-(current[i] - stored[i]).abs() / self.harmonic_sensitivity).exp();
            if alignment > best_alignment {
                best_alignment = alignment;
                best_harmonic = i;
            }
        }

        best_harmonic
    }

    /// Compute phase drift between patterns
    fn compute_phase_drift(&self, current: &[f64], stored: &[f64]) -> f64 {
        let min_len = current.len().min(stored.len());
        if min_len == 0 {
            return 0.0;
        }

        let mut total_drift = 0.0;
        for i in 0..min_len {
            total_drift += (current[i] - stored[i]).abs();
        }

        total_drift / min_len as f64
    }

    /// Generate semantic interpretation of resonance
    fn generate_resonance_interpretation(&self, strength: f64, phase_drift: f64) -> String {
        if strength > 0.9 {
            if phase_drift < 0.1 {
                "This feels exactly like...".to_string()
            } else if phase_drift < 0.5 {
                "This reminds me of...".to_string()
            } else {
                "This feels like... but something's wrong".to_string()
            }
        } else if strength > 0.7 {
            "There's something familiar here...".to_string()
        } else {
            "I sense a faint echo of...".to_string()
        }
    }

    /// Query the current feeling of the system
    pub fn query_feeling(&mut self) -> String {
        // Update current signature
        let features = self.perceiver.perceive_state(&self.neuron_grid);
        self.current_signature.timestamp = self.neuron_grid.current_time;
        self.current_signature.complexity = features.persistence_entropy;

        // Check for resonance
        if let Some(resonance) = self.detect_resonance() {
            format!(
                "{} {} (resonance: {:.2})",
                resonance.interpretation, resonance.memory_label, resonance.strength
            )
        } else {
            format!(
                "This feels like {:.1}Hz with complexity {:.2}",
                self.current_signature.dominant_frequency, self.current_signature.complexity
            )
        }
    }

    /// Get current rhythmic signature
    pub fn get_current_signature(&self) -> &RhythmicSignature {
        &self.current_signature
    }

    /// Get network access for external control
    pub fn network_mut(&mut self) -> &mut OscillatoryNetwork {
        &mut self.neuron_grid
    }

    /// Get network reference
    pub fn network(&self) -> &OscillatoryNetwork {
        &self.neuron_grid
    }

    /// Reset the oscillator
    pub fn reset(&mut self) {
        self.neuron_grid.reset();
        self.perceiver.clear();
        self.current_signature = RhythmicSignature::default();
    }
}

/// Frequency modulation pattern derived from topology
#[derive(Debug, Clone)]
struct FrequencyModulation {
    frequency_shifts: Vec<f64>,
    inhibition_pattern: Vec<f64>,
    base_frequency: f64,
}

impl Default for RhythmicSignature {
    fn default() -> Self {
        Self {
            dominant_frequency: 10.0,
            harmonics: vec![10.0],
            phase_pattern: vec![0.0],
            complexity: 0.0,
            inhibition_pattern: vec![1.0],
            timestamp: 0.0,
            label: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topological_oscillator_creation() {
        let oscillator = TopologicalOscillator::new();

        assert_eq!(oscillator.neuron_grid.size(), 256);
        assert_eq!(oscillator.phase_lock, 0.7);
        assert_eq!(oscillator.harmonic_sensitivity, 0.8);
        assert_eq!(oscillator.resonance_threshold, 0.6);
    }

    #[test]
    fn test_oscillator_with_sensitivity() {
        let oscillator = TopologicalOscillator::with_sensitivity(0.5, 0.9, 0.7);

        assert_eq!(oscillator.phase_lock, 0.5);
        assert_eq!(oscillator.harmonic_sensitivity, 0.9);
        assert_eq!(oscillator.resonance_threshold, 0.7);
    }

    #[test]
    fn test_splat_ingestion() {
        let mut oscillator = TopologicalOscillator::new();

        // Create simple test splat (cube vertices)
        let splat_points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];

        let signature = oscillator.ingest_splat(&splat_points);

        assert!(signature.dominant_frequency > 0.0);
        assert!(!signature.harmonics.is_empty());
        assert!(signature.timestamp > 0.0);
    }

    #[test]
    fn test_signature_storage_and_retrieval() {
        let mut oscillator = TopologicalOscillator::new();

        // Create and store a signature
        let signature = RhythmicSignature {
            dominant_frequency: 15.0,
            harmonics: vec![15.0, 30.0, 45.0],
            phase_pattern: vec![0.0, 1.0, 0.0],
            complexity: 0.5,
            inhibition_pattern: vec![1.0],
            timestamp: 1.0,
            label: Some("test_memory".to_string()),
        };

        oscillator.store_signature("test_memory".to_string(), signature);

        // Should have stored signature
        assert!(oscillator.resonance_memory.contains_key("test_memory"));
    }

    #[test]
    fn test_resonance_detection() {
        let mut oscillator = TopologicalOscillator::with_sensitivity(0.1, 0.1, 0.1); // Very sensitive

        // Store a signature
        let stored_signature = RhythmicSignature {
            dominant_frequency: 10.0,
            harmonics: vec![10.0, 20.0, 30.0],
            phase_pattern: vec![0.0, 0.5, 1.0],
            complexity: 0.3,
            inhibition_pattern: vec![1.0],
            timestamp: 1.0,
            label: Some("tokyo_alley".to_string()),
        };

        oscillator.store_signature("tokyo_alley".to_string(), stored_signature);

        // Set current signature to be very similar
        oscillator.current_signature = RhythmicSignature {
            dominant_frequency: 10.1, // Very close
            harmonics: vec![10.1, 20.1, 30.1],
            phase_pattern: vec![0.1, 0.6, 1.1],
            complexity: 0.31,
            inhibition_pattern: vec![1.0],
            timestamp: 2.0,
            label: None,
        };

        // Should detect resonance
        let resonance = oscillator.detect_resonance();
        assert!(resonance.is_some());

        let resonance = resonance.unwrap();
        assert_eq!(resonance.memory_label, "tokyo_alley");
        assert!(resonance.strength > 0.1);
    }

    #[test]
    fn test_feeling_query() {
        let mut oscillator = TopologicalOscillator::new();

        // Should return basic feeling without stored memories
        let feeling = oscillator.query_feeling();
        assert!(feeling.contains("Hz"));
        assert!(feeling.contains("complexity"));
    }

    #[test]
    fn test_rhythmic_signature_default() {
        let signature = RhythmicSignature::default();

        assert_eq!(signature.dominant_frequency, 10.0);
        assert_eq!(signature.harmonics, vec![10.0]);
        assert_eq!(signature.complexity, 0.0);
        assert!(signature.label.is_none());
    }

    #[test]
    fn test_oscillator_reset() {
        let mut oscillator = TopologicalOscillator::new();

        // Run network to change state
        oscillator.neuron_grid.run_steps(10);

        // Reset
        oscillator.reset();

        // Should be back to default
        assert_eq!(oscillator.neuron_grid.current_time, 0.0);
        assert_eq!(oscillator.current_signature.dominant_frequency, 10.0);
    }
}
