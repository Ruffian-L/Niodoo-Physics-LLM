//! Topological Homeostasis: Self-Regulation Through Shape-Based Control Laws
//!
//! "The system that maintains its own optimal complexity"
//!
//! This module implements control laws that use topological features as feedback
//! signals to maintain the system in its optimal complexity regime. The system
//! learns to regulate its own emergence through shape-based homeostasis.

use crate::generative::{OscillatoryNetwork, SimParams};
use crate::perceptual::{ComplexityTrend, TopologicalFeatures, TopologicalRegime};
use crate::regulation::wundt_optimizer::{IntrinsicMotivation, WundtOptimizer};
use std::collections::VecDeque;

/// Parameters for topological homeostasis control
#[derive(Debug, Clone)]
pub struct HomeostasisParams {
    /// Target complexity level (optimal topological entropy)
    pub target_complexity: f64,

    /// Complexity tolerance band
    pub complexity_tolerance: f64,

    /// Control gain for complexity regulation
    pub complexity_gain: f64,

    /// Control gain for regime stabilization
    pub regime_gain: f64,

    /// Time constant for control smoothing
    pub control_tau: f64,

    /// Maximum control action magnitude
    pub max_control_action: f64,
}

impl Default for HomeostasisParams {
    fn default() -> Self {
        Self {
            target_complexity: 0.5,    // Medium complexity is optimal
            complexity_tolerance: 0.2, // ±20% tolerance
            complexity_gain: 0.1,      // Gentle control
            regime_gain: 0.15,         // Stronger regime control
            control_tau: 0.3,          // 300ms smoothing
            max_control_action: 0.8,   // Max 80% parameter change
        }
    }
}

/// Homeostatic state of the system
#[derive(Debug, Clone)]
pub struct HomeostaticState {
    /// Current complexity level
    pub current_complexity: f64,

    /// Complexity error (target - actual)
    pub complexity_error: f64,

    /// Current topological regime
    pub current_regime: TopologicalRegime,

    /// Complexity trend
    pub complexity_trend: ComplexityTrend,

    /// Homeostatic stability (0.0 = unstable, 1.0 = stable)
    pub stability: f64,

    /// Control effort being applied
    pub control_effort: f64,

    /// Time since last regime change
    pub regime_stability_time: f64,
}

/// Control actions for homeostatic regulation
#[derive(Debug, Clone)]
pub struct HomeostaticControl {
    /// Frequency control action
    pub frequency_control: f64,

    /// Inhibition control action
    pub inhibition_control: f64,

    /// Noise control action
    pub noise_control: f64,

    /// Network size control action (if applicable)
    pub size_control: f64,

    /// Overall control magnitude
    pub control_magnitude: f64,
}

/// Topological homeostasis controller
///
/// This system monitors topological features and applies control laws to
/// maintain optimal complexity and regime stability.
pub struct TopologicalHomeostasis {
    /// Homeostasis parameters
    params: HomeostasisParams,

    /// Wundt optimizer for intrinsic motivation
    wundt_optimizer: WundtOptimizer,

    /// History of homeostatic states
    state_history: VecDeque<HomeostaticState>,

    /// Current homeostatic state
    current_state: HomeostaticState,

    /// Current control actions
    current_control: HomeostaticControl,

    /// Previous control actions (for smoothing)
    previous_control: HomeostaticControl,

    /// Maximum history size
    max_history: usize,

    /// Last update timestamp
    last_update_time: f64,
}

impl TopologicalHomeostasis {
    /// Create a new topological homeostasis controller
    pub fn new() -> Self {
        Self {
            params: HomeostasisParams::default(),
            wundt_optimizer: WundtOptimizer::new(),
            state_history: VecDeque::new(),
            current_state: HomeostaticState::default(),
            current_control: HomeostaticControl::default(),
            previous_control: HomeostaticControl::default(),
            max_history: 50,
            last_update_time: 0.0,
        }
    }

    /// Create controller with custom parameters
    pub fn with_params(params: HomeostasisParams) -> Self {
        Self {
            params,
            wundt_optimizer: WundtOptimizer::new(),
            state_history: VecDeque::new(),
            current_state: HomeostaticState::default(),
            current_control: HomeostaticControl::default(),
            previous_control: HomeostaticControl::default(),
            max_history: 50,
            last_update_time: 0.0,
        }
    }

    /// Update homeostatic control based on current system state
    pub fn update(
        &mut self,
        network: &OscillatoryNetwork,
        features: &TopologicalFeatures,
        timestamp: f64,
    ) -> HomeostaticControl {
        // 1. Update homeostatic state estimation
        self.update_state(network, features, timestamp);

        // 2. Update Wundt optimizer for intrinsic motivation
        let motivation = self.wundt_optimizer.update(network, features);

        // 3. Compute homeostatic control actions
        let control = self.compute_homeostatic_control(&motivation);

        // 4. Smooth control actions
        let smoothed_control = self.smooth_control(&control);

        // 5. Update current control
        self.previous_control = self.current_control.clone();
        self.current_control = smoothed_control.clone();

        // 6. Store state in history
        self.store_state();

        smoothed_control
    }

    /// Update homeostatic state estimation
    fn update_state(
        &mut self,
        _network: &OscillatoryNetwork,
        features: &TopologicalFeatures,
        timestamp: f64,
    ) {
        let current_complexity = features.persistence_entropy;
        let complexity_error = self.params.target_complexity - current_complexity;

        // Compute stability based on recent complexity variance
        let stability = self.compute_stability();

        // Compute control effort
        let control_effort = self.current_control.control_magnitude;

        // Update regime stability time
        let regime_stability_time = if features.persistence_entropy > 0.0 {
            timestamp - self.last_update_time
        } else {
            self.current_state.regime_stability_time
        };

        self.current_state = HomeostaticState {
            current_complexity,
            complexity_error,
            current_regime: TopologicalRegime::Simple, // Would be computed from perceiver
            complexity_trend: ComplexityTrend::Stable, // Would be computed from perceiver
            stability,
            control_effort,
            regime_stability_time,
        };

        self.last_update_time = timestamp;
    }

    /// Compute system stability from recent complexity history
    fn compute_stability(&self) -> f64 {
        if self.state_history.len() < 5 {
            return 0.5; // Unknown stability
        }

        let recent_complexities: Vec<f64> = self
            .state_history
            .iter()
            .rev()
            .take(5)
            .map(|s| s.current_complexity)
            .collect();

        let mean_complexity =
            recent_complexities.iter().sum::<f64>() / recent_complexities.len() as f64;
        let variance = recent_complexities
            .iter()
            .map(|c| (c - mean_complexity).powi(2))
            .sum::<f64>()
            / recent_complexities.len() as f64;

        // Low variance = high stability
        (1.0 - variance).clamp(0.0, 1.0)
    }

    /// Compute homeostatic control actions
    fn compute_homeostatic_control(&self, motivation: &IntrinsicMotivation) -> HomeostaticControl {
        let error = self.current_state.complexity_error;

        // 1. Complexity regulation (proportional control)
        let complexity_control = error * self.params.complexity_gain;

        // 2. Regime stabilization (if in undesirable regime)
        let regime_control = self.compute_regime_control();

        // 3. Intrinsic motivation modulation
        let motivation_control = self.compute_motivation_control(motivation);

        // 4. Combine control actions
        let frequency_control = (complexity_control
            + regime_control.frequency_control
            + motivation_control.frequency_control)
            .clamp(
                -self.params.max_control_action,
                self.params.max_control_action,
            );

        let inhibition_control =
            (regime_control.inhibition_control + motivation_control.inhibition_control).clamp(
                -self.params.max_control_action,
                self.params.max_control_action,
            );

        let noise_control = motivation_control
            .noise_control
            .clamp(0.0, self.params.max_control_action);

        let size_control = regime_control.size_control.clamp(
            -self.params.max_control_action,
            self.params.max_control_action,
        );

        let control_magnitude = (frequency_control.abs()
            + inhibition_control.abs()
            + noise_control
            + size_control.abs())
            / 4.0;

        HomeostaticControl {
            frequency_control,
            inhibition_control,
            noise_control,
            size_control,
            control_magnitude,
        }
    }

    /// Compute regime-specific control actions
    fn compute_regime_control(&self) -> HomeostaticControl {
        match self.current_state.current_regime {
            TopologicalRegime::Simple => {
                // Too simple - increase complexity
                HomeostaticControl {
                    frequency_control: 0.2,
                    inhibition_control: -0.1,
                    noise_control: 0.3,
                    size_control: 0.0,
                    control_magnitude: 0.15,
                }
            }
            TopologicalRegime::Complex => {
                // Optimal regime - minimal control
                HomeostaticControl {
                    frequency_control: 0.0,
                    inhibition_control: 0.0,
                    noise_control: 0.1,
                    size_control: 0.0,
                    control_magnitude: 0.025,
                }
            }
            TopologicalRegime::Chaotic => {
                // Too chaotic - decrease complexity
                HomeostaticControl {
                    frequency_control: -0.2,
                    inhibition_control: 0.2,
                    noise_control: 0.1,
                    size_control: 0.0,
                    control_magnitude: 0.125,
                }
            }
            TopologicalRegime::HyperChaotic => {
                // Way too chaotic - strong control
                HomeostaticControl {
                    frequency_control: -0.4,
                    inhibition_control: 0.4,
                    noise_control: 0.05,
                    size_control: -0.2, // Reduce network size
                    control_magnitude: 0.2625,
                }
            }
            TopologicalRegime::Unknown => {
                // Unknown regime - conservative control
                HomeostaticControl {
                    frequency_control: 0.0,
                    inhibition_control: 0.0,
                    noise_control: 0.2,
                    size_control: 0.0,
                    control_magnitude: 0.05,
                }
            }
        }
    }

    /// Compute motivation-based control actions
    fn compute_motivation_control(&self, motivation: &IntrinsicMotivation) -> HomeostaticControl {
        match motivation.optimal_action {
            crate::regulation::wundt_optimizer::MotivationalAction::IncreaseComplexity => {
                HomeostaticControl {
                    frequency_control: 0.1 * motivation.motivation,
                    inhibition_control: -0.1 * motivation.motivation,
                    noise_control: 0.2 * motivation.motivation,
                    size_control: 0.0,
                    control_magnitude: motivation.motivation * 0.1,
                }
            }
            crate::regulation::wundt_optimizer::MotivationalAction::DecreaseComplexity => {
                HomeostaticControl {
                    frequency_control: -0.1 * motivation.motivation,
                    inhibition_control: 0.1 * motivation.motivation,
                    noise_control: 0.05 * motivation.motivation,
                    size_control: 0.0,
                    control_magnitude: motivation.motivation * 0.0625,
                }
            }
            crate::regulation::wundt_optimizer::MotivationalAction::MaintainOptimal => {
                HomeostaticControl {
                    frequency_control: 0.0,
                    inhibition_control: 0.0,
                    noise_control: 0.1 * motivation.exploration_bias,
                    size_control: 0.0,
                    control_magnitude: motivation.exploration_bias * 0.025,
                }
            }
            crate::regulation::wundt_optimizer::MotivationalAction::ExploreNovelty => {
                HomeostaticControl {
                    frequency_control: (rand::random::<f64>() - 0.5) * 0.3 * motivation.motivation,
                    inhibition_control: (rand::random::<f64>() - 0.5) * 0.3 * motivation.motivation,
                    noise_control: 0.4 * motivation.motivation,
                    size_control: 0.0,
                    control_magnitude: motivation.motivation * 0.2,
                }
            }
            crate::regulation::wundt_optimizer::MotivationalAction::ExploitKnown => {
                HomeostaticControl {
                    frequency_control: -0.05,
                    inhibition_control: 0.05,
                    noise_control: 0.05,
                    size_control: 0.0,
                    control_magnitude: 0.0375,
                }
            }
        }
    }

    /// Smooth control actions using exponential filtering
    fn smooth_control(&self, control: &HomeostaticControl) -> HomeostaticControl {
        let alpha = 1.0 - (-0.01 / self.params.control_tau).exp(); // Discrete approximation

        HomeostaticControl {
            frequency_control: alpha * control.frequency_control
                + (1.0 - alpha) * self.previous_control.frequency_control,
            inhibition_control: alpha * control.inhibition_control
                + (1.0 - alpha) * self.previous_control.inhibition_control,
            noise_control: alpha * control.noise_control
                + (1.0 - alpha) * self.previous_control.noise_control,
            size_control: alpha * control.size_control
                + (1.0 - alpha) * self.previous_control.size_control,
            control_magnitude: alpha * control.control_magnitude
                + (1.0 - alpha) * self.previous_control.control_magnitude,
        }
    }

    /// Apply homeostatic control to network
    pub fn apply_control(&self, network: &mut OscillatoryNetwork) {
        let current_params = &network.params;

        // Apply frequency control
        let new_frequency = (current_params.frequency
            + self.current_control.frequency_control * 10.0) // Scale control
            .clamp(0.1, 100.0);

        // Apply inhibition control
        let new_inhibition = (current_params.inhib_amplitude
            + self.current_control.inhibition_control * 5.0)
            .clamp(0.0, 10.0);

        // Create new parameters
        let new_params = SimParams::new(
            new_frequency,
            new_inhibition,
            current_params.tau_activation,
            current_params.tau_refractory,
        );

        network.update_params(new_params);

        // Apply noise control
        if self.current_control.noise_control > 0.1 {
            let noise_strength = self.current_control.noise_control * 0.05;
            network.apply_network_noise(noise_strength);
        }

        // Size control would require network reconfiguration (advanced feature)
        // For now, we just log it
        if self.current_control.size_control.abs() > 0.01 {
            // Size control not implemented in this version
        }
    }

    /// Store current state in history
    fn store_state(&mut self) {
        self.state_history.push_back(self.current_state.clone());
        while self.state_history.len() > self.max_history {
            self.state_history.pop_front();
        }
    }

    /// Get current homeostatic state
    pub fn get_state(&self) -> &HomeostaticState {
        &self.current_state
    }

    /// Get current control actions
    pub fn get_control(&self) -> &HomeostaticControl {
        &self.current_control
    }

    /// Get Wundt optimizer reference
    pub fn get_wundt_optimizer(&self) -> &WundtOptimizer {
        &self.wundt_optimizer
    }

    /// Get state history
    pub fn get_state_history(&self) -> Vec<HomeostaticState> {
        self.state_history.iter().cloned().collect()
    }

    /// Check if system is in optimal regime
    pub fn is_optimal(&self) -> bool {
        self.current_state.current_regime == TopologicalRegime::Complex
            && self.current_state.complexity_error.abs() <= self.params.complexity_tolerance
            && self.current_state.stability > 0.7
    }

    /// Get homeostatic performance metrics
    pub fn get_performance_metrics(&self) -> HomeostaticMetrics {
        let recent_states: Vec<_> = self.state_history.iter().rev().take(10).collect();

        let avg_complexity = if recent_states.is_empty() {
            self.current_state.current_complexity
        } else {
            recent_states
                .iter()
                .map(|s| s.current_complexity)
                .sum::<f64>()
                / recent_states.len() as f64
        };

        let avg_stability = if recent_states.is_empty() {
            self.current_state.stability
        } else {
            recent_states.iter().map(|s| s.stability).sum::<f64>() / recent_states.len() as f64
        };

        let avg_control_effort = if recent_states.is_empty() {
            self.current_state.control_effort
        } else {
            recent_states.iter().map(|s| s.control_effort).sum::<f64>() / recent_states.len() as f64
        };

        HomeostaticMetrics {
            average_complexity: avg_complexity,
            average_stability: avg_stability,
            average_control_effort: avg_control_effort,
            target_achievement: (1.0 - self.current_state.complexity_error.abs()).max(0.0),
            regime_optimality: if self.current_state.current_regime == TopologicalRegime::Complex {
                1.0
            } else {
                0.0
            },
        }
    }

    /// Reset homeostasis controller
    pub fn reset(&mut self) {
        self.state_history.clear();
        self.current_state = HomeostaticState::default();
        self.current_control = HomeostaticControl::default();
        self.previous_control = HomeostaticControl::default();
        self.wundt_optimizer.reset();
        self.last_update_time = 0.0;
    }
}

impl Default for HomeostaticState {
    fn default() -> Self {
        Self {
            current_complexity: 0.5,
            complexity_error: 0.0,
            current_regime: TopologicalRegime::Unknown,
            complexity_trend: ComplexityTrend::InsufficientData,
            stability: 0.5,
            control_effort: 0.0,
            regime_stability_time: 0.0,
        }
    }
}

impl Default for HomeostaticControl {
    fn default() -> Self {
        Self {
            frequency_control: 0.0,
            inhibition_control: 0.0,
            noise_control: 0.1,
            size_control: 0.0,
            control_magnitude: 0.025,
        }
    }
}

/// Homeostatic performance metrics
#[derive(Debug, Clone)]
pub struct HomeostaticMetrics {
    pub average_complexity: f64,
    pub average_stability: f64,
    pub average_control_effort: f64,
    pub target_achievement: f64,
    pub regime_optimality: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generative::InputPattern;

    #[test]
    fn test_homeostasis_creation() {
        let mut homeostasis = TopologicalHomeostasis::new();
        assert_eq!(homeostasis.params.target_complexity, 0.5);
        assert_eq!(homeostasis.current_state.current_complexity, 0.5);
        assert_eq!(homeostasis.current_control.control_magnitude, 0.025);
        homeostasis.current_state.current_regime = TopologicalRegime::Simple;
    }

    #[test]
    fn test_homeostasis_with_params() {
        let params = HomeostasisParams {
            target_complexity: 0.7,
            complexity_tolerance: 0.3,
            complexity_gain: 0.2,
            regime_gain: 0.25,
            control_tau: 0.5,
            max_control_action: 0.9,
        };

        let homeostasis = TopologicalHomeostasis::with_params(params);

        assert_eq!(homeostasis.params.target_complexity, 0.7);
        assert_eq!(homeostasis.params.complexity_tolerance, 0.3);
    }

    #[test]
    fn test_state_update() {
        let mut homeostasis = TopologicalHomeostasis::new();
        let mut network = OscillatoryNetwork::with_size(10);
        let features = TopologicalFeatures {
            feature_vector: vec![0.5; 8],
            betti_numbers: crate::perceptual::topological_perceiver::BettiNumbers::default(),
            persistence_entropy: 0.6,
            max_persistence: crate::perceptual::topological_perceiver::PersistenceMeasures::default(
            ),
            timestamp: 1.0,
        };

        network.apply_input_pattern(InputPattern::Uniform(0.5));
        network.run_steps(50);

        homeostasis.update(&network, &features, 1.0);

        let state = homeostasis.get_state();
        assert_eq!(state.current_complexity, 0.6);
        assert!((state.complexity_error - -0.1).abs() < 1e-10); // 0.5 - 0.6
    }

    #[test]
    fn test_regime_control() {
        let mut homeostasis = TopologicalHomeostasis::new();

        // Test simple regime control
        homeostasis.current_state.current_regime = TopologicalRegime::Simple;
        let control = homeostasis.compute_regime_control();

        assert!(control.frequency_control > 0.0); // Should increase complexity
        assert!(control.noise_control > 0.1); // Should add noise
        assert!(control.control_magnitude > 0.0);
    }

    #[test]
    fn test_optimal_check() {
        let mut homeostasis = TopologicalHomeostasis::new();

        // Set up optimal state
        homeostasis.current_state.current_regime = TopologicalRegime::Complex;
        homeostasis.current_state.complexity_error = 0.1; // Within tolerance
        homeostasis.current_state.stability = 0.8;

        assert!(homeostasis.is_optimal());

        // Set up non-optimal state
        homeostasis.current_state.current_regime = TopologicalRegime::Simple;
        assert!(!homeostasis.is_optimal());
    }

    #[test]
    fn test_performance_metrics() {
        let homeostasis = TopologicalHomeostasis::new();
        let metrics = homeostasis.get_performance_metrics();

        assert!(metrics.average_complexity >= 0.0 && metrics.average_complexity <= 1.0);
        assert!(metrics.average_stability >= 0.0 && metrics.average_stability <= 1.0);
        assert!(metrics.target_achievement >= 0.0 && metrics.target_achievement <= 1.0);
    }

    #[test]
    fn test_control_application() {
        let mut homeostasis = TopologicalHomeostasis::new();
        let mut network = OscillatoryNetwork::with_size(10);

        // Set up control
        homeostasis.current_control.frequency_control = 0.5;
        homeostasis.current_control.inhibition_control = 0.2;

        let original_frequency = network.params.frequency;
        let original_inhibition = network.params.inhib_amplitude;

        homeostasis.apply_control(&mut network);

        assert!(network.params.inhib_amplitude != original_inhibition);
    }

    #[test]
    fn test_homeostasis_reset() {
        let mut homeostasis = TopologicalHomeostasis::new();

        // Modify state
        homeostasis.current_state.current_complexity = 0.8;
        homeostasis.current_control.control_magnitude = 0.5;
        homeostasis
            .state_history
            .push_back(HomeostaticState::default());

        // Reset
        homeostasis.reset();

        // Verify reset
        assert_eq!(homeostasis.current_state.current_complexity, 0.5);
        assert_eq!(homeostasis.current_control.control_magnitude, 0.025);
        assert!(homeostasis.state_history.is_empty());
    }
}
