//! Emergence Controller: Master Control Loop for Self-Regulating Emergence
//!
//! "The conductor that lets the orchestra regulate its own symphony"
//!
//! This is the master controller that integrates all Phase 3 components:
//! - Wundt Optimizer for intrinsic motivation
//! - Topological Homeostasis for complexity regulation
//! - Closed-loop feedback control for sustainable emergence
//! - Self-awareness and meta-cognitive monitoring

use crate::generative::{OscillatoryNetwork, SimParams};
use crate::perceptual::{TopologicalFeatures, TopologicalPerceiver};
use crate::regulation::{
    HomeostaticControl, IntrinsicMotivation, TopologicalHomeostasis, WundtOptimizer,
};
use rand;
use std::collections::VecDeque;

/// Query context for dynamic gain computation
/// Provides external hints about the current query/injection
#[derive(Debug, Clone, Default)]
pub struct QueryContext {
    /// Is this the first time seeing this topic/concept?
    pub is_new_concept: bool,
    /// Query length in tokens (longer = gentler injection)
    pub query_length: usize,
    /// Time since last query (seconds) - silence breeds exploration
    pub time_since_last: f64,
    /// Optional external novelty hint (from embedding distance)
    pub external_novelty_hint: Option<f64>,
}

/// Master controller for emergent self-regulation
///
/// This system orchestrates all control loops to maintain optimal emergence
/// while allowing the system to explore and learn autonomously.
pub struct EmergenceController {
    /// Topological perceiver for state monitoring
    perceiver: TopologicalPerceiver,

    /// Wundt optimizer for intrinsic motivation
    wundt_optimizer: WundtOptimizer,

    /// Topological homeostasis controller
    homeostasis: TopologicalHomeostasis,

    /// Control loop state
    control_state: ControlLoopState,

    /// Performance metrics
    performance_metrics: PerformanceMetrics,

    /// Meta-cognitive monitoring
    meta_monitor: MetaCognitiveMonitor,

    /// Control history
    control_history: VecDeque<ControlSnapshot>,
}

/// Control loop state
#[derive(Debug, Clone)]
pub struct ControlLoopState {
    /// Current control mode
    pub control_mode: ControlMode,

    /// Loop iteration count
    pub iteration: u64,

    /// System uptime
    pub uptime: f64,

    /// Last control timestamp
    pub last_control_time: f64,

    /// Control frequency (Hz)
    pub control_frequency: f64,

    /// System health status
    pub health_status: HealthStatus,
}

/// Control modes for different operational states
#[derive(Debug, Clone, PartialEq)]
pub enum ControlMode {
    /// Normal operation with balanced exploration/exploitation
    Normal,

    /// High exploration mode (seeking novelty)
    Exploration,

    /// High exploitation mode (consolidating knowledge)
    Exploitation,

    /// Recovery mode (returning to optimal state)
    Recovery,

    /// Learning mode (adapting control parameters)
    Learning,

    /// Safe mode (minimal control, high stability)
    Safe,
}

/// System health status
#[derive(Debug, Clone, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Warning,
    Critical,
    Recovering,
    Learning,
}

/// Performance metrics for the emergence controller
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Average complexity over time window
    pub avg_complexity: f64,

    /// Complexity stability (inverse of variance)
    pub complexity_stability: f64,

    /// Intrinsic motivation satisfaction
    pub motivation_satisfaction: f64,

    /// Homeostatic efficiency (low effort, high stability)
    pub homeostatic_efficiency: f64,

    /// Learning progress (improvement over time)
    pub learning_progress: f64,

    /// Emergence sustainability (can maintain optimal state)
    pub emergence_sustainability: f64,
}

/// Meta-cognitive monitoring
#[derive(Debug, Clone)]
pub struct MetaCognitiveMonitor {
    /// Self-awareness level
    pub self_awareness: f64,

    /// Predictive accuracy (how well can predict own state)
    pub predictive_accuracy: f64,

    /// Adaptation rate (how fast control parameters adapt)
    pub adaptation_rate: f64,

    /// Meta-learning progress
    pub meta_learning_progress: f64,

    /// Anomaly detection confidence
    pub anomaly_detection: f64,
}

/// Snapshot of control state for history tracking
#[derive(Debug, Clone)]
pub struct ControlSnapshot {
    pub timestamp: f64,
    pub complexity: f64,
    pub motivation: IntrinsicMotivation,
    pub homeostatic_control: HomeostaticControl,
    pub control_mode: ControlMode,
    pub health_status: HealthStatus,
}

impl EmergenceController {
    /// Create a new emergence controller
    pub fn new() -> Self {
        Self {
            perceiver: TopologicalPerceiver::new(),
            wundt_optimizer: WundtOptimizer::new(),
            homeostasis: TopologicalHomeostasis::new(),
            control_state: ControlLoopState::default(),
            performance_metrics: PerformanceMetrics::default(),
            meta_monitor: MetaCognitiveMonitor::default(),
            control_history: VecDeque::new(),
        }
    }

    /// Execute one control loop iteration
    pub fn control_loop_step(
        &mut self,
        network: &mut OscillatoryNetwork,
        timestamp: f64,
    ) -> ControlResult {
        // 1. Perceive current topological state
        let features = self.perceiver.perceive_state(network);

        // 2. Update control state
        self.update_control_state(timestamp);

        // 3. Update Wundt optimizer
        let motivation = self.wundt_optimizer.update(network, &features);

        // 4. Update homeostatic control
        let homeostatic_control = self.homeostasis.update(network, &features, timestamp);

        // 5. Determine control mode
        let control_mode = self.determine_control_mode(&motivation, &homeostatic_control);
        self.control_state.control_mode = control_mode.clone();

        // 6. Apply control actions
        self.apply_control_actions(network, &homeostatic_control, &control_mode);

        // 7. Update performance metrics
        self.update_performance_metrics(&features, &motivation, &homeostatic_control);

        // 8. Update meta-cognitive monitoring
        self.update_meta_monitoring(&features, &motivation);

        // 9. Store control snapshot
        self.store_control_snapshot(timestamp, &features, &motivation, &homeostatic_control);

        // 10. Update health status
        self.update_health_status();

        ControlResult {
            success: true,
            control_mode,
            motivation: motivation.clone(),
            homeostatic_control: homeostatic_control.clone(),
            performance_metrics: self.performance_metrics.clone(),
            health_status: self.control_state.health_status.clone(),
        }
    }

    /// Update control loop state
    fn update_control_state(&mut self, timestamp: f64) {
        self.control_state.iteration += 1;

        if self.control_state.last_control_time > 0.0 {
            let dt = timestamp - self.control_state.last_control_time;
            self.control_state.uptime += dt;
            self.control_state.control_frequency = 1.0 / dt;
        }

        self.control_state.last_control_time = timestamp;
    }

    /// Determine optimal control mode based on current state
    fn determine_control_mode(
        &self,
        motivation: &IntrinsicMotivation,
        homeostatic_control: &HomeostaticControl,
    ) -> ControlMode {
        // Check health status first
        if self.control_state.health_status == HealthStatus::Critical {
            return ControlMode::Recovery;
        }

        // Check if learning is needed
        if self.meta_monitor.adaptation_rate < 0.1 {
            return ControlMode::Learning;
        }

        // Check if homeostasis is struggling
        if homeostatic_control.control_magnitude > 0.7 {
            return ControlMode::Safe;
        }

        // Determine based on motivation
        match motivation.optimal_action {
            crate::regulation::wundt_optimizer::MotivationalAction::ExploreNovelty => {
                ControlMode::Exploration
            }
            crate::regulation::wundt_optimizer::MotivationalAction::ExploitKnown => {
                ControlMode::Exploitation
            }
            crate::regulation::wundt_optimizer::MotivationalAction::IncreaseComplexity => {
                if motivation.exploration_bias > 0.6 {
                    ControlMode::Exploration
                } else {
                    ControlMode::Normal
                }
            }
            crate::regulation::wundt_optimizer::MotivationalAction::DecreaseComplexity => {
                if motivation.motivation < 0.3 {
                    ControlMode::Recovery
                } else {
                    ControlMode::Normal
                }
            }
            crate::regulation::wundt_optimizer::MotivationalAction::MaintainOptimal => {
                ControlMode::Normal
            }
        }
    }

    /// Apply control actions based on control mode
    fn apply_control_actions(
        &self,
        network: &mut OscillatoryNetwork,
        _homeostatic_control: &HomeostaticControl,
        control_mode: &ControlMode,
    ) {
        // Apply base homeostatic control
        self.homeostasis.apply_control(network);

        // Apply mode-specific modifications
        match control_mode {
            ControlMode::Normal => {
                // Standard control, no modifications
            }
            ControlMode::Exploration => {
                // Increase exploration
                let exploration_params = SimParams::new(
                    network.params.frequency * (1.0 + rand::random::<f64>() * 0.2),
                    network.params.inhib_amplitude * (1.0 - rand::random::<f64>() * 0.3),
                    network.params.tau_activation * (1.0 + rand::random::<f64>() * 0.1),
                    network.params.tau_refractory * (1.0 + rand::random::<f64>() * 0.1),
                );
                network.update_params(exploration_params);
                network.apply_network_noise(0.05);
            }
            ControlMode::Exploitation => {
                // Decrease exploration, increase stability
                let exploitation_params = SimParams::new(
                    network.params.frequency * 0.95,
                    network.params.inhib_amplitude * 1.05,
                    network.params.tau_activation,
                    network.params.tau_refractory,
                );
                network.update_params(exploitation_params);
            }
            ControlMode::Recovery => {
                // Strong stabilization
                let recovery_params = SimParams::new(
                    10.0, // Return to safe frequency
                    2.0,  // Moderate inhibition
                    0.05, 0.1, // Standard time constants
                );
                network.update_params(recovery_params);
            }
            ControlMode::Learning => {
                // Adaptive parameters
                let learning_factor = 1.0 + self.meta_monitor.adaptation_rate * 0.5;
                let learning_params = SimParams::new(
                    network.params.frequency * learning_factor,
                    network.params.inhib_amplitude / learning_factor,
                    network.params.tau_activation * learning_factor,
                    network.params.tau_refractory * learning_factor,
                );
                network.update_params(learning_params);
            }
            ControlMode::Safe => {
                // Minimal control, high stability
                let safe_params = SimParams::new(
                    8.0, // Low, stable frequency
                    3.0, // Higher inhibition
                    0.1, 0.2, // Longer time constants
                );
                network.update_params(safe_params);
            }
        }
    }

    /// Update performance metrics
    fn update_performance_metrics(
        &mut self,
        features: &TopologicalFeatures,
        motivation: &IntrinsicMotivation,
        homeostatic_control: &HomeostaticControl,
    ) {
        // Update complexity metrics
        self.performance_metrics.avg_complexity = features.persistence_entropy;

        // Compute complexity stability from history
        let recent_complexities: Vec<f64> = self
            .control_history
            .iter()
            .rev()
            .take(10)
            .map(|s| s.complexity)
            .collect();

        if recent_complexities.len() > 1 {
            let mean_complexity =
                recent_complexities.iter().sum::<f64>() / recent_complexities.len() as f64;
            let variance = recent_complexities
                .iter()
                .map(|c| (c - mean_complexity).powi(2))
                .sum::<f64>()
                / recent_complexities.len() as f64;
            self.performance_metrics.complexity_stability = (1.0 - variance).max(0.0);
        }

        // Update motivation satisfaction
        self.performance_metrics.motivation_satisfaction = motivation.motivation;

        // Update homeostatic efficiency (inverse of control effort)
        self.performance_metrics.homeostatic_efficiency =
            1.0 - homeostatic_control.control_magnitude;

        // Update learning progress
        self.performance_metrics.learning_progress = self.meta_monitor.meta_learning_progress;

        // Update emergence sustainability
        self.performance_metrics.emergence_sustainability =
            self.performance_metrics.complexity_stability * 0.3
                + self.performance_metrics.motivation_satisfaction * 0.3
                + self.performance_metrics.homeostatic_efficiency * 0.2
                + self.performance_metrics.learning_progress * 0.2;
    }

    /// Update meta-cognitive monitoring
    fn update_meta_monitoring(
        &mut self,
        features: &TopologicalFeatures,
        _motivation: &IntrinsicMotivation,
    ) {
        // Update self-awareness based on prediction accuracy
        if self.control_history.len() > 5 {
            let predicted_complexity = self.predict_next_complexity();
            let actual_complexity = features.persistence_entropy;
            let prediction_error = (predicted_complexity - actual_complexity).abs();
            self.meta_monitor.predictive_accuracy = (1.0 - prediction_error).max(0.0);
            self.meta_monitor.self_awareness = self.meta_monitor.predictive_accuracy;
        }

        // Update adaptation rate
        let recent_controls: Vec<f64> = self
            .control_history
            .iter()
            .rev()
            .take(5)
            .map(|s| s.homeostatic_control.control_magnitude)
            .collect();

        if recent_controls.len() > 1 {
            let control_variance = recent_controls
                .iter()
                .map(|c| (c - recent_controls[0]).powi(2))
                .sum::<f64>()
                / recent_controls.len() as f64;
            self.meta_monitor.adaptation_rate = control_variance;
        }

        // Update meta-learning progress
        self.meta_monitor.meta_learning_progress = self.meta_monitor.self_awareness * 0.4
            + self.meta_monitor.predictive_accuracy * 0.3
            + self.meta_monitor.adaptation_rate * 0.3;

        // Update anomaly detection
        self.meta_monitor.anomaly_detection = self.detect_anomalies(features);
    }

    /// Predict next complexity level (simple linear prediction)
    fn predict_next_complexity(&self) -> f64 {
        if self.control_history.len() < 3 {
            return 0.5; // Default prediction
        }

        let recent_complexities: Vec<f64> = self
            .control_history
            .iter()
            .rev()
            .take(3)
            .map(|s| s.complexity)
            .collect();

        // Simple linear extrapolation
        let trend = recent_complexities[2] - recent_complexities[1];
        recent_complexities[0] + trend
    }

    /// Detect anomalies in current state
    fn detect_anomalies(&self, features: &TopologicalFeatures) -> f64 {
        if self.control_history.len() < 10 {
            return 0.0; // Not enough data
        }

        let recent_complexities: Vec<f64> = self
            .control_history
            .iter()
            .rev()
            .take(10)
            .map(|s| s.complexity)
            .collect();

        let mean_complexity =
            recent_complexities.iter().sum::<f64>() / recent_complexities.len() as f64;
        let std_dev = (recent_complexities
            .iter()
            .map(|c| (c - mean_complexity).powi(2))
            .sum::<f64>()
            / recent_complexities.len() as f64)
            .sqrt();

        // Z-score of current complexity
        let z_score = (features.persistence_entropy - mean_complexity) / (std_dev + 1e-6);

        // Convert to anomaly confidence (0-1)
        (z_score.abs() / 3.0).min(1.0)
    }

    /// Store control snapshot in history
    fn store_control_snapshot(
        &mut self,
        timestamp: f64,
        features: &TopologicalFeatures,
        motivation: &IntrinsicMotivation,
        homeostatic_control: &HomeostaticControl,
    ) {
        let snapshot = ControlSnapshot {
            timestamp,
            complexity: features.persistence_entropy,
            motivation: motivation.clone(),
            homeostatic_control: homeostatic_control.clone(),
            control_mode: self.control_state.control_mode.clone(),
            health_status: self.control_state.health_status.clone(),
        };

        self.control_history.push_back(snapshot);
        while self.control_history.len() > 100 {
            self.control_history.pop_front();
        }
    }

    /// Update system health status
    fn update_health_status(&mut self) {
        let health_score = self.performance_metrics.emergence_sustainability * 0.3
            + self.meta_monitor.self_awareness * 0.2
            + (1.0 - self.meta_monitor.anomaly_detection) * 0.2
            + self.performance_metrics.homeostatic_efficiency * 0.3;

        self.control_state.health_status = if health_score > 0.8 {
            HealthStatus::Healthy
        } else if health_score > 0.6 {
            HealthStatus::Warning
        } else if health_score > 0.4 {
            HealthStatus::Learning
        } else if health_score > 0.2 {
            HealthStatus::Recovering
        } else {
            HealthStatus::Critical
        };
    }

    /// Get current control state
    pub fn get_control_state(&self) -> &ControlLoopState {
        &self.control_state
    }

    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> &PerformanceMetrics {
        &self.performance_metrics
    }

    /// Get meta-cognitive monitor
    pub fn get_meta_monitor(&self) -> &MetaCognitiveMonitor {
        &self.meta_monitor
    }

    /// Get control history
    pub fn get_control_history(&self) -> Vec<ControlSnapshot> {
        self.control_history.iter().cloned().collect()
    }

    /// Check if system is self-regulating successfully
    pub fn is_self_regulating(&self) -> bool {
        self.control_state.health_status == HealthStatus::Healthy
            && self.performance_metrics.emergence_sustainability > 0.7
            && self.meta_monitor.self_awareness > 0.6
            && self.control_state.control_mode == ControlMode::Normal
    }

    /// Reset controller
    pub fn reset(&mut self) {
        self.perceiver.clear();
        self.wundt_optimizer.reset();
        self.homeostasis.reset();
        self.control_state = ControlLoopState::default();
        self.performance_metrics = PerformanceMetrics::default();
        self.meta_monitor = MetaCognitiveMonitor::default();
        self.control_history.clear();
    }

    // ========================================================================
    // DYNAMIC GAIN COMPUTATION - THE AUTO-TUNING SYSTEM
    // ========================================================================

    /// Compute dynamic gain for shadow token injection based on system state
    ///
    /// This replaces the static gain knob in SplatEngine. The gain is computed
    /// per-query using:
    /// - Wundt optimizer arousal level (targeting 0.72 peak)
    /// - Topological perceiver novelty score (new concepts need louder injection)
    /// - Persistence sharpness (fuzzy concepts need gentler push)
    /// - Optional TensorHeart pulse (chaos factor)
    ///
    /// # Arguments
    /// * `ctx` - Optional query context with external hints
    /// * `pulse_chaos` - Optional chaos level from TensorHeart (default 0.3)
    ///
    /// # Returns
    /// * `f64` - Computed gain value, clamped to [0.5, 5.0]
    pub fn compute_dynamic_gain(
        &self,
        ctx: Option<&QueryContext>,
        pulse_chaos: Option<f64>,
    ) -> f64 {
        // 1. Get arousal state from Wundt optimizer (0.0 - 1.0)
        let arousal = self.wundt_optimizer.get_statistics().current_arousal;
        let target_arousal = 0.72; // Peak of Wundt curve for optimal injection

        // 2. Get novelty and sharpness from topological perceiver
        let novelty = self.perceiver.novelty_score();
        let sharpness = self.perceiver.persistence_sharpness();

        // 3. Get pulse chaos (from TensorHeart if wired, else use default/provided)
        let chaos = pulse_chaos.unwrap_or(0.3);

        // 4. Base gain (known good center from experiments)
        let base = 2.7;

        // 5. Arousal correction (P-controller targeting Wundt peak)
        // Positive error = under-aroused → need more gain
        // Negative error = over-aroused → need less gain
        let error = target_arousal - arousal;
        let correction = error * 4.8; // P-gain tuned for stability

        // 6. Novelty boost: brand-new concepts need louder injection
        // novelty 0.0 = familiar topic → no boost
        // novelty 1.0 = completely new → +1.9 gain
        let novelty_boost = novelty * 1.9;

        // 7. Sharpness damping: fuzzy concepts need gentler push
        // sharpness 1.0 = sharp definition → no damping
        // sharpness 0.0 = fuzzy/noisy → -0.8 gain
        let sharpness_damp = (1.0 - sharpness) * 0.8;

        // 8. Chaos contribution from TensorHeart
        // Adds small exploration factor based on internal chaos level
        let chaos_contrib = chaos * 0.3;

        // 9. Optional context modifiers
        let context_mod = if let Some(context) = ctx {
            let mut mod_val = 0.0;

            // New concepts get extra boost
            if context.is_new_concept {
                mod_val += 0.5;
            }

            // Long queries get gentler injection (more context = less override)
            if context.query_length > 50 {
                mod_val -= 0.3;
            }

            // Long silence breeds exploration
            if context.time_since_last > 30.0 {
                mod_val += 0.4;
            }

            // External novelty hint overrides internal if provided
            if let Some(ext_novelty) = context.external_novelty_hint {
                mod_val += (ext_novelty - novelty) * 0.5; // Blend with internal
            }

            mod_val
        } else {
            0.0
        };

        // 10. Compute final gain
        let raw_gain =
            base + correction + novelty_boost - sharpness_damp + chaos_contrib + context_mod;

        // Clamp to sane range
        // 0.5 = barely audible whisper
        // 5.0 = screaming override (use sparingly)
        raw_gain.clamp(0.5, 5.0)
    }

    /// Simplified gain computation without context
    /// Uses internal state only - good for most use cases
    pub fn compute_gain(&self) -> f64 {
        self.compute_dynamic_gain(None, None)
    }

    /// Get current arousal level from Wundt optimizer
    pub fn current_arousal(&self) -> f64 {
        self.wundt_optimizer.get_statistics().current_arousal
    }

    /// Get current novelty score from perceiver
    pub fn current_novelty(&self) -> f64 {
        self.perceiver.novelty_score()
    }

    /// Get current sharpness from perceiver  
    pub fn current_sharpness(&self) -> f64 {
        self.perceiver.persistence_sharpness()
    }

    /// Get debug info about gain computation
    pub fn gain_debug_info(&self, pulse_chaos: Option<f64>) -> GainDebugInfo {
        let arousal = self.wundt_optimizer.get_statistics().current_arousal;
        let novelty = self.perceiver.novelty_score();
        let sharpness = self.perceiver.persistence_sharpness();
        let chaos = pulse_chaos.unwrap_or(0.3);
        let computed_gain = self.compute_dynamic_gain(None, pulse_chaos);

        GainDebugInfo {
            arousal,
            target_arousal: 0.72,
            novelty,
            sharpness,
            chaos,
            computed_gain,
            control_mode: self.control_state.control_mode.clone(),
            health_status: self.control_state.health_status.clone(),
        }
    }
}

/// Debug info for gain computation
#[derive(Debug, Clone)]
pub struct GainDebugInfo {
    pub arousal: f64,
    pub target_arousal: f64,
    pub novelty: f64,
    pub sharpness: f64,
    pub chaos: f64,
    pub computed_gain: f64,
    pub control_mode: ControlMode,
    pub health_status: HealthStatus,
}

impl std::fmt::Display for GainDebugInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Gain: {:.2} | Arousal: {:.2}/{:.2} | Nov: {:.2} | Sharp: {:.2} | Chaos: {:.2} | Mode: {:?}",
            self.computed_gain,
            self.arousal,
            self.target_arousal,
            self.novelty,
            self.sharpness,
            self.chaos,
            self.control_mode
        )
    }
}

/// Result of a control loop step
#[derive(Debug, Clone)]
pub struct ControlResult {
    pub success: bool,
    pub control_mode: ControlMode,
    pub motivation: IntrinsicMotivation,
    pub homeostatic_control: HomeostaticControl,
    pub performance_metrics: PerformanceMetrics,
    pub health_status: HealthStatus,
}

impl Default for ControlLoopState {
    fn default() -> Self {
        Self {
            control_mode: ControlMode::Normal,
            iteration: 0,
            uptime: 0.0,
            last_control_time: 0.0,
            control_frequency: 10.0, // Default 10 Hz control loop
            health_status: HealthStatus::Learning,
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            avg_complexity: 0.5,
            complexity_stability: 0.5,
            motivation_satisfaction: 0.5,
            homeostatic_efficiency: 0.5,
            learning_progress: 0.0,
            emergence_sustainability: 0.5,
        }
    }
}

impl Default for MetaCognitiveMonitor {
    fn default() -> Self {
        Self {
            self_awareness: 0.0,
            predictive_accuracy: 0.0,
            adaptation_rate: 0.1,
            meta_learning_progress: 0.0,
            anomaly_detection: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generative::InputPattern;

    #[test]
    fn test_emergence_controller_creation() {
        let controller = EmergenceController::new();

        assert_eq!(controller.control_state.control_mode, ControlMode::Normal);
        assert_eq!(controller.control_state.iteration, 0);
        assert_eq!(
            controller.control_state.health_status,
            HealthStatus::Learning
        );
    }

    #[test]
    fn test_control_loop_step() {
        let mut controller = EmergenceController::new();
        let mut network = OscillatoryNetwork::with_size(10);

        network.apply_input_pattern(InputPattern::Uniform(0.5));
        network.run_steps(50);

        let result = controller.control_loop_step(&mut network, 1.0);

        assert!(result.success);
        assert!(result.performance_metrics.avg_complexity >= 0.0);
        assert!(result.motivation.motivation >= 0.0);
    }

    #[test]
    fn test_control_mode_determination() {
        let controller = EmergenceController::new();

        let motivation = IntrinsicMotivation {
            motivation: 0.8,
            arousal_deficit: 0.1,
            exploration_bias: 0.7,
            optimal_action: crate::regulation::wundt_optimizer::MotivationalAction::ExploreNovelty,
        };

        let homeostatic_control = HomeostaticControl {
            frequency_control: 0.1,
            inhibition_control: -0.1,
            noise_control: 0.2,
            size_control: 0.0,
            control_magnitude: 0.1,
        };

        let control_mode = controller.determine_control_mode(&motivation, &homeostatic_control);

        assert_eq!(control_mode, ControlMode::Exploration);
    }

    #[test]
    fn test_health_status_update() {
        let mut controller = EmergenceController::new();

        // Set up healthy metrics
        controller.performance_metrics.emergence_sustainability = 0.9;
        controller.meta_monitor.self_awareness = 0.8;
        controller.meta_monitor.anomaly_detection = 0.1;
        controller.performance_metrics.homeostatic_efficiency = 0.8;

        controller.update_health_status();

        assert_eq!(
            controller.control_state.health_status,
            HealthStatus::Healthy
        );
    }

    #[test]
    fn test_self_regulation_check() {
        let mut controller = EmergenceController::new();

        // Set up self-regulating state
        controller.control_state.health_status = HealthStatus::Healthy;
        controller.performance_metrics.emergence_sustainability = 0.8;
        controller.meta_monitor.self_awareness = 0.7;
        controller.control_state.control_mode = ControlMode::Normal;

        assert!(controller.is_self_regulating());

        // Set up non-self-regulating state
        controller.control_state.health_status = HealthStatus::Warning;
        assert!(!controller.is_self_regulating());
    }

    #[test]
    fn test_controller_reset() {
        let mut controller = EmergenceController::new();

        // Modify state
        controller.control_state.iteration = 100;
        controller.performance_metrics.avg_complexity = 0.8;
        controller.control_history.push_back(ControlSnapshot {
            timestamp: 1.0,
            complexity: 0.6,
            motivation: IntrinsicMotivation {
                motivation: 0.7,
                arousal_deficit: 0.1,
                exploration_bias: 0.5,
                optimal_action:
                    crate::regulation::wundt_optimizer::MotivationalAction::MaintainOptimal,
            },
            homeostatic_control: HomeostaticControl::default(),
            control_mode: ControlMode::Exploration,
            health_status: HealthStatus::Healthy,
        });

        // Reset
        controller.reset();

        // Verify reset
        assert_eq!(controller.control_state.iteration, 0);
        assert_eq!(controller.performance_metrics.avg_complexity, 0.5);
        assert!(controller.control_history.is_empty());
        assert_eq!(
            controller.control_state.health_status,
            HealthStatus::Learning
        );
    }
}
