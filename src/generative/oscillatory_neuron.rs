//! OscillatoryNeuron: The fundamental unit of rhythmic intelligence
//!
//! Replaces static update rules with differential equation-driven dynamics
//! that enable temporally-based addressing and emergent computation.

use crate::generative::constants::*;
use std::f64::consts::PI;

/// Parameters governing oscillatory dynamics
/// These are the "control knobs" that will be modulated by topological feedback
#[derive(Debug, Clone)]
pub struct SimParams {
    /// Global oscillation frequency (Hz) - controls system's "clock speed"
    pub frequency: f64,

    /// Global inhibitory pulse amplitude - controls "selection pressure"  
    pub inhib_amplitude: f64,

    /// Activation time constant τₐ - controls "reaction speed"
    pub tau_activation: f64,

    /// Refractory time constant τᵣ - controls "recovery time"
    pub tau_refractory: f64,

    /// Simulation time step (seconds) - typically 10ms
    pub delta_t: f64,
}

impl Default for SimParams {
    fn default() -> Self {
        Self {
            frequency: 10.0,          // Alpha rhythm (8-12 Hz)
            inhib_amplitude: 1.0,     // Moderate inhibition
            tau_activation: 0.05,     // 50ms activation time constant
            tau_refractory: 0.1,      // 100ms refractory period
            delta_t: DEFAULT_DELTA_T, // 10ms simulation step
        }
    }
}

impl SimParams {
    /// Create parameters with biologically plausible constraints
    pub fn new(
        frequency: f64,
        inhib_amplitude: f64,
        tau_activation: f64,
        tau_refractory: f64,
    ) -> Self {
        Self {
            frequency: frequency.clamp(MIN_FREQUENCY, MAX_FREQUENCY),
            inhib_amplitude: inhib_amplitude.clamp(MIN_INHIB_AMPLITUDE, MAX_INHIB_AMPLITUDE),
            tau_activation: tau_activation.clamp(MIN_TAU, MAX_TAU),
            tau_refractory: tau_refractory.clamp(MIN_TAU, MAX_TAU),
            delta_t: DEFAULT_DELTA_T,
        }
    }

    /// Get the angular frequency ω = 2πf for the inhibitory pulse
    pub fn angular_frequency(&self) -> f64 {
        2.0 * PI * self.frequency
    }

    /// Validate parameters are within reasonable bounds
    pub fn is_valid(&self) -> bool {
        self.frequency > 0.0
            && self.inhib_amplitude >= 0.0
            && self.tau_activation > 0.0
            && self.tau_refractory > 0.0
            && self.delta_t > 0.0
    }
}

/// A single neuron with oscillatory dynamics
///
/// Behavior governed by coupled differential equations:
/// da/dt = (-a + sigmoid(net_input)) / τₐ
/// dr/dt = (-r + a) / τᵣ
///
/// Where:
/// - a = activation level
/// - r = refractory level  
/// - net_input = input_strength - refractory_level - inhibitory_pulse
#[derive(Debug, Clone)]
pub struct OscillatoryNeuron {
    /// Current activation level (0.0 to 1.0)
    pub activation: f64,

    /// Current refractory level (0.0 to 1.0)
    pub refractory_level: f64,
}

impl Default for OscillatoryNeuron {
    fn default() -> Self {
        Self {
            activation: 0.0,
            refractory_level: 0.0,
        }
    }
}

impl OscillatoryNeuron {
    /// Create a new neuron with optional initial state
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_state(activation: f64, refractory_level: f64) -> Self {
        Self {
            activation: activation.clamp(0.0, 1.0),
            refractory_level: refractory_level.clamp(0.0, 1.0),
        }
    }

    /// Update neuron state according to oscillatory dynamics
    ///
    /// # Arguments
    /// * `input_strength` - External stimulus (0.0 to 1.0)
    /// * `time_step` - Current simulation time
    /// * `params` - System parameters
    pub fn update(&mut self, input_strength: f64, time_step: f64, params: &SimParams) {
        // 1. Compute global inhibitory pulse
        // inhibitory_pulse = amplitude * sin(ω * t)
        let inhibitory_pulse =
            params.inhib_amplitude * (params.angular_frequency() * time_step).sin();

        // 2. Calculate net input
        // net_input = input - refractory - inhibition
        let net_input = input_strength - self.refractory_level - inhibitory_pulse;

        // 3. Apply sigmoid activation function
        let sigmoid_input = 1.0 / (1.0 + (-net_input).exp());

        // 4. Update activation using differential equation
        // da/dt = (-a + sigmoid(net_input)) / τₐ
        let activation_derivative = (-self.activation + sigmoid_input) / params.tau_activation;
        self.activation += activation_derivative * params.delta_t;

        // 5. Update refractory level using differential equation
        // dr/dt = (-r + a) / τᵣ
        let refractory_derivative =
            (-self.refractory_level + self.activation) / params.tau_refractory;
        self.refractory_level += refractory_derivative * params.delta_t;

        // 6. Clamp values to biologically plausible ranges
        self.activation = self.activation.max(0.0f64).min(1.0f64);
        self.refractory_level = self.refractory_level.clamp(0.0, 1.0);
    }

    /// Get the neuron's firing probability (based on activation)
    pub fn firing_probability(&self) -> f64 {
        self.activation
    }

    /// Check if neuron is in refractory period (unlikely to fire)
    pub fn is_refractory(&self, threshold: f64) -> bool {
        self.refractory_level > threshold
    }

    /// Reset neuron to resting state
    pub fn reset(&mut self) {
        self.activation = 0.0;
        self.refractory_level = 0.0;
    }

    /// Apply noise to neuron state (for exploration)
    pub fn apply_noise(&mut self, noise_level: f64) {
        let noise = (rand::random::<f64>() - 0.5) * 2.0 * noise_level;
        self.activation = (self.activation + noise).clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sim_params_default() {
        let params = SimParams::default();
        assert!(params.is_valid());
        assert_eq!(params.frequency, 10.0);
        assert_eq!(params.inhib_amplitude, 1.0);
    }

    #[test]
    fn test_sim_params_constraints() {
        // Test frequency constraints
        let params = SimParams::new(-1.0, 1.0, 0.1, 0.1);
        assert_eq!(params.frequency, MIN_FREQUENCY);

        let params = SimParams::new(1000.0, 1.0, 0.1, 0.1);
        assert_eq!(params.frequency, MAX_FREQUENCY);

        // Test inhibition constraints
        let params = SimParams::new(10.0, -5.0, 0.1, 0.1);
        assert_eq!(params.inhib_amplitude, MIN_INHIB_AMPLITUDE);

        let params = SimParams::new(10.0, 50.0, 0.1, 0.1);
        assert_eq!(params.inhib_amplitude, MAX_INHIB_AMPLITUDE);
    }

    #[test]
    fn test_oscillatory_neuron_creation() {
        let neuron = OscillatoryNeuron::new();
        assert_eq!(neuron.activation, 0.0);
        assert_eq!(neuron.refractory_level, 0.0);

        let neuron = OscillatoryNeuron::with_state(0.5, 0.3);
        assert_eq!(neuron.activation, 0.5);
        assert_eq!(neuron.refractory_level, 0.3);
    }

    #[test]
    fn test_neuron_basic_dynamics() {
        let mut neuron = OscillatoryNeuron::new();
        let params = SimParams::default();

        // Test with no input
        neuron.update(0.0, 0.0, &params);
        assert!(neuron.activation >= 0.0);

        // Test with strong input
        neuron.update(1.0, 0.0, &params);
        assert!(neuron.activation > 0.0);

        // Test refractory behavior
        assert!(neuron.refractory_level > 0.0);
    }

    #[test]
    fn test_inhibitory_pulse() {
        let params = SimParams::new(1.0, 1.0, 0.1, 0.1); // 1 Hz for easy testing

        // At t=0, sin(0) = 0, so no inhibition
        let pulse_at_0 = params.inhib_amplitude * (0.0f64).sin();
        assert!((pulse_at_0 - 0.0).abs() < 1e-10);

        // At t=0.25s, sin(2π*1*0.25) = sin(π/2) = 1, maximum inhibition
        let pulse_at_quarter =
            params.inhib_amplitude * (params.angular_frequency() * 0.25f64).sin();
        assert!((pulse_at_quarter - 1.0).abs() < 1e-10);

        // At t=0.5s, sin(π) = 0, no inhibition
        let pulse_at_half = params.inhib_amplitude * (params.angular_frequency() * 0.5f64).sin();
        assert!((pulse_at_half - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_neuron_temporal_dynamics() {
        let mut neuron = OscillatoryNeuron::new();
        let params = SimParams::new(10.0, 1.0, 0.05, 0.1); // 10 Hz oscillation

        let input_strength = 0.8;

        // Update through one complete cycle (0.1 seconds for 10 Hz)
        let steps_per_cycle = (0.1 / params.delta_t) as usize;
        let mut activations = Vec::new();

        for step in 0..steps_per_cycle {
            let time = step as f64 * params.delta_t;
            neuron.update(input_strength, time, &params);
            activations.push(neuron.activation);
        }

        // Should show oscillatory behavior
        let max_activation = activations.iter().fold(0.0f64, |a, &b| a.max(b));
        let min_activation = activations.iter().fold(1.0f64, |a, &b| a.min(b));

        assert!(max_activation > min_activation, "Should show oscillation");
        assert!(max_activation > 0.1, "Should reach significant activation");
    }

    #[test]
    fn test_frequency_effects() {
        let mut slow_neuron = OscillatoryNeuron::new();
        let mut fast_neuron = OscillatoryNeuron::new();

        let slow_params = SimParams::new(1.0, 1.0, 0.05, 0.1); // 1 Hz
        let fast_params = SimParams::new(50.0, 1.0, 0.05, 0.1); // 50 Hz

        let input = 0.5;

        // Run for same duration
        for step in 0..100 {
            let time = step as f64 * 0.01;
            slow_neuron.update(input, time, &slow_params);
            fast_neuron.update(input, time, &fast_params);
        }

        // Fast neuron should have different activation pattern
        assert!((slow_neuron.activation - fast_neuron.activation).abs() > 0.01);
    }
}
