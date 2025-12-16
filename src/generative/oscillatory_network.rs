//! OscillatoryNetwork: A network of rhythmically intelligent neurons
//!
//! Implements temporally-based addressing where time becomes a computational
//! resource for information flow, selection, and segregation.

use crate::generative::{constants::DEFAULT_NETWORK_SIZE, OscillatoryNeuron, SimParams};
use std::collections::VecDeque;

/// A network of oscillatory neurons with global rhythmic coordination
///
/// The network creates "windows of opportunity" for different neurons
/// to fire based on the interplay of global inhibition and individual refractory states.
/// This converts parallel inputs into serial temporal sequences.
pub struct OscillatoryNetwork {
    /// Individual neurons in the network
    pub neurons: Vec<OscillatoryNeuron>,

    /// External stimulus inputs for each neuron
    pub inputs: Vec<f64>,

    /// System parameters controlling dynamics
    pub params: SimParams,

    /// Current simulation time
    pub current_time: f64,

    /// History of average activations for state reconstruction
    pub activation_history: VecDeque<f64>,

    /// Maximum history size for Takens' embedding
    pub max_history_size: usize,
}

impl OscillatoryNetwork {
    /// Create a new oscillatory network with default parameters
    pub fn new() -> Self {
        Self::with_size(DEFAULT_NETWORK_SIZE)
    }

    /// Create a network with specified number of neurons
    pub fn with_size(neuron_count: usize) -> Self {
        Self::with_params(neuron_count, SimParams::default())
    }

    /// Create a network with custom parameters
    pub fn with_params(neuron_count: usize, params: SimParams) -> Self {
        Self {
            neurons: (0..neuron_count)
                .map(|_| OscillatoryNeuron::new())
                .collect(),
            inputs: vec![0.0; neuron_count],
            params,
            current_time: 0.0,
            activation_history: VecDeque::new(),
            max_history_size: 1000,
        }
    }

    /// Get the number of neurons in the network
    pub fn size(&self) -> usize {
        self.neurons.len()
    }

    /// Set external input for a specific neuron
    pub fn set_input(&mut self, neuron_index: usize, input_strength: f64) {
        if neuron_index < self.inputs.len() {
            self.inputs[neuron_index] = input_strength.clamp(0.0, 1.0);
        }
    }

    /// Set inputs for all neurons at once
    pub fn set_inputs(&mut self, inputs: &[f64]) {
        let min_len = inputs.len().min(self.inputs.len());
        for (i, &input) in inputs.iter().take(min_len).enumerate() {
            self.inputs[i] = input.clamp(0.0, 1.0);
        }
    }

    /// Apply a pattern of inputs across the network
    pub fn apply_input_pattern(&mut self, pattern: InputPattern) {
        match pattern {
            InputPattern::Uniform(strength) => {
                self.inputs.fill(strength.clamp(0.0, 1.0));
            }
            InputPattern::Gradient(start, end) => {
                let n = self.inputs.len();
                for i in 0..n {
                    let t = i as f64 / (n - 1).max(1) as f64;
                    self.inputs[i] = (start + t * (end - start)).clamp(0.0, 1.0);
                }
            }
            InputPattern::Gaussian(center, width, strength) => {
                let n = self.inputs.len();
                for i in 0..n {
                    let t = i as f64 / (n - 1).max(1) as f64;
                    let distance = (t - center).abs();
                    let gaussian = strength * (-distance.powi(2) / (2.0 * width.powi(2))).exp();
                    self.inputs[i] = gaussian.clamp(0.0, 1.0);
                }
            }
            InputPattern::Custom(values) => {
                self.set_inputs(&values);
            }
        }
    }

    /// Advance the network by one time step
    ///
    /// This is the core computation where temporally-based addressing occurs.
    /// The global inhibitory pulse creates rhythmic "windows of opportunity"
    /// that different neurons can exploit based on their input strength and refractory state.
    pub fn step(&mut self) {
        // Update each neuron with its input and the global time
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            neuron.update(self.inputs[i], self.current_time, &self.params);
        }

        // Advance simulation time
        self.current_time += self.params.delta_t;

        // Record average activation for state reconstruction
        let avg_activation = self.get_average_activation();
        self.activation_history.push_back(avg_activation);

        // Maintain history size
        while self.activation_history.len() > self.max_history_size {
            self.activation_history.pop_front();
        }
    }

    /// Run multiple steps
    pub fn run_steps(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }

    /// Get current average activation across all neurons
    pub fn get_average_activation(&self) -> f64 {
        if self.neurons.is_empty() {
            return 0.0;
        }
        self.neurons.iter().map(|n| n.activation).sum::<f64>() / self.neurons.len() as f64
    }

    /// Get current average refractory level across all neurons
    pub fn get_average_refractory(&self) -> f64 {
        if self.neurons.is_empty() {
            return 0.0;
        }
        self.neurons.iter().map(|n| n.refractory_level).sum::<f64>() / self.neurons.len() as f64
    }

    /// Get the activation vector (current state snapshot)
    pub fn get_activation_vector(&self) -> Vec<f64> {
        self.neurons.iter().map(|n| n.activation).collect()
    }

    /// Get the refractory vector
    pub fn get_refractory_vector(&self) -> Vec<f64> {
        self.neurons.iter().map(|n| n.refractory_level).collect()
    }

    /// Get the full state vector (activation + refractory for each neuron)
    pub fn get_full_state(&self) -> Vec<f64> {
        let mut state = Vec::with_capacity(self.neurons.len() * 2);
        for neuron in &self.neurons {
            state.push(neuron.activation);
            state.push(neuron.refractory_level);
        }
        state
    }

    /// Get the activation history for Takens' embedding
    pub fn get_activation_history(&self) -> Vec<f64> {
        self.activation_history.iter().copied().collect()
    }

    /// Calculate network complexity based on activation variance
    pub fn get_network_complexity(&self) -> f64 {
        let activations = self.get_activation_vector();
        if activations.len() < 2 {
            return 0.0;
        }

        let mean = activations.iter().sum::<f64>() / activations.len() as f64;
        let variance =
            activations.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / activations.len() as f64;

        variance.sqrt()
    }

    /// Get the current inhibitory pulse value
    pub fn get_inhibitory_pulse(&self) -> f64 {
        self.params.inhib_amplitude * (self.params.angular_frequency() * self.current_time).sin()
    }

    /// Identify currently "active" neurons (above threshold)
    pub fn get_active_neurons(&self, threshold: f64) -> Vec<usize> {
        self.neurons
            .iter()
            .enumerate()
            .filter(|(_, n)| n.activation > threshold)
            .map(|(i, _)| i)
            .collect()
    }

    /// Get the firing pattern (which neurons are likely to fire)
    pub fn get_firing_pattern(&self, threshold: f64) -> Vec<bool> {
        self.neurons
            .iter()
            .map(|n| n.firing_probability() > threshold)
            .collect()
    }

    /// Apply noise to all neurons for exploration
    pub fn apply_network_noise(&mut self, noise_level: f64) {
        for neuron in &mut self.neurons {
            neuron.apply_noise(noise_level);
        }
    }

    /// Reset network to initial state
    pub fn reset(&mut self) {
        for neuron in &mut self.neurons {
            neuron.reset();
        }
        self.inputs.fill(0.0);
        self.current_time = 0.0;
        self.activation_history.clear();
    }

    /// Update network parameters
    pub fn update_params(&mut self, new_params: SimParams) {
        if new_params.is_valid() {
            self.params = new_params;
        }
    }

    /// Get current network statistics
    pub fn get_network_stats(&self) -> NetworkStats {
        NetworkStats {
            average_activation: self.get_average_activation(),
            average_refractory: self.get_average_refractory(),
            network_complexity: self.get_network_complexity(),
            active_neuron_count: self.get_active_neurons(0.5).len(),
            inhibitory_pulse: self.get_inhibitory_pulse(),
            current_frequency: self.params.frequency,
            current_inhibition: self.params.inhib_amplitude,
        }
    }
}

/// Different input patterns for testing network behavior
#[derive(Debug, Clone)]
pub enum InputPattern {
    /// Same input to all neurons
    Uniform(f64),
    /// Linear gradient from start to end
    Gradient(f64, f64),
    /// Gaussian bump centered at position (0.0 to 1.0)
    Gaussian(f64, f64, f64), // (center, width, strength)
    /// Custom input vector
    Custom(Vec<f64>),
}

/// Network statistics for monitoring and analysis
#[derive(Debug, Clone)]
pub struct NetworkStats {
    pub average_activation: f64,
    pub average_refractory: f64,
    pub network_complexity: f64,
    pub active_neuron_count: usize,
    pub inhibitory_pulse: f64,
    pub current_frequency: f64,
    pub current_inhibition: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let network = OscillatoryNetwork::new();
        assert_eq!(network.size(), DEFAULT_NETWORK_SIZE);
        assert_eq!(network.inputs.len(), DEFAULT_NETWORK_SIZE);
        assert_eq!(network.neurons.len(), DEFAULT_NETWORK_SIZE);
        assert!(network.params.is_valid());
    }

    #[test]
    fn test_network_with_custom_size() {
        let network = OscillatoryNetwork::with_size(50);
        assert_eq!(network.size(), 50);
        assert_eq!(network.inputs.len(), 50);
    }

    #[test]
    fn test_input_setting() {
        let mut network = OscillatoryNetwork::with_size(5);

        // Test single input
        network.set_input(0, 0.8);
        assert_eq!(network.inputs[0], 0.8);
        assert_eq!(network.inputs[1], 0.0);

        // Test multiple inputs
        network.set_inputs(&[0.2, 0.4, 0.6, 0.8, 1.0]);
        assert_eq!(network.inputs, vec![0.2, 0.4, 0.6, 0.8, 1.0]);

        // Test input clamping
        network.set_input(0, -1.0);
        assert_eq!(network.inputs[0], 0.0);

        network.set_input(0, 2.0);
        assert_eq!(network.inputs[0], 1.0);
    }

    #[test]
    fn test_input_patterns() {
        let mut network = OscillatoryNetwork::with_size(10);

        // Test uniform pattern
        network.apply_input_pattern(InputPattern::Uniform(0.7));
        assert!(network.inputs.iter().all(|&x| (x - 0.7).abs() < 1e-10));

        // Test gradient pattern
        network.apply_input_pattern(InputPattern::Gradient(0.0, 1.0));
        assert!((network.inputs[0] - 0.0).abs() < 1e-10);
        assert!((network.inputs[9] - 1.0).abs() < 1e-10);

        // Test gaussian pattern
        network.apply_input_pattern(InputPattern::Gaussian(0.5, 0.2, 1.0));
        let center_idx = network.inputs.len() / 2;
        let center_value = network.inputs[center_idx];
        assert!(center_value > 0.8); // Should be near peak
    }

    #[test]
    fn test_network_step() {
        let mut network = OscillatoryNetwork::with_size(5);
        network.apply_input_pattern(InputPattern::Uniform(0.5));

        let initial_time = network.current_time;
        assert_eq!(initial_time, 0.0);

        network.step();

        // Time should advance
        assert!((network.current_time - initial_time - network.params.delta_t).abs() < 1e-10);

        // Activations should change
        let avg_activation = network.get_average_activation();
        assert!(avg_activation > 0.0);

        // History should be recorded
        assert_eq!(network.activation_history.len(), 1);
    }

    #[test]
    fn test_temporal_dynamics() {
        let mut network = OscillatoryNetwork::with_size(10);
        network.apply_input_pattern(InputPattern::Uniform(0.8));

        // Run for multiple steps
        let steps = 100;
        network.run_steps(steps);

        // Should have history
        assert_eq!(network.activation_history.len(), steps);

        // Should show oscillatory behavior
        let activations: Vec<f64> = network.activation_history.iter().copied().collect();
        let max_act = activations.iter().fold(0.0f64, |a, &b| a.max(b));
        let min_act = activations.iter().fold(1.0f64, |a, &b| a.min(b));

        assert!(max_act > min_act, "Should show oscillation over time");
    }

    #[test]
    fn test_network_complexity() {
        let mut network = OscillatoryNetwork::with_size(10);

        // With uniform inputs, complexity should be low
        network.apply_input_pattern(InputPattern::Uniform(0.5));
        network.step();
        let uniform_complexity = network.get_network_complexity();

        // With varied inputs, complexity should be higher
        network.apply_input_pattern(InputPattern::Gradient(0.0, 1.0));
        network.step();
        let varied_complexity = network.get_network_complexity();

        assert!(varied_complexity >= uniform_complexity);
    }

    #[test]
    fn test_active_neurons() {
        let mut network = OscillatoryNetwork::with_size(10);
        network.apply_input_pattern(InputPattern::Gaussian(0.5, 0.1, 1.0));

        // Run a few steps to let activations develop
        network.run_steps(10);

        let active_neurons = network.get_active_neurons(0.3);
        assert!(
            !active_neurons.is_empty(),
            "Should have some active neurons"
        );

        let firing_pattern = network.get_firing_pattern(0.3);
        assert_eq!(firing_pattern.len(), 10);
        assert!(
            firing_pattern.iter().any(|&x| x),
            "Should have some firing neurons"
        );
    }

    #[test]
    fn test_network_stats() {
        let mut network = OscillatoryNetwork::with_size(5);
        network.apply_input_pattern(InputPattern::Uniform(0.6));
        network.run_steps(5);

        let stats = network.get_network_stats();
        assert!(stats.average_activation > 0.0);
        assert!(stats.average_refractory >= 0.0);
        assert!(stats.network_complexity >= 0.0);
        assert_eq!(stats.current_frequency, network.params.frequency);
        assert_eq!(stats.current_inhibition, network.params.inhib_amplitude);
    }

    #[test]
    fn test_network_reset() {
        let mut network = OscillatoryNetwork::with_size(5);
        network.apply_input_pattern(InputPattern::Uniform(0.8));
        network.run_steps(10);

        // Verify network has changed
        assert!(network.current_time > 0.0);
        assert!(!network.activation_history.is_empty());
        assert!(network.get_average_activation() > 0.0);

        // Reset and verify
        network.reset();
        assert_eq!(network.current_time, 0.0);
        assert!(network.activation_history.is_empty());
        assert!(network.inputs.iter().all(|&x| x == 0.0));
        assert!(network.get_average_activation() == 0.0);
    }

    #[test]
    fn test_parameter_modulation() {
        let mut network = OscillatoryNetwork::new();

        let original_frequency = network.params.frequency;
        let new_params = SimParams::new(20.0, 2.0, 0.1, 0.2);

        network.update_params(new_params);

        assert_eq!(network.params.frequency, 20.0);
        assert_eq!(network.params.inhib_amplitude, 2.0);
        assert_ne!(network.params.frequency, original_frequency);
    }
}
