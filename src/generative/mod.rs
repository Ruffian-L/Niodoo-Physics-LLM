//! Generative Engine: Oscillatory Neural Network for Emergent Dynamics
//!
//! This module replaces static "magic numbers" with a living, breathing
//! dynamical system that generates behavior through temporal computation.

pub mod oscillatory_network;
pub mod oscillatory_neuron;
pub mod simulation_controller;

pub use oscillatory_network::{InputPattern, OscillatoryNetwork};
pub use oscillatory_neuron::{OscillatoryNeuron, SimParams};
pub use simulation_controller::{SimulationController, SynchronousController};

/// Core constants for the generative engine
pub mod constants {
    /// Default simulation time step (10ms)
    pub const DEFAULT_DELTA_T: f64 = 0.01;

    /// Default network size for cognitive processing
    pub const DEFAULT_NETWORK_SIZE: usize = 96;

    /// Minimum biologically plausible frequency (0.1 Hz)
    pub const MIN_FREQUENCY: f64 = 0.1;

    /// Maximum biologically plausible frequency (100 Hz)  
    pub const MAX_FREQUENCY: f64 = 100.0;

    /// Minimum inhibition amplitude (no inhibition)
    pub const MIN_INHIB_AMPLITUDE: f64 = 0.0;

    /// Maximum inhibition amplitude (complete suppression)
    pub const MAX_INHIB_AMPLITUDE: f64 = 10.0;

    /// Minimum time constant (fast response)
    pub const MIN_TAU: f64 = 0.001;

    /// Maximum time constant (slow integration)
    pub const MAX_TAU: f64 = 10.0;
}
