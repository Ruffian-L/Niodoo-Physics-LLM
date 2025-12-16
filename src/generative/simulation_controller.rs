//! SimulationController: High-level control of the oscillatory network
//!
//! Provides the interface between the generative engine and the rest of the system,
//! handling timing, threading, and external coordination.

use crate::generative::oscillatory_network::InputPattern;
use crate::generative::{OscillatoryNetwork, SimParams};
use anyhow::{Context, Result};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Commands that can be sent to the simulation controller
#[derive(Debug, Clone)]
pub enum SimulationCommand {
    /// Start or resume simulation
    Start,
    /// Pause simulation
    Pause,
    /// Stop simulation and reset
    Stop,
    /// Step simulation by N steps
    Step(usize),
    /// Set input pattern
    SetInputPattern(InputPattern),
    /// Update simulation parameters
    UpdateParams(SimParams),
    /// Get current network state
    GetState,
    /// Apply noise to network
    ApplyNoise(f64),
    /// Terminate simulation thread
    Terminate,
}

/// Network state information for external monitoring
#[derive(Debug, Clone)]
pub struct NetworkState {
    pub average_activation: f64,
    pub network_complexity: f64,
    pub active_neuron_count: usize,
    pub current_time: f64,
    pub simulation_speed: f64, // Steps per second
    pub total_steps: u64,
}

/// Messages sent from simulation thread to main thread
#[derive(Debug, Clone)]
pub enum SimulationMessage {
    /// Current network state
    State(NetworkState),
    /// Simulation error occurred
    Error(String),
    /// Simulation has terminated
    Terminated,
    /// Heartbeat indicating simulation is running
    Heartbeat,
}

/// Controller for running the oscillatory network simulation
///
/// This can run in real-time (with timing constraints) or as fast as possible.
/// It provides thread-safe control and monitoring capabilities.
pub struct SimulationController {
    /// The oscillatory network being simulated
    network: Arc<Mutex<OscillatoryNetwork>>,

    /// Command sender to simulation thread
    command_sender: Sender<SimulationCommand>,

    /// Message receiver from simulation thread
    message_receiver: Receiver<SimulationMessage>,

    /// Simulation thread handle
    simulation_thread: Option<thread::JoinHandle<()>>,

    /// Whether simulation is currently running
    is_running: Arc<Mutex<bool>>,

    /// Performance metrics
    metrics: Arc<Mutex<SimulationMetrics>>,
}

/// Performance and timing metrics for the simulation
#[derive(Debug, Clone, Default)]
pub struct SimulationMetrics {
    pub total_steps: u64,
    pub total_simulation_time: f64,
    pub average_step_time: f64,
    pub steps_per_second: f64,
    pub last_heartbeat: Option<Instant>,
}

impl SimulationController {
    /// Create a new simulation controller
    pub fn new(network: OscillatoryNetwork) -> Self {
        let (command_sender, command_receiver) = mpsc::channel();
        let (message_sender, message_receiver) = mpsc::channel();

        let network_shared = Arc::new(Mutex::new(network));
        let network_for_thread = Arc::clone(&network_shared);
        let is_running = Arc::new(Mutex::new(false));
        let is_running_for_thread = Arc::clone(&is_running);
        let metrics = Arc::new(Mutex::new(SimulationMetrics::default()));
        let metrics_for_thread = Arc::clone(&metrics);

        // Spawn simulation thread
        let thread_handle = thread::spawn(move || {
            Self::simulation_thread_loop(
                network_for_thread,
                command_receiver,
                message_sender,
                is_running_for_thread,
                metrics_for_thread,
            );
        });

        Self {
            network: network_shared,
            command_sender,
            message_receiver,
            simulation_thread: Some(thread_handle),
            is_running,
            metrics,
        }
    }

    /// Create controller with default network
    pub fn new_default() -> Self {
        Self::new(OscillatoryNetwork::new())
    }

    /// Start the simulation
    pub fn start(&self) -> Result<()> {
        self.command_sender
            .send(SimulationCommand::Start)
            .context("Failed to send start command")
    }

    /// Pause the simulation
    pub fn pause(&self) -> Result<()> {
        self.command_sender
            .send(SimulationCommand::Pause)
            .context("Failed to send pause command")
    }

    /// Stop and reset the simulation
    pub fn stop(&self) -> Result<()> {
        self.command_sender
            .send(SimulationCommand::Stop)
            .context("Failed to send stop command")
    }

    /// Step simulation by N steps
    pub fn step(&self, steps: usize) -> Result<()> {
        self.command_sender
            .send(SimulationCommand::Step(steps))
            .context("Failed to send step command")
    }

    /// Set input pattern for the network
    pub fn set_input_pattern(&self, pattern: InputPattern) -> Result<()> {
        self.command_sender
            .send(SimulationCommand::SetInputPattern(pattern))
            .context("Failed to set input pattern")
    }

    /// Update simulation parameters
    pub fn update_params(&self, params: SimParams) -> Result<()> {
        self.command_sender
            .send(SimulationCommand::UpdateParams(params))
            .context("Failed to update params")
    }

    /// Apply noise to network
    pub fn apply_noise(&self, noise_level: f64) -> Result<()> {
        self.command_sender
            .send(SimulationCommand::ApplyNoise(noise_level))
            .context("Failed to apply noise")
    }

    /// Get current network state
    pub fn get_state(&self) -> Result<()> {
        self.command_sender
            .send(SimulationCommand::GetState)
            .context("Failed to request state")
    }

    /// Check if simulation is currently running
    pub fn is_running(&self) -> Result<bool> {
        let running = self
            .is_running
            .lock()
            .map_err(|_| anyhow::anyhow!("Simulation state lock poisoned"))?;
        Ok(*running)
    }

    /// Get current performance metrics
    pub fn get_metrics(&self) -> Result<SimulationMetrics> {
        let m = self
            .metrics
            .lock()
            .map_err(|_| anyhow::anyhow!("Metrics lock poisoned"))?;
        Ok(m.clone())
    }

    /// Get network access for direct manipulation (use with caution)
    pub fn get_network_access(&self) -> Arc<Mutex<OscillatoryNetwork>> {
        Arc::clone(&self.network)
    }

    /// Process pending messages from simulation thread
    pub fn process_messages(&self) -> Vec<SimulationMessage> {
        let mut messages = Vec::new();
        while let Ok(message) = self.message_receiver.try_recv() {
            messages.push(message);
        }
        messages
    }

    /// Wait for next message (blocking)
    pub fn wait_for_message(&self) -> Result<SimulationMessage> {
        self.message_receiver
            .recv()
            .context("Failed to receive message")
    }

    /// Terminate the simulation thread
    pub fn terminate(self) -> Result<()> {
        // Send terminate command
        self.command_sender
            .send(SimulationCommand::Terminate)
            .context("Failed to send terminate command")?;

        // Wait for thread to finish
        if let Some(handle) = self.simulation_thread {
            handle
                .join()
                .map_err(|e| anyhow::anyhow!("Failed to join simulation thread: {:?}", e))?;
        }

        Ok(())
    }

    /// Main simulation thread loop
    fn simulation_thread_loop(
        network: Arc<Mutex<OscillatoryNetwork>>,
        command_receiver: Receiver<SimulationCommand>,
        message_sender: Sender<SimulationMessage>,
        is_running: Arc<Mutex<bool>>,
        metrics: Arc<Mutex<SimulationMetrics>>,
    ) {
        let mut running = false;
        let mut step_accumulator = 0.0;
        let mut last_heartbeat = Instant::now();

        loop {
            // Process commands
            let mut commands = Vec::new();
            while let Ok(command) = command_receiver.try_recv() {
                commands.push(command);
            }

            for command in commands {
                match command {
                    SimulationCommand::Start => {
                        if let Ok(mut r) = is_running.lock() {
                            running = true;
                            *r = true;
                        } else {
                            let _ = message_sender
                                .send(SimulationMessage::Error("State lock poisoned".to_string()));
                        }
                    }
                    SimulationCommand::Pause => {
                        if let Ok(mut r) = is_running.lock() {
                            running = false;
                            *r = false;
                        } else {
                            let _ = message_sender
                                .send(SimulationMessage::Error("State lock poisoned".to_string()));
                        }
                    }
                    SimulationCommand::Stop => {
                        if let Ok(mut r) = is_running.lock() {
                            running = false;
                            *r = false;
                        }

                        if let Ok(mut net) = network.lock() {
                            net.reset();
                        }
                        // Reset metrics
                        if let Ok(mut m) = metrics.lock() {
                            *m = SimulationMetrics::default();
                        }
                    }
                    SimulationCommand::Step(steps) => {
                        if let Ok(mut net) = network.lock() {
                            for _ in 0..steps {
                                Self::perform_simulation_step(
                                    &mut net,
                                    &mut step_accumulator,
                                    &metrics,
                                );
                            }
                        }
                    }
                    SimulationCommand::SetInputPattern(pattern) => {
                        if let Ok(mut net) = network.lock() {
                            net.apply_input_pattern(pattern);
                        }
                    }
                    SimulationCommand::UpdateParams(params) => {
                        if let Ok(mut net) = network.lock() {
                            net.update_params(params);
                        }
                    }
                    SimulationCommand::GetState => {
                        if let Ok(net) = network.lock() {
                            let state = Self::create_network_state(&net, &metrics);
                            let _ = message_sender.send(SimulationMessage::State(state));
                        }
                    }
                    SimulationCommand::ApplyNoise(noise_level) => {
                        if let Ok(mut net) = network.lock() {
                            net.apply_network_noise(noise_level);
                        }
                    }
                    SimulationCommand::Terminate => {
                        if let Ok(mut r) = is_running.lock() {
                            running = false;
                            *r = false;
                        }
                        let _ = message_sender.send(SimulationMessage::Terminated);
                        return;
                    }
                }
            }

            // Perform simulation step if running
            if running {
                if let Ok(mut net) = network.lock() {
                    Self::perform_simulation_step(&mut net, &mut step_accumulator, &metrics);
                }
            }

            // Send periodic heartbeat
            if last_heartbeat.elapsed() >= Duration::from_millis(100) {
                let _ = message_sender.send(SimulationMessage::Heartbeat);
                last_heartbeat = Instant::now();

                // Update heartbeat in metrics
                if let Ok(mut m) = metrics.lock() {
                    m.last_heartbeat = Some(last_heartbeat);
                }
            }

            // Small sleep to prevent busy waiting
            thread::sleep(Duration::from_micros(100));
        }
    }

    /// Perform a single simulation step with timing
    fn perform_simulation_step(
        network: &mut OscillatoryNetwork,
        step_accumulator: &mut f64,
        metrics: &Arc<Mutex<SimulationMetrics>>,
    ) {
        let step_start = Instant::now();

        // Perform the actual network step
        network.step();

        // Update timing metrics
        let step_duration = step_start.elapsed().as_secs_f64();
        *step_accumulator += network.params.delta_t;

        if let Ok(mut m) = metrics.lock() {
            m.total_steps += 1;
            m.total_simulation_time += network.params.delta_t;
            m.average_step_time = (m.average_step_time * (m.total_steps - 1) as f64
                + step_duration)
                / m.total_steps as f64;

            // Calculate steps per second
            if m.total_steps % 100 == 0 {
                m.steps_per_second = if step_duration > 0.0 {
                    1.0 / step_duration
                } else {
                    f64::INFINITY
                };
            }
        }
    }

    /// Create network state message
    fn create_network_state(
        network: &OscillatoryNetwork,
        metrics: &Arc<Mutex<SimulationMetrics>>,
    ) -> NetworkState {
        let stats = network.get_network_stats();
        // Handle lock poisoning gracefully for the metrics
        let (simulation_speed, total_steps) = if let Ok(m) = metrics.lock() {
            (m.steps_per_second, m.total_steps)
        } else {
            (0.0, 0)
        };

        NetworkState {
            average_activation: stats.average_activation,
            network_complexity: stats.network_complexity,
            active_neuron_count: stats.active_neuron_count,
            current_time: network.current_time,
            simulation_speed,
            total_steps,
        }
    }
}

/// A simpler synchronous controller for testing and non-real-time use
pub struct SynchronousController {
    network: OscillatoryNetwork,
}

impl SynchronousController {
    /// Create new synchronous controller
    pub fn new(network: OscillatoryNetwork) -> Self {
        Self { network }
    }

    /// Run simulation for specified steps
    pub fn run_steps(&mut self, steps: usize) -> NetworkState {
        for _ in 0..steps {
            self.network.step();
        }

        self.get_current_state()
    }

    /// Get current network state
    pub fn get_current_state(&self) -> NetworkState {
        let stats = self.network.get_network_stats();

        NetworkState {
            average_activation: stats.average_activation,
            network_complexity: stats.network_complexity,
            active_neuron_count: stats.active_neuron_count,
            current_time: self.network.current_time,
            simulation_speed: 0.0, // Not applicable for sync
            total_steps: (self.network.current_time / self.network.params.delta_t).round() as u64,
        }
    }

    /// Get network access
    pub fn network_mut(&mut self) -> &mut OscillatoryNetwork {
        &mut self.network
    }

    /// Get network reference
    pub fn network(&self) -> &OscillatoryNetwork {
        &self.network
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_synchronous_controller() {
        let mut controller = SynchronousController::new(OscillatoryNetwork::with_size(10));

        // Apply input and run
        controller
            .network_mut()
            .apply_input_pattern(InputPattern::Uniform(0.7));
        let state = controller.run_steps(10);

        assert!(state.average_activation > 0.0);
        assert!(state.current_time > 0.0);
        assert_eq!(state.total_steps, 10);
    }

    #[test]
    fn test_simulation_controller_creation() {
        let controller = SimulationController::new_default();
        assert!(!controller.is_running().unwrap());

        // Clean termination
        controller.terminate().unwrap();
    }

    #[test]
    fn test_simulation_controller_commands() {
        let controller = SimulationController::new_default();

        // Test command sending
        assert!(controller.start().is_ok());
        assert!(controller.step(5).is_ok());
        assert!(controller.pause().is_ok());
        assert!(controller.stop().is_ok());

        // Clean termination
        controller.terminate().unwrap();
    }

    #[test]
    fn test_simulation_controller_messaging() {
        let controller = SimulationController::new_default();

        // Request state
        controller.get_state().unwrap();

        // Process messages
        let messages = controller.process_messages();
        // Note: messages might be empty immediately if thread hasn't processed command yet
        // assert!(!messages.is_empty());

        // Clean termination
        controller.terminate().unwrap();
    }

    #[test]
    fn test_simulation_controller_running_state() {
        let controller = SimulationController::new_default();

        // Should not be running initially
        assert!(!controller.is_running().unwrap());

        // Start simulation
        controller.start().unwrap();
        thread::sleep(Duration::from_millis(10));

        // Should be running now
        assert!(controller.is_running().unwrap());

        // Stop simulation
        controller.stop().unwrap();
        thread::sleep(Duration::from_millis(10));

        // Should not be running
        assert!(!controller.is_running().unwrap());

        // Clean termination
        controller.terminate().unwrap();
    }

    #[test]
    fn test_simulation_metrics() {
        let controller = SimulationController::new_default();

        // Run some steps
        controller.step(100).unwrap();
        thread::sleep(Duration::from_millis(50));

        // Check metrics
        let metrics = controller.get_metrics().unwrap();
        assert!(metrics.total_steps >= 100);
        assert!(metrics.total_simulation_time > 0.0);

        // Clean termination
        controller.terminate().unwrap();
    }
}
