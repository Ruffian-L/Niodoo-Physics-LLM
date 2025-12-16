
// Stub Visualizer for Headless Release
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RenderParticle {
    pub position: [f32; 3],
    pub color: [f32; 3],
    pub radius: f32, 
}

// Dummy run function if referenced
pub async fn run_visualizer(_rx: std::sync::mpsc::Receiver<Vec<RenderParticle>>) {
    println!("Visualizer is disabled in Niodoo Headless Engine.");
}
