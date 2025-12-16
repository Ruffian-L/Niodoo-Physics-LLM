use crate::energy::compute_sheaf_energy;
use crate::sheaf::SheafGraph;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Linear;

#[derive(Debug, PartialEq)]
pub enum CuratorDecision {
    Merge,       // Safe to average vectors
    Reject,      // Contradictory noise
    Encapsulate, // High-Valence Paradox (Keep distinct)
}

pub struct Curator {
    device: Device,
}

impl Curator {
    pub fn new(device: Device) -> Self {
        Self { device }
    }

    pub fn judge(
        &self,
        new_vec: &Tensor,
        old_vec: &Tensor,
        valence: f32,
    ) -> Result<CuratorDecision> {
        let energy_threshold = 0.2;
        let valence_threshold = 0.8;

        // Create a temporary SheafGraph to measure energy between the two vectors
        let mut graph = SheafGraph::new(self.device.clone());
        let dim = new_vec.dim(1)?;

        // Node 1: Old Memory
        graph.add_node(1, old_vec.clone());

        // Node 2: New Memory
        graph.add_node(2, new_vec.clone());

        // Edge: Identity (We are testing if they are the "same" concept)
        let weight = Tensor::eye(dim, DType::F32, &self.device)?;
        let b_12 = Linear::new(weight.clone(), None);
        let b_21 = Linear::new(weight, None);

        graph.add_edge(1, 2, b_12, b_21);

        // Compute Energy
        let energy = compute_sheaf_energy(&graph)?;

        if energy < energy_threshold {
            Ok(CuratorDecision::Merge)
        } else if energy > energy_threshold && valence > valence_threshold {
            Ok(CuratorDecision::Encapsulate)
        } else {
            Ok(CuratorDecision::Reject)
        }
    }
}
