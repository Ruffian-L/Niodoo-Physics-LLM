//! Regulation System: Feedback Loop Control & Emergent Homeostasis
//!
//! "Where the system learns to regulate its own emergence"
//!
//! Phase 3 implements closed-loop control laws that allow the system to:
//! - Maintain optimal complexity through topological homeostasis
//! - Generate intrinsic motivation via Wundt curve optimization  
//! - Self-regulate emergence based on internal state monitoring
//! - Achieve sustainable complexity without external guidance

pub mod emergence_controller;
pub mod topological_homeostasis;
pub mod wundt_optimizer;

pub use emergence_controller::{
    ControlLoopState, EmergenceController, GainDebugInfo, QueryContext,
};
pub use topological_homeostasis::{HomeostaticControl, HomeostaticState, TopologicalHomeostasis};
pub use wundt_optimizer::{IntrinsicMotivation, WundtOptimizer};
