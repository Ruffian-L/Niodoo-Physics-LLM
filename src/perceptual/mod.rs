//! Perceptual System: Topological State Reconstruction
//!
//! Connects the OscillatoryNeuron engine to topological memory through
//! Takens' embedding and persistence diagram analysis.

pub mod phase_locked_oscillator;
pub mod takens_embedding;
pub mod topological_perceiver;

pub use phase_locked_oscillator::{
    ResonanceFeeling, ResonanceMemory, RhythmicSignature, TopologicalOscillator,
};
pub use takens_embedding::TakensEmbedding;
pub use topological_perceiver::{
    BettiNumbers, ComplexityTrend, PersistenceMeasures, TopologicalFeatures, TopologicalPerceiver,
    TopologicalRegime,
};
