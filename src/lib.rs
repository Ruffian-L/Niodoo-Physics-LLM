// pub mod adapter;
// pub mod bench;
// pub mod config;
pub mod constants; // Likely needed
                   // pub mod curator;
                   // pub mod embeddings;
                   // pub mod encoder;
                   // pub mod gaussian_rag;
                   // pub mod generative;
                   // pub mod genesis;
                   // pub mod gpu;
                   // pub mod hopfield;
                   // pub mod indexing;
                   // pub mod ingest;
                   // pub mod ipc;
                   // pub mod language;
                   // pub mod learning;
                   // pub mod linguistics;
                   // pub mod llm;
                   // pub mod manifold;
                   // pub mod memory;
                   // pub mod memory_system;
                   // pub mod model;
                   // pub mod splat_engine;
                   // pub mod synapse;
                   // pub mod organism;
pub mod physics;
// pub mod ranking;
// pub mod regulation;
// pub mod rendering;
// pub mod retrieval;
// pub mod search;
// pub mod server;
// pub mod energy;
// pub mod m3_compute;
// pub mod memory_topology;
// pub mod perceptual;
// pub mod shadow_logger;
// pub mod sheaf;
// pub mod storage;
pub mod structs;
// pub mod tivm;
// pub mod token_promotion;
pub mod types;
// pub mod utils;
pub mod visualizer; // Stubbed
                    // pub mod watch;

// Re-exports
// pub use config::SplatMemoryConfig;
// pub use indexing::TopologicalFingerprint;
// pub use ingest::IngestionEngine;
// pub use memory_system::MemorySystem;
// pub use search::{SearchMode, SearchResult, Searcher};
// pub use storage::TopologicalMemoryStore;
// pub use tivm::SplatRagConfig;
pub use types::{SplatId, SplatInput, SplatMeta};

// Lazy statics removed as they depend on above modules
