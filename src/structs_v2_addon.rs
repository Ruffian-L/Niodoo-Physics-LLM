// Add to structs.rs after SplatSemantics (around line 150)
// This goes right after the SplatSemantics definition

/// Manifesto-compliant discrete RVQ-based semantics (Version 2)
/// Stores 12 discrete codes instead of dense 768D embedding
/// Storage: ~48 bytes vs ~3072 bytes (64-128× compression)
#[derive(Debug, Clone, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct SplatSemanticsV2 {
    pub payload_id: u64,
    pub birth_time: f64,
    pub confidence: f32,
    
    /// RVQ discrete codes (12 layers × u16 = 24 bytes)
    /// Replaces embedding: [f32; 768] (3072 bytes)
    pub rvq_indices: [u16; crate::encoder::rvq_candle::NUM_QUANTIZERS],
    
    /// Mass derived from code[0] rarity (self-information)
    /// Rarer coarse codes = higher mass = stronger gravity wells
    pub coarse_mass: u8,
    
    /// Domain valence (kept from v1 for compatibility)
    pub domain_valence: [f32; 4],
    
    /// Manifold vector (64-dim subspace) - kept for compatibility
    #[serde(with = "BigArray")]
    pub manifold_vector: [f32; 64],
    
    // --- God Protocol Additions (kept) ---
    #[serde(default)]
    pub emotional_state: Option<EmotionalState>,
    
    #[serde(default)]
    pub fitness_metadata: Option<WeightedMemoryMetadata>,
}

impl Default for SplatSemanticsV2 {
    fn default() -> Self {
        Self {
            payload_id: 0,
            birth_time: 0.0,
            confidence: 1.0,
            rvq_indices: [0u16; crate::encoder::rvq_candle::NUM_QUANTIZERS],
            coarse_mass: 128, // Neutral mass
            domain_valence: [0.25, 0.25, 0.25, 0.25],
            manifold_vector: [0.0; 64],
            emotional_state: None,
            fitness_metadata: None,
        }
    }
}
