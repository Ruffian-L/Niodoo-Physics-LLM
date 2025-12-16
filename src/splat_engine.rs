use crate::adapter::SplatAdapter;
use crate::llm::qwen::Model as MyBaseModel;
use crate::token_promotion::dynamic_tokenizer::DynamicTokenizer;
use candle_core::{Device, IndexOp, Result, Tensor};

pub struct SplatEngine {
    pub base_model: MyBaseModel, // Kept for structure, but unused in God Protocol
    pub adapter: SplatAdapter,   // Kept for structure
    pub tokenizer: DynamicTokenizer, // Kept for decoding
    pub device: Device,
    pub ghost_vectors: Vec<Tensor>,
    pub anti_ghost_vectors: Vec<Tensor>,
    pub gain: f64,
}

impl SplatEngine {
    /// Initialize the Engine
    pub fn new(
        base_model: MyBaseModel,
        adapter: SplatAdapter,
        tokenizer: DynamicTokenizer,
        device: Device,
    ) -> Self {
        Self {
            base_model,
            adapter,
            tokenizer,
            device,
            ghost_vectors: Vec::new(),
            anti_ghost_vectors: Vec::new(),
            gain: 0.6,
        }
    }

    pub fn set_gain(&mut self, gain: f64) {
        self.gain = gain;
    }

    pub fn get_gain(&self) -> f64 {
        self.gain
    }

    pub fn clear_ghosts(&mut self) {
        self.ghost_vectors.clear();
        self.anti_ghost_vectors.clear();
    }

    pub fn add_ghost_vector(&mut self, vector: Tensor) {
        self.ghost_vectors.push(vector);
    }

    pub fn add_anti_ghost_vector(&mut self, vector: Tensor) {
        self.anti_ghost_vectors.push(vector);
    }

    pub fn inject_ghost_sequence(&mut self, sequence: &Tensor) -> Result<()> {
        Ok(())
    }

    /// The God Protocol: Physics-Based Generation
    /// Replaces token prediction with energetic displacement rendering.
    pub fn step(&mut self, input_tensor: &Tensor) -> anyhow::Result<Tensor> {
        // 1. Decode Input to Text (Reverse Tokenization)
        let input_ids: Vec<u32> = input_tensor.flatten_all()?.to_vec1()?;
        let _text = self
            .tokenizer
            .decode_batch(&[input_ids.clone()], true)
            .map(|v| v.first().cloned().unwrap_or_default())
            .unwrap_or_default();

        // 2. Run Physics Simulation (The "Thought")
        // We need access to the memory store.
        // SplatEngine doesn't own the store directly in this architecture (MemorySystem does).
        // But we are in God Protocol. We must forge a connection.
        // For now, we simulate the "thought" by returning a tensor that encodes the
        // "energy" of the response.

        // Since we can't easily access the store here without refactoring `main.rs` to pass it in,
        // we will emit a special "Physics Token" that the outer loop (in `splat_chat.rs`)
        // interprets as a signal to run the physics engine.

        // Token ID 999999 = "PHYSICS_THOUGHT"
        let physics_token = 999999u32;

        // Return a tensor with just this token
        let device = input_tensor.device();
        let out = Tensor::new(&[physics_token], device)?.unsqueeze(0)?;

        Ok(out)
    }

    pub fn step_with_injection(
        &mut self,
        input_ids: &Tensor,
        _embedding: Vec<f32>,
    ) -> Result<Tensor> {
        self.step(input_ids)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))
    }
}
