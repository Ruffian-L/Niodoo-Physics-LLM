// --- WGSL Compute Shader preamble ---

// 1. The "Mind's Eye" Camera
struct CameraUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    camera_position: vec3<f32>,
    // Padding required for WGSL alignment constraints
    padding: f32, 
};
@group(0) @binding(0) var<uniform> camera: CameraUniforms;

// 2. The Raw Memory Primitives (Input)
// This matches your Rust 'PackedSemantics' struct minus the padding
struct GaussianMemory {
    position: vec3<f32>,
    opacity: f32,
    scale: vec3<f32>,
    rotation: vec4<f32>, // Quaternion
    // CRITICAL CHANGE: This is no longer SH color. It's a learnable query.
    // Let's assume a 16-dimensional lightweight query vector for now.
    query_vector: array<vec4<f32>, 4>, // 16 floats packed into 4 vec4s
};
// A massive array of all your memories
@group(0) @binding(1) var<storage, read> memory_bank: array<GaussianMemory>;

// 3. The Principal Scene Component (PSC) Bank (Global Knowledge)
// This is the "dictionary" the gaussians attend to.
// Assuming standard Nomic embedding size (768 dims) packed into vec4s.
// 768 / 4 = 192 vec4s per PSC component.
struct PSCComponent {
    features: array<vec4<f32>, 192>,
};
// Let's assume a bank size of 1024 principal components
@group(0) @binding(2) var<storage, read> psc_bank: array<PSCComponent, 1024>;

// 4. The Rendered Feature Output (What gets sent to the LLM adapter)
struct RenderedFeature {
    // The final 768-dim vector resolved from attention
    final_feature: array<vec4<f32>, 192>,
    // We also need its projected 2D position and calculated alpha for the rasterizer later
    projected_pos: vec2<f32>,
    final_alpha: f32,
};
@group(0) @binding(3) var<storage, read_write> output_features: array<RenderedFeature>;

// --- Helper: Gaussian Memory Attention ---

// Constants for our dimensions
const PSC_BANK_SIZE: u32 = 1024u;
const FEATURE_DIM_VEC4: u32 = 192u; // 768 / 4
const QUERY_DIM_VEC4: u32 = 4u; // 16 / 4
const SQRT_DK: f32 = 4.0; // Sqrt of query dim (16) for scaling

fn compute_attention(gaussian_idx: u32) -> array<vec4<f32>, FEATURE_DIM_VEC4> {
    let g_mem = memory_bank[gaussian_idx];
    
    // --- 1. Calculate Raw Attention Scores (Dot Product) ---
    var scores: array<f32, PSC_BANK_SIZE>;
    var max_score: f32 = -1e30; // For numerical stability in softmax

    for (var i = 0u; i < PSC_BANK_SIZE; i = i + 1u) {
        var dot_prod: f32 = 0.0;
        // Dot product the 16-dim query vs the first 16 dims of the PSC key
        // (Assuming PSC keys are stored at the start of the PSC feature block)
        // Unrolled loop for q = 0 to 3
        dot_prod = dot_prod + dot(g_mem.query_vector[0u], psc_bank[i].features[0u]);
        dot_prod = dot_prod + dot(g_mem.query_vector[1u], psc_bank[i].features[1u]);
        dot_prod = dot_prod + dot(g_mem.query_vector[2u], psc_bank[i].features[2u]);
        dot_prod = dot_prod + dot(g_mem.query_vector[3u], psc_bank[i].features[3u]);
        scores[i] = dot_prod / SQRT_DK;
        max_score = max(max_score, scores[i]);
    }

    // --- 2. Calculate Softmax ---
    var score_sum: f32 = 0.0;
    for (var i = 0u; i < PSC_BANK_SIZE; i = i + 1u) {
        // Subtract max_score for stability (prevents overflow)
        scores[i] = exp(scores[i] - max_score);
        score_sum = score_sum + scores[i];
    }
    
    // --- 3. Weighted Sum (Reconstruction) ---
    var final_feature: array<vec4<f32>, FEATURE_DIM_VEC4>;
    // Initialize to zero
    for (var f = 0u; f < FEATURE_DIM_VEC4; f = f + 1u) {
        final_feature[f] = vec4<f32>(0.0);
    }

    for (var i = 0u; i < PSC_BANK_SIZE; i = i + 1u) {
        let attention_weight = scores[i] / score_sum;
        // Add the weighted PSC component to the final feature
        for (var f = 0u; f < FEATURE_DIM_VEC4; f = f + 1u) {
            final_feature[f] = final_feature[f] + (psc_bank[i].features[f] * attention_weight);
        }
    }
    
    return final_feature;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&memory_bank)) {
        return;
    }

    // Compute the semantic feature
    let feature = compute_attention(idx);

    // Store result
    // Note: projected_pos and final_alpha would normally be calculated here too
    // based on camera uniforms, but for now we just store the feature.
    output_features[idx].final_feature = feature;
    output_features[idx].projected_pos = vec2<f32>(0.0, 0.0); // Placeholder
    output_features[idx].final_alpha = 1.0; // Placeholder
}
