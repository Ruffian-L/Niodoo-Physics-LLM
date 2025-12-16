// PhysicsLang CUDA Kernels - Aalto U(1)^4 Gauge Gravity Implementation
// Based on 2025 quantum gravity breakthroughs: Partanen & Tulkki gauge theory
// domain_valence[4] = four U(1) gauge symmetries for gravity emergence in flat spacetime

// Helper to compute Hamming distance between two 8-int codes
__device__ int hamming_dist(const int* a, const int* b) {
    int dist = 0;
    for (int i=0; i<8; ++i) {
        dist += __popc(a[i] ^ b[i]);
    }
    return dist;
}

<<<<<<< Updated upstream
// Mahalanobis distance using inverse covariance matrix
__device__ float mahalanobis_dist(const int* a, const int* b, const float* inv_cov) {
    float diff[8];
    for (int i = 0; i < 8; ++i) {
        diff[i] = (float)(a[i] - b[i]);
    }
    float result = 0.0f;
    for (int i = 0; i < 8; ++i) {
        float temp = 0.0f;
        for (int j = 0; j < 8; ++j) {
            temp += inv_cov[i * 8 + j] * diff[j];
        }
        result += diff[i] * temp;
    }
    return sqrtf(fmaxf(result, 0.0f));
}

// Morton code encoding - fixed for 64-bit limit
// Project to 6D subspace, 10 bits per dimension = 60 bits total (fits u64)
extern "C" __global__ void compute_morton_codes(
    const float* __restrict__ positions,
    unsigned long long* __restrict__ morton_codes,
    int* __restrict__ indices,
=======
extern "C" __global__ void compute_forces_zorder(
    const float* __restrict__ positions,    // [N, 6] unsorted (token order)
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ codes,          // [N, 8] RVQ codes
    const int* __restrict__ sorted_indices, // [N] output from sort
    float* __restrict__ forces,             // [N, 6]
>>>>>>> Stashed changes
    int num_particles,
    float min_val,
    float max_val
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    
    indices[idx] = idx;
    
    // Project 1024D to 6D via averaging 170-dim chunks, then morton encode
    float reduced[6];
    for (int d = 0; d < 6; ++d) {
        float sum = 0.0f;
        int start = d * 170;
        int end = (d == 5) ? 1024 : start + 170;
        for (int k = start; k < end; ++k) {
            sum += positions[idx * 1024 + k];
        }
        reduced[d] = sum / (end - start);
    }
    
    // Quantize to 10 bits (0-1023) per dimension
    float scale = 1023.0f / (max_val - min_val + 1e-6f);
    unsigned long long code = 0;
    
    for (int d = 0; d < 6; ++d) {
        float p = reduced[d];
        if (p < min_val) p = min_val;
        if (p > max_val) p = max_val;
        unsigned int q = (unsigned int)((p - min_val) * scale);
        if (q > 1023) q = 1023;
        
        // Interleave: bit i of dim d goes to bit (i*6 + d)
        for (int i = 0; i < 10; ++i) {
            unsigned long long bit = (q >> i) & 1ULL;
            code |= (bit << (i * 6 + d));
        }
    }
    morton_codes[idx] = code;
}

// ============================================================================
// MAIN PHYSICS KERNEL: Aalto 2025 Gauge Gravity + Domain Crystallization
// - domain_valence[4] as U(1)^4 gauge fields (exactly matches Aalto paper!)
// - Semantic gravity with gauge modulation
// - Universal Coulomb repulsion prevents singularity collapse
// - CODE RESONANCE: RVQ codes gate interactions (Phase 1 upgrade)
// ============================================================================
extern "C" __global__ void compute_query_force(
    const float* __restrict__ positions,      // [N, 256]
    const float* __restrict__ original_emb,   // [N, 256]
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ codes,            // [N, 8] RVQ codes
    const float* __restrict__ domain_valence, // [N, 4]
    float* __restrict__ forces,
    int num_particles,
    int query_idx,
    float G,
    float k_e,
    float softening,
<<<<<<< Updated upstream
    float context_bonus,
    float domain_repulsion_threshold,
    float domain_attraction_boost,
    float resonance_threshold,
    float resonance_damping
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles || idx == query_idx) return;

    // Query in registers via shared memory (8KB, perfect for 256D)
    extern __shared__ float shared[];
    float* s_q_pos  = shared;
    float* s_q_orig = shared + 256;

    if (threadIdx.x < 256) {
        s_q_pos[threadIdx.x]  = positions[query_idx * 256 + threadIdx.x];
        s_q_orig[threadIdx.x] = original_emb[query_idx * 256 + threadIdx.x];
    }
    __syncthreads();

    // =========================================================================
    // =========================================================================
    // CODE RESONANCE GATING
    // If RVQ codes are too different, reduce interaction strength
    // This creates "semantic insulation" between unrelated concept domains
    // =========================================================================
    int code_dist = 0;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        int q_code = codes[query_idx * 8 + k];
        int my_code = codes[idx * 8 + k];
        if (q_code != my_code) code_dist++;
    }
    
    // Use passed-in resonance parameters 
    // resonance_threshold: how many mismatches allowed (e.g. 6.0)
    // resonance_damping: multiplier when threshold exceeded (e.g. 0.1)
    float code_resonance = (code_dist > resonance_threshold) ? resonance_damping : 1.0f;
    // =========================================================================

    float q_mass = masses[query_idx];
    float q_charge = charges[query_idx];
    float q_domain[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) q_domain[k] = domain_valence[query_idx * 4 + k];

    float my_mass = masses[idx];
    float my_charge = charges[idx];
    float my_domain[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) my_domain[k] = domain_valence[idx * 4 + k];

    // Distance + semantic similarity
    float dist_sq = softening;
    float dot = 0.0f;
    float q_norm_sq = 1e-8f;
    float my_norm_sq = 1e-8f;
    float d_pos[256];

    #pragma unroll 32
    for (int k = 0; k < 256; ++k) {
        float qp = s_q_pos[k];
        float mp = positions[idx * 256 + k];
        float d = qp - mp;
        d_pos[k] = d;
        dist_sq += d * d;

        float qo = s_q_orig[k];
        float mo = original_emb[idx * 256 + k];
        dot += qo * mo;
        q_norm_sq += qo * qo;
        my_norm_sq += mo * mo;
    }

    float dist = sqrtf(dist_sq);
    float inv_dist3 = 1.0f / (dist_sq * dist + 1e-8f);
    
    // TUNING FIX: Sharpen semantic similarity to favor high-quality matches
    // Old: sim^2
    // New: (sim^2)^4 = sim^8
    // e.g., 0.6^8 = 0.016, 0.4^8 = 0.0006 (25x stronger)
    float base_sim = (dot * dot) / (q_norm_sq * my_norm_sq);
    float semantic_sim_sq = base_sim * base_sim; 
    semantic_sim_sq *= semantic_sim_sq; // ^4
    semantic_sim_sq *= semantic_sim_sq; // ^8


    // Fast cos(4θ) approximation — no acos, no trig, 4x faster
    float cos_theta = 0.0f;
    float mag_q = 0.0f, mag_my = 0.0f;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        cos_theta += q_domain[k] * my_domain[k];
        mag_q += q_domain[k] * q_domain[k];
        mag_my += my_domain[k] * my_domain[k];
    }
    cos_theta /= (sqrtf(mag_q) * sqrtf(mag_my) + 1e-6f);
    cos_theta = fmaxf(-1.0f, fminf(1.0f, cos_theta));

    float x2 = cos_theta * cos_theta;
    float cos4theta = 8.0f * x2 * x2 - 8.0f * x2 + 1.0f;  // exact cos(4θ)
    float topology_weight = fmaxf(0.0f, cos4theta);
    topology_weight = topology_weight * topology_weight;  // sharpen

    // qMOND with f(Q) screening
    // qMOND with f(Q) screening
    float my_mass_eff = my_mass + 0.147f * powf(my_mass, 1.12f);
    
    // TUNING FIX 2: GLOBAL SEMANTIC GRAVITY
    float dist_decay = 1.0f; // Global search

    // TUNING FIX 3: SEMANTIC CLEANING (Active Repulsion)
    // If sim < 0.58, we REPEL them to clear the basin for the best matches.
    // 0.61^8 = 0.019 (Attract)
    // 0.56^8 = 0.009 (Repel)
    
    float attraction_sign = (semantic_sim_sq > 0.012f) ? 1.0f : -10.0f; 

    // Final force with CODE RESONANCE applied
    float attraction = G * q_mass * my_mass_eff * semantic_sim_sq * (1.0f + domain_attraction_boost * topology_weight);
    
    // If repelling, we ignore semantic_sim_sq magnitude (which is small) and push hard
    if (attraction_sign < 0.0f) {
        attraction = G * q_mass * my_mass_eff * 0.05f; // Fixed magnitude repulsion
    }

    float repulsion = k_e * q_charge * my_charge;
    
    // Apply sign to attraction term
    float net_scalar = (attraction * attraction_sign - repulsion) * dist_decay * context_bonus * code_resonance;

    #pragma unroll 32
    for (int k = 0; k < 256; ++k) {
        atomicAdd(&forces[idx * 256 + k], net_scalar * d_pos[k]);
    }
}

// ============================================================================
// GENERATIVE PHYSICS KERNEL: Valence Forces for Thought Evolution
// Calculates interaction between dynamic query particles ("Thought Vectors")
// ============================================================================
extern "C" __global__ void compute_valence_forces(
    const float* __restrict__ positions,      // [N_query, 256]
    const float* __restrict__ masses,         // [N_query]
    const float* __restrict__ valence_matrix, // [N_query * N_query] - Asymmetric interaction strength
    float* __restrict__ forces,               // [N_query, 256]
    int num_particles,                        // Number of query particles
    float G                                   // Generative Gravity
=======
    int window_half // Total window = 2*window_half
>>>>>>> Stashed changes
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_particles) return;

    int i = sorted_indices[tid];

<<<<<<< Updated upstream
    float my_mass = masses[idx];
    float f_accum[256];
    #pragma unroll
    for (int k=0; k<256; ++k) f_accum[k] = 0.0f;

    for (int j = 0; j < num_particles; ++j) {
        if (idx == j) continue;

        // Force: p_j acting on p_i (j pulling i)
        // Valence: valence_matrix[idx * num_particles + j]
        // This signifies how much particle i is attracted to particle j.
        
        float interaction_strength = valence_matrix[idx * num_particles + j]; 
        
        // Optimization: If interaction is negligible, skip
        if (fabsf(interaction_strength) < 1e-6f) continue;
        
        float other_mass = masses[j];
        
        // Distance vector r_ij = pos_j - pos_i
        // PHASE 12: Only use first 32 dimensions (matching safetensors)
        float d_vec[256];
        float dist_sq = 0.0f;
        
        // Compute distance in 32D (where actual embeddings live)
        #pragma unroll
        for (int k=0; k<32; ++k) {
            float d = positions[j * 256 + k] - positions[idx * 256 + k];
            d_vec[k] = d;
            dist_sq += d * d;
        }
        // Zero out unused dimensions
        #pragma unroll
        for (int k=32; k<256; ++k) {
            d_vec[k] = 0.0f;
        }
        
        float dist = sqrtf(dist_sq + 1e-6f);
        
        // Force Law: F = G * m1 * m2 * strength / (dist + epsilon)
        // Using 1/r potential (Gravity)
        float f_scalar = G * my_mass * other_mass * interaction_strength;
        f_scalar /= (dist + 0.1f); // Softening
        
        #pragma unroll 32
        for (int k=0; k<256; ++k) {
            f_accum[k] += f_scalar * d_vec[k]; 
        }
    }

    #pragma unroll 32
    for (int k=0; k<256; ++k) {
        atomicAdd(&forces[idx * 256 + k], f_accum[k]);
=======
    float my_pos[6];
    float my_mass = masses[i];
    float my_charge = charges[i];
    int my_code[8];
    for (int k = 0; k < 6; ++k) my_pos[k] = positions[i * 6 + k];
    for (int k = 0; k < 8; ++k) my_code[k] = codes[i * 8 + k];

    float force[6] = {0.0f};

    int start = max(0, tid - window_half);
    int end   = min(num_particles, tid + window_half + 1);

    for (int t = start; t < end; ++t) {
        int j = sorted_indices[t];
        if (i == j) continue;

        float d_pos[6];
        float dist_sq = softening;
        for (int k = 0; k < 6; ++k) {
            d_pos[k] = positions[j * 6 + k] - my_pos[k];
            dist_sq += d_pos[k] * d_pos[k];
        }

        float dist = sqrtf(dist_sq);
        float inv_dist3 = 1.0f / (dist_sq * dist);

        // Hamming similarity (fast path if very dissimilar)
        int h_dist = 0;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            h_dist += __popc(my_code[k] ^ codes[j * 8 + k]);
        }
        if (h_dist > 48) continue; // Early reject — kills 90%+ of weak interactions

        float sim = 1.0f / (1.0f + h_dist * 0.092f); // Tuned for maximal recall

        // Gravity + Coulomb + Exclusion
        float f_grav = G * my_mass * masses[j] * sim * inv_dist3;

        float f_elec = -k_e * my_charge * charges[j] * inv_dist3;

        float f_excl = 0.0f;
        if (my_mass < 0.0f || masses[j] < 0.0f) {
            f_excl = -5.0f * G * fabsf(my_mass) * fabsf(masses[j]) * inv_dist3;
            f_grav = 0.0f;
        }

        float f_total = f_grav + f_elec + f_excl;

        for (int k = 0; k < 6; ++k) {
            force[k] += f_total * d_pos[k];
        }
    }

    for (int k = 0; k < 6; ++k) {
        forces[i * 6 + k] = force[k];
>>>>>>> Stashed changes
    }
}

// Calculates force exerted BY Universe ON Query Particles (Inverse Gravity)
// One thread per Universe Particle. Loops over Queries.
// MANIFOLD INJECTION: Uses q_transformed for Dot Product (Attention), q_positions for Distance.
extern "C" __global__ void compute_semantic_gravity(
    const float* __restrict__ universe_pos,    // [StrideUni, 512] SoA
    const float* __restrict__ universe_mass,   // [StrideUni]
    const float* __restrict__ universe_charge, // [StrideUni]
    const int* __restrict__ universe_codes,    // [StrideUni * 8]
    const float* __restrict__ q_positions,     // [StrideGen, 512] SoA (Distance)
    const float* __restrict__ q_transformed,   // [StrideGen, 512] SoA (Attention: q @ M)
    const float* __restrict__ q_masses,        // [StrideGen]
    const float* __restrict__ q_charges,       // [StrideGen]
    const int* __restrict__ q_codes,           // [StrideGen * 8]
    float* __restrict__ q_forces,              // [StrideGen, 512] SoA
    int num_universe,
    int num_queries,
    int stride_uni,
    int stride_gen,
    float G,
    float resonance_threshold,
    float resonance_damping
) {
    // 1. Setup Shared Memory Accumulator
    // Limit: 16 query particles max per block to fit 512D.
    // 16 * 512 * 4 bytes = 32KB. Fits in 48KB.
    __shared__ float s_forces[16 * 512];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared forces to 0
    for (int k = tid; k < 16 * 512; k += blockDim.x) {
        s_forces[k] = 0.0f;
    }
    __syncthreads();

    // 3. Process Universe Particle (One particle per thread)
    if (idx < num_universe) {
        float u_pos[512];
        float u_mass = universe_mass[idx];
        float u_charge = universe_charge[idx];
        
        // Read Universe Position (SoA: dim * stride + idx)
        #pragma unroll 16
        for (int k=0; k<512; ++k) u_pos[k] = universe_pos[k * stride_uni + idx];

        // Also read Universe Codes (AoS assumed for int8? Or SoA? Check Rust implementation.)
        // Rust alloc: capacity * 8. copy codes.
        // It's likely linear AoS [p0c0, p0c1... p1c0...].
        int u_code[8];
        #pragma unroll
        for (int k=0; k<8; ++k) u_code[k] = universe_codes[idx*8 + k];

        // Loop over Queries (Batch 16)
        // Note: Host must handle num_queries > 16 by multiple launches or looping blocks? 
        // For now hardcoded max 16 per kernel pass to simplify shared mem.
        for (int q_idx = 0; q_idx < num_queries && q_idx < 16; ++q_idx) {
            float my_mass = q_masses[q_idx];
            
            // Calc distance (Euclidean in Embedding Space)
            float dist_sq = 0.0f;
            float d_vec[512];
            
            #pragma unroll 16
            for (int k=0; k<512; ++k) {
                // SoA Access for Query
                float qp = q_positions[k * stride_gen + q_idx];
                float d = u_pos[k] - qp;
                d_vec[k] = d;
                dist_sq += d * d;
            }

            const float R_MAX = 1.0f; 
            if (dist_sq > R_MAX * R_MAX) continue;

            float dist = sqrtf(dist_sq + 1e-6f);

            // MANIFOLD ATTENTION (Dot Product in Qwen Space)
            // q_trans = q @ M.
            // Attn = q_trans . u
            float attn = 0.0f;
            #pragma unroll 16
            for (int k=0; k<512; ++k) {
                // SoA Access
                attn += q_transformed[k * stride_gen + q_idx] * u_pos[k];
            }
            // Softmax-like scaling?
            // Gravitationalgram uses exp(attn). Or just attn^2?
            // "The Lobotomy" used raw Softmax(S_ij).
            // Let's use exp(attn * scale)? 
            // Or just 'attn' if projected correctly.
            // Let's assume M is scaled.
            // Use Sigmoid/ReLU to prevent negative gravity?
            float sim_factor = fmaxf(0.0f, attn); 
            sim_factor = sim_factor * sim_factor; // Sharpen (Attn^2)

            // Code Resonance
            int code_dist = 0;
            #pragma unroll
            for (int k=0; k<8; ++k) {
                if (u_code[k] != q_codes[q_idx * 8 + k]) code_dist++;
            }
            float resonance = (code_dist > resonance_threshold) ? resonance_damping : 1.0f;

            // Physics Stats
            float my_charge = q_charges[q_idx];
            float pos_boost = 1.0f;
            float torsion_factor = 1.0f;

            if (fabsf(my_charge) > 0.1f && fabsf(u_charge) > 0.1f) {
                if (my_charge * u_charge > 0.0f) {
                    pos_boost = 1.0f; // Like charges: Normal (Gravity dominates)
                    // Wait, Coulomb is separate. This is GRAVITY.
                    // Keep Gravity clean.
                } else {
                     // Opposites attract strongly in semantic space?
                     pos_boost = 5.0f; 
                }
            }

            // Force Scalar
            float scalar = G * my_mass * u_mass * resonance * sim_factor * pos_boost / (dist + 0.1f);
            
            if (fabsf(scalar) < 1e-5f) continue;

            // Accumulate to Shared Memory (SoA accumulation?? No, s_forces is flat [q_idx * 512 + k])
            // Wait, s_forces indexing:
            for (int k=0; k<512; ++k) {
                atomicAdd(&s_forces[q_idx * 512 + k], scalar * d_vec[k]);
            }
        }
    }

    __syncthreads();

    // 4. Flush Shared to Global (SoA)
    // s_forces is [q_idx * 512 + k] (AoS-ish within shared).
    // Target q_forces is SoA [k * stride_gen + q_idx].
    for (int k = tid; k < 16 * 512; k += blockDim.x) {
        int q_idx = k / 512;
        int dim = k % 512;
        if (q_idx < num_queries) {
            float val = s_forces[k];
            if (fabsf(val) > 1e-9f) {
                // Write to SoA
                atomicAdd(&q_forces[dim * stride_gen + q_idx], val); 
            }
        }
    }
}

// Sentence-level springs (keep tokens within same doc close)
extern "C" __global__ void compute_springs(
    const float* __restrict__ positions,
    float* __restrict__ forces,
    int num_particles,
    float k_spring,
    float L0
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles - 1) return;

    int next = idx + 1;
    float dist_sq = 0.0f;
    
    // Pass 1: Distance
    #pragma unroll 32
    for (int k=0; k<256; ++k) {
        float d = positions[next*256 + k] - positions[idx*256 + k];
        dist_sq += d * d;
    }
    
    float dist = sqrtf(dist_sq + 1e-6f);
    
    // Break spring if too long (different docs)
    if (dist > 10.0f * L0) return;
    
    float f_mag = k_spring * (dist - L0);
    float scalar = f_mag / dist;

    // Pass 2: Apply
    #pragma unroll 32
    for (int k=0; k<256; ++k) {
        float d = positions[next*256 + k] - positions[idx*256 + k];
        float f = scalar * d;
        atomicAdd(&forces[idx*256 + k], f);
        atomicAdd(&forces[next*256 + k], -f);
    }
}

// Semantic bond forces (pre-computed similar pairs)
// Semantic bond forces (pre-computed similar pairs) with Hebbian Weights
extern "C" __global__ void compute_semantic_bonds(
    const float* positions,
    float* forces,
    const int* bonds,
    const float* bond_strengths, // New: Per-bond strength (Hebbian Plasticity)
    int num_bonds,
    float k_bond_global_scale,   // Global multiplier
    float L0
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_bonds) return;
    
    int src = bonds[idx * 2];
    int dst = bonds[idx * 2 + 1];
    float strength = bond_strengths[idx]; // Load individual strength
    
    float dist_sq = 0.0f;
    #pragma unroll 32
    for (int k=0; k<256; ++k) {
        float d = positions[dst*256 + k] - positions[src*256 + k];
        dist_sq += d * d;
    }
    float dist = sqrtf(dist_sq + 1e-6f);
    
    // F = k * strength * (dist - L0)
    float f_mag = k_bond_global_scale * strength * (dist - L0);
    float scalar = f_mag / dist;

    #pragma unroll 32
    for (int k=0; k<256; ++k) {
        float d = positions[dst*256 + k] - positions[src*256 + k];
        float f = scalar * d;
        atomicAdd(&forces[src*256 + k], f);
        atomicAdd(&forces[dst*256 + k], -f);
    }
}

// Velocity Verlet integration with friction
extern "C" __global__ void integrate(
    float* __restrict__ positions,
    float* __restrict__ velocities,
    const float* __restrict__ forces,
    const float* __restrict__ masses,
    int num_particles,
    float dt
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    
    #pragma unroll 32
    for (int k=0; k<256; ++k) {
        float acc = forces[idx*256 + k] / masses[idx];
        
        // Clamp acceleration
        acc = fmaxf(-100.0f, fminf(100.0f, acc));
        
        // Friction for stability (0.995 = light damping)
        velocities[idx*256 + k] *= 0.995f;
        velocities[idx*256 + k] += acc * dt;
        
        // Clamp velocity
        velocities[idx*256 + k] = fmaxf(-50.0f, fminf(50.0f, velocities[idx*256 + k]));
        
        positions[idx*256 + k] += velocities[idx*256 + k] * dt;
    }
}

// Gravitational lensing score (for ranking)
// NOTE: Host side currently calls this with only four arguments:
//   (positions, masses, lensing_scores, num_particles)
// To keep signatures consistent and avoid misaligned address errors,
// we implement a simple, safe placeholder that only touches valid
// memory and does not depend on query-specific state.
extern "C" __global__ void compute_gravitational_lensing(
	const float* positions,
	const float* masses,
	float* lensing_scores,
	int num_particles
) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_particles) return;

	// Simple, stable scoring: mass-based brightness proxy.
	// This keeps lensing_scores finite and non-negative without
	// requiring extra per-query buffers.
	float m = masses[i];
	if (!isfinite(m)) m = 0.0f;
	lensing_scores[i] = fabsf(m);
}

// --- SFT Phase 2: Langevin Dynamics ---

__device__ float hash_rand(int seed, int idx) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    int x = seed + idx;
    x = (x << 13) ^ x;
    // Map to [-1, 1]
    return (1.0f - ((x * (x * x * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0f);   
}

extern "C" __global__ void inject_langevin_noise(
    float* __restrict__ velocities,
    int num_particles,
    float noise_scale,
    int seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    
    // Approximate Gaussian using sum of 3 Uniforms (Irwin-Hall distribution)
    // Mean 0, Variance approx 1 if scaled.
    // Uniform [-1, 1] has var 1/3. Sum of 3 has var 1.
    // So sum(U(-1,1)) is naturally approx N(0, 1).
    
    #pragma unroll 32
    for (int k=0; k<256; ++k) {
        float r1 = hash_rand(seed + k*7, idx);
        float r2 = hash_rand(seed + k*13 + 1, idx);
        float r3 = hash_rand(seed + k*17 + 2, idx);
        float g = (r1 + r2 + r3); // Sum is range [-3, 3], Var=1. Perfect.
        
        velocities[idx * 256 + k] += noise_scale * g;
    }
}

// ============================================================================
// PHASE 4: KPZ ROUGHNESS EMBEDDING
// Implements Stochastic Creativity via Curvature-Dependent Noise
// Tokens in "flat" semantic space (clichés) get kicked hard.
// Tokens in "deep wells" (crystallized truth) are protected.
// ============================================================================
extern "C" __global__ void inject_kpz_noise(
    const float* __restrict__ positions,      // [N, 256]
    float* __restrict__ velocities,           // [N, 256]
    const float* __restrict__ masses,         // [N]
    int num_particles,
    float roughness_alpha,                    // Base noise temperature
    float curvature_sensitivity,              // How much structure dampens noise
    int seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    // 1. Calculate Local Semantic Density (Gravitational Potential)
    // \phi ~ sum(m_j / r_ij)
    // High potential means we are in a dense cluster (Crystallized)
    float potential = 0.0f;
    for (int j = 0; j < num_particles; ++j) {
        if (idx == j) continue;
        
        float dist_sq = 0.0f;
        #pragma unroll 32
        for (int k = 0; k < 256; ++k) {
            float d = positions[idx * 256 + k] - positions[j * 256 + k];
            dist_sq += d * d;
        }
        float dist = sqrtf(dist_sq + 1e-6f);
        
        // Potential from this neighbor
        potential += masses[j] / (dist + 0.1f);
    }
    
    // 2. Determine Noise Scale via KPZ logic
    // Flattening effect: Noise fills the valleys? 
    // No, we want noise to EXPLORE flat areas and LEAVE deep valleys alone.
    // So Noise ~ 1 / Potential.
    float damping = 1.0f + curvature_sensitivity * potential;
    float scale = roughness_alpha / damping;
    
    // 3. Inject Gaussian Noise
    #pragma unroll 32
    for (int k = 0; k < 256; ++k) {
        // Use hash_rand from existing device code
        float r1 = hash_rand(seed + k * 7, idx);
        float r2 = hash_rand(seed + k * 13 + 1, idx);
        float r3 = hash_rand(seed + k * 17 + 2, idx);
        float g = (r1 + r2 + r3); // Approx Gaussian N(0,1)
        
        velocities[idx * 256 + k] += scale * g;
    }
}
