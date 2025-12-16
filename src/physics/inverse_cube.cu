// inverse_cube.cu
// Implementation of the Principia Cybernetica Antigravity Law

#include <cuda_runtime.h>

// Epsilon to prevent division by zero singularities when r -> 0
#define SOFTENING_EPSILON 1e-6f

extern "C" __global__ void inverse_cube_force_kernel(
    const float* __restrict__ positions, // Input: [N, 3] (x, y, z)
    const float* __restrict__ charges,   // Input: [N, 3] (valence, arousal, dominance)
    float* __restrict__ forces,          // Output: [N, 3] (fx, fy, fz)
    const int num_particles,
    const float G_sem                    // Semantic Gravitational Constant
) {
    // Determine the particle 'i' this thread is responsible for
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_particles) return;

    // Load attributes of particle i into registers (high speed memory)
    float3 pos_i = make_float3(positions[i*3], positions[i*3+1], positions[i*3+2]);
    float3 charge_i = make_float3(charges[i*3], charges[i*3+1], charges[i*3+2]);

    // Accumulator for the net force
    float3 net_force = make_float3(0.0f, 0.0f, 0.0f);

    // Loop over all other particles 'j'
    // Optimization Note: For N > 10,000, this loop should be tiled using shared memory
    // to reduce global memory bandwidth pressure. This is a baseline implementation.
    for (int j = 0; j < num_particles; ++j) {
        if (i == j) continue; // Do not calculate self-force

        float3 pos_j = make_float3(positions[j*3], positions[j*3+1], positions[j*3+2]);
        float3 charge_j = make_float3(charges[j*3], charges[j*3+1], charges[j*3+2]);

        // Displacement vector r_ij = pos_i - pos_j (Repulsive direction)
        float3 r_vec;
        r_vec.x = pos_i.x - pos_j.x;
        r_vec.y = pos_i.y - pos_j.y;
        r_vec.z = pos_i.z - pos_j.z;

        // Distance Squared: r^2
        float dist_sq = r_vec.x*r_vec.x + r_vec.y*r_vec.y + r_vec.z*r_vec.z;
        
        // Apply Softening to prevent singularity
        dist_sq += SOFTENING_EPSILON;

        // Calculate Inverse Cube Factor
        // We need F ~ 1/r^3 * (r_vec/r) = r_vec / r^4
        // r^4 = (dist_sq)^2
        float dist_fourth = dist_sq * dist_sq;
        float inv_dist_fourth = 1.0f / dist_fourth;

        // Interaction Strength (Charge Dot Product)
        // Physics Logic: 
        // If charges are aligned (dot > 0) -> Repulsion (Force in direction of r_vec)
        // If charges are opposed (dot < 0) -> Attraction (Force opposite to r_vec)
        float charge_interaction = charge_i.x * charge_j.x + 
                                   charge_i.y * charge_j.y + 
                                   charge_i.z * charge_j.z;

        float scalar_magnitude = G_sem * charge_interaction * inv_dist_fourth;

        // Accumulate Vector Force
        net_force.x += scalar_magnitude * r_vec.x;
        net_force.y += scalar_magnitude * r_vec.y;
        net_force.z += scalar_magnitude * r_vec.z;
    }

    // Write result to global memory
    forces[i*3]     = net_force.x;
    forces[i*3 + 1] = net_force.y;
    forces[i*3 + 2] = net_force.z;
}
