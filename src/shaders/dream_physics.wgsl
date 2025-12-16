// Define constants to match Rust src/constants.rs
// VALENCE_LOCK_THRESHOLD = 9.5
// In WGSL we can't easily share constants directly without preprocessing, 
// but we define it here clearly.

const VALENCE_LOCK_THRESHOLD: f32 = 9.5;

struct Particle {
    pos: vec4<f32>, // xyz, w = mass/valence
    vel: vec4<f32>, // xyz, w = padding
}

@group(0) @binding(0) var<storage, read> particles_in: array<Particle>;
@group(0) @binding(1) var<storage, read_write> particles_out: array<Particle>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&particles_in)) { return; }

    var p = particles_in[index];
    var force = vec3<f32>(0.0);

    // Valence Lock: If valence (pos.w) > VALENCE_LOCK_THRESHOLD, this is an Immortal Core Memory.
    // It ignores all physics forces and stays anchored in truth.
    if (p.pos.w > VALENCE_LOCK_THRESHOLD) {
        particles_out[index] = p;
        return;
    }

    // Semantic Gravity: Pull towards center (0,0,0)
    force -= p.pos.xyz * 0.01; 

    // N-Body Repulsion (The "Gas" Law)
    // Naive O(N) per thread -> O(N^2) total. 
    // For 10k points, 10k*10k = 100M ops. 
    // Modern GPU handles this fine. For 1M points, we need shared memory tiling.
    let count = arrayLength(&particles_in);
    for (var i = 0u; i < count; i++) {
        if (i == index) { continue; }
        let other = particles_in[i];
        let diff = p.pos.xyz - other.pos.xyz;
        let dist_sq = dot(diff, diff);
        
        // Soft softening to avoid singularity
        if (dist_sq < 25.0 && dist_sq > 0.01) {
            force += normalize(diff) / dist_sq * 0.5;
        }
    }

    // Apply Verlet Integration
    let dt = 0.016;
    p.vel.x += force.x * dt;
    p.vel.y += force.y * dt;
    p.vel.z += force.z * dt;
    
    // Dampening (Entropy)
    p.vel.x *= 0.98;
    p.vel.y *= 0.98;
    p.vel.z *= 0.98;

    p.pos.x += p.vel.x;
    p.pos.y += p.vel.y;
    p.pos.z += p.vel.z;

    particles_out[index] = p;
}
