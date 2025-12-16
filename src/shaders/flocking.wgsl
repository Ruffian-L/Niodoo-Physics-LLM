// Flocking Compute Shader for "The Dolphins"
// Implements Reynolds Boids with Semantic Tethering and Valence Repulsion

struct SplatMotion {
    velocity: vec3<f32>,
    covariance_det: f32,
    time_birth: f32,
    time_death: f32,
}

struct SplatGeometry {
    position: vec3<f32>,
    scale: vec3<f32>,
    rotation: vec4<f32>,
    color: u32,
    physics: u32,
}

@group(0) @binding(0) var<storage, read_write> geometries: array<SplatGeometry>;
@group(0) @binding(1) var<storage, read_write> motions: array<SplatMotion>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    dt: f32,
    num_boids: u32,
    separation_radius: f32,
    alignment_radius: f32,
    cohesion_radius: f32,
    tether_strength: f32,
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.num_boids) {
        return;
    }

    var pos = geometries[index].position;
    var vel = motions[index].velocity;
    
    // 1. Boid Rules (Simplified O(N) via spatial hash or just naive O(N^2) for small batches)
    // For 1M splats, we CANNOT do O(N^2).
    // We rely on "Semantic Tethering" (pull to origin) + Local Noise for "Living" feel
    // rather than full global flocking, unless we have a spatial grid buffer.
    
    // Semantic Tethering: Pull back to original position (stored where? Assuming 'pos' is current)
    // We need a 'home' position. Let's assume 'scale' stores the home or we just drift.
    // Or we assume the 'position' in SplatGeometry is the state, and we want to hover around a target.
    
    // Valence Repulsion:
    // If physics.valence (in physics packed u32) is negative (Trauma), push away.
    
    let valence_byte = (geometries[index].physics >> 8u) & 0xFFu;
    let valence = f32(valence_byte) - 127.0;
    
    var force = vec3<f32>(0.0);
    
    // A. Tether Force (keep it from flying away)
    // Assuming 'home' is implicit or we just damp velocity.
    // Let's add a random "Brownian" force for "breathing"
    
    // Pseudo-random based on index and time (params.dt accumulation needed ideally)
    let seed = f32(index) * 0.123 + params.dt; 
    let noise = vec3<f32>(sin(seed), cos(seed * 1.3), sin(seed * 0.7));
    
    force += noise * 0.1;

    // B. Valence "Expansion"
    if (valence < -20.0) {
        // Trauma expands
        force += normalize(pos) * 0.5; 
    } else if (valence > 20.0) {
        // Joy contracts/stabilizes
        force -= vel * 0.5;
    }

    // Integration
    vel += force * params.dt;
    pos += vel * params.dt;
    
    // Damping
    vel *= 0.98;

    // Update buffers
    geometries[index].position = pos;
    motions[index].velocity = vel;
}




