struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct InstanceInput {
    @location(1) position: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) speed: f32,
    @location(4) mass: f32,
    @location(5) charge: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) speed: f32,
    @location(3) mass: f32,
    @location(4) charge: f32,
};

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    var out: VertexOutput;
    
    // Billboard logic: assume camera looks down Z for now, or use view matrix to align.
    // For simplicity in this "God View", we just add the model pos to instance pos.
    // Model pos is a quad [-0.5, 0.5].
    
    // Scale particle
    let scale = 0.1; 
    let world_position = instance.position + model.position * scale;
    
    out.clip_position = camera.view_proj * vec4<f32>(world_position, 1.0);
    out.color = instance.color;
    out.uv = model.position.xy + 0.5; // Map [-0.5, 0.5] to [0, 1]
    out.speed = instance.speed;
    out.mass = instance.mass;
    out.charge = instance.charge;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Radial Gradient for "Glowing" effect
    let dist = distance(in.uv, vec2<f32>(0.5, 0.5));
    let alpha = 1.0 - smoothstep(0.0, 0.5, dist);
    
    // Add a hot core
    let core = 1.0 - smoothstep(0.0, 0.1, dist);
    
    // Velocity-driven glow: faster particles get a warmer, brighter halo
    let s = clamp(in.speed / 10.0, 0.0, 1.0);
    let warm = mix(in.color, vec3<f32>(1.0, 0.9, 0.3), s);
    let glow = warm * (0.6 + 0.4 * s);
    
    // Mass-driven brightness (rarer / heavier terms glow more)
    let mass_brightness = clamp(in.mass / 20.0, 0.0, 2.0);
    let size_glow = mass_brightness * 0.8;

    // Charge-driven tint: blue (neutral) → red (high charge)
    let charge_clamped = clamp(in.charge, 0.0, 1.0);
    let charge_tint = mix(vec3<f32>(0.3, 0.6, 1.0), vec3<f32>(1.0, 0.4, 0.4), charge_clamped);

    // Final color: speed glow * mass + white core scaled by mass
    let final_color =
        (glow * (0.7 + size_glow)) * charge_tint + vec3<f32>(1.0, 1.0, 1.0) * core * mass_brightness;
    
    if (alpha < 0.01) {
        discard;
    }
    
    return vec4<f32>(final_color, alpha);
}
