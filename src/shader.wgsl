// Camera Uniform
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_pos: vec4<f32>, // Added view position for billboarding if needed
    view_right: vec4<f32>, // Camera Right Vector
    view_up: vec4<f32>,    // Camera Up Vector
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct InstanceInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) radius: f32, // Added radius
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) v_index: u32,
    instance: InstanceInput
) -> VertexOutput {
    var out: VertexOutput;
    out.color = instance.color;

    // Hardcoded Quad Vertices (Triangle Strip order or Indexed)
    // 0: -1, -1
    // 1:  1, -1
    // 2: -1,  1
    // 3:  1,  1
    var pos = vec2<f32>(0.0, 0.0);
    var uv = vec2<f32>(0.0, 0.0);
    
    // Simple 4-vertex quad (index 0-3) using TriangleStrip topology or (0-5) TriangleList
    // Let's assume we draw 6 vertices per instance with TriangleList
    // Indices: 0, 1, 2,  2, 1, 3 (Standard Quad)
    // Vertices: (-1,-1), (1,-1), (-1,1), (1,1)
    
    // We can use a switch on vertex_index % 6
    let idx = v_index % 6u;
    
    // 0,1,2 -> Tri 1
    // 3,4,5 -> Tri 2 (which is 2,1,3)
    
    var corner = vec2<f32>(0.0);
    if (idx == 0u) { corner = vec2<f32>(-1.0, -1.0); }
    else if (idx == 1u) { corner = vec2<f32>(1.0, -1.0); }
    else if (idx == 2u) { corner = vec2<f32>(-1.0, 1.0); }
    else if (idx == 3u) { corner = vec2<f32>(-1.0, 1.0); } // Duplicate of 2
    else if (idx == 4u) { corner = vec2<f32>(1.0, -1.0); } // Duplicate of 1
    else if (idx == 5u) { corner = vec2<f32>(1.0, 1.0); }
    
    out.uv = corner;
    
    // Billboarding:
    // World Pos = InstancePos + CameraRight * x * r + CameraUp * y * r
    let right = vec3<f32>(camera.view_proj[0][0], camera.view_proj[1][0], camera.view_proj[2][0]); // Approx right from matrix? 
    // Actually, simpler to just use camera basis vectors if passed. 
    // Or just View-Aligned Quad in Clip Space?
    // Let's use View-Aligned in World Space.
    
    // Better: We need the camera Up and Right vectors to perfect billboard.
    // For now, let's just create 3D cubes/sprites aligned to axes? 
    // No, circles need to face camera.
    // Let's extract Right/Up from View Matrix.
    // View Matrix is Inverse Camera Transform.
    // Rows 0, 1, 2 of View Matrix are Right, Up, Look vectors?
    // Let's rely on updated CameraUniform or just use simple screen-space sizing (PointSprite style).
    
    // PointSprite Style:
    // Project Center to Clip.
    // Add offset in Clip Space.
    let center_clip = camera.view_proj * vec4<f32>(instance.position, 1.0);
    
    // Offset in pixels/NDC?
    // variance is spatial size. so we should scale in World Space.
    // We need billboard vectors.
    // Hack: Assume camera is mostly looking -Z, Up is +Y.
    // Just drawing flat quads on XY plane is 2D.
    // drawing spheres?
    
    // Let's assume we updated CameraUniform to send Right/Up vectors.
    // Fallback: Just draw XY quads (flat discs) for now to verify.
    // It will look 2D but show size.
    let local_pos = vec3<f32>(corner * instance.radius, 0.0);
    let world_pos = instance.position + local_pos;
    
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Circle SDF
    let dist = length(in.uv);
    if (dist > 1.0) {
        discard;
    }
    // Soft edge
    let alpha = 1.0 - smoothstep(0.8, 1.0, dist);
    
    return vec4<f32>(in.color, alpha); // Opacity implicitly 1.0 for now, combined with soft edge
}
