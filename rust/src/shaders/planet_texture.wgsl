// Shader for rendering planet texture to viewport (camera-aware)

struct PlanetTransform {
    camera_pos: vec2<f32>,
    zoom: f32,
    _padding1: f32,
    view_size: vec2<f32>,
    bounds: vec4<f32>,  // left, top, right, bottom
}

@group(0) @binding(0)
var planet_texture: texture_2d<f32>;

@group(0) @binding(1)
var planet_sampler: sampler;

@group(0) @binding(2)
var<uniform> transform: PlanetTransform;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    
    // Generate quad corners in world space based on simulation bounds
    var world_pos: vec2<f32>;
    var uv: vec2<f32>;
    
    switch vertex_index {
        case 0u: {  // Bottom-left
            world_pos = vec2<f32>(transform.bounds.x, transform.bounds.w);
            uv = vec2<f32>(0.0, 1.0);
        }
        case 1u: {  // Bottom-right
            world_pos = vec2<f32>(transform.bounds.z, transform.bounds.w);
            uv = vec2<f32>(1.0, 1.0);
        }
        case 2u: {  // Top-left
            world_pos = vec2<f32>(transform.bounds.x, transform.bounds.y);
            uv = vec2<f32>(0.0, 0.0);
        }
        default: {  // Top-right
            world_pos = vec2<f32>(transform.bounds.z, transform.bounds.y);
            uv = vec2<f32>(1.0, 0.0);
        }
    }
    
    // Transform from world space to clip space (same as bounds renderer)
    let visible_width = transform.view_size.x / transform.zoom;
    let visible_height = transform.view_size.y / transform.zoom;
    
    let relative_x = world_pos.x - transform.camera_pos.x;
    let relative_y = world_pos.y - transform.camera_pos.y;
    
    let clip_x = (relative_x / visible_width) * 2.0;
    let clip_y = -(relative_y / visible_height) * 2.0;  // Flip Y
    
    out.clip_position = vec4<f32>(clip_x, clip_y, 0.0, 1.0);
    out.uv = uv;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(planet_texture, planet_sampler, in.uv);
}

