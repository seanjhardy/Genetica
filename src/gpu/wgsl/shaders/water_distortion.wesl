import package::utils::distortion::{water_distortion};
// Water distortion composite shader
// Applies wave-like distortion to the rendered scene in world space

struct WaterUniform {
    camera_pos_zoom: vec4<f32>,      // camera_x, camera_y, zoom, _padding
    view_size: vec4<f32>,            // view_width, view_height, _padding, _padding
    bounds: vec4<f32>,               // left, top, right, bottom
    time_params: vec4<f32>,          // time, _padding, _padding, _padding
}

@group(0) @binding(0)
var scene_texture: texture_2d<f32>;

@group(0) @binding(1)
var tex_sampler: sampler;

@group(0) @binding(2)
var<uniform> uniforms: WaterUniform;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) world_pos: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    var pos: vec2<f32>;
    switch vertex_index {
        case 0u: { pos = vec2<f32>(-1.0, -1.0); }
        case 1u: { pos = vec2<f32>(1.0, -1.0); }
        case 2u: { pos = vec2<f32>(-1.0, 1.0); }
        default: { pos = vec2<f32>(1.0, 1.0); }
    }
    
    out.clip_position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = vec2<f32>(
        (pos.x + 1.0) * 0.5,
        1.0 - (pos.y + 1.0) * 0.5
    );
    
    // Convert clip space to world space for distortion calculation
    let camera_pos = uniforms.camera_pos_zoom.xy;
    let zoom = uniforms.camera_pos_zoom.z;
    let view_size = uniforms.view_size.xy;
    
    let visible_width = view_size.x / zoom;
    let visible_height = view_size.y / zoom;
    
    let world_x = camera_pos.x + pos.x * visible_width * 0.5;
    let world_y = camera_pos.y - pos.y * visible_height * 0.5;
    
    out.world_pos = vec2<f32>(world_x, world_y);
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let bounds = uniforms.bounds;
    
    // Check if we're inside the simulation bounds
    let inside_bounds = in.world_pos.x >= bounds.x && in.world_pos.x <= bounds.z 
                     && in.world_pos.y >= bounds.y && in.world_pos.y <= bounds.w;
    
    var sample_uv = in.uv;
    
    if inside_bounds {
        // Apply water distortion to world position
        let time = uniforms.time_params.x;
        let distortion = water_distortion(in.world_pos, time);
        
        // Convert world-space distortion to screen-space UV offset
        let camera_pos = uniforms.camera_pos_zoom.xy;
        let zoom = uniforms.camera_pos_zoom.z;
        let view_size = uniforms.view_size.xy;
        
        let visible_width = view_size.x / zoom;
        let visible_height = view_size.y / zoom;
        
        let uv_offset_x = distortion.x / (visible_width * 0.5);
        let uv_offset_y = -distortion.y / (visible_height * 0.5);
        
        sample_uv = in.uv + vec2<f32>(uv_offset_x, uv_offset_y);
    }
    
    // Sample the scene texture with distorted UVs
    return textureSample(scene_texture, tex_sampler, sample_uv);
}
