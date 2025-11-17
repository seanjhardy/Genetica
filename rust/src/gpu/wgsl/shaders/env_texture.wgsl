// Shader for rendering planet texture to viewport (camera-aware)

struct PlanetUniform {
    camera_pos_zoom_thickness: vec4<f32>,   // camera_x, camera_y, zoom, line_thickness_px
    view_size_grid: vec4<f32>,              // view_width, view_height, grid_spacing_px, grid_opacity
    grid_threshold_padding: vec4<f32>,      // grid_zoom_threshold, padding...
    border_color: vec4<f32>,
    bounds: vec4<f32>,                      // left, top, right, bottom
    padding: vec4<f32>,
}

@group(0) @binding(0)
var planet_texture: texture_2d<f32>;

@group(0) @binding(1)
var planet_sampler: sampler;

@group(0) @binding(2)
var<uniform> uniform_data: PlanetUniform;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) world_pos: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    
    // Generate quad corners in world space based on simulation bounds
    var world_pos: vec2<f32>;
    var uv: vec2<f32>;
    
    switch vertex_index {
        case 0u: {  // Bottom-left
            world_pos = vec2<f32>(uniform_data.bounds.x, uniform_data.bounds.w);
            uv = vec2<f32>(0.0, 1.0);
        }
        case 1u: {  // Bottom-right
            world_pos = vec2<f32>(uniform_data.bounds.z, uniform_data.bounds.w);
            uv = vec2<f32>(1.0, 1.0);
        }
        case 2u: {  // Top-left
            world_pos = vec2<f32>(uniform_data.bounds.x, uniform_data.bounds.y);
            uv = vec2<f32>(0.0, 0.0);
        }
        default: {  // Top-right
            world_pos = vec2<f32>(uniform_data.bounds.z, uniform_data.bounds.y);
            uv = vec2<f32>(1.0, 0.0);
        }
    }
    
    // Transform from world space to clip space (same as bounds renderer)
    let camera_pos = uniform_data.camera_pos_zoom_thickness.xy;
    let zoom = uniform_data.camera_pos_zoom_thickness.z;
    let line_thickness_world = uniform_data.camera_pos_zoom_thickness.w;
    let view_size = uniform_data.view_size_grid.xy;
    let grid_spacing_world = uniform_data.view_size_grid.z;
    let grid_opacity = uniform_data.view_size_grid.w;
    let grid_zoom_threshold = uniform_data.grid_threshold_padding.x;

    let visible_width = view_size.x / zoom;
    let visible_height = view_size.y / zoom;
    
    let relative_x = world_pos.x - camera_pos.x;
    let relative_y = world_pos.y - camera_pos.y;
    
    let clip_x = (relative_x / visible_width) * 2.0;
    let clip_y = -(relative_y / visible_height) * 2.0;  // Flip Y
    
    out.clip_position = vec4<f32>(clip_x, clip_y, 0.0, 1.0);
    out.uv = uv;
    out.world_pos = world_pos;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var color = textureSample(planet_texture, planet_sampler, in.uv);

    let zoom = uniform_data.camera_pos_zoom_thickness.z;
    let line_thickness_world = uniform_data.camera_pos_zoom_thickness.w;
    let grid_spacing_world = uniform_data.view_size_grid.z;
    let grid_opacity = uniform_data.view_size_grid.w;
    let grid_zoom_threshold = uniform_data.grid_threshold_padding.x;

    let thickness_world = line_thickness_world;
    let spacing_world = grid_spacing_world;

    // Grid overlay
    if zoom > grid_zoom_threshold && spacing_world > 0.0 {
        let local_x = in.world_pos.x - uniform_data.bounds.x;
        let local_y = in.world_pos.y - uniform_data.bounds.y;

        let frac_x = fract(local_x / spacing_world);
        let frac_y = fract(local_y / spacing_world);
        let dist_x = min(frac_x, 1.0 - frac_x) * spacing_world;
        let dist_y = min(frac_y, 1.0 - frac_y) * spacing_world;

        if dist_x <= thickness_world * 0.5 || dist_y <= thickness_world * 0.5 {
            let blended_rgb =
                mix(color.rgb, vec3<f32>(0.0, 0.0, 0.0), grid_opacity);
            color = vec4<f32>(blended_rgb, color.a);
        }
    }

    // Border overlay (world thickness derived from pixel thickness)
    let left_edge = abs(in.world_pos.x - uniform_data.bounds.x) <= thickness_world;
    let right_edge = abs(uniform_data.bounds.z - in.world_pos.x) <= thickness_world;
    let top_edge = abs(in.world_pos.y - uniform_data.bounds.y) <= thickness_world;
    let bottom_edge = abs(uniform_data.bounds.w - in.world_pos.y) <= thickness_world;

    if left_edge || right_edge || top_edge || bottom_edge {
        color = uniform_data.border_color.rgba;
    }

    // Clamp to viewport rectangle to avoid sampling outside due to precision
    if in.world_pos.x < uniform_data.bounds.x
        || in.world_pos.x > uniform_data.bounds.z
        || in.world_pos.y < uniform_data.bounds.y
        || in.world_pos.y > uniform_data.bounds.w {
        discard;
    }

    return color;
}

