// Render shader for drawing cells as quads
struct Cell {
    pos: vec2<f32>,
    prev_pos: vec2<f32>,
    energy: f32,
    cell_wall_thickness: f32,
    lifeform_idx: u32,
    random_force: vec2<f32>, // Random force vector that changes over time
}

// Uniforms struct must match Rust struct layout exactly (including padding)
struct Uniforms {
    delta_time: f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
    camera_pos: vec2<f32>,
    zoom: f32,
    point_radius: f32,
    bounds: vec4<f32>,
    view_size: vec2<f32>,
    cell_capacity: u32,
    free_cells_count: u32,
}


@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var<storage, read> cells: array<Cell>;

@group(0) @binding(2)
var<storage, read> cell_free_list: array<u32>;


struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) cell_index: f32,
    @location(1) uv: vec2<f32>,
    @location(2) energy: f32,
    @location(3) cell_wall_thickness: f32,
}

// Generate a quad for each cell using instanced rendering
@vertex
fn vs_main(@builtin(instance_index) instance_index: u32, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    
    // Get the cell data for this instance
    let cell_idx = instance_index;
    let quad_vertex = vertex_index;
    
    // Iterate through all cells up to capacity, but skip free cells
    if cell_idx >= uniforms.cell_capacity {
        // Discard if out of bounds
        out.clip_position = vec4<f32>(0.0, 0.0, -1.0, 1.0);
        return out;
    }
    
    for (var i: u32 = 0u; i < uniforms.free_cells_count; i++) {
        if cell_free_list[i] == cell_idx {
            out.clip_position = vec4<f32>(0.0, 0.0, -1.0, 1.0);
            return out;
        }
    }
        
    let cell = cells[cell_idx];
    
    // Transform cell from world space to clip space
    let world_pos = cell.pos;
    let relative_x = world_pos.x - uniforms.camera_pos.x;
    let relative_y = world_pos.y - uniforms.camera_pos.y;
    
    // Visible size in world units (what the camera sees)
    let view_size_x = uniforms.view_size.x / uniforms.zoom;
    let view_size_y = uniforms.view_size.y / uniforms.zoom;
    
    // Convert to clip space [-1, 1]
    let clip_x = (relative_x / view_size_x) * 2.0;
    let clip_y = -(relative_y / view_size_y) * 2.0;
    
    // DEBUG: Check if cell is way outside view (might indicate a problem)
    // Cells should be within reasonable bounds - discard if way off screen
    if abs(clip_x) > 10.0 || abs(clip_y) > 10.0 {
        // Cell is way off screen - skip rendering
        out.clip_position = vec4<f32>(0.0, 0.0, -1.0, 1.0);
        return out;
    }
    
    // Point size in world units (scales with zoom)
    let cell_radius_world = uniforms.point_radius;
    
    // Convert cell size from world units to clip space
    let cell_size_clip_x = (cell_radius_world / view_size_x) * 2.0;
    let cell_size_clip_y = (cell_radius_world / view_size_y) * 2.0;
    
    // Generate quad vertices for TriangleStrip
    var offset: vec2<f32>;
    var uv_offset: vec2<f32>;
    
    switch quad_vertex {
        case 0u {  // Bottom-left
            offset = vec2<f32>(-1.0, -1.0) * vec2<f32>(cell_size_clip_x, cell_size_clip_y);
            uv_offset = vec2<f32>(0.0, 1.0);
        }
        case 1u {  // Bottom-right
            offset = vec2<f32>(1.0, -1.0) * vec2<f32>(cell_size_clip_x, cell_size_clip_y);
            uv_offset = vec2<f32>(1.0, 1.0);
        }
        case 2u {  // Top-left
            offset = vec2<f32>(-1.0, 1.0) * vec2<f32>(cell_size_clip_x, cell_size_clip_y);
            uv_offset = vec2<f32>(0.0, 0.0);
        }
        default {  // Top-right
            offset = vec2<f32>(1.0, 1.0) * vec2<f32>(cell_size_clip_x, cell_size_clip_y);
            uv_offset = vec2<f32>(1.0, 0.0);
        }
    }
    
    out.clip_position = vec4<f32>(clip_x + offset.x, clip_y + offset.y, 0.0, 1.0);
    out.cell_index = f32(cell_idx);
    out.uv = uv_offset;
    out.energy = cell.energy;
    out.cell_wall_thickness = cell.cell_wall_thickness;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Create circular shape by discarding pixels outside the circle
    let center = vec2<f32>(0.5, 0.5);
    let dist = length(in.uv - center);
    if dist > 0.5 {
        discard;
    }
    
    // Color based on energy (subtle brightness variation)
    // Energy is normalized (assuming max 100.0 starting energy)
    let energy_normalized = clamp(in.energy / 100.0, 0.0, 1.0);
    
    // All cells should be similar bright cyan color
    // Use energy for subtle brightness variation only
    let brightness = 0.2 + energy_normalized * 0.8; // Range from 0.7 to 1.0
    let r = 0.0;
    let g = brightness;
    let b = brightness;
    let cell_color = vec4<f32>(r, g, b, 1.0);
    var final_color = cell_color;

    // Darken the border of the cell
    if dist > 0.5 - in.cell_wall_thickness {
        final_color = final_color * 0.4;
    }

    // nucleus is a smaller circle inside the cell slightly offset from the center
    let nucleus_radius = 0.15;
    let nucleus_x = 0.5 + 0.1 * sin(in.uv.x);
    let nucleus_y = 0.5 + 0.1 * cos(in.uv.y);
    let nucleus_dist = length(in.uv - vec2<f32>(nucleus_x, nucleus_y));
    if nucleus_dist < nucleus_radius {
        final_color = final_color * 0.8;
    }


    return final_color;
}
