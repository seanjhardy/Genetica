// Render shader for drawing cells as quads
struct Cell {
    pos: vec2<f32>,
    prev_pos: vec2<f32>,
    radius: f32,
    energy: f32,
    cell_wall_thickness: f32,
    lifeform_idx: u32,
    random_force: vec2<f32>, // Random force vector that changes over time
}

// Uniforms struct must match Rust struct layout exactly (including padding)
struct Uniforms {
    sim_params: vec4<f32>, // x: dt, y: zoom, z: view_w, w: view_h
    cell_count: vec4<f32>, // x: cell_count, y: reserved0, z: reserved1, w: reserved2
    camera: vec4<f32>,     // x: cam_x, y: cam_y
    bounds: vec4<f32>,     // (left, top, right, bottom)
};


@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var<storage, read> cells: array<Cell>;

struct CellFreeList {
    count: u32,
    indices: array<u32>,
}

@group(0) @binding(2)
var<storage, read> cell_free_list: CellFreeList;


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
    
    let free_cells_count = cell_free_list.count;
    for (var i: u32 = 0u; i < free_cells_count; i++) {
        if cell_free_list.indices[i] == cell_idx {
            out.clip_position = vec4<f32>(0.0, 0.0, -1.0, 1.0);
            return out;
        }
    }
        
    let cell = cells[cell_idx];
    
    // Transform cell from world space to clip space
    let world_pos = cell.pos;
    let relative_x = world_pos.x - uniforms.camera.x;
    let relative_y = world_pos.y - uniforms.camera.y;
    
    // Visible size in world units (what the camera sees)
    let view_size_x = uniforms.sim_params.z / uniforms.sim_params.y;
    let view_size_y = uniforms.sim_params.w / uniforms.sim_params.y;
    
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
    let cell_radius_world = cell.radius;
    
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
    let angle_from_center = atan2(in.uv.y - center.y, in.uv.x - center.x);
    let radius = 0.49 - sin(angle_from_center * 10.0) * 0.01;
    if dist > radius {
        discard;
    }
    
    // Color based on energy (subtle brightness variation)
    // Energy is normalized (assuming max 100.0 starting energy)
    let energy_normalized = clamp(in.energy / 100.0, 0.0, 1.0);
    
    // All cells should be similar bright cyan color
    // Use energy for subtle brightness variation only
    let brightness = 0.1 + energy_normalized * 0.9; // Range from 0.7 to 1.0
    let r = (1 - brightness) * 0.5;
    let g = brightness;
    let b = brightness;
    let cell_color = vec4<f32>(r, g, b, 1.0);
    var final_color = cell_color;

    // Darken the border of the cell
    if dist > radius - in.cell_wall_thickness {
        final_color = final_color * 0.2;
    }

    return final_color;
}
