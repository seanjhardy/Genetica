// Render shader for nutrient overlay grid
struct Uniforms {
    sim_params: vec4<f32>, // x: dt, y: zoom, z: view_w, w: view_h
    cell_count: vec4<f32>,
    camera: vec4<f32>,
    bounds: vec4<f32>,
    nutrient: vec4<u32>,// (Cell size, scale, reserved, reserved)
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct NutrientGrid {
    values: array<u32>,
}

@group(0) @binding(1)
var<storage, read> nutrient_grid: NutrientGrid;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) nutrient_value: f32,
}


const NUTRIENT_CELL_SIZE: f32 = 20.0;


@vertex
fn vs_main(@builtin(instance_index) instance_index: u32, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let total_cells = arrayLength(&nutrient_grid.values);
    if instance_index >= total_cells {
        out.clip_position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.nutrient_value = 0.0;
        return out;
    }

    let bounds_width = uniforms.bounds.z - uniforms.bounds.x;
    let bounds_height = uniforms.bounds.w - uniforms.bounds.y;
    if bounds_width <= 0.0 || bounds_height <= 0.0 {
        out.clip_position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.nutrient_value = 0.0;
        return out;
    }

    let view_size_x = uniforms.sim_params.z / uniforms.sim_params.y;
    let view_size_y = uniforms.sim_params.w / uniforms.sim_params.y;

    let grid_width_f = max(1.0, ceil(bounds_width / NUTRIENT_CELL_SIZE));
    let grid_width = u32(grid_width_f);
    let gx = instance_index % grid_width;
    let gy = instance_index / grid_width;

    let cell_left = uniforms.bounds.x + f32(gx) * NUTRIENT_CELL_SIZE;
    let cell_top = uniforms.bounds.y + f32(gy) * NUTRIENT_CELL_SIZE;
    let cell_center = vec2<f32>(
        cell_left + NUTRIENT_CELL_SIZE * 0.5,
        cell_top + NUTRIENT_CELL_SIZE * 0.5,
    );

    let relative_x = cell_center.x - uniforms.camera.x;
    let relative_y = cell_center.y - uniforms.camera.y;

    let clip_x = (relative_x / view_size_x) * 2.0;
    let clip_y = -(relative_y / view_size_y) * 2.0;

    let half_size = NUTRIENT_CELL_SIZE * 0.5;
    let size_clip_x = (half_size / view_size_x) * 2.0;
    let size_clip_y = (half_size / view_size_y) * 2.0;

    var offset: vec2<f32>;
    switch vertex_index {
        case 0u {
            offset = vec2<f32>(-size_clip_x, -size_clip_y);
        }
        case 1u {
            offset = vec2<f32>(size_clip_x, -size_clip_y);
        }
        case 2u {
            offset = vec2<f32>(-size_clip_x, size_clip_y);
        }
        default {
            offset = vec2<f32>(size_clip_x, size_clip_y);
        }
    }

    out.clip_position = vec4<f32>(clip_x + offset.x, clip_y + offset.y, 0.0, 1.0);
    out.nutrient_value = f32(nutrient_grid.values[instance_index]);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let intensity = clamp(in.nutrient_value / f32(uniforms.nutrient.y), 0.0, 1.0);
    let alpha = intensity * 0.01;
    let color = vec3<f32>(0.0, intensity, 0.0);
    return vec4<f32>(color, alpha);
}

