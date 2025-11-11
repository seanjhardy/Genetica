// Compute shader for regenerating nutrient grid values
struct Uniforms {
    sim_params: vec4<f32>, // x: dt, y: zoom, z: view_w, w: view_h
    cell_count: vec4<f32>,
    camera: vec4<f32>,
    bounds: vec4<f32>,
    nutrient: vec4<u32>,// (Cell size, scale, grid_width, grid_height)
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct NutrientGrid {
    values: array<atomic<u32>>,
}

@group(0) @binding(1)
var<storage, read_write> nutrient_grid: NutrientGrid;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let dt = uniforms.sim_params.x;
    let nutrient_scale = uniforms.nutrient.y;
    let grid_width = uniforms.nutrient.z;
    let grid_height = uniforms.nutrient.w;

    let total_cells = arrayLength(&nutrient_grid.values);
    if index >= total_cells {
        return;
    }

    if grid_width == 0u || grid_height == 0u {
        return;
    }

    let current = f32(atomicLoad(&nutrient_grid.values[index])) / f32(nutrient_scale);
    if current >= 1.0 {
        return;
    }

    let width_i = i32(grid_width);
    let height_i = i32(grid_height);
    let x = i32(index % grid_width);
    let y = i32(index / grid_width);

    var neighbour_nutrient_sum = 0.0;
    var neighbour_count = 0.0;

    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            if dx == 0 && dy == 0 {
                continue;
            }
            let nx = x + dx;
            let ny = y + dy;
            if nx < 0 || nx >= width_i || ny < 0 || ny >= height_i {
                continue;
            }
            let neighbour_index = u32(ny) * grid_width + u32(nx);
            if neighbour_index < total_cells {
                let neighbour_nutrient =
                    f32(atomicLoad(&nutrient_grid.values[neighbour_index])) / f32(nutrient_scale);
                neighbour_nutrient_sum += neighbour_nutrient;
                neighbour_count += 1.0;
            }
        }
    }

    var neighbour_average = current;
    if neighbour_count > 0.0 {
        neighbour_average = neighbour_nutrient_sum / neighbour_count;
    }
    let neighbour_bonus = max(neighbour_average - current, 0.0);

    let nutrient_growth = (0.0001 + 0.00005 * current + 0.001 * neighbour_bonus) * dt;
    let new_nutrient_level = min(current + nutrient_growth, 1.0);

    atomicStore(&nutrient_grid.values[index], u32(new_nutrient_level * f32(nutrient_scale)));
}


