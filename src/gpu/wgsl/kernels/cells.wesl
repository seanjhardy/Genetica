import package::constants::{CELL_FLAG_ACTIVE, POINT_FLAG_ACTIVE, M_PI};
import package::types::{Uniforms, Point, Cell, NutrientGrid, Link, DivisionRequest};
import package::utils::random::{rand_01};


@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> points: array<Point>;
@group(0) @binding(2) var<storage, read_write> physics_free_list: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> physics_counter: atomic<u32>;
@group(0) @binding(4) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(5) var<storage, read_write> cell_free_list: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> cell_counter: atomic<u32>;
@group(0) @binding(7) var<storage, read_write> nutrient_grid: NutrientGrid;
@group(0) @binding(8) var<storage, read_write> links: array<Link>;
@group(0) @binding(9) var<storage, read_write> link_free_list: array<atomic<u32>>;
@group(0) @binding(10) var<storage, read_write> points_counter: atomic<u32>;
@group(0) @binding(11) var<storage, read_write> division_requests: array<DivisionRequest>;
@group(0) @binding(12) var<storage, read_write> division_counter: atomic<u32>;

@group(0) @binding(13)
var<storage, read_write> link_corrections: array<atomic<i32>>;

@group(0) @binding(14)
var<uniform> step_counter: u32;

const DIVISION_CHANCE: f32 = 0.0001;

fn seed_from_u32(v: u32) -> vec2<u32> {
    return vec2<u32>(v * 1664525u + 1013904223u, v * 22695477u + 1u);
}

fn absorb_nutrients(cell_pos: vec2<f32>, radius: f32, absorption_rate: f32) -> f32 {
    let bounds_width = uniforms.bounds.z - uniforms.bounds.x;
    let bounds_height = uniforms.bounds.w - uniforms.bounds.y;
    let cell_size = f32(uniforms.nutrient.x);
    if bounds_width <= 0.0 || bounds_height <= 0.0 || cell_size <= 0.0 {
        return 0.0;
    }

    let grid_width_f = max(1.0, ceil(bounds_width / cell_size));
    let grid_height_f = max(1.0, ceil(bounds_height / cell_size));
    let grid_width = u32(grid_width_f);
    let grid_height = u32(grid_height_f);

    let local_x = clamp(cell_pos.x - uniforms.bounds.x, 0.0, bounds_width - 0.0001);
    let local_y = clamp(cell_pos.y - uniforms.bounds.y, 0.0, bounds_height - 0.0001);

    let gx = u32(clamp(floor(local_x / cell_size), 0.0, grid_width_f - 1.0));
    let gy = u32(clamp(floor(local_y / cell_size), 0.0, grid_height_f - 1.0));
    let grid_index = gy * grid_width + gx;

    if grid_index >= arrayLength(&nutrient_grid.values) {
        return 0.0;
    }

    let old_val = atomicLoad(&nutrient_grid.values[grid_index]);
    let available = f32(old_val) / f32(uniforms.nutrient.y);
    if available <= 0.0 {
        return 0.0;
    }
    let take = min(absorption_rate, available);
    let new_val = u32(max(available - take, 0.0) * f32(uniforms.nutrient.y));
    atomicStore(&nutrient_grid.values[grid_index], new_val);
    return take;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let dt = uniforms.sim_params.x;

    if idx >= arrayLength(&cells) {
        return;
    }

    var cell = cells[idx];
    if (cell.flags & CELL_FLAG_ACTIVE) == 0u {
        return;
    }

    if cell.point_idx >= arrayLength(&points) {
        return;
    }

    var point = points[cell.point_idx];
    
    if (point.flags & POINT_FLAG_ACTIVE) == 0u {
        return;
    }
    

    let seed = point.pos.x * 1000.0 + point.pos.y * 10.0 + cell.energy + f32(step_counter);


    //point.pos += rand_vec2(seed) * 0.1f;
    //point.prev_pos = point.pos;
    points[cell.point_idx] = point;


    // Simple metabolism and nutrient intake (with regeneration to keep cells alive)
    /*let metabolic_loss = (0.02 + 0.001 * point.radius) * dt;
    let nutrient_gain = absorb_nutrients(point.pos, point.radius, 0.05 * point.radius);
    cell.energy = max(cell.energy - metabolic_loss + nutrient_gain + 0.1, 0.0); // +0.1 regeneration

    if cell.energy <= 0.0 || point.radius <= 0.1 {
        //kill_cell(idx);
        return;
    }*/

    let division_seed = f32(idx) + seed + 91.0 + point.pos.x * 13.7 + point.pos.y * 7.3 + cell.energy;
    if rand_01(division_seed) < DIVISION_CHANCE { // 1% chance per cell per frame
        // Record division request for later processing by spawn_cells kernel
        let request_idx = atomicAdd(&division_counter, 1u);

        // Check if we have space for the request (should be plenty)
        if request_idx < 100000u {  // Match DIVISION_REQUEST_CAPACITY
            let division_angle = rand_01(division_seed + 91.0) * (M_PI * 2.0);

            let request = DivisionRequest(
                idx,  // parent_cell_idx
                cell.generation + 1u,  // generation
                15.0,  // energy
                division_angle  // angle
            );

            division_requests[request_idx] = request;
        }
    }

    cells[idx] = cell;
}
