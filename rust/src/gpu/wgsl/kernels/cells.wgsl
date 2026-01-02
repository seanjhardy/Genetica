@include src/gpu/wgsl/constants.wgsl;
@include src/gpu/wgsl/types.wgsl;
@include src/gpu/wgsl/utils/color.wgsl;
@include src/gpu/wgsl/utils/random.wgsl;

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var<storage, read_write> points: array<VerletPoint>;

@group(0) @binding(2)
var<storage, read_write> physics_free_list: FreeList;

@group(0) @binding(3)
var<storage, read_write> physics_counter: atomic<u32>;

@group(0) @binding(4)
var<storage, read_write> cells: array<Cell>;

@group(0) @binding(5)
var<storage, read_write> cell_free_list: FreeList;

@group(0) @binding(6)
var<storage, read_write> cell_counter: atomic<u32>;

@group(0) @binding(7)
var<storage, read_write> spawn_buffer: SpawnBuffer;

@group(0) @binding(8)
var<storage, read_write> nutrient_grid: NutrientGrid;

@group(0) @binding(9)
var<storage, read_write> links: array<Link>;

@group(0) @binding(10)
var<storage, read_write> link_free_list: FreeList;

@group(0) @binding(11)
var<storage, read_write> lifeforms: array<Lifeform>;

@group(0) @binding(12)
var<storage, read_write> lifeform_free: FreeList;

@group(0) @binding(13)
var<storage, read_write> lifeform_counter: atomic<u32>;

@group(0) @binding(14)
var<storage, read_write> species_entries: array<Species>;

@group(0) @binding(15)
var<storage, read_write> species_free: FreeList;

@group(0) @binding(16)
var<storage, read_write> species_counter: atomic<u32>;

@group(0) @binding(17)
var<storage, read_write> position_changes: array<PositionChangeEntry>;

fn compute_cell_color(radius: f32, energy: f32) -> vec4<f32> {
    let energy_normalized = clamp(energy / (radius * 50.0), 0.0, 1.0);
    let color_vec4 = mix(vec4<f32>(46.0, 133.0, 48.0, 255.0) / 255.0, 
    vec4<f32>(46.0, 133.0, 48.0, 255.0) / 255.0, energy_normalized);
    return srgb(color_vec4);
}

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

fn spawn_cells() {
    /*loop {
        let pending = atomicLoad(&spawn_buffer.counter);
        if pending == 0u {
            break;
        }
        let desired = pending - 1u;
        let exchange = atomicCompareExchangeWeak(&spawn_buffer.counter, pending, desired);
        if !exchange.exchanged {
            continue;
        }
        let request_index = desired;
        if request_index >= arrayLength(&spawn_buffer.requests) {
            continue;
        }

        let req = spawn_buffer.requests[request_index];
        let cell_slot = alloc_cell_slot();
        let phys_slot = alloc_physics_slot();
        if cell_slot < 0 || phys_slot < 0 {
            // Put the counter back if we failed to allocate
            atomicAdd(&spawn_buffer.counter, 1u);
            if cell_slot >= 0 {
                free_cell_slot(u32(cell_slot));
            }
            if phys_slot >= 0 {
                free_physics_slot(u32(phys_slot));
            }
            break;
        }

        // Physics setup
        var phys = VerletPoint(
            req.pos,
            req.pos,
            vec2<f32>(0.0),
            max(req.radius, 1.0),
            1u,
        );
        points[u32(phys_slot)] = phys;
        atomicAdd(&physics_counter, 1u);

        // Cell setup
        var cell = Cell(
            u32(phys_slot),       // point_idx
            req.lifeform_id,      // lifeform_id
            0u,                   // generation
            req.radius,           // radius
            req.energy,           // energy
            1.0,                  // cell_wall_thickness
            compute_cell_color(req.radius, req.energy),
            1u,                   // flags
            array<u32, 20>(), // noise_permutations
            vec2<f32>(0.0),       // noise_texture_offset
        );
        cells[u32(cell_slot)] = cell;
        atomicAdd(&cell_counter, 1u);

        if req.lifeform_id < arrayLength(&lifeforms) {
            atomicAdd(&lifeforms[req.lifeform_id].cell_count, 1u);
        }
    }*/
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

    var point = points[cell.point_idx];
    
    if (point.flags & POINT_FLAG_ACTIVE) == 0u {
        return;
    }

    let seed = point.pos.x * 10000.0 + cell.energy;
    point.pos += rand_vec2(seed) * 3.0f;
    point.prev_pos = point.pos;
    points[cell.point_idx] = point;


    // Simple metabolism and nutrient intake
    let metabolic_loss = (0.02 + 0.001 * point.radius) * dt;
    let nutrient_gain = absorb_nutrients(point.pos, point.radius, 0.05 * point.radius);
    cell.energy = max(cell.energy - metabolic_loss + nutrient_gain, 0.0);

    if cell.energy <= 0.0 || point.radius <= 0.1 {
        //kill_cell(idx);
        return;
    }

    cell.color = compute_cell_color(point.radius, cell.energy);

    cells[idx] = cell;
}
